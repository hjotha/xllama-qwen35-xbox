#include "dflash-gpu-sync-probe.h"

#include "../../beellama.cpp/ggml/src/ggml-quants.h"
#include "../../beellama.cpp/src/llama-kv-cache-iswa.h"
#include "../../beellama.cpp/src/llama-kv-cache.h"
#include "dflash-gpu-engine.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

double milliseconds(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

std::string json_string(const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}

std::string json_number(double value) {
    if (!std::isfinite(value))
        return "null";
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(17) << value;
    return stream.str();
}

struct comparison {
    size_t elements = 0;
    size_t exact_bits = 0;
    size_t within_absolute = 0;
    size_t within_relative = 0;
    size_t within_combined = 0;
    size_t nonfinite = 0;
    size_t mismatches = 0;
    size_t first_mismatch = static_cast<size_t>(-1);
    float first_cpu = 0;
    float first_gpu = 0;
    double first_absolute = 0;
    double first_relative = 0;
    double max_absolute = 0;
    double max_relative = 0;
};

comparison compare_all(const float* cpu, const float* gpu, size_t count, float absolute_tolerance,
                       float relative_tolerance) {
    comparison result;
    result.elements = count;
    for (size_t index = 0; index < count; ++index) {
        const float a = cpu[index];
        const float b = gpu[index];
        if (std::memcmp(&a, &b, sizeof(float)) == 0)
            ++result.exact_bits;
        const bool finite = std::isfinite(a) && std::isfinite(b);
        const double absolute =
            finite ? std::abs(double(a) - double(b)) : std::numeric_limits<double>::quiet_NaN();
        const double magnitude = finite ? std::max(std::abs(double(a)), std::abs(double(b))) : 0;
        const double relative = finite ? absolute / std::max(magnitude, 1.0e-8)
                                       : std::numeric_limits<double>::quiet_NaN();
        if (!finite)
            ++result.nonfinite;
        else {
            result.max_absolute = std::max(result.max_absolute, absolute);
            result.max_relative = std::max(result.max_relative, relative);
            if (absolute <= absolute_tolerance)
                ++result.within_absolute;
            if (relative <= relative_tolerance)
                ++result.within_relative;
        }
        const bool within = finite && absolute <= double(absolute_tolerance) +
                                                      double(relative_tolerance) * magnitude;
        if (within)
            ++result.within_combined;
        else {
            ++result.mismatches;
            if (result.first_mismatch == static_cast<size_t>(-1)) {
                result.first_mismatch = index;
                result.first_cpu = a;
                result.first_gpu = b;
                result.first_absolute = absolute;
                result.first_relative = relative;
            }
        }
    }
    return result;
}

struct phase_result {
    std::string name;
    uint32_t tokens = 0;
    double cpu_ms = 0;
    double gpu_ms = 0;
    double readback_ms = 0;
    comparison values;
    bool kv_validity = false;
};

struct fc_reference {
    std::vector<uint8_t> weight;
    std::vector<float> gamma;
    ggml_type type = GGML_TYPE_COUNT;
    float scale = 1.0f;
    float epsilon = 1.0e-6f;
};

bool load_fc_reference(const std::string& model_path, fc_reference* reference, std::string* error) {
    using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    gguf_ptr gguf(gguf_init_from_file(model_path.c_str(), {true, nullptr}), &gguf_free);
    if (!gguf) {
        *error = "GGUF metadata unavailable for FC diagnostic";
        return false;
    }
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        *error = "GGUF file unavailable for FC diagnostic";
        return false;
    }
    const uint64_t length = static_cast<uint64_t>(file.tellg());
    const uint64_t data_offset = gguf_get_data_offset(gguf.get());
    auto read_tensor = [&](const char* name, void* destination, size_t bytes, ggml_type type,
                           int64_t width, int64_t height) -> bool {
        const int64_t id = gguf_find_tensor(gguf.get(), name);
        if (id < 0 || gguf_get_tensor_type(gguf.get(), id) != type ||
            gguf_get_tensor_size(gguf.get(), id) != bytes) {
            *error = std::string("FC diagnostic tensor mismatch: ") + name;
            return false;
        }
        const int64_t* shape = gguf_get_tensor_ne(gguf.get(), id);
        if (!shape || shape[0] != width || shape[1] != height) {
            *error = std::string("FC diagnostic shape mismatch: ") + name;
            return false;
        }
        const uint64_t relative = gguf_get_tensor_offset(gguf.get(), id);
        if (relative > std::numeric_limits<uint64_t>::max() - data_offset) {
            *error = std::string("FC diagnostic offset overflow: ") + name;
            return false;
        }
        const uint64_t offset = data_offset + relative;
        if (offset > length || bytes > length - offset) {
            *error = std::string("FC diagnostic file range invalid: ") + name;
            return false;
        }
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
        if (!file) {
            *error = std::string("FC diagnostic read failed: ") + name;
            return false;
        }
        return true;
    };
    const int64_t fc_id = gguf_find_tensor(gguf.get(), "fc.weight");
    if (fc_id < 0) {
        *error = "FC diagnostic fc.weight missing";
        return false;
    }
    reference->type = gguf_get_tensor_type(gguf.get(), fc_id);
    reference->weight.resize(gguf_get_tensor_size(gguf.get(), fc_id));
    reference->gamma.resize(DflashGpuSyncEngine::kHiddenWidth);
    if (!read_tensor("fc.weight", reference->weight.data(), reference->weight.size(),
                     reference->type, DflashGpuSyncEngine::kFeatureWidth,
                     DflashGpuSyncEngine::kHiddenWidth) ||
        !read_tensor("enc.output_norm.weight", reference->gamma.data(),
                     reference->gamma.size() * sizeof(float), GGML_TYPE_F32,
                     DflashGpuSyncEngine::kHiddenWidth, 1))
        return false;
    const int64_t scale_id = gguf_find_tensor(gguf.get(), "fc.scale");
    if (scale_id >= 0 &&
        !read_tensor("fc.scale", &reference->scale, sizeof(float), GGML_TYPE_F32, 1, 1))
        return false;
    const int64_t epsilon_id = gguf_find_key(gguf.get(), "dflash.attention.layer_norm_rms_epsilon");
    if (epsilon_id < 0 || gguf_get_kv_type(gguf.get(), epsilon_id) != GGUF_TYPE_FLOAT32) {
        *error = "FC diagnostic epsilon metadata missing";
        return false;
    }
    reference->epsilon = gguf_get_val_f32(gguf.get(), epsilon_id);
    return true;
}

bool cpu_fc_matmul(const fc_reference& reference, const float* features, uint32_t count,
                   std::vector<float>* output, std::string* error) {
    using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    backend_ptr backend(ggml_backend_cpu_init(), &ggml_backend_free);
    if (!backend) {
        *error = "FC CPU backend init failed";
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 4);
    context_ptr graph_context(ggml_init({1024 * 1024, nullptr, true}), &ggml_free);
    if (!graph_context) {
        *error = "FC ggml context init failed";
        return false;
    }
    ggml_tensor* w =
        ggml_new_tensor_2d(graph_context.get(), reference.type, DflashGpuSyncEngine::kFeatureWidth,
                           DflashGpuSyncEngine::kHiddenWidth);
    ggml_tensor* x = ggml_new_tensor_2d(graph_context.get(), GGML_TYPE_F32,
                                        DflashGpuSyncEngine::kFeatureWidth, count);
    ggml_tensor* y = ggml_mul_mat(graph_context.get(), w, x);
    ggml_cgraph* graph = ggml_new_graph(graph_context.get());
    if (!w || !x || !y || !graph || ggml_nbytes(w) != reference.weight.size()) {
        *error = "FC ggml graph mismatch";
        return false;
    }
    ggml_build_forward_expand(graph, y);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(graph_context.get(), backend.get()),
                      &ggml_backend_buffer_free);
    if (!buffer) {
        *error = "FC ggml tensor allocation failed";
        return false;
    }
    ggml_backend_tensor_set(w, reference.weight.data(), 0, reference.weight.size());
    ggml_backend_tensor_set(x, features, 0,
                            size_t(count) * DflashGpuSyncEngine::kFeatureWidth * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        *error = "FC ggml matrix compute failed";
        return false;
    }
    output->resize(size_t(count) * DflashGpuSyncEngine::kHiddenWidth);
    ggml_backend_tensor_get(y, output->data(), 0, output->size() * sizeof(float));
    return true;
}

std::vector<float> cpu_norm_from_fc(const fc_reference& reference, const std::vector<float>& fc) {
    const size_t width = DflashGpuSyncEngine::kHiddenWidth;
    std::vector<float> result(fc.size());
    for (size_t row = 0; row < fc.size() / width; ++row) {
        double sum = 0;
        for (size_t col = 0; col < width; ++col) {
            const float x = fc[row * width + col] * reference.scale;
            sum += double(x * x);
        }
        const float mean = static_cast<float>(sum / double(width));
        const float factor = 1.0f / std::sqrt(mean + reference.epsilon);
        for (size_t col = 0; col < width; ++col) {
            const float x = fc[row * width + col] * reference.scale;
            result[row * width + col] = x * factor * reference.gamma[col];
        }
    }
    return result;
}

std::string phase_json(const phase_result& phase) {
    const comparison& c = phase.values;
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "{\"name\":" << json_string(phase.name) << ",\"tokens\":" << phase.tokens
           << ",\"elements\":" << c.elements << ",\"cpuMs\":" << json_number(phase.cpu_ms)
           << ",\"gpuSyncMs\":" << json_number(phase.gpu_ms)
           << ",\"readbackMs\":" << json_number(phase.readback_ms)
           << ",\"exactBits\":" << c.exact_bits << ",\"withinAbsolute\":" << c.within_absolute
           << ",\"withinRelative\":" << c.within_relative
           << ",\"withinCombined\":" << c.within_combined << ",\"mismatches\":" << c.mismatches
           << ",\"nonfinite\":" << c.nonfinite << ",\"maxAbs\":" << json_number(c.max_absolute)
           << ",\"maxRel\":" << json_number(c.max_relative)
           << ",\"kvLogicalValidity\":" << (phase.kv_validity ? "true" : "false")
           << ",\"firstMismatch\":";
    if (c.first_mismatch == static_cast<size_t>(-1)) {
        stream << "null";
    } else {
        stream << "{\"index\":" << c.first_mismatch
               << ",\"token\":" << c.first_mismatch / DflashGpuSyncEngine::kHiddenWidth
               << ",\"column\":" << c.first_mismatch % DflashGpuSyncEngine::kHiddenWidth
               << ",\"cpu\":" << json_number(c.first_cpu) << ",\"gpu\":" << json_number(c.first_gpu)
               << ",\"abs\":" << json_number(c.first_absolute)
               << ",\"rel\":" << json_number(c.first_relative) << '}';
    }
    stream << '}';
    return stream.str();
}

std::string intermediate_json(const std::string& name, const comparison& c) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "{\"name\":" << json_string(name) << ",\"elements\":" << c.elements
           << ",\"exactBits\":" << c.exact_bits << ",\"withinCombined\":" << c.within_combined
           << ",\"mismatches\":" << c.mismatches << ",\"nonfinite\":" << c.nonfinite
           << ",\"maxAbs\":" << json_number(c.max_absolute)
           << ",\"maxRel\":" << json_number(c.max_relative) << ",\"firstMismatch\":";
    if (c.first_mismatch == static_cast<size_t>(-1))
        stream << "null";
    else
        stream << "{\"index\":" << c.first_mismatch << ",\"cpu\":" << json_number(c.first_cpu)
               << ",\"gpu\":" << json_number(c.first_gpu)
               << ",\"abs\":" << json_number(c.first_absolute)
               << ",\"rel\":" << json_number(c.first_relative) << '}';
    stream << '}';
    return stream.str();
}

std::string q8_byte_parity_json(const char* name, const uint8_t* cpu, const uint8_t* gpu,
                                size_t bytes, size_t blocks_per_row, bool* numeric_pass = nullptr) {
    constexpr size_t block_bytes = sizeof(block_q8_K);
    constexpr size_t index_cap = 1024;
    constexpr size_t sample_cap_per_field = 12;
    static constexpr const char* field_names[] = {"scale", "qs", "bsums"};
    size_t mismatches = 0;
    size_t first = bytes;
    size_t field_mismatches[3] = {};
    unsigned field_max_byte_delta[3] = {};
    std::vector<size_t> mismatch_indexes;
    std::vector<size_t> field_samples[3];
    size_t mismatched_blocks = 0;
    size_t last_mismatched_block = bytes;
    uint64_t cpu_hash = 14695981039346656037ull;
    uint64_t gpu_hash = cpu_hash;
    for (size_t i = 0; i < bytes; ++i) {
        cpu_hash = (cpu_hash ^ cpu[i]) * 1099511628211ull;
        gpu_hash = (gpu_hash ^ gpu[i]) * 1099511628211ull;
        if (cpu[i] != gpu[i]) {
            ++mismatches;
            if (first == bytes)
                first = i;
            const size_t within = i % block_bytes;
            const size_t field = within < 4 ? 0 : within < 260 ? 1 : 2;
            ++field_mismatches[field];
            const unsigned delta = unsigned(std::abs(int(cpu[i]) - int(gpu[i])));
            field_max_byte_delta[field] = std::max(field_max_byte_delta[field], delta);
            if (mismatch_indexes.size() < index_cap)
                mismatch_indexes.push_back(i);
            if (field_samples[field].size() < sample_cap_per_field)
                field_samples[field].push_back(i);
            const size_t block = i / block_bytes;
            if (block != last_mismatched_block) {
                ++mismatched_blocks;
                last_mismatched_block = block;
            }
        }
    }
    size_t mismatched_scale_blocks = 0;
    size_t nonfinite_scales = 0;
    double max_absolute_scale = 0;
    double max_relative_scale = 0;
    uint64_t max_scale_ulp = 0;
    size_t scale_blocks_beyond_one_ulp = 0;
    size_t payload_mismatched_blocks = 0;
    for (size_t block = 0; block < bytes / block_bytes; ++block) {
        const size_t offset = block * block_bytes;
        if (std::memcmp(cpu + offset + 4, gpu + offset + 4, block_bytes - 4) != 0)
            ++payload_mismatched_blocks;
        uint32_t cpu_bits = 0, gpu_bits = 0;
        std::memcpy(&cpu_bits, cpu + offset, 4);
        std::memcpy(&gpu_bits, gpu + offset, 4);
        const uint64_t ulp =
            cpu_bits > gpu_bits ? uint64_t(cpu_bits - gpu_bits) : uint64_t(gpu_bits - cpu_bits);
        max_scale_ulp = std::max(max_scale_ulp, ulp);
        if (ulp > 1)
            ++scale_blocks_beyond_one_ulp;
        float cpu_scale = 0;
        float gpu_scale = 0;
        std::memcpy(&cpu_scale, cpu + offset, sizeof(float));
        std::memcpy(&gpu_scale, gpu + offset, sizeof(float));
        if (!std::isfinite(cpu_scale) || !std::isfinite(gpu_scale)) {
            ++nonfinite_scales;
            continue;
        }
        if (cpu_bits == gpu_bits)
            continue;
        ++mismatched_scale_blocks;
        const double absolute = std::abs(double(cpu_scale) - double(gpu_scale));
        const double magnitude = std::max(std::abs(double(cpu_scale)), std::abs(double(gpu_scale)));
        max_absolute_scale = std::max(max_absolute_scale, absolute);
        max_relative_scale = std::max(max_relative_scale, absolute / std::max(magnitude, 1.0e-8));
    }
    const bool passed =
        payload_mismatched_blocks == 0 && nonfinite_scales == 0 && scale_blocks_beyond_one_ulp == 0;
    if (numeric_pass)
        *numeric_pass = passed;
    auto write_mismatch = [&](std::ostringstream& out, size_t i) {
        const size_t block = i / block_bytes;
        const size_t within = i % block_bytes;
        const size_t field = within < 4 ? 0 : within < 260 ? 1 : 2;
        const int delta = int(gpu[i]) - int(cpu[i]);
        out << "{\"index\":" << i << ",\"token\":" << block / blocks_per_row
            << ",\"block\":" << block % blocks_per_row << ",\"byteInBlock\":" << within
            << ",\"field\":" << json_string(field_names[field]) << ",\"cpu\":" << unsigned(cpu[i])
            << ",\"gpu\":" << unsigned(gpu[i]) << ",\"signedByteDelta\":" << delta
            << ",\"absByteDelta\":" << std::abs(delta) << '}';
    };
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "{\"name\":" << json_string(name) << ",\"bytes\":" << bytes
           << ",\"exactBytes\":" << bytes - mismatches << ",\"mismatches\":" << mismatches
           << ",\"mismatchedBlocks\":" << mismatched_blocks
           << ",\"numericGatePassed\":" << (passed ? "true" : "false")
           << ",\"payloadMismatchedBlocks\":" << payload_mismatched_blocks
           << ",\"scaleBlocksBeyondOneUlp\":" << scale_blocks_beyond_one_ulp
           << ",\"maxScaleUlp\":" << max_scale_ulp << ",\"cpuFnv1a64\":\"" << std::hex
           << std::setfill('0') << std::setw(16) << cpu_hash << "\",\"gpuFnv1a64\":\""
           << std::setw(16) << gpu_hash << "\"" << std::dec << ",\"firstMismatch\":";
    if (first == bytes)
        stream << "null";
    else
        write_mismatch(stream, first);
    stream << ",\"fieldStats\":{";
    for (size_t field = 0; field < 3; ++field) {
        if (field)
            stream << ',';
        stream << json_string(field_names[field]) << ":{\"mismatches\":" << field_mismatches[field]
               << ",\"maxAbsByteDelta\":" << field_max_byte_delta[field] << '}';
    }
    stream << "},\"scaleFloat\":{\"mismatchedBlocks\":" << mismatched_scale_blocks
           << ",\"nonfinite\":" << nonfinite_scales
           << ",\"maxAbs\":" << json_number(max_absolute_scale)
           << ",\"maxRel\":" << json_number(max_relative_scale) << '}'
           << ",\"mismatchByteIndexes\":[";
    for (size_t i = 0; i < mismatch_indexes.size(); ++i) {
        if (i)
            stream << ',';
        stream << mismatch_indexes[i];
    }
    stream << "],\"indexesTruncated\":" << (mismatches > mismatch_indexes.size() ? "true" : "false")
           << ",\"sampleCapPerField\":" << sample_cap_per_field << ",\"fieldSamples\":{";
    for (size_t field = 0; field < 3; ++field) {
        if (field)
            stream << ',';
        stream << json_string(field_names[field]) << ":[";
        for (size_t i = 0; i < field_samples[field].size(); ++i) {
            if (i)
                stream << ',';
            write_mismatch(stream, field_samples[field][i]);
        }
        stream << ']';
    }
    stream << '}';
    stream << '}';
    return stream.str();
}

std::string q8_fc_scale_samples_json(const float* features, const uint8_t* cpu, const uint8_t* gpu,
                                     uint32_t count) {
    constexpr size_t blocks_per_row = DflashGpuSyncEngine::kFeatureWidth / 256;
    constexpr size_t block_bytes = sizeof(block_q8_K);
    auto float_bits = [](float value) {
        uint32_t result = 0;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    };
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "{\"name\":\"fc_activation_q8_scale_samples\",\"samples\":[";
    size_t samples = 0;
    bool positive_delta = false;
    bool negative_delta = false;
    for (uint32_t row = 0; row < count; ++row) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            const size_t offset = (size_t(row) * blocks_per_row + block) * block_bytes;
            if (std::memcmp(cpu + offset, gpu + offset, 4) == 0)
                continue;
            float cpu_scale = 0, gpu_scale = 0;
            std::memcpy(&cpu_scale, cpu + offset, 4);
            std::memcpy(&gpu_scale, gpu + offset, 4);
            const int64_t delta = int64_t(float_bits(gpu_scale)) - int64_t(float_bits(cpu_scale));
            if (samples >= 4 && ((delta > 0 && positive_delta) || (delta < 0 && negative_delta)))
                continue;
            float signed_max = 0, max_abs = 0;
            const float* values =
                features + size_t(row) * DflashGpuSyncEngine::kFeatureWidth + block * 256;
            for (size_t lane = 0; lane < 256; ++lane) {
                const float absolute = std::fabs(values[lane]);
                if (absolute > max_abs) {
                    max_abs = absolute;
                    signed_max = values[lane];
                }
            }
            const float cpu_iscale = signed_max == 0 ? 0 : -127.0f / signed_max;
            if (samples++)
                stream << ',';
            stream << "{\"token\":" << row << ",\"block\":" << block
                   << ",\"signedMax\":" << json_number(signed_max)
                   << ",\"signedMaxBits\":" << float_bits(signed_max)
                   << ",\"cpuIscaleBits\":" << float_bits(cpu_iscale)
                   << ",\"cpuScaleBits\":" << float_bits(cpu_scale)
                   << ",\"gpuScaleBits\":" << float_bits(gpu_scale)
                   << ",\"scaleBitDelta\":" << delta << '}';
            if (delta > 0)
                positive_delta = true;
            if (delta < 0)
                negative_delta = true;
            if (samples >= 12 && positive_delta && negative_delta) {
                stream << "]}";
                return stream.str();
            }
        }
    }
    stream << "]}";
    return stream.str();
}

std::vector<float> hadamard_cpu_reference(const std::vector<float>& input,
                                          uint32_t transform_width) {
    std::vector<float> output(input.size());
    constexpr uint32_t head_width = DflashGpuSyncEngine::kHeadWidth;
    const float factor = transform_width == 64 ? 0.125f : 0.08838834764831845f;
    for (size_t base = 0; base < input.size(); base += head_width) {
        for (uint32_t part = 0; part < head_width; part += transform_width) {
            for (uint32_t dst = 0; dst < transform_width; ++dst) {
                float sum = 0;
                for (uint32_t src = 0; src < transform_width; ++src) {
                    uint32_t common = dst & src;
                    bool negative = false;
                    while (common) {
                        negative = !negative;
                        common &= common - 1;
                    }
                    sum += input[base + part + src] * (negative ? -factor : factor);
                }
                output[base + part + dst] = sum;
            }
        }
    }
    return output;
}

std::vector<float> fwht_cpu_reference(const std::vector<float>& input, uint32_t transform_width) {
    std::vector<float> output = input;
    const float factor = transform_width == 64 ? 0.125f : 0.08838834764831845f;
    for (size_t base = 0; base < output.size(); base += transform_width) {
        for (uint32_t i = 0; i < transform_width; ++i)
            output[base + i] *= factor;
        for (uint32_t stride = 1; stride < transform_width; stride <<= 1) {
            for (uint32_t pair = 0; pair < transform_width; pair += 2 * stride) {
                for (uint32_t lane = 0; lane < stride; ++lane) {
                    const size_t lo = base + pair + lane;
                    const size_t hi = lo + stride;
                    const float a = output[lo], b = output[hi];
                    output[lo] = a + b;
                    output[hi] = a - b;
                }
            }
        }
    }
    return output;
}

void json_float_block(std::ostringstream& out, const std::vector<float>& values, size_t first) {
    out << '[';
    for (size_t i = 0; i < 32; ++i) {
        if (i)
            out << ',';
        out << json_number(values[first + i]);
    }
    out << ']';
}

void json_byte_block(std::ostringstream& out, const std::vector<uint8_t>& values, size_t first) {
    out << '[';
    for (size_t i = 0; i < 20; ++i) {
        if (i)
            out << ',';
        out << unsigned(values[first + i]);
    }
    out << ']';
}

std::vector<uint8_t> q4_gpu_stride_reference(const std::vector<float>& input) {
    static_assert(sizeof(block_q4_0) == 18, "GGML Q4_0 block size changed");
    std::vector<block_q4_0> blocks(input.size() / 32);
    quantize_row_q4_0_ref(input.data(), blocks.data(), static_cast<int64_t>(input.size()));
    std::vector<uint8_t> gpu_stride(blocks.size() * 20, 0);
    for (size_t block = 0; block < blocks.size(); ++block) {
        std::memcpy(gpu_stride.data() + block * 20, &blocks[block].d, 2);
        std::memcpy(gpu_stride.data() + block * 20 + 4, blocks[block].qs, 16);
    }
    return gpu_stride;
}

std::string q4_physical_json(const char* name, const std::vector<uint8_t>& expected,
                             const std::vector<uint8_t>& actual) {
    size_t mismatches = 0;
    size_t first = expected.size();
    size_t fields[3] = {}; // scale, aligned padding, qs
    unsigned max_byte_delta = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] == actual[i])
            continue;
        ++mismatches;
        if (first == expected.size())
            first = i;
        const size_t within = i % 20;
        ++fields[within < 2 ? 0 : within < 4 ? 1 : 2];
        max_byte_delta =
            std::max(max_byte_delta, unsigned(std::abs(int(expected[i]) - int(actual[i]))));
    }
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "{\"name\":" << json_string(name)
           << ",\"diagnosticOnly\":true,\"bytes\":" << expected.size()
           << ",\"exactBytes\":" << expected.size() - mismatches << ",\"mismatches\":" << mismatches
           << ",\"scaleMismatchBytes\":" << fields[0] << ",\"paddingMismatchBytes\":" << fields[1]
           << ",\"qsMismatchBytes\":" << fields[2] << ",\"maxAbsByteDelta\":" << max_byte_delta
           << ",\"firstMismatch\":";
    if (first == expected.size())
        stream << "null";
    else
        stream << "{\"index\":" << first << ",\"block\":" << first / 20
               << ",\"byteInBlock\":" << first % 20 << ",\"cpu\":" << unsigned(expected[first])
               << ",\"gpu\":" << unsigned(actual[first]) << '}';
    stream << '}';
    return stream.str();
}

std::string q4_cpu_gpu_rows_json(const std::string& name, const std::vector<uint8_t>& cpu,
                                 const std::vector<uint8_t>& gpu, const int32_t* positions,
                                 uint32_t count) {
    constexpr size_t row_bytes =
        (DflashGpuSyncEngine::kKvHeads * DflashGpuSyncEngine::kHeadWidth / 32) * 20;
    size_t total_mismatches = 0;
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "{\"name\":" << json_string(name)
           << ",\"diagnosticOnly\":true,\"source\":\"actual_cpu_q4_cache\""
           << ",\"rows\":[";
    for (uint32_t row = 0; row < count; ++row) {
        if (row)
            stream << ',';
        size_t mismatches = 0;
        size_t first = row_bytes;
        unsigned max_delta = 0;
        for (size_t i = 0; i < row_bytes; ++i) {
            const size_t index = size_t(row) * row_bytes + i;
            if (cpu[index] == gpu[index])
                continue;
            ++mismatches;
            if (first == row_bytes)
                first = i;
            max_delta = std::max(max_delta, unsigned(std::abs(int(cpu[index]) - int(gpu[index]))));
        }
        total_mismatches += mismatches;
        stream << "{\"position\":" << positions[row] << ",\"exactBytes\":" << row_bytes - mismatches
               << ",\"mismatches\":" << mismatches << ",\"maxAbsByteDelta\":" << max_delta
               << ",\"firstMismatch\":";
        if (first == row_bytes)
            stream << "null";
        else {
            const size_t index = size_t(row) * row_bytes + first;
            stream << "{\"block\":" << first / 20 << ",\"byteInBlock\":" << first % 20
                   << ",\"cpu\":" << unsigned(cpu[index]) << ",\"gpu\":" << unsigned(gpu[index])
                   << '}';
        }
        stream << '}';
    }
    stream << "],\"bytes\":" << size_t(count) * row_bytes
           << ",\"exactBytes\":" << size_t(count) * row_bytes - total_mismatches
           << ",\"mismatches\":" << total_mismatches << '}';
    return stream.str();
}

std::string q4_v_scale_boundary_json(const std::vector<uint8_t>& cpu,
                                     const std::vector<uint8_t>& gpu,
                                     const std::vector<float>& gpu_rotated,
                                     const int32_t* positions, uint32_t count) {
    constexpr size_t row_bytes = 640;
    constexpr size_t blocks_per_row = row_bytes / 20;
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "{\"name\":\"last_layer_v_q4_scale_boundary\",\"firstScaleMismatch\":";
    for (uint32_t row = 0; row < count; ++row) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            const size_t offset = size_t(row) * row_bytes + block * 20;
            if (std::memcmp(cpu.data() + offset, gpu.data() + offset, 2) == 0)
                continue;
            uint16_t cpu_bits = 0, gpu_bits = 0;
            std::memcpy(&cpu_bits, cpu.data() + offset, 2);
            std::memcpy(&gpu_bits, gpu.data() + offset, 2);
            float signed_max = 0, max_abs = 0;
            const size_t first = (size_t(row) * blocks_per_row + block) * 32;
            for (size_t lane = 0; lane < 32; ++lane) {
                const float value = gpu_rotated[first + lane];
                if (std::fabs(value) > max_abs) {
                    max_abs = std::fabs(value);
                    signed_max = value;
                }
            }
            const float gpu_scale_f32 = signed_max / -8.0f;
            const float cpu_scale = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(cpu_bits));
            const float gpu_scale = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(gpu_bits));
            const double midpoint = (double(cpu_scale) + double(gpu_scale)) * 0.5;
            stream << "{\"position\":" << positions[row] << ",\"block\":" << block
                   << ",\"gpuRotatedSignedMax\":" << json_number(signed_max)
                   << ",\"gpuScaleF32\":" << json_number(gpu_scale_f32)
                   << ",\"cpuScaleF16Bits\":" << cpu_bits << ",\"gpuScaleF16Bits\":" << gpu_bits
                   << ",\"cpuScaleF16\":" << json_number(cpu_scale)
                   << ",\"gpuScaleF16\":" << json_number(gpu_scale)
                   << ",\"fp16Midpoint\":" << json_number(midpoint)
                   << ",\"gpuDistanceFromMidpoint\":"
                   << json_number(double(gpu_scale_f32) - midpoint) << "}}";
            return stream.str();
        }
    }
    stream << "null}";
    return stream.str();
}

struct q4_numeric_result {
    bool passed = true;
    size_t scale_mismatched_blocks = 0;
    size_t payload_mismatched_blocks = 0;
    size_t nonfinite_scale_blocks = 0;
    size_t scale_blocks_beyond_one_ulp = 0;
    size_t scale_blocks_without_midpoint_evidence = 0;
    double max_midpoint_distance_f32_ulps = 0;
    std::string first_failure;
};

q4_numeric_result q4_numeric_gate(const std::vector<uint8_t>& cpu, const std::vector<uint8_t>& gpu,
                                  const DflashGpuSyncEngine::KvSnapshot& snapshot, uint32_t count) {
    constexpr size_t row_bytes = 640;
    constexpr size_t blocks_per_row = row_bytes / 20;
    q4_numeric_result result;
    const size_t required_bytes = size_t(DflashGpuSyncEngine::kLayers) * 2 * count * row_bytes;
    if (cpu.size() != required_bytes || gpu.size() != required_bytes) {
        result.passed = false;
        result.first_failure = "Q4 physical snapshot size mismatch";
        return result;
    }
    const auto reject = [&](const char* reason, uint32_t layer, uint32_t kind, uint32_t row,
                            size_t block) {
        result.passed = false;
        if (result.first_failure.empty()) {
            result.first_failure = std::string(reason) + " layer=" + std::to_string(layer) +
                                   " kind=" + std::to_string(kind) + " row=" + std::to_string(row) +
                                   " block=" + std::to_string(block);
        }
    };
    for (uint32_t layer = 0; layer < DflashGpuSyncEngine::kLayers; ++layer) {
        for (uint32_t kind = 0; kind < 2; ++kind) {
            for (uint32_t row = 0; row < count; ++row) {
                for (size_t block = 0; block < blocks_per_row; ++block) {
                    const size_t offset =
                        (((size_t(layer) * 2 + kind) * count + row) * blocks_per_row + block) * 20;
                    if (std::memcmp(cpu.data() + offset + 2, gpu.data() + offset + 2, 18) != 0) {
                        ++result.payload_mismatched_blocks;
                        reject("Q4 payload/layout mismatch", layer, kind, row, block);
                    }
                    uint16_t cpu_bits = 0, gpu_bits = 0;
                    std::memcpy(&cpu_bits, cpu.data() + offset, 2);
                    std::memcpy(&gpu_bits, gpu.data() + offset, 2);
                    const float cpu_scale = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(cpu_bits));
                    const float gpu_scale = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(gpu_bits));
                    if (!std::isfinite(cpu_scale) || !std::isfinite(gpu_scale)) {
                        ++result.nonfinite_scale_blocks;
                        reject("Q4 nonfinite scale", layer, kind, row, block);
                        continue;
                    }
                    if (cpu_bits == gpu_bits)
                        continue;
                    ++result.scale_mismatched_blocks;
                    const unsigned ulp = cpu_bits > gpu_bits ? unsigned(cpu_bits - gpu_bits)
                                                             : unsigned(gpu_bits - cpu_bits);
                    if (ulp > 1) {
                        ++result.scale_blocks_beyond_one_ulp;
                        reject("Q4 scale exceeds one FP16 ULP", layer, kind, row, block);
                    }
                    if (layer != DflashGpuSyncEngine::kLayers - 1) {
                        ++result.scale_blocks_without_midpoint_evidence;
                        reject("Q4 scale has no prepack snapshot", layer, kind, row, block);
                        continue;
                    }
                    const auto& rotated = kind == 0 ? snapshot.rotated_key : snapshot.rotated_value;
                    const size_t first = (size_t(row) * blocks_per_row + block) * 32;
                    if (first + 32 > rotated.size()) {
                        ++result.scale_blocks_without_midpoint_evidence;
                        reject("Q4 prepack snapshot out of range", layer, kind, row, block);
                        continue;
                    }
                    float signed_max = 0, max_abs = 0;
                    for (size_t lane = 0; lane < 32; ++lane) {
                        const float value = rotated[first + lane];
                        if (std::fabs(value) > max_abs) {
                            max_abs = std::fabs(value);
                            signed_max = value;
                        }
                    }
                    const float gpu_scale_f32 = signed_max / -8.0f;
                    if (ggml_fp32_to_fp16(gpu_scale_f32) != static_cast<ggml_fp16_t>(gpu_bits)) {
                        ++result.scale_blocks_without_midpoint_evidence;
                        reject("Q4 GPU scale does not round from prepack F32", layer, kind, row,
                               block);
                    }
                    const float midpoint = float((double(cpu_scale) + double(gpu_scale)) * 0.5);
                    const float next =
                        std::nextafter(midpoint, std::numeric_limits<float>::infinity());
                    const double f32_ulp = double(next) - double(midpoint);
                    const double distance =
                        f32_ulp > 0 ? std::abs(double(gpu_scale_f32) - double(midpoint)) / f32_ulp
                                    : std::numeric_limits<double>::infinity();
                    result.max_midpoint_distance_f32_ulps =
                        std::max(result.max_midpoint_distance_f32_ulps, distance);
                    if (!std::isfinite(distance) || distance > 8.0) {
                        ++result.scale_blocks_without_midpoint_evidence;
                        reject("Q4 scale far from FP16 midpoint", layer, kind, row, block);
                    }
                }
            }
        }
    }
    return result;
}

std::string q4_numeric_gate_json(const q4_numeric_result& result) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "{\"name\":\"kv_q4_numeric_gate\",\"passed\":" << (result.passed ? "true" : "false")
           << ",\"scaleMismatchedBlocks\":" << result.scale_mismatched_blocks
           << ",\"payloadMismatchedBlocks\":" << result.payload_mismatched_blocks
           << ",\"nonfiniteScaleBlocks\":" << result.nonfinite_scale_blocks
           << ",\"scaleBlocksBeyondOneUlp\":" << result.scale_blocks_beyond_one_ulp
           << ",\"scaleBlocksWithoutMidpointEvidence\":"
           << result.scale_blocks_without_midpoint_evidence << ",\"maxMidpointDistanceF32Ulps\":"
           << json_number(result.max_midpoint_distance_f32_ulps) << ",\"firstFailure\":"
           << (result.first_failure.empty() ? "null" : json_string(result.first_failure)) << '}';
    return stream.str();
}

std::string cpu_cache_group_json(const char* name, llama_kv_cache* cache, const int32_t* positions,
                                 uint32_t count) {
    if (!cache)
        return "null";
    std::ostringstream stream;
    stream << "{\"name\":" << json_string(name) << ",\"typeK\":" << int(cache->type_k())
           << ",\"typeV\":" << int(cache->type_v()) << ",\"streams\":" << cache->get_n_stream()
           << ",\"capacity\":" << cache->get_size() << ",\"layers\":[";
    const auto ids = cache->get_layer_ids();
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i)
            stream << ',';
        stream << ids[i];
    }
    stream << "],\"slots\":[";
    if (cache->get_n_stream() > 0) {
        for (uint32_t row = 0; row < count; ++row) {
            if (row)
                stream << ',';
            const auto cells = cache->cells_at(0, positions[row]);
            stream << "{\"position\":" << positions[row] << ",\"cells\":[";
            for (size_t i = 0; i < cells.size(); ++i) {
                if (i)
                    stream << ',';
                stream << cells[i];
            }
            stream << "]}";
        }
    }
    stream << "]}";
    return stream.str();
}

std::string cpu_tensor_geometry_json(const char* name, const ggml_tensor* tensor) {
    if (!tensor)
        return "null";
    std::ostringstream stream;
    stream << "{\"name\":" << json_string(name) << ",\"type\":" << int(tensor->type) << ",\"ne\":[";
    for (int i = 0; i < 4; ++i) {
        if (i)
            stream << ',';
        stream << tensor->ne[i];
    }
    stream << "],\"nb\":[";
    for (int i = 0; i < 4; ++i) {
        if (i)
            stream << ',';
        stream << tensor->nb[i];
    }
    stream << "],\"hasBuffer\":" << (tensor->buffer ? "true" : "false")
           << ",\"hasData\":" << (tensor->data ? "true" : "false")
           << ",\"hasViewSource\":" << (tensor->view_src ? "true" : "false") << '}';
    return stream.str();
}

bool cpu_kv_packed_snapshot(llama_context* ctx, const int32_t* positions, uint32_t count,
                            std::vector<uint8_t>* output, std::string* diagnostics,
                            std::string* error) {
    constexpr uint32_t layers = DflashGpuSyncEngine::kLayers;
    constexpr uint32_t width = DflashGpuSyncEngine::kKvHeads * DflashGpuSyncEngine::kHeadWidth;
    constexpr size_t blocks_per_row = width / 32;
    constexpr size_t cpu_row_bytes = blocks_per_row * sizeof(block_q4_0);
    constexpr size_t gpu_row_bytes = blocks_per_row * 20;
    static_assert(sizeof(block_q4_0) == 18, "GGML Q4_0 block size changed");
    llama_synchronize(ctx);
    auto* memory = llama_get_memory(ctx);
    auto* direct = dynamic_cast<llama_kv_cache*>(memory);
    auto* iswa = dynamic_cast<llama_kv_cache_iswa*>(memory);
    llama_kv_cache* groups[2] = {direct, nullptr};
    const char* memory_kind = direct ? "standard" : iswa ? "iswa" : "unsupported";
    if (iswa) {
        groups[0] = dynamic_cast<llama_kv_cache*>(iswa->get_base());
        groups[1] = dynamic_cast<llama_kv_cache*>(iswa->get_swa());
    }
    std::ostringstream geometries;
    geometries << '[';
    auto publish = [&] {
        *diagnostics = std::string("{\"name\":\"cpu_kv_geometry\",\"memoryKind\":") +
                       json_string(memory_kind) +
                       ",\"base\":" + cpu_cache_group_json("base", groups[0], positions, count) +
                       ",\"swa\":" + cpu_cache_group_json("swa", groups[1], positions, count) +
                       ",\"tensors\":" + geometries.str() + "]}";
    };
    if (!groups[0] && !groups[1]) {
        *error = "CPU KV has no readable standard cache group";
        publish();
        return false;
    }
    llama_kv_cache* route[layers] = {};
    for (auto* cache : groups) {
        if (!cache)
            continue;
        const auto ids = cache->get_layer_ids();
        if (ids.empty())
            continue; // ISWA's non-SWA group may own no DFlash layers.
        if (cache->type_k() != GGML_TYPE_Q4_0 || cache->type_v() != GGML_TYPE_Q4_0 ||
            cache->get_n_stream() != 1) {
            *error = "CPU KV group type or stream count mismatch";
            publish();
            return false;
        }
        for (const uint32_t id : ids) {
            if (id >= layers || route[id]) {
                *error = "CPU KV layer id out of range or duplicated";
                publish();
                return false;
            }
            route[id] = cache;
        }
    }
    for (auto* cache : route) {
        if (!cache) {
            *error = "CPU KV layer not owned by base/SWA cache";
            publish();
            return false;
        }
    }
    using ggml_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    ggml_ptr metadata(ggml_init({1024 * 1024, nullptr, true}), &ggml_free);
    if (!metadata) {
        *error = "CPU KV metadata context unavailable";
        publish();
        return false;
    }
    output->assign(size_t(layers) * 2 * count * gpu_row_bytes, 0);
    std::vector<uint8_t> raw(cpu_row_bytes);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        auto* cache = route[layer];
        llama_kv_cache::slot_info sinfo{};
        sinfo.s0 = cache->get_stream_for_seq(0);
        sinfo.s1 = sinfo.s0;
        sinfo.strm.push_back(sinfo.s0);
        ggml_tensor* key = cache->get_k_storage(layer);
        ggml_tensor* value = cache->get_v(metadata.get(), layer, cache->get_size(), sinfo);
        if (layer)
            geometries << ',';
        geometries << "{\"layer\":" << layer
                   << ",\"group\":" << json_string(cache == groups[0] ? "base" : "swa")
                   << ",\"key\":" << cpu_tensor_geometry_json("k", key)
                   << ",\"value\":" << cpu_tensor_geometry_json("v", value) << '}';
        if (!key || !value || key->type != GGML_TYPE_Q4_0 || value->type != GGML_TYPE_Q4_0 ||
            key->ne[0] != width || value->ne[0] != DflashGpuSyncEngine::kHeadWidth ||
            value->ne[1] != DflashGpuSyncEngine::kKvHeads || key->nb[1] != cpu_row_bytes ||
            value->nb[2] != cpu_row_bytes || !key->buffer || !key->data || !value->view_src ||
            !value->view_src->buffer || !value->data) {
            *error = "CPU KV Q4 tensor layout mismatch at layer " + std::to_string(layer);
            publish();
            return false;
        }
        std::vector<uint32_t> cells(count);
        for (uint32_t row = 0; row < count; ++row) {
            const auto matches = cache->cells_at(0, positions[row]);
            if (matches.size() != 1 || matches[0] >= cache->get_size()) {
                *error = "CPU KV position maps to zero/multiple cells at layer " +
                         std::to_string(layer) + ", position " + std::to_string(positions[row]);
                publish();
                return false;
            }
            cells[row] = matches[0];
        }
        ggml_tensor* tensors[2] = {key, value};
        for (uint32_t kind = 0; kind < 2; ++kind) {
            for (uint32_t row = 0; row < count; ++row) {
                const size_t tensor_bytes = ggml_nbytes(tensors[kind]);
                if (tensor_bytes < cpu_row_bytes ||
                    size_t(cells[row]) > (tensor_bytes - cpu_row_bytes) / cpu_row_bytes) {
                    *error = "CPU KV physical cell exceeds tensor bounds at layer " +
                             std::to_string(layer);
                    publish();
                    return false;
                }
                ggml_backend_tensor_get(tensors[kind], raw.data(),
                                        size_t(cells[row]) * cpu_row_bytes, cpu_row_bytes);
                const size_t base = (size_t(layer) * 2 + kind) * count * gpu_row_bytes +
                                    size_t(row) * gpu_row_bytes;
                for (size_t block = 0; block < blocks_per_row; ++block) {
                    const uint8_t* src = raw.data() + block * sizeof(block_q4_0);
                    uint8_t* dst = output->data() + base + block * 20;
                    std::memcpy(dst, src, 2);
                    std::memcpy(dst + 4, src + 2, 16);
                }
            }
        }
    }
    publish();
    return true;
}

bool kv_rows_match(const DflashGpuSyncEngine& engine, const int32_t* positions, uint32_t count,
                   bool expected) {
    for (uint32_t row = 0; row < count; ++row) {
        for (uint32_t layer = 0; layer < DflashGpuSyncEngine::kLayers; ++layer) {
            const uint32_t position = static_cast<uint32_t>(positions[row]);
            if (engine.kv().valid(layer, position, DflashGpuKv::Kind::Key) != expected ||
                engine.kv().valid(layer, position, DflashGpuKv::Kind::Value) != expected)
                return false;
        }
    }
    return true;
}

struct owned_batch {
    llama_batch value{};
    ~owned_batch() {
        if (value.embd || value.token)
            llama_batch_free(value);
    }
};

struct kv_cpu_trace {
    std::string key_name;
    std::string value_name;
    std::vector<float> encoder;
    std::vector<float> key;
    std::vector<float> value;
    std::string error;
};

bool kv_cpu_trace_callback(ggml_tensor* tensor, bool ask, void* user_data) {
    auto& trace = *static_cast<kv_cpu_trace*>(user_data);
    if (!tensor)
        return false;
    std::vector<float>* destination = nullptr;
    if (std::strcmp(tensor->name, "inp_g_embeddings") == 0)
        destination = &trace.encoder;
    else if (trace.key_name == tensor->name)
        destination = &trace.key;
    else if (trace.value_name == tensor->name)
        destination = &trace.value;
    if (!destination)
        return false;
    if (ask)
        return true;
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || !tensor->buffer ||
        ggml_nbytes(tensor) % sizeof(float)) {
        trace.error = std::string("CPU trace tensor unavailable: ") + tensor->name;
        return true;
    }
    destination->resize(ggml_nbytes(tensor) / sizeof(float));
    ggml_backend_tensor_get(tensor, destination->data(), 0, ggml_nbytes(tensor));
    return true;
}

bool cpu_sync(llama_context* ctx, llama_batch& batch, const int32_t* positions,
              const float* features, uint32_t count, std::vector<float>* output,
              std::string* error) {
    batch.n_tokens = static_cast<int32_t>(count);
    for (uint32_t row = 0; row < count; ++row) {
        batch.pos[row] = positions[row];
        batch.n_seq_id[row] = 1;
        batch.seq_id[row][0] = 0;
        batch.logits[row] = false;
    }
    const size_t floats = size_t(count) * DflashGpuSyncEngine::kFeatureWidth;
    std::memcpy(batch.embd, features, floats * sizeof(float));
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        if (error)
            *error = "llama_decode returned " + std::to_string(rc);
        return false;
    }
    const float* hidden = llama_get_embeddings(ctx);
    if (!hidden) {
        if (error)
            *error = "llama_get_embeddings returned null";
        return false;
    }
    output->assign(hidden, hidden + size_t(count) * DflashGpuSyncEngine::kHiddenWidth);
    return true;
}

} // namespace

dflash_gpu_sync_probe_result
run_dflash_gpu_sync_probe(const std::string& model_path, llama_context* ctx,
                          const int32_t* positions, const float* features, uint32_t count,
                          bool exercise_state, float absolute_tolerance, float relative_tolerance,
                          bool original_f32_matmul) {
    dflash_gpu_sync_probe_result result;
    std::vector<phase_result> phases;
    std::vector<std::string> intermediates;
    DflashGpuSyncEngine engine;
    bool cpu_mutated = false;
    bool gpu_initialized = false;
    bool all_passed = true;
    std::string state_cases = "not_requested";
    auto finish = [&](const std::string& stage, const std::string& message) {
        result.passed = stage == "complete" && all_passed;
        const auto diagnostics = engine.kv().diagnostics();
        std::ostringstream stream;
        stream.imbue(std::locale::classic());
        stream << "{\"passed\":" << (result.passed ? "true" : "false")
               << ",\"stage\":" << json_string(stage)
               << ",\"error\":" << (message.empty() ? "null" : json_string(message))
               << ",\"cpuContextMutated\":" << (cpu_mutated ? "true" : "false")
               << ",\"gpuBackend\":" << json_string(gpu_initialized ? "d3d12-hardware" : "none")
               << ",\"q3ActivationMode\":"
               << json_string(original_f32_matmul ? "original_f32_diagnostic" : "ggml_q8_k")
               << ",\"q4q5ValueActivationMode\":"
               << json_string(original_f32_matmul ? "original_f32_diagnostic" : "ggml_q8_k")
               << ",\"gpuSequenceValid\":" << (engine.sequence_valid() ? "true" : "false")
               << ",\"comparison\":\"encoder_f32_all_rows\""
               << ",\"kvComparison\":\"actual_cpu_q4_physical_numeric_gate\""
               << ",\"toleranceRule\":\"abs(cpu-gpu)<=absTol+relTol*max(abs(cpu),abs(gpu))\""
               << ",\"absTol\":" << json_number(absolute_tolerance)
               << ",\"relTol\":" << json_number(relative_tolerance)
               << ",\"stateCases\":" << json_string(state_cases)
               << ",\"kvPackRecords\":" << diagnostics.pack_records
               << ",\"kvAllocatedBytes\":" << diagnostics.allocated_bytes << ",\"phases\":[";
        for (size_t i = 0; i < phases.size(); ++i) {
            if (i)
                stream << ',';
            stream << phase_json(phases[i]);
        }
        stream << "],\"intermediates\":[";
        for (size_t i = 0; i < intermediates.size(); ++i) {
            if (i)
                stream << ',';
            stream << intermediates[i];
        }
        stream << "]}";
        result.json = stream.str();
        return result;
    };

    if (!ctx || model_path.empty() || !positions || !features || !count ||
        count > DflashGpuSyncEngine::kBatch || !std::isfinite(absolute_tolerance) ||
        absolute_tolerance < 0 || !std::isfinite(relative_tolerance) || relative_tolerance < 0)
        return finish("input_validation", "invalid probe arguments");
    for (uint32_t i = 0; i < count; ++i) {
        if (positions[i] < 0 || positions[i] >= static_cast<int32_t>(DflashGpuSyncEngine::kContext))
            return finish("input_validation", "position outside GPU context");
        for (uint32_t j = 0; j < i; ++j)
            if (positions[j] == positions[i])
                return finish("input_validation", "duplicate position");
    }
    owned_batch batch;
    batch.value =
        llama_batch_init(static_cast<int32_t>(count), DflashGpuSyncEngine::kFeatureWidth, 1);
    if (!batch.value.embd || !batch.value.pos || !batch.value.n_seq_id || !batch.value.seq_id ||
        !batch.value.logits)
        return finish("batch_init", "llama_batch_init failed");
    auto* memory = llama_get_memory(ctx);
    if (!memory)
        return finish("context_memory", "llama_get_memory returned null");
    llama_memory_clear(memory, false);
    cpu_mutated = true;

    std::string error;
    std::vector<float> cpu;
    const auto first_cpu_start = clock_type::now();
    if (!cpu_sync(ctx, batch.value, positions, features, count, &cpu, &error))
        return finish("cpu_initial", error);
    const double first_cpu_ms = milliseconds(first_cpu_start);
    if (!engine.initialize(model_path, &error,
                           original_f32_matmul ? DflashGpuSyncEngine::ActivationMode::OriginalF32
                                               : DflashGpuSyncEngine::ActivationMode::GgmlQ8K))
        return finish("gpu_initialize", error);
    gpu_initialized = true;
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
        const HRESULT hr = engine.device()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,
                                                                &options, sizeof(options));
        std::ostringstream metadata;
        metadata << "{\"name\":\"d3d12_fp64_support\",\"querySucceeded\":"
                 << (SUCCEEDED(hr) ? "true" : "false") << ",\"doublePrecisionFloatShaderOps\":"
                 << (SUCCEEDED(hr) && options.DoublePrecisionFloatShaderOps ? "true" : "false")
                 << '}';
        intermediates.push_back(metadata.str());
    }

    auto run_gpu_compare = [&](const std::string& name, const int32_t* rows, const float* inputs,
                               uint32_t n, bool reset_gpu, int32_t trim_to, double cpu_ms,
                               const std::vector<float>& expected,
                               std::vector<float>* captured_gpu = nullptr) -> bool {
        phase_result phase;
        phase.name = name;
        phase.tokens = n;
        phase.cpu_ms = cpu_ms;
        const auto gpu_start = clock_type::now();
        if (!engine.sync(rows, inputs, n, reset_gpu, trim_to, &error))
            return false;
        phase.gpu_ms = milliseconds(gpu_start);
        std::vector<float> actual(size_t(n) * DflashGpuSyncEngine::kHiddenWidth);
        const auto readback_start = clock_type::now();
        if (!engine.read_encoder(actual.data(), n, &error))
            return false;
        phase.readback_ms = milliseconds(readback_start);
        phase.values = compare_all(expected.data(), actual.data(), actual.size(),
                                   absolute_tolerance, relative_tolerance);
        if (captured_gpu)
            *captured_gpu = actual;
        phase.kv_validity = kv_rows_match(engine, rows, n, true);
        if (phase.values.mismatches || !phase.kv_validity)
            all_passed = false;
        phases.push_back(std::move(phase));
        return true;
    };
    std::vector<float> initial_gpu;
    if (!run_gpu_compare("initial", positions, features, count, true, -1, first_cpu_ms, cpu,
                         &initial_gpu))
        return finish("gpu_initial", error);

    if (!original_f32_matmul) {
        static_assert(sizeof(block_q8_K) == 292, "GGML Q8_K stride changed");
        constexpr size_t blocks_per_row = DflashGpuSyncEngine::kFeatureWidth / 256;
        const size_t q8_bytes = size_t(count) * blocks_per_row * sizeof(block_q8_K);
        std::vector<block_q8_K> cpu_q8(size_t(count) * blocks_per_row);
        std::vector<uint8_t> gpu_q8(q8_bytes);
        for (uint32_t row = 0; row < count; ++row)
            quantize_row_q8_K_ref(features + size_t(row) * DflashGpuSyncEngine::kFeatureWidth,
                                  cpu_q8.data() + size_t(row) * blocks_per_row,
                                  DflashGpuSyncEngine::kFeatureWidth);
        if (!engine.read_input_q8(gpu_q8.data(), count, &error))
            return finish("q8_gpu_readback", error);
        const auto* expected = reinterpret_cast<const uint8_t*>(cpu_q8.data());
        bool input_q8_numeric_pass = false;
        intermediates.push_back(q8_byte_parity_json("fc_activation_q8_k_byte_parity", expected,
                                                    gpu_q8.data(), q8_bytes, blocks_per_row,
                                                    &input_q8_numeric_pass));
        intermediates.push_back(q8_fc_scale_samples_json(features, expected, gpu_q8.data(), count));
        if (!input_q8_numeric_pass)
            all_passed = false;

        constexpr size_t encoder_blocks = DflashGpuSyncEngine::kHiddenWidth / 256;
        const size_t encoder_q8_bytes = size_t(count) * encoder_blocks * sizeof(block_q8_K);
        std::vector<block_q8_K> cpu_encoder_q8(size_t(count) * encoder_blocks);
        std::vector<block_q8_K> gpu_encoder_cpu_q8(size_t(count) * encoder_blocks);
        std::vector<uint8_t> gpu_encoder_q8(encoder_q8_bytes);
        for (uint32_t row = 0; row < count; ++row) {
            quantize_row_q8_K_ref(cpu.data() + size_t(row) * DflashGpuSyncEngine::kHiddenWidth,
                                  cpu_encoder_q8.data() + size_t(row) * encoder_blocks,
                                  DflashGpuSyncEngine::kHiddenWidth);
            quantize_row_q8_K_ref(initial_gpu.data() +
                                      size_t(row) * DflashGpuSyncEngine::kHiddenWidth,
                                  gpu_encoder_cpu_q8.data() + size_t(row) * encoder_blocks,
                                  DflashGpuSyncEngine::kHiddenWidth);
        }
        if (!engine.read_encoder_q8(gpu_encoder_q8.data(), count, &error))
            return finish("encoder_q8_gpu_readback", error);
        intermediates.push_back(
            q8_byte_parity_json("encoder_q8_cpu_encoder_vs_gpu",
                                reinterpret_cast<const uint8_t*>(cpu_encoder_q8.data()),
                                gpu_encoder_q8.data(), encoder_q8_bytes, encoder_blocks));
        bool encoder_q8_numeric_pass = false;
        intermediates.push_back(q8_byte_parity_json(
            "encoder_q8_cpu_from_gpu_encoder_vs_gpu",
            reinterpret_cast<const uint8_t*>(gpu_encoder_cpu_q8.data()), gpu_encoder_q8.data(),
            encoder_q8_bytes, encoder_blocks, &encoder_q8_numeric_pass));
        if (!encoder_q8_numeric_pass)
            all_passed = false;
    }

    DflashGpuSyncEngine::KvSnapshot kv_snapshot;
    if (!engine.read_layer_kv(positions, count, &kv_snapshot, &error))
        return finish("kv_physical_readback", error);
    const auto cpu_h128_k = hadamard_cpu_reference(kv_snapshot.pre_key, 128);
    const auto cpu_h64_v = hadamard_cpu_reference(kv_snapshot.pre_value, 64);
    intermediates.push_back(
        intermediate_json("last_layer_k_h128_cpu_matrix_vs_gpu",
                          compare_all(cpu_h128_k.data(), kv_snapshot.rotated_key.data(),
                                      cpu_h128_k.size(), absolute_tolerance, relative_tolerance)));
    intermediates.push_back(
        intermediate_json("last_layer_v_h64_cpu_matrix_vs_gpu",
                          compare_all(cpu_h64_v.data(), kv_snapshot.rotated_value.data(),
                                      cpu_h64_v.size(), absolute_tolerance, relative_tolerance)));
    intermediates.push_back(q4_physical_json("last_layer_k_q4_pack_from_gpu_hadamard",
                                             q4_gpu_stride_reference(kv_snapshot.rotated_key),
                                             kv_snapshot.packed_key));
    intermediates.push_back(q4_physical_json("last_layer_v_q4_pack_from_gpu_hadamard",
                                             q4_gpu_stride_reference(kv_snapshot.rotated_value),
                                             kv_snapshot.packed_value));
    intermediates.push_back(q4_physical_json("last_layer_k_q4_cpu_hadamard_from_gpu_pre_key",
                                             q4_gpu_stride_reference(cpu_h128_k),
                                             kv_snapshot.packed_key));
    intermediates.push_back(q4_physical_json("last_layer_v_q4_cpu_hadamard_from_gpu_pre_value",
                                             q4_gpu_stride_reference(cpu_h64_v),
                                             kv_snapshot.packed_value));

    std::vector<uint8_t> cpu_kv_bytes;
    std::string cpu_kv_geometry;
    const bool cpu_kv_read =
        cpu_kv_packed_snapshot(ctx, positions, count, &cpu_kv_bytes, &cpu_kv_geometry, &error);
    if (!cpu_kv_geometry.empty())
        intermediates.push_back(cpu_kv_geometry);
    if (!cpu_kv_read)
        return finish("cpu_kv_physical_readback", error);
    std::vector<uint8_t> gpu_kv_bytes;
    if (!engine.read_kv_packed(positions, count, &gpu_kv_bytes, &error))
        return finish("gpu_kv_physical_readback", error);
    if (gpu_kv_bytes.size() != cpu_kv_bytes.size())
        return finish("kv_physical_layout", "CPU/GPU packed KV size mismatch");
    constexpr size_t packed_row_bytes =
        (DflashGpuSyncEngine::kKvHeads * DflashGpuSyncEngine::kHeadWidth / 32) * 20;
    const size_t kind_bytes = size_t(count) * packed_row_bytes;
    for (uint32_t layer = 0; layer < DflashGpuSyncEngine::kLayers; ++layer) {
        for (uint32_t kind = 0; kind < 2; ++kind) {
            const size_t offset = (size_t(layer) * 2 + kind) * kind_bytes;
            const auto first_cpu = cpu_kv_bytes.begin() + offset;
            const auto first_gpu = gpu_kv_bytes.begin() + offset;
            const std::string name =
                "cpu_kv_layer" + std::to_string(layer) + (kind == 0 ? "_k_vs_gpu" : "_v_vs_gpu");
            intermediates.push_back(q4_cpu_gpu_rows_json(
                name, std::vector<uint8_t>(first_cpu, first_cpu + kind_bytes),
                std::vector<uint8_t>(first_gpu, first_gpu + kind_bytes), positions, count));
            if (layer == DflashGpuSyncEngine::kLayers - 1 && kind == 1)
                intermediates.push_back(q4_v_scale_boundary_json(
                    std::vector<uint8_t>(first_cpu, first_cpu + kind_bytes),
                    std::vector<uint8_t>(first_gpu, first_gpu + kind_bytes),
                    kv_snapshot.rotated_value, positions, count));
        }
    }
    const auto kv_numeric = q4_numeric_gate(cpu_kv_bytes, gpu_kv_bytes, kv_snapshot, count);
    intermediates.push_back(q4_numeric_gate_json(kv_numeric));
    if (!kv_numeric.passed)
        all_passed = false;
    if (!kv_numeric.passed) {
        // A separate CPU context proves the selected projection belongs to the
        // same execution before its F32 tensors are used as a reference.
        const auto diagnose = [&] {
            auto diagnostic_error = [&](const std::string& reason) {
                intermediates.push_back(std::string("{\"name\":\"kv_first_mismatch_stage\","
                                                    "\"diagnosticOnly\":true,\"error\":") +
                                        json_string(reason) + '}');
            };
            const auto mismatch =
                std::mismatch(cpu_kv_bytes.begin(), cpu_kv_bytes.end(), gpu_kv_bytes.begin());
            if (mismatch.first == cpu_kv_bytes.end()) {
                diagnostic_error("numeric gate failed without differing physical bytes");
                return;
            }
            const size_t index = size_t(mismatch.first - cpu_kv_bytes.begin());
            const uint32_t layer = uint32_t(index / (2 * kind_bytes));
            const uint32_t kind = uint32_t((index / kind_bytes) & 1u);
            const size_t within_kind = index % kind_bytes;
            const uint32_t row = uint32_t(within_kind / packed_row_bytes);
            const size_t block = (within_kind % packed_row_bytes) / 20;
            const size_t byte_in_block = within_kind % 20;
            DflashGpuSyncEngine::KvSnapshot selected;
            std::string diagnostic_message;
            if (!engine.read_layer_kv(positions, count, &selected, &diagnostic_message, layer)) {
                diagnostic_error("GPU selected-layer readback: " + diagnostic_message);
                return;
            }
            const auto& selected_packed = kind ? selected.packed_value : selected.packed_key;
            const size_t packed_kind_offset = (size_t(layer) * 2 + kind) * kind_bytes;
            const bool snapshot_matches_cache =
                selected_packed.size() == kind_bytes &&
                std::memcmp(selected_packed.data(), gpu_kv_bytes.data() + packed_kind_offset,
                            kind_bytes) == 0;
            const auto& selected_rotated = kind ? selected.rotated_value : selected.rotated_key;
            const size_t expected_floats =
                size_t(count) * DflashGpuSyncEngine::kKvHeads * DflashGpuSyncEngine::kHeadWidth;
            if (selected_packed.size() != kind_bytes ||
                selected.pre_key.size() != expected_floats ||
                selected.pre_value.size() != expected_floats ||
                selected.rotated_key.size() != expected_floats ||
                selected.rotated_value.size() != expected_floats) {
                diagnostic_error("GPU selected-layer snapshot shape mismatch");
                return;
            }
            const auto packed_from_gpu = q4_gpu_stride_reference(selected_rotated);
            intermediates.push_back(
                q4_physical_json("selected_layer_cpu_pack_from_gpu_hadamard_vs_gpu",
                                 packed_from_gpu, selected_packed));

            kv_cpu_trace trace;
            trace.key_name = "Kcur_injected-" + std::to_string(layer);
            trace.value_name = "Vcur_injected-" + std::to_string(layer);
            const auto* model = llama_get_model(ctx);
            if (!model) {
                diagnostic_error("CPU model unavailable for trace");
                return;
            }
            auto params = llama_context_default_params();
            const int32_t max_position = *std::max_element(positions, positions + count);
            params.n_ctx = std::min(DflashGpuSyncEngine::kContext,
                                    std::max(2048u, uint32_t(max_position) + count + 1u));
            params.n_batch = 64;
            params.n_ubatch = 64;
            params.n_seq_max = 1;
            params.n_threads = 4;
            params.n_threads_batch = 4;
            params.embeddings = true;
            params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            params.dflash_split = true;
            params.type_k = GGML_TYPE_Q4_0;
            params.type_v = GGML_TYPE_Q4_0;
            params.cb_eval = kv_cpu_trace_callback;
            params.cb_eval_user_data = &trace;
            std::unique_ptr<llama_context, decltype(&llama_free)> trace_ctx(
                llama_init_from_model(const_cast<llama_model*>(model), params), &llama_free);
            if (!trace_ctx) {
                diagnostic_error("CPU trace context init failed");
                return;
            }
            llama_set_embeddings_nextn(trace_ctx.get(), true, false);
            llama_set_causal_attn(trace_ctx.get(), false);
            owned_batch trace_batch;
            trace_batch.value = llama_batch_init(static_cast<int32_t>(count),
                                                 DflashGpuSyncEngine::kFeatureWidth, 1);
            if (!trace_batch.value.embd || !trace_batch.value.pos || !trace_batch.value.n_seq_id ||
                !trace_batch.value.seq_id || !trace_batch.value.logits) {
                diagnostic_error("CPU trace batch init failed");
                return;
            }
            std::vector<float> trace_encoder;
            if (!cpu_sync(trace_ctx.get(), trace_batch.value, positions, features, count,
                          &trace_encoder, &diagnostic_message)) {
                diagnostic_error("CPU trace SYNC: " + diagnostic_message);
                return;
            }
            std::vector<uint8_t> trace_kv_bytes;
            std::string trace_geometry;
            if (!cpu_kv_packed_snapshot(trace_ctx.get(), positions, count, &trace_kv_bytes,
                                        &trace_geometry, &diagnostic_message)) {
                diagnostic_error("CPU trace KV: " + diagnostic_message);
                return;
            }
            const bool encoder_exact =
                trace_encoder.size() == cpu.size() &&
                std::memcmp(trace_encoder.data(), cpu.data(), cpu.size() * sizeof(float)) == 0 &&
                (trace.encoder.empty() ||
                 (trace.encoder.size() == cpu.size() &&
                  std::memcmp(trace.encoder.data(), cpu.data(), cpu.size() * sizeof(float)) == 0));
            const bool kv_exact = trace_kv_bytes == cpu_kv_bytes;
            const bool trace_trusted = trace.error.empty() && encoder_exact && kv_exact &&
                                       trace.key.size() == expected_floats &&
                                       trace.value.size() == expected_floats;
            std::ostringstream details;
            details.imbue(std::locale::classic());
            details << "{\"name\":\"kv_first_mismatch_stage\",\"diagnosticOnly\":true"
                    << ",\"layer\":" << layer << ",\"kind\":" << json_string(kind ? "v" : "k")
                    << ",\"position\":" << positions[row] << ",\"row\":" << row
                    << ",\"block\":" << block << ",\"byteInBlock\":" << byte_in_block
                    << ",\"cpuByte\":" << unsigned(cpu_kv_bytes[index])
                    << ",\"gpuByte\":" << unsigned(gpu_kv_bytes[index])
                    << ",\"snapshotMatchesGpuCache\":"
                    << (snapshot_matches_cache ? "true" : "false")
                    << ",\"traceEncoderExact\":" << (encoder_exact ? "true" : "false")
                    << ",\"traceKvExact\":" << (kv_exact ? "true" : "false")
                    << ",\"traceTrusted\":" << (trace_trusted ? "true" : "false")
                    << ",\"traceKeyElements\":" << trace.key.size()
                    << ",\"traceValueElements\":" << trace.value.size() << ",\"traceError\":"
                    << (trace.error.empty() ? "null" : json_string(trace.error));
            const auto& cpu_pre = kind ? trace.value : trace.key;
            const auto& gpu_pre = kind ? selected.pre_value : selected.pre_key;
            const size_t first_float = (size_t(row) * packed_row_bytes / 20 + block) * 32;
            const size_t first_packed = size_t(row) * packed_row_bytes + block * 20;
            if (trace_trusted && cpu_pre.size() == gpu_pre.size() &&
                gpu_pre.size() == selected_rotated.size() && first_float + 32 <= cpu_pre.size() &&
                first_packed + 20 <= selected_packed.size()) {
                const auto cpu_key_rotated = fwht_cpu_reference(trace.key, 128);
                const auto cpu_value_rotated = fwht_cpu_reference(trace.value, 64);
                const auto& cpu_rotated = kind ? cpu_value_rotated : cpu_key_rotated;
                for (uint32_t projection = 0; projection < 2; ++projection) {
                    const auto& reference = projection ? trace.value : trace.key;
                    const auto& actual = projection ? selected.pre_value : selected.pre_key;
                    const auto& rotated_reference =
                        projection ? cpu_value_rotated : cpu_key_rotated;
                    const auto& rotated_actual =
                        projection ? selected.rotated_value : selected.rotated_key;
                    const std::string prefix =
                        std::string("selected_layer_") + (projection ? "v" : "k");
                    if (reference.size() == actual.size() &&
                        rotated_reference.size() == rotated_actual.size()) {
                        intermediates.push_back(intermediate_json(
                            prefix + "_pre_cpu_vs_gpu",
                            compare_all(reference.data(), actual.data(), actual.size(), 0, 0)));
                        intermediates.push_back(intermediate_json(
                            prefix + "_fwht_cpu_vs_gpu",
                            compare_all(rotated_reference.data(), rotated_actual.data(),
                                        rotated_actual.size(), 0, 0)));
                    }
                }
                const auto packed_from_cpu_fwht = q4_gpu_stride_reference(cpu_rotated);
                const std::vector<uint8_t> physical_cpu_kind(
                    cpu_kv_bytes.begin() + packed_kind_offset,
                    cpu_kv_bytes.begin() + packed_kind_offset + kind_bytes);
                intermediates.push_back(
                    q4_physical_json("selected_layer_cpu_pack_from_cpu_fwht_vs_cpu_cache",
                                     packed_from_cpu_fwht, physical_cpu_kind));
                details << ",\"cpuPre32\":";
                json_float_block(details, cpu_pre, first_float);
                details << ",\"gpuPre32\":";
                json_float_block(details, gpu_pre, first_float);
                details << ",\"cpuFwht32\":";
                json_float_block(details, cpu_rotated, first_float);
                details << ",\"cpuPacked20\":";
                json_byte_block(details, physical_cpu_kind, first_packed);
                details << ",\"gpuPacked20\":";
                json_byte_block(details, selected_packed, first_packed);
            }
            if (first_float + 32 <= selected_rotated.size()) {
                details << ",\"gpuHadamard32\":";
                json_float_block(details, selected_rotated, first_float);
            }
            if (first_packed + 20 <= packed_from_gpu.size()) {
                details << ",\"cpuPackFromGpuHadamard20\":";
                json_byte_block(details, packed_from_gpu, first_packed);
            }
            details << '}';
            intermediates.push_back(details.str());
        };
        diagnose();
    }

    // Instrument the exact boundary where the full encoder first diverges.
    // F2's synthetic FC probe does not use these recorded target features.
    fc_reference reference;
    if (!load_fc_reference(model_path, &reference, &error))
        return finish("fc_reference_load", error);
    std::vector<float> cpu_fc;
    if (!cpu_fc_matmul(reference, features, count, &cpu_fc, &error))
        return finish("fc_reference_compute", error);
    std::vector<float> gpu_fc(size_t(count) * DflashGpuSyncEngine::kHiddenWidth);
    if (!engine.read_fc(gpu_fc.data(), count, &error))
        return finish("fc_gpu_readback", error);
    const auto cpu_norm = cpu_norm_from_fc(reference, cpu_fc);
    const auto gpu_fc_cpu_norm = cpu_norm_from_fc(reference, gpu_fc);
    const size_t values = gpu_fc.size();
    intermediates.push_back(intermediate_json(
        "fc_raw_cpu_matrix_vs_gpu",
        compare_all(cpu_fc.data(), gpu_fc.data(), values, absolute_tolerance, relative_tolerance)));
    intermediates.push_back(intermediate_json(
        "llama_encoder_vs_cpu_norm_from_cpu_fc",
        compare_all(cpu.data(), cpu_norm.data(), values, absolute_tolerance, relative_tolerance)));
    intermediates.push_back(
        intermediate_json("gpu_encoder_vs_cpu_norm_from_gpu_fc",
                          compare_all(gpu_fc_cpu_norm.data(), initial_gpu.data(), values,
                                      absolute_tolerance, relative_tolerance)));
    intermediates.push_back(
        intermediate_json("llama_encoder_vs_cpu_norm_from_gpu_fc",
                          compare_all(cpu.data(), gpu_fc_cpu_norm.data(), values,
                                      absolute_tolerance, relative_tolerance)));
    {
        std::ostringstream metadata;
        metadata.imbue(std::locale::classic());
        const auto gamma_minmax =
            std::minmax_element(reference.gamma.begin(), reference.gamma.end());
        metadata << "{\"name\":\"fc_norm_parameters\",\"fcScale\":" << json_number(reference.scale)
                 << ",\"epsilon\":" << json_number(reference.epsilon)
                 << ",\"gammaMin\":" << json_number(*gamma_minmax.first)
                 << ",\"gammaMax\":" << json_number(*gamma_minmax.second) << '}';
        intermediates.push_back(metadata.str());
    }

    if (exercise_state) {
        bool ascending = true;
        for (uint32_t i = 1; i < count; ++i)
            if (positions[i] <= positions[i - 1])
                ascending = false;
        state_cases =
            ascending && count > 1 ? "trim_reinject_and_reset_reinject" : "reset_reinject_only";
        if (ascending && count > 1) {
            const uint32_t begin = count / 2;
            const uint32_t tail_count = count - begin;
            const int32_t cutoff = positions[begin];
            if (!llama_memory_seq_rm(memory, 0, cutoff, -1) ||
                !engine.trim(static_cast<uint32_t>(cutoff), &error))
                return finish("trim", "CPU or GPU suffix trim failed");
            if (!kv_rows_match(engine, positions, begin, true) ||
                !kv_rows_match(engine, positions + begin, tail_count, false))
                return finish("trim", "GPU logical KV validity incorrect after trim");
            std::vector<float> tail_cpu;
            const auto cpu_start = clock_type::now();
            if (!cpu_sync(ctx, batch.value, positions + begin,
                          features + size_t(begin) * DflashGpuSyncEngine::kFeatureWidth, tail_count,
                          &tail_cpu, &error))
                return finish("cpu_trim_reinject", error);
            const double cpu_ms = milliseconds(cpu_start);
            if (!run_gpu_compare("trim_reinject", positions + begin,
                                 features + size_t(begin) * DflashGpuSyncEngine::kFeatureWidth,
                                 tail_count, false, -1, cpu_ms, tail_cpu))
                return finish("gpu_trim_reinject", error);
        }

        llama_memory_clear(memory, false);
        engine.reset();
        if (!engine.sequence_valid())
            return finish("reset", "GPU sequence invalid");
        if (!kv_rows_match(engine, positions, count, false))
            return finish("reset", "GPU logical KV validity incorrect after reset");
        std::vector<float> reset_cpu;
        const auto cpu_start = clock_type::now();
        if (!cpu_sync(ctx, batch.value, positions, features, 1, &reset_cpu, &error))
            return finish("cpu_reset_reinject", error);
        const double cpu_ms = milliseconds(cpu_start);
        if (!run_gpu_compare("reset_reinject", positions, features, 1, false, -1, cpu_ms,
                             reset_cpu))
            return finish("gpu_reset_reinject", error);
    }
    return finish("complete", all_passed ? "" : "SYNC numeric or KV state mismatch");
}
