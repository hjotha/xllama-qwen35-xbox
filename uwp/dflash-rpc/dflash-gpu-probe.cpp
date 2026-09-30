#include "dflash-gpu-probe.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "xllama/d3d12_dyn.h"
#include "xllama/gpugemv.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

std::string json_string(const std::string& value) {
    std::string result = "\"";
    for (char c : value) {
        if (c == '"' || c == '\\')
            result += '\\';
        if (c >= ' ')
            result += c;
    }
    return result + '"';
}

std::string utf8(const wchar_t* value) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1)
        return {};
    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), size, nullptr, nullptr);
    result.pop_back();
    return result;
}

std::string error_json(const char* stage, HRESULT hr) {
    char code[16];
    std::snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned>(hr));
    return "{\"passed\":false,\"stage\":" + json_string(stage) +
           ",\"hresult\":" + json_string(code) + "}";
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>((values.size() - 1) * fraction);
    return values[index];
}

struct cpu_q4k_result {
    bool passed = false;
    std::string error;
    std::string json;
};

cpu_q4k_result benchmark_ggml_cpu_quant(const void* weights, size_t bytes, int n, int k,
                                        ggml_type type, int batch) {
    cpu_q4k_result result;
    using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    backend_ptr backend(ggml_backend_cpu_init(), &ggml_backend_free);
    if (!backend) {
        result.error = "CPU backend init";
        return result;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 4);
    context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}), &ggml_free);
    if (!ctx) {
        result.error = "CPU context init";
        return result;
    }
    ggml_tensor* w = ggml_new_tensor_2d(ctx.get(), type, k, n);
    ggml_tensor* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, batch);
    ggml_tensor* y = ggml_mul_mat(ctx.get(), w, x);
    ggml_cgraph* graph = ggml_new_graph(ctx.get());
    if (!w || !x || !y || !graph || ggml_nbytes(w) != bytes) {
        result.error = "CPU graph shape";
        return result;
    }
    ggml_build_forward_expand(graph, y);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()),
                      &ggml_backend_buffer_free);
    if (!buffer) {
        result.error = "CPU buffer allocation";
        return result;
    }
    std::vector<float> input(static_cast<size_t>(k) * batch);
    for (int b = 0; b < batch; ++b) {
        float* row = input.data() + static_cast<size_t>(b) * k;
        xllama::gpugemv_fill_x(row, k);
        for (int col = 0; col < k; ++col)
            row[col] *= 1.0f + b * 0.01f;
    }
    ggml_backend_tensor_set(w, weights, 0, bytes);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));

    std::string json = "[";
    bool passed = true;
    for (int repetition = 0; repetition < 3; ++repetition) {
        for (int warmup = 0; warmup < 50; ++warmup) {
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                result.error = "CPU warmup compute";
                return result;
            }
        }
        std::vector<double> times;
        times.reserve(200);
        for (int iteration = 0; iteration < 200; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            const auto status = ggml_backend_graph_compute(backend.get(), graph);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            if (status != GGML_STATUS_SUCCESS) {
                result.error = "CPU measured compute";
                return result;
            }
            times.push_back(ms);
        }
        std::vector<float> output(static_cast<size_t>(n) * batch);
        ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
        std::vector<float> reference(static_cast<size_t>(n) * batch);
        if (type == GGML_TYPE_Q4_K) {
            for (int b = 0; b < batch; ++b)
                xllama::gpugemv_cpu_ref(static_cast<const xllama::GpugemvQ4KBlock*>(weights),
                                        input.data() + static_cast<size_t>(b) * k,
                                        reference.data() + static_cast<size_t>(b) * n, n, k);
        }
        float max_abs_err = 0;
        for (int row = 0; row < n * batch; ++row) {
            if (type == GGML_TYPE_Q4_K)
                max_abs_err = std::max(max_abs_err, std::fabs(output[row] - reference[row]));
        }
        const bool finite = std::all_of(output.begin(), output.end(),
                                        [](float value) { return std::isfinite(value); });
        const bool correct =
            finite && (type != GGML_TYPE_Q4_K || max_abs_err <= xllama::kGpugemvMaxAbsErrTol);
        passed &= correct;
        if (repetition)
            json += ',';
        json += "{\"run\":" + std::to_string(repetition + 1) +
                ",\"correct\":" + (correct ? "true" : "false") +
                ",\"medianMs\":" + std::to_string(percentile(times, 0.5)) +
                ",\"p95Ms\":" + std::to_string(percentile(times, 0.95)) +
                ",\"referenceChecked\":" + (type == GGML_TYPE_Q4_K ? "true" : "false") +
                ",\"maxAbsErr\":" + std::to_string(max_abs_err) + "}";
        if (!correct)
            break;
    }
    json += "]";
    result.passed = passed;
    result.json = std::move(json);
    if (!passed)
        result.error = "CPU reference mismatch";
    return result;
}

} // namespace

dflash_gpu_probe_result run_dflash_gpu_probe() {
    dflash_gpu_probe_result result;
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        result.json = error_json("CreateDXGIFactory1", hr);
        return result;
    }
    const auto create_device = xllama::d3d12_dyn::CreateDevice();
    if (!create_device) {
        result.json = "{\"passed\":false,\"stage\":\"D3D12CreateDevice unavailable\"}";
        return result;
    }

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    DXGI_ADAPTER_DESC1 desc{};
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        adapter->GetDesc1(&desc);
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            hr = create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
            if (SUCCEEDED(hr))
                break;
        }
        device.Reset();
        adapter.Reset();
    }
    if (!device) {
        result.json = error_json("no hardware D3D12 adapter", hr);
        return result;
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options{};
    const bool options_ok = SUCCEEDED(
        device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options, sizeof(options)));
    D3D12_FEATURE_DATA_SHADER_MODEL shader_model{D3D_SHADER_MODEL_6_0};
    const bool shader_model_6 = SUCCEEDED(device->CheckFeatureSupport(
        D3D12_FEATURE_SHADER_MODEL, &shader_model, sizeof(shader_model)));
    DXGI_QUERY_VIDEO_MEMORY_INFO video{};
    ComPtr<IDXGIAdapter3> adapter3;
    const bool video_ok =
        SUCCEEDED(adapter.As(&adapter3)) &&
        SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &video));

    std::vector<xllama::GpugemvResult> rows;
    xllama::measure_gpugemv_each(5120, 5120, 3, xllama::GpugemvKernel::Wave32, &rows);
    bool passed = rows.size() == 3;
    std::string json =
        "{\"probe\":\"q4_k_gemv_5120x5120\",\"adapter\":" + json_string(utf8(desc.Description)) +
        ",\"vendorId\":" + std::to_string(desc.VendorId) +
        ",\"deviceId\":" + std::to_string(desc.DeviceId) +
        ",\"software\":false,\"featureLevelAtLeast11_0\":true" +
        ",\"shaderModel6\":" + (shader_model_6 ? "true" : "false") +
        ",\"waveOps\":" + (options_ok && options.WaveOps ? "true" : "false") +
        ",\"waveLaneMin\":" + std::to_string(options_ok ? options.WaveLaneCountMin : 0) +
        ",\"waveLaneMax\":" + std::to_string(options_ok ? options.WaveLaneCountMax : 0) +
        ",\"localMemoryBudgetBytes\":" + std::to_string(video_ok ? video.Budget : 0) +
        ",\"localMemoryUsageBytes\":" + std::to_string(video_ok ? video.CurrentUsage : 0) +
        ",\"runs\":[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (i)
            json += ',';
        passed &= xllama::gpugemv_passes_g1(row);
        json += "{\"run\":" + std::to_string(row.run_index) +
                ",\"d3d12\":" + (row.d3d12_ran ? "true" : "false") +
                ",\"checksumOk\":" + (row.checksum_ok ? "true" : "false") +
                ",\"gpuTimestamp\":" + (row.gpu_timestamp ? "true" : "false") +
                ",\"maxAbsErr\":" + std::to_string(row.max_abs_err) +
                ",\"packedGBs\":" + std::to_string(row.packed_gbs) +
                ",\"dispatchCpuGBs\":" + std::to_string(row.packed_gbs_cpu) +
                ",\"withCopyCpuGBs\":" + std::to_string(row.packed_gbs_h61) +
                ",\"error\":" + json_string(row.error_msg) + "}";
    }
    json += "],\"passed\":" + std::string(passed ? "true" : "false") + "}";
    result.passed = passed;
    result.json = std::move(json);
    return result;
}

dflash_gpu_probe_result run_dflash_gpu_real_quant(const std::string& model_path,
                                                  xllama::GpugemvQuant quant, int batch) {
    dflash_gpu_probe_result result;
    const char* tensor_name = quant == xllama::GpugemvQuant::Q3K   ? "fc.weight"
                              : quant == xllama::GpugemvQuant::Q4K ? "blk.0.ffn_down.weight"
                              : quant == xllama::GpugemvQuant::Q5K ? "blk.0.attn_v.weight"
                                                                   : nullptr;
    const ggml_type expected_type = quant == xllama::GpugemvQuant::Q3K   ? GGML_TYPE_Q3_K
                                    : quant == xllama::GpugemvQuant::Q4K ? GGML_TYPE_Q4_K
                                    : quant == xllama::GpugemvQuant::Q5K ? GGML_TYPE_Q5_K
                                                                         : GGML_TYPE_COUNT;
    const int k = quant == xllama::GpugemvQuant::Q3K   ? 25600
                  : quant == xllama::GpugemvQuant::Q4K ? 17408
                                                       : 5120;
    const int n = quant == xllama::GpugemvQuant::Q5K ? 1024 : 5120;
    const size_t block_bytes = quant == xllama::GpugemvQuant::Q3K   ? 110
                               : quant == xllama::GpugemvQuant::Q4K ? 144
                                                                    : 176;
    if (!tensor_name) {
        result.json = "{\"passed\":false,\"stage\":\"quant type\"}";
        return result;
    }
    if (batch < 1 || batch > 64) {
        result.json = "{\"passed\":false,\"stage\":\"batch range\"}";
        return result;
    }
    using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    gguf_ptr gguf(gguf_init_from_file(model_path.c_str(), {true, nullptr}), &gguf_free);
    if (!gguf) {
        result.json = "{\"passed\":false,\"stage\":\"gguf metadata\"}";
        return result;
    }
    const int64_t id = gguf_find_tensor(gguf.get(), tensor_name);
    if (id < 0 || gguf_get_tensor_type(gguf.get(), id) != expected_type) {
        result.json = "{\"passed\":false,\"stage\":\"quantized tensor lookup\"}";
        return result;
    }
    const int64_t* shape = gguf_get_tensor_ne(gguf.get(), id);
    const size_t bytes = gguf_get_tensor_size(gguf.get(), id);
    if (shape[0] != k || shape[1] != n || shape[2] != 1 ||
        bytes != static_cast<size_t>(n) * k / 256 * block_bytes) {
        result.json = "{\"passed\":false,\"stage\":\"quantized shape or size\"}";
        return result;
    }
    const size_t offset = gguf_get_data_offset(gguf.get()) + gguf_get_tensor_offset(gguf.get(), id);
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<uint64_t>(file.tellg()) < static_cast<uint64_t>(offset) + bytes) {
        result.json = "{\"passed\":false,\"stage\":\"GGUF bounds\"}";
        return result;
    }
    std::vector<uint8_t> weights(bytes);
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(weights.data()), static_cast<std::streamsize>(bytes));
    if (!file) {
        result.json = "{\"passed\":false,\"stage\":\"GGUF tensor read\"}";
        return result;
    }

    const auto cpu =
        benchmark_ggml_cpu_quant(weights.data(), weights.size(), n, k, expected_type, batch);
    if (!cpu.passed) {
        result.json = "{\"passed\":false,\"stage\":\"ggml CPU benchmark\",\"error\":" +
                      json_string(cpu.error) + "}";
        return result;
    }

    std::string json =
        "{\"tensor\":" + json_string(tensor_name) +
        ",\"type\":" + json_string(ggml_type_name(expected_type)) + ",\"k\":" + std::to_string(k) +
        ",\"n\":" + std::to_string(n) + ",\"batch\":" + std::to_string(batch) +
        ",\"weightBytes\":" + std::to_string(bytes) + ",\"ggufOffset\":" + std::to_string(offset) +
        ",\"warmup\":50,\"iterations\":200,\"cpuThreads\":4,\"cpuRepetitions\":" + cpu.json +
        ",\"repetitions\":[";
    bool passed = true;
    for (int repetition = 0; repetition < 3; ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<xllama::GpugemvResult> rows;
        xllama::measure_gpugemv_real_quant_batch_each(quant, n, k, batch, 50, 200,
                                                      xllama::GpugemvKernel::Wave32, weights.data(),
                                                      weights.size(), &rows);
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        std::vector<double> gpu_ms;
        std::vector<double> dispatch_ms;
        std::vector<double> with_copy_ms;
        float max_abs_err = 0;
        bool correct = rows.size() == 200;
        bool timestamps = true;
        std::string error;
        for (const auto& row : rows) {
            correct &= xllama::gpugemv_passes_g1(row);
            timestamps &= row.gpu_timestamp;
            max_abs_err = std::max(max_abs_err, row.max_abs_err);
            if (!row.error_msg.empty())
                error = row.error_msg;
            if (row.packed_gbs > 0)
                gpu_ms.push_back(bytes / (row.packed_gbs * 1e6));
            if (row.packed_gbs_cpu > 0)
                dispatch_ms.push_back(bytes / (row.packed_gbs_cpu * 1e6));
            if (row.packed_gbs_h61 > 0)
                with_copy_ms.push_back(bytes / (row.packed_gbs_h61 * 1e6));
        }
        correct &= gpu_ms.size() == 200 && dispatch_ms.size() == 200 && with_copy_ms.size() == 200;
        passed &= correct;
        if (repetition)
            json += ',';
        json += "{\"run\":" + std::to_string(repetition + 1) +
                ",\"correct\":" + (correct ? "true" : "false") +
                ",\"gpuTimestamp\":" + (timestamps ? "true" : "false") +
                ",\"gpuMedianMs\":" + std::to_string(percentile(gpu_ms, 0.5)) +
                ",\"gpuP95Ms\":" + std::to_string(percentile(gpu_ms, 0.95)) +
                ",\"dispatchMedianMs\":" + std::to_string(percentile(dispatch_ms, 0.5)) +
                ",\"dispatchP95Ms\":" + std::to_string(percentile(dispatch_ms, 0.95)) +
                ",\"withCopyMedianMs\":" + std::to_string(percentile(with_copy_ms, 0.5)) +
                ",\"withCopyP95Ms\":" + std::to_string(percentile(with_copy_ms, 0.95)) +
                ",\"maxAbsErr\":" + std::to_string(max_abs_err) +
                ",\"coldAndRunMs\":" + std::to_string(elapsed_ms) +
                ",\"error\":" + json_string(error) + "}";
        if (!correct)
            break;
    }
    json += "],\"passed\":" + std::string(passed ? "true" : "false") + "}";
    result.passed = passed;
    result.json = std::move(json);
    return result;
}

dflash_gpu_probe_result run_dflash_gpu_real_q4k(const std::string& model_path) {
    return run_dflash_gpu_real_quant(model_path, xllama::GpugemvQuant::Q4K);
}
