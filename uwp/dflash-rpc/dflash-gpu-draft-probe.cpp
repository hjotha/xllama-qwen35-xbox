#include "dflash-gpu-draft-probe.h"

#include "dedicated_protocol.h"
#include "dflash-gpu-draft-engine.h"
#include "dflash-gpu-draft-sha256.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct CpuTrace {
    bool active = false;
    uint32_t hadamard_seen = 0;
    std::unordered_map<std::string, std::vector<float>> values;
};

bool cpu_trace_callback(ggml_tensor* tensor, bool ask, void* user_data) {
    auto& trace = *static_cast<CpuTrace*>(user_data);
    if (!trace.active || !tensor)
        return false;
    const bool hadamard = tensor->op == GGML_OP_MUL_MAT &&
                          tensor->op_params[1] == static_cast<int32_t>(GGML_HINT_SRC0_IS_HADAMARD);
    const char* name = tensor->name;
    if (hadamard) {
        static const char* stages[] = {
            "q_h128-0",
            "k_h128-0",
            "v_h64-0",
            "out_h64-0",
        };
        if (trace.hadamard_seen >= 4)
            return false;
        if (ask)
            return true;
        name = stages[trace.hadamard_seen++];
    }
    const char* wanted[] = {
        "inp_noise_embd", "noise_norm-0",     "attn_conv_in-0",   "Qcur-0",    "Kcur-0",
        "Vcur-0",         "kqv_out-0",        "attn_conv_out-0",  "ffn_inp-0", "ffn_norm-0",
        "ffn_conv_in-0",  "ffn_out-0",        "ffn_conv_out-0",   "l_out-0",   "ffn_up-0",
        "ffn_gate-0",     "ffn_swiglu-0",     "ffn_conv_coeff-0", "ffn_inp-1", "ffn_norm-1",
        "ffn_conv_in-1",  "ffn_out-1",        "ffn_conv_out-1",   "ffn_up-1",  "ffn_gate-1",
        "ffn_swiglu-1",   "ffn_conv_coeff-1",
    };
    bool selected = false;
    for (const char* item : wanted)
        if (std::strcmp(name, item) == 0) {
            selected = true;
            break;
        }
    if (!selected) {
        // Keep the detailed layer-0 trace, and track where later layers first
        // amplify the same error without copying every intermediate tensor.
        for (const char* prefix : {"kqv_out-", "attn_conv_out-", "l_out-"}) {
            const size_t length = std::strlen(prefix);
            if (std::strncmp(name, prefix, length) == 0 && name[length] >= '1' &&
                name[length] <= '4' && name[length + 1] == '\0') {
                selected = true;
                break;
            }
        }
    }
    if (!selected) {
        // The Q2 path can amplify a small accumulated activation error. Keep
        // the last layer's inputs/QKV visible so that the first divergent
        // operation is distinguishable from attention itself.
        for (const char* prefix : {"noise_norm-", "attn_conv_in-", "Qcur-", "Kcur-", "Vcur-"}) {
            const size_t length = std::strlen(prefix);
            if (std::strncmp(name, prefix, length) == 0 && name[length] == '4' &&
                name[length + 1] == '\0') {
                selected = true;
                break;
            }
        }
    }
    if (!selected) {
        // Layer 4 FFN checkpoints isolate amplification after attention.
        for (const char* prefix :
             {"ffn_inp-", "ffn_norm-", "ffn_conv_coeff-", "ffn_conv_in-", "ffn_gate-", "ffn_up-",
              "ffn_swiglu-", "ffn_out-", "ffn_conv_out-"}) {
            const size_t length = std::strlen(prefix);
            if (std::strncmp(name, prefix, length) == 0 && name[length] == '4' &&
                name[length + 1] == '\0') {
                selected = true;
                break;
            }
        }
    }
    if (!selected && !hadamard)
        return false;
    if (ask)
        return true;
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || !tensor->buffer ||
        ggml_nbytes(tensor) % sizeof(float))
        return true;
    auto& values = trace.values[name];
    values.resize(ggml_nbytes(tensor) / sizeof(float));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return true;
}

struct CapturedFrame {
    uint32_t record_index = 0;
    dflash_rpc_request request{};
    dflash_rpc_response response{};
    std::vector<uint8_t> input;
    std::vector<uint8_t> expected;
};

struct CapturedSequence {
    CapturedFrame sync;
    CapturedFrame draft1;
    CapturedFrame reinject;
    CapturedFrame draft2;
};

std::string escaped(const std::string& source) {
    std::string result;
    for (char c : source) {
        if (c == '"' || c == '\\')
            result += '\\';
        if (static_cast<unsigned char>(c) >= 32)
            result += c;
    }
    return result;
}

bool read_capture_header(std::ifstream& file, std::string* error) {
    char magic[8];
    file.read(magic, sizeof(magic));
    if (!file || std::memcmp(magic, "DFLCAP01", 8)) {
        *error = "invalid DFLCAP01 header";
        return false;
    }
    return true;
}

bool read_frame(std::ifstream& file, uint32_t index, CapturedFrame* frame, std::string* error) {
    uint32_t lengths[2];
    file.read(reinterpret_cast<char*>(lengths), sizeof(lengths));
    if (!file || lengths[0] < sizeof(dflash_rpc_request) ||
        lengths[1] < sizeof(dflash_rpc_response) || lengths[0] > 8u * 1024u * 1024u ||
        lengths[1] > 8u * 1024u * 1024u) {
        *error = "capture record length invalid";
        return false;
    }
    std::vector<uint8_t> request(lengths[0]), response(lengths[1]);
    file.read(reinterpret_cast<char*>(request.data()), request.size());
    file.read(reinterpret_cast<char*>(response.data()), response.size());
    if (!file) {
        *error = "capture record truncated";
        return false;
    }
    frame->record_index = index;
    std::memcpy(&frame->request, request.data(), sizeof(frame->request));
    std::memcpy(&frame->response, response.data(), sizeof(frame->response));
    if (frame->request.magic != DFLASH_RPC_MAGIC || frame->request.version != DFLASH_RPC_VERSION ||
        frame->request.payload_bytes != lengths[0] - sizeof(frame->request) ||
        frame->response.magic != DFLASH_RPC_MAGIC ||
        frame->response.version != DFLASH_RPC_VERSION || frame->response.status ||
        frame->response.payload_bytes != lengths[1] - sizeof(frame->response)) {
        *error = "invalid captured DFL2 frame";
        return false;
    }
    frame->input.assign(request.begin() + sizeof(frame->request), request.end());
    frame->expected.assign(response.begin() + sizeof(frame->response), response.end());
    return true;
}

bool read_sequence(const std::string& path, CapturedSequence* result, std::string* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        *error = "capture open failed";
        return false;
    }
    if (!read_capture_header(file, error))
        return false;
    uint32_t phase = 0;
    for (uint32_t index = 0; index < 64 && phase < 4; ++index) {
        CapturedFrame frame;
        if (!read_frame(file, index, &frame, error))
            return false;
        if (phase == 0 && frame.request.op == DFLASH_RPC_SYNC) {
            result->sync = std::move(frame);
            ++phase;
        } else if (phase == 1 && frame.request.op == DFLASH_RPC_DRAFT) {
            result->draft1 = std::move(frame);
            ++phase;
        } else if (phase == 2 && frame.request.op == DFLASH_RPC_SYNC_TRIM &&
                   frame.request.n_tokens > 0) {
            result->reinject = std::move(frame);
            ++phase;
        } else if (phase == 3 && frame.request.op == DFLASH_RPC_DRAFT) {
            result->draft2 = std::move(frame);
            ++phase;
        }
    }
    if (phase != 4) {
        *error = "capture lacks SYNC/DRAFT/SYNC_TRIM/DRAFT";
        return false;
    }
    return true;
}

bool read_draft_prefix(const std::string& path, uint32_t ordinal,
                       std::vector<CapturedFrame>* frames, std::string* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        *error = "capture open failed";
        return false;
    }
    if (!read_capture_header(file, error))
        return false;
    uint32_t drafts = 0;
    // The two SHA-whitelisted captures have at most 369 records.
    for (uint32_t index = 0; index < 369 && file.peek() != EOF; ++index) {
        CapturedFrame frame;
        if (!read_frame(file, index, &frame, error))
            return false;
        if (frame.request.op == DFLASH_RPC_DRAFT)
            ++drafts;
        frames->push_back(std::move(frame));
        if (drafts == ordinal)
            return true;
    }
    *error = "capture lacks requested DRAFT ordinal";
    return false;
}

bool parse_sync(const CapturedFrame& frame, uint32_t prefix, std::vector<int32_t>* positions,
                std::vector<float>* features, std::string* error) {
    constexpr size_t row_bytes = 4u + 25600u * 4u;
    if (!frame.request.n_tokens || frame.request.n_tokens > 64 ||
        frame.input.size() != prefix + row_bytes * frame.request.n_tokens) {
        *error = "captured SYNC shape invalid";
        return false;
    }
    positions->resize(frame.request.n_tokens);
    features->resize(size_t(frame.request.n_tokens) * 25600u);
    for (uint32_t i = 0; i < frame.request.n_tokens; ++i) {
        const uint8_t* row = frame.input.data() + prefix + size_t(i) * row_bytes;
        std::memcpy(&(*positions)[i], row, 4);
        std::memcpy(features->data() + size_t(i) * 25600u, row + 4, 25600u * 4u);
        if ((*positions)[i] < 0 || (*positions)[i] >= 49152) {
            *error = "captured SYNC position invalid";
            return false;
        }
    }
    return true;
}

bool parse_draft(const CapturedFrame& frame, dflash_rpc_draft_request* params,
                 std::vector<float>* embeddings, std::string* error) {
    if (frame.input.size() < sizeof(*params)) {
        *error = "captured DRAFT header truncated";
        return false;
    }
    std::memcpy(params, frame.input.data(), sizeof(*params));
    const uint32_t batch = frame.request.n_tokens;
    if (params->pos0 < 0 || params->n_max != 4 || batch != 5 ||
        frame.input.size() != sizeof(*params) + size_t(batch) * 5120u * 4u ||
        frame.expected.size() != size_t(batch) * 5120u * 4u) {
        *error = "captured DRAFT shape invalid";
        return false;
    }
    embeddings->resize(size_t(batch) * 5120u);
    std::memcpy(embeddings->data(), frame.input.data() + sizeof(*params),
                embeddings->size() * sizeof(float));
    for (uint32_t row = 2; row < batch; ++row)
        if (std::memcmp(embeddings->data() + size_t(row) * 5120u, embeddings->data() + 5120u,
                        5120u * 4u)) {
            *error = "captured MASK embeddings differ";
            return false;
        }
    return true;
}

struct CpuBatches {
    llama_batch sync;
    llama_batch draft;
    explicit CpuBatches()
        : sync(llama_batch_init(64, 25600, 1)), draft(llama_batch_init(64, 5120, 1)) {
        draft.token = static_cast<llama_token*>(std::malloc(64 * sizeof(llama_token)));
    }
    ~CpuBatches() {
        if (sync.token || sync.embd)
            llama_batch_free(sync);
        if (draft.token || draft.embd)
            llama_batch_free(draft);
    }
    CpuBatches(const CpuBatches&) = delete;
    CpuBatches& operator=(const CpuBatches&) = delete;
};

bool cpu_sync(llama_context* ctx, llama_batch& batch, const std::vector<int32_t>& positions,
              const std::vector<float>& features, std::string* error) {
    if (!batch.embd || positions.size() > 64 || features.size() != positions.size() * 25600u) {
        *error = "invalid CPU SYNC batch";
        return false;
    }
    batch.n_tokens = static_cast<int32_t>(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        batch.pos[i] = positions[i];
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    std::memcpy(batch.embd, features.data(), features.size() * sizeof(float));
    if (llama_decode(ctx, batch)) {
        *error = "CPU SYNC decode failed";
        return false;
    }
    return true;
}

bool cpu_draft(llama_context* ctx, llama_batch& batch, int32_t mask,
               dflash_rpc_draft_request params, const std::vector<float>& input, uint32_t count,
               std::vector<float>* hidden, std::string* error) {
    if (!batch.token || !batch.embd || !count || count > 8 ||
        input.size() != size_t(count) * 5120u ||
        !llama_memory_seq_rm(llama_get_memory(ctx), 0, params.pos0, -1)) {
        *error = "CPU DRAFT batch/trim failed";
        return false;
    }
    batch.n_tokens = static_cast<int32_t>(count);
    for (uint32_t i = 0; i < count; ++i) {
        batch.token[i] = i ? mask : params.id_last;
        batch.pos[i] = params.pos0 + static_cast<int32_t>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    std::memcpy(batch.embd, input.data(), input.size() * sizeof(float));
    if (llama_decode(ctx, batch)) {
        *error = "CPU DRAFT decode failed";
        return false;
    }
    const float* output = llama_get_embeddings(ctx);
    if (!output) {
        *error = "CPU DRAFT hidden unavailable";
        return false;
    }
    hidden->assign(output, output + input.size());
    return true;
}

std::vector<float> expand_embeddings(const std::vector<float>& captured, uint32_t count) {
    std::vector<float> result(size_t(count) * 5120u);
    std::memcpy(result.data(), captured.data(), 5120u * 4u);
    for (uint32_t i = 1; i < count; ++i)
        std::memcpy(result.data() + size_t(i) * 5120u, captured.data() + 5120u, 5120u * 4u);
    return result;
}

// Apply every sequence mutation from a captured prefix to the CPU reference.
// The optional GPU engine receives the same mutation during the main replay.
bool apply_capture_mutation(llama_context* ctx, CpuBatches& batches, DflashGpuSyncEngine* gpu,
                            const CapturedFrame& frame, std::string* error) {
    const auto op = frame.request.op;
    if (op == DFLASH_RPC_HELLO || op == DFLASH_RPC_MOCK)
        return true;
    if (op == DFLASH_RPC_RESET) {
        if (frame.request.n_tokens || !frame.input.empty()) {
            *error = "captured RESET shape invalid";
            return false;
        }
        llama_memory_clear(llama_get_memory(ctx), false);
        if (gpu)
            gpu->reset();
        return !gpu || gpu->sequence_valid();
    }
    if (op == DFLASH_RPC_DRAFT)
        return true; // caller compares its hidden output
    if (op == DFLASH_RPC_TRIM) {
        dflash_rpc_trim_request trim{};
        if (frame.request.n_tokens || frame.input.size() != sizeof(trim)) {
            *error = "captured TRIM shape invalid";
            return false;
        }
        std::memcpy(&trim, frame.input.data(), sizeof(trim));
        if (trim.pos0 < 0 || !llama_memory_seq_rm(llama_get_memory(ctx), 0, trim.pos0, -1) ||
            (gpu && !gpu->trim(static_cast<uint32_t>(trim.pos0), error))) {
            *error = "captured TRIM failed";
            return false;
        }
        return true;
    }
    uint32_t prefix = 0;
    dflash_rpc_sync_trim_request trim{};
    if (op == DFLASH_RPC_SYNC_TRIM) {
        if (!frame.request.n_tokens && frame.input.empty())
            return true; // feature probe
        if (frame.input.size() < sizeof(trim)) {
            *error = "captured SYNC_TRIM header invalid";
            return false;
        }
        std::memcpy(&trim, frame.input.data(), sizeof(trim));
        if ((trim.reset && trim.pos0 != 0) || trim.pos0 < -1) {
            *error = "captured SYNC_TRIM position invalid";
            return false;
        }
        prefix = sizeof(trim);
        if (trim.reset)
            llama_memory_clear(llama_get_memory(ctx), false);
        else if (trim.pos0 >= 0 && !llama_memory_seq_rm(llama_get_memory(ctx), 0, trim.pos0, -1)) {
            *error = "CPU SYNC_TRIM failed";
            return false;
        }
    } else if (op != DFLASH_RPC_SYNC) {
        *error = "unsupported captured operation";
        return false;
    }
    std::vector<int32_t> positions;
    std::vector<float> features;
    if (!parse_sync(frame, prefix, &positions, &features, error) ||
        !cpu_sync(ctx, batches.sync, positions, features, error))
        return false;
    return !gpu ||
           gpu->sync(positions.data(), features.data(), static_cast<uint32_t>(positions.size()),
                     trim.reset != 0, op == DFLASH_RPC_SYNC || trim.reset ? -1 : trim.pos0, error);
}

struct Metrics {
    double max_abs = 0;
    double max_rel = 0;
    double nmse = 0;
    double cosine = 0;
    double cpu_rms = 0;
    double gpu_rms = 0;
    uint32_t nonfinite = 0;
    uint32_t mismatches = 0;
};

Metrics compare(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    Metrics m;
    double se = 0, signal = 0, dot = 0, cpu_sq = 0, gpu_sq = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        const double a = cpu[i], b = gpu[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++m.nonfinite;
            continue;
        }
        const double error = std::abs(a - b);
        const double rel = error / std::max({std::abs(a), std::abs(b), 1.0e-8});
        m.max_abs = std::max(m.max_abs, error);
        m.max_rel = std::max(m.max_rel, rel);
        if (error > 1.0e-3 + 1.0e-3 * std::max(std::abs(a), std::abs(b)))
            ++m.mismatches;
        se += (a - b) * (a - b);
        signal += a * a;
        dot += a * b;
        cpu_sq += a * a;
        gpu_sq += b * b;
    }
    m.nmse = se / std::max(signal, 1.0e-20);
    m.cosine = dot / std::sqrt(std::max(cpu_sq * gpu_sq, 1.0e-40));
    const double count = std::max<double>(1.0, static_cast<double>(cpu.size()));
    m.cpu_rms = std::sqrt(cpu_sq / count);
    m.gpu_rms = std::sqrt(gpu_sq / count);
    return m;
}

double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

} // namespace

DflashGpuDraftProbeResult run_dflash_gpu_draft_probe(const std::string& model_path,
                                                     const std::string& capture_path,
                                                     llama_context* cpu_context,
                                                     uint32_t batch_mask,
                                                     uint32_t capture_draft_ordinal) {
    DflashGpuDraftProbeResult result;
    bool cpu_context_mutated = false;
    uint64_t capture_bytes = 0;
    std::string capture_sha256;
    auto failed = [&](const std::string& stage, const std::string& detail) {
        result.json = "{\"passed\":false,\"stage\":\"" + escaped(stage) + "\",\"error\":\"" +
                      escaped(detail) + "\",\"captureBytes\":" + std::to_string(capture_bytes) +
                      ",\"captureSha256\":\"" + capture_sha256 +
                      "\",\"cpuContextMutated\":" + (cpu_context_mutated ? "true}" : "false}");
        return result;
    };
    if (!cpu_context || model_path.empty() || capture_path.empty() ||
        (batch_mask & ~uint32_t(0x1f)) || capture_draft_ordinal > 369u)
        return failed("arguments", "invalid diagnostic arguments");
    const bool replay_capture = capture_draft_ordinal != 0;
    if (!batch_mask)
        batch_mask = 0x1f;
    std::string error;
    std::error_code file_error;
    capture_bytes = std::filesystem::file_size(std::filesystem::path(capture_path), file_error);
    if (file_error || (capture_bytes != 131419032u && capture_bytes != 2048844u))
        return failed("capture_size",
                      file_error ? "capture stat failed" : "capture size differs from reference");
    const auto capture_mtime =
        std::filesystem::last_write_time(std::filesystem::path(capture_path), file_error);
    if (file_error)
        return failed("capture_mtime", "capture mtime stat failed");
    if (!dflash_draft_sha256_file(capture_path, &capture_sha256, &error))
        return failed("capture_sha256", error);
    // Trusted CPU fixtures: original Q3 and the same requests rerecorded on Q2.
    const bool trusted_capture =
        (capture_bytes == 131419032u &&
             capture_sha256 == "b55ef2a104eb37a89beb26810a31dad7a956c2039329bd41987b0b77ce58bd1f" ||
         capture_sha256 == "5ff0aeb6e85c4deb4b473905513a2e166480cb9033deeb0589033a5aa835508e") ||
        (capture_bytes == 2048844u &&
         capture_sha256 == "5e85f0e76ee4fc99730cf9becbf733cd922c7f190ca65767ebcc19edac3c64dc");
    if (!trusted_capture)
        return failed("capture_sha256", "capture SHA-256 differs from reference");
    CapturedSequence captured;
    std::vector<CapturedFrame> prefix_frames;
    if (!(replay_capture
              ? read_draft_prefix(capture_path, capture_draft_ordinal, &prefix_frames, &error)
              : read_sequence(capture_path, &captured, &error)))
        return failed("capture", error);
    const uint64_t size_after_read =
        std::filesystem::file_size(std::filesystem::path(capture_path), file_error);
    if (file_error || size_after_read != capture_bytes ||
        std::filesystem::last_write_time(std::filesystem::path(capture_path), file_error) !=
            capture_mtime ||
        file_error)
        return failed("capture_changed", "capture changed while hashing/reading");
    std::vector<int32_t> prefill_pos, reinject_pos;
    std::vector<float> prefill_features, reinject_features;
    dflash_rpc_sync_trim_request trim{};
    dflash_rpc_draft_request first{}, second{};
    std::vector<float> first_embedding, second_embedding;
    if (!replay_capture) {
        if (!parse_sync(captured.sync, 0, &prefill_pos, &prefill_features, &error) ||
            !parse_sync(captured.reinject, sizeof(dflash_rpc_sync_trim_request), &reinject_pos,
                        &reinject_features, &error))
            return failed("capture_sync", error);
        std::memcpy(&trim, captured.reinject.input.data(), sizeof(trim));
        if (!parse_draft(captured.draft1, &first, &first_embedding, &error) ||
            !parse_draft(captured.draft2, &second, &second_embedding, &error) || first.pos0 < 0 ||
            second.pos0 != first.pos0 + 1 || trim.reset || trim.pos0 != first.pos0 ||
            reinject_pos.empty() || reinject_pos.front() != first.pos0)
            return failed("capture_draft",
                          error.empty() ? "unexpected anchor/reinject positions" : error);
    }
    const auto* model = llama_get_model(cpu_context);
    if (!model || llama_model_n_embd_out(model) != 5120 ||
        llama_model_target_layer_ids_n(model) * llama_model_n_embd(model) != 25600)
        return failed("cpu_model", "CPU model geometry mismatch");
    const int32_t mask = llama_vocab_mask(llama_model_get_vocab(model));
    const uint32_t batches[5] = {2, 3, 5, 7, 8};
    uint32_t trace_batch = replay_capture ? prefix_frames.back().request.n_tokens : 2u;
    if (!replay_capture)
        for (uint32_t i = 0; i < 5; ++i) {
            if (batch_mask & (1u << i)) {
                trace_batch = batches[i];
                break;
            }
        }
    CpuTrace cpu_trace;
    std::vector<float> trace_cpu_final;
    if (batch_mask) {
        auto trace_params = llama_context_default_params();
        trace_params.n_ctx = 2048;
        trace_params.n_batch = 64;
        trace_params.n_ubatch = 64;
        trace_params.n_seq_max = 1;
        trace_params.n_threads = 4;
        trace_params.n_threads_batch = 4;
        trace_params.embeddings = true;
        trace_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        trace_params.dflash_split = true;
        trace_params.type_k = GGML_TYPE_Q4_0;
        trace_params.type_v = GGML_TYPE_Q4_0;
        trace_params.cb_eval = cpu_trace_callback;
        trace_params.cb_eval_user_data = &cpu_trace;
        std::unique_ptr<llama_context, decltype(&llama_free)> trace_ctx(
            llama_init_from_model(const_cast<llama_model*>(model), trace_params), &llama_free);
        if (!trace_ctx)
            return failed("trace_cpu_init", "CPU trace context init failed");
        llama_set_embeddings_nextn(trace_ctx.get(), true, false);
        llama_set_causal_attn(trace_ctx.get(), false);
        CpuBatches trace_batches;
        if (!trace_batches.sync.embd || !trace_batches.draft.embd || !trace_batches.draft.token)
            return failed("trace_cpu_batches", "CPU trace batch allocation failed");
        if (replay_capture) {
            for (const auto& frame : prefix_frames) {
                if (frame.request.op != DFLASH_RPC_DRAFT) {
                    if (!apply_capture_mutation(trace_ctx.get(), trace_batches, nullptr, frame,
                                                &error))
                        return failed("trace_cpu_prefix", error);
                    continue;
                }
                dflash_rpc_draft_request params{};
                std::vector<float> embedding, hidden;
                if (!parse_draft(frame, &params, &embedding, &error))
                    return failed("trace_capture_draft", error);
                const bool selected = frame.record_index == prefix_frames.back().record_index;
                cpu_trace.active = selected;
                if (!cpu_draft(trace_ctx.get(), trace_batches.draft, mask, params,
                               expand_embeddings(embedding, frame.request.n_tokens),
                               frame.request.n_tokens, &hidden, &error))
                    return failed("trace_cpu_draft", error);
                cpu_trace.active = false;
                if (frame.expected.size() != hidden.size() * sizeof(float) ||
                    std::memcmp(frame.expected.data(), hidden.data(), frame.expected.size()))
                    return failed("trace_cpu_capture", "CPU differs at capture record " +
                                                           std::to_string(frame.record_index));
                if (selected)
                    trace_cpu_final = std::move(hidden);
            }
        } else {
            if (!cpu_sync(trace_ctx.get(), trace_batches.sync, prefill_pos, prefill_features,
                          &error))
                return failed("trace_cpu_prefill", error);
            cpu_trace.active = true;
            if (!cpu_draft(trace_ctx.get(), trace_batches.draft, mask, first,
                           expand_embeddings(first_embedding, trace_batch), trace_batch,
                           &trace_cpu_final, &error))
                return failed("trace_cpu_draft", error);
            cpu_trace.active = false;
        }
    }
    CpuBatches cpu;
    if (!cpu.sync.embd || !cpu.draft.embd || !cpu.draft.token)
        return failed("cpu_batches", "CPU batch allocation failed");
    llama_set_embeddings_nextn(cpu_context, true, false);
    llama_set_causal_attn(cpu_context, false);
    cpu_context_mutated = true;

    DflashGpuSyncEngine gpu_sync;
    if (!gpu_sync.initialize(model_path, &error))
        return failed("gpu_sync_init", error);
    DflashGpuDraftEngine gpu_draft;
    if (!gpu_draft.initialize(gpu_sync, model_path, &error))
        return failed("gpu_draft_init", error);
    std::ostringstream json;
    json << std::setprecision(9) << "{\"passed\":";
    std::ostringstream cases;
    cases << std::setprecision(9) << '[';
    bool all_pass = true;
    bool first_case = true;
    uint32_t comparisons = 0;
    bool trace_done = false;
    std::string trace_json = "null";
    uint32_t capture_case_ordinal = 0;
    uint32_t capture_case_record_index = 0;
    auto do_draft = [&](const char* stage, uint32_t batch, dflash_rpc_draft_request params,
                        const std::vector<float>& captured_embedding,
                        const CapturedFrame* expected_frame) -> bool {
        const auto input = expand_embeddings(captured_embedding, batch);
        std::vector<float> cpu_hidden, gpu_hidden(input.size());
        const auto cpu_start = clock_type::now();
        if (!cpu_draft(cpu_context, cpu.draft, mask, params, input, batch, &cpu_hidden, &error))
            return false;
        const double cpu_ms = elapsed_ms(cpu_start);
        const auto gpu_start = clock_type::now();
        const bool take_trace = !trace_done && batch == trace_batch &&
                                (replay_capture ? capture_case_ordinal == capture_draft_ordinal
                                                : std::strcmp(stage, "initial") == 0);
        std::vector<DflashGpuDraftEngine::Checkpoint> gpu_trace;
        if (!gpu_draft.draft(static_cast<uint32_t>(params.pos0), input.data(), batch - 1,
                             gpu_hidden.data(), &error, take_trace ? &gpu_trace : nullptr))
            return false;
        const double gpu_ms = elapsed_ms(gpu_start);
        if (take_trace) {
            trace_done = true;
            std::ostringstream out;
            out << std::setprecision(9) << "{\"batch\":" << trace_batch
                << ",\"cpuTraceMatchesMain\":";
            if (trace_cpu_final.size() == cpu_hidden.size()) {
                const Metrics match = compare(trace_cpu_final, cpu_hidden);
                out << (match.mismatches == 0 && match.nonfinite == 0 ? "true" : "false")
                    << ",\"cpuTraceFinalMaxAbs\":" << match.max_abs;
            } else {
                out << "false,\"cpuTraceFinalMaxAbs\":null";
            }
            out << ",\"checkpoints\":[";
            bool first_checkpoint = true;
            for (const auto& checkpoint : gpu_trace) {
                if (!first_checkpoint)
                    out << ',';
                first_checkpoint = false;
                out << "{\"name\":\"" << checkpoint.name
                    << "\",\"gpuElements\":" << checkpoint.values.size()
                    << ",\"gpuIntervalMs\":" << checkpoint.gpu_ms;
                const auto found = cpu_trace.values.find(checkpoint.name);
                if (found == cpu_trace.values.end() ||
                    found->second.size() != checkpoint.values.size()) {
                    out << ",\"cpuElements\":"
                        << (found == cpu_trace.values.end() ? 0 : found->second.size())
                        << ",\"available\":false}";
                } else {
                    const Metrics local = compare(found->second, checkpoint.values);
                    out << ",\"cpuElements\":" << found->second.size()
                        << ",\"available\":true,\"maxAbs\":" << local.max_abs
                        << ",\"nmse\":" << local.nmse << ",\"cosine\":" << local.cosine
                        << ",\"cpuRms\":" << local.cpu_rms << ",\"gpuRms\":" << local.gpu_rms
                        << ",\"mismatches\":" << local.mismatches
                        << ",\"nonfinite\":" << local.nonfinite;
                    if (checkpoint.name == "kqv_out-0" &&
                        checkpoint.values.size() == 2u * 32u * 128u) {
                        out << ",\"byTokenHead\":[";
                        bool first_head = true;
                        for (uint32_t token = 0; token < 2; ++token) {
                            for (uint32_t head = 0; head < 32; ++head) {
                                if (!first_head)
                                    out << ',';
                                first_head = false;
                                const size_t offset = (size_t(token) * 32u + head) * 128u;
                                const std::vector<float> cpu_part(found->second.begin() + offset,
                                                                  found->second.begin() + offset +
                                                                      128u);
                                const std::vector<float> gpu_part(
                                    checkpoint.values.begin() + offset,
                                    checkpoint.values.begin() + offset + 128u);
                                const Metrics part = compare(cpu_part, gpu_part);
                                out << "{\"token\":" << token << ",\"head\":" << head
                                    << ",\"maxAbs\":" << part.max_abs << ",\"nmse\":" << part.nmse
                                    << ",\"cosine\":" << part.cosine
                                    << ",\"mismatches\":" << part.mismatches << '}';
                            }
                        }
                        out << ']';
                    }
                    if (checkpoint.name == "ffn_norm-1" || checkpoint.name == "ffn_conv_coeff-1" ||
                        checkpoint.name == "ffn_norm-4" || checkpoint.name == "ffn_conv_coeff-4" ||
                        checkpoint.name == "attn_conv_in-4" || checkpoint.name == "Vcur-4") {
                        out << ",\"rawCpu\":[";
                        for (size_t i = 0; i < found->second.size(); ++i) {
                            if (i)
                                out << ',';
                            out << found->second[i];
                        }
                        out << "],\"rawGpu\":[";
                        for (size_t i = 0; i < checkpoint.values.size(); ++i) {
                            if (i)
                                out << ',';
                            out << checkpoint.values[i];
                        }
                        out << ']';
                    }
                    out << '}';
                }
            }
            out << "]}";
            trace_json = out.str();
        }
        const Metrics m = compare(cpu_hidden, gpu_hidden);
        const bool compare_capture = expected_frame != nullptr;
        const bool capture_exact =
            compare_capture &&
            expected_frame->expected.size() == cpu_hidden.size() * sizeof(float) &&
            !std::memcmp(cpu_hidden.data(), expected_frame->expected.data(),
                         expected_frame->expected.size());
        const bool within =
            m.mismatches == 0 && m.nonfinite == 0 && (!compare_capture || capture_exact);
        all_pass = all_pass && within;
        if (!first_case)
            cases << ',';
        first_case = false;
        cases << "{\"stage\":\"" << stage << "\",\"batch\":" << batch
              << (replay_capture
                      ? ",\"captureDraftOrdinal\":" + std::to_string(capture_case_ordinal) +
                            ",\"captureRecordIndex\":" + std::to_string(capture_case_record_index)
                      : "")
              << (replay_capture ? ",\"pos0\":" + std::to_string(params.pos0) : "")
              << ",\"cpuMs\":" << cpu_ms << ",\"gpuMs\":" << gpu_ms << ",\"maxAbs\":" << m.max_abs
              << ",\"maxRel\":" << m.max_rel << ",\"nmse\":" << m.nmse << ",\"cosine\":" << m.cosine
              << ",\"mismatches\":" << m.mismatches << ",\"nonfinite\":" << m.nonfinite
              << ",\"captureCpuChecked\":" << (compare_capture ? "true" : "false")
              << ",\"captureCpuExact\":"
              << (compare_capture ? (capture_exact ? "true" : "false") : "null")
              << ",\"withinProvisionalTolerance\":" << (within ? "true" : "false") << '}';
        ++comparisons;
        return true;
    };

    if (replay_capture) {
        for (const auto& frame : prefix_frames) {
            if (frame.request.op != DFLASH_RPC_DRAFT) {
                if (!apply_capture_mutation(cpu_context, cpu, &gpu_sync, frame, &error))
                    return failed("capture_prefix", error);
                continue;
            }
            dflash_rpc_draft_request params{};
            std::vector<float> embedding;
            if (!parse_draft(frame, &params, &embedding, &error))
                return failed("capture_draft", error);
            ++capture_case_ordinal;
            capture_case_record_index = frame.record_index;
            if (!do_draft("capture", frame.request.n_tokens, params, embedding, &frame))
                return failed("draft_capture", error);
        }
    } else
        for (uint32_t variant = 0; variant < 5; ++variant) {
            if (!(batch_mask & (1u << variant)))
                continue;
            const uint32_t batch = batches[variant];
            for (uint32_t scenario = 0; scenario < 2; ++scenario) {
                cpu_context_mutated = true;
                llama_memory_clear(llama_get_memory(cpu_context), false);
                gpu_sync.reset();
                if (!gpu_sync.sequence_valid() || !gpu_draft.sequence_valid())
                    return failed("reset", "GPU sequence invalidated");
                if (!cpu_sync(cpu_context, cpu.sync, prefill_pos, prefill_features, &error) ||
                    !gpu_sync.sync(prefill_pos.data(), prefill_features.data(),
                                   static_cast<uint32_t>(prefill_pos.size()), false, -1, &error))
                    return failed("prefill", error);
                const CapturedFrame* expected =
                    batch == 5 && scenario == 0 ? &captured.draft1 : nullptr;
                if (!do_draft("initial", batch, first, first_embedding, expected))
                    return failed("draft_initial", error);
                if (scenario == 0) {
                    // Partial rejection: preserve anchor row, discard the rest.
                    const int32_t keep = first.pos0 + 1;
                    if (!llama_memory_seq_rm(llama_get_memory(cpu_context), 0, keep, -1) ||
                        !gpu_sync.trim(static_cast<uint32_t>(keep), &error))
                        return failed("partial_trim", error);
                } else {
                    // Actual captured SYNC_TRIM payload re-injects target features.
                    if (!llama_memory_seq_rm(llama_get_memory(cpu_context), 0, trim.pos0, -1) ||
                        !cpu_sync(cpu_context, cpu.sync, reinject_pos, reinject_features, &error) ||
                        !gpu_sync.sync(reinject_pos.data(), reinject_features.data(),
                                       static_cast<uint32_t>(reinject_pos.size()), false, trim.pos0,
                                       &error))
                        return failed("reinject", error);
                }
                if (!do_draft(scenario == 0 ? "partial_trim_next" : "reinject_next", batch, second,
                              second_embedding, nullptr))
                    return failed("draft_next", error);
            }
        }
    cases << ']';
    result.passed = all_pass;
    json << (all_pass ? "true" : "false") << ",\"gpuBackend\":\"d3d12-hardware\""
         << ",\"captureBytes\":" << capture_bytes << ",\"captureSha256\":\"" << capture_sha256
         << '"'
         << (replay_capture ? ",\"captureDraftOrdinal\":" + std::to_string(capture_draft_ordinal) +
                                  ",\"captureRecordIndex\":" +
                                  std::to_string(prefix_frames.back().record_index)
                            : "")
         << ",\"comparison\":\"cpu_llama_decode_hidden_f32\""
         << ",\"tolerance\":\"provisional: abs<=1e-3+1e-3*max(abs(cpu),abs(gpu))\""
         << ",\"fullPipelineParityGate\":false" << ",\"barrierMode\":\""
         << (gpu_draft.barrier_mode() == DflashGpuDraftEngine::BarrierMode::Strict ? "strict"
                                                                                   : "minimal")
         << "\""
         << ",\"firstBlockTrace\":" << trace_json << ",\"comparisons\":" << comparisons
         << ",\"cases\":" << cases.str() << '}';
    result.json = json.str();
    return result;
}
