#include "dedicated.h"
#include "dedicated_protocol.h"
#include "dflash-gpu-probe.h"
#include "dflash-gpu-kv-probe.h"
#include "dflash-gpu-conv-probe.h"
#include "gpu-attention-probe.h"
#include "dflash-gpu-sync-probe.h"
#include "dflash-gpu-draft-probe.h"
#include "dflash-gpu-engine.h"
#include "dflash-gpu-draft-engine.h"
#include "dflash-dml.h"

#include "llama.h"
#include "llama-ext.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using microseconds = std::chrono::microseconds;

uint64_t elapsed_us(clock_type::time_point start) {
    return (uint64_t) std::chrono::duration_cast<microseconds>(clock_type::now() - start).count();
}

void event(const std::string & path, const std::string & message) {
    std::ofstream(path, std::ios::app) << message << '\n';
}

void core_log(ggml_log_level level, const char * message, void * user_data) {
    const auto * path = static_cast<const std::string *>(user_data);
    std::ofstream(*path, std::ios::app) << static_cast<int>(level) << ':' << message;
}

bool recv_all(SOCKET sock, void * dst, size_t size) {
    char * data = static_cast<char *>(dst);
    while (size) {
        const int n = recv(sock, data, (int) std::min(size, (size_t) INT_MAX), 0);
        if (n <= 0) return false;
        data += n;
        size -= n;
    }
    return true;
}

bool send_all(SOCKET sock, const void * src, size_t size) {
    const char * data = static_cast<const char *>(src);
    while (size) {
        const int n = send(sock, data, (int) std::min(size, (size_t) INT_MAX), 0);
        if (n <= 0) return false;
        data += n;
        size -= n;
    }
    return true;
}

struct dflash_state {
    struct recorded_draft {
        int32_t pos0;
        int32_t id_last;
        int32_t n_max;
        std::vector<uint8_t> output;
    };

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch inject{};
    llama_batch draft{};
    uint32_t enc_width = 0;
    uint32_t dec_width = 0;
    uint32_t top_k = 0;
    int32_t mask = 0;
    uint64_t model_bytes = 0;
    std::string model_path;
    std::vector<recorded_draft> recorded;
    size_t replay_index = 0;
    bool mock = false;
    std::string backend = "cpu";
    std::string backend_reason;
    std::unique_ptr<DflashGpuSyncEngine> gpu_sync;
    std::unique_ptr<DflashGpuDraftEngine> gpu_draft;

    bool gpu_selected() const {
        return gpu_sync && gpu_draft;
    }

    ~dflash_state() {
        if (inject.token || inject.embd) llama_batch_free(inject);
        if (draft.token || draft.embd) llama_batch_free(draft);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    bool load(const std::string & model_path, const std::string & log_path) {
        this->model_path = model_path;
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        if (!file) {
            event(log_path, "{\"phase\":\"load\",\"status\":\"model_missing\"}");
            return false;
        }
        model_bytes = (uint64_t) file.tellg();
        file.close();

        auto mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        mparams.load_mode = LLAMA_LOAD_MODE_NONE;
        const auto start = clock_type::now();
        model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model) {
            event(log_path, "{\"phase\":\"load\",\"status\":\"model_failed\"}");
            return false;
        }
        top_k = (uint32_t) llama_model_dflash_selector_top_k(model);
        enc_width = llama_model_target_layer_ids_n(model) * (uint32_t) llama_model_n_embd(model);
        dec_width = (uint32_t) llama_model_n_embd_out(model);
        mask = llama_vocab_mask(llama_model_get_vocab(model));
        if (top_k != 16 || enc_width != 25600 || dec_width < top_k * (top_k + 1)) {
            event(log_path, "{\"phase\":\"load\",\"status\":\"unexpected_model_shape\"}");
            return false;
        }

        auto cparams = llama_context_default_params();
        cparams.n_ctx = 49152;
        cparams.n_batch = 64;
        cparams.n_ubatch = 64;
        cparams.n_seq_max = 1;
        cparams.n_threads = 4;
        cparams.n_threads_batch = 4;
        cparams.embeddings = true;
        cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cparams.dflash_split = true;
        cparams.type_k = GGML_TYPE_Q4_0;
        cparams.type_v = GGML_TYPE_Q4_0;
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            event(log_path, "{\"phase\":\"load\",\"status\":\"context_failed\"}");
            return false;
        }
        llama_set_embeddings_nextn(ctx, true, false);
        llama_set_causal_attn(ctx, false);
        inject = llama_batch_init(64, enc_width, 1);
        draft = llama_batch_init(64, dec_width, 1);
        draft.token = static_cast<llama_token *>(std::malloc(64 * sizeof(llama_token)));
        if (!draft.token) {
            event(log_path, "{\"phase\":\"load\",\"status\":\"draft_batch_allocation_failed\"}");
            return false;
        }
        event(log_path, "{\"phase\":\"load\",\"status\":\"ready\",\"modelBytes\":" +
              std::to_string(model_bytes) + ",\"elapsedUs\":" + std::to_string(elapsed_us(start)) +
              ",\"contextTokens\":49152,\"kv\":\"q4_0\"}");

        // Optional executor selection: LocalState\dflash-backend.txt with
        // "cpu" (default) or "d3d12". GPU selection is opt-in and falls back
        // to CPU before any sequence starts when initialization fails.
        const size_t separator = log_path.find_last_of("\\/");
        const std::string folder = separator == std::string::npos ? std::string()
            : log_path.substr(0, separator + 1);
        std::string requested = "cpu";
        {
            std::ifstream backend_file(folder + "dflash-backend.txt");
            if (backend_file) {
                std::getline(backend_file, requested);
                while (!requested.empty() &&
                       (requested.back() == '\r' || requested.back() == '\n' ||
                        requested.back() == ' ')) requested.pop_back();
            }
        }
        if (requested == "d3d12") {
            const auto gpu_start = clock_type::now();
            std::string gpu_error;
            auto sync = std::make_unique<DflashGpuSyncEngine>();
            auto draft = std::make_unique<DflashGpuDraftEngine>();
            if (!sync->initialize(model_path, &gpu_error)) {
                backend_reason = "sync_init_failed: " + gpu_error;
            } else if (!draft->initialize(*sync, model_path, &gpu_error)) {
                backend_reason = "draft_init_failed: " + gpu_error;
            } else {
                backend = "d3d12";
                gpu_sync = std::move(sync);
                gpu_draft = std::move(draft);
            }
            std::string message = "{\"phase\":\"backend\",\"requested\":\"d3d12\",\"effective\":\"" +
                backend + "\",\"elapsedUs\":" + std::to_string(elapsed_us(gpu_start));
            if (!backend_reason.empty()) message += ",\"reason\":\"" + backend_reason + "\"";
            event(log_path, message + "}");
        } else {
            event(log_path, "{\"phase\":\"backend\",\"requested\":\"cpu\",\"effective\":\"cpu\"}");
        }
        return true;
    }
};

int handle_request(dflash_state & state, const dflash_rpc_request & req,
                   const std::vector<uint8_t> & input, std::vector<uint8_t> & output,
                   uint64_t & prepare_us, uint64_t & compute_us, uint64_t & kv_us) {
    const auto prepare_start = clock_type::now();
    if (req.op == DFLASH_RPC_MOCK) {
        if (!input.empty() || req.n_tokens > 1) return -1;
        state.mock = req.n_tokens == 1;
        state.replay_index = 0;
        if (!state.mock) state.recorded.clear();
        prepare_us = elapsed_us(prepare_start);
        return state.mock && state.recorded.empty() ? -5 : 0;
    }
    if (req.op == DFLASH_RPC_HELLO) {
        if (req.n_tokens || !input.empty()) return -1;
        const dflash_rpc_hello hello{49152, state.enc_width, state.dec_width, state.top_k,
                                     state.mask, 4, 4, state.model_bytes};
        output.resize(sizeof(hello));
        std::memcpy(output.data(), &hello, sizeof(hello));
        prepare_us = elapsed_us(prepare_start);
        return 0;
    }
    if (req.op == DFLASH_RPC_GPU_PROBE) {
        if (req.n_tokens || !input.empty()) return -1;
        prepare_us = elapsed_us(prepare_start);
        const auto compute_start = clock_type::now();
        const auto probe = run_dflash_gpu_probe();
        compute_us = elapsed_us(compute_start);
        output.assign(probe.json.begin(), probe.json.end());
        return probe.passed ? 0 : -6;
    }
    if (req.op == DFLASH_RPC_GPU_REAL_Q4K || req.op == DFLASH_RPC_GPU_REAL_Q3K ||
        req.op == DFLASH_RPC_GPU_REAL_Q5K) {
        if (req.n_tokens) return -1;
        uint32_t batch = 1;
        if (input.size() == sizeof(batch)) {
            std::memcpy(&batch, input.data(), sizeof(batch));
        } else if (!input.empty()) {
            return -1;
        }
        if (batch < 1 || batch > 64) return -1;
        prepare_us = elapsed_us(prepare_start);
        const auto compute_start = clock_type::now();
        const auto quant = req.op == DFLASH_RPC_GPU_REAL_Q3K ? xllama::GpugemvQuant::Q3K :
                           req.op == DFLASH_RPC_GPU_REAL_Q5K ? xllama::GpugemvQuant::Q5K :
                           xllama::GpugemvQuant::Q4K;
        const auto probe = run_dflash_gpu_real_quant(state.model_path, quant, static_cast<int>(batch));
        compute_us = elapsed_us(compute_start);
        output.assign(probe.json.begin(), probe.json.end());
        return probe.passed ? 0 : -6;
    }
    if (req.op == DFLASH_RPC_DML_REAL_Q4K) {
        if (req.n_tokens || !input.empty()) return -1;
        prepare_us = elapsed_us(prepare_start);
        const auto compute_start = clock_type::now();
        const auto probe = run_dflash_dml_real_q4k(state.model_path);
        compute_us = elapsed_us(compute_start);
        output.assign(probe.json.begin(), probe.json.end());
        return probe.passed ? 0 : -6;
    }
    if (req.op == DFLASH_RPC_GPU_KV_Q40) {
        if (req.n_tokens || !input.empty()) return -1;
        prepare_us = elapsed_us(prepare_start);
        const auto compute_start = clock_type::now();
        const auto probe = run_dflash_gpu_kv_probe();
        compute_us = elapsed_us(compute_start);
        output.assign(probe.json.begin(), probe.json.end());
        return probe.passed ? 0 : -6;
    }
    if (req.op == DFLASH_RPC_GPU_CONV || req.op == DFLASH_RPC_GPU_ATTENTION) {
        if (req.n_tokens || !input.empty()) return -1;
        prepare_us = elapsed_us(prepare_start);
        const auto compute_start = clock_type::now();
        if (req.op == DFLASH_RPC_GPU_CONV) {
            const auto probe = run_dflash_gpu_conv_probe(state.model_path);
            compute_us = elapsed_us(compute_start);
            output.assign(probe.json.begin(), probe.json.end());
            return probe.passed ? 0 : -6;
        }
        const auto probe = run_dflash_gpu_attention_probe();
        compute_us = elapsed_us(compute_start);
        output.assign(probe.json.begin(), probe.json.end());
        return probe.passed ? 0 : -6;
    }
    if (req.op == DFLASH_RPC_GPU_SYNC) {
        const size_t row_bytes = sizeof(int32_t) +
                                 size_t(state.enc_width) * sizeof(float);
        if (state.mock || !req.n_tokens || req.n_tokens > 64 ||
            input.size() != size_t(req.n_tokens) * row_bytes) return -1;
        for (uint32_t i = 0; i < req.n_tokens; ++i) {
            const uint8_t * row = input.data() + size_t(i) * row_bytes;
            std::memcpy(&state.inject.pos[i], row, sizeof(int32_t));
            std::memcpy(state.inject.embd + size_t(i) * state.enc_width,
                        row + sizeof(int32_t), row_bytes - sizeof(int32_t));
        }
        prepare_us = elapsed_us(prepare_start);
        const auto compute_start = clock_type::now();
        const auto probe = run_dflash_gpu_sync_probe(state.model_path, state.ctx,
            state.inject.pos, state.inject.embd, req.n_tokens);
        compute_us = elapsed_us(compute_start);
        output.assign(probe.json.begin(), probe.json.end());
        return probe.passed ? 0 : -6;
    }
    if (req.op == DFLASH_RPC_GPU_DRAFT_PROBE) {
        if (state.mock || req.n_tokens ||
            (input.size() != 0 && input.size() != sizeof(uint32_t))) return -1;
        uint32_t batch_mask = 0;
        if (!input.empty()) std::memcpy(&batch_mask, input.data(), sizeof(batch_mask));
        if (batch_mask & ~uint32_t(0x1f)) return -1;
        const size_t last_separator = state.model_path.find_last_of("\\/");
        if (last_separator == std::string::npos) return -1;
        const std::string capture_path = state.model_path.substr(0, last_separator + 1) +
                                         "reference.cap";
        prepare_us = elapsed_us(prepare_start);
        const auto compute_start = clock_type::now();
        const auto probe = run_dflash_gpu_draft_probe(state.model_path,
            capture_path, state.ctx, batch_mask);
        compute_us = elapsed_us(compute_start);
        output.assign(probe.json.begin(), probe.json.end());
        return probe.passed ? 0 : -6;
    }
    if (req.op == DFLASH_RPC_RESET) {
        if (req.n_tokens || !input.empty()) return -1;
        llama_memory_clear(llama_get_memory(state.ctx), false);
        if (state.gpu_selected()) state.gpu_sync->reset();
        prepare_us = elapsed_us(prepare_start);
        return 0;
    }
    if (req.op == DFLASH_RPC_SYNC || req.op == DFLASH_RPC_SYNC_TRIM) {
        const bool fused = req.op == DFLASH_RPC_SYNC_TRIM;
        if (fused && req.n_tokens == 0 && input.empty()) {
            prepare_us = elapsed_us(prepare_start);
            return 0; // Capability probe: do not mutate sequence or replay state.
        }
        const size_t prefix_bytes = fused ? sizeof(dflash_rpc_sync_trim_request) : 0;
        const size_t row_bytes = sizeof(int32_t) + (size_t) state.enc_width * sizeof(float);
        if (!req.n_tokens || req.n_tokens > 64 ||
                input.size() != prefix_bytes + (size_t) req.n_tokens * row_bytes) return -1;
        dflash_rpc_sync_trim_request trim{};
        if (fused) {
            std::memcpy(&trim, input.data(), sizeof(trim));
            if (trim.pos0 < 0 || trim.reset > 1 || (trim.reset && trim.pos0 != 0)) return -1;
        }
        state.inject.n_tokens = (int32_t) req.n_tokens;
        for (uint32_t i = 0; i < req.n_tokens; ++i) {
            const uint8_t * row = input.data() + prefix_bytes + (size_t) i * row_bytes;
            std::memcpy(&state.inject.pos[i], row, sizeof(int32_t));
            std::memcpy(state.inject.embd + (size_t) i * state.enc_width, row + sizeof(int32_t), row_bytes - sizeof(int32_t));
            state.inject.n_seq_id[i] = 1;
            state.inject.seq_id[i][0] = 0;
            state.inject.logits[i] = false;
        }
        if (fused && !state.mock && !state.gpu_selected()) {
            const auto kv_start = clock_type::now();
            if (trim.reset) {
                llama_memory_clear(llama_get_memory(state.ctx), false);
            } else if (!llama_memory_seq_rm(llama_get_memory(state.ctx), 0, trim.pos0, -1)) {
                return -4;
            }
            kv_us = elapsed_us(kv_start);
        }
        prepare_us = elapsed_us(prepare_start);
        if (state.mock) return 0;
        if (state.gpu_selected()) {
            std::vector<int32_t> positions(req.n_tokens);
            std::vector<float> features((size_t) req.n_tokens * state.enc_width);
            for (uint32_t i = 0; i < req.n_tokens; ++i) {
                const uint8_t * row = input.data() + prefix_bytes + (size_t) i * row_bytes;
                std::memcpy(&positions[i], row, sizeof(int32_t));
                std::memcpy(features.data() + (size_t) i * state.enc_width,
                            row + sizeof(int32_t),
                            (size_t) state.enc_width * sizeof(float));
            }
            const bool gpu_reset = fused && trim.reset;
            const int32_t gpu_trim_to = fused && !trim.reset ? trim.pos0 : -1;
            const auto compute_start = clock_type::now();
            std::string gpu_error;
            const bool ok = state.gpu_sync->sync(positions.data(), features.data(),
                                                 req.n_tokens, gpu_reset, gpu_trim_to,
                                                 &gpu_error);
            compute_us = elapsed_us(compute_start);
            return ok ? 0 : -2;
        }
        const auto compute_start = clock_type::now();
        const int rc = llama_decode(state.ctx, state.inject);
        compute_us = elapsed_us(compute_start);
        return rc ? -2 : 0;
    }
    if (req.op == DFLASH_RPC_DRAFT) {
        if (input.size() < sizeof(dflash_rpc_draft_request)) return -1;
        dflash_rpc_draft_request params;
        std::memcpy(&params, input.data(), sizeof(params));
        if (params.pos0 < 0 || params.n_max < 1 || params.n_max > 7 ||
            req.n_tokens != (uint32_t) (params.n_max + 1) ||
            input.size() != sizeof(params) + (size_t) req.n_tokens * state.dec_width * sizeof(float)) return -1;
        state.draft.n_tokens = params.n_max + 1;
        for (int32_t i = 0; i <= params.n_max; ++i) {
            state.draft.token[i] = i == 0 ? params.id_last : state.mask;
            state.draft.pos[i] = params.pos0 + i;
            state.draft.n_seq_id[i] = 1;
            state.draft.seq_id[i][0] = 0;
            state.draft.logits[i] = false;
        }
        std::memcpy(state.draft.embd, input.data() + sizeof(params),
                    (size_t) req.n_tokens * state.dec_width * sizeof(float));
        if (state.mock) {
            if (state.replay_index >= state.recorded.size()) return -5;
            const auto & saved = state.recorded[state.replay_index++];
            if (saved.pos0 != params.pos0 || saved.id_last != params.id_last ||
                saved.n_max != params.n_max) return -5;
            output = saved.output;
            prepare_us = elapsed_us(prepare_start);
            return 0;
        }
        if (state.gpu_selected()) {
            if (!state.gpu_sync->sequence_valid() || !state.gpu_draft->sequence_valid())
                return -4;
            const float * embeddings =
                reinterpret_cast<const float *>(input.data() + sizeof(params));
            std::vector<float> hidden((size_t) req.n_tokens * state.dec_width);
            prepare_us = elapsed_us(prepare_start);
            const auto compute_start = clock_type::now();
            std::string gpu_error;
            const bool ok = state.gpu_draft->draft((uint32_t) params.pos0, embeddings,
                                                   (uint32_t) params.n_max, hidden.data(),
                                                   &gpu_error, nullptr);
            compute_us = elapsed_us(compute_start);
            if (!ok) return -2;
            output.resize(hidden.size() * sizeof(float));
            std::memcpy(output.data(), hidden.data(), output.size());
            state.recorded.push_back({params.pos0, params.id_last, params.n_max, output});
            return 0;
        }
        // SYNC can include the anchor position. Replace that target-feature KV
        // with the draft anchor instead of decoding two tokens at the same pos.
        if (!llama_memory_seq_rm(llama_get_memory(state.ctx), 0, params.pos0, -1)) return -4;
        prepare_us = elapsed_us(prepare_start);
        const auto compute_start = clock_type::now();
        const int rc = llama_decode(state.ctx, state.draft);
        compute_us = elapsed_us(compute_start);
        if (rc) return -2;
        const float * hidden = llama_get_embeddings(state.ctx);
        if (!hidden) return -3;
        output.resize((size_t) req.n_tokens * state.dec_width * sizeof(float));
        std::memcpy(output.data(), hidden, output.size());
        state.recorded.push_back({params.pos0, params.id_last, params.n_max, output});
        return 0;
    }
    if (req.op == DFLASH_RPC_TRIM) {
        if (req.n_tokens || input.size() != sizeof(dflash_rpc_trim_request)) return -1;
        dflash_rpc_trim_request params;
        std::memcpy(&params, input.data(), sizeof(params));
        if (params.pos0 < 0) return -1;
        if (state.mock) {
            prepare_us = elapsed_us(prepare_start);
            return 0;
        }
        if (state.gpu_selected()) {
            const auto kv_start = clock_type::now();
            std::string gpu_error;
            const bool ok = state.gpu_sync->trim((uint32_t) params.pos0, &gpu_error);
            kv_us = elapsed_us(kv_start);
            prepare_us = elapsed_us(prepare_start);
            return ok ? 0 : -4;
        }
        const auto kv_start = clock_type::now();
        const bool ok = llama_memory_seq_rm(llama_get_memory(state.ctx), 0, params.pos0, -1);
        kv_us = elapsed_us(kv_start);
        prepare_us = elapsed_us(prepare_start);
        if (!ok) return -4;
        return 0;
    }
    return -1;
}

void client_loop(SOCKET sock, dflash_state & state, const std::string & log_path) {
    llama_memory_clear(llama_get_memory(state.ctx), false);
    uint64_t previous_response_us = 0;
    for (;;) {
        const auto start = clock_type::now();
        dflash_rpc_request req{};
        if (!recv_all(sock, &req, sizeof(req))) break;
        const uint64_t idle_us = elapsed_us(start);
        if (req.magic != DFLASH_RPC_MAGIC || req.version != DFLASH_RPC_VERSION ||
                req.payload_bytes > sizeof(dflash_rpc_sync_trim_request) + 64u * (4u + 25600u * 4u)) break;
        std::vector<uint8_t> input(req.payload_bytes);
        const auto receive_start = clock_type::now();
        if (!recv_all(sock, input.data(), input.size())) break;
        const uint64_t receive_us = elapsed_us(receive_start);
        std::vector<uint8_t> output;
        uint64_t prepare_us = 0;
        uint64_t compute_us = 0;
        uint64_t kv_us = 0;
        const auto handle_start = clock_type::now();
        const int rc = handle_request(state, req, input, output, prepare_us, compute_us, kv_us);
        const uint64_t handle_us = elapsed_us(handle_start);
        const uint64_t post_us = handle_us > prepare_us + compute_us ? handle_us - prepare_us - compute_us : 0;
        const dflash_rpc_response rsp{DFLASH_RPC_MAGIC, DFLASH_RPC_VERSION, req.op,
                                      req.cycle_id, rc, (uint32_t) output.size(), receive_us,
                                      prepare_us, compute_us, previous_response_us};
        const auto response_start = clock_type::now();
        if (!send_all(sock, &rsp, sizeof(rsp)) || !send_all(sock, output.data(), output.size())) break;
        previous_response_us = elapsed_us(response_start);
        event(log_path, "{\"phase\":\"request\",\"cycleId\":" + std::to_string(req.cycle_id) +
              ",\"op\":" + std::to_string(req.op) + ",\"nTokens\":" + std::to_string(req.n_tokens) +
              ",\"payloadBytes\":" + std::to_string(req.payload_bytes) +
              ",\"responseBytes\":" + std::to_string(output.size()) +
              ",\"status\":" + std::to_string(rc) +
              ",\"mock\":" + std::string(state.mock ? "true" : "false") +
              ",\"idleUs\":" + std::to_string(idle_us) +
              ",\"receiveUs\":" + std::to_string(receive_us) +
              ",\"prepareUs\":" + std::to_string(prepare_us) +
              ",\"computeUs\":" + std::to_string(compute_us) +
              ",\"kvUs\":" + std::to_string(kv_us) +
              ",\"postUs\":" + std::to_string(post_us) +
              ",\"responseUs\":" + std::to_string(previous_response_us) + "}");
        if (req.op == DFLASH_RPC_GPU_DRAFT_PROBE) break;
    }
}

} // namespace

void serve_dflash_dedicated(const std::string & local_folder) {
    const std::string log_path = local_folder + "\\dflash-dedicated.jsonl";
    const std::string model_path = local_folder + "\\dflash-q3.gguf";
    static std::string core_log_path = local_folder + "\\dflash-core.log";
    llama_log_set(core_log, &core_log_path);
    dflash_state state;
    if (!state.load(model_path, log_path)) return;

    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa)) {
        event(log_path, "{\"phase\":\"listen\",\"status\":\"wsa_failed\"}");
        return;
    }
    SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listener == INVALID_SOCKET) return;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(50053);
    if (bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == SOCKET_ERROR ||
        listen(listener, 1) == SOCKET_ERROR) {
        event(log_path, "{\"phase\":\"listen\",\"status\":\"bind_failed\"}");
        closesocket(listener);
        return;
    }
    event(log_path, "{\"phase\":\"listen\",\"status\":\"ready\",\"port\":50053}");
    for (;;) {
        SOCKET client = accept(listener, nullptr, nullptr);
        if (client == INVALID_SOCKET) break;
        const int one = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&one), sizeof(one));
        client_loop(client, state, log_path);
        closesocket(client);
    }
    closesocket(listener);
    WSACleanup();
}
