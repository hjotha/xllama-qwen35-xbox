// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MTP (multi-token prediction) draft generation for the beellama fork.
//
// The fork's MTP is not a second model file: load_mtp on the target
// llama_model_params makes the loader pull the MTP head out of the same
// GGUF, and the draft context is then built from that same model with
// llama_init_from_model. So no extra weights are downloaded and no second
// mmap exists; what the draft context does cost is its own KV cache and its
// own compute per proposed token.
//
// The draft path differs from prompt-lookup in where the hidden state comes
// from. Prompt-lookup invents the next token from token_history. MTP instead
// reads the target's last hidden row via llama_get_embeddings_nextn_ith and
// feeds it back as llama_batch.embd, so the draft model predicts a real
// continuation. That is also why the target context must have been created
// with llama_set_embeddings_nextn(ctx_tgt, true, false).
//
// Everything here is optional and off unless the caller supplies a draft
// context, so a stock upstream pin links and runs unchanged.

#pragma once

#include "llama.h"

// MTP lives behind the fork's internal API rather than llama.h:
// llama_set_embeddings_nextn, llama_get_embeddings_nextn_ith and
// llama_get_ctx_other are declared in the submodule's src/llama-ext.h, and
// that internal directory is on neither the host nor the UWP include path.
// Redeclared here instead of including it, so no build file has to grow an
// include path into llama.cpp's private source tree. They are LLAMA_API, so
// the definitions are exported from the same library and this links.
struct llama_mtp_weights_info {
    bool managed = false;
    bool resident = false;
    size_t host_bytes = 0;
    size_t allocated_bytes = 0;
    size_t gpu_allocated_bytes = 0;
    size_t tensor_count = 0;
    uint64_t model_instance = 0;
    uint64_t model_load_count = 0;
    uint64_t main_gpu_upload_bytes = 0;
    uint64_t mtp_gpu_upload_bytes = 0;
    uint64_t mtp_reloads = 0;
    uint64_t backing_hash = 0;
};

extern "C" {
void llama_set_embeddings_nextn(struct llama_context* ctx, bool value, bool masked);
float* llama_get_embeddings_nextn_ith(struct llama_context* ctx, int32_t i);
struct llama_context* llama_get_ctx_other(struct llama_context* ctx);
struct llama_mtp_weights_info llama_model_mtp_weights_get_info(const struct llama_model* model);
}

#include <cstdint>
#include <vector>

namespace xllama {

// Knobs mirrored from common_params_speculative in the fork's common layer.
// The defaults match its draft defaults: up to 4 proposed tokens, minimum 1,
// and only keep drafting while the sampled candidate is at least this likely.
struct MtpDraftParams {
    int n_max = 4;
    int n_min = 1;
    float p_min = 0.75f;
};

// Owns the draft context and its sampler. One instance per Session.
class MtpDrafter {
  public:
    MtpDrafter() = default;
    ~MtpDrafter();

    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    // Builds the draft context from |model|, which must already have been
    // loaded with load_mtp = true. Shares the model's weights; only the KV
    // cache and compute buffers are additional. Returns false and logs on
    // failure, in which case draft() is a no-op and callers fall back to
    // single-token decoding.
    bool init(llama_model* model, llama_context* target_ctx, llama_context_params target_cparams,
              const MtpDraftParams& params, int n_embd);

    bool ready() const {
        return m_ctx != nullptr;
    }
    llama_context* ctx() const {
        return m_ctx;
    }

    // Proposed tokens for the next decode position. |last_token| is the token
    // just committed at |pos|; |h_row| is the target's hidden state for that
    // token (llama_get_embeddings_nextn_ith on the target context). Returns an
    // empty vector when MTP declines, which the caller must treat as "decode
    // one token classically" rather than as an error.
    std::vector<llama_token> draft(llama_token last_token, llama_pos pos, const float* h_row,
                                   int n_embd);

    const MtpDraftParams& params() const {
        return m_params;
    }

  private:
    llama_context* m_ctx = nullptr;
    llama_sampler* m_smpl = nullptr;
    llama_batch m_batch{};
    llama_context* m_target_ctx = nullptr;
    int m_n_embd = 0;
    MtpDraftParams m_params{};
    std::vector<float> m_pending_h;
    llama_pos m_pending_pos = -1;
    bool m_pending_valid = false;
};

} // namespace xllama