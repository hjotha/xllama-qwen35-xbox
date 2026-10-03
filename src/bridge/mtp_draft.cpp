// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MTP draft generation, mirroring common_speculative_impl_draft_mtp in the
// fork's common/speculative.cpp. Read that for the algorithm; this file keeps
// only the part a single-session frontend needs.
//
// Two behaviours from the reference are load-bearing and kept verbatim:
//
//   * The draft batch carries embeddings, not tokens, after the first row.
//     llama_batch_init(n, 0, 1) allocates only one of token/embd, so the token
//     array is reallocated separately. Getting this wrong yields a batch the
//     draft model reads as zeros.
//   * Drafting stops at the first candidate below p_min. MTP on a 4B target is
//     cheap but not free, and a low-confidence token costs a target decode to
//     reject, so drafting past it is a net loss.

#include "xllama/mtp_draft.h"

#include "xllama/platform.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace xllama {

namespace {

// Softmax probability of the drafted token, from the draft context's own logits
// at |idx|. llama_sampler_get_candidates would do this, but it is declared in
// the fork's common layer, which this frontend does not link.
float top_prob(llama_context* ctx, int32_t idx) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float* logits = llama_get_logits_ith(ctx, idx);
    if (!logits || n_vocab <= 0)
        return 0.0f;
    float max_l = logits[0];
    for (int i = 1; i < n_vocab; ++i)
        if (logits[i] > max_l)
            max_l = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i)
        sum += std::exp(static_cast<double>(logits[i] - max_l));
    // exp of the largest logit is 1 by construction, so it contributes exactly 1.
    return static_cast<float>(1.0 / sum);
}

} // namespace

MtpDrafter::~MtpDrafter() {
    if (m_smpl)
        llama_sampler_free(m_smpl);
    if (m_batch.token)
        llama_batch_free(m_batch);
    if (m_ctx)
        llama_free(m_ctx);
}

bool MtpDrafter::init(llama_model* model, llama_context* target_ctx,
                      llama_context_params target_cparams, const MtpDraftParams& params,
                      int n_embd) {
    if (m_ctx)
        return true;
    if (!model) {
        log_output("[xllama] mtp: no model, drafting disabled\n");
        return false;
    }

    const int32_t dft_n_embd = llama_model_n_embd_out(model);
    if (dft_n_embd != n_embd) {
        // The draft head's input row width has to match the target's, otherwise
        // the embeddings fed back in are silently reinterpreted.
        log_output("[xllama] mtp: draft n_embd_out=" + std::to_string(dft_n_embd) + " != target " +
                   std::to_string(n_embd) + "; drafting disabled\n");
        return false;
    }

    m_params = params;
    if (m_params.n_max < m_params.n_min)
        m_params.n_max = m_params.n_min;

    // Three deviations from the target params, each mirroring what the fork's
    // common layer does when it stands up a draft MTP context:
    //
    //  * ctx_other = the target, so the two contexts share one KV cache. The
    //    draft reads the target's committed prefix instead of asking for a
    //    second full-size cache, which is what made creation fail on a 4 GB
    //    console. The failure path degrades to single-token decoding, so the
    //    symptom was a flat A/B rather than an error.
    //  * ctx_type = LLAMA_CONTEXT_TYPE_MTP, which is what makes the graph
    //    builder emit the NextN head rather than a duplicate target decoder.
    //  * n_rs_seq = 0: the MTP draft holds no recurrent state of its own.
    //
    // mtp_reserve_enabled stays at its default false. It pre-allocates the
    // mixed-cache footprint the KVarN path wants; the fork keeps it off by
    // default so a model that merely has an MTP head does not phantom-reserve.
    llama_context_params cparams = target_cparams;
    // Read target_ctx directly, not m_target_ctx: the member is only
    // assigned after a successful init, so it is still null here.
    cparams.ctx_other = target_ctx;
    cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cparams.n_rs_seq = 0;

    // The draft never sees more than n_max + 1 tokens at a time: the committed
    // token whose hidden row is fed back, plus the deepest proposal. Inheriting
    // the target's n_batch made the compute reservation ask for a full-prompt
    // graph, and with the target already holding ~4000 MB on a 4 GB console that
    // allocation failed outright -- "failed to allocate compute pp buffers" --
    // so the draft context never came up.
    const int32_t n_draft_batch = m_params.n_max + 2;
    if (cparams.n_batch > n_draft_batch)
        cparams.n_batch = n_draft_batch;
    if (cparams.n_ubatch > n_draft_batch)
        cparams.n_ubatch = n_draft_batch;

    m_ctx = llama_init_from_model(model, cparams);
    if (!m_ctx) {
        log_output("[xllama] mtp: draft context creation failed (n_ctx=" +
                   std::to_string(cparams.n_ctx) + " n_batch=" + std::to_string(cparams.n_batch) +
                   " n_ubatch=" + std::to_string(cparams.n_ubatch) +
                   " ctx_type=" + std::to_string(static_cast<int>(cparams.ctx_type)) + " shared=" +
                   std::to_string(cparams.ctx_other != nullptr) + "); drafting disabled\n");
        return false;
    }

    m_target_ctx = target_ctx;
    m_n_embd = n_embd;
    m_pending_h.assign(static_cast<size_t>(n_embd), 0.0f);

    const int32_t n_b = static_cast<int32_t>(llama_n_batch(m_ctx));
    m_batch = llama_batch_init(n_b, /*embd=*/n_embd, /*n_seq_max=*/1);
    // llama_batch_init allocates only one of token/embd; MTP needs both.
    m_batch.token =
        static_cast<llama_token*>(std::malloc(sizeof(llama_token) * static_cast<size_t>(n_b)));
    if (m_batch.token)
        std::memset(m_batch.token, 0, sizeof(llama_token) * static_cast<size_t>(n_b));

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    m_smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(m_smpl, llama_sampler_init_top_k(10));

    // masked = false: the draft head sees the raw hidden row, not the selector
    // lattice the target produces.
    llama_set_embeddings_nextn(m_ctx, true, /*masked=*/true);

    log_output("[xllama] mtp: draft ready, n_max=" + std::to_string(m_params.n_max) +
               " p_min=" + std::to_string(m_params.p_min) +
               " dft_n_batch=" + std::to_string(cparams.n_batch) + "\n");
    // Confirms the draft context is wired for MTP output; if the target's nextn
    // buffer was sized to zero at creation the row read back would be zeros.
    log_output("[xllama] mtp: target nextn head layers=" +
               std::to_string(llama_model_n_layer_nextn(model)) + "\n");
    {
        // .managed is what the fork's own speculative bootstrap gates on: it
        // means load_mtp actually pulled the head's tensors into the model. A
        // GGUF with nextn tensors but no load_mtp leaves n_layer_nextn set and
        // the head unloaded, and then the target graph never emits an h_nextn
        // row at all -- which reads back as zeros.
        const llama_mtp_weights_info wi = llama_model_mtp_weights_get_info(model);
        log_output("[xllama] mtp: split_mtp_weights=" + std::to_string(wi.managed) +
                   " resident=" + std::to_string(wi.resident) +
                   " tensors=" + std::to_string(wi.tensor_count) + "\n");
    }
    return true;
}

std::vector<llama_token> MtpDrafter::draft(llama_token last_token, llama_pos pos,
                                           const float* h_row, int n_embd) {
    std::vector<llama_token> out;
    if (!m_ctx || !m_smpl || !h_row || n_embd != m_n_embd)
        return out;

    // The carry row must describe the token about to be predicted. A prompt
    // rewrite or a KV rollback moves pos backwards without the row changing, so
    // a stale carry would draft from the wrong prefix.
    if (m_pending_valid && m_pending_pos != pos - 1) {
        m_pending_valid = false;
        std::memset(m_pending_h.data(), 0, m_pending_h.size() * sizeof(float));
    }
    std::memcpy(m_pending_h.data(), h_row, static_cast<size_t>(n_embd) * sizeof(float));
    m_pending_pos = pos - 1;

    // First draft on a real run, logged once. Everything downstream depends on
    // this row being the target's hidden state, and a wrong width, a stale
    // pointer or a non-finite value here surfaces only as an abort inside
    // llama_decode with nothing pointing back at the draft.
    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        float mn = h_row[0], mx = h_row[0];
        bool finite = true;
        for (int i = 0; i < n_embd; ++i) {
            const float v = h_row[i];
            if (!std::isfinite(v)) {
                finite = false;
                break;
            }
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
        }
        log_output("[xllama] mtp: first draft pos=" + std::to_string(pos) + " n_embd=" +
                   std::to_string(n_embd) + " n_max=" + std::to_string(m_params.n_max) +
                   " row[min,max]=[" + std::to_string(mn) + "," + std::to_string(mx) + "]" +
                   (finite ? "" : " NON-FINITE") + "\n");
    }

    llama_sampler_reset(m_smpl);

    // Shared-memory layouts (the reference calls this is_mem_shared, detected
    // via llama_get_ctx_other) reuse one position for every draft row, which is
    // what the Gemma-family assistants require. With a private context each row
    // takes the next position.
    const bool shared = llama_get_ctx_other(m_ctx) != nullptr;

    m_batch.n_tokens = 0;
    m_batch.token[m_batch.n_tokens] = last_token;
    m_batch.pos[m_batch.n_tokens] = pos;
    m_batch.n_seq_id[m_batch.n_tokens] = 1;
    m_batch.seq_id[m_batch.n_tokens][0] = 0;
    m_batch.logits[m_batch.n_tokens] = 1;
    std::memcpy(m_batch.embd + static_cast<size_t>(m_batch.n_tokens) * n_embd, m_pending_h.data(),
                static_cast<size_t>(n_embd) * sizeof(float));
    // Index of the row the current step decodes, i.e. the row just decoded. It
    // changes every depth because the draft batch is rebuilt with the sampled
    // token appended. The fork tracks the same thing as i_last[seq_id].
    int i_last = m_batch.n_tokens - 1;

    const int32_t n_max = m_params.n_max;
    log_output("[xllama] mtp: draft batch n_tokens=" + std::to_string(m_batch.n_tokens) +
               " pos=" + std::to_string(m_batch.pos[0]) +
               " has_embd=" + std::to_string(m_batch.embd != nullptr) +
               " dft_n_batch=" + std::to_string(llama_n_batch(m_ctx)) + " dft_type=" +
               std::to_string(static_cast<int>(llama_get_ctx_other(m_ctx) ? 1 : 0)) + "\n");

    for (int depth = 0; depth < n_max; ++depth) {
        if (llama_decode(m_ctx, m_batch) != 0) {
            log_output("[xllama] mtp: draft decode failed at depth " + std::to_string(depth) +
                       "\n");
            m_batch.n_tokens = 0;
            return {};
        }
        m_batch.n_tokens = 0;

        {
            // Both the sampler and the nextn accessor take an index into the
            // batch that was just decoded, not a token position. The draft
            // context is configured masked (unlike the target), so the accessor
            // resolves the index through output_resolve_row -- and every draft
            // row carries logits=true, so the output index is the batch index.
            const llama_token cand = llama_sampler_sample(m_smpl, m_ctx, i_last);
            if (cand < 0)
                break;
            const float* h_next = llama_get_embeddings_nextn_ith(m_ctx, i_last);
            if (!h_next)
                break;

            // Only draft while the candidate stays confident enough that the
            // target is likely to accept it; a token below the threshold costs
            // a target decode to reject, which is a net loss.
            //
            // The probability is computed here from the draft context's logits
            // rather than through llama_sampler_get_candidates, which lives in
            // the fork's common layer and is not part of the public C API this
            // frontend links against.
            if (depth == 0) {
                float dmin = h_next[0], dmax = h_next[0];
                for (int i = 1; i < n_embd; ++i) {
                    if (h_next[i] < dmin)
                        dmin = h_next[i];
                    if (h_next[i] > dmax)
                        dmax = h_next[i];
                }
                log_output("[xllama] mtp: draft row[min,max]=[" + std::to_string(dmin) + "," +
                           std::to_string(dmax) + "]\n");
            }

            if (top_prob(m_ctx, i_last) < m_params.p_min) {
                // Stop drafting here. Anything already collected is still
                // verified; the caller falls back to a single-token decode when
                // the list comes back empty.
                break;
            }

            out.push_back(cand);

            if (static_cast<int>(out.size()) >= n_max)
                break;

            // Feed this row's embedding back with the token just sampled.
            const int i = m_batch.n_tokens;
            m_batch.token[i] = cand;
            m_batch.pos[i] = shared ? pos : pos + depth + 1;
            m_batch.n_seq_id[i] = 1;
            m_batch.seq_id[i][0] = 0;
            m_batch.logits[i] = 1;
            std::memcpy(m_batch.embd + static_cast<size_t>(i) * n_embd, h_next,
                        static_cast<size_t>(n_embd) * sizeof(float));
            m_batch.n_tokens = static_cast<int32_t>(i + 1);
            i_last = i;
        }
    }

    m_pending_valid = false;
    return out;
}

} // namespace xllama