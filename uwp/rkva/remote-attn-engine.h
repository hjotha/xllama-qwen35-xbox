#pragma once

// RemoteAttnEngine — D3D12 executor for the RKVA remote KVarN attention
// service on Xbox Series X. Owns the compressed KV (KVarN records + F16 stage
// + exact precision tail) per (session, layer) and runs the KVarN WHT / store /
// flash-attention-over-records kernels on the console APU (RDNA2, wave32).
//
// One engine serves the single persistent RKVA host connection; calls are
// strictly serialized by the listener, so the engine uses one command queue /
// allocator / list / fence and never needs internal locking.
//
// The host sends Q/K/V F32 (post-norm, post-MRoPE, PRE-WHT) plus absolute
// positions. The engine reproduces the host KVarN semantics exactly: WHT
// rotation per the negotiated domain, Sinkhorn record compression, the F16
// stage for the incomplete group, the exact precision tail, causal masking,
// and the inverse WHT on the output. Bit-parity with the CUDA fattn-kvarn path
// is the acceptance gate (see the RKVA PROBE harness).

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class RemoteAttnEngine {
  public:
    // Geometry negotiated from RKVA HELLO. Mirrors rkva_hello on the wire.
    struct Geometry {
        uint32_t n_head = 0;         // Q heads (qwen35: 24)
        uint32_t n_head_kv = 0;      // KV heads (qwen35: 4)
        uint32_t head_dim = 0;       // 256
        uint32_t head_slices = 0;    // head_dim / 128 (2)
        uint32_t cache_bits_k = 0;   // 4 for kvarn4; 16 = F16 bring-up path
        uint32_t cache_bits_v = 0;   // 4 for kvarn4; 16 = F16 bring-up path
        uint32_t group_tokens = 128; // KVarN record group (KVAR_N_GROUP)
        uint32_t sinkhorn_iters = 0;
        uint32_t tail_tokens = 0;  // exact precision tail (0 = policy default)
        uint32_t tail_groups = 2;  // rollback insurance groups
        uint32_t stage_groups = 3; // F16 stage depth (tail_groups + 1 sink)
        uint32_t domain = 0;       // rkva_domain (AUTO/ROTATED/ORIGINAL_V)
        float kq_scale = 0.0f;     // 1/sqrt(head_dim)
        uint32_t max_ctx = 0;      // per-session position capacity
        uint32_t n_layers = 0;     // remote full-attention layer count (16)
    };

    RemoteAttnEngine();
    ~RemoteAttnEngine();
    RemoteAttnEngine(const RemoteAttnEngine&) = delete;
    RemoteAttnEngine& operator=(const RemoteAttnEngine&) = delete;

    // Create the D3D12 device, root signatures, PSOs and geometry-derived
    // scratch buffers. Session KV resources are allocated per create_session.
    // A failed initialize never leaves the engine half-enabled.
    bool initialize(const Geometry& geo, std::string* error);
    bool ready() const {
        return ready_;
    }
    const Geometry& geometry() const {
        return geo_;
    }

    // Session = one host sequence's KV context. capacity is in positions
    // (0 = geo.max_ctx). Preallocates per-layer record/stage/tail resources;
    // nothing is allocated per token afterwards.
    bool create_session(uint32_t session_id, uint32_t capacity, std::string* error,
                        bool defer_allocation = false);
    bool destroy_session(uint32_t session_id);
    bool has_session(uint32_t session_id) const;
    bool reset_session(uint32_t session_id, std::string* error);
    // Remove every stored position >= pos0 (trim and speculative rewind).
    bool trim_session(uint32_t session_id, int32_t pos0, std::string* error);

    // One layer's full attention step. Appends K/V (WHT + KVarN store into the
    // session's records/stage/tail), then computes causal attention of Q over
    // the stored KV for that layer.
    //   q:  [n_tokens][n_head][head_dim]     F32, post-RoPE, pre-WHT
    //   k:  [n_tokens][n_head_kv][head_dim]  F32, post-norm/RoPE, pre-WHT
    //   v:  [n_tokens][n_head_kv][head_dim]  F32
    //   positions: [n_tokens] absolute int32 within the session
    //   out: [n_tokens][n_head][head_dim]    F32, post inverse-WHT per domain
    // layer_abs is the absolute model layer index (qwen35 full-attn: 3,7,...,63).
    // On any GPU/recording failure after mutation the session is invalidated.
    bool attend(uint32_t session_id, uint32_t layer_abs, uint32_t n_tokens,
                const int32_t* positions, const float* q, const float* k, const float* v,
                float* out, std::string* error);

    // V2 positional overflow. Physical row i represents origin+i. F16/Q4_0/Q8_0
    // uploads preserve host cache values; no transform or requantization applies.
    bool configure_range(uint32_t session_id, uint32_t origin, uint32_t capacity,
                         std::string* error, uint32_t storage = 0);
    bool upload_range(uint32_t session_id, uint32_t layer, uint32_t start_row, uint32_t n_rows,
                      const uint16_t* k, const uint16_t* v, std::string* error);
    bool attend_range(uint32_t session_id, uint32_t layer, uint32_t n_tokens,
                      const int32_t* positions, const float* q, float* out, float* lse,
                      std::string* error);

    // Diagnostics / parity harness support.
    uint64_t allocated_bytes() const {
        return allocated_bytes_;
    }
    bool session_invalid(uint32_t session_id) const;

  private:
    struct Session; // per-session, per-layer KV resources (defined in .cpp)
    struct Buffer {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        uint64_t bytes = 0;
    };

    bool fail(std::string* error, const char* stage, HRESULT hr = S_OK);
    bool create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
                       D3D12_RESOURCE_STATES state, Buffer* out, std::string* error);
    void transition(Buffer& buffer, D3D12_RESOURCE_STATES state);
    void uav_barrier(Buffer& buffer);
    bool begin(std::string* error);
    bool submit(std::string* error);
    bool create_device(std::string* error);
    bool create_pipelines(std::string* error);
    bool ensure_scratch(uint32_t n_tokens, std::string* error);
    bool ensure_range_scratch(uint32_t n_tokens, std::string* error);
    Session* find_session(uint32_t session_id);
    const Session* find_session(uint32_t session_id) const;

    // Geometry-derived sizes computed in initialize().
    uint64_t record_bytes_k_ = 0; // bytes per K record group (128 tokens, all kv heads/slices)
    uint64_t record_bytes_v_ = 0;
    uint64_t stage_row_bytes_ = 0; // F16 stage bytes per token per layer (K or V)
    uint32_t groups_per_stream_ = 0;

    bool ready_ = false;
    Geometry geo_;

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmd_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_ = nullptr;
    UINT64 fence_value_ = 0;

    // Root signatures + PSOs for the KVarN kernels.
    Microsoft::WRL::ComPtr<ID3D12RootSignature> wht_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> store_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> attn_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> store_kvarn_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> attn_kvarn_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> split_k_reduce_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> wht_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> store_k_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> store_v_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> store_kvarn_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> attn_decode_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> attn_prefill_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> attn_kvarn_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> split_k_reduce_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> materialize_pso_;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> range_attn_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> range_reduce_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_pso_[3];
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_reduce_pso_;
    Buffer range_upload_;
    Buffer range_query_;
    Buffer range_partials_;
    Buffer range_output_;
    Buffer range_readback_;
    uint32_t range_scratch_tokens_ = 0;

    // Shared scratch (Q/K/V upload ring, output readback, WHT workspace).
    Buffer qkv_upload_;
    Buffer pos_upload_;
    Buffer out_readback_;
    Buffer wht_workspace_;
    Buffer split_accum_; // DEFAULT UAV: split-K partial accumulators
    Buffer q_rotated_;   // DEFAULT UAV: WHT(Q) input to the kvarn attention

    // Absolute model layer index -> dense KV slot (assigned on first use).
    std::map<uint32_t, uint32_t> layer_slot_;

    // Scratch is sized for this many tokens per call (grow-only); decode uses 1.
    uint32_t scratch_tokens_ = 0;

    std::vector<std::unique_ptr<Session>> sessions_;
    uint64_t allocated_bytes_ = 0;
    bool terminal_failure_ = false;
    bool unconfirmed_submission_ = false;
};
