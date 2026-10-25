#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

// The dedicated transport uses little-endian x64 on both .57 and Xbox.
// One TCP connection owns one persistent DFlash sequence.
static constexpr uint32_t DFLASH_RPC_MAGIC = 0x324c4644; // "DFL2"
static constexpr uint16_t DFLASH_RPC_VERSION = 2;

enum dflash_rpc_op : uint16_t {
    DFLASH_RPC_HELLO = 1,
    DFLASH_RPC_RESET = 2,
    DFLASH_RPC_SYNC = 3,
    DFLASH_RPC_DRAFT = 4,
    DFLASH_RPC_TRIM = 5,
    DFLASH_RPC_MOCK = 6,             // benchmark only: n_tokens=0 records, 1 replays
    DFLASH_RPC_SYNC_TRIM = 7,        // optional v2 extension; empty request probes support
    DFLASH_RPC_GPU_PROBE = 8,        // explicit diagnostic; never changes the draft sequence
    DFLASH_RPC_GPU_REAL_Q4K = 9,     // explicit GGUF tensor benchmark; sequence remains CPU-owned
    DFLASH_RPC_DML_REAL_Q4K = 10,    // opt-in FP16 DirectML comparison; no sequence mutation
    DFLASH_RPC_GPU_REAL_Q3K = 11,    // real GGUF fc.weight benchmark
    DFLASH_RPC_GPU_REAL_Q5K = 12,    // real GGUF attention V benchmark
    DFLASH_RPC_GPU_KV_Q40 = 13,      // isolated hardware KV pack/trim/reinject diagnostic
    DFLASH_RPC_GPU_SYNC = 14,        // destructive diagnostic on an exclusive test connection
    DFLASH_RPC_GPU_CONV = 15,        // isolated dynamic convolution diagnostic
    DFLASH_RPC_GPU_ATTENTION = 16,   // isolated noncausal SWA attention diagnostic
    DFLASH_RPC_GPU_DRAFT_PROBE = 17, // destructive full-draft parity diagnostic
    DFLASH_RPC_SYNC_AND_DRAFT = 18,  // fused SYNC and DRAFT: single round-trip
};

#pragma pack(push, 1)
struct dflash_rpc_request {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t cycle_id;
    uint32_t n_tokens;
    uint32_t payload_bytes;
};

struct dflash_rpc_response {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t cycle_id;
    int32_t status; // 0=success, else negative error code
    uint32_t payload_bytes;
    uint64_t xbox_receive_us;
    uint64_t xbox_prepare_us;
    uint64_t xbox_compute_us;  // Wall time of the selected CPU or GPU executor.
    uint64_t xbox_response_us; // prior response; current send time is logged server-side
};

struct dflash_rpc_hello {
    uint32_t n_ctx;
    uint32_t n_embd_enc;
    uint32_t n_embd_dec;
    uint32_t selector_top_k;
    int32_t mask_token_id;
    uint32_t cache_bits_k;
    uint32_t cache_bits_v;
    uint64_t model_bytes;
};

struct dflash_rpc_draft_request {
    int32_t pos0;
    int32_t id_last;
    int32_t n_max;
    // Followed by (n_max + 1) * n_embd F32 anchor/MASK embeddings.
    // These are logical target embeddings, before draft embedding_scale.
};

struct dflash_rpc_trim_request {
    int32_t pos0; // Remove all draft KV positions >= pos0 after target verification.
};
struct dflash_rpc_sync_trim_request {
    int32_t pos0;   // Remove the suffix before injecting the following SYNC rows.
    uint32_t reset; // 1 clears the sequence instead; requires pos0=0.
};
// SYNC_AND_DRAFT: sync_trim_request (pos0=-1 means no trim), n_tokens
// SYNC rows, then draft_request and (n_max+1)*n_embd_dec F32 embeddings.
// An empty request with n_tokens=0 only probes support, without mutation.
#pragma pack(pop)

static_assert(sizeof(dflash_rpc_request) == 24, "wire request size");
static_assert(sizeof(dflash_rpc_response) == 56, "wire response size");
static_assert(sizeof(dflash_rpc_hello) == 36, "wire hello size");
static_assert(sizeof(dflash_rpc_draft_request) == 12, "wire draft size");
static_assert(sizeof(dflash_rpc_trim_request) == 4, "wire trim size");
static_assert(sizeof(dflash_rpc_sync_trim_request) == 8, "wire sync trim size");

// Validate untrusted wire data before changing either CPU or GPU KV state.
inline bool dflash_rpc_finite(const uint8_t* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float value;
        std::memcpy(&value, data + i * sizeof(value), sizeof(value));
        if (!std::isfinite(value))
            return false;
    }
    return true;
}

inline bool dflash_rpc_valid_sync(const uint8_t* rows, uint32_t count, uint32_t width,
                                  uint32_t context) {
    const size_t stride = sizeof(int32_t) + size_t(width) * sizeof(float);
    for (uint32_t i = 0; i < count; ++i) {
        int32_t pos;
        std::memcpy(&pos, rows + i * stride, sizeof(pos));
        if (pos < 0 || uint32_t(pos) >= context ||
            !dflash_rpc_finite(rows + i * stride + sizeof(pos), width))
            return false;
        for (uint32_t j = 0; j < i; ++j) {
            int32_t prior;
            std::memcpy(&prior, rows + j * stride, sizeof(prior));
            if (prior == pos)
                return false;
        }
    }
    return true;
}

inline bool dflash_rpc_valid_draft(const dflash_rpc_draft_request& draft, const uint8_t* embeddings,
                                   uint32_t width, uint32_t context) {
    return draft.pos0 >= 0 && draft.n_max >= 1 && draft.n_max <= 7 &&
           uint32_t(draft.pos0) < context &&
           uint32_t(draft.n_max) < context - uint32_t(draft.pos0) &&
           dflash_rpc_finite(embeddings, size_t(draft.n_max + 1) * width);
}
