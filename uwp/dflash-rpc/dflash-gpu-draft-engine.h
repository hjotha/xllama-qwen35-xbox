#pragma once

#include "dflash-gpu-engine.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// DFlash2 noise-block executor. The SYNC engine owns the D3D12 device, queue,
// and Q4_0 KV. Keep that engine alive and serialize calls to both engines.
// Every successful draft fences the shared queue before returning. On failure
// after a GPU mutation, sequence_valid() becomes false; the RPC connection
// must close and must not invoke SYNC, trim, reset or another DRAFT.
class DflashGpuDraftEngine {
  public:
    static constexpr uint32_t kLayers = 5;
    static constexpr uint32_t kContext = 49152;
    static constexpr uint32_t kMaxTokens = 8;
    static constexpr uint32_t kHidden = 5120;
    static constexpr uint32_t kFfn = 17408;
    static constexpr uint32_t kQHeads = 32;
    static constexpr uint32_t kKvHeads = 8;
    static constexpr uint32_t kHead = 128;

    DflashGpuDraftEngine() = default;
    ~DflashGpuDraftEngine();
    DflashGpuDraftEngine(const DflashGpuDraftEngine&) = delete;
    DflashGpuDraftEngine& operator=(const DflashGpuDraftEngine&) = delete;

    struct Checkpoint {
        std::string name;
        std::vector<float> values;
        double gpu_ms = -1.0; // since the preceding checkpoint copy completed
    };

    enum class BarrierMode { Strict, Minimal };
    bool initialize(DflashGpuSyncEngine& sync, const std::string& gguf_path, std::string* error);
    // input/output are token-major F32 [n_max+1,5120]; input is the host's
    // unscaled embedding. The anchor is pos0 and is replaced in shared KV.
    bool draft(uint32_t pos0, const float* embeddings, uint32_t n_max, float* hidden,
               std::string* error, std::vector<Checkpoint>* checkpoints = nullptr);
    bool ready() const {
        return ready_;
    }
    bool sequence_valid() const {
        return sequence_valid_;
    }
    BarrierMode barrier_mode() const {
        return barrier_mode_;
    }
    void set_barrier_mode(BarrierMode mode) {
        barrier_mode_ = mode;
    }

  private:
    struct Buffer {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        uint64_t bytes = 0;
    };
    struct Weight {
        Buffer buffer;
        uint32_t columns = 0;
        uint32_t rows = 0;
        uint32_t type = 0;
        float scale = 1.0f;
    };
    struct Layer {
        Weight q, k, v, o, gate, up, down;
        Weight attn_conv_proj, ffn_conv_proj;
        Weight attn_norm, q_norm, k_norm, ffn_norm;
        Weight attn_conv_base, ffn_conv_base;
        Weight sinks;
    };
    struct PendingCheckpoint {
        std::string name;
        Buffer readback;
        uint64_t bytes = 0;
    };
    enum Slot : size_t {
        Embedding,
        HiddenA,
        HiddenB,
        Norm,
        ConvCoeff,
        ConvIn,
        Q,
        K,
        V,
        QNorm,
        KNorm,
        QRope,
        KRope,
        QRot,
        KRot,
        VRot,
        AttentionRot,
        Attention,
        AttnOut,
        ConvOut,
        FfnIn,
        FfnNorm,
        FfnCoeff,
        FfnConv,
        Gate,
        Up,
        Activated,
        Down,
        MatRaw,
        QuantizedInput,
        Positions,
        ActivePositions,
        UploadEmbedding,
        UploadPositions,
        UploadActive,
        Readback,
        Zero,
        NoSink,
        Count
    };

    bool fail(std::string* error, const std::string& stage, HRESULT hr = S_OK);
    bool create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
                       D3D12_RESOURCE_STATES state, Buffer* out, std::string* error);
    bool create_pipelines(std::string* error);
    bool load_weights(const std::string& path, std::string* error);
    bool begin(std::string* error);
    bool submit(std::string* error);
    void quarantine_inflight();
    bool record_checkpoint(const char* name, Slot slot, uint32_t batch, uint32_t width,
                           std::string* error);
    void transition(Buffer& buffer, D3D12_RESOURCE_STATES next);
    void transition(Buffer& b1, D3D12_RESOURCE_STATES s1, Buffer& b2, D3D12_RESOURCE_STATES s2);
    void transition(Buffer& b1, D3D12_RESOURCE_STATES s1, Buffer& b2, D3D12_RESOURCE_STATES s2,
                    Buffer& b3, D3D12_RESOURCE_STATES s3);
    void barrier(Buffer& buffer);
    bool matmul(const Weight& weight, Slot input, Slot output, uint32_t batch, std::string* error);
    void norm(Slot input, const Weight& weight, Slot output, uint32_t batch, uint32_t width,
              float scale = 1.0f);
    void rope(Slot input, Slot output, uint32_t batch, uint32_t width);
    void hadamard(Slot input, Slot output, uint32_t batch, uint32_t heads,
                  uint32_t transform_width);
    void conv(Slot input, Slot coeff, const Weight& base, Slot output, uint32_t batch,
              uint32_t side);
    void scale_add(Slot input, Slot residual, Slot output, uint32_t batch, uint32_t width,
                   float scale = 1.0f);
    void swiglu(Slot gate, Slot up, Slot output, uint32_t batch);
    bool pack_kv(uint32_t layer, uint32_t pos0, uint32_t batch, std::string* error);
    bool attention(uint32_t layer, uint32_t pos0, uint32_t batch, const Weight& sinks,
                   std::string* error);

    DflashGpuSyncEngine* sync_ = nullptr; // borrowed, owns KV lifetime
    DflashGpuKv* kv_ = nullptr;           // borrowed from sync_
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmd_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE event_ = nullptr;
    uint64_t fence_value_ = 0;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> pending_uploads_;
    std::vector<PendingCheckpoint> pending_checkpoints_;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> trace_queries_;
    Buffer trace_times_;
    uint64_t trace_frequency_ = 0;
    bool fatal_inflight_ = false;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> matmul_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> ops_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> conv_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> attention_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> f32_matmul_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> q8_quantize_pso_, hadamard_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> quant_dot_psos_[6];
    Microsoft::WRL::ComPtr<ID3D12PipelineState> norm_pso_, rope_pso_, scale_add_pso_, swiglu_pso_,
        conv_pso_, attention_pso_;
    std::array<Buffer, Count> buffers_;
    std::array<Layer, kLayers> layers_;
    Weight output_norm_;
    float epsilon_ = 1.0e-6f;
    float rope_base_ = 10000000.0f;
    float embedding_scale_ = 1.0f;
    bool ready_ = false;
    bool sequence_valid_ = false;
    Slot quant_source_ = Count;
    uint32_t quant_batch_ = 0;
    uint32_t quant_width_ = 0;
    BarrierMode barrier_mode_ = BarrierMode::Minimal;
};
