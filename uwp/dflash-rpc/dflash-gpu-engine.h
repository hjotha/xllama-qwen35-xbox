#pragma once

#include "dflash-gpu-kv.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>
#include <vector>

// SYNC half of the DFlash2 graph. This is an opt-in, hardware-only executor;
// DRAFT is deliberately absent until its complete graph has GPU parity.
class DflashGpuSyncEngine {
  public:
    enum class ActivationMode { GgmlQ8K, OriginalF32 };
    static constexpr uint32_t kLayers = 5;
    static constexpr uint32_t kContext = 49152;
    static constexpr uint32_t kBatch = 64;
    static constexpr uint32_t kFeatureWidth = 25600;
    static constexpr uint32_t kHiddenWidth = 5120;
    static constexpr uint32_t kKvHeads = 8;
    static constexpr uint32_t kHeadWidth = 128;

    // The model path is the exact GGUF used by the CPU reference. A failed
    // initialize never enables the executor or a CPU fallback inside it.
    bool initialize(const std::string& model_path, std::string* error,
                    ActivationMode mode = ActivationMode::GgmlQ8K);

    // Positions are absolute, unique and within [0, 49151]. Input rows are
    // position-major, 25600 F32 values each. SYNC_TRIM calls sync with
    // reset=true or trim_to >= 0; plain SYNC passes both defaults.
    // On any GPU/recording failure after mutation, sequence_valid() is false.
    bool sync(const int32_t* positions, const float* features, uint32_t count, bool reset = false,
              int32_t trim_to = -1, std::string* error = nullptr);
    bool trim(uint32_t keep_positions, std::string* error);
    void reset();

    // Explicit diagnostic only. No readback occurs during sync().
    bool read_encoder(float* destination, uint32_t count, std::string* error);
    bool read_fc(float* destination, uint32_t count, std::string* error);
    bool read_input_q8(void* destination, uint32_t count, std::string* error);
    bool read_encoder_q8(void* destination, uint32_t count, std::string* error);
    struct KvSnapshot {
        std::vector<float> pre_key;
        std::vector<float> pre_value;
        std::vector<float> rotated_key;
        std::vector<float> rotated_value;
        std::vector<uint8_t> packed_key;
        std::vector<uint8_t> packed_value;
    };
    // Diagnostic only: latest SYNC batch. Other layers recompute projections;
    // actual packed KV is unchanged, and normal SYNC has no readback.
    bool read_layer_kv(const int32_t* positions, uint32_t count, KvSnapshot* snapshot,
                       std::string* error, uint32_t layer = kLayers - 1);
    // Raw 20-byte Q4 blocks for every layer, K then V, each requested row.
    // Layout: ((layer * 2 + kind) * count + row) * 640 bytes.
    bool read_kv_packed(const int32_t* positions, uint32_t count, std::vector<uint8_t>* destination,
                        std::string* error);

    bool ready() const {
        return ready_;
    }
    bool sequence_valid() const {
        return sequence_valid_;
    }
    // F4 must call this after any failed external GPU mutation. Recovery
    // requires a new engine/sequence; RESET cannot resurrect a partial KV.
    void invalidate_sequence() {
        sequence_valid_ = false;
        last_batch_ = 0;
    }
    ActivationMode activation_mode() const {
        return activation_mode_;
    }
    const std::string& model_path() const {
        return model_path_;
    }
    uint64_t model_file_size() const {
        return model_file_size_;
    }
    int64_t model_mtime_ticks() const {
        return model_mtime_ticks_;
    }
    const DflashGpuKv& kv() const {
        return kv_;
    }
    ID3D12Device* device() const {
        return device_.Get();
    }
    // F4 may share these resources while this engine remains alive. Its work
    // must be fenced before the next SYNC, RESET or TRIM call. Calls are serial.
    ID3D12CommandQueue* queue() const {
        return queue_.Get();
    }
    DflashGpuKv* mutable_kv() {
        return ready_ && sequence_valid_ ? &kv_ : nullptr;
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
    };
    struct Layer {
        Weight key;
        Weight value;
        Weight key_norm;
        float key_scale = 1.0f;
        float value_scale = 1.0f;
    };

    bool fail(std::string* error, const char* stage, HRESULT hr = S_OK);
    bool create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
                       D3D12_RESOURCE_STATES state, Buffer* out, std::string* error);
    void transition(Buffer& buffer, D3D12_RESOURCE_STATES state);
    void uav_barrier(Buffer& buffer);
    bool begin(std::string* error);
    bool submit(std::string* error);
    bool load_weights(const std::string& model_path, std::string* error);
    bool create_pipelines(std::string* error);
    bool read_buffer(Buffer& source, void* destination, uint64_t bytes, const char* stage,
                     std::string* error);
    bool matmul(const Weight& weight, Buffer& input, Buffer& output, uint32_t batch,
                std::string* error);
    bool quantize_q8(Buffer& input, Buffer& output, uint32_t width, uint32_t batch,
                     std::string* error);
    bool project_kv(uint32_t layer, uint32_t count, std::string* error);
    void dispatch_norm(uint32_t batch);
    void dispatch_key(uint32_t batch, const Layer& layer);
    void dispatch_hadamard(Buffer& input, Buffer& output, uint32_t batch, uint32_t transform_width);

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmd_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> matmul_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> ops_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> matmul_psos_[4][2];
    Microsoft::WRL::ComPtr<ID3D12PipelineState> q8_pack_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> q2q8_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> q3q8_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> q4q8_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> q5q8_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> q6q8_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> q8_0q8_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> norm_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> key_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> scale_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> hadamard_pso_;
    uint64_t fence_value_ = 0;
    HANDLE fence_event_ = nullptr;
    // A submitted list may still reference these on a Signal/wait failure.
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> pending_uploads_;
    Microsoft::WRL::ComPtr<ID3D12Resource> pending_readback_;
    bool unconfirmed_submission_ = false;
    bool terminal_failure_ = false;

    Weight fc_;
    Weight enc_norm_;
    Layer layers_[kLayers];
    Buffer input_;
    Buffer input_q8_;
    Buffer positions_;
    Buffer fc_output_;
    Buffer encoder_;
    Buffer encoder_q8_;
    Buffer key_;
    Buffer value_;
    Buffer scaled_value_;
    Buffer rotated_key_;
    Buffer hadamard_key_;
    Buffer hadamard_value_;
    Buffer input_upload_;
    Buffer position_upload_;
    DflashGpuKv kv_;
    float fc_scale_ = 1.0f;
    float epsilon_ = 1.0e-6f;
    float rope_base_ = 10000000.0f;
    bool ready_ = false;
    bool sequence_valid_ = false;
    ActivationMode activation_mode_ = ActivationMode::GgmlQ8K;
    std::string model_path_;
    uint64_t model_file_size_ = 0;
    int64_t model_mtime_ticks_ = 0;
    bool input_q8_ready_ = false;
    bool encoder_q8_ready_ = false;
    uint32_t last_batch_ = 0;
    int32_t last_positions_[kBatch] = {};

  public:
    ~DflashGpuSyncEngine();
    DflashGpuSyncEngine() = default;
    DflashGpuSyncEngine(const DflashGpuSyncEngine&) = delete;
    DflashGpuSyncEngine& operator=(const DflashGpuSyncEngine&) = delete;
};
