#pragma once

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#include <d3d12.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// A per-connection drafter KV store. All positions are absolute within the
// allocated context; callers decide the model's attention window.
class DflashGpuKv {
  public:
    enum class Kind : uint32_t { Key = 0, Value = 1 };
    static constexpr uint32_t kQuantBlockWidth = 32;
    static constexpr uint32_t kGpuBlockBytes = 20;

    struct Diagnostics {
        uint32_t layers = 0;
        uint32_t capacity = 0;
        uint32_t key_width = 0;
        uint32_t value_width = 0;
        uint64_t allocated_bytes = 0;
        uint64_t valid_key_positions = 0;
        uint64_t valid_value_positions = 0;
        uint64_t pack_records = 0;
        uint64_t unpack_records = 0;
        uint64_t trims = 0;
        uint64_t resets = 0;
    };

    // Widths must be positive multiples of 32. The resource remains in the
    // DEFAULT heap; each Q4_0 block uses 20 bytes (GGML's 18 plus alignment).
    bool initialize(ID3D12Device* device, uint32_t layers, uint32_t capacity, uint32_t key_width,
                    uint32_t value_width, std::string* error);

    // Records only; caller owns command submission, ordering, fences and input
    // transitions. Pack source must be a BUFFER in NON_PIXEL_SHADER_RESOURCE
    // or GENERIC_READ; unpack destination must be a BUFFER in UAV state.
    // Both byte offsets must be 4-byte aligned. Logical validity is updated
    // when Pack is recorded, so do not consume it before successful execution.
    bool record_pack(ID3D12GraphicsCommandList* cmd, uint32_t layer, uint32_t position, Kind kind,
                     ID3D12Resource* f32_source, uint64_t source_byte_offset, std::string* error);
    bool record_unpack(ID3D12GraphicsCommandList* cmd, uint32_t layer, uint32_t position, Kind kind,
                       ID3D12Resource* f32_destination, uint64_t destination_byte_offset,
                       std::string* error);
    // Transition the persistent buffer for an attention shader's SRV read.
    // gpu_byte_offset returns the 20-byte-block address for a valid row.
    bool record_prepare_read(ID3D12GraphicsCommandList* cmd, std::string* error);

    // Logical invalidation only. Dispatches already recorded must be submitted
    // and fenced before trimming/resetting, or discarded as a whole.
    bool trim(uint32_t keep_positions);
    void reset();
    bool valid(uint32_t layer, uint32_t position, Kind kind) const;
    bool gpu_byte_offset(uint32_t layer, uint32_t position, Kind kind, uint64_t* offset) const;
    ID3D12Resource* resource() const {
        return resource_.Get();
    }
    // Keep command-list dependencies alive if execution was submitted but its
    // fence cannot be confirmed. Does not change logical or resource state.
    void retain_inflight_resources(std::vector<Microsoft::WRL::ComPtr<IUnknown>>& out) const;
    Diagnostics diagnostics() const;

  private:
    uint32_t width(Kind kind) const;
    uint64_t slot_offset(uint32_t layer, uint32_t position, Kind kind) const;
    size_t flag_index(uint32_t layer, uint32_t position, Kind kind) const;
    bool check_slot(uint32_t layer, uint32_t position, Kind kind) const;
    bool record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* external, uint64_t external_offset,
                uint32_t layer, uint32_t position, Kind kind, bool pack, std::string* error);

    Microsoft::WRL::ComPtr<ID3D12Resource> resource_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pack_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> unpack_pso_;
    uint32_t layers_ = 0;
    uint32_t capacity_ = 0;
    uint32_t key_width_ = 0;
    uint32_t value_width_ = 0;
    uint64_t row_bytes_ = 0;
    uint64_t allocated_bytes_ = 0;
    D3D12_RESOURCE_STATES resource_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    std::vector<uint8_t> valid_;
    uint64_t pack_records_ = 0;
    uint64_t unpack_records_ = 0;
    uint64_t trims_ = 0;
    uint64_t resets_ = 0;
};
