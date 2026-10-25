#include "dflash-gpu-kv.h"

#include "dflash_kv_q4_0_dxil.h"
#include "xllama/d3d12_dyn.h"

#include <algorithm>
#include <cstdio>
#include <limits>

using Microsoft::WRL::ComPtr;

namespace {

bool fail(std::string* error, const char* stage, HRESULT hr = S_OK) {
    if (error) {
        char text[128];
        if (hr == S_OK)
            std::snprintf(text, sizeof(text), "%s", stage);
        else
            std::snprintf(text, sizeof(text), "%s: 0x%08X", stage, static_cast<unsigned>(hr));
        *error = text;
    }
    return false;
}

bool in_bounds(ID3D12Resource* resource, uint64_t offset, uint64_t bytes) {
    if (!resource || (offset & 3) ||
        resource->GetDesc().Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
        return false;
    const uint64_t width = resource->GetDesc().Width;
    return offset <= width && bytes <= width - offset && resource->GetGPUVirtualAddress() != 0;
}

} // namespace

bool DflashGpuKv::initialize(ID3D12Device* device, uint32_t layers, uint32_t capacity,
                             uint32_t key_width, uint32_t value_width, std::string* error) {
    if (resource_)
        return fail(error, "KV already initialized");
    if (!device || !layers || !capacity || !key_width || !value_width || key_width % 32 ||
        value_width % 32)
        return fail(error, "invalid KV shape");
    const uint64_t blocks_per_position = (uint64_t(key_width) + value_width) / 32;
    const uint64_t row_bytes = blocks_per_position * 20;
    if (key_width / 32 > 65535u * 64u || value_width / 32 > 65535u * 64u ||
        row_bytes > std::numeric_limits<uint64_t>::max() / capacity / layers ||
        uint64_t(layers) * capacity > std::numeric_limits<size_t>::max() / 2)
        return fail(error, "KV shape overflow");
    const uint64_t total_bytes = row_bytes * capacity * layers;

    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = 1;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC root_desc = {};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    auto serialize = xllama::d3d12_dyn::SerializeRootSignature();
    if (!serialize)
        return fail(error, "D3D12SerializeRootSignature unavailable");
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> messages;
    HRESULT hr = serialize(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &messages);
    if (FAILED(hr))
        return fail(error, "SerializeRootSignature", hr);
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr))
        return fail(error, "CreateRootSignature", hr);

    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline = {};
    pipeline.pRootSignature = root.Get();
    pipeline.CS = {kDflashKvQ40PackDxil, kDflashKvQ40PackDxilSize};
    ComPtr<ID3D12PipelineState> pack_pso;
    hr = device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&pack_pso));
    if (FAILED(hr))
        return fail(error, "CreateComputePipelineState Pack", hr);
    pipeline.CS = {kDflashKvQ40UnpackDxil, kDflashKvQ40UnpackDxilSize};
    ComPtr<ID3D12PipelineState> unpack_pso;
    hr = device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&unpack_pso));
    if (FAILED(hr))
        return fail(error, "CreateComputePipelineState Unpack", hr);

    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = total_bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> resource;
    hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                         IID_PPV_ARGS(&resource));
    if (FAILED(hr))
        return fail(error, "CreateCommittedResource KV", hr);

    root_ = std::move(root);
    pack_pso_ = std::move(pack_pso);
    unpack_pso_ = std::move(unpack_pso);
    resource_ = std::move(resource);
    layers_ = layers;
    capacity_ = capacity;
    key_width_ = key_width;
    value_width_ = value_width;
    row_bytes_ = row_bytes;
    allocated_bytes_ = total_bytes;
    valid_.assign(size_t(layers) * capacity * 2, 0);
    if (error)
        error->clear();
    return true;
}

uint32_t DflashGpuKv::width(Kind kind) const {
    return kind == Kind::Key ? key_width_ : value_width_;
}

bool DflashGpuKv::check_slot(uint32_t layer, uint32_t position, Kind kind) const {
    return resource_ && layer < layers_ && position < capacity_ &&
           (kind == Kind::Key || kind == Kind::Value);
}

uint64_t DflashGpuKv::slot_offset(uint32_t layer, uint32_t position, Kind kind) const {
    const uint64_t row = (uint64_t(layer) * capacity_ + position) * row_bytes_;
    return row + (kind == Kind::Value ? uint64_t(key_width_) / 32 * 20 : 0);
}

size_t DflashGpuKv::flag_index(uint32_t layer, uint32_t position, Kind kind) const {
    return (size_t(layer) * capacity_ + position) * 2 + static_cast<uint32_t>(kind);
}

bool DflashGpuKv::gpu_byte_offset(uint32_t layer, uint32_t position, Kind kind,
                                  uint64_t* offset) const {
    if (!offset || !valid(layer, position, kind))
        return false;
    *offset = slot_offset(layer, position, kind);
    return true;
}

bool DflashGpuKv::valid(uint32_t layer, uint32_t position, Kind kind) const {
    return check_slot(layer, position, kind) && valid_[flag_index(layer, position, kind)] != 0;
}

bool DflashGpuKv::record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* external,
                         uint64_t external_offset, uint32_t layer, uint32_t position, Kind kind,
                         bool pack, std::string* error) {
    if (!cmd || !check_slot(layer, position, kind) || external == resource_.Get())
        return fail(error, "invalid KV record arguments");
    if (!pack && !valid(layer, position, kind))
        return fail(error, "KV position invalid");
    const uint64_t float_bytes = uint64_t(width(kind)) * 4;
    if (!in_bounds(external, external_offset, float_bytes))
        return fail(error, "F32 buffer range invalid");
    const uint32_t block_count = width(kind) / 32;
    const uint64_t kv_offset = slot_offset(layer, position, kind);
    const D3D12_RESOURCE_STATES needed = pack ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                              : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (resource_state_ != needed) {
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource_.Get();
        barrier.Transition.StateBefore = resource_state_;
        barrier.Transition.StateAfter = needed;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd->ResourceBarrier(1, &barrier);
        resource_state_ = needed;
    }
    cmd->SetComputeRootSignature(root_.Get());
    cmd->SetPipelineState(pack ? pack_pso_.Get() : unpack_pso_.Get());
    cmd->SetComputeRoot32BitConstant(0, block_count, 0);
    cmd->SetComputeRootShaderResourceView(1,
                                          pack ? external->GetGPUVirtualAddress() + external_offset
                                               : resource_->GetGPUVirtualAddress() + kv_offset);
    cmd->SetComputeRootUnorderedAccessView(2, pack ? resource_->GetGPUVirtualAddress() + kv_offset
                                                   : external->GetGPUVirtualAddress() +
                                                         external_offset);
    cmd->Dispatch((block_count - 1) / 64 + 1, 1, 1);
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = pack ? resource_.Get() : external;
    cmd->ResourceBarrier(1, &barrier);
    if (pack) {
        valid_[flag_index(layer, position, kind)] = 1;
        ++pack_records_;
    } else {
        ++unpack_records_;
    }
    if (error)
        error->clear();
    return true;
}

bool DflashGpuKv::record_pack(ID3D12GraphicsCommandList* cmd, uint32_t layer, uint32_t position,
                              Kind kind, ID3D12Resource* f32_source, uint64_t source_byte_offset,
                              std::string* error) {
    return record(cmd, f32_source, source_byte_offset, layer, position, kind, true, error);
}

bool DflashGpuKv::record_unpack(ID3D12GraphicsCommandList* cmd, uint32_t layer, uint32_t position,
                                Kind kind, ID3D12Resource* f32_destination,
                                uint64_t destination_byte_offset, std::string* error) {
    return record(cmd, f32_destination, destination_byte_offset, layer, position, kind, false,
                  error);
}

bool DflashGpuKv::record_prepare_read(ID3D12GraphicsCommandList* cmd, std::string* error) {
    if (!cmd || !resource_)
        return fail(error, "KV read not initialized");
    if (resource_state_ != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) {
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource_.Get();
        barrier.Transition.StateBefore = resource_state_;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd->ResourceBarrier(1, &barrier);
        resource_state_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if (error)
        error->clear();
    return true;
}

bool DflashGpuKv::trim(uint32_t keep_positions) {
    if (!resource_ || keep_positions > capacity_)
        return false;
    for (uint32_t layer = 0; layer < layers_; ++layer) {
        const size_t begin = flag_index(layer, keep_positions, Kind::Key);
        const size_t end = (size_t(layer) + 1) * capacity_ * 2;
        std::fill(valid_.begin() + begin, valid_.begin() + end, uint8_t{0});
    }
    ++trims_;
    return true;
}

void DflashGpuKv::reset() {
    std::fill(valid_.begin(), valid_.end(), uint8_t{0});
    ++resets_;
}

void DflashGpuKv::retain_inflight_resources(
    std::vector<Microsoft::WRL::ComPtr<IUnknown>>& out) const {
    if (resource_)
        out.emplace_back(static_cast<IUnknown*>(resource_.Get()));
    if (root_)
        out.emplace_back(static_cast<IUnknown*>(root_.Get()));
    if (pack_pso_)
        out.emplace_back(static_cast<IUnknown*>(pack_pso_.Get()));
    if (unpack_pso_)
        out.emplace_back(static_cast<IUnknown*>(unpack_pso_.Get()));
}

DflashGpuKv::Diagnostics DflashGpuKv::diagnostics() const {
    Diagnostics result;
    result.layers = layers_;
    result.capacity = capacity_;
    result.key_width = key_width_;
    result.value_width = value_width_;
    result.allocated_bytes = allocated_bytes_;
    for (size_t index = 0; index < valid_.size(); index += 2) {
        result.valid_key_positions += valid_[index];
        result.valid_value_positions += valid_[index + 1];
    }
    result.pack_records = pack_records_;
    result.unpack_records = unpack_records_;
    result.trims = trims_;
    result.resets = resets_;
    return result;
}
