#include "dflash-gpu-draft-engine.h"
#include "dflash-gpu-rope.h"
#include "dflash-ucrt-exp-table.h"

#include "dflash_q2k_q8k_dxil.h"
#include "dflash_q3k_q8k_dxil.h"
#include "dflash_draft_quant_q8_dxil.h"
#include "dflash_q6k_q8k_dxil.h"
#include "dflash_q8_0_q8k_dxil.h"
#include "dflash_sync_ops_dxil.h"
#include "dflash_draft_ops_dxil.h"
#include "dflash_hadamard128_dxil.h"
#include "dflash_dynamic_conv_dxil.h"
#include "dflash_attention_dxil.h"
#include "gguf.h"
#include "xllama/d3d12_dyn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint64_t aligned4(uint64_t value) { return (value + 3u) & ~uint64_t(3u); }

bool range_ok(ID3D12Resource * resource, uint64_t bytes) {
    return resource && resource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
           resource->GetDesc().Width >= bytes && resource->GetGPUVirtualAddress();
}

// A failed Signal/SetEvent/timeout after ExecuteCommandLists leaves completion
// uncertain. The RPC must close and restart the process. Keep all resources
// referenced by the in-flight list alive until process exit in that rare path.
struct InflightQuarantine {
    std::mutex mutex;
    std::vector<ComPtr<IUnknown>> objects;
};

InflightQuarantine & quarantine() {
    static auto * instance = new InflightQuarantine();
    return *instance;
}

template <typename T>
void keep(std::vector<ComPtr<IUnknown>> & out, const ComPtr<T> & ptr) {
    if (!ptr) return;
    ComPtr<IUnknown> item;
    if (SUCCEEDED(ptr.As(&item))) out.push_back(std::move(item));
}
} // namespace

DflashGpuDraftEngine::~DflashGpuDraftEngine() {
    // Public draft() fences every submitted list; initialization fences every
    // weight copy. Do not destroy an engine while another thread uses it.
    // SetEventOnCompletion may still own this HANDLE after a timeout. The
    // process must restart after quarantine, so retain this one handle until
    // exit rather than close it while D3D12 can still signal it.
    if (event_ && !fatal_inflight_) CloseHandle(event_);
}

bool DflashGpuDraftEngine::fail(std::string * error, const std::string & stage,
                                HRESULT hr) {
    if (error) {
        if (hr == S_OK) *error = stage;
        else {
            char message[24];
            std::snprintf(message, sizeof(message), " 0x%08X", static_cast<unsigned>(hr));
            *error = stage + message;
        }
    }
    return false;
}

bool DflashGpuDraftEngine::create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap,
        D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
        Buffer * out, std::string * error) {
    if (!device_ || !out || !bytes || bytes > UINT64_MAX - 3)
        return fail(error, "invalid D3D12 buffer size");
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = aligned4(bytes);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    const HRESULT hr = device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE,
        &desc, state, nullptr, IID_PPV_ARGS(&resource));
    if (FAILED(hr)) return fail(error, "CreateCommittedResource", hr);
    out->resource = std::move(resource);
    out->bytes = desc.Width;
    out->state = state;
    return true;
}

void DflashGpuDraftEngine::transition(Buffer & buffer, D3D12_RESOURCE_STATES next) {
    if (buffer.state == next) return;
    D3D12_RESOURCE_BARRIER item = {};
    item.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    item.Transition.pResource = buffer.resource.Get();
    item.Transition.StateBefore = buffer.state;
    item.Transition.StateAfter = next;
    item.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_->ResourceBarrier(1, &item);
    buffer.state = next;
}

void DflashGpuDraftEngine::transition(Buffer & b1, D3D12_RESOURCE_STATES s1,
                                     Buffer & b2, D3D12_RESOURCE_STATES s2) {
    D3D12_RESOURCE_BARRIER items[2];
    uint32_t count = 0;
    if (b1.state != s1) {
        items[count] = {};
        items[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        items[count].Transition.pResource = b1.resource.Get();
        items[count].Transition.StateBefore = b1.state;
        items[count].Transition.StateAfter = s1;
        items[count].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b1.state = s1;
        ++count;
    }
    if (b2.state != s2) {
        items[count] = {};
        items[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        items[count].Transition.pResource = b2.resource.Get();
        items[count].Transition.StateBefore = b2.state;
        items[count].Transition.StateAfter = s2;
        items[count].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b2.state = s2;
        ++count;
    }
    if (count > 0) {
        cmd_->ResourceBarrier(count, items);
    }
}

void DflashGpuDraftEngine::transition(Buffer & b1, D3D12_RESOURCE_STATES s1,
                                     Buffer & b2, D3D12_RESOURCE_STATES s2,
                                     Buffer & b3, D3D12_RESOURCE_STATES s3) {
    D3D12_RESOURCE_BARRIER items[3];
    uint32_t count = 0;
    Buffer * bufs[3] = {&b1, &b2, &b3};
    D3D12_RESOURCE_STATES states[3] = {s1, s2, s3};
    for (int i = 0; i < 3; ++i) {
        if (bufs[i]->state != states[i]) {
            items[count] = {};
            items[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            items[count].Transition.pResource = bufs[i]->resource.Get();
            items[count].Transition.StateBefore = bufs[i]->state;
            items[count].Transition.StateAfter = states[i];
            items[count].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            bufs[i]->state = states[i];
            ++count;
        }
    }
    if (count > 0) {
        cmd_->ResourceBarrier(count, items);
    }
}

void DflashGpuDraftEngine::barrier(Buffer & buffer) {
    D3D12_RESOURCE_BARRIER item = {};
    item.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    item.UAV.pResource = buffer.resource.Get();
    cmd_->ResourceBarrier(1, &item);
}

bool DflashGpuDraftEngine::begin(std::string * error) {
    if (fatal_inflight_) return fail(error, "GPU work completion uncertain; restart required");
    HRESULT hr = allocator_->Reset();
    if (FAILED(hr)) return fail(error, "CommandAllocator Reset", hr);
    hr = cmd_->Reset(allocator_.Get(), nullptr);
    return SUCCEEDED(hr) ? true : fail(error, "CommandList Reset", hr);
}

bool DflashGpuDraftEngine::submit(std::string * error) {
    HRESULT hr = cmd_->Close();
    if (FAILED(hr)) return fail(error, "CommandList Close", hr);
    ID3D12CommandList * lists[] = {cmd_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    const uint64_t value = ++fence_value_;
    hr = queue_->Signal(fence_.Get(), value);
    if (FAILED(hr)) {
        quarantine_inflight();
        return fail(error, "CommandQueue Signal; restart required", hr);
    }
    if (fence_->GetCompletedValue() < value) {
        hr = fence_->SetEventOnCompletion(value, event_);
        if (FAILED(hr)) {
            quarantine_inflight();
            return fail(error, "Fence SetEventOnCompletion; restart required", hr);
        }
        if (WaitForSingleObject(event_, 60000) != WAIT_OBJECT_0) {
            quarantine_inflight();
            return fail(error, "GPU fence timeout; restart required");
        }
    }
    hr = device_->GetDeviceRemovedReason();
    if (FAILED(hr)) {
        quarantine_inflight();
        return fail(error, "D3D12 device removed; restart required", hr);
    }
    if (fence_->GetCompletedValue() < value) {
        quarantine_inflight();
        return fail(error, "GPU fence incomplete; restart required");
    }
    return true;
}

void DflashGpuDraftEngine::quarantine_inflight() {
    if (fatal_inflight_) return;
    fatal_inflight_ = true;
    std::vector<ComPtr<IUnknown>> retained;
    keep(retained, device_);
    keep(retained, queue_);
    keep(retained, allocator_);
    keep(retained, cmd_);
    keep(retained, fence_);
    keep(retained, matmul_root_);
    keep(retained, ops_root_);
    keep(retained, conv_root_);
    keep(retained, attention_root_);
    keep(retained, f32_matmul_pso_);
    keep(retained, q8_quantize_pso_);
    keep(retained, hadamard_pso_);
    for (const auto & pso : quant_dot_psos_) keep(retained, pso);
    keep(retained, norm_pso_);
    keep(retained, rope_pso_);
    keep(retained, scale_add_pso_);
    keep(retained, swiglu_pso_);
    keep(retained, conv_pso_);
    keep(retained, attention_pso_);
    keep(retained, trace_queries_);
    keep(retained, trace_times_.resource);
    for (const auto & buffer : buffers_) keep(retained, buffer.resource);
    for (const auto & upload : pending_uploads_) keep(retained, upload);
    for (const auto & checkpoint : pending_checkpoints_)
        keep(retained, checkpoint.readback.resource);
    auto hold_weight = [&](const Weight & weight) { keep(retained, weight.buffer.resource); };
    hold_weight(output_norm_);
    for (const auto & layer : layers_) {
        hold_weight(layer.q); hold_weight(layer.k); hold_weight(layer.v);
        hold_weight(layer.o); hold_weight(layer.gate); hold_weight(layer.up);
        hold_weight(layer.down); hold_weight(layer.attn_conv_proj);
        hold_weight(layer.ffn_conv_proj); hold_weight(layer.attn_norm);
        hold_weight(layer.q_norm); hold_weight(layer.k_norm);
        hold_weight(layer.ffn_norm); hold_weight(layer.attn_conv_base);
        hold_weight(layer.ffn_conv_base); hold_weight(layer.sinks);
    }
    if (kv_) kv_->retain_inflight_resources(retained);
    auto & global = quarantine();
    std::lock_guard<std::mutex> lock(global.mutex);
    global.objects.insert(global.objects.end(),
                          std::make_move_iterator(retained.begin()),
                          std::make_move_iterator(retained.end()));
}

bool DflashGpuDraftEngine::record_checkpoint(const char * name, Slot slot,
                                             uint32_t batch, uint32_t width,
                                             std::string * error) {
    PendingCheckpoint checkpoint;
    checkpoint.name = name;
    checkpoint.bytes = uint64_t(batch) * width * sizeof(float);
    const uint32_t query = 2u * static_cast<uint32_t>(pending_checkpoints_.size()) + 1u;
    if (query + 1u >= 256u) return fail(error, "DRAFT trace query capacity exceeded");
    if (trace_frequency_) cmd_->EndQuery(trace_queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query);
    if (!create_buffer(checkpoint.bytes, D3D12_HEAP_TYPE_READBACK,
                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                       &checkpoint.readback, error)) return false;
    Buffer & source = buffers_[slot];
    transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmd_->CopyBufferRegion(checkpoint.readback.resource.Get(), 0,
                           source.resource.Get(), 0, checkpoint.bytes);
    if (trace_frequency_) cmd_->EndQuery(trace_queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query + 1u);
    pending_checkpoints_.push_back(std::move(checkpoint));
    return true;
}

bool DflashGpuDraftEngine::create_pipelines(std::string * error) {
    auto serialize = xllama::d3d12_dyn::SerializeRootSignature();
    if (!serialize) return fail(error, "D3D12SerializeRootSignature unavailable");
    auto root = [&](uint32_t constants, uint32_t srvs,
                    ComPtr<ID3D12RootSignature> * target) -> bool {
        D3D12_ROOT_PARAMETER params[7] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.ShaderRegister = 0;
        params[0].Constants.Num32BitValues = constants;
        for (uint32_t i = 0; i < srvs; ++i) {
            params[i + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            params[i + 1].Descriptor.ShaderRegister = i;
        }
        params[srvs + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[srvs + 1].Descriptor.ShaderRegister = 0;
        D3D12_ROOT_SIGNATURE_DESC desc = {};
        desc.NumParameters = srvs + 2;
        desc.pParameters = params;
        ComPtr<ID3DBlob> blob, messages;
        HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &messages);
        if (FAILED(hr)) return fail(error, "SerializeRootSignature", hr);
        hr = device_->CreateRootSignature(0, blob->GetBufferPointer(),
             blob->GetBufferSize(), IID_PPV_ARGS(target->GetAddressOf()));
        return SUCCEEDED(hr) ? true : fail(error, "CreateRootSignature", hr);
    };
    if (!root(4, 2, &matmul_root_) || !root(8, 3, &ops_root_) ||
        !root(4, 3, &conv_root_) || !root(4, 5, &attention_root_)) return false;
    auto pso = [&](ID3D12RootSignature * signature, const uint8_t * code,
                   size_t bytes, ComPtr<ID3D12PipelineState> * target) -> bool {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = signature;
        desc.CS = {code, bytes};
        const HRESULT hr = device_->CreateComputePipelineState(&desc,
             IID_PPV_ARGS(target->GetAddressOf()));
        return SUCCEEDED(hr) ? true : fail(error, "CreateComputePipelineState", hr);
    };
    if (!pso(ops_root_.Get(), kDflashSyncF32MatmulDxil, kDflashSyncF32MatmulDxilSize,
             &f32_matmul_pso_) ||
        !pso(matmul_root_.Get(), kDflashQ8kQuantizeExactDxil, kDflashQ8kQuantizeExactDxilSize,
             &q8_quantize_pso_) ||
        !pso(matmul_root_.Get(), kDflashHadamard128Dxil, kDflashHadamard128DxilSize,
             &hadamard_pso_) ||
        !pso(matmul_root_.Get(), kDflashQ2kQ8kDotDxil, kDflashQ2kQ8kDotDxilSize,
             &quant_dot_psos_[5]) ||
        !pso(matmul_root_.Get(), kDflashQ3kQ8kDotDxil, kDflashQ3kQ8kDotDxilSize,
             &quant_dot_psos_[0]) ||
        !pso(matmul_root_.Get(), kDflashDraftQ4kQ8kDotDxil, kDflashDraftQ4kQ8kDotDxilSize,
             &quant_dot_psos_[1]) ||
        !pso(matmul_root_.Get(), kDflashDraftQ5kQ8kDotDxil, kDflashDraftQ5kQ8kDotDxilSize,
             &quant_dot_psos_[2]) ||
        !pso(matmul_root_.Get(), kDflashQ6kQ8kDotDxil, kDflashQ6kQ8kDotDxilSize,
             &quant_dot_psos_[3]) ||
        !pso(matmul_root_.Get(), kDflashQ8_0Q8kDotDxil, kDflashQ8_0Q8kDotDxilSize,
             &quant_dot_psos_[4]) ||
        !pso(ops_root_.Get(), kDflashDraftRmsNormDxil, kDflashDraftRmsNormDxilSize, &norm_pso_) ||
        !pso(ops_root_.Get(), kDflashDraftRope128Dxil, kDflashDraftRope128DxilSize, &rope_pso_) ||
        !pso(ops_root_.Get(), kDflashDraftScaleAddDxil, kDflashDraftScaleAddDxilSize,
             &scale_add_pso_) ||
        !pso(ops_root_.Get(), kDflashDraftSwiGluDxil, kDflashDraftSwiGluDxilSize,
             &swiglu_pso_) ||
        !pso(conv_root_.Get(), kDflashDynamicConvDxil, kDflashDynamicConvDxilSize,
             &conv_pso_) ||
        !pso(attention_root_.Get(), kDflashAttentionDxil, kDflashAttentionDxilSize,
             &attention_pso_)) return false;
    return true;
}

bool DflashGpuDraftEngine::load_weights(const std::string & path,
                                       std::string * error) {
    using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    gguf_ptr gguf(gguf_init_from_file(path.c_str(), {true, nullptr}), &gguf_free);
    if (!gguf) return fail(error, "GGUF metadata load failed");
    auto key = [&](const char * name) { return gguf_find_key(gguf.get(), name); };
    auto u32_eq = [&](const char * name, uint32_t expected) {
        const int64_t id = key(name);
        return id >= 0 && gguf_get_kv_type(gguf.get(), id) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(gguf.get(), id) == expected;
    };
    const int64_t arch = key("general.architecture");
    const int64_t causal = key("dflash.attention.causal");
    if (arch < 0 || gguf_get_kv_type(gguf.get(), arch) != GGUF_TYPE_STRING ||
        std::strcmp(gguf_get_val_str(gguf.get(), arch), "dflash") ||
        causal < 0 || gguf_get_kv_type(gguf.get(), causal) != GGUF_TYPE_BOOL ||
        gguf_get_val_bool(gguf.get(), causal) ||
        !u32_eq("dflash.block_count", kLayers) ||
        !u32_eq("dflash.embedding_length", kHidden) ||
        !u32_eq("dflash.attention.head_count", kQHeads) ||
        !u32_eq("dflash.attention.head_count_kv", kKvHeads) ||
        !u32_eq("dflash.attention.key_length", kHead) ||
        !u32_eq("dflash.attention.value_length", kHead) ||
        !u32_eq("dflash.feed_forward_length", kFfn) ||
        !u32_eq("dflash.block_size", 8) ||
        !u32_eq("dflash.conv_kernel_size", 2) ||
        !u32_eq("dflash.conv_group_size", 16) ||
        !u32_eq("dflash.attention.sliding_window", 2048))
        return fail(error, "unsupported DFlash DRAFT geometry");
    const int64_t pattern = key("dflash.attention.sliding_window_pattern");
    if (pattern >= 0) {
        const auto type = gguf_get_arr_type(gguf.get(), pattern);
        if (gguf_get_arr_n(gguf.get(), pattern) != kLayers ||
            (type != GGUF_TYPE_BOOL && type != GGUF_TYPE_INT32))
            return fail(error, "unsupported interleaved SWA pattern");
        if (type == GGUF_TYPE_BOOL) {
            const auto * values = static_cast<const uint8_t *>(gguf_get_arr_data(gguf.get(), pattern));
            if (!values) return fail(error, "empty SWA pattern");
            for (uint32_t i = 0; i < kLayers; ++i)
                if (!values[i]) return fail(error, "full-attention layer needs a separate kernel");
        } else {
            const auto * values = static_cast<const int32_t *>(gguf_get_arr_data(gguf.get(), pattern));
            if (!values) return fail(error, "empty SWA pattern");
            for (uint32_t i = 0; i < kLayers; ++i)
                if (!values[i]) return fail(error, "full-attention layer needs a separate kernel");
        }
    }
    const int64_t eps_id = key("dflash.attention.layer_norm_rms_epsilon");
    const int64_t rope_id = key("dflash.rope.freq_base");
    const int64_t sections_id = key("dflash.rope.dimension_sections");
    if (eps_id < 0 || rope_id < 0 || sections_id < 0 ||
        gguf_get_kv_type(gguf.get(), eps_id) != GGUF_TYPE_FLOAT32 ||
        gguf_get_kv_type(gguf.get(), rope_id) != GGUF_TYPE_FLOAT32 ||
        gguf_get_arr_type(gguf.get(), sections_id) != GGUF_TYPE_INT32 ||
        gguf_get_arr_n(gguf.get(), sections_id) != 4)
        return fail(error, "unsupported DRAFT norm/RoPE metadata");
    const auto * sections = static_cast<const int32_t *>(
        gguf_get_arr_data(gguf.get(), sections_id));
    if (!sections || sections[0] != 64 || sections[1] || sections[2] || sections[3])
        return fail(error, "unsupported M-RoPE sections");
    epsilon_ = gguf_get_val_f32(gguf.get(), eps_id);
    rope_base_ = gguf_get_val_f32(gguf.get(), rope_id);
    if (!std::isfinite(epsilon_) || epsilon_ <= 0 ||
        !std::isfinite(rope_base_) || rope_base_ <= 1 ||
        key("dflash.rope.scaling.type") >= 0 ||
        key("dflash.rope.scaling.factor") >= 0 ||
        key("dflash.rope.freq_scale") >= 0)
        return fail(error, "unsupported scaled RoPE");
    const int64_t embedding_id = key("dflash.embedding_scale");
    if (embedding_id >= 0) {
        if (gguf_get_kv_type(gguf.get(), embedding_id) != GGUF_TYPE_FLOAT32)
            return fail(error, "invalid embedding_scale type");
        embedding_scale_ = gguf_get_val_f32(gguf.get(), embedding_id);
        if (!std::isfinite(embedding_scale_))
            return fail(error, "invalid embedding_scale");
        if (embedding_scale_ == 0.0f) embedding_scale_ = 1.0f;
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0) return fail(error, "GGUF file open failed");
    const uint64_t file_size = static_cast<uint64_t>(file.tellg());
    const uint64_t data_offset = gguf_get_data_offset(gguf.get());
    auto load = [&](const std::string & name, uint32_t cols, uint32_t rows,
                    bool f32_only, Weight * out, bool optional_sink = false,
                    bool conv_base = false) -> bool {
        const int64_t id = gguf_find_tensor(gguf.get(), name.c_str());
        if (id < 0) return optional_sink ? true : fail(error, "missing tensor " + name);
        const auto type = gguf_get_tensor_type(gguf.get(), id);
        if (f32_only ? type != GGML_TYPE_F32 :
            (type != GGML_TYPE_F32 && type != GGML_TYPE_Q2_K && type != GGML_TYPE_Q3_K &&
             type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q5_K &&
             type != GGML_TYPE_Q6_K && type != GGML_TYPE_Q8_0))
            return fail(error, "unsupported tensor type " + name);
        const int64_t * ne = gguf_get_tensor_ne(gguf.get(), id);
        const bool shape_ok = ne && ne[0] == cols && ne[3] == 1 &&
            (conv_base ? ne[1] == 2 && ne[2] == 2 :
             ne[1] == rows && ne[2] == 1);
        if (!shape_ok)
            return fail(error, "tensor shape mismatch " + name);
        if (type != GGML_TYPE_F32 && cols % 256)
            return fail(error, "quantized tensor row misaligned " + name);
        const uint64_t bytes = gguf_get_tensor_size(gguf.get(), id);
        const uint64_t block_bytes = type == GGML_TYPE_Q2_K ? 84u : type == GGML_TYPE_Q3_K ? 110u :
            type == GGML_TYPE_Q4_K ? 144u : type == GGML_TYPE_Q5_K ? 176u :
            type == GGML_TYPE_Q6_K ? 210u : type == GGML_TYPE_Q8_0 ? 272u : 0u;
        const uint64_t expected = type == GGML_TYPE_F32 ?
            uint64_t(cols) * rows * 4u : uint64_t(cols / 256) * rows * block_bytes;
        const uint64_t tensor_offset = gguf_get_tensor_offset(gguf.get(), id);
        if (tensor_offset > UINT64_MAX - data_offset)
            return fail(error, "tensor offset overflow " + name);
        const uint64_t offset = data_offset + tensor_offset;
        if (!bytes || bytes != expected || offset > file_size ||
            bytes > file_size - offset ||
            bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
            return fail(error, "tensor file range/size invalid " + name);
        const uint64_t upload_bytes = bytes + (optional_sink ? sizeof(dflash_exp::kUcrtExpTable) : 0u);
        Buffer upload;
        if (!create_buffer(upload_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, &upload, error) ||
            !create_buffer(upload_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COPY_DEST, &out->buffer, error)) return false;
        void * mapped = nullptr;
        HRESULT hr = upload.resource->Map(0, nullptr, &mapped);
        if (FAILED(hr)) return fail(error, "GGUF upload Map", hr);
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(static_cast<char *>(mapped), static_cast<std::streamsize>(bytes));
        const bool read_ok = static_cast<bool>(file);
        if (read_ok && optional_sink)
            std::memcpy(static_cast<uint8_t *>(mapped) + bytes,
                        dflash_exp::kUcrtExpTable, sizeof(dflash_exp::kUcrtExpTable));
        if (aligned4(upload_bytes) > upload_bytes)
            std::memset(static_cast<uint8_t *>(mapped) + upload_bytes, 0,
                        static_cast<size_t>(aligned4(upload_bytes) - upload_bytes));
        upload.resource->Unmap(0, nullptr);
        if (!read_ok) return fail(error, "GGUF tensor read failed " + name);
        if (!begin(error)) return false;
        cmd_->CopyBufferRegion(out->buffer.resource.Get(), 0,
                               upload.resource.Get(), 0, aligned4(upload_bytes));
        transition(out->buffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        pending_uploads_.push_back(upload.resource);
        if (!submit(error)) return false;
        pending_uploads_.clear();
        out->columns = cols;
        out->rows = rows;
        out->type = static_cast<uint32_t>(type);
        return true;
    };
    auto scale = [&](const std::string & name, Weight * weight) -> bool {
        const int64_t id = gguf_find_tensor(gguf.get(), name.c_str());
        if (id < 0) return true;
        const int64_t * ne = gguf_get_tensor_ne(gguf.get(), id);
        if (gguf_get_tensor_type(gguf.get(), id) != GGML_TYPE_F32 ||
            !ne || ne[0] != 1 || ne[1] != 1 || gguf_get_tensor_size(gguf.get(), id) != 4)
            return fail(error, "invalid scale tensor " + name);
        const uint64_t tensor_offset = gguf_get_tensor_offset(gguf.get(), id);
        if (tensor_offset > UINT64_MAX - data_offset)
            return fail(error, "scale tensor offset overflow " + name);
        const uint64_t offset = data_offset + tensor_offset;
        if (offset > file_size || 4 > file_size - offset)
            return fail(error, "scale tensor range invalid " + name);
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(reinterpret_cast<char *>(&weight->scale), 4);
        return file && std::isfinite(weight->scale) ? true :
               fail(error, "scale tensor read failed " + name);
    };
    if (!load("output_norm.weight", kHidden, 1, true, &output_norm_)) return false;
    for (uint32_t i = 0; i < kLayers; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        Layer & layer = layers_[i];
        if (!load(prefix + "attn_norm.weight", kHidden, 1, true, &layer.attn_norm) ||
            !load(prefix + "attn_q.weight", kHidden, kQHeads * kHead, false, &layer.q) ||
            !load(prefix + "attn_k.weight", kHidden, kKvHeads * kHead, false, &layer.k) ||
            !load(prefix + "attn_v.weight", kHidden, kKvHeads * kHead, false, &layer.v) ||
            !load(prefix + "attn_output.weight", kQHeads * kHead, kHidden, false, &layer.o) ||
            !load(prefix + "attn_q_norm.weight", kHead, 1, true, &layer.q_norm) ||
            !load(prefix + "attn_k_norm.weight", kHead, 1, true, &layer.k_norm) ||
            !load(prefix + "ffn_norm.weight", kHidden, 1, true, &layer.ffn_norm) ||
            !load(prefix + "ffn_gate.weight", kHidden, kFfn, false, &layer.gate) ||
            !load(prefix + "ffn_up.weight", kHidden, kFfn, false, &layer.up) ||
            !load(prefix + "ffn_down.weight", kFfn, kHidden, false, &layer.down) ||
            !load(prefix + "attn_conv_proj.weight", kHidden, 1280, false,
                  &layer.attn_conv_proj) ||
            !load(prefix + "attn_conv_base", kHidden, 4, true,
                  &layer.attn_conv_base, false, true) ||
            !load(prefix + "ffn_conv_proj.weight", kHidden, 1280, false,
                  &layer.ffn_conv_proj) ||
            !load(prefix + "ffn_conv_base", kHidden, 4, true,
                  &layer.ffn_conv_base, false, true) ||
            !load(prefix + "attn_sinks.weight", kQHeads, 1, true,
                  &layer.sinks, true)) return false;
        const struct { const char * name; Weight * weight; } scaled[] = {
            {"attn_q.scale", &layer.q}, {"attn_k.scale", &layer.k},
            {"attn_v.scale", &layer.v}, {"attn_output.scale", &layer.o},
            {"ffn_gate.scale", &layer.gate}, {"ffn_up.scale", &layer.up},
            {"ffn_down.scale", &layer.down},
            {"attn_conv_proj.scale", &layer.attn_conv_proj},
            {"ffn_conv_proj.scale", &layer.ffn_conv_proj},
        };
        for (const auto & item : scaled)
            if (!scale(prefix + item.name, item.weight)) return false;
    }
    return true;
}

bool DflashGpuDraftEngine::initialize(DflashGpuSyncEngine & sync,
                                      const std::string & gguf_path,
                                      std::string * error) {
    if (sync_ || !sync.ready() || !sync.sequence_valid() ||
        !sync.device() || !sync.queue() || !sync.mutable_kv())
        return fail(error, "SYNC GPU engine unavailable");
    std::error_code file_error;
    // UWP LocalFolder paths are absolute but canonical() is denied by the
    // sandbox on Xbox. SYNC uses the same lexical normalization; size and
    // mtime are checked on both sides before sharing its persistent KV.
    const auto model_path = std::filesystem::path(gguf_path).lexically_normal();
    if (!model_path.is_absolute() ||
        _stricmp(model_path.u8string().c_str(), sync.model_path().c_str()) != 0)
        return fail(error, "DRAFT/SYNC GGUF path identity mismatch");
    const uint64_t initial_size = std::filesystem::file_size(model_path, file_error);
    if (file_error || initial_size != sync.model_file_size())
        return fail(error, "DRAFT/SYNC GGUF size identity mismatch");
    const auto initial_mtime = std::filesystem::last_write_time(model_path, file_error);
    if (file_error || static_cast<int64_t>(initial_mtime.time_since_epoch().count()) !=
                      sync.model_mtime_ticks())
        return fail(error, "DRAFT/SYNC GGUF mtime identity mismatch");
    sync_ = &sync;
    kv_ = sync.mutable_kv();
    device_ = sync.device();
    queue_ = sync.queue();
    barrier_mode_ = BarrierMode::Minimal;
    {
        const auto parent = model_path.parent_path();
        std::ifstream bm_file(parent / "dflash-barrier-mode.txt");
        if (bm_file) {
            std::string mode_str;
            std::getline(bm_file, mode_str);
            if (mode_str.find("strict") != std::string::npos) barrier_mode_ = BarrierMode::Strict;
        }
    }
    HRESULT hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&allocator_));
    if (FAILED(hr)) return fail(error, "CreateCommandAllocator", hr);
    hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        allocator_.Get(), nullptr, IID_PPV_ARGS(&cmd_));
    if (FAILED(hr)) return fail(error, "CreateCommandList", hr);
    hr = cmd_->Close();
    if (FAILED(hr)) return fail(error, "CommandList initial Close", hr);
    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) return fail(error, "CreateFence", hr);
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event_) return fail(error, "CreateEventW failed");
    if (!create_pipelines(error) || !load_weights(gguf_path, error)) return false;
    const uint64_t loaded_size = std::filesystem::file_size(model_path, file_error);
    if (file_error || loaded_size != initial_size)
        return fail(error, "GGUF changed during DRAFT weight load");
    const auto loaded_mtime = std::filesystem::last_write_time(model_path, file_error);
    if (file_error || loaded_mtime != initial_mtime)
        return fail(error, "GGUF changed during DRAFT weight load");

    const uint64_t hidden_bytes = uint64_t(kMaxTokens) * kHidden * 4;
    const uint64_t max_bytes = uint64_t(kMaxTokens) * kFfn * 4;
    const uint64_t position_bytes = kMaxTokens * (1 + 128) * 4;
    const uint64_t active_bytes = 2056u * 4u;
    for (size_t i = 0; i < Count; ++i) {
        const Slot slot = static_cast<Slot>(i);
        uint64_t size = max_bytes;
        if (slot == Embedding || slot == UploadEmbedding || slot == Readback)
            size = hidden_bytes;
        else if (slot == Positions || slot == UploadPositions)
            size = position_bytes;
        else if (slot == ActivePositions)
            size = active_bytes;
        else if (slot == UploadActive)
            size = active_bytes * kLayers;
        else if (slot == NoSink)
            size = kQHeads * 4u + sizeof(dflash_exp::kUcrtExpTable);
        const auto heap = slot == UploadEmbedding || slot == UploadPositions ||
            slot == UploadActive ? D3D12_HEAP_TYPE_UPLOAD :
            slot == Readback ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_DEFAULT;
        const auto flags = heap == D3D12_HEAP_TYPE_DEFAULT &&
            slot != Zero && slot != NoSink && slot != Positions &&
            slot != ActivePositions && slot != Embedding ?
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        const auto state = heap == D3D12_HEAP_TYPE_UPLOAD ?
            D3D12_RESOURCE_STATE_GENERIC_READ :
            heap == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST :
            D3D12_RESOURCE_STATE_COPY_DEST;
        if (!create_buffer(size, heap, flags, state, &buffers_[i], error)) return false;
    }
    Buffer init_zero, init_sink;
    if (!create_buffer(max_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_GENERIC_READ, &init_zero, error) ||
        !create_buffer(kQHeads * 4u + sizeof(dflash_exp::kUcrtExpTable), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_GENERIC_READ, &init_sink, error)) return false;
    void * mapped = nullptr;
    hr = init_zero.resource->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return fail(error, "zero upload Map", hr);
    std::memset(mapped, 0, static_cast<size_t>(max_bytes));
    init_zero.resource->Unmap(0, nullptr);
    hr = init_sink.resource->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return fail(error, "sink upload Map", hr);
    for (uint32_t i = 0; i < kQHeads; ++i) {
        const float no_sink = -std::numeric_limits<float>::infinity();
        std::memcpy(static_cast<uint8_t *>(mapped) + i * 4, &no_sink, 4);
    }
    std::memcpy(static_cast<uint8_t *>(mapped) + kQHeads * 4u,
                dflash_exp::kUcrtExpTable, sizeof(dflash_exp::kUcrtExpTable));
    init_sink.resource->Unmap(0, nullptr);
    if (!begin(error)) return false;
    cmd_->CopyBufferRegion(buffers_[Zero].resource.Get(), 0,
                           init_zero.resource.Get(), 0, max_bytes);
    cmd_->CopyBufferRegion(buffers_[NoSink].resource.Get(), 0,
                           init_sink.resource.Get(), 0, kQHeads * 4u + sizeof(dflash_exp::kUcrtExpTable));
    transition(buffers_[Zero], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               buffers_[NoSink], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    pending_uploads_.push_back(init_zero.resource);
    pending_uploads_.push_back(init_sink.resource);
    if (!submit(error)) return false;
    pending_uploads_.clear();
    ready_ = true;
    sequence_valid_ = true;
    if (error) error->clear();
    return true;
}

bool DflashGpuDraftEngine::matmul(const Weight & weight, Slot input,
                                  Slot output, uint32_t batch,
                                  std::string * error) {
    if (!weight.buffer.resource || !batch || batch > kMaxTokens ||
        !range_ok(buffers_[input].resource.Get(), uint64_t(batch) * weight.columns * 4) ||
        !range_ok(buffers_[output].resource.Get(), uint64_t(batch) * weight.rows * 4))
        return fail(error, "DRAFT matmul buffer range invalid");
    const Slot destination = weight.scale == 1.0f ? output : MatRaw;
    Buffer & x = buffers_[input];
    Buffer & y = buffers_[destination];
    if (&x == &y) return fail(error, "in-place DRAFT matmul unsupported");
    transition(x, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               y, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (weight.type == GGML_TYPE_F32) {
        const uint32_t constants[8] = {batch, weight.rows, weight.columns,
            bits(1.0f), bits(epsilon_), bits(rope_base_), bits(1.0f), 0};
        cmd_->SetComputeRootSignature(ops_root_.Get());
        cmd_->SetPipelineState(f32_matmul_pso_.Get());
        cmd_->SetComputeRoot32BitConstants(0, 8, constants, 0);
        cmd_->SetComputeRootShaderResourceView(1, x.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootShaderResourceView(2, weight.buffer.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootShaderResourceView(3,
            buffers_[Positions].resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootUnorderedAccessView(4, y.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(weight.rows, batch, 1);
    } else {
        uint32_t type_index = 0;
        if (weight.type == GGML_TYPE_Q3_K) type_index = 0;
        else if (weight.type == GGML_TYPE_Q4_K) type_index = 1;
        else if (weight.type == GGML_TYPE_Q5_K) type_index = 2;
        else if (weight.type == GGML_TYPE_Q6_K) type_index = 3;
        else if (weight.type == GGML_TYPE_Q8_0) type_index = 4;
        else if (weight.type == GGML_TYPE_Q2_K) type_index = 5;
        else return fail(error, "unsupported DRAFT weight type");
        if (weight.columns % 256) return fail(error, "Q8_K input width misaligned");
        const uint32_t constants[4] = {weight.rows, weight.columns,
                                       weight.columns / 256, batch};
        Buffer & q8 = buffers_[QuantizedInput];
        if (quant_source_ != input || quant_width_ != weight.columns ||
            quant_batch_ != batch) {
            transition(q8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cmd_->SetComputeRootSignature(matmul_root_.Get());
            cmd_->SetPipelineState(q8_quantize_pso_.Get());
            cmd_->SetComputeRoot32BitConstants(0, 4, constants, 0);
            cmd_->SetComputeRootShaderResourceView(1,
                buffers_[Zero].resource->GetGPUVirtualAddress());
            cmd_->SetComputeRootShaderResourceView(2, x.resource->GetGPUVirtualAddress());
            cmd_->SetComputeRootUnorderedAccessView(3, q8.resource->GetGPUVirtualAddress());
            cmd_->Dispatch(weight.columns / 256, batch, 1);
            if (barrier_mode_ == BarrierMode::Strict) barrier(q8);
            quant_source_ = input;
            quant_width_ = weight.columns;
            quant_batch_ = batch;
        }
        transition(q8, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cmd_->SetComputeRootSignature(matmul_root_.Get());
        cmd_->SetPipelineState(quant_dot_psos_[type_index].Get());
        cmd_->SetComputeRoot32BitConstants(0, 4, constants, 0);
        cmd_->SetComputeRootShaderResourceView(1,
            weight.buffer.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootShaderResourceView(2, q8.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootUnorderedAccessView(3, y.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(weight.rows, (batch + 7) / 8, 1);
    }
    if (destination != output && barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (destination != output)
        scale_add(destination, Zero, output, batch, weight.rows, weight.scale);
    if (quant_source_ == output || quant_source_ == destination)
        quant_source_ = Count;
    return true;
}

void DflashGpuDraftEngine::norm(Slot input, const Weight & weight, Slot output,
                                uint32_t rows, uint32_t width, float scale) {
    Buffer & x = buffers_[input];
    Buffer & y = buffers_[output];
    transition(x, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               y, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[8] = {rows, width, weight.columns, 0,
        bits(scale), bits(epsilon_), bits(rope_base_), 0};
    cmd_->SetComputeRootSignature(ops_root_.Get());
    cmd_->SetPipelineState(norm_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, x.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2,
        weight.buffer.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(3,
        buffers_[Positions].resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(4, y.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(rows, 1, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;
}

void DflashGpuDraftEngine::rope(Slot input, Slot output,
                                uint32_t batch, uint32_t width) {
    Buffer & x = buffers_[input];
    Buffer & y = buffers_[output];
    transition(x, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               y, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[8] = {batch, width, kHead, kMaxTokens * 4,
        bits(1.0f), bits(epsilon_), bits(rope_base_), 0};
    cmd_->SetComputeRootSignature(ops_root_.Get());
    cmd_->SetPipelineState(rope_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, x.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2,
        buffers_[Zero].resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(3,
        buffers_[Positions].resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(4, y.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(width / kHead, batch, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;
}

void DflashGpuDraftEngine::hadamard(Slot input, Slot output,
                                    uint32_t batch, uint32_t heads,
                                    uint32_t transform_width) {
    Buffer & x = buffers_[input];
    Buffer & y = buffers_[output];
    transition(x, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               y, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[4] = {batch, heads, transform_width, 0};
    cmd_->SetComputeRootSignature(matmul_root_.Get());
    cmd_->SetPipelineState(hadamard_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 4, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, x.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2,
        buffers_[Zero].resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(3, y.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(heads, batch, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;
}

void DflashGpuDraftEngine::conv(Slot input, Slot coeff, const Weight & base,
                                Slot output, uint32_t batch, uint32_t side) {
    Buffer & x = buffers_[input];
    Buffer & c = buffers_[coeff];
    Buffer & y = buffers_[output];
    transition(x, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               c, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               y, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[4] = {batch, 1, side, 0};
    cmd_->SetComputeRootSignature(conv_root_.Get());
    cmd_->SetPipelineState(conv_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 4, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, x.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2, c.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(3,
        base.buffer.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(4, y.resource->GetGPUVirtualAddress());
    cmd_->Dispatch((batch * kHidden + 63) / 64, 1, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;
}

void DflashGpuDraftEngine::scale_add(Slot input, Slot residual, Slot output,
                                     uint32_t batch, uint32_t width, float scale) {
    Buffer & x = buffers_[input];
    Buffer & r = buffers_[residual];
    Buffer & y = buffers_[output];
    transition(x, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               y, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[8] = {batch, width, 0, 0,
        bits(scale), bits(epsilon_), bits(rope_base_), 0};
    cmd_->SetComputeRootSignature(ops_root_.Get());
    cmd_->SetPipelineState(scale_add_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, x.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2, r.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(3,
        buffers_[Positions].resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(4, y.resource->GetGPUVirtualAddress());
    cmd_->Dispatch((batch * width + 127) / 128, 1, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;
}

void DflashGpuDraftEngine::swiglu(Slot gate, Slot up, Slot output,
                                  uint32_t batch) {
    Buffer & g = buffers_[gate];
    Buffer & u = buffers_[up];
    Buffer & y = buffers_[output];
    transition(g, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               u, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               y, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[8] = {batch, kFfn, 0, 0,
        bits(1.0f), bits(epsilon_), bits(rope_base_), 0};
    cmd_->SetComputeRootSignature(ops_root_.Get());
    cmd_->SetPipelineState(swiglu_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, g.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2, u.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(3,
        buffers_[Positions].resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(4, y.resource->GetGPUVirtualAddress());
    cmd_->Dispatch((batch * kFfn + 127) / 128, 1, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;
}

bool DflashGpuDraftEngine::pack_kv(uint32_t layer, uint32_t pos0,
                                   uint32_t batch, std::string * error) {
    Buffer & k = buffers_[KRot];
    Buffer & v = buffers_[VRot];
    transition(k, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               v, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    for (uint32_t row = 0; row < batch; ++row) {
        const uint64_t offset = uint64_t(row) * kKvHeads * kHead * 4;
        if (!kv_->record_pack(cmd_.Get(), layer, pos0 + row,
                              DflashGpuKv::Kind::Key, k.resource.Get(), offset, error) ||
            !kv_->record_pack(cmd_.Get(), layer, pos0 + row,
                              DflashGpuKv::Kind::Value, v.resource.Get(), offset, error))
            return false;
    }
    return true;
}

bool DflashGpuDraftEngine::attention(uint32_t layer, uint32_t pos0,
                                     uint32_t batch, const Weight & sinks,
                                     std::string * error) {
    // The SYNC and DRAFT K/V rows are both visible here, including every
    // future row of this noncausal block. The shader applies each query's
    // absolute SWA lower bound. Physical stale rows are never listed.
    std::vector<uint32_t> active;
    active.reserve(2056);
    const uint32_t lower = pos0 > 2047 ? pos0 - 2047 : 0;
    for (uint32_t pos = lower; pos < pos0 + batch; ++pos) {
        if (kv_->valid(layer, pos, DflashGpuKv::Kind::Key) &&
            kv_->valid(layer, pos, DflashGpuKv::Kind::Value))
            active.push_back(pos);
    }
    if (active.empty() || active.size() > 2056)
        return fail(error, "DRAFT attention has no valid KV rows");
    Buffer & upload = buffers_[UploadActive];
    Buffer & positions = buffers_[ActivePositions];
    const uint64_t upload_offset = uint64_t(layer) * 2056u * 4u;
    void * mapped = nullptr;
    HRESULT hr = upload.resource->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return fail(error, "active positions Map", hr);
    std::memcpy(static_cast<uint8_t *>(mapped) + upload_offset,
                active.data(), active.size() * sizeof(uint32_t));
    upload.resource->Unmap(0, nullptr);
    transition(positions, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd_->CopyBufferRegion(positions.resource.Get(), 0,
                           upload.resource.Get(), upload_offset,
                           active.size() * sizeof(uint32_t));
    transition(positions, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!kv_->record_prepare_read(cmd_.Get(), error)) return false;
    Buffer & q = buffers_[QRot];
    Buffer & y = buffers_[AttentionRot];
    transition(q, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               y, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint64_t layer_offset = uint64_t(layer) * kContext * 1280u;
    if (layer_offset > UINT32_MAX) return fail(error, "KV layer address overflow");
    const uint32_t constants[4] = {batch, static_cast<uint32_t>(active.size()),
                                   kContext, static_cast<uint32_t>(layer_offset)};
    cmd_->SetComputeRootSignature(attention_root_.Get());
    cmd_->SetPipelineState(attention_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 4, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, q.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2, kv_->resource()->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(3,
        buffers_[Positions].resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(4,
        positions.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(5,
        (sinks.buffer.resource ? sinks.buffer.resource.Get() :
         buffers_[NoSink].resource.Get())->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(6, y.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(kQHeads, batch, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == AttentionRot) quant_source_ = Count;
    return true;
}

bool DflashGpuDraftEngine::draft(uint32_t pos0, const float * embeddings,
                                 uint32_t n_max, float * hidden,
                                 std::string * error,
                                 std::vector<Checkpoint> * checkpoints) {
    if (!ready_ || !sequence_valid_ || !sync_ || !sync_->sequence_valid() ||
        sync_->mutable_kv() != kv_)
        return fail(error, "DRAFT GPU sequence unavailable");
    if (!embeddings || !hidden || n_max < 1 || n_max > 7 ||
        pos0 >= kContext || n_max >= kContext - pos0)
        return fail(error, "invalid DRAFT arguments");
    const uint32_t batch = n_max + 1;
    for (uint64_t i = 0; i < uint64_t(batch) * kHidden; ++i)
        if (!std::isfinite(embeddings[i]))
            return fail(error, "non-finite DRAFT embedding");
    const auto kv_diag = kv_->diagnostics();
    if (kv_diag.layers != kLayers || kv_diag.capacity != kContext ||
        kv_diag.key_width != kKvHeads * kHead ||
        kv_diag.value_width != kKvHeads * kHead)
        return fail(error, "shared KV geometry mismatch");
    const uint64_t input_bytes = uint64_t(batch) * kHidden * 4;
    void * mapped = nullptr;
    HRESULT hr = buffers_[UploadEmbedding].resource->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return fail(error, "DRAFT input Map", hr);
    std::memcpy(mapped, embeddings, static_cast<size_t>(input_bytes));
    buffers_[UploadEmbedding].resource->Unmap(0, nullptr);
    hr = buffers_[UploadPositions].resource->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return fail(error, "DRAFT positions Map", hr);
    for (uint32_t i = 0; i < batch; ++i) {
        const uint32_t pos = pos0 + i;
        std::memcpy(static_cast<uint8_t *>(mapped) + i * 4, &pos, 4);
        dflash_rope_cache(static_cast<float *>(mapped) + kMaxTokens + i * 128,
                          pos, rope_base_);
    }
    buffers_[UploadPositions].resource->Unmap(0, nullptr);

    auto invalidate = [&]() {
        sequence_valid_ = false;
        sync_->invalidate_sequence();
    };
    if (!begin(error)) { invalidate(); return false; }
    quant_source_ = Count;
    pending_checkpoints_.clear();
    if (checkpoints) checkpoints->clear();
    trace_frequency_ = 0;
    if (checkpoints) {
        if (!trace_queries_) {
            const D3D12_QUERY_HEAP_DESC desc = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 256, 0};
            device_->CreateQueryHeap(&desc, IID_PPV_ARGS(&trace_queries_));
        }
        if (trace_queries_ && SUCCEEDED(queue_->GetTimestampFrequency(&trace_frequency_)) && trace_frequency_) {
            if (!trace_times_.resource && !create_buffer(256 * sizeof(uint64_t), D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &trace_times_, error)) {
                invalidate(); return false;
            }
            cmd_->EndQuery(trace_queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        } else {
            trace_frequency_ = 0;
        }
    }
    auto snapshot = [&](const char * name, Slot slot, uint32_t width) {
        return !checkpoints || record_checkpoint(name, slot, batch, width, error);
    };
    auto layer_snapshot = [&](const char * name, uint32_t layer,
                              Slot slot, uint32_t width) {
        if (!checkpoints) return true;
        const std::string label = std::string(name) + '-' + std::to_string(layer);
        return snapshot(label.c_str(), slot, width);
    };
    // Equivalent to llama_memory_seq_rm(seq=0, pos0, -1) before DRAFT.
    // The logical trim precedes any new K/V record. Any failure from here
    // invalidates the shared sequence; the RPC must close rather than recover
    // on stale CPU state.
    if (!kv_->trim(pos0)) {
        invalidate();
        return fail(error, "DRAFT KV suffix trim failed");
    }
    transition(buffers_[Embedding], D3D12_RESOURCE_STATE_COPY_DEST);
    transition(buffers_[Positions], D3D12_RESOURCE_STATE_COPY_DEST);
    cmd_->CopyBufferRegion(buffers_[Embedding].resource.Get(), 0,
        buffers_[UploadEmbedding].resource.Get(), 0, input_bytes);
    cmd_->CopyBufferRegion(buffers_[Positions].resource.Get(), 0,
        buffers_[UploadPositions].resource.Get(), 0, (kMaxTokens + batch * 128) * 4);
    transition(buffers_[Positions], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    scale_add(Embedding, Zero, HiddenA, batch, kHidden, embedding_scale_);
    if (!snapshot("inp_noise_embd", HiddenA, kHidden)) { invalidate(); return false; }

    for (uint32_t il = 0; il < kLayers; ++il) {
        const Layer & layer = layers_[il];
        norm(HiddenA, layer.attn_norm, Norm, batch, kHidden);
        if (il == 0 && !snapshot("noise_norm-0", Norm, kHidden)) {
            invalidate(); return false;
        }
        if (il != 0 && !layer_snapshot("noise_norm", il, Norm, kHidden)) {
            invalidate(); return false;
        }
        if (!matmul(layer.attn_conv_proj, Norm, ConvCoeff, batch, error)) {
            invalidate(); return false;
        }
        if (!layer_snapshot("attn_conv_coeff", il, ConvCoeff, 1280)) {
            invalidate(); return false;
        }
        conv(Norm, ConvCoeff, layer.attn_conv_base, ConvIn, batch, 0);
        if (il == 0 && !snapshot("attn_conv_in-0", ConvIn, kHidden)) {
            invalidate(); return false;
        }
        if (il != 0 && !layer_snapshot("attn_conv_in", il, ConvIn, kHidden)) {
            invalidate(); return false;
        }
        if (!matmul(layer.q, ConvIn, Q, batch, error) ||
            !matmul(layer.k, ConvIn, K, batch, error) ||
            !matmul(layer.v, ConvIn, V, batch, error)) {
            invalidate(); return false;
        }
        norm(Q, layer.q_norm, QNorm, batch * kQHeads, kHead);
        norm(K, layer.k_norm, KNorm, batch * kKvHeads, kHead);
        rope(QNorm, QRope, batch, kQHeads * kHead);
        rope(KNorm, KRope, batch, kKvHeads * kHead);
        if (il == 0 && (!snapshot("Qcur-0", QRope, kQHeads * kHead) ||
                        !snapshot("Kcur-0", KRope, kKvHeads * kHead) ||
                        !snapshot("Vcur-0", V, kKvHeads * kHead))) {
            invalidate(); return false;
        }
        if (il != 0 && (!layer_snapshot("Qcur", il, QRope, kQHeads * kHead) ||
                        !layer_snapshot("Kcur", il, KRope, kKvHeads * kHead) ||
                        !layer_snapshot("Vcur", il, V, kKvHeads * kHead))) {
            invalidate(); return false;
        }
        // GGML's Q4_0 KV cache rotates K and Q with H128, but V with H64
        // independently in each half-head. H is orthonormal and involutive.
        hadamard(QRope, QRot, batch, kQHeads, 128);
        hadamard(KRope, KRot, batch, kKvHeads, 128);
        hadamard(V, VRot, batch, kKvHeads, 64);
        if (!layer_snapshot("q_h128", il, QRot, kQHeads * kHead) ||
            !layer_snapshot("k_h128", il, KRot, kKvHeads * kHead) ||
            !layer_snapshot("v_h64", il, VRot, kKvHeads * kHead)) {
            invalidate(); return false;
        }
        if (!pack_kv(il, pos0, batch, error) ||
            !attention(il, pos0, batch, layer.sinks, error)) {
            invalidate(); return false;
        }
        if (!layer_snapshot("kqv_out", il, AttentionRot, kQHeads * kHead)) {
            invalidate(); return false;
        }
        hadamard(AttentionRot, Attention, batch, kQHeads, 64);
        if (!layer_snapshot("out_h64", il, Attention, kQHeads * kHead)) {
            invalidate(); return false;
        }
        if (!matmul(layer.o, Attention, AttnOut, batch, error)) {
            invalidate(); return false;
        }
        conv(AttnOut, ConvCoeff, layer.attn_conv_base, ConvOut, batch, 1);
        if (!layer_snapshot("attn_conv_out", il, ConvOut, kHidden)) {
            invalidate(); return false;
        }
        scale_add(ConvOut, HiddenA, FfnIn, batch, kHidden);
        if (!layer_snapshot("ffn_inp", il, FfnIn, kHidden)) {
            invalidate(); return false;
        }
        norm(FfnIn, layer.ffn_norm, FfnNorm, batch, kHidden);
        if (!layer_snapshot("ffn_norm", il, FfnNorm, kHidden)) {
            invalidate(); return false;
        }
        if (!matmul(layer.ffn_conv_proj, FfnNorm, FfnCoeff, batch, error)) {
            invalidate(); return false;
        }
        if (!layer_snapshot("ffn_conv_coeff", il, FfnCoeff, 1280)) {
            invalidate(); return false;
        }
        conv(FfnNorm, FfnCoeff, layer.ffn_conv_base, FfnConv, batch, 0);
        if (!layer_snapshot("ffn_conv_in", il, FfnConv, kHidden)) {
            invalidate(); return false;
        }
        if (!matmul(layer.gate, FfnConv, Gate, batch, error) ||
            !matmul(layer.up, FfnConv, Up, batch, error)) {
            invalidate(); return false;
        }
        if (!layer_snapshot("ffn_gate", il, Gate, kFfn) ||
            !layer_snapshot("ffn_up", il, Up, kFfn)) {
            invalidate(); return false;
        }
        swiglu(Gate, Up, Activated, batch);
        if (!layer_snapshot("ffn_swiglu", il, Activated, kFfn)) {
            invalidate(); return false;
        }
        if (!matmul(layer.down, Activated, Down, batch, error)) {
            invalidate(); return false;
        }
        if (!layer_snapshot("ffn_out", il, Down, kHidden)) {
            invalidate(); return false;
        }
        conv(Down, FfnCoeff, layer.ffn_conv_base, ConvOut, batch, 1);
        if (!layer_snapshot("ffn_conv_out", il, ConvOut, kHidden)) {
            invalidate(); return false;
        }
        scale_add(ConvOut, FfnIn, HiddenB, batch, kHidden);
        if (!layer_snapshot("l_out", il, HiddenB, kHidden)) {
            invalidate(); return false;
        }
        std::swap(buffers_[HiddenA], buffers_[HiddenB]);
        quant_source_ = Count;
    }
    norm(HiddenA, output_norm_, Norm, batch, kHidden);
    if (!snapshot("output_norm", Norm, kHidden)) { invalidate(); return false; }
    transition(buffers_[Norm], D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmd_->CopyBufferRegion(buffers_[Readback].resource.Get(), 0,
                            buffers_[Norm].resource.Get(), 0, input_bytes);
    const uint32_t query_count = 2u * static_cast<uint32_t>(pending_checkpoints_.size()) + 1u;
    if (trace_frequency_) cmd_->ResolveQueryData(trace_queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
            0, query_count, trace_times_.resource.Get(), 0);
    if (!submit(error)) { invalidate(); return false; }
    if (checkpoints) {
        std::vector<uint64_t> times(query_count);
        if (trace_frequency_) {
            void * data = nullptr;
            const D3D12_RANGE read = {0, query_count * sizeof(uint64_t)};
            hr = trace_times_.resource->Map(0, &read, &data);
            if (FAILED(hr)) { invalidate(); return fail(error, "DRAFT trace timestamps Map", hr); }
            std::memcpy(times.data(), data, read.End);
            const D3D12_RANGE empty = {0, 0};
            trace_times_.resource->Unmap(0, &empty);
        }
        uint32_t query = 0;
        for (const PendingCheckpoint & checkpoint : pending_checkpoints_) {
            Checkpoint sample;
            sample.name = checkpoint.name;
            if (trace_frequency_ && times[query + 1] >= times[query])
                sample.gpu_ms = 1000.0 * double(times[query + 1] - times[query]) / double(trace_frequency_);
            query += 2;
            sample.values.resize(static_cast<size_t>(checkpoint.bytes / sizeof(float)));
            void * data = nullptr;
            const D3D12_RANGE read = {0, static_cast<SIZE_T>(checkpoint.bytes)};
            hr = checkpoint.readback.resource->Map(0, &read, &data);
            if (FAILED(hr)) {
                invalidate(); return fail(error, "DRAFT checkpoint Map", hr);
            }
            std::memcpy(sample.values.data(), data,
                        static_cast<size_t>(checkpoint.bytes));
            const D3D12_RANGE empty = {0, 0};
            checkpoint.readback.resource->Unmap(0, &empty);
            checkpoints->push_back(std::move(sample));
        }
    }
    pending_checkpoints_.clear();
    const D3D12_RANGE read_range = {0, static_cast<SIZE_T>(input_bytes)};
    hr = buffers_[Readback].resource->Map(0, &read_range, &mapped);
    if (FAILED(hr)) { invalidate(); return fail(error, "DRAFT readback Map", hr); }
    std::memcpy(hidden, mapped, static_cast<size_t>(input_bytes));
    const D3D12_RANGE empty = {0, 0};
    buffers_[Readback].resource->Unmap(0, &empty);
    for (uint64_t i = 0; i < uint64_t(batch) * kHidden; ++i)
        if (!std::isfinite(hidden[i])) {
            invalidate();
            return fail(error, "non-finite DRAFT hidden state");
        }
    if (error) error->clear();
    return true;
}
