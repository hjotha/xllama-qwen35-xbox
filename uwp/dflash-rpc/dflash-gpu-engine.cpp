#include "dflash-gpu-engine.h"

#include "dflash_q3k_t1_dxil.h"
#include "dflash_q3k_t8_dxil.h"
#include "dflash_q3k_q8k_dxil.h"
#include "dflash_draft_quant_q8_dxil.h"
#include "dflash_q6k_q8k_dxil.h"
#include "dflash_q8_0_q8k_dxil.h"
#include "dflash_q4k_t2_dxil.h"
#include "dflash_q4k_t8_dxil.h"
#include "dflash_q5k_t1_dxil.h"
#include "dflash_q5k_t8_dxil.h"
#include "dflash_sync_ops_dxil.h"
#include "dflash_hadamard128_dxil.h"
#include "gguf.h"
#include "xllama/d3d12_dyn.h"

#include <dxgi1_4.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>

using Microsoft::WRL::ComPtr;

namespace {

uint32_t bits(float value) {
    uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    return word;
}

uint64_t aligned4(uint64_t value) { return (value + 3u) & ~uint64_t(3u); }

bool buffer_range(ID3D12Resource * resource, uint64_t bytes) {
    return resource && resource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
           resource->GetDesc().Width >= bytes && resource->GetGPUVirtualAddress() != 0;
}

} // namespace

DflashGpuSyncEngine::~DflashGpuSyncEngine() {
    if (unconfirmed_submission_ && fence_ && fence_->GetCompletedValue() < fence_value_) {
        // ExecuteCommandLists has no HRESULT. Signal, event registration or
        // the wait can fail after the GPU has accepted the list. Keep every
        // referenced object alive until process exit rather than releasing a
        // possibly executing upload/readback or KV resource.
        struct Retained {
            ComPtr<IUnknown> objects[64];
            size_t used = 0;
            std::vector<ComPtr<ID3D12Resource>> uploads;
            DflashGpuKv kv;
            HANDLE event = nullptr;
        };
        auto * retained = new (std::nothrow) Retained;
        if (!retained) std::terminate();
        auto hold = [&](IUnknown * object) {
            if (!object) return;
            if (retained->used >= 64) std::terminate();
            object->AddRef();
            retained->objects[retained->used++].Attach(object);
        };
        hold(device_.Get()); hold(queue_.Get()); hold(allocator_.Get());
        hold(cmd_.Get()); hold(fence_.Get());
        hold(matmul_root_.Get()); hold(ops_root_.Get());
        for (auto & row : matmul_psos_)
            for (auto & pso : row) hold(pso.Get());
        hold(q8_pack_pso_.Get()); hold(q3q8_pso_.Get());
        hold(q4q8_pso_.Get()); hold(q5q8_pso_.Get());
        hold(q6q8_pso_.Get()); hold(q8_0q8_pso_.Get());
        hold(norm_pso_.Get()); hold(key_pso_.Get()); hold(scale_pso_.Get());
        hold(hadamard_pso_.Get());
        auto hold_buffer = [&](Buffer & buffer) { hold(buffer.resource.Get()); };
        hold_buffer(fc_.buffer); hold_buffer(enc_norm_.buffer);
        for (auto & layer : layers_) {
            hold_buffer(layer.key.buffer); hold_buffer(layer.value.buffer);
            hold_buffer(layer.key_norm.buffer);
        }
        hold_buffer(input_); hold_buffer(input_q8_); hold_buffer(positions_);
        hold_buffer(fc_output_); hold_buffer(encoder_); hold_buffer(encoder_q8_);
        hold_buffer(key_); hold_buffer(value_); hold_buffer(scaled_value_);
        hold_buffer(rotated_key_); hold_buffer(input_upload_);
        hold_buffer(hadamard_key_); hold_buffer(hadamard_value_);
        hold_buffer(position_upload_);
        retained->kv = std::move(kv_);
        hold(pending_readback_.Get());
        retained->uploads = std::move(pending_uploads_);
        retained->event = fence_event_;
        fence_event_ = nullptr;
        (void) retained; // Intentional process-lifetime quarantine.
    }
    if (fence_event_) CloseHandle(fence_event_);
}

bool DflashGpuSyncEngine::fail(std::string * error, const char * stage, HRESULT hr) {
    if (error) {
        char message[192];
        if (hr == S_OK) std::snprintf(message, sizeof(message), "%s", stage);
        else std::snprintf(message, sizeof(message), "%s: 0x%08X", stage,
                           static_cast<unsigned>(hr));
        *error = message;
    }
    return false;
}

bool DflashGpuSyncEngine::create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap,
                                       D3D12_RESOURCE_FLAGS flags,
                                       D3D12_RESOURCE_STATES state,
                                       Buffer * out, std::string * error) {
    if (!device_ || !out || !bytes || bytes > std::numeric_limits<UINT64>::max() - 3)
        return fail(error, "invalid buffer size");
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = aligned4(bytes);
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    const HRESULT hr = device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE,
        &rd, state, nullptr, IID_PPV_ARGS(&resource));
    if (FAILED(hr)) return fail(error, "CreateCommittedResource", hr);
    out->resource = std::move(resource);
    out->bytes = rd.Width;
    out->state = state;
    return true;
}

void DflashGpuSyncEngine::transition(Buffer & buffer, D3D12_RESOURCE_STATES state) {
    if (buffer.state == state) return;
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer.resource.Get();
    barrier.Transition.StateBefore = buffer.state;
    barrier.Transition.StateAfter = state;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_->ResourceBarrier(1, &barrier);
    buffer.state = state;
}

void DflashGpuSyncEngine::uav_barrier(Buffer & buffer) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = buffer.resource.Get();
    cmd_->ResourceBarrier(1, &barrier);
}

bool DflashGpuSyncEngine::begin(std::string * error) {
    if (terminal_failure_ || unconfirmed_submission_)
        return fail(error, "GPU executor terminal after unconfirmed submission");
    HRESULT hr = allocator_->Reset();
    if (FAILED(hr)) return fail(error, "CommandAllocator Reset", hr);
    hr = cmd_->Reset(allocator_.Get(), nullptr);
    if (FAILED(hr)) return fail(error, "CommandList Reset", hr);
    return true;
}

bool DflashGpuSyncEngine::submit(std::string * error) {
    if (terminal_failure_ || unconfirmed_submission_)
        return fail(error, "GPU executor terminal after unconfirmed submission");
    HRESULT hr = cmd_->Close();
    if (FAILED(hr)) return fail(error, "CommandList Close", hr);
    ID3D12CommandList * lists[] = {cmd_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    unconfirmed_submission_ = true;
    const UINT64 value = ++fence_value_;
    hr = queue_->Signal(fence_.Get(), value);
    if (FAILED(hr)) {
        terminal_failure_ = true;
        sequence_valid_ = false;
        return fail(error, "CommandQueue Signal", hr);
    }
    if (fence_->GetCompletedValue() < value) {
        hr = fence_->SetEventOnCompletion(value, fence_event_);
        if (FAILED(hr)) {
            terminal_failure_ = true;
            sequence_valid_ = false;
            return fail(error, "Fence SetEventOnCompletion", hr);
        }
        const DWORD wait = WaitForSingleObject(fence_event_, 60000);
        if (wait != WAIT_OBJECT_0) {
            terminal_failure_ = true;
            sequence_valid_ = false;
            return fail(error, "GPU fence wait failed");
        }
    }
    if (fence_->GetCompletedValue() < value) {
        const HRESULT removed = device_->GetDeviceRemovedReason();
        terminal_failure_ = true;
        sequence_valid_ = false;
        return fail(error, "GPU execution did not complete", removed);
    }
    unconfirmed_submission_ = false;
    pending_uploads_.clear();
    hr = device_->GetDeviceRemovedReason();
    if (FAILED(hr)) {
        terminal_failure_ = true;
        sequence_valid_ = false;
        return fail(error, "D3D12 device removed", hr);
    }
    return true;
}

bool DflashGpuSyncEngine::create_pipelines(std::string * error) {
    auto serialize = xllama::d3d12_dyn::SerializeRootSignature();
    if (!serialize) return fail(error, "D3D12SerializeRootSignature unavailable");
    auto root = [&](uint32_t constants, bool positions,
                    ComPtr<ID3D12RootSignature> * target) -> bool {
        D3D12_ROOT_PARAMETER params[5] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.ShaderRegister = 0;
        params[0].Constants.Num32BitValues = constants;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1].Descriptor.ShaderRegister = 0;
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[2].Descriptor.ShaderRegister = 1;
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[3].Descriptor.ShaderRegister = 0;
        if (positions) {
            params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            params[4].Descriptor.ShaderRegister = 2;
        }
        D3D12_ROOT_SIGNATURE_DESC desc = {};
        desc.NumParameters = positions ? 5 : 4;
        desc.pParameters = params;
        ComPtr<ID3DBlob> blob, messages;
        HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &messages);
        if (FAILED(hr)) return fail(error, "SerializeRootSignature", hr);
        hr = device_->CreateRootSignature(0, blob->GetBufferPointer(),
                                          blob->GetBufferSize(), IID_PPV_ARGS(target->GetAddressOf()));
        return SUCCEEDED(hr) ? true : fail(error, "CreateRootSignature", hr);
    };
    if (!root(4, false, &matmul_root_) || !root(8, true, &ops_root_)) return false;
    auto pso = [&](ID3D12RootSignature * signature, const uint8_t * code,
                   size_t bytes, ComPtr<ID3D12PipelineState> * target) -> bool {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = signature;
        desc.CS = {code, bytes};
        const HRESULT hr = device_->CreateComputePipelineState(&desc,
                                                                IID_PPV_ARGS(target->GetAddressOf()));
        return SUCCEEDED(hr) ? true : fail(error, "CreateComputePipelineState", hr);
    };
    if (!pso(matmul_root_.Get(), kDflashQ3kT1Dxil, kDflashQ3kT1DxilSize, &matmul_psos_[0][0]) ||
        !pso(matmul_root_.Get(), kDflashQ3kT8Dxil, kDflashQ3kT8DxilSize, &matmul_psos_[0][1]) ||
        !pso(matmul_root_.Get(), kDflashQ4kT2Dxil, kDflashQ4kT2DxilSize, &matmul_psos_[1][0]) ||
        !pso(matmul_root_.Get(), kDflashQ4kT8Dxil, kDflashQ4kT8DxilSize, &matmul_psos_[1][1]) ||
        !pso(matmul_root_.Get(), kDflashQ5kT1Dxil, kDflashQ5kT1DxilSize, &matmul_psos_[2][0]) ||
        !pso(matmul_root_.Get(), kDflashQ5kT8Dxil, kDflashQ5kT8DxilSize, &matmul_psos_[2][1]) ||
        !pso(matmul_root_.Get(), kDflashQ8kQuantizeExactDxil, kDflashQ8kQuantizeExactDxilSize, &q8_pack_pso_) ||
        !pso(matmul_root_.Get(), kDflashQ3kQ8kDotDxil, kDflashQ3kQ8kDotDxilSize, &q3q8_pso_) ||
        !pso(matmul_root_.Get(), kDflashDraftQ4kQ8kDotDxil, kDflashDraftQ4kQ8kDotDxilSize, &q4q8_pso_) ||
        !pso(matmul_root_.Get(), kDflashDraftQ5kQ8kDotDxil, kDflashDraftQ5kQ8kDotDxilSize, &q5q8_pso_) ||
        !pso(matmul_root_.Get(), kDflashQ6kQ8kDotDxil, kDflashQ6kQ8kDotDxilSize, &q6q8_pso_) ||
        !pso(matmul_root_.Get(), kDflashQ8_0Q8kDotDxil, kDflashQ8_0Q8kDotDxilSize, &q8_0q8_pso_) ||
        !pso(matmul_root_.Get(), kDflashHadamard128Dxil, kDflashHadamard128DxilSize, &hadamard_pso_) ||
        !pso(ops_root_.Get(), kDflashSyncF32MatmulDxil, kDflashSyncF32MatmulDxilSize, &matmul_psos_[3][0]) ||
        !pso(ops_root_.Get(), kDflashSyncFcNormDxil, kDflashSyncFcNormDxilSize, &norm_pso_) ||
        !pso(ops_root_.Get(), kDflashSyncKeyNormRopeDxil, kDflashSyncKeyNormRopeDxilSize, &key_pso_) ||
        !pso(ops_root_.Get(), kDflashSyncScaleRowsDxil, kDflashSyncScaleRowsDxilSize, &scale_pso_))
        return false;
    matmul_psos_[3][1] = matmul_psos_[3][0];
    return true;
}

bool DflashGpuSyncEngine::load_weights(const std::string & model_path,
                                      std::string * error) {
    using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    gguf_ptr gguf(gguf_init_from_file(model_path.c_str(), {true, nullptr}), &gguf_free);
    if (!gguf) return fail(error, "GGUF metadata load failed");
    auto key = [&](const char * name) { return gguf_find_key(gguf.get(), name); };
    auto required_u32 = [&](const char * name, uint32_t expected) {
        const int64_t id = key(name);
        return id >= 0 && gguf_get_kv_type(gguf.get(), id) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(gguf.get(), id) == expected;
    };
    const int64_t architecture_id = key("general.architecture");
    if (architecture_id < 0 ||
        gguf_get_kv_type(gguf.get(), architecture_id) != GGUF_TYPE_STRING ||
        std::strcmp(gguf_get_val_str(gguf.get(), architecture_id), "dflash") != 0 ||
        !required_u32("dflash.block_count", kLayers) ||
        !required_u32("dflash.embedding_length", kHiddenWidth) ||
        !required_u32("dflash.attention.head_count", 32) ||
        !required_u32("dflash.attention.head_count_kv", kKvHeads) ||
        !required_u32("dflash.attention.key_length", kHeadWidth) ||
        !required_u32("dflash.attention.value_length", kHeadWidth))
        return fail(error, "unsupported DFlash2 model geometry");
    const int64_t epsilon_id = key("dflash.attention.layer_norm_rms_epsilon");
    const int64_t rope_id = key("dflash.rope.freq_base");
    const int64_t sections_id = key("dflash.rope.dimension_sections");
    if (epsilon_id < 0 || rope_id < 0 || sections_id < 0 ||
        gguf_get_kv_type(gguf.get(), epsilon_id) != GGUF_TYPE_FLOAT32 ||
        gguf_get_kv_type(gguf.get(), rope_id) != GGUF_TYPE_FLOAT32 ||
        gguf_get_arr_type(gguf.get(), sections_id) != GGUF_TYPE_INT32 ||
        gguf_get_arr_n(gguf.get(), sections_id) != 4)
        return fail(error, "unsupported RMS/RoPE metadata");
    const auto * sections = static_cast<const int32_t *>(gguf_get_arr_data(gguf.get(), sections_id));
    if (!sections || sections[0] != 64 || sections[1] || sections[2] || sections[3])
        return fail(error, "unsupported M-RoPE sections");
    epsilon_ = gguf_get_val_f32(gguf.get(), epsilon_id);
    rope_base_ = gguf_get_val_f32(gguf.get(), rope_id);
    if (!std::isfinite(epsilon_) || epsilon_ <= 0 ||
        !std::isfinite(rope_base_) || rope_base_ <= 1)
        return fail(error, "invalid RMS/RoPE metadata");
    // This kernel implements the unscaled temporal M-RoPE used by the exact
    // reference model. Reject scaling metadata rather than silently changing it.
    if (key("dflash.rope.scaling.type") >= 0 ||
        key("dflash.rope.scaling.factor") >= 0 ||
        key("dflash.rope.freq_scale") >= 0)
        return fail(error, "scaled RoPE requires a separate validated kernel");

    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) return fail(error, "GGUF file open failed");
    const uint64_t file_size = static_cast<uint64_t>(file.tellg());
    const uint64_t data_offset = gguf_get_data_offset(gguf.get());
    if (!begin(error)) return false;
    auto read_tensor = [&](const std::string & name, uint32_t cols,
                           uint32_t rows, bool norm, Weight * output) -> bool {
        const int64_t id = gguf_find_tensor(gguf.get(), name.c_str());
        if (id < 0) return fail(error, ("missing tensor " + name).c_str());
        const auto type = gguf_get_tensor_type(gguf.get(), id);
        if (norm ? type != GGML_TYPE_F32 :
            (type != GGML_TYPE_Q3_K && type != GGML_TYPE_Q4_K &&
             type != GGML_TYPE_Q5_K && type != GGML_TYPE_Q6_K &&
             type != GGML_TYPE_Q8_0 && type != GGML_TYPE_F32))
            return fail(error, ("unsupported tensor type " + name).c_str());
        const int64_t * ne = gguf_get_tensor_ne(gguf.get(), id);
        if (!ne || ne[0] != cols || ne[1] != rows || ne[2] != 1 || ne[3] != 1)
            return fail(error, ("tensor shape mismatch " + name).c_str());
        const uint64_t bytes = gguf_get_tensor_size(gguf.get(), id);
        const uint64_t expected = norm || type == GGML_TYPE_F32 ? uint64_t(cols) * rows * 4 :
            uint64_t(cols) * rows / 256 *
            (type == GGML_TYPE_Q3_K ? 110 : type == GGML_TYPE_Q4_K ? 144 :
             type == GGML_TYPE_Q5_K ? 176 : type == GGML_TYPE_Q6_K ? 210 : 272);
        if (!bytes || bytes != expected) return fail(error, ("tensor byte count mismatch " + name).c_str());
        const uint64_t relative = gguf_get_tensor_offset(gguf.get(), id);
        if (relative > std::numeric_limits<uint64_t>::max() - data_offset)
            return fail(error, ("tensor offset overflow " + name).c_str());
        const uint64_t offset = data_offset + relative;
        if (offset > file_size || bytes > file_size - offset ||
            bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
            return fail(error, ("tensor file range invalid " + name).c_str());
        Buffer upload;
        if (!create_buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, &upload, error) ||
            !create_buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COPY_DEST, &output->buffer, error))
            return false;
        void * mapped = nullptr;
        HRESULT hr = upload.resource->Map(0, nullptr, &mapped);
        if (FAILED(hr)) return fail(error, "GGUF upload Map", hr);
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(static_cast<char *>(mapped), static_cast<std::streamsize>(bytes));
        const bool read_ok = static_cast<bool>(file);
        if (aligned4(bytes) > bytes)
            std::memset(static_cast<uint8_t *>(mapped) + bytes, 0,
                        static_cast<size_t>(aligned4(bytes) - bytes));
        upload.resource->Unmap(0, nullptr);
        if (!read_ok) return fail(error, ("GGUF tensor read failed " + name).c_str());
        cmd_->CopyBufferRegion(output->buffer.resource.Get(), 0,
                               upload.resource.Get(), 0, aligned4(bytes));
        transition(output->buffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        pending_uploads_.push_back(std::move(upload.resource));
        output->columns = cols;
        output->rows = rows;
        output->type = static_cast<uint32_t>(type);
        return true;
    };
    auto read_scalar = [&](const std::string & name, float * value) -> bool {
        const int64_t id = gguf_find_tensor(gguf.get(), name.c_str());
        if (id < 0) { *value = 1.0f; return true; }
        const int64_t * ne = gguf_get_tensor_ne(gguf.get(), id);
        if (gguf_get_tensor_type(gguf.get(), id) != GGML_TYPE_F32 ||
            !ne || ne[0] != 1 || ne[1] != 1 || gguf_get_tensor_size(gguf.get(), id) != 4)
            return fail(error, ("invalid scalar " + name).c_str());
        const uint64_t relative = gguf_get_tensor_offset(gguf.get(), id);
        if (relative > std::numeric_limits<uint64_t>::max() - data_offset)
            return fail(error, ("scalar offset overflow " + name).c_str());
        const uint64_t offset = data_offset + relative;
        if (offset > file_size || 4 > file_size - offset)
            return fail(error, ("scalar file range invalid " + name).c_str());
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(reinterpret_cast<char *>(value), 4);
        if (!file || !std::isfinite(*value))
            return fail(error, ("scalar read failed " + name).c_str());
        return true;
    };
    if (!read_tensor("fc.weight", kFeatureWidth, kHiddenWidth, false, &fc_) ||
        !read_tensor("enc.output_norm.weight", kHiddenWidth, 1, true, &enc_norm_) ||
        !read_scalar("fc.scale", &fc_scale_)) return false;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        Layer & current = layers_[layer];
        if (!read_tensor(prefix + "attn_k.weight", kHiddenWidth, kKvHeads * kHeadWidth,
                         false, &current.key) ||
            !read_tensor(prefix + "attn_v.weight", kHiddenWidth, kKvHeads * kHeadWidth,
                         false, &current.value) ||
            !read_tensor(prefix + "attn_k_norm.weight", kHeadWidth, 1,
                         true, &current.key_norm) ||
            !read_scalar(prefix + "attn_k.scale", &current.key_scale) ||
            !read_scalar(prefix + "attn_v.scale", &current.value_scale)) return false;
    }
    if (!submit(error)) return false;
    // submit() releases pending uploads only after the fence completes.
    return true;
}

bool DflashGpuSyncEngine::initialize(const std::string & model_path,
                                     std::string * error, ActivationMode mode) {
    if (device_) return fail(error, "GPU executor already initialized");
    std::error_code file_error;
    const uint64_t initial_size = std::filesystem::file_size(model_path, file_error);
    if (file_error) return fail(error, "GGUF file size unavailable");
    const auto initial_mtime = std::filesystem::last_write_time(model_path, file_error);
    if (file_error) return fail(error, "GGUF modification time unavailable");
    // UWP can read/stat LocalState files while canonical() is unavailable.
    // The RPC supplies an absolute physical path; normalize it without an
    // additional filesystem resolution and pair it with size + mtime below.
    const auto normalized_path = std::filesystem::path(model_path).lexically_normal();
    if (!normalized_path.is_absolute() || normalized_path.empty())
        return fail(error, "GGUF absolute path unavailable");
    // The CPU cache creates these rotations for quantized Q4_0 K/V unless
    // LLAMA_ATTN_ROT_DISABLE is set. Never share a KV with different bases.
    char * rotation_disabled = nullptr;
    size_t rotation_env_bytes = 0;
    if (_dupenv_s(&rotation_disabled, &rotation_env_bytes,
                  "LLAMA_ATTN_ROT_DISABLE") != 0)
        return fail(error, "GPU KV rotation environment unavailable");
    const bool disable_rotation = rotation_disabled && std::atoi(rotation_disabled) != 0;
    std::free(rotation_disabled);
    if (disable_rotation)
        return fail(error, "GPU KV rotation disabled in CPU runtime");
    activation_mode_ = mode;
    auto create_device = xllama::d3d12_dyn::CreateDevice();
    if (!create_device) return fail(error, "D3D12CreateDevice unavailable");
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return fail(error, "CreateDXGIFactory1", hr);
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        hr = create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_));
        if (SUCCEEDED(hr)) break;
    }
    if (!device_) return fail(error, "no hardware D3D12 adapter");
    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_));
    if (FAILED(hr)) return fail(error, "CreateCommandQueue", hr);
    hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          IID_PPV_ARGS(&allocator_));
    if (FAILED(hr)) return fail(error, "CreateCommandAllocator", hr);
    hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(),
                                    nullptr, IID_PPV_ARGS(&cmd_));
    if (FAILED(hr)) return fail(error, "CreateCommandList", hr);
    hr = cmd_->Close();
    if (FAILED(hr)) return fail(error, "CommandList initial Close", hr);
    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) return fail(error, "CreateFence", hr);
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event_) return fail(error, "CreateEventW failed");
    if (!create_pipelines(error) || !load_weights(model_path, error)) return false;
    constexpr uint64_t input_bytes = uint64_t(kBatch) * kFeatureWidth * 4;
    constexpr uint64_t encoder_bytes = uint64_t(kBatch) * kHiddenWidth * 4;
    constexpr uint64_t kv_bytes = uint64_t(kBatch) * kKvHeads * kHeadWidth * 4;
    constexpr uint64_t input_q8_bytes = uint64_t(kBatch) * (kFeatureWidth / 256) * 292;
    constexpr uint64_t encoder_q8_bytes = uint64_t(kBatch) * (kHiddenWidth / 256) * 292;
    if (!create_buffer(input_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COPY_DEST, &input_, error) ||
        !create_buffer(input_q8_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &input_q8_, error) ||
        !create_buffer(kBatch * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COPY_DEST, &positions_, error) ||
        !create_buffer(input_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_GENERIC_READ, &input_upload_, error) ||
        !create_buffer(kBatch * 4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_GENERIC_READ, &position_upload_, error) ||
        !create_buffer(encoder_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &fc_output_, error) ||
        !create_buffer(encoder_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &encoder_, error) ||
        !create_buffer(encoder_q8_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &encoder_q8_, error) ||
        !create_buffer(kv_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &key_, error) ||
        !create_buffer(kv_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &value_, error) ||
        !create_buffer(kv_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &rotated_key_, error) ||
        !create_buffer(kv_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &hadamard_key_, error) ||
        !create_buffer(kv_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &hadamard_value_, error) ||
        !create_buffer(kv_bytes, D3D12_HEAP_TYPE_DEFAULT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &scaled_value_, error) ||
        !kv_.initialize(device_.Get(), kLayers, kContext,
                        kKvHeads * kHeadWidth, kKvHeads * kHeadWidth, error)) return false;
    const uint64_t final_size = std::filesystem::file_size(model_path, file_error);
    if (file_error || final_size != initial_size)
        return fail(error, "GGUF changed during GPU initialization");
    const auto final_mtime = std::filesystem::last_write_time(model_path, file_error);
    if (file_error || final_mtime != initial_mtime)
        return fail(error, "GGUF changed during GPU initialization");
    model_path_ = normalized_path.u8string();
    model_file_size_ = initial_size;
    model_mtime_ticks_ = static_cast<int64_t>(initial_mtime.time_since_epoch().count());
    ready_ = true;
    sequence_valid_ = true;
    if (error) error->clear();
    return true;
}

bool DflashGpuSyncEngine::matmul(const Weight & weight, Buffer & input,
                                Buffer & output, uint32_t batch,
                                std::string * error) {
    if (!weight.buffer.resource || !buffer_range(input.resource.Get(),
            uint64_t(weight.columns) * batch * 4) ||
        !buffer_range(output.resource.Get(), uint64_t(weight.rows) * batch * 4))
        return fail(error, "matmul buffer range invalid");
    transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if ((weight.type == GGML_TYPE_Q3_K || weight.type == GGML_TYPE_Q4_K ||
         weight.type == GGML_TYPE_Q5_K || weight.type == GGML_TYPE_Q6_K ||
         weight.type == GGML_TYPE_Q8_0) &&
        activation_mode_ == ActivationMode::GgmlQ8K) {
        Buffer * q8 = nullptr;
        bool * packed = nullptr;
        if (&input == &input_) {
            q8 = &input_q8_;
            packed = &input_q8_ready_;
        } else if (&input == &encoder_) {
            q8 = &encoder_q8_;
            packed = &encoder_q8_ready_;
        } else {
            return fail(error, "quantized matmul activation has no Q8 scratch buffer");
        }
        if (!*packed) {
            if (!quantize_q8(input, *q8, weight.columns, batch, error)) return false;
            *packed = true;
        }
        const uint32_t constants[4] = {weight.rows, weight.columns,
                                       weight.columns / 256, batch};
        cmd_->SetComputeRootSignature(matmul_root_.Get());
        ID3D12PipelineState * dot_pso = weight.type == GGML_TYPE_Q3_K ? q3q8_pso_.Get() :
                                        weight.type == GGML_TYPE_Q4_K ? q4q8_pso_.Get() :
                                        weight.type == GGML_TYPE_Q5_K ? q5q8_pso_.Get() :
                                        weight.type == GGML_TYPE_Q6_K ? q6q8_pso_.Get() :
                                                                        q8_0q8_pso_.Get();
        cmd_->SetPipelineState(dot_pso);
        cmd_->SetComputeRoot32BitConstants(0, 4, constants, 0);
        cmd_->SetComputeRootShaderResourceView(1, weight.buffer.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootShaderResourceView(2, q8->resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootUnorderedAccessView(3, output.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(weight.rows, (batch + 7) / 8, 1);
        uav_barrier(output);
        return true;
    }
    if (weight.type == GGML_TYPE_F32) {
        const uint32_t constants[8] = {batch, weight.rows, weight.columns,
                                       bits(1.0f), bits(epsilon_), bits(rope_base_),
                                       bits(1.0f), 0};
        cmd_->SetComputeRootSignature(ops_root_.Get());
        cmd_->SetPipelineState(matmul_psos_[3][0].Get());
        cmd_->SetComputeRoot32BitConstants(0, 8, constants, 0);
        cmd_->SetComputeRootShaderResourceView(1, input.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootShaderResourceView(2, weight.buffer.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootUnorderedAccessView(3, output.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootShaderResourceView(4, positions_.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(weight.rows, batch, 1);
    } else {
        uint32_t type_index = 0;
        if (weight.type == GGML_TYPE_Q3_K) type_index = 0;
        else if (weight.type == GGML_TYPE_Q4_K) type_index = 1;
        else if (weight.type == GGML_TYPE_Q5_K) type_index = 2;
        else return fail(error, "unsupported matmul type");
        const uint32_t constants[4] = {weight.rows, weight.columns,
                                       weight.columns / 256, batch};
        cmd_->SetComputeRootSignature(matmul_root_.Get());
        cmd_->SetPipelineState(matmul_psos_[type_index][batch == 1 ? 0 : 1].Get());
        cmd_->SetComputeRoot32BitConstants(0, 4, constants, 0);
        cmd_->SetComputeRootShaderResourceView(1, weight.buffer.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootShaderResourceView(2, input.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootUnorderedAccessView(3, output.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(weight.rows, batch == 1 ? 1 : (batch + 7) / 8, 1);
    }
    uav_barrier(output);
    return true;
}

bool DflashGpuSyncEngine::quantize_q8(Buffer & input, Buffer & output,
                                      uint32_t width, uint32_t batch,
                                      std::string * error) {
    if (!width || width % 256 || !batch || batch > kBatch ||
        !buffer_range(input.resource.Get(), uint64_t(batch) * width * 4) ||
        !buffer_range(output.resource.Get(), uint64_t(batch) * (width / 256) * 292))
        return fail(error, "Q8_K activation buffer range invalid");
    transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[4] = {0, width, width / 256, batch};
    cmd_->SetComputeRootSignature(matmul_root_.Get());
    cmd_->SetPipelineState(q8_pack_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 4, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, input.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2, input.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(3, output.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(width / 256, batch, 1);
    uav_barrier(output);
    transition(output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return true;
}

void DflashGpuSyncEngine::dispatch_norm(uint32_t batch) {
    transition(fc_output_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(encoder_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[8] = {batch, kHiddenWidth, 0,
                                   bits(fc_scale_), bits(epsilon_),
                                   bits(rope_base_), bits(1.0f), 0};
    cmd_->SetComputeRootSignature(ops_root_.Get());
    cmd_->SetPipelineState(norm_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, fc_output_.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2, enc_norm_.buffer.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(3, encoder_.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(4, positions_.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(batch, 1, 1);
    uav_barrier(encoder_);
    transition(encoder_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

void DflashGpuSyncEngine::dispatch_key(uint32_t batch, const Layer & layer) {
    transition(key_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(rotated_key_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[8] = {batch, kKvHeads * kHeadWidth, 0,
                                   bits(layer.key_scale), bits(epsilon_),
                                   bits(rope_base_), bits(1.0f), 0};
    cmd_->SetComputeRootSignature(ops_root_.Get());
    cmd_->SetPipelineState(key_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, key_.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(2, layer.key_norm.buffer.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(3, rotated_key_.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootShaderResourceView(4, positions_.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(kKvHeads, batch, 1);
    uav_barrier(rotated_key_);
    transition(rotated_key_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

void DflashGpuSyncEngine::dispatch_hadamard(Buffer & input, Buffer & output,
                                            uint32_t batch, uint32_t transform_width) {
    transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[4] = {batch, kKvHeads, transform_width, 0};
    cmd_->SetComputeRootSignature(matmul_root_.Get());
    cmd_->SetPipelineState(hadamard_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 4, constants, 0);
    cmd_->SetComputeRootShaderResourceView(1, input.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(3, output.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(kKvHeads, batch, 1);
    uav_barrier(output);
    transition(output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

bool DflashGpuSyncEngine::sync(const int32_t * positions, const float * features,
                               uint32_t count, bool reset, int32_t trim_to,
                               std::string * error) {
    if (!ready_ || !sequence_valid_) return fail(error, "GPU sequence unavailable");
    if (!positions || !features || !count || count > kBatch ||
        trim_to < -1 || trim_to > static_cast<int32_t>(kContext) ||
        (reset && trim_to != -1))
        return fail(error, "invalid SYNC arguments");
    for (uint32_t i = 0; i < count; ++i) {
        if (positions[i] < 0 || positions[i] >= static_cast<int32_t>(kContext))
            return fail(error, "SYNC position outside context");
        for (uint32_t j = 0; j < i; ++j)
            if (positions[j] == positions[i])
                return fail(error, "duplicate SYNC position");
    }
    const uint64_t input_bytes = uint64_t(count) * kFeatureWidth * sizeof(float);
    const uint64_t position_bytes = uint64_t(count) * sizeof(int32_t);
    void * mapped = nullptr;
    HRESULT hr = input_upload_.resource->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return fail(error, "SYNC input Map", hr);
    std::memcpy(mapped, features, static_cast<size_t>(input_bytes));
    input_upload_.resource->Unmap(0, nullptr);
    hr = position_upload_.resource->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return fail(error, "SYNC positions Map", hr);
    std::memcpy(mapped, positions, static_cast<size_t>(position_bytes));
    position_upload_.resource->Unmap(0, nullptr);
    // A recording or device error from this point invalidates the whole
    // sequence: DflashGpuKv marks slots valid when Pack is recorded.
    if (!begin(error)) { sequence_valid_ = false; return false; }
    input_q8_ready_ = false;
    encoder_q8_ready_ = false;
    if (reset) kv_.reset();
    else if (trim_to >= 0 && !kv_.trim(static_cast<uint32_t>(trim_to))) {
        sequence_valid_ = false;
        return fail(error, "KV trim failed");
    }
    transition(input_, D3D12_RESOURCE_STATE_COPY_DEST);
    transition(positions_, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd_->CopyBufferRegion(input_.resource.Get(), 0, input_upload_.resource.Get(), 0,
                           input_bytes);
    cmd_->CopyBufferRegion(positions_.resource.Get(), 0,
                           position_upload_.resource.Get(), 0, position_bytes);
    transition(input_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(positions_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!matmul(fc_, input_, fc_output_, count, error)) {
        sequence_valid_ = false; return false;
    }
    dispatch_norm(count);
    for (uint32_t il = 0; il < kLayers; ++il) {
        const Layer & layer = layers_[il];
        if (!matmul(layer.key, encoder_, key_, count, error) ||
            !matmul(layer.value, encoder_, value_, count, error)) {
            sequence_valid_ = false; return false;
        }
        dispatch_key(count, layer);
        // Quantized llama KV rotates the RoPE'd K with H128 before Q4_0.
        dispatch_hadamard(rotated_key_, hadamard_key_, count, kHeadWidth);
        transition(value_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Buffer * value_source = &value_;
        if (layer.value_scale != 1.0f) {
            transition(scaled_value_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            const uint32_t constants[8] = {count, kKvHeads * kHeadWidth, 0,
                                           bits(layer.value_scale), bits(epsilon_),
                                           bits(rope_base_), bits(1.0f), 0};
            cmd_->SetComputeRootSignature(ops_root_.Get());
            cmd_->SetPipelineState(scale_pso_.Get());
            cmd_->SetComputeRoot32BitConstants(0, 8, constants, 0);
            cmd_->SetComputeRootShaderResourceView(1, value_.resource->GetGPUVirtualAddress());
            cmd_->SetComputeRootShaderResourceView(2, enc_norm_.buffer.resource->GetGPUVirtualAddress());
            cmd_->SetComputeRootUnorderedAccessView(3,
                scaled_value_.resource->GetGPUVirtualAddress());
            cmd_->SetComputeRootShaderResourceView(4, positions_.resource->GetGPUVirtualAddress());
            cmd_->Dispatch((count * kKvHeads * kHeadWidth + 127) / 128, 1, 1);
            uav_barrier(scaled_value_);
            transition(scaled_value_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            value_source = &scaled_value_;
        }
        // V uses H64 independently on each half of its 128-wide head.
        dispatch_hadamard(*value_source, hadamard_value_, count, 64);
        for (uint32_t row = 0; row < count; ++row) {
            const uint64_t source_offset = uint64_t(row) * kKvHeads * kHeadWidth * 4;
            if (!kv_.record_pack(cmd_.Get(), il, static_cast<uint32_t>(positions[row]),
                                 DflashGpuKv::Kind::Key, hadamard_key_.resource.Get(),
                                 source_offset, error) ||
                !kv_.record_pack(cmd_.Get(), il, static_cast<uint32_t>(positions[row]),
                                 DflashGpuKv::Kind::Value, hadamard_value_.resource.Get(),
                                 source_offset, error)) {
                sequence_valid_ = false; return false;
            }
        }
    }
    if (!submit(error)) { sequence_valid_ = false; return false; }
    last_batch_ = count;
    std::memcpy(last_positions_, positions, size_t(count) * sizeof(int32_t));
    if (error) error->clear();
    return true;
}

bool DflashGpuSyncEngine::trim(uint32_t keep_positions, std::string * error) {
    if (!ready_ || !sequence_valid_) return fail(error, "GPU sequence unavailable");
    if (keep_positions > kContext) return fail(error, "invalid trim position");
    if (!kv_.trim(keep_positions)) return fail(error, "KV trim failed");
    if (error) error->clear();
    return true;
}

void DflashGpuSyncEngine::reset() {
    if (ready_ && sequence_valid_) {
        kv_.reset();
        last_batch_ = 0;
    }
}

bool DflashGpuSyncEngine::read_encoder(float * destination, uint32_t count,
                                       std::string * error) {
    if (!count || count > last_batch_) return fail(error, "encoder readback count invalid");
    return read_buffer(encoder_, destination, uint64_t(count) * kHiddenWidth * 4,
                       "encoder", error);
}

bool DflashGpuSyncEngine::read_fc(float * destination, uint32_t count,
                                  std::string * error) {
    if (!count || count > last_batch_) return fail(error, "FC readback count invalid");
    return read_buffer(fc_output_, destination, uint64_t(count) * kHiddenWidth * 4,
                       "FC", error);
}

bool DflashGpuSyncEngine::read_input_q8(void * destination, uint32_t count,
                                        std::string * error) {
    if (activation_mode_ != ActivationMode::GgmlQ8K || !input_q8_ready_ ||
        !count || count > last_batch_)
        return fail(error, "Q8 input readback unavailable");
    return read_buffer(input_q8_, destination,
                       uint64_t(count) * (kFeatureWidth / 256) * 292,
                       "Q8 input", error);
}

bool DflashGpuSyncEngine::read_encoder_q8(void * destination, uint32_t count,
                                          std::string * error) {
    if (activation_mode_ != ActivationMode::GgmlQ8K || !encoder_q8_ready_ ||
        !count || count > last_batch_)
        return fail(error, "Q8 encoder readback unavailable");
    return read_buffer(encoder_q8_, destination,
                       uint64_t(count) * (kHiddenWidth / 256) * 292,
                       "Q8 encoder", error);
}

bool DflashGpuSyncEngine::read_last_layer_kv(const int32_t * positions,
                                              uint32_t count, KvSnapshot * snapshot,
                                              std::string * error) {
    if (!ready_ || !sequence_valid_ || !positions || !snapshot ||
        !count || count != last_batch_ ||
        std::memcmp(positions, last_positions_, size_t(count) * sizeof(int32_t)) != 0)
        return fail(error, "KV diagnostic batch unavailable");
    constexpr uint32_t layer = kLayers - 1;
    constexpr uint64_t packed_row_bytes = uint64_t(kKvHeads * kHeadWidth / 32) * 20;
    const uint64_t float_bytes = uint64_t(count) * kKvHeads * kHeadWidth * 4;
    const uint64_t packed_bytes = uint64_t(count) * packed_row_bytes;
    const uint64_t total_bytes = 4 * float_bytes + 2 * packed_bytes;
    std::vector<uint64_t> key_offsets(count), value_offsets(count);
    for (uint32_t row = 0; row < count; ++row) {
        if (!kv_.gpu_byte_offset(layer, static_cast<uint32_t>(positions[row]),
                                 DflashGpuKv::Kind::Key, &key_offsets[row]) ||
            !kv_.gpu_byte_offset(layer, static_cast<uint32_t>(positions[row]),
                                 DflashGpuKv::Kind::Value, &value_offsets[row]))
            return fail(error, "KV diagnostic slot unavailable");
    }
    Buffer readback;
    if (!create_buffer(total_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COPY_DEST, &readback, error) ||
        !begin(error)) { sequence_valid_ = false; return false; }
    Buffer * float_sources[4] = {
        &rotated_key_, layers_[layer].value_scale == 1.0f ? &value_ : &scaled_value_,
        &hadamard_key_, &hadamard_value_};
    for (uint32_t i = 0; i < 4; ++i) {
        transition(*float_sources[i], D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd_->CopyBufferRegion(readback.resource.Get(), uint64_t(i) * float_bytes,
                               float_sources[i]->resource.Get(), 0, float_bytes);
        transition(*float_sources[i], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    if (!kv_.record_prepare_read(cmd_.Get(), error)) {
        sequence_valid_ = false;
        return false;
    }
    D3D12_RESOURCE_BARRIER kv_barrier = {};
    kv_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    kv_barrier.Transition.pResource = kv_.resource();
    kv_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    kv_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    kv_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cmd_->ResourceBarrier(1, &kv_barrier);
    for (uint32_t row = 0; row < count; ++row) {
        cmd_->CopyBufferRegion(readback.resource.Get(), 4 * float_bytes +
                               uint64_t(row) * packed_row_bytes,
                               kv_.resource(), key_offsets[row], packed_row_bytes);
        cmd_->CopyBufferRegion(readback.resource.Get(), 4 * float_bytes + packed_bytes +
                               uint64_t(row) * packed_row_bytes,
                               kv_.resource(), value_offsets[row], packed_row_bytes);
    }
    std::swap(kv_barrier.Transition.StateBefore, kv_barrier.Transition.StateAfter);
    cmd_->ResourceBarrier(1, &kv_barrier);
    pending_readback_ = std::move(readback.resource);
    if (!submit(error)) { sequence_valid_ = false; return false; }
    void * mapped = nullptr;
    const D3D12_RANGE range = {0, static_cast<SIZE_T>(total_bytes)};
    const HRESULT hr = pending_readback_->Map(0, &range, &mapped);
    if (FAILED(hr)) { sequence_valid_ = false; return fail(error, "KV diagnostic Map", hr); }
    const size_t floats = size_t(count) * kKvHeads * kHeadWidth;
    snapshot->pre_key.resize(floats);
    snapshot->pre_value.resize(floats);
    snapshot->rotated_key.resize(floats);
    snapshot->rotated_value.resize(floats);
    snapshot->packed_key.resize(static_cast<size_t>(packed_bytes));
    snapshot->packed_value.resize(static_cast<size_t>(packed_bytes));
    const auto * data = static_cast<const uint8_t *>(mapped);
    std::memcpy(snapshot->pre_key.data(), data, static_cast<size_t>(float_bytes));
    std::memcpy(snapshot->pre_value.data(), data + float_bytes,
                static_cast<size_t>(float_bytes));
    std::memcpy(snapshot->rotated_key.data(), data + 2 * float_bytes,
                static_cast<size_t>(float_bytes));
    std::memcpy(snapshot->rotated_value.data(), data + 3 * float_bytes,
                static_cast<size_t>(float_bytes));
    std::memcpy(snapshot->packed_key.data(), data + 4 * float_bytes,
                static_cast<size_t>(packed_bytes));
    std::memcpy(snapshot->packed_value.data(), data + 4 * float_bytes + packed_bytes,
                static_cast<size_t>(packed_bytes));
    const D3D12_RANGE empty = {0, 0};
    pending_readback_->Unmap(0, &empty);
    pending_readback_.Reset();
    if (error) error->clear();
    return true;
}

bool DflashGpuSyncEngine::read_kv_packed(const int32_t * positions,
                                         uint32_t count,
                                         std::vector<uint8_t> * destination,
                                         std::string * error) {
    if (!ready_ || !sequence_valid_ || !positions || !destination ||
        !count || count > kBatch)
        return fail(error, "KV packed diagnostic arguments invalid");
    constexpr uint64_t row_bytes = uint64_t(kKvHeads * kHeadWidth / 32) * 20;
    const uint64_t total_bytes = uint64_t(kLayers) * 2 * count * row_bytes;
    std::vector<uint64_t> offsets(size_t(kLayers) * 2 * count);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        for (uint32_t kind = 0; kind < 2; ++kind) {
            for (uint32_t row = 0; row < count; ++row) {
                if (positions[row] < 0 || positions[row] >= static_cast<int32_t>(kContext))
                    return fail(error, "KV packed diagnostic position invalid");
                const size_t index = (size_t(layer) * 2 + kind) * count + row;
                if (!kv_.gpu_byte_offset(layer, static_cast<uint32_t>(positions[row]),
                    static_cast<DflashGpuKv::Kind>(kind), &offsets[index]))
                    return fail(error, "KV packed diagnostic slot unavailable");
            }
        }
    }
    Buffer readback;
    if (!create_buffer(total_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COPY_DEST, &readback, error) ||
        !begin(error)) { sequence_valid_ = false; return false; }
    if (!kv_.record_prepare_read(cmd_.Get(), error)) {
        sequence_valid_ = false;
        return false;
    }
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = kv_.resource();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cmd_->ResourceBarrier(1, &barrier);
    for (size_t index = 0; index < offsets.size(); ++index)
        cmd_->CopyBufferRegion(readback.resource.Get(), uint64_t(index) * row_bytes,
                               kv_.resource(), offsets[index], row_bytes);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    cmd_->ResourceBarrier(1, &barrier);
    pending_readback_ = std::move(readback.resource);
    if (!submit(error)) { sequence_valid_ = false; return false; }
    void * mapped = nullptr;
    const D3D12_RANGE range = {0, static_cast<SIZE_T>(total_bytes)};
    const HRESULT hr = pending_readback_->Map(0, &range, &mapped);
    if (FAILED(hr)) { sequence_valid_ = false; return fail(error, "KV packed Map", hr); }
    destination->resize(static_cast<size_t>(total_bytes));
    std::memcpy(destination->data(), mapped, static_cast<size_t>(total_bytes));
    const D3D12_RANGE empty = {0, 0};
    pending_readback_->Unmap(0, &empty);
    pending_readback_.Reset();
    if (error) error->clear();
    return true;
}

bool DflashGpuSyncEngine::read_buffer(Buffer & source, void * destination,
                                     uint64_t bytes, const char * stage,
                                     std::string * error) {
    if (!ready_ || !sequence_valid_ || !destination || !bytes || bytes > source.bytes)
        return fail(error, "diagnostic readback unavailable");
    Buffer readback;
    if (!create_buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COPY_DEST, &readback, error) ||
        !begin(error)) { sequence_valid_ = false; return false; }
    transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmd_->CopyBufferRegion(readback.resource.Get(), 0, source.resource.Get(), 0, bytes);
    transition(source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    pending_readback_ = std::move(readback.resource);
    if (!submit(error)) { sequence_valid_ = false; return false; }
    void * mapped = nullptr;
    const D3D12_RANGE range = {0, static_cast<SIZE_T>(bytes)};
    const HRESULT hr = pending_readback_->Map(0, &range, &mapped);
    if (FAILED(hr)) { sequence_valid_ = false; return fail(error, stage, hr); }
    std::memcpy(destination, mapped, static_cast<size_t>(bytes));
    const D3D12_RANGE empty = {0, 0};
    pending_readback_->Unmap(0, &empty);
    pending_readback_.Reset();
    if (error) error->clear();
    return true;
}
