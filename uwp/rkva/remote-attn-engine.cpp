// RemoteAttnEngine — D3D12 executor for the RKVA service (Xbox Series X).
//
// F16 bring-up path: stores K/V as F16 (GGML RNE) in a per-session, per-layer
// DEFAULT-heap buffer and runs causal GQA flash attention on the GPU. This
// validates the full split-execution pipeline (RPC -> session -> store ->
// attention -> readback) independently of KVarN numerics, and remains the
// materialize/correctness fallback once the native KVarN-over-records kernels
// land. One engine serves the single persistent RKVA connection; calls are
// serialized by the listener, so no internal locking is needed.

#include "remote-attn-engine.h"

#include "rkva_flash_attn_dxil.h"
#include "rkva_store_f16_dxil.h"
#include "rkva_wht_dxil.h"
#include "rkva_store_kvarn4_dxil.h"
#include "rkva_flash_attn_kvarn4_dxil.h"
#include "rkva_flash_attn_split_k_reduce_dxil.h"
#include "xllama/d3d12_dyn.h"

static const uint8_t* kRkvaStoreKvarn4Dxil = g_rkva_store_kvarn4_dxil;
static const size_t kRkvaStoreKvarn4DxilSize = sizeof(g_rkva_store_kvarn4_dxil);

static const uint8_t* kRkvaFlashAttnKvarn4Dxil = g_rkva_flash_attn_kvarn4_dxil;
static const size_t kRkvaFlashAttnKvarn4DxilSize = sizeof(g_rkva_flash_attn_kvarn4_dxil);

static const uint8_t* kRkvaFlashAttnSplitKReduceDxil = g_rkva_flash_attn_split_k_reduce_dxil;
static const size_t kRkvaFlashAttnSplitKReduceDxilSize = sizeof(g_rkva_flash_attn_split_k_reduce_dxil);

#include <dxgi1_4.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

using Microsoft::WRL::ComPtr;

namespace {

uint64_t aligned4(uint64_t bytes) {
    return (bytes + 3ull) & ~3ull;
}

bool fail(std::string* error, const char* stage, HRESULT hr = S_OK) {
    if (error) {
        char text[160];
        if (hr == S_OK) {
            std::snprintf(text, sizeof(text), "%s", stage);
        } else {
            std::snprintf(text, sizeof(text), "%s: 0x%08X", stage, static_cast<unsigned>(hr));
        }
        *error = text;
    }
    return false;
}

}  // namespace

// Per-session KV storage. Each remote layer has its own dedicated committed DEFAULT
// resource in `layer_kv`, preventing >2 GiB single-resource allocation failures
// and 32-bit offset overflow in ByteAddressBuffer at 128K-256K+ context.
// F16 path: [kind: 0=K,1=V][capacity][n_head_kv][head_dim].
// KVarN path (cache_bits 2..8): per layer slot the buffer is laid out as
// [records K][stage K][records V][stage V], where a record group covers 128
// positions (KVAR_N_GROUP) and the stage holds stage_groups*128 positions of
// F16 (group 0 sink + tail_groups live groups).
struct RemoteAttnEngine::Session {
    uint32_t id = 0;
    uint32_t capacity = 0;
    int32_t high_water = -1;  // highest stored position; -1 = empty
    bool valid = true;
    std::vector<Buffer> layer_kv; // 1 committed buffer per layer slot
    uint64_t kv_bytes = 0;
    uint64_t layer_stride = 0;  // bytes per (slot) covering K+V
    uint64_t half_stride = 0;   // bytes per K or V plane (F16 path)
    // KVarN path sizes (bits>8 none; cache_bits 2..8).
    uint64_t record_words = 0;   // per record group, u32 words
    uint64_t records_bytes = 0;  // per layer per kind (K or V)
    uint64_t stage_bytes = 0;    // per layer per kind (K or V)
    uint64_t groups_per_stream = 0;
    uint64_t rec_off = 0;        // byte offset of records K within a layer slot
    uint64_t stg_off = 0;        // byte offset of stage K within a layer slot
    uint64_t rec_v_off = 0;      // byte offset of records V within a layer slot
    uint64_t stg_v_off = 0;      // byte offset of stage V within a layer slot
};

RemoteAttnEngine::RemoteAttnEngine() = default;

RemoteAttnEngine::~RemoteAttnEngine() {
    if (fence_event_) {
        CloseHandle(fence_event_);
        fence_event_ = nullptr;
    }
}

bool RemoteAttnEngine::fail(std::string* error, const char* stage, HRESULT hr) {
    ready_ = false;
    return ::fail(error, stage, hr);
}

bool RemoteAttnEngine::create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap,
                                     D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                     Buffer* out, std::string* error) {
    if (!device_ || !out || !bytes) {
        return fail(error, "invalid buffer size");
    }
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
    const HRESULT hr = device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state,
                                                        nullptr, IID_PPV_ARGS(&resource));
    if (FAILED(hr)) {
        return fail(error, "CreateCommittedResource", hr);
    }
    out->resource = std::move(resource);
    out->bytes = rd.Width;
    out->state = state;
    return true;
}

void RemoteAttnEngine::transition(Buffer& buffer, D3D12_RESOURCE_STATES state) {
    if (buffer.state == state || !buffer.resource) {
        return;
    }
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer.resource.Get();
    barrier.Transition.StateBefore = buffer.state;
    barrier.Transition.StateAfter = state;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_->ResourceBarrier(1, &barrier);
    buffer.state = state;
}

void RemoteAttnEngine::uav_barrier(Buffer& buffer) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = buffer.resource.Get();
    cmd_->ResourceBarrier(1, &barrier);
}

bool RemoteAttnEngine::begin(std::string* error) {
    if (terminal_failure_ || unconfirmed_submission_) {
        return fail(error, "engine terminal after unconfirmed submission");
    }
    HRESULT hr = allocator_->Reset();
    if (FAILED(hr)) {
        return fail(error, "CommandAllocator Reset", hr);
    }
    hr = cmd_->Reset(allocator_.Get(), nullptr);
    if (FAILED(hr)) {
        return fail(error, "CommandList Reset", hr);
    }
    return true;
}

bool RemoteAttnEngine::submit(std::string* error) {
    if (terminal_failure_ || unconfirmed_submission_) {
        return fail(error, "engine terminal after unconfirmed submission");
    }
    HRESULT hr = cmd_->Close();
    if (FAILED(hr)) {
        return fail(error, "CommandList Close", hr);
    }
    ID3D12CommandList* lists[] = {cmd_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    unconfirmed_submission_ = true;
    const UINT64 value = ++fence_value_;
    hr = queue_->Signal(fence_.Get(), value);
    if (FAILED(hr)) {
        terminal_failure_ = true;
        return fail(error, "CommandQueue Signal", hr);
    }
    if (fence_->GetCompletedValue() < value) {
        hr = fence_->SetEventOnCompletion(value, fence_event_);
        if (FAILED(hr)) {
            terminal_failure_ = true;
            return fail(error, "Fence SetEventOnCompletion", hr);
        }
        if (WaitForSingleObject(fence_event_, 60000) != WAIT_OBJECT_0) {
            terminal_failure_ = true;
            return fail(error, "GPU fence wait failed");
        }
    }
    if (fence_->GetCompletedValue() < value) {
        terminal_failure_ = true;
        return fail(error, "GPU execution did not complete", device_->GetDeviceRemovedReason());
    }
    unconfirmed_submission_ = false;
    hr = device_->GetDeviceRemovedReason();
    if (FAILED(hr)) {
        terminal_failure_ = true;
        return fail(error, "D3D12 device removed", hr);
    }
    return true;
}

bool RemoteAttnEngine::create_device(std::string* error) {
    auto create_device_fn = xllama::d3d12_dyn::CreateDevice();
    if (!create_device_fn) {
        return fail(error, "D3D12CreateDevice unavailable");
    }
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        return fail(error, "CreateDXGIFactory1", hr);
    }
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        hr = create_device_fn(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_));
        if (SUCCEEDED(hr)) {
            break;
        }
    }
    if (!device_) {
        return fail(error, "no hardware D3D12 adapter");
    }
    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_));
    if (FAILED(hr)) {
        return fail(error, "CreateCommandQueue", hr);
    }
    hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_));
    if (FAILED(hr)) {
        return fail(error, "CreateCommandAllocator", hr);
    }
    hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr,
                                    IID_PPV_ARGS(&cmd_));
    if (FAILED(hr)) {
        return fail(error, "CreateCommandList", hr);
    }
    hr = cmd_->Close();
    if (FAILED(hr)) {
        return fail(error, "CommandList initial Close", hr);
    }
    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) {
        return fail(error, "CreateFence", hr);
    }
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event_) {
        return fail(error, "CreateEventW failed");
    }
    return true;
}

bool RemoteAttnEngine::create_pipelines(std::string* error) {
    auto serialize = xllama::d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        return fail(error, "D3D12SerializeRootSignature unavailable");
    }
    auto make_root = [&](uint32_t n_const, uint32_t n_srv, uint32_t n_uav,
                         ID3D12RootSignature** out) -> bool {
        // Use a fixed-size array (max 8 params) to avoid vector pointer math.
        D3D12_ROOT_PARAMETER p[8] = {};
        uint32_t idx = 0;
        if (n_const) {
            p[idx].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            p[idx].Constants.ShaderRegister = 0;
            p[idx].Constants.Num32BitValues = n_const;
            p[idx].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            ++idx;
        }
        for (uint32_t s = 0; s < n_srv; ++s) {
            p[idx].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            p[idx].Descriptor.ShaderRegister = s;
            p[idx].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            ++idx;
        }
        for (uint32_t u = 0; u < n_uav; ++u) {
            p[idx].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            p[idx].Descriptor.ShaderRegister = u;
            p[idx].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            ++idx;
        }
        D3D12_ROOT_SIGNATURE_DESC desc = {};
        desc.NumParameters = idx;
        desc.pParameters = p;
        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> messages;
        HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &messages);
        if (FAILED(hr)) {
            return fail(error, "SerializeRootSignature", hr);
        }
        hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          IID_PPV_ARGS(out));
        if (FAILED(hr)) {
            return fail(error, "CreateRootSignature", hr);
        }
        return true;
    };

    // Attention: 8 constants + SRV t0..t3 (4) + UAV u0 (1).
    if (!make_root(8, 4, 1, attn_root_.GetAddressOf())) {
        return false;
    }
    // Store: 4 constants + SRV t0..t1 (2) + UAV u0 (1).
    if (!make_root(4, 2, 1, store_root_.GetAddressOf())) {
        return false;
    }
    // KVarN WHT: 3 constants + SRV t0 (1) + UAV u0 (1).
    if (!make_root(3, 1, 1, wht_root_.GetAddressOf())) {
        return false;
    }
    // KVarN store: 11 constants + SRV t0..t1 (2) + UAV u0..u1 (2).
    if (!make_root(11, 2, 2, store_kvarn_root_.GetAddressOf())) {
        return false;
    }
    // KVarN attention: 14 constants (including split_k) + SRV t0..t5 (6) + UAV u0 (1).
    if (!make_root(14, 6, 1, attn_kvarn_root_.GetAddressOf())) {
        return false;
    }
    // Split-K reduction: 4 constants + SRV t0 (1) + UAV u0 (1).
    if (!make_root(4, 1, 1, split_k_reduce_root_.GetAddressOf())) {
        return false;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
    pso.pRootSignature = attn_root_.Get();
    pso.CS = {kRkvaFlashAttnDxil, kRkvaFlashAttnDxilSize};
    HRESULT hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&attn_decode_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState attn", hr);
    }
    attn_prefill_pso_ = attn_decode_pso_;  // same kernel; grid differs

    pso.pRootSignature = store_root_.Get();
    pso.CS = {kRkvaStoreF16Dxil, kRkvaStoreF16DxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&store_k_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState store", hr);
    }
    store_v_pso_ = store_k_pso_;

    pso.pRootSignature = wht_root_.Get();
    pso.CS = {kRkvaWhtDxil, kRkvaWhtDxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&wht_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState wht", hr);
    }

    pso.pRootSignature = store_kvarn_root_.Get();
    pso.CS = {kRkvaStoreKvarn4Dxil, kRkvaStoreKvarn4DxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&store_kvarn_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState store kvarn", hr);
    }

    pso.pRootSignature = attn_kvarn_root_.Get();
    pso.CS = {kRkvaFlashAttnKvarn4Dxil, kRkvaFlashAttnKvarn4DxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&attn_kvarn_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState attn kvarn", hr);
    }

    pso.pRootSignature = split_k_reduce_root_.Get();
    pso.CS = {kRkvaFlashAttnSplitKReduceDxil, kRkvaFlashAttnSplitKReduceDxilSize};
    hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&split_k_reduce_pso_));
    if (FAILED(hr)) {
        return fail(error, "CreateComputePipelineState split_k reduce", hr);
    }
    return true;
}

bool RemoteAttnEngine::initialize(const Geometry& geo, std::string* error) {
    if (ready_) {
        return fail(error, "engine already initialized");
    }
    if (geo.n_head == 0 || geo.n_head_kv == 0 || geo.head_dim == 0 || geo.max_ctx == 0 ||
        geo.n_head % geo.n_head_kv != 0) {
        return fail(error, "invalid geometry");
    }
    geo_ = geo;
    if (!create_device(error) || !create_pipelines(error)) {
        return false;
    }
    // Initial scratch for a modest prefill chunk; grown on demand.
    scratch_tokens_ = 0;
    ready_ = true;
    return true;
}

RemoteAttnEngine::Session* RemoteAttnEngine::find_session(uint32_t session_id) {
    for (auto& s : sessions_) {
        if (s && s->id == session_id) {
            return s.get();
        }
    }
    return nullptr;
}

const RemoteAttnEngine::Session* RemoteAttnEngine::find_session(uint32_t session_id) const {
    for (const auto& s : sessions_) {
        if (s && s->id == session_id) {
            return s.get();
        }
    }
    return nullptr;
}

bool RemoteAttnEngine::create_session(uint32_t session_id, uint32_t capacity, std::string* error) {
    if (!ready_) {
        return fail(error, "engine not ready");
    }
    if (find_session(session_id)) {
        return true;  // idempotent
    }
    const uint32_t cap = capacity ? capacity : geo_.max_ctx;
    auto session = std::make_unique<Session>();
    session->id = session_id;
    session->capacity = cap;
    const uint32_t n_layers = geo_.n_layers ? geo_.n_layers : 1;
    const bool kvarn = geo_.cache_bits_k >= 2 && geo_.cache_bits_k <= 8;
    if (kvarn) {
        // KVarN4 layout per layer slot:
        //   records K: groups_per_stream * n_heads * record_words * 4
        //   stage   K: stage_groups*128 * n_heads * 64 * 4  (F16 packed pairs)
        //   records V / stage V: same sizes
        const uint32_t bits = geo_.cache_bits_k;
        const uint32_t n_heads = geo_.n_head_kv * geo_.head_slices;
        session->groups_per_stream = (uint64_t) (cap + 127) / 128;
        session->record_words = (uint64_t) bits * 512 + 192;
        session->records_bytes = session->groups_per_stream * n_heads *
                                 session->record_words * 4;
        session->stage_bytes = (uint64_t) geo_.stage_groups * 128 * n_heads * 64 * 4;
        session->rec_off = 0;
        session->stg_off = session->records_bytes;
        session->rec_v_off = session->stg_off + session->stage_bytes;
        session->stg_v_off = session->rec_v_off + session->records_bytes;
        session->layer_stride = session->stg_v_off + session->stage_bytes;
        session->half_stride = 0;
    } else {
        session->half_stride = uint64_t(cap) * geo_.n_head_kv * geo_.head_dim * 2;  // F16
        session->layer_stride = session->half_stride * 2;                            // K + V
    }
    session->kv_bytes = session->layer_stride * n_layers;
    session->layer_kv.resize(n_layers);
    for (uint32_t l = 0; l < n_layers; ++l) {
        if (!create_buffer(session->layer_stride, D3D12_HEAP_TYPE_DEFAULT,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &session->layer_kv[l], error)) {
            session->layer_kv.clear();
            return false;
        }
    }
    allocated_bytes_ += session->kv_bytes;
    sessions_.push_back(std::move(session));
    return true;
}

bool RemoteAttnEngine::destroy_session(uint32_t session_id) {
    for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
        if (*it && (*it)->id == session_id) {
            allocated_bytes_ -= (*it)->kv_bytes;
            (*it)->layer_kv.clear();
            sessions_.erase(it);
            return true;
        }
    }
    return false;
}

bool RemoteAttnEngine::has_session(uint32_t session_id) const {
    return find_session(session_id) != nullptr;
}

bool RemoteAttnEngine::session_invalid(uint32_t session_id) const {
    const Session* s = find_session(session_id);
    return s ? !s->valid : true;
}

bool RemoteAttnEngine::reset_session(uint32_t session_id, std::string* error) {
    Session* s = find_session(session_id);
    if (!s) {
        return fail(error, "reset: no session");
    }
    s->high_water = -1;
    s->valid = true;
    return true;
}

bool RemoteAttnEngine::trim_session(uint32_t session_id, int32_t pos0, std::string* error) {
    Session* s = find_session(session_id);
    if (!s) {
        return fail(error, "trim: no session");
    }
    if (pos0 < 0) {
        return fail(error, "trim: negative position");
    }
    if (s->high_water >= pos0) {
        s->high_water = pos0 - 1;
    }
    return true;
}

// Ensure upload/out/readback scratch fits n_tokens. Grow-only.
bool RemoteAttnEngine::ensure_scratch(uint32_t n_tokens, std::string* error) {
    if (n_tokens <= scratch_tokens_ && qkv_upload_.resource) {
        return true;
    }
    const uint32_t tokens = std::max(n_tokens, std::max(scratch_tokens_, 256u));
    const uint64_t q_bytes = uint64_t(tokens) * geo_.n_head * geo_.head_dim * 4;
    const uint64_t kv_bytes = uint64_t(tokens) * geo_.n_head_kv * geo_.head_dim * 4;
    const uint64_t pos_bytes = uint64_t(tokens) * 4;
    const uint64_t upload_bytes = q_bytes + 2 * kv_bytes + pos_bytes;
    const uint64_t out_bytes = q_bytes;

    qkv_upload_ = Buffer{};
    wht_workspace_ = Buffer{};   // DEFAULT UAV: attention output before readback
    split_accum_ = Buffer{};     // DEFAULT UAV: split-K partial accumulators
    q_rotated_ = Buffer{};       // DEFAULT UAV: WHT(Q) input / inverse-WHT output
    out_readback_ = Buffer{};    // READBACK: CPU-visible copy of the output

    if (!create_buffer(upload_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_GENERIC_READ, &qkv_upload_, error)) {
        return false;
    }
    if (!create_buffer(out_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &wht_workspace_, error)) {
        return false;
    }
    const uint64_t split_accum_bytes = 32ull * out_bytes + 32ull * (uint64_t(tokens) * geo_.n_head * 2 * 4);
    if (!create_buffer(split_accum_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &split_accum_, error)) {
        return false;
    }
    if (!create_buffer(out_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &q_rotated_, error)) {
        return false;
    }
    if (!create_buffer(out_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COPY_DEST, &out_readback_, error)) {
        return false;
    }
    scratch_tokens_ = tokens;
    return true;
}

bool RemoteAttnEngine::attend(uint32_t session_id, uint32_t layer_abs, uint32_t n_tokens,
                              const int32_t* positions, const float* q, const float* k,
                              const float* v, float* out, std::string* error) {
    if (!ready_) {
        return fail(error, "engine not ready");
    }
    Session* s = find_session(session_id);
    if (!s || !s->valid) {
        return fail(error, "attend: invalid session");
    }
    if (n_tokens == 0 || !positions || !q || !k || !v || !out) {
        return fail(error, "attend: bad arguments");
    }

    // Assign a dense KV slot for this absolute layer index.
    uint32_t slot;
    auto it = layer_slot_.find(layer_abs);
    if (it == layer_slot_.end()) {
        slot = (uint32_t) layer_slot_.size();
        if (geo_.n_layers && slot >= geo_.n_layers) {
            return fail(error, "attend: too many remote layers");
        }
        layer_slot_[layer_abs] = slot;
    } else {
        slot = it->second;
    }

    // Validate positions are within capacity.
    int32_t max_pos = s->high_water;
    for (uint32_t t = 0; t < n_tokens; ++t) {
        if (positions[t] < 0 || (uint32_t) positions[t] >= s->capacity) {
            return fail(error, "attend: position out of range");
        }
        max_pos = std::max(max_pos, positions[t]);
    }

    if (!ensure_scratch(n_tokens, error)) {
        return false;
    }

    const uint64_t q_bytes = uint64_t(n_tokens) * geo_.n_head * geo_.head_dim * 4;
    const uint64_t kv_bytes = uint64_t(n_tokens) * geo_.n_head_kv * geo_.head_dim * 4;
    const uint64_t q_off = 0;
    const uint64_t k_off = q_off + q_bytes;
    const uint64_t v_off = k_off + kv_bytes;
    const uint64_t pos_off = v_off + kv_bytes;

    if (!begin(error)) {
        return false;
    }

    // Upload Q/K/V/positions into the persistent UPLOAD buffer.
    {
        void* mapped = nullptr;
        HRESULT hr = qkv_upload_.resource->Map(0, nullptr, &mapped);
        if (FAILED(hr)) {
            return fail(error, "Map upload", hr);
        }
        auto* base = static_cast<uint8_t*>(mapped);
        std::memcpy(base + q_off, q, q_bytes);
        std::memcpy(base + k_off, k, kv_bytes);
        std::memcpy(base + v_off, v, kv_bytes);
        std::memcpy(base + pos_off, positions, uint64_t(n_tokens) * 4);
        qkv_upload_.resource->Unmap(0, nullptr);
    }

    if (slot >= s->layer_kv.size()) {
        return fail(error, "slot out of range for session layers");
    }
    Buffer& layer_buf = s->layer_kv[slot];
    const D3D12_GPU_VIRTUAL_ADDRESS upload_addr = qkv_upload_.resource->GetGPUVirtualAddress();
    const uint64_t k_plane = layer_buf.resource->GetGPUVirtualAddress();
    const uint64_t v_plane = k_plane + s->half_stride;

    // Ensure layer KV is in UAV state for the store.
    transition(layer_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    s->high_water = max_pos;
    const uint32_t active_count = (uint32_t) (s->high_water + 1);

    const bool kvarn = geo_.cache_bits_k >= 2 && geo_.cache_bits_k <= 8;
    Buffer* readback_src = &wht_workspace_;
    if (kvarn) {
        // KVarN path: WHT(Q) -> store K/V (records + stage) -> attention over
        // records -> inverse WHT(out). All buffers are DEFAULT UAV; SRV reads
        // happen after UAV barriers (the data is already ordered).
        const uint64_t rec_k = k_plane + s->rec_off;
        const uint64_t stg_k = k_plane + s->stg_off;
        const uint64_t rec_v = k_plane + s->rec_v_off;
        const uint64_t stg_v = k_plane + s->stg_v_off;
        const uint32_t n_heads_phys = geo_.n_head_kv * geo_.head_slices;
        const uint32_t gqa = geo_.n_head / geo_.n_head_kv;
        const uint32_t gqa_group_size = std::min(gqa, 4u);
        const uint32_t gqa_groups = (gqa + gqa_group_size - 1) / gqa_group_size;

        // 1) Rotate Q: rows = n_tokens * n_head, F32 in -> F32 out (q_rotated_).
        transition(q_rotated_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd_->SetComputeRootSignature(wht_root_.Get());
        cmd_->SetPipelineState(wht_pso_.Get());
        cmd_->SetComputeRoot32BitConstant(0, n_tokens * geo_.n_head, 0);
        cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 1);
        cmd_->SetComputeRoot32BitConstant(0, 0u, 2);
        cmd_->SetComputeRootShaderResourceView(1, upload_addr + q_off);
        cmd_->SetComputeRootUnorderedAccessView(2,
            q_rotated_.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(n_tokens * geo_.n_head, 1, 1);
        uav_barrier(q_rotated_);

        // 2) Store K and V (separate dispatches; `value` flips orientation).
        cmd_->SetComputeRootSignature(store_kvarn_root_.Get());
        cmd_->SetPipelineState(store_kvarn_pso_.Get());
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const bool is_v = pass == 1;
            cmd_->SetComputeRoot32BitConstant(0, n_heads_phys, 0);
            cmd_->SetComputeRoot32BitConstant(0, n_tokens, 1);
            cmd_->SetComputeRoot32BitConstant(0, (uint32_t) s->groups_per_stream, 2);
            cmd_->SetComputeRoot32BitConstant(0, (uint32_t) s->record_words, 3);
            cmd_->SetComputeRoot32BitConstant(0, geo_.cache_bits_k, 4);
            cmd_->SetComputeRoot32BitConstant(0, geo_.sinkhorn_iters, 5);
            cmd_->SetComputeRoot32BitConstant(0, is_v ? 1u : 0u, 6);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_slices, 7);
            cmd_->SetComputeRoot32BitConstant(0, geo_.stage_groups, 8);
            cmd_->SetComputeRoot32BitConstant(0, geo_.tail_groups, 9);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 10);
            cmd_->SetComputeRootShaderResourceView(1, upload_addr + pos_off);
            cmd_->SetComputeRootShaderResourceView(2, upload_addr + (is_v ? v_off : k_off));
            cmd_->SetComputeRootUnorderedAccessView(3, is_v ? stg_v : stg_k);
            cmd_->SetComputeRootUnorderedAccessView(4, is_v ? rec_v : rec_k);
            cmd_->Dispatch(geo_.n_head_kv, 1, 1);
            uav_barrier(layer_buf);
        }

        // 3) Attention over records+stage (rotated domain). Records/stage and
        // the rotated Q are read as SRVs: transition out of UAV first.
        transition(layer_buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(q_rotated_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Split-K parallelization on Xbox Series X 52 CUs:
        uint32_t split_k = 1;
        if (n_tokens == 1 && active_count > 2048) {
            split_k = (active_count >= 32768) ? 32 : 16;
        }

        Buffer* attn_target = (split_k > 1) ? &split_accum_ : &wht_workspace_;
        transition(*attn_target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        cmd_->SetComputeRootSignature(attn_kvarn_root_.Get());
        cmd_->SetPipelineState(attn_kvarn_pso_.Get());
        cmd_->SetComputeRoot32BitConstant(0, geo_.n_head, 0);
        cmd_->SetComputeRoot32BitConstant(0, geo_.n_head_kv, 1);
        cmd_->SetComputeRoot32BitConstant(0, n_heads_phys, 2);
        cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 3);
        cmd_->SetComputeRoot32BitConstant(0, n_tokens, 4);
        cmd_->SetComputeRoot32BitConstant(0, active_count, 5);
        cmd_->SetComputeRoot32BitConstant(0, (uint32_t) s->groups_per_stream, 6);
        cmd_->SetComputeRoot32BitConstant(0, geo_.cache_bits_k, 7);
        cmd_->SetComputeRoot32BitConstant(0, geo_.head_slices, 8);
        cmd_->SetComputeRoot32BitConstant(0, geo_.stage_groups, 9);
        cmd_->SetComputeRoot32BitConstant(0, geo_.tail_groups, 10);
        cmd_->SetComputeRoot32BitConstant(0, (uint32_t) s->record_words, 11);
        cmd_->SetComputeRoot32BitConstant(0, *(const uint32_t*) &geo_.kq_scale, 12);
        cmd_->SetComputeRoot32BitConstant(0, split_k, 13);
        cmd_->SetComputeRootShaderResourceView(1,
            q_rotated_.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootShaderResourceView(2, rec_k);
        cmd_->SetComputeRootShaderResourceView(3, stg_k);
        cmd_->SetComputeRootShaderResourceView(4, rec_v);
        cmd_->SetComputeRootShaderResourceView(5, stg_v);
        cmd_->SetComputeRootShaderResourceView(6, upload_addr + pos_off);
        cmd_->SetComputeRootUnorderedAccessView(7,
            attn_target->resource->GetGPUVirtualAddress());
        cmd_->Dispatch(n_tokens, geo_.n_head_kv * gqa_groups, split_k);
        uav_barrier(*attn_target);

        if (split_k > 1) {
            transition(split_accum_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(wht_workspace_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            cmd_->SetComputeRootSignature(split_k_reduce_root_.Get());
            cmd_->SetPipelineState(split_k_reduce_pso_.Get());
            cmd_->SetComputeRoot32BitConstant(0, n_tokens, 0);
            cmd_->SetComputeRoot32BitConstant(0, geo_.n_head, 1);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 2);
            cmd_->SetComputeRoot32BitConstant(0, split_k, 3);
            cmd_->SetComputeRootShaderResourceView(1,
                split_accum_.resource->GetGPUVirtualAddress());
            cmd_->SetComputeRootUnorderedAccessView(2,
                wht_workspace_.resource->GetGPUVirtualAddress());
            cmd_->Dispatch(n_tokens, geo_.n_head, 1);
            uav_barrier(wht_workspace_);
            transition(split_accum_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        // 4) Inverse WHT on the output (rotated -> original domain); the WHT is
        // self-inverse, so the same kernel rotates out back into q_rotated_.
        transition(wht_workspace_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(q_rotated_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd_->SetComputeRootSignature(wht_root_.Get());
        cmd_->SetPipelineState(wht_pso_.Get());
        cmd_->SetComputeRoot32BitConstant(0, n_tokens * geo_.n_head, 0);
        cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 1);
        cmd_->SetComputeRoot32BitConstant(0, 0u, 2);
        cmd_->SetComputeRootShaderResourceView(1,
            wht_workspace_.resource->GetGPUVirtualAddress());
        cmd_->SetComputeRootUnorderedAccessView(2,
            q_rotated_.resource->GetGPUVirtualAddress());
        cmd_->Dispatch(n_tokens * geo_.n_head, 1, 1);
        uav_barrier(q_rotated_);
        readback_src = &q_rotated_;
    } else {
        // Store K and V (F32 upload -> packed F16 KV at positions). The store
        // kernel packs two F16 dims per thread, so the element count is halved.
        const uint32_t store_pairs = n_tokens * geo_.n_head_kv * (geo_.head_dim / 2);
        const uint32_t store_groups = (store_pairs + 255) / 256;
        {
            cmd_->SetComputeRootSignature(store_root_.Get());
            cmd_->SetPipelineState(store_k_pso_.Get());
            cmd_->SetComputeRoot32BitConstant(0, n_tokens, 0);
            cmd_->SetComputeRoot32BitConstant(0, geo_.n_head_kv, 1);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 2);
            cmd_->SetComputeRoot32BitConstant(0, s->capacity, 3);
            cmd_->SetComputeRootShaderResourceView(1, upload_addr + pos_off);
            cmd_->SetComputeRootShaderResourceView(2, upload_addr + k_off);
            cmd_->SetComputeRootUnorderedAccessView(3, k_plane);
            cmd_->Dispatch(store_groups, 1, 1);
            uav_barrier(layer_buf);

            cmd_->SetComputeRootShaderResourceView(2, upload_addr + v_off);
            cmd_->SetComputeRootUnorderedAccessView(3, v_plane);
            cmd_->Dispatch(store_groups, 1, 1);
            uav_barrier(layer_buf);
        }

        // Attention: KV transitions to SRV; out buffer (wht_workspace_) is UAV.
        transition(layer_buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(wht_workspace_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        {
            cmd_->SetComputeRootSignature(attn_root_.Get());
            cmd_->SetPipelineState(n_tokens == 1 ? attn_decode_pso_.Get() : attn_prefill_pso_.Get());
            cmd_->SetComputeRoot32BitConstant(0, n_tokens, 0);       // query_count
            cmd_->SetComputeRoot32BitConstant(0, active_count, 1);   // active_count
            cmd_->SetComputeRoot32BitConstant(0, geo_.n_head, 2);
            cmd_->SetComputeRoot32BitConstant(0, geo_.n_head_kv, 3);
            cmd_->SetComputeRoot32BitConstant(0, geo_.head_dim, 4);
            cmd_->SetComputeRoot32BitConstant(0, *(const uint32_t*) &geo_.kq_scale, 5);
            cmd_->SetComputeRoot32BitConstant(0, s->capacity, 6);
            cmd_->SetComputeRoot32BitConstant(0, 0u, 7);
            cmd_->SetComputeRootShaderResourceView(1, upload_addr + q_off);   // queries
            cmd_->SetComputeRootShaderResourceView(2, k_plane);               // keys
            cmd_->SetComputeRootShaderResourceView(3, v_plane);               // values
            cmd_->SetComputeRootShaderResourceView(4, upload_addr + pos_off); // query_positions
            cmd_->SetComputeRootUnorderedAccessView(5,
                wht_workspace_.resource->GetGPUVirtualAddress());
            cmd_->Dispatch(geo_.n_head, n_tokens, 1);
            uav_barrier(wht_workspace_);
        }
    }

    // Readback out (DEFAULT UAV -> READBACK).
    transition(*readback_src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(out_readback_, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd_->CopyBufferRegion(out_readback_.resource.Get(), 0, readback_src->resource.Get(), 0,
                           q_bytes);
    // Restore states for the next call.
    transition(*readback_src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(layer_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    if (!submit(error)) {
        s->valid = false;
        return false;
    }

    {
        void* mapped = nullptr;
        D3D12_RANGE range {0, q_bytes};
        HRESULT hr = out_readback_.resource->Map(0, &range, &mapped);
        if (FAILED(hr)) {
            return fail(error, "Map readback", hr);
        }
        std::memcpy(out, mapped, q_bytes);
        D3D12_RANGE empty {0, 0};
        out_readback_.resource->Unmap(0, &empty);
    }
    return true;
}
