#include "gpu-attention-probe.h"

#include "dflash-gpu-kv.h"
#include "dflash-ucrt-exp-table.h"
#include "dflash_attention_dxil.h"
#include "xllama/d3d12_dyn.h"

#include <DirectXPackedVector.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint32_t kLayer = 1;
constexpr uint32_t kLayers = 2;
constexpr uint32_t kCapacity = 2052;
constexpr uint32_t kWidth = 1024;
constexpr uint32_t kQHeads = 32;
constexpr uint32_t kDim = 128;
constexpr uint32_t kQueries = 2;
constexpr uint32_t kPositions[] = {1, 2, 3, 2050, 2051};
constexpr uint32_t kQueryPositions[] = {2050, 2051};
constexpr uint64_t kRowBytes = 1280;
constexpr uint64_t kSourceRowBytes = kWidth * sizeof(float);
constexpr uint64_t kOutputBytes = kQueries * kQHeads * kDim * sizeof(float);

// Diagnose the native console CRT, independently of the Windows host CRT.
std::string cpu_expf_check() {
    constexpr uint32_t count = 1u << 20;
    uint32_t polynomial_differences = 0, double_differences = 0;
    std::string examples;
    for (uint32_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(-104.0 * double(i) / double(count - 1u));
        const double y = double(x) * 92.33248261689366;
        const int n = static_cast<int>(std::nearbyint(y));
        const int j = n & 63;
        const int k = (n - j) / 64;
        volatile double nr = -double(n) * 0.010830424696249145;
        volatile double r = nr + double(x);
        volatile double p0 = r * (1.0 / 6.0);
        volatile double p = p0 + 0.5;
        volatile double rr = r * r;
        volatile double z0 = rr * p;
        volatile double z = z0 + r;
        double t;
        std::memcpy(&t, &dflash_exp::kUcrtExpTable[j], sizeof(t));
        volatile double v0 = z * t;
        volatile double v = v0 + t;
        const float polynomial = static_cast<float>(std::ldexp(v, k));
        const float native = std::exp(x);
        const float from_double = static_cast<float>(std::exp(double(x)));
        if (native != from_double)
            ++double_differences;
        if (native != polynomial) {
            if (polynomial_differences < 8) {
                char item[256];
                std::snprintf(
                    item, sizeof(item),
                    "%s{\"x\":%.9g,\"native\":%.9g,\"polynomial\":%.9g,\"fromDouble\":%.9g}",
                    examples.empty() ? "" : ",", x, native, polynomial, from_double);
                examples += item;
            }
            ++polynomial_differences;
        }
    }
    return "{\"samples\":" + std::to_string(count) +
           ",\"polynomialDifferences\":" + std::to_string(polynomial_differences) +
           ",\"doubleDifferences\":" + std::to_string(double_differences) + ",\"examples\":[" +
           examples + "]}";
}

std::string quote(const std::string& input) {
    std::string result = "\"";
    for (unsigned char c : input) {
        if (c == '"' || c == '\\')
            result += '\\';
        if (c >= 32)
            result += char(c);
    }
    return result + '"';
}

std::string utf8(const wchar_t* input) {
    const int count = WideCharToMultiByte(CP_UTF8, 0, input, -1, nullptr, 0, nullptr, nullptr);
    if (count <= 1)
        return {};
    std::string result(size_t(count), '\0');
    WideCharToMultiByte(CP_UTF8, 0, input, -1, result.data(), count, nullptr, nullptr);
    result.pop_back();
    return result;
}

struct FenceEvent {
    HANDLE handle = nullptr;
    ~FenceEvent() {
        if (handle)
            CloseHandle(handle);
    }
};

ComPtr<ID3D12Resource> buffer(ID3D12Device* device, uint64_t size, D3D12_HEAP_TYPE type,
                              D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                              HRESULT* hr) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = type;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> result;
    *hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                          IID_PPV_ARGS(&result));
    return result;
}

void transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->ResourceBarrier(1, &barrier);
}

bool submit(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* cmd,
            ID3D12Fence* fence, HANDLE event_handle, uint64_t value, HRESULT* hr) {
    *hr = cmd->Close();
    if (FAILED(*hr))
        return false;
    ID3D12CommandList* lists[] = {cmd};
    queue->ExecuteCommandLists(1, lists);
    *hr = queue->Signal(fence, value);
    if (FAILED(*hr))
        return false;
    if (fence->GetCompletedValue() < value) {
        *hr = fence->SetEventOnCompletion(value, event_handle);
        if (FAILED(*hr))
            return false;
        const DWORD wait = WaitForSingleObject(event_handle, 30000);
        if (wait != WAIT_OBJECT_0) {
            *hr = wait == WAIT_TIMEOUT ? HRESULT_FROM_WIN32(WAIT_TIMEOUT)
                                       : HRESULT_FROM_WIN32(GetLastError());
            return false;
        }
    }
    *hr = device->GetDeviceRemovedReason();
    return SUCCEEDED(*hr);
}

// All values are exactly representable by a Q4_0 block with d = -1/8.
// Each block begins with q=0 (value +1), which fixes the max-abs tie rule.
uint32_t nibble(uint32_t position, uint32_t kind, uint32_t element, uint32_t version) {
    if (element % 32 == 0)
        return 0;
    return (element * 7u + position * 3u + kind * 5u + version * 11u) & 15u;
}

float kv_value(uint32_t position, uint32_t kind, uint32_t element, uint32_t version) {
    return (int(nibble(position, kind, element, version)) - 8) * -0.125f;
}

float query_value(uint32_t query, uint32_t head, uint32_t d) {
    return (int((d * 13u + head * 7u + query * 3u) % 19u) - 9) * 0.03125f;
}

float sink_value(uint32_t head) {
    return -0.4f + head * 0.02f;
}

std::vector<uint32_t> active_rows(const DflashGpuKv& kv) {
    std::vector<uint32_t> result;
    for (uint32_t p : kPositions) {
        if (kv.valid(kLayer, p, DflashGpuKv::Kind::Key) &&
            kv.valid(kLayer, p, DflashGpuKv::Kind::Value))
            result.push_back(p);
    }
    return result;
}

// Mirror the shader Q8_0 query packing before computing attention scores.
// The quantization multiplier uses the unrounded maximum; the stored scale is FP16.
std::array<float, kDim> packed_query(uint32_t query, uint32_t head) {
    std::array<float, kDim> values{};
    for (uint32_t block = 0; block < kDim / 32u; ++block) {
        float maximum = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i)
            maximum = std::max(maximum, std::fabs(query_value(query, head, block * 32u + i)));
        const float scale = DirectX::PackedVector::XMConvertHalfToFloat(
            DirectX::PackedVector::XMConvertFloatToHalf(maximum / 127.0f));
        const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) {
            const float scaled = query_value(query, head, block * 32u + i) * inverse;
            const int quant = std::clamp(int(std::nearbyint(scaled)), -127, 127);
            values[block * 32u + i] = float(quant) * scale;
        }
    }
    return values;
}

float reference(uint32_t query, uint32_t head, uint32_t d, const std::vector<uint32_t>& active,
                uint32_t version) {
    const uint32_t qp = kQueryPositions[query];
    const uint32_t kv_head = head / 4u;
    const auto q8_query = packed_query(query, head);
    const double sink = sink_value(head);
    std::vector<double> scores;
    std::vector<uint32_t> visible;
    for (uint32_t p : active) {
        if (p <= qp && qp - p >= 2048u)
            continue;
        double dot = 0.0;
        for (uint32_t i = 0; i < kDim; ++i)
            dot +=
                double(q8_query[i]) * kv_value(p, 0, kv_head * kDim + i, p == 2051 ? version : 0);
        scores.push_back(dot / std::sqrt(128.0));
        visible.push_back(p);
    }
    double maximum = sink;
    for (double score : scores)
        maximum = std::max(maximum, score);
    double denominator = std::exp(sink - maximum);
    double numerator = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        const double weight = std::exp(scores[i] - maximum);
        denominator += weight;
        numerator +=
            weight * kv_value(visible[i], 1, kv_head * kDim + d, visible[i] == 2051 ? version : 0);
    }
    return float(numerator / denominator);
}

} // namespace

dflash_gpu_attention_probe_result run_dflash_gpu_attention_probe() {
    dflash_gpu_attention_probe_result result;
    std::string stages = "[";
    auto fail = [&](const char* stage, const std::string& detail, HRESULT hr = S_OK) {
        char code[16];
        std::snprintf(code, sizeof(code), "0x%08X", unsigned(hr));
        result.json = "{\"passed\":false,\"stage\":" + quote(stage) +
                      ",\"error\":" + quote(detail) + ",\"hresult\":" + quote(code) +
                      ",\"stages\":" + stages + "]}";
        return result;
    };
    auto add_stage = [&](const char* name, float max_error, uint32_t rows) {
        if (stages.size() > 1)
            stages += ',';
        stages += "{\"name\":" + quote(name) +
                  ",\"passed\":true,\"activeRows\":" + std::to_string(rows) +
                  ",\"maxAbsErr\":" + std::to_string(max_error) + "}";
    };

    auto create = xllama::d3d12_dyn::CreateDevice();
    auto serialize = xllama::d3d12_dyn::SerializeRootSignature();
    if (!create || !serialize)
        return fail("D3D12 runtime", "entry point unavailable");
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr))
        return fail("CreateDXGIFactory1", "factory", hr);
    ComPtr<ID3D12Device> device;
    DXGI_ADAPTER_DESC1 desc = {};
    HRESULT last_device_hr = DXGI_ERROR_NOT_FOUND;
    uint32_t attempted = 0;
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        hr = factory->EnumAdapters1(i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND)
            break;
        if (FAILED(hr))
            return fail("EnumAdapters1", "adapter", hr);
        hr = adapter->GetDesc1(&desc);
        if (FAILED(hr))
            return fail("GetDesc1", "adapter", hr);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ||
            utf8(desc.Description).find("Microsoft Basic Render Driver") != std::string::npos)
            continue; // reject WARP and Microsoft Basic Render Driver
        ++attempted;
        last_device_hr = create(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (SUCCEEDED(last_device_hr))
            break;
    }
    if (!device)
        return fail("hardware device",
                    "no usable D3D12 adapter; attempted=" + std::to_string(attempted) +
                        "; last=" + utf8(desc.Description),
                    last_device_hr);
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_0};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))))
        return fail("shader model", "SM 6.0 unavailable");

    DflashGpuKv kv;
    std::string error;
    if (!kv.initialize(device.Get(), kLayers, kCapacity, kWidth, kWidth, &error))
        return fail("initialize KV", error);
    if (kv.diagnostics().allocated_bytes != uint64_t(kLayers) * kCapacity * kRowBytes)
        return fail("KV layout", "unexpected 20-byte block allocation");

    D3D12_ROOT_PARAMETER params[7] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = 4;
    for (uint32_t i = 0; i < 5; ++i) {
        params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1 + i].Descriptor.ShaderRegister = i;
    }
    params[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[6].Descriptor.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC root_desc = {};
    root_desc.NumParameters = 7;
    root_desc.pParameters = params;
    ComPtr<ID3DBlob> blob, messages;
    hr = serialize(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &messages);
    if (FAILED(hr))
        return fail("SerializeRootSignature", "attention", hr);
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr))
        return fail("CreateRootSignature", "attention", hr);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline = {};
    pipeline.pRootSignature = root.Get();
    pipeline.CS = {kDflashAttentionDxil, kDflashAttentionDxilSize};
    ComPtr<ID3D12PipelineState> pso;
    hr = device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&pso));
    if (FAILED(hr))
        return fail("CreateComputePipelineState", "attention DXIL", hr);

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
    if (FAILED(hr))
        return fail("CreateCommandQueue", "direct", hr);
    ComPtr<ID3D12CommandAllocator> allocator;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    if (FAILED(hr))
        return fail("CreateCommandAllocator", "direct", hr);
    ComPtr<ID3D12GraphicsCommandList> cmd;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                   IID_PPV_ARGS(&cmd));
    if (FAILED(hr))
        return fail("CreateCommandList", "direct", hr);
    ComPtr<ID3D12Fence> fence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr))
        return fail("CreateFence", "direct", hr);
    FenceEvent fence_event;
    fence_event.handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event.handle)
        return fail("CreateEventW", "fence event", HRESULT_FROM_WIN32(GetLastError()));

    std::vector<float> source(12 * kWidth);
    for (uint32_t p = 0; p < 5; ++p) {
        for (uint32_t kind = 0; kind < 2; ++kind) {
            for (uint32_t d = 0; d < kWidth; ++d)
                source[(p * 2 + kind) * kWidth + d] = kv_value(kPositions[p], kind, d, 0);
        }
    }
    for (uint32_t kind = 0; kind < 2; ++kind)
        for (uint32_t d = 0; d < kWidth; ++d)
            source[(10 + kind) * kWidth + d] = kv_value(2051, kind, d, 1);
    std::vector<float> queries(kOutputBytes / sizeof(float));
    for (uint32_t q = 0; q < kQueries; ++q)
        for (uint32_t h = 0; h < kQHeads; ++h)
            for (uint32_t d = 0; d < kDim; ++d)
                queries[(q * kQHeads + h) * kDim + d] = query_value(q, h, d);
    float sinks[kQHeads];
    for (uint32_t h = 0; h < kQHeads; ++h)
        sinks[h] = sink_value(h);

    auto source_upload = buffer(device.Get(), source.size() * sizeof(float), D3D12_HEAP_TYPE_UPLOAD,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &hr);
    if (!source_upload)
        return fail("source upload", "buffer", hr);
    auto query_upload = buffer(device.Get(), kOutputBytes, D3D12_HEAP_TYPE_UPLOAD,
                               D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &hr);
    if (!query_upload)
        return fail("query upload", "buffer", hr);
    auto query_pos_upload =
        buffer(device.Get(), sizeof(kQueryPositions), D3D12_HEAP_TYPE_UPLOAD,
               D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &hr);
    if (!query_pos_upload)
        return fail("query position upload", "buffer", hr);
    auto active_upload = buffer(device.Get(), sizeof(kPositions), D3D12_HEAP_TYPE_UPLOAD,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &hr);
    if (!active_upload)
        return fail("active position upload", "buffer", hr);
    std::array<uint8_t, sizeof(sinks) + sizeof(dflash_exp::kUcrtExpTable)> sink_payload{};
    std::memcpy(sink_payload.data(), sinks, sizeof(sinks));
    std::memcpy(sink_payload.data() + sizeof(sinks), dflash_exp::kUcrtExpTable,
                sizeof(dflash_exp::kUcrtExpTable));
    auto sink_upload = buffer(device.Get(), sink_payload.size(), D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &hr);
    if (!sink_upload)
        return fail("sink upload", "buffer", hr);
    auto output = buffer(device.Get(), kOutputBytes, D3D12_HEAP_TYPE_DEFAULT,
                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &hr);
    if (!output)
        return fail("attention output", "buffer", hr);
    auto readback = buffer(device.Get(), kOutputBytes, D3D12_HEAP_TYPE_READBACK,
                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &hr);
    if (!readback)
        return fail("attention readback", "buffer", hr);
    auto decoded = buffer(device.Get(), 2 * kSourceRowBytes, D3D12_HEAP_TYPE_DEFAULT,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &hr);
    if (!decoded)
        return fail("Q4 decoded output", "buffer", hr);
    auto decoded_readback = buffer(device.Get(), 2 * kSourceRowBytes, D3D12_HEAP_TYPE_READBACK,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &hr);
    if (!decoded_readback)
        return fail("Q4 decoded readback", "buffer", hr);
    auto upload = [&](ID3D12Resource* resource, const void* data, size_t bytes) {
        void* mapped = nullptr;
        const HRESULT status = resource->Map(0, nullptr, &mapped);
        if (FAILED(status))
            return status;
        std::memcpy(mapped, data, bytes);
        resource->Unmap(0, nullptr);
        return S_OK;
    };
    if (FAILED(hr = upload(source_upload.Get(), source.data(), source.size() * sizeof(float))) ||
        FAILED(hr = upload(query_upload.Get(), queries.data(), kOutputBytes)) ||
        FAILED(hr = upload(query_pos_upload.Get(), kQueryPositions, sizeof(kQueryPositions))) ||
        FAILED(hr = upload(sink_upload.Get(), sink_payload.data(), sink_payload.size())))
        return fail("upload", "Map/copy", hr);

    auto pack = [&](uint32_t position_index, uint32_t source_row) {
        const uint32_t p = kPositions[position_index];
        return kv.record_pack(cmd.Get(), kLayer, p, DflashGpuKv::Kind::Key, source_upload.Get(),
                              uint64_t(source_row) * kSourceRowBytes, &error) &&
               kv.record_pack(cmd.Get(), kLayer, p, DflashGpuKv::Kind::Value, source_upload.Get(),
                              uint64_t(source_row + 1) * kSourceRowBytes, &error);
    };
    for (uint32_t p = 0; p < 5; ++p)
        if (!pack(p, p * 2))
            return fail("record_pack", error);

    auto run_stage = [&](const char* name, uint32_t version, uint64_t fence_value,
                         bool check_layout) -> bool {
        const auto active = active_rows(kv);
        if (active.empty()) {
            error = "empty active positions";
            return false;
        }
        if (FAILED(hr = upload(active_upload.Get(), active.data(),
                               active.size() * sizeof(uint32_t)))) {
            error = "upload active positions";
            return false;
        }
        if (!kv.record_prepare_read(cmd.Get(), &error))
            return false;
        cmd->SetComputeRootSignature(root.Get());
        cmd->SetPipelineState(pso.Get());
        const uint32_t constants[4] = {kQueries, uint32_t(active.size()), kCapacity,
                                       uint32_t(uint64_t(kLayer) * kCapacity * kRowBytes)};
        cmd->SetComputeRoot32BitConstants(0, 4, constants, 0);
        cmd->SetComputeRootShaderResourceView(1, query_upload->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(2, kv.resource()->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(3, query_pos_upload->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(4, active_upload->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(5, sink_upload->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(6, output->GetGPUVirtualAddress());
        cmd->Dispatch(kQHeads, kQueries, 1);
        if (check_layout) {
            uint64_t key_offset = 0, value_offset = 0;
            if (!kv.gpu_byte_offset(kLayer, 3, DflashGpuKv::Kind::Key, &key_offset) ||
                !kv.gpu_byte_offset(kLayer, 3, DflashGpuKv::Kind::Value, &value_offset) ||
                key_offset != (uint64_t(kLayer) * kCapacity + 3) * kRowBytes ||
                value_offset != key_offset + 640u) {
                error = "20-byte Q4 layout offsets";
                return false;
            }
            if (!kv.record_unpack(cmd.Get(), kLayer, 3, DflashGpuKv::Kind::Key, decoded.Get(), 0,
                                  &error) ||
                !kv.record_unpack(cmd.Get(), kLayer, 3, DflashGpuKv::Kind::Value, decoded.Get(),
                                  kSourceRowBytes, &error))
                return false;
            transition(cmd.Get(), decoded.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            cmd->CopyResource(decoded_readback.Get(), decoded.Get());
            transition(cmd.Get(), decoded.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        transition(cmd.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd->CopyResource(readback.Get(), output.Get());
        transition(cmd.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (!submit(device.Get(), queue.Get(), cmd.Get(), fence.Get(), fence_event.handle,
                    fence_value, &hr)) {
            error = "submit/fence/device removal";
            return false;
        }
        void* mapped = nullptr;
        hr = readback->Map(0, nullptr, &mapped);
        if (FAILED(hr)) {
            error = "Map attention readback";
            return false;
        }
        const float* actual = static_cast<const float*>(mapped);
        float max_error = 0.0f;
        bool correct = true;
        for (uint32_t q = 0; q < kQueries; ++q)
            for (uint32_t h = 0; h < kQHeads; ++h)
                for (uint32_t d = 0; d < kDim; ++d) {
                    const size_t index = (q * kQHeads + h) * kDim + d;
                    const float expected = reference(q, h, d, active, version);
                    if (!std::isfinite(actual[index]))
                        correct = false;
                    else
                        max_error = std::max(max_error, std::fabs(actual[index] - expected));
                }
        readback->Unmap(0, nullptr);
        if (!correct || max_error > 3e-5f) {
            error = "attention reference mismatch, maxAbsErr=" + std::to_string(max_error);
            return false;
        }
        if (check_layout) {
            hr = decoded_readback->Map(0, nullptr, &mapped);
            if (FAILED(hr)) {
                error = "Map decoded Q4 readback";
                return false;
            }
            const float* values = static_cast<const float*>(mapped);
            bool exact = true;
            for (uint32_t kind = 0; kind < 2; ++kind)
                for (uint32_t d = 0; d < kWidth; ++d)
                    if (values[kind * kWidth + d] != kv_value(3, kind, d, 0))
                        exact = false;
            decoded_readback->Unmap(0, nullptr);
            if (!exact) {
                error = "Q4 pack/unpack layout mismatch";
                return false;
            }
        }
        add_stage(name, max_error, uint32_t(active.size()));
        return true;
    };

    if (!run_stage("initial_q4_swa", 0, 1, true))
        return fail("initial_q4_swa", error, hr);
    // The previous command list is fenced before logical invalidation.
    if (!kv.trim(2051))
        return fail("trim", "valid trim refused");
    if (kv.valid(kLayer, 2051, DflashGpuKv::Kind::Key) ||
        kv.valid(kLayer, 2051, DflashGpuKv::Kind::Value))
        return fail("trim", "stale suffix remained valid");
    hr = allocator->Reset();
    if (FAILED(hr))
        return fail("Reset allocator", "trimmed stage", hr);
    hr = cmd->Reset(allocator.Get(), nullptr);
    if (FAILED(hr))
        return fail("Reset command list", "trimmed stage", hr);
    if (!run_stage("trimmed_after_fence", 0, 2, false))
        return fail("trimmed_after_fence", error, hr);

    hr = allocator->Reset();
    if (FAILED(hr))
        return fail("Reset allocator", "reinjection stage", hr);
    hr = cmd->Reset(allocator.Get(), nullptr);
    if (FAILED(hr))
        return fail("Reset command list", "reinjection stage", hr);
    if (!pack(4, 10))
        return fail("record reinjection", error);
    if (!run_stage("reinject_after_fence", 1, 3, false))
        return fail("reinject_after_fence", error, hr);

    result.passed = true;
    result.json = "{\"passed\":true,\"probe\":\"dflash_attention_q4_0\",\"adapter\":" +
                  quote(utf8(desc.Description)) +
                  ",\"hardware\":true,\"shaderModel\":\"6.0\",\"queryCount\":2" +
                  ",\"swa\":2048,\"fencesCompleted\":3,\"stages\":" + stages +
                  "],\"cpuExpfCheck\":" + cpu_expf_check() + "}";
    return result;
}
