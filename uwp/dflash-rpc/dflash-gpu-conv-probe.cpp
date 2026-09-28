#include "dflash-gpu-conv-probe.h"

#include "dflash_dynamic_conv_dxil.h"
#include "xllama/d3d12_dyn.h"
#include "gguf.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint32_t kHidden = 5120;
constexpr uint32_t kGroups = 320;
constexpr uint32_t kGroupSize = 16;
constexpr uint32_t kTaps = 2;
constexpr uint32_t kSides = 2;
constexpr uint32_t kMaxTokens = 8;
constexpr uint64_t kBaseBytes = uint64_t(kHidden) * kTaps * kSides * sizeof(float);
constexpr uint64_t kHiddenBytes = uint64_t(kHidden) * kMaxTokens * sizeof(float);
constexpr uint64_t kDynamicBytes = uint64_t(kGroups) * kTaps * kSides * kMaxTokens * sizeof(float);

std::string quote(const std::string & value) {
    std::string result = "\"";
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') result += '\\';
        if (ch >= 32) result += static_cast<char>(ch);
    }
    return result + '"';
}

std::string utf8(const wchar_t * value) {
    const int length = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (length <= 1) return {};
    std::string result(static_cast<size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), length, nullptr, nullptr);
    result.pop_back();
    return result;
}

dflash_gpu_conv_probe_result failure(const char * stage, const char * detail, HRESULT hr = S_OK) {
    dflash_gpu_conv_probe_result result;
    result.json = "{\"passed\":false,\"stage\":" + quote(stage) +
                  ",\"error\":" + quote(detail);
    if (FAILED(hr)) {
        char buffer[16];
        std::snprintf(buffer, sizeof(buffer), "0x%08X", static_cast<unsigned>(hr));
        result.json += ",\"hresult\":" + quote(buffer);
    }
    result.json += '}';
    return result;
}

ComPtr<ID3D12Resource> make_buffer(ID3D12Device * device, uint64_t bytes,
                                   D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_FLAGS flags,
                                   D3D12_RESOURCE_STATES state, HRESULT * hr) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = heap_type;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    *hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                          state, nullptr, IID_PPV_ARGS(&resource));
    return resource;
}

struct Event {
    HANDLE handle = nullptr;
    ~Event() { if (handle) CloseHandle(handle); }
};

void transition(ID3D12GraphicsCommandList * cmd, ID3D12Resource * resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->ResourceBarrier(1, &barrier);
}

// Equivalent to dflash.cpp's coeff reshape/view/repeat, base-side view, and
// concat([zeros, previous], token axis) independently of the shader addresses.
float cpu_value(const std::vector<float> & hidden, const std::vector<float> & dynamic,
                const float * base, uint32_t token, uint32_t channel,
                uint32_t block_size, uint32_t side) {
    const uint32_t group = channel / kGroupSize;
    const uint32_t local = token % block_size;
    float result = 0.0f;
    for (uint32_t tap = 0; tap < kTaps; ++tap) {
        const float coefficient = dynamic[(((token * kSides + side) * kTaps + tap) * kGroups) + group];
        const float weight = base[((side * kTaps + tap) * kHidden) + channel] + coefficient;
        const float value = local >= tap ? hidden[(token - tap) * kHidden + channel] : 0.0f;
        const float term = weight * value;
        result = tap == 0 ? term : result + term;
    }
    return result;
}

void fill_inputs(std::vector<float> * hidden, std::vector<float> * dynamic,
                 uint32_t tokens, uint32_t blocks) {
    const uint32_t block_size = tokens / blocks;
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < kHidden; ++c) {
            const int value = static_cast<int>((c * 73u + t * 109u + 23u) % 997u) - 498;
            // A large last row makes an incorrect cross-block tap conspicuous.
            (*hidden)[t * kHidden + c] = (blocks > 1 && t + 1 < tokens &&
                (t + 1) % block_size == 0) ? 50.0f : value * 0.00317f;
        }
        for (uint32_t side = 0; side < kSides; ++side) {
            for (uint32_t tap = 0; tap < kTaps; ++tap) {
                for (uint32_t group = 0; group < kGroups; ++group) {
                    const int value = static_cast<int>((group * 131u + t * 47u +
                        side * 79u + tap * 61u + 17u) % 503u) - 251;
                    (*dynamic)[(((t * kSides + side) * kTaps + tap) * kGroups) + group] =
                        value * 0.00271f;
                }
            }
        }
    }
}

} // namespace

dflash_gpu_conv_probe_result run_dflash_gpu_conv_probe(const float * base, size_t base_bytes) {
    if (!base || base_bytes != kBaseBytes)
        return failure("base input", "expected GGUF F32 [5120,2,2] tensor");
    for (uint32_t i = 0; i < kHidden * kTaps * kSides; ++i)
        if (!std::isfinite(base[i])) return failure("base input", "nonfinite weight");

    const auto create_device = xllama::d3d12_dyn::CreateDevice();
    const auto serialize = xllama::d3d12_dyn::SerializeRootSignature();
    if (!create_device || !serialize)
        return failure("D3D12 entry points", "runtime functions unavailable");
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return failure("CreateDXGIFactory1", "factory", hr);
    ComPtr<ID3D12Device> device;
    DXGI_ADAPTER_DESC1 adapter_desc = {};
    HRESULT last_hr = DXGI_ERROR_NOT_FOUND;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        hr = factory->EnumAdapters1(index, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(hr)) return failure("EnumAdapters1", "adapter enumeration", hr);
        hr = adapter->GetDesc1(&adapter_desc);
        if (FAILED(hr)) return failure("GetDesc1", "adapter description", hr);
        if (adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        last_hr = create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (SUCCEEDED(last_hr)) break;
    }
    if (!device) return failure("hardware D3D12 device", "no usable hardware adapter", last_hr);
    D3D12_FEATURE_DATA_SHADER_MODEL shader_model = {D3D_SHADER_MODEL_6_0};
    hr = device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shader_model, sizeof(shader_model));
    if (FAILED(hr)) return failure("shader model", "SM6.0 unavailable", hr);

    D3D12_ROOT_PARAMETER params[5] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (uint32_t slot = 0; slot < 3; ++slot) {
        params[slot + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[slot + 1].Descriptor.ShaderRegister = slot;
        params[slot + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[4].Descriptor.ShaderRegister = 0;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC root_desc = {};
    root_desc.NumParameters = 5;
    root_desc.pParameters = params;
    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> messages;
    hr = serialize(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &messages);
    if (FAILED(hr)) return failure("SerializeRootSignature", "root signature", hr);
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr)) return failure("CreateRootSignature", "root signature", hr);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline = {};
    pipeline.pRootSignature = root.Get();
    pipeline.CS = {kDflashDynamicConvDxil, kDflashDynamicConvDxilSize};
    ComPtr<ID3D12PipelineState> pso;
    hr = device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) return failure("CreateComputePipelineState", "dynamic conv DXIL", hr);

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) return failure("CreateCommandQueue", "direct queue", hr);
    ComPtr<ID3D12CommandAllocator> allocator;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    if (FAILED(hr)) return failure("CreateCommandAllocator", "direct allocator", hr);
    ComPtr<ID3D12GraphicsCommandList> cmd;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                   IID_PPV_ARGS(&cmd));
    if (FAILED(hr)) return failure("CreateCommandList", "direct command list", hr);
    ComPtr<ID3D12Fence> fence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) return failure("CreateFence", "GPU fence", hr);
    Event event;
    event.handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event.handle) return failure("CreateEventW", "GPU fence event", HRESULT_FROM_WIN32(GetLastError()));

    auto hidden_gpu = make_buffer(device.Get(), kHiddenBytes, D3D12_HEAP_TYPE_UPLOAD,
                                  D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &hr);
    if (!hidden_gpu) return failure("hidden upload", "CreateCommittedResource", hr);
    auto dynamic_gpu = make_buffer(device.Get(), kDynamicBytes, D3D12_HEAP_TYPE_UPLOAD,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &hr);
    if (!dynamic_gpu) return failure("dynamic upload", "CreateCommittedResource", hr);
    auto base_gpu = make_buffer(device.Get(), kBaseBytes, D3D12_HEAP_TYPE_UPLOAD,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &hr);
    if (!base_gpu) return failure("base upload", "CreateCommittedResource", hr);
    auto output_gpu = make_buffer(device.Get(), kHiddenBytes, D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &hr);
    if (!output_gpu) return failure("output UAV", "CreateCommittedResource", hr);
    auto readback = make_buffer(device.Get(), kHiddenBytes, D3D12_HEAP_TYPE_READBACK,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &hr);
    if (!readback) return failure("readback", "CreateCommittedResource", hr);
    void * mapped = nullptr;
    hr = base_gpu->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return failure("Map base", "upload", hr);
    std::memcpy(mapped, base, kBaseBytes);
    base_gpu->Unmap(0, nullptr);

    const uint32_t token_counts[] = {2, 3, 5, 7, 8};
    std::string cases = "[";
    uint32_t case_count = 0;
    uint32_t boundary_cases = 0;
    float global_max_error = 0.0f;
    uint64_t fence_value = 0;
    for (uint32_t tokens : token_counts) {
        const uint32_t block_counts[] = {1, tokens, tokens == 8 ? 2u : 0u,
                                         tokens == 8 ? 4u : 0u};
        for (uint32_t blocks : block_counts) {
            if (!blocks) continue;
            std::vector<float> hidden(size_t(tokens) * kHidden);
            std::vector<float> dynamic(size_t(tokens) * kSides * kTaps * kGroups);
            fill_inputs(&hidden, &dynamic, tokens, blocks);
            hr = hidden_gpu->Map(0, nullptr, &mapped);
            if (FAILED(hr)) return failure("Map hidden", "upload", hr);
            std::memcpy(mapped, hidden.data(), hidden.size() * sizeof(float));
            hidden_gpu->Unmap(0, nullptr);
            hr = dynamic_gpu->Map(0, nullptr, &mapped);
            if (FAILED(hr)) return failure("Map dynamic", "upload", hr);
            std::memcpy(mapped, dynamic.data(), dynamic.size() * sizeof(float));
            dynamic_gpu->Unmap(0, nullptr);

            for (uint32_t side = 0; side < kSides; ++side) {
                if (case_count) {
                    hr = allocator->Reset();
                    if (FAILED(hr)) return failure("Reset allocator", "after fence", hr);
                    hr = cmd->Reset(allocator.Get(), nullptr);
                    if (FAILED(hr)) return failure("Reset command list", "after fence", hr);
                }
                cmd->SetComputeRootSignature(root.Get());
                cmd->SetPipelineState(pso.Get());
                const uint32_t constants[4] = {tokens, blocks, side, 0};
                cmd->SetComputeRoot32BitConstants(0, 4, constants, 0);
                cmd->SetComputeRootShaderResourceView(1, hidden_gpu->GetGPUVirtualAddress());
                cmd->SetComputeRootShaderResourceView(2, dynamic_gpu->GetGPUVirtualAddress());
                cmd->SetComputeRootShaderResourceView(3, base_gpu->GetGPUVirtualAddress());
                cmd->SetComputeRootUnorderedAccessView(4, output_gpu->GetGPUVirtualAddress());
                cmd->Dispatch((tokens * kHidden + 63) / 64, 1, 1);
                transition(cmd.Get(), output_gpu.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                cmd->CopyBufferRegion(readback.Get(), 0, output_gpu.Get(), 0,
                                      uint64_t(tokens) * kHidden * sizeof(float));
                transition(cmd.Get(), output_gpu.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                hr = cmd->Close();
                if (FAILED(hr)) return failure("Close command list", "dispatch", hr);
                ID3D12CommandList * lists[] = {cmd.Get()};
                queue->ExecuteCommandLists(1, lists);
                hr = queue->Signal(fence.Get(), ++fence_value);
                if (FAILED(hr)) return failure("Signal queue", "dispatch", hr);
                if (fence->GetCompletedValue() < fence_value) {
                    hr = fence->SetEventOnCompletion(fence_value, event.handle);
                    if (FAILED(hr)) return failure("SetEventOnCompletion", "dispatch", hr);
                    const DWORD wait = WaitForSingleObject(event.handle, 30000);
                    if (wait != WAIT_OBJECT_0)
                        return failure("GPU fence wait", "timeout or wait failure",
                                       wait == WAIT_TIMEOUT ? HRESULT_FROM_WIN32(WAIT_TIMEOUT) :
                                           HRESULT_FROM_WIN32(GetLastError()));
                }
                hr = device->GetDeviceRemovedReason();
                if (FAILED(hr)) return failure("device removed", "dispatch", hr);
                hr = readback->Map(0, nullptr, &mapped);
                if (FAILED(hr)) return failure("Map readback", "output", hr);
                const float * actual = static_cast<const float *>(mapped);
                const uint32_t block_size = tokens / blocks;
                uint32_t nonfinite = 0;
                uint32_t mismatch = 0;
                float max_error = 0.0f;
                for (uint32_t token = 0; token < tokens; ++token) {
                    for (uint32_t channel = 0; channel < kHidden; ++channel) {
                        const size_t index = size_t(token) * kHidden + channel;
                        const float expected = cpu_value(hidden, dynamic, base, token,
                                                         channel, block_size, side);
                        const float value = actual[index];
                        if (!std::isfinite(value)) { ++nonfinite; continue; }
                        const float error = std::fabs(value - expected);
                        max_error = std::max(max_error, error);
                        if (error > 2e-6f + 2e-6f * std::fabs(expected)) ++mismatch;
                    }
                }
                readback->Unmap(0, nullptr);
                global_max_error = std::max(global_max_error, max_error);
                if (blocks > 1) ++boundary_cases;
                if (case_count) cases += ',';
                cases += "{\"tokens\":" + std::to_string(tokens) +
                         ",\"blocks\":" + std::to_string(blocks) +
                         ",\"side\":" + std::to_string(side) +
                         ",\"elements\":" + std::to_string(size_t(tokens) * kHidden) +
                         ",\"boundaryPadding\":" + (blocks > 1 ? "true" : "false") +
                         ",\"nonFinite\":" + std::to_string(nonfinite) +
                         ",\"mismatches\":" + std::to_string(mismatch) +
                         ",\"maxAbsErr\":" + std::to_string(max_error) + "}";
                ++case_count;
                if (mismatch || nonfinite) {
                    dflash_gpu_conv_probe_result result;
                    result.json = "{\"passed\":false,\"stage\":\"compare\",\"hardware\":true," 
                                  "\"adapter\":" + quote(utf8(adapter_desc.Description)) +
                                  ",\"cases\":" + cases + "]}";
                    return result;
                }
            }
        }
    }
    cases += ']';
    dflash_gpu_conv_probe_result result;
    result.passed = true;
    result.json = "{\"passed\":true,\"stage\":\"complete\",\"probe\":\"dflash_dynamic_conv\"," 
                  "\"hardware\":true,\"adapter\":" + quote(utf8(adapter_desc.Description)) +
                  ",\"shaderModel6\":true,\"stages\":{\"device\":true,\"pipeline\":true," 
                  "\"dispatch\":true,\"readback\":true,\"compare\":true}," 
                  "\"caseCount\":" + std::to_string(case_count) +
                  ",\"boundaryPaddingCases\":" + std::to_string(boundary_cases) +
                  ",\"maxAbsErr\":" + std::to_string(global_max_error) +
                  ",\"cases\":" + cases + "}";
    return result;
}

dflash_gpu_conv_probe_result run_dflash_gpu_conv_probe(const std::string & model_path) {
    if (model_path.empty()) return failure("GGUF path", "empty model path");
    using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    gguf_ptr gguf(gguf_init_from_file(model_path.c_str(), {true, nullptr}), &gguf_free);
    if (!gguf) return failure("GGUF metadata", "could not open model");
    const int64_t id = gguf_find_tensor(gguf.get(), "blk.0.attn_conv_base");
    if (id < 0) return failure("GGUF tensor", "blk.0.attn_conv_base missing");
    const int64_t * shape = gguf_get_tensor_ne(gguf.get(), id);
    const size_t bytes = gguf_get_tensor_size(gguf.get(), id);
    if (gguf_get_tensor_type(gguf.get(), id) != GGML_TYPE_F32 ||
        shape[0] != kHidden || shape[1] != kTaps || shape[2] != kSides ||
        shape[3] != 1 || bytes != kBaseBytes)
        return failure("GGUF shape", "base must be F32 [5120,2,2] and 81920 bytes");

    const size_t data_offset = gguf_get_data_offset(gguf.get());
    const size_t tensor_offset = gguf_get_tensor_offset(gguf.get(), id);
    if (tensor_offset > std::numeric_limits<size_t>::max() - data_offset)
        return failure("GGUF bounds", "tensor offset overflow");
    const size_t absolute_offset = data_offset + tensor_offset;
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) return failure("GGUF file", "could not open model bytes");
    const std::streampos end = file.tellg();
    if (end < 0 || static_cast<uint64_t>(end) < absolute_offset ||
        bytes > static_cast<uint64_t>(end) - absolute_offset)
        return failure("GGUF bounds", "tensor exceeds file size");
    std::vector<float> base(static_cast<size_t>(kHidden) * kTaps * kSides);
    file.seekg(static_cast<std::streamoff>(absolute_offset));
    file.read(reinterpret_cast<char *>(base.data()), static_cast<std::streamsize>(bytes));
    if (!file) return failure("GGUF read", "short base tensor read");
    return run_dflash_gpu_conv_probe(base.data(), bytes);
}
