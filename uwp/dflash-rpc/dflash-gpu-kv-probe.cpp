#include "dflash-gpu-kv-probe.h"

#include "dflash-gpu-kv.h"
#include "ggml-quants.h"
#include "xllama/d3d12_dyn.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint32_t kWidth = 1024;
constexpr uint32_t kLayers = 5;
constexpr uint32_t kCapacity = 8;
constexpr uint32_t kLayer = 2;
constexpr uint32_t kKeepPosition = 1;
constexpr uint32_t kSuffixPosition = 4;
constexpr uint64_t kRowBytes = uint64_t(kWidth) * sizeof(float);

std::string json_string(const std::string& input) {
    std::string output = "\"";
    for (unsigned char c : input) {
        if (c == '"' || c == '\\')
            output += '\\';
        if (c >= 32)
            output += static_cast<char>(c);
    }
    return output + '"';
}

std::string utf8(const wchar_t* input) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, input, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1)
        return {};
    std::string output(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, input, -1, output.data(), size, nullptr, nullptr);
    output.pop_back();
    return output;
}

dflash_gpu_kv_probe_result failure(const char* stage, const std::string& detail,
                                   HRESULT hr = S_OK) {
    dflash_gpu_kv_probe_result result;
    result.json =
        "{\"passed\":false,\"stage\":" + json_string(stage) + ",\"error\":" + json_string(detail);
    if (hr != S_OK) {
        char code[16];
        std::snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned>(hr));
        result.json += ",\"hresult\":" + json_string(code);
    }
    result.json += '}';
    return result;
}

ComPtr<ID3D12Resource> make_buffer(ID3D12Device* device, uint64_t bytes, D3D12_HEAP_TYPE heap_type,
                                   D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                   HRESULT* result) {
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
    *result = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                              IID_PPV_ARGS(&resource));
    return resource;
}

struct FenceEvent {
    HANDLE handle = nullptr;
    ~FenceEvent() {
        if (handle)
            CloseHandle(handle);
    }
};

bool submit_and_wait(ID3D12Device* device, ID3D12CommandQueue* queue,
                     ID3D12GraphicsCommandList* cmd, ID3D12Fence* fence, HANDLE event_handle,
                     uint64_t fence_value, std::string* error, HRESULT* status) {
    *status = cmd->Close();
    if (FAILED(*status)) {
        *error = "Close command list";
        return false;
    }
    ID3D12CommandList* lists[] = {cmd};
    queue->ExecuteCommandLists(1, lists);
    *status = queue->Signal(fence, fence_value);
    if (FAILED(*status)) {
        *error = "Signal queue";
        return false;
    }
    if (fence->GetCompletedValue() < fence_value) {
        *status = fence->SetEventOnCompletion(fence_value, event_handle);
        if (FAILED(*status)) {
            *error = "SetEventOnCompletion";
            return false;
        }
        const DWORD wait = WaitForSingleObject(event_handle, 30000);
        if (wait != WAIT_OBJECT_0) {
            *status = wait == WAIT_TIMEOUT ? HRESULT_FROM_WIN32(WAIT_TIMEOUT)
                                           : HRESULT_FROM_WIN32(GetLastError());
            *error = "GPU fence wait";
            return false;
        }
    }
    *status = device->GetDeviceRemovedReason();
    if (FAILED(*status)) {
        *error = "D3D12 device removed";
        return false;
    }
    return true;
}

void fill_row(float* row, uint32_t seed) {
    for (uint32_t i = 0; i < kWidth; ++i) {
        const int value = static_cast<int>((i * 73u + seed * 41u) % 997u) - 498;
        row[i] = static_cast<float>(value) * 0.00317f;
    }
    if (seed == 1) {
        // Rounding the F32 addition gives 11 before integer truncation;
        // truncating the unrounded sum would incorrectly store nibble 10.
        for (uint32_t i = 0; i < 32; ++i)
            row[i] = 0.0f;
        row[0] = -8.0f;
        row[30] = std::nextafter(2.5f, 0.0f);
    }
}

} // namespace

dflash_gpu_kv_probe_result run_dflash_gpu_kv_probe() {
    auto create_device = xllama::d3d12_dyn::CreateDevice();
    if (!create_device)
        return failure("D3D12CreateDevice", "entry point unavailable");
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr))
        return failure("CreateDXGIFactory1", "factory", hr);
    ComPtr<ID3D12Device> device;
    DXGI_ADAPTER_DESC1 adapter_desc = {};
    HRESULT last_device_hr = DXGI_ERROR_NOT_FOUND;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        hr = factory->EnumAdapters1(index, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND)
            break;
        if (FAILED(hr))
            return failure("EnumAdapters1", "adapter enumeration", hr);
        hr = adapter->GetDesc1(&adapter_desc);
        if (FAILED(hr))
            return failure("GetDesc1", "adapter description", hr);
        if (adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;
        last_device_hr =
            create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (SUCCEEDED(last_device_hr))
            break;
    }
    if (!device)
        return failure("hardware D3D12 device", "no usable hardware adapter", last_device_hr);

    DflashGpuKv kv;
    std::string error;
    if (!kv.initialize(device.Get(), kLayers, kCapacity, kWidth, kWidth, &error))
        return failure("initialize KV", error);
    const auto initialized = kv.diagnostics();
    const uint64_t expected_kv_bytes = uint64_t(kLayers) * kCapacity * 2 *
                                       (kWidth / DflashGpuKv::kQuantBlockWidth) *
                                       DflashGpuKv::kGpuBlockBytes;
    if (initialized.allocated_bytes != expected_kv_bytes)
        return failure("KV resource size", "unexpected allocation size");

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
    if (FAILED(hr))
        return failure("CreateCommandQueue", "direct queue", hr);
    ComPtr<ID3D12CommandAllocator> allocator;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    if (FAILED(hr))
        return failure("CreateCommandAllocator", "direct allocator", hr);
    ComPtr<ID3D12GraphicsCommandList> cmd;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                   IID_PPV_ARGS(&cmd));
    if (FAILED(hr))
        return failure("CreateCommandList", "direct command list", hr);
    ComPtr<ID3D12Fence> fence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr))
        return failure("CreateFence", "GPU fence", hr);
    FenceEvent fence_event;
    fence_event.handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event.handle)
        return failure("CreateEventW", "GPU fence event", HRESULT_FROM_WIN32(GetLastError()));

    // Rows 0/1: retained prefix; 2/3: stale suffix; 4/5: reinjected suffix.
    std::vector<float> input(6 * kWidth);
    for (uint32_t row = 0; row < 6; ++row)
        fill_row(input.data() + row * kWidth, row + 1);
    auto upload = make_buffer(device.Get(), input.size() * sizeof(float), D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &hr);
    if (!upload)
        return failure("CreateCommittedResource upload", "F32 input", hr);
    auto output = make_buffer(device.Get(), 4 * kRowBytes, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &hr);
    if (!output)
        return failure("CreateCommittedResource output", "F32 output", hr);
    auto readback = make_buffer(device.Get(), 4 * kRowBytes, D3D12_HEAP_TYPE_READBACK,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &hr);
    if (!readback)
        return failure("CreateCommittedResource readback", "F32 readback", hr);
    void* mapped = nullptr;
    hr = upload->Map(0, nullptr, &mapped);
    if (FAILED(hr))
        return failure("Map upload", "F32 input", hr);
    std::memcpy(mapped, input.data(), input.size() * sizeof(float));
    upload->Unmap(0, nullptr);

    auto pack = [&](uint32_t position, DflashGpuKv::Kind kind, uint32_t row,
                    uint32_t layer = kLayer) {
        return kv.record_pack(cmd.Get(), layer, position, kind, upload.Get(),
                              uint64_t(row) * kRowBytes, &error);
    };
    if (!pack(kKeepPosition, DflashGpuKv::Kind::Key, 0) ||
        !pack(kKeepPosition, DflashGpuKv::Kind::Value, 1) ||
        !pack(kSuffixPosition, DflashGpuKv::Kind::Key, 2) ||
        !pack(kSuffixPosition, DflashGpuKv::Kind::Value, 3) ||
        !pack(kSuffixPosition, DflashGpuKv::Kind::Key, 2, kLayers - 1))
        return failure("record initial pack", error);
    if (!submit_and_wait(device.Get(), queue.Get(), cmd.Get(), fence.Get(), fence_event.handle, 1,
                         &error, &hr))
        return failure("submit initial pack", error, hr);

    // Keep the prefix and reject the still-resident suffix before overwrite.
    if (!kv.trim(kKeepPosition + 1))
        return failure("trim", "valid trim refused");
    const bool prefix_preserved = kv.valid(kLayer, kKeepPosition, DflashGpuKv::Kind::Key) &&
                                  kv.valid(kLayer, kKeepPosition, DflashGpuKv::Kind::Value);
    const bool stale_invalid = !kv.valid(kLayer, kSuffixPosition, DflashGpuKv::Kind::Key) &&
                               !kv.valid(kLayer, kSuffixPosition, DflashGpuKv::Kind::Value) &&
                               !kv.valid(kLayers - 1, kSuffixPosition, DflashGpuKv::Kind::Key);
    uint64_t stale_offset = 0;
    const bool stale_address_rejected =
        !kv.gpu_byte_offset(kLayer, kSuffixPosition, DflashGpuKv::Kind::Key, &stale_offset);
    if (!prefix_preserved || !stale_invalid || !stale_address_rejected)
        return failure("trim stale suffix", "invalid position remained addressable");

    hr = allocator->Reset();
    if (FAILED(hr))
        return failure("Reset allocator", "after initial fence", hr);
    hr = cmd->Reset(allocator.Get(), nullptr);
    if (FAILED(hr))
        return failure("Reset command list", "after initial fence", hr);
    const bool stale_read_rejected = !kv.record_unpack(
        cmd.Get(), kLayer, kSuffixPosition, DflashGpuKv::Kind::Key, output.Get(), 0, &error);
    if (!stale_read_rejected)
        return failure("stale read", "invalid suffix was readable");
    if (!pack(kSuffixPosition, DflashGpuKv::Kind::Key, 4) ||
        !pack(kSuffixPosition, DflashGpuKv::Kind::Value, 5))
        return failure("record reinjection", error);
    const bool reinjected_valid = kv.valid(kLayer, kSuffixPosition, DflashGpuKv::Kind::Key) &&
                                  kv.valid(kLayer, kSuffixPosition, DflashGpuKv::Kind::Value) &&
                                  !kv.valid(kLayers - 1, kSuffixPosition, DflashGpuKv::Kind::Key);
    if (!reinjected_valid)
        return failure("reinjection validity", "wrong layer or kind became valid");

    const DflashGpuKv::Kind kinds[4] = {DflashGpuKv::Kind::Key, DflashGpuKv::Kind::Value,
                                        DflashGpuKv::Kind::Key, DflashGpuKv::Kind::Value};
    const uint32_t positions[4] = {kKeepPosition, kKeepPosition, kSuffixPosition, kSuffixPosition};
    for (uint32_t row = 0; row < 4; ++row) {
        if (!kv.record_unpack(cmd.Get(), kLayer, positions[row], kinds[row], output.Get(),
                              uint64_t(row) * kRowBytes, &error))
            return failure("record unpack", error);
    }
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = output.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->ResourceBarrier(1, &barrier);
    cmd->CopyResource(readback.Get(), output.Get());
    if (!submit_and_wait(device.Get(), queue.Get(), cmd.Get(), fence.Get(), fence_event.handle, 2,
                         &error, &hr))
        return failure("submit reinjection and readback", error, hr);

    std::vector<float> expected(4 * kWidth);
    const uint32_t source_rows[4] = {0, 1, 4, 5};
    std::vector<block_q4_0> blocks(kWidth / 32);
    for (uint32_t row = 0; row < 4; ++row) {
        quantize_row_q4_0_ref(input.data() + source_rows[row] * kWidth, blocks.data(), kWidth);
        dequantize_row_q4_0(blocks.data(), expected.data() + row * kWidth, kWidth);
    }
    hr = readback->Map(0, nullptr, &mapped);
    if (FAILED(hr))
        return failure("Map readback", "F32 output", hr);
    const float* actual = static_cast<const float*>(mapped);
    uint32_t exact = 0;
    uint32_t first_mismatch = 4 * kWidth;
    uint32_t signed_zero_mismatches = 0;
    uint32_t other_mismatches = 0;
    uint32_t first_actual_bits = 0;
    uint32_t first_expected_bits = 0;
    uint32_t nonfinite = 0;
    float max_abs_err = 0.0f;
    for (uint32_t i = 0; i < 4 * kWidth; ++i) {
        if (std::memcmp(actual + i, expected.data() + i, sizeof(float)) == 0)
            ++exact;
        else {
            if (first_mismatch == 4 * kWidth) {
                first_mismatch = i;
                std::memcpy(&first_actual_bits, actual + i, sizeof(uint32_t));
                std::memcpy(&first_expected_bits, expected.data() + i, sizeof(uint32_t));
            }
            if (actual[i] == 0.0f && expected[i] == 0.0f)
                ++signed_zero_mismatches;
            else
                ++other_mismatches;
        }
        if (std::isfinite(actual[i]))
            max_abs_err = std::fmax(max_abs_err, std::fabs(actual[i] - expected[i]));
        else
            ++nonfinite;
    }
    readback->Unmap(0, nullptr);

    kv.reset(); // second fence completed before logical reset
    const bool reset_invalid = !kv.valid(kLayer, kKeepPosition, DflashGpuKv::Kind::Key) &&
                               !kv.valid(kLayer, kSuffixPosition, DflashGpuKv::Kind::Value) &&
                               !kv.valid(kLayers - 1, kSuffixPosition, DflashGpuKv::Kind::Key);
    const bool reset_read_rejected = !kv.record_unpack(
        cmd.Get(), kLayer, kKeepPosition, DflashGpuKv::Kind::Key, output.Get(), 0, &error);
    const auto diagnostic = kv.diagnostics();
    const bool passed = exact == 4 * kWidth && nonfinite == 0 && reset_invalid &&
                        reset_read_rejected && diagnostic.valid_key_positions == 0 &&
                        diagnostic.valid_value_positions == 0;
    const char* final_stage =
        passed ? "complete" : (exact != 4 * kWidth || nonfinite ? "compare" : "state");
    char actual_bits_text[11];
    char expected_bits_text[11];
    std::snprintf(actual_bits_text, sizeof(actual_bits_text), "0x%08X", first_actual_bits);
    std::snprintf(expected_bits_text, sizeof(expected_bits_text), "0x%08X", first_expected_bits);
    dflash_gpu_kv_probe_result result;
    result.passed = passed;
    result.json =
        "{\"passed\":" + std::string(passed ? "true" : "false") +
        ",\"stage\":" + json_string(final_stage) +
        ",\"probe\":\"kv_q4_0_1024\",\"adapter\":" + json_string(utf8(adapter_desc.Description)) +
        ",\"hardware\":true,\"layers\":" + std::to_string(kLayers) +
        ",\"capacity\":" + std::to_string(kCapacity) + ",\"keyWidth\":" + std::to_string(kWidth) +
        ",\"valueWidth\":" + std::to_string(kWidth) +
        ",\"gpuBlockBytes\":" + std::to_string(DflashGpuKv::kGpuBlockBytes) +
        ",\"kvResourceBytes\":" + std::to_string(diagnostic.allocated_bytes) +
        ",\"exactFloatCount\":" + std::to_string(exact) +
        ",\"totalFloatCount\":" + std::to_string(4 * kWidth) + ",\"firstMismatch\":" +
        (first_mismatch == 4 * kWidth ? "null" : std::to_string(first_mismatch)) +
        ",\"firstActualBits\":" +
        (first_mismatch == 4 * kWidth ? "null" : json_string(actual_bits_text)) +
        ",\"firstExpectedBits\":" +
        (first_mismatch == 4 * kWidth ? "null" : json_string(expected_bits_text)) +
        ",\"signedZeroMismatches\":" + std::to_string(signed_zero_mismatches) +
        ",\"otherMismatches\":" + std::to_string(other_mismatches) +
        ",\"nonFiniteCount\":" + std::to_string(nonfinite) +
        ",\"maxAbsErr\":" + std::to_string(max_abs_err) +
        ",\"prefixPreserved\":" + (prefix_preserved ? "true" : "false") +
        ",\"staleAddressRejected\":" + (stale_address_rejected ? "true" : "false") +
        ",\"staleReadRejected\":" + (stale_read_rejected ? "true" : "false") +
        ",\"reinjectedValid\":" + (reinjected_valid ? "true" : "false") +
        ",\"resetInvalid\":" + (reset_invalid ? "true" : "false") +
        ",\"resetReadRejected\":" + (reset_read_rejected ? "true" : "false") +
        ",\"packRecords\":" + std::to_string(diagnostic.pack_records) +
        ",\"unpackRecords\":" + std::to_string(diagnostic.unpack_records) +
        ",\"trims\":" + std::to_string(diagnostic.trims) +
        ",\"resets\":" + std::to_string(diagnostic.resets) + ",\"fencesCompleted\":2}";
    return result;
}
