#include "dflash-dml.h"

#include "gguf.h"
#include "xllama/d3d12_dyn.h"
#include "xllama/gpugemv.h"

#include <DirectML.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

using clock_type = std::chrono::steady_clock;
constexpr int kRows = 5120;
constexpr int kColumns = 17408;
constexpr int kWarmup = 50;
constexpr int kIterations = 200;
constexpr int kRepetitions = 3;
constexpr const char* kTensorName = "blk.0.ffn_down.weight";
// Fixed operator gate for this FP16 conversion, against the original Q4_K
// CPU reference. Conversion alone measured maxAbs=0.001395 and NMSE=8.8e-8
// on the frozen tensor/input. These limits are set before selecting a kernel.
constexpr double kMaxAbsError = 1e-2;
constexpr double kMaxNmse = 1e-5;
constexpr double kMinCosine = 0.9999;

double milliseconds(clock_type::time_point first, clock_type::time_point last) {
    return std::chrono::duration<double, std::milli>(last - first).count();
}

std::string json_string(const std::string& value) {
    std::string escaped = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            escaped += '\\';
            escaped += static_cast<char>(c);
        } else if (c == '\n') {
            escaped += "\\n";
        } else if (c == '\r') {
            escaped += "\\r";
        } else if (c == '\t') {
            escaped += "\\t";
        } else if (c >= 0x20) {
            escaped += static_cast<char>(c);
        }
    }
    return escaped + '"';
}

std::string hresult_string(HRESULT hr) {
    std::ostringstream out;
    out << "0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0')
        << static_cast<unsigned>(hr);
    return out.str();
}

void check(HRESULT hr, const char* stage) {
    if (FAILED(hr))
        throw std::runtime_error(std::string(stage) + ": " + hresult_string(hr));
}

struct module_handle {
    HMODULE handle = nullptr;
    ~module_handle() {
        if (handle)
            FreeLibrary(handle);
    }
};

struct event_handle {
    HANDLE handle = nullptr;
    ~event_handle() {
        if (handle)
            CloseHandle(handle);
    }
};

std::string utf8(const wchar_t* value) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1)
        return {};
    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, &result[0], size, nullptr, nullptr);
    result.pop_back();
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>((values.size() - 1) * fraction)];
}

struct accuracy {
    double max_abs = 0;
    double max_relative = 0;
    double nmse = 0;
    double cosine = 0;
    bool finite = true;
};

accuracy compare(const std::vector<float>& reference, const std::vector<float>& output) {
    accuracy result;
    double squared_error = 0;
    double squared_reference = 0;
    double squared_output = 0;
    double dot = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double a = reference[i];
        const double b = output[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            result.finite = false;
            continue;
        }
        const double delta = a - b;
        result.max_abs = std::max(result.max_abs, std::abs(delta));
        result.max_relative =
            std::max(result.max_relative, std::abs(delta) / std::max(1e-6, std::abs(a)));
        squared_error += delta * delta;
        squared_reference += a * a;
        squared_output += b * b;
        dot += a * b;
    }
    result.nmse = squared_error / std::max(1e-30, squared_reference);
    result.cosine = dot / std::sqrt(std::max(1e-30, squared_reference * squared_output));
    return result;
}

bool passes_accuracy(const accuracy& value) {
    return value.finite && value.max_abs <= kMaxAbsError && value.nmse <= kMaxNmse &&
           value.cosine >= kMinCosine;
}

void retain_worst(accuracy& aggregate, const accuracy& current) {
    aggregate.max_abs = std::max(aggregate.max_abs, current.max_abs);
    aggregate.max_relative = std::max(aggregate.max_relative, current.max_relative);
    aggregate.nmse = std::max(aggregate.nmse, current.nmse);
    aggregate.cosine = std::min(aggregate.cosine, current.cosine);
    aggregate.finite &= current.finite;
}

std::string accuracy_json(const accuracy& value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(10) << "{\"finite\":" << (value.finite ? "true" : "false")
        << ",\"maxAbs\":" << value.max_abs << ",\"maxRelative\":" << value.max_relative
        << ",\"nmse\":" << value.nmse << ",\"cosine\":" << value.cosine << '}';
    return out.str();
}

ComPtr<ID3D12Resource> make_buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heap_type,
                                   D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = heap_type;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::max<UINT64>(bytes, 4);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                          IID_PPV_ARGS(&resource)),
          "CreateCommittedResource");
    return resource;
}

void upload(ID3D12Resource* resource, const void* data, size_t bytes) {
    void* mapped = nullptr;
    check(resource->Map(0, nullptr, &mapped), "Map upload");
    std::memcpy(mapped, data, bytes);
    resource->Unmap(0, nullptr);
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
}

void uav_barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource;
    list->ResourceBarrier(1, &barrier);
}

UINT64 local_usage(IDXGIAdapter1* adapter) {
    ComPtr<IDXGIAdapter3> adapter3;
    if (FAILED(adapter->QueryInterface(IID_PPV_ARGS(&adapter3))))
        return 0;
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        return 0;
    return info.CurrentUsage;
}

struct gguf_weights {
    std::vector<xllama::GpugemvQ4KBlock> packed;
    size_t offset = 0;
};

gguf_weights read_weights(const std::string& model_path) {
    gguf_init_params params{};
    params.no_alloc = true;
    using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    gguf_ptr metadata(gguf_init_from_file(model_path.c_str(), params), &gguf_free);
    if (!metadata)
        throw std::runtime_error("GGUF metadata could not be read");
    const int64_t tensor = gguf_find_tensor(metadata.get(), kTensorName);
    if (tensor < 0 || gguf_get_tensor_type(metadata.get(), tensor) != GGML_TYPE_Q4_K)
        throw std::runtime_error("GGUF Q4_K tensor missing");
    const int64_t* ne = gguf_get_tensor_ne(metadata.get(), tensor);
    const size_t packed_bytes =
        static_cast<size_t>(kRows) * kColumns / xllama::kGpugemvQK * xllama::kGpugemvBlockBytes;
    if (ne[0] != kColumns || ne[1] != kRows || ne[2] != 1 ||
        gguf_get_tensor_size(metadata.get(), tensor) != packed_bytes)
        throw std::runtime_error("GGUF Q4_K tensor shape or size differs from benchmark");
    gguf_weights result;
    result.offset =
        gguf_get_data_offset(metadata.get()) + gguf_get_tensor_offset(metadata.get(), tensor);
    result.packed.resize(packed_bytes / sizeof(xllama::GpugemvQ4KBlock));
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file ||
        static_cast<uint64_t>(file.tellg()) < static_cast<uint64_t>(result.offset) + packed_bytes)
        throw std::runtime_error("GGUF Q4_K tensor exceeds file bounds");
    file.seekg(static_cast<std::streamoff>(result.offset));
    file.read(reinterpret_cast<char*>(result.packed.data()),
              static_cast<std::streamsize>(packed_bytes));
    if (!file)
        throw std::runtime_error("GGUF Q4_K tensor read failed");
    return result;
}

struct run_result {
    std::vector<double> boundary_ms;
    std::vector<double> gpu_ms;
    accuracy original;
    accuracy converted;
    bool finite = true;
};

} // namespace

dflash_dml_result run_dflash_dml_real_q4k(const std::string& model_path) {
    dflash_dml_result result;
    try {
        const auto begin = clock_type::now();
        const gguf_weights gguf = read_weights(model_path);
        const auto loaded = clock_type::now();
        const size_t elements = static_cast<size_t>(kRows) * kColumns;
        const UINT64 weight_bytes = static_cast<UINT64>(elements * sizeof(uint16_t));
        const UINT64 input_bytes = static_cast<UINT64>(kColumns * sizeof(uint16_t));
        const UINT64 output_bytes = static_cast<UINT64>(kRows * sizeof(uint16_t));

        std::vector<uint16_t> weight_half(elements);
        std::vector<float> input_float(kColumns);
        std::vector<uint16_t> input_half(kColumns);
        std::vector<float> expected_q4(kRows);
        std::vector<float> expected_half(kRows);
        xllama::gpugemv_fill_x(input_float.data(), kColumns);
        xllama::gpugemv_cpu_ref(gguf.packed.data(), input_float.data(), expected_q4.data(), kRows,
                                kColumns);
        for (int i = 0; i < kColumns; ++i)
            input_half[i] = xllama::gpugemv_float_to_half(input_float[i]);
        float block[256];
        for (size_t b = 0; b < gguf.packed.size(); ++b) {
            xllama::gpugemv_dequant_block(gguf.packed[b], block);
            for (int j = 0; j < xllama::kGpugemvQK; ++j) {
                if (!std::isfinite(block[j]))
                    throw std::runtime_error("GGUF dequant produced nonfinite weight");
                weight_half[b * xllama::kGpugemvQK + j] = xllama::gpugemv_float_to_half(block[j]);
            }
        }
        for (int row = 0; row < kRows; ++row) {
            double sum = 0;
            const size_t base = static_cast<size_t>(row) * kColumns;
            for (int column = 0; column < kColumns; ++column)
                sum +=
                    static_cast<double>(xllama::gpugemv_half_to_float(weight_half[base + column])) *
                    xllama::gpugemv_half_to_float(input_half[column]);
            expected_half[row] = static_cast<float>(sum);
        }
        const accuracy conversion_error = compare(expected_q4, expected_half);
        if (!conversion_error.finite)
            throw std::runtime_error("FP16 conversion produced nonfinite reference");
        const auto converted = clock_type::now();

        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        auto create_device = xllama::d3d12_dyn::CreateDevice();
        if (!create_device)
            throw std::runtime_error("D3D12CreateDevice unavailable");
        ComPtr<IDXGIAdapter1> adapter;
        ComPtr<ID3D12Device> device;
        DXGI_ADAPTER_DESC1 adapter_desc{};
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            check(adapter->GetDesc1(&adapter_desc), "GetDesc1");
            if (!(adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(
                    create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
                break;
            adapter.Reset();
        }
        if (!device)
            throw std::runtime_error("no hardware D3D12 adapter");
        const UINT64 gpu_usage_before = local_usage(adapter.Get());

        module_handle dml_module;
        dml_module.handle = LoadPackagedLibrary(L"DirectML.dll", 0);
        if (!dml_module.handle)
            throw std::runtime_error("DirectML.dll is not available in the app package");
        using create_dml_device = HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS,
                                                   DML_FEATURE_LEVEL, REFIID, void**);
        auto dml_create = reinterpret_cast<create_dml_device>(
            GetProcAddress(dml_module.handle, "DMLCreateDevice1"));
        if (!dml_create)
            throw std::runtime_error("DMLCreateDevice1 unavailable");
        ComPtr<IDMLDevice> dml;
        check(dml_create(device.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_2_0,
                         IID_PPV_ARGS(&dml)),
              "DMLCreateDevice1");

        const UINT shape_a[4] = {1, 1, 1, static_cast<UINT>(kColumns)};
        const UINT shape_b[4] = {1, 1, static_cast<UINT>(kRows), static_cast<UINT>(kColumns)};
        const UINT shape_y[4] = {1, 1, 1, static_cast<UINT>(kRows)};
        const DML_BUFFER_TENSOR_DESC desc_a{DML_TENSOR_DATA_TYPE_FLOAT16,
                                            DML_TENSOR_FLAG_NONE,
                                            4,
                                            shape_a,
                                            nullptr,
                                            input_bytes,
                                            0};
        const DML_BUFFER_TENSOR_DESC desc_b{DML_TENSOR_DATA_TYPE_FLOAT16,
                                            DML_TENSOR_FLAG_NONE,
                                            4,
                                            shape_b,
                                            nullptr,
                                            weight_bytes,
                                            0};
        const DML_BUFFER_TENSOR_DESC desc_y{DML_TENSOR_DATA_TYPE_FLOAT16,
                                            DML_TENSOR_FLAG_NONE,
                                            4,
                                            shape_y,
                                            nullptr,
                                            output_bytes,
                                            0};
        const DML_TENSOR_DESC tensor_a{DML_TENSOR_TYPE_BUFFER, &desc_a};
        const DML_TENSOR_DESC tensor_b{DML_TENSOR_TYPE_BUFFER, &desc_b};
        const DML_TENSOR_DESC tensor_y{DML_TENSOR_TYPE_BUFFER, &desc_y};
        const DML_GEMM_OPERATOR_DESC gemm{&tensor_a,
                                          &tensor_b,
                                          nullptr,
                                          &tensor_y,
                                          DML_MATRIX_TRANSFORM_NONE,
                                          DML_MATRIX_TRANSFORM_TRANSPOSE,
                                          1.0f,
                                          0.0f,
                                          nullptr};
        const DML_OPERATOR_DESC op_desc{DML_OPERATOR_GEMM, &gemm};
        ComPtr<IDMLOperator> op;
        check(dml->CreateOperator(&op_desc, IID_PPV_ARGS(&op)), "DirectML CreateOperator GEMM");
        ComPtr<IDMLCompiledOperator> compiled;
        check(dml->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&compiled)),
              "DirectML CompileOperator GEMM");
        IDMLCompiledOperator* const compiled_raw = compiled.Get();
        ComPtr<IDMLOperatorInitializer> initializer;
        check(dml->CreateOperatorInitializer(1, &compiled_raw, IID_PPV_ARGS(&initializer)),
              "DirectML CreateOperatorInitializer");
        ComPtr<IDMLCommandRecorder> recorder;
        check(dml->CreateCommandRecorder(IID_PPV_ARGS(&recorder)),
              "DirectML CreateCommandRecorder");
        const DML_BINDING_PROPERTIES init_props = initializer->GetBindingProperties();
        const DML_BINDING_PROPERTIES op_props = compiled->GetBindingProperties();

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue;
        check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        ComPtr<ID3D12CommandAllocator> allocator;
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&allocator)),
              "CreateCommandAllocator");
        ComPtr<ID3D12GraphicsCommandList> list;
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&list)),
              "CreateCommandList");
        check(list->Close(), "initial command list Close");
        ComPtr<ID3D12Fence> fence;
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        event_handle event;
        event.handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event.handle)
            throw std::runtime_error("CreateEventW failed");
        UINT64 fence_value = 0;
        auto record = [&]() {
            check(allocator->Reset(), "command allocator Reset");
            check(list->Reset(allocator.Get(), nullptr), "command list Reset");
        };
        auto submit = [&]() {
            check(list->Close(), "command list Close");
            ID3D12CommandList* commands[] = {list.Get()};
            queue->ExecuteCommandLists(1, commands);
            check(queue->Signal(fence.Get(), ++fence_value), "command queue Signal");
            if (fence->GetCompletedValue() < fence_value) {
                check(fence->SetEventOnCompletion(fence_value, event.handle),
                      "SetEventOnCompletion");
                if (WaitForSingleObject(event.handle, 30000) != WAIT_OBJECT_0)
                    throw std::runtime_error("DirectML GPU fence timed out after 30 seconds");
            }
        };

        constexpr D3D12_RESOURCE_FLAGS uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        auto weight_gpu = make_buffer(device.Get(), weight_bytes, D3D12_HEAP_TYPE_DEFAULT, uav,
                                      D3D12_RESOURCE_STATE_COPY_DEST);
        auto input_gpu = make_buffer(device.Get(), input_bytes, D3D12_HEAP_TYPE_DEFAULT, uav,
                                     D3D12_RESOURCE_STATE_COPY_DEST);
        auto output_gpu = make_buffer(device.Get(), output_bytes, D3D12_HEAP_TYPE_DEFAULT, uav,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto weight_upload =
            make_buffer(device.Get(), weight_bytes, D3D12_HEAP_TYPE_UPLOAD,
                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto input_upload =
            make_buffer(device.Get(), input_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_GENERIC_READ);
        auto output_readback =
            make_buffer(device.Get(), output_bytes, D3D12_HEAP_TYPE_READBACK,
                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        const UINT64 temporary_bytes =
            std::max(init_props.TemporaryResourceSize, op_props.TemporaryResourceSize);
        const UINT64 persistent_bytes = op_props.PersistentResourceSize;
        ComPtr<ID3D12Resource> temporary;
        ComPtr<ID3D12Resource> persistent;
        if (temporary_bytes)
            temporary = make_buffer(device.Get(), temporary_bytes, D3D12_HEAP_TYPE_DEFAULT, uav,
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (persistent_bytes)
            persistent = make_buffer(device.Get(), persistent_bytes, D3D12_HEAP_TYPE_DEFAULT, uav,
                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        heap_desc.NumDescriptors =
            std::max({1u, init_props.RequiredDescriptorCount, op_props.RequiredDescriptorCount});
        ComPtr<ID3D12DescriptorHeap> heap;
        check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)),
              "CreateDescriptorHeap");
        const DML_BINDING_TABLE_DESC init_table_desc{
            initializer.Get(), heap->GetCPUDescriptorHandleForHeapStart(),
            heap->GetGPUDescriptorHandleForHeapStart(), heap_desc.NumDescriptors};
        ComPtr<IDMLBindingTable> binding_table;
        check(dml->CreateBindingTable(&init_table_desc, IID_PPV_ARGS(&binding_table)),
              "DirectML CreateBindingTable");
        const DML_BUFFER_BINDING persistent_buffer{persistent.Get(), 0, persistent_bytes};
        const DML_BINDING_DESC persistent_binding{DML_BINDING_TYPE_BUFFER, &persistent_buffer};
        const DML_BUFFER_BINDING temporary_buffer{temporary.Get(), 0, temporary_bytes};
        const DML_BINDING_DESC temporary_binding{DML_BINDING_TYPE_BUFFER, &temporary_buffer};
        binding_table->BindInputs(0, nullptr);
        if (persistent_bytes)
            binding_table->BindOutputs(1, &persistent_binding);
        else
            binding_table->BindOutputs(0, nullptr);
        if (temporary_bytes)
            binding_table->BindTemporaryResource(&temporary_binding);
        const auto setup_done = clock_type::now();

        upload(weight_upload.Get(), weight_half.data(), static_cast<size_t>(weight_bytes));
        upload(input_upload.Get(), input_half.data(), static_cast<size_t>(input_bytes));
        record();
        list->CopyBufferRegion(weight_gpu.Get(), 0, weight_upload.Get(), 0, weight_bytes);
        list->CopyBufferRegion(input_gpu.Get(), 0, input_upload.Get(), 0, input_bytes);
        transition(list.Get(), weight_gpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(list.Get(), input_gpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        submit();
        weight_upload.Reset();
        input_upload.Reset();

        record();
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        recorder->RecordDispatch(list.Get(), initializer.Get(), binding_table.Get());
        if (persistent_bytes)
            uav_barrier(list.Get(), persistent.Get());
        submit();

        const DML_BINDING_TABLE_DESC op_table_desc{
            compiled.Get(), heap->GetCPUDescriptorHandleForHeapStart(),
            heap->GetGPUDescriptorHandleForHeapStart(), heap_desc.NumDescriptors};
        check(binding_table->Reset(&op_table_desc), "DirectML binding table Reset");
        const DML_BUFFER_BINDING input_buffers[2] = {{input_gpu.Get(), 0, input_bytes},
                                                     {weight_gpu.Get(), 0, weight_bytes}};
        // DirectML requires an input binding for every operator input, even an
        // optional null CTensor. Omitting the third GEMM slot made dispatch a
        // no-op on the Xbox (all-zero output, ~40 ns timestamp).
        const DML_BINDING_DESC input_bindings[3] = {{DML_BINDING_TYPE_BUFFER, &input_buffers[0]},
                                                    {DML_BINDING_TYPE_BUFFER, &input_buffers[1]},
                                                    {DML_BINDING_TYPE_NONE, nullptr}};
        const DML_BUFFER_BINDING output_buffer{output_gpu.Get(), 0, output_bytes};
        const DML_BINDING_DESC output_binding{DML_BINDING_TYPE_BUFFER, &output_buffer};
        binding_table->BindInputs(3, input_bindings);
        binding_table->BindOutputs(1, &output_binding);
        if (temporary_bytes)
            binding_table->BindTemporaryResource(&temporary_binding);
        if (persistent_bytes)
            binding_table->BindPersistentResource(&persistent_binding);
        const auto ready = clock_type::now();
        const UINT64 gpu_usage_ready = local_usage(adapter.Get());

        UINT64 timestamp_frequency = 0;
        check(queue->GetTimestampFrequency(&timestamp_frequency), "GetTimestampFrequency");
        D3D12_QUERY_HEAP_DESC query_desc{};
        query_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        query_desc.Count = 2;
        ComPtr<ID3D12QueryHeap> timestamp_query;
        check(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&timestamp_query)),
              "CreateQueryHeap");
        auto timestamp_readback =
            make_buffer(device.Get(), 2 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK,
                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        D3D12_RESOURCE_STATES output_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        std::vector<float> output_float(kRows);
        std::vector<run_result> runs;
        runs.reserve(kRepetitions);
        auto dispatch = [&](double* boundary_ms, double* kernel_ms) {
            const auto started = clock_type::now();
            record();
            list->SetDescriptorHeaps(1, heaps);
            if (output_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                transition(list.Get(), output_gpu.Get(), output_state,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                output_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
            list->EndQuery(timestamp_query.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            recorder->RecordDispatch(list.Get(), compiled.Get(), binding_table.Get());
            list->EndQuery(timestamp_query.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            uav_barrier(list.Get(), output_gpu.Get());
            transition(list.Get(), output_gpu.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            output_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
            list->CopyBufferRegion(output_readback.Get(), 0, output_gpu.Get(), 0, output_bytes);
            list->ResolveQueryData(timestamp_query.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                                   timestamp_readback.Get(), 0);
            submit();
            const D3D12_RANGE output_range{0, static_cast<SIZE_T>(output_bytes)};
            void* raw = nullptr;
            check(output_readback->Map(0, &output_range, &raw), "output readback Map");
            const uint16_t* bits = static_cast<const uint16_t*>(raw);
            for (int i = 0; i < kRows; ++i)
                output_float[i] = xllama::gpugemv_half_to_float(bits[i]);
            const D3D12_RANGE empty_range{0, 0};
            output_readback->Unmap(0, &empty_range);
            check(timestamp_readback->Map(0, nullptr, &raw), "timestamp readback Map");
            UINT64 ticks[2];
            std::memcpy(ticks, raw, sizeof(ticks));
            timestamp_readback->Unmap(0, &empty_range);
            const auto finished = clock_type::now();
            if (boundary_ms)
                *boundary_ms = milliseconds(started, finished);
            if (kernel_ms)
                *kernel_ms =
                    timestamp_frequency && ticks[1] >= ticks[0]
                        ? 1000.0 * static_cast<double>(ticks[1] - ticks[0]) / timestamp_frequency
                        : 0;
        };

        for (int repetition = 0; repetition < kRepetitions; ++repetition) {
            run_result run;
            run.boundary_ms.reserve(kIterations);
            run.gpu_ms.reserve(kIterations);
            for (int i = 0; i < kWarmup; ++i)
                dispatch(nullptr, nullptr);
            for (int i = 0; i < kIterations; ++i) {
                double boundary_ms = 0;
                double kernel_ms = 0;
                dispatch(&boundary_ms, &kernel_ms);
                run.boundary_ms.push_back(boundary_ms);
                run.gpu_ms.push_back(kernel_ms);
                run.finite &= std::all_of(output_float.begin(), output_float.end(),
                                          [](float value) { return std::isfinite(value); });
                const accuracy original = compare(expected_q4, output_float);
                const accuracy converted_accuracy = compare(expected_half, output_float);
                if (i == 0) {
                    run.original = original;
                    run.converted = converted_accuracy;
                } else {
                    retain_worst(run.original, original);
                    retain_worst(run.converted, converted_accuracy);
                }
            }
            run.finite &= run.original.finite && run.converted.finite;
            runs.push_back(std::move(run));
        }
        check(device->GetDeviceRemovedReason(), "D3D12 device removed");

        std::ostringstream json;
        json.imbue(std::locale::classic());
        json << std::setprecision(10)
             << "{\"probe\":\"directml_q4_k_to_fp16_gemv\",\"tensor\":" << json_string(kTensorName)
             << ",\"sourceType\":\"Q4_K\",\"computeType\":\"FP16\",\"accumulation\":\"DirectML "
                "default\""
             << ",\"adapter\":" << json_string(utf8(adapter_desc.Description))
             << ",\"software\":false,\"rows\":" << kRows << ",\"columns\":" << kColumns
             << ",\"packedBytes\":" << gguf.packed.size() * sizeof(xllama::GpugemvQ4KBlock)
             << ",\"fp16WeightBytes\":" << weight_bytes << ",\"ggufOffset\":" << gguf.offset
             << ",\"warmup\":" << kWarmup << ",\"iterations\":" << kIterations
             << ",\"repetitions\":" << kRepetitions
             << ",\"accuracyGate\":{\"maxAbs\":" << kMaxAbsError << ",\"maxNmse\":" << kMaxNmse
             << ",\"minCosine\":" << kMinCosine << ",\"reference\":\"original Q4_K CPU\"}"
             << ",\"fp16ConversionVsQ4\":" << accuracy_json(conversion_error)
             << ",\"cold\":{\"ggufReadMs\":" << milliseconds(begin, loaded)
             << ",\"dequantAndCpuReferenceMs\":" << milliseconds(loaded, converted)
             << ",\"dmlSetupAndCompileMs\":" << milliseconds(converted, setup_done)
             << ",\"uploadAndInitMs\":" << milliseconds(setup_done, ready)
             << ",\"readyMs\":" << milliseconds(begin, ready) << '}'
             << ",\"memory\":{\"gpuWeightBytes\":" << weight_bytes
             << ",\"gpuInputBytes\":" << input_bytes << ",\"gpuOutputBytes\":" << output_bytes
             << ",\"temporaryBytes\":" << temporary_bytes
             << ",\"persistentBytes\":" << persistent_bytes
             << ",\"localUsageBeforeBytes\":" << gpu_usage_before
             << ",\"localUsageReadyBytes\":" << gpu_usage_ready
             << ",\"cpuPackedBytes\":" << gguf.packed.size() * sizeof(xllama::GpugemvQ4KBlock)
             << ",\"cpuConvertedWeightBytes\":" << weight_bytes << '}' << ",\"runs\":[";
        bool passed = true;
        for (size_t i = 0; i < runs.size(); ++i) {
            const run_result& run = runs[i];
            if (i)
                json << ',';
            const bool accuracy_passed = passes_accuracy(run.original);
            passed &= run.finite && accuracy_passed && run.boundary_ms.size() == kIterations &&
                      run.gpu_ms.size() == kIterations;
            json << "{\"run\":" << (i + 1) << ",\"finite\":" << (run.finite ? "true" : "false")
                 << ",\"accuracyPassed\":" << (accuracy_passed ? "true" : "false")
                 << ",\"boundaryMedianMs\":" << percentile(run.boundary_ms, 0.5)
                 << ",\"boundaryP95Ms\":" << percentile(run.boundary_ms, 0.95)
                 << ",\"gpuMedianMs\":" << percentile(run.gpu_ms, 0.5)
                 << ",\"gpuP95Ms\":" << percentile(run.gpu_ms, 0.95)
                 << ",\"vsQ4\":" << accuracy_json(run.original)
                 << ",\"vsConvertedFP16\":" << accuracy_json(run.converted) << '}';
        }
        json << "],\"passed\":" << (passed ? "true" : "false") << '}';
        result.passed = passed;
        result.json = json.str();
    } catch (const std::exception& error) {
        result.json = "{\"probe\":\"directml_q4_k_to_fp16_gemv\",\"passed\":false,\"error\":" +
                      json_string(error.what()) + "}";
    }
    return result;
}
