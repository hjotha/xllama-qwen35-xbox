// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Included in remote-attn-engine.cpp after Session is complete.
// Native KVarN4 bulk FP16 Q/OUT staging, GPU conversion, and readback helpers.

#if __has_include("rkva_range_bulk_unpack_q_f16_dxil.h") && \
                  __has_include("rkva_range_bulk_pack_out_f16_dxil.h")
    #include "rkva_range_bulk_pack_out_f16_dxil.h"
    #include "rkva_range_bulk_unpack_q_f16_dxil.h"
    #define RKVA_HAS_BULK_F16_DXIL 1
#else
    #define RKVA_HAS_BULK_F16_DXIL 0
#endif

void RemoteAttnEngine::create_bulk_f16_pipelines() {
#if RKVA_HAS_BULK_F16_DXIL
    if (!device_ || !range_reduce_root_) {
        bulk_f16_status_ = "range reduce root signature unavailable";
        return;
    }
    char text[96];
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
    pso.pRootSignature = range_reduce_root_.Get();
    pso.CS = {g_rkva_range_bulk_unpack_q_f16_dxil, sizeof(g_rkva_range_bulk_unpack_q_f16_dxil)};
    HRESULT hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&bulk_unpack_q_f16_pso_));
    if (SUCCEEDED(hr)) {
        pso.CS = {g_rkva_range_bulk_pack_out_f16_dxil, sizeof(g_rkva_range_bulk_pack_out_f16_dxil)};
        hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&bulk_pack_out_f16_pso_));
    }
    if (FAILED(hr)) {
        std::snprintf(text, sizeof(text), "bulk f16 pipeline state: 0x%08X", unsigned(hr));
        bulk_unpack_q_f16_pso_.Reset();
        bulk_pack_out_f16_pso_.Reset();
        bulk_f16_status_ = text;
        return;
    }
    bulk_f16_status_ = "ready";
#else
    bulk_f16_status_ = "deferred_dxil_host_fallback";
#endif
}

bool RemoteAttnEngine::attend_kvarn_bulk_f16(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                             const uint8_t* data, size_t bytes, uint16_t* out_f16,
                                             float* lse, std::string* error) {
    Session* s = find_session(id);
    if (!ready_ || !s || !s->valid || !s->range_configured || s->range_storage != 3 ||
        layer >= s->layer_kv.size() || !out_f16 || !lse || !range_attn_kvarn4_prefix_pso_ ||
        bulk_recording_)
        return ::fail(error, "native bulk f16: invalid session/arguments");
    std::vector<rkva_kvarn_bulk::Segment> segments;
    if (!rkva_kvarn_bulk::validate_wire(
            data, bytes, n_tokens, s->range_descriptor, geo_.n_head, geo_.n_head_kv,
            s->range_counts[layer], s->range_record_groups[layer], s->range_ready[layer],
            s->prefix_coverage[layer], s->native_generation, RKVA_WIRE_F16, segments))
        return ::fail(error, "native bulk f16: invalid envelope/coverage/query");
    if (bulk_f16_status_ == "not initialized") {
        create_bulk_f16_pipelines();
    }
    const bool gpu_convert = bulk_f16_gpu_available();
    // Admission and all grow-only allocation precede the first native mutation.
    uint32_t max_queries = 0;
    for (const auto& segment : segments)
        max_queries = std::max(max_queries, segment.n_queries);
    if (!ensure_range_scratch(max_queries, error))
        return false;
    if (gpu_convert) {
        const uint64_t stage_out_bytes = uint64_t(max_queries) * geo_.n_head * (256 * 4 + 4);
        if (bulk_f16_stage_out_.bytes < stage_out_bytes &&
            !create_buffer(stage_out_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COPY_DEST, &bulk_f16_stage_out_, error))
            return false;
    }
    if (bulk_transfers_.size() < segments.size())
        bulk_transfers_.resize(segments.size());
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        auto& transfer = bulk_transfers_[i];
        const uint64_t q_heads = uint64_t(segment.n_queries) * geo_.n_head;
        const uint64_t q_f32_bytes = q_heads * 256 * sizeof(float);
        const uint64_t q_f16_bytes = q_heads * 256 * sizeof(uint16_t);
        const uint64_t pos_bytes = uint64_t(segment.n_queries) * sizeof(int32_t);
        const uint64_t lse_bytes = q_heads * sizeof(float);
        if (transfer.upload.bytes < segment.data_bytes &&
            !create_buffer(segment.data_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, &transfer.upload, error))
            return false;
        if (gpu_convert) {
            const uint64_t query_upload_bytes = q_f16_bytes + pos_bytes;
            const uint64_t query_default_bytes = q_f32_bytes + pos_bytes;
            const uint64_t packed_output_bytes = q_f16_bytes + lse_bytes;
            if ((transfer.query_f16_upload.bytes < query_upload_bytes &&
                 !create_buffer(query_upload_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                &transfer.query_f16_upload, error)) ||
                (transfer.query_f32_default.bytes < query_default_bytes &&
                 !create_buffer(query_default_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &transfer.query_f32_default,
                                error)) ||
                (transfer.packed_out_f16.bytes < packed_output_bytes &&
                 !create_buffer(packed_output_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &transfer.packed_out_f16,
                                error)) ||
                (transfer.readback.bytes < packed_output_bytes &&
                 !create_buffer(packed_output_bytes, D3D12_HEAP_TYPE_READBACK,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                &transfer.readback, error)))
                return false;
        } else {
            const uint64_t query_bytes = q_f32_bytes + pos_bytes;
            const uint64_t output_bytes = q_f32_bytes + lse_bytes;
            if ((transfer.query.bytes < query_bytes &&
                 !create_buffer(query_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_GENERIC_READ, &transfer.query, error)) ||
                (transfer.readback.bytes < output_bytes &&
                 !create_buffer(output_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_COPY_DEST, &transfer.readback, error)))
                return false;
        }
    }
    std::vector<float> fallback_q_f32;
    if (!gpu_convert) {
        fallback_q_f32.resize(size_t(max_queries) * geo_.n_head * 256);
    }
    if (!begin(error))
        return false;
    bulk_recording_ = true;
    float dummy_f32 = 0.0f;
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        auto& transfer = bulk_transfers_[i];
        const uint64_t q_heads = uint64_t(segment.n_queries) * geo_.n_head;
        const uint64_t q_elems = q_heads * 256;
        const uint64_t q_f16_bytes = q_elems * sizeof(uint16_t);
        const uint64_t pos_bytes = uint64_t(segment.n_queries) * sizeof(int32_t);
        const uint64_t lse_bytes = q_heads * sizeof(float);
        bool ok = false;
        if (gpu_convert) {
            void* mapped_q = nullptr;
            const HRESULT map_hr = transfer.query_f16_upload.resource->Map(0, nullptr, &mapped_q);
            if (SUCCEEDED(map_hr) && mapped_q) {
                std::memcpy(mapped_q, data + segment.query_offset, q_f16_bytes);
                std::memcpy(static_cast<uint8_t*>(mapped_q) + q_f16_bytes,
                            data + segment.positions_offset, pos_bytes);
                transfer.query_f16_upload.resource->Unmap(0, nullptr);
                std::swap(range_upload_, transfer.upload);
                ok = upload_kvarn_range(id, layer, segment.upload, data + segment.data_offset,
                                        segment.data_bytes, error);
                std::swap(range_upload_, transfer.upload);
            } else {
                ::fail(error, "Map bulk f16 query upload", map_hr);
            }
            if (ok) {
                const uint32_t q_pairs = uint32_t(q_elems / 2);
                const uint32_t unpack_params[4] = {q_pairs, segment.n_queries, 0u, 0u};
                transition(transfer.query_f32_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                cmd_->SetComputeRootSignature(range_reduce_root_.Get());
                cmd_->SetPipelineState(bulk_unpack_q_f16_pso_.Get());
                cmd_->SetComputeRoot32BitConstants(0, 4, unpack_params, 0);
                cmd_->SetComputeRootShaderResourceView(
                    1, transfer.query_f16_upload.resource->GetGPUVirtualAddress());
                cmd_->SetComputeRootUnorderedAccessView(
                    2, transfer.query_f32_default.resource->GetGPUVirtualAddress());
                const uint32_t unpack_groups = (std::max(q_pairs, segment.n_queries) + 255u) / 256u;
                cmd_->Dispatch(unpack_groups, 1, 1);
                uav_barrier(transfer.query_f32_default);
                transition(transfer.query_f32_default,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(bulk_f16_stage_out_, D3D12_RESOURCE_STATE_COPY_DEST);
                std::swap(range_query_, transfer.query_f32_default);
                std::swap(range_readback_, bulk_f16_stage_out_);
                bulk_gpu_query_ = true;
                ok = attend_range(id, layer, segment.n_queries,
                                  reinterpret_cast<const int32_t*>(data + segment.positions_offset),
                                  &dummy_f32, &dummy_f32, lse, error);
                bulk_gpu_query_ = false;
                std::swap(range_query_, transfer.query_f32_default);
                std::swap(range_readback_, bulk_f16_stage_out_);
            }
            if (ok) {
                const uint32_t q_pairs = uint32_t(q_elems / 2);
                const uint32_t q_heads_u32 = uint32_t(q_heads);
                const uint32_t pack_params[4] = {q_pairs, q_heads_u32, 0u, 0u};
                transition(range_output_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(transfer.packed_out_f16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                cmd_->SetComputeRootSignature(range_reduce_root_.Get());
                cmd_->SetPipelineState(bulk_pack_out_f16_pso_.Get());
                cmd_->SetComputeRoot32BitConstants(0, 4, pack_params, 0);
                cmd_->SetComputeRootShaderResourceView(
                    1, range_output_.resource->GetGPUVirtualAddress());
                cmd_->SetComputeRootUnorderedAccessView(
                    2, transfer.packed_out_f16.resource->GetGPUVirtualAddress());
                const uint32_t pack_groups = (std::max(q_pairs, q_heads_u32) + 255u) / 256u;
                cmd_->Dispatch(pack_groups, 1, 1);
                uav_barrier(transfer.packed_out_f16);
                transition(transfer.packed_out_f16, D3D12_RESOURCE_STATE_COPY_SOURCE);
                cmd_->CopyBufferRegion(transfer.readback.resource.Get(), 0,
                                       transfer.packed_out_f16.resource.Get(), 0,
                                       q_f16_bytes + lse_bytes);
            }
        } else {
            const auto* q_f16 = reinterpret_cast<const uint16_t*>(data + segment.query_offset);
            for (size_t e = 0; e < q_elems; ++e) {
                fallback_q_f32[e] = rkva_kvarn_bulk::f16_bits_to_f32(q_f16[e]);
            }
            std::swap(range_upload_, transfer.upload);
            std::swap(range_query_, transfer.query);
            std::swap(range_readback_, transfer.readback);
            ok = upload_kvarn_range(id, layer, segment.upload, data + segment.data_offset,
                                    segment.data_bytes, error) &&
                 attend_range(id, layer, segment.n_queries,
                              reinterpret_cast<const int32_t*>(data + segment.positions_offset),
                              fallback_q_f32.data(), &dummy_f32, lse, error);
            std::swap(range_upload_, transfer.upload);
            std::swap(range_query_, transfer.query);
            std::swap(range_readback_, transfer.readback);
        }
        if (!ok) {
            bulk_recording_ = false;
            bulk_gpu_query_ = false;
            s->valid = false;
            terminal_failure_ = true;
            return false;
        }
    }
    bulk_recording_ = false;
    if (!submit(error)) {
        s->valid = false;
        return false;
    }
    size_t row = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        const uint64_t heads = uint64_t(segments[i].n_queries) * geo_.n_head;
        const uint64_t q_elems = heads * 256;
        const uint64_t q_f16_bytes = q_elems * sizeof(uint16_t);
        const uint64_t q_f32_bytes = q_elems * sizeof(float);
        const uint64_t lse_bytes = heads * sizeof(float);
        const uint64_t readback_bytes = (gpu_convert ? q_f16_bytes : q_f32_bytes) + lse_bytes;
        void* mapped = nullptr;
        D3D12_RANGE range{0, static_cast<SIZE_T>(readback_bytes)};
        const HRESULT hr = bulk_transfers_[i].readback.resource->Map(0, &range, &mapped);
        if (FAILED(hr) || !mapped) {
            s->valid = false;
            return fail(error, "Map native bulk f16 readback", hr);
        }
        uint16_t* dst_out = out_f16 + row * geo_.n_head * 256;
        float* dst_lse = lse + row * geo_.n_head;
        bool finite_ok = true;
        if (gpu_convert) {
            std::memcpy(dst_out, mapped, q_f16_bytes);
            std::memcpy(dst_lse, static_cast<const uint8_t*>(mapped) + q_f16_bytes, lse_bytes);
            for (uint64_t e = 0; e < q_elems; ++e) {
                if (!rkva_kvarn_bulk::f16_bits_finite(dst_out[e])) {
                    finite_ok = false;
                    break;
                }
            }
        } else {
            const auto* src_out = static_cast<const float*>(mapped);
            for (uint64_t e = 0; e < q_elems; ++e) {
                if (!std::isfinite(src_out[e])) {
                    finite_ok = false;
                    break;
                }
                const uint16_t packed = rkva_kvarn_bulk::f32_to_f16_bits(src_out[e]);
                if (!rkva_kvarn_bulk::f16_bits_finite(packed)) {
                    finite_ok = false;
                    break;
                }
                dst_out[e] = packed;
            }
            std::memcpy(dst_lse, static_cast<const uint8_t*>(mapped) + q_f32_bytes, lse_bytes);
        }
        for (uint64_t h = 0; finite_ok && h < heads; ++h) {
            if (!std::isfinite(dst_lse[h])) {
                finite_ok = false;
            }
        }
        D3D12_RANGE empty{0, 0};
        bulk_transfers_[i].readback.resource->Unmap(0, &empty);
        if (!finite_ok) {
            s->valid = false;
            return ::fail(error, "native bulk f16: non-finite or overflowed attention output");
        }
        row += segments[i].n_queries;
    }
    return true;
}
