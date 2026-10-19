#pragma once

#include "rkva-kvarn-range.h"
#include <cmath>
#include <cstring>
#include <vector>

// Pure, bounded admission. No live metadata or GPU resource is changed until
// every segment has passed descriptor, coverage, size and finite-query checks.
namespace rkva_kvarn_bulk {
struct Segment {
    rkva_kvarn_range_upload upload{};
    uint32_t n_queries = 0;
    size_t data_offset = 0, data_bytes = 0, positions_offset = 0, query_offset = 0;
};
inline bool validate(const uint8_t* data, size_t bytes, uint32_t n_queries,
                     const rkva_kvarn_range_config& descriptor, uint32_t n_head, uint32_t n_head_kv,
                     uint32_t live, uint32_t groups, bool ready,
                     rkva_kvarn_range::prefix_coverage coverage, uint64_t generation,
                     std::vector<Segment>& segments) {
    segments.clear();
    if (!data || bytes < sizeof(rkva_kvarn_range_bulk) || bytes > (256u << 20) ||
        !rkva_kvarn_range::valid_descriptor(descriptor, 256) ||
        !rkva_kvarn_range::serial_prefix(descriptor) || !n_head || !n_head_kv || n_head > 256 ||
        n_head_kv > 256 || !n_queries || n_queries > 256)
        return false;
    rkva_kvarn_range_bulk header{};
    std::memcpy(&header, data, sizeof(header));
    if (header.version != 1 || !header.segment_count || header.segment_count > 32 ||
        header.total_queries != n_queries || header.reserved ||
        generation > UINT64_MAX - header.segment_count)
        return false;
    size_t offset = sizeof(header);
    uint32_t total = 0;
    for (uint32_t i = 0; i < header.segment_count; ++i) {
        if (bytes - offset < sizeof(rkva_kvarn_range_bulk_segment))
            return false;
        rkva_kvarn_range_bulk_segment wire{};
        std::memcpy(&wire, data + offset, sizeof(wire));
        offset += sizeof(wire);
        if (!wire.n_queries || wire.n_queries > 9 || wire.n_queries > n_queries - total ||
            wire.upload_bytes < sizeof(rkva_kvarn_range_upload) ||
            wire.upload_bytes > bytes - offset)
            return false;
        Segment part;
        part.n_queries = wire.n_queries;
        std::memcpy(&part.upload, data + offset, sizeof(part.upload));
        const auto& u = part.upload;
        if (!rkva_kvarn_range::valid_upload(u, descriptor.capacity, live, groups, ready))
            return false;
        const bool reset = (u.flags & 1) != 0;
        if ((reset && i) || (!reset && u.append_rows != wire.n_queries))
            return false;
        const uint64_t row_bytes = uint64_t(n_head_kv) * 256 * 2;
        const uint64_t records = uint64_t(u.record_groups) * n_head_kv * 2 * 8960;
        const uint64_t stage = uint64_t(reset ? 384 : u.append_rows) * row_bytes;
        const uint64_t tail_rows =
            descriptor.precision_tail_tokens
                ? (reset ? std::min(u.live_count, descriptor.precision_tail_tokens) : u.append_rows)
                : 0;
        const uint64_t expected = 2 * (records + stage + tail_rows * row_bytes);
        if (uint64_t(wire.upload_bytes) != sizeof(u) + expected)
            return false;
        part.data_offset = offset + sizeof(u);
        part.data_bytes = size_t(expected);
        offset += wire.upload_bytes;
        const uint64_t query_bytes = uint64_t(wire.n_queries) * n_head * 256 * sizeof(float);
        const uint64_t attention_bytes =
            sizeof(rkva_range_attn) + uint64_t(wire.n_queries) * 4 + query_bytes;
        if (attention_bytes > bytes - offset)
            return false;
        rkva_range_attn attention{};
        std::memcpy(&attention, data + offset, sizeof(attention));
        if (attention.wire != RKVA_WIRE_F32 || attention.flags || attention.reserved[0] ||
            attention.reserved[1])
            return false;
        part.positions_offset = offset + sizeof(attention);
        part.query_offset = part.positions_offset + wire.n_queries * 4;
        int32_t positions[9];
        std::memcpy(positions, data + part.positions_offset, wire.n_queries * 4);
        for (uint32_t j = 0; j < wire.n_queries; ++j)
            if (positions[j] < 0)
                return false;
        coverage = reset ? rkva_kvarn_range::coverage_after_reset(u.live_count,
                                                                  descriptor.precision_tail_tokens)
                         : rkva_kvarn_range::coverage_after_append(
                               coverage, u.live_count, descriptor.precision_tail_tokens);
        if (!rkva_kvarn_range::valid_prefix_queries(positions, wire.n_queries, descriptor.origin,
                                                    u.live_count, coverage,
                                                    descriptor.precision_tail_tokens))
            return false;
        for (size_t j = 0; j < query_bytes; j += sizeof(float)) {
            float value;
            std::memcpy(&value, data + part.query_offset + j, sizeof(value));
            if (!std::isfinite(value))
                return false;
        }
        offset += size_t(attention_bytes);
        groups = reset ? u.record_groups : std::max(groups, u.record_begin + u.record_groups);
        live = u.live_count;
        ready = true;
        total += wire.n_queries;
        segments.push_back(part);
    }
    return offset == bytes && total == n_queries;
}
} // namespace rkva_kvarn_bulk
