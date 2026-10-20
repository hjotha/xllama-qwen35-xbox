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

inline bool f16_bits_finite(uint16_t bits) {
    return (bits & 0x7c00u) != 0x7c00u;
}

inline uint16_t f32_to_f16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return uint16_t(sign | (mantissa ? 0x7e00u : 0x7c00u));
    }
    int32_t half_exp = int32_t(exponent) - 127 + 15;
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return uint16_t(sign);
        }
        mantissa |= 0x800000u;
        const uint32_t shift = uint32_t(14 - half_exp);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return uint16_t(sign | rounded);
    }
    if (half_exp >= 31) {
        return uint16_t(sign | 0x7c00u);
    }
    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
    }
    if (rounded == 0x400u) {
        rounded = 0;
        ++half_exp;
    }
    if (half_exp >= 31) {
        return uint16_t(sign | 0x7c00u);
    }
    return uint16_t(sign | (uint32_t(half_exp) << 10) | rounded);
}

inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t m = mant;
            int32_t e = -14;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03ffu;
            bits = sign | (uint32_t(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline bool peek_wire(const uint8_t* data, size_t bytes, uint32_t* out_wire) {
    if (!data || !out_wire ||
        bytes < sizeof(rkva_kvarn_range_bulk) + sizeof(rkva_kvarn_range_bulk_segment) +
                    sizeof(rkva_kvarn_range_upload) + sizeof(rkva_range_attn)) {
        return false;
    }
    rkva_kvarn_range_bulk header{};
    std::memcpy(&header, data, sizeof(header));
    if (header.version != 1 || !header.segment_count || header.segment_count > 32 ||
        header.reserved) {
        return false;
    }
    rkva_kvarn_range_bulk_segment seg{};
    std::memcpy(&seg, data + sizeof(header), sizeof(seg));
    const size_t upload_offset = sizeof(header) + sizeof(seg);
    if (seg.upload_bytes < sizeof(rkva_kvarn_range_upload) ||
        seg.upload_bytes > bytes - upload_offset ||
        bytes - upload_offset - seg.upload_bytes < sizeof(rkva_range_attn)) {
        return false;
    }
    rkva_range_attn attention{};
    std::memcpy(&attention, data + upload_offset + seg.upload_bytes, sizeof(attention));
    if (attention.wire != RKVA_WIRE_F32 && attention.wire != RKVA_WIRE_F16) {
        return false;
    }
    *out_wire = attention.wire;
    return true;
}

inline bool validate_wire(const uint8_t* data, size_t bytes, uint32_t n_queries,
                          const rkva_kvarn_range_config& descriptor, uint32_t n_head,
                          uint32_t n_head_kv, uint32_t live, uint32_t groups, bool ready,
                          rkva_kvarn_range::prefix_coverage coverage, uint64_t generation,
                          uint32_t expected_wire, std::vector<Segment>& segments) {
    segments.clear();
    if (!data || bytes < sizeof(rkva_kvarn_range_bulk) || bytes > (256u << 20) ||
        (expected_wire != RKVA_WIRE_F32 && expected_wire != RKVA_WIRE_F16) ||
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
    const uint64_t query_elem_bytes =
        expected_wire == RKVA_WIRE_F16 ? sizeof(uint16_t) : sizeof(float);
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
        const uint64_t query_bytes = uint64_t(wire.n_queries) * n_head * 256 * query_elem_bytes;
        const uint64_t attention_bytes =
            sizeof(rkva_range_attn) + uint64_t(wire.n_queries) * 4 + query_bytes;
        if (attention_bytes > bytes - offset)
            return false;
        rkva_range_attn attention{};
        std::memcpy(&attention, data + offset, sizeof(attention));
        if (attention.wire != expected_wire || attention.flags || attention.reserved[0] ||
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
        if (expected_wire == RKVA_WIRE_F16) {
            for (size_t j = 0; j < query_bytes; j += sizeof(uint16_t)) {
                uint16_t value;
                std::memcpy(&value, data + part.query_offset + j, sizeof(value));
                if (!f16_bits_finite(value))
                    return false;
            }
        } else {
            for (size_t j = 0; j < query_bytes; j += sizeof(float)) {
                float value;
                std::memcpy(&value, data + part.query_offset + j, sizeof(value));
                if (!std::isfinite(value))
                    return false;
            }
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

inline bool validate(const uint8_t* data, size_t bytes, uint32_t n_queries,
                     const rkva_kvarn_range_config& descriptor, uint32_t n_head, uint32_t n_head_kv,
                     uint32_t live, uint32_t groups, bool ready,
                     rkva_kvarn_range::prefix_coverage coverage, uint64_t generation,
                     std::vector<Segment>& segments) {
    return validate_wire(data, bytes, n_queries, descriptor, n_head, n_head_kv, live, groups, ready,
                         coverage, generation, RKVA_WIRE_F32, segments);
}
} // namespace rkva_kvarn_bulk
