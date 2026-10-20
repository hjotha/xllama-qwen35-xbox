#include "../uwp/rkva/rkva-kvarn-bulk.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
using namespace rkva_kvarn_bulk;
namespace {
template <class T> void append(std::vector<uint8_t>& data, const T& value) {
    const size_t offset = data.size();
    data.resize(offset + sizeof(T));
    std::memcpy(data.data() + offset, &value, sizeof(T));
}
rkva_kvarn_range_config descriptor(uint32_t tail) {
    return {96256, 34816, RKVA_RANGE_STORAGE_KVARN4,
            3,     128,   128,
            256,   2,     4,
            4,     8960,  3,
            2,     128,   tail,
            1,     1,     1};
}
std::vector<uint8_t> payload(uint32_t begin, uint32_t count, uint32_t tail,
                             uint32_t wire = RKVA_WIRE_F32) {
    std::vector<uint8_t> data;
    const uint32_t segments = (count + 7) / 8;
    const size_t q_elem_bytes = (wire == RKVA_WIRE_F16) ? sizeof(uint16_t) : sizeof(float);
    append(data, rkva_kvarn_range_bulk{1, segments, count, 0});
    for (uint32_t first = 0; first < count; first += 8) {
        const uint32_t n = std::min(8u, count - first);
        const uint32_t previous = begin + first, live = previous + n;
        const bool reset = previous == 0;
        const uint32_t groups = live / 128 - previous / 128;
        const rkva_kvarn_range_upload u{
            live,           previous / 128, groups, reset ? 1u : 0u, reset ? 0u : previous,
            reset ? 0u : n, {0, 0}};
        const size_t records = size_t(groups) * 4 * 2 * 8960;
        const size_t rows = (reset ? 384 : n) + (tail ? (reset ? std::min(live, tail) : n) : 0);
        const size_t bytes = 2 * (records + rows * 4 * 256 * 2);
        append(data, rkva_kvarn_range_bulk_segment{uint32_t(sizeof(u) + bytes), n});
        append(data, u);
        data.resize(data.size() + bytes);
        append(data, rkva_range_attn{wire, 0, {0, 0}});
        for (uint32_t row = 0; row < n; ++row)
            append(data, int32_t(96256 + previous + row));
        data.resize(data.size() + size_t(n) * 24 * 256 * q_elem_bytes);
    }
    return data;
}
bool valid(const std::vector<uint8_t>& data, uint32_t begin, uint32_t count, uint32_t tail,
           uint64_t generation = 1, uint32_t wire = RKVA_WIRE_F32) {
    std::vector<Segment> segments;
    return validate_wire(
        data.data(), data.size(), count, descriptor(tail), 24, 4, begin, begin / 128, begin != 0,
        rkva_kvarn_range::coverage_after_reset(begin, tail), generation, wire, segments);
}
void test_f16_helpers() {
    assert(f32_to_f16_bits(0.0f) == 0x0000u);
    assert(f32_to_f16_bits(-0.0f) == 0x8000u);
    assert(f32_to_f16_bits(1.0f) == 0x3c00u);
    assert(f32_to_f16_bits(-1.0f) == 0xbc00u);
    assert(f16_bits_to_f32(0x3c00u) == 1.0f);
    assert(f16_bits_to_f32(0xbc00u) == -1.0f);
    assert(f16_bits_finite(0x3c00u));
    assert(f16_bits_finite(0x7bffu));  // max finite FP16 (65504.0)
    assert(!f16_bits_finite(0x7c00u)); // +Inf
    assert(!f16_bits_finite(0xfc00u)); // -Inf
    assert(!f16_bits_finite(0x7e00u)); // qNaN
    const uint16_t overflow_bits = f32_to_f16_bits(70000.0f);
    assert(!f16_bits_finite(overflow_bits));
    // Subnormal roundtrip check.
    const uint16_t min_sub = 0x0001u;
    const float sub_f32 = f16_bits_to_f32(min_sub);
    assert(std::isfinite(sub_f32) && sub_f32 > 0.0f);
    assert(f32_to_f16_bits(sub_f32) == min_sub);
}
} // namespace
int main() {
    test_f16_helpers();
    // Every segment's own prefix image remains authoritative across two seals,
    // tail wrap, arbitrary unaligned live reuse, and maximum b/ub=256 for both
    // F32 and FP16 Q/OUT wire formats.
    for (uint32_t tail : {0u, 128u}) {
        for (uint32_t wire : {uint32_t(RKVA_WIRE_F32), uint32_t(RKVA_WIRE_F16)}) {
            for (uint32_t begin : {0u, 1u, 120u, 127u, 128u, 255u, 8191u}) {
                for (uint32_t count : {1u, 8u, 9u, 64u, 256u}) {
                    auto data = payload(begin, count, tail, wire);
                    uint32_t peeked = 99u;
                    assert(peek_wire(data.data(), data.size(), &peeked) && peeked == wire);
                    assert(valid(data, begin, count, tail, 1, wire));
                    // Wrong expected wire must fail closed.
                    const uint32_t other_wire =
                        (wire == RKVA_WIRE_F16) ? RKVA_WIRE_F32 : RKVA_WIRE_F16;
                    assert(!valid(data, begin, count, tail, 1, other_wire));
                    auto truncated = data;
                    truncated.pop_back();
                    assert(!valid(truncated, begin, count, tail, 1, wire));
                    data.push_back(0);
                    assert(!valid(data, begin, count, tail, 1, wire));
                }
            }
        }
        auto data = payload(120, 256, tail, RKVA_WIRE_F32);
        std::vector<Segment> segments;
        assert(validate(data.data(), data.size(), 256, descriptor(tail), 24, 4, 120, 0, true,
                        rkva_kvarn_range::coverage_after_reset(120, tail), 1, segments));
        // Last-segment corruption is rejected before any first-segment writes.
        int32_t bad_position = -1;
        std::memcpy(data.data() + segments.back().positions_offset, &bad_position, 4);
        assert(!valid(data, 120, 256, tail));
        data = payload(120, 256, tail, RKVA_WIRE_F32);
        float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(data.data() + segments.back().query_offset, &nan, 4);
        assert(!valid(data, 120, 256, tail));
        data = payload(120, 256, tail, RKVA_WIRE_F32);
        assert(!valid(data, 120, 256, tail, UINT64_MAX - 31));
        auto wrong_revision = descriptor(tail);
        wrong_revision.layout_version = 2;
        assert(!validate(data.data(), data.size(), 256, wrong_revision, 24, 4, 120, 0, true,
                         rkva_kvarn_range::coverage_after_reset(120, tail), 1, segments));

        // FP16 bulk admission: NaN/Inf in any segment Q or mixed segment wire is rejected.
        auto data_f16 = payload(120, 256, tail, RKVA_WIRE_F16);
        std::vector<Segment> segs_f16;
        assert(validate_wire(data_f16.data(), data_f16.size(), 256, descriptor(tail), 24, 4, 120, 0,
                             true, rkva_kvarn_range::coverage_after_reset(120, tail), 1,
                             RKVA_WIRE_F16, segs_f16));
        for (uint16_t bad_f16 : {uint16_t(0x7c00u), uint16_t(0xfc00u), uint16_t(0x7e00u)}) {
            auto bad_q = data_f16;
            std::memcpy(bad_q.data() + segs_f16.back().query_offset, &bad_f16, sizeof(bad_f16));
            assert(!valid(bad_q, 120, 256, tail, 1, RKVA_WIRE_F16));
        }
        // Flip the last segment's rkva_range_attn.wire from F16 to F32: must reject mixed wire.
        auto mixed_wire = data_f16;
        const uint32_t wire_f32 = RKVA_WIRE_F32;
        std::memcpy(mixed_wire.data() + segs_f16.back().positions_offset - sizeof(rkva_range_attn),
                    &wire_f32, sizeof(wire_f32));
        assert(!valid(mixed_wire, 120, 256, tail, 1, RKVA_WIRE_F16));
    }
    std::puts("native bulk envelope/coverage/f16 admission: passed");
}
