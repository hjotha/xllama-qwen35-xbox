#include "../uwp/rkva/rkva-kvarn-bulk.h"
#include <cassert>
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
std::vector<uint8_t> payload(uint32_t begin, uint32_t count, uint32_t tail) {
    std::vector<uint8_t> data;
    const uint32_t segments = (count + 7) / 8;
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
        append(data, rkva_range_attn{RKVA_WIRE_F32, 0, {0, 0}});
        for (uint32_t row = 0; row < n; ++row)
            append(data, int32_t(96256 + previous + row));
        data.resize(data.size() + size_t(n) * 24 * 256 * 4);
    }
    return data;
}
bool valid(const std::vector<uint8_t>& data, uint32_t begin, uint32_t count, uint32_t tail,
           uint64_t generation = 1) {
    std::vector<Segment> segments;
    return validate(data.data(), data.size(), count, descriptor(tail), 24, 4, begin, begin / 128,
                    begin != 0, rkva_kvarn_range::coverage_after_reset(begin, tail), generation,
                    segments);
}
} // namespace
int main() {
    // Every segment's own prefix image remains authoritative across two seals,
    // tail wrap, arbitrary unaligned live reuse, and maximum b/ub=256.
    for (uint32_t tail : {0u, 128u}) {
        for (uint32_t begin : {0u, 1u, 120u, 127u, 128u, 255u, 8191u}) {
            for (uint32_t count : {1u, 8u, 9u, 64u, 256u}) {
                auto data = payload(begin, count, tail);
                assert(valid(data, begin, count, tail));
                auto truncated = data;
                truncated.pop_back();
                assert(!valid(truncated, begin, count, tail));
                data.push_back(0);
                assert(!valid(data, begin, count, tail));
            }
        }
        auto data = payload(120, 256, tail);
        std::vector<Segment> segments;
        assert(validate(data.data(), data.size(), 256, descriptor(tail), 24, 4, 120, 0, true,
                        rkva_kvarn_range::coverage_after_reset(120, tail), 1, segments));
        // Last-segment corruption is rejected before any first-segment writes.
        int32_t bad_position = -1;
        std::memcpy(data.data() + segments.back().positions_offset, &bad_position, 4);
        assert(!valid(data, 120, 256, tail));
        data = payload(120, 256, tail);
        float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(data.data() + segments.back().query_offset, &nan, 4);
        assert(!valid(data, 120, 256, tail));
        data = payload(120, 256, tail);
        assert(!valid(data, 120, 256, tail, UINT64_MAX - 31));
        auto wrong_revision = descriptor(tail);
        wrong_revision.layout_version = 2;
        assert(!validate(data.data(), data.size(), 256, wrong_revision, 24, 4, 120, 0, true,
                         rkva_kvarn_range::coverage_after_reset(120, tail), 1, segments));
    }
    std::puts("native bulk envelope/coverage admission: passed");
}
