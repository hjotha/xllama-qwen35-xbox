#include "../uwp/rkva/rkva-kvarn-range.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
using namespace rkva_kvarn_range;
int main() {
    rkva_kvarn_range_config c{102400, 28672, 4, 1, 128, 128, 256, 2, 4,
                              4,      8960,  3, 2, 128, 128, 1,   1, 1};
    assert(valid_descriptor(c, 256));
    auto generation_descriptor = c;
    generation_descriptor.layout_version = 2;
    assert(valid_descriptor(generation_descriptor, 256));
    generation_descriptor.layout_version = 3;
    assert(!valid_descriptor(generation_descriptor, 256));
    for (unsigned i = 0; i < 18; ++i) {
        auto bad = c;
        auto* words = reinterpret_cast<uint32_t*>(&bad);
        words[i] ^= 1;
        if (i == 1)
            continue;
        assert(!valid_descriptor(bad, 256));
    }
    auto zero = c;
    zero.precision_tail_tokens = 0;
    assert(valid_descriptor(zero, 256));
    zero = c;
    zero.precision_tail_tokens = 64;
    assert(!valid_descriptor(zero, 256));
    assert(!valid_descriptor(c, 128));
    auto oversized = c;
    oversized.capacity = UINT32_MAX;
    assert(!valid_descriptor(oversized, 256));
    rkva_kvarn_range_upload u{512, 0, 4, 1, 0, 0, {0, 0}};
    assert(valid_upload(u, c.capacity, 0, 0, false));
    u.flags = 0;
    assert(!valid_upload(u, c.capacity, 0, 0, false));
    u = {641, 4, 1, 0, 512, 129, {0, 0}};
    assert(valid_upload(u, c.capacity, 512, 4, true));
    u.record_groups = 0;
    assert(!valid_upload(u, c.capacity, 512, 4, true));
    u = {641, 0, 5, 1, 0, 0, {0, 0}};
    assert(valid_upload(u, c.capacity, 0, 0, false));
    u = {513, 0, 4, 1, 0, 0, {0, 0}};
    assert(valid_upload(u, c.capacity, 0, 0, false));
    u.record_groups = 5;
    assert(!valid_upload(u, c.capacity, 0, 0, false)); // no future groups
    u = {512, 0, 4, 0, 641, 0, {0, 0}};
    assert(!valid_upload(u, c.capacity, 641, 4, true));
    u = {641, 0, 5, 0, 0, 641, {0, 0}};
    assert(!valid_upload(u, c.capacity, 0, 0, false)); // rewind needs reset
    std::array<int, 384> stage;
    stage.fill(-1);
    std::array<int, 128> tail;
    tail.fill(-1);
    unsigned previous = 0;
    for (unsigned live : {1u, 127u, 128u, 129u, 255u, 256u, 257u, 512u, 641u, 769u, 1025u}) {
        stage_chunks(102400 + previous, live - previous,
                     [&](unsigned source, unsigned target, unsigned rows) {
                         for (unsigned i = 0; i < rows; ++i)
                             stage[target + i] = previous + source + i;
                     });
        tail_chunks(previous, live - previous,
                    [&](unsigned source, unsigned target, unsigned rows) {
                        for (unsigned i = 0; i < rows; ++i)
                            tail[target + i] = previous + source + i;
                    });
        for (unsigned t = 0; t < live; ++t) {
            unsigned g = (102400 + t) / 128, lg = (102400 + live - 1) / 128;
            if (g == 0 || g + 1 >= lg) {
                unsigned slot = g == 0 ? 0 : 1 + (g - 1) % 2;
                assert(stage[slot * 128 + t % 128] == int(t));
            }
        }
        for (unsigned t = live - std::min(live, 128u); t < live; ++t)
            assert(tail[t % 128] == int(t));
        previous = live;
    }
    // Full restored-tail snapshot at a nonaligned position overwrites both ring halves.
    unsigned restored = 513, begin = restored - 128;
    tail_chunks(begin, 128, [&](unsigned source, unsigned target, unsigned rows) {
        for (unsigned i = 0; i < rows; ++i)
            tail[target + i] = begin + source + i;
    });
    for (unsigned t = begin; t < restored; ++t)
        assert(tail[t % 128] == int(t));
    std::puts("KVarN descriptor, coverage, append/stage/tail ring and rewind gates passed");
}
