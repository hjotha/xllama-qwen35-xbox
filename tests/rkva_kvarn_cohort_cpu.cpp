// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "../uwp/rkva/rkva-kvarn-cohort.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>

namespace {
using rkva_kvarn_cohort::LayerState;
using rkva_kvarn_cohort::Plan;
uint32_t checks = 0;
void check(bool value) {
    ++checks;
    assert(value);
}
template <class T> void append(std::vector<uint8_t>& bytes, const T& value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
template <class T> void store(std::vector<uint8_t>& bytes, size_t offset, const T& value) {
    assert(offset <= bytes.size() && sizeof(value) <= bytes.size() - offset);
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
rkva_kvarn_range_config descriptor() {
    return {
        96256, 34816, RKVA_RANGE_STORAGE_KVARN4, 3, 128, 128, 256, 2, 4, 4, 8960, 3, 2, 128, 128, 1,
        1,     1};
}
std::vector<uint8_t> bulk(uint32_t begin, uint32_t count, uint32_t wire) {
    std::vector<uint8_t> bytes;
    append(bytes, rkva_kvarn_range_bulk{1, (count + 7) / 8, count, 0});
    const uint32_t q_bytes = wire == RKVA_WIRE_FP8 ? 1 : wire == RKVA_WIRE_F16 ? 2 : 4;
    for (uint32_t first = 0; first < count; first += 8) {
        const uint32_t n = std::min(8u, count - first), before = begin + first, live = before + n;
        const bool reset = before == 0;
        const uint32_t groups = live / 128 - before / 128;
        const rkva_kvarn_range_upload upload{live,
                                             before / 128,
                                             groups,
                                             reset ? RKVA_KVARN_UPLOAD_RESET : 0u,
                                             reset ? 0u : before,
                                             reset ? 0u : n,
                                             {0, 0}};
        const size_t records = size_t(groups) * 2 * 8960;
        const size_t planes =
            2 * (records + ((reset ? 384 : n) + (reset ? std::min(live, 128u) : n)) * 256 * 2);
        append(bytes, rkva_kvarn_range_bulk_segment{uint32_t(sizeof(upload) + planes), n});
        append(bytes, upload);
        bytes.resize(bytes.size() + planes, 0);
        append(bytes, rkva_range_attn{wire, 0, {0, 0}});
        for (uint32_t q = 0; q < n; ++q)
            append(bytes, int32_t(96256 + before + q));
        bytes.resize(bytes.size() + size_t(n) * 256 * q_bytes, 0);
    }
    return bytes;
}
struct Fixture {
    std::array<LayerState, 4> resident{};
    rkva_kvarn_range_config config = descriptor();
    uint64_t generation = uint64_t(7) << 32;
    uint32_t n_head = 1, n_head_kv = 1, n_layers = 4, unavailable_layer = UINT32_MAX;
    std::vector<rkva_kvarn_bulk::Segment> scratch;
    uint32_t snapshot_calls = 0, executions = 0;
    Fixture() {
        for (uint32_t layer = 0; layer < 4; ++layer) {
            const uint32_t live = layer * 128;
            resident[layer] = {live, live / 128, live != 0,
                               rkva_kvarn_range::coverage_after_reset(live, 128)};
        }
    }
    std::vector<uint8_t> cohort(uint32_t count = 4, uint32_t nq = 64,
                                uint32_t wire = RKVA_WIRE_FP8) const {
        std::vector<uint8_t> bytes;
        append(bytes, rkva_kvarn_cohort_header{1, count, count * nq, 0});
        for (uint32_t layer = 0; layer < count; ++layer) {
            const auto body = bulk(resident[layer].live, nq, wire);
            append(bytes, rkva_kvarn_cohort_entry{layer, nq, uint32_t(body.size()), 0,
                                                  uint64_t(layer + 1) * 2, 77});
            bytes.insert(bytes.end(), body.begin(), body.end());
        }
        return bytes;
    }
    bool validate(const uint8_t* data, size_t bytes, uint32_t nq, Plan& plan) {
        return rkva_kvarn_cohort::validate(
            data, bytes, nq, n_layers, n_head, n_head_kv, generation,
            [&](uint32_t layer, rkva_kvarn_range_config& d, LayerState& state) {
                ++snapshot_calls;
                if (layer >= resident.size() || layer == unavailable_layer)
                    return false;
                d = config;
                state = resident[layer];
                return true;
            },
            plan, scratch);
    }
    bool validate(const std::vector<uint8_t>& bytes, uint32_t nq, Plan& plan) {
        return validate(bytes.data(), bytes.size(), nq, plan);
    }
    bool admit_then_execute(const std::vector<uint8_t>& bytes, uint32_t nq) {
        Plan plan;
        if (!validate(bytes, nq, plan))
            return false;
        for (uint32_t i = 0; i < plan.header.entry_count; ++i) {
            ++executions;
            const auto& e = plan.entries[i];
            resident[e.wire.layer].live = e.live_after;
            generation = e.generation_after;
        }
        return true;
    }
};
bool same_state(const LayerState& a, const LayerState& b) {
    return a.live == b.live && a.groups == b.groups && a.ready == b.ready &&
           a.coverage.tail_begin == b.coverage.tail_begin &&
           a.coverage.stage_begin == b.coverage.stage_begin;
}
size_t entry_offset(const Plan& plan, uint32_t i) {
    return plan.entries[i].body_offset - sizeof(rkva_kvarn_cohort_entry);
}
void reject_without_mutation(Fixture& fixture, const std::vector<uint8_t>& bytes,
                             uint32_t queries = 256) {
    const auto before = fixture.resident;
    const uint64_t generation = fixture.generation;
    const uint32_t executions = fixture.executions;
    Plan plan;
    plan.header.entry_count = 123;
    check(!fixture.validate(bytes, queries, plan));
    check(plan.header.entry_count == 0 && plan.reply_bytes == 0 && plan.segments == 0);
    check(!fixture.admit_then_execute(bytes, queries));
    check(fixture.generation == generation && fixture.executions == executions);
    for (size_t i = 0; i < before.size(); ++i)
        check(same_state(fixture.resident[i], before[i]));
}
void positive_shapes() {
    for (uint32_t wire : {RKVA_WIRE_FP8, RKVA_WIRE_F16, RKVA_WIRE_F32}) {
        for (uint32_t count = 1; count <= 4; ++count) {
            for (uint32_t nq : {1u, 8u, 9u, 16u, 64u}) {
                Fixture fixture;
                const auto bytes = fixture.cohort(count, nq, wire);
                const auto before = fixture.resident;
                Plan plan;
                check(fixture.validate(bytes, count * nq, plan));
                check(plan.header.entry_count == count && plan.header.total_queries == count * nq);
                check(plan.segments == count * ((nq + 7) / 8));
                check(plan.generation_after == fixture.generation + plan.segments);
                for (uint32_t i = 0; i < count; ++i) {
                    check(same_state(fixture.resident[i], before[i]));
                    check(plan.entries[i].wire.ticket == uint64_t(i + 1) * 2);
                    check(plan.entries[i].wire.epoch == 77);
                    check(plan.entries[i].live_after == before[i].live + nq);
                    std::vector<uint8_t> reply(plan.entries[i].reply_bytes, 0);
                    check(rkva_kvarn_cohort::valid_reply(plan.entries[i], reply.data(),
                                                         reply.size(), fixture.n_head));
                }
                check(fixture.admit_then_execute(bytes, count * nq));
                check(fixture.executions == count);
            }
        }
    }
    // Existing record group sealing and tail coverage are retained at a64-tile boundary.
    for (uint32_t wire : {RKVA_WIRE_FP8, RKVA_WIRE_F16, RKVA_WIRE_F32}) {
        Fixture seal;
        for (uint32_t layer = 0; layer < 4; ++layer) {
            const uint32_t live = 120 + layer * 128;
            seal.resident[layer] = {live, live / 128, true,
                                    rkva_kvarn_range::coverage_after_reset(live, 128)};
        }
        Plan sealed;
        check(seal.validate(seal.cohort(4, 64, wire), 256, sealed));
        check(sealed.segments == 32);
    }
    // Wire headers and all payload reads permit an unaligned receive span.
    Fixture fixture;
    const auto bytes = fixture.cohort();
    std::vector<uint8_t> unaligned(1, 0);
    unaligned.insert(unaligned.end(), bytes.begin(), bytes.end());
    Plan plan;
    check(fixture.validate(unaligned.data() + 1, bytes.size(), 256, plan));
}
void malformed_envelope() {
    Fixture fixture;
    const auto bytes = fixture.cohort();
    Plan valid;
    check(fixture.validate(bytes, 256, valid));
    for (uint32_t value : {0u, 2u, UINT32_MAX}) {
        auto bad = bytes;
        store(bad, 0, value);
        reject_without_mutation(fixture, bad);
    }
    for (uint32_t value : {0u, 5u, UINT32_MAX}) {
        auto bad = bytes;
        store(bad, 4, value);
        reject_without_mutation(fixture, bad);
    }
    for (uint32_t value : {0u, 255u, 257u, UINT32_MAX}) {
        auto bad = bytes;
        store(bad, 8, value);
        reject_without_mutation(fixture, bad);
    }
    {
        auto bad = bytes;
        store(bad, 12, 1u);
        reject_without_mutation(fixture, bad);
        reject_without_mutation(fixture, bytes, 255);
        bad = bytes;
        bad.push_back(0);
        reject_without_mutation(fixture, bad);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        const size_t offset = entry_offset(valid, i);
        for (size_t field : {size_t(0), size_t(4), size_t(8), size_t(12)}) {
            auto bad = bytes;
            store(bad, offset + field, UINT32_MAX);
            reject_without_mutation(fixture, bad);
        }
        for (size_t field : {size_t(16), size_t(24)}) {
            auto bad = bytes;
            store(bad, offset + field, uint64_t(0));
            reject_without_mutation(fixture, bad);
        }
        if (i) {
            auto bad = bytes;
            store(bad, offset, 0u); // duplicate the first layer
            reject_without_mutation(fixture, bad);
            bad = bytes;
            store(bad, offset + 16, valid.entries[i - 1].wire.ticket);
            reject_without_mutation(fixture, bad);
            bad = bytes;
            store(bad, offset + 24, uint64_t(78));
            reject_without_mutation(fixture, bad);
        }
        for (size_t cut : {offset, offset + 1, offset + 31, valid.entries[i].body_offset,
                           valid.entries[i].body_offset + 1,
                           valid.entries[i].body_offset + valid.entries[i].wire.body_bytes - 1}) {
            std::vector<uint8_t> truncated(bytes.begin(), bytes.begin() + cut);
            reject_without_mutation(fixture, truncated);
        }
    }
    for (size_t size = 0; size < sizeof(rkva_kvarn_cohort_header); ++size) {
        std::vector<uint8_t> truncated(bytes.begin(), bytes.begin() + size);
        reject_without_mutation(fixture, truncated);
    }
    check(!fixture.validate(nullptr, bytes.size(), 256, valid));
    check(!fixture.validate(bytes.data(), size_t(rkva_kvarn_cohort::max_bytes + 1), 256, valid));
}
void invalid_resident_and_generation() {
    for (uint64_t generation : {uint64_t(0), UINT64_MAX, (uint64_t(7) << 32) | UINT32_MAX,
                                (uint64_t(7) << 32) | (UINT32_MAX - 31u)}) {
        Fixture fixture;
        fixture.generation = generation;
        reject_without_mutation(fixture, fixture.cohort());
    }
    {
        Fixture fixture;
        fixture.generation |= UINT32_MAX - 32u;
        Plan plan;
        check(fixture.validate(fixture.cohort(), 256, plan));
        check(uint32_t(plan.generation_after) == UINT32_MAX); // final legal mutation
    }
    for (uint32_t broken = 0; broken < 9; ++broken) {
        Fixture fixture;
        const auto bytes = fixture.cohort();
        switch (broken) {
        case 0:
            fixture.unavailable_layer = 3;
            break;
        case 1:
            fixture.resident[3].live += 1;
            break;
        case 2:
            fixture.resident[3].ready = false;
            break;
        case 3:
            fixture.resident[3].coverage = {4096, 4096};
            break;
        case 4:
            fixture.config.layout_version = 2;
            break;
        case 5:
            fixture.n_head = 0;
            break;
        case 6:
            fixture.n_head = 257;
            break;
        case 7:
            fixture.n_head_kv = 0;
            break;
        case 8:
            fixture.n_layers = 129;
            break;
        }
        reject_without_mutation(fixture, bytes);
    }
    // Each entry individually permits9 queries, but33 aggregate segments do not.
    Fixture fixture;
    auto bytes = fixture.cohort(3, 80);
    auto fourth = bulk(fixture.resident[3].live, 9, RKVA_WIRE_FP8);
    store(bytes, 4, 4u);
    store(bytes, 8, 249u);
    append(bytes, rkva_kvarn_cohort_entry{3, 9, uint32_t(fourth.size()), 0, 8, 77});
    bytes.insert(bytes.end(), fourth.begin(), fourth.end());
    // 30+2=32 is legal; add a third inner segment by using17queries instead.
    Plan plan;
    check(fixture.validate(bytes, 249, plan));
    bytes = fixture.cohort(3, 72);                              // 27 segments
    fourth = bulk(fixture.resident[3].live, 33, RKVA_WIRE_FP8); // 5 ->32 legal
    store(bytes, 4, 4u);
    store(bytes, 8, 249u);
    append(bytes, rkva_kvarn_cohort_entry{3, 33, uint32_t(fourth.size()), 0, 8, 77});
    bytes.insert(bytes.end(), fourth.begin(), fourth.end());
    check(fixture.validate(bytes, 249, plan));
    // 33 aggregate1-query segments, while each inner body stays below32.
    std::vector<uint8_t> fragmented;
    append(fragmented, rkva_kvarn_cohort_header{1, 4, 33, 0});
    for (uint32_t layer = 0; layer < 4; ++layer) {
        const uint32_t n = layer == 3 ? 9 : 8;
        std::vector<uint8_t> body;
        append(body, rkva_kvarn_range_bulk{1, n, n, 0});
        for (uint32_t row = 0; row < n; ++row) {
            const auto one = bulk(fixture.resident[layer].live + row, 1, RKVA_WIRE_FP8);
            body.insert(body.end(), one.begin() + sizeof(rkva_kvarn_range_bulk), one.end());
        }
        append(fragmented, rkva_kvarn_cohort_entry{layer, n, uint32_t(body.size()), 0,
                                                   uint64_t(layer + 1) * 2, 77});
        fragmented.insert(fragmented.end(), body.begin(), body.end());
    }
    reject_without_mutation(fixture, fragmented, 33);
}
void bad_suffix_and_replies() {
    for (uint32_t wire : {RKVA_WIRE_FP8, RKVA_WIRE_F16, RKVA_WIRE_F32}) {
        Fixture fixture;
        const auto bytes = fixture.cohort(4, 64, wire);
        Plan plan;
        check(fixture.validate(bytes, 256, plan));
        const auto& last = plan.entries[3];
        std::vector<rkva_kvarn_bulk::Segment> segments;
        check(rkva_kvarn_bulk::validate_wire(
            bytes.data() + last.body_offset, last.wire.body_bytes, 64, fixture.config, 1, 1,
            fixture.resident[3].live, fixture.resident[3].groups, true,
            fixture.resident[3].coverage, fixture.generation, wire, segments));
        auto bad = bytes;
        const size_t q_offset = last.body_offset + segments.back().query_offset;
        if (wire == RKVA_WIRE_FP8)
            bad[q_offset] = 0x7c;
        else if (wire == RKVA_WIRE_F16)
            store(bad, q_offset, uint16_t(0x7c00));
        else
            store(bad, q_offset, std::numeric_limits<float>::quiet_NaN());
        reject_without_mutation(fixture, bad); // 3 valid entries must not execute
        bad = bytes;
        store(bad, last.body_offset + segments.back().positions_offset, int32_t(-1));
        reject_without_mutation(fixture, bad);
        std::vector<uint8_t> reply(last.reply_bytes, 0);
        const size_t out_bytes = size_t(last.wire.n_queries) * 256 *
                                 (wire == RKVA_WIRE_FP8   ? 1
                                  : wire == RKVA_WIRE_F16 ? 2
                                                          : 4);
        const float finite_lse = -2097152.0f;
        store(reply, out_bytes, finite_lse);
        check(rkva_kvarn_cohort::valid_reply(last, reply.data(), reply.size(), 1));
        for (float value :
             {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::quiet_NaN()}) {
            store(reply, out_bytes, value);
            check(!rkva_kvarn_cohort::valid_reply(last, reply.data(), reply.size(), 1));
        }
        store(reply, out_bytes, finite_lse);
        if (wire == RKVA_WIRE_FP8)
            reply[0] = 0xfc;
        else if (wire == RKVA_WIRE_F16)
            store(reply, 0, uint16_t(0x7e00));
        else
            store(reply, 0, std::numeric_limits<float>::infinity());
        check(!rkva_kvarn_cohort::valid_reply(last, reply.data(), reply.size(), 1));
        check(!rkva_kvarn_cohort::valid_reply(last, reply.data(), reply.size() - 1, 1));
        check(!rkva_kvarn_cohort::valid_reply(last, nullptr, reply.size(), 1));
    }
}
} // namespace

int main() {
    positive_shapes();
    malformed_envelope();
    invalid_resident_and_generation();
    bad_suffix_and_replies();
    std::printf("rkva_kvarn_cohort_cpu: %u checks passed\n", checks);
}
