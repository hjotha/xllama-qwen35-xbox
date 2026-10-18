// Host checks for uwp/rkva/rkva-range-profile.h: kernel-mode policy, split
// math, ring behaviour and JSON report shape. Build and run:
//   g++ -std=c++17 -Wall -Wextra -Werror tests/rkva_range_profile_cpu.cpp -o /tmp/rkva_profile
//   /tmp/rkva_profile > /tmp/rkva_profile.json && python3 -m json.tool /tmp/rkva_profile.json
#include "../uwp/rkva/rkva-range-profile.h"
#include <cassert>
#include <cstdio>
#include <cstring>

using namespace rkva_rprof;

int main() {
    // Mode validation and names.
    for (uint32_t m = 0; m < kModeCount; ++m) {
        assert(valid_mode(m));
        assert(std::strcmp(mode_name(m), "invalid") != 0);
    }
    assert(!valid_mode(4) && !valid_mode(0xffffffffu));

    // Reference live259 split policy is unchanged.
    assert(reference_split_k(0) == 1 && reference_split_k(1) == 1);
    assert(reference_split_k(256) == 1 && reference_split_k(257) == 2);
    assert(reference_split_k(28672) == 32 && reference_split_k(1u << 20) == 32);

    // Optimized split policies stay within the opt reducer and partial buffer.
    for (uint32_t m = 1; m < kModeCount; ++m) {
        for (uint32_t a : {0u, 1u, 31u, 32u, 33u, 128u, 129u, 512u, 4096u, 28672u, 131072u}) {
            const uint32_t k = optimized_split_k(a, m);
            assert(k >= 1 && k <= kOptMaxSplits && k <= split_policy(m).max_splits);
            const uint32_t rows = split_rows(a, k);
            assert(uint64_t(rows) * k >= a);                // every row is covered
            assert(a == 0 || uint64_t(rows) * (k - 1) < a); // no split starts past the end
        }
    }
    assert(optimized_split_k(1, kModeOptBalanced) == 1); // direct-output path
    assert(optimized_split_k(28672, kModeOptBalanced) == 128);
    assert(optimized_split_k(28672, kModeOptFine) == 256);
    assert(optimized_split_k(28672, kModeOptCoarse) == 56);

    // Per-call gate: qualified Q4 decode shape only.
    assert(optimized_q4_supported(1, 1, 1, 24, 4, 256));
    assert(!optimized_q4_supported(0, 1, 1, 24, 4, 256)); // reference mode
    assert(!optimized_q4_supported(4, 1, 1, 24, 4, 256)); // invalid mode
    assert(!optimized_q4_supported(1, 0, 1, 24, 4, 256)); // F16
    assert(!optimized_q4_supported(1, 2, 1, 24, 4, 256)); // Q8
    assert(!optimized_q4_supported(1, 3, 1, 24, 4, 256)); // KVarN4
    assert(!optimized_q4_supported(1, 1, 2, 24, 4, 256)); // multi-query
    assert(!optimized_q4_supported(1, 1, 0, 24, 4, 256));
    assert(!optimized_q4_supported(1, 1, 1, 32, 4, 256)); // GQA 8
    assert(!optimized_q4_supported(1, 1, 1, 24, 4, 128)); // head_dim 128
    assert(!optimized_q4_supported(1, 1, 1, 24, 0, 256));

    // Storage-qualified kernel names: Q4 keeps the 261 names, mode 0 is reference.
    for (uint32_t m = 0; m < kModeCount; ++m) {
        assert(std::strcmp(kernel_name(kStorageQ4, m), mode_name(m)) == 0);
        for (uint32_t st = 0; st < kStorageCount; ++st) {
            assert(std::strcmp(kernel_name(st, m), "invalid") != 0);
        }
    }
    assert(std::strcmp(kernel_name(kStorageF16, 0), "reference") == 0);
    assert(std::strcmp(kernel_name(kStorageF16, 1), "f16_gqa_balanced") == 0);
    assert(std::strcmp(kernel_name(kStorageQ8, 2), "q8_gqa_fine") == 0);
    assert(std::strcmp(kernel_name(kStorageKvarn4, 3), "kvarn4_gqa_coarse") == 0);
    assert(std::strcmp(kernel_name(7, 1), "invalid") == 0);

    // F16/Q8 gate: same decode shape, only storages 0 and 2.
    for (uint32_t m = 1; m < kModeCount; ++m) {
        assert(optimized_format_supported(m, 0, 1, 24, 4, 256));
        assert(optimized_format_supported(m, 2, 1, 24, 4, 256));
    }
    assert(!optimized_format_supported(0, 0, 1, 24, 4, 256));
    assert(!optimized_format_supported(4, 2, 1, 24, 4, 256));
    assert(!optimized_format_supported(1, 1, 1, 24, 4, 256)); // Q4 has its own gate
    assert(!optimized_format_supported(1, 3, 1, 24, 4, 256)); // KVarN4 has its own gate
    assert(!optimized_format_supported(1, 0, 2, 24, 4, 256));
    assert(!optimized_format_supported(1, 2, 1, 32, 4, 256));
    assert(!optimized_format_supported(1, 2, 1, 24, 4, 128));

    // KVarN4 gate: descriptor-v1 origin/tail shapes only.
    assert(optimized_kvarn4_supported(1, 3, 1, 24, 4, 256, 102400, 0));
    assert(optimized_kvarn4_supported(2, 3, 1, 24, 4, 256, 102400, 128));
    assert(optimized_kvarn4_supported(3, 3, 1, 24, 4, 256, 0, 128));
    assert(!optimized_kvarn4_supported(0, 3, 1, 24, 4, 256, 102400, 128));
    assert(!optimized_kvarn4_supported(1, 1, 1, 24, 4, 256, 102400, 128));
    assert(!optimized_kvarn4_supported(1, 3, 1, 24, 4, 256, 102464, 128)); // unaligned origin
    assert(!optimized_kvarn4_supported(1, 3, 1, 24, 4, 256, 102400, 64));  // other tail
    assert(!optimized_kvarn4_supported(1, 3, 2, 24, 4, 256, 102400, 128));
    assert(!optimized_kvarn4_supported(1, 3, 1, 24, 4, 128, 102400, 128));
    for (uint32_t m = 1; m < kModeCount; ++m) {
        for (uint32_t a : {0u, 1u, 7u, 8u, 9u, 127u, 128u, 129u, 1000u, 4133u, 28672u, 131072u}) {
            const KvarnSplit ks = kvarn4_split(a, m);
            assert(ks.split_k >= 1 && ks.split_k <= kOptMaxSplits);
            assert(ks.split_rows % kKvarnTile == 0);
            assert(uint64_t(ks.split_rows) * ks.split_k >= a);
            assert(a == 0 || uint64_t(ks.split_rows) * (ks.split_k - 1) < a);
        }
    }
    assert(kvarn4_split(1, 1).split_k == 1);         // direct output
    assert(kvarn4_split(1000, 1).split_rows == 128); // 125 -> 128 rows, 8 splits
    assert(kvarn4_split(1000, 1).split_k == 8);

    // Bucket keys separate warm decode from prefill and cold calls.
    Sample s;
    s.kind = kAttention;
    s.storage = 1;
    s.n_tokens = 1;
    s.rows = 28672;
    assert(bucket_key(s) == "attention/q4_0/reference/nq1/r16k-32k/warm");
    s.optimized = 1;
    s.mode = 2;
    s.alloc_grew = 1;
    s.n_tokens = 256;
    assert(bucket_key(s) == "attention/q4_0/optimized-q4_gqa_fine/nq129+/r16k-32k/cold");
    s.storage = 3;
    s.mode = 1;
    s.alloc_grew = 0;
    s.n_tokens = 1;
    assert(bucket_key(s) == "attention/kvarn4/optimized-kvarn4_gqa_balanced/nq1/r16k-32k/warm");
    s.storage = 2;
    assert(bucket_key(s) == "attention/q8_0/optimized-q8_gqa_balanced/nq1/r16k-32k/warm");

    // Ring: fixed capacity, overwrites oldest, counts everything.
    Recorder r;
    assert(r.capacity() == 0);
    r.record(Sample{}); // unconfigured: counted, not stored
    assert(r.recorded() == 1 && r.stored() == 0);
    r.configure(4);
    r.reset();
    for (uint32_t i = 0; i < 6; ++i) {
        Sample x;
        x.kind = i & 1 ? kAttention : kUpload;
        x.storage = 1;
        x.n_tokens = 1;
        x.rows = 1000 + i;
        x.split_k = 4;
        x.groups = 96;
        x.gpu_valid = 1;
        x.gpu_ticks[0] = 1000 * (i + 1);
        x.gpu_ticks[1] = 10;
        x.gpu_ticks[2] = 5;
        for (uint32_t span = 0; span < kCpuSpanCount; ++span) {
            x.cpu_ns[span] = 1000 * (span + 1);
        }
        r.record(x);
    }
    assert(r.recorded() == 6 && r.stored() == 4 && r.capacity() == 4);
    r.counters.calls[kUpload] = 3;
    r.counters.calls[kAttention] = 3;

    Header h;
    h.profiling = true;
    h.q4_kernel_mode = 1;
    h.gpu_timestamps = true;
    h.gpu_timestamp_frequency = 1000000; // 1 tick == 1 us
    h.gpu_timestamp_status = "ready \"quoted\"\n";
    h.q4_optimized_available = true;
    h.q4_optimized_status = "ready";
    h.optimized_available[kStorageQ4] = true;
    h.optimized_status[kStorageQ4] = "ready";
    h.optimized_status[kStorageKvarn4] = "kvarn4 optimized pipeline state: 0x80070057";
    r.counters.optimized_by_storage[kStorageKvarn4] = 5;
    r.counters.reference_by_storage[kStorageF16] = 2;
    h.wave_lane_min = 32;
    h.wave_lane_max = 64;
    const std::string json = r.json(h);
    assert(json.find("\"schema\":\"rkva-range-profile-v1\"") != std::string::npos);
    assert(json.find("\"samples_overwritten\":2") != std::string::npos);
    assert(json.find("upload/q4_0/nq1/r257-1k/warm") != std::string::npos);
    assert(json.find("attention/q4_0/reference/nq1/r257-1k/warm") != std::string::npos);
    assert(json.find("\"readback_copy\"") != std::string::npos);
    assert(json.find("\"kernel_mode_scope\":\"legacy name q4_kernel_mode") != std::string::npos);
    assert(json.find("\"q4_0\":{\"available\":true,\"status\":\"ready\",\"optimized_calls\":0") !=
           std::string::npos);
    assert(json.find("\"kvarn4\":{\"available\":false,\"status\":\"kvarn4 optimized pipeline "
                     "state: 0x80070057\",\"optimized_calls\":5,\"reference_calls\":0}") !=
           std::string::npos);
    assert(json.find("\"f16\":{\"available\":false,\"status\":\"\",\"optimized_calls\":0,"
                     "\"reference_calls\":2}") != std::string::npos);
    assert(json.find("\\\"quoted\\\"\\u000a") != std::string::npos);
    std::printf("%s\n", json.c_str());

    // Reset clears samples and counters but keeps the reserved ring.
    r.reset();
    assert(r.recorded() == 0 && r.stored() == 0 && r.capacity() == 4);
    assert(r.counters.calls[kAttention] == 0);
    const std::string empty = Recorder().json(Header{});
    assert(empty.find("\"buckets\":{}") != std::string::npos);
    std::fprintf(stderr, "rkva_range_profile_cpu: ok\n");
    return 0;
}
