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
    h.wave_lane_min = 32;
    h.wave_lane_max = 64;
    const std::string json = r.json(h);
    assert(json.find("\"schema\":\"rkva-range-profile-v1\"") != std::string::npos);
    assert(json.find("\"samples_overwritten\":2") != std::string::npos);
    assert(json.find("upload/q4_0/nq1/r257-1k/warm") != std::string::npos);
    assert(json.find("attention/q4_0/reference/nq1/r257-1k/warm") != std::string::npos);
    assert(json.find("\"readback_copy\"") != std::string::npos);
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
