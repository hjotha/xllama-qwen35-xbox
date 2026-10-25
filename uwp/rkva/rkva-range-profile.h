#pragma once
// RKVA v2 range-attention kernel selection and in-RAM profiling aggregation.
// WinRT/D3D12-free so the policy and the JSON report are host-testable.
// See docs/rkva-range-profiling.md for the schema and timing semantics.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace rkva_rprof {

// Range kernel modes. The wire/API name stays the legacy "q4_kernel_mode":
// mode 0 is the reference shader of every storage and the default. Modes 1..3
// select the optimized GQA decode kernel of the session's storage when it is
// qualified and its pipeline exists (Q4_0 since 261; F16, Q8_0 and native
// KVarN4 since the future 263 candidate), with the split policies below.
// Unsupported calls and missing pipelines keep the reference kernel.
enum : uint32_t {
    kModeReference = 0,
    kModeOptBalanced = 1,
    kModeOptFine = 2,
    kModeOptCoarse = 3,
    kModeCount = 4,
};

// Shape the optimized Q4 kernel is specialized for (Qwen35 full attention).
constexpr uint32_t kOptHeadDim = 256;
constexpr uint32_t kOptGqa = 6; // Q heads per KV head
constexpr uint32_t kOptMaxSplits = 256;
// Eight proposed MTP tokens plus the verification anchor. Each query gets
// independent workgroups; larger prefill batches retain the reference path.
constexpr uint32_t kOptMaxQueries = 9;
constexpr uint32_t kOptMinWaveLanes = 32;
constexpr uint32_t kOptMaxWaveLanes = 128;
constexpr uint32_t kReferenceMaxSplits = 32;

inline bool valid_mode(uint32_t mode) {
    return mode < kModeCount;
}

inline const char* mode_name(uint32_t mode) {
    switch (mode) {
    case kModeReference:
        return "reference";
    case kModeOptBalanced:
        return "q4_gqa_balanced";
    case kModeOptFine:
        return "q4_gqa_fine";
    case kModeOptCoarse:
        return "q4_gqa_coarse";
    default:
        return "invalid";
    }
}

enum Storage : uint32_t {
    kStorageF16 = 0,
    kStorageQ4 = 1,
    kStorageQ8 = 2,
    kStorageKvarn4 = 3,
    kStorageCount = 4,
};

// Bucket/report kernel name. Q4_0 keeps the 261 names byte for byte.
inline const char* kernel_name(uint32_t storage, uint32_t mode) {
    if (mode == kModeReference || !valid_mode(mode)) {
        return mode_name(mode);
    }
    static const char* names[kStorageCount][3] = {
        {"f16_gqa_balanced", "f16_gqa_fine", "f16_gqa_coarse"},
        {"q4_gqa_balanced", "q4_gqa_fine", "q4_gqa_coarse"},
        {"q8_gqa_balanced", "q8_gqa_fine", "q8_gqa_coarse"},
        {"kvarn4_gqa_balanced", "kvarn4_gqa_fine", "kvarn4_gqa_coarse"},
    };
    return storage < kStorageCount ? names[storage][mode - 1] : "invalid";
}

struct SplitPolicy {
    uint32_t target_rows;
    uint32_t max_splits;
};

inline SplitPolicy split_policy(uint32_t mode) {
    switch (mode) {
    case kModeOptFine:
        return {32, 256};
    case kModeOptCoarse:
        return {512, 64};
    default:
        return {128, 128};
    }
}

// Reference live259 policy: one split per 256 live rows, at most 32.
inline uint32_t reference_split_k(uint32_t active_count) {
    return std::max(1u, std::min(kReferenceMaxSplits, (active_count + 255) / 256));
}

inline uint32_t optimized_split_k(uint32_t active_count, uint32_t mode) {
    const SplitPolicy p = split_policy(mode);
    const uint32_t wanted = (active_count + p.target_rows - 1) / p.target_rows;
    return std::max(1u, std::min(std::min(p.max_splits, kOptMaxSplits), wanted));
}

inline uint32_t split_rows(uint32_t active_count, uint32_t split_k) {
    return split_k ? (active_count + split_k - 1) / split_k : 0;
}

// Per-call gate. Every unsupported call keeps the reference kernel.
inline bool optimized_q4_supported(uint32_t mode, uint32_t storage, uint32_t n_tokens,
                                   uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim) {
    return mode != kModeReference && valid_mode(mode) && storage == 1 && n_tokens >= 1 &&
           n_tokens <= kOptMaxQueries && head_dim == kOptHeadDim && n_head_kv > 0 &&
           n_head == kOptGqa * n_head_kv;
}

inline bool optimized_shape_supported(uint32_t mode, uint32_t n_tokens, uint32_t n_head,
                                      uint32_t n_head_kv, uint32_t head_dim) {
    return mode != kModeReference && valid_mode(mode) && n_tokens >= 1 &&
           n_tokens <= kOptMaxQueries && head_dim == kOptHeadDim && n_head_kv > 0 &&
           n_head == kOptGqa * n_head_kv;
}

// F16 and Q8_0 share one GQA kernel source (RKVA_OPT_STORAGE specialization)
// and the Q4 split policy.
inline bool optimized_format_supported(uint32_t mode, uint32_t storage, uint32_t n_tokens,
                                       uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim) {
    return (storage == kStorageF16 || storage == kStorageQ8) &&
           optimized_shape_supported(mode, n_tokens, n_head, n_head_kv, head_dim);
}

// Native KVarN4 descriptor v1 shapes the optimized kernel reads directly:
// 128-aligned origin and a 0 or 128 token precision tail.
constexpr uint32_t kKvarnGroup = 128;
constexpr uint32_t kKvarnTile = 8;

inline bool optimized_kvarn4_supported(uint32_t mode, uint32_t storage, uint32_t n_tokens,
                                       uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
                                       uint32_t origin, uint32_t precision_tail_tokens) {
    return storage == kStorageKvarn4 && origin % kKvarnGroup == 0 &&
           (precision_tail_tokens == 0 || precision_tail_tokens == kKvarnGroup) &&
           optimized_shape_supported(mode, n_tokens, n_head, n_head_kv, head_dim);
}

struct KvarnSplit {
    uint32_t split_k;
    uint32_t split_rows; // multiple of kKvarnTile
};

// Mode split policy with split rows rounded up to whole 8-token record tiles,
// then the split count recomputed so every split is non-empty.
inline KvarnSplit kvarn4_split(uint32_t active_count, uint32_t mode) {
    const uint32_t k0 = optimized_split_k(active_count, mode);
    uint32_t rows = split_rows(active_count, k0);
    rows = (rows + kKvarnTile - 1) / kKvarnTile * kKvarnTile;
    const uint32_t k = rows ? (active_count + rows - 1) / rows : 1;
    return {std::max(1u, k), rows};
}

enum Kind : uint8_t { kUpload = 0, kAttention = 1, kKindCount = 2 };

// Disjoint CPU spans; their sum is the engine wall time of the call (profiling
// bookkeeping after the work is reported separately as overhead).
enum CpuSpan : uint8_t {
    kCpuValidate = 0, // argument/session checks
    kCpuAlloc,        // grow-only scratch/upload buffer creation (cold calls)
    kCpuMapCopy,      // Map + memcpy inputs + Unmap
    kCpuRecord,       // allocator/list reset + command recording
    kCpuSubmit,       // Close + ExecuteCommandLists + Signal
    kCpuFenceWait,    // fence completion wait (GPU execution + queue latency)
    kCpuReadback,     // Map + memcpy outputs + Unmap (attention only)
    kCpuSpanCount,
};

// Upload uses stage 0 (copy). Attention: pass1, reduce, readback copy.
enum GpuStage : uint8_t { kGpuStage0 = 0, kGpuStage1, kGpuStage2, kGpuStageCount };

struct Sample {
    uint8_t kind = kUpload;
    uint8_t storage = 0;
    uint8_t optimized = 0;
    uint8_t mode = 0;
    uint8_t gpu_valid = 0;
    uint8_t direct_output = 0;
    uint8_t fence_blocked = 0;
    uint8_t alloc_grew = 0;
    uint32_t n_tokens = 0;
    uint32_t rows = 0; // upload rows, or attention active_count
    uint32_t split_k = 0;
    uint32_t groups = 0; // pass-1 threadgroups
    uint64_t bytes = 0;  // upload payload, or attention Q+positions in / OUT+LSE out
    uint64_t gpu_ticks[kGpuStageCount] = {};
    uint64_t cpu_ns[kCpuSpanCount] = {};
    uint64_t overhead_ns = 0;
};

struct Counters {
    uint64_t calls[kKindCount] = {};
    uint64_t optimized_calls = 0; // optimized Q4_0 attention calls (261 meaning)
    uint64_t reference_q4_calls = 0;
    uint64_t optimized_by_storage[kStorageCount] = {};
    uint64_t reference_by_storage[kStorageCount] = {};
    uint64_t failed_calls = 0;
    uint64_t submits = 0;
    uint64_t fence_blocking_waits = 0;
    uint64_t fence_already_complete = 0;
    uint64_t dispatches = 0;
    uint64_t copy_regions = 0;
    uint64_t buffer_creates = 0;
    uint64_t buffer_create_bytes = 0;
    uint64_t timestamp_resolves = 0;
    uint64_t timestamp_map_failures = 0;
};

struct Header {
    bool profiling = false;
    uint32_t q4_kernel_mode = 0;
    bool gpu_timestamps = false;
    uint64_t gpu_timestamp_frequency = 0;
    std::string gpu_timestamp_status;
    bool q4_optimized_available = false;
    std::string q4_optimized_status;
    // Per storage (index = Storage); the Q4 entry mirrors the fields above.
    bool optimized_available[kStorageCount] = {};
    std::string optimized_status[kStorageCount];
    uint32_t wave_lane_min = 0;
    uint32_t wave_lane_max = 0;
};

inline const char* storage_name(uint32_t storage) {
    switch (storage) {
    case 0:
        return "f16";
    case 1:
        return "q4_0";
    case 2:
        return "q8_0";
    case 3:
        return "kvarn4";
    default:
        return "unknown";
    }
}

// Coarse bands keep decode (nq1, warm) apart from prefill and cold calls.
inline const char* nq_band(uint32_t n_tokens) {
    return n_tokens <= 1     ? "nq1"
           : n_tokens <= 16  ? "nq2-16"
           : n_tokens <= 128 ? "nq17-128"
                             : "nq129+";
}

inline const char* rows_band(uint32_t rows) {
    return rows == 0       ? "r0"
           : rows <= 256   ? "r1-256"
           : rows <= 1024  ? "r257-1k"
           : rows <= 4096  ? "r1k-4k"
           : rows <= 16384 ? "r4k-16k"
           : rows <= 32768 ? "r16k-32k"
                           : "r32k+";
}

// kind/storage/kernel[-mode]/nq band/rows band/warm|cold. rows is the
// attention active_count or the upload row count; cold means the call grew a
// scratch or upload buffer.
inline std::string bucket_key(const Sample& s) {
    std::string key = s.kind == kUpload ? "upload/" : "attention/";
    key += storage_name(s.storage);
    if (s.kind == kAttention) {
        key += s.optimized ? "/optimized-" : "/reference";
        if (s.optimized) {
            key += kernel_name(s.storage, s.mode);
        }
    }
    key += '/';
    key += nq_band(s.n_tokens);
    key += '/';
    key += rows_band(s.rows);
    key += s.alloc_grew ? "/cold" : "/warm";
    return key;
}

inline std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

// Fixed-capacity sample ring. Storage is reserved by configure(), so record()
// never allocates during a timed RPC.
class Recorder {
  public:
    static constexpr size_t kDefaultCapacity = 16384;

    void configure(size_t capacity = kDefaultCapacity) {
        if (ring_.size() != capacity) {
            ring_.assign(capacity, Sample{});
            next_ = 0;
            stored_ = 0;
        }
    }
    void reset() {
        next_ = 0;
        stored_ = 0;
        recorded_ = 0;
        counters = Counters{};
    }
    void record(const Sample& sample) {
        ++recorded_;
        if (ring_.empty()) {
            return;
        }
        ring_[next_] = sample;
        next_ = (next_ + 1) % ring_.size();
        stored_ = std::min(stored_ + 1, ring_.size());
    }
    size_t capacity() const {
        return ring_.size();
    }
    size_t stored() const {
        return stored_;
    }
    uint64_t recorded() const {
        return recorded_;
    }

    std::string json(const Header& h) const {
        std::string o;
        o.reserve(8192);
        char buf[512];
        o += "{\"schema\":\"rkva-range-profile-v1\"";
        std::snprintf(buf, sizeof(buf),
                      ",\"profiling\":%s,\"q4_kernel_mode\":%u,\"q4_kernel_mode_name\":\"%s\"",
                      h.profiling ? "true" : "false", h.q4_kernel_mode,
                      mode_name(h.q4_kernel_mode));
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      ",\"gpu_timestamps\":%s,\"gpu_timestamp_frequency_hz\":%llu"
                      ",\"gpu_timestamp_status\":\"%s\"",
                      h.gpu_timestamps ? "true" : "false",
                      static_cast<unsigned long long>(h.gpu_timestamp_frequency),
                      json_escape(h.gpu_timestamp_status).c_str());
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      ",\"q4_optimized_available\":%s,\"q4_optimized_status\":\"%s\""
                      ",\"wave_lane_min\":%u,\"wave_lane_max\":%u",
                      h.q4_optimized_available ? "true" : "false",
                      json_escape(h.q4_optimized_status).c_str(), h.wave_lane_min, h.wave_lane_max);
        o += buf;
        o += ",\"cpu_clock\":\"std::chrono::steady_clock\"";
        std::snprintf(buf, sizeof(buf),
                      ",\"sample_capacity\":%llu,\"samples_recorded\":%llu,"
                      "\"samples_in_ring\":%llu,\"samples_overwritten\":%llu",
                      static_cast<unsigned long long>(ring_.size()),
                      static_cast<unsigned long long>(recorded_),
                      static_cast<unsigned long long>(stored_),
                      static_cast<unsigned long long>(recorded_ - stored_));
        o += buf;
        const Counters& c = counters;
        std::snprintf(
            buf, sizeof(buf),
            ",\"counters\":{\"upload_calls\":%llu,\"attention_calls\":%llu,"
            "\"optimized_q4_calls\":%llu,\"reference_q4_calls\":%llu,\"failed_calls\":%llu,"
            "\"submits\":%llu,\"fence_blocking_waits\":%llu,\"fence_already_complete\":%llu,"
            "\"dispatches\":%llu,\"copy_regions\":%llu,",
            ull(c.calls[kUpload]), ull(c.calls[kAttention]), ull(c.optimized_calls),
            ull(c.reference_q4_calls), ull(c.failed_calls), ull(c.submits),
            ull(c.fence_blocking_waits), ull(c.fence_already_complete), ull(c.dispatches),
            ull(c.copy_regions));
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      "\"buffer_creates\":%llu,\"buffer_create_bytes\":%llu,"
                      "\"timestamp_resolves\":%llu,\"timestamp_map_failures\":%llu}",
                      ull(c.buffer_creates), ull(c.buffer_create_bytes), ull(c.timestamp_resolves),
                      ull(c.timestamp_map_failures));
        o += buf;
        o += ",\"kernel_mode_scope\":\"legacy name q4_kernel_mode; modes 1..3 select the "
             "optimized kernel of each storage listed available in optimized_kernels, "
             "otherwise reference\"";
        o += ",\"optimized_kernels\":{";
        for (uint32_t st = 0; st < kStorageCount; ++st) {
            std::snprintf(
                buf, sizeof(buf),
                "%s\"%s\":{\"available\":%s,\"status\":\"%s\","
                "\"optimized_calls\":%llu,\"reference_calls\":%llu,\"fast_nq_max\":%u}",
                st ? "," : "", storage_name(st), h.optimized_available[st] ? "true" : "false",
                json_escape(h.optimized_status[st]).c_str(), ull(c.optimized_by_storage[st]),
                ull(c.reference_by_storage[st]), h.optimized_available[st] ? kOptMaxQueries : 0u);
            o += buf;
        }
        o += '}';

        std::map<std::string, std::vector<const Sample*>> buckets;
        for (size_t i = 0; i < stored_; ++i) {
            const size_t index = (next_ + ring_.size() - stored_ + i) % ring_.size();
            const Sample& s = ring_[index];
            buckets[bucket_key(s)].push_back(&s);
        }
        o += ",\"buckets\":{";
        bool first = true;
        for (const auto& bucket : buckets) {
            if (!first) {
                o += ',';
            }
            first = false;
            append_bucket(o, bucket.first, bucket.second, h);
        }
        o += "}}";
        return o;
    }

    Counters counters;

  private:
    static unsigned long long ull(uint64_t v) {
        return static_cast<unsigned long long>(v);
    }

    // values are sorted in place; nearest-rank percentiles.
    static void append_stats(std::string& o, const char* name, std::vector<double>& values) {
        char buf[320];
        if (values.empty()) {
            std::snprintf(buf, sizeof(buf), "\"%s\":{\"n\":0}", name);
            o += buf;
            return;
        }
        std::sort(values.begin(), values.end());
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        auto rank = [&](double p) {
            size_t r = static_cast<size_t>(p * static_cast<double>(values.size()) + 0.999999);
            r = std::max<size_t>(1, std::min(values.size(), r));
            return values[r - 1];
        };
        std::snprintf(buf, sizeof(buf),
                      "\"%s\":{\"n\":%llu,\"mean\":%.3f,\"min\":%.3f,\"p50\":%.3f,\"p90\":%.3f,"
                      "\"p99\":%.3f,\"max\":%.3f,\"sum\":%.3f}",
                      name, static_cast<unsigned long long>(values.size()),
                      sum / static_cast<double>(values.size()), values.front(), rank(0.5),
                      rank(0.9), rank(0.99), values.back(), sum);
        o += buf;
    }

    static void append_bucket(std::string& o, const std::string& key,
                              const std::vector<const Sample*>& samples, const Header& h) {
        const bool attention = !samples.empty() && samples.front()->kind == kAttention;
        o += '"';
        o += json_escape(key);
        o += "\":{";
        char buf[160];
        uint64_t gpu_valid = 0, fence_blocked = 0, alloc_grew = 0, direct = 0;
        for (const Sample* s : samples) {
            gpu_valid += s->gpu_valid;
            fence_blocked += s->fence_blocked;
            alloc_grew += s->alloc_grew;
            direct += s->direct_output;
        }
        std::snprintf(buf, sizeof(buf),
                      "\"samples\":%llu,\"gpu_samples\":%llu,\"fence_blocked\":%llu,"
                      "\"alloc_grew\":%llu,\"direct_output\":%llu,",
                      ull(samples.size()), ull(gpu_valid), ull(fence_blocked), ull(alloc_grew),
                      ull(direct));
        o += buf;
        std::vector<double> v;
        auto collect = [&](auto fn) {
            v.clear();
            for (const Sample* s : samples) {
                v.push_back(fn(*s));
            }
        };
        collect([](const Sample& s) { return double(s.n_tokens); });
        append_stats(o, "n_tokens", v);
        o += ',';
        collect([](const Sample& s) { return double(s.rows); });
        append_stats(o, attention ? "active_rows" : "rows", v);
        o += ',';
        collect([](const Sample& s) { return double(s.bytes); });
        append_stats(o, "bytes", v);
        if (attention) {
            o += ',';
            collect([](const Sample& s) { return double(s.split_k); });
            append_stats(o, "split_k", v);
            o += ',';
            collect([](const Sample& s) { return double(s.groups); });
            append_stats(o, "pass1_groups", v);
        }
        o += ",\"gpu_us\":{";
        if (h.gpu_timestamp_frequency) {
            const double us_per_tick = 1e6 / static_cast<double>(h.gpu_timestamp_frequency);
            static const char* upload_names[] = {"copy"};
            static const char* attn_names[] = {"pass1", "reduce", "readback_copy"};
            const uint32_t stages = attention ? 3 : 1;
            for (uint32_t st = 0; st < stages; ++st) {
                v.clear();
                for (const Sample* s : samples) {
                    if (s->gpu_valid) {
                        v.push_back(double(s->gpu_ticks[st]) * us_per_tick);
                    }
                }
                append_stats(o, attention ? attn_names[st] : upload_names[st], v);
                o += ',';
            }
            v.clear();
            for (const Sample* s : samples) {
                if (s->gpu_valid) {
                    uint64_t total = 0;
                    for (uint32_t st = 0; st < stages; ++st) {
                        total += s->gpu_ticks[st];
                    }
                    v.push_back(double(total) * us_per_tick);
                }
            }
            append_stats(o, "total", v);
        }
        o += "},\"cpu_us\":{";
        static const char* cpu_names[] = {"validate", "alloc",      "map_copy", "record",
                                          "submit",   "fence_wait", "readback"};
        for (uint32_t span = 0; span < kCpuSpanCount; ++span) {
            if (!attention && span == kCpuReadback) {
                continue;
            }
            collect([span](const Sample& s) { return double(s.cpu_ns[span]) / 1000.0; });
            append_stats(o, cpu_names[span], v);
            o += ',';
        }
        collect([](const Sample& s) {
            uint64_t total = 0;
            for (uint32_t span = 0; span < kCpuSpanCount; ++span) {
                total += s.cpu_ns[span];
            }
            return double(total) / 1000.0;
        });
        append_stats(o, "engine_total", v);
        o += "},";
        collect([](const Sample& s) { return double(s.overhead_ns) / 1000.0; });
        append_stats(o, "profile_overhead_us", v);
        o += '}';
    }

    std::vector<Sample> ring_;
    size_t next_ = 0;
    size_t stored_ = 0;
    uint64_t recorded_ = 0;
};

} // namespace rkva_rprof
