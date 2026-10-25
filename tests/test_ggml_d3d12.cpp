// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/ggml_d3d12.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace xllama;

namespace {

D3d12MatmulDesc decode_desc(ggml_type t, std::int64_t n, std::int64_t k) {
    D3d12MatmulDesc d;
    d.src0_type = t;
    d.ne00 = k;
    d.ne01 = n;
    d.ne10 = k;
    d.ne11 = 1;
    d.src0_in_weight_buffer = true;
    return d;
}

// Max |emulated - reference| / max |reference| for one type and shape, with
// padded activation/output strides. Reference: ggml's own dequantizer, double sums.
double emulate_rel_err(ggml_type t, int n, int k, int ncols, std::size_t x_pad, std::size_t y_pad) {
    std::mt19937 rng(42u + static_cast<unsigned>(n * 7 + k + ncols));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(n) * k);
    for (float& v : wf)
        v = uni(rng);
    const std::size_t row_bytes = ggml_row_size(t, k);
    std::vector<std::uint8_t> q(row_bytes * n + 8); // slack: 2-byte-aligned dword reads
    ggml_quantize_chunk(t, wf.data(), q.data(), 0, n, k, nullptr);

    const std::size_t x_stride = static_cast<std::size_t>(k) + x_pad;
    const std::size_t y_stride = static_cast<std::size_t>(n) + y_pad;
    std::vector<float> x(x_stride * ncols, 1e30f); // padding must never be read
    for (int c = 0; c < ncols; ++c)
        for (int i = 0; i < k; ++i)
            x[c * x_stride + i] = uni(rng);
    std::vector<float> y(y_stride * ncols, -7.f);
    d3d12_mmv_emulate(t, q.data(), row_bytes, x.data(), x_stride, y.data(), y_stride, n, k, ncols);

    const auto* traits = ggml_get_type_traits(t);
    std::vector<float> wrow(static_cast<std::size_t>(k));
    double max_ref = 0.0, max_diff = 0.0;
    for (int r = 0; r < n; ++r) {
        traits->to_float(q.data() + r * row_bytes, wrow.data(), k);
        for (int c = 0; c < ncols; ++c) {
            double acc = 0.0;
            for (int i = 0; i < k; ++i)
                acc += static_cast<double>(wrow[i]) * x[c * x_stride + i];
            max_ref = std::max(max_ref, std::fabs(acc));
            max_diff = std::max(max_diff, std::fabs(acc - y[c * y_stride + r]));
        }
    }
    // Output padding untouched.
    for (int c = 0; c < ncols; ++c)
        for (std::size_t p = n; p < y_stride; ++p)
            if (y[c * y_stride + p] != -7.f)
                return 1e9;
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

} // namespace

TEST_CASE("ggml_d3d12: weight types") {
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q4_0));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q4_K));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q5_K));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q6_K));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_Q8_0));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_F16));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_F32));
}

TEST_CASE("ggml_d3d12: supports_op rules for MUL_MAT") {
    D3d12MatmulDesc d = decode_desc(GGML_TYPE_Q4_K, 2048, 2048);
    CHECK(d3d12_mm_supported(d));

    // The buft probe: a weight still in D3D12_Host is refused, so llama.cpp
    // moves on to D3D12_Weights.
    D3d12MatmulDesc host = d;
    host.src0_in_weight_buffer = false;
    CHECK_FALSE(d3d12_mm_supported(host));

    D3d12MatmulDesc prefill = d; // the loader probes with 512 columns
    prefill.ne11 = 512;
    CHECK(d3d12_mm_supported(prefill));

    D3d12MatmulDesc t = d;
    t.src0_type = GGML_TYPE_Q8_0;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.src1_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.dst_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne00 = t.ne10 = 2080; // not a multiple of 256
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne10 = 1024; // K mismatch
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne12 = 4; // batched attention-style matmul
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne02 = 2; // 3-D weight
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.src1_contiguous = false;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.src0_contiguous = false;
    CHECK_FALSE(d3d12_mm_supported(t));

    // Coder-3B tied lm_head: 151936 rows → 37984 groups, inside the limit.
    CHECK(d3d12_mm_supported(decode_desc(GGML_TYPE_Q6_K, 151936, 2048)));
    CHECK_FALSE(d3d12_mm_supported(decode_desc(GGML_TYPE_Q6_K, 4 * 65536, 2048)));
}

TEST_CASE("ggml_d3d12: dispatch planner") {
    D3d12Dispatch d = d3d12_mm_dispatch(2048, 1);
    CHECK(d.ok);
    CHECK(d.groups_x == 512);
    CHECK(d.groups_y == 1);
    d = d3d12_mm_dispatch(6, 512);
    CHECK(d.groups_x == 2);
    CHECK(d.groups_y == 512);
    CHECK(d3d12_mm_dispatch(4 * 65535, 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch(4 * 65535 + 1, 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch(256, 65536).ok);
    CHECK_FALSE(d3d12_mm_dispatch(0, 1).ok);
}

TEST_CASE("ggml_d3d12: two-column dispatch planner") {
    // Same row groups as the base planner, half (ceil) the column groups.
    D3d12Dispatch d = d3d12_mm_dispatch_2col(2048, 1);
    CHECK(d.ok);
    CHECK(d.groups_x == 512);
    CHECK(d.groups_y == 1);
    d = d3d12_mm_dispatch_2col(9216, 5);
    CHECK(d.ok);
    CHECK(d.groups_x == 2304);
    CHECK(d.groups_y == 3); // tail group covers column 4 alone
    d = d3d12_mm_dispatch_2col(1023, 2);
    CHECK(d.ok);
    CHECK(d.groups_x == 256); // ceil(1023/4): the row-clamp group exists
    CHECK(d.groups_y == 1);
    CHECK(d3d12_mm_dispatch_2col(6, 512).groups_y == 256);
    CHECK(d3d12_mm_dispatch_2col(4 * 65535, 2).ok);
    CHECK_FALSE(d3d12_mm_dispatch_2col(4 * 65535 + 1, 2).ok);
    CHECK_FALSE(d3d12_mm_dispatch_2col(256, 2 * 65535 + 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch_2col(0, 2).ok);
    CHECK_FALSE(d3d12_mm_dispatch_2col(256, 0).ok);
}

TEST_CASE("ggml_d3d12: padded layouts match a strided double reference") {
    // Mirrors run_paired_case's padded setup: flat generation, strided memcpy
    // over a sentinel fill, sentinel-padded X, then the shader-math emulation
    // against a double reference that pitches rows at nb[1] (NOT the packed
    // size — passing the packed size misaligns every row past the first, the
    // exact rev58 bench bug this test pins).
    const ggml_type t = GGML_TYPE_Q4_K;
    const int n = 35, k = 512, ncols = 3;
    const int x_pad = 13, y_pad = 7;
    const std::size_t w_pad = 64;
    const std::size_t row_bytes = ggml_row_size(t, k);
    const std::size_t w_nb1 = row_bytes + w_pad;
    const std::size_t xs = static_cast<std::size_t>(k) + x_pad;
    const std::size_t ys = static_cast<std::size_t>(n) + y_pad;
    std::mt19937 rng(1234u + static_cast<unsigned>(n + k));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<std::uint8_t> qflat(static_cast<std::size_t>(n) * row_bytes);
    REQUIRE(d3d12_gen_weights(t, n, k, rng, qflat.data()));
    // nbytes-style sizing (like the device buffers): no storage past the last
    // valid row/column, so padding checks skip the final row/column exactly as
    // check_float_pad/check_byte_pad do on device readback.
    std::vector<std::uint8_t> q(row_bytes + static_cast<std::size_t>(n - 1) * w_nb1, 0xAB);
    for (int r = 0; r < n; ++r)
        std::memcpy(q.data() + static_cast<std::size_t>(r) * w_nb1,
                    qflat.data() + static_cast<std::size_t>(r) * row_bytes, row_bytes);
    std::vector<float> xf(static_cast<std::size_t>(k) + static_cast<std::size_t>(ncols - 1) * xs,
                          1e30f);
    for (int c = 0; c < ncols; ++c)
        for (int i = 0; i < k; ++i)
            xf[static_cast<std::size_t>(c) * xs + i] = uni(rng);
    std::vector<float> y(static_cast<std::size_t>(n) + static_cast<std::size_t>(ncols - 1) * ys,
                         -7.f);
    d3d12_mmv_emulate(t, q.data(), w_nb1, xf.data(), xs, y.data(), ys, n, k, ncols);

    const auto* traits = ggml_get_type_traits(t);
    std::vector<float> wrow(k);
    double max_ref = 0.0, max_diff = 0.0;
    for (int r = 0; r < n; ++r) {
        traits->to_float(q.data() + static_cast<std::size_t>(r) * w_nb1, wrow.data(), k);
        for (int c = 0; c < ncols; ++c) {
            double acc = 0.0;
            for (int i = 0; i < k; ++i)
                acc += static_cast<double>(wrow[i]) * xf[static_cast<std::size_t>(c) * xs + i];
            max_ref = std::max(max_ref, std::fabs(acc));
            max_diff = std::max(max_diff, std::fabs(acc - y[static_cast<std::size_t>(c) * ys + r]));
        }
    }
    CHECK(max_ref > 0.0);
    CHECK(max_diff / max_ref <= 1e-4);
    // Padding untouched by construction here; the device run proves the kernel.
    // Final row/column carry no padding storage (nbytes sizing), mirrored skips.
    for (int c = 0; c + 1 < ncols; ++c) {
        for (std::size_t i = k; i < xs; ++i)
            CHECK(xf[static_cast<std::size_t>(c) * xs + i] == 1e30f);
        for (std::size_t i = n; i < ys; ++i)
            CHECK(y[static_cast<std::size_t>(c) * ys + i] == -7.f);
    }
    for (int r = 0; r + 1 < n; ++r)
        for (std::size_t i = row_bytes; i < w_nb1; ++i)
            CHECK(q[static_cast<std::size_t>(r) * w_nb1 + i] == 0xAB);
}

TEST_CASE("ggml_d3d12: two-column allowlist is explicit, not a rule") {
    // Rev57 same-run pair winners: five q4_k shapes at B=2/3/5.
    for (int b : {2, 3, 5}) {
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 2560, 4096, b));
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 4096, 2560, b));
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 8192, 2560, b));
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 2560, 9216, b));
        CHECK(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 9216, 2560, b));
    }
    // Everything else stays OLD: B=1 (nothing to tile), B=4 and prefill
    // widths (unmeasured), small shapes (measured regressions), other types.
    for (int b : {1, 4, 7, 512}) {
        CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 9216, 2560, b));
        CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 2560, 4096, b));
    }
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 1024, 2560, 2));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 1023, 2560, 2));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 1, 256, 2));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q5_K, 8192, 2560, 5));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q6_K, 2560, 9216, 5));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_0, 1024, 1024, 2));
    CHECK_FALSE(d3d12_2col_allowlisted(GGML_TYPE_Q4_K, 0, 2560, 2));
}

TEST_CASE("ggml_d3d12: kernel width follows K (D2a runs 1 and 2)") {
    CHECK(d3d12_mm_threads(1024) == 64); // lm_head of LFM2.5-350M
    CHECK(d3d12_mm_threads(2048) == 64); // most projections
    CHECK(d3d12_mm_threads(3840) == 64);
    CHECK(d3d12_mm_threads(4096) == 128);  // first width with 16 chunks
    CHECK(d3d12_mm_threads(8192) == 128);  // LFM2.5-1.2B ffn_down
    CHECK(d3d12_mm_threads(11008) == 128); // Coder-3B ffn_down
}

TEST_CASE("ggml_d3d12: kernel emulation matches ggml dequantizers") {
    struct Case {
        ggml_type t;
        int n, k;
    };
    const Case cases[] = {
        {GGML_TYPE_Q4_0, 13, 1024},
        {GGML_TYPE_Q4_K, 13, 1024},
        {GGML_TYPE_Q5_K, 13, 1024},
        {GGML_TYPE_Q6_K, 13, 1024},
        // 9 super-blocks: 1890-byte rows, odd rows start 2-byte aligned.
        {GGML_TYPE_Q6_K, 6, 2304},
        // Coder-3B ffn_down width: 43 super-blocks, 9030-byte rows.
        {GGML_TYPE_Q6_K, 3, 11008},
        {GGML_TYPE_Q4_0, 5, 2304},
        // Qwen3.5-4B attention and SSM projections: 2560 and 4096 wide.
        {GGML_TYPE_Q5_K, 5, 2560},
        {GGML_TYPE_Q5_K, 3, 4096},
        {GGML_TYPE_Q5_K, 11, 5120},
        // Two-column tile inputs: odd row counts with several columns and pads.
        {GGML_TYPE_Q4_K, 13, 2560},
        {GGML_TYPE_Q4_K, 1023, 2560},
    };
    for (const auto& c : cases) {
        CAPTURE(ggml_type_name(c.t));
        CAPTURE(c.k);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 1, 0, 0) <= 1e-4);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 7, 12, 3) <= 1e-4);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 3, 5, 7) <= 1e-4);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 5, 1, 9) <= 1e-4);
    }
    CHECK(ggml_row_size(GGML_TYPE_Q6_K, 11008) % 4 == 2); // the case the ld32 trick covers
    // Q5_K blocks are 176 B, so every dword the shader reads stays inside a block
    // and its loads need no alignment trick.
    CHECK(ggml_row_size(GGML_TYPE_Q5_K, 2560) % 4 == 0);
}

TEST_CASE("ggml_d3d12: selftest CSV and non-Windows behaviour") {
    D3d12SelftestRow r;
    r.type = "q6_k";
    r.n = 2048;
    r.k = 11008;
    r.ncols = 1;
    r.error = "a,b";
    const std::string line = format_d3d12_selftest_row(r, "host");
    int commas = 0, hcommas = 0;
    for (char ch : line)
        commas += ch == ',' ? 1 : 0;
    for (const char* p = d3d12_selftest_csv_header(); *p; ++p)
        hcommas += *p == ',' ? 1 : 0;
    CHECK(commas == hcommas);
#if !defined(_WIN32)
    CHECK_FALSE(ggml_d3d12_register());
    std::vector<D3d12SelftestRow> rows;
    run_d3d12_selftest(&rows);
    REQUIRE(rows.size() == 1);
    CHECK_FALSE(rows[0].d3d12_ran);
    CHECK_FALSE(rows[0].error.empty());
#endif
}

TEST_CASE("ggml_d3d12: bounded weight generation is byte-identical to one-shot") {
    const struct {
        ggml_type t;
        int n, k;
    } cases[] = {
        {GGML_TYPE_Q4_0, 64, 1024},   // small: the Q4_0 32-element block path
        {GGML_TYPE_Q4_0, 5000, 1024}, // two 16 MiB chunks
        {GGML_TYPE_Q4_K, 8, 2560},    // one chunk
        {GGML_TYPE_Q4_K, 4096, 2560}, // three chunks
        {GGML_TYPE_Q5_K, 3000, 4096}, // three chunks
        {GGML_TYPE_Q6_K, 4096, 2560}, // three chunks
    };
    for (const auto& c : cases) {
        const std::size_t bytes = ggml_row_size(c.t, c.k) * static_cast<std::size_t>(c.n);
        std::vector<std::uint8_t> got(bytes), want(bytes);
        // The run_case seed encoding: shape only, never ncols.
        std::mt19937 a(1234u + static_cast<unsigned>(c.n + c.k));
        std::mt19937 b(1234u + static_cast<unsigned>(c.n + c.k));
        REQUIRE(d3d12_gen_weights_reference(c.t, c.n, c.k, a, want.data()));
        REQUIRE(d3d12_gen_weights(c.t, c.n, c.k, b, got.data()));
        CHECK(got == want);
        // The stream continues identically after the weights: run_case draws
        // the activations next, and both paths must hand it the same numbers.
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        for (int i = 0; i < 64; ++i)
            CHECK(uni(a) == uni(b));
    }
}

TEST_CASE("ggml_d3d12: bounded weight generation rejects bad inputs") {
    std::mt19937 rng(7);
    std::vector<std::uint8_t> buf(64 * 1024, 0xA5);
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_Q4_K, 8, 100, rng, buf.data())); // k % 256 != 0
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_Q4_0, 8, 48, rng, buf.data()));  // k % 32 != 0
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_F32, 8, 1024, rng, buf.data())); // not a weight type
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_Q4_K, 0, 2560, rng, buf.data()));
    CHECK_FALSE(d3d12_gen_weights(GGML_TYPE_Q4_K, 8, 2560, rng, nullptr));
    CHECK(std::all_of(buf.begin(), buf.end(), [](std::uint8_t v) { return v == 0xA5; }));
}

TEST_CASE("ggml_d3d12: shape-cost CSV keeps the D2a schema untouched") {
    CHECK(d3d12_block_median({}) == 0.0);
    CHECK(d3d12_block_range({}) == 0.0);
    CHECK(d3d12_block_median({2.0}) == doctest::Approx(2.0));
    CHECK(d3d12_block_median({3.0, 1.0, 2.0}) == doctest::Approx(2.0));
    CHECK(d3d12_block_median({5.0, 1.0, 4.0, 2.0, 3.0}) == doctest::Approx(3.0));
    CHECK(d3d12_block_median({4.0, 1.0, 3.0, 2.0}) == doctest::Approx(2.0)); // lower middle
    CHECK(d3d12_block_range({5.0, 1.0, 4.0, 2.0, 3.0}) == doctest::Approx(4.0));
    CHECK(std::string(d3d12_shapecost_csv_header()).find("gpu_ms_range") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find("wall_ms_range") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find(",variant,") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find(",pair,") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find(",twocol,") != std::string::npos);
    CHECK(std::string(d3d12_shapecost_csv_header()).find(",pads,") != std::string::npos);
    D3d12SelftestRow r;
    r.type = "q6_k";
    r.n = 248320;
    r.k = 2560;
    r.ncols = 5;
    r.wall_ms = 12.5;
    r.weight_mb = 521.472;
    r.peak_ws_mb = 600.0;
    r.error = "x,y";
    // The new fields ride along in the diagnostic row...
    const std::string sc = format_d3d12_shapecost_row(r, "host");
    CHECK(sc.find("248320,2560,5") != std::string::npos);
    CHECK(sc.find("521.472") != std::string::npos);
    int commas = 0, hcommas = 0;
    for (char ch : sc)
        commas += ch == ',' ? 1 : 0;
    for (const char* p = d3d12_shapecost_csv_header(); *p; ++p)
        hcommas += *p == ',' ? 1 : 0;
    CHECK(commas == hcommas);
    // ...and never leak into the gated D2a formatter.
    const std::string be = format_d3d12_selftest_row(r, "host");
    CHECK(be.find("wall_ms") == std::string::npos);
    CHECK(be.find("521.472") == std::string::npos);
    CHECK(be.find("600.0") == std::string::npos);
}

// The one place llama params get the GPU-layer request (src/bridge/llama_gpu.h).
#include "../src/bridge/llama_gpu.h"

TEST_CASE("ggml_d3d12: GPU-layer request maps to llama params") {
    llama_model_params mp = llama_model_default_params();
    CHECK(apply_gguf_gpu_layers(0, mp) == 0);
    CHECK(mp.n_gpu_layers == 0);
    CHECK_FALSE(mp.no_host); // a CPU load keeps llama's default weight bufts
    REQUIRE(mp.devices != nullptr);
    CHECK(mp.devices[0] == nullptr); // explicit empty device list: CPU only

    llama_context_params cp = llama_context_default_params();
    const bool kqv_default = cp.offload_kqv;
    apply_gguf_gpu_context(0, cp);
    CHECK(cp.offload_kqv == kqv_default);
    apply_gguf_gpu_context(28, cp);
    CHECK_FALSE(cp.offload_kqv); // KV and attention stay on the CPU

#if !defined(_WIN32)
    // No D3D12 on Linux: a request falls back to the CPU load unchanged.
    mp = llama_model_default_params();
    CHECK(apply_gguf_gpu_layers(99, mp) == 0);
    CHECK(mp.n_gpu_layers == 0);
    CHECK(mp.devices[0] == nullptr);
#endif
}

TEST_CASE("ggml_d3d12: z-0 weight identity is exact and narrow") {
    // Exact gate weight only.
    CHECK(d3d12_is_z0_weight("blk.0.attn_gate.weight", 2560, 4096, true));
    // Wrong name (other layers, other tensors, empty, null).
    CHECK_FALSE(d3d12_is_z0_weight("blk.1.attn_gate.weight", 2560, 4096, true));
    CHECK_FALSE(d3d12_is_z0_weight("blk.0.attn_gate.weight ", 2560, 4096, true));
    CHECK_FALSE(d3d12_is_z0_weight("", 2560, 4096, true));
    CHECK_FALSE(d3d12_is_z0_weight(nullptr, 2560, 4096, true));
    CHECK_FALSE(d3d12_is_z0_weight("blk.0.wqkv.weight", 2560, 4096, true));
    // Wrong shape or quant: never broaden.
    CHECK_FALSE(d3d12_is_z0_weight("blk.0.attn_gate.weight", 4096, 2560, true));
    CHECK_FALSE(d3d12_is_z0_weight("blk.0.attn_gate.weight", 2560, 4096, false));
}
