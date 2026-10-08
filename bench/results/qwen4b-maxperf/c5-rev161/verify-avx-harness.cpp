// Plan 006 C5 AVX harness (runs on .193/AVX1 and on any AVX host): compiles
// the real src/bridge/mtp_top_prob.cpp and checks mtp_logits_max_avx /
// mtp_top_prob_core(avx=true) bit-for-bit against an independent scalar
// reference over adversarial patterns, tails and p_min threshold boundaries.
// Build: cl /nologo /O2 /std:c++17 /EHsc /arch:AVX /I <repo>\include ^
//          /Fe:verify-avx.exe <repo>\src\bridge\mtp_top_prob.cpp verify-avx-harness.cpp
#include <cmath>
#if defined(_MSC_VER)
#include <windows.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "xllama/mtp_top_prob.h"

using namespace xllama;

static int g_fails = 0;
static long long g_checks = 0;

static uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

static float ref_max(const float* a, int n) {
    float m = a[0];
    for (int i = 1; i < n; ++i)
        if (a[i] > m)
            m = a[i];
    return m;
}

static float ref_core(const float* a, int n, float p_min) {
    const float m = ref_max(a, n);
    const double limit = 1.0 / static_cast<double>(p_min);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::exp(static_cast<double>(a[i] - m));
        if (sum > limit)
            return 0.0f;
    }
    return static_cast<float>(1.0 / sum);
}

static void check_max(const std::vector<float>& a, const char* what) {
    ++g_checks;
    const float s = mtp_logits_max_scalar(a.data(), (int)a.size());
    const float v = mtp_logits_max_avx(a.data(), (int)a.size());
    const float r = ref_max(a.data(), (int)a.size());
    if (bits(s) != bits(r) || bits(v) != bits(s)) {
        ++g_fails;
        std::printf("MAX_FAIL %s n=%zu scalar=%08x avx=%08x ref=%08x\n", what, a.size(), bits(s),
                    bits(v), bits(r));
    }
}

static void check_core(const std::vector<float>& a, float p_min, const char* what) {
    ++g_checks;
    const int n = (int)a.size();
    const float ref = ref_core(a.data(), n, p_min);
    const float sc = mtp_top_prob_core(a.data(), n, p_min, false);
    const float av = mtp_top_prob_core(a.data(), n, p_min, true);
    if (bits(sc) != bits(ref) || bits(av) != bits(ref)) {
        ++g_fails;
        std::printf("CORE_FAIL %s n=%d pmin=%g scalar=%08x avx=%08x ref=%08x\n", what, n, p_min,
                    bits(sc), bits(av), bits(ref));
    }
}

int main() {
    // 1. Adversarial patterns, every tail length 1..33.
    for (int n = 1; n <= 33; ++n) {
        for (int t = 0; t < 64; ++t) {
            std::vector<float> a(n);
            for (int i = 0; i < n; ++i)
                a[i] = (float)((t * 7 + i * 13) % 5) - 2.0f;
            const int p = (t * 3) % n;
            switch ((t / 3) % 6) {
            case 1: a[p] = NAN; break;
            case 2: a[p] = (t % 2) ? -0.0f : 0.0f; break;
            case 3: a[p] = (t % 2) ? INFINITY : -INFINITY; break;
            case 4: a[0] = NAN; break;
            case 5: a[n - 1] = NAN; break;
            default: break;
            }
            check_max(a, "pattern");
            check_core(a, 50.0f, "pattern");
        }
    }
    // 2. Randomized arrays with specials.
    uint64_t s = 0x9E3779B97F4A7C15ull;
    auto rnd = [&s]() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t)(s >> 33);
    };
    for (int rep = 0; rep < 400; ++rep) {
        const int n = 1 + (int)(rnd() % 300);
        std::vector<float> a(n);
        for (int i = 0; i < n; ++i)
            a[i] = (float)((int)(rnd() % 2001) - 1000) * 0.001f;
        for (int k = 0; k < 3; ++k) {
            const int p = (int)(rnd() % n);
            switch (rnd() % 4) {
            case 0: a[p] = NAN; break;
            case 1: a[p] = -0.0f; break;
            case 2: a[p] = INFINITY; break;
            default: a[p] = -INFINITY; break;
            }
        }
        check_max(a, "random");
    }
    {
        std::vector<float> big(248320);
        for (size_t i = 0; i < big.size(); ++i)
            big[i] = (float)((int)(rnd() % 2001) - 1000) * 0.01f;
        big[0] = -0.0f;
        big[1] = NAN;
        big[12345] = INFINITY;
        big[big.size() - 1] = 0.0f;
        check_max(big, "big");
    }
    // 3. p_min threshold boundaries.
    {
        std::vector<float> a(16, 0.0f);
        check_core(a, 0.5f, "all-equal");
        check_core(a, 2.0f, "all-equal");
        check_core(a, 50.0f, "all-equal");
        check_core(a, 1.0f, "all-equal");
        std::vector<float> b = {0.0f, 0.0f, -INFINITY, -INFINITY, -INFINITY};
        check_core(b, 0.5f, "exact-limit");
        std::vector<float> c = {0.0f, 0.0f, 0.0f, -INFINITY, -INFINITY};
        check_core(c, 0.5f, "over-limit");
        std::vector<float> d = {0.0f, -0.6931472f, -1.3862944f, -2.0794415f,
                                -2.7725887f, -3.465736f};
        for (float pm : {0.5f, 0.7f, 0.8f, 1.0f, 1.3f, 1.5f, 50.0f})
            check_core(d, pm, "half-scale");
        std::vector<float> e(12, -INFINITY);
        check_core(e, 50.0f, "all-neg-inf");
        std::vector<float> f(12, INFINITY);
        check_core(f, 50.0f, "all-pos-inf");
        std::vector<float> g(12, -INFINITY);
        g[7] = 0.0f;
        check_core(g, 50.0f, "one-zero");
        check_core(g, 0.5f, "one-zero");
        std::vector<float> h(12, 1.0f);
        h[0] = NAN;
        check_core(h, 50.0f, "nan-seed");
    }
    // 4. Signed-zero ties at every position.
    for (int pos = 0; pos < 9; ++pos) {
        std::vector<float> a(9, -1.0f);
        a[pos] = -0.0f;
        a[(pos + 3) % 9] = 0.0f;
        check_max(a, "zero-tie");
        check_core(a, 50.0f, "zero-tie");
    }
    // 5. Signed-zero maxima crossing lane order (first zero later in lane0
    // than a zero in lane1) plus a scalar-tail zero.
    {
        std::vector<float> a(17, -1.0f);
        a[2] = 0.0f;
        a[9] = -0.0f;
        check_max(a, "lane-order");
        check_core(a, 50.0f, "lane-order");
        std::vector<float> b(17, -1.0f);
        b[2] = -0.0f;
        b[9] = 0.0f;
        check_max(b, "lane-order");
        check_core(b, 50.0f, "lane-order");
        std::vector<float> c(33, -1.0f);
        c[0] = -0.0f;
        c[8] = 0.0f;
        c[16] = -0.0f;
        c[32] = 0.0f;
        check_max(c, "lane-order-tail");
        check_core(c, 50.0f, "lane-order-tail");
    }
    // 6. nextafter p_min boundaries around the returned probability, n > 8.
    for (int n : {9, 10, 16, 17, 33, 64}) {
        std::vector<float> a(n, -INFINITY);
        for (int i = 0; i < 9; ++i)
            a[i] = 0.0f;
        const float prob = (float)(1.0 / 9);
        const float cands[5] = {
            std::nextafter(std::nextafter(prob, -INFINITY), -INFINITY),
            std::nextafter(prob, -INFINITY),
            prob,
            std::nextafter(prob, INFINITY),
            std::nextafter(std::nextafter(prob, INFINITY), INFINITY),
        };
        int exits = 0, completes = 0;
        for (float pm : cands) {
            check_core(a, pm, "nextafter-exact");
            const float r = mtp_top_prob_core(a.data(), n, pm, true);
            (r == 0.0f ? exits : completes)++;
        }
        if (!(exits > 0 && completes > 0)) {
            ++g_fails;
            std::printf("BOUNDARY_NOT_CROSSED n=%d exits=%d completes=%d\n", n, exits, completes);
        }
    }
    std::printf("c5_avx_checks=%lld fails=%d\n", g_checks, g_fails);

    // 7. Isolated timings on this host (AVX1/AVX2 depending on /arch).
    {
        const int V = 248320;
        std::vector<float> big(V);
        for (int i = 0; i < V; ++i)
            big[i] = (float)((int)(rnd() % 2001) - 1000) * 0.01f;
        big[rnd() % V] = 8.0f;
        volatile float sink = 0;
        const int REPS = 200;
        LARGE_INTEGER f, t0, t1;
        QueryPerformanceFrequency(&f);
        auto now = [&]() {
            LARGE_INTEGER t;
            QueryPerformanceCounter(&t);
            return (double)t.QuadPart / (double)f.QuadPart;
        };
        double a0 = now();
        for (int r = 0; r < REPS; ++r) {
            big[r % V] += 1e-9f;
            sink += mtp_logits_max_scalar(big.data(), V);
        }
        double a1 = now();
        for (int r = 0; r < REPS; ++r) {
            big[r % V] += 1e-9f;
            sink += mtp_logits_max_avx(big.data(), V);
        }
        double a2 = now();
        int scanned = 0;
        for (int r = 0; r < REPS; ++r) {
            big[r % V] += 1e-9f;
            sink += mtp_top_prob_core(big.data(), V, 50.0f, false);
        }
        double a3 = now();
        for (int r = 0; r < REPS; ++r) {
            big[r % V] += 1e-9f;
            sink += mtp_top_prob_core(big.data(), V, 50.0f, true);
        }
        double a4 = now();
        (void)scanned;
        const double scan_s = (a1 - a0) / REPS * 1e6;
        const double scan_a = (a2 - a1) / REPS * 1e6;
        const double core_s = (a3 - a2) / REPS * 1e6;
        const double core_a = (a4 - a3) / REPS * 1e6;
        std::printf("timing_us scan_scalar=%.1f scan_avx=%.1f speedup=%.2fx core_scalar=%.1f "
                    "core_avx=%.1f speedup=%.2fx\n",
                    scan_s, scan_a, scan_s / scan_a, core_s, core_a, core_s / core_a);
    }
    return g_fails ? 1 : 0;
}
