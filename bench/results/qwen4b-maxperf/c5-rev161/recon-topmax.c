// Isolated C5 recon: scalar (MSVC-style 4x chain) vs AVX2 block-fold max scan,
// plus the exact exp/early-exit loop, with bit-exact equivalence checks.
// Build (on .193): cl /O2 /arch:AVX2 /Fe:topmax.exe topmax.c
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#include <windows.h>
static double now_s(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
#include <immintrin.h>
#include <time.h>
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

static float max_scalar(const float* a, int n) {
    float m = a[0];
    int i = 1;
    for (; i + 4 <= n; i += 4) {
        if (a[i] > m) m = a[i];
        if (a[i + 1] > m) m = a[i + 1];
        if (a[i + 2] > m) m = a[i + 2];
        if (a[i + 3] > m) m = a[i + 3];
    }
    for (; i < n; ++i)
        if (a[i] > m) m = a[i];
    return m;
}

static float max_avx2_blockfold(const float* a, int n) {
#if defined(__AVX2__)
    float m = a[0];
    int i = 1;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        v = _mm256_blendv_ps(v, _mm256_set1_ps(-INFINITY), _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 r = _mm_blendv_ps(lo, hi, _mm_cmp_ps(hi, lo, _CMP_GT_OQ));
        __m128 t = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 3, 0, 1));
        r = _mm_blendv_ps(r, t, _mm_cmp_ps(t, r, _CMP_GT_OQ));
        t = _mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 0, 3, 2));
        r = _mm_blendv_ps(r, t, _mm_cmp_ps(t, r, _CMP_GT_OQ));
        float bm = _mm_cvtss_f32(r);
        if (bm > m) m = bm;
    }
    for (; i < n; ++i)
        if (a[i] > m) m = a[i];
    return m;
#else
    return max_scalar(a, n);
#endif
}

// Per-lane running max (ordered GT: NaN ignored, ties keep earlier) + lane-order
// fold + exact +/-0 canonicalization. Bit-identical to max_scalar.
static float max_avx2(const float* a, int n) {
#if defined(__AVX2__)
    float m = a[0];
    __m256 vmax = _mm256_set1_ps(a[0]);
    int i = 1;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        __m256 gt = _mm256_cmp_ps(v, vmax, _CMP_GT_OQ);
        vmax = _mm256_blendv_ps(vmax, v, gt);
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, vmax);
    for (int k = 0; k < 8; ++k)
        if (lanes[k] > m) m = lanes[k];
    for (; i < n; ++i)
        if (a[i] > m) m = a[i];
    if (m == 0.0f) {
        for (int j = 0; j < n; ++j)
            if (a[j] == 0.0f) return a[j];
    }
    return m;
#else
    return max_scalar(a, n);
#endif
}

// Exact current top_prob semantics; returns the sum-scan length too.
static float top_prob(const float* logits, int n, float p_min, int* scanned, float (*mx)(const float*, int)) {
    float m = mx(logits, n);
    const double limit = 1.0 / (double)p_min;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += exp((double)(logits[i] - m));
        if (sum > limit) {
            if (scanned) *scanned = i + 1;
            return 0.0f;
        }
    }
    if (scanned) *scanned = n;
    return (float)(1.0 / sum);
}

static uint32_t bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static int equiv_case(const float* a, int n, float pmin) {
    float ms = max_scalar(a, n), ma = max_avx2(a, n), mb = max_avx2_blockfold(a, n);
    if (bits(ms) != bits(mb)) { printf("BLOCKFOLD_BITS_DIFF n=%d scalar=%08x bf=%08x\n", n, bits(ms), bits(mb)); return 1; }
    int ss, sa;
    float ps = top_prob(a, n, pmin, &ss, max_scalar);
    float pa = top_prob(a, n, pmin, &sa, max_avx2);
    if (bits(ms) != bits(ma)) {
        printf("MAX_BITS_DIFF n=%d scalar=%08x avx2=%08x\n", n, bits(ms), bits(ma));
        return 1;
    }
    if (bits(ps) != bits(pa) || ss != sa) {
        printf("TOPPROB_DIFF n=%d ps=%08x pa=%08x ss=%d sa=%d\n", n, bits(ps), bits(pa), ss, sa);
        return 1;
    }
    return 0;
}

int main(void) {
    int fails = 0;
    // 1. Adversarial: NaN/+-0/+-inf/ties at every position, tails n=1..33.
    for (int n = 1; n <= 33; ++n) {
        float a[33];
        for (int t = 0; t < 200; ++t) {
            for (int i = 0; i < n; ++i) a[i] = (float)((t * 7 + i * 13) % 5) - 2.0f; // ties
            int p = (t * 3) % n;
            switch ((t / 3) % 5) {
            case 1: a[p] = NAN; break;
            case 2: a[p] = (t % 2) ? -0.0f : 0.0f; break;
            case 3: a[p] = (t % 2) ? INFINITY : -INFINITY; break;
            case 4: a[0] = NAN; break;
            default: break;
            }
            fails += equiv_case(a, n, 50.0f);
            fails += equiv_case(a, n, 0.25f);
        }
    }
    printf("adversarial_fails=%d\n", fails);

    // 2. Random arrays with injected specials, n=248320.
    float* big = (float*)malloc(sizeof(float) * 248320);
    srand(12345);
    for (int rep = 0; rep < 20; ++rep) {
        for (int i = 0; i < 248320; ++i) big[i] = (float)((rand() / (double)RAND_MAX) * 6.0 - 3.0);
        big[rand() % 248320] = NAN;
        big[rand() % 248320] = -0.0f;
        big[rand() % 248320] = INFINITY;
        fails += equiv_case(big, 248320, 50.0f);
    }
    printf("random_fails=%d\n", fails);

    // 3. Timing: scan only, then full top_prob, over realistic-ish logits.
    const int REPS = 400;
    double t0, t1;
    // peaked distribution (max 8) and flatter (std 2.5)
    for (int mode = 0; mode < 2; ++mode) {
        for (int i = 0; i < 248320; ++i) {
            double g = 0;
            for (int k = 0; k < 4; ++k) g += rand() / (double)RAND_MAX; // ~N(0,1)-ish
            g = (g - 2.0) * (mode ? 2.5 : 1.0);
            big[i] = (float)g;
        }
        big[rand() % 248320] = mode ? 2.0f : 8.0f;
        volatile float sink = 0;
        t0 = now_s();
        for (int r = 0; r < REPS; ++r) { big[r % 248320] = big[r % 248320] + 1e-9f; sink += max_scalar(big, 248320); }
        t1 = now_s();
        double scan_scalar_us = (t1 - t0) / REPS * 1e6;
        t0 = now_s();
        for (int r = 0; r < REPS; ++r) { big[r % 248320] = big[r % 248320] + 1e-9f; sink += max_avx2(big, 248320); }
        t1 = now_s();
        double scan_avx_us = (t1 - t0) / REPS * 1e6;
        t0 = now_s();
        for (int r = 0; r < REPS; ++r) { big[r % 248320] = big[r % 248320] + 1e-9f; sink += max_avx2_blockfold(big, 248320); }
        t1 = now_s();
        double scan_bf_us = (t1 - t0) / REPS * 1e6;
        int scanned = 0;
        t0 = now_s();
        float p = 0;
        long long total_scan = 0;
        for (int r = 0; r < REPS; ++r) {
            big[r % 248320] = big[r % 248320] + 1e-9f;
            p += top_prob(big, 248320, 50.0f, &scanned, max_scalar);
            total_scan += scanned;
        }
        t1 = now_s();
        double tp_scalar_us = (t1 - t0) / REPS * 1e6;
        t0 = now_s();
        float pa = 0;
        for (int r = 0; r < REPS; ++r) { big[r % 248320] = big[r % 248320] + 1e-9f; pa += top_prob(big, 248320, 50.0f, &scanned, max_avx2); }
        t1 = now_s();
        double tp_avx_us = (t1 - t0) / REPS * 1e6;
        printf("mode=%d scan_scalar=%.1fus scan_avx2=%.1fus scan_blockfold=%.1fus speedup=%.2fx avg_sum_scan=%lld "
               "topprob_scalar=%.1fus topprob_avx2=%.1fus saved=%.1fus max_frac=%.1f%% sum_frac=%.1f%%\n",
               mode, scan_scalar_us, scan_avx_us, scan_bf_us, scan_scalar_us / scan_avx_us,
               total_scan / REPS, tp_scalar_us, tp_avx_us, tp_scalar_us - tp_avx_us,
               100.0 * scan_scalar_us / tp_scalar_us, 100.0 * (tp_scalar_us - scan_scalar_us) / tp_scalar_us);
        (void)p;
        (void)pa;
    }
    free(big);
    printf("done fails=%d\n", fails);
    return fails ? 1 : 0;
}
