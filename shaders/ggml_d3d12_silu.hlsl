// ggml_d3d12_silu.hlsl — exact-port probe kernel for the D3D12 selftest
// (plan 006 C4 feasibility): silu(x) = x / (1 + exp_avx2(-x)) using the same
// AVX2 ggml_v_expf polynomial (vec.h:1215+) evaluated with explicit float
// mad() chains (dxc -Gis emits dx.op.tertiary FMad + precise fdiv). Not wired
// into any product graph: the selftest dispatches it directly and compares
// per element against the CPU kernel.
//
// Root: u0 = output (root param 2), u1 = input (root param 3). No root
// constants: the host predicate requires nelements % 256 == 0 and the dispatch
// covers exactly one thread per element, so no bounds guard is needed.

RWStructuredBuffer<float> xs : register(u1);
RWStructuredBuffer<float> ys : register(u0);

static const float C_r = asfloat(0x4b400000u); // 0x1.8p23f
static const float C_lo = 0x1.715476p+0f;
static const float C_c1 = 0x1.7f7d1cp-20f;
static const float C_c2 = 0x1.62e4p-1f;
static const float C_p0 = 0x1.0e4020p-7f;
static const float C_p1 = 0x1.573e2ep-5f;
static const float C_p2 = 0x1.555e66p-3f;
static const float C_p3 = 0x1.fffdb6p-2f;
static const float C_p4 = 0x1.ffffecp-1f;

float expf_avx2(float x) {
    const float r = C_r;
    const float z = mad(x, C_lo, r);
    const float n = z - r;
    const float b = mad(-n, C_c1, mad(-n, C_c2, x));
    const uint e = asuint(z) << 23;
    const float k = asfloat(e + asuint(1.0f));
    const uint c = (abs(n) > 126.0f) ? 1u : 0u;
    const float u = b * b;
    const float j = mad(mad(mad(C_p0, b, C_p1), u, mad(C_p2, b, C_p3)), u, C_p4 * b);
    if (c == 0u)
        return mad(j, k, k);
    const uint g = (n <= 0.0f) ? 0x82000000u : 0u;
    const float s1 = asfloat(g + 0x7f000000u);
    const float s2 = asfloat(e - g);
    const uint d = (abs(n) > 192.0f) ? 1u : 0u;
    return (d != 0u) ? (s1 * s1) : (mad(s2, j, s2) * s1);
}

float silu_avx2(float x) {
    const float neg_x = 0.0f - x;
    const float e = expf_avx2(neg_x);
    return x / (1.0f + e);
}

[numthreads(256, 1, 1)] void main(uint3 tid : SV_DispatchThreadID) {
    ys[tid.x] = silu_avx2(xs[tid.x]);
}
