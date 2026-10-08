// SWIGLU microkernel feasibility probe: exact port of the AVX2 ggml_v_expf /
// ggml_v_silu (vec.h:1215+) to HLSL. Not wired into the backend; this file
// exists to inspect the DXIL dxc emits for fma/div precision before any
// production plumbing.
StructuredBuffer<float> xs : register(t0);
StructuredBuffer<float> gs : register(t1);
RWStructuredBuffer<float> dst : register(u0);

static const float C_r  = asfloat(0x4b400000u); // 0x1.8p23f
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
    const float z = x * C_lo + r;
    const float n = z - r;
    const float b = -n * C_c1 + (-n * C_c2 + x);
    const uint e = asuint(z) << 23;
    const float k = asfloat(e + asuint(1.0f));
    const uint c = (asuint(abs(n)) > asuint(126.0f)) ? 1u : 0u;
    const float u = b * b;
    const float j = (C_p0 * b + C_p1) * u + (C_p2 * b + C_p3);
    const float j2 = j * u + C_p4 * b;
    if (c == 0u)
        return j2 * k + k;
    const uint g = (n <= 0.0f) ? 0x82000000u : 0u;
    const float s1 = asfloat(g + 0x7f000000u);
    const float s2 = asfloat(e - g);
    const uint d = (asuint(abs(n)) > asuint(192.0f)) ? 1u : 0u;
    return (d != 0u) ? (s1 * s1) : ((s2 * j2 + s2) * s1);
}

float silu_avx2(float x) {
    const float neg_x = 0.0f - x;
    const float e = expf_avx2(neg_x);
    return x / (1.0f + e);
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    dst[i] = silu_avx2(xs[i]) * gs[i];
}
