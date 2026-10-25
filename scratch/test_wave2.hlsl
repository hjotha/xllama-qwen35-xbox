RWStructuredBuffer<float> Y : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint lane : SV_GroupIndex) {
    float accum = float(lane);
    uint b = lane / 8u;
    uint avx_lane = lane & 7u;

    // Convergent 8-lane reduction matching AVX2 hsum order: (a + e) + (c + g)
    // a = lane0 + lane4, e = lane2 + lane6, c = lane1 + lane5, g = lane3 + lane7
    precise float pair = accum + WaveReadLaneAt(accum, lane ^ 4u);
    precise float quad = pair + WaveReadLaneAt(pair, lane ^ 2u);
    precise float oct = quad + WaveReadLaneAt(quad, lane ^ 1u);

    if (b < 8u && avx_lane == 0u) {
        Y[b] = oct;
    }
}
