RWStructuredBuffer<float> Y : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint lane : SV_GroupIndex) {
    float accum = float(lane);
    uint avx_lane = lane & 7u;
    if (avx_lane == 0u) {
        precise float a = accum + WaveReadLaneAt(accum, lane + 4u);
        precise float c = WaveReadLaneAt(accum, lane + 1u) + WaveReadLaneAt(accum, lane + 5u);
        precise float e = WaveReadLaneAt(accum, lane + 2u) + WaveReadLaneAt(accum, lane + 6u);
        precise float g = WaveReadLaneAt(accum, lane + 3u) + WaveReadLaneAt(accum, lane + 7u);
        precise float result = (a + e) + (c + g);
        Y[lane / 8u] = result;
    }
}
