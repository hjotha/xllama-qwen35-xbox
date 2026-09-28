// DFlash2 Q4_K/Q5_K W[N,K] x GGML Q8_K X[batch,K] -> F32 Y[batch,N].
// Q8_K layout: float d, int8 qs[256], int16 bsums[16] = 292 bytes.
// The Q8 pack kernel is shared with SYNC.
// Arithmetic matches ggml_vec_dot_q4_K_q8_K / ggml_vec_dot_q5_K_q8_K:
// (scale * quant) and min correction are exact integers per block, applied
// in F32 per block as (d8*dw)*sum_dot - (d8*dminw)*sum_corr.
// b0={N,K,blocks_per_row,batch}, W=t0, Q8=t1, Y=u0.
#if !defined(DFLASH_Q4_K) && !defined(DFLASH_Q5_K)
#error Select DFLASH_Q4_K or DFLASH_Q5_K
#endif

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb;
    uint batch;
};
ByteAddressBuffer W : register(t0);
ByteAddressBuffer Q8 : register(t1);
RWStructuredBuffer<float> Y : register(u0);

#include "quant_k_decode.hlsli"

groupshared int partial[2 * 8 * 64];

int q8_value(uint row_base, uint lane) {
    const uint packed = Q8.Load(row_base + 4u + (lane >> 2u) * 4u);
    const uint byte_value = (packed >> ((lane & 3u) * 8u)) & 255u;
    return int(byte_value << 24u) >> 24;
}

void decode_ints(uint base, uint element, out int scale, out int minimum,
                 out int quant) {
    const uint scale_index = element >> 5u;
    if (scale_index < 4u) {
        scale = int(dflash_load_u8(base + 4u + scale_index) & 63u);
        minimum = int(dflash_load_u8(base + 8u + scale_index) & 63u);
    } else {
        scale = int((dflash_load_u8(base + 8u + scale_index) & 15u)
                    | ((dflash_load_u8(base + scale_index) >> 6u) << 4u));
        minimum = int((dflash_load_u8(base + 8u + scale_index) >> 4u)
                      | ((dflash_load_u8(base + 4u + scale_index) >> 6u) << 4u));
    }
#if defined(DFLASH_Q4_K)
    const uint packed = dflash_load_u8(base + 16u + (element >> 6u) * 32u
                                       + (element & 31u));
    quant = int((packed >> ((element & 32u) != 0u ? 4u : 0u)) & 15u);
#else
    const uint lane32 = element & 31u;
    const uint packed = dflash_load_u8(base + 48u + (element >> 6u) * 32u + lane32);
    const uint low4 = (packed >> ((element & 32u) != 0u ? 4u : 0u)) & 15u;
    const uint high1 = (dflash_load_u8(base + 16u + lane32)
                        >> (element >> 5u)) & 1u;
    quant = int(low4 | (high1 << 4u));
#endif
}

[numthreads(256,1,1)]
void CSDot(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint row = group.x;
    const uint first = group.y * 8u;
    if (row >= n || first >= batch) return;
    const uint wave_size = WaveGetLaneCount();
    const uint wave_id = lane / wave_size;
    const uint wave_count = 256u / wave_size;
    float sum[8];
    [unroll] for (uint b = 0u; b < 8u; ++b) sum[b] = 0.0f;
    [loop] for (uint block = 0u; block < nb; ++block) {
#if defined(DFLASH_Q4_K)
        const uint wb = (row * nb + block) * 144u;
#else
        const uint wb = (row * nb + block) * 176u;
#endif
        int scale, minimum, quant;
        decode_ints(wb, lane, scale, minimum, quant);
        [unroll] for (uint b = 0u; b < 8u; ++b) {
            if (first + b < batch) {
                const uint ab = ((first + b) * nb + block) * 292u;
                const int q8 = q8_value(ab, lane);
                const int dot = WaveActiveSum(scale * quant * q8);
                const int corr = WaveActiveSum(minimum * q8);
                if (WaveIsFirstLane()) {
                    partial[(b * 64u + wave_id) * 2u + 0u] = dot;
                    partial[(b * 64u + wave_id) * 2u + 1u] = corr;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            const uint scales = W.Load(wb);
            const float dw = f16tof32(scales & 65535u);
            const float dminw = f16tof32(scales >> 16u);
            [unroll] for (uint b = 0u; b < 8u; ++b) {
                if (first + b < batch) {
                    int isum_dot = 0;
                    int isum_corr = 0;
                    for (uint wave = 0u; wave < wave_count; ++wave) {
                        isum_dot += partial[(b * 64u + wave) * 2u + 0u];
                        isum_corr += partial[(b * 64u + wave) * 2u + 1u];
                    }
                    const uint ab = ((first + b) * nb + block) * 292u;
                    const float d8 = asfloat(Q8.Load(ab));
                    sum[b] += (d8 * dw) * float(isum_dot)
                              - (d8 * dminw) * float(isum_corr);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        [unroll] for (uint b = 0u; b < 8u; ++b)
            if (first + b < batch)
                Y[(first + b) * n + row] = sum[b];
    }
}
