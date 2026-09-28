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

#if defined(DFLASH_Q4_K)
#define BLOCK_BYTES 144u
#define BLOCK_U32_COUNT 36u
#define BLOCK_U32X4_COUNT 9u
#else
#define BLOCK_BYTES 176u
#define BLOCK_U32_COUNT 44u
#define BLOCK_U32X4_COUNT 11u
#endif

groupshared int partial[2 * 8 * 8]; // 8 batch rows x 8 waves x 2 (dot, corr)
groupshared uint s_w[BLOCK_U32_COUNT];
groupshared uint s_q8[512];
groupshared float s_d8[8];

uint load_shared_w_u8(uint offset) {
    return (s_w[offset >> 2u] >> ((offset & 3u) * 8u)) & 255u;
}

void decode_ints(uint element, out int scale, out int minimum,
                 out int quant) {
    const uint scale_index = element >> 5u;
    if (scale_index < 4u) {
        scale = int(load_shared_w_u8(4u + scale_index) & 63u);
        minimum = int(load_shared_w_u8(8u + scale_index) & 63u);
    } else {
        scale = int((load_shared_w_u8(8u + scale_index) & 15u)
                    | ((load_shared_w_u8(scale_index) >> 6u) << 4u));
        minimum = int((load_shared_w_u8(8u + scale_index) >> 4u)
                      | ((load_shared_w_u8(4u + scale_index) >> 6u) << 4u));
    }
#if defined(DFLASH_Q4_K)
    const uint packed = load_shared_w_u8(16u + (element >> 6u) * 32u
                                       + (element & 31u));
    quant = int((packed >> ((element & 32u) != 0u ? 4u : 0u)) & 15u);
#else
    const uint lane32 = element & 31u;
    const uint packed = load_shared_w_u8(48u + (element >> 6u) * 32u + lane32);
    const uint low4 = (packed >> ((element & 32u) != 0u ? 4u : 0u)) & 15u;
    const uint high1 = (load_shared_w_u8(16u + lane32)
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
    const uint n_batch_in_group = min(8u, batch - first);
    const uint total_q8_vec4 = n_batch_in_group * 16u;
    float sum = 0.0f;
    [loop] for (uint block = 0u; block < nb; ++block) {
        const uint wb = (row * nb + block) * BLOCK_BYTES;
        if (lane < BLOCK_U32X4_COUNT) {
            const uint4 v = W.Load4(wb + lane * 16u);
            const uint base_idx = lane * 4u;
            s_w[base_idx + 0u] = v.x;
            s_w[base_idx + 1u] = v.y;
            s_w[base_idx + 2u] = v.z;
            s_w[base_idx + 3u] = v.w;
        }
        if (lane < n_batch_in_group) {
            s_d8[lane] = asfloat(Q8.Load(((first + lane) * nb + block) * 292u));
        }
        if (lane < total_q8_vec4) {
            const uint b = lane >> 4u;
            const uint q = lane & 15u;
            const uint4 v = Q8.Load4(((first + b) * nb + block) * 292u + 4u + q * 16u);
            const uint dst = b * 64u + q * 4u;
            s_q8[dst + 0u] = v.x;
            s_q8[dst + 1u] = v.y;
            s_q8[dst + 2u] = v.z;
            s_q8[dst + 3u] = v.w;
        }
        GroupMemoryBarrierWithGroupSync();

        int scale, minimum, quant;
        decode_ints(lane, scale, minimum, quant);
        const int sc_quant = scale * quant;
        [unroll] for (uint b = 0u; b < 8u; ++b) {
            if (first + b < batch) {
                const uint packed = s_q8[b * 64u + (lane >> 2u)];
                const uint byte_val = (packed >> ((lane & 3u) * 8u)) & 255u;
                const int q8 = int(byte_val << 24u) >> 24;
                const int dot = WaveActiveSum(sc_quant * q8);
                const int corr = WaveActiveSum(minimum * q8);
                if (WaveIsFirstLane()) {
                    partial[(b * 8u + wave_id) * 2u + 0u] = dot;
                    partial[(b * 8u + wave_id) * 2u + 1u] = corr;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane < 8u && first + lane < batch) {
            const uint b = lane;
            const uint scales = s_w[0];
            const float dw = f16tof32(scales & 65535u);
            const float dminw = f16tof32(scales >> 16u);
            int isum_dot = 0;
            int isum_corr = 0;
            for (uint wave = 0u; wave < wave_count; ++wave) {
                isum_dot += partial[(b * 8u + wave) * 2u + 0u];
                isum_corr += partial[(b * 8u + wave) * 2u + 1u];
            }
            const float d8 = s_d8[b];
            sum += (d8 * dw) * float(isum_dot)
                  - (d8 * dminw) * float(isum_corr);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane < 8u && first + lane < batch) {
        Y[(first + lane) * n + row] = sum;
    }
}
