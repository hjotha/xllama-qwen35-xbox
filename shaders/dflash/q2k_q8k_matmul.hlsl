// GGML Q2_K W[N,K] times Q8_K X[batch,K] -> F32 Y[batch,N].
// Q2_K: 84 bytes/block. Q8_K: 292 bytes/block. K must divide 256.
// Dispatch(N,ceil(batch/8),1); one 64-thread group per output row/tile.
// Match GGML's x86 AVX2 path: eight FMA accumulators and its final hsum.
#include "ggml_float.hlsli"

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb;
    uint batch;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

groupshared float lane_sum[8 * 8]; // batch item x AVX2 accumulator
groupshared uint s_w2[21];        // 21 uint32s = 84 bytes; blocks are DWORD aligned
groupshared uint s_q8[512];
groupshared float s_d8[8];
groupshared float s_q2_d;
groupshared float s_q2_dmin;
groupshared int scaled_weights[256];

uint load_shared_q2_u8(uint offset, uint shift) {
    const uint byte_pos = shift + offset;
    return (s_w2[byte_pos >> 2u] >> ((byte_pos & 3u) * 8u)) & 255u;
}

uint load_shared_q2_u16(uint even_offset, uint shift) {
    const uint byte_pos = shift + even_offset;
    return (s_w2[byte_pos >> 2u] >> ((byte_pos & 2u) * 8u)) & 65535u;
}

int2 q2_scaled_integer(uint element, uint shift) {
    const uint half = element >> 7u;
    const uint quarter = (element >> 5u) & 3u;
    // Q2_K layout: scales (0..15), qs (16..79), d/dmin (80..83).
    const uint quant = (load_shared_q2_u8(16u + half * 32u + (element & 31u), shift)
                        >> (quarter * 2u)) & 3u;
    const uint scale = load_shared_q2_u8(element >> 4u, shift);
    return int2(int((scale & 15u) * quant), int(scale >> 4u));
}

[numthreads(64, 1, 1)]
void CSDot(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row = gid.x;
    const uint batch_start = gid.y * 8u;
    if (row >= n || batch_start >= batch)
        return; // group-uniform exit before any barrier

    const uint n_batch_in_group = min(8u, batch - batch_start);
    const uint total_q8_vec4 = n_batch_in_group * 16u;
    const uint b = lane / 8u;
    const uint avx_lane = lane & 7u;
    precise float accum = 0.0f;

    [loop]
    for (uint block = 0u; block < nb; ++block) {
        const uint q2_base = (row * nb + block) * 84u;
        const uint start_u32 = q2_base & ~3u;
        const uint shift = q2_base & 3u;

        if (lane < 5u) {
            const uint4 v = W.Load4(start_u32 + lane * 16u);
            const uint base_idx = lane * 4u;
            s_w2[base_idx + 0u] = v.x;
            s_w2[base_idx + 1u] = v.y;
            s_w2[base_idx + 2u] = v.z;
            s_w2[base_idx + 3u] = v.w;
        } else if (lane == 5u) {
            s_w2[20u] = W.Load(start_u32 + 80u);
        }

        if (lane < n_batch_in_group) {
            s_d8[lane] = asfloat(X.Load(((batch_start + lane) * nb + block) * 292u));
        }
        for (uint load = lane; load < total_q8_vec4; load += 64u) {
            const uint b_idx = load >> 4u;
            const uint q = load & 15u;
            const uint4 v = X.Load4(((batch_start + b_idx) * nb + block) * 292u + 4u + q * 16u);
            const uint dst = b_idx * 64u + q * 4u;
            s_q8[dst + 0u] = v.x;
            s_q8[dst + 1u] = v.y;
            s_q8[dst + 2u] = v.z;
            s_q8[dst + 3u] = v.w;
        }
        GroupMemoryBarrierWithGroupSync();

        if (lane == 0u) {
            s_q2_d = f16tof32(load_shared_q2_u16(80u, shift));
            s_q2_dmin = f16tof32(load_shared_q2_u16(82u, shift));
        }

        // Reuse each decoded weight across tokens; retain CPU FMA ordering.
        [unroll] for (uint part = 0u; part < 4u; ++part) {
            const uint element = lane + part * 64u;
            scaled_weights[element] = q2_scaled_integer(element, shift).x;
        }
        GroupMemoryBarrierWithGroupSync();

        if (b < n_batch_in_group) {
            int integer_sum = 0;
            [unroll] for (uint segment = 0u; segment < 8u; ++segment) {
                const uint first = segment * 32u + avx_lane * 4u;
                const uint packed = s_q8[b * 64u + first / 4u];
                [unroll] for (uint j = 0u; j < 4u; ++j) {
                    const int q8 = int((packed >> (j * 8u)) << 24u) >> 24;
                    integer_sum += scaled_weights[first + j] * q8;
                }
            }
            const uint sums = X.Load(((batch_start + b) * nb + block) * 292u +
                                     260u + avx_lane * 4u);
            const int sum0 = int(sums << 16u) >> 16;
            const int sum1 = int(sums) >> 16;
            const int minimum = sum0 * int(load_shared_q2_u8(avx_lane * 2u, shift) >> 4u)
                              + sum1 * int(load_shared_q2_u8(avx_lane * 2u + 1u, shift) >> 4u);
            const float d8 = s_d8[b];
            precise float d = d8 * s_q2_d;
            precise float dmin = -d8 * s_q2_dmin;
            // HLSL mad splits the rounding on both Xbox and WARP. F64 keeps
            // the product until the single F32 rounding required by AVX2 FMA.
            accum = ggml_fma_f32(dmin, float(minimum), accum);
            accum = ggml_fma_f32(d, float(integer_sum), accum);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (b < n_batch_in_group) lane_sum[lane] = accum;
    GroupMemoryBarrierWithGroupSync();
    if (b < n_batch_in_group && avx_lane == 0u) {
        precise float a = lane_sum[lane] + lane_sum[lane + 4u];
        precise float c = lane_sum[lane + 1u] + lane_sum[lane + 5u];
        precise float e = lane_sum[lane + 2u] + lane_sum[lane + 6u];
        precise float g = lane_sum[lane + 3u] + lane_sum[lane + 7u];
        precise float result = (a + e) + (c + g);
        Y[(batch_start + b) * n + row] = result;
    }
}
