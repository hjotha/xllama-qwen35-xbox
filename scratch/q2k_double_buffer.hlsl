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
groupshared uint s_w2[2][21];      // 21 uint32s = 84 bytes; blocks are DWORD aligned
groupshared uint s_q8[2][512];
groupshared uint s_sums[2][64];
groupshared float s_d8[2][8];
groupshared float s_q2_d[2];
groupshared float s_q2_dmin[2];
groupshared int scaled_weights[2][256];

uint load_shared_q2_u8(uint buf, uint offset, uint shift) {
    const uint byte_pos = shift + offset;
    return (s_w2[buf][byte_pos >> 2u] >> ((byte_pos & 3u) * 8u)) & 255u;
}

uint load_shared_q2_u16(uint buf, uint even_offset, uint shift) {
    const uint byte_pos = shift + even_offset;
    return (s_w2[buf][byte_pos >> 2u] >> ((byte_pos & 2u) * 8u)) & 65535u;
}

int2 q2_scaled_integer(uint buf, uint element, uint shift) {
    const uint half = element >> 7u;
    const uint quarter = (element >> 5u) & 3u;
    // Q2_K layout: scales (0..15), qs (16..79), d/dmin (80..83).
    const uint quant = (load_shared_q2_u8(buf, 16u + half * 32u + (element & 31u), shift)
                        >> (quarter * 2u)) & 3u;
    const uint scale = load_shared_q2_u8(buf, element >> 4u, shift);
    return int2(int((scale & 15u) * quant), int(scale >> 4u));
}

void load_q2_block(uint buf, uint block_idx, uint row, uint batch_start,
                   uint n_batch_in_group, uint total_q8_vec4,
                   uint lane, uint b, uint avx_lane) {
    const uint q2_base = (row * nb + block_idx) * 84u;
    const uint start_u32 = q2_base & ~3u;

    if (lane < 5u) {
        const uint4 v = W.Load4(start_u32 + lane * 16u);
        const uint base_idx = lane * 4u;
        s_w2[buf][base_idx + 0u] = v.x;
        s_w2[buf][base_idx + 1u] = v.y;
        s_w2[buf][base_idx + 2u] = v.z;
        s_w2[buf][base_idx + 3u] = v.w;
    } else if (lane == 5u) {
        s_w2[buf][20u] = W.Load(start_u32 + 80u);
    }

    if (lane < n_batch_in_group) {
        s_d8[buf][lane] = asfloat(X.Load(((batch_start + lane) * nb + block_idx) * 292u));
    }
    for (uint load = lane; load < total_q8_vec4; load += 64u) {
        const uint b_idx = load >> 4u;
        const uint q = load & 15u;
        const uint4 v = X.Load4(((batch_start + b_idx) * nb + block_idx) * 292u + 4u + q * 16u);
        const uint dst = b_idx * 64u + q * 4u;
        s_q8[buf][dst + 0u] = v.x;
        s_q8[buf][dst + 1u] = v.y;
        s_q8[buf][dst + 2u] = v.z;
        s_q8[buf][dst + 3u] = v.w;
    }
    if (b < n_batch_in_group) {
        s_sums[buf][lane] = X.Load(((batch_start + b) * nb + block_idx) * 292u + 260u + avx_lane * 4u);
    }
}

void decode_q2_weights(uint buf, uint block_idx, uint row, uint lane) {
    const uint q2_base = (row * nb + block_idx) * 84u;
    const uint shift = q2_base & 3u;
    if (lane == 0u) {
        s_q2_d[buf] = f16tof32(load_shared_q2_u16(buf, 80u, shift));
        s_q2_dmin[buf] = f16tof32(load_shared_q2_u16(buf, 82u, shift));
    }
    [unroll] for (uint part = 0u; part < 4u; ++part) {
        const uint element = lane + part * 64u;
        scaled_weights[buf][element] = q2_scaled_integer(buf, element, shift).x;
    }
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

    if (nb > 0u) {
        load_q2_block(0u, 0u, row, batch_start, n_batch_in_group, total_q8_vec4, lane, b, avx_lane);
        GroupMemoryBarrierWithGroupSync();
        decode_q2_weights(0u, 0u, row, lane);
        GroupMemoryBarrierWithGroupSync();
    }

    [loop]
    for (uint block = 0u; block < nb; ++block) {
        const uint curr = block & 1u;
        const uint next = 1u - curr;
        const uint next_block = block + 1u;

        if (next_block < nb) {
            load_q2_block(next, next_block, row, batch_start, n_batch_in_group, total_q8_vec4, lane, b, avx_lane);
        }

        if (b < n_batch_in_group) {
            const uint q2_base = (row * nb + block) * 84u;
            const uint shift = q2_base & 3u;
            int integer_sum = 0;
            [unroll] for (uint segment = 0u; segment < 8u; ++segment) {
                const uint first = segment * 32u + avx_lane * 4u;
                const uint packed = s_q8[curr][b * 64u + first / 4u];
                [unroll] for (uint j = 0u; j < 4u; ++j) {
                    const int q8 = int((packed >> (j * 8u)) << 24u) >> 24;
                    integer_sum += scaled_weights[curr][first + j] * q8;
                }
            }
            const uint sums = s_sums[curr][lane];
            const int sum0 = int(sums << 16u) >> 16;
            const int sum1 = int(sums) >> 16;
            const int minimum = sum0 * int(load_shared_q2_u8(curr, avx_lane * 2u, shift) >> 4u)
                              + sum1 * int(load_shared_q2_u8(curr, avx_lane * 2u + 1u, shift) >> 4u);
            const float d8 = s_d8[curr][b];
            precise float d = d8 * s_q2_d[curr];
            precise float dmin = -d8 * s_q2_dmin[curr];
            accum = ggml_fma_f32(dmin, float(minimum), accum);
            accum = ggml_fma_f32(d, float(integer_sum), accum);
        }

        if (next_block < nb) {
            GroupMemoryBarrierWithGroupSync();
            decode_q2_weights(next, next_block, row, lane);
            GroupMemoryBarrierWithGroupSync();
        }
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
