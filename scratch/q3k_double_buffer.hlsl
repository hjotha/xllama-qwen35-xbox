// GGML Q3_K W[N,K] times Q8_K X[batch,K] -> F32 Y[batch,N].
// Q3_K: 110 bytes/block. Q8_K: 292 bytes/block. K must divide 256.
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

groupshared float lane_sum[8 * 8];
groupshared uint s_w3[2][28];        // 28 uint32s = 112 bytes (covers full 110 bytes of Q3_K block)
groupshared uint s_q8[2][512];
groupshared float s_d8[2][8];
groupshared float s_q3_d[2];
groupshared int scaled_weights[2][256];

uint load_shared_q3_u8(uint buf, uint offset, uint shift) {
    const uint byte_pos = shift + offset;
    return (s_w3[buf][byte_pos >> 2u] >> ((byte_pos & 3u) * 8u)) & 255u;
}

uint load_shared_q3_u16(uint buf, uint even_offset, uint shift) {
    const uint byte_pos = shift + even_offset;
    return (s_w3[buf][byte_pos >> 2u] >> ((byte_pos & 2u) * 8u)) & 65535u;
}

int q3_scaled_integer(uint buf, uint element, uint shift) {
    const uint half = element >> 7u;
    const uint quarter = (element >> 5u) & 3u;
    const uint lane32 = element & 31u;
    const uint low2 = (load_shared_q3_u8(buf, 32u + half * 32u + lane32, shift)
                       >> (quarter * 2u)) & 3u;
    const uint high_bit = (load_shared_q3_u8(buf, lane32, shift)
                           >> (half * 4u + quarter)) & 1u;
    const int quant = int(low2) - (high_bit == 0u ? 4 : 0);

    const uint scale_index = element >> 4u;
    const uint scale_byte = load_shared_q3_u8(buf, 96u + (scale_index & 7u), shift);
    const uint high_byte = load_shared_q3_u8(buf, 104u + (scale_index & 3u), shift);
    const uint low4 = (scale_byte >> (scale_index >= 8u ? 4u : 0u)) & 15u;
    const uint high2 = (high_byte >> (2u * (scale_index >> 2u))) & 3u;
    const int scale = int(low4 | (high2 << 4u)) - 32;
    return scale * quant;
}

void load_q3_block(uint buf, uint block_idx, uint row, uint batch_start,
                   uint n_batch_in_group, uint total_q8_vec4,
                   uint lane) {
    const uint q3_base = (row * nb + block_idx) * 110u;
    const uint start_u32 = q3_base & ~3u;

    if (lane < 7u) {
        const uint4 v = W.Load4(start_u32 + lane * 16u);
        const uint base_idx = lane * 4u;
        s_w3[buf][base_idx + 0u] = v.x;
        s_w3[buf][base_idx + 1u] = v.y;
        s_w3[buf][base_idx + 2u] = v.z;
        s_w3[buf][base_idx + 3u] = v.w;
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
}

void decode_q3_weights(uint buf, uint block_idx, uint row, uint lane) {
    const uint q3_base = (row * nb + block_idx) * 110u;
    const uint shift = q3_base & 3u;
    if (lane == 0u) {
        s_q3_d[buf] = f16tof32(load_shared_q3_u16(buf, 108u, shift));
    }
    [unroll] for (uint part = 0u; part < 4u; ++part) {
        const uint element = lane + part * 64u;
        scaled_weights[buf][element] = q3_scaled_integer(buf, element, shift);
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
        load_q3_block(0u, 0u, row, batch_start, n_batch_in_group, total_q8_vec4, lane);
        GroupMemoryBarrierWithGroupSync();
        decode_q3_weights(0u, 0u, row, lane);
        GroupMemoryBarrierWithGroupSync();
    }

    [loop]
    for (uint block = 0u; block < nb; ++block) {
        const uint curr = block & 1u;
        const uint next = 1u - curr;
        const uint next_block = block + 1u;

        if (next_block < nb) {
            load_q3_block(next, next_block, row, batch_start, n_batch_in_group, total_q8_vec4, lane);
        }

        if (b < n_batch_in_group) {
            int integer_sum = 0;
            [unroll] for (uint segment = 0u; segment < 8u; ++segment) {
                const uint first = segment * 32u + avx_lane * 4u;
                const uint packed = s_q8[curr][b * 64u + first / 4u];
                const int q0 = int(packed << 24u) >> 24;
                const int q1 = int((packed >> 8u) << 24u) >> 24;
                const int q2 = int((packed >> 16u) << 24u) >> 24;
                const int q3 = int(packed) >> 24;
                integer_sum += scaled_weights[curr][first + 0u] * q0 +
                               scaled_weights[curr][first + 1u] * q1 +
                               scaled_weights[curr][first + 2u] * q2 +
                               scaled_weights[curr][first + 3u] * q3;
            }
            precise float d = s_d8[curr][b] * s_q3_d[curr];
            accum = ggml_fma_f32(d, float(integer_sum), accum);
        }

        if (next_block < nb) {
            GroupMemoryBarrierWithGroupSync();
            decode_q3_weights(next, next_block, row, lane);
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
