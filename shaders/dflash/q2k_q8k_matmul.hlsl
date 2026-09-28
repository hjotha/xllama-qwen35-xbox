// GGML Q2_K W[N,K] times Q8_K X[batch,K] -> F32 Y[batch,N].
// Q2_K: 84 bytes/block. Q8_K: 292 bytes/block. K must divide 256.
// Dispatch(N,ceil(batch/8),1); one 256-thread group per output row/tile.
// Integer products are summed exactly per QK block, then scaled in F32.
// Requires SM6 WaveOps; initialization fails if unavailable.

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb;
    uint batch;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

groupshared int2 wave_sum[8 * 8];  // 8 batch items x 8 waves (wave32/64 on SM6)
groupshared uint s_w2[21];        // 21 uint32s = 84 bytes; blocks are DWORD aligned
groupshared uint s_q8[512];
groupshared float s_d8[8];

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

[numthreads(256, 1, 1)]
void CSDot(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row = gid.x;
    const uint batch_start = gid.y * 8u;
    if (row >= n || batch_start >= batch)
        return; // group-uniform exit before any barrier

    const uint wave_size = WaveGetLaneCount();
    const uint wave_id = lane / wave_size;
    const uint wave_count = 256u / wave_size;
    const uint n_batch_in_group = min(8u, batch - batch_start);
    const uint total_q8_vec4 = n_batch_in_group * 16u;
    float accum = 0.0f;

    [loop]
    for (uint block = 0u; block < nb; ++block) {
        const uint q2_base = (row * nb + block) * 84u;
        const uint start_u32 = q2_base & ~3u;
        const uint shift = q2_base & 3u;

        if (lane < 21u)
            s_w2[lane] = W.Load(start_u32 + lane * 4u);
        if (lane < n_batch_in_group) {
            s_d8[lane] = asfloat(X.Load(((batch_start + lane) * nb + block) * 292u));
        }
        if (lane < total_q8_vec4) {
            const uint b = lane >> 4u;
            const uint q = lane & 15u;
            const uint4 v = X.Load4(((batch_start + b) * nb + block) * 292u + 4u + q * 16u);
            const uint dst = b * 64u + q * 4u;
            s_q8[dst + 0u] = v.x;
            s_q8[dst + 1u] = v.y;
            s_q8[dst + 2u] = v.z;
            s_q8[dst + 3u] = v.w;
        }
        GroupMemoryBarrierWithGroupSync();

        const int2 weight = q2_scaled_integer(lane, shift);
        [unroll]
        for (uint b = 0u; b < 8u; ++b) {
            if (batch_start + b < batch) {
                const uint packed = s_q8[b * 64u + (lane >> 2u)];
                const uint byte_value = (packed >> ((lane & 3u) * 8u)) & 255u;
                const int q8 = int(byte_value << 24u) >> 24;
                const int2 sum_val = WaveActiveSum(weight * q8);
                if (WaveIsFirstLane())
                    wave_sum[b * 8u + wave_id] = sum_val;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (lane < 8u && batch_start + lane < batch) {
            const uint b = lane;
            const float d2 = f16tof32(load_shared_q2_u16(80u, shift));
            const float dmin = f16tof32(load_shared_q2_u16(82u, shift));
            int2 integer_sum = int2(0, 0);
            for (uint wave = 0u; wave < wave_count; ++wave)
                integer_sum += wave_sum[b * 8u + wave];
            const float d8 = s_d8[b];
            accum += d8 * (d2 * float(integer_sum.x) - dmin * float(integer_sum.y));
        }
        GroupMemoryBarrierWithGroupSync(); // wave_sum can be reused next block
    }
    if (lane < 8u && batch_start + lane < batch) {
        Y[(batch_start + lane) * n + row] = accum;
    }
}
