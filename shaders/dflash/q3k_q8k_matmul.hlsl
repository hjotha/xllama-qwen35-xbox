// GGML Q3_K W[N,K] times Q8_K X[batch,K] -> F32 Y[batch,N].
// Q3_K: 110 bytes/block. Q8_K: 292 bytes/block. K must divide 256.
// Dispatch(N,ceil(batch/8),1); one 256-thread group per output row/tile.
// Integer products are summed exactly per QK block, then scaled in F32.
// Requires SM6 WaveOps; CPU path remains the fallback if unavailable.

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb;
    uint batch;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

groupshared int wave_sum[8 * 64]; // Up to 64 waves/group if wave size is 4.

uint load_w_u8(uint offset) {
    return (W.Load(offset & ~3u) >> ((offset & 3u) * 8u)) & 255u;
}

uint load_w_u16(uint even_offset) {
    return (W.Load(even_offset & ~3u) >> ((even_offset & 2u) * 8u)) & 65535u;
}

int load_q8_i8(uint offset) {
    const uint byte_value = (X.Load(offset & ~3u) >> ((offset & 3u) * 8u)) & 255u;
    return int(byte_value << 24u) >> 24;
}

int q3_scaled_integer(uint base, uint element) {
    const uint half = element >> 7u;
    const uint quarter = (element >> 5u) & 3u;
    const uint lane32 = element & 31u;
    const uint low2 = (load_w_u8(base + 32u + half * 32u + lane32)
                       >> (quarter * 2u)) & 3u;
    const uint high_bit = (load_w_u8(base + lane32)
                           >> (half * 4u + quarter)) & 1u;
    const int quant = int(low2) - (high_bit == 0u ? 4 : 0);

    const uint scale_index = half * 8u + quarter * 2u + (lane32 >> 4u);
    const uint scale_byte = load_w_u8(base + 96u + (scale_index & 7u));
    const uint high_byte = load_w_u8(base + 104u + (scale_index & 3u));
    const uint low4 = (scale_byte >> (scale_index >= 8u ? 4u : 0u)) & 15u;
    const uint high2 = (high_byte >> (2u * (scale_index >> 2u))) & 3u;
    const int scale = int(low4 | (high2 << 4u)) - 32;
    return scale * quant;
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
    float accum[8];
    [unroll]
    for (uint b = 0u; b < 8u; ++b)
        accum[b] = 0.0f;

    [loop]
    for (uint block = 0u; block < nb; ++block) {
        const uint q3_base = (row * nb + block) * 110u;
        const int weight = q3_scaled_integer(q3_base, lane);
        [unroll]
        for (uint b = 0u; b < 8u; ++b) {
            if (batch_start + b < batch) {
                const uint q8_base = ((batch_start + b) * nb + block) * 292u;
                const int q8 = load_q8_i8(q8_base + 4u + lane);
                const int sum = WaveActiveSum(weight * q8);
                if (WaveIsFirstLane())
                    wave_sum[b * 64u + wave_id] = sum;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (lane == 0u) {
            const float d3 = f16tof32(load_w_u16(q3_base + 108u));
            [unroll]
            for (uint b = 0u; b < 8u; ++b) {
                if (batch_start + b < batch) {
                    int integer_sum = 0;
                    for (uint wave = 0u; wave < wave_count; ++wave)
                        integer_sum += wave_sum[b * 64u + wave];
                    const uint q8_base = ((batch_start + b) * nb + block) * 292u;
                    const float d8 = asfloat(X.Load(q8_base));
                    accum[b] += (d8 * d3) * float(integer_sum);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync(); // wave_sum can be reused next block
    }
    if (lane == 0u) {
        [unroll]
        for (uint b = 0u; b < 8u; ++b) {
            if (batch_start + b < batch)
                Y[(batch_start + b) * n + row] = accum[b];
        }
    }
}
