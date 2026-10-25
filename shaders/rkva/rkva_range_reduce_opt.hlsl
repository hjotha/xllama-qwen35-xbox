// RKVA v2 split-K reduction for up to 256 splits (optimized Q4 path).
// Same input/output layout and arithmetic as rkva_range_reduce.hlsl: the split
// maximum, weights exp(m_s - M), denominator and numerator sums are evaluated
// in the same split order, but each split's (max, denom) and weight are loaded
// or computed once per group instead of once per output element.
#define RKVA_REDUCE_MAX_SPLITS 256
cbuffer Params : register(b0) {
    uint n_query;
    uint n_head;
    uint head_dim;
    uint split_k;
};
ByteAddressBuffer partials : register(t0);
RWByteAddressBuffer output : register(u0);
groupshared float gs_split_max[RKVA_REDUCE_MAX_SPLITS];
groupshared float gs_split_den[RKVA_REDUCE_MAX_SPLITS];
groupshared float gs_split_weight[RKVA_REDUCE_MAX_SPLITS];
groupshared float gs_inv;

[numthreads(128, 1, 1)]
void CSRangeReduceOpt(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint qh = group.y * n_head + group.x;
    uint total = n_query * n_head;
    uint stride = head_dim + 2;
    uint splits = min(split_k, RKVA_REDUCE_MAX_SPLITS);
    for (uint s = lane; s < splits; s += 128) {
        uint base = (s * total + qh) * stride;
        gs_split_max[s] = asfloat(partials.Load((base + head_dim) * 4));
        gs_split_den[s] = asfloat(partials.Load((base + head_dim + 1) * 4));
    }
    GroupMemoryBarrierWithGroupSync();
    float maximum = -3.402823466e38f;
    for (uint s = 0; s < splits; ++s) {
        if (gs_split_den[s] > 0) maximum = max(maximum, gs_split_max[s]);
    }
    for (uint s = lane; s < splits; s += 128) {
        gs_split_weight[s] = gs_split_den[s] > 0 ? exp(gs_split_max[s] - maximum) : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
        float sum = 0;
        for (uint s = 0; s < splits; ++s) {
            if (gs_split_den[s] > 0) sum += gs_split_den[s] * gs_split_weight[s];
        }
        gs_inv = sum > 0 ? 1 / sum : 0;
        float lse = sum > 0 ? maximum + log(sum) : asfloat(0xff800000);
        output.Store((total * head_dim + qh) * 4, asuint(lse));
    }
    GroupMemoryBarrierWithGroupSync();
    float inv = gs_inv;
    for (uint d = lane; d < head_dim; d += 128) {
        float acc = 0;
        for (uint s = 0; s < splits; ++s) {
            if (gs_split_den[s] > 0)
                acc += asfloat(partials.Load(((s * total + qh) * stride + d) * 4)) * gs_split_weight[s];
        }
        output.Store((qh * head_dim + d) * 4, asuint(acc * inv));
    }
}
