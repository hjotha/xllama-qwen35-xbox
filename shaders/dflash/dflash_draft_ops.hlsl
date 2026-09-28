// DFlash2 DRAFT scalar operators. All rows are contiguous F32 in GGML order.
// The quantized matmuls, dynamic convolution, attention, and KV pack are
// separate shaders so that no intermediate hidden row is read back to CPU.
cbuffer Params : register(b0) {
    uint rows;
    uint width;
    uint head_width;
    uint reserved0;
    float scale;
    float epsilon;
    float rope_base;
    float reserved1;
};

ByteAddressBuffer X : register(t0);
ByteAddressBuffer W : register(t1);
ByteAddressBuffer Positions : register(t2);
RWByteAddressBuffer Y : register(u0);

groupshared float partial[128];

// Dispatch(rows,1,1). Supports hidden norm (5120) and per-head Q/K norm (128).
[numthreads(128,1,1)]
void RmsNorm(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint row = group.x;
    if (row >= rows || width == 0u) return;
    float ss = 0.0f;
    for (uint col = lane; col < width; col += 128u) {
        const float value = asfloat(X.Load((row * width + col) * 4u)) * scale;
        ss += value * value;
    }
    partial[lane] = ss;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 64u; stride != 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float inv = rsqrt(partial[0] / float(width) + epsilon);
    for (uint col = lane; col < width; col += 128u) {
        const float value = asfloat(X.Load((row * width + col) * 4u)) * scale;
        Y.Store((row * width + col) * 4u,
                asuint(value * inv * asfloat(W.Load((col % head_width) * 4u))));
    }
}

// Dispatch(heads, tokens, 1). DFlash temporal M-RoPE sections [64,0,0,0]
// pair (d,d+64), not adjacent pairs. Width is heads*128.
[numthreads(64,1,1)]
void Rope128(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    const uint head = group.x;
    const uint token = group.y;
    if (token >= rows || head * 128u >= width) return;
    const uint index = token * width + head * 128u + lane;
    const float a = asfloat(X.Load(index * 4u));
    const float b = asfloat(X.Load((index + 64u) * 4u));
    const float pos = float(Positions.Load(token * 4u));
    const float angle = pos * pow(rope_base, -float(lane) / 64.0f);
    float sn, cs;
    sincos(angle, sn, cs);
    Y.Store(index * 4u, asuint(a * cs - b * sn));
    Y.Store((index + 64u) * 4u, asuint(a * sn + b * cs));
}

// Y=X*scale+W. A zero W resource supplies the no-residual variant. The
// caller must never bind X or W as Y in the same dispatch.
[numthreads(128,1,1)]
void ScaleAdd(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= rows * width) return;
    Y.Store(i * 4u, asuint(asfloat(X.Load(i * 4u)) * scale +
                            asfloat(W.Load(i * 4u))));
}

// Gate and up use [token,17408]. Dispatch ceil(rows*width/128).
[numthreads(128,1,1)]
void SwiGlu(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= rows * width) return;
    const float gate = asfloat(X.Load(i * 4u));
    const float up = asfloat(W.Load(i * 4u));
    Y.Store(i * 4u, asuint((gate / (1.0f + exp(-gate))) * up));
}
