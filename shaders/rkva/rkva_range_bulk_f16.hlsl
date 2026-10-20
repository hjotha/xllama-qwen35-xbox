// RKVA v2 native bulk FP16 Q/OUT GPU conversion shaders (Xbox Series X|S).
//
// Shares the 4-constant + 1-SRV (t0) + 1-UAV (u0) root signature layout of
// range_reduce_root_ so no additional root signature object is required:
//   b0 = { q_pairs, aux_words, reserved0, reserved1 }
//   t0 = source ByteAddressBuffer
//   u0 = destination RWByteAddressBuffer
//
// CSBulkUnpackQF16 (after upload, before attention dispatch):
//   src (UPLOAD):  Q_f16 [q_pairs * 2] packed 2xF16 words, then positions [n_queries] i32
//   dst (DEFAULT): Q_f32 [q_pairs * 2] F32 words, then positions [n_queries] i32
//
// CSBulkPackOutF16 (after attention/reduce, before readback DMA copy):
//   src (DEFAULT): OUT_f32 [q_pairs * 2] F32 words, then LSE_f32 [q_heads] F32 words
//   dst (DEFAULT): OUT_f16 [q_pairs * 2] packed 2xF16 words, then LSE_f32 [q_heads] F32 words

cbuffer Params : register(b0) {
    uint q_pairs;
    uint aux_words;
    uint reserved0;
    uint reserved1;
};

ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

// GGML F32->F16 round-to-nearest-even (bit-exact with ggml_fp32_to_fp16).
uint ggml_half_bits(float value) {
    const uint bits = asuint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const uint exponent = (bits >> 23) & 255u;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 255u) {
        return sign | (mantissa == 0 ? 0x7c00u : 0x7e00u);
    }
    int half_exponent = int(exponent) - 127 + 15;
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000u;
        const uint shift = uint(14 - half_exponent);
        uint rounded = mantissa >> shift;
        const uint remainder = mantissa & ((1u << shift) - 1u);
        const uint halfway = uint(1u << (shift - 1u));
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return sign | rounded;
    }
    if (half_exponent >= 31) {
        return sign | 0x7c00u;
    }
    uint rounded = mantissa >> 13;
    const uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
    }
    if (rounded == 0x400u) {
        rounded = 0;
        ++half_exponent;
    }
    if (half_exponent >= 31) {
        return sign | 0x7c00u;
    }
    return sign | (uint(half_exponent) << 10) | rounded;
}

[numthreads(256, 1, 1)]
void CSBulkUnpackQF16(uint3 tid : SV_DispatchThreadID) {
    const uint idx = tid.x;
    if (idx < q_pairs) {
        const uint packed = src.Load(idx * 4);
        const float lo = f16tof32(packed & 0xffffu);
        const float hi = f16tof32(packed >> 16);
        dst.Store((idx * 2) * 4, asuint(lo));
        dst.Store((idx * 2 + 1) * 4, asuint(hi));
    }
    if (idx < aux_words) {
        const uint pos_word = src.Load((q_pairs + idx) * 4);
        dst.Store((q_pairs * 2 + idx) * 4, pos_word);
    }
}

[numthreads(256, 1, 1)]
void CSBulkPackOutF16(uint3 tid : SV_DispatchThreadID) {
    const uint idx = tid.x;
    if (idx < q_pairs) {
        const float lo = asfloat(src.Load((idx * 2) * 4));
        const float hi = asfloat(src.Load((idx * 2 + 1) * 4));
        const uint packed = (ggml_half_bits(lo) & 0xffffu) | (ggml_half_bits(hi) << 16);
        dst.Store(idx * 4, packed);
    }
    if (idx < aux_words) {
        const uint lse_word = src.Load((q_pairs * 2 + idx) * 4);
        dst.Store((q_pairs + idx) * 4, lse_word);
    }
}
