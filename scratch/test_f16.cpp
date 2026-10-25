#include <DirectXPackedVector.h>
#include <cmath>
#include <cstdint>
#include <iostream>

// GGML CPU conversion from ggml-impl.h:
static inline float ggml_compute_fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = exp_scale * (float)(int32_t)(two_w + exp_offset);
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = magic_bias * (float)(int32_t)((two_w >> 17) | magic_mask);
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? *(uint32_t*)&denormalized_value
                                                                : *(uint32_t*)&normalized_value);
    return *(float*)&result;
}

int main() {
    int diff_count = 0;
    for (uint32_t u = 0; u < 65536; ++u) {
        uint16_t h = (uint16_t)u;
        float f_dx = DirectX::PackedVector::XMConvertHalfToFloat(h);
        float f_ggml = ggml_compute_fp16_to_fp32(h);
        uint32_t u_dx = *(uint32_t*)&f_dx;
        uint32_t u_ggml = *(uint32_t*)&f_ggml;
        if (u_dx != u_ggml) {
            if (diff_count < 10) {
                std::cout << "Diff at 0x" << std::hex << u << ": DX=0x" << u_dx << " (" << f_dx
                          << ") vs GGML=0x" << u_ggml << " (" << f_ggml << ")" << std::dec
                          << std::endl;
            }
            diff_count++;
        }
    }
    std::cout << "Total differences out of 65536: " << diff_count << std::endl;
    return 0;
}
