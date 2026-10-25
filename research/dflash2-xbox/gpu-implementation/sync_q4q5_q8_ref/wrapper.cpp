#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"

extern "C" __declspec(dllexport) void dflash_sync_quantize(int type, const float* input,
                                                           void* output, int k) {
    if (type == 4)
        quantize_row_q4_K_ref(input, static_cast<block_q4_K*>(output), k);
    if (type == 5)
        quantize_row_q5_K_ref(input, static_cast<block_q5_K*>(output), k);
    if (type == 8)
        quantize_row_q8_K_ref(input, static_cast<block_q8_K*>(output), k);
}

extern "C" __declspec(dllexport) float dflash_sync_dot(int type, const void* weight,
                                                       const void* activation, int k) {
    ggml_cpu_init();
    float result = 0.0f;
    if (type == 4)
        ggml_vec_dot_q4_K_q8_K_generic(k, &result, 0, weight, 0, activation, 0, 1);
    if (type == 5)
        ggml_vec_dot_q5_K_q8_K_generic(k, &result, 0, weight, 0, activation, 0, 1);
    return result;
}
