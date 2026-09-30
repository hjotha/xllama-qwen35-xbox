#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"

// The generic dot uses ggml-cpu's FP16 lookup table; ggml_cpu_init fills it.
extern "C" __declspec(dllexport) float dflash_ggml_q3q8_dot(const void * q3,
                                                               const void * q8,
                                                               int k) {
    ggml_cpu_init();
    float result = 0.0f;
    ggml_vec_dot_q3_K_q8_K_generic(k, &result, 0, q3, 0, q8, 0, 1);
    return result;
}

extern "C" __declspec(dllexport) void dflash_ggml_q8_quantize(const float * x,
                                                                  void * q8,
                                                                  int k) {
    quantize_row_q8_K_ref(x, static_cast<block_q8_K *>(q8), k);
}
