#pragma once

#include <string>

// Isolated DirectML comparison. This does not select a generation backend or
// mutate the DFlash RPC sequence. The caller should keep it on a diagnostic
// opcode, outside the inference hot path.
struct dflash_dml_result {
    bool passed = false; // execution, finite output, and fixed numeric gate
    std::string json;
};

// Measures the real GGUF blk.0.ffn_down.weight Q4_K tensor as a single-vector
// GEMM (1 x 17408 by 17408 x 5120). The tensor is dequantized once to FP16;
// the JSON reports conversion error against the original Q4_K CPU reference.
// Each of three runs has 50 warmups and 200 CPU-to-GPU-to-CPU measurements.
dflash_dml_result run_dflash_dml_real_q4k(const std::string& model_path);
