#pragma once

#include <cstddef>
#include <string>

struct dflash_gpu_conv_probe_result {
    bool passed = false;
    std::string json;
};

// Runs the isolated DFlash2 dynamic-convolution DXIL on a hardware D3D12
// adapter and compares all output elements with the CPU graph semantics.
// base points to the original contiguous GGUF F32 [5120, 2, 2] tensor bytes.
// No RPC or production drafter state is accessed. Caller owns base lifetime.
dflash_gpu_conv_probe_result run_dflash_gpu_conv_probe(const float * base, size_t base_bytes);

// Opens the GGUF and reads blk.0.attn_conv_base after validating its F32
// [5120,2,2] metadata and the tensor's absolute file bounds.
dflash_gpu_conv_probe_result run_dflash_gpu_conv_probe(const std::string & model_path);
