#pragma once

#include <string>
#include "xllama/gpugemv.h"

struct dflash_gpu_probe_result {
    bool passed = false;
    std::string json;
};

dflash_gpu_probe_result run_dflash_gpu_probe();
dflash_gpu_probe_result run_dflash_gpu_real_q4k(const std::string & model_path);
dflash_gpu_probe_result run_dflash_gpu_real_quant(const std::string & model_path,
                                                  xllama::GpugemvQuant quant, int batch = 1);
