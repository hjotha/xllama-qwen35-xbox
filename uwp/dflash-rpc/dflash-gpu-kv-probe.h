#pragma once

#include <string>

struct dflash_gpu_kv_probe_result {
    bool passed = false;
    std::string json;
};

// Isolated hardware diagnostic; no RPC or production cache state is touched.
dflash_gpu_kv_probe_result run_dflash_gpu_kv_probe();
