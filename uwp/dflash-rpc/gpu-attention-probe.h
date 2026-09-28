#pragma once

#include <string>

struct dflash_gpu_attention_probe_result {
    bool passed = false;
    std::string json;
};

// Hardware-only, isolated diagnostic. The caller may log .json verbatim.
// Does not mutate the RPC executor or any production KV state.
dflash_gpu_attention_probe_result run_dflash_gpu_attention_probe();
