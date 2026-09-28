#pragma once

#include <cstdint>
#include <string>

struct llama_context;

struct DflashGpuDraftProbeResult {
    bool passed = false;
    std::string json;
};

// Destructive, exclusive diagnostic. Uses the already loaded CPU context as
// reference, then leaves it mutated. The RPC handler must close the test
// connection after this call; never run this in a production sequence.
// batch_mask bits 0..4 select batches {2,3,5,7,8}; zero selects all.
// capture_draft_ordinal selects a 1-based DRAFT in the trusted capture; zero
// keeps the short synthetic scenarios. The selected mode replays its prefix.
DflashGpuDraftProbeResult run_dflash_gpu_draft_probe(
    const std::string & model_path, const std::string & capture_path,
    llama_context * cpu_context, uint32_t batch_mask = 0,
    uint32_t capture_draft_ordinal = 0);
