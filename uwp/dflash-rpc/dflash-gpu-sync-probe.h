#pragma once

#include <cstdint>
#include <string>

struct llama_context;

struct dflash_gpu_sync_probe_result {
    bool passed = false;
    std::string json;
};

// Destructive diagnostic for an exclusive CPU DFlash split context: clears its
// sequence, executes real SYNC input through llama_decode, then compares the
// encoder's entire F32 output against a hardware-only D3D12 SYNC executor.
// Optional state cases repeat after trim and reset. This does not validate
// packed KV bytes or the DRAFT/attention graph.
dflash_gpu_sync_probe_result run_dflash_gpu_sync_probe(
    const std::string & model_path, llama_context * ctx,
    const int32_t * positions, const float * features, uint32_t count,
    bool exercise_state = true,
    float absolute_tolerance = 1.0e-3f,
    float relative_tolerance = 1.0e-3f,
    bool original_f32_matmul = false);
