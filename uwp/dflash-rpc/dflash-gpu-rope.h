#pragma once

#include <cmath>
#include <cstdint>

// Match GGML's temporal M-RoPE [64,0,0,0] cache. Only position constants
// cross the CPU/GPU boundary; hidden states stay on the GPU.
inline void dflash_rope_cache(float* cache, uint32_t position, float base) {
    const float step = std::pow(base, -2.0f / 128.0f);
    float theta = static_cast<float>(position);
    for (uint32_t i = 0; i < 64; ++i) {
        cache[2 * i] = std::cos(theta);
        cache[2 * i + 1] = std::sin(theta);
        theta *= step;
    }
}
