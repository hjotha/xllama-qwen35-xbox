// Standalone: cl /EHsc /std:c++17 test_rpc_validation.cpp && test_rpc_validation.exe
#include "../../uwp/dflash-rpc/dedicated_protocol.h"
#include <cassert>
#include <limits>

int main() {
    float embeddings[2] = {0.25f, -0.5f};
    const auto* bytes = reinterpret_cast<const uint8_t*>(embeddings);
    assert(dflash_rpc_valid_draft({49150, 1, 1}, bytes, 1, 49152));
    assert(!dflash_rpc_valid_draft({49151, 1, 1}, bytes, 1, 49152));
    assert(!dflash_rpc_valid_draft({INT32_MAX, 1, 7}, bytes, 1, 49152));
    assert(!dflash_rpc_valid_draft({-1, 1, 1}, bytes, 1, 49152));
    embeddings[1] = std::numeric_limits<float>::quiet_NaN();
    assert(!dflash_rpc_valid_draft({0, 1, 1}, bytes, 1, 49152));
    struct row {
        int32_t pos;
        float feature;
    } rows[] = {{0, 1}, {49151, 2}};
    const auto* sync = reinterpret_cast<const uint8_t*>(rows);
    assert(dflash_rpc_valid_sync(sync, 2, 1, 49152));
    rows[1].pos = 0;
    assert(!dflash_rpc_valid_sync(sync, 2, 1, 49152));
    rows[1].pos = 49152;
    assert(!dflash_rpc_valid_sync(sync, 2, 1, 49152));
    rows[1].pos = -1;
    assert(!dflash_rpc_valid_sync(sync, 2, 1, 49152));
    rows[1].pos = 1;
    rows[1].feature = std::numeric_limits<float>::infinity();
    assert(!dflash_rpc_valid_sync(sync, 2, 1, 49152));
}
