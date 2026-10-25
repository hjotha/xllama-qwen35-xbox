#pragma once
#include "remote-attn-protocol.h"
#include <algorithm>
#include <cstdint>

namespace rkva_kvarn_range {
inline bool valid_descriptor(const rkva_kvarn_range_config& c, uint32_t dimension) {
    return c.storage == RKVA_RANGE_STORAGE_KVARN4 &&
           (c.layout_version == 1 || c.layout_version == 2) && c.origin % 128 == 0 &&
           c.capacity > 0 && c.capacity <= uint32_t(INT32_MAX) &&
           c.origin <= uint32_t(INT32_MAX) - c.capacity && dimension == 256 &&
           c.token_group == 128 && c.record_dim == 128 && c.head_dim == 256 && c.head_slices == 2 &&
           c.key_bits == 4 && c.value_bits == 4 && c.record_bytes == 8960 && c.stage_groups == 3 &&
           c.tail_groups == 2 && c.sink_tokens == 128 &&
           (c.precision_tail_tokens == 0 || c.precision_tail_tokens == 128) && c.wht_version == 1 &&
           c.domain == 1 && c.record_policy == 1;
}
inline uint32_t required_record_groups(uint32_t live) {
    return live / 128; // eager: every completed overflow group, including relative group 0
}
inline bool valid_upload(const rkva_kvarn_range_upload& u, uint32_t capacity,
                         uint32_t previous_live, uint32_t previous_groups, bool ready) {
    if (!u.live_count || u.live_count > capacity || (u.flags & ~1u) || u.reserved[0] ||
        u.reserved[1])
        return false;
    const uint32_t total_groups = (capacity + 127) / 128;
    if (u.record_begin > total_groups || u.record_groups > total_groups - u.record_begin)
        return false;
    const bool reset = (u.flags & 1) != 0;
    if (reset ? u.record_begin != 0
              : (!ready || u.live_count < previous_live || u.record_begin > previous_groups))
        return false;
    if (reset ? (u.append_begin || u.append_rows)
              : (u.append_begin != previous_live || u.append_rows != u.live_count - previous_live))
        return false;
    if (u.record_begin + u.record_groups > required_record_groups(u.live_count))
        return false;
    const uint32_t covered =
        reset ? u.record_groups : std::max(previous_groups, u.record_begin + u.record_groups);
    return covered >= required_record_groups(u.live_count);
}
template <typename Copy> void stage_chunks(uint32_t begin, uint32_t rows, Copy copy) {
    for (uint32_t row = 0; row < rows;) {
        const uint32_t token = begin + row;
        const uint32_t group = token / 128;
        const uint32_t slot = group == 0 ? 0 : 1 + ((group - 1) % 2);
        const uint32_t chunk = std::min(rows - row, 128 - token % 128);
        copy(row, slot * 128 + token % 128, chunk);
        row += chunk;
    }
}
template <typename Copy> void tail_chunks(uint32_t begin, uint32_t rows, Copy copy) {
    for (uint32_t row = rows > 128 ? rows - 128 : 0; row < rows;) {
        const uint32_t token = begin + row;
        const uint32_t chunk = std::min(rows - row, 128 - token % 128);
        copy(row, token % 128, chunk);
        row += chunk;
    }
}
} // namespace rkva_kvarn_range
