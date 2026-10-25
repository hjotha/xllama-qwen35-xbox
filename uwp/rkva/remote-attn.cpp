// RKVA remote KVarN attention service — Xbox Series X side.
//
// Control-plane listener skeleton. The attention data plane (D3D12 KVarN
// store + flash-attention-over-records kernels) is implemented separately in
// remote-attn-engine.* and wired in once it passes the kernel-parity gate; until
// then ATTN_DECODE/ATTN_PREFILL are deliberately NOT advertised in HELLO and
// return RKVA_ERR_UNSUPPORTED_OP, so the host fails fast instead of consuming
// silently-wrong attention.
//
// Modeled on the proven DFL2 dedicated server (dedicated.cpp): single persistent
// client, strict request/response ordering, jsonl event log, TCP_NODELAY.

#include "remote-attn.h"
#include "remote-attn-engine.h"
#include "remote-attn-protocol.h"

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#include <immintrin.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include "xllama/session_hub.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#pragma comment(lib, "Ws2_32.lib")

namespace {

void evict_local_session() {
    std::lock_guard<std::mutex> lk(xllama::session_hub().mtx);
    xllama::session_hub().reset_locked();
}

using clock_type = std::chrono::steady_clock;

static inline float f16_to_f32(uint16_t h) {
    __m128i in = _mm_cvtsi32_si128(static_cast<int>(h));
    __m128 out = _mm_cvtph_ps(in);
    return _mm_cvtss_f32(out);
}

static inline uint16_t f32_to_f16(float f) {
    __m128 in = _mm_set_ss(f);
    __m128i out = _mm_cvtps_ph(in, 0);
    return static_cast<uint16_t>(_mm_cvtsi128_si32(out));
}

static inline void convert_f16_to_f32(const uint16_t* src, float* dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 out = _mm256_cvtph_ps(in);
        _mm256_storeu_ps(dst + i, out);
    }
    for (; i < count; ++i) {
        dst[i] = f16_to_f32(src[i]);
    }
}

static inline void convert_f32_to_f16(const float* src, uint16_t* dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 in = _mm256_loadu_ps(src + i);
        __m128i out = _mm256_cvtps_ph(in, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), out);
    }
    for (; i < count; ++i) {
        dst[i] = f32_to_f16(src[i]);
    }
}

uint64_t elapsed_us(clock_type::time_point start) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() -
                                                                           start)
        .count();
}

void event(const std::string& path, const std::string& message) {
    std::ofstream out(path, std::ios::app);
    if (out) {
        out << message << "\n";
    }
}

// Log path for the active connection, so handle_request (which has no path
// argument) can record engine/GPU errors.
std::string g_log_path;
std::string g_snapshot_local_folder;

void event_engine_error(const std::string& message) {
    std::string safe;
    safe.reserve(message.size());
    for (char c : message) {
        if (c == '"' || c == '\\') {
            safe += '\\';
        }
        safe += (c == '\n' || c == '\r') ? ' ' : c;
    }
    event(g_log_path, "{\"phase\":\"engine_error\",\"error\":\"" + safe + "\"}");
}

bool recv_all(SOCKET sock, void* dst, size_t size) {
    auto* p = static_cast<uint8_t*>(dst);
    while (size) {
        const int n = ::recv(sock, reinterpret_cast<char*>(p), (int)size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= (size_t)n;
    }
    return true;
}

bool send_all(SOCKET sock, const void* src, size_t size) {
    const auto* p = static_cast<const uint8_t*>(src);
    while (size) {
        const int n = ::send(sock, reinterpret_cast<const char*>(p), (int)size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= (size_t)n;
    }
    return true;
}

// One host context's KV state. The D3D12 engine (next phase) owns the actual
// record/stage/tail buffers; this skeleton tracks logical session state so the
// control plane (create/destroy/reset/trim/rewind) is fully exercisable.
struct remote_session {
    uint32_t session_id = 0;
    uint32_t seq_id = 0;
    uint32_t capacity = 0;
    int32_t high_water = -1; // highest stored position, -1 = empty
    uint64_t trims = 0;
    uint64_t resets = 0;
    bool valid = true;
    bool range_configured = false;
    bool range_session = false;
    uint32_t range_storage = 0;
    uint32_t range_origin = 0;
};

struct rpc_profile_counter {
    uint64_t calls = 0, request_bytes = 0, response_bytes = 0;
    uint64_t payload_receive_us = 0, handle_us = 0, send_enqueue_us = 0;
};

struct remote_state {
    bool hello_done = false;
    bool native_rewind_v2_client = false;
    rkva_hello geo{};
    std::map<uint32_t, remote_session> sessions;
    std::unique_ptr<RemoteAttnEngine> engine;
    bool engine_f16 = false; // cache_bits 0/16 -> F16 path; 2..8 -> KVarN (pending)
    std::mutex mutex;
    bool profile_enabled = false;
    bool trace_requests = false;
    rpc_profile_counter rpc_profile[32]{};
};

std::string rpc_profile_json(const remote_state& state) {
    std::string text = "{\"enabled\":" + std::string(state.profile_enabled ? "true" : "false") +
                       ",\"trace_requests\":" + (state.trace_requests ? "true" : "false") +
                       ",\"ops\":[";
    bool first = true;
    for (uint32_t op = 0; op < 32; ++op) {
        const auto& c = state.rpc_profile[op];
        if (!c.calls)
            continue;
        if (!first)
            text += ",";
        first = false;
        text += "{\"op\":" + std::to_string(op) + ",\"calls\":" + std::to_string(c.calls) +
                ",\"request_bytes\":" + std::to_string(c.request_bytes) +
                ",\"response_bytes\":" + std::to_string(c.response_bytes) +
                ",\"payload_receive_us\":" + std::to_string(c.payload_receive_us) +
                ",\"handle_us\":" + std::to_string(c.handle_us) +
                ",\"send_enqueue_us\":" + std::to_string(c.send_enqueue_us) + "}";
    }
    return text + "]}";
}

// Ops this server actually serves. The F16 attention path is live; the KVarN
// native path (cache_bits 2..8) advertises ATTN too but returns UNSUPPORTED at
// call time until its D3D12 kernels land, so a KVarN host fails fast rather
// than consuming wrong output.
constexpr uint32_t supported_ops_mask() {
    return (1u << RKVA_HELLO) | (1u << RKVA_CREATE_SESSION) | (1u << RKVA_DESTROY_SESSION) |
           (1u << RKVA_RESET) | (1u << RKVA_TRIM) | (1u << RKVA_REWIND) | (1u << RKVA_PING) |
           (1u << RKVA_GET_STATS) | (1u << RKVA_ATTN_DECODE) | (1u << RKVA_ATTN_PREFILL) |
           (1u << RKVA_RANGE_CONFIG) | (1u << RKVA_RANGE_UPLOAD) | (1u << RKVA_RANGE_ATTN) |
           (1u << RKVA_KVARN_RANGE_CONFIG) | (1u << RKVA_KVARN_RANGE_UPLOAD) |
           (1u << RKVA_SNAPSHOT_SAVE) | (1u << RKVA_SNAPSHOT_RESTORE) |
           (1u << RKVA_RANGE_UPLOAD_ATTN) | (1u << RKVA_RANGE_PROFILE) |
           (1u << RKVA_KVARN_RANGE_REWIND);
}

int handle_request(remote_state& state, const rkva_request& req, const std::vector<uint8_t>& input,
                   std::vector<uint8_t>& output, uint64_t& prepare_us, uint64_t& compute_us) {
    const auto t0 = clock_type::now();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (req.op >= RKVA_RANGE_CONFIG && req.version != RKVA_VERSION) {
        return RKVA_ERR_UNSUPPORTED_OP;
    }
    switch (req.op) {
    case RKVA_HELLO: {
        if (req.payload_bytes != sizeof(rkva_hello)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        rkva_hello hello{};
        std::memcpy(&hello, input.data(), sizeof(hello));
        // Validate the geometry we can actually serve.
        if (hello.head_dim == 0 || hello.n_head == 0 || hello.n_head_kv == 0 ||
            hello.n_head % hello.n_head_kv != 0 || hello.max_ctx == 0 ||
            (hello.head_dim != 128 && hello.head_dim != 256) || !hello.n_layer_remote ||
            hello.n_layer_remote > 128 || hello.n_head > 256 || hello.n_head_kv > 256 ||
            hello.max_ctx > uint32_t(INT32_MAX) || hello.has_sinks || hello.swa) {
            return RKVA_ERR_BAD_GEOMETRY;
        }
        state.sessions.clear();
        state.profile_enabled = state.trace_requests = false;
        for (auto& counter : state.rpc_profile)
            counter = {};
        state.geo = hello;
        state.hello_done = true;
        state.native_rewind_v2_client = (hello.proto_flags & RKVA_PROTO_NATIVE_REWIND_V2) != 0;

        evict_local_session();

        // cache_bits 0 or 16 -> F16 bring-up path; 2..8 -> KVarN native.
        state.engine_f16 = (hello.cache_bits_k == 0 || hello.cache_bits_k >= 16);

        RemoteAttnEngine::Geometry g{};
        g.n_head = hello.n_head;
        g.n_head_kv = hello.n_head_kv;
        g.head_dim = hello.head_dim;
        g.head_slices = hello.head_dim / 128;
        g.cache_bits_k = state.engine_f16 ? 16 : hello.cache_bits_k;
        g.cache_bits_v = state.engine_f16 ? 16 : hello.cache_bits_v;
        g.group_tokens = hello.group_tokens ? hello.group_tokens : 128;
        g.sinkhorn_iters = hello.sinkhorn_iters;
        g.tail_tokens = hello.tail_tokens;
        g.tail_groups = hello.tail_groups ? hello.tail_groups : 2;
        g.stage_groups = g.tail_groups + 1;
        g.domain = hello.domain;
        g.kq_scale = hello.kq_scale;
        g.max_ctx = hello.max_ctx;
        g.n_layers = hello.n_layer_remote;

        state.engine = std::make_unique<RemoteAttnEngine>();
        std::string engine_error;
        if (!state.engine->initialize(g, &engine_error)) {
            state.engine.reset();
            state.hello_done = false;
            event_engine_error(engine_error);
            return RKVA_ERR_GPU;
        }

        rkva_hello_ack ack{};
        ack.supported_ops = supported_ops_mask();
        if (req.version == RKVA_VERSION_LEGACY || !state.engine_f16 ||
            hello.domain != RKVA_DOMAIN_AUTO) {
            ack.supported_ops &= ~((1u << RKVA_RANGE_CONFIG) | (1u << RKVA_RANGE_UPLOAD) |
                                   (1u << RKVA_RANGE_ATTN) | (1u << RKVA_KVARN_RANGE_CONFIG) |
                                   (1u << RKVA_KVARN_RANGE_UPLOAD) | (1u << RKVA_SNAPSHOT_SAVE) |
                                   (1u << RKVA_SNAPSHOT_RESTORE) | (1u << RKVA_RANGE_UPLOAD_ATTN) |
                                   (1u << RKVA_RANGE_PROFILE) | (1u << RKVA_KVARN_RANGE_REWIND));
        }
        // Choose wire format: if host supports F16 wire (proto_flags bit 0), negotiate F16 for 2x
        // bandwidth savings
        ack.chosen_wire =
            (hello.proto_flags & (1u << RKVA_WIRE_F16)) ? RKVA_WIRE_F16 : RKVA_WIRE_F32;
        if (req.version == RKVA_VERSION && state.engine_f16 && hello.domain == RKVA_DOMAIN_AUTO) {
            ack.reserved[0] = RKVA_CAP_RANGE_Q4_0 | RKVA_CAP_RANGE_Q8_0 | RKVA_CAP_RANGE_KVARN4 |
                              RKVA_CAP_DISK_SNAPSHOT | RKVA_CAP_RANGE_UPLOAD_ATTN |
                              RKVA_CAP_RANGE_PROFILE;
            if (state.native_rewind_v2_client)
                ack.reserved[0] |= RKVA_CAP_KVARN_REWIND;
        }
        if (!state.native_rewind_v2_client)
            ack.supported_ops &= ~(1u << RKVA_KVARN_RANGE_REWIND);
        ack.max_sessions = 64;
        ack.server_version = 262; // bumped with the UWP package
        // ~5 GiB game heap; leave headroom for the D3D12 runtime + buffers.
        ack.mem_budget_bytes = 4ull * 1024 * 1024 * 1024;
        output.resize(sizeof(ack));
        std::memcpy(output.data(), &ack, sizeof(ack));
        return RKVA_OK;
    }
    case RKVA_PING: {
        return RKVA_OK;
    }
    case RKVA_CREATE_SESSION: {
        if (req.payload_bytes != sizeof(rkva_session_req)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        evict_local_session();
        rkva_session_req sr{};
        std::memcpy(&sr, input.data(), sizeof(sr));
        if (!state.hello_done || !state.engine) {
            return RKVA_ERR_STATE;
        }
        if (req.version == RKVA_VERSION && (sr.flags & ~RKVA_SESSION_RANGE)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const bool defer = req.version == RKVA_VERSION && (sr.flags & RKVA_SESSION_RANGE);
        if (defer && (!state.engine_f16 || state.geo.domain != RKVA_DOMAIN_AUTO)) {
            return RKVA_ERR_BAD_GEOMETRY;
        }
        remote_session s;
        s.range_session = defer;
        s.session_id = sr.session_id;
        s.seq_id = sr.seq_id;
        s.capacity = sr.capacity ? sr.capacity : state.geo.max_ctx;
        std::string engine_error;
        if (!state.engine->create_session(sr.session_id, s.capacity, &engine_error, defer)) {
            event_engine_error(engine_error);
            return RKVA_ERR_MEMORY;
        }
        state.sessions[sr.session_id] = s;
        return RKVA_OK;
    }
    case RKVA_DESTROY_SESSION: {
        state.sessions.erase(req.session_id);
        if (state.engine) {
            state.engine->destroy_session(req.session_id);
        }
        return RKVA_OK;
    }
    case RKVA_RESET: {
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end()) {
            return RKVA_ERR_BAD_SESSION;
        }
        it->second.high_water = -1;
        it->second.resets++;
        std::string engine_error;
        if (state.engine && !state.engine->reset_session(req.session_id, &engine_error)) {
            event_engine_error(engine_error);
            return RKVA_ERR_GPU;
        }
        // Disk restore can fail after partial GPU copies. A successful RESET
        // makes all old rows unreachable before authoritative host reupload.
        it->second.valid = true;
        return RKVA_OK;
    }
    case RKVA_TRIM:
    case RKVA_REWIND: {
        if (req.payload_bytes != sizeof(rkva_trim_req)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        rkva_trim_req tr{};
        std::memcpy(&tr, input.data(), sizeof(tr));
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end()) {
            return RKVA_ERR_BAD_SESSION;
        }
        if (tr.pos0 < 0) {
            return RKVA_ERR_BAD_POSITION;
        }
        // Remove every stored position >= pos0 (logical, mirrors DFL2 trim).
        if (it->second.high_water >= tr.pos0) {
            it->second.high_water = tr.pos0 - 1;
        }
        it->second.trims++;
        std::string engine_error;
        if (state.engine && !state.engine->trim_session(req.session_id, tr.pos0, &engine_error)) {
            event_engine_error(engine_error);
            return RKVA_ERR_GPU;
        }
        return RKVA_OK;
    }
    case RKVA_GET_STATS: {
        uint64_t trims = 0, resets = 0;
        for (const auto& kv : state.sessions) {
            trims += kv.second.trims;
            resets += kv.second.resets;
        }
        const unsigned long long engine_bytes =
            state.engine ? (unsigned long long)state.engine->allocated_bytes() : 0ull;
        char buf[640];
        snprintf(buf, sizeof(buf),
                 "{\"service\":\"rkva\",\"hello\":%s,\"sessions\":%zu,\"trims\":%llu,"
                 "\"resets\":%llu,\"engine\":\"%s\",\"engine_kv_bytes\":%llu}",
                 state.hello_done ? "true" : "false", state.sessions.size(),
                 (unsigned long long)trims, (unsigned long long)resets,
                 state.engine_f16 ? "d3d12-f16" : "d3d12-kvarn", engine_bytes);
        const size_t n = std::strlen(buf);
        output.assign(buf, buf + n);
        return RKVA_OK;
    }
    case RKVA_ATTN_DECODE:
    case RKVA_ATTN_PREFILL: {
        if (!state.hello_done || !state.engine) {
            return RKVA_ERR_STATE;
        }
        // KVarN (cache_bits 2..8) and F16 paths both run the D3D12 engine.
        if (state.sessions.find(req.session_id) == state.sessions.end()) {
            return RKVA_ERR_BAD_SESSION;
        }
        if (state.sessions.at(req.session_id).range_configured ||
            state.sessions.at(req.session_id).range_session) {
            return RKVA_ERR_STATE;
        }
        if (input.size() < sizeof(rkva_attn_req)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint32_t n_tokens = req.n_tokens;
        const uint32_t n_head = state.geo.n_head;
        const uint32_t n_head_kv = state.geo.n_head_kv;
        const uint32_t head_dim = state.geo.head_dim;
        if (n_tokens == 0) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint64_t q_elems = uint64_t(n_tokens) * n_head * head_dim;
        const uint64_t kv_elems = uint64_t(n_tokens) * n_head_kv * head_dim;

        rkva_attn_req areq{};
        std::memcpy(&areq, input.data(), sizeof(areq));
        const bool is_f16 = (areq.wire == RKVA_WIRE_F16);
        if (areq.wire != RKVA_WIRE_F32 && areq.wire != RKVA_WIRE_F16) {
            return RKVA_ERR_UNSUPPORTED_OP;
        }
        const uint64_t elem_size = is_f16 ? 2ull : 4ull;
        const uint64_t expected =
            sizeof(rkva_attn_req) + uint64_t(n_tokens) * 4 + (q_elems + 2 * kv_elems) * elem_size;
        if (req.payload_bytes != expected) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint8_t* p = input.data() + sizeof(rkva_attn_req);
        const int32_t* positions = reinterpret_cast<const int32_t*>(p);
        p += uint64_t(n_tokens) * 4;

        std::vector<float> q_f32, k_f32, v_f32;
        const float* q_ptr = nullptr;
        const float* k_ptr = nullptr;
        const float* v_ptr = nullptr;

        if (is_f16) {
            const uint16_t* q_f16 = reinterpret_cast<const uint16_t*>(p);
            p += q_elems * 2;
            const uint16_t* k_f16 = reinterpret_cast<const uint16_t*>(p);
            p += kv_elems * 2;
            const uint16_t* v_f16 = reinterpret_cast<const uint16_t*>(p);

            q_f32.resize(q_elems);
            k_f32.resize(kv_elems);
            v_f32.resize(kv_elems);

            convert_f16_to_f32(q_f16, q_f32.data(), q_elems);
            convert_f16_to_f32(k_f16, k_f32.data(), kv_elems);
            convert_f16_to_f32(v_f16, v_f32.data(), kv_elems);
            q_ptr = q_f32.data();
            k_ptr = k_f32.data();
            v_ptr = v_f32.data();
        } else {
            q_ptr = reinterpret_cast<const float*>(p);
            p += q_elems * 4;
            k_ptr = reinterpret_cast<const float*>(p);
            p += kv_elems * 4;
            v_ptr = reinterpret_cast<const float*>(p);
        }

        std::vector<float> out_f32_buf;
        float* out_target = nullptr;
        if (is_f16) {
            out_f32_buf.resize(q_elems);
            out_target = out_f32_buf.data();
        } else {
            output.resize(q_elems * 4);
            out_target = reinterpret_cast<float*>(output.data());
        }

        const auto compute_start = clock_type::now();
        std::string engine_error;
        const bool ok = state.engine->attend(req.session_id, req.layer_id, n_tokens, positions,
                                             q_ptr, k_ptr, v_ptr, out_target, &engine_error);
        compute_us = elapsed_us(compute_start);
        if (!ok) {
            output.clear();
            event_engine_error(engine_error);
            auto it = state.sessions.find(req.session_id);
            if (it != state.sessions.end()) {
                it->second.valid = false;
            }
            return RKVA_ERR_GPU;
        }

        if (is_f16) {
            output.resize(q_elems * 2);
            uint16_t* out_f16 = reinterpret_cast<uint16_t*>(output.data());
            convert_f32_to_f16(out_target, out_f16, q_elems);
        }
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_RANGE_CONFIG: {
        if (input.size() != sizeof(rkva_range_config)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !state.engine || !state.engine_f16) {
            return RKVA_ERR_BAD_SESSION;
        }
        rkva_range_config config{};
        std::memcpy(&config, input.data(), sizeof(config));
        if ((config.reserved[0] & ~(RKVA_RANGE_STORAGE_Q4_0 | RKVA_RANGE_STORAGE_Q8_0)) ||
            config.reserved[0] == (RKVA_RANGE_STORAGE_Q4_0 | RKVA_RANGE_STORAGE_Q8_0) ||
            config.reserved[1] || config.capacity != it->second.capacity ||
            config.origin > uint32_t(INT32_MAX) - config.capacity) {
            return RKVA_ERR_BAD_POSITION;
        }
        std::string error;
        if (!state.engine->configure_range(req.session_id, config.origin, config.capacity, &error,
                                           config.reserved[0])) {
            event_engine_error(error);
            return RKVA_ERR_STATE;
        }
        it->second.range_configured = true;
        it->second.range_storage = config.reserved[0];
        it->second.range_origin = config.origin;
        return RKVA_OK;
    }
    case RKVA_RANGE_UPLOAD: {
        if (input.size() < sizeof(rkva_range_upload)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !it->second.range_configured || !state.engine ||
            !it->second.valid) {
            return RKVA_ERR_BAD_SESSION;
        }
        rkva_range_upload upload{};
        std::memcpy(&upload, input.data(), sizeof(upload));
        if (it->second.range_storage == RKVA_RANGE_STORAGE_KVARN4 ||
            upload.wire != (it->second.range_storage == RKVA_RANGE_STORAGE_Q8_0 ? RKVA_WIRE_Q8_0
                            : it->second.range_storage == RKVA_RANGE_STORAGE_Q4_0
                                ? RKVA_WIRE_Q4_0
                                : RKVA_WIRE_F16) ||
            upload.reserved || !upload.n_rows || upload.start_row > it->second.capacity ||
            upload.n_rows > it->second.capacity - upload.start_row ||
            req.layer_id >= state.geo.n_layer_remote) {
            return RKVA_ERR_BAD_POSITION;
        }
        const uint64_t plane_bytes =
            uint64_t(upload.n_rows) * state.geo.n_head_kv *
            (it->second.range_storage
                 ? uint64_t(state.geo.head_dim / 32) *
                       (it->second.range_storage == RKVA_RANGE_STORAGE_Q4_0 ? 18 : 34)
                 : uint64_t(state.geo.head_dim) * 2);
        if (input.size() != sizeof(upload) + 2 * plane_bytes) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint8_t* data = input.data() + sizeof(upload);
        std::string error;
        const auto start = clock_type::now();
        if (!state.engine->upload_range(req.session_id, req.layer_id, upload.start_row,
                                        upload.n_rows, reinterpret_cast<const uint16_t*>(data),
                                        reinterpret_cast<const uint16_t*>(data + plane_bytes),
                                        &error)) {
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_POSITION;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_KVARN_RANGE_CONFIG: {
        if (req.version != RKVA_VERSION || input.size() != sizeof(rkva_kvarn_range_config))
            return RKVA_ERR_BAD_PAYLOAD;
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !state.engine || !state.engine_f16 || !it->second.valid)
            return RKVA_ERR_BAD_SESSION;
        rkva_kvarn_range_config c{};
        std::memcpy(&c, input.data(), sizeof(c));
        if (c.layout_version == 2 && !state.native_rewind_v2_client)
            return RKVA_ERR_BAD_PAYLOAD;
        std::string error;
        if (!state.engine->configure_kvarn_range(req.session_id, c, &error)) {
            event_engine_error(error);
            return RKVA_ERR_BAD_GEOMETRY;
        }
        it->second.range_configured = true;
        it->second.range_storage = RKVA_RANGE_STORAGE_KVARN4;
        if (c.layout_version == 2) {
            const uint64_t generation = state.engine->native_range_generation(req.session_id);
            output.resize(sizeof(generation));
            std::memcpy(output.data(), &generation, sizeof(generation));
        }
        return RKVA_OK;
    }
    case RKVA_KVARN_RANGE_UPLOAD: {
        if (req.version != RKVA_VERSION || input.size() < sizeof(rkva_kvarn_range_upload))
            return RKVA_ERR_BAD_PAYLOAD;
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !state.engine || !it->second.valid ||
            !it->second.range_configured || it->second.range_storage != RKVA_RANGE_STORAGE_KVARN4)
            return RKVA_ERR_BAD_SESSION;
        rkva_kvarn_range_upload u{};
        std::memcpy(&u, input.data(), sizeof(u));
        std::string error;
        const auto start = clock_type::now();
        if (!state.engine->upload_kvarn_range(req.session_id, req.layer_id, u,
                                              input.data() + sizeof(u), input.size() - sizeof(u),
                                              &error)) {
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_PAYLOAD;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_RANGE_ATTN: {
        if (input.size() < sizeof(rkva_range_attn)) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !it->second.range_configured || !state.engine ||
            !it->second.valid) {
            return RKVA_ERR_BAD_SESSION;
        }
        rkva_range_attn attn{};
        std::memcpy(&attn, input.data(), sizeof(attn));
        if (attn.wire != RKVA_WIRE_F32 || attn.flags || attn.reserved[0] || attn.reserved[1] ||
            !req.n_tokens || req.n_tokens > 1024 || req.layer_id >= state.geo.n_layer_remote) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const uint64_t q_heads = uint64_t(req.n_tokens) * state.geo.n_head;
        const uint64_t q_bytes = q_heads * state.geo.head_dim * 4;
        if (input.size() != sizeof(attn) + uint64_t(req.n_tokens) * 4 + q_bytes) {
            return RKVA_ERR_BAD_PAYLOAD;
        }
        const auto* positions = reinterpret_cast<const int32_t*>(input.data() + sizeof(attn));
        const auto* q = reinterpret_cast<const float*>(input.data() + sizeof(attn) +
                                                       uint64_t(req.n_tokens) * 4);
        for (uint32_t t = 0; t < req.n_tokens; ++t) {
            if (positions[t] < 0)
                return RKVA_ERR_BAD_POSITION;
        }
        output.resize(q_bytes + q_heads * 4);
        auto* out = reinterpret_cast<float*>(output.data());
        auto* lse = reinterpret_cast<float*>(output.data() + q_bytes);
        std::string error;
        const auto start = clock_type::now();
        if (!state.engine->attend_range(req.session_id, req.layer_id, req.n_tokens, positions, q,
                                        out, lse, &error)) {
            output.clear();
            event_engine_error(error);
            return RKVA_ERR_GPU;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_KVARN_RANGE_REWIND: {
        if (!state.hello_done || !state.native_rewind_v2_client ||
            input.size() < sizeof(rkva_kvarn_range_rewind) || req.layer_id || req.n_tokens ||
            !state.engine || !state.engine_f16)
            return RKVA_ERR_BAD_PAYLOAD;
        const auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !it->second.valid || !it->second.range_configured)
            return RKVA_ERR_BAD_SESSION;
        rkva_kvarn_range_rewind rewind{};
        std::memcpy(&rewind, input.data(), sizeof(rewind));
        std::string error;
        const auto start = clock_type::now();
        if (!state.engine->rewind_kvarn_range(req.session_id, rewind, input.data() + sizeof(rewind),
                                              input.size() - sizeof(rewind), &error)) {
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_PAYLOAD;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_RANGE_PROFILE: {
        if (!state.hello_done || !state.engine || !state.engine_f16 ||
            input.size() != sizeof(rkva_range_profile) || req.n_tokens || req.layer_id)
            return RKVA_ERR_BAD_PAYLOAD;
        rkva_range_profile control{};
        std::memcpy(&control, input.data(), sizeof(control));
        constexpr uint32_t allowed = RKVA_PROFILE_ENABLE | RKVA_PROFILE_RESET | RKVA_PROFILE_TRACE |
                                     RKVA_PROFILE_REPORT_ONLY;
        if (control.reserved || (control.flags & ~allowed) || control.q4_kernel_mode > 3 ||
            ((control.flags & RKVA_PROFILE_REPORT_ONLY) &&
             (control.flags != RKVA_PROFILE_REPORT_ONLY || control.q4_kernel_mode)))
            return RKVA_ERR_BAD_PAYLOAD;
        if (!(control.flags & RKVA_PROFILE_REPORT_ONLY)) {
            std::string error;
            if (!state.engine->set_range_profiling((control.flags & RKVA_PROFILE_ENABLE) != 0,
                                                   (control.flags & RKVA_PROFILE_RESET) != 0,
                                                   control.q4_kernel_mode, error)) {
                event_engine_error(error);
                return RKVA_ERR_STATE;
            }
            state.profile_enabled = (control.flags & RKVA_PROFILE_ENABLE) != 0;
            state.trace_requests = (control.flags & RKVA_PROFILE_TRACE) != 0;
            if (control.flags & RKVA_PROFILE_RESET)
                for (auto& counter : state.rpc_profile)
                    counter = {};
        }
        const std::string engine = state.engine->get_range_profile_json();
        const std::string report = "{\"engine\":" + (engine.empty() ? "{}" : engine) +
                                   ",\"transport\":" + rpc_profile_json(state) + "}";
        if (report.size() > (1u << 20))
            return RKVA_ERR_MEMORY;
        output.assign(report.begin(), report.end());
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_RANGE_UPLOAD_ATTN: {
        if (input.size() < sizeof(rkva_range_upload_attn))
            return RKVA_ERR_BAD_PAYLOAD;
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !it->second.range_configured || !state.engine ||
            !it->second.valid || it->second.range_storage == RKVA_RANGE_STORAGE_KVARN4)
            return RKVA_ERR_BAD_SESSION;
        rkva_range_upload_attn combined{};
        std::memcpy(&combined, input.data(), sizeof(combined));
        const auto& upload = combined.upload;
        const auto& attn = combined.attention;
        const uint32_t wire = it->second.range_storage == RKVA_RANGE_STORAGE_Q8_0   ? RKVA_WIRE_Q8_0
                              : it->second.range_storage == RKVA_RANGE_STORAGE_Q4_0 ? RKVA_WIRE_Q4_0
                                                                                    : RKVA_WIRE_F16;
        if (upload.wire != wire || upload.reserved || !upload.n_rows ||
            upload.start_row > it->second.capacity ||
            upload.n_rows > it->second.capacity - upload.start_row || attn.wire != RKVA_WIRE_F32 ||
            attn.flags || attn.reserved[0] || attn.reserved[1] || !req.n_tokens ||
            req.n_tokens > 1024 || req.layer_id >= state.geo.n_layer_remote)
            return RKVA_ERR_BAD_PAYLOAD;
        const uint64_t row_bytes = uint64_t(state.geo.n_head_kv) *
                                   (wire == RKVA_WIRE_F16 ? uint64_t(state.geo.head_dim) * 2
                                                          : uint64_t(state.geo.head_dim / 32) *
                                                                (wire == RKVA_WIRE_Q4_0 ? 18 : 34));
        const uint64_t plane_bytes = uint64_t(upload.n_rows) * row_bytes;
        const uint64_t q_heads = uint64_t(req.n_tokens) * state.geo.n_head;
        const uint64_t q_bytes = q_heads * state.geo.head_dim * sizeof(float);
        const uint64_t positions_bytes = uint64_t(req.n_tokens) * sizeof(int32_t);
        if (input.size() != sizeof(combined) + 2 * plane_bytes + positions_bytes + q_bytes)
            return RKVA_ERR_BAD_PAYLOAD;
        const auto* k = input.data() + sizeof(combined);
        const auto* v = k + plane_bytes;
        const auto* positions = reinterpret_cast<const int32_t*>(v + plane_bytes);
        const auto* q = reinterpret_cast<const float*>(v + plane_bytes + positions_bytes);
        const uint64_t end = uint64_t(it->second.range_origin) + upload.start_row + upload.n_rows;
        for (uint32_t t = 0; t < req.n_tokens; ++t) {
            if (positions[t] < int64_t(it->second.range_origin) || uint64_t(positions[t]) >= end ||
                (t && int64_t(positions[t]) != int64_t(positions[t - 1]) + 1))
                return RKVA_ERR_BAD_POSITION;
        }
        if (uint64_t(positions[req.n_tokens - 1]) + 1 != end)
            return RKVA_ERR_BAD_POSITION;
        output.resize(q_bytes + q_heads * sizeof(float));
        auto* out = reinterpret_cast<float*>(output.data());
        auto* lse = reinterpret_cast<float*>(output.data() + q_bytes);
        std::string error;
        const auto start = clock_type::now();
        // First isolate network batching: retain the qualified upload/attention
        // engine calls. Queue/fence coalescing is measured as a separate change.
        if (!state.engine->upload_range(req.session_id, req.layer_id, upload.start_row,
                                        upload.n_rows, reinterpret_cast<const uint16_t*>(k),
                                        reinterpret_cast<const uint16_t*>(v), &error) ||
            !state.engine->attend_range(req.session_id, req.layer_id, req.n_tokens, positions, q,
                                        out, lse, &error)) {
            output.clear();
            event_engine_error(error);
            return state.engine->session_invalid(req.session_id) ? RKVA_ERR_GPU
                                                                 : RKVA_ERR_BAD_POSITION;
        }
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        return RKVA_OK;
    }
    case RKVA_SNAPSHOT_SAVE:
    case RKVA_SNAPSHOT_RESTORE: {
        if (input.size() != sizeof(rkva_snapshot_req) || req.n_tokens || req.layer_id)
            return RKVA_ERR_BAD_PAYLOAD;
        auto it = state.sessions.find(req.session_id);
        if (it == state.sessions.end() || !state.engine || !it->second.valid ||
            !it->second.range_configured)
            return RKVA_ERR_BAD_SESSION;
        rkva_snapshot_req snapshot{};
        std::memcpy(&snapshot, input.data(), sizeof(snapshot));
        if (snapshot.revision != 1 || snapshot.flags || snapshot.reserved || !snapshot.live_rows ||
            snapshot.live_rows > it->second.capacity)
            return RKVA_ERR_BAD_PAYLOAD;
        rkva_snapshot_ack ack{};
        std::string error;
        const bool restoring = req.op == RKVA_SNAPSHOT_RESTORE;
        const auto start = clock_type::now();
        if (!state.engine->disk_snapshot(req.session_id, g_snapshot_local_folder, snapshot,
                                         restoring, &ack, &error)) {
            event_engine_error(error);
            if (state.engine->session_invalid(req.session_id)) {
                it->second.valid = false;
                return RKVA_ERR_STATE;
            }
            return restoring ? RKVA_ERR_SNAPSHOT_MISS : RKVA_ERR_SNAPSHOT_IO;
        }
        // Engine publishes counters only after all GPU transfers complete.
        it->second.high_water = -1; // absolute range origin is owned by the engine
        compute_us = elapsed_us(start);
        prepare_us = elapsed_us(t0);
        output.resize(sizeof(ack));
        std::memcpy(output.data(), &ack, sizeof(ack));
        return RKVA_OK;
    }
    case RKVA_PROBE:
    default:
        // Parity harness lands with the KVarN kernels; explicit failure now.
        (void)compute_us;
        prepare_us = elapsed_us(t0);
        return RKVA_ERR_UNSUPPORTED_OP;
    }
    prepare_us = elapsed_us(t0);
}

void client_loop(SOCKET sock, remote_state& state, const std::string& log_path) {
    uint64_t previous_response_us = 0;
    std::vector<uint8_t> input, output;
    for (;;) {
        const auto start = clock_type::now();
        rkva_request req{};
        if (!recv_all(sock, &req, sizeof(req))) {
            break;
        }
        const uint64_t idle_us = elapsed_us(start);
        // Generous cap: prefill batches can be multi-MB (Q/K/V rows).
        static constexpr uint32_t MAX_PAYLOAD = 256u * 1024u * 1024u;
        if (req.magic != RKVA_MAGIC ||
            (req.version != RKVA_VERSION && req.version != RKVA_VERSION_LEGACY) ||
            req.payload_bytes > MAX_PAYLOAD) {
            break;
        }
        input.resize(req.payload_bytes);
        const auto payload_start = clock_type::now();
        if (req.payload_bytes && !recv_all(sock, input.data(), input.size())) {
            break;
        }
        const uint64_t payload_receive_us = elapsed_us(payload_start);
        output.clear();
        uint64_t prepare_us = 0, compute_us = 0;
        const auto handle_start = clock_type::now();
        const int rc = handle_request(state, req, input, output, prepare_us, compute_us);
        const uint64_t handle_us = elapsed_us(handle_start);

        const rkva_response rsp{
            RKVA_MAGIC, req.version, req.op,     req.cycle_id,        rc, (uint32_t)output.size(),
            idle_us,    prepare_us,  compute_us, previous_response_us};
        const auto response_start = clock_type::now();
        if (!send_all(sock, &rsp, sizeof(rsp)) ||
            (output.size() && !send_all(sock, output.data(), output.size()))) {
            break;
        }
        previous_response_us = elapsed_us(response_start);
        if (state.profile_enabled && req.op != RKVA_RANGE_PROFILE && req.op < 32) {
            auto& counter = state.rpc_profile[req.op];
            ++counter.calls;
            counter.request_bytes += sizeof(req) + input.size();
            counter.response_bytes += sizeof(rsp) + output.size();
            counter.payload_receive_us += payload_receive_us;
            counter.handle_us += handle_us;
            counter.send_enqueue_us += previous_response_us;
        }
        if (state.trace_requests)
            event(log_path, "{\"phase\":\"request\",\"cycleId\":" + std::to_string(req.cycle_id) +
                                ",\"op\":" + std::to_string(req.op) +
                                ",\"session\":" + std::to_string(req.session_id) +
                                ",\"layer\":" + std::to_string(req.layer_id) +
                                ",\"nTokens\":" + std::to_string(req.n_tokens) +
                                ",\"payloadBytes\":" + std::to_string(req.payload_bytes) +
                                ",\"responseBytes\":" + std::to_string(output.size()) +
                                ",\"status\":" + std::to_string(rc) +
                                ",\"idleUs\":" + std::to_string(idle_us) +
                                ",\"prepareUs\":" + std::to_string(prepare_us) +
                                ",\"computeUs\":" + std::to_string(compute_us) +
                                ",\"handleUs\":" + std::to_string(handle_us) +
                                ",\"responseUs\":" + std::to_string(previous_response_us) + "}");
    }
}

} // namespace

void serve_remote_attn(const std::string& local_folder) {
    const std::string log_path = local_folder + "\\remote-attn.jsonl";
    g_log_path = log_path;
    g_snapshot_local_folder = local_folder;
    constexpr uint16_t port = 50054;

    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa)) {
        event(log_path, "{\"phase\":\"listen\",\"status\":\"wsa_failed\"}");
        return;
    }
    SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listener == INVALID_SOCKET) {
        event(log_path, "{\"phase\":\"listen\",\"status\":\"socket_failed\"}");
        WSACleanup();
        return;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR ||
        listen(listener, 1) == SOCKET_ERROR) {
        event(log_path, "{\"phase\":\"listen\",\"status\":\"bind_failed\",\"port\":" +
                            std::to_string(port) + "}");
        closesocket(listener);
        WSACleanup();
        return;
    }
    event(log_path,
          "{\"phase\":\"listen\",\"status\":\"ready\",\"port\":" + std::to_string(port) + "}");

    remote_state state;
    for (;;) {
        SOCKET client = accept(listener, nullptr, nullptr);
        if (client == INVALID_SOCKET) {
            break;
        }
        const int one = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one),
                   sizeof(one));
        int rcvbuf = 8 << 20, sndbuf = 8 << 20;
        setsockopt(client, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&rcvbuf),
                   sizeof(rcvbuf));
        setsockopt(client, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&sndbuf),
                   sizeof(sndbuf));
        event(log_path, "{\"phase\":\"accept\",\"status\":\"connected\"}");
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.hello_done = false;
            state.sessions.clear();
            state.engine.reset(); // freed and re-created at the next HELLO
            state.engine_f16 = false;
        }
        client_loop(client, state, log_path);
        closesocket(client);
        event(log_path, "{\"phase\":\"accept\",\"status\":\"disconnected\"}");
    }
    closesocket(listener);
    WSACleanup();
}
