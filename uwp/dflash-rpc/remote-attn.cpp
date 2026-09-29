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
#include "remote-attn-protocol.h"

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#include <winsock2.h>
#include <ws2tcpip.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#pragma comment(lib, "Ws2_32.lib")

namespace {

using clock_type = std::chrono::steady_clock;

uint64_t elapsed_us(clock_type::time_point start) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        clock_type::now() - start).count();
}

void event(const std::string& path, const std::string& message) {
    std::ofstream out(path, std::ios::app);
    if (out) {
        out << message << "\n";
    }
}

bool recv_all(SOCKET sock, void* dst, size_t size) {
    auto* p = static_cast<uint8_t*>(dst);
    while (size) {
        const int n = ::recv(sock, reinterpret_cast<char*>(p), (int) size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= (size_t) n;
    }
    return true;
}

bool send_all(SOCKET sock, const void* src, size_t size) {
    const auto* p = static_cast<const uint8_t*>(src);
    while (size) {
        const int n = ::send(sock, reinterpret_cast<const char*>(p), (int) size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= (size_t) n;
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
    int32_t  high_water = -1;   // highest stored position, -1 = empty
    uint64_t trims = 0;
    uint64_t resets = 0;
    bool     valid = true;
};

struct remote_state {
    bool hello_done = false;
    rkva_hello geo {};
    std::map<uint32_t, remote_session> sessions;
    std::mutex mutex;
};

// Ops this skeleton actually serves. ATTN_DECODE/ATTN_PREFILL are intentionally
// omitted until the D3D12 KVarN kernels land (host then fails fast at HELLO).
constexpr uint32_t supported_ops_mask() {
    return (1u << RKVA_HELLO) | (1u << RKVA_CREATE_SESSION) |
           (1u << RKVA_DESTROY_SESSION) | (1u << RKVA_RESET) |
           (1u << RKVA_TRIM) | (1u << RKVA_REWIND) |
           (1u << RKVA_PING) | (1u << RKVA_GET_STATS);
}

int handle_request(remote_state& state, const rkva_request& req,
                   const std::vector<uint8_t>& input, std::vector<uint8_t>& output,
                   uint64_t& prepare_us, uint64_t& compute_us) {
    const auto t0 = clock_type::now();
    std::lock_guard<std::mutex> lock(state.mutex);

    switch (req.op) {
        case RKVA_HELLO: {
            if (req.payload_bytes != sizeof(rkva_hello)) {
                return RKVA_ERR_BAD_PAYLOAD;
            }
            rkva_hello hello {};
            std::memcpy(&hello, input.data(), sizeof(hello));
            // Validate the geometry we can actually serve.
            if (hello.head_dim == 0 || hello.n_head == 0 || hello.n_head_kv == 0 ||
                hello.n_head % hello.n_head_kv != 0 || hello.max_ctx == 0) {
                return RKVA_ERR_BAD_GEOMETRY;
            }
            state.geo = hello;
            state.hello_done = true;

            rkva_hello_ack ack {};
            ack.supported_ops = supported_ops_mask();
            ack.chosen_wire = (hello.proto_flags & (1u << 0)) ? RKVA_WIRE_F16 : RKVA_WIRE_F32;
            ack.max_sessions = 64;
            ack.server_version = 1;  // bumped with the UWP package
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
            rkva_session_req sr {};
            std::memcpy(&sr, input.data(), sizeof(sr));
            if (!state.hello_done) {
                return RKVA_ERR_STATE;
            }
            remote_session s;
            s.session_id = sr.session_id;
            s.seq_id = sr.seq_id;
            s.capacity = sr.capacity ? sr.capacity : state.geo.max_ctx;
            state.sessions[sr.session_id] = s;
            return RKVA_OK;
        }
        case RKVA_DESTROY_SESSION: {
            state.sessions.erase(req.session_id);
            return RKVA_OK;
        }
        case RKVA_RESET: {
            auto it = state.sessions.find(req.session_id);
            if (it == state.sessions.end()) {
                return RKVA_ERR_BAD_SESSION;
            }
            it->second.high_water = -1;
            it->second.resets++;
            // engine.reset(session) lands with the D3D12 phase.
            return RKVA_OK;
        }
        case RKVA_TRIM:
        case RKVA_REWIND: {
            if (req.payload_bytes != sizeof(rkva_trim_req)) {
                return RKVA_ERR_BAD_PAYLOAD;
            }
            rkva_trim_req tr {};
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
            // engine.trim(session, pos0) lands with the D3D12 phase.
            return RKVA_OK;
        }
        case RKVA_GET_STATS: {
            uint64_t trims = 0, resets = 0;
            for (const auto& kv : state.sessions) {
                trims += kv.second.trims;
                resets += kv.second.resets;
            }
            char buf[512];
            snprintf(buf, sizeof(buf),
                "{\"service\":\"rkva\",\"hello\":%s,\"sessions\":%zu,\"trims\":%llu,"
                "\"resets\":%llu,\"attn_engine\":\"pending-d3d12-kvarn\"}",
                state.hello_done ? "true" : "false", state.sessions.size(),
                (unsigned long long) trims, (unsigned long long) resets);
            const size_t n = std::strlen(buf);
            output.assign(buf, buf + n);
            return RKVA_OK;
        }
        case RKVA_ATTN_DECODE:
        case RKVA_ATTN_PREFILL:
        case RKVA_PROBE:
        default:
            // Not advertised in HELLO; explicit failure rather than wrong output.
            (void) compute_us;
            prepare_us = elapsed_us(t0);
            return RKVA_ERR_UNSUPPORTED_OP;
    }
    prepare_us = elapsed_us(t0);
}

void client_loop(SOCKET sock, remote_state& state, const std::string& log_path) {
    uint64_t previous_response_us = 0;
    for (;;) {
        const auto start = clock_type::now();
        rkva_request req {};
        if (!recv_all(sock, &req, sizeof(req))) {
            break;
        }
        const uint64_t idle_us = elapsed_us(start);
        // Generous cap: prefill batches can be multi-MB (Q/K/V rows).
        static constexpr uint32_t MAX_PAYLOAD = 256u * 1024u * 1024u;
        if (req.magic != RKVA_MAGIC || req.version != RKVA_VERSION ||
            req.payload_bytes > MAX_PAYLOAD) {
            break;
        }
        std::vector<uint8_t> input(req.payload_bytes);
        if (req.payload_bytes && !recv_all(sock, input.data(), input.size())) {
            break;
        }
        std::vector<uint8_t> output;
        uint64_t prepare_us = 0, compute_us = 0;
        const auto handle_start = clock_type::now();
        const int rc = handle_request(state, req, input, output, prepare_us, compute_us);
        const uint64_t handle_us = elapsed_us(handle_start);

        const rkva_response rsp {
            RKVA_MAGIC, RKVA_VERSION, req.op, req.cycle_id, rc,
            (uint32_t) output.size(), idle_us, prepare_us, compute_us,
            previous_response_us
        };
        const auto response_start = clock_type::now();
        if (!send_all(sock, &rsp, sizeof(rsp)) ||
            (output.size() && !send_all(sock, output.data(), output.size()))) {
            break;
        }
        previous_response_us = elapsed_us(response_start);
        event(log_path,
              "{\"phase\":\"request\",\"cycleId\":" + std::to_string(req.cycle_id) +
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

}  // namespace

void serve_remote_attn(const std::string& local_folder) {
    const std::string log_path = local_folder + "\\remote-attn.jsonl";
    constexpr uint16_t port = 50054;

    WSADATA wsa {};
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
    sockaddr_in address {};
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
    event(log_path, "{\"phase\":\"listen\",\"status\":\"ready\",\"port\":" +
                    std::to_string(port) + "}");

    remote_state state;
    for (;;) {
        SOCKET client = accept(listener, nullptr, nullptr);
        if (client == INVALID_SOCKET) {
            break;
        }
        const int one = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char*>(&one), sizeof(one));
        int rcvbuf = 8 << 20, sndbuf = 8 << 20;
        setsockopt(client, SOL_SOCKET, SO_RCVBUF,
                   reinterpret_cast<const char*>(&rcvbuf), sizeof(rcvbuf));
        setsockopt(client, SOL_SOCKET, SO_SNDBUF,
                   reinterpret_cast<const char*>(&sndbuf), sizeof(sndbuf));
        event(log_path, "{\"phase\":\"accept\",\"status\":\"connected\"}");
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.hello_done = false;
            state.sessions.clear();
        }
        client_loop(client, state, log_path);
        closesocket(client);
        event(log_path, "{\"phase\":\"accept\",\"status\":\"disconnected\"}");
    }
    closesocket(listener);
    WSACleanup();
}
