#!/usr/bin/env bash
set -euo pipefail

evidence=/tmp/dflash-gpu-f0-20260928
binary=/tmp/dflash-trim-opt/bin-after/llama-server
library_dir=/home/hjotha/beellama.cpp/build-dflash-xbox-dl/bin
target=/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
draft=/home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf
service=llama-server-root.service
test_pid=
proxy_pid=

mkdir -p "$evidence"
exec 9>"$evidence/run.lock"
flock -n 9
[[ $(systemctl is-active "$service") == active ]]

restore() {
    rc=$?
    trap - EXIT
    if [[ -n "$test_pid" ]]; then
        kill "$test_pid" 2>/dev/null || true
        wait "$test_pid" 2>/dev/null || true
    fi
    if [[ -n "$proxy_pid" ]]; then
        kill "$proxy_pid" 2>/dev/null || true
        wait "$proxy_pid" 2>/dev/null || true
    fi
    python3 -c 'import socket,struct; s=socket.create_connection(("192.168.1.26",50053),timeout=15); s.sendall(struct.pack("<IHHQII",0x324c4644,2,6,1,0,0)); data=b"";
while len(data)<56:
 chunk=s.recv(56-len(data))
 if not chunk: raise RuntimeError("short mock-disable reply")
 data+=chunk
assert struct.unpack("<IHHQiIQQQQ",data)[4]==0; s.close()' || rc=1
    systemctl start "$service" || rc=1
    curl -fsS --retry 90 --retry-delay 1 --retry-connrefused --retry-all-errors \
        --max-time 5 http://127.0.0.1:8090/health -o "$evidence/production-health.json" || rc=1
    systemctl is-active "$service" || rc=1
    chmod -R a+rX "$evidence"
    echo "CAPTURE_EXIT=$rc"
    exit "$rc"
}
trap restore EXIT

python3 /tmp/dflash-reference-capture-proxy.py capture \
    --listen 127.0.0.1:50054 --target 192.168.1.26:50053 \
    --output "$evidence/reference.cap" > "$evidence/proxy.jsonl" 2>&1 &
proxy_pid=$!

systemctl stop "$service"

LD_LIBRARY_PATH="/tmp/dflash-trim-opt/bin-after:$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
DFLASH_XBOX_RPC=127.0.0.1:50054 \
DFLASH_XBOX_METRICS="$evidence/cycles.jsonl" \
DFLASH_XBOX_MOCK_MODE=real DFLASH_XBOX_LEGACY_TRIM=0 \
DFLASH_XBOX_TRACE=1 GGML_MTP_PROF=1 \
    "$binary" \
    --model "$target" --spec-draft-model "$draft" \
    --spec-type draft-dflash --spec-draft-n-max 4 --draft-p-min 0 \
    --spec-draft-ngl 0 --spec-draft-device none \
    --device CUDA0 --n-gpu-layers 99 --fit off --load-mode none \
    --cache-type-k kvarn4 --cache-type-v kvarn4 \
    --flash-attn on --ctx-size 8192 --parallel 1 \
    --host 127.0.0.1 --port 59567 \
    > "$evidence/server.log" 2>&1 &
test_pid=$!

curl -fsS --retry 90 --retry-delay 1 --retry-connrefused --retry-all-errors \
    --max-time 5 http://127.0.0.1:59567/health -o "$evidence/test-health.json"
python3 /tmp/dflash-profile-request.py gpu-f0 > "$evidence/request.json"
cat "$evidence/request.json"
kill "$test_pid"
wait "$test_pid" 2>/dev/null || true
test_pid=
wait "$proxy_pid"
proxy_pid=
cat "$evidence/proxy.jsonl"
