#!/usr/bin/env bash
set -euo pipefail
root=/tmp/dflash-trim-opt
build=/home/hjotha/beellama.cpp/build-dflash-xbox-dl/bin
prod=llama-server-root.service
target=/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
draft=/home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf
test_pid=
exec 9>"$root/run.lock"
flock -n 9
[[ $(systemctl is-active "$prod") == active ]]
# The user explicitly authorized stopping production during this benchmark.

restore() {
    rc=$?
    trap - EXIT
    if [[ -n "$test_pid" ]]; then
        kill "$test_pid" 2>/dev/null || true
        wait "$test_pid" 2>/dev/null || true
    fi
    # Always leave real inference selected, including after a mock benchmark failure.
    python3 -c 'import socket,struct; s=socket.create_connection(("192.168.1.26",50053),timeout=15); s.sendall(struct.pack("<IHHQII",0x324c4644,2,6,1,0,0)); b=b"";
while len(b)<56:
 c=s.recv(56-len(b))
 if not c: raise RuntimeError("short mock-disable reply")
 b+=c
assert struct.unpack("<IHHQiIQQQQ",b)[4]==0; s.close()' || rc=1
    systemctl start "$prod" || rc=1
    curl -fsS --retry 90 --retry-delay 1 --retry-connrefused --retry-all-errors \
        --max-time 5 http://127.0.0.1:8090/health -o "$root/production-health.json" || rc=1
    systemctl is-active "$prod" || rc=1
    chmod -R a+rX "$root"
    echo "TRIM_RUN_EXIT=$rc"
    exit "$rc"
}
trap restore EXIT
systemctl stop "$prod"

run_one() {
    local label=$1 mode=$2 variant=$3 legacy=$4
    echo "TRIM_START=$label"
    LD_LIBRARY_PATH="$root/bin-$variant:$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    DFLASH_XBOX_RPC=192.168.1.26:50053 \
    DFLASH_XBOX_METRICS="/tmp/dflash-profile-$label.jsonl" \
    DFLASH_XBOX_CAPTURE="$root/capture.bin" \
    DFLASH_XBOX_MOCK_MODE="$mode" DFLASH_XBOX_LEGACY_TRIM="$legacy" \
    DFLASH_XBOX_TRACE=1 GGML_MTP_PROF=1 \
        "$root/bin-$variant/llama-server" \
        --model "$target" --spec-draft-model "$draft" \
        --spec-type draft-dflash --spec-draft-n-max 4 --draft-p-min 0 \
        --spec-draft-ngl 0 --spec-draft-device none \
        --device CUDA0 --n-gpu-layers 99 --fit off --load-mode none \
        --cache-type-k kvarn4 --cache-type-v kvarn4 \
        --flash-attn on --ctx-size 8192 --parallel 1 \
        --host 127.0.0.1 --port 59567 \
        > "/tmp/dflash-profile-$label-server.log" 2>&1 &
    test_pid=$!
    curl -fsS --retry 90 --retry-delay 1 --retry-connrefused --retry-all-errors \
        --max-time 5 http://127.0.0.1:59567/health \
        -o "/tmp/dflash-profile-$label-health.json"
    python3 /tmp/dflash-profile-request.py "$label" > "/tmp/dflash-profile-$label-summary.json"
    cat "/tmp/dflash-profile-$label-summary.json"
    kill "$test_pid"
    wait "$test_pid" 2>/dev/null || true
    test_pid=
    echo "TRIM_DONE=$label"
}

run_one trim-before-real real before 1
run_one trim-before-xbox xbox before 1
run_one trim-tcp-real real after 1
run_one trim-tcp-xbox xbox after 1
for repetition in 1 2 3; do
    run_one "trim-fused-real-$repetition" real after 0
    run_one "trim-fused-xbox-$repetition" xbox after 0
done
run_one trim-fused-local local after 0
