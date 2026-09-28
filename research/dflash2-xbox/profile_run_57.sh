#!/usr/bin/env bash
set -euo pipefail

prod=llama-server-root.service
server=/home/hjotha/beellama.cpp/build-dflash-xbox-dl/bin/llama-server
target=/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
draft=/home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf
test_pid=

restore() {
    rc=$?
    trap - EXIT
    if [[ -n "$test_pid" ]]; then
        kill "$test_pid" 2>/dev/null || true
        wait "$test_pid" 2>/dev/null || true
    fi
    rm -f /tmp/dflash-profile-running.lock
    systemctl start "$prod" || rc=1
    if ! curl -fsS --retry 90 --retry-delay 1 --retry-connrefused --retry-all-errors \
            --max-time 5 http://127.0.0.1:8090/health -o /tmp/dflash-profile-production-health.json; then
        rc=1
    fi
    systemctl is-active "$prod" || rc=1
    chmod 644 /tmp/dflash-profile-* 2>/dev/null || true
    echo "PROFILE_RUN_EXIT=$rc"
    exit "$rc"
}
trap restore EXIT

[[ $(systemctl is-active "$prod") == active ]]
[[ ! -e /tmp/dflash-profile-running.lock ]]
touch /tmp/dflash-profile-running.lock
systemctl stop "$prod"

run_one() {
    local mode=$1
    echo "PROFILE_MODE_START=$mode"
    DFLASH_XBOX_RPC=192.168.1.26:50053 \
    DFLASH_XBOX_METRICS="/tmp/dflash-profile-$mode.jsonl" \
    DFLASH_XBOX_CAPTURE=/tmp/dflash-profile-capture.bin \
    DFLASH_XBOX_MOCK_MODE="$mode" \
    DFLASH_XBOX_TRACE=1 GGML_MTP_PROF=1 \
        "$server" \
        --model "$target" --spec-draft-model "$draft" \
        --spec-type draft-dflash --spec-draft-n-max 4 --draft-p-min 0 \
        --spec-draft-ngl 0 --spec-draft-device none \
        --device CUDA0 --n-gpu-layers 99 --fit off --load-mode none \
        --cache-type-k kvarn4 --cache-type-v kvarn4 \
        --flash-attn on --ctx-size 8192 --parallel 1 \
        --host 127.0.0.1 --port 59567 \
        > "/tmp/dflash-profile-$mode-server.log" 2>&1 &
    test_pid=$!
    curl -fsS --retry 90 --retry-delay 1 --retry-connrefused --retry-all-errors \
        --max-time 5 http://127.0.0.1:59567/health \
        -o "/tmp/dflash-profile-$mode-health.json"
    python3 /tmp/dflash-profile-request.py "$mode" \
        > "/tmp/dflash-profile-$mode-summary.json"
    cat "/tmp/dflash-profile-$mode-summary.json"
    kill "$test_pid"
    wait "$test_pid" 2>/dev/null || true
    test_pid=
    echo "PROFILE_MODE_DONE=$mode"
}

run_one real
run_one xbox
run_one local
rm /tmp/dflash-profile-running.lock
