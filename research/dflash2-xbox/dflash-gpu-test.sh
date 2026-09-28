#!/usr/bin/env bash
set -euo pipefail
mode=$1
nmax=${2:-4}
draft=${3:-/home/hjotha/models/Qwen3.8-27B-DFlash2-Q3_K_M.gguf}
prod=llama-server-root.service
server=/home/hjotha/beellama.cpp/build-dflash-xbox-dl/bin/llama-server
target=/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
test_pid=
restore() {
    rc=$?
    trap - EXIT
    if [[ -n "$test_pid" ]]; then kill "$test_pid" 2>/dev/null || true; wait "$test_pid" 2>/dev/null || true; fi
    echo amdk6x | sudo -S systemctl start "$prod" || rc=1
    curl -fsS --retry 90 --retry-delay 1 --retry-connrefused --retry-all-errors --max-time 5 http://127.0.0.1:8090/health -o /tmp/health-restore.json || rc=1
    systemctl is-active "$prod" || rc=1
    echo "RESTORE_EXIT=$rc"
    exit "$rc"
}
trap restore EXIT
echo amdk6x | sudo -S systemctl stop "$prod"
DFLASH_XBOX_RPC=192.168.1.26:50053 \
DFLASH_XBOX_METRICS="/tmp/dflash-$mode-metrics.jsonl" \
DFLASH_XBOX_CAPTURE=/tmp/dflash-$mode-capture.bin \
DFLASH_XBOX_TRACE=1 GGML_MTP_PROF=1 \
    "$server" --model "$target" --spec-draft-model "$draft" \
    --spec-type draft-dflash --spec-draft-n-max "$nmax" --draft-p-min 0 \
    --spec-draft-ngl 0 --spec-draft-device none \
    --device CUDA0 --n-gpu-layers 99 --fit off --load-mode none \
    --cache-type-k kvarn4 --cache-type-v kvarn4 \
    --flash-attn on --ctx-size 8192 --parallel 1 \
    --host 127.0.0.1 --port 59567 \
    > "/tmp/dflash-$mode-server.log" 2>&1 &
test_pid=$!
curl -fsS --retry 90 --retry-delay 1 --retry-connrefused --retry-all-errors --max-time 5 http://127.0.0.1:59567/health -o "/tmp/dflash-$mode-health.json"
python3 /tmp/dflash-profile-request.py "$mode"
kill "$test_pid"; wait "$test_pid" 2>/dev/null || true; test_pid=
