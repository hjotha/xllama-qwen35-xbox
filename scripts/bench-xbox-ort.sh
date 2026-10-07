#!/usr/bin/env bash
# bench-xbox-ort.sh — ORT GenAI benchmark orchestrator for Xbox Series S
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-xbox-ort.sh <model-dir-name> [--threads N] [--runs N] [--prompt file]
#                                [--out FILE] [--gpu-sample] [--ctx N] [--n-predict N]
#                                [--max-length N] [--keep-config] [--prompt-lookup]
#
# Arguments:
#   model-dir-name   Model directory name in LocalState/models/ (e.g. smollm2-360m-cpu-int4)
#   --threads N      Upload genai_config-threads-N.json and tag CSV row with t<N>.
#                    The device config is backed up and RESTORED on exit (a swap
#                    left in place would silently affect every later run).
#   --keep-config    Leave the --threads config on the device (skip the restore)
#   --runs N         Number of bench runs (default: 4, warmup run 1 is dropped;
#                    runs 2..N are each appended individually with their run_index
#                    so the summary can report a spread instead of a pre-averaged
#                    point — W1.1). Default 4 yields 3 recorded measurement runs.
#   --prompt file    Path to prompt file (default: bench/prompts/standard-512.txt)
#   --out FILE       Results CSV to append the per-run rows to
#                    (default: bench/results/phase1-cpu.csv; DML runs → phase2-dml.csv)
#   --gpu-sample     Sample Device Portal GPU telemetry (xbox-gpu-sample.sh)
#                    across the runs and print a per-engine summary at the end
#   --ctx N          Override n_ctx via bench_ctx.txt (0 = engine default 2048)
#   --n-predict N    Override n_predict via bench_npredict.txt (0 = default 512)
#   --max-length N   Override max_length via bench_maxlen.txt. 0 = derive as
#                    min(n_ctx, prompt+n_predict); -1 = saturate to n_ctx (what
#                    the shipping app does since #135); >0 = explicit value.
#                    On DirectML this is THE variable governing prefill (#130).
#   --ubatch N       Override llama.cpp n_ubatch (physical prefill chunk) via
#                    bench_ubatch.txt (#172). 0 = llama default (512). GGUF
#                    models only; the ORT path ignores it. The device tags the
#                    CSV host column with -uN so the row carries the variable.
#   --batch N        Override llama.cpp n_batch (prompt batch) via
#                    bench_n_batch.txt. 0 = llama default (512). GGUF models
#                    only. Host column tagged -bN. Exists so an actual
#                    64/64 configuration can be produced AND verified from the
#                    effective-config log line (never inferred from the flag).
#   --kv-q8          q8_0 KV cache + flash attention via bench_kvq8.txt (#171).
#                    GGUF models only. Host column tagged -kvq8 (same rationale
#                    as -uN); the guard fails if the MSIX ignores the knob.
#   --gpu-layers N   GGUF GPU decode D2b: layers on the d3d12 backend via
#                    bench_gpu_layers.txt (0 = CPU, default). GGUF models only.
#                    Host column tagged -gN; the guard fails if the MSIX ignores it.
#   --ignore-eog     Decode exactly n_predict tokens via bench_ignore_eog.txt
#                    (no EOG / stop-sequence end). Host column tagged -noeog.
#                    Use for A/Bs whose arms reach EOG at different points.
#   --prompt-lookup  Phase 15 W2 (#210): draft-free n-gram speculative decoding
#                    via bench_prompt_lookup.txt=1. Host column tagged -plookup.
#                    Off (file deleted) when the flag is absent so a prior on
#                    run cannot leak into the next.
#   --mtp N          MTP drafting against the beellama MTP head, n_max = N.
#                    Requires a model whose GGUF carries the head (qwen35-4b-mtp);
#                    on any other model it is a no-op. Host tag -mtpN.
#   --mtp-pmin PCT   Draft stops below this candidate probability (percent,
#                    0-100; default 75). 0 also skips the top_prob softmax.
#                    Host tag -pminPCT. Only meaningful together with --mtp.
#
# Required env: XBOX_IP, XBOX_USER, XBOX_PASS
#
# Output: one row per recorded run appended to the --out CSV (run_index column);
#         the summary generator computes the median and min-max spread from them.
#
# Notes:
#   - The model must already be in LocalState\models\<name>\ on the console
#     (copied from MSIX on first launch, or uploaded via deploy.sh upload-dir).
#   - bench.flag is consumed by the app and deleted; this script re-uploads it
#     for each run.
#   - Thread variants require bench-xbox-ort.sh from a v0.3.1+ MSIX that reads
#     bench_threads.txt. With v0.3.0 the n_threads CSV column shows detect_threads().

set -euo pipefail

MODEL_NAME="${1:-smollm2-360m-cpu-int4}"
N_THREADS=0
N_CTX=0         # 0 = engine default (2048)
N_PREDICT=0     # 0 = engine default (512)
MAX_LEN=0       # 0 = derive min(n_ctx, prompt+n_predict); -1 = saturate to n_ctx; >0 = explicit
UBATCH=0        # 0 = llama default (512); #172 sweep knob, GGUF only
BATCH=0         # 0 = llama default (512); MVP n_batch knob (with --ubatch -> 64/64)
KVQ8=0          # 1 = q8_0 KV + flash attention; #171 A/B knob, GGUF only
GPU_LAYERS=0    # D2b: GGUF layers on the d3d12 backend; 0 = CPU
TWOCOL="off"    # plan 004: off = clean OLD arm; auto = d3d12twocol.txt ("auto"), allowlisted NEW
IGNORE_EOG=0    # D2b: 1 = decode exactly n_predict tokens
PROMPT_LOOKUP=0 # 1 = W2 prompt-lookup; #210 A/B knob, GGUF only
MTP_N=0         # 0 = MTP off; N>0 = MTP on with n_max = N
MTP_PMIN=-1     # -1 = bridge default (0.75); 0..100 = draft p_min in percent
GREEDY=0        # 1 = deterministic argmax decode (plan 003, stage 1: parity)
# Plan 003 F3.5: Session parity gate. The throughput bench makes ONE generation
# per run, so reset / edited-prefix / delta / multi-chunk had no harness on the
# console at all. --mtp-session switches to the four Session scenarios.
PROFILE_PHASES=1    # phase instrumentation ON/OFF arm for the cost measurement
MTP_SESSION=0        # 1 = run the Session scenarios instead of the throughput bench
SESSION_NPREDICT=12  # tokens per turn in those scenarios
SESSION_NBATCH=16    # session n_batch for the multi-chunk scenario (0 = default)
SEED=0          # 0 = engine default seed; >0 = fixed sampling seed
DUMP_TOKENS=0   # 1 = save the accepted token ids per run (bench-tokens-<run>.txt)
N_RUNS=4        # warmup run 1 dropped; runs 2..N recorded individually (W1.1) → 3 by default
PROMPT_FILE=""
OUT_CSV=""
GPU_SAMPLE=false
KEEP_CONFIG=false # --keep-config: leave a --threads genai_config.json on the device

shift || true
while [[ $# -gt 0 ]]; do
	case "$1" in
	--threads)
		N_THREADS="${2:?--threads requires a value}"
		shift 2
		;;
	--ctx)
		N_CTX="${2:?--ctx requires a value}"
		shift 2
		;;
	--n-predict)
		N_PREDICT="${2:?--n-predict requires a value}"
		shift 2
		;;
	--max-length)
		MAX_LEN="${2:?--max-length requires a value}"
		shift 2
		;;
	--ubatch)
		UBATCH="${2:?--ubatch requires a value}"
		shift 2
		;;
	--batch)
		BATCH="${2:?--batch requires a value}"
		shift 2
		;;
	--kv-q8)
		KVQ8=1
		shift
		;;
	--gpu-layers)
		GPU_LAYERS="${2:?--gpu-layers requires a value}"
		shift 2
		;;
	--twocol)
		TWOCOL="${2:?--twocol requires off|auto}"
		shift 2
		;;
	--ignore-eog)
		IGNORE_EOG=1
		shift
		;;
	--prompt-lookup)
		PROMPT_LOOKUP=1
		shift
		;;
	--mtp)
		MTP_N="${2:?--mtp requires a value}"
		shift 2
		;;
	--mtp-pmin)
		MTP_PMIN="${2:?--mtp-pmin requires a value}"
		shift 2
		;;
	--greedy)
		GREEDY=1
		shift
		;;
	--profile-phases)
		PROFILE_PHASES="${2:?--profile-phases requires 0 or 1}"
		shift 2
		;;
	--mtp-session)
		MTP_SESSION=1
		shift
		;;
	--session-npredict)
		SESSION_NPREDICT="${2:?--session-npredict requires a value}"
		shift 2
		;;
	--session-nbatch)
		SESSION_NBATCH="${2:?--session-nbatch requires a value}"
		shift 2
		;;
	--seed)
		SEED="${2:?--seed requires a value}"
		shift 2
		;;
	--tokens)
		DUMP_TOKENS=1
		shift
		;;
	--runs)
		N_RUNS="${2:?--runs requires a value}"
		shift 2
		;;
	--prompt)
		PROMPT_FILE="${2:?--prompt requires a file}"
		shift 2
		;;
	--out)
		OUT_CSV="${2:?--out requires a file}"
		shift 2
		;;
	--gpu-sample)
		GPU_SAMPLE=true
		shift
		;;
	--keep-config)
		KEEP_CONFIG=true
		shift
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

echo "=== xllama bench-xbox-ort ==="
echo "  Model:   $MODEL_NAME"
echo "  Threads: ${N_THREADS:-auto}"
echo "  n_ctx:   $([[ $N_CTX -gt 0 ]] && echo "$N_CTX" || echo "default")"
echo "  n_predict: $([[ $N_PREDICT -gt 0 ]] && echo "$N_PREDICT" || echo "default")"
if ((MAX_LEN == 0)); then
	MAX_LEN_LABEL="derived"
elif ((MAX_LEN < 0)); then
	MAX_LEN_LABEL="saturated (n_ctx)"
else
	MAX_LEN_LABEL="$MAX_LEN"
fi
echo "  max_length: $MAX_LEN_LABEL"
echo "  Runs:    $N_RUNS (run 1 warmup, dropped from median)"
echo "  Xbox:    $XBOX_IP"

# ---------------------------------------------------------------------------
# CSRF token (required for POST/DELETE)
# ---------------------------------------------------------------------------
CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n1)
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE may fail" >&2

# ---------------------------------------------------------------------------
# Package full name
# ---------------------------------------------------------------------------
PFN=$("${DEPLOY}" pfn 2>/dev/null)

verify_expected_package() {
	[[ -z "${XLLAMA_EXPECTED_PFN:-}" ]] && return 0
	if [[ "$PFN" != "$XLLAMA_EXPECTED_PFN" ]]; then
		echo "Error: initial package $PFN differs from trial package $XLLAMA_EXPECTED_PFN" >&2
		return 1
	fi
	local current_pfn
	current_pfn=$("${DEPLOY}" pfn)
	if [[ "$current_pfn" != "$XLLAMA_EXPECTED_PFN" ]]; then
		echo "Error: installed package $current_pfn differs from trial package $XLLAMA_EXPECTED_PFN" >&2
		return 1
	fi
}
verify_expected_package
[[ -z "$PFN" ]] && {
	echo "Error: xllama not found — deploy it first" >&2
	exit 1
}
echo "  PFN: $PFN"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TMPDIR_LOCAL=$(mktemp -d)
CONFIG_SWAPPED="" # set when --threads overwrites the device genai_config.json
MODEL_TXT_ORIG="" # "saved" or "absent" once the device model.txt was backed up
TWOCOL_ORIG=""

# One cleanup path for the whole script. --threads overwrites genai_config.json
# on the device (models\<name>\); without a restore, the last thread variant
# stays in force for every later run of the app and every bench that does not
# pass --threads (observed: a t8 sweep left the console on t8). Mirror the
# backup/restore that profile-dml-run.sh already does, and run it on ANY exit,
# so a mid-run failure still restores. --keep-config opts out.
cleanup() {
	# Plan 003 F3.5: the Session gate knobs make the app skip the throughput bench
	# on its next start, so they must never survive this script. Left in force they
	# would hijack the next bench into session mode, which writes a different CSV.
	for k in bench_mtp_session.txt bench_session_npredict.txt bench_session_nbatch.txt \
		bench_profile.txt bench_mtp.txt bench_mtp_pmin.txt; do
		delete_from_localstate "$k" >/dev/null 2>&1 || true
	done
	if [[ "$TWOCOL_ORIG" == "saved" ]]; then
		upload_as "${TMPDIR_LOCAL}/twocol_orig.txt" "" "d3d12twocol.txt" >/dev/null 2>&1 || true
	elif [[ "$TWOCOL_ORIG" == "absent" ]]; then
		delete_from_localstate "d3d12twocol.txt"
	fi
	if [[ -n "$CONFIG_SWAPPED" && "$KEEP_CONFIG" != "true" ]]; then
		echo "  Restoring original genai_config.json on the device..." >&2
		upload_as "${TMPDIR_LOCAL}/genai_config_orig.json" "models\\${MODEL_NAME}" "genai_config.json" >/dev/null 2>&1 || true
	elif [[ -n "$CONFIG_SWAPPED" ]]; then
		echo "  --keep-config: t${N_THREADS} genai_config.json left on the device" >&2
	fi
	# model.txt also selects the model for the LAN API and the next app launch.
	# Left pointing at the benched dir, it breaks both once that dir is removed
	# (observed after the shape gate: model.txt = a deleted shape-* dir).
	if [[ "$MODEL_TXT_ORIG" == "saved" ]]; then
		echo "  Restoring original model.txt on the device..." >&2
		upload_as "${TMPDIR_LOCAL}/model_orig.txt" "" "model.txt" >/dev/null 2>&1 || true
	elif [[ "$MODEL_TXT_ORIG" == "absent" ]]; then
		delete_from_localstate "model.txt"
	fi
	rm -rf "$TMPDIR_LOCAL"
}
trap cleanup EXIT

_curl_post_file() {
	local local_path="$1" path_param="$2" filename="${3:-}"
	local form_entry
	if [[ -n "$filename" ]]; then
		form_entry="${local_path};filename=${filename};type=application/octet-stream"
	else
		form_entry="${local_path};type=application/octet-stream"
	fi
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST \
		-F "file=@${form_entry}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${path_param}" \
		>/dev/null
}

# Upload file to LocalState root (or subdir via remote_dir like "models\\<name>")
upload_to_localstate() {
	local local_path="$1" remote_dir="${2:-}"
	local path_param="%5CLocalState"
	[[ -n "$remote_dir" ]] && path_param="%5CLocalState%5C${remote_dir//\\/%5C}"
	echo "  Uploading $(basename "$local_path") → LocalState\\${remote_dir} ..."
	_curl_post_file "$local_path" "$path_param"
}

# Upload file to LocalState with an explicit remote filename
upload_as() {
	local local_path="$1" remote_dir="${2:-}" remote_name="$3"
	local path_param="%5CLocalState"
	[[ -n "$remote_dir" ]] && path_param="%5CLocalState%5C${remote_dir//\\/%5C}"
	printf '  Uploading %s → LocalState\\%s\\%s ...\n' "$(basename "$local_path")" "$remote_dir" "$remote_name"
	_curl_post_file "$local_path" "$path_param" "$remote_name"
}

# Download file from LocalState root
download_from_localstate() {
	local remote_name="$1" dest="$2"
	curl "${CURL_AUTH[@]}" -o "$dest" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${remote_name}" \
		2>/dev/null || true
}

# Delete file from LocalState root
delete_from_localstate() {
	local remote_name="$1"
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${remote_name}" \
		>/dev/null 2>&1 || true
}

# Poll until a LocalState file is confirmed absent (GET returns 404 / error body).
# delete_from_localstate is fire-and-forget; without this confirmation a stale
# marker still present when wait_for_done first polls makes it return 0 instantly
# off the PREVIOUS run's bench-result.csv.done — a silent wrong row, not a timeout.
verify_deleted() {
	local remote_name="$1" tries="${2:-15}" resp
	while ((tries-- > 0)); do
		resp=$(curl "${CURL_AUTH[@]}" \
			"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${remote_name}" \
			2>/dev/null) || true
		if [[ -z "$resp" || "$resp" == *"404"* || "$resp" == *"error"* ]]; then
			return 0
		fi
		delete_from_localstate "$remote_name" # retry the delete, then re-check
		sleep 1
	done
	echo "  Warning: could not confirm ${remote_name} was deleted — result may be stale" >&2
	return 1
}

# Restart app via Device Portal
restart_app() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/taskmanager/app?package=${PFN}" >/dev/null 2>&1 || true
	sleep 2
	local pfamily
	# shellcheck disable=SC2001
	pfamily=$(echo "$PFN" | sed 's/_[0-9][0-9.]*_[^_]*__/_/')
	local aumid
	aumid=$(printf '%s!xllama' "$pfamily" | base64 -w0)
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST -d "" \
		"${BASE_URL}/api/taskmanager/app?appid=${aumid}" >/dev/null 2>&1 || true
}

# Wait for a done-marker in LocalState (polling). Parameterised because the
# Session parity gate waits on its own marker, not bench-result.csv.done.
wait_for_marker() {
	local name="$1" timeout_s="${2:-300}" elapsed=0
	echo "  Waiting for ${name} (timeout ${timeout_s}s)..."
	while ((elapsed < timeout_s)); do
		local resp
		resp=$(curl "${CURL_AUTH[@]}" \
			"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${name}" \
			2>/dev/null) || true
		if [[ -n "$resp" && "$resp" != *"404"* && "$resp" != *"error"* ]]; then
			echo "  Done after ${elapsed}s."
			return 0
		fi
		sleep 10
		((elapsed += 10))
	done
	echo "  Timeout waiting for ${name}" >&2
	return 1
}

# Wait for bench-result.csv.done (polling)
wait_for_done() {
	wait_for_marker "bench-result.csv.done" "${1:-300}"
}

# ---------------------------------------------------------------------------
# Upload genai_config variant for thread tuning (if --threads N specified)
# ORT GenAI only — skip when the model dir is GGUF (llama.cpp uses
# bench_threads.txt / params.n_threads; a genai_config.json next to a .gguf
# is noise and confused earlier campaigns).
# ---------------------------------------------------------------------------
model_dir_has_gguf() {
	local listing
	listing=$(curl "${CURL_AUTH[@]}" \
		"${BASE_URL}/api/filesystem/apps/files?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cmodels%5C${MODEL_NAME}" \
		2>/dev/null) || true
	[[ "$listing" == *".gguf"* ]]
}

if [[ "$N_THREADS" -gt 0 ]] 2>/dev/null; then
	if model_dir_has_gguf; then
		echo ""
		echo "--- Skipping genai_config (GGUF model; threads via bench_threads.txt) ---"
	else
		THREADS_CONFIG="${REPO_ROOT}/bench/configs/genai_config-threads-${N_THREADS}.json"
		if [[ -f "$THREADS_CONFIG" ]]; then
			echo ""
			echo "--- Uploading genai_config (intra_op_num_threads=${N_THREADS}) ---"
			# Back up the device's current config FIRST, so the EXIT trap can put
			# it back. Without this the swap is permanent (see cleanup()).
			curl "${CURL_AUTH[@]}" -o "${TMPDIR_LOCAL}/genai_config_orig.json" \
				"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cmodels%5C${MODEL_NAME}&filename=genai_config.json" \
				2>/dev/null || true
			if [[ -s "${TMPDIR_LOCAL}/genai_config_orig.json" ]]; then
				CONFIG_SWAPPED=1
			else
				echo "Warning: could not back up the device genai_config.json — it will NOT be restored" >&2
			fi
			upload_as "$THREADS_CONFIG" "models\\${MODEL_NAME}" "genai_config.json"
		else
			echo "Warning: $THREADS_CONFIG not found — using existing genai_config.json on device" >&2
		fi
	fi
fi

# ---------------------------------------------------------------------------
# Prepare local bench files
# ---------------------------------------------------------------------------
PROMPT_SRC="${PROMPT_FILE:-${REPO_ROOT}/bench/prompts/standard-512.txt}"
[[ ! -f "$PROMPT_SRC" ]] && {
	echo "Error: prompt file not found: $PROMPT_SRC" >&2
	exit 1
}
cp "$PROMPT_SRC" "${TMPDIR_LOCAL}/prompt.txt"

# model.txt — tells inference-bridge which model dir to use
printf '%s' "$MODEL_NAME" >"${TMPDIR_LOCAL}/model.txt"
# Back it up before the first overwrite so cleanup() can put it back.
if curl "${CURL_AUTH[@]}" --fail -o "${TMPDIR_LOCAL}/model_orig.txt" \
	"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=model.txt" \
	2>/dev/null; then
	MODEL_TXT_ORIG="saved"
else
	MODEL_TXT_ORIG="absent"
fi

# bench_threads.txt — tells inference-bridge what n_threads to write in CSV (v0.3.1+)
if curl "${CURL_AUTH[@]}" --fail -o "${TMPDIR_LOCAL}/twocol_orig.txt" \
	"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=d3d12twocol.txt" \
	2>/dev/null; then
	TWOCOL_ORIG="saved"
else
	TWOCOL_ORIG="absent"
fi

# bench_threads.txt — tells inference-bridge what n_threads to write in CSV (v0.3.1+)
printf '%d' "$N_THREADS" >"${TMPDIR_LOCAL}/bench_threads.txt"
# 0 = leave the engine default; #130 varies these to test the band hypothesis.
printf '%d' "$N_CTX" >"${TMPDIR_LOCAL}/bench_ctx.txt"
printf '%d' "$N_PREDICT" >"${TMPDIR_LOCAL}/bench_npredict.txt"
# Always written, even when 0 — main_loop only ever overwrites these files, so a
# value left by a previous run would otherwise stay in force.
printf '%d' "$MAX_LEN" >"${TMPDIR_LOCAL}/bench_maxlen.txt"
printf '%d' "$UBATCH" >"${TMPDIR_LOCAL}/bench_ubatch.txt"
printf '%d' "$BATCH" >"${TMPDIR_LOCAL}/bench_n_batch.txt"
printf '%d' "$KVQ8" >"${TMPDIR_LOCAL}/bench_kvq8.txt"
printf '%d' "$GPU_LAYERS" >"${TMPDIR_LOCAL}/bench_gpu_layers.txt"
printf '%d' "$IGNORE_EOG" >"${TMPDIR_LOCAL}/bench_ignore_eog.txt"
printf '%d' "$PROMPT_LOOKUP" >"${TMPDIR_LOCAL}/bench_prompt_lookup.txt"
# Plan 003 stage 2: 0 keeps every counter but drops the chrono snapshots, so an
# ON/OFF pair on the same package/GGUF measures the instrumentation's own cost.
printf '%d' "$PROFILE_PHASES" >"${TMPDIR_LOCAL}/bench_profile.txt"
printf '%d' "$MTP_N" >"${TMPDIR_LOCAL}/bench_mtp.txt"
printf '%d' "$MTP_PMIN" >"${TMPDIR_LOCAL}/bench_mtp_pmin.txt"
# Plan 003, stage 1: parity knobs. Greedy (argmax) and a fixed seed make the
# sampling deterministic so baseline and MTP runs can be compared token by token.
printf '%d' "$GREEDY" >"${TMPDIR_LOCAL}/bench_greedy.txt"
printf '%u' "$SEED" >"${TMPDIR_LOCAL}/bench_seed.txt"
printf '%d' "$DUMP_TOKENS" >"${TMPDIR_LOCAL}/bench_tokens.txt"

# bench.flag — consumed by app on each start; must be re-uploaded per run
printf 'bench' >"${TMPDIR_LOCAL}/bench.flag"

# ---------------------------------------------------------------------------
# Run N_RUNS iterations
# ---------------------------------------------------------------------------
declare -a CSV_ROWS=()

# Defined before the run loop, not just before the median: each downloaded row is
# checked against this schema's field count as it arrives (see below).
CSV_HEADER="model,quant,backend,n_ctx,n_threads,prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,gpu_mem_mb,gpu_budget_mb,n_prompt_tok,n_gen_tok,max_length,host,date,run_index,prefill_ms,ttft_ms"

SAMPLER_PID=""
if [[ "$GPU_SAMPLE" == "true" ]]; then
	echo "  Starting GPU sampler (systemperf)..."
	"${SCRIPT_DIR}/xbox-gpu-sample.sh" --out "${TMPDIR_LOCAL}/gpu-sample.csv" \
		>"${TMPDIR_LOCAL}/gpu-summary.txt" 2>&1 &
	SAMPLER_PID=$!
fi

# ---------------------------------------------------------------------------
# Plan 003 F3.5: Session parity gate (reset / edited-prefix / delta / multi-chunk).
# One shot, not N runs: the device writes bench-mtp-session.csv plus per-turn
# token dumps, and the ids are ALSO compared here, so the verdict does not rest
# on the device's own arithmetic. A scenario passes only when every generate
# succeeded, the drafter was active AND proposed something after the
# reset/rewind, and the ids matched the reference.
# ---------------------------------------------------------------------------
mtp_session_gate() {
	local evidence="${EVIDENCE_DIR:-/tmp/opencode/xbox-mtp-session-$(date -u +%Y%m%dT%H%M%SZ)}"
	local dumps=(
		bench-mtp-s1-t2.txt bench-mtp-s1-t2.txt.prefill
		bench-mtp-s1-ref.txt bench-mtp-s1-ref.txt.prefill
		bench-mtp-s2-t2.txt bench-mtp-s2-t2.txt.prefill
		bench-mtp-s2-ref.txt bench-mtp-s2-ref.txt.prefill
		bench-mtp-s3-t1.txt bench-mtp-s3-t1.txt.prefill
		bench-mtp-s3-t2.txt bench-mtp-s3-t2.txt.prefill
		bench-mtp-s3-seq-t1.txt bench-mtp-s3-seq-t1.txt.prefill
		bench-mtp-s3-seq-t2.txt bench-mtp-s3-seq-t2.txt.prefill
		bench-mtp-s3-ref.txt bench-mtp-s3-ref.txt.prefill
		bench-mtp-s4-t.txt bench-mtp-s4-t.txt.prefill
		bench-mtp-s4-ref.txt bench-mtp-s4-ref.txt.prefill
	)
	local f
	mkdir -p "$evidence"
	echo "  Session evidence dir: $evidence"

	# The trap removes TMPDIR_LOCAL and this gate can fail before the driver
	# appends anything, so the CSV and every sidecar are copied out on EVERY exit
	# path — a failed gate with no artefacts is not evidence.
	persist_session_evidence() {
		if [[ -f "${TMPDIR_LOCAL}/session.csv" ]]; then
			cp -f "${TMPDIR_LOCAL}/session.csv" "$evidence/bench-mtp-session.csv" 2>/dev/null || true
		fi
		for f in "${dumps[@]}"; do
			[[ -f "${TMPDIR_LOCAL}/$f" ]] && cp -f "${TMPDIR_LOCAL}/$f" "$evidence/$f" 2>/dev/null
		done
		{
			echo "package_pfn=$PFN"
			echo "model=$MODEL_NAME"
			echo "msix_sha256=${XLLAMA_MSIX_SHA256:-unknown}"
			echo "requested_mtp_n=$MTP_N requested_mtp_pmin=$MTP_PMIN requested_gpu_layers=$GPU_LAYERS requested_threads=$N_THREADS requested_ctx=$N_CTX requested_twocol=$TWOCOL"
			echo "session_npredict=$SESSION_NPREDICT session_nbatch=$SESSION_NBATCH run_id=${SESSION_RUN_ID:-1}"
			echo "recorded_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
			if [[ -n "${SESSION_LOG_OUT:-}" && -s "${SESSION_LOG_OUT}" ]]; then
				echo "--- effective config / scenarios from the device log ---"
				grep -aE 'MTP_SESSION_CONFIG|MPT_SCENARIO|mtp: draft ready|MTP_DIVERGENCE' \
					"$SESSION_LOG_OUT" || echo "(no matching lines)"
			else
				echo "device_log=unavailable"
			fi
		} >"$evidence/config.txt" 2>/dev/null || true
	}

	# Diagnostic mode: p_min=0 (or any value below the gate) separates "the drafter
    # cannot propose" from "the confidence filter rejected every proposal". It is
    # NOT an approval path and never weakens a scenario's n_drafted>0
    # requirement — the same asserts run, only the threshold changes.
    printf '1' >"${TMPDIR_LOCAL}/bench_mtp_session.txt"
	printf '%d' "$SESSION_NPREDICT" >"${TMPDIR_LOCAL}/bench_session_npredict.txt"
	printf '%d' "$SESSION_NBATCH" >"${TMPDIR_LOCAL}/bench_session_nbatch.txt"
	printf '%d' "${SESSION_RUN_ID:-1}" >"${TMPDIR_LOCAL}/bench_run_index.txt"

	verify_expected_package
	delete_from_localstate "bench-mtp-session.csv"
	delete_from_localstate "bench-mtp-session.csv.done"
	# Conditional uploads below (threads/mtp/pmin) must not inherit a previous
	# run's files: a stale bench_mtp_pmin.txt=0 once leaked pmin=0 into a
	# default run. Delete first, upload only when set.
	delete_from_localstate "bench_threads.txt"
	delete_from_localstate "bench_n_batch.txt"
	delete_from_localstate "bench_mtp.txt"
	delete_from_localstate "bench_mtp_pmin.txt"
	# A stale d3d12twocol.txt would silently flip this run to the NEW arm.
	delete_from_localstate "d3d12twocol.txt"
	# bench_turns.txt would hijack the app into kv-bench mode, which never writes
	# the marker this gate waits for.
	delete_from_localstate "bench_turns.txt"
	delete_from_localstate "bench-result.csv.done"
	verify_deleted "bench-mtp-session.csv.done"
	for f in "${dumps[@]}"; do
		delete_from_localstate "$f"
	done
	sleep 1

	upload_to_localstate "${TMPDIR_LOCAL}/bench.flag"
	upload_to_localstate "${TMPDIR_LOCAL}/model.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_mtp_session.txt"
	if [[ "$TWOCOL" == auto ]]; then
		printf 'auto' >"${TMPDIR_LOCAL}/d3d12twocol.txt"
		upload_to_localstate "${TMPDIR_LOCAL}/d3d12twocol.txt"
		echo "  twocol=auto: session scenarios run the allowlisted NEW tile"
	elif [[ "$TWOCOL" != off ]]; then
		echo "Error: --twocol must be off|auto" >&2
		return 1
	else
		echo "  twocol=off: clean OLD arm"
	fi
	upload_to_localstate "${TMPDIR_LOCAL}/bench_session_npredict.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_session_nbatch.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_run_index.txt"
	if ((N_THREADS > 0)); then
		printf '%d' "$N_THREADS" >"${TMPDIR_LOCAL}/bench_threads.txt"
		upload_to_localstate "${TMPDIR_LOCAL}/bench_threads.txt"
	fi
	if ((MTP_N != 0)); then
		printf '%d' "$MTP_N" >"${TMPDIR_LOCAL}/bench_mtp.txt"
		upload_to_localstate "${TMPDIR_LOCAL}/bench_mtp.txt"
	fi
	if ((MTP_PMIN >= 0)); then
		printf '%d' "$MTP_PMIN" >"${TMPDIR_LOCAL}/bench_mtp_pmin.txt"
		upload_to_localstate "${TMPDIR_LOCAL}/bench_mtp_pmin.txt"
	fi
	upload_to_localstate "${TMPDIR_LOCAL}/bench_gpu_layers.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_ctx.txt"
	# The ON/OFF arm of the instrumentation-cost measurement. This gate has its own
	# upload block, so without it the knob never reached the device and the OFF arm
	# silently measured ON — the PHASE line prints profile= to make that visible.
	printf '%d' "$PROFILE_PHASES" >"${TMPDIR_LOCAL}/bench_profile.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_profile.txt"

	restart_app
	if ! wait_for_marker "bench-mtp-session.csv.done" 1800; then
		# A timeout still leaves partial evidence: whatever the device wrote is
		# collected before persisting, otherwise the failure is indistinguishable
		# from "the console did nothing".
		echo "  Marker timeout; collecting partial evidence before failing." >&2
		download_from_localstate "bench-mtp-session.csv" "${TMPDIR_LOCAL}/session.csv" || true
		for f in "${dumps[@]}"; do
			download_from_localstate "$f" "${TMPDIR_LOCAL}/$f" || true
		done
		if [[ -n "${SESSION_LOG_OUT:-}" ]]; then
			download_from_localstate "xllama.log" "$SESSION_LOG_OUT" || true
		fi
		persist_session_evidence
		echo "Error: the console never wrote bench-mtp-session.csv.done." >&2
		echo "  Partial evidence kept in $evidence" >&2
		return 1
	fi
	download_from_localstate "bench-mtp-session.csv" "${TMPDIR_LOCAL}/session.csv" || true
	for f in "${dumps[@]}"; do
		download_from_localstate "$f" "${TMPDIR_LOCAL}/$f" || true
	done
	# The device log is the evidence for the effective backend and knobs: the host
	# tag records what was REQUESTED, not what ran.
	if [[ -n "${SESSION_LOG_OUT:-}" ]]; then
		# fetch-file gives the WHOLE device log, which accumulates across earlier runs
		# and invocations. The markers written above bracket this run exactly, so the
		# evidence keeps the full log AND the delimited slice — never the accumulation.
		download_from_localstate "xllama.log" "$SESSION_LOG_OUT" || true
		python3 - "$SESSION_LOG_OUT" "$SESSION_LOG_OUT.slice.log" <<'PYSLICE'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src, errors="replace").read().splitlines()
starts = [i for i, l in enumerate(lines) if "MTP_SESSION_CONFIG" in l]
if not starts:
    open(dst, "w").write("")
    raise SystemExit(0)
# Everything after the LAST MTP_SESSION_CONFIG: this invocation only.
seg = lines[starts[-1]:]
cfg = seg[0]
body, tail = [], []
for l in seg[1:]:
    if "bench-mtp-session.csv written" in l:
        tail.append(l)
        break
    body.append(l)
out = [f"# run-scoped slice (last MTP_SESSION_CONFIG .. csv written) of {src}"]
out.append(cfg)
out.append(f"# segments in whole log: {len(starts)} (only the last belongs to this run)")
out.extend(body)
out.extend(tail)
open(dst, "w").write("\n".join(out) + "\n")
print(f"  log slice: {len(body)} lines from the last of {len(starts)} segment(s)")
PYSLICE
	fi
	persist_session_evidence
	echo "  Effective-config lines from the device log:"
	if [[ -n "${SESSION_LOG_OUT:-}" && -s "${SESSION_LOG_OUT}" ]]; then
		grep -aE 'MTP_SESSION_CONFIG|MPT_SCENARIO|mtp: draft ready|d3d12' "$SESSION_LOG_OUT" \
			|tail -20 | sed 's/^/    /' || true
	else
		echo "    (log unavailable; effective backend not evidenced)" >&2
	fi

	local expect_session=1 expect_mtp=0 expect_pmin=-1 expect_gpu=0
	((MTP_N != 0)) && expect_mtp="$MTP_N"
	((MTP_PMIN >= 0)) && expect_pmin="$MTP_PMIN"
	((GPU_LAYERS > 0)) && expect_gpu="$GPU_LAYERS"

	echo ""
	echo "=== MTP Session parity gate ==="
	python3 - "$TMPDIR_LOCAL" "$expect_session" "$expect_mtp" "$expect_pmin" "$expect_gpu" <<'PY'
import csv, pathlib, sys

d = pathlib.Path(sys.argv[1])
want_session = sys.argv[2] == "1"
want_mtp = int(sys.argv[3])
want_pmin = int(sys.argv[4])
want_gpu = int(sys.argv[5])
pairs = [
    ("reset_prompt_swap", "bench-mtp-s1-t2.txt", "bench-mtp-s1-ref.txt"),
    ("edited_prefix", "bench-mtp-s2-t2.txt", "bench-mtp-s2-ref.txt"),
    ("delta_continuation", "bench-mtp-s3-t2.txt", "bench-mtp-s3-ref.txt"),
    ("multi_chunk", "bench-mtp-s4-t.txt", "bench-mtp-s4-ref.txt"),
]

def load(path):
    """ids, or None. Missing / empty / unparsable is None — never [] — so an
    absent dump can never compare equal and pass as parity."""
    p = d / path
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        vals = [int(x) for x in p.read_text().split()]
    except ValueError:
        return None
    return vals or None

csv_path = d / "session.csv"
if not csv_path.exists():
    print("FAIL: bench-mtp-session.csv was not downloaded")
    sys.exit(1)
with csv_path.open() as fh:
    # Strict: a malformed row (e.g. an unquoted comma shifting columns, once
    # observed as int() on a reason fragment) must FAIL loudly here, never be
    # silently reinterpreted into shifted columns.
    try:
        raw = list(csv.DictReader(fh, strict=True))
    except csv.Error as e:
        print(f"FAIL: bench-mtp-session.csv is malformed: {e}")
        sys.exit(1)
    ncols = 17
    bad = [i + 2 for i, r in enumerate(raw) if r is None or len(r) != ncols]
    if bad:
        print(f"FAIL: bench-mtp-session.csv has malformed rows (want {ncols} fields): {bad}")
        sys.exit(1)
    device = {row["scenario"]: row for row in raw}

print(f"{'scenario':>20} {'ok':>3} {'active':>6} {'agg_draft':>9} {'mtp_draft':>9} "
      f"{'rounds':>6} {'lk_draft':>8} {'n_ids':>6} {'n_ref':>6} "
      f"{'parity':>8} {'first_diff':>11}  reason")
failures = []
for name, got_f, ref_f in pairs:
    row = device.get(name)
    if row is None:
        print(f"{name:>20} {'-':>3} {'-':>6} {'-':>10} {'-':>6} {'-':>6} {'NO_ROW':>8} {'-':>11}")
        failures.append(f"{name}: absent from the device CSV")
        continue
    a, b = load(got_f), load(ref_f)
    reason = row.get("reason", "-") or "-"
    agg = row.get("n_drafted", "?")
    mtp_d = row.get("mtp_drafted", "0") or "0"
    mtp_r = row.get("mtp_rounds", "0") or "0"
    lk_d = row.get("lookup_drafted", "0") or "0"
    if a is None or b is None:
        print(f"{name:>20} {row.get('ok','?'):>3} {row.get('mtp_active','?'):>6} "
              f"{agg:>9} {mtp_d:>9} {mtp_r:>6} {lk_d:>8} {'-':>6} {'-':>6} "
              f"{'NO_DUMP':>8} {'-':>11}  {reason}")
        failures.append(f"{name}: token dump missing, empty or unparsable")
        continue
    first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    parity = first is None and len(a) == len(b)
    print(f"{name:>20} {row.get('ok','?'):>3} {row.get('mtp_active','?'):>6} {agg:>9} "
          f"{mtp_d:>9} {mtp_r:>6} {lk_d:>8} {len(a):>6} {len(b):>6} "
          f"{'OK' if parity else 'DIVERGE':>8} {('-' if first is None else first):>11}  {reason}")
    if first is not None:
        print(f"    first divergence at index {first}: got={a[first]} ref={b[first]}")
        print("    NOTE: the logit margin at that index is NOT instrumented (open gap) —")
        print("          this is a reported failure, never an approval")
    if row.get("ok") != "1":
        failures.append(f"{name}: device reported ok=0 ({reason})")
    if row.get("mtp_active") != "1":
        failures.append(f"{name}: MTP inactive after the reset/rewind")
    # MTP must have PROPOSED, not merely been active: with the n-gram lookup also
    # on, the aggregate n_drafted can be non-zero from lookup tokens alone.
    if int(mtp_r) <= 0:
        failures.append(f"{name}: no MTP round reached a verify batch")
    if int(mtp_d) <= 0:
        failures.append(f"{name}: the MTP head proposed nothing "
                        f"(aggregate={agg}, lookup={lk_d})")
    if not parity:
        failures.append(f"{name}: ids differ (first_diff={first})")
    host = row.get("host", "")
    if want_session and "-session" not in host:
        failures.append(f"{name}: device host tag lacks -session ({host})")
    if want_mtp and f"-mtp{want_mtp}" not in host:
        failures.append(f"{name}: device host tag lacks -mtp{want_mtp} ({host})")
    if want_mtp and want_pmin >= 0 and f"-pmin{want_pmin}" not in host:
        failures.append(f"{name}: device host tag lacks -pmin{want_pmin} ({host})")
    if want_gpu and f"-g{want_gpu}" not in host:
        failures.append(f"{name}: device host tag lacks -g{want_gpu} ({host})")

    # The delta verdict only counts when the histories are provably the same
    # token sequence. Recomputed here from the dumps, so a device-side "ok" is
    # corroborated rather than trusted.
    if name == "delta_continuation":
        t1_pre = load("bench-mtp-s3-t1.txt.prefill")
        t1_out = load("bench-mtp-s3-t1.txt")
        t2_pre = load("bench-mtp-s3-t2.txt.prefill")
        seq_t1_pre = load("bench-mtp-s3-seq-t1.txt.prefill")
        seq_t1_out = load("bench-mtp-s3-seq-t1.txt")
        seq_t2_pre = load("bench-mtp-s3-seq-t2.txt.prefill")
        seq_t2_out = load("bench-mtp-s3-seq-t2.txt")
        cold_pre = load("bench-mtp-s3-ref.txt.prefill")
        if not all((t1_pre, t1_out, t2_pre, seq_t1_pre, seq_t1_out, seq_t2_pre,
                    seq_t2_out, cold_pre)):
            failures.append("delta_continuation: a history dump is missing or empty")
        else:
            if t1_pre != seq_t1_pre:
                failures.append("delta_continuation: MTP-off control has different turn-1 prefill ids")
            if t2_pre != seq_t2_pre:
                failures.append("delta_continuation: MTP-off control has different turn-2 prefill ids")
            # BOTH turns' emitted ids, not just the delta text: "same history"
            # means the whole sequence the control produced.
            if t1_out != seq_t1_out:
                n = min(len(t1_out), len(seq_t1_out))
                od = next((i for i in range(n) if t1_out[i] != seq_t1_out[i]), min(n, 0))
                failures.append(
                    "delta_continuation: MTP-off control emitted different turn-1 ids "
                    f"(first_diff={od}) — not the same history")
            if a != seq_t2_out:
                failures.append("delta_continuation: MTP output ids differ from the MTP-off control")
            expected = t1_pre + t1_out + t2_pre
            if expected != cold_pre:
                n = min(len(expected), len(cold_pre))
                hd = next((i for i in range(n) if expected[i] != cold_pre[i]),
                          min(len(expected), len(cold_pre)))
                failures.append(
                    "delta_continuation: cold reference history is not token-for-token the "
                    f"resident history (first_diff={hd}, {len(expected)} vs {len(cold_pre)} ids) "
                    "— reference failure, not an MTP verdict")

print("")
if failures:
    for f in failures:
        print(f"FAIL {f}")
    sys.exit(1)
print("OK: every scenario kept MTP active with n_drafted>0 and matched the reference ids")
PY
	local gate_rc=$?
	persist_session_evidence
	return $gate_rc
}

if ((MTP_SESSION != 0)); then
	mtp_session_gate
	exit $?
fi

for ((run = 1; run <= N_RUNS; run++)); do
	verify_expected_package
	echo ""
	echo "--- Run $run / $N_RUNS ---"

	delete_from_localstate "bench-result.csv"
	delete_from_localstate "bench-result.csv.done"
	# Confirm the marker is actually gone before starting the app, so wait_for_done
	# can't return off a stale bench-result.csv.done from the previous run.
	verify_deleted "bench-result.csv.done"
	# Remove any leftover bench_turns.txt: main_loop treats its presence as the
	# KV-reuse bench trigger, which would hijack this standard bench into kv-bench
	# mode (writes bench-kv-result.csv, never bench-result.csv → this run times out).
	delete_from_localstate "bench_turns.txt"
	# Same for the Session parity mode: it takes precedence over this bench and
	# writes bench-mtp-session.csv, so a leftover from a previous gate would make
	# this run time out instead of measuring.
	delete_from_localstate "bench_mtp_session.txt"
	verify_deleted "bench_mtp_session.txt" 5 || true
	sleep 1

	# run_index changes per iteration (unlike the other bench_*.txt knobs), so it
	# is written inside the loop: the device echoes it into the CSV row so repeats
	# are individually recoverable rather than pre-averaged here (W1.1).
	printf '%d' "$run" >"${TMPDIR_LOCAL}/bench_run_index.txt"

	echo "  Uploading bench artifacts..."
	upload_to_localstate "${TMPDIR_LOCAL}/bench.flag"
	upload_to_localstate "${TMPDIR_LOCAL}/prompt.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/model.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_threads.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_ctx.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_npredict.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_maxlen.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_ubatch.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_n_batch.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_kvq8.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_gpu_layers.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_ignore_eog.txt"
	if ((MTP_N != 0)); then
		upload_to_localstate "${TMPDIR_LOCAL}/bench_mtp.txt"
	else
		delete_from_localstate "bench_mtp.txt"
		verify_deleted "bench_mtp.txt" 5 || true
	fi
	if ((MTP_PMIN >= 0)); then
		upload_to_localstate "${TMPDIR_LOCAL}/bench_mtp_pmin.txt"
	else
		delete_from_localstate "bench_mtp_pmin.txt"
		verify_deleted "bench_mtp_pmin.txt" 5 || true
	fi
	# Plan 004: two-column knob, same conditional pattern (a stale auto file
	# would silently flip the run to NEW).
	if [[ "$TWOCOL" == auto ]]; then
		printf 'auto' >"${TMPDIR_LOCAL}/d3d12twocol.txt"
		upload_to_localstate "${TMPDIR_LOCAL}/d3d12twocol.txt"
	elif [[ "$TWOCOL" != off ]]; then
		echo "Error: --twocol must be off|auto" >&2
		return 1
	else
		delete_from_localstate "d3d12twocol.txt"
		verify_deleted "d3d12twocol.txt" 5 || true
	fi
	# Plan 003, stage 1: greedy/seed/tokens knobs. Always uploaded so a previous
	# run cannot leave them in force; the bridge treats 0 as "not set".
	upload_to_localstate "${TMPDIR_LOCAL}/bench_greedy.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_seed.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_tokens.txt"

	if ((PROMPT_LOOKUP != 0)); then
		upload_to_localstate "${TMPDIR_LOCAL}/bench_prompt_lookup.txt"
	else
		# A prior --prompt-lookup run must not leave the knob on.
		delete_from_localstate "bench_prompt_lookup.txt"
		verify_deleted "bench_prompt_lookup.txt" 5 || true
	fi
	# Always sent, so the ON/OFF arm is explicit instead of inherited: a missing
	# knob reads as ON and would make the OFF run measure ON.
	upload_to_localstate "${TMPDIR_LOCAL}/bench_profile.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_run_index.txt"

	echo "  Starting app..."
	restart_app

	if ! wait_for_done 300; then
		echo "  Run $run timed out — skipping." >&2
		continue
	fi

	local_csv="${TMPDIR_LOCAL}/run${run}.csv"
	download_from_localstate "bench-result.csv" "$local_csv"
	# Device log per run when asked: the PHASE lines live only there, and the ON/OFF
	# comparison needs them for the same package/GGUF/knobs as the CSV row.
	if [[ -n "${RUN_LOG_DIR:-}" ]]; then
		mkdir -p "$RUN_LOG_DIR"
		download_from_localstate "xllama.log" "$RUN_LOG_DIR/run${run}.log" || true
	fi
	# Plan 003, stage 1: per-run token sidecar (accepted ids + text hash). The
	# bridge writes bench-tokens-<run_index>.txt when bench_tokens.txt=1; a
	# missing sidecar on a run that asked for one is a parity failure, not a
	# skip (same rule as a drafter that did not activate).
	if ((DUMP_TOKENS != 0)); then
		tok_local="${TMPDIR_LOCAL}/run${run}.tokens"
		if download_from_localstate "bench-tokens-${run}.txt" "$tok_local" \
			&& [[ -s "$tok_local" ]]; then
			echo "  Tokens: $tok_local ($(wc -l <"$tok_local") lines)"
		else
			echo "Error: bench_tokens.txt=1 but bench-tokens-${run}.txt is missing." >&2
			echo "  The installed MSIX predates the token sidecar — redeploy before parity runs." >&2
			exit 1
		fi
	fi
	data_row=$(tail -n +2 "$local_csv" 2>/dev/null | head -1)
	if [[ -n "$data_row" ]]; then
		# The device writes the row; this script owns the header. A build older
		# than the current schema emits fewer fields and they get appended under
		# our header anyway, shifting every column silently (a 14-field row from a
		# pre-n_gen_tok build puts `host` under n_gen_tok and `date` under host).
		# assert_header_matches only guards the local file, so nothing else here
		# would notice. Observed 2026-07-21 with MSIX 1.4.0.615.
		n_fields=$(awk -F, '{print NF}' <<<"$data_row")
		want_fields=$(awk -F, '{print NF}' <<<"$CSV_HEADER")
		if ((n_fields != want_fields)); then
			echo "Error: the console wrote a ${n_fields}-field row; this script expects ${want_fields}." >&2
			echo "  row:    $data_row" >&2
			echo "  header: $CSV_HEADER" >&2
			echo "The installed MSIX is older than the CSV schema — redeploy before benchmarking." >&2
			exit 1
		fi
		# Before anything else: did the device measure the model we ASKED for?
		# model.txt reaches the console over WDP, and a WDP POST can report
		# success while writing nothing (the missing-X-CSRF-Token failure mode
		# documented in uwp-constraints). A build older than 1.5.2 answered a
		# missing model.txt by silently benching smollm2-360m-cpu-int4, so a lost
		# upload produced a genuine row for the wrong model, appended to the
		# results file of the run that was requested. Current builds refuse and
		# write no CSV; this check also covers the older ones, and any future way
		# the two can drift apart.
		got_model=$(awk -F, '{print $1}' <<<"$data_row")
		if [[ "$got_model" != "$MODEL_NAME" ]]; then
			echo "Error: the console benched '${got_model}', not the requested '${MODEL_NAME}'." >&2
			echo "  model.txt likely never reached the device (WDP writes can fail silently)." >&2
			echo "  Nothing was appended — re-run rather than trusting this row." >&2
			exit 1
		fi
		# The CSV schema does NOT change with --max-length, so the arity check
		# above cannot catch an MSIX that ignores bench_maxlen.txt — it would
		# record DERIVED max_lengths under a "saturated" label and the whole
		# experiment would be quietly wrong. Compare what we asked for against
		# what the device reports in the max_length column.
		if ((MAX_LEN != 0)); then
			got=$(awk -F, '{print $14}' <<<"$data_row")
			if ((MAX_LEN < 0)); then
				want=$((N_CTX > 0 ? N_CTX : 2048))
			else
				want=$MAX_LEN
			fi
			if ((got != want)); then
				echo "Error: the console ignored --max-length: row says ${got}, asked ${want}." >&2
				echo "  The installed MSIX predates bench_maxlen.txt — redeploy before measuring." >&2
				exit 1
			fi
		fi
		# Same hazard for --ubatch: the schema has no ubatch column, so an MSIX
		# that ignores bench_ubatch.txt would record default-512 rows under a
		# sweep label. The device tags the host column with -uN; require it.
		if ((UBATCH != 0)); then
			got_host=$(awk -F, '{print $15}' <<<"$data_row")
			if [[ "$got_host" != *"-u${UBATCH}"* ]]; then
				echo "Error: the console ignored --ubatch: host column says '${got_host}', asked -u${UBATCH}." >&2
				echo "  The installed MSIX predates bench_ubatch.txt — redeploy before measuring." >&2
				exit 1
			fi
		fi
		# And for --kv-q8 (#171): require the -kvq8 host tag.
		if ((KVQ8 != 0)); then
			got_host=$(awk -F, '{print $15}' <<<"$data_row")
			if [[ "$got_host" != *"-kvq8"* ]]; then
				echo "Error: the console ignored --kv-q8: host column says '${got_host}'." >&2
				echo "  The installed MSIX predates bench_kvq8.txt — redeploy before measuring." >&2
				exit 1
			fi
		fi
		if ((IGNORE_EOG != 0)); then
			got_host=$(awk -F, '{print $15}' <<<"$data_row")
			if [[ "$got_host" != *"-noeog"* ]]; then
				echo "Error: the console ignored --ignore-eog: host column says '${got_host}'." >&2
				exit 1
			fi
		fi
		# Plan 003, stage 1: parity knobs must be honoured — a bench that asked
		# for greedy or a fixed seed but got the engine default would produce a
		# row that cannot be diffed token-by-token. The host column carries the
		# tags, same contract as -uN / -kvq8 / -noeog.
		if ((GREEDY != 0)); then
			got_host=$(awk -F, '{print $15}' <<<"$data_row")
			if [[ "$got_host" != *"-greedy"* ]]; then
				echo "Error: the console ignored --greedy: host column says '${got_host}'." >&2
				echo "  The installed MSIX predates bench_greedy.txt — redeploy before parity runs." >&2
				exit 1
			fi
		fi
		if ((SEED != 0)); then
			got_host=$(awk -F, '{print $15}' <<<"$data_row")
			if [[ "$got_host" != *"-s${SEED}"* ]]; then
				echo "Error: the console ignored --seed ${SEED}: host column says '${got_host}'." >&2
				echo "  The installed MSIX predates bench_seed.txt — redeploy before parity runs." >&2
				exit 1
			fi
		fi
		# And for --gpu-layers (D2b): require the -gN tag and the d3d12 backend
		# column — a CPU fallback (device unavailable) must not pass as a GPU row.
		if ((GPU_LAYERS > 0)); then
			got_host=$(awk -F, '{print $15}' <<<"$data_row")
			got_backend=$(awk -F, '{print $3}' <<<"$data_row")
			if [[ "$got_host" != *"-g${GPU_LAYERS}"* || "$got_backend" != "d3d12" ]]; then
				echo "Error: --gpu-layers ${GPU_LAYERS} not honoured: host '${got_host}', backend '${got_backend}'." >&2
				echo "  Old MSIX (no bench_gpu_layers.txt) or the d3d12 device failed — see get-log." >&2
				exit 1
			fi
		fi
		CSV_ROWS+=("$data_row")
		echo "  Row: $data_row"
	else
		echo "  Warning: bench-result.csv empty or missing" >&2
	fi
done

if [[ -n "$SAMPLER_PID" ]]; then
	kill -TERM "$SAMPLER_PID" 2>/dev/null || true
	wait "$SAMPLER_PID" 2>/dev/null || true
	if [[ -s "${TMPDIR_LOCAL}/gpu-summary.txt" ]]; then
		echo ""
		cat "${TMPDIR_LOCAL}/gpu-summary.txt"
	fi
fi

# Persist the per-run token sidecars next to the CSV when --tokens was asked:
# TMPDIR_LOCAL is removed by the EXIT trap, and the pipeline diffs these files
# between baseline and MTP arms (plan 003, stage 1).
if ((DUMP_TOKENS != 0)); then
	RESULT_CSV="${OUT_CSV:-${REPO_ROOT}/bench/results/phase1-cpu.csv}"
	RESULT_DIR="$(dirname "$RESULT_CSV")"
	for tok in "${TMPDIR_LOCAL}"/run*.tokens; do
		[[ -e "$tok" ]] || continue
		run=$(basename "$tok" .tokens)
		cp "$tok" "${RESULT_DIR}/$(basename "$RESULT_CSV" .csv).${run}.tokens"
	done
fi

# ---------------------------------------------------------------------------
# Append each recorded run to the results CSV
# ---------------------------------------------------------------------------
# W1.1: the driver no longer pre-averages. Every recorded run is appended with the
# run_index the device wrote, so the spread is recoverable from the committed CSV;
# scripts/generate-benchmark-summary.py computes the median and min-max from them.
echo ""
echo "--- Appending recorded runs ---"
RESULT_CSV="${OUT_CSV:-${REPO_ROOT}/bench/results/phase1-cpu.csv}"

# Warmup drop matches the previous median's warmup exclusion: run 1 (CSV_ROWS[0])
# is a cold cache and is not recorded, UNLESS it is the only run collected.
if ((${#CSV_ROWS[@]} == 0)); then
	echo "Error: no successful runs collected." >&2
	exit 1
elif ((${#CSV_ROWS[@]} < 2)); then
	echo "Warning: only 1 run collected (warmup not dropped)." >&2
	ROWS_TO_APPEND=("${CSV_ROWS[0]}")
else
	ROWS_TO_APPEND=("${CSV_ROWS[@]:1}") # skip warmup (run 1 = index 0)
fi

# Route DML rows to phase2-dml.csv when --out was not given. All rows in one
# campaign share a backend, so decide from the first row to append.
if [[ -z "$OUT_CSV" && "$(cut -d, -f3 <<<"${ROWS_TO_APPEND[0]}")" == *dml* ]]; then
	RESULT_CSV="${REPO_ROOT}/bench/results/phase2-dml.csv"
fi
[[ ! -f "$RESULT_CSV" ]] && printf '%s\n' "$CSV_HEADER" >"$RESULT_CSV"

# Refuse to append to a file written under an older schema: the row arity would
# not match its header, and csv.DictReader would silently misalign every field
# after the mismatch (host/date landing under the wrong keys). Pick a new --out.
assert_header_matches() {
	local csv="$1" existing
	existing=$(head -1 "$csv")
	if [[ "$existing" != "$CSV_HEADER" ]]; then
		echo "Error: $csv uses a different CSV schema." >&2
		echo "  file:     $existing" >&2
		echo "  expected: $CSV_HEADER" >&2
		echo "Append to a new file with --out, or migrate the old one first." >&2
		exit 1
	fi
}
assert_header_matches "$RESULT_CSV"

for row in "${ROWS_TO_APPEND[@]}"; do
	printf '%s\n' "$row" >>"$RESULT_CSV"
	echo "  appended: $row"
done
echo "Appended ${#ROWS_TO_APPEND[@]} run(s) to $RESULT_CSV"

echo ""
echo "Done. ${RESULT_CSV} updated."
