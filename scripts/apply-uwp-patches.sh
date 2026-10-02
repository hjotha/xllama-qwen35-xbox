#!/usr/bin/env bash
# apply-uwp-patches.sh — apply the UWP build patches to the llama.cpp submodule.
#
# The UWP (Xbox) build compiles ggml/llama sources directly (uwp/ggml-uwp.vcxproj),
# so a few upstream sources need local changes to build and run inside the
# AppContainer, and one needs a change to build under clang-cl at all. Each patch
# and its reason: patches/README.md.
#
# Applies every patches/0*-*.patch in order. Idempotent per patch, so re-running
# after adding a new one applies only the new one. Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT/llama.cpp"
applied=0
skipped=0
for patch in "$ROOT"/patches/0*-*.patch; do
	name="$(basename "$patch")"
	if git apply --reverse --check "$patch" 2>/dev/null; then
		echo "apply-uwp-patches: ${name} already applied."
		skipped=$((skipped + 1))
		continue
	fi
	# -3 (3-way) is required against the beellama fork: plain git apply fails
	# on ggml-backend-dl.cpp, ggml-cpu.c and ggml-cpu.cpp, which drifted on both
	# sides since the upstream pin. On src/llama-mmap.cpp the 3-way merge still
	# conflicts — the fork guards the Windows branch with a LLAMA_WINDOWS_DESKTOP
	# macro it never defines — so that one hunk is resolved in favour of the
	# patch's AppContainer-aware test (correct on desktop and on Xbox alike).
	if ! git apply -3 --check "$patch" 2>/dev/null; then
		echo "apply-uwp-patches: ${name} does not apply (3-way failed); aborting." >&2
		exit 1
	fi
	git apply -3 "$patch"
	if git ls-files -u --error-unmatch src/llama-mmap.cpp >/dev/null 2>&1; then
		python3 - <<'PYEOF'
import re
p = "src/llama-mmap.cpp"
s = open(p).read()
good = ("#if defined(_POSIX_MEMLOCK_RANGE) || (defined(_WIN32) && (!defined(WINAPI_FAMILY) || "
        "WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)))")
s, n = re.subn(
    r"<<<<<<< ours\n#if defined\(_POSIX_MEMLOCK_RANGE\) \|\| defined\(LLAMA_WINDOWS_DESKTOP\)\n"
    r"=======\n#if defined[^\n]*\n>>>>>>> theirs",
    good, s)
if "<<<<<<<" in s:
    raise SystemExit("apply-uwp-patches: unresolved conflict left in " + p)
open(p, "w").write(s)
print("apply-uwp-patches: resolved %d llama-mmap.cpp hunk(s)" % n)
PYEOF
		git add src/llama-mmap.cpp
	fi
	echo "apply-uwp-patches: applied ${name}."
	applied=$((applied + 1))
done
echo "apply-uwp-patches: ${applied} applied, ${skipped} already present, llama.cpp @ $(git rev-parse --short HEAD)"
