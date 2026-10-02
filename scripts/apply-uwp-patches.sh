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
	# git apply -3 returns non-zero both when the 3-way merge conflicts and when
	# it cannot run at all (missing base blobs in a shallow clone), so the exit
	# code alone cannot gate this. Apply, then decide from the index: unmerged
	# entries mean "resolvable conflict", no entries and non-zero means the
	# submodule is missing the base commit and the build must not continue.
	if ! git apply -3 "$patch" 2>/dev/null && ! git ls-files -u | grep -q .; then
		echo "apply-uwp-patches: ${name} does not apply (3-way could not run); aborting." >&2
		exit 1
	fi
	if git ls-files -u --error-unmatch src/llama-mmap.cpp >/dev/null 2>&1; then
		# "python", not "python3": on Windows python3 is the Microsoft Store
		# alias stub, which prints "Python was not found" and exits non-zero.
		PY_BIN=python
		command -v python >/dev/null 2>&1 || PY_BIN=python3
		"$PY_BIN" - <<'PYEOF'
# Resolve the llama-mmap.cpp 3-way conflict without regex: the fork guards
# the Windows branch with LLAMA_WINDOWS_DESKTOP, a macro it never defines,
# while the patch uses the AppContainer-aware test that is correct on desktop
# and on Xbox. Line-oriented so no escaping survives the shell heredoc.
path = "src/llama-mmap.cpp"
good = ("#if defined(_POSIX_MEMLOCK_RANGE) || (defined(_WIN32) && "
        "(!defined(WINAPI_FAMILY) || WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)))")
lines = open(path).read().splitlines(True)
out = []
i = 0
fixed = 0
while i < len(lines):
    if (lines[i].startswith("<<<<<<<") and i + 4 < len(lines)
            and lines[i + 1].startswith("#if") and "LLAMA_WINDOWS_DESKTOP" in lines[i + 1]
            and lines[i + 2].startswith("=======") and lines[i + 3].startswith("#if")
            and lines[i + 4].startswith(">>>>>>>")):
        out.append(good + "\n")
        fixed += 1
        i += 5
        continue
    out.append(lines[i])
    i += 1
if any(l.startswith("<<<<<<<") for l in out):
    raise SystemExit("apply-uwp-patches: unresolved conflict left in " + path)
if fixed == 0:
    raise SystemExit("apply-uwp-patches: no llama-mmap.cpp hunk matched in " + path)
open(path, "w").write("".join(out))
print("apply-uwp-patches: resolved %d llama-mmap.cpp hunk(s)" % fixed)
PYEOF'
import re

path = "src/llama-mmap.cpp"
text = open(path).read()

# Keep the patch's AppContainer-aware test instead of the fork's
# LLAMA_WINDOWS_DESKTOP, which the fork never defines anywhere.
#
# The pattern is built by concatenation on purpose: inside a raw string "\n"
# is a backslash followed by "n" and would never match a newline, and a single
# non-raw "\n" here would already be consumed one escaping level too early by
# writing it through a shell heredoc.
NL = chr(10)
good = ("#if defined(_POSIX_MEMLOCK_RANGE) || (defined(_WIN32) && (!defined(WINAPI_FAMILY) || "
        "WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)))")
BS = chr(92)   # backslash, so "\n" can be written as BS+"n"
ANY = "[^" + BS + "n]*"
pattern = ("<<<<<<< ours" + NL + "#if" + ANY + "LLAMA_WINDOWS_DESKTOP" + ANY + NL +
           "=======" + NL + "#if" + ANY + NL + ">>>>>>> theirs")
]*LLAMA_WINDOWS_DESKTOP[^
]*" + NL +
           "=======" + NL + r"#if[^
]*" + NL + ">>>>>>> theirs")
text, n = re.subn(pattern, good, text)

if "<<<<<<<" in text:
    raise SystemExit("apply-uwp-patches: unresolved conflict left in " + path)
if n == 0:
    raise SystemExit("apply-uwp-patches: no llama-mmap.cpp hunk matched in " + path)
open(path, "w").write(text)
print("apply-uwp-patches: resolved %d llama-mmap.cpp hunk(s)" % n)
PYEOF'
import re

path = "src/llama-mmap.cpp"
text = open(path).read()

# Keep the patch's AppContainer-aware test instead of the fork's
# LLAMA_WINDOWS_DESKTOP, which the fork never defines anywhere. Non-raw regex:
# Plain concatenated strings, not a raw string: the newlines in the pattern
# have to be real \n escapes for re, and the heredoc is quoted so the shell
# leaves them alone.
good = ("#if defined(_POSIX_MEMLOCK_RANGE) || (defined(_WIN32) && (!defined(WINAPI_FAMILY) || "
        "WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)))")
pattern = ("<<<<<<< ours" + chr(10) + r"#if[^
]*LLAMA_WINDOWS_DESKTOP[^
]*" + chr(10) +
           "=======" + chr(10) + r"#if[^
]*" + chr(10) + ">>>>>>> theirs")
text, n = re.subn(pattern, good, text)

if "<<<<<<<" in text:
    raise SystemExit("apply-uwp-patches: unresolved conflict left in " + path)
if n == 0:
    raise SystemExit("apply-uwp-patches: no llama-mmap.cpp hunk matched in " + path)
open(path, "w").write(text)
print("apply-uwp-patches: resolved %d llama-mmap.cpp hunk(s)" % n)
PYEOF'
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
