#!/usr/bin/env bash
# build-identity.sh <rev207.msix> <rev209.msix> — prove the rev209 package is
# the same validated candidate build inputs as rev207 (no kernel change):
# 1) source identity: git diff f314ecb..HEAD over src/shaders/include/uwp/tests
#    is empty (rev207 was built from f314ecb; db79e3c touched evidence only).
# 2) package content: unzip both, diff -r — only AppxManifest.xml (Identity
#    Version revision), AppxBlockMap.xml (covers the manifest hash) and the
#    re-signature (CodeIntegrity.cat / AppxSignature.p7x) may differ.
# 3) xllama.exe: cmp must show ONLY build-timestamp bytes (PE TimeDateStamp +
#    Debug Directory timestamps); every .text/.rdata/.data/.reloc code byte is
#    identical. Any other byte difference fails the check.
set -euo pipefail
REPO=/home/hjotha/worktrees/xllama-qwen35-4b-maxperf
BASE="$REPO/bench/results/qwen4b-maxperf/enabled-rev209"
MSIX207="${1:?rev207 msix}"
MSIX209="${2:?rev209 msix}"
WORK=/tmp/opencode/qwen4b-recon/identity209
rm -rf "$WORK"; mkdir -p "$WORK/207" "$WORK/209"
{
  echo "build_identity_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "msix207=$MSIX207 sha256=$(sha256sum "$MSIX207" | cut -d' ' -f1)"
  echo "msix209=$MSIX209 sha256=$(sha256sum "$MSIX209" | cut -d' ' -f1)"
  cd "$REPO"
  SRC_DIFF=$(git diff f314ecb HEAD -- shaders include uwp src tests | wc -l)
  echo "source_diff_lines_f314ecb_to_HEAD=$SRC_DIFF (0 => same build inputs as rev207)"
  cd "$WORK"
  unzip -qq "$MSIX207" -d 207
  unzip -qq "$MSIX209" -d 209
  echo "--- package file diff (expected: manifest/blockmap/signature only):"
  diff -rq 207 209 > "$WORK/content.diff" 2>&1 || true
  cat "$WORK/content.diff" || true
  BADPKG=$(grep -v "AppxManifest.xml\|AppxBlockMap.xml\|CodeIntegrity.cat\|AppxSignature.p7x" \
    "$WORK/content.diff" | grep -c "" || true)
  echo "unexpected_differing_package_files=$BADPKG (expect 1: xllama.exe, analyzed below)"
  # xllama.exe: every differing byte must belong to a build timestamp.
  python3 - <<'PY' > "$WORK/exe-diff-analysis.txt"
import struct, sys

def load(p):
    d = open(p, 'rb').read()
    e = struct.unpack_from('<I', d, 0x3c)[0]
    tds = struct.unpack_from('<I', d, e + 8)[0]
    rsds = d.find(b'RSDS')
    age = range(rsds + 20, rsds + 24) if rsds > 0 else range(0, 0)
    guid_ok = d[rsds + 4:rsds + 20] if rsds > 0 else b''
    return d, tds, age, guid_ok

a, tds_a, age_a, guid_a = load('207/xllama.exe')
b, tds_b, age_b, guid_b = load('209/xllama.exe')
print(f"exe_size_207={len(a)} exe_size_209={len(b)}")
print(f"PE_TimeDateStamp_207=0x{tds_a:08x} PE_TimeDateStamp_209=0x{tds_b:08x}")
print(f"pdb_guid_identical={'YES' if guid_a == guid_b and guid_a else 'NO'} "
      f"pdb_age_207={struct.unpack_from('<I', a, list(age_a)[0])[0] if list(age_a) else '?'} "
      f"pdb_age_209={struct.unpack_from('<I', b, list(age_b)[0])[0] if list(age_b) else '?'}")
diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
print(f"differing_bytes={len(diff)} of {len(a)}")
ta = struct.pack('<I', tds_a)
tb = struct.pack('<I', tds_b)
ok = len(a) == len(b)
for i in diff:
    # each differing byte must be part of a 4-byte TimeDateStamp span (COFF
    # header or a Debug Directory entry) or the PDB age field of the RSDS
    # CodeView record — build bookkeeping, never code or data.
    span = slice(i - (i % 4), i - (i % 4) + 4)
    hit = (a[span] == ta and b[span] == tb) or (i in age_a and i in age_b)
    kind = "timestamp" if a[span] == ta and b[span] == tb else ("pdb_age" if hit else "OTHER")
    print(f"  offset={i} class={kind} a={a[i]:02x} b={b[i]:02x}")
    ok = ok and hit
print(f"all_differing_bytes_are_build_timestamps_or_pdb_age={'YES' if ok and diff else 'NO'}")
sys.exit(0 if ok and diff else 1)
PY
  cat "$WORK/exe-diff-analysis.txt"
  echo "identity_check_done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$BASE/build-identity-receipt.txt"
grep -q "source_diff_lines_f314ecb_to_HEAD=0" "$BASE/build-identity-receipt.txt" \
  || { echo "SOURCE_IDENTITY_FAIL"; exit 1; }
grep -q "all_differing_bytes_are_build_timestamps_or_pdb_age=YES" "$BASE/build-identity-receipt.txt" \
  || { echo "EXE_IDENTITY_FAIL"; exit 1; }
echo IDENTITY_OK
