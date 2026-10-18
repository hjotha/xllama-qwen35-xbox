# Optional RKVA disk snapshots (future, unqualified candidate)

This implementation has **not been built, deployed, or executed**. Keep the live 259 application and the separately prepared 260 package unchanged during TPS measurements. Its files live in a dedicated source branch; no package version has been assigned.

The host's native cache file remains authoritative on GAMES. Xbox snapshots are an optional accelerator for restoring the overflow cache; they do not replace the host file, CPU companion, prefix KV, recurrent state, token list, logits, model validation, or sampling state. No speed claim is supported until the future measurements below run.

## Contract and lifecycle

RKVA v2 reserves operation 17 for SAVE, operation 18 for RESTORE, and capability bit 3 for local disk snapshots. Existing v1 and 259 clients remain compatible. Operations 15/16 and capability bit 2 remain native KVarN4, without collision. Requests are fixed 48-byte structures containing SHA256 of the complete native host sequence file, the number of live overflow rows, revision 1, and zero flags/reserved. No client-provided pathname reaches the filesystem.

Use only a quiescent evaluated checkpoint. The native host file includes the exact evaluated token prefix and all native KV state. An unprocessed sampled token is not part of that prefix. The Bee patch requires a single sequence, position minimum 0, position maximum equal to `evaluated_tokens - 1`, and exact per-layer uploaded row counts in the current epoch. A host rollback with an older checkpoint and newer remote rows therefore skips SAVE; it never attaches future KV to the older host identity. Neither snapshot operation is sent at or below P.

The console rechecks all layer counts before saving. For native KVarN4 it also requires ready stage/tail state and exactly `floor(live_rows/128)` completed eager record groups. A manifest records the request identity, complete negotiated geometry, global origin P, capacity C-P, actual storage format, descriptor/layout version, WHT/domain/record policy, stage groups, tail groups, precision-tail policy, payload extents, and SHA256 for each payload chunk. The manifest itself has SHA256 integrity protection.

The immutable directory name is the 64 hexadecimal characters of the host digest. Data is streamed through 8 MiB D3D12 READBACK or UPLOAD resources, so a whole overflow copy is never allocated in staging RAM. Standard F16/Q4_0/Q8_0 snapshots store only live K/V rows; native KVarN4 stores completed records, the canonical physical three-slot stage, and the valid precision-tail ring. Every file is bounded to 8 MiB, avoiding the UWP per-file constraint described in [UWP constraints](uwp-constraints.md). No network KV payload is involved in a disk hit.

SAVE writes a temporary directory under the app's LocalState `rkva-snapshots-v1`, closes every checked file, writes the complete checksummed manifest, then atomically renames the directory. A published identity is immutable. The store retains at most four snapshots under an 8 GiB quota, with at most 4 GiB payload per snapshot; oldest unpinned entries and abandoned temporary directories are collected. These are optional cache entries: eviction never deletes a host GAMES snapshot. File close/rename supplies consistent publication, not a claim of durability through arbitrary power loss. Truncated/corrupt files after a crash fail the restore integrity checks.

RESTORE first validates the entire manifest and every payload checksum against the current configuration without mutating live GPU KV. It then checks chunks again while loading, and publishes all layer counters together only after the final successful GPU fence. Stale future rows remain unreachable. A later disk read failure leaves the session invalid; an explicit RESET is required before full host reupload. Missing, corrupt, incompatible, or evicted disk entries fall back to the authoritative host companion. Transport/protocol/GPU failures fail closed.

The Bee integration is opt-in via `GGML_RKVA_DISK_SNAPSHOTS=1` and capability negotiation. It adds SHA256 to the existing host checksum pass during canonical disk publication and native streaming restore; no extra host disk scan is introduced. The Xbox restore runs only after the verified host restore commits. A validated hit updates every backend upload tracker to the new epoch and exact row count; the next attention uploads only appended rows. A miss leaves all upload trackers invalid, forcing the existing full companion upload. Source cache and actual storage remain distinguished.

This first host patch connects canonical persistent disk snapshots and their native streaming import. It intentionally does not save each speculative or in-memory checkpoint, and the legacy non-streaming sequence reader retains the full-upload path. A restored file with no Xbox entry can still be used correctly; the following cold decode uploads its overflow normally. Native KVarN4 console persistence is implemented, but the Bee native KVarN range producer remains a separate future dependency.

## Future validation (not performed)

After TPS work finishes, reserve the console and review/build a new package in an isolated Windows source tree. Do not overlay the signed 260 artifact or the live 259 installation. Apply `bee-disk-snapshots.patch` to a reviewed Bee branch/checkpoint, compile with the existing 20 W governor/fan policy, and use the explicit opt-in only for qualification.

`tests/rkva_disk_snapshot_probe.py` is inert without `--execute`. Future primitive gates can run with:

```bash
python3 tests/rkva_disk_snapshot_probe.py --execute --host XBOX_ADDRESS --storage f16
python3 tests/rkva_disk_snapshot_probe.py --execute --host XBOX_ADDRESS --storage q4_0
python3 tests/rkva_disk_snapshot_probe.py --execute --host XBOX_ADDRESS --storage q8_0
```

The probe prepares two different layers, rejects mixed-prefix SAVE, saves atomically, resets and restores, compares complete OUT/LSE exactly, restores over stale future rows, checks append counters, and rejects missing identity, geometry mismatch, zero rows, and unknown revision. Its synthetic fixture key is only for the primitive gate; the real client always uses the complete host native-file digest.

Also qualify SHA256 against published vectors, native KVarN4 record/stage/tail round trips at global group 800 and at record boundaries, cross-process restore, missing/evicted/corrupt/shortened payloads, interrupted SAVE/RESTORE, out-of-space/quota/GC, and protocol capability fallback. Measure SAVE/RESTORE wall time and disk bytes on the actual console storage, proving zero KV upload on a hit and one full resync on a miss. Finally compare model tokens/full-vocabulary log probabilities and measure real long-context TPS with the same model/prompt/settings as the current benchmark. None of those results exist yet.
