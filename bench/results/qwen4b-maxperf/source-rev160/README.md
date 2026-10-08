# Source reproducibility — rev160 final source

The parent repository commit `d960126` (`feat/qwen4b-maxperf`) does **not** pin
the full final source: `git status` reports `m llama.cpp` (modified submodule
content). This directory records the exact `llama.cpp` state used by the final
build so the tree is reproducible.

- Submodule HEAD: `982eaadaa9dbe761f00205bc20b6c04b7329b58d` (detached),
  describe `b11089-12087-g982eaadaa` — `llama-cpp-head.txt`.
- Worktree status + diffstat — `llama-cpp-status.txt` (8 modified files, no
  untracked/added/deleted).
- Binary-safe worktree diff — `llama-cpp-worktree.patch`
  (`git diff --binary HEAD`), sha256 in `llama-cpp-sha256.txt`:
  `7cd11abc97cc2bb4fe196fb73a4465633f1b7d5aa5b7fa97b44184478b4eb144`
  (373 lines). Per-file sha256 of the eight modified files is in
  `llama-cpp-sha256.txt`.

Reproduce:

```sh
cd llama.cpp
git checkout 982eaadaa9dbe761f00205bc20b6c04b7329b58d
git apply --binary ../bench/results/qwen4b-maxperf/source-rev160/llama-cpp-worktree.patch
```

These are the AppContainer/local-split working-tree modifications the build
expects (the build mirror also re-runs `scripts/apply-uwp-patches.sh`).

Excluded raw evidence logs are **preserved, gzipped**, in the same evidence
tree (`10 GiB` raw -> compression to a few GiB; not committed because of size).
Disk at the time of writing: **20 GiB free** (`df -h /`).
