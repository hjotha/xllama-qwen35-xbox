# Plan 003 Paired MTP Diagnostic Validator

Offline diagnostics for the Plan 003 paired MTP (Multi-Token Prediction) protocol.

This is a **diagnostic-only draft**, not a product certification tool. Non-synthetic data can return `FAIL`, `INCOMPLETE_NOT_PROVEN`, or `DIAGNOSTIC_ONLY_NOT_CERTIFIED`; it cannot return product `PASS`. `SYNTHETIC_PASS` applies only to generated fixtures and does not establish device correctness or performance.

The reader ingests CSVs, manifests, gate reports, and full token-ID sidecars across exactly three prompts, two output lengths, and three arms (18 configurations, each with warmup and measured repetitions). It checks available CSV/token file hashes, declared identity formats, sequence parity, and throughput statistics. Model/package/config/prompt digest declarations are not bound to running-device bytes; declared block order is not physical execution attestation, and incomplete gate reports remain unproven. Distribution-free 95% median interval support requires at least six paired repetitions; reported statistics do not enable certification in this draft.

---

## 1. Protocol Architecture & Exact Grid

The protocol evaluates three arms under identical prompts, seeds, contexts, and hardware configurations:

- `off`: Baseline sequential decoding (MTP disabled).
- `old`: MTP enabled with the standard single-column D3D12 kernel (`d3d12twocol=off`, variant 0).
- `auto`: MTP enabled with the allowlisted two-column D3D12 kernel (`d3d12twocol=auto`, variant 2).

### Strict 3 × 2 Grid (Exactly 6 Cells, 18 Configurations)

The validator enforces **exactly three distinct representative prompts** and **exactly two output lengths (64 and 256)**:

| Cell Key             | Prompt Identifier | Expected Generated Tokens (`n_gen_tok`) | Arms Evaluated       |
| :------------------- | :---------------- | :-------------------------------------- | :------------------- |
| `spec-chat-open_64`  | `spec-chat-open`  | 64                                      | `off`, `old`, `auto` |
| `spec-chat-open_256` | `spec-chat-open`  | 256                                     | `off`, `old`, `auto` |
| `spec-code-edit_64`  | `spec-code-edit`  | 64                                      | `off`, `old`, `auto` |
| `spec-code-edit_256` | `spec-code-edit`  | 256                                     | `off`, `old`, `auto` |
| `standard-512_64`    | `standard-512`    | 64                                      | `off`, `old`, `auto` |
| `standard-512_256`   | `standard-512`    | 256                                     | `off`, `old`, `auto` |

Grid reduction, expansion, duplicate prompts, duplicate lengths, or extra unapproved cells are strictly rejected.

---

## 2. Manifest Schema Specification

A protocol execution is described by a manifest in JSON format (`schema_version: "plan003-mtp-v2"`). The example below contains illustrative identity strings, not measured provenance. Correct digest syntax alone is not proof of the model, package, configuration, prompt, or device execution:

```json
{
  "schema_version": "plan003-mtp-v2",
  "protocol": "plan003-paired-mtp",
  "synthetic": false,
  "title": "Plan 003 Paired MTP Product Protocol Validation",
  "provenance": {
    "model": {
      "declared_name": "qwen35-4b-mtp",
      "file_sha256": "3874209241c9a397e2f62cd3f70f80fd00000000000000000000000000000000"
    },
    "package": {
      "declared_identity": "GianlucaMazza.xllama_1.6.0.64_x64__pj67f1fcj4n14",
      "digest_sha256": "069c52431faef633f8cfb99dbbb242eb0e5d4cb058e39f7be7308d5b8e97f59d"
    },
    "config": {
      "config_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
      "effective_params": {
        "backend": "d3d12",
        "gpu_layers": 28,
        "n_threads": 4,
        "quant": "q4_k_m"
      }
    },
    "prompts": {
      "spec-chat-open": {
        "prompt_hash": "a000000000000000000000000000000000000000000000000000000000000001",
        "length_tokens": 64
      },
      "spec-code-edit": {
        "prompt_hash": "a000000000000000000000000000000000000000000000000000000000000002",
        "length_tokens": 64
      },
      "standard-512": {
        "prompt_hash": "a000000000000000000000000000000000000000000000000000000000000003",
        "length_tokens": 512
      }
    },
    "backend": "d3d12",
    "gpu_layers": 28,
    "sampling": {
      "greedy": 1,
      "seed": 1,
      "temperature": 0.0
    },
    "device_host": "xbox-series-s (192.168.1.26)",
    "warmup_runs": 1,
    "recorded_runs": 3
  },
  "grid": {
    "prompts": ["spec-chat-open", "spec-code-edit", "standard-512"],
    "output_lengths": [64, 256],
    "arms": ["off", "old", "auto"]
  },
  "mandatory_gates": {
    "build_and_host_tests": {
      "status": "PASS",
      "evidence_type": "structured_report",
      "scope": {
        "package_identity": "GianlucaMazza.xllama_1.6.0.64_x64__pj67f1fcj4n14",
        "model_file_sha256": "3874209241c9a397e2f62cd3f70f80fd00000000000000000000000000000000",
        "config_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0"
      },
      "verification": {
        "assertions_passed": 8,
        "assertions_total": 8,
        "artifact_path": "gate_reports/build_and_host_tests.json",
        "artifact_sha256": "..."
      },
      "details": "Linux host test suite doctest mtp:* passed (8/8 assertions)"
    }
  },
  "cells": {
    "spec-chat-open_64": {
      "arms": {
        "off": {
          "csv": "bench/results/protocol-20261005-spec-chat-open-64-off.csv",
          "csv_sha256": "7f4aa1eaa991b290b727f9c805b2ab95b111eaed93b8adf9efc97d1c27384fce",
          "knob": "--greedy --seed 1",
          "token_sidecars": {
            "1": {"path": "...", "sha256": "..."},
            "2": {"path": "...", "sha256": "..."},
            "3": {"path": "...", "sha256": "..."}
          }
        },
        "old": { "csv": "...", "csv_sha256": "...", "token_sidecars": {...} },
        "auto": { "csv": "...", "csv_sha256": "...", "token_sidecars": {...} }
      },
      "blocks": [
        {
          "block_id": "block-1",
          "arm_order": ["off", "old", "auto"],
          "order_evidence": {
            "execution_sequence": ["off", "old", "auto"],
            "randomized_or_rotated": true,
            "timestamps": {
              "off": "2026-10-06T12:00:00Z",
              "old": "2026-10-06T12:01:00Z",
              "auto": "2026-10-06T12:02:00Z"
            }
          },
          "runs": {"off": 2, "old": 2, "auto": 2}
        },
        {
          "block_id": "block-2",
          "arm_order": ["old", "auto", "off"],
          "order_evidence": {
            "execution_sequence": ["old", "auto", "off"],
            "randomized_or_rotated": true,
            "timestamps": {
              "old": "2026-10-06T12:03:00Z",
              "auto": "2026-10-06T12:04:00Z",
              "off": "2026-10-06T12:05:00Z"
            }
          },
          "runs": {"off": 3, "old": 3, "auto": 3}
        }
      ]
    }
  }
}
```

---

## 3. Statistical Formulation & Exact Median Interval

### A. Paired Percentage Speedup Definition

For paired block $b \in \{1, \dots, N\}$, the paired gain is defined per block:
$$g_b = \left( \frac{\text{candidate\_speed}_b}{\text{baseline\_speed}_b} - 1 \right) \times 100\%$$

### B. Mean-Gain Confidence Interval (Student's $t$)

$$\bar{g} \pm t_{1 - \alpha/2, N - 1} \cdot \frac{s_g}{\sqrt{N}}$$

- Assumes approximate normality of paired gains.
- Requires sample support $N \ge 3$ recorded repetitions.
- If $N < 3$, the Student-$t$ interval cannot be formed; only the empirical range $[\min g_b, \max g_b]$ is reported and the product throughput gate is set to `INCOMPLETE_NOT_PROVEN`.

### C. Distribution-Free Median Confidence Interval (Binomial Order Statistics)

Let $X_{(1)} \le X_{(2)} \le \dots \le X_{(N)}$ be the ordered sample of paired gains.
The confidence interval $[X_{(r)}, X_{(N - r + 1)}]$ covers the population median with exact coverage:
$$\text{Coverage}(N, r) = \sum_{k=r}^{N-r} \binom{N}{k} (0.5)^N = 1 - 2 \sum_{k=0}^{r-1} \binom{N}{k} (0.5)^N$$

**Sample Support Thresholds for Median Coverage**:

- $N = 5$: Max achievable coverage is $r=1 \implies 93.75\% < 95\%$.
- $N = 6$: $r=1 \implies [X_{(1)}, X_{(6)}]$ achieves $96.88\% \ge 95\%$.
- $N = 7$: $r=1 \implies [X_{(1)}, X_{(7)}]$ achieves $98.44\% \ge 95\%$ ($r=2$ is $87.5\% < 95\%$).
- $N = 10$: $r=2 \implies [X_{(2)}, X_{(9)}]$ achieves $97.85\% \ge 95\%$ ($r=3$ is $89.06\% < 95\%$).

_For distribution-free $\ge 95\%$ median interval proof, $N \ge 6$ recorded repetitions are required._

---

## 4. Operational Gate Taxonomy

| Gate Name                          | Requirement                    | Proof / Evidence Required                                                                                          |
| :--------------------------------- | :----------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| `build_and_host_tests`             | Host test suite clean pass     | Structured report linking doctest / unit test log and hash.                                                        |
| `active_mtp_engagement`            | Active drafter execution       | Structured report with dispatch counters (e.g. 2-col GEMV invocations).                                            |
| `cache_reuse_parity`               | KV-cache reuse parity          | Structured test log confirming KV-reuse parity across batch sweeps.                                                |
| `rollback_under_mismatch`          | Rollback parity on mismatch    | Isolated rollback test evidence verifying bit-exact logits under mismatch.                                         |
| `phase_instrumentation_accounting` | Phase breakdown reconciliation | Reconciled phase timer log matching total generated tokens.                                                        |
| `eog_handling`                     | Natural EOS stop parity        | Test log with natural EOS observation. _(Non-EOS length saturation is strictly classified as `PARTIALLY_PROVEN`)_. |
| `cancel_policy`                    | Session cancellation safety    | Test log confirming mid-generation cancel cleanly cleans state.                                                    |

_Assertion-only strings or unstructured evidence are strictly rejected as `NOT_PROVEN`._

---

## 5. CLI Usage

```bash
# Validate manifest with standard text report
python3 scripts/validate-mtp-protocol.py manifest.json

# Markdown formatted output for PRs and issues
python3 scripts/validate-mtp-protocol.py manifest.json --format markdown

# Machine-readable JSON output
python3 scripts/validate-mtp-protocol.py manifest.json --format json
```

## 6. Historical negative regression fixture

`bench/results/protocol-20261005-provenance.json` combines real historical CSV/token sidecars with auto-generated placeholder identity fields and unverified legacy gate assertions. Its model digest is a historical FNV64 label padded to SHA-256 length; config/prompt identities and some gate digests are fabricated fixture placeholders, not measured hashes. These values are retained to exercise rejection and diagnostic reporting, never to establish provenance. The manifest explicitly labels this distinction, keeps `synthetic: false` for the real console data, and is expected to return `FAIL`. The companion 18 CSVs and 54 token sidecars are required for those historical negative tests.

The positive fixture under `tests/fixtures/mtp_protocol/synthetic_positive/` is generated data (`synthetic: true`). It contains 18 CSVs, 72 token sidecars, and seven synthetic gate reports. Its `SYNTHETIC_PASS` checks the reader only; its package/model identities and gate reports are not console evidence.
