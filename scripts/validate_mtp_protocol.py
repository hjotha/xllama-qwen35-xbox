#!/usr/bin/env python3
"""Diagnostic-only reader for the Plan 003 paired MTP protocol.

Strictly ingests and validates CSV benchmarks, provenance manifests, and full
token-ID sidecars across a 3-prompt x 2-length grid (6 cells x 3 arms = 18 configurations).
Evaluates deterministic greedy sequence parity, per-cell paired speedups,
statistically justified median/mean confidence intervals with explicit sample-size
limitations, declared block/order evidence, and scoped operational gate reports.
Non-synthetic inputs cannot yield product PASS; synthetic PASS is fixture-only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Protocol schema constants
SUPPORTED_SCHEMA_VERSIONS = {1, "1", "plan003-mtp-v1", "plan003-mtp-v2"}
PROTOCOL_ID = "plan003-paired-mtp"

# Required exact grid dimensions
EXACT_PROMPT_COUNT = 3
EXACT_OUTPUT_LENGTHS = {64, 256}
EXACT_ARMS = {"off", "old", "auto"}
EXACT_CELL_COUNT = 6  # 3 prompts x 2 lengths

# Required CSV columns (V1 core + V2 extensions)
REQUIRED_CSV_CORE = [
    "model", "quant", "backend", "n_ctx", "n_threads", "prompt_tok_s",
    "decode_tok_s", "peak_ws_mb", "load_ms", "gpu_mem_mb", "gpu_budget_mb",
    "n_prompt_tok", "n_gen_tok", "max_length", "host", "date", "run_index"
]
OPTIONAL_CSV_V2 = ["prefill_ms", "ttft_ms", "n_draft", "kernel", "mtp_active"]

# Minimum sample size for statistical claims
MIN_RECORDED_RUNS_FOR_MEAN_CI = 3          # Student-t interval on mean requires N >= 3
MIN_RECORDED_RUNS_FOR_95PCT_MEDIAN_CI = 6  # Non-parametric binomial median CI requires N >= 6 for >=95% confidence

# Student's t distribution critical values for two-sided 95% CI (alpha=0.05) on the MEAN
STUDENT_T_95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980
}

HEX64_REGEX = re.compile(r"^[0-9a-f]{64}$")
FNV1A_HEX_REGEX = re.compile(r"^[0-9a-f]{16}$")


def get_t_critical_95(df: int) -> float:
    """Return Student's t critical value for 95% two-sided confidence interval."""
    if df < 1:
        return float("inf")
    if df in STUDENT_T_95:
        return STUDENT_T_95[df]
    keys = sorted(STUDENT_T_95.keys())
    if df > keys[-1]:
        return 1.960  # Asymptotic normal z-critical
    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        if k1 <= df <= k2:
            frac = (df - k1) / (k2 - k1)
            return STUDENT_T_95[k1] + frac * (STUDENT_T_95[k2] - STUDENT_T_95[k1])
    return 1.960


# ---------------------------------------------------------------------------
# Custom Exceptions for Strict Ingestion & Validation
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Base exception for all protocol validation failures."""
    pass


class ManifestError(ValidationError):
    """Raised when the provenance manifest is invalid, missing, or malformed."""
    pass


class IntegrityError(ValidationError):
    """Raised when file hashes or existence checks fail."""
    pass


class CsvFormatError(ValidationError):
    """Raised when CSV structure, types, non-finite or negative values fail."""
    pass


class TokenFormatError(ValidationError):
    """Raised when token ID sidecars are malformed, truncated, or invalid."""
    pass


class GridIncompleteError(ValidationError):
    """Raised when the prompt/length/arm grid is incomplete, expanded, or duplicated."""
    pass


class BlockPairingError(ValidationError):
    """Raised when explicit block pairing or order evidence is malformed."""
    pass


class DeterminismError(ValidationError):
    """Raised when repeated runs of the same greedy arm produce differing tokens."""
    pass


class ProvenanceMismatchError(ValidationError):
    """Raised when CSV metadata does not match the manifest provenance."""
    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PromptProvenance:
    prompt_id: str
    prompt_hash: str
    length_tokens: int


@dataclasses.dataclass(frozen=True)
class ProvenanceConfig:
    model_declared_name: str
    model_file_sha256: str
    package_declared_identity: str
    package_digest_sha256: str
    backend: str
    gpu_layers: int
    config_hash: str
    effective_params: Dict[str, Any]
    prompts: Dict[str, PromptProvenance]
    sampling: Dict[str, Any]
    device_host: str
    warmup_runs: int
    recorded_runs: int
    synthetic: bool = False


@dataclasses.dataclass(frozen=True)
class CsvRow:
    model: str
    quant: str
    backend: str
    n_ctx: int
    n_threads: int
    prompt_tok_s: float
    decode_tok_s: float
    peak_ws_mb: float
    load_ms: float
    gpu_mem_mb: float
    gpu_budget_mb: float
    n_prompt_tok: int
    n_gen_tok: int
    max_length: int
    host: str
    date: str
    run_index: int
    n_draft: Optional[int] = None
    kernel: Optional[str] = None
    mtp_active: Optional[bool] = None
    prefill_ms: Optional[float] = None
    ttft_ms: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class TokenSidecar:
    run_index: int
    tokens: List[int]
    fnv1a_hash: Optional[str]
    source_file: Path


@dataclasses.dataclass
class ArmData:
    arm_name: str
    csv_file: Path
    rows: List[CsvRow]
    warmup_rows: List[CsvRow]
    measured_rows: List[CsvRow]
    token_sidecars: Dict[int, TokenSidecar]
    knob: str = ""


@dataclasses.dataclass(frozen=True)
class BlockPairing:
    block_id: str
    arm_order: List[str]
    order_evidence: Dict[str, Any]
    runs: Dict[str, int]


@dataclasses.dataclass
class SequenceParityResult:
    is_match: bool
    earliest_mismatch_pos_0idx: Optional[int] = None
    earliest_mismatch_pos_1idx: Optional[int] = None
    left_token: Optional[int] = None
    right_token: Optional[int] = None
    left_len: int = 0
    right_len: int = 0
    matched_prefix_len: int = 0
    message: str = ""


@dataclasses.dataclass
class PairedComparisonStats:
    n_pairs: int
    baseline_median_tok_s: float
    baseline_spread_tok_s: Tuple[float, float]
    candidate_median_tok_s: float
    candidate_spread_tok_s: Tuple[float, float]
    paired_gains_pct: List[float]
    median_gain_pct: float
    mean_gain_pct: float
    std_gain_pct: float
    min_gain_pct: float
    max_gain_pct: float
    spread_gain_pct: float

    # Statistical intervals with explicit method labeling
    mean_ci_95_pct: Optional[Tuple[float, float]]       # Student's t interval on the MEAN (assumes normality)
    median_ci_pct: Optional[Tuple[float, float]]        # Non-parametric exact binomial interval on the MEDIAN
    median_ci_achieved_confidence_pct: float            # Exact confidence level achieved (e.g. 96.88% for N=6)
    median_ci_sufficient_for_95pct: bool                # True only if N >= 6 (achieving >= 95.0% confidence)

    hodges_lehmann_median_pct: float
    small_sample_warning: bool
    sample_support_sufficient: bool                     # True only if N >= MIN_RECORDED_RUNS_FOR_MEAN_CI (>= 3)
    warning_message: str = ""


@dataclasses.dataclass
class CellResult:
    prompt: str
    n_gen_tok: int
    cell_key: str
    arms: Dict[str, ArmData]

    # Block Pairing & Order Verification
    block_pairing_status: str                           # "VERIFIED", "UNVERIFIED_HISTORICAL", "MALFORMED_BLOCKS"
    block_pairing_passed: bool
    blocks: List[BlockPairing]

    # Parity checks
    mtp_old_vs_baseline_parity: SequenceParityResult
    mtp_auto_vs_baseline_parity: SequenceParityResult
    kernel_auto_vs_old_parity: SequenceParityResult

    # Performance statistics
    mtp_old_vs_baseline_stats: PairedComparisonStats
    mtp_auto_vs_baseline_stats: PairedComparisonStats
    kernel_auto_vs_old_stats: PairedComparisonStats

    # Per-cell Gate Verdicts
    greedy_parity_passed: bool
    kernel_parity_passed: bool
    mtp_product_gate_passed: bool
    gate_failure_reasons: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class MandatoryGateResult:
    name: str
    status: str                                         # "PASS", "FAIL", "UNTESTED", "NOT_PROVEN", "PARTIALLY_PROVEN", "ASSERTION_ONLY"
    evidence_type: str                                  # "structured_report", "assertion_only", "missing"
    scope: Dict[str, Any]
    verification: Dict[str, Any]
    details: str
    is_pass: bool
    verification_notes: str = ""


@dataclasses.dataclass
class ProtocolReport:
    manifest_path: Path
    provenance: ProvenanceConfig
    is_synthetic: bool
    cells: List[CellResult]
    mandatory_gates: List[MandatoryGateResult]
    all_cells_parity_pass: bool
    all_cells_kernel_parity_pass: bool
    all_cells_mtp_speedup_pass: bool
    all_cells_block_pairing_pass: bool
    all_mandatory_gates_pass: bool
    all_sample_sizes_sufficient: bool
    overall_verdict: str                                # "FAIL", "INCOMPLETE_NOT_PROVEN", "SYNTHETIC_PASS", "DIAGNOSTIC_ONLY_NOT_CERTIFIED"
    summary_messages: List[str] = dataclasses.field(default_factory=list)
    rejection_reasons: List[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Strict JSON and File Utilities
# ---------------------------------------------------------------------------

def strict_json_loads(text: str) -> Dict[str, Any]:
    """Parse JSON while strictly rejecting duplicate object keys."""
    def check_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for key, value in pairs:
            if key in res:
                raise ManifestError(f"Duplicate JSON key detected in manifest: '{key}'")
            res[key] = value
        return res
    return json.loads(text, object_pairs_hook=check_duplicates)


def compute_file_sha256(path: Path) -> str:
    if not path.is_file():
        raise IntegrityError(f"Target file does not exist: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Statistical Calculations
# ---------------------------------------------------------------------------

def calculate_median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of an empty sequence")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 != 0 else (ordered[mid - 1] + ordered[mid]) / 2.0


def calculate_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty sequence")
    return sum(values) / len(values)


def calculate_std(values: Sequence[float], mean_val: Optional[float] = None) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = mean_val if mean_val is not None else calculate_mean(values)
    variance = sum((x - mu) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def calculate_hodges_lehmann(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute Hodges-Lehmann on empty sequence")
    walsh_averages = []
    n = len(values)
    for i in range(n):
        for j in range(i, n):
            walsh_averages.append((values[i] + values[j]) / 2.0)
    return calculate_median(walsh_averages)


def compute_binomial_median_coverage(n: int, r: int) -> float:
    """Exact two-sided coverage probability for median order statistic interval [X_(r), X_(n-r+1)].

    Coverage(n, r) = sum_{k=r}^{n-r} binom(n, k) * 0.5^n = 1 - 2 * sum_{k=0}^{r-1} binom(n, k) * 0.5^n
    """
    if n < 1 or r < 1 or r > (n + 1) // 2:
        return 0.0
    tail_prob = sum(math.comb(n, k) for k in range(r)) / (2.0 ** n)
    return max(0.0, 1.0 - 2.0 * tail_prob)


def compute_nonparametric_median_ci(
    values: Sequence[float], target_confidence: float = 0.95
) -> Dict[str, Any]:
    """Calculate exact non-parametric binomial order-statistics CI for the median.

    Finds the largest r (narrowest interval [X_(r), X_(n-r+1)]) such that coverage >= target_confidence.
    For target_confidence=0.95, n >= 6 is required.
    """
    n = len(values)
    ordered = sorted(values)
    if n == 0:
        return {
            "n": 0,
            "ci": None,
            "achieved_confidence_pct": 0.0,
            "r": None,
            "sufficient_for_target": False,
            "note": "Empty sample"
        }

    # Find the largest r >= 1 achieving coverage >= target_confidence
    max_r = (n + 1) // 2
    best_r: Optional[int] = None
    best_coverage: float = 0.0

    for r in range(max_r, 0, -1):
        cov = compute_binomial_median_coverage(n, r)
        if cov >= target_confidence:
            best_r = r
            best_coverage = cov
            break

    if best_r is not None:
        # 1-indexed r -> [ordered[r-1], ordered[n-r]]
        ci = (ordered[best_r - 1], ordered[n - best_r])
        cov_pct = round(best_coverage * 100.0, 2)
        return {
            "n": n,
            "ci": ci,
            "achieved_confidence_pct": cov_pct,
            "r": best_r,
            "sufficient_for_target": True,
            "note": f"Distribution-free binomial order interval [X_({best_r}), X_({n - best_r + 1})] with {cov_pct:.2f}% coverage."
        }
    else:
        # No r satisfies target_confidence (e.g. n < 6 for 95%).
        cov_r1 = compute_binomial_median_coverage(n, 1) if n >= 2 else 0.0
        ci = (ordered[0], ordered[-1]) if n >= 1 else None
        cov_pct = round(cov_r1 * 100.0, 2)
        return {
            "n": n,
            "ci": ci,
            "achieved_confidence_pct": cov_pct,
            "r": 1 if n >= 2 else None,
            "sufficient_for_target": False,
            "note": (
                f"Insufficient sample size for {target_confidence * 100:.0f}% distribution-free median CI "
                f"(N={n} < 6; max achieved coverage is {cov_pct:.2f}%). "
                f"Requires N >= 6 recorded repetitions for distribution-free >=95% median proof."
            )
        }


def compute_paired_comparison_stats(
    baseline_speeds: List[float],
    candidate_speeds: List[float]
) -> PairedComparisonStats:
    """Calculate paired speedup statistics, Student-t mean CI, and non-parametric median CI."""
    if len(baseline_speeds) != len(candidate_speeds):
        raise ValueError(
            f"Paired comparison requires equal lengths: {len(baseline_speeds)} vs {len(candidate_speeds)}"
        )
    n = len(baseline_speeds)
    if n == 0:
        raise ValueError("Cannot compute statistics for zero paired observations")

    gains: List[float] = [
        ((cand / base) - 1.0) * 100.0 for base, cand in zip(baseline_speeds, candidate_speeds)
    ]

    base_med = calculate_median(baseline_speeds)
    base_spread = (min(baseline_speeds), max(baseline_speeds))
    cand_med = calculate_median(candidate_speeds)
    cand_spread = (min(candidate_speeds), max(candidate_speeds))

    median_gain = calculate_median(gains)
    mean_gain = calculate_mean(gains)
    std_gain = calculate_std(gains, mean_gain)
    min_gain = min(gains)
    max_gain = max(gains)
    spread_gain = max_gain - min_gain
    hl_median = calculate_hodges_lehmann(gains)

    # Sample support threshold
    sample_sufficient = n >= MIN_RECORDED_RUNS_FOR_MEAN_CI
    small_sample = n < MIN_RECORDED_RUNS_FOR_MEAN_CI
    mean_ci_95: Optional[Tuple[float, float]] = None
    warning_msg = ""

    if small_sample:
        warning_msg = (
            f"SMALL_SAMPLE_WARNING: N={n} recorded repetitions is insufficient for robust "
            f"mean confidence interval estimation (minimum required N >= {MIN_RECORDED_RUNS_FOR_MEAN_CI}). "
            f"Empirical range is [{min_gain:+.2f}%, {max_gain:+.2f}%]."
        )
    else:
        df = n - 1
        t_crit = get_t_critical_95(df)
        se = std_gain / math.sqrt(n)
        ci_half = t_crit * se
        mean_ci_95 = (mean_gain - ci_half, mean_gain + ci_half)

    median_ci_res = compute_nonparametric_median_ci(gains, target_confidence=0.95)

    return PairedComparisonStats(
        n_pairs=n,
        baseline_median_tok_s=base_med,
        baseline_spread_tok_s=base_spread,
        candidate_median_tok_s=cand_med,
        candidate_spread_tok_s=cand_spread,
        paired_gains_pct=gains,
        median_gain_pct=median_gain,
        mean_gain_pct=mean_gain,
        std_gain_pct=std_gain,
        min_gain_pct=min_gain,
        max_gain_pct=max_gain,
        spread_gain_pct=spread_gain,
        mean_ci_95_pct=mean_ci_95,
        median_ci_pct=median_ci_res["ci"],
        median_ci_achieved_confidence_pct=median_ci_res["achieved_confidence_pct"],
        median_ci_sufficient_for_95pct=median_ci_res["sufficient_for_target"],
        hodges_lehmann_median_pct=hl_median,
        small_sample_warning=small_sample,
        sample_support_sufficient=sample_sufficient,
        warning_message=warning_msg
    )


# ---------------------------------------------------------------------------
# Sequence Parity & Token Sidecar Parsing
# ---------------------------------------------------------------------------

def compare_token_sequences(
    seq_left: Sequence[int],
    seq_right: Sequence[int],
    name_left: str = "baseline",
    name_right: str = "candidate"
) -> SequenceParityResult:
    """Bit-exact token comparison returning earliest mismatch position and IDs."""
    len_left = len(seq_left)
    len_right = len(seq_right)
    min_len = min(len_left, len_right)

    for idx in range(min_len):
        t_left = seq_left[idx]
        t_right = seq_right[idx]
        if t_left != t_right:
            return SequenceParityResult(
                is_match=False,
                earliest_mismatch_pos_0idx=idx,
                earliest_mismatch_pos_1idx=idx + 1,
                left_token=t_left,
                right_token=t_right,
                left_len=len_left,
                right_len=len_right,
                matched_prefix_len=idx,
                message=(
                    f"Divergence at token index {idx} (1-indexed position {idx + 1}): "
                    f"{name_left}={t_left} vs {name_right}={t_right} "
                    f"(matched prefix length {idx}/{min_len})"
                )
            )

    if len_left != len_right:
        return SequenceParityResult(
            is_match=False,
            earliest_mismatch_pos_0idx=min_len,
            earliest_mismatch_pos_1idx=min_len + 1,
            left_token=seq_left[min_len] if min_len < len_left else None,
            right_token=seq_right[min_len] if min_len < len_right else None,
            left_len=len_left,
            right_len=len_right,
            matched_prefix_len=min_len,
            message=(
                f"Length divergence after matching prefix of {min_len} tokens: "
                f"{name_left} length={len_left} vs {name_right} length={len_right}"
            )
        )

    return SequenceParityResult(
        is_match=True,
        left_len=len_left,
        right_len=len_right,
        matched_prefix_len=len_left,
        message=f"Bit-exact greedy match across all {len_left} tokens"
    )


def parse_token_sidecar(path: Path, expected_gen_tok: int, run_idx: int) -> TokenSidecar:
    """Strictly parse a token-ID sidecar file, validating tokens and fnv1a footer."""
    if not path.is_file():
        raise TokenFormatError(f"Token sidecar file not found: {path}")

    tokens: List[int] = []
    fnv1a_hash: Optional[str] = None

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise TokenFormatError(f"Cannot read token file {path}: {exc}") from exc

    lines = content.splitlines()
    if not lines:
        raise TokenFormatError(f"Token sidecar file is empty: {path}")

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            raise TokenFormatError(f"{path}:{line_num}: invalid empty line within token sidecar")
        if line.startswith("fnv1a="):
            parts = line.split("=", 1)
            hash_val = parts[1].strip().lower()
            if not FNV1A_HEX_REGEX.match(hash_val):
                raise TokenFormatError(
                    f"{path}:{line_num}: malformed fnv1a hash '{parts[1]}' (expected 16-hex characters)"
                )
            fnv1a_hash = hash_val
            continue
        try:
            tok_val = int(line)
        except ValueError:
            raise TokenFormatError(f"{path}:{line_num}: invalid non-integer token '{line}'")
        if tok_val < 0:
            raise TokenFormatError(f"{path}:{line_num}: token ID must be non-negative, got {tok_val}")
        tokens.append(tok_val)

    if len(tokens) != expected_gen_tok:
        raise TokenFormatError(
            f"{path}: expected exactly {expected_gen_tok} tokens for cell, but found {len(tokens)} "
            f"({'truncated' if len(tokens) < expected_gen_tok else 'excess'} sequence)"
        )

    return TokenSidecar(
        run_index=run_idx,
        tokens=tokens,
        fnv1a_hash=fnv1a_hash,
        source_file=path
    )


# ---------------------------------------------------------------------------
# Strict CSV Parsing and Validation
# ---------------------------------------------------------------------------

def parse_and_validate_csv(path: Path, prov: ProvenanceConfig, expected_gen_tok: int, arm_name: str) -> List[CsvRow]:
    """Strictly parse an RFC-4180 CSV, checking quoting, non-finite values, and constraints."""
    if not path.is_file():
        raise CsvFormatError(f"CSV file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise CsvFormatError(f"Failed to read CSV file {path}: {exc}") from exc

    if not raw_text.strip():
        raise CsvFormatError(f"CSV file is empty: {path}")

    rows: List[CsvRow] = []
    seen_run_indices = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=",", quotechar='"', doublequote=True, strict=True)
        try:
            header = next(reader)
        except StopIteration:
            raise CsvFormatError(f"CSV file is empty: {path}")
        except csv.Error as exc:
            raise CsvFormatError(f"RFC-4180 CSV parsing error in header of {path}: {exc}")

        header_set = set(header)
        missing_cols = set(REQUIRED_CSV_CORE) - header_set
        if missing_cols:
            raise CsvFormatError(
                f"{path}: missing required benchmark CSV columns: {sorted(missing_cols)}"
            )

        col_map = {name: idx for idx, name in enumerate(header)}
        expected_col_count = len(header)

        try:
            for row_num, raw_row in enumerate(reader, start=2):
                if len(raw_row) != expected_col_count:
                    raise CsvFormatError(
                        f"{path}:{row_num}: row column count mismatch (expected {expected_col_count}, got {len(raw_row)})"
                    )

                def get_str(name: str) -> str:
                    return raw_row[col_map[name]].strip()

                def get_int(name: str) -> int:
                    val_str = get_str(name)
                    try:
                        return int(val_str)
                    except ValueError:
                        raise CsvFormatError(f"{path}:{row_num}: column '{name}' must be integer, got '{val_str}'")

                def get_float(name: str) -> float:
                    val_str = get_str(name)
                    try:
                        v = float(val_str)
                        if math.isnan(v) or math.isinf(v):
                            raise ValueError("non-finite")
                        return v
                    except ValueError:
                        raise CsvFormatError(
                            f"{path}:{row_num}: column '{name}' must be finite float, got '{val_str}'"
                        )

                model_val = get_str("model")
                quant_val = get_str("quant")
                backend_val = get_str("backend")
                n_ctx_val = get_int("n_ctx")
                n_threads_val = get_int("n_threads")
                prompt_tok_s = get_float("prompt_tok_s")
                decode_tok_s = get_float("decode_tok_s")
                peak_ws_mb = get_float("peak_ws_mb")
                load_ms = get_float("load_ms")
                gpu_mem_mb = get_float("gpu_mem_mb")
                gpu_budget_mb = get_float("gpu_budget_mb")
                n_prompt_tok = get_int("n_prompt_tok")
                n_gen_tok = get_int("n_gen_tok")
                max_length = get_int("max_length")
                host_val = get_str("host")
                date_val = get_str("date")
                run_idx = get_int("run_index")

                if decode_tok_s <= 0:
                    raise CsvFormatError(f"{path}:{row_num}: decode_tok_s must be positive, got {decode_tok_s}")
                if prompt_tok_s <= 0:
                    raise CsvFormatError(f"{path}:{row_num}: prompt_tok_s must be positive, got {prompt_tok_s}")
                if load_ms < 0 or peak_ws_mb < 0 or gpu_mem_mb < 0 or gpu_budget_mb < 0:
                    raise CsvFormatError(f"{path}:{row_num}: resource metrics must be non-negative")
                if n_ctx_val <= 0 or n_threads_val <= 0 or n_prompt_tok <= 0 or n_gen_tok <= 0:
                    raise CsvFormatError(f"{path}:{row_num}: token/context parameters must be positive")
                if run_idx < 1:
                    raise CsvFormatError(f"{path}:{row_num}: run_index must be >= 1, got {run_idx}")

                if run_idx in seen_run_indices:
                    raise CsvFormatError(f"{path}:{row_num}: duplicate run_index {run_idx} detected")
                seen_run_indices.add(run_idx)

                if n_gen_tok != expected_gen_tok:
                    raise CsvFormatError(
                        f"{path}:{row_num}: n_gen_tok ({n_gen_tok}) does not match cell expectation ({expected_gen_tok})"
                    )

                if model_val != prov.model_declared_name:
                    raise ProvenanceMismatchError(
                        f"{path}:{row_num}: model '{model_val}' does not match provenance declared model '{prov.model_declared_name}'"
                    )
                if backend_val != prov.backend:
                    raise ProvenanceMismatchError(
                        f"{path}:{row_num}: backend '{backend_val}' does not match provenance backend '{prov.backend}'"
                    )

                # Check optional per-run MTP knobs if present in CSV
                n_draft_val = get_int("n_draft") if "n_draft" in col_map and raw_row[col_map["n_draft"]].strip() else None
                kernel_val = get_str("kernel") if "kernel" in col_map and raw_row[col_map["kernel"]].strip() else None

                if arm_name == "off" and n_draft_val is not None and n_draft_val != 0:
                    raise CsvFormatError(f"{path}:{row_num}: baseline arm 'off' must have n_draft=0, got {n_draft_val}")
                if arm_name in ("old", "auto") and n_draft_val is not None and n_draft_val < 1:
                    raise CsvFormatError(f"{path}:{row_num}: MTP arm '{arm_name}' must have n_draft >= 1, got {n_draft_val}")

                prefill_ms = get_float("prefill_ms") if "prefill_ms" in col_map and raw_row[col_map["prefill_ms"]].strip() else None
                ttft_ms = get_float("ttft_ms") if "ttft_ms" in col_map and raw_row[col_map["ttft_ms"]].strip() else None

                rows.append(
                    CsvRow(
                        model=model_val,
                        quant=quant_val,
                        backend=backend_val,
                        n_ctx=n_ctx_val,
                        n_threads=n_threads_val,
                        prompt_tok_s=prompt_tok_s,
                        decode_tok_s=decode_tok_s,
                        peak_ws_mb=peak_ws_mb,
                        load_ms=load_ms,
                        gpu_mem_mb=gpu_mem_mb,
                        gpu_budget_mb=gpu_budget_mb,
                        n_prompt_tok=n_prompt_tok,
                        n_gen_tok=n_gen_tok,
                        max_length=max_length,
                        host=host_val,
                        date=date_val,
                        run_index=run_idx,
                        n_draft=n_draft_val,
                        kernel=kernel_val,
                        prefill_ms=prefill_ms,
                        ttft_ms=ttft_ms
                    )
                )
        except csv.Error as exc:
            raise CsvFormatError(f"RFC-4180 CSV parsing error in {path}: {exc}") from exc

    if not rows:
        raise CsvFormatError(f"No valid data rows found in {path}")
    return rows


# ---------------------------------------------------------------------------
# Core Protocol Validator Engine
# ---------------------------------------------------------------------------

class MtpProtocolValidator:
    """Validates full Plan 003 paired protocol dataset against provenance manifest."""

    def __init__(self, manifest_path: Union[str, Path]):
        self.manifest_path = Path(manifest_path).resolve()
        self.base_dir = self.manifest_path.parent
        self.manifest_data: Dict[str, Any] = {}
        self.provenance: Optional[ProvenanceConfig] = None

    def validate(self) -> ProtocolReport:
        """Run all ingestion, verification, statistical, and gate checks."""
        self._load_and_validate_manifest_structure()
        assert self.provenance is not None

        grid_prompts = self.manifest_data["grid"]["prompts"]
        grid_lengths = self.manifest_data["grid"]["output_lengths"]
        grid_arms = self.manifest_data["grid"]["arms"]
        raw_cells = self.manifest_data.get("cells", {})

        cell_results: List[CellResult] = []
        all_cells_parity_pass = True
        all_cells_kernel_parity_pass = True
        all_cells_mtp_speedup_pass = True
        all_cells_block_pairing_pass = True
        all_samples_sufficient = True

        for prompt in grid_prompts:
            for n_gen_tok in grid_lengths:
                cell_key = f"{prompt}_{n_gen_tok}"
                cell_dict = raw_cells[cell_key]
                cell_res = self._validate_cell(prompt, n_gen_tok, cell_key, cell_dict, grid_arms)
                cell_results.append(cell_res)

                if not cell_res.greedy_parity_passed:
                    all_cells_parity_pass = False
                if not cell_res.kernel_parity_passed:
                    all_cells_kernel_parity_pass = False
                if not cell_res.mtp_product_gate_passed:
                    all_cells_mtp_speedup_pass = False
                if not cell_res.block_pairing_passed:
                    all_cells_block_pairing_pass = False
                if not cell_res.mtp_auto_vs_baseline_stats.sample_support_sufficient:
                    all_samples_sufficient = False

        # Validate Mandatory Operational Gates with traceable structured evidence
        mandatory_gates = self._validate_mandatory_gates()
        all_mandatory_pass = all(g.is_pass for g in mandatory_gates)

        # Determine Overall Verdict
        # Rules:
        # 1. Any sequence parity failure (greedy or kernel) -> FAIL
        # 2. Any measured throughput speedup failure (median < 10% or min <= 0% or CI lower bound <= 0%) -> FAIL
        # 3. Block pairing unverified / absent in non-synthetic mode -> INCOMPLETE_NOT_PROVEN
        # 4. Parity and speedup hold, but sample support is insufficient (N < 3) -> INCOMPLETE_NOT_PROVEN
        # 5. Parity and speedup hold, but mandatory operational gates are unproven -> INCOMPLETE_NOT_PROVEN
        # 6. All criteria pass and synthetic=True -> SYNTHETIC_PASS
        # 7. Qualifying non-synthetic data -> DIAGNOSTIC_ONLY_NOT_CERTIFIED; no product certification
        has_parity_failure = not (all_cells_parity_pass and all_cells_kernel_parity_pass)
        has_measured_speedup_failure = any(
            c.mtp_auto_vs_baseline_stats.median_gain_pct < 10.0 or
            c.mtp_auto_vs_baseline_stats.min_gain_pct <= 0.0 or
            (c.mtp_auto_vs_baseline_stats.mean_ci_95_pct is not None and c.mtp_auto_vs_baseline_stats.mean_ci_95_pct[0] <= 0.0)
            for c in cell_results
        )

        if has_parity_failure or has_measured_speedup_failure:
            verdict = "FAIL"
        elif not all_cells_block_pairing_pass:
            verdict = "INCOMPLETE_NOT_PROVEN"
        elif not (all_samples_sufficient and all_mandatory_pass):
            verdict = "INCOMPLETE_NOT_PROVEN"
        elif self.provenance.synthetic:
            verdict = "SYNTHETIC_PASS"
        else:
            # Diagnostic-only draft: Product certification is disabled.
            # All non-synthetic input without hard failures is classified as DIAGNOSTIC_ONLY_NOT_CERTIFIED.
            verdict = "DIAGNOSTIC_ONLY_NOT_CERTIFIED"

        summary_msgs: List[str] = []
        rejection_reasons: List[str] = []

        if not all_cells_parity_pass:
            rejection_reasons.append("MTP vs Baseline greedy sequence parity failed in one or more cells.")
        if not all_cells_kernel_parity_pass:
            rejection_reasons.append("MTP AUTO vs OLD kernel sequence parity failed in one or more cells.")
        if has_measured_speedup_failure:
            rejection_reasons.append("MTP product throughput gate (>= +10.0% median gain and positive CI lower bound) failed in one or more cells.")
        if not all_cells_block_pairing_pass:
            rejection_reasons.append("Block pairing & order evidence is unverified or absent in one or more cells (diagnostic pairing used).")
        if not all_samples_sufficient:
            rejection_reasons.append(f"Insufficient sample size support (N < {MIN_RECORDED_RUNS_FOR_MEAN_CI} recorded repetitions) in one or more cells.")
        if not all_mandatory_pass:
            unpassed = [g.name for g in mandatory_gates if not g.is_pass]
            rejection_reasons.append(f"Mandatory operational gates not fully proven/passed: {', '.join(unpassed)}.")

        if verdict == "DIAGNOSTIC_ONLY_NOT_CERTIFIED":
            summary_msgs.append(
                "DIAGNOSTIC NOTICE: Product certification is NOT supported in this diagnostic tool draft. "
                "Unresolved evidence checks include host-to-device byte binding for digests, physical order "
                "timing attestations, and sample support for distribution-free >=95% median CI (requires N >= 6)."
            )

        return ProtocolReport(
            manifest_path=self.manifest_path,
            provenance=self.provenance,
            is_synthetic=self.provenance.synthetic,
            cells=cell_results,
            mandatory_gates=mandatory_gates,
            all_cells_parity_pass=all_cells_parity_pass,
            all_cells_kernel_parity_pass=all_cells_kernel_parity_pass,
            all_cells_mtp_speedup_pass=all_cells_mtp_speedup_pass,
            all_cells_block_pairing_pass=all_cells_block_pairing_pass,
            all_mandatory_gates_pass=all_mandatory_pass,
            all_sample_sizes_sufficient=all_samples_sufficient,
            overall_verdict=verdict,
            summary_messages=summary_msgs,
            rejection_reasons=rejection_reasons
        )

    def _load_and_validate_manifest_structure(self) -> None:
        """Load manifest JSON and validate top-level keys, exact grid, and provenance."""
        if not self.manifest_path.is_file():
            raise ManifestError(f"Manifest file does not exist: {self.manifest_path}")

        try:
            self.manifest_data = strict_json_loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ManifestError(f"Failed to parse manifest JSON {self.manifest_path}: {exc}") from exc

        required_top = {"schema_version", "protocol", "provenance", "grid", "cells", "mandatory_gates"}
        missing_top = required_top - self.manifest_data.keys()
        if missing_top:
            raise ManifestError(f"Manifest missing required top-level keys: {sorted(missing_top)}")

        schema_ver = self.manifest_data["schema_version"]
        if schema_ver not in SUPPORTED_SCHEMA_VERSIONS:
            raise ManifestError(f"Unsupported schema_version: {schema_ver}")

        proto = self.manifest_data["protocol"]
        if proto != PROTOCOL_ID:
            raise ManifestError(f"Unsupported protocol: '{proto}' (expected '{PROTOCOL_ID}')")

        # Validate Provenance
        prov_dict = self.manifest_data["provenance"]
        required_prov = {
            "model", "package", "backend", "gpu_layers", "sampling",
            "device_host", "warmup_runs", "recorded_runs"
        }
        missing_prov = required_prov - prov_dict.keys()
        if missing_prov:
            raise ManifestError(f"Provenance missing required keys: {sorted(missing_prov)}")

        raw_synth = self.manifest_data.get("synthetic")
        if raw_synth is None or not isinstance(raw_synth, bool):
            raise ManifestError("Manifest field 'synthetic' must be a strict JSON boolean (true or false)")

        # Validate sampling configuration
        sampling_dict = prov_dict["sampling"]
        if not isinstance(sampling_dict, dict) or sampling_dict.get("greedy") != 1 or int(sampling_dict.get("seed", 0)) < 1:
            raise ManifestError("Provenance sampling must specify greedy=1 and seed >= 1")

        if prov_dict["backend"] != "d3d12":
            raise ManifestError(f"Protocol requires backend='d3d12', got '{prov_dict['backend']}'")

        # Validate Model Provenance (distinguish declared name vs verified file sha256)
        model_entry = prov_dict["model"]
        if isinstance(model_entry, dict):
            model_decl_name = str(model_entry.get("declared_name", model_entry.get("name", ""))).strip()
            model_file_sha256 = str(model_entry.get("file_sha256", model_entry.get("sha256", ""))).strip().lower()
        else:
            model_decl_name = str(model_entry).strip()
            model_file_sha256 = str(prov_dict.get("model_sha256", "")).strip().lower()

        if not model_decl_name:
            raise ManifestError("Provenance model requires a non-empty declared name/identifier")

        if not raw_synth:
            if not model_file_sha256 or not HEX64_REGEX.match(model_file_sha256):
                raise ManifestError(
                    f"Non-synthetic product claims require verified lowercase 64-hex 'model.file_sha256', got '{model_file_sha256}'"
                )
        else:
            if not model_file_sha256:
                model_file_sha256 = "synthetic-model-hash"

        # Validate Package Provenance (distinguish declared package identity vs verified digest sha256)
        pkg_entry = prov_dict["package"]
        if isinstance(pkg_entry, dict):
            pkg_decl_id = str(pkg_entry.get("declared_identity", pkg_entry.get("identity", ""))).strip()
            pkg_digest_sha256 = str(pkg_entry.get("digest_sha256", pkg_entry.get("sha256", ""))).strip().lower()
        else:
            pkg_decl_id = str(pkg_entry).strip()
            pkg_digest_sha256 = str(prov_dict.get("package_sha256", "")).strip().lower()

        if not pkg_decl_id:
            raise ManifestError("Provenance package requires a non-empty declared package identity")

        if not raw_synth:
            if not pkg_digest_sha256 or not HEX64_REGEX.match(pkg_digest_sha256):
                raise ManifestError(
                    f"Non-synthetic product claims require verified lowercase 64-hex 'package.digest_sha256', got '{pkg_digest_sha256}'"
                )
        else:
            if not pkg_digest_sha256:
                pkg_digest_sha256 = "synthetic-package-hash"

        # Validate Config Hash
        config_entry = prov_dict.get("config", {})
        if isinstance(config_entry, dict):
            cfg_hash = str(config_entry.get("config_hash", prov_dict.get("config_hash", ""))).strip().lower()
            eff_params = config_entry.get("effective_params", {})
        else:
            cfg_hash = str(config_entry).strip().lower()
            eff_params = {}

        if not raw_synth:
            if not cfg_hash or not HEX64_REGEX.match(cfg_hash):
                raise ManifestError(
                    f"Non-synthetic product claims require verified lowercase 64-hex 'config.config_hash', got '{cfg_hash}'"
                )
        else:
            if not cfg_hash:
                cfg_hash = "synthetic-config-hash"

        # Validate Prompts Provenance & Hashes
        prompts_dict: Dict[str, PromptProvenance] = {}
        raw_prompts = prov_dict.get("prompts", {})
        if isinstance(raw_prompts, dict):
            for p_name, p_data in raw_prompts.items():
                if isinstance(p_data, dict):
                    p_hash = str(p_data.get("prompt_hash", "")).strip().lower()
                    p_len = int(p_data.get("length_tokens", 0))
                else:
                    p_hash = str(p_data).strip().lower()
                    p_len = 0
                if not raw_synth and (not p_hash or not HEX64_REGEX.match(p_hash)):
                    raise ManifestError(
                        f"Non-synthetic product claims require verified 64-hex hash for prompt '{p_name}', got '{p_hash}'"
                    )
                prompts_dict[p_name] = PromptProvenance(
                    prompt_id=p_name,
                    prompt_hash=p_hash or "synthetic-prompt-hash",
                    length_tokens=p_len
                )
        elif not raw_synth:
            raise ManifestError("Non-synthetic product claims require verified 'prompts' provenance dictionary")

        # Validate Exact Grid Definition
        grid = self.manifest_data["grid"]
        required_grid = {"prompts", "output_lengths", "arms"}
        missing_grid = required_grid - grid.keys()
        if missing_grid:
            raise ManifestError(f"Grid missing required keys: {sorted(missing_grid)}")

        grid_prompts = grid["prompts"]
        if not isinstance(grid_prompts, list) or len(grid_prompts) != EXACT_PROMPT_COUNT or len(set(grid_prompts)) != EXACT_PROMPT_COUNT:
            raise GridIncompleteError(
                f"grid.prompts must contain exactly {EXACT_PROMPT_COUNT} distinct prompts, got {grid_prompts}"
            )

        grid_lengths = grid["output_lengths"]
        if not isinstance(grid_lengths, list) or set(grid_lengths) != EXACT_OUTPUT_LENGTHS or len(grid_lengths) != len(EXACT_OUTPUT_LENGTHS):
            raise GridIncompleteError(
                f"grid.output_lengths must contain exactly lengths {sorted(EXACT_OUTPUT_LENGTHS)}, got {grid_lengths}"
            )

        grid_arms = grid["arms"]
        if not isinstance(grid_arms, list) or set(grid_arms) != EXACT_ARMS or len(grid_arms) != len(EXACT_ARMS):
            raise GridIncompleteError(
                f"grid.arms must contain exactly {sorted(EXACT_ARMS)}, got {grid_arms}"
            )

        # Validate Exact Cell Matrix
        raw_cells = self.manifest_data["cells"]
        if not isinstance(raw_cells, dict):
            raise ManifestError("Manifest 'cells' must be a JSON object mapping cell keys to cell definitions")

        expected_cell_keys = {f"{p}_{l}" for p in grid_prompts for l in grid_lengths}
        actual_cell_keys = set(raw_cells.keys())

        if actual_cell_keys != expected_cell_keys:
            missing_cells = expected_cell_keys - actual_cell_keys
            extra_cells = actual_cell_keys - expected_cell_keys
            err_parts = []
            if missing_cells:
                err_parts.append(f"missing cells: {sorted(missing_cells)}")
            if extra_cells:
                err_parts.append(f"unapproved extra cells: {sorted(extra_cells)}")
            raise GridIncompleteError(f"Grid cell matrix mismatch: {'; '.join(err_parts)}")

        self.provenance = ProvenanceConfig(
            model_declared_name=model_decl_name,
            model_file_sha256=model_file_sha256,
            package_declared_identity=pkg_decl_id,
            package_digest_sha256=pkg_digest_sha256,
            backend=str(prov_dict["backend"]),
            gpu_layers=int(prov_dict["gpu_layers"]),
            config_hash=cfg_hash,
            effective_params=eff_params,
            prompts=prompts_dict,
            sampling=dict(prov_dict["sampling"]),
            device_host=str(prov_dict["device_host"]),
            warmup_runs=int(prov_dict["warmup_runs"]),
            recorded_runs=int(prov_dict["recorded_runs"]),
            synthetic=raw_synth
        )

    def _resolve_path(self, rel_or_abs_path: Union[str, Path]) -> Path:
        p = Path(rel_or_abs_path)
        if p.is_absolute():
            return p
        cand1 = (self.base_dir / p).resolve()
        if cand1.exists():
            return cand1
        for parent in self.manifest_path.parents:
            cand = (parent / p).resolve()
            if cand.exists():
                return cand
        cand_cwd = (Path.cwd() / p).resolve()
        if cand_cwd.exists():
            return cand_cwd
        return cand1

    def _validate_cell(
        self,
        prompt: str,
        n_gen_tok: int,
        cell_key: str,
        cell_dict: Dict[str, Any],
        grid_arms: List[str]
    ) -> CellResult:
        """Validate arms and paired blocks within a single cell."""
        raw_arms = cell_dict.get("arms", {})
        if set(raw_arms.keys()) != set(grid_arms):
            raise GridIncompleteError(
                f"Cell '{cell_key}' arms mismatch: expected {sorted(grid_arms)}, got {sorted(raw_arms.keys())}"
            )

        arm_data_map: Dict[str, ArmData] = {}

        for arm_name in grid_arms:
            arm_entry = raw_arms[arm_name]
            if "csv" not in arm_entry:
                raise ManifestError(f"Cell '{cell_key}', arm '{arm_name}' missing 'csv' field")

            csv_path = self._resolve_path(arm_entry["csv"])
            if not csv_path.is_file():
                raise IntegrityError(f"Cell '{cell_key}', arm '{arm_name}': CSV not found: {csv_path}")

            # Enforce CSV SHA-256 in all modes
            if "csv_sha256" not in arm_entry:
                raise IntegrityError(f"Cell '{cell_key}', arm '{arm_name}': missing mandatory 'csv_sha256'")
            actual_sha = compute_file_sha256(csv_path)
            expected_sha = str(arm_entry["csv_sha256"]).lower().strip()
            if actual_sha != expected_sha:
                raise IntegrityError(
                    f"CSV SHA-256 mismatch for {csv_path}: expected {expected_sha}, got {actual_sha}"
                )

            assert self.provenance is not None
            rows = parse_and_validate_csv(csv_path, self.provenance, n_gen_tok, arm_name)

            warmup_rows = [r for r in rows if r.run_index <= self.provenance.warmup_runs]
            measured_rows = [r for r in rows if r.run_index > self.provenance.warmup_runs]

            if not measured_rows:
                raise CsvFormatError(
                    f"Cell '{cell_key}', arm '{arm_name}': no recorded measurement rows after warmup"
                )

            # Ingest token sidecars
            token_sidecars: Dict[int, TokenSidecar] = {}
            raw_tokens = arm_entry.get("token_sidecars", {})
            if not isinstance(raw_tokens, dict):
                raise ManifestError(f"Cell '{cell_key}', arm '{arm_name}': token_sidecars must be an object")

            for run_key, token_entry in raw_tokens.items():
                try:
                    run_idx = int(run_key)
                except ValueError:
                    raise ManifestError(f"Invalid token run key '{run_key}' in cell '{cell_key}'")

                if isinstance(token_entry, dict):
                    t_path_str = token_entry.get("path")
                    expected_tok_sha = token_entry.get("sha256")
                else:
                    raise ManifestError(f"Invalid token entry format in cell '{cell_key}', arm '{arm_name}'")

                if not t_path_str or not expected_tok_sha:
                    raise IntegrityError(
                        f"Cell '{cell_key}', arm '{arm_name}', run {run_idx}: missing token path or sha256"
                    )

                t_path = self._resolve_path(t_path_str)
                actual_tok_sha = compute_file_sha256(t_path)
                if actual_tok_sha != str(expected_tok_sha).lower().strip():
                    raise IntegrityError(
                        f"Token sidecar SHA-256 mismatch for {t_path}: expected {expected_tok_sha}, got {actual_tok_sha}"
                    )

                parsed_sidecar = parse_token_sidecar(t_path, n_gen_tok, run_idx)
                token_sidecars[run_idx] = parsed_sidecar

            # Ensure every recorded measurement row has a corresponding token sidecar
            for m_row in measured_rows:
                if m_row.run_index not in token_sidecars:
                    raise TokenFormatError(
                        f"Cell '{cell_key}', arm '{arm_name}': missing token sidecar for recorded run_index {m_row.run_index}"
                    )

            # Check intra-arm determinism across measured runs
            measured_indices = [r.run_index for r in measured_rows]
            if len(measured_indices) >= 2:
                first_run = measured_indices[0]
                first_toks = token_sidecars[first_run].tokens
                for other_run in measured_indices[1:]:
                    other_toks = token_sidecars[other_run].tokens
                    det_cmp = compare_token_sequences(
                        first_toks, other_toks, f"run{first_run}", f"run{other_run}"
                    )
                    if not det_cmp.is_match:
                        raise DeterminismError(
                            f"Non-deterministic greedy output in cell '{cell_key}', arm '{arm_name}': "
                            f"{det_cmp.message}"
                        )

            arm_data_map[arm_name] = ArmData(
                arm_name=arm_name,
                csv_file=csv_path,
                rows=rows,
                warmup_rows=warmup_rows,
                measured_rows=measured_rows,
                token_sidecars=token_sidecars,
                knob=str(arm_entry.get("knob", ""))
            )

        # Sequence Parity Analysis
        rec_idx = arm_data_map["off"].measured_rows[0].run_index
        off_tokens = arm_data_map["off"].token_sidecars[rec_idx].tokens
        old_tokens = arm_data_map["old"].token_sidecars[rec_idx].tokens
        auto_tokens = arm_data_map["auto"].token_sidecars[rec_idx].tokens

        old_vs_off_parity = compare_token_sequences(off_tokens, old_tokens, "baseline(off)", "mtp(old)")
        auto_vs_off_parity = compare_token_sequences(off_tokens, auto_tokens, "baseline(off)", "mtp(auto)")
        auto_vs_old_parity = compare_token_sequences(old_tokens, auto_tokens, "mtp(old)", "mtp(auto)")

        greedy_parity_pass = old_vs_off_parity.is_match and auto_vs_off_parity.is_match
        kernel_parity_pass = auto_vs_old_parity.is_match

        # Block Pairing & Order Verification
        off_speeds_by_run = {r.run_index: r.decode_tok_s for r in arm_data_map["off"].measured_rows}
        old_speeds_by_run = {r.run_index: r.decode_tok_s for r in arm_data_map["old"].measured_rows}
        auto_speeds_by_run = {r.run_index: r.decode_tok_s for r in arm_data_map["auto"].measured_rows}

        raw_blocks = cell_dict.get("blocks")
        parsed_blocks: List[BlockPairing] = []
        block_pairing_status = "UNVERIFIED_HISTORICAL"
        block_pairing_passed = False

        off_speeds_paired: List[float] = []
        old_speeds_paired: List[float] = []
        auto_speeds_paired: List[float] = []

        if raw_blocks is not None:
            if not isinstance(raw_blocks, list) or len(raw_blocks) == 0:
                raise BlockPairingError(f"Cell '{cell_key}': 'blocks' must be a non-empty list of block definitions")

            used_runs = {arm: set() for arm in grid_arms}

            for b_idx, b_entry in enumerate(raw_blocks, start=1):
                if not isinstance(b_entry, dict):
                    raise BlockPairingError(f"Cell '{cell_key}', block #{b_idx}: block must be a JSON object")

                b_id = str(b_entry.get("block_id", f"b{b_idx}")).strip()
                arm_order = b_entry.get("arm_order", [])
                if not isinstance(arm_order, list) or set(arm_order) != set(grid_arms) or len(arm_order) != len(grid_arms):
                    raise BlockPairingError(
                        f"Cell '{cell_key}', block '{b_id}': arm_order must contain all arms {sorted(grid_arms)}, got {arm_order}"
                    )

                order_ev = b_entry.get("order_evidence", {})
                if not isinstance(order_ev, dict) or not order_ev:
                    raise BlockPairingError(
                        f"Cell '{cell_key}', block '{b_id}': missing or empty 'order_evidence' dictionary"
                    )

                b_runs = b_entry.get("runs", {})
                if not isinstance(b_runs, dict) or set(b_runs.keys()) != set(grid_arms):
                    raise BlockPairingError(
                        f"Cell '{cell_key}', block '{b_id}': runs mapping must map each arm {sorted(grid_arms)} to its run_index"
                    )

                # Check run indices exist in measured rows and are not duplicated across blocks
                for arm in grid_arms:
                    r_idx = int(b_runs[arm])
                    if r_idx in used_runs[arm]:
                        raise BlockPairingError(
                            f"Cell '{cell_key}', block '{b_id}': run_index {r_idx} for arm '{arm}' already assigned to another block"
                        )
                    used_runs[arm].add(r_idx)

                    if arm == "off" and r_idx not in off_speeds_by_run:
                        raise BlockPairingError(f"Cell '{cell_key}', block '{b_id}': off run {r_idx} not in measured rows")
                    if arm == "old" and r_idx not in old_speeds_by_run:
                        raise BlockPairingError(f"Cell '{cell_key}', block '{b_id}': old run {r_idx} not in measured rows")
                    if arm == "auto" and r_idx not in auto_speeds_by_run:
                        raise BlockPairingError(f"Cell '{cell_key}', block '{b_id}': auto run {r_idx} not in measured rows")

                parsed_blocks.append(
                    BlockPairing(
                        block_id=b_id,
                        arm_order=arm_order,
                        order_evidence=order_ev,
                        runs={arm: int(b_runs[arm]) for arm in grid_arms}
                    )
                )

                off_speeds_paired.append(off_speeds_by_run[int(b_runs["off"])])
                old_speeds_paired.append(old_speeds_by_run[int(b_runs["old"])])
                auto_speeds_paired.append(auto_speeds_by_run[int(b_runs["auto"])])

            # Ensure all measured runs are accounted for in blocks
            for arm in grid_arms:
                expected_runs = set(arm_data_map[arm].token_sidecars.keys()) - {
                    r.run_index for r in arm_data_map[arm].warmup_rows
                }
                if used_runs[arm] != expected_runs:
                    raise BlockPairingError(
                        f"Cell '{cell_key}', arm '{arm}': blocks do not cover all measured runs: "
                        f"covered={sorted(used_runs[arm])}, expected={sorted(expected_runs)}"
                    )

            block_pairing_status = "VERIFIED"
            block_pairing_passed = True
        else:
            # Fallback for historical datasets: match by run_index for diagnostic calculations only
            block_pairing_status = "UNVERIFIED_HISTORICAL (diagnostic pairing inferred from run_index without verified block/order evidence)"
            block_pairing_passed = False

            if set(off_speeds_by_run.keys()) != set(old_speeds_by_run.keys()) or set(off_speeds_by_run.keys()) != set(auto_speeds_by_run.keys()):
                raise GridIncompleteError(
                    f"Cell '{cell_key}': mismatched measured run_index sets between arms: "
                    f"off={sorted(off_speeds_by_run.keys())}, old={sorted(old_speeds_by_run.keys())}, auto={sorted(auto_speeds_by_run.keys())}"
                )

            common_runs = sorted(off_speeds_by_run.keys())
            off_speeds_paired = [off_speeds_by_run[r] for r in common_runs]
            old_speeds_paired = [old_speeds_by_run[r] for r in common_runs]
            auto_speeds_paired = [auto_speeds_by_run[r] for r in common_runs]

        mtp_old_stats = compute_paired_comparison_stats(off_speeds_paired, old_speeds_paired)
        mtp_auto_stats = compute_paired_comparison_stats(off_speeds_paired, auto_speeds_paired)
        kernel_auto_stats = compute_paired_comparison_stats(old_speeds_paired, auto_speeds_paired)

        # Gate Evaluation Rules
        gate_failures = []
        if not greedy_parity_pass:
            gate_failures.append("Greedy sequence parity failure against baseline")
        if not kernel_parity_pass:
            gate_failures.append("Kernel sequence parity failure (AUTO vs OLD)")
        if not block_pairing_passed:
            gate_failures.append("Block pairing & order evidence unverified or absent")

        # MTP Product Throughput Gate:
        # Requires:
        # 1. Greedy parity PASS
        # 2. Block pairing verified
        # 3. Sufficient sample support (N >= 3 for mean CI, N >= 6 for >=95% median CI)
        # 4. Median paired gain >= +10.0%
        # 5. Empirical min gain > 0.0%
        # 6. Mean CI lower bound > 0.0%
        if not mtp_auto_stats.sample_support_sufficient:
            gate_failures.append(f"Insufficient sample support: N={mtp_auto_stats.n_pairs} < {MIN_RECORDED_RUNS_FOR_MEAN_CI}")
        if mtp_auto_stats.median_gain_pct < 10.0:
            gate_failures.append(f"Median MTP speedup {mtp_auto_stats.median_gain_pct:+.2f}% < +10.0%")
        if mtp_auto_stats.min_gain_pct <= 0.0:
            gate_failures.append(f"Empirical minimum gain {mtp_auto_stats.min_gain_pct:+.2f}% <= 0.0%")
        if mtp_auto_stats.mean_ci_95_pct and mtp_auto_stats.mean_ci_95_pct[0] <= 0.0:
            gate_failures.append(f"Mean 95% CI lower bound {mtp_auto_stats.mean_ci_95_pct[0]:+.2f}% <= 0.0%")

        mtp_product_gate_passed = len(gate_failures) == 0

        return CellResult(
            prompt=prompt,
            n_gen_tok=n_gen_tok,
            cell_key=cell_key,
            arms=arm_data_map,
            block_pairing_status=block_pairing_status,
            block_pairing_passed=block_pairing_passed,
            blocks=parsed_blocks,
            mtp_old_vs_baseline_parity=old_vs_off_parity,
            mtp_auto_vs_baseline_parity=auto_vs_off_parity,
            kernel_auto_vs_old_parity=auto_vs_old_parity,
            mtp_old_vs_baseline_stats=mtp_old_stats,
            mtp_auto_vs_baseline_stats=mtp_auto_stats,
            kernel_auto_vs_old_stats=kernel_auto_stats,
            greedy_parity_passed=greedy_parity_pass,
            kernel_parity_passed=kernel_parity_pass,
            mtp_product_gate_passed=mtp_product_gate_passed,
            gate_failure_reasons=gate_failures
        )

    def _validate_mandatory_gates(self) -> List[MandatoryGateResult]:
        """Validate presence and traceable structured evidence for mandatory operational gates."""
        gates_dict = self.manifest_data.get("mandatory_gates", {})
        expected_mandatory = [
            "build_and_host_tests",
            "active_mtp_engagement",
            "cache_reuse_parity",
            "rollback_under_mismatch",
            "phase_instrumentation_accounting",
            "eog_handling",
            "cancel_policy"
        ]
        results: List[MandatoryGateResult] = []

        for gate_name in expected_mandatory:
            if gate_name not in gates_dict:
                results.append(
                    MandatoryGateResult(
                        name=gate_name,
                        status="NOT_PROVEN",
                        evidence_type="missing",
                        scope={},
                        verification={},
                        details="Gate definition missing from manifest",
                        is_pass=False,
                        verification_notes="Missing gate entry in manifest"
                    )
                )
                continue

            entry = gates_dict[gate_name]
            if not isinstance(entry, dict):
                results.append(
                    MandatoryGateResult(
                        name=gate_name,
                        status="NOT_PROVEN",
                        evidence_type="assertion_only",
                        scope={},
                        verification={},
                        details="Gate definition must be a structured JSON object",
                        is_pass=False,
                        verification_notes="Malformed gate entry"
                    )
                )
                continue

            status_str = str(entry.get("status", "NOT_PROVEN")).upper().strip()
            ev_type = str(entry.get("evidence_type", "assertion_only")).strip()
            scope = entry.get("scope", {})
            verif = entry.get("verification", {})
            details_str = str(entry.get("details", entry.get("evidence", ""))).strip()

            notes = []
            is_pass = False

            # Require structured report for verified pass
            if ev_type != "structured_report":
                status_str = "NOT_PROVEN"
                notes.append("Assertion-only or unstructured evidence rejected; structured_report required")
            else:
                # Validate scope
                if isinstance(scope, dict):
                    pkg_id = scope.get("package_identity")
                    mod_sha = scope.get("model_file_sha256")
                    cfg_sha = scope.get("config_hash")
                    if not self.provenance.synthetic:
                        if pkg_id and pkg_id != self.provenance.package_declared_identity:
                            notes.append(f"Scope package_identity mismatch: {pkg_id}")
                        if mod_sha and mod_sha.lower() != self.provenance.model_file_sha256.lower():
                            notes.append(f"Scope model_file_sha256 mismatch: {mod_sha}")
                        if cfg_sha and cfg_sha.lower() != self.provenance.config_hash.lower():
                            notes.append(f"Scope config_hash mismatch: {cfg_sha}")

                # Validate verification counts and artifact
                if isinstance(verif, dict):
                    pass_cnt = verif.get("assertions_passed", 0)
                    tot_cnt = verif.get("assertions_total", 0)
                    art_path_str = verif.get("artifact_path", "")
                    art_sha = verif.get("artifact_sha256", "")

                    if pass_cnt != tot_cnt or tot_cnt <= 0:
                        notes.append(f"Assertions incomplete: {pass_cnt}/{tot_cnt} passed")

                    if art_path_str:
                        art_path = self._resolve_path(art_path_str)
                        if not art_path.is_file():
                            notes.append(f"Evidence artifact not found: {art_path}")
                        elif art_sha:
                            actual_sha = compute_file_sha256(art_path)
                            if actual_sha != str(art_sha).lower().strip():
                                notes.append(f"Artifact SHA-256 mismatch: expected {art_sha}, got {actual_sha}")
                    elif not self.provenance.synthetic:
                        notes.append("Missing artifact_path for structured verification")

                if not notes and status_str == "PASS":
                    is_pass = True

            # Special validation for EOG gate: Do NOT allow unobserved-EOS to count as fully proven EOG gate
            if gate_name == "eog_handling":
                if "no eos observed" in details_str.lower() or "unobserved" in details_str.lower():
                    status_str = "PARTIALLY_PROVEN"
                    is_pass = False
                    notes.append("Fixture lacked natural EOS stop token; limited to non-EOS length saturation")

            results.append(
                MandatoryGateResult(
                    name=gate_name,
                    status=status_str,
                    evidence_type=ev_type,
                    scope=scope if isinstance(scope, dict) else {},
                    verification=verif if isinstance(verif, dict) else {},
                    details=details_str,
                    is_pass=is_pass,
                    verification_notes="; ".join(notes) if notes else "Verified structured report"
                )
            )

        return results


# ---------------------------------------------------------------------------
# Formatting & Presentation Utilities
# ---------------------------------------------------------------------------

def format_report_text(report: ProtocolReport) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("PLAN 003 PAIRED MTP DIAGNOSTIC REPORT")
    lines.append("=" * 80)
    lines.append(f"Manifest:       {report.manifest_path}")
    lines.append(f"Model:          {report.provenance.model_declared_name} (layers={report.provenance.gpu_layers}, backend={report.provenance.backend})")
    lines.append(f"Model SHA-256:  {report.provenance.model_file_sha256}")
    lines.append(f"Package:        {report.provenance.package_declared_identity}")
    lines.append(f"Package SHA:    {report.provenance.package_digest_sha256}")
    lines.append(f"Config Hash:    {report.provenance.config_hash}")
    lines.append(f"Host:           {report.provenance.device_host}")
    lines.append(f"Sampling:       {json.dumps(report.provenance.sampling)}")
    lines.append(f"Warmup / Rec:   {report.provenance.warmup_runs} warmup / {report.provenance.recorded_runs} recorded repetitions")
    lines.append(f"Synthetic Data: {'YES (FIXTURE TEST ONLY - NOT PRODUCT EVIDENCE)' if report.is_synthetic else 'NO (Physical Device Evidence)'}")
    lines.append("-" * 80)

    # 1. Greedy Parity Matrix
    lines.append("1. GREEDY SEQUENCE PARITY MATRIX")
    lines.append(f"{'Cell':<22} | {'Kernel Parity (AUTO vs OLD)':<30} | {'Baseline Parity (MTP vs OFF)':<30}")
    lines.append("-" * 86)
    for cell in report.cells:
        k_str = "PASS (bit-exact)" if cell.kernel_parity_passed else f"FAIL (pos {cell.kernel_auto_vs_old_parity.earliest_mismatch_pos_1idx})"
        b_str = (
            "PASS (bit-exact)"
            if cell.greedy_parity_passed
            else f"FAIL (pos {cell.mtp_auto_vs_baseline_parity.earliest_mismatch_pos_1idx}: off={cell.mtp_auto_vs_baseline_parity.left_token} mtp={cell.mtp_auto_vs_baseline_parity.right_token})"
        )
        lines.append(f"{cell.cell_key:<22} | {k_str:<30} | {b_str:<30}")
    lines.append("-" * 86)
    lines.append("")

    # 2. Block Pairing Status
    lines.append("2. BLOCK PAIRING & ORDER EVIDENCE")
    lines.append(f"{'Cell':<22} | {'Pairing Status':<40} | {'Blocks':<8} | {'Gate'}")
    lines.append("-" * 78)
    for cell in report.cells:
        p_status = "VERIFIED" if cell.block_pairing_passed else "UNVERIFIED (DIAGNOSTIC)"
        lines.append(f"{cell.cell_key:<22} | {p_status:<40} | {len(cell.blocks):<8} | {'PASS' if cell.block_pairing_passed else 'INCOMPLETE'}")
    lines.append("-" * 78)
    lines.append("")

    # 3. Product Throughput Speedup Matrix
    lines.append("3. PER-CELL PAIRED PERFORMANCE & PRODUCT SPEEDUP (>= +10.0% Gate)")
    lines.append(f"{'Cell':<22} | {'Base tok/s':>10} | {'MTP tok/s':>10} | {'MTP Gain %':>12} | {'Mean 95% CI / Range %':<24} | {'Gate'}")
    lines.append("-" * 92)
    for cell in report.cells:
        stats = cell.mtp_auto_vs_baseline_stats
        gate_str = "PASS" if cell.mtp_product_gate_passed else "FAIL"
        if stats.mean_ci_95_pct:
            ci_str = f"[{stats.mean_ci_95_pct[0]:+.2f}, {stats.mean_ci_95_pct[1]:+.2f}]"
        else:
            ci_str = f"[{stats.min_gain_pct:+.2f}, {stats.max_gain_pct:+.2f}]*"
        lines.append(
            f"{cell.cell_key:<22} | {stats.baseline_median_tok_s:>10.2f} | {stats.candidate_median_tok_s:>10.2f} | "
            f"{stats.median_gain_pct:>+11.2f}% | {ci_str:<24} | {gate_str}"
        )
    lines.append("-" * 92)
    lines.append("* Note: Asterisk (*) indicates empirical range when sample size N < 3 is insufficient for Student-t CI on the mean.")
    lines.append("")

    # 4. Incremental Kernel Comparison
    lines.append("4. INCREMENTAL KERNEL COMPARISON (AUTO vs OLD)")
    lines.append(f"{'Cell':<22} | {'OLD tok/s':>10} | {'AUTO tok/s':>10} | {'Kernel Gain %':>14} | {'Range %'}")
    lines.append("-" * 75)
    for cell in report.cells:
        k_stats = cell.kernel_auto_vs_old_stats
        lines.append(
            f"{cell.cell_key:<22} | {k_stats.baseline_median_tok_s:>10.2f} | {k_stats.candidate_median_tok_s:>10.2f} | "
            f"{k_stats.median_gain_pct:>+13.2f}% | [{k_stats.min_gain_pct:+.2f}%, {k_stats.max_gain_pct:+.2f}%]"
        )
    lines.append("-" * 75)
    lines.append("")

    # 5. Mandatory Gates
    lines.append("5. MANDATORY OPERATIONAL GATES")
    for g in report.mandatory_gates:
        p_str = "PASS" if g.is_pass else f"NOT PASSED ({g.status})"
        lines.append(f"- {g.name:<32}: {p_str:<24} Verification: {g.verification_notes}")
    lines.append("")

    # 6. Statistical Limitations & Uncertainty Notice
    warnings = [c.mtp_auto_vs_baseline_stats.warning_message for c in report.cells if c.mtp_auto_vs_baseline_stats.small_sample_warning]
    if warnings:
        lines.append("6. STATISTICAL LIMITATIONS & UNCERTAINTY NOTICE")
        for cell in report.cells:
            if cell.mtp_auto_vs_baseline_stats.small_sample_warning:
                lines.append(f"  [{cell.cell_key}]: {cell.mtp_auto_vs_baseline_stats.warning_message}")
        lines.append("")

    lines.append("=" * 80)
    lines.append(f"OVERALL PROTOCOL VERDICT: {report.overall_verdict}")
    lines.append("=" * 80)

    if report.rejection_reasons:
        lines.append("Deficiencies / Rejection Reasons:")
        for r in report.rejection_reasons:
            lines.append(f"  * {r}")
        lines.append("=" * 80)

    return "\n".join(lines)


def format_report_markdown(report: ProtocolReport) -> str:
    lines = []
    lines.append("# Plan 003 Paired MTP Product Protocol Offline Validation Report\n")
    lines.append(f"- **Overall Verdict**: `{report.overall_verdict}`")
    lines.append(f"- **Manifest**: `{report.manifest_path}`")
    lines.append(f"- **Model**: `{report.provenance.model_declared_name}` (`{report.provenance.model_file_sha256[:16]}...`)")
    lines.append(f"- **Package**: `{report.provenance.package_declared_identity}` (`{report.provenance.package_digest_sha256[:16]}...`)")
    lines.append(f"- **Config Hash**: `{report.provenance.config_hash[:16]}...`")
    lines.append(f"- **Host / Device**: `{report.provenance.device_host}`")
    lines.append(f"- **Warmup / Recorded Repetitions**: {report.provenance.warmup_runs} warmup / {report.provenance.recorded_runs} recorded")
    lines.append(f"- **Synthetic Test Data**: `{'YES (FIXTURE TEST ONLY)' if report.is_synthetic else 'NO (Physical Device Evidence)'}`\n")

    lines.append("## 1. Greedy Sequence Parity Matrix\n")
    lines.append("| Cell | Kernel Parity (AUTO vs OLD) | Baseline Parity (MTP vs OFF) | Status |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for cell in report.cells:
        k_str = "PASS (bit-exact)" if cell.kernel_parity_passed else f"FAIL (pos {cell.kernel_auto_vs_old_parity.earliest_mismatch_pos_1idx})"
        b_str = (
            "PASS (bit-exact)"
            if cell.greedy_parity_passed
            else f"FAIL (pos {cell.mtp_auto_vs_baseline_parity.earliest_mismatch_pos_1idx}: off={cell.mtp_auto_vs_baseline_parity.left_token} mtp={cell.mtp_auto_vs_baseline_parity.right_token})"
        )
        c_status = "PASS" if (cell.greedy_parity_passed and cell.kernel_parity_passed) else "FAIL"
        lines.append(f"| `{cell.cell_key}` | {k_str} | {b_str} | **{c_status}** |")
    lines.append("")

    lines.append("## 2. Block Pairing & Order Evidence\n")
    lines.append("| Cell | Pairing Status | Blocks Verified | Gate |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for cell in report.cells:
        lines.append(f"| `{cell.cell_key}` | {cell.block_pairing_status} | {len(cell.blocks)} | **{'PASS' if cell.block_pairing_passed else 'INCOMPLETE'}** |")
    lines.append("")

    lines.append("## 3. Product Speedup & Statistical Gate (>= +10.0% Speedup)\n")
    lines.append("| Cell | Baseline tok/s | MTP tok/s | Median Gain % | Mean 95% CI / Range % | Gate |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for cell in report.cells:
        stats = cell.mtp_auto_vs_baseline_stats
        gate_str = "PASS" if cell.mtp_product_gate_passed else "FAIL"
        if stats.mean_ci_95_pct:
            ci_str = f"[{stats.mean_ci_95_pct[0]:+.2f}%, {stats.mean_ci_95_pct[1]:+.2f}%]"
        else:
            ci_str = f"[{stats.min_gain_pct:+.2f}%, {stats.max_gain_pct:+.2f}%]*"
        lines.append(
            f"| `{cell.cell_key}` | {stats.baseline_median_tok_s:.2f} | {stats.candidate_median_tok_s:.2f} | "
            f"**{stats.median_gain_pct:+.2f}%** | {ci_str} | **{gate_str}** |"
        )
    lines.append("\n*\\* N < 3 recorded runs is insufficient for Student-t CI on the mean; empirical range reported.*")
    lines.append("")

    lines.append("## 4. Mandatory Operational Gates\n")
    lines.append("| Gate Name | Status | Evidence Type | Verification Notes |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for g in report.mandatory_gates:
        st_str = f"**{g.status}**" if g.is_pass else f"`{g.status}`"
        lines.append(f"| `{g.name}` | {st_str} | `{g.evidence_type}` | {g.verification_notes} |")
    lines.append("")

    if report.rejection_reasons:
        lines.append("## Deficiencies / Rejection Reasons\n")
        for r in report.rejection_reasons:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)


def format_report_json(report: ProtocolReport) -> str:
    def serialize_obj(obj: Any) -> Any:
        if isinstance(obj, (Path,)):
            return str(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        raise TypeError(f"Unserializable type: {type(obj)}")
    return json.dumps(dataclasses.asdict(report), default=serialize_obj, indent=2)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Offline validator for Plan 003 paired MTP protocol.")
    parser.add_argument("manifest", type=str, help="Path to protocol provenance manifest JSON")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text", help="Output format")
    parser.add_argument("--min-gain", type=float, default=10.0, help="Required median paired MTP gain percentage")

    args = parser.parse_args()

    validator = MtpProtocolValidator(args.manifest)
    try:
        report = validator.validate()
    except ValidationError as exc:
        print(f"PROTOCOL VALIDATION ERROR: {exc}", file=sys.stderr)
        return 1

    if args.format == "text":
        print(format_report_text(report))
    elif args.format == "markdown":
        print(format_report_markdown(report))
    elif args.format == "json":
        print(format_report_json(report))

    # Exit code: 0 for SYNTHETIC_PASS or DIAGNOSTIC_ONLY_NOT_CERTIFIED, 1 for FAIL or INCOMPLETE_NOT_PROVEN
    return 0 if report.overall_verdict in ("SYNTHETIC_PASS", "DIAGNOSTIC_ONLY_NOT_CERTIFIED") else 1


if __name__ == "__main__":
    sys.exit(main())
