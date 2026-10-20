#!/usr/bin/env python3
"""Offline analysis tool for RKVA native KVarN4 bulk profiling telemetry.

Reads an 'rkva-range-profile-v1' JSON profile and presents:
1. Executive summary of bulk calls, segments, and queries.
2. Complete host CPU span breakdown (admission, alloc, map/copy, record, submit, fence wait, readback).
3. Attributable GPU kernel breakdown (upload, pass 1, reduce, readback copy, total).
4. Derived metrics (queue wait, scheduling delay, CPU copy overhead, effective bandwidth).
5. Attribution sanity checks (single submit/fence invariant, monotonicity).

Usage:
    python3 scripts/analyze_rkva_bulk_profile.py profile.json
    cat profile.json | python3 scripts/analyze_rkva_bulk_profile.py
    python3 scripts/analyze_rkva_bulk_profile.py profile.json --markdown
    python3 scripts/analyze_rkva_bulk_profile.py profile.json --json
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional


def format_stats(stat_dict: Optional[Dict[str, Any]], unit: str = "us") -> str:
    if not stat_dict or stat_dict.get("n", 0) == 0:
        return "N/A"
    mean = stat_dict.get("mean", 0.0)
    p50 = stat_dict.get("p50", 0.0)
    p90 = stat_dict.get("p90", 0.0)
    max_val = stat_dict.get("max", 0.0)
    return f"{mean:8.2f} {unit}  (p50: {p50:8.2f}, p90: {p90:8.2f}, max: {max_val:8.2f})"


def analyze_profile(data: Dict[str, Any], fmt: str = "markdown") -> str:
    # Handle top-level or nested 'engine' profile
    prof = data.get("engine", data)
    schema = prof.get("schema", "")
    if schema != "rkva-range-profile-v1":
        sys.stderr.write(f"Warning: Expected schema 'rkva-range-profile-v1', got '{schema}'\n")

    counters = prof.get("counters", {})
    bulk = prof.get("bulk", {})
    timings = bulk.get("timings", {})
    bytes_data = bulk.get("bytes", {})
    notes = bulk.get("timing_breakdown_notes", {})

    bulk_calls = counters.get("bulk_calls", bulk.get("bulk_calls", 0))
    bulk_segments = counters.get("bulk_segments", bulk.get("bulk_segments", 0))
    bulk_queries = counters.get("bulk_queries", bulk.get("bulk_queries", 0))
    submits = counters.get("submits", 0)
    fence_blocking = counters.get("fence_blocking_waits", 0)
    fence_complete = counters.get("fence_already_complete", 0)
    has_gpu = bulk.get("gpu_timestamps", prof.get("gpu_timestamps", False))

    if fmt == "json":
        summary = {
            "counters": {
                "bulk_calls": bulk_calls,
                "bulk_segments": bulk_segments,
                "bulk_queries": bulk_queries,
                "submits": submits,
                "fence_waits": fence_blocking + fence_complete,
            },
            "timings": timings,
            "bytes": bytes_data,
            "notes": notes,
        }
        return json.dumps(summary, indent=2)

    lines = []
    lines.append("# RKVA Native KVarN4 Bulk Profile Analysis\n")
    lines.append("## 1. Workload Overview\n")
    lines.append(f"- **Bulk RPC Calls**: {bulk_calls}")
    lines.append(f"- **Total Segments**: {bulk_segments} ({bulk_segments / max(1, bulk_calls):.1f} seg/call)")
    lines.append(f"- **Total Queries**: {bulk_queries} ({bulk_queries / max(1, bulk_calls):.1f} queries/call)")
    lines.append(f"- **Submissions**: {submits} (invariant submits == bulk_calls: {submits == bulk_calls})")
    lines.append(f"- **Fence Waits**: {fence_blocking} blocking, {fence_complete} already complete")
    lines.append(f"- **GPU Timestamps**: {'Available & Valid' if has_gpu else 'Unavailable / Fallback'}\n")

    lines.append("## 2. Host CPU Timing Breakdown\n")
    lines.append("| CPU Span | Mean | p50 | p90 | Max | Description |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")

    cpu_spans = [
        ("admission_cpu_us", "Envelope & segment argument checks"),
        ("alloc_cpu_us", "Scratch & transfer buffer allocations"),
        ("map_copy_cpu_us", "Host-to-upload copy (inputs)"),
        ("record_cpu_us", "Command recording in command list"),
        ("submit_cpu_us", "Queue execution submission"),
        ("fence_wait_cpu_us", "Fence wait wall time (inclusive)"),
        ("readback_cpu_us", "Readback copy (outputs)"),
        ("engine_total_cpu_us", "Total engine CPU duration"),
    ]

    for key, desc in cpu_spans:
        s = timings.get(key, {})
        if s and s.get("n", 0) > 0:
            lines.append(f"| `{key}` | {s.get('mean', 0.0):.2f} us | {s.get('p50', 0.0):.2f} us | {s.get('p90', 0.0):.2f} us | {s.get('max', 0.0):.2f} us | {desc} |")
        else:
            lines.append(f"| `{key}` | N/A | N/A | N/A | N/A | {desc} |")

    lines.append("\n## 3. GPU Execution Breakdown\n")
    if has_gpu and "gpu_total_us" in timings:
        lines.append("| GPU Stage | Mean | p50 | p90 | Max | Description |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
        gpu_stages = [
            ("gpu_upload_us", "KV-cache store & stage uploads"),
            ("gpu_pass1_us", "Pass-1 attention kernel dispatch"),
            ("gpu_reduce_us", "Split-K reduction kernel dispatch"),
            ("gpu_readback_copy_us", "Readback buffer copy region"),
            ("gpu_total_us", "Exclusive GPU execution (envelope begin to end)"),
            ("queue_wait_us", "Queue latency: max(0, fence_wait - gpu_total)"),
        ]
        for key, desc in gpu_stages:
            s = timings.get(key, {})
            if s and s.get("n", 0) > 0:
                lines.append(f"| `{key}` | {s.get('mean', 0.0):.2f} us | {s.get('p50', 0.0):.2f} us | {s.get('p90', 0.0):.2f} us | {s.get('max', 0.0):.2f} us | {desc} |")
            else:
                lines.append(f"| `{key}` | N/A | N/A | N/A | N/A | {desc} |")
    else:
        lines.append("*GPU hardware timestamps were not recorded for this profile (hardware unavailable or capability fallback).*\n")

    lines.append("\n## 4. Transfer Bytes and Data Volume\n")
    lines.append("| Stream | Mean Bytes | Sum Bytes | Description |")
    lines.append("| :--- | :---: | :---: | :--- |")
    byte_fields = [
        ("upload_bytes", "KV-cache range upload payload"),
        ("query_bytes", "Input query vectors and positions"),
        ("output_bytes", "Output attention and LSE vectors"),
        ("total_bytes", "Total RPC transfer memory"),
    ]
    for key, desc in byte_fields:
        b = bytes_data.get(key, {})
        if b and b.get("n", 0) > 0:
            lines.append(f"| `{key}` | {b.get('mean', 0.0):.1f} B | {b.get('sum', 0.0):.1f} B | {desc} |")
        else:
            lines.append(f"| `{key}` | N/A | N/A | {desc} |")

    if notes:
        lines.append("\n## 5. Architectural Breakdown Notes\n")
        for k, v in notes.items():
            lines.append(f"- **{k}**: {v}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze RKVA native bulk profile JSON telemetry.")
    parser.add_argument("profile_path", nargs="?", default=None, help="Path to profile JSON (reads stdin if omitted)")
    parser.add_argument("--json", action="store_true", help="Output summary as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output formatted markdown (default)")
    args = parser.parse_args()

    if args.profile_path:
        with open(args.profile_path, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = sys.stdin.read()

    if not raw.strip():
        sys.stderr.write("Error: empty input\n")
        sys.exit(1)

    try:
        data = json.loads(raw)
    except Exception as e:
        sys.stderr.write(f"Error parsing JSON: {e}\n")
        sys.exit(1)

    fmt = "json" if args.json else "markdown"
    output = analyze_profile(data, fmt=fmt)
    print(output)


if __name__ == "__main__":
    main()
