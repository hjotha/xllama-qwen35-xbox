"""Turn Xbox real-tensor probe JSON into a run-level CSV without inventing TPS."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = (
    "source", "source_sha256", "tensor", "type", "batch", "backend", "run",
    "warmup", "iterations", "weight_bytes", "median_ms", "p95_ms",
    "gpu_median_ms", "gpu_p95_ms", "max_abs_vs_reference", "passed",
)


def rows_for(path: Path):
    raw = path.read_bytes()
    probe = json.loads(raw)["probe"]
    if not probe.get("passed"):
        return []
    common = {
        "source": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor": probe["tensor"],
        "type": probe.get("type", probe.get("sourceType")),
        "batch": probe.get("batch", 1),
        "warmup": probe["warmup"],
        "iterations": probe["iterations"],
        "weight_bytes": probe.get("weightBytes", probe.get("packedBytes")),
        "passed": True,
    }
    rows = []
    if "cpuRepetitions" in probe:
        for run in probe["cpuRepetitions"]:
            rows.append({**common, "backend": "ggml_cpu_4_threads", "run": run["run"],
                         "median_ms": run["medianMs"], "p95_ms": run["p95Ms"],
                         "max_abs_vs_reference": run["maxAbsErr"] if run.get("referenceChecked") else ""})
        for run in probe["repetitions"]:
            rows.append({**common, "backend": "d3d12_hlsl_packed", "run": run["run"],
                         "median_ms": run["withCopyMedianMs"], "p95_ms": run["withCopyP95Ms"],
                         "gpu_median_ms": run["gpuMedianMs"], "gpu_p95_ms": run["gpuP95Ms"],
                         "max_abs_vs_reference": run["maxAbsErr"]})
    elif probe.get("probe") == "directml_q4_k_to_fp16_gemv":
        if not all(run.get("accuracyPassed") is True for run in probe["runs"]):
            return []
        for run in probe["runs"]:
            rows.append({**common, "backend": "directml_fp16_converted", "run": run["run"],
                         "median_ms": run["boundaryMedianMs"], "p95_ms": run["boundaryP95Ms"],
                         "gpu_median_ms": run["gpuMedianMs"], "gpu_p95_ms": run["gpuP95Ms"],
                         "max_abs_vs_reference": run["vsQ4"]["maxAbs"],
                         "passed": run["accuracyPassed"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((*args.evidence.glob("real-*.json"), *args.evidence.glob("q4k-batch*.json")))
    rows = [row for path in files for row in rows_for(path)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"files": len(files), "rows": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
