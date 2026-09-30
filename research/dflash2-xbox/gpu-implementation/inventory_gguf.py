"""Inventory DFlash GGUF tensors using the vendored gguf-py reader."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.gguf_py))
    import gguf

    reader = gguf.GGUFReader(str(args.model))
    tensors = []
    counts = Counter()
    bytes_by_type = Counter()
    for tensor in reader.tensors:
        quant_type = gguf.GGMLQuantizationType(tensor.tensor_type)
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        shape = [int(value) for value in tensor.shape]
        # GGUF shape is ordered from the contiguous GGML axis outward.
        strides = [type_size]
        for axis in shape[:-1]:
            strides.append(strides[-1] * axis // (block_size if len(strides) == 1 else 1))
        record = {"name": tensor.name, "shape": shape, "type": quant_type.name,
                  "block_size": block_size, "type_size": type_size,
                  "logical_strides_bytes": strides, "offset": int(tensor.data_offset),
                  "bytes": int(tensor.n_bytes)}
        tensors.append(record)
        counts[quant_type.name] += 1
        bytes_by_type[quant_type.name] += int(tensor.n_bytes)
    report = {"model": str(args.model), "model_bytes": args.model.stat().st_size,
              "model_sha256": digest(args.model), "tensor_count": len(tensors),
              "counts_by_type": dict(counts), "bytes_by_type": dict(bytes_by_type),
              "tensors": tensors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model_bytes", "model_sha256", "tensor_count", "counts_by_type", "bytes_by_type")}))


if __name__ == "__main__":
    main()
