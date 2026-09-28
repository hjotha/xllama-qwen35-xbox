"""Check DRAFT weight names, shapes, and GGUF types against the frozen model inventory.

This verifies the loader contract only; it does not execute inference.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "research/dflash2-xbox/gpu-implementation/f0-2026-09-28/tensor-inventory.json"
HIDDEN = 5120
FFN = 17408
QUANT = {"Q3_K", "Q4_K", "Q5_K", "F32"}


def main():
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    tensors = {item["name"]: item for item in data["tensors"]}
    expected = {"output_norm.weight": ([HIDDEN], {"F32"})}
    for layer in range(5):
        p = f"blk.{layer}."
        expected.update({
            p + "attn_norm.weight": ([HIDDEN], {"F32"}),
            p + "attn_q.weight": ([HIDDEN, 4096], QUANT),
            p + "attn_k.weight": ([HIDDEN, 1024], QUANT),
            p + "attn_v.weight": ([HIDDEN, 1024], QUANT),
            p + "attn_output.weight": ([4096, HIDDEN], QUANT),
            p + "attn_q_norm.weight": ([128], {"F32"}),
            p + "attn_k_norm.weight": ([128], {"F32"}),
            p + "ffn_norm.weight": ([HIDDEN], {"F32"}),
            p + "ffn_gate.weight": ([HIDDEN, FFN], QUANT),
            p + "ffn_up.weight": ([HIDDEN, FFN], QUANT),
            p + "ffn_down.weight": ([FFN, HIDDEN], QUANT),
            p + "attn_conv_proj.weight": ([HIDDEN, 1280], QUANT),
            p + "attn_conv_base": ([HIDDEN, 2, 2], {"F32"}),
            p + "ffn_conv_proj.weight": ([HIDDEN, 1280], QUANT),
            p + "ffn_conv_base": ([HIDDEN, 2, 2], {"F32"}),
        })
    for name, (shape, types) in expected.items():
        item = tensors.get(name)
        if item is None or item["shape"] != shape or item["type"] not in types:
            raise AssertionError(f"DRAFT weight mismatch: {name}: {item}")
    optional = [name for name in tensors if name.endswith(".attn_sinks.weight")]
    for name in optional:
        item = tensors[name]
        if item["shape"] != [32] or item["type"] != "F32":
            raise AssertionError(f"attention sink mismatch: {name}: {item}")
    print(json.dumps({
        "inventory_sha256": data["model_sha256"],
        "required_tensors": len(expected),
        "optional_sinks_present": len(optional),
        "pass": True,
    }))


if __name__ == "__main__":
    main()
