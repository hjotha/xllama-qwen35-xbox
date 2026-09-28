"""Compile DFlash Q6_K and Q8_0 x GGML Q8_K HLSL shaders."""

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "shaders" / "dflash"
GENERATED = ROOT / "shaders" / "generated"

TARGETS = [
    ("q6k_q8k_matmul.hlsl", "CSDot", "kDflashQ6kQ8kDotDxil", "dflash_q6k_q8k_dxil.h"),
    ("q8_0_q8k_matmul.hlsl", "CSDot", "kDflashQ8_0Q8kDotDxil", "dflash_q8_0_q8k_dxil.h"),
]


def find_dxc() -> str:
    found = shutil.which("dxc") or shutil.which("dxc.exe")
    if found:
        return found
    sdk = Path(r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe")
    if sdk.is_file():
        return str(sdk)
    raise SystemExit("dxc not found")


def compile_and_emit(dxc: str, source: str, entry: str, symbol: str, out_header: str) -> None:
    src_file = SOURCES / source
    out_file = GENERATED / out_header
    with tempfile.TemporaryDirectory() as temp_dir:
        dxil_bin = Path(temp_dir) / "kernel.dxil"
        subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", entry, "-O3", "-WX",
             "-Fo", str(dxil_bin), str(src_file)],
            check=True,
        )
        data = dxil_bin.read_bytes()
        digest = hashlib.sha256(data).hexdigest()

        lines = [
            f"// AUTO-GENERATED from {source}; do not edit.",
            "#pragma once",
            "#include <cstddef>",
            "#include <cstdint>",
            "// clang-format off",
            f"// {source} {entry}, cs_6_0, {len(data)} bytes, sha256 {digest}",
            f"static const uint8_t {symbol}[] = {{",
        ]
        for offset in range(0, len(data), 12):
            lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in data[offset:offset + 12]) + ",")
        lines.extend([
            "};",
            f"static const size_t {symbol}Size = {len(data)};",
            "// clang-format on",
        ])
        GENERATED.mkdir(parents=True, exist_ok=True)
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Generated {out_file.name}: {len(data)} bytes, sha256={digest}")


def main() -> None:
    dxc = find_dxc()
    for source, entry, symbol, out_header in TARGETS:
        compile_and_emit(dxc, source, entry, symbol, out_header)


if __name__ == "__main__":
    main()
