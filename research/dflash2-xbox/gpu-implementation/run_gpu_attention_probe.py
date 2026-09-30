"""Build and run the isolated attention probe on this Windows D3D12 host.

No app packaging, Xbox install, RPC changes, or repository build outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VCVARS = Path(r"C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not VCVARS.is_file():
        raise SystemExit(f"MSVC environment missing: {VCVARS}")
    with tempfile.TemporaryDirectory(prefix="dflash-attention-") as directory:
        temp = Path(directory)
        main_cpp = temp / "main.cpp"
        main_cpp.write_text(
            '#include "gpu-attention-probe.h"\n'
            '#include <cstdio>\n'
            'int main() { const auto r = run_dflash_gpu_attention_probe(); '
            'std::puts(r.json.c_str()); return r.passed ? 0 : 1; }\n',
            encoding="utf-8",
        )
        exe = temp / "attention_probe.exe"
        command = [
            "cl", "/nologo", "/std:c++17", "/EHsc", "/MD", "/O2", "/W3", "/utf-8",
            f"/I{ROOT / 'include'}", f"/I{ROOT / 'shaders/generated'}",
            f"/I{ROOT / 'uwp/dflash-rpc'}", str(main_cpp),
            str(ROOT / "uwp/dflash-rpc/gpu-attention-probe.cpp"),
            str(ROOT / "uwp/dflash-rpc/dflash-gpu-kv.cpp"),
            f"/Fe:{exe}", "/link", "dxgi.lib", "dxguid.lib", "uuid.lib",
        ]
        batch = temp / "build.cmd"
        batch.write_text(
            f'@echo off\r\ncall "{VCVARS}" >nul 2>nul\r\n'
            'if errorlevel 1 exit /b 1\r\n'
            + subprocess.list2cmdline(command) + '\r\n'
            'exit /b %errorlevel%\r\n',
            encoding="ascii",
        )
        subprocess.run(["cmd.exe", "/d", "/c", str(batch)], cwd=temp, check=True)
        if args.compile_only:
            print(json.dumps({"compiled": True, "ran": False, "platform": "Windows host"}))
            return
        run = subprocess.run([str(exe)], cwd=temp, capture_output=True, text=True)
        print(run.stdout, end="")
        if run.stderr:
            print(run.stderr, end="", file=__import__("sys").stderr)
        if run.returncode:
            raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
