from pathlib import Path

p = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\dflash-gpu-draft-probe.cpp")
text = p.read_text(encoding="utf-8")
lines = text.splitlines()
for i, line in enumerate(lines):
    if "fullPipelineParityGate" in line:
        lines[i] = '         << ",\\"fullPipelineParityGate\\":false"' + ' << ",\\"barrierMode\\":\\"" << (gpu_draft.barrier_mode() == DflashGpuDraftEngine::BarrierMode::Strict ? "strict" : "minimal") << "\\""'
        break
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Fixed line 805 successfully!")
