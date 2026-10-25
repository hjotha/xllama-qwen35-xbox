from pathlib import Path

p = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\dflash-gpu-draft-probe.cpp")
text = p.read_text(encoding="utf-8")
for line in text.splitlines():
    if "fullPipelineParityGate" in line:
        print("Found line:", repr(line))
        text = text.replace(line, line + ' << ",\"barrierMode\":\"" << (gpu_draft.barrier_mode() == DflashGpuDraftEngine::BarrierMode::Strict ? "strict" : "minimal") << "\\""', 1)
        break
p.write_text(text, encoding="utf-8")
print("Probe patched successfully!")
