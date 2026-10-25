from pathlib import Path

target = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\dflash-gpu-draft-engine.cpp")
text = target.read_text(encoding="utf-8")

# 1. Initialize barrier_mode_
old_init = "    queue_ = sync.queue();"
new_init = """    queue_ = sync.queue();
    barrier_mode_ = BarrierMode::Minimal;
    {
        const auto parent = model_path.parent_path();
        std::ifstream bm_file(parent / "dflash-barrier-mode.txt");
        if (bm_file) {
            std::string mode_str;
            std::getline(bm_file, mode_str);
            if (mode_str.find("strict") != std::string::npos) barrier_mode_ = BarrierMode::Strict;
        }
    }"""
assert old_init in text, "old_init not found"
text = text.replace(old_init, new_init, 1)

# 2. matmul barrier(q8)
old_q8 = """            cmd_->Dispatch(weight.columns / 256, batch, 1);
            barrier(q8);
            quant_source_ = input;"""
new_q8 = """            cmd_->Dispatch(weight.columns / 256, batch, 1);
            if (barrier_mode_ == BarrierMode::Strict) barrier(q8);
            quant_source_ = input;"""
assert old_q8 in text, "old_q8 not found"
text = text.replace(old_q8, new_q8, 1)

# 3. matmul destination != output
old_dest = "    if (destination != output) barrier(y);"
new_dest = "    if (destination != output && barrier_mode_ == BarrierMode::Strict) barrier(y);"
assert old_dest in text, "old_dest not found"
text = text.replace(old_dest, new_dest, 1)

# 4. norm
old_norm = """    cmd_->Dispatch(rows, 1, 1);
    barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
new_norm = """    cmd_->Dispatch(rows, 1, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
assert old_norm in text, "old_norm not found"
text = text.replace(old_norm, new_norm, 1)

# 5. rope
old_rope = """    cmd_->Dispatch(width / kHead, batch, 1);
    barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
new_rope = """    cmd_->Dispatch(width / kHead, batch, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
assert old_rope in text, "old_rope not found"
text = text.replace(old_rope, new_rope, 1)

# 6. hadamard
old_hadamard = """    cmd_->Dispatch(heads, batch, 1);
    barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
new_hadamard = """    cmd_->Dispatch(heads, batch, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
assert old_hadamard in text, "old_hadamard not found"
text = text.replace(old_hadamard, new_hadamard, 1)

# 7. conv
old_conv = """    cmd_->Dispatch((batch * kHidden + 63) / 64, 1, 1);
    barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
new_conv = """    cmd_->Dispatch((batch * kHidden + 63) / 64, 1, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
assert old_conv in text, "old_conv not found"
text = text.replace(old_conv, new_conv, 1)

# 8. scale_add
old_scale_add = """    cmd_->Dispatch((batch * width + 127) / 128, 1, 1);
    barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
new_scale_add = """    cmd_->Dispatch((batch * width + 127) / 128, 1, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
assert old_scale_add in text, "old_scale_add not found"
text = text.replace(old_scale_add, new_scale_add, 1)

# 9. swiglu
old_swiglu = """    cmd_->Dispatch((batch * kFfn + 127) / 128, 1, 1);
    barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
new_swiglu = """    cmd_->Dispatch((batch * kFfn + 127) / 128, 1, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == output) quant_source_ = Count;"""
assert old_swiglu in text, "old_swiglu not found"
text = text.replace(old_swiglu, new_swiglu, 1)

# 10. attention
old_attn = """    cmd_->Dispatch(kQHeads, batch, 1);
    barrier(y);
    if (quant_source_ == AttentionRot) quant_source_ = Count;"""
new_attn = """    cmd_->Dispatch(kQHeads, batch, 1);
    if (barrier_mode_ == BarrierMode::Strict) barrier(y);
    if (quant_source_ == AttentionRot) quant_source_ = Count;"""
assert old_attn in text, "old_attn not found"
text = text.replace(old_attn, new_attn, 1)

target.write_text(text, encoding="utf-8")
print("Engine successfully patched with barrier_mode_!")
