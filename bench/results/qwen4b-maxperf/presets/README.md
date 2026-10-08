# Device presets (rev160)

- `llama-ini-mtp2-production.txt` — documented production profile (mtp=2,
  FFN knob OFF, profiling OFF). This is what the device runs now.
- `llama-ini-mtp0-seq.txt` — optional sequential preset (mtp=0).

Seq activation: upload `llama-ini-mtp0-seq.txt` as `llama.ini`,
upload `d3d12swiglu.txt=1`, restart the app; the first context binds the
seq profile and FFN dispatches engage. Revert: upload the mtp2 file, set
`d3d12swiglu.txt=0`, restart.
