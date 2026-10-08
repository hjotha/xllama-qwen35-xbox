# Sequential-only FFN SWIGLU process profile (rev159) — enforced

Package: GianlucaMazza.xllama_1.6.0.159_x64__pj67f1fcj4n14

Design: SwigluModePolicy binds the capability at the FIRST context and
never changes it while the process lives. A later context with the other
MTP mode is rejected BEFORE creation (d3d12_swiglu_profile_accepts ->
"restart required"); a rejected creation never mutates backend state
(conflict check precedes the target-layer set). Once bound ON, a later
knob=0 cannot open the gate (host test: CHECK_FALSE(seq.accepts(0,true))).

Device verification (knob d3d12swiglu.txt=1, fresh processes):

- MTP arm: `FFN SWIGLU profile bound: off (mtp_capable=1)`; zero FFN
  SWIGLU dispatches. FFN stays OFF whenever MTP is active.
- seq arm: `FFN SWIGLU profile bound: on (mtp_capable=0)`; 2112 FFN
  SWIGLU dispatches.
- Token parity both arms: 84e081965076a0c947bb (baseline chat64 digest).

Host tests: SwigluModePolicy latch 16/16 (bind-once, conflicting MTP
rejected, knob=0 after bind does not open the gate, MTP-first stays off);
full ctest green. Knob remains default OFF; seq candidate is experimental
until the API/session/cancel/stop/EOS gates and the final optimized
build pass.
