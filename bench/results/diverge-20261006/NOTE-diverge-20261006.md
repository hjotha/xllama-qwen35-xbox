# diverge-20261006 evidence notes (diagnostic-only annotations, not gates)

- rev75/rev76-replay-result.csv: A/B/C/D/E remainder runs. Raw measurements,
  superseded in interpretation only (see below); files preserved byte-intact.
- rev80-diverge-result.csv: error marker only (UWP pairing bug: split-block
  assumption vs interleaved device layout). Device capture itself was valid
  (paired=1354, first_diff=30 both reps, in xllama-rev80-full.log).
- rev82-diverge-result.csv + rev82-diverge-log-slice.txt: INVALID FOR
  RANKING. The extent-only alignment compared reverse-chronological snapshot
  slots (node59 ' (view)' [524288,1,3,1], node61 CPY, and per-layer
  recurrences) at different logical positions (B3 slots 0..2 = pos 144/143/
  142 vs B4 slots 0..2 = pos 145/144/143), so their mismatches are expected
  positional shifts, not width effects. Raw bytes preserved for audit.
  Same-token rows (z-0/node62 MUL_MAT 4.76837e-06 onward) remain valid
  measurements. See ALIGN-LOGICAL-SNAPSHOT-POSITIONS-157-20261006T0957Z.
- E-branch inference RETRACTED (rev76 audit): E's single-11 decode rewrites
  only snapshot slot 0, so post-trim restore reads stale slot 1
  (through-143, missing token 144). E tested trim + stale restore, NOT trim
  alone; its "safe" outcome must not be read as trim-alone evidence. What
  stands: D (width4, no trim) near-flips without reproducing; B (width4 +
  trim, correct slot-1 restore) reproduces output33=1204. Trim's solo effect
  is unknown from E. (B-vs-D is NOT a pure trim contrast either: the 4th
  token differs in-batch (11 vs 2261) and the correction batching differs.)
  See TRACE-RESTORATION-CAUSE-157-20261006T0434Z.
- Pairing/export repairs (rev81 interleaved layout, rev82→83 multi-axis
  token rule, CSV quoting, drift identity logging) are HARNESS root causes,
  not numerical model root causes; that distinction is kept throughout.
