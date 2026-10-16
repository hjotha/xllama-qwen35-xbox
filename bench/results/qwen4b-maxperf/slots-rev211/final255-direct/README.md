# Final255 campaign scripts (prepared, not run)

These root Python and shell scripts were copied from final251-direct as a
script-only starting point for the Final255 campaign. Revision-specific
output paths, package PFN values, slot identifiers, and Final251 labels were
updated to Final255. Historical Final247/Final209 comparison inputs and the
strict 93/87 canonical VROUND validation logic remain unchanged.

verify-blocks-v3.py is byte-for-byte identical to its Final251 source.
campaign.py and longcampaign.py are intentionally omitted; use the
separate groups only: python3 run-group.py functional, then b1, then b2.
The numeric gate is separate (numeric.sh / the numeric stage). The
coverage aggregator is present for later review and has not been run.

The manifest covers only the copied root scripts and this README; it excludes
other files that may be created here by the parallel build/package owner. I did
not copy outcome JSON, CSV, logs, captures, build outputs, package-verification
data, or test results. No campaign script in this directory has been executed.
