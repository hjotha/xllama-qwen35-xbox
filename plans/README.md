# MTP implementation plan

Prepared 2026-10-03 for the existing OpenCode task on optiplex. Source review was scoped to Xbox Qwen3.5 MTP correctness, performance, rollback memory, and benchmark evidence. Other subsystems were not audited.

| Plan                                                                                 | Priority                               | Depends on                               | Status                                                                       |
| ------------------------------------------------------------------------------------ | -------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------- |
| [001: correct MTP and demonstrate an Xbox speedup](001-mtp-correctness-and-speed.md) | P1                                     | Existing task branch and .193 build flow | IN PROGRESS: OpenCode read the plan and started its drift check              |
| [002: Rev32 review and build steer](002-mtp-review-and-build-steer.md)               | Historical                             | 001                                      | SUPERSEDED: use 003 for carry indexing and current verification requirements |
| [003: Rev37 review and next steps](003-mtp-rev37-review-next-steps.md)               | P0 validation, then P1/P2 improvements | Current Rev37 implementation             | TODO: review complete; implementation and new Xbox validation not started    |

Current review (2026-10-04, HEAD `051b05e`): the Rev37 CSV medians were independently confirmed, but the >=10% speed gate was not reached and functional counters do not establish output parity. Plan 003 reconciles the remaining private-draft state, session carry, corrective decode and D3D12 batch costs. It is the current handoff. Plans 001/002 remain historical; their existence or prior receipt does not imply their acceptance gates passed.

Execute correctness before performance tuning, then require a matched hardware A/B. Update status with actual evidence.

Handoff evidence from the OpenCode SQLite transcript on 2026-10-03 (UTC): steer `STEER-MTP-REV29-REVIEW-20261003` recorded at 20:29:10; full plan read completed at 20:29:30; acknowledgment "Plano lido. Executando o drift check primeiro" at 20:29:35; drift-check command completed at 20:29:36. This proves receipt and start, not completion of fixes or an Xbox speedup.

Considered and rejected:

- Closing MTP as impossible from Rev29: the port omits known state/rollback mechanisms and has a confirmed verifier bug.
- Declaring depth 1 intrinsically slower: the canonical combined verifier can emit an accepted draft plus an additional target token; hardware cost still requires measurement.
- Blind `chain_heads`/layer-offset port: Qwen3.5 has one MTP head.
- Treating `ctx_other` as shared Qwen KV: the original context setup clears it; hardware logs show `shared=0`.
- Treating missing TOP_K/ARGMAX GPU sampling as the only bottleneck: CPU sampling works and must be profiled.
- Claiming the 4029/4147 ratio proves UWP OOM: those fields represent different memory-accounting domains.
- Updating the whole BeeLlama pin: relevant .57 MTP/rollback source matches the local pin.
