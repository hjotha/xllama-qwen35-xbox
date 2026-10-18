#!/usr/bin/env python3
"""Descriptor-3 (serial prefix) row-representation partition model for a future
native KVarN4 multi-query tile kernel. CPU only; no kernel is implemented here.

For a query with window W = q_count = min(active_count, pos - origin + 1), the
revision-3 kernels (rkva_range_attn_kvarn4[_opt].hlsl with RKVA_SERIAL_PREFIX=1)
read row r < W as
    tail    if r >= W - min(W, precision_tail_tokens)      (ORIGINAL F16, ring 256)
    stage   if global group of r is 0 (sink) or r is in W's live group and W % 128 != 0
    record  otherwise                                       (sealed KVarN4 record)
A multi-query tile may share one K/V decode of row r across its queries only if
every query reads r from the same representation. This model checks, for the
real origins (72704, and 0 with the sink group), tails 0/128, windows around
every seal boundary and spans 1..9, that

* the shared record region [record_lo, min(record_hi, tail_begin)) of the
  SMALLEST window is a record row for every query in the tile (so it never
  reads a group sealed after the smallest query's prefix), and it is maximal
  among record rows common to all queries below the smallest window;
* rows outside it are classified per query (and can differ, e.g. a group
  sealed between two queries' windows is stage for the earlier one and record
  for the later one), so they must use each query's own path;
* the per-query residual stays bounded (<= 128 stage + 128 tail + 8 alignment
  + span rows), which is what makes a shared record-tile kernel worthwhile at
  ~58k live rows.
Run: python3 tests/rkva_kvarn4_mq_partition_model.py
"""
import sys

GROUP, TILE = 128, 8


def representation(r, w, origin, ptt):
    assert 0 <= r < w
    if r >= w - min(w, ptt):
        return "tail"
    group = r // GROUP
    global_group = origin // GROUP + group
    live_group = (w - 1) // GROUP
    if global_group == 0 or (group == live_group and w % GROUP != 0):
        return "stage"
    return "record"


def shared_record_region(windows, origin, ptt):
    """[lo, hi) rows read as records by every query: bounded by the smallest window."""
    w = min(windows)
    record_lo = GROUP if origin == 0 else 0
    tail_begin = w - min(w, ptt)
    record_hi = w & ~(GROUP - 1)
    hi = min(record_hi, tail_begin)
    return record_lo, max(record_lo, hi)


def check(origin, ptt, w0, span):
    windows = [w0 + i for i in range(span)]
    lo, hi = shared_record_region(windows, origin, ptt)
    w_min = min(windows)
    for r in range(lo, hi):
        reps = {representation(r, w, origin, ptt) for w in windows}
        assert reps == {"record"}, (origin, ptt, windows, r, reps)
    # Maximality: no row of the smallest window outside [lo, hi) is a record for all queries.
    differ = 0
    for r in range(w_min):
        if lo <= r < hi:
            continue
        reps = [representation(r, w, origin, ptt) for w in windows]
        assert reps.count("record") != len(reps), (origin, ptt, windows, r)
        differ += len(set(reps)) > 1
    residual = max(w - (hi - lo) for w in windows)
    bound = (GROUP if origin == 0 else 0) + GROUP + ptt + TILE + span
    assert residual <= bound, (origin, ptt, windows, residual, bound)
    return residual, differ


def main():
    worst = {}
    differ_cases = 0
    for origin in (72704, 0):
        for ptt in (0, 128):
            for base in (127, 128, 129, 255, 256, 257, 383, 384, 385):
                for span in range(1, 10):
                    for shift in range(-9, 2):
                        w0 = base + shift
                        if w0 < 1:
                            continue
                        residual, differ = check(origin, ptt, w0, span)
                        key = (origin, ptt)
                        worst[key] = max(worst.get(key, 0), residual)
                        differ_cases += differ > 0
    # Real filled range: live 58128, nq 3 / 5 / 9 contiguous MTP verify windows.
    for ptt in (0, 128):
        for nq in (3, 5, 9):
            windows = [58128 - nq + 1 + i for i in range(nq)]
            lo, hi = shared_record_region(windows, 72704, ptt)
            print(f"live=58128 ptt={ptt:3d} nq={nq}: shared record rows {hi - lo} "
                  f"({100.0 * (hi - lo) / 58128:.2f}%), per-query residual <= "
                  f"{max(windows) - (hi - lo)} rows")
    for (origin, ptt), value in sorted(worst.items()):
        print(f"origin={origin:5d} ptt={ptt:3d}: worst per-query residual {value} rows")
    assert differ_cases > 0, "boundary cases where queries disagree must exist"
    print(f"{differ_cases} tile configurations have rows whose representation differs "
          "between queries (handled per query, never shared)")
    print("rkva_kvarn4_mq_partition_model: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
