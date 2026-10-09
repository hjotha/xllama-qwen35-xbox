#!/usr/bin/env python3
"""extract-segments.py <vround-dir> — delimited VROUND trace extraction.

Segment = last line containing "main_loop:" (run start) .. last "[xllama] done:"
line of the freshly downloaded accumulated log (run2 = the measured run). The
schedule/state canon covers only pertinent fields (width, reuse, accepted, mm,
feed, gen); the execution-binding counter `calls` is reported separately and is
not part of the canon. Counts are per new run (chat93/code87 expectation).
"""
import hashlib
import pathlib
import re
import sys

FIELDS = ["width", "reuse", "accepted", "mm", "feed", "gen"]


def segment(path):
    lines = path.read_text(errors="replace").splitlines()
    dones = [i for i, l in enumerate(lines) if "[xllama] done: " in l]
    if not dones:
        raise SystemExit(f"no done marker in {path}")
    starts = [i for i, l in enumerate(lines) if "main_loop:" in l and i < dones[-1]]
    if not starts:
        raise SystemExit(f"no main_loop start before done in {path}")
    return lines, lines[starts[-1]: dones[-1] + 1]


def main(d):
    d = pathlib.Path(d)
    fails = []
    canon_hashes = {}
    for cell in ("chat256-mtp", "code256-mtp"):
        for sw in (0, 1):
            log = d / f"{cell}-sw{sw}" / "run2.log"
            if not log.exists():
                fails.append(f"LOG_MISSING {log.name}")
                continue
            lines, seg = segment(log)
            seg_file = d / f"{cell}-sw{sw}.segment.log"
            seg_file.write_text(
                f"# run-scoped segment of {log} (last main_loop: start .. last "
                f"done: end)\n# whole-log lines: {len(lines)}; segment "
                f"lines: {len(seg)}; segment: {seg[0][:80]}\n"
                + "\n".join(seg) + "\n")
            vround = [l for l in seg if "[xllama] VROUND " in l]
            print(f"{cell}-sw{sw}: vround_lines={len(vround)} "
                  f"(whole_log={len(lines)} segment={len(seg)})")
            canon, calls = [], None
            for l in vround:
                kv = dict(p.split("=", 1) for p in
                          re.search(r"VROUND (.*)$", l).group(1).split())
                canon.append(" ".join(f"{k}={kv[k]}" for k in FIELDS))
                calls = kv.get("calls")
            ctext = "\n".join(canon) + "\n"
            cfile = d / f"{cell}-sw{sw}.schedule-canon.txt"
            cfile.write_text(ctext)
            h = hashlib.sha256(ctext.encode()).hexdigest()
            canon_hashes[(cell, sw)] = h
            (d / f"{cell}-sw{sw}.calls.txt").write_text(calls or "")
            print(f"  schedule-canon sha256={h} calls={calls}")
    for cell in ("chat256-mtp", "code256-mtp"):
        a = canon_hashes.get((cell, 0))
        b = canon_hashes.get((cell, 1))
        if a and b:
            status = "IDENTICAL" if a == b else "DIFFER"
            print(f"{cell}: schedule/state sw0 vs sw1 = {status} ({a[:16]})")
            if a != b:
                fails.append(f"CANON_DIFF {cell}")
    print(f"failures={len(fails)}")
    for f in fails:
        print("  ", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
