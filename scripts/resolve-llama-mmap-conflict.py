#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Resolve the src/llama-mmap.cpp 3-way conflict left by apply-uwp-patches.sh.

Against the beellama fork the UWP patch cannot merge cleanly on this one file:
the fork guards the Windows branch with LLAMA_WINDOWS_DESKTOP, a macro it never
defines anywhere, while the patch replaces the test with an AppContainer-aware
expression that is correct on desktop Windows and inside the Xbox AppContainer.

The upstream side of the conflict is therefore kept. Resolution is line-oriented
on purpose: a regex would need escapes that do not survive a shell heredoc, and
that failure mode is silent and easy to misread as "no conflict to resolve".

Exits non-zero, without touching the file, when the file has no such conflict or
when a conflict remains that this does not understand.
"""

import sys

PATH = "src/llama-mmap.cpp"
GOOD = (
    "#if defined(_POSIX_MEMLOCK_RANGE) || (defined(_WIN32) && "
    "(!defined(WINAPI_FAMILY) || WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)))"
)


def main() -> int:
    with open(PATH, "r", newline="") as handle:
        lines = handle.read().splitlines()

    out = []
    i = 0
    fixed = 0
    while i < len(lines):
        # A conflict block here is exactly five lines: ours' #if, the ====,
        # theirs' #if, the >>>> terminator, and the marker line above.
        if (
            lines[i].startswith("<<<<<<<")
            and i + 4 < len(lines)
            and lines[i + 1].startswith("#if")
            and "LLAMA_WINDOWS_DESKTOP" in lines[i + 1]
            and lines[i + 2].startswith("=======")
            and lines[i + 3].startswith("#if")
            and lines[i + 4].startswith(">>>>>>>")
        ):
            out.append(GOOD)
            fixed += 1
            i += 5
            continue
        out.append(lines[i])
        i += 1

    if fixed == 0:
        print("resolve-llama-mmap-conflict: no llama-mmap.cpp hunk matched", file=sys.stderr)
        return 1
    if any(line.startswith(("<<<<<<<", "=======", ">>>>>>>")) for line in out):
        print("resolve-llama-mmap-conflict: unresolved conflict left in " + PATH, file=sys.stderr)
        return 1

    with open(PATH, "w", newline="") as handle:
        handle.write("\n".join(out) + "\n")
    print("resolve-llama-mmap-conflict: resolved %d llama-mmap.cpp hunk(s)" % fixed)
    return 0


if __name__ == "__main__":
    sys.exit(main())