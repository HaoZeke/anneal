#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Apply the measurement patches to one unpacked GMIN source tree.

The cap counts one potential entry as one energy and gradient call and
stops at the paper budget. The Morse and Gupta timers record time inside
those potentials. One hunk only set the replica index to 1. This source
already loops that index from 1, and the hunk does not apply, so it is
left out. Nothing here reads or checks a source revision.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

H2H = Path("bench/lj104-lattice/v2/h2h-gmin.patch")
MORSE = Path("scripts/morse_h2h/morse-timer.patch")
GUPTA = Path("bench/metal-cluster-h2h/gmin-bgupta.patch")
JP_ADDED = ["#ifndef GMIN_MPI\n", "   JP=1\n", "#endif\n"]


def drop_replica_hunk(text: str) -> str:
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("@@"):
            hunk = [lines[index]]
            index += 1
            while index < len(lines) and not lines[index].startswith("@@") and not lines[index].startswith("--- "):
                hunk.append(lines[index])
                index += 1
            added = [line[1:] for line in hunk[1:] if line.startswith("+") and not line.startswith("+++")]
            if added == JP_ADDED:
                continue
            kept.extend(hunk)
        else:
            kept.append(lines[index])
            index += 1
    return "".join(kept)


def apply_patch(dest: Path, text: str, label: str) -> None:
    proc = subprocess.run(
        ["patch", "-p1", "--forward", "--batch", "-d", str(dest)],
        input=text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"patch failed: {label}")


def markers_present(dest: Path) -> bool:
    source = dest / "source"
    cap = source / "h2h_cap.f90"
    morse = source / "morsegh.f90"
    gupta = source / "BGupta.f90"
    if not cap.is_file() or not morse.is_file() or not gupta.is_file():
        return False
    return "H2H_POT_ENTER" in morse.read_text() and "H2H_POT_ENTER" in gupta.read_text()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    dest = args.dest
    if markers_present(dest):
        print("patches already present")
        return 0
    root = args.root
    apply_patch(dest, drop_replica_hunk((root / H2H).read_text()), "cap")
    apply_patch(dest, (root / MORSE).read_text(), "morse")
    apply_patch(dest, (root / GUPTA).read_text(), "gupta")
    if not markers_present(dest):
        raise SystemExit("measurement markers missing after patch")
    print("patches applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
