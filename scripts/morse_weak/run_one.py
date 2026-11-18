#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One seed of the same-session baseline (recommended) vs adaptive population
(recommended_adaptive) across the 12 Morse clusters and 5 Lennard-Jones clusters.

Usage: run_one.py SYSTEM ARM SEED
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj104": 4_000_000,
    "m38r3": 400_000,
    "m38r6": 400_000,
    "m38r10": 400_000,
    "m38r14": 400_000,
    "m55r3": 1_000_000,
    "m55r6": 1_000_000,
    "m55r10": 1_000_000,
    "m55r14": 1_000_000,
    "m75r6": 4_000_000,
    "m75r14": 4_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
}
SEEDS = tuple(range(2000, 2012))
ARMS = ("recommended", "recommended_adaptive")


def hostname() -> str:
    return Path("/proc/sys/kernel/hostname").read_text().splitlines()[0].strip()


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().splitlines()[0].strip()


def main() -> int:
    if hostname() != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    if len(sys.argv) != 4:
        print("usage: run_one.py SYSTEM ARM SEED", file=sys.stderr)
        return 2
    system, arm, seed_text = sys.argv[1:]
    if system not in PAPER_BUDGETS or arm not in ARMS:
        print("unknown system or arm", file=sys.stderr)
        return 2
    seed = int(seed_text)
    if seed not in SEEDS:
        print("seed is outside 2000-2011", file=sys.stderr)
        return 2
    revision = os.environ.get("MORSE_WEAK_REVISION", "")
    if not revision:
        print("MORSE_WEAK_REVISION is unset", file=sys.stderr)
        return 2
    budget = PAPER_BUDGETS[system]
    binary = os.environ.get("MORSE_WEAK_BIN", "/work/target-panel/release/examples/morse_weak")
    proc = subprocess.run(
        [binary, "run", system, str(budget), str(seed), arm],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    fields = {}
    for line in proc.stdout.splitlines():
        if line.startswith("system="):
            fields = dict(part.split("=", 1) for part in line.split())
    if fields.get("system") != system or fields.get("seed") != str(seed) or fields.get("arm") != arm:
        raise RuntimeError("morse_weak stdout has no row")
    hit = fields["hit"] == "1"
    evals = "" if fields["first_calls"] == "-" else fields["first_calls"]
    print(
        ",".join(
            [
                system,
                arm,
                str(seed),
                str(budget),
                str(int(hit)),
                evals,
                fields["charged"],
                f"{float(fields['best']):.8f}",
                f"{float(fields['wall_s']):.6f}",
                f"{float(fields['inside_s']):.6f}",
                f"{float(fields['outside_s']):.6f}",
                revision,
                hostname(),
                fields["stop"],
                fields["restarts"],
                fields["orbits"],
                fields["hops"],
            ]
        ),
        flush=True,
    )
    print(f"boot_id={boot_id()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
