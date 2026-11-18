#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run one seed of the current and proposed recommended search."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PAPER_BUDGETS = {
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
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj102": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "lj150": 4_000_000,
    "au55": 1_000_000,
}
SEEDS = tuple(range(2000, 2012))
ARMS = ("current_recommended", "new_recommended")
LEGACY_MORSE = frozenset(PAPER_BUDGETS) - {"lj102", "lj103", "lj150", "au55"}


def hostname() -> str:
    return Path("/proc/sys/kernel/hostname").read_text().splitlines()[0].strip()


def key_values(text: str, system: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith(f"system={system} "):
            return dict(part.split("=", 1) for part in line.split())
    raise RuntimeError(f"no result for {system}")


def baseline(system: str, budget: int, seed: int) -> dict[str, str]:
    if system in LEGACY_MORSE:
        binary = os.environ["MORSE_WEAK2_BASELINE"]
        proc = subprocess.run(
            [binary, "run", system, str(budget), str(seed), "recommended"],
            check=True,
            capture_output=True,
            text=True,
        )
        return key_values(proc.stdout, system)
    if system == "au55":
        binary = os.environ["MORSE_WEAK2_AU_BASELINE"]
        proc = subprocess.run(
            [binary, system, str(budget), str(seed), "recommended"],
            check=True,
            capture_output=True,
            text=True,
        )
        row = key_values(proc.stdout, system)
        row.update(restarts="0", orbits="0", hops="0")
        return row
    binary = os.environ["MORSE_WEAK2_LJ_BASELINE"]
    env = {**os.environ, "H2H_BUDGET": str(budget)}
    proc = subprocess.run(
        [binary, "run", system, str(seed)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith(f"{system},")]
    if len(lines) != 1:
        raise RuntimeError(f"no baseline row for {system}")
    row = dict(
        zip(
            "system code seed budget hit evals_to_hit charged wall_s best head tree cpu pins pot_s".split(),
            lines[0].split(","),
        )
    )
    return {
        "hit": row["hit"],
        "first_calls": row["evals_to_hit"] or "-",
        "charged": row["charged"],
        "best": row["best"],
        "wall_s": row["wall_s"],
        "inside_s": row["pot_s"],
        "outside_s": f"{max(0.0, float(row['wall_s']) - float(row['pot_s'])):.6f}",
        "stop": "hit" if row["hit"] == "1" else "budget",
        "restarts": "0",
        "orbits": "0",
        "hops": "0",
    }


def candidate(system: str, budget: int, seed: int) -> dict[str, str]:
    proc = subprocess.run(
        [os.environ["MORSE_WEAK2_CANDIDATE"], "run", system, str(budget), str(seed), "recommended"],
        check=True,
        capture_output=True,
        text=True,
    )
    return key_values(proc.stdout, system)


def main() -> int:
    if hostname() != "bench" or len(sys.argv) != 4:
        return 2
    system, arm, seed_text = sys.argv[1:]
    if system not in PAPER_BUDGETS or arm not in ARMS:
        return 2
    seed = int(seed_text)
    if seed not in SEEDS:
        return 2
    budget = PAPER_BUDGETS[system]
    row = baseline(system, budget, seed) if arm == ARMS[0] else candidate(system, budget, seed)
    revision = (
        os.environ["MORSE_WEAK2_BASE_REVISION"]
        if arm == ARMS[0]
        else os.environ["MORSE_WEAK2_FIX_REVISION"]
    )
    first = row["first_calls"]
    if first == "-":
        first = ""
    print(
        ",".join(
            [
                system,
                arm,
                str(seed),
                str(budget),
                row["hit"],
                first,
                row["charged"],
                f"{float(row['best']):.8f}",
                f"{float(row['wall_s']):.6f}",
                f"{float(row['inside_s']):.6f}",
                f"{float(row['outside_s']):.6f}",
                revision,
                hostname(),
                row["stop"],
                row["restarts"],
                row["orbits"],
                row["hops"],
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
