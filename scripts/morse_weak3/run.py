#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append the serial same-session morse-weak3 gate."""

from __future__ import annotations

import csv
import os
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "bench" / "morse-weak3" / "rows.csv"
SEEDS = tuple(range(2000, 2012))
SYSTEMS = {
    "m38r10": 400_000,
    "m38r14": 400_000,
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
    "lj104": 4_000_000,
}
ARMS = ("current_default", "adaptive_one_shot_twin")
FIELDS = (
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "inside_s",
    "outside_s",
    "code_revision",
    "host",
    "stop",
    "restarts",
    "orbits",
    "hops",
)


def parse_result(text: str, system: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith(f"system={system} "):
            return dict(field.split("=", 1) for field in line.split())
    raise RuntimeError(f"no result for {system}")


def existing() -> set[tuple[str, str, int]]:
    if not ROWS.exists():
        return set()
    with ROWS.open(newline="") as handle:
        return {
            (row["system"], row["arm"], int(row["seed"]))
            for row in csv.DictReader(handle)
        }


def append(row: dict[str, str]) -> None:
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    fresh = not ROWS.exists()
    with ROWS.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    if socket.gethostname() != "bench":
        raise SystemExit("run inside the bench UTS namespace")
    binaries = {
        "current_default": os.environ["MORSE_WEAK3_BASELINE"],
        "adaptive_one_shot_twin": os.environ["MORSE_WEAK3_FIX"],
    }
    revisions = {
        "current_default": os.environ["MORSE_WEAK3_BASE_REVISION"],
        "adaptive_one_shot_twin": os.environ["MORSE_WEAK3_FIX_REVISION"],
    }
    done = existing()
    for system, budget in SYSTEMS.items():
        for seed in SEEDS:
            for arm in ARMS:
                if (system, arm, seed) in done:
                    continue
                mode = "recommended" if arm == ARMS[0] else "recommended_adaptive"
                proc = subprocess.run(
                    [binaries[arm], "run", system, str(budget), str(seed), mode],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result = parse_result(proc.stdout, system)
                append(
                    {
                        "system": system,
                        "arm": arm,
                        "seed": str(seed),
                        "budget": str(budget),
                        "hit": result["hit"],
                        "evals_to_hit": (
                            "" if result["first_calls"] == "-" else result["first_calls"]
                        ),
                        "charged": result["charged"],
                        "best": f"{float(result['best']):.8f}",
                        "wall_s": f"{float(result['wall_s']):.6f}",
                        "inside_s": f"{float(result['inside_s']):.6f}",
                        "outside_s": f"{float(result['outside_s']):.6f}",
                        "code_revision": revisions[arm],
                        "host": "bench",
                        "stop": result["stop"],
                        "restarts": result["restarts"],
                        "orbits": result["orbits"],
                        "hops": result["hops"],
                    }
                )
                done.add((system, arm, seed))


if __name__ == "__main__":
    main()
