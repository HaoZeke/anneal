#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Time the hop between quenches on M80 at rho = 3.

The parent rows were measured on this VM, hostname bench, one core,
against the binary before the common-neighbour trail change. A fresh
row uses the binary just built. The quenched energy and the move arms
are the search; they matched the parent on both budgets.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

PARENT = (
    ("m80r3", 2000, 20000, "parent", -690.13974055, 10.633751, 0.631093, 10.002658, 1),
    ("m80r3", 2000, 200000, "parent", -690.42850638, 94.303019, 6.206821, 88.096198, 1),
)
FIELDS = [
    "system",
    "seed",
    "budget",
    "binary",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "search_matches_parent",
]
PARENT_BEST = {20000: -690.13974055, 200000: -690.42850638}
# Full paper budget on the changed binary, seed 2000, same host. Not
# remeasured when the short rows are refreshed.
RECORDED = (
    ("m80r3", 2000, 4000000, "this", -690.51891025, 202.979855, 115.129077, 87.850779, ""),
)


def parse_row(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("m80r3,")]
    if len(lines) != 1:
        raise SystemExit(f"expected one result row, saw {text!r}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s"
    )
    return next(csv.DictReader([header, lines[0]]))


def run_one(binary: Path, budget: int, seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "OMP_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", "m80r3", str(seed)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-400:] or proc.stdout[-400:])
    return parse_row(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()
    root = args.root
    work = Path("/work/morse-hop")
    target = work / "target"
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["TMPDIR"] = str(work / "rustc-tmp")
    (work / "rustc-tmp").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--offline",
            "-j",
            "1",
            "--example",
            "h2h_morse",
        ],
        cwd=root,
        check=True,
        env=env,
    )
    binary = target / "release" / "examples" / "h2h_morse"
    budgets = (args.budget,) if args.budget else (20000, 200000)
    fresh = []
    for budget in budgets:
        row = run_one(binary, budget, args.seed)
        best = float(row["best"])
        parent = PARENT_BEST.get(budget)
        matches = 1 if parent is not None and abs(best - parent) < 1e-8 else 0
        fresh.append(
            (
                "m80r3",
                args.seed,
                budget,
                "this",
                best,
                float(row["wall_s"]),
                float(row["pot_s"]),
                float(row["outside_s"]),
                matches,
            )
        )
        print(
            f"m80r3 seed {args.seed} budget {budget} best {best:.8f} "
            f"wall {float(row['wall_s']):.3f} pot {float(row['pot_s']):.3f} "
            f"outside {float(row['outside_s']):.3f}",
            file=sys.stderr,
        )
    if args.budget:
        return
    out = root / "bench" / "morse-hop" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(PARENT)
        writer.writerows(fresh)
        writer.writerows(RECORDED)


if __name__ == "__main__":
    main()
