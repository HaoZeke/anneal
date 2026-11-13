#!/usr/bin/env python3
# Copyright (c) anneal developers
"""GMIN on the ablation systems that have no recorded GMIN row.

LJ38, M80 at rho = 6, and TIP4P with 4 and 8 waters. Each call is the
harness's own run_gmin: scripts/gmin_h2h for LJ38, scripts/morse_h2h for
M80, scripts/tip4p_h2h for the waters. Starts are the anneal binaries'
`coords` output for the same seed, and for TIP4P the `.start.gmin` the
ablation's default arm wrote. The GMIN build is the one those harnesses
use. This process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
HERE = Path(__file__).resolve().parent
FIELDS = ["system", "arm", "seed", "budget", "hit", "evals_to_hit", "wall_s", "best", "pot_s", "outside_s"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--bins", type=Path, required=True)
    parser.add_argument("--ablation-runs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "bench":
        print(f"hostname is {socket.gethostname()}, not bench", file=sys.stderr)
        return 2
    scripts = args.root / "scripts"
    lj = load("lj_h2h", scripts / "gmin_h2h" / "run_campaign.py")
    morse = load("morse_h2h", scripts / "morse_h2h" / "run_campaign.py")
    tip4p = load("tip4p_h2h", scripts / "tip4p_h2h" / "run_campaign.py")
    tip4p.REFERENCES.update({4: -116.5904186160, 8: -305.5183221676})
    examples = args.root / "examples"
    done = set()
    if args.rows.is_file():
        with args.rows.open() as handle:
            done = {(r["system"], r["seed"]) for r in csv.DictReader(handle)}
    new_file = not args.rows.is_file()
    with args.rows.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for system in ("lj38", "m80r6", "tip4p4", "tip4p8"):
            for seed in SEEDS:
                if (system, str(seed)) in done:
                    continue
                work = args.work / system / f"s{seed}"
                work.mkdir(parents=True, exist_ok=True)
                if system == "lj38":
                    budget = 400_000
                    coords = work / "start.xyz"
                    subprocess.run(
                        [str(args.bins / "h2h_gmin_proof"), "coords", "lj38", str(seed), str(coords)],
                        check=True,
                    )
                    row = lj.run_gmin(
                        args.gmin, examples / "h2h_gmin_lj.data", 38, seed, budget,
                        -173.928427, coords, work / "run", work / "log.txt", work / "row.txt",
                    )
                elif system == "m80r6":
                    budget = 4_000_000
                    coords = work / "start.xyz"
                    subprocess.run(
                        [str(args.bins / "h2h_morse"), "coords", "m80r6", str(seed), str(coords)],
                        check=True,
                    )
                    row = morse.run_gmin(
                        args.gmin, examples / "h2h_gmin_morse.data", "m80r6", 6.0, seed, budget,
                        -378.333471, coords, work / "run", work / "log.txt", work / "row.txt",
                    )
                else:
                    n = int(system.removeprefix("tip4p"))
                    budget = 2_000_000
                    start = args.ablation_runs / system / f"default-s{seed}" / "out.start.gmin"
                    row = tip4p.run_gmin(
                        args.gmin, examples / "h2h_gmin_tip4p.data", n, seed, budget, start, work,
                    )
                wall = float(row["wall_s"])
                pot = float(row.get("pot_s", "nan"))
                outside = row.get("outside_s") or f"{max(wall - pot, 0.0):.6f}"
                hit = row["hit"].lower() in ("1", "true")
                writer.writerow(
                    {
                        "system": system,
                        "arm": "gmin",
                        "seed": seed,
                        "budget": budget,
                        "hit": "true" if hit else "false",
                        "evals_to_hit": row["evals_to_hit"],
                        "wall_s": row["wall_s"],
                        "best": row.get("best", ""),
                        "pot_s": row.get("pot_s", ""),
                        "outside_s": outside,
                    }
                )
                handle.flush()
                print(f"done gmin {system} seed {seed} hit={hit} evals={row['evals_to_hit']} wall={wall}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
