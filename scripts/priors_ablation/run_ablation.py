#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial anneal arms with and without structural priors.

Three arms run the same binary at one commit. `default` leaves
ANNEAL_STRUCTURAL_PRIORS unset, `priors_off` sets it to 0, and
`general_only` sets it to general_only. Each (system, seed, arm) is one
`taskset -c 0` process. Within a system the (seed, arm) pairs are
shuffled. The binaries, budgets, start files and environment are those of
the head-to-head harnesses: h2h_gmin_proof for Lennard-Jones, h2h_morse,
h2h_metal (Pt-Cu with H2H_BASIN_STRIDE=1, as run_ptcu.py does), h2h_sc,
and h2h_seed for TIP4P. A hit is a quenched energy within 0.001 of the
reference. Pt-Cu has no Cambridge row, so its reference is the lowest
energy seen in this session or the recorded Pt-Cu session. Medians are
Hyndman-Fan type 7. This process refuses to run unless the UTS hostname
is bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = {"default": None, "priors_off": "0", "general_only": "general_only"}
SCHEDULE_SEED = 20261113
HIT_TOL = 0.001

# (name, binary, binary system, budget, TIP4P stop reference, extra env).
# The other binaries score the hit against their own tables.
SYSTEMS = (
    ("lj38", "h2h_gmin_proof", "lj38", 400_000, None, {}),
    ("lj75", "h2h_gmin_proof", "lj75", 4_000_000, None, {}),
    ("lj98", "h2h_gmin_proof", "lj98", 4_000_000, None, {}),
    ("lj104", "h2h_gmin_proof", "lj104", 4_000_000, None, {}),
    ("lj150", "h2h_gmin_proof", "lj150", 4_000_000, None, {}),
    ("m38r6", "h2h_morse", "m38r6", 400_000, None, {}),
    ("m55r6", "h2h_morse", "m55r6", 1_000_000, None, {}),
    ("m80r6", "h2h_morse", "m80r6", 4_000_000, None, {}),
    ("m38r14", "h2h_morse", "m38r14", 400_000, None, {}),
    ("m55r14", "h2h_morse", "m55r14", 1_000_000, None, {}),
    ("m80r14", "h2h_morse", "m80r14", 4_000_000, None, {}),
    ("sc126-38", "h2h_sc", "sc126-38", 400_000, None, {}),
    ("sc126-55", "h2h_sc", "sc126-55", 1_000_000, None, {}),
    ("au38", "h2h_metal", "au38", 400_000, None, {}),
    ("au55", "h2h_metal", "au55", 1_000_000, None, {}),
    ("cu38", "h2h_metal", "cu38", 400_000, None, {"H2H_BASIN_STRIDE": "1"}),
    ("cu6au32", "h2h_metal", "cu6au32", 400_000, None, {}),
    ("cu27au28", "h2h_metal", "cu27au28", 1_000_000, None, {}),
    ("pt12cu11", "h2h_metal", "pt12cu11", 200_000, None, {"H2H_BASIN_STRIDE": "1"}),
    ("pt6cu17", "h2h_metal", "pt6cu17", 200_000, None, {"H2H_BASIN_STRIDE": "1"}),
    ("tip4p4", "h2h_seed", "4", 2_000_000, -116.5904186160, {}),
    ("tip4p8", "h2h_seed", "8", 2_000_000, -305.5183221676, {}),
    ("tip4p15", "h2h_seed", "15", 2_000_000, -628.3722525077, {}),
)
SESSION_BEST = {"cu38", "pt12cu11", "pt6cu17"}

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "hit_binary",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
]


def parse_row(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            return {
                "hit_binary": "true" if fields["hit"] == "1" else "false",
                "evals_to_hit": fields["evals_to_hit"],
                "charged": fields["charged"],
                "wall_s": fields["wall_s"],
                "best": fields["best"],
                "pot_s": fields["pot_s"],
                "outside_s": fields["outside_s"],
            }
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("system,code,seed,") and i + 1 < len(lines):
            row = dict(zip(line.split(","), lines[i + 1].split(",")))
            wall = float(row["wall_s"])
            pot = float(row["pot_s"])
            outside = row.get("outside_s") or f"{max(wall - pot, 0.0):.6f}"
            return {
                "hit_binary": row["hit"],
                "evals_to_hit": row["evals_to_hit"],
                "charged": row["charged"],
                "wall_s": row["wall_s"],
                "best": row["best"],
                "pot_s": row["pot_s"],
                "outside_s": outside,
            }
    raise ValueError(f"no anneal row in {text[-800:]!r}")


def run_one(bins: Path, runs: Path, spec, seed: int, arm: str) -> dict[str, str]:
    name, binary, target, budget, reference, extra = spec
    env = os.environ.copy()
    env.pop("ANNEAL_STRUCTURAL_PRIORS", None)
    if ARMS[arm] is not None:
        env["ANNEAL_STRUCTURAL_PRIORS"] = ARMS[arm]
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "OMP_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    env.update(extra)
    work = runs / name / f"{arm}-s{seed}"
    work.mkdir(parents=True, exist_ok=True)
    exe = str(bins / binary)
    if binary == "h2h_seed":
        stop = reference + HIT_TOL
        cmd = [exe, "tip4p", target, "0", str(budget), str(seed), "recommended", repr(stop), str(work / "out")]
    else:
        coords = work / "start.xyz"
        if binary == "h2h_morse":
            subprocess.run([exe, "coords", target, str(seed), str(coords)], check=True, env=env)
        cmd = [exe, "run", target, str(seed), str(coords)]
    log = work / "log.txt"
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", *cmd],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"{name} {arm} seed {seed} exited {proc.returncode}\n{text[-2000:]}")
    row = parse_row(text)
    row.update({"system": name, "arm": arm, "seed": str(seed), "budget": str(budget)})
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--systems", nargs="*")
    parser.add_argument("--arms", nargs="*", choices=tuple(ARMS))
    args = parser.parse_args()
    if socket.gethostname() != "bench":
        print(f"hostname is {socket.gethostname()}, not bench", file=sys.stderr)
        return 2
    chosen = [s for s in SYSTEMS if not args.systems or s[0] in args.systems]
    arms = tuple(a for a in ARMS if not args.arms or a in args.arms)
    done = set()
    if args.rows.is_file():
        with args.rows.open() as handle:
            done = {(r["system"], r["arm"], r["seed"]) for r in csv.DictReader(handle)}
    new_file = not args.rows.is_file()
    rng = random.Random(SCHEDULE_SEED)
    with args.rows.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        if new_file:
            writer.writeheader()
        for spec in chosen:
            jobs = [(seed, arm) for seed in SEEDS for arm in arms]
            rng.shuffle(jobs)
            for seed, arm in jobs:
                if (spec[0], arm, str(seed)) in done:
                    continue
                row = run_one(args.bins, args.runs, spec, seed, arm)
                writer.writerow({k: row[k] for k in ROW_FIELDS})
                handle.flush()
                print(
                    f"done {spec[0]} {arm} seed {seed} hit={row['hit_binary']} "
                    f"evals={row['evals_to_hit']} best={row['best']} wall={row['wall_s']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
