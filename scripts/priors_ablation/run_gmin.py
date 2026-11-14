#!/usr/bin/env python3
# Copyright (c) anneal developers
"""GMIN on every ablation system, on the anneal arms' seeds.

Each call is the head-to-head harness's own run_gmin: scripts/gmin_h2h
for Lennard-Jones, scripts/morse_h2h for Morse, scripts/metal_h2h's
Darby driver for Gupta Cu-Au (BGUPTAT, the recorded 90000000 steps and
swaps on a true alloy), and scripts/tip4p_h2h for the waters. Starts are
the anneal binaries' `coords` output for the same seed, and for TIP4P the
`.start.gmin` the ablation's default arm wrote. --gmin is the Lennard-Jones,
Morse and TIP4P build those harnesses use; --gmin-gupta is the Darby
build. Within a system the seeds are shuffled. Every row carries
--code-revision, the anneal commit whose harness and starts it ran with,
and host, the first twelve hex digits of /etc/machine-id. A (system, seed)
already recorded at that revision on this host is not run again. This
process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import socket
import subprocess
import sys
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
SCHEDULE_SEED = 20261113
HERE = Path(__file__).resolve().parent
FIELDS = [
    "system", "arm", "seed", "budget", "hit", "evals_to_hit", "wall_s", "best", "pot_s", "outside_s",
    "code_revision", "host",
]

LJ = {
    "lj38": (38, 400_000, -173.928427),
    "lj75": (75, 4_000_000, -397.492331),
    "lj98": (98, 4_000_000, -543.665361),
    "lj104": (104, 4_000_000, -582.086642),
    "lj150": (150, 4_000_000, -893.310258),
}
MORSE = ("m55r6", "m80r6", "m55r14", "m80r14")
GUPTA = ("au55", "cu6au32", "cu27au28")
TIP4P = {"tip4p4": -116.5904186160, "tip4p8": -305.5183221676, "tip4p15": -628.3722525077}
SYSTEMS = (*LJ, *MORSE, *GUPTA, *TIP4P)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coords_of(binary: Path, system: str, seed: int, path: Path) -> Path:
    if not path.is_file():
        subprocess.run([str(binary), "coords", system, str(seed), str(path)], check=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--gmin-gupta", type=Path, required=True)
    parser.add_argument("--bins", type=Path, required=True)
    parser.add_argument("--ablation-runs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--systems", nargs="*", choices=SYSTEMS)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    if socket.gethostname() != "bench":
        print(f"hostname is {socket.gethostname()}, not bench", file=sys.stderr)
        return 2
    host = Path("/etc/machine-id").read_text().strip()[:12]
    scripts = args.root / "scripts"
    lj = load("lj_h2h", scripts / "gmin_h2h" / "run_campaign.py")
    morse = load("morse_h2h", scripts / "morse_h2h" / "run_campaign.py")
    darby = load("darby_h2h", scripts / "metal_h2h" / "run_darby.py")
    tip4p = load("tip4p_h2h", scripts / "tip4p_h2h" / "run_campaign.py")
    tip4p.REFERENCES.update({int(name.removeprefix("tip4p")): ref for name, ref in TIP4P.items()})
    morse_spec = {row[0]: row for row in morse.SYSTEMS}
    examples = args.root / "examples"
    done = set()
    if args.rows.is_file():
        with args.rows.open() as handle:
            done = {
                (r["system"], r["seed"], r["code_revision"], r["host"])
                for r in csv.DictReader(handle)
            }
    new_file = not args.rows.is_file()
    rng = random.Random(SCHEDULE_SEED)
    with args.rows.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        for system in SYSTEMS:
            seeds = list(SEEDS)
            rng.shuffle(seeds)
            if args.systems and system not in args.systems:
                continue
            for seed in seeds:
                if (system, str(seed), args.code_revision, host) in done:
                    continue
                work = args.work / system / f"s{seed}"
                work.mkdir(parents=True, exist_ok=True)
                if system in LJ:
                    n, budget, reference = LJ[system]
                    coords = coords_of(args.bins / "h2h_gmin_proof", system, seed, work / "start.xyz")
                    row = lj.run_gmin(
                        args.gmin, examples / "h2h_gmin_lj.data", n, seed, budget,
                        reference, coords, work / "run", work / "log.txt", work / "row.txt",
                    )
                elif system in MORSE:
                    _name, _n, rho, budget, reference = morse_spec[system]
                    coords = coords_of(args.bins / "h2h_morse", system, seed, work / "start.xyz")
                    row = morse.run_gmin(
                        args.gmin, examples / "h2h_gmin_morse.data", system, rho, seed, budget,
                        reference, coords, work / "run", work / "log.txt", work / "row.txt",
                    )
                elif system in GUPTA:
                    budget = darby.PAPER_BUDGETS[system]
                    coords = coords_of(args.bins / "h2h_metal", system, seed, work / "start.xyz")
                    row = darby.run_gmin(
                        args.gmin_gupta, system, seed, budget, darby.REFERENCES[system], coords,
                        work / "run", work / "log.txt", steps=90_000_000,
                        swaps=0 < darby.COPPER_COUNT[system] < darby.N_ATOMS[system],
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
                        "code_revision": args.code_revision,
                        "host": host,
                    }
                )
                handle.flush()
                print(f"done gmin {system} seed {seed} hit={hit} evals={row['evals_to_hit']} wall={wall}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
