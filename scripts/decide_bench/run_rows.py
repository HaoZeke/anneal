#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the walker-allocation and move-bandit panel and append raw rows.

One row per (variant, system, seed). Seeds 2000-2011, the paper budget
of each system, one core, hostname bench. A finished row is never
rewritten: a rerun skips every (variant, system, seed) already in the
table and appends the rest. ``code_revision`` is the git revision the
binaries were built from and ``host`` is the UTS hostname.

A variant is the value of ``H2H_DECIDE`` handed to every binary;
``baseline`` leaves it unset. A miss is stored as the full budget.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(2000, 2012))
ROWS = Path("bench/decide/rows.csv")

# (system, binary, budget, reference). The budgets are the paper budgets
# of bench/lj-n12-500aaf65, bench/au55-scale, bench/morse-panel,
# bench/metal-cluster-h2h/sc-n12 and bench/tip4p-h2h.
PANEL = (
    ("lj38", "h2h_gmin_proof", 400_000, -173.928427),
    ("lj55", "h2h_gmin_proof", 1_000_000, -279.248470),
    ("au55", "h2h_metal", 1_000_000, -191.03085280),
    ("tip4p15", "h2h_seed", 2_000_000, -628.3722525077),
    ("tip4p20", "h2h_seed", 2_000_000, -872.9887544952),
    ("m80r3", "h2h_morse", 4_000_000, -690.577890),
    ("m80r6", "h2h_morse", 4_000_000, -378.333471),
    ("sc108-55", "h2h_sc", 1_000_000, -14814.5225),
    ("sc96-80", "h2h_sc", 1_000_000, -19560.8966),
    ("sc108-80", "h2h_sc", 1_000_000, -21819.9952),
    ("sc126-80", "h2h_sc", 1_000_000, -78414.6271),
    ("lj75", "h2h_gmin_proof", 4_000_000, -397.492331),
    ("lj76", "h2h_gmin_proof", 4_000_000, -402.894866),
    ("lj77", "h2h_gmin_proof", 4_000_000, -409.083517),
    ("lj98", "h2h_gmin_proof", 4_000_000, -543.665361),
    ("lj102", "h2h_gmin_proof", 4_000_000, -569.363652),
    ("lj103", "h2h_gmin_proof", 4_000_000, -575.766131),
    ("lj104", "h2h_gmin_proof", 4_000_000, -582.086642),
    ("lj150", "h2h_gmin_proof", 4_000_000, -893.310258),
)
TIP4P_TOL = 0.001

FIELDS = (
    "variant",
    "system",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
)

# Spec handed to H2H_DECIDE for each named variant.
VARIANTS = {
    "baseline": None,
    "hyperband": "hyperband",
    "thompson": "thompson",
    "ducb": "ducb",
    "adapt": "adapt",
}


def csv_row(text: str, system: str) -> dict[str, str]:
    """The result line of the CSV-printing binaries, keyed by its header."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(system + ",") and index > 0:
            for header in reversed(lines[:index]):
                if header.startswith("system,"):
                    return dict(zip(header.split(","), line.split(",")))
    raise RuntimeError(f"no result row for {system}\n{text[-1500:]}")


def kv_row(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            return dict(part.split("=", 1) for part in line.split()[1:])
    raise RuntimeError(f"no h2h_row line\n{text[-1500:]}")


def command(bindir: Path, system: str, binary: str, seed: int, budget: int,
            reference: float, scratch: Path) -> list[str]:
    path = str(bindir / binary)
    if binary == "h2h_seed":
        n = system.removeprefix("tip4p")
        stop = repr(reference + TIP4P_TOL)
        prefix = str(scratch / f"{system}-s{seed}")
        return [path, "tip4p", n, "0", str(budget), str(seed), "recommended", stop, prefix]
    return [path, "run", system, str(seed)]


def run_one(bindir: Path, system: str, binary: str, seed: int, budget: int,
            reference: float, spec: str | None, scratch: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS"):
        env[key] = "1"
    env.pop("H2H_DECIDE", None)
    env.pop("H2H_BUDGET", None)
    if spec is not None:
        env["H2H_DECIDE"] = spec
    proc = subprocess.run(
        ["taskset", "-c", "0", *command(bindir, system, binary, seed, budget,
                                        reference, scratch)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=scratch,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{system} seed {seed} exited {proc.returncode}\n{proc.stderr[-1500:]}"
        )
    parsed = kv_row(proc.stdout) if binary == "h2h_seed" else csv_row(proc.stdout, system)
    if int(parsed["budget"]) != budget:
        raise RuntimeError(f"{system} ran at {parsed['budget']}, not {budget}")
    wall = float(parsed["wall_s"])
    pot = float(parsed["pot_s"])
    outside = float(parsed["outside_s"]) if "outside_s" in parsed else wall - pot
    hit = parsed["hit"] in ("1", "true")
    return {
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": parsed["evals_to_hit"] if hit else str(budget),
        "charged": parsed["charged"],
        "wall_s": f"{wall:.6f}",
        "best": f"{float(parsed['best']):.8f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
    }


def load_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    return {(r["variant"], r["system"], r["seed"]) for r in csv.DictReader(path.open())}


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({key: row[key] for key in FIELDS})


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", choices=sorted(VARIANTS))
    parser.add_argument("--bindir", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--system", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    args = parser.parse_args(argv)
    host = os.uname().nodename
    if host != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    revision = args.head_file.read_text().strip()
    path = args.root.resolve() / ROWS
    args.scratch.mkdir(parents=True, exist_ok=True)
    done = load_done(path)
    seeds = tuple(args.seed) or SEEDS
    for variant in args.variants:
        for system, binary, budget, reference in PANEL:
            if args.system and system not in args.system:
                continue
            for seed in seeds:
                key = (variant, system, str(seed))
                if key in done:
                    continue
                print(f"run {variant} {system} {seed}", flush=True)
                row = run_one(args.bindir, system, binary, seed, budget, reference,
                              VARIANTS[variant], args.scratch)
                row.update(variant=variant, system=system, seed=str(seed),
                           code_revision=revision, host=host)
                append_row(path, row)
                done.add(key)
                print(f"done {variant} {system} {seed} hit={row['hit']} "
                      f"calls={row['evals_to_hit']} wall={row['wall_s']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
