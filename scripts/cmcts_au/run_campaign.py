#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Sutton–Chen Au35, Au38 and Au55 against public PyPhysTree c-MCTS.

The shared budget is energy-and-gradient calls. One local minimisation
on seeds 2000–2011 costs the median below (Hyndman–Fan type 7). Banik
et al. report convergence in under 12,000 such evaluations. Doye and
Wales used 25,000 local minimisations. The shared budget is the larger
anchor, 25,000 minimisations priced in calls. A miss stores that budget
as its calls to first hit.

PyPhysTree commit d6eb071a7404550487654a28ec6181a60af44f05. The crystal
driver hyperparameters are the ones in examples/02_Crystal_Structure/codes/Crystal_search.py.
The box is the supplementary Au cell, 20 Å, divided by the Todd–Lynden-Bell
gold lattice parameter 4.08 Å. The search itself is their MCTSBatch.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "pyphystree"
SEEDS = list(range(2000, 2012))
SYSTEMS = ("au35", "au38", "au55")
ARMS = ("recommended", "for_cluster", "cmcts")
SIZES = {"au35": 35, "au38": 38, "au55": 55}
REFERENCES = {"au35": -9276.2927, "au38": -10117.2454, "au55": -14814.5225}
BH_MINIMISATIONS = 25000
PAPER_EVALUATIONS = 12000
CALLS_PER_MINIMISATION = {"au35": 1225.5, "au38": 1209.0, "au55": 1209.5}
PAPER_BUDGETS = {
    "au35": 30637500,
    "au38": 30225000,
    "au55": 30237500,
}
HIT_TOL = 1e-3
MAX_ITER = 1000
BOX_ANGSTROM = 20.0
AU_LATTICE_A = 4.08
PYPHYSTREE_COMMIT = (VENDOR / "UPSTREAM").read_text().strip()
ROW_FIELDS = [
    "system",
    "arm",
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
    "node_evals",
    "calls_per_minimisation",
    "paper_anchor_calls",
    "bh_anchor_calls",
]


def anneal_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, rows: list[dict[str, str]]) -> None:
    existing = read_rows(path)
    seen = {
        (row["system"], row["arm"], row["seed"], row["code_revision"]) for row in existing
    }
    fresh = [
        row
        for row in rows
        if (row["system"], row["arm"], row["seed"], row["code_revision"]) not in seen
    ]
    if not fresh:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        for row in fresh:
            writer.writerow({field: row.get(field, "") for field in ROW_FIELDS})


def anchor_fields(system: str) -> dict[str, str]:
    per = CALLS_PER_MINIMISATION[system]
    return {
        "calls_per_minimisation": f"{per:.1f}",
        "paper_anchor_calls": str(int(PAPER_EVALUATIONS * per)),
        "bh_anchor_calls": str(PAPER_BUDGETS[system]),
    }


def run_anneal(binary: Path, arm: str, system: str, seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [
            "taskset",
            "-c",
            "0",
            str(binary),
            "search",
            arm,
            str(SIZES[system]),
            str(seed),
            str(PAPER_BUDGETS[system]),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    text = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not text:
        raise SystemExit(proc.stderr[-800:] or proc.stdout[-800:])
    cols = text[-1].split(",")
    hit = cols[4] == "true"
    return {
        "system": cols[1],
        "arm": cols[0],
        "seed": cols[2],
        "budget": cols[3],
        "hit": "1" if hit else "0",
        "evals_to_hit": cols[5],
        "charged": cols[6],
        "wall_s": cols[7],
        "best": cols[8],
        "pot_s": cols[9],
        "outside_s": cols[10],
        "node_evals": "",
        **anchor_fields(system),
    }


class Potential:
    def __init__(self, binary: Path, n: int):
        self.n = n
        self.proc = subprocess.Popen(
            ["taskset", "-c", "0", str(binary), "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.pot_ns = 0

    def quench(self, values: list[float], max_iter: int, budget: int) -> tuple[float, list[float], int]:
        line = " ".join(f"{value:.16g}" for value in values)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(f"quench {self.n} {max_iter} {budget}\n{line}\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline()
        coords = self.proc.stdout.readline()
        if not header or not coords:
            raise SystemExit("the potential server closed")
        energy_s, calls_s, pot_s = header.split()
        self.pot_ns += int(pot_s)
        return float(energy_s), [float(value) for value in coords.split()], int(calls_s)

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        self.proc.wait(timeout=10)


class Objective:
    def __init__(self, potential: Potential, reference: float, budget: int):
        self.potential = potential
        self.reference = reference
        self.budget = budget
        self.best = math.inf
        self.calls = 0
        self.hit_calls: int | None = None
        self.node_evals = 0

    def __call__(self, values):
        self.node_evals += 1
        left = self.budget - self.calls
        if left <= 0:
            raise BudgetSpent()
        energy, coords, used = self.potential.quench(list(values), MAX_ITER, left)
        self.calls += used
        if math.isfinite(energy) and energy < self.best:
            self.best = energy
        if self.hit_calls is None and math.isfinite(energy) and abs(energy - self.reference) <= HIT_TOL:
            self.hit_calls = self.calls
            raise Found()
        if self.calls >= self.budget:
            raise BudgetSpent()
        return coords, float(energy)


class BudgetSpent(Exception):
    pass


class Found(Exception):
    pass


def run_cmcts(binary: Path, system: str, seed: int) -> dict[str, str]:
    sys.path.insert(0, str(VENDOR))
    import logging

    import numpy as np
    from lgtree.utils import _latin

    n = SIZES[system]
    budget = PAPER_BUDGETS[system]
    reference = REFERENCES[system]
    side = BOX_ANGSTROM / AU_LATTICE_A
    dim = 3 * n
    np.random.seed(seed)
    random.seed(seed)
    lb = np.zeros(dim)
    ub = np.full(dim, side)
    potential = Potential(binary, n)
    objective = Objective(potential, reference, budget)
    logger = logging.getLogger("cmcts")
    logger.addHandler(logging.NullHandler())
    log_path = ROOT / "bench" / "cmcts-au" / "logs" / f"{system}-{seed}-cmcts.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        ntrees = 10
        top_k = 5
        head_points = _latin(ntrees, lb.tolist(), ub.tolist())
        a_values = np.linspace(0.005, 0.1, ntrees)
        config = {
            i: {
                "a": float(a_values[i]),
                "b": 0.5,
                "root_data": head_points[i],
                "head_data": None,
                "lower_bounds": lb,
                "upper_bounds": ub,
            }
            for i in range(ntrees)
        }
        # Hyperparameters copied from the released crystal-structure driver.
        with log_path.open("a") as log, contextlib.redirect_stdout(log):
            _drive_cmcts(objective, logger, config, lb, ub, top_k, reference)
    except (Found, BudgetSpent):
        pass
    finally:
        potential.close()
    wall = time.perf_counter() - started
    pot_s = potential.pot_ns / 1.0e9
    outside = max(0.0, wall - pot_s)
    hit = objective.hit_calls is not None
    best = objective.best if math.isfinite(objective.best) else math.inf
    return {
        "system": system,
        "arm": "cmcts",
        "seed": str(seed),
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(objective.hit_calls if hit else budget),
        "charged": str(objective.calls),
        "wall_s": f"{wall:.6f}",
        "best": f"{best:.8f}" if math.isfinite(best) else "inf",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{outside:.6f}",
        "node_evals": str(objective.node_evals),
        **anchor_fields(system),
    }


def _drive_cmcts(objective, logger, config, lb, ub, top_k, reference) -> None:
    from lgtree.lgtree import MCTSBatch, run_global_tree_batch, run_local_tree_batch

    del lb, ub
    global_batch = MCTSBatch(
        objfunc=objective,
        config_dict=config,
        logger=logger,
        nplayouts=5,
        explore_constant_max=1e10,
        explore_constant_min=0.5,
        max_depth=12,
        niterations=20,
        sampling_mode="hypersphere",
    )
    top_stats, _ = run_global_tree_batch(global_batch, top_k=top_k)
    if not top_stats:
        return
    local_config = {
        idx: {
            "a": res["a"],
            "b": res["b"],
            "root_data": res["parameter"],
            "head_data": None,
            "lower_bounds": res["lb"],
            "upper_bounds": res["ub"],
        }
        for idx, res in top_stats.items()
    }
    local_batch = MCTSBatch(
        objfunc=objective,
        config_dict=local_config,
        logger=logger,
        nplayouts=5,
        max_depth=100,
        niterations=10,
        explore_constant_max=0.0,
        explore_constant_min=0.0,
    )
    run_local_tree_batch(
        local_batch=local_batch,
        top_stats=top_stats,
        objfunc=objective,
        logger=logger,
        n_iterations=10**12,
        total_evals_sofar=0,
        target=reference,
    )


def run_one(binary: Path, system: str, seed: int, arm: str) -> dict[str, str]:
    if arm == "cmcts":
        return run_cmcts(binary, system, seed)
    return run_anneal(binary, arm, system, seed)


def campaign(binary: Path, seeds_path: Path, only: tuple[str, int, str] | None) -> None:
    head = anneal_revision()
    host = os.uname().nodename
    jobs = []
    if only is None:
        for system in SYSTEMS:
            for seed in SEEDS:
                for arm in ARMS:
                    jobs.append((system, seed, arm))
    else:
        jobs.append(only)
    for system, seed, arm in jobs:
        print(f"start {system} {arm} {seed}", file=sys.stderr, flush=True)
        row = run_one(binary, system, seed, arm)
        row["code_revision"] = head if arm != "cmcts" else PYPHYSTREE_COMMIT
        row["host"] = host
        append_rows(seeds_path, [row])
        print(
            f"done {system} {arm} {seed} hit={row['hit']} calls={row['evals_to_hit']} "
            f"wall={row['wall_s']} pot={row['pot_s']} outside={row['outside_s']} best={row['best']}",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--seeds-out", type=Path, default=ROOT / "bench" / "cmcts-au" / "seeds.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "bench" / "cmcts-au" / "summary.csv")
    parser.add_argument("--only", nargs=3, metavar=("SYSTEM", "SEED", "ARM"))
    args = parser.parse_args()
    only = None
    if args.only:
        system, seed_s, arm = args.only
        if system not in SYSTEMS or arm not in ARMS:
            raise SystemExit(f"unknown job {args.only}")
        only = (system, int(seed_s), arm)
    campaign(args.binary, args.seeds_out, only)
    from summarize import write_summary

    write_summary(args.seeds_out, args.summary)


if __name__ == "__main__":
    main()
