#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One seed of the ASE panel.

Usage: run_one.py SYSTEM ARM SEED

`anneal` is Config::for_cluster. `recommended` is Config::recommended.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults.
Lennard-Jones is the untruncated pair. Au55 is Gupta::copper_gold through
asg_eval serve, the same potential the anneal arms evaluate.
A hit is a quenched energy within 0.001 of the reference. A miss is
censored at the paper budget. One calculator call is one energy and one
gradient. An ASE seed, and a recommended seed, stop at the wall cap stored
on the ASE rows. The row records the calls, the wall, the time inside and
outside the potential, the cap, code_revision and host.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from evaluator import evaluator_spec

sys.path.insert(0, str(ROOT / "scripts" / "ase_gofee_h2h"))
from potential import (  # noqa: E402
    AnnealCalculator,
    AnnealServer,
    BudgetReached,
    CapReached,
    HitReached,
    LjCalculator,
    lj_value_and_gradient,
    warmup,
)

PAPER_BUDGETS = {
    38: 400_000,
    55: 1_000_000,
    98: 4_000_000,
    104: 4_000_000,
    150: 4_000_000,
}
SYSTEMS = {
    "lj38": {"n": 38, "budget": 400_000, "reference": -173.928427, "kind": "lj"},
    "lj55": {"n": 55, "budget": 1_000_000, "reference": -279.248470, "kind": "lj"},
    "lj98": {"n": 98, "budget": 4_000_000, "reference": -543.665361, "kind": "lj"},
    "lj104": {"n": 104, "budget": 4_000_000, "reference": -582.086642, "kind": "lj"},
    "lj150": {"n": 150, "budget": 4_000_000, "reference": -893.310258, "kind": "lj"},
    "au55": {
        "n": 55,
        "budget": 1_000_000,
        "reference": -191.03085280,
        "kind": "gupta",
        "symbol": "Au",
    },
}
HIT_TOL = 0.001
MIN_SEP = 0.85
DENSITY = 0.7
SEEDS = tuple(range(2000, 2012))
FMAX = {"ase_bh": 0.1, "ase_mh": 0.05}
ARMS = ("anneal", "recommended", "ase_bh", "ase_mh")
ASE_VERSION = "3.29.0"


def hostname() -> str:
    return Path("/proc/sys/kernel/hostname").read_text().splitlines()[0].strip()


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().splitlines()[0].strip()


def place(n: int, seed: int, length: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    radius = length * (3.0 * n / (4.0 * np.pi * DENSITY)) ** (1.0 / 3.0)
    gap = MIN_SEP * length
    points: list[np.ndarray] = []
    draws = 0
    while len(points) < n:
        draws += 1
        if draws > 1_000_000:
            raise RuntimeError("could not place the cluster")
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        point = direction * (radius * (float(rng.random()) ** (1.0 / 3.0)) / norm)
        if all(float(np.linalg.norm(point - other)) >= gap for other in points):
            points.append(point)
    return np.vstack(points)


def run_ase(system: str, arm: str, seed: int, work: Path, cap_s: float, budget: int) -> dict:
    import ase
    from ase import Atoms
    from ase.optimize.basin import BasinHopping
    from ase.optimize.minimahopping import MinimaHopping

    if ase.__version__ != ASE_VERSION:
        raise RuntimeError(f"ASE {ASE_VERSION} required, found {ase.__version__}")
    spec = SYSTEMS[system]
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    symbol = spec.get("symbol", "Ar")
    binary = os.environ.get("ASG_EVAL", "/work/target-panel/release/examples/asg_eval")
    if spec["kind"] == "gupta":
        n_atoms, paper_budget, reference, length = evaluator_spec(binary, system)
        if (n_atoms, paper_budget) != (spec["n"], spec["budget"]) or reference != spec["reference"]:
            raise RuntimeError("asg_eval spec differs from the panel")
    else:
        length = 1.0
    atoms = Atoms(f"{symbol}{spec['n']}", positions=place(spec["n"], seed, length), pbc=False)
    started = time.perf_counter()
    server = None
    if spec["kind"] == "lj":
        warmup()
        calc = LjCalculator(budget, spec["reference"], HIT_TOL, FMAX[arm], deadline=started + cap_s)
    else:
        server = AnnealServer(binary, system)
        calc = AnnealCalculator(
            server, budget, spec["reference"], HIT_TOL, FMAX[arm], deadline=started + cap_s
        )
    atoms.calc = calc
    np.random.seed(seed)
    stop = "budget"
    try:
        if arm == "ase_bh":
            BasinHopping(
                atoms,
                logfile=None,
                optimizer_logfile=None,
                trajectory=None,
                local_minima_trajectory=None,
            ).run(10**9)
        else:
            MinimaHopping(atoms)(totalsteps=10**9)
    except CapReached:
        stop = "wall"
    except (BudgetReached, HitReached):
        stop = "budget"
    finally:
        if server is not None:
            server.close()
    wall = time.perf_counter() - started
    best = calc.best_quenched if calc.best_quenched < 1.0e299 else calc.best
    hit = calc.hit_calls is not None
    return {
        "hit": int(hit),
        "evals": calc.hit_calls if hit else "",
        "charged": calc.calls,
        "best": float(best),
        "wall": wall,
        "inside": calc.inside_s,
        "outside": max(0.0, wall - calc.inside_s),
        "stop": "hit" if hit else stop,
        "cap_s": cap_s,
    }


def run_anneal(system: str, seed: int, budget: int, preset: str, cap_s: str) -> dict:
    binary = os.environ.get("ASE_PANEL", "/work/target-panel/release/examples/ase_panel")
    command = [binary, system, str(budget), str(seed), preset]
    if cap_s:
        command.append(cap_s)
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    fields = {}
    for line in proc.stdout.splitlines():
        if line.startswith("system="):
            fields = dict(part.split("=", 1) for part in line.split())
    if fields.get("system") != system or fields.get("seed") != str(seed) or fields.get("arm") != preset:
        raise RuntimeError("anneal stdout has no row")
    hit = fields["hit"] == "1"
    stop = fields.get("stop") or ("hit" if hit else "budget")
    return {
        "hit": int(hit),
        "evals": "" if fields["first_calls"] == "-" else fields["first_calls"],
        "charged": int(fields["charged"]),
        "best": float(fields["best"]),
        "wall": float(fields["wall_s"]),
        "inside": float(fields["inside_s"]),
        "outside": float(fields["outside_s"]),
        "stop": stop,
        "cap_s": cap_s,
    }


def check_lj() -> None:
    r = 2.0 ** (1.0 / 6.0)
    x = np.zeros(6)
    x[3] = r
    energy, grad = lj_value_and_gradient(x)
    if abs(float(energy) + 1.0) > 1.0e-8 or float(np.linalg.norm(grad)) > 1.0e-8:
        raise RuntimeError(f"LJ pair drifted {energy}")


def main() -> int:
    if hostname() != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    if len(sys.argv) != 4:
        print("usage: run_one.py SYSTEM ARM SEED", file=sys.stderr)
        return 2
    system, arm, seed_text = sys.argv[1:]
    if system not in SYSTEMS or arm not in ARMS:
        print("unknown system or arm", file=sys.stderr)
        return 2
    seed = int(seed_text)
    if seed not in SEEDS:
        print("seed is outside 2000-2011", file=sys.stderr)
        return 2
    if SYSTEMS[system]["kind"] == "lj":
        check_lj()
    revision = os.environ.get("ASE_PANEL_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        print("ASE_PANEL_REVISION must be the full revision used to build the binaries", file=sys.stderr)
        return 2
    spec = SYSTEMS[system]
    budget = int(os.environ.get("ASE_PANEL_BUDGET", spec["budget"]))
    if not 0 < budget <= spec["budget"]:
        print(f"ASE_PANEL_BUDGET must be in 1-{spec['budget']}", file=sys.stderr)
        return 2
    if arm in ("anneal", "recommended"):
        cap_text = ""
        if arm == "recommended":
            cap_text = os.environ.get("ASE_PANEL_WALL_CAP", "")
            if not cap_text:
                print("ASE_PANEL_WALL_CAP is unset", file=sys.stderr)
                return 2
        preset = "for_cluster" if arm == "anneal" else "recommended"
        row = run_anneal(system, seed, budget, preset, cap_text)
    else:
        cap_text = os.environ.get("ASE_PANEL_WALL_CAP", "")
        if not cap_text:
            print("ASE_PANEL_WALL_CAP is unset", file=sys.stderr)
            return 2
        row = run_ase(
            system,
            arm,
            seed,
            Path(os.environ.get("ASE_PANEL_WORK", "/work/ase-panel/runs")) / f"{system}-{arm}-{seed}",
            float(cap_text),
            budget,
        )
    cap = "" if row["cap_s"] == "" else f"{float(row['cap_s']):.6f}"
    print(
        ",".join(
            [
                system,
                arm,
                str(seed),
                str(budget),
                str(row["hit"]),
                "" if row["evals"] == "" else str(row["evals"]),
                str(row["charged"]),
                f"{row['best']:.8f}",
                f"{row['wall']:.6f}",
                f"{row['inside']:.6f}",
                f"{row['outside']:.6f}",
                revision,
                hostname(),
                row["stop"],
                cap,
            ]
        ),
        flush=True,
    )
    print(f"boot_id={boot_id()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
