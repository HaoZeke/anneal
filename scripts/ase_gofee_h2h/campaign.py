#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial head-to-head on LJ38 and LJ75.

Arms, one shared seed list, one force-call budget per size:

* paper: ``Config::recommended`` via ``h2h_gmin_proof``
* ase_bh: ``ase.optimize.basin.BasinHopping`` at the 3.29.0 constructor defaults
  (FIRE, dr=0.1, fmax=0.1, temperature=100 kB)
* ase_mh: ``ase.optimize.minimahopping.MinimaHopping`` at its constructor defaults
* agox_gofee: the GOFEE script on the AGOX algorithm page (kappa=2, the
  documented kernel, KMeans sample 10, the candidate schedule, a five-step
  surrogate relaxation from iteration 8, true-potential fmax 0.05)

The page's true-potential step cap is 1, and the page says that cap is too
low for a real search. This run keeps fmax at 0.05 and lets that relaxation
run until the force-call budget, so a charged candidate can reach the
threshold. The collector and
the surrogate relaxation run in this process. The GPR fit stays on this
process (``use_ray`` false) and does not restart kernel hyperparameters, and
it trains on at most 512 stored structures, which is what lets a force-call
budget be spent. Generator bond windows are set so a reduced Lennard-Jones
nearest neighbour is allowed; the page's windows are for the EMT example.

A hit is a quenched energy within 0.001 of the Cambridge minimum. Misses
stop at the paper force-call budget. Jobs run one at a time in an order
shuffled with seed 20261111.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from potential import BudgetReached, HitReached, LjCalculator, lj_value_and_gradient, warmup

HERE = Path(__file__).resolve().parent
PAPER_BUDGETS = {"lj38": 400_000, "lj75": 4_000_000}
REFERENCES = {"lj38": -173.928427, "lj75": -397.492331}
HIT_TOL = 0.001
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("paper", "ase_bh", "ase_mh", "agox_gofee")
ORDER_SEED = 20261111
FMAX = {"ase_bh": 0.1, "ase_mh": 0.05, "agox_gofee": 0.05}
ASE_VERSION = "3.29.0"
AGOX_VERSION = "3.11.1"


def type7(sorted_values: list[float], p: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    h = 1.0 + (n - 1) * p
    lower = math.floor(h)
    frac = h - lower
    index = int(lower) - 1
    if frac == 0.0:
        return float(sorted_values[index])
    return float(sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac)


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return type7(ordered, 0.5), type7(ordered, 0.75) - type7(ordered, 0.25)


def check_type7() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12:
        raise RuntimeError("type 7 quartile drifted")
    if abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise RuntimeError("type 7 median drifted")
    median, iqr = median_iqr(sample)
    if abs(iqr - 2.5) > 1e-12 or abs(median - 3.5) > 1e-12:
        raise RuntimeError("iqr drifted")


def schedule(comparison: str | None) -> list[tuple[str, int, str]]:
    arms = ("paper", comparison) if comparison else ARMS
    jobs = [(system, seed, arm) for system in PAPER_BUDGETS for seed in SEEDS for arm in arms]
    random.Random(ORDER_SEED).shuffle(jobs)
    return jobs


def require_versions() -> None:
    import importlib.metadata as metadata

    ase_version = metadata.version("ase")
    agox_version = metadata.version("agox")
    if ase_version != ASE_VERSION or agox_version != AGOX_VERSION:
        raise SystemExit(f"expected ase {ASE_VERSION} and agox {AGOX_VERSION}, found {ase_version} and {agox_version}")


def install_constraint_shim() -> None:
    import ase.constraints as constraints
    from ase.constraints.constraint import IndexedConstraint, slice2enlist

    if not hasattr(constraints, "IndexedConstraint"):
        constraints.IndexedConstraint = IndexedConstraint
    if not hasattr(constraints, "slice2enlist"):
        constraints.slice2enlist = slice2enlist


def read_coords(path: Path, n: int):
    import numpy as np

    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if len(rows) != n:
        raise RuntimeError(f"{path} has {len(rows)} atoms, expected {n}")
    return np.asarray(rows, dtype=np.float64)


def ensure_start(binary: Path, repo: Path, system: str, seed: int, path: Path) -> None:
    n = int(system[2:])
    if path.is_file():
        try:
            read_coords(path, n)
            return
        except RuntimeError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(binary), "coords", system, str(seed), str(path)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    read_coords(path, n)


def check_potential(binary: Path, repo: Path, start: Path) -> None:
    proc = subprocess.run(
        [str(binary), "energy", "lj38", str(start)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    rust = float(proc.stdout.strip().splitlines()[-1])
    coords = read_coords(start, 38).ravel()
    python, _ = lj_value_and_gradient(coords)
    if abs(float(python) - rust) > 1e-8:
        raise RuntimeError(f"LJ mismatch rust {rust} python {python}")


def run_paper(binary: Path, repo: Path, system: str, seed: int, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    coords = work / "start.txt"
    proc = subprocess.run(
        [str(binary), "run", system, str(seed), str(coords)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    (work / "stdout.txt").write_text(proc.stdout)
    (work / "stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"paper {system} {seed} failed\n{proc.stderr[-2000:]}")
    row = {}
    for line in proc.stdout.splitlines():
        if line.startswith(f"{system},anneal,"):
            fields = line.split(",")
            header = proc.stdout.splitlines()[0].split(",")
            row = dict(zip(header, fields))
    if not row:
        raise RuntimeError(f"paper {system} {seed} produced no row")
    inside = None
    for line in proc.stderr.splitlines():
        if line.startswith("potential_inside_s="):
            inside = float(line.split("=", 1)[1])
    if inside is None:
        raise RuntimeError("paper run did not report potential time")
    hit = row["hit"] in {"1", "true"}
    wall = float(row["wall_s"])
    return {
        "hit": int(hit),
        "calls_to_hit": int(row["evals_to_hit"]) if hit else "",
        "charged": int(row["charged"]),
        "wall_s": wall,
        "inside_s": inside,
        "outside_s": max(0.0, wall - inside),
        "best": float(row["best"]),
        "head": row["head"],
    }


def _finish(calc: LjCalculator, wall: float, head: str) -> dict:
    hit = calc.hit_calls is not None
    best = calc.best_quenched if calc.best_quenched < 1.0e299 else calc.best
    return {
        "hit": int(hit),
        "calls_to_hit": calc.hit_calls if hit else "",
        "charged": calc.calls,
        "wall_s": wall,
        "inside_s": calc.inside_s,
        "outside_s": max(0.0, wall - calc.inside_s),
        "best": best,
        "head": head,
    }


def run_ase_bh(system: str, seed: int, budget: int, work: Path, start: Path) -> dict:
    import numpy as np
    from ase import Atoms
    from ase.optimize.basin import BasinHopping

    n = int(system[2:])
    os.chdir(work)
    np.random.seed(seed)
    atoms = Atoms(f"Ar{n}", positions=read_coords(start, n), pbc=False)
    calc = LjCalculator(budget, REFERENCES[system], HIT_TOL, FMAX["ase_bh"])
    atoms.calc = calc
    started = time.perf_counter()
    try:
        dyn = BasinHopping(
            atoms,
            logfile=str(work / "basin.log"),
            optimizer_logfile="/dev/null",
            trajectory=str(work / "lowest.traj"),
            local_minima_trajectory=str(work / "local_minima.traj"),
        )
        dyn.run(10**9)
    except (HitReached, BudgetReached):
        pass
    return _finish(calc, time.perf_counter() - started, git_head())


def run_ase_mh(system: str, seed: int, budget: int, work: Path, start: Path) -> dict:
    import numpy as np
    from ase import Atoms
    from ase.optimize.minimahopping import MinimaHopping

    n = int(system[2:])
    os.chdir(work)
    np.random.seed(seed)
    atoms = Atoms(f"Ar{n}", positions=read_coords(start, n), pbc=False)
    calc = LjCalculator(budget, REFERENCES[system], HIT_TOL, FMAX["ase_mh"])
    atoms.calc = calc
    started = time.perf_counter()
    try:
        MinimaHopping(atoms)(totalsteps=10**9)
    except (HitReached, BudgetReached):
        pass
    return _finish(calc, time.perf_counter() - started, git_head())


def run_gofee(system: str, seed: int, budget: int, work: Path) -> dict:
    import contextlib

    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    from ase import Atoms
    from ase.data import atomic_numbers, covalent_radii

    install_constraint_shim()
    from agox import AGOX
    from agox.acquisitors import LowerConfidenceBoundAcquisitor
    from agox.collectors import StandardCollector
    from agox.databases import Database
    from agox.environments import Environment
    from agox.evaluators import LocalOptimizationEvaluator
    from agox.generators import RandomGenerator, RattleGenerator
    from agox.models.GPR import GPR
    from agox.models.GPR.kernels import RBF, Noise
    from agox.models.GPR.kernels import Constant as C
    from agox.models.GPR.priors import Repulsive
    from agox.postprocessors import RelaxPostprocess
    from agox.samplers import KMeansSampler
    from agox.utils.filters.ABC_filter import FilterBaseClass

    class BoundedTrainingFilter(FilterBaseClass):
        name = "BoundedTraining"

        def __init__(self, m_points=512):
            super().__init__()
            self.m_points = m_points

        def _filter(self, atoms):
            count = len(atoms)
            if count <= self.m_points:
                return np.arange(count)
            energies = np.array([structure.get_potential_energy() for structure in atoms])
            chosen = {int(np.argmin(energies)), count - 1}
            pool = np.array([index for index in range(count) if index not in chosen])
            draw = np.random.choice(pool, size=self.m_points - len(chosen), replace=False)
            return np.sort(np.concatenate([np.fromiter(chosen, dtype=int), draw]))

    n = int(system[2:])
    os.chdir(work)
    radius = (3.0 * n / (4.0 * math.pi * 0.7)) ** (1.0 / 3.0)
    conf = 2.0 * radius + 6.0
    cell = conf + 2.0
    corner = (cell - conf) / 2.0
    rsum = 2.0 * float(covalent_radii[atomic_numbers["Ar"]])
    pair_min = 2.0 ** (1.0 / 6.0)
    bond = {"c1": 0.90 * pair_min / rsum, "c2": 2.50 / rsum}
    calc = LjCalculator(budget, REFERENCES[system], HIT_TOL, FMAX["agox_gofee"])
    template = Atoms("", cell=np.eye(3) * cell, pbc=False)
    environment = Environment(
        template=template,
        symbols=f"Ar{n}",
        confinement_cell=np.eye(3) * conf,
        confinement_corner=np.full(3, corner),
        print_report=False,
    )
    db_path = work / "gofee.db"
    if db_path.exists():
        db_path.unlink()
    database = Database(filename=str(db_path), order=5, initialize=True)
    descriptor = Fingerprint = __import__(
        "agox.models.descriptors.fingerprint", fromlist=["Fingerprint"]
    ).Fingerprint
    descriptor = Fingerprint(environment=environment)
    beta = 0.01
    k0 = C(beta, (beta, beta)) * RBF()
    k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
    kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(0.01, (0.01, 0.01))
    model = GPR(
        descriptor=descriptor,
        kernel=kernel,
        database=database,
        prior=Repulsive(),
        filter=BoundedTrainingFilter(512),
        n_optimize=0,
        use_ray=False,
    )
    sampler = KMeansSampler(descriptor=descriptor, database=database, sample_size=10)
    confinement = environment.get_confinement()
    generators = [
        RandomGenerator(**confinement, **bond),
        RattleGenerator(**confinement, **bond),
    ]
    collector = StandardCollector(
        generators=generators,
        sampler=sampler,
        environment=environment,
        num_candidates={0: [10, 0], 5: [3, 7]},
        order=1,
    )
    acquisitor = LowerConfidenceBoundAcquisitor(model=model, kappa=2, order=3)
    relaxer = RelaxPostprocess(
        model=acquisitor.get_acquisition_calculator(),
        constraints=environment.get_constraints(),
        optimizer_run_kwargs={"steps": 5},
        start_relax=8,
        order=2,
    )
    evaluator = LocalOptimizationEvaluator(
        calc,
        gets={"get_key": "prioritized_candidates"},
        optimizer_kwargs={"logfile": None},
        optimizer_run_kwargs={"fmax": 0.05, "steps": budget},
        constraints=environment.get_constraints(),
        store_trajectory=False,
        order=4,
    )
    search = AGOX(collector, relaxer, acquisitor, evaluator, database, seed=seed)
    started = time.perf_counter()
    log = (work / "gofee.stdio").open("w")
    try:
        with contextlib.redirect_stdout(log):
            try:
                search.run(N_iterations=budget, verbose=False, hide_log=True)
            finally:
                import atexit

                atexit.unregister(search.on_failure)
    except (HitReached, BudgetReached):
        pass
    finally:
        log.close()
    return _finish(calc, time.perf_counter() - started, git_head())


def git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def run_job(binary, repo, runs, system, seed, arm, head) -> dict:
    budget = PAPER_BUDGETS[system]
    work = runs / system / arm / f"seed{seed}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    start = runs / "starts" / system / f"seed{seed}.txt"
    ensure_start(binary, repo, system, seed, start)
    if arm == "paper":
        row = run_paper(binary, repo, system, seed, work)
    elif arm == "ase_bh":
        row = run_ase_bh(system, seed, budget, work, start)
    elif arm == "ase_mh":
        row = run_ase_mh(system, seed, budget, work, start)
    elif arm == "agox_gofee":
        row = run_gofee(system, seed, budget, work)
    else:
        raise RuntimeError(arm)
    row.update(
        {
            "system": system,
            "arm": arm,
            "seed": seed,
            "budget": budget,
            "ase": ASE_VERSION,
            "agox": AGOX_VERSION,
        }
    )
    if not row.get("head"):
        row["head"] = head
    return row


COLUMNS = [
    "order",
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "calls_to_hit",
    "charged",
    "wall_s",
    "inside_s",
    "outside_s",
    "best",
    "ase",
    "agox",
    "head",
]


def format_results(rows: list[dict]) -> str:
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["order"]),
                    row["system"],
                    row["arm"],
                    str(row["seed"]),
                    str(row["budget"]),
                    str(row["hit"]),
                    "" if row["calls_to_hit"] == "" else str(row["calls_to_hit"]),
                    str(row["charged"]),
                    f"{row['wall_s']:.6f}",
                    f"{row['inside_s']:.6f}",
                    f"{row['outside_s']:.6f}",
                    f"{row['best']:.8f}",
                    row["ase"],
                    row["agox"],
                    row["head"],
                ]
            )
        )
    return "\n".join(lines) + "\n"


def summarise(rows: list[dict]) -> str:
    header = (
        "system,arm,n,hits,budget,calls_to_hit_median,calls_to_hit_iqr,"
        "wall_median_s,wall_ratio,inside_median_s,outside_median_s"
    )
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["arm"]), []).append(row)
    paper_wall = {}
    for system in PAPER_BUDGETS:
        sample = [item["wall_s"] for item in grouped.get((system, "paper"), [])]
        paper_wall[system] = median_iqr(sample)[0] if sample else None
    lines = [header]
    for system in PAPER_BUDGETS:
        for arm in ARMS:
            group = grouped.get((system, arm), [])
            if not group:
                continue
            hits = [float(item["calls_to_hit"]) for item in group if item["hit"]]
            walls = [item["wall_s"] for item in group]
            insides = [item["inside_s"] for item in group]
            outsides = [item["outside_s"] for item in group]
            wall_median = median_iqr(walls)[0]
            ratio = ""
            if paper_wall[system]:
                ratio = f"{wall_median / paper_wall[system]:.6f}"
            if hits:
                call_median, call_iqr = median_iqr(hits)
                call_text = f"{call_median:.6f},{call_iqr:.6f}"
            else:
                call_text = ","
            lines.append(
                ",".join(
                    [
                        system,
                        arm,
                        str(len(group)),
                        str(sum(item["hit"] for item in group)),
                        str(group[0]["budget"]),
                        call_text,
                        f"{wall_median:.6f}",
                        ratio,
                        f"{median_iqr(insides)[0]:.6f}",
                        f"{median_iqr(outsides)[0]:.6f}",
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def comparison_slice(summary: str, arm: str) -> str:
    lines = [line for line in summary.splitlines() if line]
    kept = [lines[0]]
    for line in lines[1:]:
        fields = line.split(",")
        if fields[1] in {"paper", arm}:
            kept.append(line)
    return "\n".join(kept) + "\n"


def load_journal(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    if not lines:
        return []
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split(",")
        row = dict(zip(header, fields))
        row["order"] = int(row["order"])
        row["seed"] = int(row["seed"])
        row["budget"] = int(row["budget"])
        row["hit"] = int(row["hit"])
        row["charged"] = int(row["charged"])
        row["calls_to_hit"] = int(row["calls_to_hit"]) if row["calls_to_hit"] else ""
        row["wall_s"] = float(row["wall_s"])
        row["inside_s"] = float(row["inside_s"])
        row["outside_s"] = float(row["outside_s"])
        row["best"] = float(row["best"])
        rows.append(row)
    return rows


def write_tables(rows: list[dict], repo: Path, comparison: str | None) -> None:
    out = repo / "bench" / "ase-gofee-h2h"
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(rows)
    if comparison is None:
        (out / "results.csv").write_text(format_results(rows))
        (out / "summary.csv").write_text(summary)
        for arm in ("ase_bh", "ase_mh", "agox_gofee"):
            (out / f"{arm}.csv").write_text(comparison_slice(summary, arm))
    else:
        (out / f"{comparison}.csv").write_text(comparison_slice(summary, comparison))


def main() -> int:
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    check_type7()
    require_versions()
    warmup()
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--comparison", choices=["ase_bh", "ase_mh", "agox_gofee"])
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if not args.binary.is_file():
        print(f"missing binary {args.binary}", file=sys.stderr)
        return 2
    head = git_head()
    start = args.runs / "starts" / "lj38" / "seed2000.txt"
    ensure_start(args.binary, args.repo, "lj38", 2000, start)
    check_potential(args.binary, args.repo, start)

    if args.size or args.seed or args.arm:
        if not (args.size and args.seed is not None and args.arm):
            print("a single run needs --size, --seed and --arm", file=sys.stderr)
            return 2
        system = f"lj{args.size}"
        if system not in PAPER_BUDGETS:
            print(f"unknown size {args.size}", file=sys.stderr)
            return 2
        row = run_job(args.binary, args.repo, args.runs, system, args.seed, args.arm, head)
        row["order"] = 0
        print(format_results([row]), end="")
        return 0

    jobs = schedule(args.comparison)
    journal_path = args.runs / ("journal.csv" if args.comparison is None else f"journal-{args.comparison}.csv")
    rows = load_journal(journal_path)
    done = {(row["system"], row["seed"], row["arm"]) for row in rows}
    print(f"jobs {len(jobs)} already {len(done)}", flush=True)
    for index, (system, seed, arm) in enumerate(jobs, start=1):
        if (system, seed, arm) in done:
            continue
        print(f"start {index}/{len(jobs)} {system} {arm} seed={seed}", flush=True)
        try:
            row = run_job(args.binary, args.repo, args.runs, system, seed, arm, head)
        except Exception:
            traceback.print_exc()
            return 1
        row["order"] = index
        rows.append(row)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(format_results(rows))
        write_tables(rows, args.repo, args.comparison)
        print(
            f"done {index}/{len(jobs)} {system} {arm} seed={seed} hit={row['hit']} "
            f"charged={row['charged']} calls_to_hit={row['calls_to_hit']} "
            f"wall={row['wall_s']:.3f} inside={row['inside_s']:.3f} best={row['best']:.6f}",
            flush=True,
        )
    write_tables(rows, args.repo, args.comparison)
    print(summarise(rows), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
