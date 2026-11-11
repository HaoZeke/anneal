#!/usr/bin/env python3
"""AGOX basin hopping as the 3.11.1 cluster example documents it.

The search modules follow script_bh_cluster_emt.py and BasinHoppingConfig.
BoxConstraint is the stock class from Environment.get_constraints.
The Lennard-Jones calculator, the 400000-call budget, and the hit rule
are the board's.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

print("wrapper_header hostname bench was set via unshare --uts", flush=True)
print(f"uts_hostname={os.uname().nodename}", flush=True)
print("arm=agox-documented", flush=True)
if os.uname().nodename != "bench":
    print("uts hostname is not bench", file=sys.stderr)
    sys.exit(2)

os.chdir("/work/anneal")
os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
        "PYTHONUNBUFFERED": "1",
        "LJ_BOARD_ROOT": "results/lj-board",
        "HOME": "/work/home",
        "TMPDIR": "/work/scratch",
        "MPLCONFIGDIR": "/work/mpl",
        "NUMBA_CACHE_DIR": "/work/numba",
        "XDG_CACHE_HOME": "/work/cache",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "/work/anneal",
    }
)

sys.path.insert(0, "scripts/lj_board")
import run_arm
from boardlib import (
    FORCE_BAR,
    HIT_TOL,
    PAPER_BUDGETS,
    REFERENCES,
    BudgetReached,
    HitReached,
    Stopwatch,
    WallReached,
    held_out_seeds,
)

stats = {"calls": 0, "inside_s": 0.0}
_lj_grad = run_arm.lj_grad


def timed(x):
    started = time.perf_counter()
    result = _lj_grad(x)
    stats["inside_s"] += time.perf_counter() - started
    stats["calls"] += 1
    return result


run_arm.lj_grad = timed

import ase.constraints as ase_constraints
import numpy as np
from ase.constraints.constraint import IndexedConstraint, slice2enlist

# ASE 3.29 moved these names. AGOX 3.11.1 still imports them from ase.constraints.
if not hasattr(ase_constraints, "IndexedConstraint"):
    ase_constraints.IndexedConstraint = IndexedConstraint
if not hasattr(ase_constraints, "slice2enlist"):
    ase_constraints.slice2enlist = slice2enlist

from ase import Atoms
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator, RattleGenerator
from agox.samplers import MetropolisSampler


def main() -> int:
    seed = int(sys.argv[1])
    if seed not in set(held_out_seeds()):
        print(f"seed {seed} is outside the held-out list", file=sys.stderr)
        return 2
    n = 38
    budget = PAPER_BUDGETS[n]
    reference = REFERENCES[n]
    # script_bh_cluster_emt.py lines 31-33 use cell 12, confinement 6, corner 3
    # for Au8Ni8: the box edge is half the cell and the corner centers it.
    # Ar38 keeps that ratio. The confinement edge is the sphere diameter at
    # density 0.7 plus 4, so the 38 atoms fit.
    radius = (3.0 * n / (4.0 * np.pi * 0.7)) ** (1.0 / 3.0)
    confinement_edge = 2.0 * radius + 4.0
    cell_edge = 2.0 * confinement_edge
    corner = confinement_edge / 2.0
    template = Atoms(cell=np.eye(3) * cell_edge, pbc=False)
    environment = Environment(
        template=template,
        symbols=f"Ar{n}",
        confinement_cell=np.eye(3) * confinement_edge,
        confinement_corner=np.full(3, corner),
        print_report=False,
    )
    root = Path("results/lj-board")
    work = root / "agox-documented" / f"lj{n}-seed{seed}"
    work.mkdir(parents=True, exist_ok=True)
    calc = run_arm.BoundCalculator(budget, reference).new()
    db_path = work / "agox.db"
    if db_path.exists():
        db_path.unlink()
    database = Database(filename=str(db_path), order=4, initialize=True)
    # BasinHoppingConfig sampler default, and script_bh_cluster_emt.py line 49.
    sampler = MetropolisSampler(temperature=0.25, order=3)
    # RattleGenerator defaults: n_rattle=3, rattle_amplitude=3.
    generator = RattleGenerator(
        **environment.get_confinement(),
        environment=environment,
        sampler=sampler,
        order=1,
    )
    # BasinHoppingConfig evaluator: LocalOptConfig fmax=0.05, max_steps=100.
    # constraints=environment.get_constraints() is script_bh_cluster_emt.py line 60,
    # which attaches the stock BoxConstraint.
    evaluator = LocalOptimizationEvaluator(
        calc,
        gets={"get_key": "candidates"},
        store_trajectory=False,
        optimizer_run_kwargs={"fmax": 0.05, "steps": 100},
        optimizer_kwargs={"logfile": None},
        order=2,
        constraints=environment.get_constraints(),
    )
    # RattleGenerator asks for one parent. An empty sampler returns none,
    # so the first structure comes from RandomGenerator, the documented
    # generator that takes no parent.
    starter = RandomGenerator(**environment.get_confinement(), environment=environment)
    seeded = None
    for _ in range(20):
        made = starter.get_candidates(sampler, environment)
        if made:
            seeded = made[0]
            break
    if seeded is None:
        print("RandomGenerator produced no start", file=sys.stderr)
        return 2
    seeded.calc = calc
    try:
        from ase.optimize import BFGS

        BFGS(seeded, logfile=None).run(fmax=0.05, steps=100)
    except (BudgetReached, HitReached, WallReached):
        pass
    if calc.hit_calls is None and calc.calls < budget:
        sampler.sample = [seeded]
    search = AGOX(generator, database, sampler, evaluator, seed=seed, report=False)

    def dispatch(*args, **kwargs):
        if calc.hit_calls is not None or calc.calls >= budget:
            raise BudgetReached()
        before = calc.calls
        AGOX.dispatch_to_observers(search, *args, **kwargs)
        if calc.hit_calls is not None or calc.calls >= budget:
            raise BudgetReached()
        if calc.calls == before:
            dispatch.stall += 1
            if dispatch.stall >= 3:
                raise RuntimeError("the search added no charged calls")
        else:
            dispatch.stall = 0

    dispatch.stall = 0
    search.dispatch_to_observers = dispatch
    clock = Stopwatch()
    import contextlib

    try:
        with (work / "agox.stdio").open("w") as handle:
            with contextlib.redirect_stdout(handle):
                try:
                    search.run(N_iterations=budget, verbose=False, hide_log=True)
                finally:
                    import atexit

                    atexit.unregister(search.on_failure)
    except (BudgetReached, HitReached, WallReached, RuntimeError):
        pass
    clock.stop()
    note = (
        "agox-documented BasinHoppingConfig "
        "temperature=0.25 n_rattle=3 rattle_amplitude=3 "
        "fmax=0.05 steps=100 constraints=environment.get_constraints "
        "box=stock no_force_patch"
    )
    payload = run_arm.finish_calculator(calc, note, "agox-3.11.1", reference, budget, work)
    payload.update(
        {"code": "agox-documented", "n": n, "seed": seed, "budget": budget, "reference": reference}
    )
    command = f"python3 scripts/lj38_paired/agox_documented.py {seed}"
    log_path = root / "agox-documented" / f"lj{n}-seed{seed}.log"
    run_arm.write_log(log_path, payload, clock, command)
    inside = stats["inside_s"]
    wall = clock.wall_s
    text = "\n".join(
        [
            f"energy_calls={stats['calls']}",
            f"gradient_calls={stats['calls']}",
            f"potential_calls={stats['calls']}",
            f"charged={calc.calls}",
            f"inside_s={inside:.6f}",
            f"outside_s={wall - inside:.6f}",
            f"wall_s={wall:.6f}",
            "fused=1",
            "setup=documented",
            "",
        ]
    )
    (root / "agox-documented" / f"lj{n}-seed{seed}.potential.txt").write_text(text)
    print(
        f"agox-documented LJ{n} seed {seed} hit {payload['hit']} calls {payload['calls']} "
        f"charged {payload['charged']} stop {payload['stop']} wall {wall:.3f}",
        flush=True,
    )
    print(text, end="", flush=True)
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
