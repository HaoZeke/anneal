#!/usr/bin/env python3
"""Run one ASE or AGOX arm of the Lennard-Jones board.

Usage: python3 scripts/lj_board/run_arm.py CODE SIZE SEED
CODE is ase or agox.
The budget is the paper ceiling for that size. It is not an argument.
Exit 75 means a wall stopped the run. The log keeps the charged count.
"""

from __future__ import annotations

import os
import signal
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boardlib import (
    FORCE_BAR,
    HIT_TOL,
    PAPER_BUDGETS,
    REFERENCES,
    WALES_STEP,
    WALES_TEMPERATURE,
    BudgetReached,
    HitReached,
    Journal,
    Stopwatch,
    WallReached,
    git_lines,
    held_out_seeds,
    host_lines,
    independent_check,
    bfgs_apply,
    bfgs_inverse_update,
    lj_grad,
    random_cluster,
    rattle_c1,
    utc_now,
    write_xyz,
)

import numpy as np

WALL = {"stop": False}
_STDERR = os.dup(2)


def _progress(text: str) -> None:
    os.write(_STDERR, (text + "\n").encode())


def _on_wall(signum, frame):
    WALL["stop"] = True
    raise WallReached()


def install_fast_bfgs() -> None:
    """Apply ASE's own BFGS step with the inverse Hessian.

    ASE 3.29 diagonalizes the Hessian on every step. The inverse update
    matches that step while the curvature stays positive, which it did on
    quenched Lennard-Jones clusters. The Hessian update itself is unchanged.
    """
    from ase._4.optimize.bfgs import BFGSMethod

    if getattr(BFGSMethod, "_board_inverse", False):
        return
    original_init = BFGSMethod.__init__

    def _init(self, hessian):
        original_init(self, hessian)
        diagonal = np.diag(self.hessian)
        width = len(diagonal)
        self._board_inv = np.eye(width, dtype=np.float64) * (1.0 / float(diagonal[0]))

    def _reset_inv(self):
        width = self._board_inv.shape[0]
        self._board_inv[:, :] = 0.0
        diag = np.diag_indices(width)
        self._board_inv[diag] = 1.0 / 70.0

    def _update(self, pos, gradient, pos0, gradient0):
        step = np.ascontiguousarray(pos - pos0, dtype=np.float64)
        slope_src = np.ascontiguousarray(gradient - gradient0, dtype=np.float64)
        if (
            step.shape != slope_src.shape
            or not np.isfinite(step).all()
            or not np.isfinite(slope_src).all()
        ):
            self._reset_inv()
            return
        if np.abs(step).max() < 1e-7:
            return
        curvature = float(step @ slope_src)
        if not np.isfinite(curvature) or abs(curvature) <= 1.0e-16:
            return
        bfgs_inverse_update(self._board_inv, step, slope_src)
        if not np.isfinite(self._board_inv).all():
            self._reset_inv()

    def _compute_step(self, gradient):
        grad = np.ascontiguousarray(gradient, dtype=np.float64)
        if grad.shape != (self._board_inv.shape[0],) or not np.isfinite(grad).all():
            self._reset_inv()
            return np.zeros_like(grad)
        direction = bfgs_apply(self._board_inv, grad)
        if not np.isfinite(direction).all():
            self._reset_inv()
            scale = float(np.max(np.abs(grad)))
            if not np.isfinite(scale) or scale == 0.0:
                return np.zeros_like(grad)
            return -grad / scale * 0.1
        return direction

    BFGSMethod.__init__ = _init
    BFGSMethod.update = _update
    BFGSMethod.compute_step = _compute_step
    BFGSMethod._board_inverse = True


def out_dir() -> Path:
    root = Path(os.environ.get("LJ_BOARD_ROOT", "results/lj-board"))
    if root.is_absolute():
        raise SystemExit("LJ_BOARD_ROOT must be a relative path")
    return root


def prepare_agox_import():
    import ase.constraints as ase_constraints
    from ase.constraints.constraint import IndexedConstraint, slice2enlist

    if not hasattr(ase_constraints, "IndexedConstraint"):
        ase_constraints.IndexedConstraint = IndexedConstraint
    if not hasattr(ase_constraints, "slice2enlist"):
        ase_constraints.slice2enlist = slice2enlist
    from agox.generators.ABC_generator import GeneratorBaseClass
    from agox.helpers.confinement import Confinement
    from agox.utils.constraints.box_constraint import BoxConstraint
    from ase.data import covalent_radii

    if getattr(BoxConstraint, "_board_fast", False):
        return
    original_check = GeneratorBaseClass.check_new_position
    original_projection = Confinement.get_projection_coefficients

    def get_projection_coefficients(self, positions):
        cell = np.asarray(self.effective_cell, dtype=float)
        diagonal = np.diag(cell)
        if cell.shape == (3, 3) and np.count_nonzero(cell - np.diag(diagonal)) == 0:
            coords = np.asarray(positions, dtype=float).reshape(-1, 3)
            width = self.dimensionality
            return (coords[:, :width] - self.corner[:width]) / diagonal[:width]
        return original_projection(self, positions)

    def check_new_position(self, candidate, new_position, number, skipped_indices=None):
        # Same accept and reject tests as the generator, on whole arrays.
        if self.use_mic:
            return original_check(self, candidate, new_position, number, skipped_indices)
        coords = np.asarray(candidate.positions, dtype=float)
        numbers = np.asarray(candidate.numbers)
        if skipped_indices:
            keep = np.ones(len(coords), dtype=bool)
            keep[list(skipped_indices)] = False
            coords = coords[keep]
            numbers = numbers[keep]
        if len(coords) == 0:
            return False
        delta = coords - np.asarray(new_position, dtype=float)
        distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        span = covalent_radii[numbers] + covalent_radii[number]
        near = self.c1 * span
        far = self.c2 * span
        if np.any(distance < near):
            return False
        return bool(np.any(distance <= far))

    GeneratorBaseClass.check_new_position = check_new_position
    Confinement.get_projection_coefficients = get_projection_coefficients
    original_adjust = BoxConstraint.adjust_forces

    def adjust_forces(self, atoms, forces):
        coeff = np.asarray(
            self.get_projection_coefficients(atoms.positions[self.indices]),
            dtype=float,
        )
        periodic = np.asarray(self.pbc, dtype=bool)
        if periodic.any():
            coeff = np.array(coeff, copy=True)
            coeff[:, periodic] = 0.5
        if coeff.size == 0:
            return
        # Interior atoms keep their forces. The original loop only rescales
        # a component that has crossed a soft wall.
        if (
            coeff.min() >= self.lower_soft_boundary
            and coeff.max() <= self.upper_soft_boundary
        ):
            return
        original_adjust(self, atoms, forces)

    BoxConstraint.adjust_forces = adjust_forces
    BoxConstraint._board_fast = True


class BoundCalculator:
    """Counts every new potential evaluation against the paper ceiling."""

    def __init__(self, budget: int, reference: float):
        from ase.calculators.calculator import Calculator

        outer = self

        class _Bound(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self):
                Calculator.__init__(self)
                self.use_cache = False
                self.budget = budget
                self.reference = reference
                self.calls = 0
                self.best = 1.0e300
                self.hit_calls = None
                self.hit_energy = None
                self.hit_force = None
                self.hit_positions = None
                self._pos = None

            def check_state(self, atoms, tol=1e-15):
                pos = atoms.positions
                cached = self._pos
                if (
                    cached is not None
                    and cached.shape == pos.shape
                    and np.array_equal(cached, pos)
                ):
                    return []
                self._pos = np.array(pos, copy=True)
                return ["positions"]

            def calculate(self, atoms=None, properties=None, system_changes=None):
                if WALL["stop"]:
                    raise WallReached()
                if self.calls >= self.budget:
                    raise BudgetReached()
                positions = np.ascontiguousarray(atoms.positions.ravel(), dtype=np.float64)
                energy, grad = lj_grad(positions)
                self.calls += 1
                forces = (-grad).reshape(-1, 3)
                self.results["energy"] = float(energy)
                self.results["forces"] = forces
                fmax = float(np.max(np.abs(forces)))
                if energy < self.best:
                    self.best = float(energy)
                if self.calls % 50000 == 0:
                    _progress(
                        f"progress {utc_now()} calls={self.calls} best={self.best:.6f}"
                    )
                if (
                    self.hit_calls is None
                    and energy < self.reference + HIT_TOL
                    and fmax <= FORCE_BAR
                ):
                    self.hit_calls = self.calls
                    self.hit_energy = float(energy)
                    self.hit_force = fmax
                    self.hit_positions = np.array(atoms.positions, dtype=float, copy=True)
                    raise HitReached()

        outer.cls = _Bound

    def new(self):
        return self.cls()


def _stop_name(calc, budget: int) -> str:
    if calc.hit_calls is not None:
        return "hit"
    if WALL["stop"]:
        return "wall"
    if calc.calls >= budget:
        return "ceiling"
    return "stall"


def run_ase(n, budget, reference, seed, work: Path):
    install_fast_bfgs()
    from ase import Atoms
    from ase.optimize import BFGS
    from ase.optimize.basin import BasinHopping

    import importlib.metadata as metadata

    coords = random_cluster(n, seed)
    np.savetxt(work / "start.xyz", coords)
    atoms = Atoms(f"Ar{n}", positions=coords)
    calc = BoundCalculator(budget, reference).new()
    atoms.calc = calc
    np.random.seed(seed)
    ase_version = metadata.version("ase")

    class CappedBasinHopping(BasinHopping):
        """Same basin hop, with a finite local-step cap.

        A broken curvature would otherwise keep the local optimizer
        stepping without moving, up to its default of 1e8 iterations.
        Healthy quenches converge well below this cap.
        """

        def get_energy(self, positions):
            if np.any(self.positions != positions):
                self.positions = positions
                self.optimizable.set_x(positions.ravel())
                with self.optimizer(self.optimizable, logfile=self.optimizer_logfile) as opt:
                    opt.run(fmax=self.fmax, steps=5000)
                if self.lm_trajectory is not None:
                    self.lm_trajectory.write(self.optimizable)
                self.energy = self.optimizable.get_value()
            return self.energy

    hop = CappedBasinHopping(
        atoms,
        temperature=WALES_TEMPERATURE,
        dr=WALES_STEP,
        optimizer=BFGS,
        fmax=FORCE_BAR,
        logfile=None,
        trajectory=None,
        optimizer_logfile=None,
        local_minima_trajectory=None,
    )
    note = (
        f"ase={ase_version} BasinHopping temperature={WALES_TEMPERATURE} "
        f"dr={WALES_STEP} optimizer=BFGS fmax={FORCE_BAR} local_steps=5000 "
        f"numpy_seed={seed} "
        "citation=Wales,Doye,J.Phys.Chem.A,101,5111,1997"
    )
    stalled = 0
    try:
        while calc.calls < budget and calc.hit_calls is None and not WALL["stop"]:
            before = calc.calls
            hop.run(1)
            if calc.calls == before:
                stalled += 1
                if stalled >= 3:
                    raise RuntimeError("basin hopping added no charged calls")
            else:
                stalled = 0
    except (BudgetReached, HitReached, WallReached):
        pass
    return calc, note, "ase-basinhopping"


def run_agox(n, budget, reference, seed, work: Path):
    install_fast_bfgs()
    prepare_agox_import()
    import contextlib
    import importlib.metadata as metadata

    from ase import Atoms
    from ase.optimize import BFGS
    from agox import AGOX
    from agox.candidates.standard import StandardCandidate
    from agox.databases import Database
    from agox.environments import Environment
    from agox.evaluators import LocalOptimizationEvaluator
    from agox.generators import RattleGenerator
    from agox.samplers import MetropolisSampler
    from ase.calculators.singlepoint import SinglePointCalculator

    version = metadata.version("agox")
    if version != "3.11.1":
        raise SystemExit(f"agox {version} is not 3.11.1")
    radius = (3.0 * n / (4.0 * np.pi * 0.7)) ** (1.0 / 3.0)
    span = max(2.0 * radius + 4.0, 8.0)
    margin = 1.0
    template = Atoms(cell=np.eye(3) * span, pbc=False)
    environment = Environment(
        template=template,
        symbols=f"Ar{n}",
        confinement_cell=np.eye(3) * (span - 2.0 * margin),
        confinement_corner=np.full(3, margin),
        print_report=False,
    )
    calc = BoundCalculator(budget, reference).new()
    db_path = work / "agox.db"
    if db_path.exists():
        db_path.unlink()
    database = Database(filename=str(db_path), order=4, initialize=True)
    sampler = MetropolisSampler(temperature=WALES_TEMPERATURE, order=3)
    c1 = rattle_c1()
    generator = RattleGenerator(
        **environment.get_confinement(),
        n_rattle=n,
        rattle_amplitude=WALES_STEP,
        environment=environment,
        sampler=sampler,
        c1=c1,
        use_mic=False,
        order=1,
    )
    evaluator = LocalOptimizationEvaluator(
        calc,
        optimizer=BFGS,
        optimizer_run_kwargs={"fmax": FORCE_BAR, "steps": 200},
        optimizer_kwargs={"logfile": None},
        gets={"get_key": "candidates"},
        store_trajectory=False,
        order=2,
        constraints=environment.get_constraints(),
    )

    def _observe(candidate, steps):
        evaluator.check_callback(candidate)
        if steps() == 0:
            evaluator.E_initial = candidate.get_potential_energy(apply_constraint=False)

    evaluator._observer = _observe
    search = AGOX(generator, database, sampler, evaluator, seed=seed, report=False)
    center = np.full(3, margin + 0.5 * (span - 2.0 * margin))
    placed = random_cluster(n, seed) + center
    np.savetxt(work / "start.xyz", placed)
    parent = StandardCandidate(template=template, positions=placed, numbers=[18] * n)
    parent.calc = calc
    try:
        energy = float(parent.get_potential_energy())
        forces = parent.get_forces()
    except (BudgetReached, HitReached, WallReached):
        energy = None
        forces = None
    if energy is not None and calc.hit_calls is None:
        parent.calc = SinglePointCalculator(parent, energy=energy, forces=forces)
        parent.add_meta_information("uuid", f"seed-{seed}")
        parent.add_meta_information("spawn_uuids", [f"seed-{seed}"])
        parent.add_meta_information("search_iteration", 0)
        sampler.sample = [parent]
    stalled = {"count": 0}

    def dispatch(*args, **kwargs):
        if WALL["stop"] or calc.hit_calls is not None or calc.calls >= budget:
            raise BudgetReached()
        before = calc.calls
        AGOX.dispatch_to_observers(search, *args, **kwargs)
        if WALL["stop"] or calc.hit_calls is not None or calc.calls >= budget:
            raise BudgetReached()
        if calc.calls == before:
            stalled["count"] += 1
            if stalled["count"] >= 3:
                raise RuntimeError("the native search added no charged calls")
        else:
            stalled["count"] = 0

    search.dispatch_to_observers = dispatch
    note = (
        f"agox={version} AGOX+Database+Environment+RattleGenerator+"
        "MetropolisSampler+LocalOptimizationEvaluator "
        f"temperature={WALES_TEMPERATURE} rattle_amplitude={WALES_STEP} "
        f"n_rattle={n} c1={c1:.6f} use_mic=0 position_check=array "
        f"fmax={FORCE_BAR} steps=200 "
        f"seed={seed} "
        "citation=Christiansen,Ronne,Hammer,J.Chem.Phys.157,054701,2022;"
        "Wales,Doye,J.Phys.Chem.A,101,5111,1997"
    )
    if calc.hit_calls is None and not WALL["stop"] and calc.calls < budget:
        stdio_path = work / "agox.stdio"
        try:
            with stdio_path.open("w") as handle:
                with contextlib.redirect_stdout(handle):
                    try:
                        search.run(N_iterations=budget, verbose=False, hide_log=True)
                    finally:
                        import atexit

                        atexit.unregister(search.on_failure)
        except (BudgetReached, HitReached, WallReached):
            pass
    return calc, note, f"agox-{version}"


def finish_calculator(calc, note, native, reference, budget, work: Path):
    hit = 1 if calc.hit_calls is not None else 0
    calls = calc.hit_calls if hit else calc.calls
    energy = calc.hit_energy
    force = calc.hit_force
    exit_code = 0
    stop = _stop_name(calc, budget)
    if hit and calc.hit_positions is not None:
        check_energy, check_force, ok = independent_check(
            calc.hit_positions, reference, calls, budget
        )
        write_xyz(work / "hit.xyz", calc.hit_positions, check_energy)
        if not ok or abs(check_energy - energy) > 1.0e-6:
            hit = 0
            calls = calc.calls
            stop = "audit"
            exit_code = 1
        else:
            energy = check_energy
            force = check_force
    if stop == "stall":
        exit_code = 1
    return {
        "hit": hit,
        "calls": calls,
        "charged": calc.calls,
        "best": calc.best,
        "hit_energy": energy,
        "hit_force": force,
        "note": note,
        "exit_code": exit_code,
        "native": native,
        "stop": stop,
    }


def write_log(path: Path, payload: dict, clock: Stopwatch, command: str) -> None:
    lines = [
        f"start={clock.started}",
        *git_lines(),
        *host_lines(),
        f"command={command}",
        f"code={payload['code']} n={payload['n']} seed={payload['seed']}",
        f"budget={payload['budget']} reference={payload['reference']:.6f}",
        f"hit_tol={HIT_TOL:.6e} force_bar={FORCE_BAR:.7f}",
        payload["note"],
        f"native={payload['native']}",
        f"best={payload['best']:.6f}",
    ]
    if payload.get("hit_energy") is not None:
        lines.append(f"hit_energy={payload['hit_energy']:.8f}")
        lines.append(f"hit_force={payload['hit_force']:.6e}")
    lines.append(f"hit={payload['hit']}")
    lines.append(f"calls={payload['calls']}")
    lines.append(f"charged={payload['charged']}")
    lines.append(f"wall_s={clock.wall_s:.6f}")
    lines.append(f"end={clock.ended}")
    lines.append(f"stop={payload['stop']}")
    lines.append(f"exit_code={payload['exit_code']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_arm.py CODE SIZE SEED")
    code = sys.argv[1]
    n = int(sys.argv[2])
    seed = int(sys.argv[3])
    if seed not in set(held_out_seeds()):
        raise SystemExit(f"seed {seed} is not in the frozen held-out list")
    if n not in PAPER_BUDGETS:
        raise SystemExit(f"size {n} has no paper budget")
    if code not in {"ase", "agox"}:
        raise SystemExit(f"unknown arm {code}")
    budget = PAPER_BUDGETS[n]
    reference = REFERENCES[n]
    root = out_dir()
    work = root / code / f"lj{n}-seed{seed}"
    work.mkdir(parents=True, exist_ok=True)
    log_path = root / code / f"lj{n}-seed{seed}.log"
    journal = Journal(root / "provenance" / "launch-journal.tsv")
    command = f"python3 scripts/lj_board/run_arm.py {code} {n} {seed}"
    journal.append("start", code=code, n=n, seed=seed, budget=budget, head=git_lines()[0])
    clock = Stopwatch()
    exit_code = 1
    signal.signal(signal.SIGTERM, _on_wall)
    signal.signal(signal.SIGINT, _on_wall)
    payload = None
    try:
        if code == "ase":
            calc, note, native = run_ase(n, budget, reference, seed, work)
        else:
            calc, note, native = run_agox(n, budget, reference, seed, work)
        payload = finish_calculator(calc, note, native, reference, budget, work)
        payload.update(
            {"code": code, "n": n, "seed": seed, "budget": budget, "reference": reference}
        )
        exit_code = int(payload["exit_code"])
    except WallReached:
        WALL["stop"] = True
        payload = {
            "code": code,
            "n": n,
            "seed": seed,
            "budget": budget,
            "reference": reference,
            "hit": 0,
            "calls": 0,
            "charged": 0,
            "best": 1.0e300,
            "hit_energy": None,
            "hit_force": None,
            "note": "wall before the first charged geometry",
            "exit_code": 0,
            "native": code,
            "stop": "wall",
        }
        exit_code = 0
    except Exception:
        traceback.print_exc()
        payload = {
            "code": code,
            "n": n,
            "seed": seed,
            "budget": budget,
            "reference": reference,
            "hit": 0,
            "calls": 0,
            "charged": 0,
            "best": 1.0e300,
            "hit_energy": None,
            "hit_force": None,
            "note": "exception",
            "exit_code": 1,
            "native": code,
            "stop": "error",
        }
        exit_code = 1
    finally:
        clock.stop()
        if payload is None:
            payload = {
                "code": code,
                "n": n,
                "seed": seed,
                "budget": budget,
                "reference": reference,
                "hit": 0,
                "calls": 0,
                "charged": 0,
                "best": 1.0e300,
                "hit_energy": None,
                "hit_force": None,
                "note": "exception",
                "exit_code": 1,
                "native": code,
                "stop": "error",
            }
            exit_code = 1
        if payload.get("stop") == "wall":
            exit_code = 75
            payload["exit_code"] = 75
        write_log(log_path, payload, clock, command)
        journal.append(
            "end",
            code=code,
            n=n,
            seed=seed,
            budget=budget,
            hit=payload["hit"],
            calls=payload["calls"],
            charged=payload["charged"],
            stop=payload["stop"],
            wall_s=f"{clock.wall_s:.6f}",
            exit_code=exit_code,
        )
    print(
        f"{code} LJ{n} seed {seed} hit {payload['hit']} calls {payload['calls']} "
        f"charged {payload['charged']} stop {payload['stop']} "
        f"wall {clock.wall_s:.3f} exit {exit_code}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
