# Copyright (c) anneal developers
"""ASE-shaped optimizer: ``from anneal.ase import Anneal``.

A local quench uses the same call site as ASE ``BFGS``::

    from anneal.ase import Anneal
    opt = Anneal(atoms)                 # mode="local"
    opt.run(fmax=0.05)

That stays in the starting basin. It is not a class rename for hops.
``mode="search"`` hops from the current geometry via ``search_from``,
then quenches. ``mode="auto"`` hops only when there are four or more
free atoms. Periodic cells and cell filters are refused on hop.
"""

from __future__ import annotations

import sys

import numpy as np

from anneal import cluster_search, polish

__all__ = ["Anneal", "PotentialCalculator"]

_MODES = frozenset({"auto", "local", "search"})


def _base_ase_calculator():
    try:
        from ase.calculators.calculator import Calculator
    except ImportError:
        return object
    return Calculator


class PotentialCalculator(_base_ase_calculator()):
    """ASE-compatible calculator wrapping an ``anneal`` potential or objective.

    ``potential`` may be any built-in potential (``PairPotential``,
    ``BinaryLennardJones``, ``SuttonChen``, ``Gupta``), a ``PyObjective``
    constructed with ``grad_fn``, or an object providing
    ``value_and_gradient(x) -> (energy, gradient)``. Forces are returned
    with ASE's minus-gradient sign convention and shape ``(N, 3)``.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, potential, **kwargs):
        if _base_ase_calculator() is not object:
            super().__init__(**kwargs)
        if not (
            hasattr(potential, "value_and_gradient")
            or (hasattr(potential, "eval") and hasattr(potential, "grad"))
        ):
            raise TypeError(
                "potential must implement value_and_gradient(x) or eval(x) and grad(x)"
            )
        self.potential = potential
        self.results: dict[str, object] = {}
        self.atoms = None

    def reset(self) -> None:
        self.results = {}

    def _eval_and_grad(self, flat: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.ascontiguousarray(flat, dtype=np.float64).reshape(-1)
        if hasattr(self.potential, "value_and_gradient"):
            energy, grad = self.potential.value_and_gradient(x)
        else:
            energy = self.potential.eval(x)
            grad = self.potential.grad(x)
        return float(energy), np.asarray(grad, dtype=np.float64).reshape(-1)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=None,
    ) -> None:
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("atoms must be provided to calculate")
        self.atoms = atoms
        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        energy, grad = self._eval_and_grad(pos.reshape(-1))
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -grad.reshape(pos.shape),
        }

    def get_potential_energy(self, atoms=None, force_consistent: bool = False) -> float:
        target = self.atoms if atoms is None else atoms
        if target is None:
            raise ValueError("atoms must be provided")
        self.calculate(target, properties=("energy", "forces"))
        return float(self.results["energy"])

    def get_forces(self, atoms=None) -> np.ndarray:
        target = self.atoms if atoms is None else atoms
        if target is None:
            raise ValueError("atoms must be provided")
        self.calculate(target, properties=("energy", "forces"))
        return np.asarray(self.results["forces"], dtype=np.float64).copy()


def _positions(atoms) -> np.ndarray:
    return np.asarray(atoms.get_positions(), dtype=np.float64)


def _set_positions(atoms, x: np.ndarray) -> None:
    atoms.set_positions(np.asarray(x, dtype=np.float64).reshape(-1, 3))


def _max_force(atoms) -> float:
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if forces.size == 0:
        return 0.0
    return float(np.sqrt((forces * forces).sum(axis=1)).max())


def _callbacks(atoms):
    """Energy and plus-gradient on a flat 3N vector.

    ASE ``get_forces`` is minus grad E. ``polish`` wants plus grad E.
    ``cluster_search`` probes orientation, so the same sign is safe there.
    """

    def energy(x):
        _set_positions(atoms, x)
        return float(atoms.get_potential_energy())

    def grad(x):
        _set_positions(atoms, x)
        return -np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1)

    return energy, grad


def _bounds_about(x: np.ndarray, pad: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    span = max(pad, float(np.ptp(pts, axis=0).max()) + pad)
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    return flat - span, flat + span


def _has_constraints(atoms) -> bool:
    if getattr(atoms, "frozen", None):
        return True
    return bool(getattr(atoms, "constraints", None))


def _has_pbc(atoms) -> bool:
    pbc = getattr(atoms, "pbc", None)
    if pbc is None:
        return False
    return bool(np.any(np.asarray(pbc)))


def _is_cell_filter(atoms) -> bool:
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        return True
    inner = getattr(atoms, "atoms", None)
    if inner is None or not hasattr(inner, "get_positions"):
        return False
    inner_n = int(np.asarray(inner.get_positions()).shape[0])
    return int(pos.shape[0]) != inner_n


def _nn_length_scale(atoms) -> float:
    pts = _positions(atoms)
    if len(pts) < 2:
        return 1.0
    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)
    scale = float(np.median(nn[np.isfinite(nn)]))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def _free_count(atoms) -> int:
    n = int(len(atoms))
    frozen = getattr(atoms, "frozen", None)
    if frozen:
        return n - len({int(i) for i in frozen})
    constraints = getattr(atoms, "constraints", None)
    if not constraints:
        return n
    if all(isinstance(item, (int, np.integer)) for item in constraints):
        return n - len({int(i) for i in constraints})
    mask = np.ones(n, dtype=bool)
    for cons in constraints:
        index = getattr(cons, "index", None)
        if index is None:
            continue
        mask[np.asarray(index, dtype=int).reshape(-1)] = False
    return int(mask.sum())


def _require_calculator(atoms) -> None:
    if getattr(atoms, "calc", None) is not None:
        return
    if hasattr(atoms, "numbers"):
        raise ValueError("atoms need a calculator")
    if not hasattr(atoms, "get_potential_energy"):
        raise ValueError("atoms need a calculator")


class Anneal:
    """ASE ``Optimizer`` stand-in. Default is a local quench.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` (or a duck type with ``get/set_positions``,
        ``get_potential_energy``, ``get_forces``, and ``__len__``).
    logfile, trajectory, master
        Same names as ASE. ``attach`` fires observers. ``trajectory`` is
        written at the start of ``run`` and after each step unless
        ``master=False``; errors raise.
    restart
        Restart files are not supported and a non-``None`` value raises
        ``NotImplementedError``.
    force_consistent, append_trajectory
        Accepted for the BFGS constructor shape. ``append_trajectory``
        is passed to the final/step write. ``force_consistent`` is unused
        (ASE >= 3.23 dropped it on ``Optimizer``).
    seed
        Hop RNG seed.
    budget
        Charged evaluations for the hop phase. Default scales with atom count.
    mode
        ``local`` (default), ``search``, or ``auto``.
    """

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        master=None,
        force_consistent=None,
        append_trajectory=False,
        seed: int = 0,
        budget: int | None = None,
        mode: str = "local",
    ):
        _require_calculator(atoms)
        if restart is not None:
            raise NotImplementedError("Anneal restart files are not supported")
        mode = str(mode)
        if mode not in _MODES:
            raise ValueError(f"mode must be auto, local, or search, got {mode!r}")
        self.atoms = atoms
        self.seed = int(seed)
        self.budget = budget
        self.mode = mode
        self.logfile = logfile
        self.trajectory = trajectory
        self.restart = None
        self.master = None if master is None else bool(master)
        self.force_consistent = force_consistent
        self.append_trajectory = bool(append_trajectory)
        self.nsteps = 0
        self.fmax = 0.05
        self.charged = 0
        self._did_search = False
        self._observers: list[tuple] = []

    def get_number_of_steps(self) -> int:
        return self.nsteps

    def converged(self, forces=None) -> bool:
        if forces is None:
            return _max_force(self.atoms) < self.fmax
        forces = np.asarray(forces, dtype=np.float64)
        if forces.ndim == 1:
            forces = forces.reshape(-1, 3)
        return float(np.sqrt((forces * forces).sum(axis=1)).max()) < self.fmax

    def log(self, *args, **kwargs) -> None:
        if self.logfile is None:
            return
        e = float(self.atoms.get_potential_energy())
        fmax = _max_force(self.atoms)
        line = f"{self.nsteps:4d}  E = {e:12.6f}  fmax = {fmax:8.4f}\n"
        if self.logfile == "-":
            sys.stdout.write(line)
            return
        if hasattr(self.logfile, "write"):
            self.logfile.write(line)
            return
        with open(self.logfile, "a", encoding="utf-8") as handle:
            handle.write(line)

    def attach(self, function, interval: int = 1, *args, **kwargs) -> None:
        if not callable(function):
            function = getattr(function, "write", None)
        if not callable(function):
            raise TypeError("attach function must be callable")
        self._observers.append((function, int(interval), args, kwargs))

    def _call_observers(self) -> None:
        for function, interval, args, kwargs in self._observers:
            if (interval > 0 and self.nsteps % interval == 0) or (
                interval <= 0 and self.nsteps == abs(interval)
            ):
                function(*args, **kwargs)

    def _write_trajectory(self) -> None:
        if self.trajectory is None or self.master is False:
            return
        if hasattr(self.trajectory, "write"):
            self.trajectory.write(self.atoms)
            return
        from ase.io import write

        write(
            self.trajectory,
            self.atoms,
            append=self.append_trajectory or self.nsteps > 0,
        )

    def _search_budget(self, n: int) -> int:
        if self.budget is not None:
            return int(self.budget)
        return int(max(2000, 400 * n))

    def _should_search(self) -> bool:
        if self.mode == "local":
            return False
        if self.mode == "search":
            return True
        return _free_count(self.atoms) >= 4

    def _refuse_hop_geometry(self) -> None:
        if _is_cell_filter(self.atoms):
            raise ValueError("cell filters are not a free-space cluster hop")
        if _has_pbc(self.atoms):
            raise ValueError("periodic cells are not a free-space cluster hop")
        if _has_constraints(self.atoms):
            raise ValueError("constrained atoms are not a free-space cluster hop")

    def _quench(self, max_fevals: int) -> None:
        energy, grad = _callbacks(self.atoms)
        x0 = _positions(self.atoms).reshape(-1)
        low, high = _bounds_about(x0)
        out = polish(
            energy,
            grad,
            low,
            high,
            x0,
            max_fevals=int(max_fevals),
            grad_tol=float(self.fmax),
        )
        _set_positions(self.atoms, out["best_pos"])
        self.nsteps += 1
        self.log()
        self._call_observers()
        self._write_trajectory()

    def _hop(self) -> None:
        self._refuse_hop_geometry()
        n = len(self.atoms)
        energy, grad = _callbacks(self.atoms)
        start = _positions(self.atoms).reshape(-1)
        out = cluster_search(
            energy,
            grad,
            n,
            self._search_budget(n),
            seed=self.seed,
            recommended=True,
            start=start,
            length_scale=_nn_length_scale(self.atoms),
        )
        _set_positions(self.atoms, out["best"])
        self.charged += int(out.get("charged", out.get("spent", 0)) or 0)
        self._did_search = True
        self.nsteps += 1
        self.log()
        self._call_observers()
        self._write_trajectory()

    def step(self) -> None:
        """One ASE dynamics step: a hop once, then quench bursts."""
        if self._should_search() and not self._did_search:
            self._hop()
            return
        self._quench(max_fevals=40)

    def irun(self, fmax: float = 0.05, steps: int | None = None):
        """Yield after the initial residual and after each ``step``."""
        self.fmax = float(fmax)
        limit = 10**9 if steps is None else int(steps)
        self.log()
        self._call_observers()
        self._write_trajectory()
        yield self.converged()
        while not self.converged() and self.nsteps < limit:
            self.step()
            yield self.converged()

    def run(self, fmax: float = 0.05, steps: int | None = None) -> bool:
        """Match ``ase.optimize.Optimizer.run``."""
        for _ in self.irun(fmax=fmax, steps=steps):
            pass
        return self.converged()
