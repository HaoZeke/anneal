# Copyright (c) anneal developers
"""Untruncated Lennard-Jones potential, the same pair term as anneal.

Energy is ``4 (r^-12 - r^-6)``. The gradient matches
``PairPotential::lennard_jones``: ``(dV/dr) / r`` times the separation.
"""

from __future__ import annotations

import time

import numpy as np
from ase.calculators.calculator import Calculator
from numba import njit


class BudgetReached(BaseException):
    """The force-call budget is spent. A BaseException so search loops do not catch it."""


class HitReached(BaseException):
    """A quenched energy is within the hit window."""


class CapReached(BaseException):
    """The wall cap for this seed is spent."""


@njit(cache=True)
def lj_value_and_gradient(x):
    n = x.shape[0] // 3
    energy = 0.0
    grad = np.zeros(x.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[3 * i] - x[3 * j]
            dy = x[3 * i + 1] - x[3 * j + 1]
            dz = x[3 * i + 2] - x[3 * j + 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0:
                continue
            inv = 1.0 / r2
            inv6 = inv * inv * inv
            inv12 = inv6 * inv6
            energy += 4.0 * (inv12 - inv6)
            coef = -24.0 * inv * (2.0 * inv12 - inv6)
            grad[3 * i] += coef * dx
            grad[3 * i + 1] += coef * dy
            grad[3 * i + 2] += coef * dz
            grad[3 * j] -= coef * dx
            grad[3 * j + 1] -= coef * dy
            grad[3 * j + 2] -= coef * dz
    return energy, grad


def warmup():
    lj_value_and_gradient(np.zeros(6))


class LjCalculator(Calculator):
    """One charge per geometry. Stops on the budget or on a quenched hit."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        budget: int,
        reference: float,
        tol: float,
        fmax: float,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        deadline: float | None = None,
    ):
        super().__init__()
        self.budget = budget
        self.reference = reference
        self.tol = tol
        self.fmax = fmax
        self.sigma = sigma
        self.epsilon = epsilon
        self.deadline = deadline
        self.calls = 0
        self.inside_s = 0.0
        self.hit_calls = None
        self.best = 1.0e300
        self.best_quenched = 1.0e300

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise CapReached()
        if self.calls >= self.budget:
            raise BudgetReached()
        positions = np.ascontiguousarray(atoms.positions.ravel(), dtype=np.float64)
        started = time.perf_counter()
        reduced_energy, grad = lj_value_and_gradient(positions / self.sigma)
        self.inside_s += time.perf_counter() - started
        self.calls += 1
        forces = (-grad * self.epsilon / self.sigma).reshape(-1, 3)
        energy = float(reduced_energy) * self.epsilon
        reduced_energy = float(reduced_energy)
        self.results = {"energy": energy, "forces": forces}
        if reduced_energy < self.best:
            self.best = reduced_energy
        peak = float(np.max(np.abs(forces))) if forces.size else 0.0
        if peak <= self.fmax and reduced_energy < self.best_quenched:
            self.best_quenched = reduced_energy
        if (
            self.hit_calls is None
            and peak <= self.fmax
            and reduced_energy <= self.reference + self.tol
        ):
            self.hit_calls = self.calls
            raise HitReached()
        if self.calls % 20000 == 0:
            print(
                f"progress calls={self.calls} best={self.best:.6f}",
                flush=True,
            )


class AnnealServer:
    """One persistent ``asg_eval serve`` process.

    Stdin is a geometry. Stdout is the energy, the seconds inside the
    potential, and the gradient. The potential is the one anneal evaluates
    for that system.
    """

    def __init__(self, binary: str, system: str):
        import subprocess

        self.proc = subprocess.Popen(
            [binary, "serve", system],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def eval(self, coords: np.ndarray):
        if self.proc.poll() is not None:
            raise RuntimeError(f"asg_eval exited {self.proc.returncode}")
        text = " ".join(f"{float(value):.16e}" for value in coords)
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("asg_eval closed stdout")
        values = np.fromstring(line, sep=" ")
        if values.size < 2:
            raise RuntimeError(f"asg_eval reply {line!r}")
        return float(values[0]), float(values[1]), values[2:]

    def close(self):
        if self.proc.poll() is None and self.proc.stdin is not None:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
            except BrokenPipeError:
                pass
            self.proc.wait(timeout=5)


class AnnealCalculator(Calculator):
    """ASE view of an anneal potential.

    ``scale`` is 1 for basin hopping and minima hopping, which stay in the
    potential's own units. GOFEE passes the factor that puts the pair
    minimum on the Angstrom length its repulsive prior expects. The energy
    and the hit are always the native ones.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        server: AnnealServer,
        budget: int,
        reference: float,
        tol: float,
        fmax: float,
        scale: float = 1.0,
        deadline: float | None = None,
    ):
        super().__init__()
        self.server = server
        self.budget = budget
        self.reference = reference
        self.tol = tol
        self.fmax = fmax
        self.scale = scale
        self.deadline = deadline
        self.calls = 0
        self.inside_s = 0.0
        self.hit_calls = None
        self.best = 1.0e300
        self.best_quenched = 1.0e300

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise CapReached()
        if self.calls >= self.budget:
            raise BudgetReached()
        native = np.ascontiguousarray(atoms.positions.ravel(), dtype=np.float64) / self.scale
        energy, inside, grad = self.server.eval(native)
        self.inside_s += inside
        self.calls += 1
        if not np.isfinite(energy) or not np.all(np.isfinite(grad)):
            energy = 1.0e8
            grad = np.zeros_like(native)
            native_energy = 1.0e300
        else:
            native_energy = float(energy)
        # pos = native * scale and E_ase = E_native * scale, so the force
        # ASE sees is the native gradient.
        forces = (-np.asarray(grad, dtype=np.float64)).reshape(-1, 3)
        self.results = {"energy": native_energy * self.scale, "forces": forces}
        if native_energy < self.best:
            self.best = native_energy
        peak = float(np.max(np.abs(forces))) if forces.size else 0.0
        if peak <= self.fmax and native_energy < self.best_quenched:
            self.best_quenched = native_energy
        if (
            self.hit_calls is None
            and peak <= self.fmax
            and native_energy <= self.reference + self.tol
        ):
            self.hit_calls = self.calls
            raise HitReached()
        if self.calls % 20000 == 0:
            print(
                f"progress calls={self.calls} best={self.best:.6f}",
                flush=True,
            )
