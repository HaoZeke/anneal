# Copyright (c) anneal developers
"""Untruncated Lennard-Jones potential, the same pair term as anneal.

Energy is ``4 (r^-12 - r^-6)``. The gradient matches
``PairPotential::lennard_jones``: ``(dV/dr) / r`` times the separation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator
from numba import njit

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from target_tol import energy_hits  # noqa: E402


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
    gupta_value_and_gradient(np.array([0.0, 0.0, 0.0, 2.8, 0.0, 0.0]), np.zeros(2, dtype=np.int64))


# Cleri–Rosato Cu–Au rows, the same numbers as `src/potentials/gupta.rs`.
# Order is Cu–Cu, Cu–Au, Au–Cu, Au–Au. Each row is A, xi, p, q, r0.
_GUPTA = np.array(
    [
        [0.0855, 1.224, 10.960, 2.278, 2.556],
        [0.1539, 1.5605, 11.05, 3.0475, 2.556],
        [0.1539, 1.5605, 11.05, 3.0475, 2.556],
        [0.2061, 1.790, 10.229, 4.036, 2.884],
    ],
    dtype=np.float64,
)


@njit(cache=True)
def gupta_value_and_gradient(x, slot):
    """All-pairs Gupta energy and gradient. `slot` is 0 for Cu and 1 for Au."""
    n = x.shape[0] // 3
    energy = 0.0
    rho = np.zeros(n)
    # i, j, dx, dy, dz, r, dphi_dr, dpsi_dr
    pairs_i = np.empty(n * (n - 1) // 2, dtype=np.int64)
    pairs_j = np.empty(n * (n - 1) // 2, dtype=np.int64)
    stored = np.empty((n * (n - 1) // 2, 6))
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[3 * i] - x[3 * j]
            dy = x[3 * i + 1] - x[3 * j + 1]
            dz = x[3 * i + 2] - x[3 * j + 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0:
                return np.inf, np.zeros(x.shape[0])
            row = slot[i] * 2 + slot[j]
            a = _GUPTA[row, 0]
            xi = _GUPTA[row, 1]
            p = _GUPTA[row, 2]
            q = _GUPTA[row, 3]
            r0 = _GUPTA[row, 4]
            r = np.sqrt(r2)
            u = r / r0 - 1.0
            phi = 2.0 * a * np.exp(-p * u)
            psi = xi * xi * np.exp(-2.0 * q * u)
            energy += phi
            rho[i] += psi
            rho[j] += psi
            pairs_i[count] = i
            pairs_j[count] = j
            stored[count, 0] = dx
            stored[count, 1] = dy
            stored[count, 2] = dz
            stored[count, 3] = r
            stored[count, 4] = -(p / r0) * phi
            stored[count, 5] = -(2.0 * q / r0) * psi
            count += 1
    for i in range(n):
        energy -= np.sqrt(rho[i])
    grad = np.zeros(x.shape[0])
    for index in range(count):
        i = pairs_i[index]
        j = pairs_j[index]
        embed = -0.5 * (1.0 / np.sqrt(rho[i]) + 1.0 / np.sqrt(rho[j]))
        coef = (stored[index, 4] + embed * stored[index, 5]) / stored[index, 3]
        grad[3 * i] += coef * stored[index, 0]
        grad[3 * i + 1] += coef * stored[index, 1]
        grad[3 * i + 2] += coef * stored[index, 2]
        grad[3 * j] -= coef * stored[index, 0]
        grad[3 * j + 1] -= coef * stored[index, 1]
        grad[3 * j + 2] -= coef * stored[index, 2]
    return energy, grad


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
        self.evaluate = None
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
        if self.evaluate is None:
            reduced_energy, grad = lj_value_and_gradient(positions / self.sigma)
            forces = (-grad * self.epsilon / self.sigma).reshape(-1, 3)
            energy = float(reduced_energy) * self.epsilon
        else:
            reduced_energy, grad = self.evaluate(positions)
            forces = (-grad).reshape(-1, 3)
            energy = float(reduced_energy)
        self.inside_s += time.perf_counter() - started
        self.calls += 1
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
            and energy_hits(reduced_energy, self.reference, self.tol)
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
            and energy_hits(native_energy, self.reference, self.tol)
        ):
            self.hit_calls = self.calls
            raise HitReached()
        if self.calls % 20000 == 0:
            print(
                f"progress calls={self.calls} best={self.best:.6f}",
                flush=True,
            )
