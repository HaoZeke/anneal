"""Shared pieces for the ASE and AGOX Lennard-Jones arms.

One hit rule, one budget map, and one set of thread pins.
A miss keeps the charged count. Nothing substitutes the paper budget.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

# Set before NumPy or Numba read the variables.
THREAD_PINS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_PROC_BIND": "TRUE",
    "OMP_PLACES": "cores",
}
for _key, _value in THREAD_PINS.items():
    os.environ[_key] = _value

import numpy as np
from numba import njit

REPO = Path(__file__).resolve().parents[2]
SEED_FILE = REPO / "experiments" / "lj_held_out_seeds.txt"

# Hard ceilings from scripts/elja_submit_jcc_lj.sh. Confirmed there:
# 38 is 400000, 55 is 1000000, 75, 98, and 104 are 4000000.
PAPER_BUDGETS = {
    38: 400_000,
    55: 1_000_000,
    75: 4_000_000,
    98: 4_000_000,
    104: 4_000_000,
}
REFERENCES = {
    38: -173.928427,
    55: -279.248470,
    75: -397.492331,
    98: -543.665361,
    104: -582.086642,
}
HIT_TOL = 1.0e-4
FORCE_BAR = 0.0514221
# Midpoint of the 0.36 to 0.40 band in Wales and Doye, J. Phys. Chem. A
# 101, 5111 (1997). The same value is LennardJonesPreset::ALL_POINTS_STEP.
WALES_STEP = 0.38
WALES_TEMPERATURE = 0.8
# Minimum pair distance used when the starting cluster is placed.
PLACEMENT_FLOOR = 0.85


class BudgetReached(BaseException):
    """The charged-call ceiling stopped the run."""


class HitReached(BaseException):
    """The shared predicate already holds on a charged geometry."""


class WallReached(BaseException):
    """The process was asked to stop before the ceiling."""


def held_out_seeds() -> list[int]:
    seeds = []
    for line in SEED_FILE.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            seeds.append(int(line))
    if seeds != [6400, 6401, 6402, 6403, 6404, 6405]:
        raise SystemExit("the frozen seed file is not 6400 through 6405")
    return seeds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@njit(cache=True)
def lj_grad(x):
    n = x.shape[0] // 3
    energy = 0.0
    grad = np.zeros(x.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[3 * i] - x[3 * j]
            dy = x[3 * i + 1] - x[3 * j + 1]
            dz = x[3 * i + 2] - x[3 * j + 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < 1.0e-12:
                continue
            inv = 1.0 / r2
            inv6 = inv * inv * inv
            inv12 = inv6 * inv6
            energy += 4.0 * (inv12 - inv6)
            coef = 24.0 * inv * (2.0 * inv12 - inv6)
            grad[3 * i] -= coef * dx
            grad[3 * i + 1] -= coef * dy
            grad[3 * i + 2] -= coef * dz
            grad[3 * j] += coef * dx
            grad[3 * j + 1] += coef * dy
            grad[3 * j + 2] += coef * dz
    return energy, grad


@njit(cache=True)
def bfgs_inverse_update(inv, step, slope):
    """In-place inverse BFGS update. inv is the approximate inverse Hessian."""
    n = step.shape[0]
    curvature = 0.0
    for i in range(n):
        curvature += step[i] * slope[i]
    if curvature < 1.0e-16 and curvature > -1.0e-16:
        return
    mapped = np.zeros(n)
    for i in range(n):
        acc = 0.0
        row = inv[i]
        for j in range(n):
            acc += row[j] * slope[j]
        mapped[i] = acc
    quad = 0.0
    for i in range(n):
        quad += slope[i] * mapped[i]
    scale = 1.0 / curvature
    grow = (curvature + quad) * scale * scale
    for i in range(n):
        row = inv[i]
        si = step[i]
        mi = mapped[i]
        for j in range(n):
            row[j] += grow * si * step[j] - scale * (si * mapped[j] + mi * step[j])


@njit(cache=True)
def bfgs_apply(inv, gradient):
    n = gradient.shape[0]
    out = np.empty(n)
    for i in range(n):
        acc = 0.0
        row = inv[i]
        for j in range(n):
            acc += row[j] * gradient[j]
        out[i] = -acc
    return out


def predicate(energy: float, force: float, calls: int, reference: float, budget: int) -> bool:
    return energy < reference + HIT_TOL and force <= FORCE_BAR and calls <= budget


def random_cluster(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    radius = (3.0 * n / (4.0 * np.pi * 0.7)) ** (1.0 / 3.0)
    points = []
    draws = 0
    while len(points) < n and draws < 200000:
        draws += 1
        vec = rng.normal(size=3)
        vec *= rng.random() ** (1.0 / 3.0) * radius / np.linalg.norm(vec)
        if all(np.linalg.norm(vec - old) >= PLACEMENT_FLOOR for old in points):
            points.append(vec)
    if len(points) < n:
        extra = rng.normal(size=(n - len(points), 3)) * (radius / 2.0)
        points.extend(list(extra))
    return np.asarray(points[:n], dtype=float)


def rattle_c1() -> float:
    """Lower bond factor so the rattle gate matches the placement floor.

    Argon's tabulated covalent diameter is far larger than one Lennard-Jones
    length. The generator default rejects every move inside a reduced cluster.
    """
    from ase.data import covalent_radii

    diameter = 2.0 * float(covalent_radii[18])
    return PLACEMENT_FLOOR / diameter


def type7_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("empty sample")
    if count == 1:
        return float(ordered[0])
    rank = 1.0 + (count - 1) * probability
    lower = int(math.floor(rank))
    weight = rank - lower
    left = ordered[lower - 1]
    if lower >= count:
        return float(left)
    right = ordered[lower]
    return float(left + weight * (right - left))


def type7_iqr(values: list[float]) -> float:
    return type7_quantile(values, 0.75) - type7_quantile(values, 0.25)


def median(values: list[float]) -> float:
    return type7_quantile(values, 0.5)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_first(path: str) -> str:
    file = Path(path)
    if not file.is_file():
        return "absent"
    return file.read_text().strip().splitlines()[0]


def git_lines() -> list[str]:
    import subprocess

    def rev(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()

    return [f"HEAD={rev('rev-parse', 'HEAD')}", f"TREE={rev('rev-parse', 'HEAD^{tree}')}"]


def host_lines() -> list[str]:
    model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = " ".join(line.split(":", 1)[1].split())
                break
    cores = os.cpu_count()
    return [
        f"host={socket.gethostname()}",
        f"machine_id={read_first('/etc/machine-id')}",
        f"boot_id={read_first('/proc/sys/kernel/random/boot_id')}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"processor_count={cores}",
        f"model={model}",
        "thread_pins=" + ",".join(f"{key}={value}" for key, value in THREAD_PINS.items()),
    ]


class Journal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **fields: object) -> None:
        import fcntl

        line = "\t".join(
            [utc_now(), kind, f"pid={os.getpid()}"]
            + [f"{key}={value}" for key, value in fields.items()]
        )
        with self.path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_xyz(path: Path, coords: np.ndarray, energy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["{}".format(len(coords)), f"energy={energy:.8f}"]
    for point in coords:
        rows.append(f"Ar {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}")
    path.write_text("\n".join(rows) + "\n")


def independent_check(coords: np.ndarray, reference: float, calls: int, budget: int):
    flat = np.ascontiguousarray(coords.ravel(), dtype=np.float64)
    energy, grad = lj_grad(flat)
    force = float(np.max(np.abs(grad)))
    ok = predicate(float(energy), force, calls, reference, budget)
    return float(energy), force, ok


class Stopwatch:
    def __init__(self):
        self.started = utc_now()
        self._t0 = time.perf_counter()
        self.ended = ""
        self.wall_s = 0.0

    def stop(self) -> None:
        self.wall_s = time.perf_counter() - self._t0
        self.ended = utc_now()
