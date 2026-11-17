"""Threaded energy and force calls at the minimum-image sizes.

``threads`` is applied with ``omp_set_num_threads`` before the geometry
is built, so the same file runs on a build that ignores the count and
on one that folds pairs across that many threads. asv does not time or
track ``setup``.
"""

from __future__ import annotations

import ctypes
import math
import os

import numpy as np

REPEAT = 6
ATOMS = (256, 1024, 4096)

_DIAMOND = (
    (0.00, 0.00, 0.00),
    (0.50, 0.50, 0.00),
    (0.50, 0.00, 0.50),
    (0.00, 0.50, 0.50),
    (0.25, 0.25, 0.25),
    (0.75, 0.75, 0.25),
    (0.75, 0.25, 0.75),
    (0.25, 0.75, 0.75),
)
_FCC = (
    (0.00, 0.00, 0.00),
    (0.50, 0.50, 0.00),
    (0.50, 0.00, 0.50),
    (0.00, 0.50, 0.50),
)


def _sibling_key(cpu: int) -> str:
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return str(cpu)


def _exclusive_cpus() -> list[int]:
    """One logical CPU per physical core inside the current affinity.

    CPUs that share a ``thread_siblings_list`` are SMT siblings. Only
    the first of each list is offered to a team.
    """
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = list(range(os.cpu_count() or 1))
    chosen = []
    seen = set()
    for cpu in allowed:
        key = _sibling_key(cpu)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(cpu)
    return chosen


def _thread_counts() -> tuple[int, ...]:
    """Counts that fit one core per thread after the parent keeps one."""
    room = max(0, len(_exclusive_cpus()) - 1)
    return tuple(count for count in (1, 2, 4, 8) if count <= room)


THREADS = _thread_counts()


def _parent_core() -> int:
    cpus = _exclusive_cpus()
    return cpus[0] if cpus else 0


def _worker_cores(threads: int) -> list[int]:
    """N exclusive cores for a team of N. The parent keeps a different one.

    The same count always receives the same cores, on every arm. When
    the exclusive CPUs are 0 through 7, one thread uses 1, two use 2-3
    and four use 4-7. A count that does not leave the parent a core of
    its own is not a benchmark parameter.
    """
    cpus = _exclusive_cpus()
    pool = cpus[1:]
    count = int(threads)
    if count > len(pool) or count < 1:
        raise RuntimeError(
            f"{count} threads need {count} cores and the parent needs one more"
        )
    if cpus[:8] == list(range(8)):
        if count == 1:
            return [1]
        if count == 2:
            return [2, 3]
        if count == 4:
            return [4, 5, 6, 7]
    return pool[:count]


def band_wider_than_effect(ratio: float, low: float, high: float) -> bool:
    """True when the develop-vs-develop band is wider than the effect.

    A row in that state is a failure and is not reported as a ratio.
    """
    if ratio is None or low is None or high is None:
        return True
    return abs(high - low) > abs(ratio - 1.0)


def _pin_cores(threads: int) -> None:
    """Pin this process to ``_worker_cores`` and the parent to the rest."""
    worker = _worker_cores(threads)
    ncpu = os.cpu_count() or 1
    other = [cpu for cpu in range(ncpu) if cpu not in worker] or [0]
    try:
        os.sched_setaffinity(0, set(worker))
        os.sched_setaffinity(os.getppid(), set(other))
    except (AttributeError, OSError, ProcessLookupError):
        pass
    os.environ["OMP_NUM_THREADS"] = str(int(threads))
    os.environ["OMP_DYNAMIC"] = "false"
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = ",".join(f"{{{cpu}}}" for cpu in worker)


def _set_threads(threads: int) -> None:
    _pin_cores(threads)
    lib = ctypes.CDLL("libgomp.so.1")
    lib.omp_set_dynamic(0)
    lib.omp_set_num_threads(int(threads))


def _lattice(atoms: int, lattice: float, basis, atomic_number: int):
    per_cell = len(basis)
    ncell = 1
    while per_cell * ncell * ncell * ncell < atoms:
        ncell += 1
    coords = []
    for ix in range(ncell):
        for iy in range(ncell):
            for iz in range(ncell):
                for site in basis:
                    if len(coords) >= atoms:
                        break
                    coords.append(
                        (
                            (ix + site[0]) * lattice,
                            (iy + site[1]) * lattice,
                            (iz + site[2]) * lattice,
                        )
                    )
                if len(coords) >= atoms:
                    break
            if len(coords) >= atoms:
                break
        if len(coords) >= atoms:
            break
    positions = np.ascontiguousarray(coords, dtype=np.float64)
    atom_types = np.full(atoms, atomic_number, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * (ncell * lattice)
    return positions, atom_types, box


def _diamond(atoms: int):
    return _lattice(atoms, 5.431, _DIAMOND, 14)


def _aluminium(atoms: int):
    return _lattice(atoms, 4.05, _FCC, 13)


def _lj_cluster(atoms: int):
    side = 1
    while side * side * side < atoms:
        side += 1
    spacing = 2.0 ** (1.0 / 6.0)
    coords = []
    for index in range(side * side * side):
        if len(coords) == atoms:
            break
        x = index % side
        y = (index // side) % side
        z = index // (side * side)
        coords.append((x * spacing, y * spacing, z * spacing))
    positions = np.ascontiguousarray(coords, dtype=np.float64)
    atom_types = np.ones(atoms, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 80.0
    return positions, atom_types, box


def _energy_force(pot, positions, atom_types, box) -> None:
    energy, forces, _variance = pot(positions, atom_types, box)
    if getattr(forces, "shape", None) != positions.shape:
        raise AssertionError(getattr(forces, "shape", None))
    if not math.isfinite(float(energy)):
        raise AssertionError("energy is not finite")


def _loops(atoms: int) -> int:
    """Calls per sample so the timed stretch is at least about 20 ms."""
    if atoms >= 4096:
        return 12
    if atoms >= 1024:
        return 40
    return 200


def _escape(source: str) -> str:
    return source.replace("{", "{{").replace("}", "}}")


_CHILD = r'''
import os
cores = set(int(part) for part in os.environ["BENCH_CORES"].split(",") if part)
os.sched_setaffinity(0, cores)
import math
import numpy as np
import rgpot

DIAMOND = (
    (0.00, 0.00, 0.00),
    (0.50, 0.50, 0.00),
    (0.50, 0.00, 0.50),
    (0.00, 0.50, 0.50),
    (0.25, 0.25, 0.25),
    (0.75, 0.75, 0.25),
    (0.75, 0.25, 0.75),
    (0.25, 0.75, 0.75),
)
FCC = (
    (0.00, 0.00, 0.00),
    (0.50, 0.50, 0.00),
    (0.50, 0.00, 0.50),
    (0.00, 0.50, 0.50),
)


def lattice(atoms, lattice_a, basis, atomic_number):
    per_cell = len(basis)
    ncell = 1
    while per_cell * ncell * ncell * ncell < atoms:
        ncell += 1
    coords = []
    for ix in range(ncell):
        for iy in range(ncell):
            for iz in range(ncell):
                for site in basis:
                    if len(coords) >= atoms:
                        break
                    coords.append(
                        (
                            (ix + site[0]) * lattice_a,
                            (iy + site[1]) * lattice_a,
                            (iz + site[2]) * lattice_a,
                        )
                    )
                if len(coords) >= atoms:
                    break
            if len(coords) >= atoms:
                break
        if len(coords) >= atoms:
            break
    positions = np.ascontiguousarray(coords, dtype=np.float64)
    atom_types = np.full(atoms, atomic_number, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * (ncell * lattice_a)
    return positions, atom_types, box


def lj_cluster(atoms):
    side = 1
    while side * side * side < atoms:
        side += 1
    spacing = 2.0 ** (1.0 / 6.0)
    coords = []
    for index in range(side * side * side):
        if len(coords) == atoms:
            break
        x = index % side
        y = (index // side) % side
        z = index // (side * side)
        coords.append((x * spacing, y * spacing, z * spacing))
    positions = np.ascontiguousarray(coords, dtype=np.float64)
    atom_types = np.ones(atoms, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 80.0
    return positions, atom_types, box


def build(kind, atoms):
    if kind == "LJCluster":
        return lj_cluster(atoms), rgpot.LJClusterPot(cutoff=15.0)
    if kind == "LJ":
        # Nearest neighbour at the Lennard-Jones minimum. Cutoff 2 A
        # plus the kernel's 1 A skin puts five bins on each axis of the
        # 4096-atom cube, which is where the orthogonal reject used to run.
        spacing = 2.0 ** (1.0 / 6.0) * math.sqrt(2.0)
        return lattice(atoms, spacing, FCC, 1), rgpot.LJPot(cutoff=2.0)
    if kind == "Morse":
        # Cutoff 6 A on the 4.05 A FCC cube. The 4096-atom cell then has
        # six bins on each axis.
        return lattice(atoms, 4.05, FCC, 1), rgpot.MorsePot(cutoff=6.0)
    if kind == "EAM":
        pot = rgpot.EAMAlPot
        if pot is None:
            raise SystemExit("this build has no stored EAM potential")
        return lattice(atoms, 4.05, FCC, 13), pot()
    if kind == "StillingerWeber":
        return lattice(atoms, 5.431, DIAMOND, 14), rgpot.StillingerWeberSi()
    if kind == "Tersoff":
        return lattice(atoms, 5.431, DIAMOND, 14), rgpot.TersoffSiC()
    raise SystemExit("unknown kernel %s" % kind)


def call(pot, positions, atom_types, box):
    energy, forces, _variance = pot(positions, atom_types, box)
    if getattr(forces, "shape", None) != positions.shape:
        raise SystemExit("bad force shape")
    if not math.isfinite(float(energy)):
        raise SystemExit("energy is not finite")


def _cpus(token):
    found = set()
    for part in token.split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            found.update(range(int(lo), int(hi) + 1))
        else:
            found.add(int(part))
    return found


def cores_collapsed(requested):
    # A short call may occupy only the first core of its set. That is
    # not a collapsed team. Fail when a thread leaves the set, or when
    # several bound threads share one core.
    wanted = set(int(part) for part in os.environ["BENCH_CORES"].split(",") if part)
    singles = []
    for name in os.listdir("/proc/self/task"):
        status = open("/proc/self/task/%s/status" % name, encoding="ascii", errors="replace").read()
        for line in status.splitlines():
            if not line.startswith("Cpus_allowed_list:"):
                continue
            allowed = _cpus(line.split()[-1])
            if not allowed or not allowed.issubset(wanted):
                return True
            if len(allowed) == 1:
                singles.append(next(iter(allowed)))
    if requested > 1 and len(singles) >= 2 and len(set(singles)) == 1:
        return True
    return False
'''


def _timeraw_pair(kind: str, atoms: int, threads: int) -> tuple[str, str]:
    setup = _escape(_CHILD) + "\n" + _escape(
        f"""
kind = {kind!r}
atoms = {int(atoms)}
requested = {int(threads)}
pot_built = build(kind, atoms)
positions, atom_types, box = pot_built[0]
pot = pot_built[1]
for _ in range(3):
    call(pot, positions, atom_types, box)
if cores_collapsed(requested):
    raise SystemExit("team collapsed off its exclusive cores")
"""
    )
    stmt = _escape("call(pot, positions, atom_types, box)\n")
    return stmt, setup


class _Timed:
    params = (list(ATOMS), list(THREADS))
    param_names = ["atoms", "threads"]
    repeat = REPEAT
    sample_time = 0.02
    min_run_count = REPEAT
    timeout = 600
    rounds = 20

    kind = ""

    def setup(self, atoms: int, threads: int) -> None:
        worker = _worker_cores(int(threads))
        # The child is born with these variables, before the runtime starts.
        self.env = {
            "OMP_NUM_THREADS": str(int(threads)),
            "OMP_DYNAMIC": "false",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": ",".join(f"{{{cpu}}}" for cpu in worker),
            "BENCH_CORES": ",".join(str(cpu) for cpu in worker),
        }
        self.number = _loops(int(atoms))
        # This process only launches the timed child. It stays on the
        # core reserved for the parent, off the team's cores.
        try:
            os.sched_setaffinity(0, {_parent_core()})
        except (AttributeError, OSError):
            pass

    def timeraw_energy_force(self, atoms: int, threads: int) -> tuple[str, str]:
        return _timeraw_pair(self.kind, int(atoms), int(threads))


class _Tracked:
    params = (list(ATOMS), list(THREADS))
    param_names = ["atoms", "threads"]
    number = 1
    repeat = REPEAT
    timeout = 600

    def ray_energy_force(self, atoms: int, threads: int) -> None:
        _energy_force(self.pot, self.positions, self.atom_types, self.box)


def _prepare(self, atoms: int, threads: int, geometry, pot) -> None:
    _set_threads(threads)
    self.positions, self.atom_types, self.box = geometry(atoms)
    self.pot = pot
    _energy_force(self.pot, self.positions, self.atom_types, self.box)


class LJCluster(_Timed):
    """Lennard-Jones cluster at 256, 1024 and 4096 atoms."""

    kind = "LJCluster"


class LJ(_Timed):
    """Periodic Lennard-Jones FCC cell at 256, 1024 and 4096 atoms."""

    kind = "LJ"


class RayLJCluster(_Tracked):
    """Memray peak of one threaded Lennard-Jones cluster call."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        _prepare(self, atoms, threads, _lj_cluster, rgpot.LJClusterPot(cutoff=15.0))


class StillingerWeber(_Timed):
    """Stillinger-Weber silicon at 256, 1024 and 4096 atoms."""

    kind = "StillingerWeber"


class RayStillingerWeber(_Tracked):
    """Memray peak of one threaded Stillinger-Weber call."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        _prepare(self, atoms, threads, _diamond, rgpot.StillingerWeberSi())


class Tersoff(_Timed):
    """C++ Tersoff silicon-carbon kernel at 256, 1024 and 4096 atoms."""

    kind = "Tersoff"


class Morse(_Timed):
    """Periodic Morse FCC cell at 256, 1024 and 4096 atoms."""

    kind = "Morse"


class RayTersoff(_Tracked):
    """Memray peak of one threaded Tersoff call."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        _prepare(self, atoms, threads, _diamond, rgpot.TersoffSiC())


class StoredEAM(_Timed):
    """Stored aluminium EAM on a periodic FCC cell."""

    kind = "EAM"


class RayStoredEAM(_Tracked):
    """Memray peak of one threaded stored-EAM call."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        if rgpot.EAMAlPot is None:
            raise RuntimeError("this build has no stored EAM potential")
        _prepare(self, atoms, threads, _aluminium, rgpot.EAMAlPot())
