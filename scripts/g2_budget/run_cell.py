#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run one pre-registered G2 budget-stop cell.

The call budget and the process-CPU cap come from the protocol module.
ASE minima hopping and basin hopping add the evaluator child's user
and system time to the parent. The shape keeps both parts. A seed that
already has a row is not run again. A capped seed that stopped short
of the pre-registered cap continues from its checkpoint. An anneal
search keeps one worker process across those segments.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import argparse
import hashlib
import importlib.util
import json
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREREG_REVISION = "1308d8678a8912de95ded423291a25430bb073fc"
_THREAD_LIMIT = None
_PANEL = None
_POTENTIAL = None
_GUPTA = None
_SCALES = None


def _load(name: str, path: Path):
    loaded = sys.modules.get(name)
    if loaded is not None and Path(loaded.__file__).resolve() == path.resolve():
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _protocol():
    return _load("g2_budget_protocol", Path(__file__).resolve().with_name("protocol.py"))


def _rows():
    return _load("g2_budget_rows", Path(__file__).resolve().with_name("rows.py"))


protocol = _protocol()
rows = _rows()


def segment_cap_s(cell) -> float | None:
    """Process-CPU budget of one invocation.

    Production reads the pre-registered cap and nothing else. A test may
    replace this function; the replacement is not a command-line default.
    """
    cap = protocol.cpu_cap_s(cell.system, cell.arm)
    if cap is None:
        return None
    return float(cap)


def checkpoint_path(root, system: str, arm: str, seed: int) -> Path:
    return (
        Path(root)
        / "bench"
        / "g2-budget"
        / "work"
        / system
        / arm
        / f"seed{int(seed)}"
        / "checkpoint.json"
    )


def _libraries():
    global _PANEL, _POTENTIAL
    if _PANEL is not None:
        return _PANEL, _POTENTIAL
    panel_dir = ROOT / "scripts" / "ase_panel"
    potential_dir = ROOT / "scripts" / "ase_gofee_h2h"
    for entry in (str(potential_dir), str(panel_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import potential

    panel = _load("ase_panel_run_one", panel_dir / "run_one.py")
    _PANEL = panel
    _POTENTIAL = potential
    return panel, potential


def _limit_threads() -> None:
    global _THREAD_LIMIT
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        import numba

        numba.set_num_threads(1)
    except ImportError:
        pass
    if _THREAD_LIMIT is None:
        try:
            import threadpoolctl

            _THREAD_LIMIT = threadpoolctl.threadpool_limits(limits=1)
            _THREAD_LIMIT.__enter__()
        except ImportError:
            _THREAD_LIMIT = False


def _pin() -> int:
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("this process has no CPU")
    cpu = int(allowed[0])
    os.sched_setaffinity(0, {cpu})
    return cpu


def _model_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("cpu model is missing")


def _machine_id_sha256() -> str:
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if path.is_file():
            text = path.read_text().strip()
            if text:
                return hashlib.sha256(text.encode()).hexdigest()
    raise RuntimeError("machine id is missing")


def _runner_revision() -> str:
    text = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if len(text) != 40:
        raise RuntimeError("runner revision is not a commit")
    return text


def _shape_extra(cpu: int) -> dict:
    count = os.cpu_count()
    if count is None:
        raise RuntimeError("cpu count is missing")
    return {
        "runner_revision": _runner_revision(),
        "nproc": int(count),
        "model_name": _model_name(),
        "machine_id_sha256": _machine_id_sha256(),
        "core_affinity": int(cpu),
    }


def _stored_row(root: Path, cell) -> dict | None:
    _text, found = rows._read_rows(protocol.rows_path(root, cell.system))
    matches = [
        item
        for item in found
        if item["system"] == cell.system
        and item["arm"] == cell.arm
        and item["seed"] == str(cell.seed)
    ]
    if len(matches) > 1:
        raise RuntimeError("seed has more than one row")
    return matches[0] if matches else None


def _write_stamp(cell, stamp: Path) -> None:
    text = json.dumps(protocol.cell_record(cell), indent=2, sort_keys=True) + "\n"
    rows._atomic_write(Path(stamp), text.encode())


def _dump_rng(rng) -> list:
    kind, keys, pos, has_gauss, cached = rng.get_state()
    return [
        str(kind),
        [int(value) for value in keys],
        int(pos),
        int(has_gauss),
        float(cached),
    ]


def _load_rng(payload):
    kind, keys, pos, has_gauss, cached = payload
    rng = np.random.RandomState(0)
    rng.set_state(
        (
            str(kind),
            np.asarray(keys, dtype=np.uint32),
            int(pos),
            int(has_gauss),
            float(cached),
        )
    )
    return rng


def _rng_token(rng) -> tuple:
    kind, keys, pos, has_gauss, cached = rng.get_state()
    return (str(kind), keys.tobytes(), int(pos), int(has_gauss), float(cached))


def _load_checkpoint(root: Path, cell) -> dict | None:
    path = checkpoint_path(root, cell.system, cell.arm, cell.seed)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _write_checkpoint(root: Path, cell, outcome: dict, segments: list[dict]) -> None:
    payload = {
        "calls": int(outcome["calls"]),
        "best_e": float(outcome["best_e"]),
        "cpu_s": float(outcome["cpu_s"]),
        "wall_s": float(outcome["wall_s"]),
        "segments": len(segments),
        "prior": segments,
        "positions": outcome["positions"],
        "completed": int(outcome.get("completed", 0)),
    }
    if outcome.get("rng_state") is not None:
        payload["rng_state"] = outcome["rng_state"]
    if outcome.get("worker_pid") is not None:
        payload["worker_pid"] = int(outcome["worker_pid"])
    if outcome.get("trajectory") is not None:
        payload["trajectory"] = str(outcome["trajectory"])
    if outcome.get("segment") is not None:
        payload["segment"] = int(outcome["segment"])
    path = checkpoint_path(root, cell.system, cell.arm, cell.seed)
    rows._atomic_write(path, (json.dumps(payload) + "\n").encode())


def _best_energy(calc, previous: float | None) -> float:
    values = []
    if calc.best_quenched < 1.0e299:
        values.append(float(calc.best_quenched))
    if calc.best < 1.0e299:
        values.append(float(calc.best))
    if previous is not None and math.isfinite(previous):
        values.append(float(previous))
    if not values:
        raise RuntimeError("the search did not record an energy")
    return min(values)


def _positions(values, n: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(n, 3)
    return np.ascontiguousarray(array)


def _ase_setup(cell, panel, potential, state):
    import ase
    from ase import Atoms

    if ase.__version__ != panel.ASE_VERSION:
        raise RuntimeError(f"ASE {panel.ASE_VERSION} required, found {ase.__version__}")
    spec = panel.SYSTEMS[cell.system]
    server = None
    if spec["kind"] == "lj":
        panel.check_lj()
        potential.warmup()
        length = 1.0
        symbol = spec.get("symbol", "Ar")
        calc = potential.LjCalculator(
            int(cell.budget),
            spec["reference"],
            panel.HIT_TOL,
            panel.FMAX[cell.arm],
        )
    else:
        binary = panel.asg_eval_binary()
        n_atoms, paper_budget, reference, length = panel.evaluator_spec(binary, cell.system)
        if (n_atoms, paper_budget) != (spec["n"], spec["budget"]) or reference != spec["reference"]:
            raise RuntimeError("asg_eval spec differs from the panel")
        symbol = spec["symbol"]
        server = potential.AnnealServer(binary, cell.system)
        calc = potential.AnnealCalculator(
            server,
            int(cell.budget),
            spec["reference"],
            panel.HIT_TOL,
            panel.FMAX[cell.arm],
        )
    previous = None if state is None else float(state["best_e"])
    if state is not None:
        calc.calls = int(state["calls"])
        if previous is not None and math.isfinite(previous):
            calc.best = previous
            calc.best_quenched = previous
        positions = _positions(state["positions"], spec["n"])
    else:
        positions = panel.place(spec["n"], int(cell.seed), float(length))
    atoms = Atoms(f"{symbol}{spec['n']}", positions=positions, pbc=False)
    atoms.calc = calc
    return spec, atoms, calc, server, previous


def _outcome(stop: str, calls: int, best: float, cpu_s: float, wall_s: float, **extra) -> dict:
    payload = {
        "stop": stop,
        "calls": int(calls),
        "best_e": float(best),
        "cpu_s": float(cpu_s),
        "wall_s": float(wall_s),
    }
    payload.update(extra)
    return payload


def _process_cpu_s(pid: int) -> float:
    """User plus system CPU seconds from ``/proc/<pid>/stat``."""
    text = Path(f"/proc/{int(pid)}/stat").read_text()
    end = text.rfind(")")
    if end < 0 or end + 2 >= len(text):
        raise RuntimeError(f"process {pid} has no stat")
    fields = text[end + 2:].split()
    if len(fields) < 13:
        raise RuntimeError(f"process {pid} stat is short")
    ticks = int(fields[11]) + int(fields[12])
    return ticks / float(os.sysconf("SC_CLK_TCK"))


class _SearchCpu:
    """Parent process CPU plus one live evaluator child's user and system CPU.

    A search with no child records zero. The child is read while it is
    running, so a cap sees the same sum that is recorded.
    """

    def __init__(self, pid: int | None) -> None:
        self.pid = None if pid is None else int(pid)
        self._parent0 = time.process_time()
        self._wall0 = time.perf_counter()
        self._child0 = 0.0 if self.pid is None else _process_cpu_s(self.pid)

    def snapshot(self) -> tuple[float, float, float]:
        parent = time.process_time() - self._parent0
        if parent < 0.0:
            parent = 0.0
        if self.pid is None:
            child = 0.0
        else:
            child = _process_cpu_s(self.pid) - self._child0
            if child < 0.0:
                child = 0.0
        wall = time.perf_counter() - self._wall0
        total = parent + child
        if wall < total:
            wall = total
        return parent, child, wall


def _raise_on_cap(calc, meter: _SearchCpu, cap_s: float, potential) -> None:
    original = calc.calculate

    def calculate(atoms=None, properties=None, system_changes=None):
        parent, child, _wall = meter.snapshot()
        if calc.calls > 0 and parent + child >= cap_s:
            raise potential.CapReached()
        return original(atoms, properties, system_changes)

    calc.calculate = calculate


def _run_bh(cell, cap_s, state) -> dict:
    if state is not None:
        raise RuntimeError("basin hopping is budget-stopped in one invocation")
    panel, potential = _libraries()
    from ase.optimize.basin import BasinHopping

    _spec, atoms, calc, server, previous = _ase_setup(cell, panel, potential, None)
    np.random.seed(int(cell.seed))
    meter = _SearchCpu(None if server is None else server.proc.pid)
    if cap_s is not None:
        _raise_on_cap(calc, meter, float(cap_s), potential)
    stop = None
    try:
        try:
            BasinHopping(
                atoms,
                logfile=None,
                optimizer_logfile=None,
                trajectory=None,
                local_minima_trajectory=None,
            ).run(10**9)
        except potential.BudgetReached:
            stop = "budget"
        except potential.HitReached:
            stop = "hit"
        except potential.CapReached:
            stop = "cap"
        if stop is None:
            if calc.hit_calls is not None:
                stop = "hit"
            elif calc.calls >= cell.budget:
                stop = "budget"
            else:
                raise RuntimeError("basin hopping stopped before the budget")
        calls = int(calc.hit_calls if stop == "hit" and calc.hit_calls is not None else calc.calls)
        if stop == "cap" and calls >= int(cell.budget):
            stop = "budget"
        if stop == "budget" and calls != int(cell.budget):
            raise RuntimeError(f"budget stop recorded {calls} calls")
        parent, child, wall = meter.snapshot()
        return _outcome(
            stop,
            calls,
            _best_energy(calc, previous),
            parent + child,
            wall,
            positions=atoms.get_positions().reshape(-1).tolist(),
            parent_cpu_s=parent,
            child_cpu_s=child,
        )
    finally:
        if server is not None:
            server.close()


def _run_mh(cell, root: Path, cap_s, state) -> dict:
    panel, potential = _libraries()
    from ase.optimize.minimahopping import MinimaHopping

    work = checkpoint_path(root, cell.system, cell.arm, cell.seed).parent
    work.mkdir(parents=True, exist_ok=True)
    previous_dir = Path.cwd()
    server = None
    try:
        os.chdir(work)
        _spec, atoms, calc, server, previous = _ase_setup(cell, panel, potential, state)
        if state is None:
            rng = np.random.RandomState(int(cell.seed))
            completed = 0
        else:
            rng = _load_rng(state["rng_state"])
            completed = int(state["completed"])
        meter = _SearchCpu(None if server is None else server.proc.pid)
        stop = None
        while stop is None:
            if calc.hit_calls is not None:
                stop = "hit"
                break
            if calc.calls >= int(cell.budget):
                stop = "budget"
                break
            token = _rng_token(rng)
            calls_before = int(calc.calls)
            try:
                MinimaHopping(
                    atoms,
                    logfile="hop.log",
                    minima_traj="minima.traj",
                    fmax=panel.FMAX["ase_mh"],
                    rng=rng,
                )(totalsteps=completed + 1)
            except potential.HitReached:
                stop = "hit"
            except potential.BudgetReached:
                stop = "budget"
            else:
                completed += 1
                moved = _rng_token(rng) != token
                parent, child, _wall = meter.snapshot()
                if cap_s is not None and parent + child >= float(cap_s) and moved:
                    stop = "cap"
                elif not moved and int(calc.calls) == calls_before:
                    raise RuntimeError("minima hopping made no progress")
        calls = int(calc.hit_calls if stop == "hit" and calc.hit_calls is not None else calc.calls)
        if stop == "cap" and calls >= int(cell.budget):
            stop = "budget"
        if stop == "budget" and calls != int(cell.budget):
            raise RuntimeError(f"budget stop recorded {calls} calls")
        parent, child, wall = meter.snapshot()
        return _outcome(
            stop,
            calls,
            _best_energy(calc, previous),
            parent + child,
            wall,
            positions=atoms.get_positions().reshape(-1).tolist(),
            rng_state=_dump_rng(rng),
            completed=completed,
            parent_cpu_s=parent,
            child_cpu_s=child,
        )
    finally:
        os.chdir(previous_dir)
        if server is not None:
            server.close()


def _classify(exc: BaseException) -> str | None:
    current = exc
    seen = []
    while current is not None and current not in seen:
        seen.append(current)
        name = type(current).__name__
        if name == "BudgetReached":
            return "budget"
        if name == "HitReached":
            return "hit"
        if name == "CapReached":
            return "cap"
        current = current.__cause__ or current.__context__
    return None


def _gupta():
    global _GUPTA
    if _GUPTA is None:
        metal = ROOT / "scripts" / "metal_h2h"
        entry = str(metal)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        import gupta_calc

        _GUPTA = gupta_calc
    return _GUPTA


def gupta_pair_scales() -> tuple[float, float]:
    """Panel length and well: r_min / 2^(1/6) and -E at the Gupta minimum."""
    global _SCALES
    if _SCALES is None:
        gupta = _gupta()
        numbers = np.array([79, 79], dtype=int)
        best_r = 0.0
        best_e = math.inf
        distance = 1.8
        while distance <= 4.5:
            xyz = np.zeros((2, 3), dtype=np.float64)
            xyz[1, 0] = distance
            energy, _grad = gupta.evaluate(numbers, xyz)
            if energy < best_e:
                best_e = float(energy)
                best_r = float(distance)
            distance += 0.002
        length = best_r / 2.0 ** (1.0 / 6.0)
        well = -best_e
        if not (math.isfinite(length) and math.isfinite(well) and length > 0.0 and well > 0.0):
            raise RuntimeError("gupta scales are not positive")
        _SCALES = (length, well)
    return _SCALES


def au55_scaled_energy_gradient(coordinates):
    """Gupta energy and gradient in the panel's reduced units.

    The search coordinate ``x`` is evaluated at ``x * length``. Energy is
    divided by the well and the Cartesian gradient is multiplied by
    ``length / well``.
    """
    length, well = gupta_pair_scales()
    xyz = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    if xyz.shape != (55, 3):
        raise ValueError("au55 coordinates are 55 atoms")
    energy, grad = _gupta().evaluate(np.full(55, 79, dtype=int), xyz * length)
    reduced = float(energy) / well
    reduced_grad = np.asarray(grad, dtype=np.float64).reshape(-1) * (length / well)
    return reduced, reduced_grad


class _Meter:
    def __init__(self) -> None:
        self.cpu = 0.0
        self.wall = 0.0
        self._mark()

    def _mark(self) -> None:
        self._cpu = time.process_time()
        self._wall = time.perf_counter()

    def running_cpu(self) -> float:
        return self.cpu + (time.process_time() - self._cpu)

    def snapshot_and_reset(self) -> tuple[float, float]:
        self.cpu += time.process_time() - self._cpu
        self.wall += time.perf_counter() - self._wall
        cpu = float(self.cpu)
        wall = float(self.wall)
        if wall < cpu:
            wall = cpu
        self.cpu = 0.0
        self.wall = 0.0
        return cpu, wall

    def restart(self) -> None:
        self._mark()


def _alive(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        text = stat.read_text()
    except FileNotFoundError:
        return False
    end = text.rfind(")")
    if end < 0 or end + 2 >= len(text):
        return False
    return text[end + 2] not in {"Z", "X"}


def _reap(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return


def _tail(path: Path) -> str:
    if not path.is_file():
        return ""
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    return " | ".join(lines[-5:])


def _read_status(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_status(work: Path, payload: dict) -> None:
    rows._atomic_write(work / "status.json", (json.dumps(payload) + "\n").encode())


def _reduced_pair(kind: str, n: int, potential):
    if kind == "lj":
        def pair(flat: np.ndarray):
            energy, grad = potential.lj_value_and_gradient(flat)
            return float(energy), np.asarray(grad, dtype=np.float64).reshape(-1)

        return pair, 1.0, 1.0
    if kind != "gupta":
        raise RuntimeError(f"unknown potential {kind}")
    length, well = gupta_pair_scales()
    numbers = np.full(int(n), 79, dtype=int)
    gupta = _gupta()

    def pair(flat: np.ndarray):
        xyz = np.asarray(flat, dtype=np.float64).reshape(-1, 3) * length
        energy, grad = gupta.evaluate(numbers, xyz)
        reduced = float(energy) / well
        reduced_grad = np.asarray(grad, dtype=np.float64).reshape(-1) * (length / well)
        return reduced, reduced_grad

    return pair, length, well


def _worker_search(work: Path) -> None:
    job = json.loads((work / "job.json").read_text())
    _pin()
    _limit_threads()
    panel, potential = _libraries()
    from anneal import Config, cluster_search

    kind = str(job["kind"])
    n = int(job["n"])
    if kind == "lj":
        panel.check_lj()
        potential.warmup()
    preset = Config.for_cluster(n)
    if int(preset.n_points) != n:
        raise RuntimeError("for_cluster built a different system")
    pair, _length, well = _reduced_pair(kind, n, potential)
    budget = int(job["budget"])
    level = (float(job["reference"]) + float(job["hit_tol"])) / well
    cap = None if job.get("cap_s") is None else float(job["cap_s"])
    meter = _Meter()
    counter = {
        "calls": 0,
        "best": math.inf,
        "best_x": None,
        "x": None,
        "energy": None,
        "grad": None,
    }
    hasher = hashlib.sha256()
    segment = 0

    def publish(phase: str, stop: str) -> None:
        nonlocal segment
        segment += 1
        cpu_s, wall_s = meter.snapshot_and_reset()
        best_x = counter["best_x"]
        _write_status(
            work,
            {
                "phase": phase,
                "stop": stop,
                "segment": segment,
                "calls": int(counter["calls"]),
                "best_e": float(counter["best"] * well),
                "cpu_s": cpu_s,
                "wall_s": wall_s,
                "trajectory": hasher.hexdigest(),
                "positions": (
                    []
                    if best_x is None
                    else np.asarray(best_x, dtype=np.float64).reshape(-1).tolist()
                ),
                "pid": os.getpid(),
            },
        )

    def park() -> float | None:
        publish("paused", "cap")
        resume = work / "resume"
        stop_file = work / "stop"
        while not resume.exists():
            if stop_file.exists():
                raise potential.CapReached()
            time.sleep(0.01)
        resume.unlink()
        fresh = json.loads((work / "job.json").read_text())
        meter.restart()
        nxt = fresh.get("cap_s")
        return None if nxt is None else float(nxt)

    def evaluate(x) -> None:
        nonlocal cap
        flat = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
        if (
            counter["x"] is not None
            and flat.shape == counter["x"].shape
            and np.array_equal(flat, counter["x"])
        ):
            return
        if counter["calls"] >= budget:
            raise potential.BudgetReached()
        if (
            cap is not None
            and counter["calls"] > 0
            and math.isfinite(counter["best"])
            and meter.running_cpu() >= cap
        ):
            cap = park()
        energy, grad = pair(flat)
        if grad.shape != flat.shape or not math.isfinite(energy) or not np.all(np.isfinite(grad)):
            energy = 1.0e8
            grad = np.zeros_like(flat)
        counter["calls"] += 1
        counter["x"] = flat.copy()
        counter["energy"] = float(energy)
        counter["grad"] = np.ascontiguousarray(grad, dtype=np.float64)
        hasher.update(np.ascontiguousarray(counter["x"], dtype=np.float64).tobytes())
        if counter["energy"] < counter["best"]:
            counter["best"] = counter["energy"]
            counter["best_x"] = counter["x"].copy()

    def energy(x):
        evaluate(x)
        return counter["energy"]

    def grad(x):
        evaluate(x)
        return counter["grad"]

    stop = None
    try:
        out = cluster_search(
            energy,
            grad,
            n,
            budget + 64,
            seed=int(job["seed"]),
            recommended=False,
            target=level,
        )
    except potential.CapReached:
        raise
    except BaseException as exc:
        kind_stop = _classify(exc)
        if kind_stop is None:
            if counter["calls"] >= budget:
                kind_stop = "budget"
            else:
                raise
        stop = kind_stop
        out = None
    else:
        reported = float(out["best_energy"])
        if math.isfinite(reported) and reported < counter["best"]:
            counter["best"] = reported
            best = out.get("best")
            if best is not None:
                counter["best_x"] = np.asarray(best, dtype=np.float64).reshape(-1).copy()
        if counter["best"] <= level:
            stop = "hit"
        elif counter["calls"] >= budget:
            stop = "budget"
        else:
            raise RuntimeError(
                f"anneal stopped after {counter['calls']} calls at {counter['best']}"
            )
    if counter["best_x"] is None or not math.isfinite(counter["best"]):
        raise RuntimeError("anneal did not record an energy")
    if stop == "cap" and counter["calls"] >= budget:
        stop = "budget"
    if stop == "budget" and counter["calls"] != budget:
        raise RuntimeError(f"budget stop recorded {counter['calls']} calls")
    publish("done", stop)


def _worker_main(workdir: str) -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    work = Path(workdir)
    try:
        _worker_search(work)
    except BaseException as exc:
        if _classify(exc) == "cap":
            return 0
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        try:
            _write_status(
                work,
                {
                    "phase": "error",
                    "segment": 1 << 30,
                    "message": f"{type(exc).__name__}: {exc}",
                    "pid": os.getpid(),
                },
            )
        except Exception:
            pass
        return 1
    return 0


def _publish_job(work: Path, cell, cap_s) -> None:
    panel, _potential = _libraries()
    spec = panel.SYSTEMS[cell.system]
    payload = {
        "system": cell.system,
        "seed": int(cell.seed),
        "budget": int(cell.budget),
        "cap_s": None if cap_s is None else float(cap_s),
        "n": int(spec["n"]),
        "kind": spec["kind"],
        "reference": float(spec["reference"]),
        "hit_tol": float(panel.HIT_TOL),
    }
    rows._atomic_write(work / "job.json", (json.dumps(payload) + "\n").encode())


def _spawn_worker(work: Path) -> int:
    script = str(Path(__file__).resolve())
    with (work / "worker.err").open("ab") as err:
        proc = subprocess.Popen(
            [sys.executable, script, "--worker", str(work)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=err,
            start_new_session=True,
            cwd=str(ROOT),
        )
    return int(proc.pid)


def _matching_status(status: dict | None, segment: int) -> bool:
    return (
        status is not None
        and int(status.get("segment", 0)) >= int(segment)
        and status.get("phase") in {"paused", "done", "error"}
    )


def _raise_worker_status(status: dict, pid: int) -> None:
    if status.get("phase") == "error":
        message = status.get("message") or f"anneal worker {pid} failed"
        raise RuntimeError(str(message))


def _wait_worker(work: Path, pid: int, segment: int) -> dict:
    status_path = work / "status.json"
    while True:
        status = _read_status(status_path)
        if _matching_status(status, segment):
            _raise_worker_status(status, pid)
            return status
        if not _alive(pid):
            status = _read_status(status_path)
            if _matching_status(status, segment):
                _raise_worker_status(status, pid)
                _reap(pid)
                return status
            _reap(pid)
            detail = f"anneal worker {pid} is gone; refusing to reseed"
            tail = _tail(work / "worker.err")
            if tail:
                detail = f"{detail}: {tail}"
            raise RuntimeError(detail)
        time.sleep(0.01)


def _stop_worker(pid: int, work: Path) -> None:
    (work / "stop").write_text("\n")
    for _ in range(500):
        if not _alive(pid):
            _reap(pid)
            return
        time.sleep(0.01)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _reap(pid)


def _clear_worker_flags(work: Path) -> None:
    for name in ("resume", "stop", "status.json"):
        path = work / name
        if path.exists():
            path.unlink()


def _run_anneal(cell, root: Path, cap_s, state) -> dict:
    work = checkpoint_path(root, cell.system, cell.arm, cell.seed).parent
    work.mkdir(parents=True, exist_ok=True)
    if state is not None:
        pid = state.get("worker_pid")
        if pid is None or not _alive(int(pid)):
            shown = "missing" if pid is None else int(pid)
            raise RuntimeError(f"anneal worker {shown} is gone; refusing to reseed")
        pid = int(pid)
        _publish_job(work, cell, cap_s)
        (work / "resume").write_text("\n")
        status = _wait_worker(work, pid, int(state.get("segment", 1)) + 1)
    else:
        _clear_worker_flags(work)
        _publish_job(work, cell, cap_s)
        pid = _spawn_worker(work)
        status = _wait_worker(work, pid, 1)
    if status.get("phase") == "paused":
        stop = "cap"
    elif status.get("phase") == "done":
        stop = str(status["stop"])
    else:
        raise RuntimeError(f"anneal worker {pid} ended in {status.get('phase')}")
    cpu_s = float(status["cpu_s"])
    return _outcome(
        stop,
        int(status["calls"]),
        float(status["best_e"]),
        cpu_s,
        float(status["wall_s"]),
        positions=status["positions"],
        worker_pid=pid,
        trajectory=str(status["trajectory"]),
        segment=int(status["segment"]),
        parent_cpu_s=cpu_s,
        child_cpu_s=0.0,
    )


def _finish(root: Path, cell, stamp: Path, state, outcome: dict, cpu: int):
    prior = list(state["prior"]) if state else []
    summed_cpu = math.fsum([float(item["cpu_s"]) for item in prior] + [float(outcome["cpu_s"])])
    summed_wall = math.fsum([float(item["wall_s"]) for item in prior] + [float(outcome["wall_s"])])
    stop = outcome["stop"]
    terminal = stop in {"hit", "budget"} or (stop == "cap" and summed_cpu >= float(protocol.CPU_CAP_S))
    record_stop = stop if terminal else "checkpoint"
    segments = prior + [
        {
            "calls_reached": int(outcome["calls"]),
            "best_e": float(outcome["best_e"]),
            "cpu_s": float(outcome["cpu_s"]),
            "wall_s": float(outcome["wall_s"]),
            "stop": record_stop,
            "parent_cpu_s": float(outcome["parent_cpu_s"]),
            "child_cpu_s": float(outcome["child_cpu_s"]),
        }
    ]
    if not terminal:
        _write_checkpoint(root, cell, outcome, segments)
        return None
    record = segments
    if cell.stop_rule != "cap" and len(segments) > 1:
        record = [
            {
                "calls_reached": int(outcome["calls"]),
                "best_e": min(float(item["best_e"]) for item in segments),
                "cpu_s": summed_cpu,
                "wall_s": summed_wall,
                "stop": record_stop,
            }
        ]
    built = [
        rows.Segment(
            int(item["calls_reached"]),
            float(item["best_e"]),
            float(item["cpu_s"]),
            float(item["wall_s"]),
            str(item["stop"]),
        )
        for item in record
    ]
    extra = _shape_extra(cpu)
    extra["parent_cpu_s"] = format(
        math.fsum(float(item["parent_cpu_s"]) for item in segments),
        ".6f",
    )
    extra["child_cpu_s"] = format(
        math.fsum(float(item["child_cpu_s"]) for item in segments),
        ".6f",
    )
    if outcome.get("trajectory") is not None:
        extra["trajectory_sha256"] = str(outcome["trajectory"])
    if cell.arm == "anneal":
        extra["segment_count"] = len(segments)
    row = rows.append_result(
        root,
        cell,
        built,
        PREREG_REVISION,
        shape_extra=extra,
    )
    _write_stamp(cell, stamp)
    return row


def run_cell(system: str, arm: str, seed: int, root, stamp) -> dict | None:
    root = Path(root)
    stamp = Path(stamp)
    cpu = _pin()
    _limit_threads()
    cell = protocol.find_cell(system, arm, int(seed))
    stored = _stored_row(root, cell)
    if stored is not None:
        _write_stamp(cell, stamp)
        return stored
    cap_s = segment_cap_s(cell)
    if cell.arm == "anneal" or cell.stop_rule == "cap":
        state = _load_checkpoint(root, cell)
    else:
        state = None
    if cell.arm == "ase_bh":
        outcome = _run_bh(cell, cap_s, state)
    elif cell.arm == "ase_mh":
        outcome = _run_mh(cell, root, cap_s, state)
    elif cell.arm == "anneal":
        outcome = _run_anneal(cell, root, cap_s, state)
    else:
        raise RuntimeError(f"unknown arm {cell.arm}")
    row = _finish(root, cell, stamp, state, outcome, cpu)
    if cell.arm == "anneal" and row is not None and outcome.get("worker_pid") is not None:
        pid = int(outcome["worker_pid"])
        work = checkpoint_path(root, cell.system, cell.arm, cell.seed).parent
        if _alive(pid):
            _stop_worker(pid, work)
        else:
            _reap(pid)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one pre-registered G2 budget-stop cell.")
    parser.add_argument("--system")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stamp")
    parser.add_argument("--root", default=None)
    parser.add_argument("--worker", default=None)
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_main(args.worker)
    if not args.system or not args.arm or args.seed is None or not args.stamp:
        parser.error("--system, --arm, --seed and --stamp are required")
    root = Path(args.root) if args.root else ROOT
    run_cell(args.system, args.arm, args.seed, root, Path(args.stamp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
