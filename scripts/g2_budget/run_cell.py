#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run one pre-registered G2 budget-stop cell.

The call budget and the process-CPU cap come from the protocol module.
A seed that already has a row is not run again. A capped seed that
stopped short of the pre-registered cap continues from its checkpoint.
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


def _run_bh(cell, cap_s, state) -> dict:
    if cap_s is not None or state is not None:
        raise RuntimeError("basin hopping is budget-stopped in one invocation")
    panel, potential = _libraries()
    from ase.optimize.basin import BasinHopping

    _spec, atoms, calc, server, previous = _ase_setup(cell, panel, potential, None)
    np.random.seed(int(cell.seed))
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
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
        if stop is None:
            if calc.hit_calls is not None:
                stop = "hit"
            elif calc.calls >= cell.budget:
                stop = "budget"
            else:
                raise RuntimeError("basin hopping stopped before the budget")
        calls = int(calc.hit_calls if stop == "hit" and calc.hit_calls is not None else calc.calls)
        if stop == "budget" and calls != int(cell.budget):
            raise RuntimeError(f"budget stop recorded {calls} calls")
        return _outcome(
            stop,
            calls,
            _best_energy(calc, previous),
            time.process_time() - cpu0,
            time.perf_counter() - wall0,
            positions=atoms.get_positions().reshape(-1).tolist(),
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
        cpu0 = time.process_time()
        wall0 = time.perf_counter()
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
                if cap_s is not None and (time.process_time() - cpu0) >= float(cap_s) and moved:
                    stop = "cap"
                elif not moved and int(calc.calls) == calls_before:
                    raise RuntimeError("minima hopping made no progress")
        calls = int(calc.hit_calls if stop == "hit" and calc.hit_calls is not None else calc.calls)
        if stop == "cap" and calls >= int(cell.budget):
            stop = "budget"
        if stop == "budget" and calls != int(cell.budget):
            raise RuntimeError(f"budget stop recorded {calls} calls")
        return _outcome(
            stop,
            calls,
            _best_energy(calc, previous),
            time.process_time() - cpu0,
            time.perf_counter() - wall0,
            positions=atoms.get_positions().reshape(-1).tolist(),
            rng_state=_dump_rng(rng),
            completed=completed,
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


def _pair_potential(cell, panel, potential):
    spec = panel.SYSTEMS[cell.system]
    if spec["kind"] == "lj":
        panel.check_lj()
        potential.warmup()

        def pair(flat: np.ndarray):
            energy, grad = potential.lj_value_and_gradient(flat)
            return float(energy), np.asarray(grad, dtype=np.float64).reshape(-1)

        return spec, pair, None
    binary = panel.asg_eval_binary()
    n_atoms, paper_budget, reference, _length = panel.evaluator_spec(binary, cell.system)
    if (n_atoms, paper_budget) != (spec["n"], spec["budget"]) or reference != spec["reference"]:
        raise RuntimeError("asg_eval spec differs from the panel")
    server = potential.AnnealServer(binary, cell.system)

    def pair(flat: np.ndarray):
        energy, _inside, grad = server.eval(flat)
        return float(energy), np.asarray(grad, dtype=np.float64).reshape(-1)

    return spec, pair, server


def _run_anneal(cell, cap_s, state) -> dict:
    panel, potential = _libraries()
    from anneal import Config, cluster_search

    spec, pair, server = _pair_potential(cell, panel, potential)
    preset = Config.for_cluster(int(spec["n"]))
    if int(preset.n_points) != int(spec["n"]):
        raise RuntimeError("for_cluster built a different system")
    start_calls = 0 if state is None else int(state["calls"])
    counter = {
        "calls": start_calls,
        "best": math.inf if state is None else float(state["best_e"]),
        "best_x": None if state is None else np.asarray(state["positions"], dtype=np.float64).reshape(-1).copy(),
        "x": None,
        "energy": None,
        "grad": None,
    }
    cpu0 = time.process_time()
    wall0 = time.perf_counter()

    def evaluate(x) -> None:
        flat = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
        if (
            counter["x"] is not None
            and flat.shape == counter["x"].shape
            and np.array_equal(flat, counter["x"])
        ):
            return
        if counter["calls"] >= int(cell.budget):
            raise potential.BudgetReached()
        if (
            cap_s is not None
            and counter["calls"] > start_calls
            and (time.process_time() - cpu0) >= float(cap_s)
        ):
            raise potential.CapReached()
        energy, grad = pair(flat)
        if grad.shape != flat.shape or not math.isfinite(energy) or not np.all(np.isfinite(grad)):
            energy = 1.0e8
            grad = np.zeros_like(flat)
        counter["calls"] += 1
        counter["x"] = flat.copy()
        counter["energy"] = float(energy)
        counter["grad"] = np.ascontiguousarray(grad, dtype=np.float64)
        if counter["energy"] < counter["best"]:
            counter["best"] = counter["energy"]
            counter["best_x"] = counter["x"].copy()

    def energy(x):
        evaluate(x)
        return counter["energy"]

    def grad(x):
        evaluate(x)
        return counter["grad"]

    level = float(spec["reference"]) + float(panel.HIT_TOL)
    start = None if state is None else np.asarray(state["positions"], dtype=np.float64).reshape(-1)
    stop = None
    try:
        try:
            out = cluster_search(
                energy,
                grad,
                int(spec["n"]),
                int(cell.budget) + 64,
                seed=int(cell.seed),
                recommended=False,
                start=start,
                target=level,
            )
        except BaseException as exc:
            kind = _classify(exc)
            if kind is None:
                elapsed = time.process_time() - cpu0
                if counter["calls"] >= int(cell.budget):
                    kind = "budget"
                elif cap_s is not None and counter["calls"] > start_calls and elapsed >= float(cap_s):
                    kind = "cap"
                else:
                    raise
            stop = kind
            out = None
        else:
            reported = float(out["best_energy"])
            if math.isfinite(reported) and reported < counter["best"]:
                counter["best"] = reported
                counter["best_x"] = np.asarray(out["best"], dtype=np.float64).reshape(-1).copy()
            if counter["best"] <= level:
                stop = "hit"
            elif counter["calls"] >= int(cell.budget):
                stop = "budget"
            elif cap_s is not None and (time.process_time() - cpu0) >= float(cap_s):
                stop = "cap"
            else:
                raise RuntimeError(
                    f"anneal stopped after {counter['calls']} calls at {counter['best']}"
                )
        if counter["best_x"] is None or not math.isfinite(counter["best"]):
            raise RuntimeError("anneal did not record an energy")
        calls = int(counter["calls"])
        if stop == "cap" and calls >= int(cell.budget):
            stop = "budget"
        if stop == "budget" and calls != int(cell.budget):
            raise RuntimeError(f"budget stop recorded {calls} calls")
        return _outcome(
            stop,
            calls,
            float(counter["best"]),
            time.process_time() - cpu0,
            time.perf_counter() - wall0,
            positions=np.asarray(counter["best_x"], dtype=np.float64).reshape(-1).tolist(),
        )
    finally:
        if server is not None:
            server.close()


def _finish(root: Path, cell, stamp: Path, state, outcome: dict, cpu: int):
    prior = list(state["prior"]) if state else []
    summed = math.fsum([float(item["cpu_s"]) for item in prior] + [float(outcome["cpu_s"])])
    stop = outcome["stop"]
    terminal = stop in {"hit", "budget"} or (stop == "cap" and summed >= float(protocol.CPU_CAP_S))
    record_stop = stop if terminal else "checkpoint"
    segments = prior + [
        {
            "calls_reached": int(outcome["calls"]),
            "best_e": float(outcome["best_e"]),
            "cpu_s": float(outcome["cpu_s"]),
            "wall_s": float(outcome["wall_s"]),
            "stop": record_stop,
        }
    ]
    if not terminal:
        _write_checkpoint(root, cell, outcome, segments)
        return None
    built = [
        rows.Segment(
            int(item["calls_reached"]),
            float(item["best_e"]),
            float(item["cpu_s"]),
            float(item["wall_s"]),
            str(item["stop"]),
        )
        for item in segments
    ]
    row = rows.append_result(
        root,
        cell,
        built,
        PREREG_REVISION,
        shape_extra=_shape_extra(cpu),
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
    state = _load_checkpoint(root, cell) if cell.stop_rule == "cap" else None
    if cell.arm == "ase_bh":
        outcome = _run_bh(cell, cap_s, state)
    elif cell.arm == "ase_mh":
        outcome = _run_mh(cell, root, cap_s, state)
    elif cell.arm == "anneal":
        outcome = _run_anneal(cell, cap_s, state)
    else:
        raise RuntimeError(f"unknown arm {cell.arm}")
    return _finish(root, cell, stamp, state, outcome, cpu)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one pre-registered G2 budget-stop cell.")
    parser.add_argument("--system", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else ROOT
    run_cell(args.system, args.arm, args.seed, root, Path(args.stamp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
