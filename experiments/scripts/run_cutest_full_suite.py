"""Run anneal drivers across a PyCUTEst-selected benchmark suite.

The runner enumerates unconstrained and bound-constrained CUTEst
problems through PyCUTEst, filters by dimension, and emits a
long-form CSV with one row per ``(problem, driver, seed)`` cell. The
CSV is compatible with the plotting script and the CUTEst summary
checker.

Example:
    pixi run -e verify python experiments/scripts/run_cutest_full_suite.py \
        --out data/cutest_full.csv --seeds 3 --dim-cap 50 \
        --per-problem-timeout 120
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty

import numpy as np

from experiments.benchmarks.cutest_runner import (
    CutestConfig,
    configured_pycutest,
    default_cutest_config,
    load,
)
from experiments.scripts.run_cutest_benchmarks import (
    DRIVERS as MANIFEST_DRIVERS,
    SCIPY_DRIVERS,
    _bgsa_run,
    _cutest_gradient,
    _run_cutest_additive_independence,
    _run_cutest_gle_langevin,
    bayesian_mixing_sa,
    classical_sa,
    mcmc_sa,
    mcmc_sa_budgeted,
    portfolio_sa,
    pt_sa_budgeted,
)


FULL_SUITE_DRIVERS = tuple(MANIFEST_DRIVERS)
FIELD_NAMES = [
    "problem",
    "kind",
    "dim",
    "driver",
    "seed",
    "fevals",
    "best_val",
    "wall_time_s",
    "f_x0",
    "solved",
    "status",
]


@dataclass(frozen=True, order=True)
class TargetProblem:
    """A CUTEst problem selected for the full-suite sweep."""

    name: str
    kind: str
    dim: int


@contextmanager
def per_problem_timeout(seconds: int):
    """Bound a problem cell with SIGALRM and restore the handler."""

    def _on_alarm(_signum, _frame):
        raise TimeoutError(f"timeout after {seconds}s")

    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _property_dim(props: dict) -> int | None:
    raw = props.get("n")
    if raw is None:
        return None
    if isinstance(raw, str):
        if not raw.isdigit():
            return None
        raw = int(raw)
    try:
        dim = int(raw)
    except (TypeError, ValueError):
        return None
    return dim if dim > 0 else None


def _collect_kind(
    pycutest, kind: str, dim_cap: int, exclude_names: set[str]
) -> list[TargetProblem]:
    selected: list[TargetProblem] = []
    for name in pycutest.find_problems(constraints=kind):
        if name in exclude_names:
            continue
        try:
            dim = _property_dim(pycutest.problem_properties(name))
        except Exception:
            continue
        if dim is None or dim > dim_cap:
            continue
        selected.append(TargetProblem(name=name, kind=kind, dim=dim))
    return selected


def cutest_config_from_args(args=None) -> CutestConfig:
    if args is None:
        return default_cutest_config()
    return default_cutest_config(
        getattr(args, "bench_root", None),
        cache_dir=getattr(args, "pycutest_cache", None),
    )


def list_target_problems(
    dim_cap: int,
    exclude_names: Iterable[str] | None = None,
    config: CutestConfig | None = None,
) -> list[TargetProblem]:
    """Enumerate unique unconstrained and bound-constrained CUTEst targets."""

    pycutest = configured_pycutest(config)

    excluded = set(exclude_names or ())
    by_name: dict[str, TargetProblem] = {}
    for kind in ("unconstrained", "bound"):
        for target in _collect_kind(pycutest, kind, dim_cap, excluded):
            by_name.setdefault(target.name, target)
    return sorted(by_name.values(), key=lambda target: (target.name, target.kind))


def shard_targets(
    targets: Iterable[TargetProblem],
    shard_index: int,
    shard_count: int,
) -> list[TargetProblem]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return [
        target for idx, target in enumerate(targets) if idx % shard_count == shard_index
    ]


def parse_drivers(raw: str | None) -> tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return FULL_SUITE_DRIVERS
    drivers = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(drivers) - set(FULL_SUITE_DRIVERS))
    if unknown:
        raise ValueError(f"unknown CUTEst driver(s): {', '.join(unknown)}")
    return drivers


def evict_pycutest_cache(config: CutestConfig | None = None) -> None:
    cache_root = (
        default_cutest_config() if config is None else config.validate()
    ).cache_dir
    cache_root.mkdir(parents=True, exist_ok=True)
    for child in cache_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (cache_root / "pycutest_cache_holder").mkdir(parents=True, exist_ok=True)


def resume_keys(rows: Iterable[dict]) -> set[tuple[str, str, str]]:
    return {(str(row["problem"]), str(row["driver"]), str(row["seed"])) for row in rows}


def load_existing_rows(path: str, resume: bool) -> list[dict]:
    if not resume or not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for row in rows:
            normalised = {key: row.get(key, "") for key in FIELD_NAMES}
            writer.writerow(normalised)


def append_target_timeout_rows(
    rows: list[dict],
    seen: set[tuple[str, str, str]],
    target: TargetProblem,
    cells: Iterable[tuple[int, str]],
    f0: float,
) -> int:
    appended = 0
    for seed, driver in cells:
        key = (target.name, driver, str(seed))
        if key in seen:
            continue
        rows.append(
            {
                "problem": target.name,
                "kind": target.kind,
                "dim": target.dim,
                "driver": driver,
                "seed": seed,
                "fevals": 0,
                "best_val": float("nan"),
                "wall_time_s": "0.000",
                "f_x0": f0,
                "solved": 0,
                "status": "target-timeout",
            }
        )
        seen.add(key)
        appended += 1
    return appended


def append_target_error_rows(
    rows: list[dict],
    seen: set[tuple[str, str, str]],
    target: TargetProblem,
    cells: Iterable[tuple[int, str]],
    status: str,
    f0: float,
) -> int:
    appended = 0
    for seed, driver in cells:
        key = (target.name, driver, str(seed))
        if key in seen:
            continue
        rows.append(
            {
                "problem": target.name,
                "kind": target.kind,
                "dim": target.dim,
                "driver": driver,
                "seed": seed,
                "fevals": 0,
                "best_val": float("nan"),
                "wall_time_s": "0.000",
                "f_x0": f0,
                "solved": 0,
                "status": status,
            }
        )
        seen.add(key)
        appended += 1
    return appended


# Pilot work units spent fitting each problem's reduction, charged into the
# reported fevals of every cell that runs in the reduced space. Keyed by name;
# read back in the same process that fit it (the forked cell worker).
_REDUCTION_PILOT_WORK: dict[str, float] = {}


def maybe_reduce(prob, args):
    """Collapse a high-dimensional problem onto an active subspace.

    When ``--reduce`` is set and ``prob.dim`` exceeds the threshold, fit the
    dimension-collapse + Chebyshev surrogate (deterministically, with a fixed
    pilot seed so the reduced coordinate system is identical across seeds and
    drivers) and return a reduced ``CutestProblem``. The reduced ``fn``
    evaluates the true objective at the decoded point, so ``best_val`` stays
    honest; the reduced ``grad`` is the surrogate's cheap analytic gradient,
    which the HMC driver consumes in place of an ``n+1`` finite difference.
    """
    import dataclasses

    if not getattr(args, "reduce", False):
        return prob
    threshold = int(getattr(args, "surrogate_dim_threshold", 0) or 0)
    if threshold <= 0 or prob.dim <= threshold:
        return prob

    from experiments.surrogate import fit_reduced_surrogate

    fit = fit_reduced_surrogate(
        prob.fn,
        prob.grad,
        prob.low,
        prob.high,
        prob.dim,
        k=int(args.surrogate_k),
        degree=int(args.surrogate_degree),
        n_pilot=getattr(args, "surrogate_pilot", None),
        rng=12345,
    )
    decode = fit.encoder.decode
    true_fn = prob.fn
    surrogate_grad = fit.surrogate.grad

    def reduced_fn(r, _fn=true_fn, _dec=decode):
        return float(_fn(_dec(r)))

    def reduced_grad(r, _g=surrogate_grad):
        return _g(r)

    reduced_low = np.asarray(fit.reduced_low, dtype=float)
    reduced_high = np.asarray(fit.reduced_high, dtype=float)
    reduced_x0 = np.clip(
        fit.encoder.encode(np.asarray(prob.x0, dtype=float)),
        reduced_low,
        reduced_high,
    )
    reduced = dataclasses.replace(
        prob,
        dim=fit.surrogate.k,
        fn=reduced_fn,
        grad=reduced_grad,
        low=reduced_low,
        high=reduced_high,
        x0=reduced_x0,
        design_low=reduced_low.copy(),
        design_high=reduced_high.copy(),
    )
    _REDUCTION_PILOT_WORK[prob.name] = float(fit.pilot_work_units)
    return reduced


def load_problem(name: str, args=None):
    prob = load(name, sif_params=None, config=cutest_config_from_args(args))
    if args is not None:
        prob = maybe_reduce(prob, args)
    return prob


def load_target_problem(target: TargetProblem, args):
    """Load a CUTEst target outside the per-driver timer."""
    seconds = int(getattr(args, "load_timeout", 0) or 0)
    if seconds > 0:
        with per_problem_timeout(seconds):
            return load_problem(target.name, args)
    return load_problem(target.name, args)


def _evict_problem_cache(name: str, args) -> None:
    """Stream the suite: drop a problem's compiled pycutest cache once its cells
    are done. Each CUTEst problem is decoded and compiled to a shared library;
    keeping all of them caches the whole set on disk and exhausts it on the
    large native-size problems. Evicting after each problem bounds peak disk to
    a single problem (one per shard) rather than the full catalogue.
    """
    if not getattr(args, "evict_cache", False):
        return
    try:
        pycutest = configured_pycutest(cutest_config_from_args(args))
        pycutest.clear_cache(name)
    except Exception:
        pass
    cache = cutest_config_from_args(args).cache_dir
    for sub in (cache / "pycutest_cache" / name, cache / name):
        shutil.rmtree(sub, ignore_errors=True)


def centre_value(prob) -> float:
    f0 = float(prob.fn((prob.low + prob.high) / 2))
    return f0 if np.isfinite(f0) else 0.0


def solved_flag(best_val: float, f0: float) -> int:
    if not np.isfinite(best_val):
        return 0
    if f0 > 0:
        return int(best_val < 0.95 * f0)
    return int(best_val < 1.05 * f0)


def is_timeout_status(status: str | None) -> bool:
    return status in {"timeout", "target-timeout"}


def run_driver(prob, driver: str, seed: int, args) -> tuple[float, int]:
    if driver == "classical":
        return classical_sa(prob, seed, args.n_epochs, args.k_fixed)
    if driver == "mcmc_sa":
        return mcmc_sa(
            prob,
            seed,
            args.n_epochs,
            args.n_chains,
            args.k_min,
            args.k_check,
            args.k_max,
            args.rhat_threshold,
        )
    if driver == "mcmc_sa_sparse":
        return mcmc_sa(
            prob,
            seed,
            args.n_epochs,
            args.n_chains,
            args.k_min,
            args.k_check,
            args.k_max,
            args.rhat_threshold,
            sparse=True,
            straggler_top_k=args.straggler_top_k,
        )
    if driver == "mcmc_sa_budgeted":
        return mcmc_sa_budgeted(
            prob,
            seed,
            args.n_epochs,
            args.n_chains,
            args.k_fixed,
            args.k_min,
            args.k_check,
            args.rhat_threshold,
        )
    if driver == "mcmc_sa_sparse_budgeted":
        return mcmc_sa_budgeted(
            prob,
            seed,
            args.n_epochs,
            args.n_chains,
            args.k_fixed,
            args.k_min,
            args.k_check,
            args.rhat_threshold,
            sparse=True,
            straggler_top_k=args.straggler_top_k,
        )
    if driver == "pt_sa_budgeted":
        return pt_sa_budgeted(
            prob,
            seed,
            args.n_epochs,
            args.n_chains,
            args.k_fixed,
        )
    if driver == "bayesian_mixing_sa":
        return bayesian_mixing_sa(
            prob,
            seed,
            1 + args.n_epochs * args.k_fixed,
        )
    if driver == "portfolio":
        return portfolio_sa(
            prob,
            seed,
            1 + args.n_epochs * args.k_fixed,
        )
    if driver == "additive_indep":
        result = _run_cutest_additive_independence(
            _anneal_module(),
            prob,
            seed,
            args.n_epochs,
            args.k_fixed,
        )
        if result is None:
            raise RuntimeError("additive independence driver returned no result")
        return result
    if driver == "gle_langevin":
        grad_fn, grad_kind = _cutest_gradient(prob)
        result = _run_cutest_gle_langevin(
            _anneal_module(),
            prob,
            grad_fn,
            grad_kind,
            seed,
            args.n_epochs,
            args.k_fixed,
        )
        if result is None:
            raise RuntimeError("GLE Langevin driver returned no result")
        return result
    if driver in SCIPY_DRIVERS:
        return SCIPY_DRIVERS[driver](
            prob,
            seed,
            1 + args.n_epochs * args.k_fixed,
        )
    return _bgsa_run(prob, seed, args.n_epochs, args.k_fixed, args.n_chains, driver)


def _anneal_module():
    import anneal

    return anneal


def _load_cell_problem(problem_ref, args=None):
    if isinstance(problem_ref, str):
        return load_problem(problem_ref, args)
    return problem_ref


def _pilot_work(prob) -> int:
    return int(_REDUCTION_PILOT_WORK.get(getattr(prob, "name", ""), 0))


def _driver_worker(queue, problem_ref, driver: str, seed: int, args) -> None:
    try:
        prob = _load_cell_problem(problem_ref, args)
        f0 = centre_value(prob)
        best_val, fevals = run_driver(prob, driver, seed, args)
        queue.put(("ok", (best_val, fevals + _pilot_work(prob), f0)))
    except Exception as exc:
        queue.put(("err", type(exc).__name__))


def _fork_context():
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def run_driver_cell(
    problem_ref, driver: str, seed: int, args
) -> tuple[float, int, float, str]:
    """Load and run one driver cell with a timeout that can stop compiled code."""
    ctx = _fork_context()
    if ctx is None:
        try:
            with per_problem_timeout(args.per_problem_timeout):
                prob = _load_cell_problem(problem_ref, args)
                f0 = centre_value(prob)
                best_val, fevals = run_driver(prob, driver, seed, args)
            return best_val, fevals + _pilot_work(prob), f0, "ok"
        except TimeoutError:
            return float("nan"), 0, float("nan"), "timeout"
        except Exception as exc:
            return float("nan"), 0, float("nan"), f"err:{type(exc).__name__}"

    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_driver_worker, args=(queue, problem_ref, driver, seed, args)
    )
    proc.start()
    proc.join(args.per_problem_timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return float("nan"), 0, float("nan"), "timeout"
    if proc.exitcode != 0:
        return float("nan"), 0, float("nan"), f"err:exit{proc.exitcode}"
    try:
        status, payload = queue.get_nowait()
    except Empty:
        return float("nan"), 0, float("nan"), "err:no-result"
    if status == "ok":
        best_val, fevals, f0 = payload
        return best_val, fevals, f0, "ok"
    return float("nan"), 0, float("nan"), f"err:{payload}"


def run_suite(args) -> list[dict]:
    drivers = parse_drivers(args.drivers)
    config = cutest_config_from_args(args)
    rows = load_existing_rows(args.out, resume=not args.no_resume)
    seen = resume_keys(rows)

    if args.evict_cache:
        evict_pycutest_cache(config)

    targets = list_target_problems(
        args.dim_cap,
        exclude_names=args.exclude_problem,
        config=config,
    )
    if args.max_problems is not None:
        targets = targets[: args.max_problems]
    targets = shard_targets(targets, args.shard_index, args.shard_count)

    print(f"Found {len(targets)} CUTEst targets with dim <= {args.dim_cap}", flush=True)
    if args.shard_count > 1:
        print(f"Shard {args.shard_index + 1}/{args.shard_count}", flush=True)
    print(f"Drivers: {', '.join(drivers)}", flush=True)
    if rows:
        print(f"Resuming from {len(rows)} existing row(s)", flush=True)

    t0_global = time.perf_counter()
    completed_targets = 0
    for target in targets:
        cells = tuple(
            (seed, driver) for seed in range(args.seeds) for driver in drivers
        )
        all_done = all(
            (target.name, driver, str(seed)) in seen for seed, driver in cells
        )
        if all_done:
            continue
        existing_target_rows = [
            row for row in rows if row.get("problem") == target.name
        ]
        target_timeouts = sum(
            1 for row in existing_target_rows if is_timeout_status(row.get("status"))
        )
        target_f0 = float("nan")
        for row in existing_target_rows:
            try:
                candidate_f0 = float(row.get("f_x0", "nan"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(candidate_f0):
                target_f0 = candidate_f0
                break
        if (
            args.max_timeouts_per_target > 0
            and target_timeouts >= args.max_timeouts_per_target
        ):
            appended = append_target_timeout_rows(rows, seen, target, cells, target_f0)
            if appended:
                write_rows(args.out, rows)
            continue
        try:
            target_problem = load_target_problem(target, args)
        except TimeoutError:
            appended = append_target_timeout_rows(rows, seen, target, cells, target_f0)
            if appended:
                write_rows(args.out, rows)
            _evict_problem_cache(target.name, args)
            continue
        except Exception as exc:
            appended = append_target_error_rows(
                rows,
                seen,
                target,
                cells,
                f"err:{type(exc).__name__}",
                target_f0,
            )
            if appended:
                write_rows(args.out, rows)
            _evict_problem_cache(target.name, args)
            continue

        for idx, (seed, driver) in enumerate(cells):
            key = (target.name, driver, str(seed))
            if key in seen:
                continue

            start = time.perf_counter()
            best_val, fevals, f0, status = run_driver_cell(
                target_problem, driver, seed, args
            )
            if np.isfinite(float(f0)):
                target_f0 = float(f0)

            row = {
                "problem": target.name,
                "kind": target.kind,
                "dim": target.dim,
                "driver": driver,
                "seed": seed,
                "fevals": fevals,
                "best_val": best_val,
                "wall_time_s": f"{time.perf_counter() - start:.3f}",
                "f_x0": f0,
                "solved": solved_flag(float(best_val), f0),
                "status": status,
            }
            rows.append(row)
            seen.add(key)
            if is_timeout_status(status):
                target_timeouts += 1
            if (
                args.max_timeouts_per_target > 0
                and target_timeouts >= args.max_timeouts_per_target
            ):
                append_target_timeout_rows(
                    rows, seen, target, cells[idx + 1 :], target_f0
                )
                write_rows(args.out, rows)
                break
            write_rows(args.out, rows)

        _evict_problem_cache(target.name, args)
        completed_targets += 1
        if completed_targets % args.checkpoint_every == 0:
            write_rows(args.out, rows)
            elapsed = time.perf_counter() - t0_global
            print(
                f"  checkpoint {completed_targets}/{len(targets)} target(s), "
                f"{len(rows)} row(s), {elapsed:.0f}s",
                flush=True,
            )

    write_rows(args.out, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cutest_full.csv")
    parser.add_argument(
        "--bench-root",
        default=None,
        help="Project root containing .bench/ with CUTEst, SIFDecode, and sif.",
    )
    parser.add_argument(
        "--pycutest-cache",
        default=None,
        help="Explicit PyCUTEst cache directory; defaults to .bench/cache.",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dim-cap", type=int, default=50)
    parser.add_argument(
        "--reduce",
        action="store_true",
        help="Collapse problems above the dimension threshold onto an "
        "active subspace with a Chebyshev surrogate gradient.",
    )
    parser.add_argument(
        "--surrogate-dim-threshold",
        type=int,
        default=0,
        help="Wrap problems with dim above this (0 disables).",
    )
    parser.add_argument(
        "--surrogate-k",
        type=int,
        default=4,
        help="Reduced (active-subspace) dimension.",
    )
    parser.add_argument(
        "--surrogate-degree",
        type=int,
        default=6,
        help="Total-degree of the Chebyshev surrogate.",
    )
    parser.add_argument(
        "--surrogate-pilot",
        type=int,
        default=None,
        help="Pilot sample size (default scales with the basis size).",
    )
    parser.add_argument(
        "--evict-cache",
        action="store_true",
        help="Stream the suite: drop each problem's compiled pycutest "
        "cache once its cells finish, bounding peak disk to one "
        "problem (one per shard) instead of the whole catalogue.",
    )
    parser.add_argument(
        "--load-timeout",
        type=int,
        default=0,
        help="Maximum seconds for CUTEst load/setup; 0 disables this setup bound.",
    )
    parser.add_argument("--per-problem-timeout", type=int, default=180)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--k-fixed", type=int, default=200)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=30)
    parser.add_argument("--k-check", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--rhat-threshold", type=float, default=1.2)
    parser.add_argument("--straggler-top-k", type=int, default=2)
    parser.add_argument("--drivers", default=None)
    parser.add_argument("--exclude-problem", action="append", default=[])
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--max-timeouts-per-target", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def print_tally(rows: list[dict]) -> None:
    print(f"\nWrote {len(rows)} row(s)")
    for driver in sorted({row["driver"] for row in rows}):
        sub = [row for row in rows if row["driver"] == driver]
        ok = [
            row
            for row in sub
            if row.get("status", "ok") == "ok" and np.isfinite(float(row["best_val"]))
        ]
        solved = sum(int(row["solved"]) for row in sub)
        timeouts = sum(1 for row in sub if is_timeout_status(row.get("status")))
        errors = sum(1 for row in sub if str(row.get("status", "")).startswith("err:"))
        mean_fevals = np.mean([float(row["fevals"]) for row in ok]) if ok else 0.0
        print(
            f"  {driver:<16}: solved {solved}/{len(sub)} cells, "
            f"{len(ok)} ok, {timeouts} timeout, {errors} err, "
            f"mean fevals = {mean_fevals:.0f}"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rows = run_suite(args)
    print_tally(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
