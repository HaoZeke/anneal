"""Score anneal's cluster search on an optbench global-optimisation problem.

The harness is optbench's lane-B one, imported from an optbench checkout:
``optbench.lane_budget.Budget`` counts one fresh coordinate as one force
call, a repeated coordinate is free, and a call at the cap is refused and
not counted. Starts come from the problem archive through
``optbench.lane_load``, the per-start record is ``optbench.lane_methods``'s,
and the campaign files are written by ``optbench.runners.harness``. Nothing
here counts a call.

A start is a hit when a fresh evaluation lands inside the problem's energy
window with every gradient component at most ``CONVERGED`` in magnitude:
a converged quench of the global minimum, counted when it happened. The
runner then raises a ``BaseException``; anneal ends the run at its next
budget check and the call count at the hit is the start's cost.

Lennard-Jones and binary Lennard-Jones reach anneal as the atom positions.
Rigid TIP4P reaches it as a rigid-body state through ``water_sites``: the
centres of mass, then exponential-map rotations, which are GMIN's angle-axis
vectors with the sign flipped because GMIN rotates passively and anneal
actively. The harness only ever sees GMIN's chart.

Run from the optbench checkout's environment::

    uv run --project $OPTBENCH python scripts/optbench/anneal_globalopt.py \\
        --optbench $OPTBENCH --problem lj38 --preset communicating \\
        --workers 8 --output results/anneal-lj38-communicating.dat
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tomllib
from pathlib import Path

import numpy as np

CONVERGED = 1.0e-3
PRESETS = ("communicating", "recommended")


class _Stop(BaseException):
    """Ends an anneal run from inside a callback."""


def _optbench(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def lennard_jones_size(problem: dict) -> int | None:
    """Atom count of a pure Lennard-Jones problem, when its name is ``lj<N>``."""
    import re

    kind = str((problem.get("potential") or {}).get("kind") or "lennard-jones")
    if kind != "lennard-jones":
        return None
    name = str(problem.get("problem") or "")
    match = re.fullmatch(r"lj(\d+)", name)
    if match is None:
        match = re.fullmatch(r"globalopt/lj(\d+)", str(problem.get("id") or ""))
    if match is None:
        return None
    return int(match.group(1))


def load_problem(root: Path, name: str) -> dict:
    path = root / "problems" / "globalopt" / f"{name}.toml"
    with path.open("rb") as handle:
        problem = tomllib.load(handle)
    convergence = problem.setdefault("convergence", {})
    if "max_force_calls" not in convergence:
        raise ValueError(f"{path} states no max_force_calls")
    reference = (problem.get("reference") or {}).get("energy")
    if reference is None:
        reference = convergence.get("energy_minimum", (problem.get("potential") or {}).get("energy_minimum"))
    if reference is None:
        raise ValueError(f"{path} states no reference energy")
    convergence["energy_minimum"] = float(reference)
    size = lennard_jones_size(problem)
    if size is not None:
        import anneal

        anneal.check_stop_limit(size, float(convergence["energy_minimum"]))
    return problem


def kind_of(problem: dict) -> str:
    kind = str((problem.get("potential") or {}).get("kind") or "lennard-jones")
    if kind == "binary-lennard-jones" or problem["id"] == "globalopt/lj100-binary":
        return "binary"
    if kind == "tip4p":
        return "tip4p"
    return "lj"


def load_tip4p_starts(root: Path, problem: dict, work: Path, limit: int) -> list[dict]:
    """``.aa`` starts: centre-of-mass rows, then angle-axis rows."""
    import tarfile

    from optbench.campaign import geometry_archive

    archive = geometry_archive(root, problem)
    dest = work / problem["id"].replace("/", "-")
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    if not marker.is_file():
        with tarfile.open(archive) as packed:
            packed.extractall(dest, filter="data")
        marker.write_text(archive.name, encoding="utf-8")
    paths = sorted(dest.rglob("*.aa"), key=lambda path: int(path.stem))
    rows = []
    for index, path in enumerate(paths):
        values = np.array(path.read_text(encoding="utf-8").split(), dtype=np.float64)
        if values.size % 6 != 0:
            raise ValueError(f"{path} is not a centre and angle-axis block")
        rows.append(
            {
                "index": index,
                "start": path.name,
                "coords": values,
                "box": None,
                "species": None,
                "mode": None,
            }
        )
    expected = (problem.get("sample") or {}).get("starts")
    if limit <= 0 and expected is not None and len(rows) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(rows)}")
    return rows[:limit] if limit > 0 else rows


def load_binary_starts(root: Path, problem: dict, work: Path, output: Path, limit: int) -> list[dict]:
    """``coords.N`` starts with the problem's composition, ``n_A`` then ``n_B``.

    The archive also holds ``.con`` files whose compositions range from 30/70
    to 73/27; a start of the wrong composition cannot reach the stated
    minimum. GMIN's row runs the ``coords.N`` files with the first ``n_A``
    atoms of species A, and so does this one.
    """
    from optbench.lane_load import load_starts

    potential = problem["potential"]
    n_a, n_b = int(potential["n_A"]), int(potential["n_B"])
    sample = dict(problem.get("sample") or {}, layout="coords")
    rows = load_starts(root, dict(problem, sample=sample), work=work, output=output, limit=limit)
    species = np.array([1] * n_a + [2] * n_b, dtype=np.int64)
    for row in rows:
        if np.asarray(row["coords"]).size != 3 * (n_a + n_b):
            raise ValueError(f"{row['start']} does not hold {n_a + n_b} atoms")
        row["species"] = species
    return rows


def layout_of(root: Path, problem: dict) -> None:
    """Archives of ``coords.<n>`` blocks (and no ``.con``) load as coordinates."""
    import re
    import tarfile

    from optbench.campaign import geometry_archive

    sample = problem.setdefault("sample", {})
    if sample.get("layout"):
        return
    names = [Path(name).name for name in tarfile.open(geometry_archive(root, problem)).getnames()]
    if not any(name.endswith(".con") for name in names) and any(re.fullmatch(r"coords\.\d+", n) for n in names):
        sample["layout"] = "coords"


def run_start(job: dict) -> dict:
    """One start, scored by the harness. Returns an ``optbench.start/v2`` row."""
    _optbench(Path(job["optbench"]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import anneal

    import fast_kernels
    from optbench.counter import ForceCallCap
    from optbench.lane_budget import Budget, Hit
    from optbench.lane_methods import _record

    kind, species = job["kind"], job.get("species")
    start = np.asarray(job["coords"], dtype=np.float64)
    fast_kernels.check(kind, start, species)
    evaluate = fast_kernels.native(kind, species)
    python_kernel = fast_kernels.reference(kind, species)
    target = float(job["target"])
    window = float(job["energy_window"])
    budget = Budget(evaluate, int(job["cap"]), target=target, window=window)
    last: dict = {}
    reason: list[str] = []
    rigid = {}
    if kind == "tip4p":
        from optbench.tip4p import reference_sites

        rigid["water_sites"] = reference_sites()[:3]

    def chart(x):
        """GMIN's rotation is passive and anneal's active: the rotation block flips sign.

        The map is its own inverse and applies to a gradient the same way.
        """
        if kind != "tip4p":
            return x
        y = np.array(x, dtype=np.float64)
        y[y.size // 2 :] *= -1.0
        return y

    def fresh(x):
        x = np.asarray(x, dtype=np.float64)
        point = last.get("x")
        if point is not None and np.array_equal(point, x):
            return last["e"], last["g"]
        harness_x = chart(x)
        try:
            energy, gradient = budget.fg(harness_x)
        except ForceCallCap:
            reason.append("cap")
            raise _Stop() from None
        last.update(x=x.copy(), e=energy, g=chart(gradient))
        if energy <= target + window and float(np.max(np.abs(gradient))) <= CONVERGED:
            energy, gradient = python_kernel(harness_x)
            if not (energy <= target + window and float(np.max(np.abs(gradient))) <= CONVERGED):
                return last["e"], last["g"]
            try:
                budget.note_quench(harness_x, energy, gradient, converged=True)
            except Hit:
                reason.append("hit")
                raise _Stop() from None
        return last["e"], last["g"]

    error = ""
    out = None
    try:
        out = anneal.cluster_search(
            lambda x: fresh(x)[0],
            lambda x: fresh(x)[1],
            int(job["n_points"]),
            int(job["cap"]),
            seed=int(job["seed"]),
            communicating=job["preset"] == "communicating",
            recommended=job["preset"] == "recommended",
            start=chart(start),
            **rigid,
        )
    except _Stop:
        pass
    except Exception as exc:  # a crash is a recorded failure, not a lost start
        error = f"{type(exc).__name__}: {exc}"
    if reason[:1] == ["cap"] and not error:
        error = "ForceCallCap: the harness refused a call at the cap"
    if out is not None and not budget.hit:
        best = chart(np.asarray(out["best"], dtype=np.float64))
        if np.all(np.isfinite(best)) and math.isfinite(float(out["best_energy"])):
            with budget.counter.verification():
                energy, gradient = budget.fg(best)
            try:
                budget.note_quench(
                    best,
                    energy,
                    gradient,
                    converged=float(np.max(np.abs(gradient))) <= CONVERGED,
                )
            except Hit:
                pass
    if budget.calls >= int(job["cap"]) and not error:
        error = "ForceCallCap: the search reached the cap"
    record = _record(job, budget, error, None, None)
    record["preset"] = job["preset"]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--optbench", type=Path, required=True)
    parser.add_argument("--problem", required=True, help="lj38, lj75, lj98, lj150, lj100-binary, tip4p10, tip4p20")
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=0, help="first N starts only; the row is then partial")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.optbench.resolve()
    _optbench(root)
    import anneal

    from optbench.lane_load import load_starts
    from optbench.runners import harness

    problem = load_problem(root, args.problem)
    layout_of(root, problem)
    kind = kind_of(problem)
    cap = int(problem["convergence"]["max_force_calls"])
    window = float(problem["convergence"].get("energy_above_minimum") or 0.001)
    target = float(problem["convergence"]["energy_minimum"])
    work = (args.work or args.output.parent / "geometries").resolve()
    if kind == "tip4p":
        rows = load_tip4p_starts(root, problem, work, args.limit)
    elif kind == "binary":
        rows = load_binary_starts(root, problem, work, args.output, args.limit)
    else:
        rows = load_starts(root, problem, work=work, output=args.output, limit=args.limit)

    jobs = []
    for row in rows:
        coords = np.asarray(row["coords"], dtype=np.float64).reshape(-1)
        jobs.append(
            {
                "optbench": str(root),
                "index": int(row["index"]),
                "start": row["start"],
                "coords": coords.tolist(),
                "species": None if row["species"] is None else np.asarray(row["species"]).tolist(),
                "box": None,
                "mode": None,
                "kind": kind,
                "family": "globalopt",
                "method": "anneal-cluster-search",
                "preset": args.preset,
                "n_points": coords.size // (6 if kind == "tip4p" else 3),
                "cap": cap,
                "gtol": 0.0,
                "seed": 314159 + int(row["index"]),
                "target": target,
                "energy_window": window,
                "sigma": 0.0,
            }
        )

    checkpoint = args.output.with_suffix(".checkpoint.jsonl")
    done: dict[str, dict] = {}
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["start"])] = row
    pending = [job for job in jobs if str(job["start"]) not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def store(record: dict) -> None:
        done[str(record["start"])] = record
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"{args.problem} {args.preset} {record['start']} {len(done)}/{len(jobs)} "
            f"calls={record['force_calls']} success={record['success']} energy={record.get('energy')}",
            flush=True,
        )

    if pending and args.workers <= 1:
        for job in pending:
            store(run_start(job))
    elif pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(run_start, job) for job in pending]):
                store(future.result())

    ordered = [done[str(job["start"])] for job in jobs]
    for record in ordered:
        record.pop("index", None)
        record.pop("_point", None)
    version = anneal.__version__
    commit = os.environ.get("ANNEAL_COMMIT", "")
    template = {
        "id": f"anneal/{args.problem}-{args.preset}",
        "code": "anneal",
        "code_version": version,
        "commit": commit,
        "algorithm": f"cluster_search {args.preset}",
        "profile": "default",
        "tuning": "default",
        "ranked": True,
        "search": {
            "family": "globalopt",
            "method": "anneal-cluster-search",
            "preset": args.preset,
            "quench": "rgmin L-BFGS (anneal WarmLbfgs)",
            "hit_gradient": CONVERGED,
        },
        "note": (
            f"anneal {version} cluster_search, Config.{args.preset}(n), stock. "
            "Starts are the archive geometries; the seed is 314159 plus the start index. "
            f"A hit is a fresh evaluation within {window} of {target} with every gradient "
            f"component at most {CONVERGED}."
        ),
    }
    partial = args.limit > 0
    summary = harness._write(problem, template, ordered, cap, version, commit, partial, args.output)
    print(json.dumps({k: summary.get(k) for k in ("par2_mean", "v2_nfailed", "starts", "median")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
