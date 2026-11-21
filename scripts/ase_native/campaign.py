# Copyright (c) anneal developers
"""Resumable runner for the native-unit ASE head-to-head.

The process pins itself to one core and one thread before importing the
numeric libraries. A finished (system, arm, seed) row is skipped. A row
is appended only after that seed returns.
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench" / "ase-native"
ROWS = DATA / "rows.csv"
SESSIONS = DATA / "sessions.csv"

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
    "NUMBA_NUM_THREADS",
    "TBB_NUM_THREADS",
    "GOTO_NUM_THREADS",
)
BLOB_PATHS = (
    "scripts/ase_native/mapping.py",
    "scripts/ase_native/search.py",
    "scripts/ase_native/controls.py",
    "scripts/ase_native/potentials.py",
    "scripts/ase_native/eval_client.py",
    "scripts/ase_native/campaign.py",
    "scripts/ase_native/report.py",
    "scripts/ase_native/test_mapping.py",
    "scripts/ase_native/test_agree.py",
    "examples/ase_native_eval.rs",
    "examples/ase_panel.rs",
    "examples/ase_nonlj.rs",
)
UNCHANGED = ("examples/ase_panel.rs", "examples/ase_nonlj.rs")
COLUMNS = [
    "system",
    "arm",
    "seed",
    "slot",
    "core",
    "evals",
    "evals_to_hit",
    "best_energy",
    "hit",
    "wall_s",
    "machine_id",
    "budget",
    "stop",
    "best_infinity",
    "best_ase",
    "base_sha",
    "base_tree",
    "blobs",
]


def _pin() -> None:
    for name in THREAD_VARS:
        os.environ[name] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["LJOS_SEAT"] = "benchbot"
    if __name__ != "__main__":
        return
    allowed = _cpus_allowed()
    if allowed == "0":
        return
    os.execvpe("taskset", ["taskset", "-c", "0", sys.executable, "-u", *sys.argv], os.environ)


def _cpus_allowed() -> str:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("affinity is not visible")


_pin()

import numpy as np  # noqa: E402
from ase import __version__ as ASE_VERSION  # noqa: E402

from mapping import BASE_SHA, SYSTEMS, balance_ok, schedule  # noqa: E402
from report import render  # noqa: E402
from search import run_anneal, run_ase  # noqa: E402
from test_agree import check_agreement  # noqa: E402


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    if not raw:
        raw = Path("/proc/sys/kernel/random/boot_id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def provenance() -> dict[str, str]:
    head = _git("rev-parse", "HEAD")
    base_type = _git("cat-file", "-t", BASE_SHA)
    if base_type != "commit":
        raise RuntimeError("base is not a commit")
    base_tree = _git("rev-parse", f"{BASE_SHA}^{{tree}}")
    _git("merge-base", "--is-ancestor", BASE_SHA, head)
    dirty = subprocess.run(["git", "diff", "--quiet", "--", *BLOB_PATHS], cwd=ROOT)
    if dirty.returncode != 0:
        raise RuntimeError("harness files differ from HEAD")
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *BLOB_PATHS], cwd=ROOT)
    if staged.returncode != 0:
        raise RuntimeError("harness files are staged and uncommitted")
    blobs: dict[str, str] = {}
    for path in BLOB_PATHS:
        blob = _git("rev-parse", f"HEAD:{path}")
        _git("cat-file", "-e", blob)
        blobs[path] = blob
        if path in UNCHANGED:
            base_blob = _git("rev-parse", f"{BASE_SHA}:{path}")
            if base_blob != blob:
                raise RuntimeError(f"{path} differs from the base")
    return {
        "base_sha": BASE_SHA,
        "base_tree": base_tree,
        "blobs": ";".join(f"{path}={blob}" for path, blob in blobs.items()),
    }


def _load_rows() -> list[dict[str, str]]:
    if not ROWS.is_file():
        return []
    with ROWS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _append(path: Path, columns: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _session(event: str, ident: dict[str, str], extra: dict[str, object]) -> None:
    columns = [
        "event",
        "ase_version",
        "numpy_version",
        "anneal_sha",
        "anneal_tree",
        "machine_id",
        "core",
        "seeds",
        "balance",
        "base_sha",
        "base_tree",
        "blobs",
    ]
    row = {
        "event": event,
        "ase_version": ASE_VERSION,
        "numpy_version": np.__version__,
        "anneal_sha": ident["base_sha"],
        "anneal_tree": ident["base_tree"],
        "machine_id": machine_id(),
        "core": _cpus_allowed(),
        "seeds": extra.get("seeds", ""),
        "balance": extra.get("balance", ""),
        "base_sha": ident["base_sha"],
        "base_tree": ident["base_tree"],
        "blobs": ident["blobs"],
    }
    _append(SESSIONS, columns, row)


def _log_dir() -> Path:
    raw = os.environ.get("ASE_NATIVE_LOG_DIR")
    path = Path(raw) if raw else Path(tempfile.gettempdir()) / "ase-native-logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_campaign() -> int:
    if _cpus_allowed() != "0":
        raise RuntimeError("the campaign is not pinned to core 0")
    ident = provenance()
    check_agreement()
    jobs = schedule()
    if not balance_ok(jobs):
        raise RuntimeError("the schedule is not balanced")
    done = {(row["system"], row["arm"], row["seed"]) for row in _load_rows()}
    for row in _load_rows():
        match = [
            job
            for job in jobs
            if job["system"] == row["system"] and job["arm"] == row["arm"] and str(job["seed"]) == row["seed"]
        ]
        if len(match) != 1 or str(match[0]["slot"]) != row["slot"]:
            raise RuntimeError("an existing row does not match the schedule")
    counts: dict[str, int] = {}
    for job in jobs:
        key = f"{job['slot'] % len(set(job['arm'] for job in jobs))}:{job['arm']}"
        counts[key] = counts.get(key, 0) + 1
    balance = ",".join(f"{key}={counts[key]}" for key in sorted(counts))
    seeds = ",".join(f"{name}:{len(set(job['seed'] for job in jobs if job['system'] == name))}" for name in SYSTEMS)
    _session("start", ident, {"seeds": seeds, "balance": balance})
    log = _log_dir() / "campaign.log"
    work_root = Path(tempfile.mkdtemp(prefix="ase-native-"))
    for job in jobs:
        key = (str(job["system"]), str(job["arm"]), str(job["seed"]))
        if key in done:
            continue
        system = str(job["system"])
        spec = SYSTEMS[system]
        if job["arm"] == "anneal":
            result = run_anneal(system, int(job["seed"]), spec["budget"], spec["kind"])
        else:
            result = run_ase(
                system,
                str(job["arm"]),
                int(job["seed"]),
                spec["budget"],
                spec["kind"],
                spec["reference"],
                work_root / f"{system}-{job['arm']}-{job['seed']}",
            )
        if int(result["evals"]) > spec["budget"]:
            raise RuntimeError("a run exceeded the paper budget")
        row = {
            "system": system,
            "arm": job["arm"],
            "seed": job["seed"],
            "slot": job["slot"],
            "core": _cpus_allowed(),
            "evals": result["evals"],
            "evals_to_hit": result["evals_to_hit"],
            "best_energy": _number(result["best_energy"]),
            "hit": result["hit"],
            "wall_s": _number(result["wall_s"]),
            "machine_id": machine_id(),
            "budget": spec["budget"],
            "stop": result["stop"],
            "best_infinity": _number(result["best_infinity"]) if result["best_infinity"] != "" else "",
            "best_ase": _number(result["best_ase"]) if result["best_ase"] != "" else "",
            "base_sha": ident["base_sha"],
            "base_tree": ident["base_tree"],
            "blobs": ident["blobs"],
        }
        _append(ROWS, COLUMNS, row)
        line = f"{system} {job['arm']} seed={job['seed']} evals={result['evals']} hit={result['hit']}\n"
        sys.stderr.write(line)
        with log.open("a") as handle:
            handle.write(line)
        done.add(key)
    _session("end", ident, {"seeds": seeds, "balance": balance})
    report = DATA / "report.md"
    report.write_text(render())
    return 0


def _number(value: object) -> str:
    number = float(value)  # type: ignore[arg-type]
    if number != number or number == float("inf") or number == float("-inf"):
        return "inf"
    return format(number, ".16g")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        sys.stdout.write(render())
        return 0
    return run_campaign()


if __name__ == "__main__":
    raise SystemExit(main())
