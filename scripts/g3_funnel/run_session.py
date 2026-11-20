#!/usr/bin/env python3
# Copyright (c) anneal developers
"""The G3 funnel panel: anneal and ASE minima hopping, traced, in one session.

Usage: run_session.py BIN_DIR OUT_DIR

LJ38, LJ75, LJ98, LJ150 and Morse M80 at rho 6, seeds 2000-2011, the
``PAPER_BUDGETS`` of ``scripts/gmin_panel/run_panel.py`` and its references.
For each system the seeds run in order, and the method order alternates per
seed: anneal then minima hopping on even positions, the reverse on odd ones
(ABBA). Every run is serial, in this process's affinity, which must be core 0
alone, with every thread variable at 1 and no ``H2H_`` or ``ANNEAL_``
variable.

anneal runs ``h2h_gmin_proof`` (Lennard-Jones) or ``h2h_morse`` from BIN_DIR
with ``--trace``. Minima hopping runs ``run_mh.py``; its wall cap is three
times the slowest anneal wall stored for the system, the rule and the values
of the committed ASE panels (``CAP_S``). Rows go to OUT_DIR/rows.csv after
every run and are never rewritten; a stored (system, method, seed) is
skipped. A run that fails leaves a row with status ``error`` and hit 0.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gmin_panel = load("g3_gmin_panel", ROOT / "scripts" / "gmin_panel" / "run_panel.py")

SYSTEMS = ("lj38", "lj75", "lj98", "lj150", "m80r6")
SEEDS = tuple(range(2000, 2012))
METHODS = ("anneal", "ase_mh")
# bench/ase-lj38, ase-lj75, ase-lj98 and ase-lj150 rows; m80r6 is three times
# the slowest anneal wall_s in bench/gmin-panel/rows.csv (178.269596 s).
CAP_S = {
    "lj38": 4.653972,
    "lj75": 167.806728,
    "lj98": 246.546693,
    "lj150": 544.418154,
    "m80r6": 534.808788,
}
FIELDS = [
    "system",
    "method",
    "seed",
    "order",
    "budget",
    "status",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "stop",
    "cap_s",
    "minima",
    "trace_sha256",
    "code_revision",
    "machine",
    "utc",
]


def machine_id() -> str:
    return hashlib.sha256(Path("/etc/machine-id").read_bytes().strip()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def environment(out: Path) -> dict[str, str]:
    env = gmin_panel.single_thread_env()
    env["NUMBA_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMBA_CACHE_DIR"] = str(out / "numba-cache")
    return env


def run_anneal(bin_dir: Path, out: Path, system: str, seed: int, trace: Path) -> dict[str, str]:
    binary = "h2h_morse" if system.startswith("m") else "h2h_gmin_proof"
    command = [str(bin_dir / binary), "run", system, str(seed), "--trace", str(trace)]
    proc = subprocess.run(command, cwd=ROOT, env=environment(out), capture_output=True, text=True, check=False)
    (out / "logs" / f"{system}-anneal-{seed}.log").write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}")
    row = gmin_panel.anneal_result(proc.stdout, system, seed, gmin_panel.PAPER_BUDGETS[system])
    return {
        "hit": row["hit"],
        "evals_to_hit": row["evals_to_hit"],
        "charged": row["charged"],
        "best": row["best"],
        "wall_s": row["wall_s"],
        "pot_s": row["pot_s"],
        "stop": "hit" if row["hit"] == "1" else "budget",
        "cap_s": "",
    }


def run_mh(out: Path, system: str, seed: int, trace: Path) -> dict[str, str]:
    work = out / "work" / f"{system}-ase_mh-{seed}"
    command = [sys.executable, str(HERE / "run_mh.py"), system, str(seed), f"{CAP_S[system]:.6f}", str(work), str(trace)]
    proc = subprocess.run(command, cwd=ROOT, env=environment(out), capture_output=True, text=True, check=False)
    (out / "logs" / f"{system}-ase_mh-{seed}.log").write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    keep = out / "logs" / f"{system}-ase_mh-{seed}"
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("hop.log", "minima.traj"):
        if (work / name).exists():
            shutil.copy2(work / name, keep / name)
    shutil.rmtree(work, ignore_errors=True)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}")
    lines = [line for line in proc.stdout.splitlines() if line.startswith("g3_row ")]
    if len(lines) != 1:
        raise RuntimeError("no g3_row")
    fields = dict(part.split("=", 1) for part in lines[0].split()[1:])
    if fields["system"] != system or int(fields["seed"]) != seed:
        raise RuntimeError(f"row {lines[0]}")
    if int(fields["unmatched"]) != 0:
        raise RuntimeError(f"{fields['unmatched']} minima without a matching force call")
    return {key: fields[key] for key in ("hit", "evals_to_hit", "charged", "best", "wall_s", "pot_s", "stop", "cap_s")}


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    bin_dir = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    if os.sched_getaffinity(0) != {0}:
        sys.exit("pin this process to core 0 alone (taskset -c 0)")
    for key in os.environ:
        if key.startswith(("H2H_", "ANNEAL_")):
            sys.exit(f"{key} is set")
    for sub in ("logs", "traces", "work"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.csv"
    done = set()
    if rows_path.exists():
        with rows_path.open() as handle:
            done = {(r["system"], r["method"], r["seed"]) for r in csv.DictReader(handle)}
    else:
        with rows_path.open("w") as handle:
            csv.writer(handle, lineterminator="\n").writerow(FIELDS)
    code_revision = revision()
    machine = machine_id()
    for system in SYSTEMS:
        budget = gmin_panel.PAPER_BUDGETS[system]
        for position, seed in enumerate(SEEDS):
            order = METHODS if position % 2 == 0 else METHODS[::-1]
            for slot, method in enumerate(order):
                if (system, method, str(seed)) in done:
                    continue
                trace = out / "traces" / f"{system}-{method}-{seed}.xyz"
                started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                print(f"{started} {system} {method} {seed}", flush=True)
                try:
                    if method == "anneal":
                        result = run_anneal(bin_dir, out, system, seed, trace)
                    else:
                        result = run_mh(out, system, seed, trace)
                    status = "ok"
                except Exception as error:  # noqa: BLE001
                    print(f"  error: {error}", flush=True)
                    result = {
                        "hit": "0",
                        "evals_to_hit": "",
                        "charged": "",
                        "best": "",
                        "wall_s": "",
                        "pot_s": "",
                        "stop": "error",
                        "cap_s": f"{CAP_S[system]:.6f}" if method == "ase_mh" else "",
                    }
                    status = "error"
                minima = ""
                digest = ""
                if trace.exists():
                    with trace.open() as handle:
                        minima = str(sum(1 for line in handle if line.startswith("call=")))
                    digest = sha256(trace)
                row = {
                    "system": system,
                    "method": method,
                    "seed": seed,
                    "order": slot,
                    "budget": budget,
                    "status": status,
                    "minima": minima,
                    "trace_sha256": digest,
                    "code_revision": code_revision,
                    "machine": machine,
                    "utc": started,
                    **result,
                }
                with rows_path.open("a") as handle:
                    csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n").writerow(row)
    print("SESSION-DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
