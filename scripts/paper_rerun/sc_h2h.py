#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fresh Sutton-Chen head-to-head of the printed anneal and GMIN cells.

The cells, arms, seeds and budgets are the manifest. Budgets are
``scripts/metal_h2h/run_sc.py`` ``PAPER_BUDGETS``. Seeds are 2000-2011.
Arms alternate ABBA by seed: an even offset runs anneal then GMIN, an odd
offset runs GMIN then anneal. One job at a time. The process tree is bound
to cpu 0. A job that exceeds ``JOB_TIMEOUT_S`` is an error, not a rerun.

A hit is ``scripts/target_tol.py`` ``energy_hits``: best <= reference + 0.001.
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "metal_h2h"))

import run_sc as sc  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

CODE_REVISION = "61fa67cf380e46876b569da45e7cc091647a775b"
GMIN_PIN = "9cbec1a446352bc9ef064bd2b1a5767194655919"
SEEDS = tuple(range(2000, 2012))
ARMS = ("anneal", "gmin")
# Set before any job. A full 80-atom budget is far under this.
JOB_TIMEOUT_S = 21600
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

BASE_FIELDS = (
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "best",
    "charged",
    "evals_to_hit",
    "wall_s",
    "pot_s",
    "outside_s",
    "pot_per_eval_s",
    "outside_per_eval_s",
    "final_e",
    "hops",
    "stall_escapes",
    "trail_escapes",
    "restarts",
    "paths",
    "path_improvements",
    "code_revision",
    "host",
    "harness_revision",
)
FIELDS = BASE_FIELDS + ("gmin_pin", "harness_blob", "session")
MANIFEST_FIELDS = ("seq", "system", "arm", "seed", "budget", "reference")


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def blob_of(path: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "hash-object", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def head_of() -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def jobs() -> list[tuple[str, int, str]]:
    ordered: list[tuple[str, int, str]] = []
    for system in sc.PAPER_BUDGETS:
        for seed in SEEDS:
            if (seed - SEEDS[0]) % 2 == 0:
                arms = ("anneal", "gmin")
            else:
                arms = ("gmin", "anneal")
            for arm in arms:
                ordered.append((system, seed, arm))
    return ordered


def manifest_rows() -> list[dict[str, str]]:
    rows = []
    for seq, (system, seed, arm) in enumerate(jobs(), start=1):
        rows.append(
            {
                "seq": str(seq),
                "system": system,
                "arm": arm,
                "seed": str(seed),
                "budget": str(sc.PAPER_BUDGETS[system]),
                "reference": f"{sc.REFERENCES[system]:.4f}",
            }
        )
    return rows


def write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows())


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [tuple(row[name] for name in MANIFEST_FIELDS) for row in rows] != [
        tuple(row[name] for name in MANIFEST_FIELDS) for row in manifest_rows()
    ]:
        raise SystemExit(f"{path} is not the PAPER_BUDGETS manifest")
    return rows


def pin_affinity() -> None:
    if hasattr(os, "sched_getaffinity") and os.sched_getaffinity(0) == {0}:
        return
    os.execvp("taskset", ["taskset", "-c", "0", sys.executable, *sys.argv])


def thread_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    return env


def load_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {(row["system"], row["arm"], row["seed"]) for row in csv.DictReader(handle)}


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def parse_diag(text: str) -> dict[str, str]:
    found = {
        "final_e": "",
        "hops": "",
        "stall_escapes": "",
        "trail_escapes": "",
        "restarts": "",
        "paths": "",
        "path_improvements": "",
    }
    for line in text.splitlines():
        if not line.startswith("diag "):
            continue
        for part in line.split()[1:]:
            key, value = part.split("=", 1)
            if key == "final":
                found["final_e"] = value
            elif key in found:
                found[key] = value
    return found


def run_anneal(binary: Path, system: str, seed: int, budget: int, coords: Path, log: Path, work: Path) -> dict[str, str]:
    env = thread_env()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "H2H_BEST": str(work / f"best-{seed}.xyz"),
            "H2H_FINAL": str(work / f"final-{seed}.xyz"),
        }
    )
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "run", system, str(seed), str(coords)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=JOB_TIMEOUT_S,
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal {system} seed {seed} exited {proc.returncode}\n{text[-2000:]}")
    parsed = sc.parse_anneal(text)
    parsed["_diag"] = parse_diag(text)
    return parsed


def run_gmin(
    binary: Path,
    system: str,
    seed: int,
    budget: int,
    coords: Path,
    work: Path,
    log: Path,
) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    (work / "data").write_text(sc.data_text(system, 90_000_000) + f"RANSEED {seed}\n")
    (work / "coords").write_bytes(coords.read_bytes())
    env = thread_env()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "H2H_SYSTEM": system,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{sc.REFERENCES[system]:.10f}",
            "H2H_TOL": f"{DEFAULT_HIT_TOL}",
            "H2H_ROW": str(work / "row.txt"),
        }
    )
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=work,
            timeout=JOB_TIMEOUT_S,
        )
    row_path = work / "row.txt"
    if not row_path.is_file():
        tail = log.read_text()[-2000:]
        raise RuntimeError(f"gmin {system} seed {seed} exited {proc.returncode}\n{tail}")
    return sc.parse_gmin(row_path.read_text())


def as_row(
    system: str,
    arm: str,
    parsed: dict[str, str],
    diag: dict[str, str],
    host: str,
    harness_revision: str,
    harness_blob: str,
    session: str,
    code_revision: str,
) -> dict[str, str]:
    reference = sc.REFERENCES[system]
    best = float(parsed["best"])
    hit = energy_hits(best, reference, DEFAULT_HIT_TOL)
    charged = max(int(float(parsed["charged"])), 1)
    wall = float(parsed["wall_s"])
    pot = float(parsed["pot_s"])
    outside = max(wall - pot, 0.0)
    evals = int(float(parsed.get("evals_to_hit", charged)))
    if not hit:
        evals = int(float(parsed["budget"]))
    return {
        "system": system,
        "arm": arm,
        "seed": str(int(float(parsed["seed"]))),
        "budget": str(int(float(parsed["budget"]))),
        "hit": "1" if hit else "0",
        "best": f"{best:.8f}",
        "charged": str(charged),
        "evals_to_hit": str(evals),
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "pot_per_eval_s": f"{pot / charged:.9f}",
        "outside_per_eval_s": f"{outside / charged:.9f}",
        "final_e": diag.get("final_e", ""),
        "hops": diag.get("hops", ""),
        "stall_escapes": diag.get("stall_escapes", ""),
        "trail_escapes": diag.get("trail_escapes", ""),
        "restarts": diag.get("restarts", ""),
        "paths": diag.get("paths", ""),
        "path_improvements": diag.get("path_improvements", ""),
        "code_revision": code_revision,
        "host": host,
        "harness_revision": harness_revision,
        "gmin_pin": GMIN_PIN,
        "harness_blob": harness_blob,
        "session": session,
    }


def session_id(path: Path) -> str:
    if path.is_file():
        return path.read_text().strip()
    value = hashlib.sha256(os.urandom(32)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")
    return value


def gmin_revision(source: Path | None) -> str | None:
    if source is None or not source.is_file():
        return None
    for line in source.read_text().splitlines():
        if line.startswith("svnrev="):
            return line.split("=", 1)[1].strip()
    return None


def main(argv: list[str]) -> int:
    pin_affinity()
    os.environ.update(THREAD_ENV)
    manifest = ROOT / "bench" / "paper-rerun" / "sc-h2h" / "manifest.csv"
    rows_path = ROOT / "bench" / "paper-rerun" / "sc-h2h" / "rows.csv"
    if argv == ["--write-manifest"]:
        write_manifest(manifest)
        print(f"wrote {manifest} cells={len(manifest_rows())}")
        return 0
    rows = read_manifest(manifest)
    work = Path(os.environ["SC_H2H_WORK"])
    logs = Path(os.environ["SC_H2H_LOGS"])
    binary = Path(os.environ["SC_H2H_ANNEAL"])
    gmin = Path(os.environ["SC_H2H_GMIN"]) if os.environ.get("SC_H2H_GMIN") else None
    source = Path(os.environ["SC_H2H_GMIN_SOURCE"]) if os.environ.get("SC_H2H_GMIN_SOURCE") else None
    built = gmin_revision(source)
    gmin_ok = bool(gmin and gmin.is_file() and built == GMIN_PIN)
    if not gmin_ok:
        print(
            f"gmin arm not runnable: pin {GMIN_PIN} binary source is {built or 'absent'}",
            flush=True,
        )
    host = machine_id()
    harness_revision = head_of()
    harness_blob = blob_of(HERE / "sc_h2h.py")
    session = session_id(work / "session")
    done = load_done(rows_path)
    for row in rows:
        key = (row["system"], row["arm"], row["seed"])
        if key in done:
            continue
        system, arm, seed = row["system"], row["arm"], int(row["seed"])
        budget = int(row["budget"])
        if budget != sc.PAPER_BUDGETS[system] or arm not in ARMS:
            raise SystemExit(f"manifest drift {system} {arm} {budget}")
        if arm == "gmin" and not gmin_ok:
            print(f"skip {system} gmin seed {seed}: pin unavailable", flush=True)
            continue
        runs = work / "runs" / system
        runs.mkdir(parents=True, exist_ok=True)
        coords = runs / f"start-{seed}.xyz"
        if not coords.is_file():
            subprocess.run(
                ["taskset", "-c", "0", str(binary), "coords", system, str(seed), str(coords)],
                check=True,
                env=thread_env(),
                timeout=JOB_TIMEOUT_S,
            )
        log = logs / system / f"{arm}-{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"run {system} {arm} seed {seed}", flush=True)
        if arm == "anneal":
            parsed = run_anneal(binary, system, seed, budget, coords, log, runs)
            diag = parsed.pop("_diag")
            code = CODE_REVISION
        else:
            assert gmin is not None
            parsed = run_gmin(gmin, system, seed, budget, coords, runs / f"gmin-{seed}", log)
            diag = {name: "" for name in ("final_e", "hops", "stall_escapes", "trail_escapes", "restarts", "paths", "path_improvements")}
            code = GMIN_PIN
        written = as_row(system, arm, parsed, diag, host, harness_revision, harness_blob, session, code)
        if written["budget"] != row["budget"] or written["seed"] != row["seed"]:
            raise SystemExit(f"row does not match the manifest {system} {arm} {seed}")
        append_row(rows_path, written)
        print(
            f"done {system} {arm} seed {seed} hit={written['hit']} calls={written['evals_to_hit']}",
            flush=True,
        )
    print(f"rows {rows_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
