#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Gate runner comparing arm A (develop dd054749) with arm B (no-size-constants-r3).

Arm A is develop dd054749 unmodified. Arm B is that commit plus the
no-size-constants-r3 patch. This runner does not edit either arm.

The panel, presets, budgets and seeds are the no-size-constants-b2 gate.
Every supported system uses n=12 seeds 2000-2011. That gate's measured rows
are n=12 on every supported system, so this run keeps n=12, fixed before
any result.

Runs are serial. Each arm and seed is a fresh process, pinned to CPU core 0,
with OpenMP, OpenBLAS, MKL and Rayon threads set to 1. Even seeds run A then
B. Odd seeds run B then A. There is no warm-up and no rescue rerun. Rows are
appended once. Each row records both commit SHAs, both trees, and a hashed
host id.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# n=12 on every supported system. The b2 gate measured n=12 everywhere and
# declared no smaller per-system n.
SEEDS = tuple(range(2000, 2012))

ROW_FIELDS = [
    "panel",
    "system",
    "driver",
    "arm",
    "seed",
    "order",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "best",
    "reference",
    "tol",
    "trace_sha256",
    "code_revision",
    "tree",
    "a_revision",
    "b_revision",
    "a_tree",
    "b_tree",
    "host",
    "utc",
]

ARMS: dict[str, dict[str, str]] = {}
HOST_ID = ""


def host_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def stamp_row(row: dict[str, str]) -> dict[str, str]:
    arm = row.get("arm", "")
    out = dict(row)
    if arm in ARMS and HOST_ID:
        out["code_revision"] = ARMS[arm]["rev"]
        out["tree"] = ARMS[arm]["tree"]
        out["a_revision"] = ARMS["A"]["rev"]
        out["b_revision"] = ARMS["B"]["rev"]
        out["a_tree"] = ARMS["A"]["tree"]
        out["b_tree"] = ARMS["B"]["tree"]
        out["host"] = HOST_ID
    return out

LJ_SPECS = [
    ("lj", "lj38", 38, 400_000, -173.928427, 0.001),
    ("lj", "lj75", 75, 4_000_000, -397.492331, 0.001),
    ("lj", "lj98", 98, 4_000_000, -543.665361, 0.001),
    ("lj", "lj103", 103, 4_000_000, -575.766131, 0.001),
    ("lj", "lj104", 104, 4_000_000, -582.086642, 0.001),
]

MORSE_SPECS = [
    ("morse", "m75r3", 75, 3, 4_000_000, -633.513370, 0.001),
    ("morse", "m75r6", 75, 6, 4_000_000, -351.472365, 0.001),
    ("morse", "m75r10", 75, 10, 4_000_000, -322.643558, 0.001),
    ("morse", "m75r14", 75, 14, 4_000_000, -318.407330, 0.001),
    ("morse", "m80r3", 80, 3, 4_000_000, -690.577890, 0.001),
    ("morse", "m80r6", 80, 6, 4_000_000, -378.333471, 0.001),
    ("morse", "m80r10", 80, 10, 4_000_000, -344.911381, 0.001),
    ("morse", "m80r14", 80, 14, 4_000_000, -340.811371, 0.001),
]

TIP4P_SPECS = [
    ("tip4p", "tip4p8", 8, 2_000_000, -305.5183221676, 0.001),
    ("tip4p", "tip4p12", 12, 2_000_000, -492.9082357122, 0.001),
    ("tip4p", "tip4p15", 15, 2_000_000, -628.3722525077, 0.001),
    ("tip4p", "tip4p16", 16, 2_000_000, -681.1925945898, 0.001),
    ("tip4p", "tip4p20", 20, 2_000_000, -872.9887544952, 0.001),
    ("tip4p", "tip4p21", 21, 2_000_000, -916.7074302940, 0.001),
    ("tip4p", "tip4p25", 25, 2_000_000, None, 0.001),
]

SLAB_SPECS = [
    ("slab", "cuh2_fcc_slab", "examples/fixtures/cuh2_fcc_slab.con", 2500, None, 0.001),
    ("slab", "cuh2_fcc_slab_h6", "examples/fixtures/cuh2_fcc_slab_h6.con", 5000, None, 0.0001),
]

SLAB_SEED_RE = re.compile(
    r"seed (?P<seed>\d+): best (?P<best>[-+0-9.eE]+|[-+]?inf|nan) eV\s+hops (?P<hops>\d+)\s+"
    r"charged (?P<charged>\d+)\s+basins (?P<basins>\d+)",
    re.IGNORECASE,
)
SLAB_IMPROVE_RE = re.compile(
    r"improve hops=\d+ charged=(?P<charged>\d+) basins=\d+ e=(?P<energy>[-+0-9.eE]+)"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes()) if path.is_file() else ""


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_", "TWO_PHASE_", "BANK_")):
            del env[key]
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def verify_references(root: Path) -> None:
    morse_path = root / "examples" / "fixtures" / "morse_ccd.csv"
    morse_table: dict[tuple[int, int], float] = {}
    with morse_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            morse_table[(int(row["n"]), int(float(row["rho"])))] = float(row["energy"])
    for _panel, name, n, rho, _budget, ref, _tol in MORSE_SPECS:
        want = morse_table.get((n, rho))
        if want is None or abs(want - ref) > 1e-9:
            raise SystemExit(f"{name}: ref {ref} mismatches morse_ccd.csv {want}")

    tip4p_path = root / "data" / "tip4p_ccd.csv"
    tip4p_table: dict[int, float] = {}
    with tip4p_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            tip4p_table[int(row["n"])] = float(row["energy"])
    for _panel, name, n, _budget, ref, _tol in TIP4P_SPECS:
        if ref is None:
            continue
        want = tip4p_table.get(n)
        if want is None or abs(want - ref) > 1e-10:
            raise SystemExit(f"{name}: ref {ref} mismatches tip4p_ccd.csv {want}")


def load_conformer_specs(root: Path) -> list[tuple[str, str, int, float]]:
    mols_path = root / "bench" / "openconf-h2h" / "molecules.csv"
    budgets_path = root / "bench" / "openconf-h2h" / "budgets.csv"
    with mols_path.open(newline="") as handle:
        names = [row["name"] for row in csv.DictReader(handle)]
    with budgets_path.open(newline="") as handle:
        budgets = {row["molecule"]: int(row["anneal_budget"]) for row in csv.DictReader(handle)}
    return [("conformer", name, budgets[name], 0.5) for name in names]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        stamped = stamp_row(row)
        writer.writerow({k: str(stamped.get(k, "")) for k in ROW_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def append_raw(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_raw(path: Path) -> dict[tuple[str, str, int], dict]:
    found: dict[tuple[str, str, int], dict] = {}
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            found[(rec["system"], rec["arm"], int(rec["seed"]))] = rec
    return found


def parse_h2h_seed_row(text: str, kind: str, n: int, seed: int, budget: int) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"{kind}{n} seed {seed}: expected one h2h_row\n{text[-2000:]}")
    values = dict(item.split("=", 1) for item in rows[0].split()[1:])
    if values["system"] != f"{kind}{n}" or int(values["seed"]) != seed or int(values["budget"]) != budget:
        raise RuntimeError(f"{kind}{n} seed {seed}: bad row {rows[0]}")
    return values


def first_trace_hit(trace_path: Path, level: float) -> int | None:
    if not trace_path.is_file():
        return None
    with trace_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["energy"]) <= level:
                return int(row["calls"])
    return None


def run_h2h_seed(
    binary: Path,
    kind: str,
    n: int,
    rho: int,
    budget: int,
    seed: int,
    stop: float | None,
    run_dir: Path,
    revision: str,
    panel: str,
    system: str,
    arm: str,
    order: int,
    tol: float,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_dir / f"{system}-{arm}-{seed}"
    stop_arg = "-" if stop is None else repr(stop)
    cmd = [
        "taskset",
        "-c",
        "0",
        str(binary),
        kind,
        str(n),
        str(rho),
        str(budget),
        str(seed),
        "recommended",
        stop_arg,
        str(prefix),
    ]
    proc = subprocess.run(cmd, cwd=run_dir, env=single_thread_env(), capture_output=True, text=True, check=False)
    (run_dir / "run.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} {arm} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    values = parse_h2h_seed_row(proc.stdout, kind, n, seed, budget)
    trace_file = Path(f"{prefix}.trace.csv")
    trace_sha = sha256_file(trace_file)
    return {
        "panel": panel,
        "system": system,
        "driver": "h2h_seed",
        "arm": arm,
        "seed": seed,
        "order": order,
        "budget": budget,
        "hit": int(values["hit"]),
        "evals_to_hit": int(values["evals_to_hit"]),
        "charged": int(values["charged"]),
        "wall_s": f"{float(values['wall_s']):.6f}",
        "pot_s": f"{float(values['pot_s']):.6f}",
        "outside_s": f"{float(values['outside_s']):.6f}",
        "best": values["best"],
        "reference": "" if stop is None else f"{stop - tol:.10f}",
        "tol": f"{tol}",
        "trace_sha256": trace_sha,
        "trace_path": str(trace_file),
        "code_revision": revision,
        "host": socket.gethostname(),
        "utc": utc_now(),
    }


def run_slab(
    binary: Path,
    root: Path,
    con_rel: str,
    budget: int,
    seed: int,
    run_dir: Path,
    cuh2_timed_lib: Path,
    cuh2_real_lib: Path,
    revision: str,
    system: str,
    arm: str,
    order: int,
    tol: float,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    timer_out = run_dir / "pot.ns"
    timer_out.unlink(missing_ok=True)
    env = single_thread_env()
    env["SEED_OFFSET"] = str(seed)
    env["RGPOT_CUH2_LIBRARY"] = str(cuh2_timed_lib)
    env["GATE_REAL_CUH2_LIB"] = str(cuh2_real_lib)
    env["GATE_POT_TIMER_OUT"] = str(timer_out)
    con_path = root / con_rel
    cmd = ["taskset", "-c", "0", str(binary), str(con_path), str(budget), "1", "recommended"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, check=False)
    wall_s = time.perf_counter() - t0
    (run_dir / "run.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} {arm} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    if not timer_out.is_file():
        raise RuntimeError(f"{system} {arm} seed {seed}: missing pot timer output")
    pot_s = int(timer_out.read_text().strip()) / 1.0e9
    outside_s = max(wall_s - pot_s, 0.0)
    seed_m = SLAB_SEED_RE.search(proc.stdout)
    if not seed_m:
        raise RuntimeError(f"{system} {arm} seed {seed}: no seed line\n{proc.stdout[-2000:]}")
    best = float(seed_m.group("best"))
    charged = int(seed_m.group("charged"))
    improves = [
        (int(m.group("charged")), float(m.group("energy")))
        for m in SLAB_IMPROVE_RE.finditer(proc.stdout)
    ]
    trace_payload = json.dumps({"best": best, "charged": charged, "improves": improves}, sort_keys=True).encode()
    return {
        "panel": "slab",
        "system": system,
        "driver": "slab_adsorption",
        "arm": arm,
        "seed": seed,
        "order": order,
        "budget": budget,
        "charged": charged,
        "wall_s": f"{wall_s:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{outside_s:.6f}",
        "best": f"{best:.10f}",
        "tol": f"{tol}",
        "improves": improves,
        "trace_sha256": sha256_bytes(trace_payload),
        "code_revision": revision,
        "host": socket.gethostname(),
        "utc": utc_now(),
    }


def score_slab_system(records: list[dict], budget: int, tol: float) -> list[dict[str, str]]:
    valid = [float(r["best"]) for r in records if float(r["best"]) < 1.0e5]
    ref = min(valid) if valid else None
    rows: list[dict[str, str]] = []
    for r in sorted(records, key=lambda rec: (int(rec["seed"]), int(rec["order"]))):
        best = float(r["best"])
        hit = 0
        evals = budget
        if ref is not None and best < 1.0e5 and best <= ref + tol:
            hit = 1
            at_hit = next((c for c, e in r["improves"] if e <= ref + tol), None)
            evals = min(at_hit, budget) if at_hit is not None else min(int(r["charged"]), budget)
        rows.append(
            {
                "panel": "slab",
                "system": r["system"],
                "driver": r["driver"],
                "arm": r["arm"],
                "seed": str(r["seed"]),
                "order": str(r["order"]),
                "budget": str(budget),
                "hit": str(hit),
                "evals_to_hit": str(evals),
                "charged": str(r["charged"]),
                "wall_s": r["wall_s"],
                "pot_s": r["pot_s"],
                "outside_s": r["outside_s"],
                "best": r["best"],
                "reference": "" if ref is None else f"{ref:.10f}",
                "tol": f"{tol}",
                "trace_sha256": r["trace_sha256"],
                "code_revision": r["code_revision"],
                "host": r["host"],
                "utc": r["utc"],
            }
        )
    return rows


def run_conformer(
    python_exe: Path,
    root: Path,
    py_dir: Path,
    rdff_lib: Path,
    molecule: str,
    anneal_budget: int,
    seed: int,
    run_dir: Path,
    revision: str,
    arm: str,
    order: int,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    openconf_scripts = root / "scripts" / "openconf_h2h"
    env = single_thread_env()
    env["LD_PRELOAD"] = str(rdff_lib)
    env["PYTHONPATH"] = f"{py_dir}{os.pathsep}{openconf_scripts}"
    cmd = [
        "taskset",
        "-c",
        "0",
        str(python_exe),
        "-u",
        str(openconf_scripts / "h2h_job.py"),
        "anneal",
        molecule,
        str(seed),
        "--out",
        str(run_dir),
        "--budget",
        str(anneal_budget),
    ]
    proc = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, check=False)
    (run_dir / "run.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{molecule} {arm} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    info = json.loads((run_dir / "job.json").read_text())
    if info["hostname"] != "bench":
        raise RuntimeError(f"{molecule} {arm} seed {seed}: hostname {info['hostname']}")
    calls = int(info["energy_calls"]) + int(info["grad_calls"])
    cap = 2 * anneal_budget
    return {
        "panel": "conformer",
        "system": molecule,
        "driver": "openconf_h2h",
        "arm": arm,
        "seed": seed,
        "order": order,
        "anneal_budget": anneal_budget,
        "budget": cap,
        "charged": calls,
        "wall_s": f"{float(info['wall_s']):.6f}",
        "pot_s": f"{float(info['pot_s']):.6f}",
        "outside_s": f"{float(info['outside_s']):.6f}",
        "best_raw_kcal": info.get("best_kcal"),
        "error": info.get("error") or "",
        "trace_sha256": sha256_file(run_dir / "trace.xyz"),
        "run_dir": str(run_dir),
        "code_revision": revision,
        "host": socket.gethostname(),
        "utc": utc_now(),
    }


def _polish_job(args: tuple[str, dict, str]) -> tuple[str, object, object, object]:
    root_str, mol_row, job_dir_str = args
    sys.path.insert(0, str(Path(root_str) / "scripts" / "openconf_h2h"))
    import h2h_common as hc
    from h2h_score import Molecule
    import numpy as np

    job_dir = Path(job_dir_str)
    cache = job_dir / "polished.npz"
    if cache.is_file():
        data = np.load(cache)
        return job_dir_str, data["x"], data["e"], data["ok"]
    mol = Molecule(mol_row)
    path = job_dir / "candidates.xyz"
    frames = hc.read_xyz(path) if path.is_file() else []
    xs, es, ok = [], [], []
    for _, x, _ in frames:
        px, pe, pv = mol.polish(x)
        xs.append(px)
        es.append(pe)
        ok.append(pv)
    xs_arr = np.asarray(xs).reshape(-1, mol.mol.GetNumAtoms(), 3)
    es_arr, ok_arr = np.asarray(es), np.asarray(ok, dtype=bool)
    np.savez(cache, x=xs_arr, e=es_arr, ok=ok_arr)
    return job_dir_str, xs_arr, es_arr, ok_arr


def score_conformer_system(root: Path, molecule: str, records: list[dict], workers: int) -> list[dict[str, str]]:
    sys.path.insert(0, str(root / "scripts" / "openconf_h2h"))
    import h2h_common as hc
    from h2h_score import ENERGY_TIE, RMSD, Molecule

    mol_row = hc.molecules()[molecule]
    mol = Molecule(mol_row)
    jobs_args = [(str(root), mol_row, r["run_dir"]) for r in records]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        polished_list = list(pool.map(_polish_job, jobs_args))
    by_dir: dict[str, tuple] = {}
    all_valid = []
    for job_dir_str, xs, es, ok in polished_list:
        mine = mol.dedupe([(x, float(e)) for x, e, v in zip(xs, es, ok, strict=True) if v])
        by_dir[job_dir_str] = (xs, es, ok, mine)
        all_valid.extend(mine)
    if not all_valid:
        raise RuntimeError(f"{molecule}: no valid polished conformers")
    lowest = min(all_valid, key=lambda item: item[1])
    e_ref = float(lowest[1])
    rows: list[dict[str, str]] = []
    for r in sorted(records, key=lambda rec: (int(rec["seed"]), int(rec["order"]))):
        job_dir = Path(r["run_dir"])
        xs, es, ok, mine = by_dir[r["run_dir"]]
        cap = int(r["budget"])
        calls = int(r["charged"])
        hit = 1 if any(mol.same(lowest, item) for item in mine) else 0
        to_hit = cap
        if hit:
            trace = hc.read_xyz(job_dir / "trace.xyz")
            found_call = None
            for (_, _, comment), (px, pe, pv) in zip(trace, zip(xs, es, ok, strict=True)):
                if pv and mol.same(lowest, (px, float(pe))):
                    found_call = int(comment.split()[0])
                    break
            to_hit = min(found_call if found_call is not None else calls, cap)
        best_valid = min((e for _, e in mine), default=float("nan"))
        rows.append(
            {
                "panel": "conformer",
                "system": molecule,
                "driver": r["driver"],
                "arm": r["arm"],
                "seed": str(r["seed"]),
                "order": str(r["order"]),
                "budget": str(cap),
                "hit": str(hit),
                "evals_to_hit": str(to_hit),
                "charged": str(calls),
                "wall_s": r["wall_s"],
                "pot_s": r["pot_s"],
                "outside_s": r["outside_s"],
                "best": f"{best_valid:.8f}" if best_valid == best_valid else "nan",
                "reference": f"{e_ref:.8f}",
                "tol": f"rmsd<={RMSD},dE<={ENERGY_TIE}",
                "trace_sha256": r["trace_sha256"],
                "code_revision": r["code_revision"],
                "host": r["host"],
                "utc": r["utc"],
            }
        )
    return rows


def self_test() -> None:
    root = Path(__file__).resolve().parents[2]
    verify_references(root)
    confs = load_conformer_specs(root)
    assert len(confs) == 8
    assert confs[0][1] == "rq3" and confs[0][2] == 28457
    text = (
        "system: lj\n"
        "h2h_row system=lj38 code=anneal seed=2000 budget=400000 hit=1 "
        "evals_to_hit=1234 charged=1234 best=-173.9280000000 wall_s=0.100000 "
        "pot_s=0.050000 outside_s=0.050000 basins=3\n"
    )
    kv = parse_h2h_seed_row(text, "lj", 38, 2000, 400_000)
    assert kv["hit"] == "1" and kv["evals_to_hit"] == "1234"
    slab_out = (
        "seed 2000: best -415.971489 eV  hops 480  charged 2500  basins 1  converged 980/981\n"
        "    improve hops=0 charged=19 basins=0 e=-415.96304809\n"
        "    improve hops=268 charged=1432 basins=1 e=-415.97134228\n"
    )
    m = SLAB_SEED_RE.search(slab_out)
    assert m and float(m.group("best")) == -415.971489
    imps = [(int(x.group("charged")), float(x.group("energy"))) for x in SLAB_IMPROVE_RE.finditer(slab_out)]
    assert imps == [(19, -415.96304809), (1432, -415.97134228)]
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("build/no-size-constants-r3-gate"))
    parser.add_argument("--arm-a-bin", type=Path, required=True)
    parser.add_argument("--arm-b-bin", type=Path, required=True)
    parser.add_argument("--arm-a-py", type=Path, required=True)
    parser.add_argument("--arm-b-py", type=Path, required=True)
    parser.add_argument("--arm-a-rev", required=True)
    parser.add_argument("--arm-b-rev", required=True)
    parser.add_argument("--arm-a-tree", required=True)
    parser.add_argument("--arm-b-tree", required=True)
    parser.add_argument("--cuh2-timed-lib", type=Path, required=True)
    parser.add_argument("--cuh2-real-lib", type=Path, required=True)
    parser.add_argument("--rdff-lib", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--panels", nargs="+", default=["slab", "conformer", "tip4p", "lj", "morse"])
    parser.add_argument("--systems", nargs="*")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--polish-workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv[1:])

    if socket.gethostname() != "bench":
        print("uts nodename is not the gate sentinel", file=sys.stderr)
        return 2
    if os.sched_getaffinity(0) != {0}:
        print(f"pinned to {sorted(os.sched_getaffinity(0))}, expected core 0", file=sys.stderr)
        return 2

    global HOST_ID
    HOST_ID = host_id()
    ARMS.clear()
    ARMS["A"] = {"rev": args.arm_a_rev, "tree": args.arm_a_tree}
    ARMS["B"] = {"rev": args.arm_b_rev, "tree": args.arm_b_tree}
    print(f"host_id={HOST_ID}", flush=True)

    verify_references(args.root)
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    raw_path = args.work / "raw_rows.jsonl"
    raw_done = load_raw(raw_path)
    committed_rows = read_rows(args.rows)
    committed_done = {(r["system"], r["arm"], int(r["seed"]), r["code_revision"]) for r in committed_rows}

    arms = {
        "A": {"bin": args.arm_a_bin, "py": args.arm_a_py, "rev": args.arm_a_rev, "tree": args.arm_a_tree},
        "B": {"bin": args.arm_b_bin, "py": args.arm_b_py, "rev": args.arm_b_rev, "tree": args.arm_b_tree},
    }

    def seed_order(seed: int) -> tuple[str, str]:
        return ("A", "B") if seed % 2 == 0 else ("B", "A")

    def wanted(panel: str, system: str) -> bool:
        if panel not in args.panels:
            return False
        return not args.systems or system in args.systems

    # 1. Surface / slab panel
    for panel, system, con_rel, budget, _ref, tol in SLAB_SPECS:
        if not wanted(panel, system):
            continue
        sys_records: list[dict] = []
        for seed in seeds:
            for order, arm in enumerate(seed_order(seed)):
                rev = arms[arm]["rev"]
                key = (system, arm, seed)
                if key in raw_done and raw_done[key]["code_revision"] == rev:
                    sys_records.append(raw_done[key])
                    print(f"skip {system} {arm} {seed}", flush=True)
                    continue
                print(f"start {system} {arm} {seed} order={order}", flush=True)
                rec = run_slab(
                    arms[arm]["bin"] / "slab_adsorption",
                    args.root,
                    con_rel,
                    budget,
                    seed,
                    args.work / "runs" / system / f"{arm}-{seed}",
                    args.cuh2_timed_lib,
                    args.cuh2_real_lib,
                    rev,
                    system,
                    arm,
                    order,
                    tol,
                )
                append_raw(raw_path, rec)
                raw_done[key] = rec
                sys_records.append(rec)
                print(
                    f"done {system} {arm} {seed} best={rec['best']} wall_s={rec['wall_s']} "
                    f"pot_s={rec['pot_s']} outside_s={rec['outside_s']}",
                    flush=True,
                )
        for row in score_slab_system(sys_records, budget, tol):
            rkey = (row["system"], row["arm"], int(row["seed"]), row["code_revision"])
            if rkey not in committed_done:
                append_row(args.rows, row)
                committed_done.add(rkey)

    # 2. Molecular conformer panel
    for panel, system, anneal_budget, _tol in load_conformer_specs(args.root):
        if not wanted(panel, system):
            continue
        sys_records = []
        for seed in seeds:
            for order, arm in enumerate(seed_order(seed)):
                rev = arms[arm]["rev"]
                key = (system, arm, seed)
                if key in raw_done and raw_done[key]["code_revision"] == rev:
                    sys_records.append(raw_done[key])
                    print(f"skip {system} {arm} {seed}", flush=True)
                    continue
                print(f"start {system} {arm} {seed} order={order}", flush=True)
                rec = run_conformer(
                    args.python,
                    args.root,
                    arms[arm]["py"],
                    args.rdff_lib,
                    system,
                    anneal_budget,
                    seed,
                    args.work / "runs" / system / f"{arm}-{seed}",
                    rev,
                    arm,
                    order,
                )
                append_raw(raw_path, rec)
                raw_done[key] = rec
                sys_records.append(rec)
                print(
                    f"done {system} {arm} {seed} calls={rec['charged']} wall_s={rec['wall_s']} "
                    f"pot_s={rec['pot_s']} outside_s={rec['outside_s']}",
                    flush=True,
                )
        for row in score_conformer_system(args.root, system, sys_records, args.polish_workers):
            rkey = (row["system"], row["arm"], int(row["seed"]), row["code_revision"])
            if rkey not in committed_done:
                append_row(args.rows, row)
                committed_done.add(rkey)

    # 3. TIP4P water clusters
    for panel, system, n, budget, ref, tol in TIP4P_SPECS:
        if not wanted(panel, system):
            continue
        stop = None if ref is None else ref + tol
        sys_records = []
        for seed in seeds:
            for order, arm in enumerate(seed_order(seed)):
                rev = arms[arm]["rev"]
                rkey = (system, arm, seed, rev)
                if stop is not None and rkey in committed_done:
                    print(f"skip {system} {arm} {seed}", flush=True)
                    continue
                key = (system, arm, seed)
                if stop is None and key in raw_done and raw_done[key]["code_revision"] == rev:
                    sys_records.append(raw_done[key])
                    print(f"skip {system} {arm} {seed}", flush=True)
                    continue
                print(f"start {system} {arm} {seed} order={order}", flush=True)
                rec = run_h2h_seed(
                    arms[arm]["bin"] / "h2h_seed",
                    "tip4p",
                    n,
                    0,
                    budget,
                    seed,
                    stop,
                    args.work / "runs" / system / f"{arm}-{seed}",
                    rev,
                    panel,
                    system,
                    arm,
                    order,
                    tol,
                )
                append_raw(raw_path, rec)
                raw_done[key] = rec
                if stop is not None:
                    append_row(args.rows, rec)
                    committed_done.add(rkey)
                else:
                    sys_records.append(rec)
                print(
                    f"done {system} {arm} {seed} hit={rec['hit']} calls={rec['evals_to_hit']} "
                    f"best={rec['best']} wall_s={rec['wall_s']} pot_s={rec['pot_s']} outside_s={rec['outside_s']}",
                    flush=True,
                )
        if stop is None:
            pooled_ref = min(float(r["best"]) for r in sys_records)
            level = pooled_ref + tol
            for r in sorted(sys_records, key=lambda rec: (int(rec["seed"]), int(rec["order"]))):
                best = float(r["best"])
                hit = 1 if best <= level else 0
                evals = int(r["budget"])
                if hit:
                    at_hit = first_trace_hit(Path(r["trace_path"]), level)
                    evals = min(at_hit if at_hit is not None else int(r["charged"]), int(r["budget"]))
                row = dict(r, hit=str(hit), evals_to_hit=str(evals), reference=f"{pooled_ref:.10f}")
                rkey = (row["system"], row["arm"], int(row["seed"]), row["code_revision"])
                if rkey not in committed_done:
                    append_row(args.rows, row)
                    committed_done.add(rkey)

    # 4. LJ panel
    for panel, system, n, budget, ref, tol in LJ_SPECS:
        if not wanted(panel, system):
            continue
        stop = ref + tol
        for seed in seeds:
            for order, arm in enumerate(seed_order(seed)):
                rev = arms[arm]["rev"]
                rkey = (system, arm, seed, rev)
                if rkey in committed_done:
                    print(f"skip {system} {arm} {seed}", flush=True)
                    continue
                print(f"start {system} {arm} {seed} order={order}", flush=True)
                rec = run_h2h_seed(
                    arms[arm]["bin"] / "h2h_seed",
                    "lj",
                    n,
                    0,
                    budget,
                    seed,
                    stop,
                    args.work / "runs" / system / f"{arm}-{seed}",
                    rev,
                    panel,
                    system,
                    arm,
                    order,
                    tol,
                )
                append_raw(raw_path, rec)
                append_row(args.rows, rec)
                committed_done.add(rkey)
                print(
                    f"done {system} {arm} {seed} hit={rec['hit']} calls={rec['evals_to_hit']} "
                    f"best={rec['best']} wall_s={rec['wall_s']} pot_s={rec['pot_s']} outside_s={rec['outside_s']}",
                    flush=True,
                )

    # 5. Morse panel (M75 and M80 at rho = 3, 6, 10, 14)
    for panel, system, n, rho, budget, ref, tol in MORSE_SPECS:
        if not wanted(panel, system):
            continue
        stop = ref + tol
        for seed in seeds:
            for order, arm in enumerate(seed_order(seed)):
                rev = arms[arm]["rev"]
                rkey = (system, arm, seed, rev)
                if rkey in committed_done:
                    print(f"skip {system} {arm} {seed}", flush=True)
                    continue
                print(f"start {system} {arm} {seed} order={order}", flush=True)
                rec = run_h2h_seed(
                    arms[arm]["bin"] / "h2h_seed",
                    "morse",
                    n,
                    rho,
                    budget,
                    seed,
                    stop,
                    args.work / "runs" / system / f"{arm}-{seed}",
                    rev,
                    panel,
                    system,
                    arm,
                    order,
                    tol,
                )
                append_raw(raw_path, rec)
                append_row(args.rows, rec)
                committed_done.add(rkey)
                print(
                    f"done {system} {arm} {seed} hit={rec['hit']} calls={rec['evals_to_hit']} "
                    f"best={rec['best']} wall_s={rec['wall_s']} pot_s={rec['pot_s']} outside_s={rec['outside_s']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
