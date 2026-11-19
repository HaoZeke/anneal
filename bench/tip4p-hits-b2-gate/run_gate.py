#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial same-session A/B, second-build A/A, and water_hexamer placement-band gate."""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SEEDS = tuple(range(2000, 2012))

ROW_FIELDS = [
    "session_id",
    "band",
    "panel",
    "system",
    "driver",
    "arm",
    "seed",
    "arm_order",
    "run_seq",
    "base_ref",
    "base_tree",
    "harness_run_gate",
    "harness_summarize",
    "harness_snakefile",
    "core",
    "affinity",
    "machine_id_sha256",
    "budget",
    "hit",
    "calls",
    "charged",
    "best",
    "final_energy_bits",
    "reference",
    "tol",
    "wall_s",
    "pot_s",
    "outside_s",
    "trace_sha256",
    "utc",
]

HOST_FIELDS = [
    "session_id",
    "utc",
    "machine_id_sha256",
    "arch",
    "kernel",
    "cpu_model",
    "cpu_count",
    "affinity",
    "mem_kib",
    "rustc",
    "cargo",
    "python_version",
]

TIP4P_SPECS = [
    ("tip4p", "tip4p8", 8, 2_000_000, -305.5183221676, DEFAULT_HIT_TOL),
    ("tip4p", "tip4p15", 15, 2_000_000, -628.3722525077, DEFAULT_HIT_TOL),
    ("tip4p", "tip4p21", 21, 2_000_000, -916.7074302940, DEFAULT_HIT_TOL),
    ("tip4p", "tip4p25", 25, 2_000_000, None, DEFAULT_HIT_TOL),
]

LJ_SPECS = [
    ("lj", "lj38", 38, 400_000, -173.928427, DEFAULT_HIT_TOL),
    ("lj", "lj75", 75, 4_000_000, -397.492331, DEFAULT_HIT_TOL),
    ("lj", "lj98", 98, 4_000_000, -543.665361, DEFAULT_HIT_TOL),
]

CONFORMER_SYSTEMS = ("rq3", "xld")

WATER_HEXAMER_MOLECULES = 6
WATER_HEXAMER_BUDGET = 4000
WATER_HEXAMER_REF_EV = -829.8522275088
WATER_OH_WALL_ONSET_A = 1.15
WATER_OH_WALL_EV_PER_A2 = 200.0
WATER_OVERLAP_LIMIT_A = 0.90

AA_CELLS = ("tip4p8", "tip4p15", "rq3", "lj38", "water_hexamer")

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_session_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes()) if path.is_file() else ""


def float_bits_hex(value: float) -> str:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{raw:016x}"


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return sha256_bytes(raw.encode("utf-8"))
    raise RuntimeError("cannot read machine-id")


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return " ".join(line.split(":", 1)[1].split())
    return platform.processor() or "unknown"


def mem_kib() -> str:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    return ""


def first_line(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None:
        return "absent"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    return lines[0].strip() if proc.returncode == 0 and lines else "absent"


def read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def single_thread_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_", "TWO_PHASE_", "BANK_")):
            del env[key]
    for key in THREAD_ENV_KEYS:
        env[key] = "1"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    env["PYTHONHASHSEED"] = "0"
    env["LJOS_SEAT"] = os.environ.get("LJOS_SEAT", "benchbot")
    if extra:
        env.update(extra)
    return env


def verify_references(root: Path) -> None:
    tip4p_path = root / "data" / "tip4p_ccd.csv"
    tip4p_table: dict[int, float] = {}
    with tip4p_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tip4p_table[int(row["n"])] = float(row["energy"])
    for _panel, name, n, _budget, ref, _tol in TIP4P_SPECS:
        if ref is None:
            continue
        want = tip4p_table.get(n)
        if want is None or abs(want - ref) > 1e-10:
            raise SystemExit(f"{name}: ref {ref} mismatches tip4p_ccd.csv {want}")


def load_conformer_specs(root: Path) -> dict[str, tuple[int, float]]:
    mols_path = root / "bench" / "openconf-h2h" / "molecules.csv"
    budgets_path = root / "bench" / "openconf-h2h" / "budgets.csv"
    with mols_path.open(newline="", encoding="utf-8") as handle:
        names = {row["name"] for row in csv.DictReader(handle)}
    with budgets_path.open(newline="", encoding="utf-8") as handle:
        budgets = {row["molecule"]: int(row["anneal_budget"]) for row in csv.DictReader(handle)}
    out: dict[str, tuple[int, float]] = {}
    for name in CONFORMER_SYSTEMS:
        if name not in names or name not in budgets:
            raise SystemExit(f"missing conformer spec for {name}")
        out[name] = (budgets[name], 0.5)
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle))
        if existing_header != fields:
            raise SystemExit(f"{path} header mismatch; records are append-only")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({k: str(row.get(k, "")) for k in fields})
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, obj: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def append_raw(path: Path, record: dict) -> None:
    append_jsonl(path, record)


def load_raw(path: Path) -> dict[tuple[str, str, str, int], dict]:
    found: dict[tuple[str, str, str, int], dict] = {}
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            found[(rec["band"], rec["system"], rec["arm"], int(rec["seed"]))] = rec
    return found


def parse_h2h_seed_row(text: str, kind: str, n: int, seed: int, budget: int) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"{kind}{n} seed {seed}: expected one h2h_row\n{text[-2000:]}")
    values = dict(item.split("=", 1) for item in rows[0].split()[1:])
    if values["system"] != f"{kind}{n}" or int(values["seed"]) != seed or int(values["budget"]) != budget:
        raise RuntimeError(f"{kind}{n} seed {seed}: bad row {rows[0]}")
    return values


def first_trace_hit(trace_path: Path, target: float, tol: float) -> int | None:
    if not trace_path.is_file():
        return None
    with trace_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if energy_hits(float(row["energy"]), target, tol):
                return int(row["calls"])
    return None


def affinity_text() -> str:
    return ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))


def verify_pinned() -> int:
    if socket.gethostname() != "bench":
        raise RuntimeError(f"uts hostname is {socket.gethostname()}, expected bench")
    aff = sorted(os.sched_getaffinity(0))
    if aff != [0]:
        raise RuntimeError(f"cpu affinity is {aff}, expected [0]")
    return 0


def run_h2h_seed(
    *,
    session_id: str,
    band: str,
    panel: str,
    system: str,
    kind: str,
    n: int,
    budget: int,
    seed: int,
    ref: float | None,
    tol: float,
    arm: str,
    arm_order: str,
    run_seq: int,
    arm_info: dict[str, object],
    run_dir: Path,
    log_path: Path,
    machine_id_sha256: str,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = run_dir / f"{system}-{arm}-{seed}"
    stop_arg = "-" if ref is None else repr(ref + tol)
    cmd = [
        "taskset",
        "-c",
        "0",
        str(Path(arm_info["bin"]) / "h2h_seed"),
        kind,
        str(n),
        "0",
        str(budget),
        str(seed),
        "recommended",
        stop_arg,
        str(prefix),
    ]
    proc = subprocess.run(cmd, cwd=run_dir, env=single_thread_env(), capture_output=True, text=True, check=False)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{band} {system} {arm} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    values = parse_h2h_seed_row(proc.stdout, kind, n, seed, budget)
    trace_file = Path(f"{prefix}.trace.csv")
    trace_sha = sha256_file(trace_file)
    best_str = values["best"]
    best_val = float(best_str)
    hit = 0 if ref is None else int(energy_hits(best_val, ref, tol))
    calls = int(values["evals_to_hit"]) if ref is not None else budget
    return {
        "session_id": session_id,
        "band": band,
        "panel": panel,
        "system": system,
        "driver": "h2h_seed",
        "arm": arm,
        "seed": seed,
        "arm_order": arm_order,
        "run_seq": run_seq,
        "base_ref": str(arm_info["base_ref"]),
        "base_tree": str(arm_info["base_tree"]),
        "harness_run_gate": str(arm_info["harness_run_gate"]),
        "harness_summarize": str(arm_info["harness_summarize"]),
        "harness_snakefile": str(arm_info["harness_snakefile"]),
        "core": "0",
        "affinity": affinity_text(),
        "machine_id_sha256": machine_id_sha256,
        "budget": budget,
        "hit": hit,
        "calls": calls,
        "charged": int(values["charged"]),
        "best": best_str,
        "final_energy_bits": float_bits_hex(best_val),
        "reference": "" if ref is None else f"{ref:.10f}",
        "tol": f"{tol}",
        "wall_s": f"{float(values['wall_s']):.6f}",
        "pot_s": f"{float(values['pot_s']):.6f}",
        "outside_s": f"{float(values['outside_s']):.6f}",
        "trace_sha256": trace_sha,
        "trace_path": str(trace_file),
        "utc": utc_now(),
    }


def run_conformer(
    *,
    session_id: str,
    band: str,
    python_exe: Path,
    root: Path,
    rdff_lib: Path,
    molecule: str,
    anneal_budget: int,
    seed: int,
    arm: str,
    arm_order: str,
    run_seq: int,
    arm_info: dict[str, object],
    run_dir: Path,
    log_path: Path,
    machine_id_sha256: str,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    openconf_scripts = root / "scripts" / "openconf_h2h"
    env = single_thread_env(
        {
            "LD_PRELOAD": str(rdff_lib),
            "PYTHONPATH": f"{arm_info['py']}{os.pathsep}{openconf_scripts}",
        }
    )
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
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{band} {molecule} {arm} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    info = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    if info["hostname"] != "bench":
        raise RuntimeError(f"{molecule} {arm} seed {seed}: hostname {info['hostname']}")
    calls = int(info["energy_calls"]) + int(info["grad_calls"])
    cap = 2 * anneal_budget
    return {
        "session_id": session_id,
        "band": band,
        "panel": "conformer",
        "system": molecule,
        "driver": "openconf_h2h",
        "arm": arm,
        "seed": seed,
        "arm_order": arm_order,
        "run_seq": run_seq,
        "base_ref": str(arm_info["base_ref"]),
        "base_tree": str(arm_info["base_tree"]),
        "harness_run_gate": str(arm_info["harness_run_gate"]),
        "harness_summarize": str(arm_info["harness_summarize"]),
        "harness_snakefile": str(arm_info["harness_snakefile"]),
        "core": "0",
        "affinity": affinity_text(),
        "machine_id_sha256": machine_id_sha256,
        "anneal_budget": anneal_budget,
        "budget": cap,
        "charged": calls,
        "wall_s": f"{float(info['wall_s']):.6f}",
        "pot_s": f"{float(info['pot_s']):.6f}",
        "outside_s": f"{float(info['outside_s']):.6f}",
        "trace_sha256": sha256_file(run_dir / "trace.xyz"),
        "run_dir": str(run_dir),
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
    for r in sorted(records, key=lambda rec: int(rec["run_seq"])):
        job_dir = Path(r["run_dir"])
        xs, es, ok, mine = by_dir[r["run_dir"]]
        cap = int(r["budget"])
        charged = int(r["charged"])
        hit = 1 if any(mol.same(lowest, item) for item in mine) else 0
        to_hit = cap
        if hit:
            trace = hc.read_xyz(job_dir / "trace.xyz")
            found_call = None
            for (_, _, comment), (px, pe, pv) in zip(trace, zip(xs, es, ok, strict=True)):
                if pv and mol.same(lowest, (px, float(pe))):
                    found_call = int(comment.split()[0])
                    break
            to_hit = min(found_call if found_call is not None else charged, cap)
        best_valid = min((e for _, e in mine), default=float("nan"))
        best_str = f"{best_valid:.8f}" if best_valid == best_valid else "nan"
        bits = float_bits_hex(float(best_str)) if best_str != "nan" else float_bits_hex(float("nan"))
        rows.append(
            {
                "session_id": r["session_id"],
                "band": r["band"],
                "panel": "conformer",
                "system": molecule,
                "driver": r["driver"],
                "arm": r["arm"],
                "seed": str(r["seed"]),
                "arm_order": r["arm_order"],
                "run_seq": str(r["run_seq"]),
                "base_ref": r["base_ref"],
                "base_tree": r["base_tree"],
                "harness_run_gate": r["harness_run_gate"],
                "harness_summarize": r["harness_summarize"],
                "harness_snakefile": r["harness_snakefile"],
                "core": r["core"],
                "affinity": r["affinity"],
                "machine_id_sha256": r["machine_id_sha256"],
                "budget": str(cap),
                "hit": str(hit),
                "calls": str(to_hit),
                "charged": str(charged),
                "best": best_str,
                "final_energy_bits": bits,
                "reference": f"{e_ref:.8f}",
                "tol": f"rmsd<={RMSD},dE<={ENERGY_TIE}",
                "wall_s": r["wall_s"],
                "pot_s": r["pot_s"],
                "outside_s": r["outside_s"],
                "trace_sha256": r["trace_sha256"],
                "utc": r["utc"],
            }
        )
    return rows


def read_xyz_coords(path: Path) -> list[float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    n = int(lines[0].split()[0])
    coords: list[float] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        coords.extend(float(v) for v in parts[1:4])
    return coords


def oh_wall(coords_flat):
    import numpy as np

    p = coords_flat.reshape(WATER_HEXAMER_MOLECULES, 3, 3)
    energy = 0.0
    grad = np.zeros_like(p)
    for mol_idx in range(WATER_HEXAMER_MOLECULES):
        oxygen = p[mol_idx, 0]
        for h_idx in (1, 2):
            delta = p[mol_idx, h_idx] - oxygen
            dist = float(np.linalg.norm(delta))
            if dist > WATER_OH_WALL_ONSET_A and dist >= 1e-12:
                excess = dist - WATER_OH_WALL_ONSET_A
                energy += WATER_OH_WALL_EV_PER_A2 * excess * excess
                slope = 2.0 * WATER_OH_WALL_EV_PER_A2 * excess / dist
                grad[mol_idx, h_idx] += delta * slope
                grad[mol_idx, 0] -= delta * slope
    return energy, grad.reshape(-1)


def hexamer_worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xtb-engine", type=Path, required=True)
    parser.add_argument("--start-xyz", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    args = parser.parse_args(argv)

    verify_pinned()
    import anneal
    import numpy as np

    lib = ctypes.CDLL(str(args.xtb_engine))

    class XtbCfg(ctypes.Structure):
        _fields_ = [
            ("method", ctypes.c_int),
            ("accuracy", ctypes.c_double),
            ("etemp", ctypes.c_double),
            ("max_iter", ctypes.c_int),
            ("charge", ctypes.c_double),
            ("uhf", ctypes.c_int),
        ]

    lib.rgpot_xtb_create.argtypes = [ctypes.POINTER(XtbCfg), ctypes.c_char_p, ctypes.c_size_t]
    lib.rgpot_xtb_create.restype = ctypes.c_void_p
    lib.rgpot_xtb_destroy.argtypes = [ctypes.c_void_p]
    lib.rgpot_xtb_force.argtypes = [
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rgpot_xtb_force.restype = ctypes.c_int

    cfg = XtbCfg(3, 0.01, 300.0, 500, 0.0, 0)
    err = ctypes.create_string_buffer(256)
    pot = lib.rgpot_xtb_create(ctypes.byref(cfg), err, 256)
    if not pot:
        raise RuntimeError(f"rgpot_xtb_create failed: {err.value.decode(errors='replace')}")

    try:
        x0 = np.asarray(read_xyz_coords(args.start_xyz), dtype=np.float64)
        n_atoms = 3 * WATER_HEXAMER_MOLECULES
        z = (ctypes.c_int * n_atoms)(*([8, 1, 1] * WATER_HEXAMER_MOLECULES))
        box = (ctypes.c_double * 9)(60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0)
        forces = np.zeros(3 * n_atoms, dtype=np.float64)
        ev = ctypes.c_double()
        vv = ctypes.c_double()
        groups = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(WATER_HEXAMER_MOLECULES)]
        species = [8, 1, 1] * WATER_HEXAMER_MOLECULES

        state = {
            "x": None,
            "e": None,
            "g": None,
            "calls": 0,
            "lowest": float("inf"),
            "first_hit": None,
            "pot_s": 0.0,
            "trace": [],
        }

        def eval_both(x):
            arr = np.ascontiguousarray(x, dtype=np.float64)
            if state["x"] is not None and np.array_equal(arr, state["x"]):
                return state["e"], state["g"]
            pts = arr.reshape(-1, 3)
            d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
            np.fill_diagonal(d2, np.inf)
            if float(np.sqrt(np.min(d2))) <= WATER_OVERLAP_LIMIT_A:
                return float("nan"), np.full(3 * n_atoms, float("nan"), dtype=np.float64)
            t_mark = time.perf_counter()
            rc = lib.rgpot_xtb_force(
                pot,
                n_atoms,
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                z,
                forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(ev),
                ctypes.byref(vv),
                box,
            )
            state["pot_s"] += time.perf_counter() - t_mark
            state["calls"] += 1
            if rc != 0 or not np.isfinite(ev.value):
                return float("nan"), np.full(3 * n_atoms, float("nan"), dtype=np.float64)
            we, wg = oh_wall(arr)
            e_tot = float(ev.value) + we
            g_tot = -forces + wg
            if e_tot < state["lowest"]:
                if e_tot < state["lowest"] - 1e-7:
                    state["trace"].append((state["calls"], e_tot))
                state["lowest"] = e_tot
            if state["first_hit"] is None and energy_hits(e_tot, WATER_HEXAMER_REF_EV, DEFAULT_HIT_TOL):
                state["first_hit"] = state["calls"]
            state["x"] = arr.copy()
            state["e"] = e_tot
            state["g"] = g_tot.copy()
            return e_tot, g_tot

        t0 = time.perf_counter()
        result = anneal.cluster_search(
            lambda x: eval_both(x)[0],
            lambda x: eval_both(x)[1],
            n_atoms,
            args.budget,
            seed=args.seed,
            recommended=True,
            start=x0,
            species=species,
            groups=groups,
            target=WATER_HEXAMER_REF_EV + DEFAULT_HIT_TOL,
        )
        wall_s = time.perf_counter() - t0
    finally:
        lib.rgpot_xtb_destroy(pot)

    best_energy = float(result["best_energy"])
    hit = int(energy_hits(best_energy, WATER_HEXAMER_REF_EV, DEFAULT_HIT_TOL))
    charged = int(state["calls"])
    calls_to_hit = int(state["first_hit"]) if (hit and state["first_hit"] is not None) else args.budget
    pot_s = float(state["pot_s"])
    outside_s = max(wall_s - pot_s, 0.0)

    args.trace_csv.parent.mkdir(parents=True, exist_ok=True)
    trace_lines = ["calls,energy\n"]
    for c_idx, c_e in state["trace"]:
        trace_lines.append(f"{c_idx},{c_e:.10f}\n")
    args.trace_csv.write_text("".join(trace_lines), encoding="utf-8")

    payload = {
        "seed": args.seed,
        "budget": args.budget,
        "hit": hit,
        "calls": calls_to_hit,
        "charged": charged,
        "best": f"{best_energy:.10f}",
        "final_energy_bits": float_bits_hex(best_energy),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{outside_s:.6f}",
        "hops": int(result["hops"]),
        "trace_sha256": sha256_file(args.trace_csv),
        "affinity": affinity_text(),
        "hostname": socket.gethostname(),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


def ensure_hexamer_start(gfn2_bin: Path, seed: int, start_dir: Path) -> Path:
    start_xyz = start_dir / "start.xyz"
    if start_xyz.is_file():
        return start_xyz
    start_dir.mkdir(parents=True, exist_ok=True)
    env = single_thread_env({"H2H_START_ONLY": "1"})
    cmd = ["taskset", "-c", "0", str(gfn2_bin), str(WATER_HEXAMER_MOLECULES), str(WATER_HEXAMER_BUDGET), str(seed), str(start_dir)]
    proc = subprocess.run(cmd, cwd=start_dir, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not start_xyz.is_file():
        raise RuntimeError(f"hexamer start seed {seed} failed: {proc.stderr[-1000:]}")
    return start_xyz


def run_hexamer(
    *,
    session_id: str,
    band: str,
    python_exe: Path,
    xtb_engine: Path,
    start_xyz: Path,
    seed: int,
    arm: str,
    arm_order: str,
    run_seq: int,
    arm_info: dict[str, object],
    run_dir: Path,
    log_path: Path,
    machine_id_sha256: str,
    driver: str = "rgpot_xtb_py",
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "hexamer.json"
    trace_csv = run_dir / f"water_hexamer-{arm}-{seed}.trace.csv"
    env = single_thread_env(
        {
            "PYTHONPATH": str(arm_info["py"]),
            "LD_LIBRARY_PATH": f"{python_exe.parent.parent / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":"),
        }
    )
    cmd = [
        "taskset",
        "-c",
        "0",
        str(python_exe),
        "-u",
        str(Path(__file__).resolve()),
        "--hexamer-worker",
        "--xtb-engine",
        str(xtb_engine),
        "--start-xyz",
        str(start_xyz),
        "--seed",
        str(seed),
        "--budget",
        str(WATER_HEXAMER_BUDGET),
        "--out-json",
        str(out_json),
        "--trace-csv",
        str(trace_csv),
    ]
    proc = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, check=False)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{band} water_hexamer {arm} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    if data["hostname"] != "bench":
        raise RuntimeError(f"hexamer worker hostname {data['hostname']}")
    return {
        "session_id": session_id,
        "band": band,
        "panel": "molecular",
        "system": "water_hexamer",
        "driver": driver,
        "arm": arm,
        "seed": seed,
        "arm_order": arm_order,
        "run_seq": run_seq,
        "base_ref": str(arm_info["base_ref"]),
        "base_tree": str(arm_info["base_tree"]),
        "harness_run_gate": str(arm_info["harness_run_gate"]),
        "harness_summarize": str(arm_info["harness_summarize"]),
        "harness_snakefile": str(arm_info["harness_snakefile"]),
        "core": "0",
        "affinity": data["affinity"],
        "machine_id_sha256": machine_id_sha256,
        "budget": WATER_HEXAMER_BUDGET,
        "hit": int(data["hit"]),
        "calls": int(data["calls"]),
        "charged": int(data["charged"]),
        "best": data["best"],
        "final_energy_bits": data["final_energy_bits"],
        "reference": f"{WATER_HEXAMER_REF_EV:.10f}",
        "tol": f"{DEFAULT_HIT_TOL}",
        "wall_s": data["wall_s"],
        "pot_s": data["pot_s"],
        "outside_s": data["outside_s"],
        "trace_sha256": data["trace_sha256"],
        "utc": utc_now(),
    }


def self_test() -> None:
    assert float_bits_hex(-173.928427) == "0xc065bdb5ac8a3726"
    assert energy_hits(-305.518, -305.5183221676, DEFAULT_HIT_TOL)
    verify_references(ROOT)
    confs = load_conformer_specs(ROOT)
    assert confs["rq3"][0] == 28457 and confs["xld"][0] == 22494
    assert len(SEEDS) == 12 and SEEDS[0] == 2000 and SEEDS[-1] == 2011
    print("ok")


def verify_abba(rows: list[dict[str, str]]) -> None:
    """A finished cell is balanced when each order of the pair occurs on six seeds."""
    orders: dict[tuple[str, str], dict[str, int]] = {}
    seen: dict[tuple[str, str, str], str] = {}
    for row in rows:
        cell = (row["band"], row["system"])
        key = (row["band"], row["system"], row["seed"])
        order = row["arm_order"]
        previous = seen.get(key)
        if previous is None:
            seen[key] = order
            bucket = orders.setdefault(cell, {})
            bucket[order] = bucket.get(order, 0) + 1
        elif previous != order:
            raise SystemExit(f"ABBA order mismatch {cell} seed {row['seed']}: {previous} vs {order}")
    problems: list[str] = []
    for cell, counts in sorted(orders.items()):
        if sum(counts.values()) != len(SEEDS):
            continue
        if sorted(counts.values()) != [len(SEEDS) // 2, len(SEEDS) // 2]:
            problems.append(f"{cell[0]}/{cell[1]} {counts}")
    if problems:
        raise SystemExit("ABBA order is not balanced:\n" + "\n".join(problems))
    if orders:
        print(f"abba_balanced cells={len(orders)}", flush=True)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--hexamer-worker":
        return hexamer_worker_main(argv[2:])
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if "--quick" in argv:
        raise SystemExit("--quick is prohibited by the gate protocol")

    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--rows", type=Path, default=here / "rows.csv")
    parser.add_argument("--sessions", type=Path, default=here / "sessions.jsonl")
    parser.add_argument("--hosts", type=Path, default=here / "hosts.csv")
    parser.add_argument("--work", type=Path, default=Path("build/tip4p-hits-b2-gate"))
    parser.add_argument("--logs", type=Path, default=Path("build/tip4p-hits-b2-gate/logs"))
    parser.add_argument("--a-bin", type=Path, required=True)
    parser.add_argument("--b-bin", type=Path, required=True)
    parser.add_argument("--a2-bin", type=Path, required=True)
    parser.add_argument("--a-py", type=Path, required=True)
    parser.add_argument("--b-py", type=Path, required=True)
    parser.add_argument("--a2-py", type=Path, required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--base-tree", required=True)
    parser.add_argument("--b-kit-sha256", required=True)
    parser.add_argument("--b-kit-tree", required=True)
    parser.add_argument("--harness-run-gate", required=True)
    parser.add_argument("--harness-summarize", required=True)
    parser.add_argument("--harness-snakefile", required=True)
    parser.add_argument("--rdff-lib", type=Path, required=True)
    parser.add_argument("--xtb-engine", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--bands", nargs="+", default=["AB", "AA"], choices=["AB", "AA", "PL"])
    parser.add_argument("--placement-py", nargs=5, type=Path)
    parser.add_argument("--placement-kind", nargs=5)
    parser.add_argument("--systems", nargs="*")
    parser.add_argument("--polish-workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv[1:])

    verify_pinned()
    verify_references(args.root)
    conformer_specs = load_conformer_specs(args.root)
    if "PL" in args.bands:
        if args.placement_py is None or args.placement_kind is None:
            raise SystemExit("PL band requires five --placement-py paths and five --placement-kind labels")
        for kind in args.placement_kind:
            if not kind or any(ch.isspace() for ch in kind) or "/" in kind or "\\" in kind:
                raise SystemExit(f"placement kind {kind!r} must be a single path-free token")

    session_id = now_session_id()
    mid_sha = hashed_machine_id()
    aff = affinity_text()
    utc = utc_now()

    append_csv(
        args.hosts,
        HOST_FIELDS,
        {
            "session_id": session_id,
            "utc": utc,
            "machine_id_sha256": mid_sha,
            "arch": platform.machine(),
            "kernel": platform.release(),
            "cpu_model": cpu_model(),
            "cpu_count": str(os.cpu_count() or 1),
            "affinity": aff,
            "mem_kib": mem_kib(),
            "rustc": first_line(["rustc", "-V"]),
            "cargo": first_line(["cargo", "-V"]),
            "python_version": platform.python_version(),
        },
    )

    append_jsonl(
        args.sessions,
        {
            "schema": "anneal-gate-session-v2",
            "session_id": session_id,
            "utc": utc,
            "machine_id_sha256": mid_sha,
            "kernel": platform.release(),
            "cpu_model": cpu_model(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "base_ref": args.base_ref,
            "base_tree": args.base_tree,
            "b_kit_sha256": args.b_kit_sha256,
            "b_kit_tree": args.b_kit_tree,
            "harness_run_gate": args.harness_run_gate,
            "harness_summarize": args.harness_summarize,
            "harness_snakefile": args.harness_snakefile,
            "a2": "second_release_build",
            "compilers": {
                "rustc": first_line(["rustc", "-V"]),
                "cargo": first_line(["cargo", "-V"]),
                "cc": first_line(["cc", "--version"]),
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "thread_env": {k: "1" for k in THREAD_ENV_KEYS},
            "bands": args.bands,
            "n_seeds": len(SEEDS),
            "seeds": [SEEDS[0], SEEDS[-1]],
            "arm_order_protocol": "ABBA_interleaved_per_seed",
            "placement_kinds": list(args.placement_kind or []),
            "placement_bands": ["AA", "PL1", "PL2", "PL3", "PL4", "PL5"] if "PL" in args.bands else [],
        },
    )

    stamp = {
        "base_ref": args.base_ref,
        "base_tree": args.base_tree,
        "harness_run_gate": args.harness_run_gate,
        "harness_summarize": args.harness_summarize,
        "harness_snakefile": args.harness_snakefile,
    }

    def spec(binary: Path, py_dir: Path) -> dict[str, object]:
        return {"bin": binary, "py": py_dir, **stamp}

    arm_specs = {
        "A": spec(args.a_bin, args.a_py),
        "B": spec(args.b_bin, args.b_py),
        "A1": spec(args.a_bin, args.a_py),
        "A2": spec(args.a2_bin, args.a2_py),
    }

    raw_path = args.work / "raw_rows.jsonl"
    raw_done = load_raw(raw_path)
    committed_rows = read_rows(args.rows)
    committed_done = {
        (r["band"], r["system"], r["arm"], int(r["seed"]))
        for r in committed_rows
        if r.get("base_ref") == args.base_ref
    }

    def wanted(system: str) -> bool:
        return not args.systems or system in args.systems

    run_seq = len(committed_rows)

    def next_seq() -> int:
        nonlocal run_seq
        run_seq += 1
        return run_seq

    def run_h2h_cell(band: str, pair_arms: tuple[str, str], panel: str, system: str, kind: str, n: int, budget: int, ref: float | None, tol: float) -> None:
        if not wanted(system):
            return
        sys_records: list[dict] = []
        for offset, seed in enumerate(SEEDS):
            order = pair_arms if offset % 2 == 0 else (pair_arms[1], pair_arms[0])
            order_tag = "".join(order)
            for arm_name in order:
                info = arm_specs[arm_name]
                rkey = (band, system, arm_name, seed)
                if ref is not None and rkey in committed_done:
                    continue
                key = (band, system, arm_name, seed)
                if ref is None and key in raw_done and raw_done[key].get("base_ref") == args.base_ref:
                    sys_records.append(raw_done[key])
                    continue
                seq = next_seq()
                rec = run_h2h_seed(
                    session_id=session_id,
                    band=band,
                    panel=panel,
                    system=system,
                    kind=kind,
                    n=n,
                    budget=budget,
                    seed=seed,
                    ref=ref,
                    tol=tol,
                    arm=arm_name,
                    arm_order=order_tag,
                    run_seq=seq,
                    arm_info=info,
                    run_dir=args.work / "runs" / band / system / f"{arm_name}-{seed}",
                    log_path=args.logs / band / system / f"{seed}-{arm_name}.log",
                    machine_id_sha256=mid_sha,
                )
                append_raw(raw_path, rec)
                raw_done[key] = rec
                if ref is not None:
                    append_csv(args.rows, ROW_FIELDS, rec)
                    committed_done.add(rkey)
                else:
                    sys_records.append(rec)
                print(
                    f"[{band}] {system} seed={seed} arm={arm_name} hit={rec['hit']} calls={rec['calls']} "
                    f"best={rec['best']} wall={rec['wall_s']} out={rec['outside_s']}",
                    flush=True,
                )
        if ref is None and sys_records:
            pooled_ref = min(float(r["best"]) for r in sys_records)
            for r in sorted(sys_records, key=lambda rec: int(rec["run_seq"])):
                best_val = float(r["best"])
                hit = 1 if energy_hits(best_val, pooled_ref, tol) else 0
                calls = int(r["budget"])
                if hit:
                    at_hit = first_trace_hit(Path(r["trace_path"]), pooled_ref, tol)
                    calls = min(at_hit if at_hit is not None else int(r["charged"]), int(r["budget"]))
                row = dict(r, hit=str(hit), calls=str(calls), reference=f"{pooled_ref:.10f}")
                rkey = (row["band"], row["system"], row["arm"], int(row["seed"]))
                if rkey not in committed_done:
                    append_csv(args.rows, ROW_FIELDS, row)
                    committed_done.add(rkey)

    def run_conformer_cell(band: str, pair_arms: tuple[str, str], molecule: str) -> None:
        if not wanted(molecule):
            return
        anneal_budget, _tol = conformer_specs[molecule]
        sys_records: list[dict] = []
        for offset, seed in enumerate(SEEDS):
            order = pair_arms if offset % 2 == 0 else (pair_arms[1], pair_arms[0])
            order_tag = "".join(order)
            for arm_name in order:
                info = arm_specs[arm_name]
                key = (band, molecule, arm_name, seed)
                if key in raw_done and raw_done[key].get("base_ref") == args.base_ref:
                    sys_records.append(raw_done[key])
                    continue
                seq = next_seq()
                rec = run_conformer(
                    session_id=session_id,
                    band=band,
                    python_exe=args.python,
                    root=args.root,
                    rdff_lib=args.rdff_lib,
                    molecule=molecule,
                    anneal_budget=anneal_budget,
                    seed=seed,
                    arm=arm_name,
                    arm_order=order_tag,
                    run_seq=seq,
                    arm_info=info,
                    run_dir=args.work / "runs" / band / molecule / f"{arm_name}-{seed}",
                    log_path=args.logs / band / molecule / f"{seed}-{arm_name}.log",
                    machine_id_sha256=mid_sha,
                )
                append_raw(raw_path, rec)
                raw_done[key] = rec
                sys_records.append(rec)
                print(
                    f"[{band}] {molecule} seed={seed} arm={arm_name} charged={rec['charged']} "
                    f"wall={rec['wall_s']} out={rec['outside_s']}",
                    flush=True,
                )
        for row in score_conformer_system(args.root, molecule, sys_records, args.polish_workers):
            rkey = (row["band"], row["system"], row["arm"], int(row["seed"]))
            if rkey not in committed_done:
                append_csv(args.rows, ROW_FIELDS, row)
                committed_done.add(rkey)

    def run_hexamer_cell(band: str, pair_arms: tuple[str, str], drivers: dict[str, str] | None = None) -> None:
        if not wanted("water_hexamer"):
            return
        for offset, seed in enumerate(SEEDS):
            start_xyz = ensure_hexamer_start(
                args.a_bin / "gfn2_water_h2h",
                seed,
                args.work / "starts" / f"hexamer-{seed}",
            )
            order = pair_arms if offset % 2 == 0 else (pair_arms[1], pair_arms[0])
            order_tag = "".join(order)
            for arm_name in order:
                info = arm_specs[arm_name]
                rkey = (band, "water_hexamer", arm_name, seed)
                if rkey in committed_done:
                    continue
                seq = next_seq()
                rec = run_hexamer(
                    session_id=session_id,
                    band=band,
                    python_exe=args.python,
                    xtb_engine=args.xtb_engine,
                    start_xyz=start_xyz,
                    seed=seed,
                    arm=arm_name,
                    arm_order=order_tag,
                    run_seq=seq,
                    arm_info=info,
                    run_dir=args.work / "runs" / band / "water_hexamer" / f"{arm_name}-{seed}",
                    log_path=args.logs / band / "water_hexamer" / f"{seed}-{arm_name}.log",
                    machine_id_sha256=mid_sha,
                    driver=(drivers or {}).get(arm_name, "rgpot_xtb_py"),
                )
                append_raw(raw_path, rec)
                append_csv(args.rows, ROW_FIELDS, rec)
                committed_done.add(rkey)
                print(
                    f"[{band}] water_hexamer seed={seed} arm={arm_name} hit={rec['hit']} calls={rec['calls']} "
                    f"best={rec['best']} wall={rec['wall_s']} out={rec['outside_s']}",
                    flush=True,
                )

    if "AA" in args.bands:
        for spec_name in AA_CELLS:
            if spec_name == "rq3":
                run_conformer_cell("AA", ("A1", "A2"), "rq3")
            elif spec_name.startswith("tip4p"):
                for panel, system, n, budget, ref, tol in TIP4P_SPECS:
                    if system == spec_name:
                        run_h2h_cell("AA", ("A1", "A2"), panel, system, "tip4p", n, budget, ref, tol)
            elif spec_name == "water_hexamer":
                run_hexamer_cell("AA", ("A1", "A2"))
            elif spec_name.startswith("lj"):
                for panel, system, n, budget, ref, tol in LJ_SPECS:
                    if system == spec_name:
                        run_h2h_cell("AA", ("A1", "A2"), panel, system, "lj", n, budget, ref, tol)

    if "AB" in args.bands:
        for panel, system, n, budget, ref, tol in TIP4P_SPECS:
            run_h2h_cell("AB", ("A", "B"), panel, system, "tip4p", n, budget, ref, tol)
        run_hexamer_cell("AB", ("A", "B"))
        for molecule in CONFORMER_SYSTEMS:
            run_conformer_cell("AB", ("A", "B"), molecule)
        for panel, system, n, budget, ref, tol in LJ_SPECS:
            run_h2h_cell("AB", ("A", "B"), panel, system, "lj", n, budget, ref, tol)

    if "PL" in args.bands:
        # The water_hexamer A/A is the second-build cell above. Variants differ
        # from A only by object size or section alignment.
        for index, (py_dir, kind) in enumerate(zip(args.placement_py, args.placement_kind, strict=True), start=1):
            arm_specs["V"] = spec(args.a_bin, py_dir)
            run_hexamer_cell(f"PL{index}", ("A", "V"), {"V": kind})

    verify_abba(read_rows(args.rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
