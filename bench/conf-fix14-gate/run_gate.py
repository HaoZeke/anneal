#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session gate for the molecular quench default.

One serial session on core 0. Arms A and B, and A' on the noise-band
cells, run in ABBA blocks. Molecules use the paired conformer hit rule.
Guards use the default cluster presets and must match bit for bit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "openconf_h2h"))
sys.path.insert(0, str(ROOT / "scripts"))
import h2h_common as hc  # noqa: E402
from h2h_score import Molecule  # noqa: E402

BASE_REF = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
B_TREE = "2ca02a84d3a5ed6452263d23c0cb810d6a0d3dcc"
KIT_SHA = "8e5600c1070c8d98d2d9936a4c9d70f5afa46f10d35beb61c7f723171cd5fd85"
SEEDS = list(range(2000, 2024))
MOLECULES = ["rq3", "915", "xld", "fluvastatin", "z34", "amprenavir", "lopinavir", "lorlatinib"]
GUARDS = ["lj38", "lj75", "m80r6", "au55", "hexamer"]
AA_CELLS = ("rq3", "xld", "lj38", "hexamer")
CELL_ORDER = MOLECULES + GUARDS
GUARD_BUDGET = {"lj38": 400_000, "lj75": 4_000_000, "m80r6": 4_000_000, "au55": 1_000_000, "hexamer": 500_000}
LAYOUTS = (16, 32, 64, 128, 256)
FAMILY = len(MOLECULES) * 3 + len(GUARDS)
ALPHA = 0.05
HARNESS_FILES = (
    "bench/conf-fix14-gate/run_gate.py",
    "bench/conf-fix14-gate/guard_trace.rs",
)
ROW_FIELDS = [
    "cell", "kind", "phase", "seed", "arm", "slot", "core", "hit", "calls", "charged",
    "wall_s", "energy", "energy_bits", "trajectory", "machine_id", "base_ref", "base_tree",
    "harness_blobs", "budget", "error",
]


def kind_of(cell: str) -> str:
    return "molecule" if cell in MOLECULES else "guard"


def block(cell: str, pair: int) -> list[tuple[str, int, int]]:
    first = 2000 + 2 * pair
    second = first + 1
    if cell in AA_CELLS:
        return [
            ("A", first, 0),
            ("B", first, 1),
            ("Ap", first, 2),
            ("Ap", second, 3),
            ("B", second, 4),
            ("A", second, 5),
        ]
    return [("A", first, 0), ("B", first, 1), ("B", second, 2), ("A", second, 3)]


def schedule() -> list[dict]:
    jobs = []
    index = 0
    for pair in range(12):
        for cell in CELL_ORDER:
            for arm, seed, slot in block(cell, pair):
                jobs.append(
                    {
                        "cell": cell,
                        "kind": kind_of(cell),
                        "phase": "session",
                        "seed": seed,
                        "arm": arm,
                        "slot": slot,
                        "index": index,
                        "block": pair,
                    }
                )
                index += 1
    return jobs


def schedule_id(jobs: list[dict]) -> str:
    blob = json.dumps(
        [(j["cell"], j["seed"], j["arm"], j["slot"], j["index"]) for j in jobs],
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def expected_balance(cell: str) -> Counter:
    if cell in AA_CELLS:
        pairs = [(0, "A"), (5, "A"), (1, "B"), (4, "B"), (2, "Ap"), (3, "Ap")]
    else:
        pairs = [(0, "A"), (3, "A"), (1, "B"), (2, "B")]
    return Counter({pair: 12 for pair in pairs})


def check_balance(jobs: list[dict]) -> dict[str, Counter]:
    found = {}
    for cell in CELL_ORDER:
        mine = [j for j in jobs if j["cell"] == cell]
        counts = Counter((j["slot"], j["arm"]) for j in mine)
        want = expected_balance(cell)
        if counts != want:
            raise SystemExit(f"unbalanced {cell}: {counts} != {want}")
        arms = sorted({j["arm"] for j in mine})
        means = []
        for arm in arms:
            idx = [j["index"] for j in mine if j["arm"] == arm]
            means.append(sum(idx) / len(idx))
            if len(idx) != 24:
                raise SystemExit(f"{cell} {arm} has {len(idx)} runs")
        if max(means) - min(means) > 1e-9:
            raise SystemExit(f"unbalanced position {cell}: {means}")
        found[cell] = counts
    return found


def load_budgets() -> dict[tuple[str, int], int]:
    files = [
        ROOT / "bench" / "openconf-h2h" / "conf-fix4" / "n24" / "seeds.csv",
        ROOT / "bench" / "openconf-h2h" / "conf-fix4" / "seeds.csv",
        ROOT / "bench" / "openconf-h2h" / "conf-fix3" / "seeds.csv",
        ROOT / "bench" / "openconf-h2h" / "conf-fix" / "seeds.csv",
    ]
    found: dict[tuple[str, int], int] = {}
    for path in files:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["arm"] != "anneal" or not row["budget"]:
                    continue
                key = (row["molecule"], int(row["seed"]))
                found.setdefault(key, int(float(row["budget"])))
    budgets = {}
    for molecule in MOLECULES:
        recorded = [found[(molecule, seed)] for seed in SEEDS if (molecule, seed) in found]
        if not recorded:
            raise SystemExit(f"no recorded budget for {molecule}")
        centre = int(statistics.median(recorded))
        for seed in SEEDS:
            budgets[(molecule, seed)] = found.get((molecule, seed), centre)
    return budgets


def thread_env(extra: dict[str, str]) -> dict[str, str]:
    kept = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "RUSTUP_HOME", "CARGO_HOME")
    env = {key: os.environ[key] for key in kept if key in os.environ}
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "GOTO_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "NUMBA_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
            "JULIA_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    env.update(extra)
    return env


def affinity() -> str:
    text = Path("/proc/self/status").read_text()
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def require_core() -> None:
    if affinity() != "0":
        raise SystemExit("this process is not pinned to core 0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def require_object(repo: Path, obj: str) -> None:
    proc = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", obj], capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"git object missing: {obj}")


def provenance(repo: Path) -> dict[str, str]:
    require_object(repo, BASE_REF)
    base_tree = git_text(repo, "rev-parse", f"{BASE_REF}^{{tree}}")
    require_object(repo, base_tree)
    blobs = []
    for rel in HARNESS_FILES:
        digest = git_text(repo, "hash-object", rel)
        require_object(repo, digest)
        blobs.append(f"{Path(rel).name}:{digest}")
    return {"base_ref": BASE_REF, "base_tree": base_tree, "harness_blobs": " ".join(blobs)}


def machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/sys/class/dmi/id/product_uuid")):
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes().strip()).hexdigest()
    raise SystemExit("machine id is unavailable")


def job_dir(run_dir: Path, job: dict) -> Path:
    return run_dir / "jobs" / job["phase"] / job["cell"] / str(job["seed"]) / job["arm"]


def result_ok(path: Path, ident: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return data.get("complete") is True and data.get("schedule") == ident


def classify_error(text: str) -> str:
    if "no finite start" in text or "finite start" in text:
        return "no finite start"
    if text.strip():
        return "run failed"
    return ""


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def log_line(journal: Path, text: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{stamp} {text}\n"
    with journal.open("a") as handle:
        handle.write(line)
    print(text, flush=True)


def run_molecule(args, job: dict, budget: int, ident: str, journal: Path) -> None:
    out = job_dir(args.run_dir, job)
    dest = out / "result.json"
    if result_ok(dest, ident):
        return
    out.mkdir(parents=True, exist_ok=True)
    env = thread_env(
        {
            "PYTHONPATH": str(args.py[job["arm"]]),
            "LD_PRELOAD": str(args.counter),
            "LD_LIBRARY_PATH": str(args.rdkit_lib),
        }
    )
    cmd = [
        "taskset", "-c", "0",
        sys.executable, str(Path(__file__).resolve()),
        "worker-molecule",
        "--cell", job["cell"],
        "--seed", str(job["seed"]),
        "--budget", str(budget),
        "--out", str(out),
        "--schedule", ident,
        "--arm", job["arm"],
        "--slot", str(job["slot"]),
        "--index", str(job["index"]),
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if proc.stdout:
        journal_raw(journal, proc.stdout)
    if proc.stderr:
        journal_raw(journal, proc.stderr)
    if not result_ok(dest, ident):
        write_result(
            dest,
            {
                "complete": True,
                "schedule": ident,
                "cell": job["cell"],
                "kind": "molecule",
                "phase": "session",
                "seed": job["seed"],
                "arm": job["arm"],
                "slot": job["slot"],
                "index": job["index"],
                "core": affinity(),
                "budget": budget,
                "calls": 0,
                "energy_calls": 0,
                "grad_calls": 0,
                "wall_s": elapsed,
                "error": classify_error(proc.stderr) or "run failed",
                "returncode": proc.returncode,
            },
        )
    log_line(journal, f"molecule {job['cell']} seed {job['seed']} {job['arm']} slot {job['slot']} exit {proc.returncode}")


def journal_raw(journal: Path, text: str) -> None:
    with journal.open("a") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def worker_molecule(args) -> int:
    require_core()
    import h2h_job

    row = hc.molecules()[args.cell]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    error = ""
    info: dict = {}
    try:
        info = h2h_job.run_anneal(row, args.seed, args.budget, out)
    except Exception as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)
        error = classify_error(str(err)) or "run failed"
        info = {"wall_s": time.perf_counter() - started, "energy_calls": 0, "grad_calls": 0, "error": error}
    calls = int(info.get("energy_calls") or 0) + int(info.get("grad_calls") or 0)
    raw_error = info.get("error") or error or ""
    if raw_error and not error:
        print(str(raw_error), file=sys.stderr)
    payload = {
        "complete": True,
        "schedule": args.schedule,
        "cell": args.cell,
        "kind": "molecule",
        "phase": "session",
        "seed": args.seed,
        "arm": args.arm,
        "slot": args.slot,
        "index": args.index,
        "core": affinity(),
        "budget": args.budget,
        "calls": calls,
        "energy_calls": int(info.get("energy_calls") or 0),
        "grad_calls": int(info.get("grad_calls") or 0),
        "wall_s": float(info.get("wall_s") or 0.0),
        "best_kcal": None if info.get("best_kcal") is None else float(info["best_kcal"]),
        "candidates": None if info.get("candidates") is None else int(info["candidates"]),
        "error": classify_error(str(raw_error)) if raw_error else "",
    }
    if payload["core"] != "0":
        payload["error"] = "wrong core"
        payload["complete"] = True
        write_result(out / "result.json", payload)
        return 3
    write_result(out / "result.json", payload)
    return 0 if not payload["error"] else 1


def run_guard(args, job: dict, binary: Path, ident: str, journal: Path) -> None:
    out = job_dir(args.run_dir, job)
    dest = out / "result.json"
    if result_ok(dest, ident):
        return
    out.mkdir(parents=True, exist_ok=True)
    env = thread_env({})
    cmd = ["taskset", "-c", "0", str(binary), job["cell"], str(job["seed"])]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.stderr:
        journal_raw(journal, proc.stderr)
    gate = parse_gate(proc.stdout)
    if gate is None or gate.get("core") != "0" or int(gate.get("budget", "0")) != GUARD_BUDGET[job["cell"]]:
        write_result(
            dest,
            {
                "complete": True,
                "schedule": ident,
                "cell": job["cell"],
                "kind": "guard",
                "phase": job["phase"],
                "seed": job["seed"],
                "arm": job["arm"],
                "slot": job["slot"],
                "index": job["index"],
                "core": gate.get("core") if gate else "",
                "budget": GUARD_BUDGET[job["cell"]],
                "hit": 0,
                "calls": 0,
                "charged": 0,
                "wall_s": 0.0,
                "energy_bits": "",
                "trajectory": "",
                "error": classify_error(proc.stderr) or "run failed",
                "returncode": proc.returncode,
            },
        )
    else:
        write_result(
            dest,
            {
                "complete": True,
                "schedule": ident,
                "cell": job["cell"],
                "kind": "guard",
                "phase": job["phase"],
                "seed": job["seed"],
                "arm": job["arm"],
                "slot": job["slot"],
                "index": job["index"],
                "core": gate["core"],
                "budget": int(gate["budget"]),
                "hit": int(gate["hit"]),
                "calls": int(gate["calls"]),
                "charged": int(gate["charged"]),
                "wall_s": float(gate["wall_s"]),
                "energy_bits": gate["best_bits"],
                "trajectory": gate["trajectory"],
                "error": "",
            },
        )
    log_line(
        journal,
        f"guard {job['cell']} seed {job['seed']} {job['arm']} slot {job['slot']} exit {proc.returncode}",
    )


def parse_gate(text: str) -> dict[str, str] | None:
    for line in reversed(text.splitlines()):
        if line.startswith("GATE "):
            fields = {}
            for part in line.split()[1:]:
                key, value = part.split("=", 1)
                fields[key] = value
            return fields
    return None


def bits_to_float(text: str) -> float:
    return struct.unpack("<d", int(text, 16).to_bytes(8, "little"))[0]


def polish_frames(mol: Molecule, directory: Path):
    cache = directory / "polished.npz"
    if cache.is_file():
        data = np.load(cache)
        return data["x"], data["e"], data["ok"]
    path = directory / "candidates.xyz"
    frames = hc.read_xyz(path) if path.is_file() else []
    xs, es, ok = [], [], []
    for _, coords, _ in frames:
        px, pe, pv = mol.polish(coords)
        xs.append(px)
        es.append(pe)
        ok.append(pv)
    xs_arr = np.asarray(xs).reshape(-1, mol.mol.GetNumAtoms(), 3) if xs else np.zeros((0, mol.mol.GetNumAtoms(), 3))
    es_arr = np.asarray(es, dtype=float)
    ok_arr = np.asarray(ok, dtype=bool)
    np.savez(cache, x=xs_arr, e=es_arr, ok=ok_arr)
    return xs_arr, es_arr, ok_arr


def score_molecules(run_dir: Path, jobs: list[dict]) -> dict[tuple, dict]:
    rows = hc.molecules()
    scored = {}
    for cell in MOLECULES:
        mol = Molecule(rows[cell])
        mine = [j for j in jobs if j["cell"] == cell]
        kept = {}
        raw = {}
        for job in mine:
            directory = job_dir(run_dir, job)
            info = json.loads((directory / "result.json").read_text())
            xs, es, ok = polish_frames(mol, directory)
            valid = [(x, float(e)) for x, e, flag in zip(xs, es, ok, strict=True) if flag]
            key = (job["arm"], job["seed"])
            kept[key] = mol.dedupe(valid)
            raw[key] = info
        pool = [item for items in kept.values() for item in items]
        lowest = min(pool, key=lambda item: item[1]) if pool else None
        for key, items in kept.items():
            info = raw[key]
            hit = bool(lowest is not None and any(mol.same(lowest, item) for item in items))
            energy = min((e for _, e in items), default=None)
            scored[(cell, key[0], key[1])] = {
                "hit": int(hit),
                "energy": energy,
                "calls": int(info.get("calls") or 0),
                "charged": int(info.get("calls") or 0),
                "wall_s": float(info.get("wall_s") or 0.0),
                "budget": int(info.get("budget") or 0),
                "error": info.get("error") or "",
                "core": info.get("core") or "",
                "energy_bits": "",
                "trajectory": hash_trace(job_dir(run_dir, {"phase": "session", "cell": cell, "seed": key[1], "arm": key[0]})),
            }
    return scored


def hash_trace(directory: Path) -> str:
    path = directory / "trace.xyz"
    if not path.is_file():
        return ""
    return sha256_file(path)


def p_of(result) -> float:
    return float(result.pvalue if hasattr(result, "pvalue") else result[1])


def fisher_worse(b_hits: int, a_hits: int, n: int) -> float:
    return p_of(fisher_exact([[b_hits, n - b_hits], [a_hits, n - a_hits]], alternative="less"))


def mw_worse(b_values: list[float], a_values: list[float]) -> float:
    if not a_values or not b_values:
        return 1.0
    if len(set(a_values + b_values)) == 1:
        return 1.0
    return p_of(mannwhitneyu(b_values, a_values, alternative="greater", method="exact"))


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def rows_for(records: list[dict], cell: str, arm: str, phase: str = "session") -> list[dict]:
    return [row for row in records if row["cell"] == cell and row["arm"] == arm and row["phase"] == phase]


def numeric(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def ratio(num: float, den: float) -> float:
    if den == 0:
        return float("inf") if num > 0 else 1.0
    return num / den


def aa_wall_ratios(records: list[dict], cells: list[str]) -> list[float]:
    ratios = []
    for cell in cells:
        a = {int(row["seed"]): float(row["wall_s"]) for row in rows_for(records, cell, "A")}
        ap = {int(row["seed"]): float(row["wall_s"]) for row in rows_for(records, cell, "Ap")}
        for seed, wall_a in a.items():
            if seed in ap:
                ratios.append(ratio(ap[seed], wall_a))
    return ratios


def layout_ratios(records: list[dict], cell: str) -> list[float]:
    a = {int(row["seed"]): float(row["wall_s"]) for row in rows_for(records, cell, "A")}
    ratios = []
    arms = sorted({row["arm"] for row in records if row["cell"] == cell and row["phase"] == "placement"})
    for arm in arms:
        for row in rows_for(records, cell, arm, "placement"):
            seed = int(row["seed"])
            if seed in a and row["error"] == "" and identical_pair(records, cell, arm, seed, phase_b="placement"):
                ratios.append(ratio(float(row["wall_s"]), a[seed]))
    return ratios


def identical_pair(records: list[dict], cell: str, arm: str, seed: int, phase_b: str = "session") -> bool:
    a_rows = [row for row in rows_for(records, cell, "A") if int(row["seed"]) == seed and row["error"] == ""]
    b_rows = [row for row in rows_for(records, cell, arm, phase_b) if int(row["seed"]) == seed and row["error"] == ""]
    if len(a_rows) != 1 or len(b_rows) != 1:
        return False
    a, b = a_rows[0], b_rows[0]
    return (
        a["trajectory"] == b["trajectory"]
        and a["trajectory"] != ""
        and a["energy_bits"] == b["energy_bits"]
        and int(float(a["charged"])) == int(float(b["charged"]))
        and int(float(a["calls"])) == int(float(b["calls"]))
    )


def guard_identity(records: list[dict], cell: str, arm: str = "B") -> tuple[bool, list[int]]:
    bad = []
    for seed in SEEDS:
        if not identical_pair(records, cell, arm, seed):
            bad.append(seed)
    return (not bad), bad


def adjust(p: float) -> float:
    return min(1.0, p * FAMILY)


def endpoint_loss(p: float, worse: bool, outside: bool) -> bool:
    return worse and outside and adjust(p) < ALPHA


def band_limit(ratios: list[float]) -> float:
    return max(ratios) if ratios else 1.0


def judge_cell(records: list[dict], cell: str, wall_limits: dict[str, float]) -> dict:
    a = rows_for(records, cell, "A")
    b = rows_for(records, cell, "B")
    n = 24
    hits_a = sum(int(row["hit"]) for row in a)
    hits_b = sum(int(row["hit"]) for row in b)
    calls_a = numeric(a, "calls")
    calls_b = numeric(b, "calls")
    wall_a = numeric(a, "wall_s")
    wall_b = numeric(b, "wall_s")
    p_hits = fisher_worse(hits_b, hits_a, n) if kind_of(cell) == "molecule" else 1.0
    p_calls = mw_worse(calls_b, calls_a) if kind_of(cell) == "molecule" else 1.0
    p_wall = mw_worse(wall_b, wall_a)
    med_calls_a, med_calls_b = median(calls_a), median(calls_b)
    med_wall_a, med_wall_b = median(wall_a), median(wall_b)
    wall_ratio = ratio(med_wall_b, med_wall_a)
    call_ratio = ratio(med_calls_b, med_calls_a)
    outside_wall = wall_ratio > wall_limits[cell]
    outside_calls = call_ratio > wall_limits.get(f"calls:{cell}", 1.0)
    hit_band = wall_limits.get(f"hits:{cell}", 0.0)
    outside_hits = (hits_a - hits_b) > hit_band
    identical, bad = (True, [])
    if kind_of(cell) == "guard":
        identical, bad = guard_identity(records, cell)
    losses = []
    if kind_of(cell) == "molecule":
        if endpoint_loss(p_hits, hits_b < hits_a, outside_hits):
            losses.append("hits")
        if endpoint_loss(p_calls, med_calls_b > med_calls_a, outside_calls):
            losses.append("calls")
        if endpoint_loss(p_wall, med_wall_b > med_wall_a, outside_wall):
            losses.append("wall")
        verdict = "FAIL" if losses else "PASS"
    else:
        if not identical:
            losses.append("identity")
        if endpoint_loss(p_wall, med_wall_b > med_wall_a, outside_wall):
            losses.append("wall")
        verdict = "FAIL" if losses else "PASS"
    ap = rows_for(records, cell, "Ap")
    return {
        "cell": cell,
        "kind": kind_of(cell),
        "hits_a": hits_a,
        "hits_b": hits_b,
        "hits_ap": sum(int(row["hit"]) for row in ap) if ap else "",
        "p_hits": p_hits,
        "p_calls": p_calls,
        "p_wall": p_wall,
        "calls_a": med_calls_a,
        "calls_b": med_calls_b,
        "wall_a": med_wall_a,
        "wall_b": med_wall_b,
        "wall_ratio": wall_ratio,
        "wall_limit": wall_limits[cell],
        "identical": identical,
        "mismatch": bad,
        "losses": losses,
        "verdict": verdict,
        "n_a": len(a),
        "n_b": len(b),
    }


def hit_bands(records: list[dict]) -> dict[str, float]:
    own = {}
    for cell in AA_CELLS:
        if kind_of(cell) != "molecule":
            continue
        a = sum(int(row["hit"]) for row in rows_for(records, cell, "A"))
        ap = sum(int(row["hit"]) for row in rows_for(records, cell, "Ap"))
        own[cell] = abs(a - ap)
    pool = max(own.values()) if own else 0.0
    return {cell: own.get(cell, pool) for cell in MOLECULES}


def call_bands(records: list[dict]) -> dict[str, float]:
    own = {}
    for cell in ("rq3", "xld"):
        a = {int(row["seed"]): float(row["calls"]) for row in rows_for(records, cell, "A")}
        ap = {int(row["seed"]): float(row["calls"]) for row in rows_for(records, cell, "Ap")}
        ratios = [ratio(ap[seed], a[seed]) for seed in a if seed in ap]
        own[cell] = band_limit(ratios)
    pool = max(own.values()) if own else 1.0
    return {cell: own.get(cell, pool) for cell in MOLECULES}


def wall_bands(records: list[dict], placement_cells: set[str]) -> dict[str, float]:
    mol_own = {}
    for cell in ("rq3", "xld"):
        ratios = []
        a = {int(row["seed"]): float(row["wall_s"]) for row in rows_for(records, cell, "A")}
        ap = {int(row["seed"]): float(row["wall_s"]) for row in rows_for(records, cell, "Ap")}
        ratios.extend(ratio(ap[seed], a[seed]) for seed in a if seed in ap)
        mol_own[cell] = band_limit(ratios)
    mol_pool = max(mol_own.values()) if mol_own else 1.0
    guard_pool_ratios = aa_wall_ratios(records, ["lj38", "hexamer"])
    limits = {}
    for cell in MOLECULES:
        limits[cell] = mol_own.get(cell, mol_pool)
    for cell in GUARDS:
        ratios = aa_wall_ratios(records, [cell] if cell in AA_CELLS else ["lj38", "hexamer"])
        if not ratios:
            ratios = guard_pool_ratios
        if cell in placement_cells:
            ratios = ratios + layout_ratios(records, cell)
        limits[cell] = band_limit(ratios)
    return limits


def assemble_records(run_dir: Path, jobs: list[dict], scored: dict, prov: dict, machine: str) -> list[dict]:
    records = []
    for job in jobs:
        info = json.loads((job_dir(run_dir, job) / "result.json").read_text())
        if job["kind"] == "molecule" and job["phase"] == "session":
            extra = scored[(job["cell"], job["arm"], job["seed"])]
            hit = extra["hit"]
            calls = extra["calls"]
            charged = extra["charged"]
            wall = extra["wall_s"]
            energy = "" if extra["energy"] is None else f"{extra['energy']:.6f}"
            bits = ""
            trajectory = extra["trajectory"]
            error = extra["error"]
            budget = extra["budget"]
            core = extra["core"]
        else:
            hit = int(info.get("hit") or 0)
            calls = int(info.get("calls") or 0)
            charged = int(info.get("charged") or 0)
            wall = float(info.get("wall_s") or 0.0)
            bits = info.get("energy_bits") or ""
            energy = f"{bits_to_float(bits):.16e}" if bits else ""
            trajectory = info.get("trajectory") or ""
            error = info.get("error") or ""
            budget = int(info.get("budget") or 0)
            core = info.get("core") or ""
        records.append(
            {
                "cell": job["cell"],
                "kind": job["kind"],
                "phase": job["phase"],
                "seed": job["seed"],
                "arm": job["arm"],
                "slot": job["slot"],
                "core": core,
                "hit": hit,
                "calls": calls,
                "charged": charged,
                "wall_s": f"{wall:.9f}",
                "energy": energy,
                "energy_bits": bits,
                "trajectory": trajectory,
                "machine_id": machine,
                "base_ref": prov["base_ref"],
                "base_tree": prov["base_tree"],
                "harness_blobs": prov["harness_blobs"],
                "budget": budget,
                "error": error,
            }
        )
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def fmt_p(value: float) -> str:
    return f"{value:.4e}"


def fmt_num(value: float) -> str:
    if value != value:
        return ""
    if abs(value - round(value)) < 1e-9 and abs(value) > 1:
        return str(int(round(value)))
    return f"{value:.6g}"


def write_report(path: Path, verdicts: list[dict], records: list[dict], sessions: list[tuple[str, str]], placement_note: str) -> None:
    lines = [
        "Copyright (c) anneal developers",
        "",
        "Same-session comparison of develop and the conf-fix14 tree.",
        "A is develop " + BASE_REF + ".",
        "B is that commit plus the conf-fix14 kit, tree " + B_TREE + ", kit sha256 " + KIT_SHA + ".",
        "A' is a second release build of A. The kit is not part of this tree.",
        "",
        "Seeds are 2000-2023, twenty-four per arm. Molecule caps are the per-seed",
        "anneal budgets of the paired conformer benchmark; a seed with no recorded",
        "budget uses the median of that molecule's recorded budgets. The search is",
        "cluster_search on the recommended molecular preset. No preset field is overridden.",
        "",
        "A molecule hit is a reported conformer that polishes to the lowest valid",
        "conformer of the union of the session's arms and seeds. Polish is MMFF94",
        "at dielectric one until the minimiser reports convergence. A conformer",
        "whose stereocentres or double bonds differ from the SMILES is invalid and",
        "is dropped. Two valid geometries are the same conformer when their",
        "heavy-atom RMSD, minimised over graph automorphisms after a Kabsch fit,",
        "is under 0.25 angstrom and their energies differ by less than 0.5 kcal/mol.",
        "Molecule calls are energy plus gradient force-field calls. A guard call",
        "is the charged count at the published minimum, or the whole budget on a miss.",
        "A guard hit uses DEFAULT_HIT_TOL (0.001) against the published minimum.",
        "The guard pass rule is bit-identity of the trajectory hash, the charged",
        "count, the call count and the best-energy bits on every seed.",
        "",
        "Tests are one-sided and exact: Fisher on hits, in the direction of fewer",
        "B hits, and Mann-Whitney on calls and wall, in the direction of larger B",
        "values. The family is the eight molecules' three endpoints and the five",
        f"guard wall endpoints, {FAMILY} tests. A loss requires an adjusted p below",
        f"{ALPHA}, a worse median or hit count, and a median ratio outside the A/A",
        "or placement band. The band limit is the largest per-seed A'/A ratio on",
        "that cell, or the largest such ratio pooled from the measured A/A cells",
        "when the cell has no A'.",
        "",
        "## Verdict",
        "",
        "| cell | hits A | hits B | hits A' | p hits | calls A | calls B | p calls | wall A | wall B | p wall | wall limit | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in verdicts:
        lines.append(
            "| {cell} | {ha} | {hb} | {hp} | {ph} | {ca} | {cb} | {pc} | {wa} | {wb} | {pw} | {lim} | {verdict} |".format(
                cell=row["cell"],
                ha=f"{row['hits_a']}/24",
                hb=f"{row['hits_b']}/24",
                hp="" if row["hits_ap"] == "" else f"{row['hits_ap']}/24",
                ph=fmt_p(row["p_hits"]),
                ca=fmt_num(row["calls_a"]),
                cb=fmt_num(row["calls_b"]),
                pc=fmt_p(row["p_calls"]),
                wa=fmt_num(row["wall_a"]),
                wb=fmt_num(row["wall_b"]),
                pw=fmt_p(row["p_wall"]),
                lim=fmt_num(row["wall_limit"]),
                verdict=row["verdict"],
            )
        )
    overall = "PASS" if all(row["verdict"] == "PASS" and row["n_a"] == 24 and row["n_b"] == 24 for row in verdicts) else "FAIL"
    lines += ["", f"Overall: {overall}", "", "## Guard bit-identity", ""]
    lines.append("| cell | identical to A | mismatched seeds |")
    lines.append("|---|---|---|")
    for row in verdicts:
        if row["kind"] != "guard":
            continue
        seeds = " ".join(str(seed) for seed in row["mismatch"]) if row["mismatch"] else ""
        lines.append(f"| {row['cell']} | {str(row['identical']).lower()} | {seeds} |")
    lines += ["", "## A/A band", ""]
    lines.append("| cell | arm | median wall ratio to A | median call ratio to A | hit difference |")
    lines.append("|---|---|---|---|---|")
    for cell in AA_CELLS:
        for arm in ("Ap", "B"):
            a = rows_for(records, cell, "A")
            other = rows_for(records, cell, arm)
            if not a or not other:
                continue
            lines.append(
                "| {cell} | {arm} | {wall} | {calls} | {hits} |".format(
                    cell=cell,
                    arm="A'" if arm == "Ap" else arm,
                    wall=fmt_num(ratio(median(numeric(other, "wall_s")), median(numeric(a, "wall_s")))),
                    calls=fmt_num(ratio(median(numeric(other, "calls")), median(numeric(a, "calls")))),
                    hits=sum(int(r["hit"]) for r in other) - sum(int(r["hit"]) for r in a),
                )
            )
    if placement_note:
        lines += ["", "## Placement", "", placement_note]
    lines += ["", "## Session", ""]
    for key, value in sessions:
        lines.append(f"- {key}: {value}")
    lines.append("")
    path.write_text("\n".join(lines))


def session_rows(prov: dict, machine: str, balances: dict[str, Counter], builds: dict[str, str], kit: str) -> list[tuple[str, str]]:
    rows = [
        ("base_ref", prov["base_ref"]),
        ("base_tree", prov["base_tree"]),
        ("b_tree", B_TREE),
        ("kit_sha256", kit),
        ("harness_blobs", prov["harness_blobs"]),
        ("machine_id", machine),
        ("seeds", "2000-2023"),
        ("n", "24"),
        ("core", "0"),
        ("threads", "1"),
        ("family", str(FAMILY)),
        ("alpha", str(ALPHA)),
        ("build_a", builds["A"]),
        ("build_b", builds["B"]),
        ("build_ap", builds["Ap"]),
    ]
    for cell, counts in balances.items():
        text = " ".join(f"{slot}:{arm}={count}" for (slot, arm), count in sorted(counts.items()))
        rows.append((f"balance_{cell}", text))
    for cell, budget in GUARD_BUDGET.items():
        rows.append((f"budget_{cell}", str(budget)))
    return rows


def write_sessions(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def patch_ledger(path: Path, align: int) -> None:
    text = path.read_text()
    marker = "pub struct Ledger {"
    attr = f"#[repr(align({align}))]"
    if marker not in text:
        raise SystemExit("Ledger struct was not found")
    if f"{attr}\n{marker}" not in text:
        text = text.replace(marker, attr + "\n" + marker, 1)
        path.write_text(text)


def build_layout(args, align: int, journal: Path) -> Path:
    dest = args.run_dir / "layout" / f"align-{align}"
    binary = dest / "target" / "release" / "examples" / "guard_trace"
    if binary.is_file() and (dest / "src" / "methods" / "cluster_hopping.rs").read_text().count(f"#[repr(align({align}))]"):
        return binary
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    subprocess.run(
        ["cp", "-a", str(args.src_a), str(dest)],
        check=True,
    )
    git_file = dest / ".git"
    if git_file.exists() or git_file.is_symlink():
        git_file.unlink()
    example = dest / "examples" / "guard_trace.rs"
    if not example.is_file():
        raise SystemExit("layout tree has no guard example")
    patch_ledger(dest / "src" / "methods" / "cluster_hopping.rs", align)
    env = os.environ.copy()
    env["RUSTUP_TOOLCHAIN"] = args.toolchain
    env["CARGO_TARGET_DIR"] = str(dest / "target")
    env.pop("RAYON_NUM_THREADS", None)
    env.pop("LD_PRELOAD", None)
    env.pop("LD_LIBRARY_PATH", None)
    cpus = os.cpu_count() or 1
    span = "0" if cpus == 1 else f"0-{cpus - 1}"
    log_line(journal, f"build layout align {align}")
    proc = subprocess.run(
        ["taskset", "-c", span, "cargo", "build", "--release", "--locked", "--example", "guard_trace"],
        cwd=dest,
        env=env,
        capture_output=True,
        text=True,
    )
    journal_raw(journal, proc.stderr[-4000:])
    if proc.returncode != 0 or not binary.is_file():
        raise SystemExit(f"layout build failed for align {align}")
    return binary


def placement_jobs(cell: str) -> list[dict]:
    jobs = []
    index = 0
    for seed in SEEDS:
        for slot, align in enumerate(LAYOUTS):
            jobs.append(
                {
                    "cell": cell,
                    "kind": "guard",
                    "phase": "placement",
                    "seed": seed,
                    "arm": f"L{align}",
                    "slot": slot,
                    "index": index,
                    "block": 0,
                    "align": align,
                }
            )
            index += 1
    return jobs


def needs_placement(records: list[dict], cell: str, limit: float) -> bool:
    identical, _ = guard_identity(records, cell)
    if not identical:
        return False
    a = median(numeric(rows_for(records, cell, "A"), "wall_s"))
    b = median(numeric(rows_for(records, cell, "B"), "wall_s"))
    return ratio(b, a) > limit


def self_check() -> None:
    jobs = schedule()
    check_balance(jobs)
    if len(jobs) != 720:
        raise SystemExit(f"schedule length {len(jobs)}")
    tiny = fisher_worse(0, 24, 24)
    if tiny > 1e-6:
        raise SystemExit(f"fisher self-check {tiny}")
    slow = mw_worse([5, 6, 7, 8], [1, 2, 3, 4])
    if slow > 0.05:
        raise SystemExit(f"mannwhitney self-check {slow}")
    flat = mw_worse([1, 1, 1, 1], [1, 1, 1, 1])
    if flat != 1.0:
        raise SystemExit(f"tie self-check {flat}")
    load_budgets()


def run(args) -> int:
    require_core()
    self_check()
    kit = sha256_file(args.kit)
    if kit != KIT_SHA:
        raise SystemExit("kit sha256 does not match")
    b_tree = git_text(args.src_b, "rev-parse", "HEAD^{tree}")
    if b_tree != B_TREE:
        raise SystemExit("B tree does not match")
    a_tree = git_text(args.src_a, "rev-parse", "HEAD^{tree}")
    ap_tree = git_text(args.src_ap, "rev-parse", "HEAD^{tree}")
    if a_tree != ap_tree or a_tree != git_text(args.repo, "rev-parse", f"{BASE_REF}^{{tree}}"):
        raise SystemExit("A and A' are not the develop tree")
    prov = provenance(args.repo)
    machine = machine_id()
    jobs = schedule()
    balances = check_balance(jobs)
    ident = schedule_id(jobs)
    budgets = load_budgets()
    journal = args.run_dir / "journal.log"
    journal.parent.mkdir(parents=True, exist_ok=True)
    log_line(journal, f"session {ident[:12]} jobs {len(jobs)}")
    for position, job in enumerate(jobs, 1):
        if job["kind"] == "molecule":
            run_molecule(args, job, budgets[(job["cell"], job["seed"])], ident, journal)
        else:
            run_guard(args, job, args.bin[job["arm"]], ident, journal)
        if position % 20 == 0:
            log_line(journal, f"progress {position}/{len(jobs)}")
    log_line(journal, "scoring molecules")
    scored = score_molecules(args.run_dir, jobs)
    records = assemble_records(args.run_dir, jobs, scored, prov, machine)
    # Provisional bands from A/A only. Placement, if required, widens guard walls.
    call_limit = call_bands(records)
    hit_limit = hit_bands(records)
    wall_limit = wall_bands(records, set())
    limits = dict(wall_limit)
    for cell, value in call_limit.items():
        limits[f"calls:{cell}"] = value
    for cell, value in hit_limit.items():
        limits[f"hits:{cell}"] = value
    placement_cells = [cell for cell in GUARDS if needs_placement(records, cell, wall_limit[cell])]
    note = ""
    if placement_cells:
        log_line(journal, "placement " + " ".join(placement_cells))
        extra_jobs = []
        binaries = {}
        for align in LAYOUTS:
            binaries[align] = build_layout(args, align, journal)
        for cell in placement_cells:
            for job in placement_jobs(cell):
                extra_jobs.append(job)
                run_guard(args, job, binaries[job["align"]], ident, journal)
        records = assemble_records(args.run_dir, jobs + extra_jobs, scored, prov, machine)
        wall_limit = wall_bands(records, set(placement_cells))
        for cell, value in wall_limit.items():
            limits[cell] = value
        note = (
            "Five release builds of A change only the alignment of Ledger "
            "(16, 32, 64, 128, 256). A variant enters the band only when its "
            "trajectory, calls and energy bits match A on that seed. "
            + "Cells: "
            + " ".join(placement_cells)
            + "."
        )
    verdicts = [judge_cell(records, cell, limits) for cell in CELL_ORDER]
    builds = {arm: sha256_file(args.bin[arm]) for arm in ("A", "B", "Ap")}
    sessions = session_rows(prov, machine, balances, builds, kit)
    sessions.append(("a_tree", a_tree))
    sessions.append(("placement", " ".join(placement_cells) if placement_cells else "none"))
    write_csv(args.run_dir / "rows.csv", records)
    write_csv(args.rows, records)
    write_sessions(args.sessions, sessions)
    write_report(args.report, verdicts, records, sessions, note)
    log_line(journal, "overall " + ("PASS" if all(v["verdict"] == "PASS" for v in verdicts) else "FAIL"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("schedule")

    run_p = sub.add_parser("run")
    run_p.add_argument("--repo", type=Path, required=True)
    run_p.add_argument("--run-dir", type=Path, required=True)
    run_p.add_argument("--src-a", type=Path, required=True)
    run_p.add_argument("--src-b", type=Path, required=True)
    run_p.add_argument("--src-ap", type=Path, required=True)
    run_p.add_argument("--bin-a", type=Path, required=True)
    run_p.add_argument("--bin-b", type=Path, required=True)
    run_p.add_argument("--bin-ap", type=Path, required=True)
    run_p.add_argument("--py-a", type=Path, required=True)
    run_p.add_argument("--py-b", type=Path, required=True)
    run_p.add_argument("--py-ap", type=Path, required=True)
    run_p.add_argument("--counter", type=Path, required=True)
    run_p.add_argument("--rdkit-lib", type=Path, required=True)
    run_p.add_argument("--kit", type=Path, required=True)
    run_p.add_argument("--rows", type=Path, required=True)
    run_p.add_argument("--sessions", type=Path, required=True)
    run_p.add_argument("--report", type=Path, required=True)
    run_p.add_argument("--toolchain", default="1.88.0")

    worker = sub.add_parser("worker-molecule")
    worker.add_argument("--cell", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--budget", type=int, required=True)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--schedule", required=True)
    worker.add_argument("--arm", required=True)
    worker.add_argument("--slot", type=int, required=True)
    worker.add_argument("--index", type=int, required=True)

    args = parser.parse_args()
    if args.cmd == "schedule":
        self_check()
        jobs = schedule()
        counts = check_balance(jobs)
        for cell, counter in counts.items():
            text = " ".join(f"{slot}:{arm}={n}" for (slot, arm), n in sorted(counter.items()))
            print(f"{cell} {text}")
        print(f"jobs {len(jobs)} id {schedule_id(jobs)[:12]}")
        return 0
    if args.cmd == "worker-molecule":
        return worker_molecule(args)
    args.bin = {"A": args.bin_a, "B": args.bin_b, "Ap": args.bin_ap}
    args.py = {"A": args.py_a, "B": args.py_b, "Ap": args.py_ap}
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
