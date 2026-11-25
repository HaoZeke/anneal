#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session restack gate for the conf-miss4 merge.

One serial session on core 0. Arms A, B and A2 (a second release build
of A) run on every cell in palindromic blocks. Molecules use the paired
conformer hit rule. Guards use the default cluster presets. Each row
records outside time, calls and in-process RssAnon. The statistics are
in analyze.py and are fixed by PROTOCOL.md before any data.
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
from typing import TYPE_CHECKING

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "openconf_h2h"))
sys.path.insert(0, str(ROOT / "scripts"))
from bench_host import host_label  # noqa: E402
from machine_provenance import machine_provenance, require_csv_schema  # noqa: E402

if TYPE_CHECKING:
    from h2h_score import Molecule

N_SEEDS = 12
SEEDS = list(range(2000, 2000 + N_SEEDS))
MOLECULES = ["rq3", "915", "xld", "fluvastatin", "z34", "amprenavir", "lopinavir", "lorlatinib"]
GUARDS = ["lj38", "m38r14d", "lj75", "lj98", "m80r6", "au55", "hexamer"]
CELL_ORDER = MOLECULES + GUARDS
GUARD_BUDGET = {
    "lj38": 400_000,
    "m38r14d": 400_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "m80r6": 4_000_000,
    "au55": 1_000_000,
    "hexamer": 500_000,
}
HARNESS_FILES = (
    "bench/gate-pr186-restack/PROTOCOL.md",
    "bench/gate-pr186-restack/run_gate.py",
    "bench/gate-pr186-restack/guard_trace.rs",
    "bench/gate-pr186-restack/analyze.py",
)
ROW_FIELDS = [
    "cell", "kind", "seed", "arm", "slot", "core", "hit", "budget",
    "callback_calls", "calls_to_hit", "charged",
    "wall_s", "inside_s", "outside_per_call_s", "rss_anon_kb", "peak_rss_anon_kb",
    "branch_on_stall", "b_path_draws", "b_path_spent",
    "energy", "energy_bits", "trajectory",
    "arm_revision", "arm_tree", "arm_binary", "arm_binary_sha256",
    "host", "machine_id", "cpu_model", "a_ref", "a_tree", "b_ref", "b_tree",
    "harness_revision", "harness_tree", "harness_blobs", "error",
]


def kind_of(cell: str) -> str:
    return "molecule" if cell in MOLECULES else "guard"


def block(cell: str, pair: int) -> list[tuple[str, int, int]]:
    first = 2000 + 2 * pair
    second = first + 1
    return [
        ("A", first, 0),
        ("B", first, 1),
        ("Ap", first, 2),
        ("Ap", second, 3),
        ("B", second, 4),
        ("A", second, 5),
    ]


def schedule() -> list[dict]:
    jobs = []
    index = 0
    for pair in range(N_SEEDS // 2):
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
    pairs = [(0, "A"), (5, "A"), (1, "B"), (4, "B"), (2, "Ap"), (3, "Ap")]
    return Counter({pair: N_SEEDS // 2 for pair in pairs})


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
            if len(idx) != N_SEEDS:
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


def outside_per_call(wall: float, inside: float, calls: int) -> float:
    if calls <= 0:
        return float("nan")
    return (wall - inside) / calls


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


def resolve_ref(repo: Path, ref: str) -> tuple[str, str]:
    revision = git_text(repo, "rev-parse", f"{ref}^{{commit}}")
    tree = git_text(repo, "rev-parse", f"{revision}^{{tree}}")
    require_object(repo, revision)
    require_object(repo, tree)
    return revision, tree


def provenance(repo: Path, a_ref: str, b_ref: str) -> dict[str, str]:
    a_revision, a_tree = resolve_ref(repo, a_ref)
    b_revision, b_tree = resolve_ref(repo, b_ref)
    harness_revision, harness_tree = resolve_ref(repo, "HEAD")
    blobs = []
    for rel in HARNESS_FILES:
        digest = git_text(repo, "hash-object", rel)
        require_object(repo, digest)
        blobs.append(f"{Path(rel).name}:{digest}")
    return {
        "a_ref": a_revision,
        "a_tree": a_tree,
        "b_ref": b_revision,
        "b_tree": b_tree,
        "harness_revision": harness_revision,
        "harness_tree": harness_tree,
        "harness_blobs": " ".join(blobs),
    }


def ext_binary(py_dir: Path) -> Path:
    matches = sorted((py_dir / "anneal").glob("_core*.so"))
    if len(matches) != 1:
        raise SystemExit(f"expected one _core*.so under {py_dir / 'anneal'}, found {len(matches)}")
    return matches[0]


def arm_source_meta(src: Path, expected_tree: str) -> tuple[str, str]:
    head_tree = git_text(src, "rev-parse", "HEAD^{tree}")
    if head_tree == expected_tree:
        return git_text(src, "rev-parse", "HEAD"), head_tree
    parent_tree = git_text(src, "rev-parse", "HEAD~1^{tree}")
    if parent_tree == expected_tree:
        return git_text(src, "rev-parse", "HEAD~1"), parent_tree
    raise SystemExit(f"arm source tree {head_tree} / {parent_tree} does not match {expected_tree}")


def build_arm_provenance(args, prov: dict[str, str]) -> dict[str, dict[str, str]]:
    sources = {"A": args.src_a, "B": args.src_b, "Ap": args.src_ap}
    expected = {"A": prov["a_tree"], "B": prov["b_tree"], "Ap": prov["a_tree"]}
    out: dict[str, dict[str, str]] = {}
    for arm in ("A", "B", "Ap"):
        rev, tree = arm_source_meta(sources[arm], expected[arm])
        guard_bin = args.bin[arm]
        ext_bin = ext_binary(args.py[arm])
        out[arm] = {
            "arm_revision": rev,
            "arm_tree": tree,
            "guard_binary": guard_bin.name,
            "guard_sha256": sha256_file(guard_bin),
            "ext_binary": ext_bin.name,
            "ext_sha256": sha256_file(ext_bin),
        }
    return out


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


def run_molecule(args, job: dict, budget: int, ident: str, journal: Path, arm_meta: dict[str, str]) -> None:
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
        "--arm-revision", arm_meta["arm_revision"],
        "--arm-tree", arm_meta["arm_tree"],
        "--arm-binary", arm_meta["ext_binary"],
        "--arm-binary-sha256", arm_meta["ext_sha256"],
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
                "budget": 2 * budget,
                "search_budget": budget,
                "callback_calls": 0,
                "calls_to_hit": None,
                "charged": 0,
                "energy_calls": 0,
                "grad_calls": 0,
                "wall_s": elapsed,
                "inside_s": 0.0,
                "rss_anon_kb": 0,
                "peak_rss_anon_kb": 0,
                "branch_on_stall": None,
                "b_path_draws": None,
                "b_path_spent": None,
                "trajectory": hash_trace(out),
                "arm_revision": arm_meta["arm_revision"],
                "arm_tree": arm_meta["arm_tree"],
                "arm_binary": arm_meta["ext_binary"],
                "arm_binary_sha256": arm_meta["ext_sha256"],
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


def rss_anon_kb() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1])
    return 0


def worker_molecule(args) -> int:
    require_core()
    import anneal
    import h2h_common as hc
    import h2h_job

    row = hc.molecules()[args.cell]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    error = ""
    info: dict = {}
    peak_rss = [rss_anon_kb()]
    captured: dict = {}
    orig_cluster_search = anneal.cluster_search

    def wrapped_cluster_search(energy_fn, grad_fn, *cs_args, **cs_kwargs):
        sample_n = [0]

        def sampled_energy(x):
            n = sample_n[0]
            sample_n[0] = n + 1
            if (n & 255) == 0:
                val = rss_anon_kb()
                if val > peak_rss[0]:
                    peak_rss[0] = val
            return energy_fn(x)

        val0 = rss_anon_kb()
        if val0 > peak_rss[0]:
            peak_rss[0] = val0
        res = orig_cluster_search(sampled_energy, grad_fn, *cs_args, **cs_kwargs)
        val1 = rss_anon_kb()
        if val1 > peak_rss[0]:
            peak_rss[0] = val1
        captured["result"] = res
        return res

    anneal.cluster_search = wrapped_cluster_search
    try:
        info = h2h_job.run_anneal(row, args.seed, args.budget, out)
    except Exception as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)
        error = classify_error(str(err)) or "run failed"
        info = {"wall_s": time.perf_counter() - started, "energy_calls": 0, "grad_calls": 0, "error": error}
    finally:
        anneal.cluster_search = orig_cluster_search
    end_rss = rss_anon_kb()
    if end_rss > peak_rss[0]:
        peak_rss[0] = end_rss
    cs_res = captured.get("result") or {}
    branch_on_stall, b_path_draws, b_path_spent = molecular_path_fields(
        cs_res, info
    )
    energy_calls = int(info.get("energy_calls") or 0)
    grad_calls = int(info.get("grad_calls") or 0)
    ledger = info.get("ledger") or {}
    # One unit: a together energy-and-gradient return is one charge.
    # The raw callback total stays in callback_calls.
    charged = int(ledger.get("spent") or info.get("oracle_calls") or 0)
    callback_calls = int(info.get("callback_calls") or (energy_calls + grad_calls))
    raw_error = info.get("error") or error or ""
    if raw_error and not error:
        print(str(raw_error), file=sys.stderr)
    # Phase 0 of the force-field counter is the search potential. Embedding
    # is a different phase and is not part of this inside time.
    inside = float(info.get("pot_s") or 0.0)
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
        "budget": 2 * args.budget,
        "search_budget": args.budget,
        "callback_calls": callback_calls,
        "calls_to_hit": None,
        "charged": charged,
        "energy_calls": energy_calls,
        "grad_calls": grad_calls,
        "wall_s": float(info.get("wall_s") or 0.0),
        "inside_s": inside,
        "best_kcal": None if info.get("best_kcal") is None else float(info["best_kcal"]),
        "candidates": None if info.get("candidates") is None else int(info["candidates"]),
        "rss_anon_kb": end_rss,
        "peak_rss_anon_kb": peak_rss[0],
        "branch_on_stall": branch_on_stall,
        "b_path_draws": b_path_draws,
        "b_path_spent": b_path_spent,
        "trajectory": hash_trace(out),
        "arm_revision": args.arm_revision,
        "arm_tree": args.arm_tree,
        "arm_binary": args.arm_binary,
        "arm_binary_sha256": args.arm_binary_sha256,
        "error": classify_error(str(raw_error)) if raw_error else "",
    }
    if payload["core"] != "0":
        payload["error"] = "wrong core"
        payload["complete"] = True
        write_result(out / "result.json", payload)
        return 3
    write_result(out / "result.json", payload)
    return 0 if not payload["error"] else 1


def molecular_path_fields(cs_res: dict, info: dict) -> tuple[bool | None, int | None, int | None]:
    ledger = cs_res.get("ledger") if isinstance(cs_res, dict) else None
    if not isinstance(ledger, dict):
        candidate = info.get("ledger")
        ledger = candidate if isinstance(candidate, dict) else None
    flag = cs_res.get("branch_on_stall") if isinstance(cs_res, dict) else None
    if not isinstance(flag, bool):
        candidate = info.get("branch_on_stall")
        flag = candidate if isinstance(candidate, bool) else None
    draws = optional_ledger_count(ledger, "branch_draws")
    spent = optional_ledger_count(ledger, "branch_spent")
    return flag, draws, spent


def optional_ledger_count(ledger: dict | None, key: str) -> int | None:
    if ledger is None or key not in ledger or ledger[key] is None:
        return None
    return int(ledger[key])


def run_guard(args, job: dict, binary: Path, ident: str, journal: Path, arm_meta: dict[str, str]) -> None:
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
    invalid_counts = (
        gate is not None
        and (
            int(gate.get("charged", "0")) > GUARD_BUDGET[job["cell"]]
            or int(gate.get("calls", "0")) > GUARD_BUDGET[job["cell"]]
            or gate.get("branch_on_stall") not in ("0", "1")
            or (
                int(gate.get("hit", "0"))
                and int(gate.get("calls", "0")) > int(gate.get("charged", "0"))
            )
        )
    )
    if (
        gate is None
        or gate.get("core") != "0"
        or int(gate.get("budget", "0")) != GUARD_BUDGET[job["cell"]]
        or invalid_counts
    ):
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
                "callback_calls": 0,
                "calls_to_hit": None,
                "charged": 0,
                "wall_s": 0.0,
                "inside_s": 0.0,
                "rss_anon_kb": 0,
                "peak_rss_anon_kb": 0,
                "branch_on_stall": None,
                "b_path_draws": None,
                "b_path_spent": None,
                "energy_bits": "",
                "trajectory": "",
                "arm_revision": arm_meta["arm_revision"],
                "arm_tree": arm_meta["arm_tree"],
                "arm_binary": arm_meta["guard_binary"],
                "arm_binary_sha256": arm_meta["guard_sha256"],
                "error": classify_error(proc.stderr) or "run failed",
                "returncode": proc.returncode,
            },
        )
    else:
        charge_names = (
            "screen", "full", "check", "closure", "quench", "settle", "lattice", "reoccupy", "probe",
        )
        if "screen" in gate:
            named = sum(int(gate[name]) for name in charge_names)
            charged = int(gate["charged"])
            if named != charged:
                raise ValueError(
                    f"{job['cell']} seed {job['seed']}: charged {charged} "
                    f"is not the sum of the named counters {named}"
                )
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
                "callback_calls": int(gate["charged"]),
                "calls_to_hit": int(gate["calls"]) if int(gate["hit"]) else None,
                "charged": int(gate["charged"]),
                "wall_s": float(gate["wall_s"]),
                "inside_s": float(gate["inside_s"]),
                "rss_anon_kb": int(gate["rss_anon_kb"]),
                "peak_rss_anon_kb": int(gate.get("peak_rss_anon_kb") or gate["rss_anon_kb"]),
                "branch_on_stall": int(gate["branch_on_stall"]),
                "b_path_draws": optional_gate_count(gate, "branch_draws"),
                "b_path_spent": optional_gate_count(gate, "branch_spent"),
                "energy_bits": gate["best_bits"],
                "trajectory": gate["trajectory"],
                "arm_revision": arm_meta["arm_revision"],
                "arm_tree": arm_meta["arm_tree"],
                "arm_binary": arm_meta["guard_binary"],
                "arm_binary_sha256": arm_meta["guard_sha256"],
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


def optional_gate_count(gate: dict[str, str], key: str) -> int | None:
    value = gate.get(key)
    return None if value in (None, "na") else int(value)


def bits_to_float(text: str) -> float:
    return struct.unpack("<d", int(text, 16).to_bytes(8, "little"))[0]


def polish_frames(mol: Molecule, directory: Path):
    import h2h_common as hc

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


def first_hit_calls(
    mol: Molecule,
    directory: Path,
    polished: tuple[np.ndarray, np.ndarray, np.ndarray],
    lowest: tuple[np.ndarray, float],
) -> int | None:
    """Return the charged call at which the matching minimum first appeared."""
    import h2h_common as hc

    path = directory / "trace.xyz"
    if not path.is_file():
        return None
    trace = hc.read_xyz(path)
    xs, es, ok = polished
    if len(trace) != len(xs):
        raise ValueError(f"{path}: trace and candidate frame counts differ")
    for (_, _, comment), px, pe, valid in zip(trace, xs, es, ok, strict=True):
        if not valid or not mol.same(lowest, (px, float(pe))):
            continue
        return int(comment.split()[0])
    return None


def score_molecules(run_dir: Path, jobs: list[dict]) -> dict[tuple, dict]:
    import h2h_common as hc
    from h2h_score import Molecule

    rows = hc.molecules()
    scored = {}
    for cell in MOLECULES:
        mol = Molecule(rows[cell])
        mine = [j for j in jobs if j["cell"] == cell]
        kept = {}
        raw = {}
        polished = {}
        for job in mine:
            directory = job_dir(run_dir, job)
            info = json.loads((directory / "result.json").read_text())
            xs, es, ok = polish_frames(mol, directory)
            valid = [(x, float(e)) for x, e, flag in zip(xs, es, ok, strict=True) if flag]
            key = (job["arm"], job["seed"])
            kept[key] = mol.dedupe(valid)
            raw[key] = info
            polished[key] = (xs, es, ok)
        pool = [item for items in kept.values() for item in items]
        lowest = min(pool, key=lambda item: item[1]) if pool else None
        for key, items in kept.items():
            info = raw[key]
            hit = bool(lowest is not None and any(mol.same(lowest, item) for item in items))
            energy = min((e for _, e in items), default=None)
            callback_calls = int(info.get("callback_calls") or 0)
            charged = int(info.get("charged") or 0)
            calls_to_hit = (
                first_hit_calls(mol, job_dir(run_dir, {
                    "phase": "session", "cell": cell, "seed": key[1], "arm": key[0],
                }), polished[key], lowest)
                if hit and lowest is not None
                else None
            )
            budget = int(info.get("budget") or 0)
            if charged > budget:
                raise ValueError(f"{cell} {key}: charged {charged} exceeds budget {budget}")
            if hit and calls_to_hit is None:
                raise ValueError(f"{cell} {key}: hit has no first-hit trace call")
            if calls_to_hit is not None and calls_to_hit > min(charged, budget):
                raise ValueError(
                    f"{cell} {key}: calls to hit {calls_to_hit} exceeds "
                    f"charged {charged} or budget {budget}"
                )
            wall = float(info.get("wall_s") or 0.0)
            inside = float(info.get("inside_s") or 0.0)
            scored[(cell, key[0], key[1])] = {
                "hit": int(hit),
                "energy": energy,
                "callback_calls": callback_calls,
                "calls_to_hit": calls_to_hit,
                "charged": charged,
                "wall_s": wall,
                "inside_s": inside,
                "outside_per_call_s": outside_per_call(wall, inside, callback_calls),
                "budget": budget,
                "rss_anon_kb": int(info.get("rss_anon_kb") or 0),
                "peak_rss_anon_kb": int(info.get("peak_rss_anon_kb") or info.get("rss_anon_kb") or 0),
                "branch_on_stall": info.get("branch_on_stall"),
                "b_path_draws": info.get("b_path_draws"),
                "b_path_spent": info.get("b_path_spent"),
                "error": info.get("error") or "",
                "core": info.get("core") or "",
                "energy_bits": "",
                "trajectory": info.get("trajectory") or hash_trace(job_dir(run_dir, {"phase": "session", "cell": cell, "seed": key[1], "arm": key[0]})),
            }
    return scored


def hash_trace(directory: Path) -> str:
    path = directory / "trace.xyz"
    if not path.is_file():
        return ""
    return sha256_file(path)


def rows_for(records: list[dict], cell: str, arm: str, phase: str = "session") -> list[dict]:
    return [row for row in records if row["cell"] == cell and row["arm"] == arm and row["phase"] == phase]


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
        and int(float(a["callback_calls"])) == int(float(b["callback_calls"]))
        and a["calls_to_hit"] == b["calls_to_hit"]
    )


def guard_identity(records: list[dict], cell: str, arm: str = "B") -> tuple[bool, list[int]]:
    bad = []
    for seed in SEEDS:
        if not identical_pair(records, cell, arm, seed):
            bad.append(seed)
    return (not bad), bad


def assemble_records(
    run_dir: Path,
    jobs: list[dict],
    scored: dict,
    prov: dict,
    arm_prov: dict[str, dict[str, str]],
    machine: dict[str, str],
) -> list[dict]:
    records = []
    for job in jobs:
        info = json.loads((job_dir(run_dir, job) / "result.json").read_text())
        meta = arm_prov[job["arm"]]
        if job["kind"] == "molecule" and job["phase"] == "session":
            extra = scored[(job["cell"], job["arm"], job["seed"])]
            hit = extra["hit"]
            callback_calls = extra["callback_calls"]
            calls_to_hit = extra["calls_to_hit"]
            charged = extra["charged"]
            wall = extra["wall_s"]
            inside = extra["inside_s"]
            energy = "" if extra["energy"] is None else f"{extra['energy']:.6f}"
            bits = ""
            trajectory = extra["trajectory"]
            error = extra["error"]
            core = extra["core"]
            budget = extra["budget"]
            rss = extra["rss_anon_kb"]
            peak_rss = extra["peak_rss_anon_kb"]
            branch_on_stall = extra["branch_on_stall"]
            b_path_draws = extra["b_path_draws"]
            b_path_spent = extra["b_path_spent"]
            arm_binary = info.get("arm_binary") or meta["ext_binary"]
            arm_binary_sha256 = info.get("arm_binary_sha256") or meta["ext_sha256"]
        else:
            hit = int(info.get("hit") or 0)
            callback_calls = int(info.get("callback_calls") or 0)
            calls_to_hit = info.get("calls_to_hit")
            charged = int(info.get("charged") or 0)
            wall = float(info.get("wall_s") or 0.0)
            inside = float(info.get("inside_s") or 0.0)
            bits = info.get("energy_bits") or ""
            energy = f"{bits_to_float(bits):.16e}" if bits else ""
            trajectory = info.get("trajectory") or ""
            error = info.get("error") or ""
            core = info.get("core") or ""
            budget = int(info.get("budget") or 0)
            rss = int(info.get("rss_anon_kb") or 0)
            peak_rss = int(info.get("peak_rss_anon_kb") or rss)
            branch_on_stall = info.get("branch_on_stall")
            b_path_draws = info.get("b_path_draws")
            b_path_spent = info.get("b_path_spent")
            arm_binary = info.get("arm_binary") or meta["guard_binary"]
            arm_binary_sha256 = info.get("arm_binary_sha256") or meta["guard_sha256"]
        outside = outside_per_call(wall, inside, callback_calls)
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
                "budget": budget,
                "callback_calls": callback_calls,
                "calls_to_hit": "" if calls_to_hit is None else calls_to_hit,
                "charged": charged,
                "wall_s": f"{wall:.9f}",
                "inside_s": f"{inside:.9f}",
                "outside_per_call_s": f"{outside:.9e}",
                "rss_anon_kb": rss,
                "peak_rss_anon_kb": peak_rss,
                "branch_on_stall": (
                    "" if branch_on_stall is None else int(branch_on_stall)
                ),
                "b_path_draws": b_path_draws,
                "b_path_spent": b_path_spent,
                "energy": energy,
                "energy_bits": bits,
                "trajectory": trajectory,
                "arm_revision": info.get("arm_revision") or meta["arm_revision"],
                "arm_tree": info.get("arm_tree") or meta["arm_tree"],
                "arm_binary": arm_binary,
                "arm_binary_sha256": arm_binary_sha256,
                "host": host_label(),
                "machine_id": machine["machine_id"],
                "cpu_model": machine["cpu_model"],
                "a_ref": prov["a_ref"],
                "a_tree": prov["a_tree"],
                "b_ref": prov["b_ref"],
                "b_tree": prov["b_tree"],
                "harness_revision": prov["harness_revision"],
                "harness_tree": prov["harness_tree"],
                "harness_blobs": prov["harness_blobs"],
                "error": error,
            }
        )
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        require_csv_schema(path, ROW_FIELDS)
        with path.open(newline="") as handle:
            existing = {(r["cell"], str(r["seed"]), r["arm"]) for r in csv.DictReader(handle)}
        new_rows = [r for r in records if (r["cell"], str(r["seed"]), r["arm"]) not in existing]
        if not new_rows:
            return
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n", extrasaction="ignore")
            for row in new_rows:
                writer.writerow(row)
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def self_check() -> None:
    jobs = schedule()
    check_balance(jobs)
    want = len(CELL_ORDER) * N_SEEDS * 3
    if len(jobs) != want:
        raise SystemExit(f"schedule length {len(jobs)} != {want}")
    load_budgets()


def write_sessions(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def run(args) -> int:
    require_core()
    self_check()
    prov = provenance(args.repo, args.a_ref, args.b_ref)
    arm_prov = build_arm_provenance(args, prov)
    b_head_tree = git_text(args.src_b, "rev-parse", "HEAD^{tree}")
    machine_id, cpu_model = machine_provenance()
    machine = {"machine_id": machine_id, "cpu_model": cpu_model}
    jobs = schedule()
    balances = check_balance(jobs)
    ident = schedule_id(jobs)
    budgets = load_budgets()
    journal = args.run_dir / "journal.log"
    journal.parent.mkdir(parents=True, exist_ok=True)
    log_line(journal, f"seeds {SEEDS[0]}-{SEEDS[-1]} n {len(SEEDS)} cells {len(CELL_ORDER)}")
    log_line(journal, f"session {ident[:12]} jobs {len(jobs)}")
    for position, job in enumerate(jobs, 1):
        meta = arm_prov[job["arm"]]
        if job["kind"] == "molecule":
            run_molecule(args, job, budgets[(job["cell"], job["seed"])], ident, journal, meta)
        else:
            run_guard(args, job, args.bin[job["arm"]], ident, journal, meta)
        if position % 20 == 0:
            log_line(journal, f"progress {position}/{len(jobs)}")
    log_line(journal, "scoring molecules")
    scored = score_molecules(args.run_dir, jobs)
    records = assemble_records(args.run_dir, jobs, scored, prov, arm_prov, machine)
    sessions = [
        ("a_ref", prov["a_ref"]),
        ("a_tree", prov["a_tree"]),
        ("b_ref", prov["b_ref"]),
        ("b_tree", prov["b_tree"]),
        ("b_harness_tree", b_head_tree),
        ("harness_revision", prov["harness_revision"]),
        ("harness_tree", prov["harness_tree"]),
        ("harness_blobs", prov["harness_blobs"]),
        ("host", host_label()),
        ("machine_id", machine["machine_id"]),
        ("cpu_model", machine["cpu_model"]),
        ("seeds", f"{SEEDS[0]}-{SEEDS[-1]}"),
        ("n", str(N_SEEDS)),
        ("core", "0"),
        ("threads", "1"),
        ("schedule", ident),
        ("a_revision", arm_prov["A"]["arm_revision"]),
        ("a_tree_resolved", arm_prov["A"]["arm_tree"]),
        ("b_revision", arm_prov["B"]["arm_revision"]),
        ("b_tree_resolved", arm_prov["B"]["arm_tree"]),
        ("a2_revision", arm_prov["Ap"]["arm_revision"]),
        ("a2_tree", arm_prov["Ap"]["arm_tree"]),
        ("build_a", arm_prov["A"]["guard_sha256"]),
        ("build_b", arm_prov["B"]["guard_sha256"]),
        ("build_a2", arm_prov["Ap"]["guard_sha256"]),
        ("ext_a", arm_prov["A"]["ext_sha256"]),
        ("ext_b", arm_prov["B"]["ext_sha256"]),
        ("ext_a2", arm_prov["Ap"]["ext_sha256"]),
    ]
    for cell, counts in balances.items():
        text = " ".join(f"{slot}:{arm}={count}" for (slot, arm), count in sorted(counts.items()))
        sessions.append((f"balance_{cell}", text))
    for cell, budget in GUARD_BUDGET.items():
        sessions.append((f"budget_{cell}", str(budget)))
    write_csv(args.run_dir / "rows.csv", records)
    write_csv(args.rows, records)
    write_sessions(args.sessions, sessions)
    log_line(journal, "rows written; run analyze.py on rows.csv")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("schedule")

    run_p = sub.add_parser("run")
    run_p.add_argument("--repo", type=Path, required=True)
    run_p.add_argument("--a-ref", required=True)
    run_p.add_argument("--b-ref", required=True)
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
    run_p.add_argument("--rows", type=Path, required=True)
    run_p.add_argument("--sessions", type=Path, required=True)

    worker = sub.add_parser("worker-molecule")
    worker.add_argument("--cell", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--budget", type=int, required=True)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--schedule", required=True)
    worker.add_argument("--arm", required=True)
    worker.add_argument("--slot", type=int, required=True)
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--arm-revision", default="")
    worker.add_argument("--arm-tree", default="")
    worker.add_argument("--arm-binary", default="")
    worker.add_argument("--arm-binary-sha256", default="")

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
