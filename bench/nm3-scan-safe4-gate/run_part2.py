#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Part 2 default-on loses-nowhere panel runner for the nm3-scan-safe4 gate."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import math
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
HIT_TOL = 1e-3
JOB_TIMEOUT_S = 43200.0

# Balanced 6-row mirror-pair schedule over 3 arms (BASE, SAFE4, A').
# Across 12 seeds per cell, every (slot, arm) count is exactly 4.
ARM_SCHEDULE = (
    ("BASE", "SAFE4", "A'"),
    ("A'", "SAFE4", "BASE"),
    ("SAFE4", "A'", "BASE"),
    ("BASE", "A'", "SAFE4"),
    ("A'", "BASE", "SAFE4"),
    ("SAFE4", "BASE", "A'"),
)

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "CARGO_BUILD_JOBS",
)

# Panel cells ordered fast -> slow across all nm3-gate families
# (LJ, Metal/Au, SC, Morse, BLJ, TIP4P; m75r14+m80r14+tip4p20 also cover the MW shard).
CELLS: list[tuple[str, str, str]] = [
    ("lj38", "LJ", "h2h_gmin_proof"),
    ("lj55", "LJ", "h2h_gmin_proof"),
    ("au38", "Metal", "h2h_metal"),
    ("sc96-38", "SC", "h2h_sc"),
    ("sc126-55", "SC", "h2h_sc"),
    ("m80r6", "Morse", "h2h_morse"),
    ("blj115n38", "BLJ", "nm3-gate-driver"),
    ("blj130n38", "BLJ", "nm3-gate-driver"),
    ("blj105n38", "BLJ", "nm3-gate-driver"),
    ("blj105n55", "BLJ", "nm3-gate-driver"),
    ("blj130n55", "BLJ", "nm3-gate-driver"),
    ("blj115n55", "BLJ", "nm3-gate-driver"),
    *[(f"tip4p{n}", "TIP4P", "h2h_seed") for n in range(8, 22)],
    ("lj75", "LJ", "h2h_gmin_proof"),
    ("m75r14", "Morse", "nm3-gate-driver"),
    ("lj98", "LJ", "h2h_gmin_proof"),
    ("lj104", "LJ", "h2h_gmin_proof"),
    ("m80r14", "Morse", "h2h_morse"),
    ("lj103", "LJ", "h2h_gmin_proof"),
    ("blj130n98", "BLJ", "nm3-gate-driver"),
]

FIELDS = (
    "system",
    "arm",
    "seed",
    "slot",
    "core",
    "hit",
    "calls",
    "charged",
    "wall",
    "inside_s",
    "outside_s",
    "best",
    "best_bits",
    "budget",
    "hashed_machine_id",
    "library_tree",
    "harness_blob",
    "panel",
    "driver",
    "reference",
    "outside_per_call_s",
    "arm_order",
    "binary_sha256",
    "session_id",
    "status",
    "utc",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def machine_id_sha256() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    if not raw:
        raise RuntimeError("/etc/machine-id is empty")
    return hashlib.sha256(raw).hexdigest()


def float_bits(val: float) -> str:
    return struct.pack(">d", val).hex()


def child_env(rgpot_sc_lib: Path | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("H2H_")}
    for k in THREAD_ENV_VARS:
        env[k] = "1"
    if rgpot_sc_lib is not None:
        env["RGPOT_SC_LIB"] = str(rgpot_sc_lib)
    return env


def git_blob(repo: Path, rel: str) -> str:
    proc = subprocess.run(
        ["git", "hash-object", str(repo / rel)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def read_tip4p_refs(repo: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with (repo / "data" / "tip4p_ccd.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            if 8 <= n <= 21:
                out[n] = float(row["energy"])
    if tuple(sorted(out)) != tuple(range(8, 22)):
        raise RuntimeError(f"missing tip4p refs: {sorted(out)}")
    return out


def read_tip4p_budget(repo: Path) -> int:
    scripts = repo / "scripts"
    tip4p = scripts / "tip4p_h2h"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(tip4p))
    import importlib
    mod = importlib.import_module("run_h2h")
    return int(mod.BUDGET)


def query_budget(binary: Path, system: str, env: dict[str, str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [str(binary), "budget", system],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"unexpected budget output from {binary} {system}: {proc.stdout!r}")
    return int(lines[0]), lines[1]


def run_job(
    argv: list[str],
    cpu: str,
    env: dict[str, str],
    cwd: Path,
    log: Path,
) -> tuple[str, str]:
    pinned = argv
    if hasattr(os, "sched_getaffinity"):
        if os.sched_getaffinity(0) != {int(cpu)}:
            pinned = ["taskset", "-c", cpu, *argv]
    proc = subprocess.Popen(
        pinned,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=JOB_TIMEOUT_S)
        status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        out, _ = proc.communicate()
        status = "timeout"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"# status={status}\n" + (out or ""), encoding="utf-8")
    return status, out or ""


def parse_result(
    text: str,
    system: str,
    seed: int,
    budget: int,
    reference: float,
) -> dict[str, str]:
    if "h2h_row " in text:
        for line in text.splitlines():
            if line.startswith("h2h_row "):
                parsed = dict(part.split("=", 1) for part in line.split()[1:])
                if parsed.get("system") != system:
                    raise ValueError(f"unexpected system {parsed.get('system')}")
                if int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
                    raise ValueError(f"row mismatch {system} {seed} {budget}")
                best_val = float(parsed["best"])
                hit = "1" if (math.isfinite(best_val) and best_val <= reference + HIT_TOL) else "0"
                calls = int(float(parsed["evals_to_hit"]))
                charged = int(float(parsed["charged"]))
                wall = float(parsed["wall_s"])
                pot = float(parsed["pot_s"])
                outside = max(float(parsed.get("outside_s", wall - pot)), 0.0)
                return {
                    "hit": hit,
                    "calls": str(calls),
                    "charged": str(charged),
                    "wall": f"{wall:.6f}",
                    "inside_s": f"{pot:.6f}",
                    "outside_s": f"{outside:.6f}",
                    "best": parsed["best"],
                    "best_bits": parsed.get("best_bits") or float_bits(best_val),
                }
        raise ValueError("missing h2h_row")

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not line.startswith("system,code,seed,"):
            continue
        header = line.split(",")
        for candidate in lines[idx + 1 :]:
            vals = candidate.split(",")
            if len(vals) != len(header) or vals[0] != system or vals[2] != str(seed):
                continue
            row = dict(zip(header, vals, strict=True))
            if int(float(row["budget"])) != budget:
                raise ValueError(f"budget {row['budget']} != {budget}")
            best_val = float(row["best"])
            hit = "1" if (math.isfinite(best_val) and best_val <= reference + HIT_TOL) else "0"
            calls = int(float(row["evals_to_hit"]))
            charged = int(float(row["charged"]))
            wall = float(row["wall_s"])
            pot = float(row["pot_s"])
            outside = max(float(row.get("outside_s", str(wall - pot))), 0.0)
            return {
                "hit": hit,
                "calls": str(calls),
                "charged": str(charged),
                "wall": f"{wall:.6f}",
                "inside_s": f"{pot:.6f}",
                "outside_s": f"{outside:.6f}",
                "best": row["best"],
                "best_bits": row.get("best_bits") or float_bits(best_val),
            }
    raise ValueError("missing CSV result row")


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--bindir-base", type=Path, required=True)
    ap.add_argument("--bindir-base2", type=Path, required=True)
    ap.add_argument("--bindir-safe4", type=Path, required=True)
    ap.add_argument("--tree-base", required=True)
    ap.add_argument("--tree-safe4", required=True)
    ap.add_argument("--rgpot-sc-lib", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--cpu", default="0")
    args = ap.parse_args()

    arm_dirs = {
        "BASE": args.bindir_base,
        "A'": args.bindir_base2,
        "SAFE4": args.bindir_safe4,
    }
    arm_trees = {
        "BASE": args.tree_base,
        "A'": args.tree_base,
        "SAFE4": args.tree_safe4,
    }

    env = child_env(args.rgpot_sc_lib)
    args.scratch.mkdir(parents=True, exist_ok=True)
    (args.scratch / "tip4p").mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    args.rows.parent.mkdir(parents=True, exist_ok=True)

    mid = machine_id_sha256()
    tip4p_refs = read_tip4p_refs(args.repo)
    tip4p_budget = read_tip4p_budget(args.repo)

    harness_blobs = ":".join(
        [
            f"run_part2.py={git_blob(args.repo, 'bench/nm3-scan-safe4-gate/run_part2.py')}",
            f"driver/src/main.rs={git_blob(args.repo, 'bench/nm3-scan-safe4-gate/driver/src/main.rs')}",
        ]
    )

    bin_sha: dict[tuple[str, str], str] = {}
    for arm, bdir in arm_dirs.items():
        for drv in ("h2h_gmin_proof", "h2h_metal", "h2h_sc", "h2h_morse", "nm3-gate-driver", "h2h_seed"):
            p = bdir / drv
            if not p.is_file():
                raise RuntimeError(f"missing binary {p}")
            bin_sha[(arm, drv)] = sha256_file(p)

    # Preflight budgets & references
    budgets: dict[str, int] = {}
    references: dict[str, tuple[float, str]] = {}
    for system, _panel, driver in CELLS:
        if driver == "h2h_seed":
            n = int(system.removeprefix("tip4p"))
            budgets[system] = tip4p_budget
            ref = tip4p_refs[n]
            references[system] = (ref, f"{ref:.16g}")
        else:
            seen = {
                query_budget(arm_dirs[arm] / driver, system, env, args.scratch)
                for arm in ("BASE", "A'", "SAFE4")
            }
            if len(seen) != 1:
                raise RuntimeError(f"arms disagree on budget/ref for {system}: {seen}")
            b, ref_str = seen.pop()
            budgets[system] = b
            references[system] = (float(ref_str), ref_str)

    existing = read_existing(args.rows)
    done = {
        (r["system"], r["arm"], int(r["seed"]))
        for r in existing
        if r.get("status") == "ok"
    }
    session_id = existing[0]["session_id"] if existing else dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total = len(CELLS) * len(SEEDS) * 3
    print(
        f"session={session_id} machine={mid[:12]} arch={platform.machine()} "
        f"cells={len(CELLS)} total_jobs={total} done={len(done)}",
        flush=True,
    )

    for system, panel, driver in CELLS:
        budget = budgets[system]
        ref_val, ref_str = references[system]
        for seed_idx, seed in enumerate(SEEDS):
            order = ARM_SCHEDULE[seed_idx % 6]
            order_label = ">".join(order)
            for slot, arm in enumerate(order, start=1):
                if (system, arm, seed) in done:
                    continue
                binary = arm_dirs[arm] / driver
                safe_arm = "Ap" if arm == "A'" else arm
                if driver == "h2h_seed":
                    size = system.removeprefix("tip4p")
                    stop = repr(ref_val + HIT_TOL)
                    prefix = args.scratch / "tip4p" / f"{system}-{safe_arm}-s{seed}"
                    argv = [
                        str(binary),
                        "tip4p",
                        size,
                        "0",
                        str(budget),
                        str(seed),
                        "recommended",
                        stop,
                        str(prefix),
                    ]
                else:
                    argv = [str(binary), "run", system, str(seed)]

                log_path = args.logs / f"{system}-{safe_arm}-s{seed}.log"
                status, out = run_job(argv, args.cpu, env, args.scratch, log_path)
                if status != "ok":
                    raise RuntimeError(f"job failed: {system} {arm} seed={seed} status={status}")

                parsed = parse_result(out, system, seed, budget, ref_val)
                outside = float(parsed["outside_s"])
                charged = int(parsed["charged"])
                per_call = outside / charged if charged else float("nan")
                row = {
                    "system": system,
                    "arm": arm,
                    "seed": str(seed),
                    "slot": str(slot),
                    "core": args.cpu,
                    "hit": parsed["hit"],
                    "calls": parsed["calls"],
                    "charged": parsed["charged"],
                    "wall": parsed["wall"],
                    "inside_s": parsed["inside_s"],
                    "outside_s": parsed["outside_s"],
                    "best": parsed["best"],
                    "best_bits": parsed["best_bits"],
                    "budget": str(budget),
                    "hashed_machine_id": mid,
                    "library_tree": arm_trees[arm],
                    "harness_blob": harness_blobs,
                    "panel": panel,
                    "driver": driver,
                    "reference": ref_str,
                    "outside_per_call_s": f"{per_call:.12g}",
                    "arm_order": order_label,
                    "binary_sha256": bin_sha[(arm, driver)],
                    "session_id": session_id,
                    "status": "ok",
                    "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
                }
                append_row(args.rows, row)
                done.add((system, arm, seed))
                print(
                    f"[{len(done)}/{total}] {system} seed={seed} slot={slot} arm={arm} "
                    f"hit={row['hit']} calls={row['calls']} wall={row['wall']} "
                    f"out/call={row['outside_per_call_s']}",
                    flush=True,
                )

    print(f"COMPLETE session={session_id} rows={len(done)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
