#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Part 1 symmetry-detection microbench runner for the nm3-scan-safe4 gate."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ARMS = ("BASE", "A'", "SAFE2", "SAFE4")

# Balanced 4x4 Williams design (rows 0+2 and 1+3 are exact reverses, giving ABBA symmetry
# and equal (slot, arm) counts across every 4-round block).
WILLIAMS_4 = (
    ("BASE", "A'", "SAFE2", "SAFE4"),
    ("A'", "SAFE4", "BASE", "SAFE2"),
    ("SAFE4", "SAFE2", "A'", "BASE"),
    ("SAFE2", "BASE", "SAFE4", "A'"),
)

CASES = (
    ("detect_all:random33", 2000),
    ("detect_all:random55", 2000),
    ("detect_all:random38", 1000),
    ("detect_all:ring55", 200),
    ("detect_all:packed80", 200),
    ("detect_orders:random33", 2000),
    ("detect_orders:random55", 2000),
    ("detect_orders:random38", 1000),
    ("detect_orders:ring55", 200),
    ("detect_orders:packed80", 200),
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
)

FIELDS = (
    "session_id",
    "seq",
    "case",
    "op",
    "geom",
    "round",
    "slot",
    "arm",
    "core",
    "iters",
    "total_s",
    "per_call_s",
    "digest_hex",
    "candidate_count",
    "bit_identical",
    "machine_id_sha256",
    "library_tree",
    "binary_sha256",
    "harness_blob",
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


def child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("H2H_")}
    for k in THREAD_ENV_VARS:
        env[k] = "1"
    return env


def git_blob_or_sha(repo: Path, rel: str) -> str:
    p = repo / rel
    proc = subprocess.run(
        ["git", "hash-object", str(p)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def run_pinned(argv: list[str], cpu: str, env: dict[str, str], cwd: Path) -> str:
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        if allowed != {int(cpu)}:
            argv = ["taskset", "-c", cpu, *argv]
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--bin-base", type=Path, required=True)
    ap.add_argument("--bin-base2", type=Path, required=True)
    ap.add_argument("--bin-safe2", type=Path, required=True)
    ap.add_argument("--bin-safe4", type=Path, required=True)
    ap.add_argument("--tree-base", required=True)
    ap.add_argument("--tree-safe2", required=True)
    ap.add_argument("--tree-safe4", required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--cpu", default="0")
    args = ap.parse_args()

    if args.rounds < 8 or args.rounds % 4 != 0:
        raise SystemExit("rounds must be >= 8 and divisible by 4 for balanced (slot, arm) counts")

    bins = {
        "BASE": args.bin_base,
        "A'": args.bin_base2,
        "SAFE2": args.bin_safe2,
        "SAFE4": args.bin_safe4,
    }
    trees = {
        "BASE": args.tree_base,
        "A'": args.tree_base,
        "SAFE2": args.tree_safe2,
        "SAFE4": args.tree_safe4,
    }
    bin_sha = {arm: sha256_file(p) for arm, p in bins.items()}
    mid = machine_id_sha256()
    env = child_env()
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.rows.parent.mkdir(parents=True, exist_ok=True)

    harness_blob = ":".join(
        [
            f"run_part1.py={git_blob_or_sha(args.repo, 'bench/nm3-scan-safe4-gate/run_part1.py')}",
            f"microbench/src/main.rs={git_blob_or_sha(args.repo, 'bench/nm3-scan-safe4-gate/microbench/src/main.rs')}",
        ]
    )

    session_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_lines: list[str] = [
        f"session_id={session_id}",
        f"machine_id_sha256={mid}",
        f"arch={platform.machine()} kernel={platform.release()} cpu={args.cpu}",
    ]
    for arm in ARMS:
        log_lines.append(f"arm={arm} tree={trees[arm]} bin_sha256={bin_sha[arm]}")

    # Bit-identity dump verification across all 4 arms for every case.
    bit_ok: dict[str, bool] = {}
    for case, _iters in CASES:
        dumps = {}
        for arm in ARMS:
            dumps[arm] = run_pinned([str(bins[arm]), case, "1", "--dump"], args.cpu, env, args.scratch)
        ref = dumps["BASE"]
        identical = all(dumps[arm] == ref for arm in ARMS)
        bit_ok[case] = identical
        head = ref.splitlines()[:2]
        log_lines.append(f"dump case={case} bit_identical={identical} {' '.join(head)}")
        if not identical:
            log_lines.append(f"MISMATCH on {case}!")

    seq = 0
    for rnd in range(1, args.rounds + 1):
        order = WILLIAMS_4[(rnd - 1) % 4]
        for case, iters in CASES:
            op, geom = case.split(":", 1)
            for slot, arm in enumerate(order, start=1):
                seq += 1
                out = run_pinned([str(bins[arm]), case, str(iters)], args.cpu, env, args.scratch).strip()
                parts = out.split(",")
                if len(parts) != 6 or parts[0] != case or int(parts[1]) != iters:
                    raise RuntimeError(f"unexpected microbench output: {out!r}")
                row = {
                    "session_id": session_id,
                    "seq": str(seq),
                    "case": case,
                    "op": op,
                    "geom": geom,
                    "round": str(rnd),
                    "slot": str(slot),
                    "arm": arm,
                    "core": args.cpu,
                    "iters": str(iters),
                    "total_s": parts[2],
                    "per_call_s": parts[3],
                    "digest_hex": parts[4],
                    "candidate_count": parts[5],
                    "bit_identical": "1" if bit_ok[case] else "0",
                    "machine_id_sha256": mid,
                    "library_tree": trees[arm],
                    "binary_sha256": bin_sha[arm],
                    "harness_blob": harness_blob,
                    "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                append_row(args.rows, row)
        log_lines.append(f"round={rnd}/{args.rounds} done seq={seq}")

    args.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # Quick stdout summary
    with args.rows.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for case, _iters in CASES:
        med = {}
        for arm in ARMS:
            vals = [float(r["per_call_s"]) for r in rows if r["case"] == case and r["arm"] == arm]
            med[arm] = statistics.median(vals)
        ap_val = med["A'"]
        print(
            f"{case:24s} BASE={med['BASE']*1e6:8.2f}us A'={ap_val*1e6:8.2f}us "
            f"SAFE2={med['SAFE2']*1e6:8.2f}us SAFE4={med['SAFE4']*1e6:8.2f}us "
            f"S4/B={med['SAFE4']/med['BASE']:.4f} S4/S2={med['SAFE4']/med['SAFE2']:.4f} "
            f"A'/B={ap_val/med['BASE']:.4f} bit_id={bit_ok[case]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
