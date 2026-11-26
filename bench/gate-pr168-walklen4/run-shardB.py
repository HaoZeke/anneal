#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial ABBA gate for LJ75 and LJ38, shard B.

One process at a time, pinned to core 0. Both arms take the same
``h2h_seed`` arguments: paper budget, recommended preset, stop at the
reference plus the repository default hit tolerance. A miss is stored
as budget + 1. Outside-potential CPU time to a hit is process CPU
(user + system) minus the seconds the driver reports inside energy and
force calls. That difference is recorded only when the run hits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "paper_0120"))

from machine_provenance import host_label  # noqa: E402
from run_panel import PAPER_BUDGETS  # noqa: E402
from target_tol import DEFAULT_HIT_TOL  # noqa: E402

SEEDS = tuple(range(2120, 2132))
SYSTEMS = (
    ("lj75", 75, -397.492331),
    ("lj38", 38, -173.928427),
)
FIELDS = [
    "arm",
    "system",
    "seed",
    "hit",
    "calls_to_hit",
    "outside_cpu_to_hit_s",
    "cpu_s",
    "wall_s",
    "peak_rss_kb",
    "code_revision",
    "host_id",
    "core",
    "budget",
    "best",
    "pot_s",
    "charged",
    "harness_evals",
    "harness_outside_s",
    "search_wall_s",
    "tree",
    "revision_kind",
    "binary_sha256",
    "order_index",
    "slot",
]
CORE = "0"


def schedule() -> list[tuple[str, int, float, int, int, str, int]]:
    """LJ75 then LJ38. Even seeds run A then B; odd seeds run B then A."""
    jobs = []
    for system, n, reference in SYSTEMS:
        budget = int(PAPER_BUDGETS[system])
        for index, seed in enumerate(SEEDS):
            arms = ("A", "B") if index % 2 == 0 else ("B", "A")
            for slot, arm in enumerate(arms, start=1):
                jobs.append((system, n, reference, budget, seed, arm, slot))
    return jobs


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[key] = "1"
    env["LC_ALL"] = "C"
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_h2h(text: str) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"expected one h2h_row, saw {len(rows)}")
    return dict(item.split("=", 1) for item in rows[0].split()[1:])


def run_pinned(binary: Path, args: list[str], env: dict[str, str]) -> tuple[int, str, str, dict[str, float]]:
    """Run ``binary`` on core 0 and return its own rusage.

    ``wait4`` reports that process only: user plus system CPU, and
    ``ru_maxrss`` in kilobytes. Wall time is the parent's clock around
    the child.
    """
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    started = time.perf_counter()
    pid = os.fork()
    if pid == 0:
        try:
            os.dup2(out_w, 1)
            os.dup2(err_w, 2)
            for fd in (out_r, out_w, err_r, err_w):
                os.close(fd)
            os.sched_setaffinity(0, {0})
            os.execve(str(binary), [str(binary), *args], env)
        finally:
            os._exit(127)
    os.close(out_w)
    os.close(err_w)
    chunks = {out_r: [], err_r: []}
    pending = {out_r, err_r}
    while pending:
        readable, _, _ = select.select(list(pending), [], [])
        for fd in readable:
            data = os.read(fd, 1 << 16)
            if data:
                chunks[fd].append(data)
            else:
                os.close(fd)
                pending.remove(fd)
    _wpid, status, usage = os.wait4(pid, 0)
    timing = {
        "cpu_s": usage.ru_utime + usage.ru_stime,
        "wall_s": time.perf_counter() - started,
        "peak_rss_kb": int(usage.ru_maxrss),
    }
    code = os.waitstatus_to_exitcode(status)
    stdout = b"".join(chunks[out_r]).decode()
    stderr = b"".join(chunks[err_r]).decode()
    return code, stdout, stderr, timing


def have(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {(row["arm"], row["system"], row["seed"]) for row in csv.DictReader(handle)}


def append_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def check_pin() -> None:
    proc = subprocess.run(
        ["taskset", "-c", CORE, sys.executable, "-c", "import os; print(sorted(os.sched_getaffinity(0)))"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or proc.stdout.strip() != "[0]":
        raise SystemExit("taskset did not pin a process to core 0")


def run_one(
    binary: Path,
    system: str,
    n: int,
    reference: float,
    budget: int,
    seed: int,
    log_path: Path,
) -> tuple[dict[str, str], dict[str, float]]:
    stop = f"{reference + DEFAULT_HIT_TOL:.6f}"
    with tempfile.TemporaryDirectory() as scratch:
        prefix = str(Path(scratch) / "run")
        args = ["lj", str(n), "0", str(budget), str(seed), "recommended", stop, prefix]
        code, stdout, stderr, timing = run_pinned(binary, args, single_thread_env())
        logical = " ".join(["h2h_seed", *args[:-1]])
        log_path.write_text(logical + "\n" + stdout + stderr)
        if code != 0:
            raise RuntimeError(f"{system} seed {seed} exited {code}")
    parsed = parse_h2h(stdout)
    if parsed.get("system") != f"lj{n}" or int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
        raise RuntimeError(f"row does not match the request: {parsed}")
    return parsed, timing


def row_from(
    parsed: dict[str, str],
    timing: dict[str, float],
    *,
    arm: str,
    system: str,
    seed: int,
    budget: int,
    meta: dict[str, str],
    order_index: int,
    slot: int,
) -> dict[str, str]:
    hit = parsed["hit"] == "1"
    harness_evals = int(parsed["evals_to_hit"])
    if hit and harness_evals > budget:
        raise RuntimeError(f"{system} {arm} seed {seed} hit past the budget")
    if not hit and harness_evals != budget:
        raise RuntimeError(f"{system} {arm} seed {seed} miss evals {harness_evals} != {budget}")
    pot_s = float(parsed["pot_s"])
    outside = timing["cpu_s"] - pot_s
    return {
        "arm": arm,
        "system": system,
        "seed": str(seed),
        "hit": "1" if hit else "0",
        "calls_to_hit": str(harness_evals if hit else budget + 1),
        "outside_cpu_to_hit_s": f"{outside:.6f}" if hit else "",
        "cpu_s": f"{timing['cpu_s']:.6f}",
        "wall_s": f"{timing['wall_s']:.6f}",
        "peak_rss_kb": str(timing["peak_rss_kb"]),
        "code_revision": meta["revision"],
        "host_id": meta["host_id"],
        "core": CORE,
        "budget": str(budget),
        "best": parsed["best"],
        "pot_s": f"{pot_s:.6f}",
        "charged": parsed["charged"],
        "harness_evals": str(harness_evals),
        "harness_outside_s": f"{float(parsed['outside_s']):.6f}",
        "search_wall_s": f"{float(parsed['wall_s']):.6f}",
        "tree": meta["tree"],
        "revision_kind": meta["kind"],
        "binary_sha256": meta["sha256"],
        "order_index": str(order_index),
        "slot": str(slot),
    }


def main() -> int:
    if len(SEEDS) != 12:
        raise SystemExit("seed count")
    if DEFAULT_HIT_TOL != 1.0e-3:
        raise SystemExit("hit tolerance is not the repository default")
    if PAPER_BUDGETS["lj75"] != 4_000_000 or PAPER_BUDGETS["lj38"] != 400_000:
        raise SystemExit("paper budgets moved")
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-bin", type=Path, required=True)
    parser.add_argument("--b-bin", type=Path, required=True)
    parser.add_argument("--a-revision", required=True)
    parser.add_argument("--b-revision", required=True)
    parser.add_argument("--a-tree", required=True)
    parser.add_argument("--b-tree", required=True)
    parser.add_argument("--a-kind", default="commit")
    parser.add_argument("--b-kind", default="tree")
    parser.add_argument("--rows", type=Path, default=Path(__file__).resolve().parent / "rows-shardB.csv")
    parser.add_argument("--sessions", type=Path, default=Path(__file__).resolve().parent / "sessions-shardB.jsonl")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--dry-schedule", action="store_true")
    args = parser.parse_args()
    jobs = schedule()
    if args.dry_schedule:
        for order, job in enumerate(jobs, start=1):
            system, _n, _reference, budget, seed, arm, slot = job
            print(f"{order} {system} seed {seed} arm {arm} slot {slot} budget {budget}")
        return 0
    check_pin()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    host = host_label()
    metas = {}
    for arm, binary, revision, tree, kind in (
        ("A", args.a_bin, args.a_revision, args.a_tree, args.a_kind),
        ("B", args.b_bin, args.b_revision, args.b_tree, args.b_kind),
    ):
        if not binary.is_file():
            raise SystemExit(f"missing binary for {arm}")
        metas[arm] = {
            "revision": revision,
            "tree": tree,
            "kind": kind,
            "sha256": sha256_file(binary),
            "host_id": host,
        }
    done = have(args.rows)
    binaries = {"A": args.a_bin, "B": args.b_bin}
    for order, (system, n, reference, budget, seed, arm, slot) in enumerate(jobs, start=1):
        if (arm, system, str(seed)) in done:
            print(f"{order}/{len(jobs)} {system} {arm} seed {seed} kept", flush=True)
            continue
        log_path = args.log_dir / f"{order:03d}-{system}-{arm}-{seed}.log"
        parsed, timing = run_one(binaries[arm], system, n, reference, budget, seed, log_path)
        row = row_from(
            parsed,
            timing,
            arm=arm,
            system=system,
            seed=seed,
            budget=budget,
            meta=metas[arm],
            order_index=order,
            slot=slot,
        )
        session = dict(row)
        session["status"] = "ok"
        session["cli"] = (
            f"h2h_seed lj {n} 0 {budget} {seed} recommended "
            f"{reference + DEFAULT_HIT_TOL:.6f}"
        )
        append_csv(args.rows, row)
        append_jsonl(args.sessions, session)
        print(
            f"{order}/{len(jobs)} {system} {arm} seed {seed} hit {row['hit']} "
            f"calls {row['calls_to_hit']} cpu {row['cpu_s']} outside {row['outside_cpu_to_hit_s']} "
            f"best {row['best']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
