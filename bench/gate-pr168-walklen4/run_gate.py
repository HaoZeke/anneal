#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial LJ98 walklen4 gate driver.

One `h2h_seed` at a time, pinned to one core. Even seeds run A then B.
Odd seeds run B then A. Process CPU and peak RSS come from `wait4`.
Outside-potential CPU time is that CPU minus the harness `pot_s`.
The parent process stays off the pinned core.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from machine_provenance import machine_provenance  # noqa: E402
from target_tol import energy_hits  # noqa: E402

from protocol import (  # noqa: E402
    BUDGET,
    CORE,
    MISS_CALLS,
    REFERENCE,
    SEEDS,
    STOP,
    revision_of,
    schedule,
)

FIELDS = [
    "arm",
    "seed",
    "hit",
    "calls_to_hit",
    "outside_cpu_s",
    "cpu_s",
    "wall_s",
    "peak_rss_kb",
    "code_revision",
    "code_tree",
    "host_id_hash",
    "core",
    "core_observed",
    "budget",
    "pot_s",
    "harness_outside_s",
    "harness_wall_s",
    "best",
    "charged",
    "potential_calls",
    "best_agrees",
    "calls_agrees",
    "exit_code",
    "seq",
    "patch_tip",
    "binary_sha256",
    "governor",
]

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def governor(core: int) -> str:
    path = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor")
    if not path.is_file():
        return ""
    return path.read_text().strip()


def parse_harness(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    rows = []
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            rows.append(line)
            continue
        if line.startswith("arm "):
            continue
        if ": " in line:
            key, value = line.split(": ", 1)
            found[key] = value
    if len(rows) != 1:
        raise RuntimeError(f"expected one h2h_row, found {len(rows)}")
    for token in rows[0].split()[1:]:
        key, value = token.split("=", 1)
        found[key] = value
    return found


def agrees(hit: int, best: str) -> int:
    energy = float(best)
    printed = energy_hits(energy, REFERENCE, 1.0e-3)
    near = abs(energy - (REFERENCE + 1.0e-3)) <= 5e-10
    if hit == 1:
        return int(printed or near)
    return int((not printed) or near)


def wait_measured(proc: subprocess.Popen, core: int) -> tuple[float, object, list[int]]:
    cores: list[int] = []

    def grab() -> None:
        try:
            stat = Path(f"/proc/{proc.pid}/stat").read_text()
        except OSError:
            return
        cores.append(int(stat.rsplit(" ", 1)[-1]))

    grab()
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(0.1):
            grab()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        _wpid, status, usage = os.wait4(proc.pid, 0)
    finally:
        stop.set()
        thread.join()
    proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - started
    if sorted(set(cores)) != [core]:
        raise RuntimeError(f"core samples {sorted(set(cores))}")
    return wall, usage, cores


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if new_file:
        writer.writerow(FIELDS)
    writer.writerow([row[name] for name in FIELDS])
    with path.open("a", newline="") as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())


def append_session(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({name: row[name] for name in FIELDS}, sort_keys=True)
    with path.open("a") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def finished(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {(row["arm"], int(row["seed"])) for row in csv.DictReader(handle)}


def run_job(
    job: dict,
    binary: Path,
    scratch: Path,
    budget: int,
    stop: str,
    core: int,
    host_id: str,
    binary_id: str,
    gov: str,
) -> dict[str, str]:
    arm = job["arm"]
    seed = job["seed"]
    prefix = scratch / f"s{seed}-{arm}"
    stdout_path = scratch / f"s{seed}-{arm}.out"
    stderr_path = scratch / f"s{seed}-{arm}.err"
    cmd = [
        str(binary),
        "lj",
        "98",
        "0",
        str(budget),
        str(seed),
        "recommended",
        stop,
        str(prefix),
    ]
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            env=child_env(),
            preexec_fn=lambda: os.sched_setaffinity(0, {core}),
        )
        try:
            wall, usage, _cores = wait_measured(proc, core)
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            raise
    text = stdout_path.read_text()
    if proc.returncode != 0:
        tail = stderr_path.read_text()[-1500:]
        raise RuntimeError(f"exit {proc.returncode}\n{tail}")
    parsed = parse_harness(text)
    if parsed["system"] != "lj98" or int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
        raise RuntimeError(f"harness identity {parsed.get('system')} {parsed.get('seed')} {parsed.get('budget')}")
    if parsed["arm"] != "recommended":
        raise RuntimeError(f"harness arm {parsed['arm']}")
    hit = int(parsed["hit"])
    potential = int(parsed["calls"])
    harness_evals = int(parsed["evals_to_hit"])
    if hit == 1:
        calls = harness_evals
        calls_ok = calls == min(potential, budget) and 1 <= calls <= budget
    else:
        calls = MISS_CALLS if budget == BUDGET else budget + 1
        calls_ok = harness_evals == budget
    cpu = float(usage.ru_utime) + float(usage.ru_stime)
    pot = float(parsed["pot_s"])
    revision, tree, tip = revision_of(arm)
    row = {
        "arm": arm,
        "seed": str(seed),
        "hit": str(hit),
        "calls_to_hit": str(calls),
        "outside_cpu_s": f"{(cpu - pot):.6f}",
        "cpu_s": f"{cpu:.6f}",
        "wall_s": f"{wall:.6f}",
        "peak_rss_kb": str(int(usage.ru_maxrss)),
        "code_revision": revision,
        "code_tree": tree,
        "host_id_hash": host_id,
        "core": str(core),
        "core_observed": str(core),
        "budget": str(budget),
        "pot_s": f"{pot:.6f}",
        "harness_outside_s": f"{float(parsed['outside_s']):.6f}",
        "harness_wall_s": f"{float(parsed['wall_s']):.6f}",
        "best": parsed["best"],
        "charged": str(int(parsed["charged"])),
        "potential_calls": str(potential),
        "best_agrees": str(agrees(hit, parsed["best"])),
        "calls_agrees": "1" if calls_ok else "0",
        "exit_code": "0",
        "seq": str(job["seq"]),
        "patch_tip": tip,
        "binary_sha256": binary_id,
        "governor": gov,
    }
    return row


def paper_mode(seeds: tuple[int, ...], budget: int, stop: str, core: int) -> bool:
    return seeds == SEEDS and budget == BUDGET and stop == STOP and core == CORE


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--a-bin", type=Path, required=True)
    parser.add_argument("--b-bin", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--stop", default=STOP)
    parser.add_argument("--core", type=int, default=CORE)
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",")) if args.seeds else SEEDS
    rows_path = args.rows or (HERE / "rows.csv")
    sessions_path = args.sessions or (HERE / "sessions.jsonl")
    if not paper_mode(seeds, args.budget, args.stop, args.core):
        default_rows = HERE / "rows.csv"
        if rows_path.resolve() == default_rows.resolve():
            raise SystemExit("a non-paper run must set --rows")
    jobs = schedule(seeds)
    if seeds == SEEDS:
        arms = [job["arm"] for job in jobs[:8]]
        if arms != ["A", "B", "B", "A", "A", "B", "B", "A"]:
            raise SystemExit("ABBA schedule is wrong")
    parent = 1 if args.core == 0 else 0
    os.sched_setaffinity(0, {parent})
    host_id, _cpu_model = machine_provenance()
    bins = {
        "A": args.a_bin.resolve(),
        "B": args.b_bin.resolve(),
    }
    for path in bins.values():
        if not path.is_file():
            raise SystemExit(f"missing binary {path.name}")
    ids = {arm: sha256_file(path) for arm, path in bins.items()}
    if ids["A"] == ids["B"]:
        raise SystemExit("arm binaries are identical")
    gov = governor(args.core)
    done = finished(rows_path)
    args.scratch.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        key = (job["arm"], job["seed"])
        if key in done:
            continue
        row = run_job(
            job,
            bins[job["arm"]],
            args.scratch,
            args.budget,
            args.stop,
            args.core,
            host_id,
            ids[job["arm"]],
            gov,
        )
        append_row(rows_path, row)
        append_session(sessions_path, row)
        print(
            f"seq={row['seq']} arm={row['arm']} seed={row['seed']} hit={row['hit']} "
            f"calls={row['calls_to_hit']} outside={row['outside_cpu_s']} "
            f"cpu={row['cpu_s']} wall={row['wall_s']} rss_kb={row['peak_rss_kb']}",
            flush=True,
        )
        if paper_mode(seeds, args.budget, args.stop, args.core) and (
            row["best_agrees"] != "1" or row["calls_agrees"] != "1"
        ):
            raise SystemExit(
                f"score disagrees arm={row['arm']} seed={row['seed']} "
                f"best={row['best']} hit={row['hit']} calls={row['calls_to_hit']} "
                f"potential={row['potential_calls']}"
            )
    if paper_mode(seeds, args.budget, args.stop, args.core) and len(finished(rows_path)) == len(jobs):
        import analyze

        result = analyze.write_report(rows_path, HERE / "REPORT.md")
        print(result["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
