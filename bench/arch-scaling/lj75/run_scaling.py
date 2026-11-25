#!/usr/bin/env python3
"""Run LJ75 worker-scaling rungs and append rows to rows.csv."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "paper_0120"))
import run_panel  # noqa: E402

def load_budgets():
    return run_panel.PAPER_BUDGETS


BUDGET = load_budgets()["lj75"]
if BUDGET != 4_000_000:
    raise SystemExit(f"lj75 budget {BUDGET} is not 4000000")
SEEDS = list(range(2000, 2012))
DEFAULT_RUNGS = [8, 16, 4, 2, 1]

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "workers",
    "total_oracle_calls",
    "call_budget",
    "hit",
    "calls_to_hit",
    "wall_to_hit_s",
    "best_energy",
    "wall_s",
    "cpu_s",
    "peak_rss_anon_kib",
    "replica_charged",
    "hops",
    "basins",
    "screened_out",
    "returned",
    "history_obs",
    "history_new",
    "ledger_sum",
    "evals_observed",
    "ckpt_calls_to_hit",
    "revision",
    "nproc",
    "cpu_model",
    "pinned_cores",
    "oversubscribed",
    "session",
]


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def target_binary() -> Path:
    meta = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=REPO,
            text=True,
        )
    )
    return Path(meta["target_directory"]) / "release" / "examples" / "arch_lj75_bench"


def load_done(path: Path) -> set[tuple[str, str, int]]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {
            ((r.get("session") or ""), r["arm"], int(r["seed"]))
            for r in csv.DictReader(handle)
        }


def ensure_header(path: Path) -> None:
    """Add trailing columns. Do not rewrite existing data lines."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    raw = path.read_bytes()
    nl = raw.find(b"\n")
    if nl < 0:
        raise RuntimeError(f"{path} has no header newline")
    header = raw[:nl].decode("ascii")
    if header.endswith("\r"):
        raise RuntimeError(f"{path} header is CRLF")
    want = ",".join(ROW_FIELDS)
    if header == want:
        return
    old = header.split(",")
    if ROW_FIELDS[: len(old)] != old:
        raise RuntimeError(f"unexpected header in {path}: {header}")
    path.write_bytes(want.encode("ascii") + raw[nl:])


def append_row(path: Path, row: dict[str, object]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_job(
    binary: Path,
    workers: int,
    seed: int,
    revision: str,
    nproc: int,
    model: str,
    rows_path: Path,
    session: str,
) -> None:
    oversubscribed = workers > nproc
    max_core = min(workers, nproc) - 1
    pinned = "0" if max_core == 0 else f"0-{max_core}"
    cmd = [
        "taskset",
        "-c",
        pinned,
        str(binary),
        "scaling",
        str(workers),
        str(seed),
        str(BUDGET),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"workers={workers} seed={seed} failed ({proc.returncode}): {proc.stderr}"
        )
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            payload = json.loads(line)
    if payload is None:
        raise RuntimeError(f"missing JSON output for workers={workers} seed={seed}")
    if float(payload["wall_s"]) < 0.050:
        raise RuntimeError(
            f"sample wall_s={payload['wall_s']} < 0.050 s for workers={workers} seed={seed}"
        )
    if int(payload["ledger_sum"]) != int(payload["total_oracle_calls"]):
        raise RuntimeError(
            f"ledger mismatch for workers={workers} seed={seed}: {payload}"
        )
    row = {
        "system": payload["system"],
        "arm": payload["arm"],
        "seed": payload["seed"],
        "workers": payload["workers"],
        "total_oracle_calls": payload["total_oracle_calls"],
        "call_budget": payload["call_budget"],
        "hit": payload["hit"],
        "calls_to_hit": payload["calls_to_hit"],
        "wall_to_hit_s": (
            "" if payload["wall_to_hit_s"] is None else f"{payload['wall_to_hit_s']:.6f}"
        ),
        "best_energy": f"{payload['best_energy']:.10f}",
        "wall_s": f"{payload['wall_s']:.6f}",
        "cpu_s": f"{payload['cpu_s']:.2f}",
        "peak_rss_anon_kib": payload["peak_rss_anon_kib"],
        "replica_charged": payload["replica_charged"],
        "hops": payload["hops"],
        "basins": payload["basins"],
        "screened_out": payload["screened_out"],
        "returned": payload["returned"],
        "history_obs": payload["history_obs"],
        "history_new": payload["history_new"],
        "ledger_sum": payload["ledger_sum"],
        "evals_observed": payload["evals_observed"],
        "ckpt_calls_to_hit": (
            "" if payload["ckpt_calls_to_hit"] is None else payload["ckpt_calls_to_hit"]
        ),
        "revision": revision,
        "nproc": nproc,
        "cpu_model": model,
        "pinned_cores": pinned,
        "oversubscribed": "true" if oversubscribed else "false",
        "session": session,
    }
    append_row(rows_path, row)
    print(
        f"session={session or '-'} comm-p{workers} seed={seed} hit={row['hit']} "
        f"calls={row['calls_to_hit']} wall={row['wall_s']}s cpu={row['cpu_s']}s "
        f"rss={row['peak_rss_anon_kib']}KiB pin={pinned} best={row['best_energy']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rungs",
        default=",".join(str(r) for r in DEFAULT_RUNGS),
        help="Comma-separated worker rungs in execution order",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Rotate rung order per seed across all listed rungs",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path(__file__).resolve().parent / "rows.csv",
    )
    parser.add_argument(
        "--session",
        default="",
        help="Session tag stored on rows that are not the control rung",
    )
    parser.add_argument(
        "--control-workers",
        type=int,
        default=0,
        help="Worker count whose rows receive --control-session",
    )
    parser.add_argument(
        "--control-session",
        default="",
        help="Session tag for the same-VM control rung",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the serial order and do not launch the driver",
    )
    args = parser.parse_args()
    if args.control_session and args.control_workers <= 0:
        raise SystemExit("--control-session requires --control-workers")

    rungs = [int(x) for x in args.rungs.split(",") if x]
    revision = git_head()
    nproc = os.cpu_count() or 1
    model = cpu_model()
    done = load_done(args.rows)

    if args.rotate:
        jobs = []
        for idx, seed in enumerate(SEEDS):
            rot = idx % len(rungs)
            for workers in rungs[rot:] + rungs[:rot]:
                jobs.append((workers, seed))
    else:
        jobs = [(workers, seed) for workers in rungs for seed in SEEDS]

    planned: list[tuple[int, int, str, str]] = []
    for workers, seed in jobs:
        if args.control_session and workers == args.control_workers:
            session = args.control_session
        else:
            session = args.session
        arm = f"comm-p{workers}"
        planned.append((workers, seed, session, arm))

    print(
        f"revision={revision} nproc={nproc} cpu_model={model} budget={BUDGET} "
        f"jobs={len(planned)}",
        flush=True,
    )
    if args.dry_run:
        for workers, seed, session, arm in planned:
            oversubscribed = workers > nproc
            max_core = min(workers, nproc) - 1
            pinned = "0" if max_core == 0 else f"0-{max_core}"
            skip = (session, arm, seed) in done
            print(
                f"plan session={session or '-'} arm={arm} seed={seed} "
                f"pin={pinned} oversubscribed={str(oversubscribed).lower()} "
                f"skip={str(skip).lower()}",
                flush=True,
            )
        return

    binary = target_binary()
    if not binary.is_file():
        raise SystemExit(f"missing binary: {binary}")
    ensure_header(args.rows)
    for workers, seed, session, arm in planned:
        if (session, arm, seed) in done:
            continue
        run_job(
            binary, workers, seed, revision, nproc, model, args.rows, session
        )
        done.add((session, arm, seed))


if __name__ == "__main__":
    main()
