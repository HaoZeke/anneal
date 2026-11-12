#!/usr/bin/env python3
"""Serial depth-posterior ablation on the paper Lennard-Jones sizes.

Both arms share the seed list and PAPER_BUDGETS. The schedule is one
shuffled list, so the arms are interleaved rather than run as two blocks.
One process at a time. A hit is within 0.001 of the Cambridge minimum;
force calls to first hit are summarised with the type-7 median and IQR
on the hits only.
"""

from __future__ import annotations

import argparse
import math
import random
import socket
import subprocess
import sys
from pathlib import Path

SIZES = (38, 55, 75, 98, 104)
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("posterior", "uniform")
PAPER_BUDGETS = {
    38: 400_000,
    55: 1_000_000,
    75: 4_000_000,
    98: 4_000_000,
    104: 4_000_000,
}
SCHEDULE_SEED = 20261111
HEADER = "order,system,arm,seed,budget,hit,force_calls_to_hit,charged,best,wall_s"
SYSTEM = {38: "lj38", 55: "lj55", 75: "lj75", 98: "lj98", 104: "lj104"}


def schedule() -> list[tuple[int, int, str]]:
    jobs = [(size, seed, arm) for size in SIZES for seed in SEEDS for arm in ARMS]
    random.Random(SCHEDULE_SEED).shuffle(jobs)
    return jobs


def type7(sorted_values: list[float], p: float) -> float:
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    height = 1.0 + (count - 1) * p
    lower = math.floor(height)
    frac = height - lower
    index = int(lower) - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return type7(ordered, 0.5), type7(ordered, 0.75) - type7(ordered, 0.25)


def parse_row(line: str, size: int, seed: int, arm: str) -> dict[str, str]:
    fields = line.strip().split(",")
    if len(fields) != 9:
        raise SystemExit(f"expected 9 fields, got {len(fields)}: {line!r}")
    system, got_arm, got_seed, budget, hit, calls, charged, best, wall = fields
    if system != SYSTEM[size] or got_arm != arm or int(got_seed) != seed:
        raise SystemExit(f"row does not match the job: {line!r}")
    if int(budget) != PAPER_BUDGETS[size]:
        raise SystemExit(f"budget {budget} is not the paper budget for LJ{size}")
    if hit not in {"0", "1"}:
        raise SystemExit(f"hit flag {hit}")
    if hit == "0" and calls != "":
        raise SystemExit(f"a miss recorded force calls: {line!r}")
    if hit == "1" and calls == "":
        raise SystemExit(f"a hit omitted force calls: {line!r}")
    return {
        "system": system,
        "arm": got_arm,
        "seed": got_seed,
        "budget": budget,
        "hit": hit,
        "force_calls_to_hit": calls,
        "charged": charged,
        "best": best,
        "wall_s": wall,
    }


def run_job(binary: Path, log_dir: Path, order: int, size: int, seed: int, arm: str) -> dict[str, str]:
    log_path = log_dir / f"{order:02d}-lj{size}-seed{seed}-{arm}.log"
    proc = subprocess.run(
        [str(binary), SYSTEM[size], str(seed), arm],
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        f"hostname={socket.gethostname()}\norder={order}\nsize={size}\nseed={seed}\narm={arm}\n"
        f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SystemExit(f"LJ{size} seed {seed} {arm} exited {proc.returncode}; see {log_path}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SystemExit(f"LJ{size} seed {seed} {arm} did not print one row: {proc.stdout!r}")
    return parse_row(lines[0], size, seed, arm)


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    body = [HEADER]
    for index, row in enumerate(rows, start=1):
        body.append(
            ",".join(
                [
                    str(index),
                    row["system"],
                    row["arm"],
                    row["seed"],
                    row["budget"],
                    row["hit"],
                    row["force_calls_to_hit"],
                    row["charged"],
                    row["best"],
                    row["wall_s"],
                ]
            )
        )
    text = "\n".join(body) + "\n"
    path.write_text(text, encoding="utf-8")


def report(rows: list[dict[str, str]]) -> None:
    print("system arm hits median_calls iqr_calls", flush=True)
    for size in SIZES:
        for arm in ARMS:
            group = [row for row in rows if row["system"] == SYSTEM[size] and row["arm"] == arm]
            hits = [row for row in group if row["hit"] == "1"]
            if hits:
                median, iqr = median_iqr([float(row["force_calls_to_hit"]) for row in hits])
                calls = f"{median:.1f} {iqr:.1f}"
            else:
                calls = "na na"
            print(f"LJ{size} {arm} {len(hits)}/{len(group)} {calls}", flush=True)


def main() -> None:
    if socket.gethostname() != "bench":
        raise SystemExit(f"hostname is {socket.gethostname()}, expected bench")
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.binary.is_file() is False:
        raise SystemExit(f"missing binary {args.binary}")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    single = (args.size, args.seed, args.arm)
    if any(value is not None for value in single) and any(value is None for value in single):
        raise SystemExit("pass size, seed, and arm together")
    if args.size is not None:
        if args.size not in PAPER_BUDGETS or args.seed not in SEEDS:
            raise SystemExit("size or seed is outside the paper list")
        row = run_job(args.binary, args.log_dir, 1, args.size, args.seed, args.arm)
        print(
            f"LJ{args.size} {args.arm} seed {args.seed} hit={row['hit']} "
            f"calls={row['force_calls_to_hit'] or '-'} wall={row['wall_s']}",
            flush=True,
        )
        return
    jobs = schedule()
    rows: list[dict[str, str]] = []
    progress = args.log_dir / "progress.txt"
    for order, (size, seed, arm) in enumerate(jobs, start=1):
        print(f"start {order}/{len(jobs)} LJ{size} seed {seed} {arm}", flush=True)
        row = run_job(args.binary, args.log_dir, order, size, seed, arm)
        rows.append(row)
        write_table(args.log_dir / "rows.csv", rows)
        progress.write_text(f"{order}/{len(jobs)} LJ{size} seed {seed} {arm} hit={row['hit']}\n", encoding="utf-8")
        print(
            f"done {order}/{len(jobs)} LJ{size} seed {seed} {arm} hit={row['hit']} "
            f"calls={row['force_calls_to_hit'] or '-'} wall={row['wall_s']}",
            flush=True,
        )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    write_table(args.results, rows)
    report(rows)
    print(f"wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
