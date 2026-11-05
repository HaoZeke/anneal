#!/usr/bin/env python3
"""Deposit strong scaling tables from the raw cell logs.

The quartile is Hyndman-Fan type 7. Speedup and parallel efficiency use
the median wall time against one worker. Every repeat is kept.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


def quantile_type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("empty sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    if lower >= count:
        return ordered[-1]
    if lower < 1:
        return ordered[0]
    left = ordered[lower - 1]
    if fraction == 0.0:
        return left
    return left + fraction * (ordered[lower] - left)


def check_type7() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if quantile_type7(sample, 0.25) != 2.25:
        raise SystemExit("type 7 lower quartile mismatch")
    if quantile_type7(sample, 0.5) != 3.5:
        raise SystemExit("type 7 median mismatch")
    if quantile_type7(sample, 0.75) != 4.75:
        raise SystemExit("type 7 upper quartile mismatch")


def parse_log(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "stderr:" or line.startswith("stderr:"):
            break
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key] = value
    return fields


def slug(cpu_model: str, logical_cpus: str) -> str:
    letters = []
    for character in cpu_model.lower():
        if character.isalnum():
            letters.append(character)
        else:
            letters.append("-")
    compact = "".join(letters)
    while "--" in compact:
        compact = compact.replace("--", "-")
    return f"{compact.strip('-')}-{logical_cpus}cpu"


def main() -> int:
    check_type7()
    root = Path("experiments/scaling/results/raw")
    logs = sorted(root.glob("*/*/r*.txt"))
    if not logs:
        print("no raw logs", file=sys.stderr)
        return 1
    rows = []
    for path in logs:
        fields = parse_log(path)
        if fields.get("exit") != "0":
            print(f"nonzero exit in {path.name}", file=sys.stderr)
            return 1
        if fields.get("scaling") != "strong":
            print(f"not a strong scaling log: {path.name}", file=sys.stderr)
            return 1
        rows.append(fields)
    required = [
        "cpu_model",
        "logical_cpus",
        "memory_kib",
        "gpu",
        "atoms",
        "walkers",
        "generations",
        "quench_steps",
        "dmc_budget",
    ]
    anchor = rows[0]
    for fields in rows[1:]:
        for key in required:
            if fields.get(key) != anchor.get(key):
                print(f"logs disagree on {key}", file=sys.stderr)
                return 1
    identity: dict[tuple[str, str], dict[str, str]] = {}
    for fields in rows:
        key = (fields["method"], fields["repeat"])
        signature = {
            "best_energy": fields["best_energy"],
            "evaluations": fields["evaluations"],
            "decisions": fields["decisions"],
        }
        previous = identity.setdefault(key, signature)
        if previous != signature:
            print(
                f"{fields['method']} repeat {fields['repeat']} changed with the worker count",
                file=sys.stderr,
            )
            return 1
    grouped: dict[tuple[str, str], list[float]] = {}
    for fields in rows:
        key = (fields["method"], fields["workers"])
        grouped.setdefault(key, []).append(float(fields["wall_s"]))
    for times in grouped.values():
        if len(times) < 6:
            print("fewer than 6 repeats", file=sys.stderr)
            return 1
    methods = sorted({key[0] for key in grouped})
    out_dir = Path("experiments/scaling/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "method,workers,n,median_wall_s,q1_wall_s,q3_wall_s,iqr_wall_s,speedup,parallel_efficiency"
    ]
    long_lines = ["method,workers,repeat,wall_s"]
    for fields in rows:
        long_lines.append(
            f"{fields['method']},{fields['workers']},{fields['repeat']},{fields['wall_s']}"
        )
    table_rows = []
    for method in methods:
        baseline = grouped[(method, "1")]
        base_median = quantile_type7(baseline, 0.5)
        worker_ids = sorted({int(key[1]) for key in grouped if key[0] == method})
        for workers in worker_ids:
            times = grouped[(method, str(workers))]
            median = quantile_type7(times, 0.5)
            q1 = quantile_type7(times, 0.25)
            q3 = quantile_type7(times, 0.75)
            iqr = q3 - q1
            speedup = base_median / median
            efficiency = speedup / workers
            summary_lines.append(
                f"{method},{workers},{len(times)},{median:.9f},{q1:.9f},{q3:.9f},"
                f"{iqr:.9f},{speedup:.6f},{efficiency:.6f}"
            )
            table_rows.append(
                (
                    method,
                    workers,
                    len(times),
                    median,
                    q1,
                    q3,
                    iqr,
                    speedup,
                    efficiency,
                )
            )
    cpu = anchor["cpu_model"]
    logical = anchor["logical_cpus"]
    memory = anchor["memory_kib"]
    gib = int(memory) / (1024 * 1024)
    header = [
        f"# Strong scaling on {cpu}",
        "",
        (
            f"Host: {cpu}, {logical} logical CPUs, {memory} KiB memory "
            f"({gib:.2f} GiB). GPU: {anchor['gpu']}."
        ),
        (
            "Fixed problem: Lennard-Jones, "
            f"{anchor['atoms']} atoms, {anchor['walkers']} walkers, "
            f"{anchor['generations']} generations, "
            f"quench length {anchor['quench_steps']}, "
            f"diffusion Monte Carlo budget {anchor['dmc_budget']}."
        ),
        (
            "The worker count is the thread pool. "
            "Six repeats. The quartile is Hyndman-Fan type 7. "
            "Speedup and parallel efficiency use the median wall time against one worker."
        ),
        "This is a scaling table.",
        "",
        (
            "| method | workers | n | median wall s | Q1 | Q3 | IQR | speedup | efficiency |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    body = []
    for row in table_rows:
        method, workers, count, median, q1, q3, iqr, speedup, efficiency = row
        body.append(
            f"| {method} | {workers} | {count} | {median:.6f} | {q1:.6f} | "
            f"{q3:.6f} | {iqr:.6f} | {speedup:.4f} | {efficiency:.4f} |"
        )
    text = "\n".join(header + body) + "\n"
    stable_csv = out_dir / "pop_strong_scaling.csv"
    stable_md = out_dir / "pop_strong_scaling.md"
    long_csv = out_dir / "pop_strong_scaling_long.csv"
    stable_csv.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    stable_md.write_text(text, encoding="utf-8")
    long_csv.write_text("\n".join(long_lines) + "\n", encoding="utf-8")
    named = slug(cpu, logical)
    (out_dir / f"{named}.csv").write_text(stable_csv.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / f"{named}.md").write_text(text, encoding="utf-8")
    print(f"wrote {stable_csv}")
    print(f"wrote {named}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
