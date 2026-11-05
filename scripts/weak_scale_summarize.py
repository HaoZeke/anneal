#!/usr/bin/env python3
"""Aggregate weak-scaling logs into a hardware-named table.

The IQR is the type 7 interquartile range. Weak efficiency is the
one-worker median wall time divided by the cell median.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "weak_scale" / "raw"
OUT_CSV = ROOT / "results" / "weak_scale" / "summary.csv"
OUT_MD = ROOT / "results" / "weak_scale" / "table.md"
FAMILIES = ("pt", "population", "dmc")
WORKERS = (1, 2, 4, 8, 16)
SEEDS = tuple(range(6))


def quantile_type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("empty sample")
    if count == 1:
        return ordered[0]
    position = (count - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] + weight * (ordered[high] - ordered[low])


def parse_log(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ": " in line and not line.startswith("family="):
            key, value = line.split(": ", 1)
            fields[key] = value
        if line.startswith("family="):
            for item in line.split():
                key, value = item.split("=", 1)
                fields[key] = value
    return fields


def load() -> list[dict[str, str]]:
    rows = []
    for family in FAMILIES:
        for workers in WORKERS:
            for seed in SEEDS:
                path = RAW / family / f"w{workers}" / f"seed{seed}.log"
                if not path.is_file():
                    raise SystemExit(f"missing {path}")
                fields = parse_log(path)
                fields["path"] = str(path.relative_to(ROOT))
                rows.append(fields)
    return rows


def schedule_line(summary: list[dict], family: str, label: str) -> str:
    row = next(item for item in summary if item["family"] == family and item["workers"] == 1)
    return (
        f"{label} keeps {row['replicas']} replicas and "
        f"{row['charged_per_worker']:g} charged evaluations on each worker."
    )


def main() -> int:
    rows = load()
    heads = {row["HEAD"] for row in rows}
    trees = {row["tree"] for row in rows}
    if len(heads) != 1 or len(trees) != 1:
        raise SystemExit(f"mixed revisions heads={heads} trees={trees}")
    exits = {row["exit"] for row in rows}
    if exits != {"0"}:
        raise SystemExit(f"nonzero exits {exits}")
    cpu = rows[0]["cpu_model"]
    cores = rows[0]["cores"]
    if any(row["cpu_model"] != cpu or row["gpu"] != "none" for row in rows):
        raise SystemExit("hardware lines differ")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], int(row["workers"]))].append(row)
    summary = []
    for family in FAMILIES:
        for workers in WORKERS:
            sample = grouped[(family, workers)]
            if len(sample) < 6:
                raise SystemExit(f"{family} w{workers} has {len(sample)} runs")
            walls = [float(row["wall_s"]) for row in sample]
            numpy_q1 = float(np.quantile(walls, 0.25, method="linear"))
            numpy_q3 = float(np.quantile(walls, 0.75, method="linear"))
            q1 = quantile_type7(walls, 0.25)
            q3 = quantile_type7(walls, 0.75)
            if abs(q1 - numpy_q1) > 1e-12 or abs(q3 - numpy_q3) > 1e-12:
                raise SystemExit("type 7 quantile disagrees with numpy")
            median = quantile_type7(walls, 0.5)
            charged = [int(row["charged"]) for row in sample]
            replicas = [int(row["replicas"]) for row in sample]
            atoms = [int(row["atoms"]) for row in sample]
            pools = [int(row["pool_threads"]) for row in sample]
            if any(pool != workers for pool in pools):
                raise SystemExit(f"pool width {pools} at {workers} workers")
            if len(set(charged)) != 1 or len(set(replicas)) != 1 or len(set(atoms)) != 1:
                raise SystemExit(f"schedule drifted for {family} w{workers}")
            per_worker = charged[0] / workers
            record = {
                "family": family,
                "workers": workers,
                "atoms": atoms[0],
                "replicas": replicas[0],
                "charged": charged[0],
                "charged_per_worker": per_worker,
                "median_wall_s": median,
                "iqr_s": q3 - q1,
                "n": len(sample),
                "cpu_model": cpu,
                "cores": int(cores),
                "gpu": "none",
                "cuda": "none",
            }
            summary.append(record)
        family_rows = [row for row in summary if row["family"] == family]
        unit = family_rows[0]["charged_per_worker"]
        if any(row["charged_per_worker"] != unit for row in family_rows):
            raise SystemExit(f"{family} charge per worker is not constant")
        base_wall = next(row["median_wall_s"] for row in family_rows if row["workers"] == 1)
        for row in family_rows:
            row["weak_efficiency"] = base_wall / row["median_wall_s"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "workers",
        "atoms",
        "replicas",
        "charged",
        "charged_per_worker",
        "median_wall_s",
        "iqr_s",
        "weak_efficiency",
        "n",
        "cpu_model",
        "cores",
        "gpu",
        "cuda",
    ]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Weak scaling",
        "",
        f"Host CPU: {cpu}, {cores} cores.",
        "GPU: none. CUDA: none.",
        "Workers are threads in one process on this host.",
        "Snakemake exports OMP_NUM_THREADS as 1.",
        "The Rayon pools and the diffusion threads use the worker count.",
        "Worker counts are 1, 2, 4, 8, and 16.",
        f"Sixteen threads share {cores} cores.",
        "Each cell is six seeds, numbered 0 through 5.",
        "The wall time is the median.",
        "The IQR is the type 7 interquartile range of those six wall times.",
        "Weak efficiency is the one-worker median divided by the cell median.",
        schedule_line(summary, "pt", "Parallel tempering"),
        schedule_line(summary, "population", "Population annealing"),
        schedule_line(summary, "dmc", "The diffusion population"),
        "Atoms per cluster stay fixed, so the pair count of one evaluation stays fixed.",
        "A multi-node path needs an MPI launcher or the bank RPC sockets,",
        "one chain group or one population per rank, and an exchange of energies",
        "at each swap or resampling step.",
        "This host has no second node and no MPI library, so those rows are absent.",
        "",
        "| family | workers | atoms | replicas | charged | median wall s | type 7 IQR s | weak efficiency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {family} | {workers} | {atoms} | {replicas} | {charged} | {median_wall_s:.6f} | {iqr_s:.6f} | {weak_efficiency:.6f} |".format(
                **row
            )
        )
    lines.append("")
    lines.append(f"HEAD: {next(iter(heads))}")
    lines.append(f"tree: {next(iter(trees))}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
