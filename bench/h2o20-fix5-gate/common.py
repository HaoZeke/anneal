# Copyright (c) anneal developers
"""Rows, identities and statistics shared by the h2o20-fix5 gate."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import struct
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

ROW_FIELDS = [
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
    "machine_id",
    "library_tree",
    "harness_blob",
    "driver_blob",
    "status",
]

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def machine_id() -> str:
    """sha256 of /etc/machine-id without its trailing newline."""
    raw = Path("/etc/machine-id").read_text().strip()
    return hashlib.sha256(raw.encode()).hexdigest()


def git(tree: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(tree), *args], text=True).strip()


def tree_of(tree: Path) -> str:
    return git(tree, "rev-parse", "HEAD^{tree}")


def clean(tree: Path) -> bool:
    return git(tree, "status", "--porcelain", "--untracked-files=all") == ""


def blob_of(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()


def run_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Process environment with one thread everywhere and no H2H_* override."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("H2H_")}
    env.pop("LD_LIBRARY_PATH", None)
    for key in THREAD_KEYS:
        env[key] = "1"
    if extra:
        env.update(extra)
    return env


def bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str], fields: list[str] = ROW_FIELDS) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="") as handle:
            if next(csv.reader(handle)) != fields:
                raise SystemExit(f"{path} has another header; rows are appended, never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def quantile7(values: list[float], prob: float) -> float:
    """Hyndman-Fan type 7 quantile."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    h = (len(ordered) - 1) * prob
    low = math.floor(h)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def median7(values: list[float]) -> float:
    return quantile7(values, 0.5)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Exact two-sided Fisher test on a 2x2 table, point-probability rule."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U with tie and continuity correction.

    Returns U for ``left`` and the normal-approximation p value.
    """
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        width = j - i + 1
        ties += width**3 - width
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = abs(u - n1 * n2 / 2.0) - 0.5
    return u, min(1.0, math.erfc(max(dev, 0.0) / math.sqrt(var) / math.sqrt(2.0)))


def ranked_calls(rows: list[dict[str, str]]) -> list[float]:
    """Calls to the hit; a miss ranks worse than every hit."""
    worst = max(float(row["budget"]) for row in rows) + 1.0
    return [float(row["calls"]) if row["hit"] == "1" else worst for row in rows]


def ratio(top: float, bottom: float) -> float:
    if bottom == 0.0 or not math.isfinite(top) or not math.isfinite(bottom):
        return math.nan
    return top / bottom


def fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"
