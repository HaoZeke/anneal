# Copyright (c) anneal developers
"""Rows, identities, and statistics shared by the no-builders14 gate."""

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
    "session",
    "system",
    "family",
    "arm",
    "seed",
    "block",
    "repeat",
    "slot",
    "core",
    "hit",
    "calls_to_hit",
    "charged",
    "wall_s",
    "inside_s",
    "outside_s",
    "best",
    "best_bits",
    "final",
    "final_bits",
    "budget",
    "reference",
    "exit_code",
    "arm_tree",
    "machine",
    "rss_anon_kib",
    "ru_maxrss_kib",
    "vm_hwm_kib",
    "harness_blob",
    "driver_blob",
    "started_utc",
    "ended_utc",
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
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "GOTO_NUM_THREADS",
    "TBB_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "CARGO_BUILD_JOBS",
)

SEEDS = tuple(range(2000, 2024))
ARM_TREES = {
    "A": "cff796c507a8fa1d0bfb28888c3570b61988055e",
    "A2": "cff796c507a8fa1d0bfb28888c3570b61988055e",
    "B": "5d21b6f92713b82b421fb44f2b5edd932619819b",
    "G": "5d21b6f92713b82b421fb44f2b5edd932619819b",
}

# family, system, driver, paper budget, reference
CELLS = (
    ("lj", "lj38", "gate_h2h_gmin_proof", 400_000, -173.928427),
    ("lj", "lj55", "gate_h2h_gmin_proof", 1_000_000, -279.248470),
    ("lj", "lj75", "gate_h2h_gmin_proof", 4_000_000, -397.492331),
    ("lj", "lj98", "gate_h2h_gmin_proof", 4_000_000, -543.665361),
    ("blj", "blj105n38", "gate_h2h_blj", 400_000, -177.2090846748),
    ("blj", "blj115n38", "gate_h2h_blj", 400_000, -187.3676867453),
    ("blj", "blj130n38", "gate_h2h_blj", 400_000, -191.1840192602),
    ("blj", "blj105n55", "gate_h2h_blj", 1_000_000, -283.1545313810),
    ("blj", "blj115n55", "gate_h2h_blj", 1_000_000, -289.2931853058),
    ("blj", "blj130n55", "gate_h2h_blj", 1_000_000, -306.7337643721),
    ("blj", "blj130n98", "gate_h2h_blj", 4_000_000, -590.7874132760),
    ("morse", "m55r6", "gate_h2h_morse", 1_000_000, -250.286609),
    ("morse", "m55r14", "gate_h2h_morse", 1_000_000, -220.646208),
    ("morse", "m80r6", "gate_h2h_morse", 4_000_000, -378.333471),
    ("morse", "m80r14", "gate_h2h_morse", 4_000_000, -340.811371),
    ("sc", "sc126-38", "gate_h2h_sc", 400_000, -35419.9804),
    ("sc", "sc108-38", "gate_h2h_sc", 400_000, -10117.2454),
    ("sc", "sc96-38", "gate_h2h_sc", 400_000, -8917.7056),
    ("sc", "sc126-55", "gate_h2h_sc", 1_000_000, -52884.6806),
    ("sc", "sc108-55", "gate_h2h_sc", 1_000_000, -14814.5225),
    ("sc", "sc96-55", "gate_h2h_sc", 1_000_000, -13217.8963),
    ("sc", "sc126-80", "gate_h2h_sc", 1_000_000, -78414.6271),
    ("sc", "sc108-80", "gate_h2h_sc", 1_000_000, -21819.9952),
    ("sc", "sc96-80", "gate_h2h_sc", 1_000_000, -19560.8966),
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
    """One-thread environment without search overrides."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("H2H_", "ANNEAL_"))
    }
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
    if ordered[low] == ordered[high]:
        return ordered[low]
    if math.isinf(ordered[high]):
        return ordered[high]
    if math.isinf(ordered[low]):
        return ordered[low]
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
    return [
        float(row["calls_to_hit"]) if row["hit"] == "1" else math.inf
        for row in rows
    ]


def ratio(top: float, bottom: float) -> float:
    if bottom == 0.0 or not math.isfinite(top) or not math.isfinite(bottom):
        return math.nan
    return top / bottom


def fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"
