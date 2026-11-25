#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run chain-ensemble arms and append rows.

One arm at a time, pinned to CPU 0. Even seeds run comm, indep, single.
Odd seeds run that order reversed. Existing (system, arm, seed) rows are
left in place.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
CURVE = HERE / "success_vs_calls.csv"
REPORT = HERE / "REPORT.md"


def release_binary() -> Path:
    cargo = os.environ.get("CARGO", "cargo")
    out = subprocess.run(
        [cargo, "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    target = Path(json.loads(out.stdout)["target_directory"])
    return target / "release" / "examples" / "arch_chains"
SEEDS = tuple(range(2000, 2012))
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261123
FIELDS = (
    "system",
    "arm",
    "seed",
    "chains",
    "calls",
    "budget",
    "hit",
    "calls_to_hit",
    "best_energy",
    "wall_s",
    "cpu_s",
    "rss_anon_kb",
    "ledger_aggregate",
    "ledger_replica_sum",
    "ledger_history_obs",
    "ledger_history_refusals",
    "ledger_history_minima",
    "ledger_history_accepted",
    "ledger_history_visits",
    "ledger_bias_published",
    "ledger_exchange_published",
    "ledger_exchange_delivered",
    "ledger_shared_deposits",
    "ledger_gossip_rounds",
    "ledger_two_choice_restarts",
    "ledger_sum",
    "history_stores",
    "revision",
    "nproc",
    "cpu_model",
)
ARMS = ("comm", "indep", "single")
LEDGER_SUM = "ledger_sum"


def revision() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def arm_order(seed: int) -> tuple[str, ...]:
    if (seed - 2000) % 2 == 0:
        return ARMS
    return tuple(reversed(ARMS))


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow({key: row[key] for key in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def run_one(binary: Path, system: str, arm: str, seed: int, rev: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", system, arm, str(seed), rev],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"{system} {arm} {seed} exited {proc.returncode}\n{proc.stderr}"
        )
    payload = json.loads(proc.stdout)
    row = {key: str(payload[key]) for key in FIELDS}
    if int(row["hit"]) not in (0, 1):
        raise SystemExit(f"bad hit flag {row}")
    if float(row["wall_s"]) < 0.05:
        raise SystemExit(f"{system} {arm} {seed} wall {row['wall_s']} is under 50 ms")
    return row


def campaign(binary: Path, systems: list[str]) -> None:
    rev = revision()
    done = {(row["system"], row["arm"], row["seed"]) for row in load_rows(ROWS)}
    for system in systems:
        for seed in SEEDS:
            for arm in arm_order(seed):
                key = (system, arm, str(seed))
                if key in done:
                    print(f"keep {system} {arm} {seed}", flush=True)
                    continue
                row = run_one(binary, system, arm, seed, rev)
                append_row(ROWS, row)
                done.add(key)
                print(
                    f"row {system} {arm} {seed} hit={row['hit']} "
                    f"calls={row['calls']} wall={row['wall_s']}",
                    flush=True,
                )


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    value = 1
    for i in range(k):
        value = value * (n - i) // (i + 1)
    return value


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a = hits_a
    b = n_a - hits_a
    c = hits_b
    d = n_b - hits_b
    row1 = a + b
    col1 = a + c
    total = row1 + c + d
    lo = max(0, row1 - (total - col1))
    hi = min(row1, col1)

    def hyper(x: int) -> float:
        return (
            comb(col1, x)
            * comb(total - col1, row1 - x)
            / comb(total, row1)
        )

    observed = hyper(a)
    return sum(hyper(x) for x in range(lo, hi + 1) if hyper(x) <= observed * (1 + 1e-12))


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def mann_whitney(left: list[float], right: list[float]) -> float:
    """Two-sided Mann-Whitney via the normal approximation with tie correction."""
    n1 = len(left)
    n2 = len(right)
    pooled = left + right
    ranks = rank_values(pooled)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mean = n1 * n2 / 2.0
    n = n1 + n2
    tie_term = 0.0
    i = 0
    ordered = sorted(pooled)
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        t = j - i + 1
        tie_term += t * t * t - t
        i = j + 1
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0.0:
        return 1.0
    z = (abs(u1 - mean) - 0.5) / math.sqrt(var)
    # Two-sided survival of the standard normal.
    return math.erfc(z / math.sqrt(2.0))


def bootstrap_median(values: list[float], rng) -> tuple[float, float, float]:
    import numpy as np

    data = np.asarray(values, dtype=float)
    point = float(np.median(data))
    draws = rng.integers(0, len(data), size=(BOOTSTRAP, len(data)))
    medians = np.median(data[draws], axis=1)
    lo, hi = np.quantile(medians, [0.025, 0.975])
    return point, float(lo), float(hi)


def bootstrap_rate(hits: list[int], rng) -> tuple[float, float, float]:
    import numpy as np

    data = np.asarray(hits, dtype=float)
    point = float(data.mean())
    draws = rng.integers(0, len(data), size=(BOOTSTRAP, len(data)))
    rates = data[draws].mean(axis=1)
    lo, hi = np.quantile(rates, [0.025, 0.975])
    return point, float(lo), float(hi)


def grouped(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault((row["system"], row["arm"]), []).append(row)
    for group in out.values():
        group.sort(key=lambda row: int(row["seed"]))
    return out


def write_curve(rows: list[dict[str, str]]) -> None:
    groups = grouped(rows)
    lines = ["system,arm,calls,success_fraction"]
    for system in sorted({row["system"] for row in rows}):
        for arm in ARMS:
            group = groups.get((system, arm), [])
            if len(group) != len(SEEDS):
                continue
            budget = int(group[0]["budget"])
            hits = sorted(
                int(row["calls_to_hit"]) for row in group if int(row["hit"]) == 1
            )
            n = len(group)
            points = [(0, 0.0)]
            seen = 0
            i = 0
            while i < len(hits):
                call = hits[i]
                while i < len(hits) and hits[i] == call:
                    seen += 1
                    i += 1
                points.append((call, seen / n))
            if points[-1][0] != budget:
                points.append((budget, seen / n))
            for call, fraction in points:
                lines.append(f"{system},{arm},{call},{fraction:.6f}")
    CURVE.write_text("\n".join(lines) + "\n")


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_report(rows: list[dict[str, str]]) -> None:
    import numpy as np

    groups = grouped(rows)
    systems = []
    for system in ("tip4p20", "au55"):
        if all(len(groups.get((system, arm), [])) == len(SEEDS) for arm in ARMS):
            systems.append(system)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lines = [
        "# Chain ensemble results",
        "",
        "Seeds 2000-2011. A miss counts as budget + 1 in the calls-to-hit",
        "median and in the Mann-Whitney test. Bootstrap intervals are 95%",
        f"percentiles from {BOOTSTRAP} resamples, seed {BOOTSTRAP_SEED}.",
        "Fisher and Mann-Whitney compare comm against indep, then comm",
        "against single. Both tests are two-sided.",
        "",
        "| system | arm | hits | hit rate | rate CI | median calls | calls CI | ledger sum |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    tests = ["", "## Tests", ""]
    for system in systems:
        for arm in ARMS:
            group = groups[(system, arm)]
            hits = [int(row["hit"]) for row in group]
            calls = [float(row["calls_to_hit"]) for row in group]
            rate, rate_lo, rate_hi = bootstrap_rate(hits, rng)
            med, med_lo, med_hi = bootstrap_median(calls, rng)
            ledger = sum(int(row[LEDGER_SUM]) for row in group)
            lines.append(
                f"| {system} | {arm} | {sum(hits)}/{len(hits)} | {fmt(rate)} | "
                f"[{fmt(rate_lo)}, {fmt(rate_hi)}] | {med:.1f} | "
                f"[{med_lo:.1f}, {med_hi:.1f}] | {ledger} |"
            )
        comm = groups[(system, "comm")]
        for other in ("indep", "single"):
            alt = groups[(system, other)]
            fisher_p = fisher(
                sum(int(row["hit"]) for row in comm),
                len(comm),
                sum(int(row["hit"]) for row in alt),
                len(alt),
            )
            mwu_p = mann_whitney(
                [float(row["calls_to_hit"]) for row in comm],
                [float(row["calls_to_hit"]) for row in alt],
            )
            tests.append(
                f"- {system} comm vs {other}: Fisher p {fisher_p:.4f}, "
                f"Mann-Whitney p {mwu_p:.4f}"
            )
    if not systems:
        lines.append("| none |  |  |  |  |  |  |  |")
        tests.append("- no system has all 12 seeds on every arm")
    tests.extend(
        [
            "",
            "Openconf cell rq3 is not in this table. The ensemble has no",
            "MMFF objective, so that ligand was not run.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines + tests))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--systems", nargs="*", default=["tip4p20", "au55"])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.summarize_only:
        if args.binary is None:
            args.binary = release_binary()
        if not args.binary.is_file():
            raise SystemExit(f"missing binary {args.binary.name}")
        campaign(args.binary, args.systems)
    rows = load_rows(ROWS)
    write_curve(rows)
    write_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
