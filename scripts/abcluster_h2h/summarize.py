#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarise the anneal against ABCluster rows.

Reads ``seeds.csv``, ``parity.csv``, ``counter.csv`` and ``pilot.csv`` in
``bench/abcluster-h2h``. Writes ``summary.csv`` and the table between the
markers of ``README.md``. Does not start a search.

Per system and arm: hits, the median calls to first hit, the median time to
first hit, and the median time inside and outside the potential. A miss
counts as the full budget in calls and as the whole run in seconds, so the
medians and the rank tests see every seed. Each anneal arm is tested against
ABCluster with Fisher's exact test on the hit counts (two-sided) and with
the Mann-Whitney U test (normal approximation, tie-corrected, continuity
correction) on calls to first hit and on time to first hit.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_campaign as rc  # noqa: E402

ROOT = HERE.parents[1]
BENCH = ROOT / "bench" / "abcluster-h2h"
BEGIN = "<!-- BEGIN-SUMMARY -->"
END = "<!-- END-SUMMARY -->"
LABELS = {
    "abcluster": "ABCluster",
    "anneal-recommended": "anneal recommended",
    "anneal-cluster": "anneal for_cluster",
}
SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "hits",
    "median_calls_to_hit",
    "median_calls_to_hit_hits_only",
    "median_time_to_hit_s",
    "median_time_to_hit_hits_only_s",
    "median_wall_s",
    "median_pot_s",
    "median_outside_s",
    "fisher_p",
    "mwu_calls_p",
    "mwu_time_p",
]


def log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test of the table [[a, b], [c, d]].

    The sum of every table with the observed margins whose probability does
    not exceed the observed one, with the relative slack scipy uses.
    """
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    low, high = max(0, col1 - row2), min(col1, row1)

    def log_prob(x: int) -> float:
        return log_choose(row1, x) + log_choose(row2, col1 - x) - log_choose(total, col1)

    observed = math.exp(log_prob(a))
    cutoff = observed * (1.0 + 1e-7)
    p = sum(math.exp(log_prob(x)) for x in range(low, high + 1) if math.exp(log_prob(x)) <= cutoff)
    return min(1.0, p)


def average_ranks(values: list[float]) -> tuple[list[float], list[int]]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    ties: list[int] = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def mann_whitney(x: list[float], y: list[float]) -> float:
    """Two-sided Mann-Whitney U, normal approximation, tie and continuity corrected."""
    n1, n2 = len(x), len(y)
    n = n1 + n2
    ranks, ties = average_ranks(list(x) + list(y))
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    tie_term = sum(t**3 - t for t in ties) / (n * (n - 1))
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term)
    if var <= 0.0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    return min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2.0)))


def load_seeds() -> list[dict[str, str]]:
    path = BENCH / "seeds.csv"
    rows = rc.read_rows(path)
    if not rows:
        raise SystemExit(f"{path} has no rows")
    seen: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["system"], row["arm"], row["seed"])
        if key in seen:
            raise SystemExit(f"duplicate row {key}")
        seen[key] = row
        if row["host"] != "bench":
            raise SystemExit(f"{key} host is {row['host']}, expected bench")
        if int(row["budget"]) != rc.PAPER_BUDGETS[row["system"]]:
            raise SystemExit(f"{key} budget {row['budget']}")
    for system in rc.PAPER_BUDGETS:
        for arm in rc.ARMS:
            for seed in rc.SEEDS:
                if (system, arm, str(seed)) not in seen:
                    raise SystemExit(f"missing {system} {arm} {seed}")
    return rows


def calls_value(row: dict[str, str]) -> float:
    return float(row["calls_to_hit"]) if row["hit"] == "1" else float(row["budget"])


def time_value(row: dict[str, str]) -> float:
    return float(row["time_to_hit_s"]) if row["hit"] == "1" else float(row["wall_s"])


def group(rows: list[dict[str, str]], system: str, arm: str) -> list[dict[str, str]]:
    chosen = [r for r in rows if r["system"] == system and r["arm"] == arm]
    chosen.sort(key=lambda r: int(r["seed"]))
    return chosen


def optional_median(values: list[float]) -> float | None:
    return rc.median(values) if values else None


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for system in rc.PAPER_BUDGETS:
        abc = group(rows, system, "abcluster")
        for arm in rc.ARMS:
            g = group(rows, system, arm)
            hits = [r for r in g if r["hit"] == "1"]
            record = {
                "system": system,
                "arm": arm,
                "n": str(len(g)),
                "hits": str(len(hits)),
                "median_calls_to_hit": f"{rc.median([calls_value(r) for r in g]):.1f}",
                "median_calls_to_hit_hits_only": fmt(optional_median([calls_value(r) for r in hits]), "{:.1f}"),
                "median_time_to_hit_s": f"{rc.median([time_value(r) for r in g]):.6f}",
                "median_time_to_hit_hits_only_s": fmt(optional_median([time_value(r) for r in hits]), "{:.6f}"),
                "median_wall_s": f"{rc.median([float(r['wall_s']) for r in g]):.6f}",
                "median_pot_s": f"{rc.median([float(r['pot_s']) for r in g]):.6f}",
                "median_outside_s": f"{rc.median([float(r['outside_s']) for r in g]):.6f}",
                "fisher_p": "",
                "mwu_calls_p": "",
                "mwu_time_p": "",
            }
            if arm != "abcluster":
                abc_hits = sum(r["hit"] == "1" for r in abc)
                record["fisher_p"] = f"{fisher_exact(len(hits), len(g) - len(hits), abc_hits, len(abc) - abc_hits):.6g}"
                record["mwu_calls_p"] = f"{mann_whitney([calls_value(r) for r in g], [calls_value(r) for r in abc]):.6g}"
                record["mwu_time_p"] = f"{mann_whitney([time_value(r) for r in g], [time_value(r) for r in abc]):.6g}"
            out.append(record)
    return out


def fmt(value: float | None, pattern: str) -> str:
    return "" if value is None else pattern.format(value)


def seconds(text: str) -> str:
    if text == "":
        return "-"
    value = float(text)
    return f"{value:.3f}" if value < 100 else f"{value:.1f}"


def count(text: str) -> str:
    return "-" if text == "" else f"{float(text):,.0f}"


def p_text(text: str) -> str:
    if text == "":
        return ""
    value = float(text)
    return f"{value:.2g}" if value < 0.001 else f"{value:.3f}"


def table(summary: list[dict[str, str]]) -> list[str]:
    lines = [
        "| System | Budget | Arm | Hits | Median calls to first hit | Median calls, hits only "
        "| Median s to first hit | Median s in potential | Median s outside | Fisher p | MWU p calls | MWU p time |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['system']} | {rc.PAPER_BUDGETS[r['system']]:,} | {LABELS[r['arm']]} | {r['hits']}/{r['n']} "
            f"| {count(r['median_calls_to_hit'])} | {count(r['median_calls_to_hit_hits_only'])} "
            f"| {seconds(r['median_time_to_hit_s'])} | {seconds(r['median_pot_s'])} | {seconds(r['median_outside_s'])} "
            f"| {p_text(r['fisher_p'])} | {p_text(r['mwu_calls_p'])} | {p_text(r['mwu_time_p'])} |"
        )
    return lines


def checks(rows_dir: Path) -> list[str]:
    lines: list[str] = []
    parity = rc.read_rows(rows_dir / "parity.csv")
    if parity:
        worst = max(float(r["rel_diff"]) for r in parity)
        lines.append(
            f"Kernel parity: {len(parity)} geometries across {len({r['system'] for r in parity})} systems, "
            f"largest relative energy difference between `atom` and anneal {worst:.3e} "
            f"(limit {rc.PARITY_TOL:g}). Gold Gupta parameters: A, xi, d, p, q = "
            f"{' '.join(str(v) for v in rc.GOLD_PARAMETERS)}."
        )
    counter = rc.read_rows(rows_dir / "counter.csv")
    if counter:
        agree = sum(r["match"] == "1" for r in counter)
        lines.append(
            f"Call counter: shim count equals a debugger breakpoint count on {agree} of {len(counter)} "
            f"checked runs ({', '.join(sorted({r['system'] for r in counter}))})."
        )
    pilot = rc.read_rows(rows_dir / "pilot.csv")
    if pilot:
        chosen = ", ".join(f"{s} SN={rc.ABC_SN[s]}" for s in rc.PAPER_BUDGETS)
        lines.append(f"ABCluster population size, chosen on pilot seeds {rc.PILOT_SEEDS[0]} to {rc.PILOT_SEEDS[-1]}: {chosen}.")
    return lines


def splice(readme: Path, block: list[str]) -> None:
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{readme} lacks the summary markers")
    head, rest = text.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    readme.write_text(head + BEGIN + "\n" + "\n".join(block) + "\n" + END + tail)


def main() -> None:
    if abs(rc.median([1.0, 2.0, 3.0, 4.0]) - 2.5) > 1e-12:
        raise SystemExit("type 7 median drifted")
    rows = load_seeds()
    summary = summarise(rows)
    with (BENCH / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    revisions = sorted({r["code_revision"] for r in rows if r["arm"] == "abcluster"})
    heads = sorted({r["anneal_head"] for r in rows})
    cpus = sorted({r["cpu"] for r in rows})
    block = table(summary) + [""]
    block += checks(BENCH)
    block += [
        "",
        f"ABCluster revision: {'; '.join(revisions)}.",
        f"anneal revision: {'; '.join(heads)}.",
        f"CPU: {'; '.join(cpus)}.",
    ]
    splice(BENCH / "README.md", block)
    print("\n".join(block))


if __name__ == "__main__":
    main()
