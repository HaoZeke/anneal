#!/usr/bin/env python3
"""Pooled LJ75 arch-chains comparison. Stdlib, numpy, and scipy only."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
RESULTS = HERE / "results.json"
REPORT = HERE / "REPORT-lj75.md"

BUDGET = 4_000_000
MISS_CALLS = BUDGET + 1
ARMS = ("comm", "indep", "single")
PAIRS = (("comm", "single"), ("indep", "single"), ("comm", "indep"))
SHARDS = (
    ("2000-2002", range(2000, 2003)),
    ("2003-2005", range(2003, 2006)),
    ("2006-2008", range(2006, 2009)),
    ("2009-2011", range(2009, 2012)),
)
REPORTED = {
    "single": [3, 1, 3, 2],
    "comm": [2, 2, 1, 2],
    "indep": [1, 1, 2, 2],
}
RESAMPLES = 10_000
SEED = 20261123


def load_lj75() -> dict[str, list[dict[str, int]]]:
    grouped: dict[str, list[dict[str, int]]] = {arm: [] for arm in ARMS}
    corrections = 0
    with ROWS.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["system"] != "lj75":
                continue
            arm = row["arm"]
            if arm not in grouped:
                raise SystemExit(f"unknown arm {arm}")
            if row["hit"] not in ("0", "1"):
                raise SystemExit(f"bad hit field {row['hit']}")
            hit = int(row["hit"])
            calls = int(row["calls_to_hit"])
            if hit == 0 and calls != MISS_CALLS:
                calls = MISS_CALLS
                corrections += 1
            if hit == 1 and calls == MISS_CALLS:
                raise SystemExit(f"hit recorded at miss calls for seed {row['seed']} {arm}")
            grouped[arm].append(
                {"seed": int(row["seed"]), "hit": hit, "calls": calls}
            )
    for arm, rows in grouped.items():
        seeds = [row["seed"] for row in rows]
        if len(rows) != 12 or len(set(seeds)) != 12 or set(seeds) != set(range(2000, 2012)):
            raise SystemExit(f"seed set for {arm}: {sorted(seeds)}")
        rows.sort(key=lambda item: item["seed"])
    total = sum(len(rows) for rows in grouped.values())
    keys = {(arm, row["seed"]) for arm, rows in grouped.items() for row in rows}
    if total != 36 or len(keys) != 36:
        raise SystemExit(f"expected 36 unique rows, found {total} keys {len(keys)}")
    if corrections:
        print(f"miss_calls_corrections {corrections}", file=sys.stderr)
    return grouped


def shard_hits(grouped: dict[str, list[dict[str, int]]]) -> dict[str, list[int]]:
    observed = {arm: [] for arm in ("single", "comm", "indep")}
    for _name, seeds in SHARDS:
        wanted = set(seeds)
        for arm in observed:
            hits = sum(row["hit"] for row in grouped[arm] if row["seed"] in wanted)
            observed[arm].append(hits)
    return observed


def median_ratio_ci(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    point_right = float(np.median(right))
    if point_right == 0.0:
        raise SystemExit("median calls on the right arm is 0")
    point = float(np.median(left) / point_right)
    rng = np.random.default_rng(SEED)
    stats = np.empty(RESAMPLES, dtype=float)
    n_left = left.size
    n_right = right.size
    for i in range(RESAMPLES):
        sample_left = rng.choice(left, size=n_left, replace=True)
        sample_right = rng.choice(right, size=n_right, replace=True)
        denom = float(np.median(sample_right))
        if denom == 0.0:
            raise SystemExit("bootstrap median calls is 0")
        stats[i] = float(np.median(sample_left) / denom)
    lo, hi = np.quantile(stats, [0.025, 0.975], method="linear")
    return point, float(lo), float(hi)


def ci_fact(lo: float, hi: float) -> str:
    if lo > 1.0:
        return "The 95% interval lies entirely above 1."
    if hi < 1.0:
        return "The 95% interval lies entirely below 1."
    return "The 95% interval includes 1."


def fmt_calls(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.4f}"


def fmt_num(value: float) -> str:
    return f"{value:.6g}"


def fmt_p(value: float) -> str:
    return f"{value:.6e}"


def seq(values: list[int]) -> str:
    return ", ".join(str(int(value)) for value in values)


def analyze(grouped: dict[str, list[dict[str, int]]]) -> dict:
    observed = shard_hits(grouped)
    reported = {arm: list(REPORTED[arm]) for arm in ("single", "comm", "indep")}
    match = observed == reported
    arms_out = {}
    calls = {}
    for arm in ARMS:
        rows = grouped[arm]
        sample = np.asarray([row["calls"] for row in rows], dtype=float)
        calls[arm] = sample
        arms_out[arm] = {
            "hits": int(sum(row["hit"] for row in rows)),
            "median_calls_to_hit": float(np.median(sample)),
            "n": len(rows),
        }
    pairs = []
    for left, right in PAIRS:
        hits_left = arms_out[left]["hits"]
        hits_right = arms_out[right]["hits"]
        table = [
            [hits_left, 12 - hits_left],
            [hits_right, 12 - hits_right],
        ]
        odds, fisher_p = fisher_exact(table, alternative="two-sided")
        try:
            mwu = mannwhitneyu(
                calls[left],
                calls[right],
                alternative="two-sided",
                use_continuity=True,
                method="auto",
            )
            mwu_method = "auto"
        except ValueError:
            mwu = mannwhitneyu(
                calls[left],
                calls[right],
                alternative="two-sided",
                use_continuity=True,
                method="asymptotic",
            )
            mwu_method = "asymptotic"
        ratio, lo, hi = median_ratio_ci(calls[left], calls[right])
        pairs.append(
            {
                "contrast": f"{left} vs {right}",
                "fisher_odds_ratio_left": float(odds),
                "fisher_p": float(fisher_p),
                "fisher_table": table,
                "left": left,
                "median_calls_ratio": ratio,
                "median_calls_ratio_ci95": [lo, hi],
                "mwu_method": mwu_method,
                "mwu_p": float(mwu.pvalue),
                "mwu_u_left": float(mwu.statistic),
                "right": right,
            }
        )
    return {
        "arms": arms_out,
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "budget": BUDGET,
        "global_minimum": "-397.4923309829",
        "miss_calls": MISS_CALLS,
        "numpy": np.__version__,
        "observed_shard_hits": observed,
        "pairs": pairs,
        "reported_shard_hits": reported,
        "scipy": __import__("scipy").__version__,
        "shard_hits_match": match,
        "shard_order": [name for name, _seeds in SHARDS],
    }


def write_report(payload: dict) -> None:
    lines = [
        "# LJ75 arch-chains pooled comparison",
        "",
        "Pooled rows: 36 (12 seeds, arms comm, indep, and single). "
        "Budget 4000000 oracle calls. A miss is counted as 4000001 calls. "
        "Hits are the recorded hit field. The campaign global minimum is "
        "-397.4923309829.",
        "",
        "## Per arm",
        "",
        "| arm | hits | median calls_to_hit |",
        "| --- | --- | --- |",
    ]
    for arm in ARMS:
        info = payload["arms"][arm]
        lines.append(
            f"| {arm} | {info['hits']}/12 | {fmt_calls(info['median_calls_to_hit'])} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise tests",
            "",
            "The median-calls ratio is median(left) / median(right). "
            "Fisher's exact test is two-sided on the hit/miss table, and the "
            "odds ratio is the left arm's hit odds divided by the right arm's "
            "hit odds. The Mann-Whitney statistic is the two-sided U for the "
            "left arm from scipy.stats.mannwhitneyu, with the continuity "
            "correction and method auto. The 95% interval is the 2.5 and 97.5 "
            "percentiles of 10000 resamples of the median-calls ratio. Each "
            "contrast uses numpy Generator seed 20261123 and resamples the two "
            "arms independently with replacement.",
            "",
        ]
    )
    for pair in payload["pairs"]:
        lo, hi = pair["median_calls_ratio_ci95"]
        lines.append(f"### {pair['contrast']}")
        lines.append("")
        lines.append(
            f"Hits {pair['left']} {payload['arms'][pair['left']]['hits']}/12, "
            f"{pair['right']} {payload['arms'][pair['right']]['hits']}/12. "
            f"Fisher odds ratio {fmt_num(pair['fisher_odds_ratio_left'])}, "
            f"two-sided p {fmt_p(pair['fisher_p'])}. "
            f"Mann-Whitney U {fmt_num(pair['mwu_u_left'])}, "
            f"two-sided p {fmt_p(pair['mwu_p'])}. "
            f"Median-calls ratio {fmt_num(pair['median_calls_ratio'])}, "
            f"95% interval [{fmt_num(lo)}, {fmt_num(hi)}]. "
            f"{ci_fact(lo, hi)}"
        )
        lines.append("")
    lines.extend(
        [
            "## Shard hit check",
            "",
            "Reported hits for shards 2000, 2003, 2006, and 2009: "
            f"single {seq(payload['reported_shard_hits']['single'])}; "
            f"comm {seq(payload['reported_shard_hits']['comm'])}; "
            f"indep {seq(payload['reported_shard_hits']['indep'])}.",
            "",
            "Observed hits in that shard order: "
            f"single {seq(payload['observed_shard_hits']['single'])}; "
            f"comm {seq(payload['observed_shard_hits']['comm'])}; "
            f"indep {seq(payload['observed_shard_hits']['indep'])}.",
            "",
        ]
    )
    if payload["shard_hits_match"]:
        lines.append("Observed shard hits match the reported check.")
    else:
        lines.append(
            "MISMATCH: observed shard hits differ from the reported check. "
            "The counts above are the data."
        )
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    payload = analyze(load_lj75())
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    RESULTS.write_text(encoded, encoding="utf-8")
    write_report(payload)
    for arm in ARMS:
        info = payload["arms"][arm]
        print(
            f"arm {arm} hits {info['hits']}/12 "
            f"median_calls {info['median_calls_to_hit']:.16g}"
        )
    for pair in payload["pairs"]:
        lo, hi = pair["median_calls_ratio_ci95"]
        print(
            f"pair {pair['contrast']} fisher_p={pair['fisher_p']:.16g} "
            f"mwu_p={pair['mwu_p']:.16g} mwu_u={pair['mwu_u_left']:.16g} "
            f"ratio={pair['median_calls_ratio']:.16g} "
            f"ci={lo:.16g},{hi:.16g}"
        )
    print(f"shard_hits_match {payload['shard_hits_match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
