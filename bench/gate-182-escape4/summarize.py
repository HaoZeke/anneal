#!/usr/bin/env python3
"""Analyze PR 182 escape-4 gate rows and produce summary CSV and Markdown."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20261123
UPPER_LIMIT = 1.10

CELL_ORDER = (
    ("lj98_short_d0", 2500),
    ("lj98_short_d4", 2500),
    ("lj98_traj_d0", 5000),
    ("lj98_traj_d4", 5000),
    ("lj38_short_d0", 2500),
    ("lj38_short_d4", 2500),
    ("lj38_d0", 400_000),
    ("lj38_d4", 400_000),
    ("lj55", 1_000_000),
    ("lj75", 4_000_000),
    ("lj98", 4_000_000),
    ("lj102", 4_000_000),
    ("lj103", 4_000_000),
    ("lj104", 4_000_000),
    ("lj150", 4_000_000),
)

SUMMARY_FIELDS = (
    "cell",
    "n",
    "budget",
    "deferral",
    "hits_a",
    "hits_b",
    "hits_a2",
    "hits_b2",
    "fisher_two_sided_p",
    "fisher_a_greater_p",
    "med_calls_a",
    "med_calls_b",
    "mwu_u_b",
    "mwu_two_sided_p",
    "mwu_b_slower_p",
    "med_best_a",
    "med_best_b",
    "min_best_a",
    "min_best_b",
    "med_hops_a",
    "med_hops_b",
    "outside_med_ba",
    "outside_ci_lo_ba",
    "outside_ci_hi_ba",
    "outside_med_a2a",
    "outside_ci_lo_a2a",
    "outside_ci_hi_a2a",
    "outside_real_reg",
    "rss_med_ba",
    "rss_ci_lo_ba",
    "rss_ci_hi_ba",
    "rss_med_a2a",
    "rss_ci_lo_a2a",
    "rss_ci_hi_a2a",
    "rss_real_reg",
    "hit_pass",
    "calls_pass",
    "outside_pass",
    "rss_pass",
    "verdict",
)


def type7(values: list[float], prob: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n < 1:
        raise ValueError("type7 needs a non-empty sample")
    if n == 1:
        return ordered[0]
    pos = (n - 1) * prob
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def bootstrap_median_ci(values: list[float], rng: random.Random) -> tuple[float, float, float]:
    est = type7(values, 0.5)
    n = len(values)
    boots = [
        type7([values[rng.randrange(n)] for _ in range(n)], 0.5)
        for _ in range(BOOTSTRAPS)
    ]
    return est, type7(boots, 0.025), type7(boots, 0.975)


def fisher_exact(hits_a: int, hits_b: int, n: int) -> tuple[float, float]:
    total_hits = hits_a + hits_b
    denom = math.comb(2 * n, total_hits)

    def prob(k: int) -> float:
        other = total_hits - k
        if not 0 <= other <= n:
            return 0.0
        return math.comb(n, k) * math.comb(n, other) / denom

    p_obs = prob(hits_a)
    p_two = min(1.0, sum(prob(k) for k in range(n + 1) if prob(k) <= p_obs + 1e-12))
    p_a_gt = min(1.0, sum(prob(k) for k in range(hits_a, n + 1)))
    return p_two, p_a_gt


def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float, float]:
    pooled = sorted(a + b)
    ranks: dict[float, float] = {}
    tie_term = 0
    idx = 0
    while idx < len(pooled):
        end = idx
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[idx]:
            end += 1
        ranks[pooled[idx]] = (idx + end + 2) / 2.0
        w = end - idx + 1
        tie_term += w**3 - w
        idx = end + 1
    n_a, n_b = len(a), len(b)
    u_b = sum(ranks[v] for v in b) - n_b * (n_b + 1) / 2.0
    mean = n_a * n_b / 2.0
    total = n_a + n_b
    var = n_a * n_b / 12.0 * (total + 1 - tie_term / (total * (total - 1)))
    if var <= 0.0:
        return u_b, 1.0, 1.0
    sd = math.sqrt(var)
    diff = abs(u_b - mean) - 0.5
    z_two = max(0.0, diff) / sd
    p_two = min(1.0, math.erfc(z_two / math.sqrt(2.0)))
    z_b_gt = (u_b - mean - 0.5) / sd
    p_b_gt = 0.5 * math.erfc(z_b_gt / math.sqrt(2.0))
    return u_b, p_two, p_b_gt


def fmt(val: float, digits: int = 6) -> str:
    return f"{val:.{digits}g}"


def summarize(rows_path: Path) -> list[dict[str, str]]:
    with rows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_cell_arm: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_cell_arm.setdefault((row["cell"], row["arm"]), []).append(row)

    rng = random.Random(BOOTSTRAP_SEED)
    out: list[dict[str, str]] = []

    for cell, budget in CELL_ORDER:
        if (cell, "A") not in by_cell_arm:
            continue
        arms = {arm: by_cell_arm[(cell, arm)] for arm in ("A", "B", "A2", "B2")}
        n = len(arms["A"])
        if any(len(arms[a]) != n for a in arms):
            raise RuntimeError(f"{cell}: mismatched arm counts {[len(arms[a]) for a in arms]}")

        for i in range(n):
            seeds = {arms[a][i]["seed"] for a in arms}
            reps = {arms[a][i]["inner_reps"] for a in arms}
            if len(seeds) != 1 or len(reps) != 1:
                raise RuntimeError(f"{cell} round {i}: seed/reps mismatch {seeds} {reps}")

        hits = {a: sum(int(r["hit"]) for r in arms[a]) for a in arms}
        calls = {
            a: [
                float(r["calls_to_hit"]) if int(r["hit"]) == 1 else float(budget + 1)
                for r in arms[a]
            ]
            for a in arms
        }
        best = {a: [float(r["best_energy"]) for r in arms[a]] for a in arms}
        hops = {a: [float(r["hops"]) for r in arms[a]] for a in arms}
        outside_ba = [
            float(b["outside_s"]) / float(a["outside_s"])
            for a, b in zip(arms["A"], arms["B"], strict=True)
        ]
        outside_a2a = [
            float(a2["outside_s"]) / float(a["outside_s"])
            for a, a2 in zip(arms["A"], arms["A2"], strict=True)
        ]
        rss_ba = [
            float(b["rss_anon_kib"]) / float(a["rss_anon_kib"])
            for a, b in zip(arms["A"], arms["B"], strict=True)
        ]
        rss_a2a = [
            float(a2["rss_anon_kib"]) / float(a["rss_anon_kib"])
            for a, a2 in zip(arms["A"], arms["A2"], strict=True)
        ]

        f_two, f_a_gt = fisher_exact(hits["A"], hits["B"], n)
        u_b, mwu_two, mwu_b_gt = mann_whitney(calls["A"], calls["B"])

        out_ba_med, out_ba_lo, out_ba_hi = bootstrap_median_ci(outside_ba, rng)
        out_a2a_med, out_a2a_lo, out_a2a_hi = bootstrap_median_ci(outside_a2a, rng)
        rss_ba_med, rss_ba_lo, rss_ba_hi = bootstrap_median_ci(rss_ba, rng)
        rss_a2a_med, rss_a2a_lo, rss_a2a_hi = bootstrap_median_ci(rss_a2a, rng)

        med_calls_a = type7(calls["A"], 0.5)
        med_calls_b = type7(calls["B"], 0.5)

        out_real_reg = out_ba_lo > 1.0 and out_ba_lo > out_a2a_hi
        rss_real_reg = rss_ba_lo > 1.0 and rss_ba_lo > rss_a2a_hi

        hit_pass = hits["B"] >= hits["A"]
        calls_pass = not (mwu_b_gt < 0.05 and med_calls_b > med_calls_a) and med_calls_b <= med_calls_a
        outside_pass = out_ba_hi <= UPPER_LIMIT
        rss_pass = rss_ba_hi <= UPPER_LIMIT
        verdict = "PASS" if (hit_pass and calls_pass and outside_pass and rss_pass) else "FAIL"

        out.append(
            {
                "cell": cell,
                "n": str(n),
                "budget": str(budget),
                "deferral": arms["A"][0]["deferral"],
                "hits_a": str(hits["A"]),
                "hits_b": str(hits["B"]),
                "hits_a2": str(hits["A2"]),
                "hits_b2": str(hits["B2"]),
                "fisher_two_sided_p": fmt(f_two),
                "fisher_a_greater_p": fmt(f_a_gt),
                "med_calls_a": fmt(med_calls_a, 8),
                "med_calls_b": fmt(med_calls_b, 8),
                "mwu_u_b": fmt(u_b, 6),
                "mwu_two_sided_p": fmt(mwu_two),
                "mwu_b_slower_p": fmt(mwu_b_gt),
                "med_best_a": f"{type7(best['A'], 0.5):.12g}",
                "med_best_b": f"{type7(best['B'], 0.5):.12g}",
                "min_best_a": f"{min(best['A']):.12g}",
                "min_best_b": f"{min(best['B']):.12g}",
                "med_hops_a": fmt(type7(hops["A"], 0.5), 6),
                "med_hops_b": fmt(type7(hops["B"], 0.5), 6),
                "outside_med_ba": fmt(out_ba_med, 6),
                "outside_ci_lo_ba": fmt(out_ba_lo, 6),
                "outside_ci_hi_ba": fmt(out_ba_hi, 6),
                "outside_med_a2a": fmt(out_a2a_med, 6),
                "outside_ci_lo_a2a": fmt(out_a2a_lo, 6),
                "outside_ci_hi_a2a": fmt(out_a2a_hi, 6),
                "outside_real_reg": "1" if out_real_reg else "0",
                "rss_med_ba": fmt(rss_ba_med, 6),
                "rss_ci_lo_ba": fmt(rss_ba_lo, 6),
                "rss_ci_hi_ba": fmt(rss_ba_hi, 6),
                "rss_med_a2a": fmt(rss_a2a_med, 6),
                "rss_ci_lo_a2a": fmt(rss_a2a_lo, 6),
                "rss_ci_hi_a2a": fmt(rss_a2a_hi, 6),
                "rss_real_reg": "1" if rss_real_reg else "0",
                "hit_pass": "1" if hit_pass else "0",
                "calls_pass": "1" if calls_pass else "0",
                "outside_pass": "1" if outside_pass else "0",
                "rss_pass": "1" if rss_pass else "0",
                "verdict": verdict,
            }
        )
    return out


def write_summary(summaries: list[dict[str, str]]) -> None:
    with (HERE / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    overall = "PASS" if summaries and all(r["verdict"] == "PASS" for r in summaries) else "FAIL"
    lines = [
        "# PR 182 Escape-4 Performance Gate Summary",
        "",
        f"Overall verdict: **{overall}**",
        "",
        "| Cell | Hits A/B | Fisher p | Med calls A/B | MWU p | Med best A/B | Med hops A/B | Outside B/A [95% CI] (A2/A band) | RSS B/A [95% CI] (A2/A band) | Verdict |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for r in summaries:
        lines.append(
            f"| {r['cell']} | {r['hits_a']}/{r['hits_b']} | {r['fisher_two_sided_p']} | "
            f"{r['med_calls_a']}/{r['med_calls_b']} | {r['mwu_two_sided_p']} | "
            f"{r['med_best_a']}/{r['med_best_b']} | {r['med_hops_a']}/{r['med_hops_b']} | "
            f"{r['outside_med_ba']} [{r['outside_ci_lo_ba']}, {r['outside_ci_hi_ba']}] "
            f"({r['outside_ci_lo_a2a']}..{r['outside_ci_hi_a2a']}) | "
            f"{r['rss_med_ba']} [{r['rss_ci_lo_ba']}, {r['rss_ci_hi_ba']}] "
            f"({r['rss_ci_lo_a2a']}..{r['rss_ci_hi_a2a']}) | {r['verdict']} |"
        )
    lines.append("")
    (HERE / "SUMMARY.md").write_text("\n".join(lines))


def main() -> int:
    summaries = summarize(HERE / "rows.csv")
    write_summary(summaries)
    overall = "PASS" if summaries and all(r["verdict"] == "PASS" for r in summaries) else "FAIL"
    print(f"overall={overall} cells={len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
