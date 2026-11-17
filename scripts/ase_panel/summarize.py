#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Reprint the ASE panel from its committed rows.

Misses are censored at the call budget. Does not start a search.
``recommended`` is compared with both ASE arms. The for_cluster
comparison of those ASE arms is reprinted from the same rows.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(2000, 2012))
SYSTEMS = ("lj38", "lj55", "lj98", "lj104", "lj150", "au55")
ARMS = ("anneal", "recommended", "ase_bh", "ase_mh")
ASE_ARMS = ("ase_bh", "ase_mh")
LABEL = {
    "anneal": "anneal",
    "recommended": "recommended",
    "ase_bh": "basin hopping",
    "ase_mh": "minima hopping",
}
REFERENCE = {
    "lj38": -173.928427,
    "lj55": -279.248470,
    "lj98": -543.665361,
    "lj104": -582.086642,
    "lj150": -893.310258,
    "au55": -191.03085280,
}
BUDGET = {
    "lj38": "400000",
    "lj55": "1000000",
    "lj98": "4000000",
    "lj104": "4000000",
    "lj150": "4000000",
    "au55": "1000000",
}
POTENTIAL = {
    "lj38": "Anneal is Config::for_cluster. Recommended is Config::recommended.",
    "lj55": "Anneal is Config::for_cluster. Recommended is Config::recommended.",
    "lj98": "Anneal is Config::for_cluster. Recommended is Config::recommended.",
    "lj104": "Anneal is Config::for_cluster. Recommended is Config::recommended.",
    "lj150": "Anneal is Config::for_cluster. Recommended is Config::recommended.",
    "au55": "Anneal is Config::for_cluster on Gupta::copper_gold(0, 55), with the "
    "pair length and well depth as the scales. Recommended is "
    "Config::recommended_with_scales at those scales. ASE uses the same Gupta through asg_eval.",
}


def type7(ordered: list[float], probability: float) -> float:
    count = len(ordered)
    rank = 1.0 + (count - 1) * probability
    lower = math.floor(rank)
    weight = rank - lower
    left = ordered[int(lower) - 1]
    if weight == 0.0 or int(lower) >= count:
        return float(left)
    return float(left * (1.0 - weight) + ordered[int(lower)] * weight)


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def calls(row: dict[str, str]) -> float:
    if row["evals_to_hit"]:
        return float(row["evals_to_hit"])
    return float(row["budget"])


def g6(value: float) -> str:
    return f"{value:.6g}"


def reference_text(value: float) -> str:
    text = f"{value:.8f}"
    return text[:-2] if text.endswith("00") else text


def load(system: str) -> dict[str, list[dict[str, str]]]:
    path = ROOT / "bench" / f"ase-{system}" / "rows.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {arm: [] for arm in ARMS}
    for row in rows:
        if row["system"] != system or row["budget"] != BUDGET[system] or row["host"] != "bench":
            raise SystemExit(f"unexpected row {row}")
        if row["arm"] not in grouped:
            raise SystemExit(f"unknown arm {row['arm']}")
        grouped[row["arm"]].append(row)
    for arm in ARMS:
        seeds = tuple(sorted(int(row["seed"]) for row in grouped[arm]))
        if seeds != SEEDS:
            raise SystemExit(f"{system} {arm} seeds {seeds}")
        grouped[arm].sort(key=lambda row: int(row["seed"]))
    return grouped


def fisher(hits: int, base: int) -> float:
    return float(fisher_exact([[hits, 12 - hits], [base, 12 - base]])[1])


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    if len(set(left + right)) == 1:
        return len(left) * len(right) / 2.0, 1.0
    result = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic", use_continuity=True)
    return float(result.statistic), float(result.pvalue)


def paired_ratio(left: list[dict[str, str]], right: list[dict[str, str]]) -> tuple[float, float]:
    over = median([float(a["wall_s"]) / float(b["wall_s"]) for a, b in zip(left, right)])
    back = median([float(b["wall_s"]) / float(a["wall_s"]) for a, b in zip(left, right)])
    return over, back


def against(group: list[dict[str, str]], other: list[dict[str, str]]) -> tuple[float, float, float]:
    hits = sum(row["hit"] == "1" for row in group)
    base = sum(row["hit"] == "1" for row in other)
    u_stat, mw_p = mann_whitney([calls(row) for row in group], [calls(row) for row in other])
    return fisher(hits, base), u_stat, mw_p


def join_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def seed_list(rows: list[dict[str, str]]) -> str:
    return join_list([row["seed"] for row in rows])


def write_readme(system: str, grouped: dict[str, list[dict[str, str]]], reference: float) -> None:
    anneal = grouped["anneal"]
    caps = sorted({row["cap_s"] for arm in ASE_ARMS for row in grouped[arm]})
    if len(caps) != 1:
        raise SystemExit(f"{system} ASE cap is not unique: {caps}")
    cap = caps[0]
    slowest = max(anneal, key=lambda row: float(row["wall_s"]))
    triple = f"{3.0 * float(slowest['wall_s']):.6f}"
    cap_sentence = f"The wall cap is {cap} s, the cap stored on the ASE rows."
    if triple == cap:
        cap_sentence = (
            f"The wall cap is {cap} s, three times the slowest for_cluster wall, "
            f"seed {slowest['seed']} at {float(slowest['wall_s']):.6f} s. "
            "The recommended arm uses that stored cap."
        )
    title = "Au55" if system == "au55" else system.upper()
    lines = [
        "Copyright (c) anneal developers",
        "",
        f"{title}, seeds 2000-2011, budget {int(BUDGET[system]):,}, host bench, one core.",
        f"The reference minimum is {reference_text(reference)}.",
        POTENTIAL[system],
        "ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults.",
        "The ASE rows are the committed rows and were not rerun.",
    ]
    for arm in ARMS:
        revisions = sorted({row["code_revision"] for row in grouped[arm]})
        lines.append(f"{LABEL[arm]} code_revision {', '.join(revisions)}.")
    lines.append(cap_sentence)
    for arm in ARMS:
        stops = sum(row["stop"] == "wall" for row in grouped[arm])
        budget_stops = sum(row["stop"] == "budget" for row in grouped[arm])
        hits = [row for row in grouped[arm] if row["hit"] == "1"]
        misses = [row for row in grouped[arm] if row["hit"] != "1"]
        lines.append(
            f"{LABEL[arm]} hit on {len(hits)} of 12 seeds, "
            f"spent the call budget on {budget_stops} of 12, "
            f"and reached the wall cap on {stops} of 12."
        )
        if hits and len(hits) < 12:
            lines.append(f"{LABEL[arm]} hits are seeds {seed_list(hits)}.")
        if misses and len(misses) < 12:
            bests = join_list([f"{float(row['best']):.8f}" for row in misses])
            lines.append(f"{LABEL[arm]} misses are seeds {seed_list(misses)}, best {bests}.")
    lines.extend(
        [
            "",
            "| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | "
            f"gap to {reference_text(reference)} | median wall s | over anneal | anneal over | "
            "median inside s | median outside s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    base_calls = [calls(row) for row in anneal]
    base_hits = sum(row["hit"] == "1" for row in anneal)
    for arm in ARMS:
        group = grouped[arm]
        hits = sum(row["hit"] == "1" for row in group)
        best = median([float(row["best"]) for row in group])
        wall = median([float(row["wall_s"]) for row in group])
        inside = median([float(row["inside_s"]) for row in group])
        outside = median([float(row["outside_s"]) for row in group])
        if arm == "anneal":
            over = back = fisher_p = 1.0
            u_stat = len(base_calls) * len(base_calls) / 2.0
            mw_p = 1.0
        else:
            over, back = paired_ratio(group, anneal)
            fisher_p = fisher(hits, base_hits)
            u_stat, mw_p = mann_whitney([calls(row) for row in group], base_calls)
        lines.append(
            f"| {LABEL[arm]} | {hits}/12 | {g6(fisher_p)} | {g6(u_stat)} | {g6(mw_p)} | "
            f"{best:.8f} | {best - reference:.8f} | {wall:.6f} | {over:.6f} | {back:.6f} | "
            f"{inside:.6f} | {outside:.6f} |"
        )
    lines.extend(
        [
            "",
            "Fisher and Mann-Whitney below compare recommended with each ASE arm. "
            "Misses are censored at the call budget.",
            "",
            "| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | "
            "median wall ratio | inverse ratio |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    recommended = grouped["recommended"]
    rec_hits = sum(row["hit"] == "1" for row in recommended)
    for arm in ASE_ARMS:
        other = grouped[arm]
        other_hits = sum(row["hit"] == "1" for row in other)
        fisher_p, u_stat, mw_p = against(recommended, other)
        over, back = paired_ratio(recommended, other)
        lines.append(
            f"| recommended against {LABEL[arm]} | {rec_hits}/12 vs {other_hits}/12 | "
            f"{g6(fisher_p)} | {g6(u_stat)} | {g6(mw_p)} | {over:.6f} | {back:.6f} |"
        )
    lines.extend(
        [
            "",
            "Misses are censored at the call budget for the rank test.",
            "Medians are type 7. Wall ratios are paired by seed.",
            "Fisher is the two-sided exact test. Mann-Whitney is two-sided with",
            "the tie correction and the continuity correction.",
            "Inside is time in the potential. Outside is the rest of the wall.",
            "",
        ]
    )
    (ROOT / "bench" / f"ase-{system}" / "README.md").write_text("\n".join(lines))


def summarise(system: str) -> list[str]:
    reference = REFERENCE[system]
    grouped = load(system)
    anneal = grouped["anneal"]
    lines = [
        "arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,median_outside_s,"
        "median_over_anneal,median_anneal_over,fisher_p,mann_whitney_U,mann_whitney_p,wall_stops,wall_cap_s,"
        "fisher_p_bh,mann_whitney_U_bh,mann_whitney_p_bh,fisher_p_mh,mann_whitney_U_mh,mann_whitney_p_mh",
    ]
    base_calls = [calls(row) for row in anneal]
    base_hits = sum(row["hit"] == "1" for row in anneal)
    print(f"{system} seeds 2000-2011, budget {BUDGET[system]}, host bench, reference {reference}")
    combined = []
    for arm in ARMS:
        group = grouped[arm]
        hits = sum(row["hit"] == "1" for row in group)
        best = median([float(row["best"]) for row in group])
        wall = median([float(row["wall_s"]) for row in group])
        inside = median([float(row["inside_s"]) for row in group])
        outside = median([float(row["outside_s"]) for row in group])
        wall_stops = sum(row.get("stop") == "wall" for row in group)
        caps = [row["cap_s"] for row in group if row.get("cap_s")]
        cap = caps[0] if caps else ""
        if arm == "anneal":
            over = back = fisher_p = 1.0
            u_stat, mw_p = len(base_calls) * len(base_calls) / 2.0, 1.0
            cap = ""
        else:
            over, back = paired_ratio(group, anneal)
            fisher_p = fisher(hits, base_hits)
            u_stat, mw_p = mann_whitney([calls(row) for row in group], base_calls)
        versus = {}
        for other_arm in ASE_ARMS:
            fisher_o, u_o, mw_o = against(group, grouped[other_arm])
            versus[other_arm] = (fisher_o, u_o, mw_o)
        print(
            f"{arm} hits {hits}/12 median_best {best:.8f} gap {best - reference:.8f} "
            f"wall {wall:.6f} inside {inside:.6f} outside {outside:.6f} "
            f"over {over:.6f} back {back:.6f} fisher {g6(fisher_p)} U {g6(u_stat)} mw {g6(mw_p)} "
            f"wall_stops {wall_stops} cap {cap} "
            f"bh {g6(versus['ase_bh'][0])} {g6(versus['ase_bh'][1])} {g6(versus['ase_bh'][2])} "
            f"mh {g6(versus['ase_mh'][0])} {g6(versus['ase_mh'][1])} {g6(versus['ase_mh'][2])}"
        )
        lines.append(
            f"{arm},{hits},12,{best:.8f},{best - reference:.8f},{wall:.6f},{inside:.6f},{outside:.6f},"
            f"{over:.6f},{back:.6f},{g6(fisher_p)},{g6(u_stat)},{g6(mw_p)},{wall_stops},{cap},"
            f"{g6(versus['ase_bh'][0])},{g6(versus['ase_bh'][1])},{g6(versus['ase_bh'][2])},"
            f"{g6(versus['ase_mh'][0])},{g6(versus['ase_mh'][1])},{g6(versus['ase_mh'][2])}"
        )
        combined.append(
            f"{system},{arm},{hits},12,{best:.8f},{best - reference:.8f},{wall:.6f},{inside:.6f},{outside:.6f},"
            f"{g6(versus['ase_bh'][0])},{g6(versus['ase_bh'][1])},{g6(versus['ase_bh'][2])},"
            f"{g6(versus['ase_mh'][0])},{g6(versus['ase_mh'][1])},{g6(versus['ase_mh'][2])},{cap}"
        )
    (ROOT / "bench" / f"ase-{system}" / "summary.csv").write_text("\n".join(lines) + "\n")
    write_readme(system, grouped, reference)
    return combined


def main() -> None:
    systems = sys.argv[1:]
    if systems == ["--all"]:
        systems = list(SYSTEMS)
    if not systems or any(system not in REFERENCE for system in systems):
        raise SystemExit("usage: summarize.py SYSTEM | --all")
    combined = [
        "system,arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,median_outside_s,"
        "fisher_p_bh,mann_whitney_U_bh,mann_whitney_p_bh,fisher_p_mh,mann_whitney_U_mh,mann_whitney_p_mh,wall_cap_s",
    ]
    for system in systems:
        combined.extend(summarise(system))
    if systems == list(SYSTEMS):
        out = ROOT / "bench" / "ase-recommended" / "summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(combined) + "\n")


if __name__ == "__main__":
    main()
