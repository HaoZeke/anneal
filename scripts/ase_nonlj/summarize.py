#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize the non-LJ (Morse) ASE panel from its committed rows.

Misses are censored at the call budget. Does not start a search.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(2000, 2012))
SYSTEMS = ("m38r6", "m38r14", "m55r6", "m55r14")
ARMS = ("for_cluster", "recommended", "ase_bh", "ase_mh")
ASE_ARMS = ("ase_bh", "ase_mh")
LABEL = {
    "for_cluster": "for_cluster",
    "recommended": "recommended",
    "ase_bh": "basin hopping",
    "ase_mh": "minima hopping",
}
SPEC = {
    "m38r6": (38, 6.0, "400000", -157.477108, "38E"),
    "m38r14": (38, 14.0, "400000", -144.321054, "38D"),
    "m55r6": (55, 6.0, "1000000", -250.286609, "55B"),
    "m55r14": (55, 14.0, "1000000", -220.646208, "55C"),
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


def join_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def seed_list(rows: list[dict[str, str]]) -> str:
    return join_list([row["seed"] for row in rows])


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


def load_rows() -> dict[str, dict[str, list[dict[str, str]]]]:
    path = ROOT / "bench" / "ase-nonlj" / "rows.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {system: {arm: [] for arm in ARMS} for system in SYSTEMS}
    for row in rows:
        system = row["system"]
        arm = row["arm"]
        if system not in SPEC or arm not in ARMS:
            raise SystemExit(f"unexpected row {row}")
        if row["budget"] != SPEC[system][2] or row["host"] != "bench":
            raise SystemExit(f"unexpected budget or host {row}")
        if not row["code_revision"].strip():
            raise SystemExit(f"missing code_revision {row}")
        grouped[system][arm].append(row)
    for system in SYSTEMS:
        for arm in ARMS:
            seeds = tuple(sorted(int(row["seed"]) for row in grouped[system][arm]))
            if seeds != SEEDS:
                raise SystemExit(f"{system} {arm} seeds {seeds}")
            grouped[system][arm].sort(key=lambda row: int(row["seed"]))
    return grouped


def load_identity() -> list[dict[str, str]]:
    path = ROOT / "bench" / "ase-nonlj" / "identity.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_sys = {row["system"]: row for row in rows}
    if tuple(by_sys) != SYSTEMS:
        raise SystemExit(f"identity systems {tuple(by_sys)}")
    for row in rows:
        if float(row["abs_energy_diff"]) > 1.0e-8 or float(row["max_force_diff"]) > 1.0e-8:
            raise SystemExit(f"identity check exceeded 1e-8: {row}")
    return rows


def main() -> None:
    grouped = load_rows()
    identity = load_identity()
    revisions = sorted({row["code_revision"] for s in SYSTEMS for a in ARMS for row in grouped[s][a]})
    summary_lines = [
        "system,arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,median_outside_s,"
        "fisher_p_bh,mann_whitney_U_bh,mann_whitney_p_bh,fisher_p_mh,mann_whitney_U_mh,mann_whitney_p_mh,"
        "wall_over_bh,wall_over_mh,wall_stops,wall_cap_s",
    ]
    readme = [
        "Copyright (c) anneal developers",
        "",
        "# Non-LJ ASE head-to-head (Cambridge Morse clusters)",
        "",
        "Four Cambridge Morse systems across two sizes and two ranges: M38 and M55 at rho = 6 and rho = 14.",
        "Seeds 2000-2011, host bench, serial one core (`taskset -c 0`).",
        f"code_revision {', '.join(revisions)}.",
        "Potential: untruncated Morse (`epsilon = 1`, `r0 = 1`, `rho0 = rho`).",
        "Anneal evaluates `PairPotential::morse` (`rgpot_pair::cambridge_morse`).",
        "ASE evaluates `ase.calculators.morse.MorsePotential` with `rcut1 = 1e6`, `rcut2 = 2e6`.",
        "Anneal `for_cluster` uses `Config::with_scales` at `PairKind::length_scale()` plus `for_cluster` Bayes screening.",
        "Anneal `recommended` uses `Config::recommended_with_scales` at `PairKind::length_scale()`.",
        "ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults (`fmax` 0.1 and 0.05).",
        "Hits reach `reference + 0.001`. Misses are censored at the paper budget (400,000 at 38 atoms, 1,000,000 at 55).",
        "The wall cap for `recommended`, `ase_bh` and `ase_mh` is three times the slowest `for_cluster` wall per system.",
        "",
        "## Cross-code potential agreement (seed 2000 start)",
        "",
        "| system | N | rho | E anneal | E ASE | |dE| | max |dF| |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in identity:
        n, rho, _, _, _ = SPEC[row["system"]]
        readme.append(
            f"| {row['system']} | {n} | {rho:g} | {float(row['energy_anneal']):.10f} | "
            f"{float(row['energy_ase']):.10f} | {float(row['abs_energy_diff']):.3e} | "
            f"{float(row['max_force_diff']):.3e} |"
        )

    readme.extend(
        [
            "",
            "## Summary",
            "",
            "| system | arm | hits | median best | gap | median wall s | median inside s | median outside s | "
            "stops (hit/budget/wall) | cap s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    vs_lines = [
        "",
        "## Fisher and Mann-Whitney against ASE arms",
        "",
        "| system | comparison | hits | Fisher p | U | Mann-Whitney p | wall ratio | inverse ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for system in SYSTEMS:
        n, rho, budget, reference, label = SPEC[system]
        base = grouped[system]["for_cluster"]
        caps = sorted({row["cap_s"] for row in grouped[system]["recommended"] if row["cap_s"]})
        if len(caps) != 1:
            raise SystemExit(f"{system} cap not unique: {caps}")
        cap = caps[0]
        for arm in ARMS:
            group = grouped[system][arm]
            hits = sum(row["hit"] == "1" for row in group)
            best = median([float(row["best"]) for row in group])
            wall = median([float(row["wall_s"]) for row in group])
            inside = median([float(row["inside_s"]) for row in group])
            outside = median([float(row["outside_s"]) for row in group])
            hit_stops = sum(row["stop"] == "hit" for row in group)
            budget_stops = sum(row["stop"] == "budget" for row in group)
            wall_stops = sum(row["stop"] == "wall" for row in group)
            f_bh, u_bh, mw_bh = against(group, grouped[system]["ase_bh"])
            f_mh, u_mh, mw_mh = against(group, grouped[system]["ase_mh"])
            w_bh, inv_bh = paired_ratio(group, grouped[system]["ase_bh"])
            w_mh, inv_mh = paired_ratio(group, grouped[system]["ase_mh"])
            arm_cap = "" if arm == "for_cluster" else cap
            summary_lines.append(
                f"{system},{arm},{hits},12,{best:.8f},{best - reference:.8f},"
                f"{wall:.6f},{inside:.6f},{outside:.6f},"
                f"{g6(f_bh)},{g6(u_bh)},{g6(mw_bh)},{g6(f_mh)},{g6(u_mh)},{g6(mw_mh)},"
                f"{w_bh:.6f},{w_mh:.6f},{wall_stops},{arm_cap}"
            )
            readme.append(
                f"| {system} ({label}, {reference:.6f}) | {LABEL[arm]} | {hits}/12 | {best:.8f} | "
                f"{best - reference:.8f} | {wall:.6f} | {inside:.6f} | {outside:.6f} | "
                f"{hit_stops}/{budget_stops}/{wall_stops} | {arm_cap} |"
            )
            if arm in ("for_cluster", "recommended"):
                for other_arm, (fp, u_val, mwp, wr, ir) in (
                    ("ase_bh", (f_bh, u_bh, mw_bh, w_bh, inv_bh)),
                    ("ase_mh", (f_mh, u_mh, mw_mh, w_mh, inv_mh)),
                ):
                    other_hits = sum(row["hit"] == "1" for row in grouped[system][other_arm])
                    vs_lines.append(
                        f"| {system} | {LABEL[arm]} vs {LABEL[other_arm]} | {hits}/12 vs {other_hits}/12 | "
                        f"{g6(fp)} | {g6(u_val)} | {g6(mwp)} | {wr:.6f} | {ir:.6f} |"
                    )

    readme.extend(vs_lines)
    readme.extend(
        [
            "",
            "Medians are Hyndman-Fan type 7. Wall ratios are paired by seed.",
            "Fisher is two-sided exact. Mann-Whitney is two-sided asymptotic with",
            "the tie correction and the continuity correction; misses are censored at the call budget.",
            "Inside is time in the potential. Outside is the rest of the wall.",
            "",
        ]
    )
    out = ROOT / "bench" / "ase-nonlj"
    (out / "summary.csv").write_text("\n".join(summary_lines) + "\n")
    (out / "README.md").write_text("\n".join(readme))


if __name__ == "__main__":
    main()
