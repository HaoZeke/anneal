#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Generate the non-LJ (Morse) ASE panel report from its committed rows.

Wall-censored misses do not enter a rank test. Does not start a search.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(2000, 2012))
SYSTEMS = (
    "m38r3",
    "m38r6",
    "m38r10",
    "m38r14",
    "m55r3",
    "m55r6",
    "m55r10",
    "m55r14",
    "m75r6",
    "m75r14",
    "m80r6",
    "m80r14",
)
ARMS = ("for_cluster", "recommended", "ase_bh", "ase_mh")
ASE_ARMS = ("ase_bh", "ase_mh")
LABEL = {
    "for_cluster": "for_cluster",
    "recommended": "recommended",
    "ase_bh": "basin hopping",
    "ase_mh": "minima hopping",
}
SPEC = {
    "m38r3": (38, 3.0, "400000", -249.159174, "38A"),
    "m38r6": (38, 6.0, "400000", -157.477108, "38E"),
    "m38r10": (38, 10.0, "400000", -145.849817, "38D"),
    "m38r14": (38, 14.0, "400000", -144.321054, "38D"),
    "m55r3": (55, 3.0, "1000000", -417.918562, "55A"),
    "m55r6": (55, 6.0, "1000000", -250.286609, "55B"),
    "m55r10": (55, 10.0, "1000000", -225.814286, "55B"),
    "m55r14": (55, 14.0, "1000000", -220.646208, "55C"),
    "m75r6": (75, 6.0, "4000000", -351.472365, "75C"),
    "m75r14": (75, 14.0, "4000000", -318.407330, "75C"),
    "m80r6": (80, 6.0, "4000000", -378.333471, "80B"),
    "m80r14": (80, 14.0, "4000000", -340.811371, "80G"),
}
ROW_FIELDS = (
    "system", "arm", "seed", "budget", "hit", "evals_to_hit", "charged", "best",
    "wall_s", "inside_s", "outside_s", "code_revision", "host", "stop", "cap_s",
)
HIT_TOL = 0.001


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
    row = 12
    col = hits + base
    total = 24

    def probability(value: int) -> float:
        return math.comb(row, value) * math.comb(row, col - value) / math.comb(total, col)

    observed = probability(hits)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(max(0, col - row), min(row, col) + 1)
            if probability(value) <= observed * (1.0 + 1.0e-12)
        ),
    )


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    n_left, n_right = len(left), len(right)
    pooled = sorted([(value, 0) for value in left] + [(value, 1) for value in right])
    ranks = [0.0] * len(pooled)
    tie_term = 0
    index = 0
    while index < len(pooled):
        end = index + 1
        while end < len(pooled) and pooled[end][0] == pooled[index][0]:
            end += 1
        average = 0.5 * (index + 1 + end)
        ranks[index:end] = [average] * (end - index)
        width = end - index
        tie_term += width**3 - width
        index = end
    rank_left = sum(rank for rank, (_, arm) in zip(ranks, pooled) if arm == 0)
    statistic = rank_left - n_left * (n_left + 1) / 2.0
    total = n_left + n_right
    variance = n_left * n_right / 12.0 * (
        total + 1 - tie_term / (total * (total - 1))
    )
    if variance <= 0.0:
        return statistic, 1.0
    z = max(0.0, (abs(statistic - n_left * n_right / 2.0) - 0.5) / math.sqrt(variance))
    return statistic, min(1.0, math.erfc(z / math.sqrt(2.0)))


def wall_ratio(left: list[dict[str, str]], right: list[dict[str, str]]) -> tuple[float, float]:
    left_wall = median([float(row["wall_s"]) for row in left])
    right_wall = median([float(row["wall_s"]) for row in right])
    return left_wall / right_wall, right_wall / left_wall


def wall_censored(rows: list[dict[str, str]]) -> bool:
    return any(row["stop"] == "wall" and row["hit"] == "0" for row in rows)


def against(group: list[dict[str, str]], other: list[dict[str, str]]) -> tuple[str, str, str]:
    hits = sum(row["hit"] == "1" for row in group)
    base = sum(row["hit"] == "1" for row in other)
    fisher_text = g6(fisher(hits, base))
    if wall_censored(group) or wall_censored(other):
        return fisher_text, "", ""
    u_stat, mw_p = mann_whitney([calls(row) for row in group], [calls(row) for row in other])
    return fisher_text, g6(u_stat), g6(mw_p)


def load_rows() -> dict[str, dict[str, list[dict[str, str]]]]:
    path = ROOT / "bench" / "ase-nonlj" / "rows.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_FIELDS:
            raise SystemExit(f"{path}: unexpected columns {reader.fieldnames}")
        rows = list(reader)
    grouped = {system: {arm: [] for arm in ARMS} for system in SYSTEMS}
    seen = set()
    for row in rows:
        system = row["system"]
        arm = row["arm"]
        if system not in SPEC or arm not in ARMS:
            raise SystemExit(f"unexpected row {row}")
        if row["budget"] != SPEC[system][2] or row["host"] != "bench":
            raise SystemExit(f"unexpected budget or host {row}")
        key = (system, arm, row["seed"])
        if key in seen:
            raise SystemExit(f"duplicate row {key}")
        seen.add(key)
        budget = int(row["budget"])
        charged = int(row["charged"])
        if not 0 < charged <= budget:
            raise SystemExit(f"invalid charged count {row}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["code_revision"]):
            raise SystemExit(f"invalid revision {row}")
        hit = row["hit"] == "1"
        if row["hit"] not in {"0", "1"} or hit != bool(row["evals_to_hit"]):
            raise SystemExit(f"hit/calls mismatch {row}")
        if hit:
            evals_to_hit = int(row["evals_to_hit"])
            if row["stop"] != "hit" or not 0 < evals_to_hit <= charged:
                raise SystemExit(f"invalid hit row {row}")
            if float(row["best"]) > SPEC[system][3] + HIT_TOL + 5.0e-9:
                raise SystemExit(f"hit above threshold {row}")
        elif row["stop"] == "budget":
            if charged != budget:
                raise SystemExit(f"invalid budget stop {row}")
        elif row["stop"] == "wall":
            if charged >= budget or not row["cap_s"]:
                raise SystemExit(f"invalid wall stop {row}")
        else:
            raise SystemExit(f"invalid miss stop {row}")
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
    if set(by_sys) != set(SYSTEMS):
        raise SystemExit(f"identity systems {tuple(by_sys)}")
    for row in rows:
        if float(row["abs_energy_diff"]) > 1.0e-8 or float(row["max_force_diff"]) > 1.0e-8:
            raise SystemExit(f"identity check exceeded 1e-8: {row}")
    return [by_sys[system] for system in SYSTEMS]


def main() -> None:
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--check"):
        raise SystemExit("usage: summarize.py [--check]")
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
        "Twelve Cambridge Morse systems: M38 and M55 at rho = 3, 6, 10 and 14, plus M75 and M80 at rho = 6 and 14.",
        "Seeds 2000-2011, host bench, serial one core (`taskset -c 0`).",
        f"code_revision {', '.join(revisions)}.",
        "Potential: untruncated Morse (`epsilon = 1`, `r0 = 1`, `rho0 = rho`).",
        "Anneal evaluates `PairPotential::morse` (`rgpot_pair::cambridge_morse`).",
        "ASE evaluates `ase.calculators.morse.MorsePotential` with `rcut1 = 1e6`, `rcut2 = 2e6`.",
        "Anneal `for_cluster` uses `Config::for_cluster_with_scales` at `PairKind::length_scale()` (earlier M38/M55 rho 6 and 14 rows retain their committed runs).",
        "Anneal `recommended` uses `Config::recommended_with_scales` at `PairKind::length_scale()`.",
        "ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults (`fmax` 0.1 and 0.05).",
        "Hits reach `reference + 0.001`. Misses are censored at the paper budget (400,000 at 38 atoms, 1,000,000 at 55, 4,000,000 at 75 and 80).",
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
            w_bh, inv_bh = wall_ratio(group, grouped[system]["ase_bh"])
            w_mh, inv_mh = wall_ratio(group, grouped[system]["ase_mh"])
            arm_cap = "" if arm == "for_cluster" else cap
            summary_lines.append(
                f"{system},{arm},{hits},12,{best:.8f},{best - reference:.8f},"
                f"{wall:.6f},{inside:.6f},{outside:.6f},"
                f"{f_bh},{u_bh},{mw_bh},{f_mh},{u_mh},{mw_mh},"
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
                        f"{fp} | {u_val or 'not estimated'} | {mwp or 'not estimated'} | "
                        f"{wr:.6f} | {ir:.6f} |"
                    )

    readme.extend(vs_lines)
    readme.extend(
        [
            "",
            "Medians are Hyndman-Fan type 7. Wall ratios are ratios of arm medians.",
            "The Rust and NumPy random generators do not make a shared starting",
            "geometry from the same numeric seed, so seed labels are not treated as pairs.",
            "Fisher is two-sided exact. Mann-Whitney is two-sided asymptotic with",
            "the tie correction and the continuity correction. It is not estimated",
            "when either arm has wall-censored misses; assigning those misses the",
            "larger paper budget would manufacture observations that were never run.",
            "Inside is time in the potential. Outside is the rest of the wall.",
            "",
        ]
    )
    out = ROOT / "bench" / "ase-nonlj"
    outputs = {
        out / "summary.csv": "\n".join(summary_lines) + "\n",
        out / "README.md": "\n".join(readme),
    }
    if len(sys.argv) == 2:
        stale = [
            str(path.relative_to(ROOT))
            for path, text in outputs.items()
            if path.read_text() != text
        ]
        if stale:
            raise SystemExit("stale generated report: " + ", ".join(stale))
    else:
        for path, text in outputs.items():
            path.write_text(text)


if __name__ == "__main__":
    main()
