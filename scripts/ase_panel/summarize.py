#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Reprint the ASE panel from its committed rows.

Wall-censored misses do not enter a rank test. Does not start a search.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SEEDS = tuple(range(2000, 2012))
ARMS = ("anneal", "recommended", "ase_bh", "ase_mh")
ASE_ARMS = ("ase_bh", "ase_mh")
LABELS = {
    "anneal": "anneal",
    "recommended": "recommended",
    "ase_bh": "basin hopping",
    "ase_mh": "minima hopping",
}
SYSTEMS = {
    "lj38": (38, 400_000, -173.928427, "LJ38"),
    "lj55": (55, 1_000_000, -279.248470, "LJ55"),
    "lj98": (98, 4_000_000, -543.665361, "LJ98"),
    "lj104": (104, 4_000_000, -582.086642, "LJ104"),
    "lj150": (150, 4_000_000, -893.310258, "LJ150"),
    "au55": (55, 1_000_000, -191.03085280, "Au55"),
}
TEX_LABELS = {
    "lj38": "LJ\\(_{38}\\)",
    "lj55": "LJ\\(_{55}\\)",
    "lj98": "LJ\\(_{98}\\)",
    "lj104": "LJ\\(_{104}\\)",
    "lj150": "LJ\\(_{150}\\)",
    "au55": "Au\\(_{55}\\)",
}
DOCUMENT = ROOT / "paper" / "jctc" / "si.tex"
TEX_MARKS = ("% ase-panel table: start", "% ase-panel table: end")
ASE_VERSION = "3.29.0"
HIT_TOL = DEFAULT_HIT_TOL
HEADER = (
    "arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,"
    "median_outside_s,median_over_anneal,median_anneal_over,fisher_p,"
    "mann_whitney_U,mann_whitney_p,wall_stops,wall_cap_s,"
    "fisher_p_bh,mann_whitney_U_bh,mann_whitney_p_bh,"
    "fisher_p_mh,mann_whitney_U_mh,mann_whitney_p_mh"
)
ROW_FIELDS = (
    "system", "arm", "seed", "budget", "hit", "evals_to_hit", "charged", "best",
    "wall_s", "inside_s", "outside_s", "code_revision", "host", "stop", "cap_s",
)


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


def reference_text(value: float) -> str:
    text = f"{value:.8f}"
    return text[:-2] if text.endswith("00") else text


def calls(row: dict[str, str]) -> float:
    if row["evals_to_hit"]:
        return float(row["evals_to_hit"])
    return float(row["budget"])


def wall_censored(rows: list[dict[str, str]]) -> bool:
    return any(row["stop"] == "wall" and row["hit"] == "0" for row in rows)


def load(system: str) -> dict[str, list[dict[str, str]]]:
    directory = ROOT / "bench" / f"ase-{system}"
    paths = [directory / "rows.csv"]
    if system == "au55":
        paths.append(directory / "rows-for-cluster.csv")
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ROW_FIELDS:
                raise SystemExit(f"{path}: unexpected columns {reader.fieldnames}")
            for row in reader:
                if system != "au55" or (
                    (path.name == "rows.csv" and row["arm"] == "recommended")
                    or (path.name == "rows-for-cluster.csv" and row["arm"] != "recommended")
                ):
                    rows.append(row)
    _, budget, reference, _ = SYSTEMS[system]
    grouped = {arm: [] for arm in ARMS}
    by_arm: dict[str, list[tuple[int, dict[str, str]]]] = {arm: [] for arm in ARMS}
    for row in rows:
        if (
            row["system"] != system
            or row["arm"] not in grouped
            or row["budget"] != str(budget)
            or row["host"] != "bench"
        ):
            raise SystemExit(f"unexpected row {row}")
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
            if not energy_hits(float(row["best"]), reference, HIT_TOL + 5.0e-9):
                raise SystemExit(f"hit above threshold {row}")
        elif row["stop"] == "budget":
            if charged != budget:
                raise SystemExit(f"invalid budget stop {row}")
        elif row["stop"] == "wall":
            if charged >= budget or not row["cap_s"]:
                raise SystemExit(f"invalid wall stop {row}")
        else:
            raise SystemExit(f"invalid miss stop {row}")
        by_arm[row["arm"]].append((len(by_arm[row["arm"]]), row))

    def choose_campaign(arm: str) -> list[dict[str, str]]:
        # Append-only rows can hold multiple campaigns. Choose the latest full
        # 12-seed campaign of this arm, grouped by code revision and cap.
        groups: dict[tuple[str, str], tuple[int, dict[int, dict[str, str]]]] = {}
        for index, row in by_arm[arm]:
            key = (row["code_revision"], row["cap_s"])
            last, seeds = groups.get(key, (-1, {}))
            seeds[int(row["seed"])] = row
            groups[key] = (index, seeds)
        full = [
            (last, seeds)
            for last, seeds in groups.values()
            if tuple(sorted(seeds)) == SEEDS
        ]
        if full:
            _last, seeds = max(full, key=lambda item: item[0])
            return [seeds[seed] for seed in SEEDS]
        # Older tables (for example LJ55 ASE-BH) can be a complete campaign
        # split across multiple revisions. Fall back to the latest row of each
        # seed in append order.
        latest_by_seed: dict[int, dict[str, str]] = {}
        for _index, row in by_arm[arm]:
            latest_by_seed[int(row["seed"])] = row
        if tuple(sorted(latest_by_seed)) != SEEDS:
            raise SystemExit(f"{system} {arm}: no complete 2000-2011 campaign")
        seeds = latest_by_seed
        return [seeds[seed] for seed in SEEDS]

    for arm in ARMS:
        grouped[arm] = choose_campaign(arm)
    for arm in ARMS:
        seeds = tuple(sorted(int(row["seed"]) for row in grouped[arm]))
        if seeds != SEEDS:
            raise SystemExit(f"{arm} seeds {seeds}")
        grouped[arm].sort(key=lambda row: int(row["seed"]))
    return grouped


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


def comparison(
    left: list[dict[str, str]], right: list[dict[str, str]]
) -> tuple[str, str, str]:
    left_hits = sum(row["hit"] == "1" for row in left)
    right_hits = sum(row["hit"] == "1" for row in right)
    fisher_text = f"{fisher(left_hits, right_hits):.6g}"
    if wall_censored(left) or wall_censored(right):
        return fisher_text, "", ""
    statistic, probability = mann_whitney(
        [calls(row) for row in left], [calls(row) for row in right]
    )
    return fisher_text, f"{statistic:.6g}", f"{probability:.6g}"


def render(system: str) -> tuple[str, str]:
    _, budget, reference, title = SYSTEMS[system]
    grouped = load(system)
    anneal = grouped["anneal"]
    lines = [HEADER]
    reports = []
    base_wall = median([float(row["wall_s"]) for row in anneal])
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
            over = back = 1.0
            fisher_p = "1"
            u_text, mw_text = "72", "1"
            cap = ""
        else:
            over = wall / base_wall
            back = base_wall / wall
            fisher_p, u_text, mw_text = comparison(group, anneal)
        versus = {other: comparison(group, grouped[other]) for other in ASE_ARMS}
        lines.append(
            f"{arm},{hits},12,{best:.8f},{best - reference:.8f},{wall:.6f},{inside:.6f},{outside:.6f},"
            f"{over:.6f},{back:.6f},{fisher_p},{u_text},{mw_text},{wall_stops},{cap},"
            f"{','.join(versus['ase_bh'])},{','.join(versus['ase_mh'])}"
        )
        reports.append(
            (arm, hits, best, wall, inside, outside, over, back, fisher_p, u_text, mw_text,
             wall_stops, cap, min(int(row["charged"]) for row in group),
             max(int(row["charged"]) for row in group), versus)
        )

    revisions = list(dict.fromkeys(row["code_revision"] for arm in ARMS for row in grouped[arm]))
    cap_details = []
    for arm in ARMS:
        arm_caps = sorted({row["cap_s"] for row in grouped[arm] if row["cap_s"]})
        if arm_caps:
            cap_details.append(f"{LABELS[arm]} {', '.join(arm_caps)} s")
    report = [
        "Copyright (c) anneal developers",
        "",
        f"{title}, seeds 2000-2011, maximum budget {budget:,}, host bench, one core.",
        f"The reference minimum is {reference:.8f}; a hit is at most {HIT_TOL:g} above it.",
        "Anneal uses the cluster preset. Recommended uses Config::recommended.",
        "ASE basin hopping and minima hopping use",
        "the ASE 3.29 constructor defaults.",
        "Rows carry code_revision " + ", ".join(revisions) + ".",
    ]
    if system == "au55":
        report.extend([
            "The anneal, basin-hopping and minima-hopping rows of `rows.csv` at 33bed75f are kept",
            "and superseded by `rows-for-cluster.csv`; `superseded.csv` lists each of them.",
            "This report reads the recommended rows of `rows.csv` and every row of `rows-for-cluster.csv`.",
        ])
    if cap_details:
        report.extend([
            f"The recorded wall caps are {'; '.join(cap_details)}. A wall-stopped row did not receive",
            "the full call budget, so its call count is right-censored at `charged`.",
        ])
    report.extend([
        "",
        "| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | "
        f"gap to {reference_text(reference)} | median wall s | over anneal | anneal over | "
        "median inside s | median outside s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    stop_notes = []
    for (
        arm, hits, best, wall, inside, outside, over, back, fisher_p, u_text, mw_text,
        wall_stops, cap, low_charged, high_charged, versus,
    ) in reports:
        report.append(
            f"| {LABELS[arm]} | {hits}/12 | {fisher_p} | {u_text or 'not estimated'} | "
            f"{mw_text or 'not estimated'} | {best:.8f} | {best - reference:.8f} | "
            f"{wall:.6f} | {over:.6f} | {back:.6f} | {inside:.6f} | {outside:.6f} |"
        )
        if wall_stops:
            stop_notes.append(
                f"{LABELS[arm].capitalize()} stopped at the wall cap on {wall_stops}/12 seeds; "
                f"those rows charged {low_charged:,}-{high_charged:,} calls."
            )
    recommended = next(item for item in reports if item[0] == "recommended")
    recommended_hits = recommended[1]
    recommended_versus = recommended[-1]
    report.extend(["", *stop_notes, "",
        "Fisher and Mann-Whitney below compare recommended with each ASE arm.",
        "",
        "| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | "
        "median wall ratio | inverse ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for other in ASE_ARMS:
        other_report = next(item for item in reports if item[0] == other)
        other_hits = other_report[1]
        fisher_text, u_text, mw_text = recommended_versus[other]
        wall_ratio = recommended[3] / other_report[3]
        inverse_ratio = other_report[3] / recommended[3]
        report.append(
            f"| recommended against {LABELS[other]} | {recommended_hits}/12 vs {other_hits}/12 | "
            f"{fisher_text} | {u_text or 'not estimated'} | {mw_text or 'not estimated'} | "
            f"{wall_ratio:.6f} | {inverse_ratio:.6f} |"
        )
    report.extend(["",
        "Fisher is the two-sided exact test on hits under the recorded stopping rule.",
        "Mann-Whitney is two-sided with tie and continuity corrections. It is not",
        "estimated for an arm with wall-censored misses; imputing those misses at the",
        "larger paper budget would manufacture observations that were never run.",
        "Medians are type 7. Wall ratios are ratios of arm medians. The Rust and",
        "NumPy random generators do not make a shared starting geometry from the",
        "same numeric seed, so the report does not treat seed labels as pairs.",
        "",
    ])
    return "\n".join(lines) + "\n", "\n".join(report)


AGGREGATE_HEADER = (
    "system,arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,"
    "median_outside_s,fisher_p_bh,mann_whitney_U_bh,mann_whitney_p_bh,"
    "fisher_p_mh,mann_whitney_U_mh,mann_whitney_p_mh,wall_cap_s,code_revision,host"
)


def aggregate_rows(system: str, summary: str) -> list[str]:
    fields = (
        "arm", "hits", "n", "median_best", "gap_to_reference", "median_wall_s",
        "median_inside_s", "median_outside_s", "fisher_p_bh", "mann_whitney_U_bh",
        "mann_whitney_p_bh", "fisher_p_mh", "mann_whitney_U_mh",
        "mann_whitney_p_mh", "wall_cap_s",
    )
    revisions = {
        arm: {row["code_revision"] for row in rows} for arm, rows in load(system).items()
    }
    out = []
    for row in csv.DictReader(summary.splitlines()):
        found = ";".join(sorted(revisions[row["arm"]]))
        out.append(",".join([system, *(row[field] for field in fields), found, "bench"]))
    return out


def show_p(p: float) -> str:
    """Two significant figures; 1 only when p is exactly 1."""
    if p >= 1.0:
        return "1"
    if p >= 0.01:
        return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def show_budget(budget: int) -> str:
    mantissa, exponent = f"{budget:.0e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def show_wall(value: float) -> str:
    return f"{value:.3f}" if value < 100.0 else f"{value:.1f}"


def show_cap(cap: str) -> str:
    value = float(cap)
    return f"{value:.3f}" if value < 100.0 else f"{value:.1f}"


def tex_table() -> str:
    """Table tab:ase: recommended against both ASE arms, Fisher on hits only."""
    body = []
    revisions = set()
    for system in SYSTEMS:
        _, budget, _, _ = SYSTEMS[system]
        grouped = load(system)
        revisions |= {row["code_revision"] for row in grouped["recommended"]}
        if any(row["stop"] == "wall" for row in grouped["recommended"]):
            raise SystemExit(f"{system}: a recommended row stopped at the wall cap")
        caps = {row["cap_s"] for arm in ASE_ARMS for row in grouped[arm]}
        if len(caps) != 1:
            raise SystemExit(f"{system}: the ASE wall cap is not unique: {sorted(caps)}")
        slowest = max(float(row["wall_s"]) for row in grouped["anneal"])
        if abs(float(next(iter(caps))) - 3.0 * slowest) > 1.0e-6:
            raise SystemExit(f"{system}: the ASE wall cap is not three times the slowest anneal wall")

        def hits(arm: str) -> int:
            return sum(row["hit"] == "1" for row in grouped[arm])

        def stops(arm: str) -> int:
            return sum(row["stop"] == "wall" for row in grouped[arm])

        body.append(" & ".join([
            TEX_LABELS[system], show_budget(budget), show_cap(caps.pop()),
            f"{hits('recommended')}/12", f"{hits('ase_bh')}/12", f"{hits('ase_mh')}/12",
            str(stops("ase_bh")), str(stops("ase_mh")),
            show_wall(median([float(row["wall_s"]) for row in grouped["recommended"]])),
            show_wall(median([float(row["wall_s"]) for row in grouped["ase_bh"]])),
            show_wall(median([float(row["wall_s"]) for row in grouped["ase_mh"]])),
            show_p(fisher(hits("recommended"), hits("ase_bh"))),
            show_p(fisher(hits("recommended"), hits("ase_mh"))),
        ]) + " \\\\")
    if not revisions:
        raise SystemExit("no recommended revisions found")
    rev_text = ", ".join(sorted(revisions))
    return "\n".join([
        "\\begin{table}",
        "\\caption{\\raggedright This work against the basin hopping (BH) and minima hopping",
        f"(MH) of ASE {ASE_VERSION} at its constructor defaults.",
        f"This work is anneal \\texttt{{{rev_text}}}.",
        "Seeds 2000--2011, one core, budget in force calls.",
        "Cap is the wall in seconds at which a BH or MH run stops, three times",
        "the slowest wall of the cluster-preset arm of the panel.",
        "Stops counts the BH and MH runs that ended at the cap; those runs",
        "are misses with fewer calls than the budget.",
        "No run of this work ended at a cap.",
        "Wall is the median over the 12 runs of the seconds each run took;",
        "a BH or MH run that ended at the cap is counted at its wall.",
        "\\(p_\\mathrm{BH}\\) and \\(p_\\mathrm{MH}\\) are the two-sided Fisher",
        "exact test on hits against each ASE arm.}",
        "\\label{tab:ase}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrrrrrrr@{}}",
        "\\toprule",
        " & & & \\multicolumn{3}{c}{Hits} & \\multicolumn{2}{c}{Stops} & "
        "\\multicolumn{3}{c}{Wall (s)} & & \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}",
        "Cluster & Budget & Cap & This work & BH & MH & BH & MH & This work & BH & MH & "
        "\\(p_\\mathrm{BH}\\) & \\(p_\\mathrm{MH}\\) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def splice(text: str, block: str) -> str:
    start, end = TEX_MARKS
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{DOCUMENT} needs one '{start}' and one '{end}'")
    head, rest = text.split(start, 1)
    _old, tail = rest.split(end, 1)
    return f"{head}{start}\n{block}\n{end}{tail}"


def main() -> None:
    args = sys.argv[1:]
    flags = [arg for arg in args if arg.startswith("--") and arg != "--all"]
    selection = [arg for arg in args if arg not in flags]
    if len(selection) != 1 or any(flag not in ("--check", "--tex") for flag in flags) or len(set(flags)) != len(flags):
        raise SystemExit("usage: summarize.py SYSTEM|--all [--check] [--tex]")
    selected = list(SYSTEMS) if selection[0] == "--all" else [selection[0]]
    if any(system not in SYSTEMS for system in selected):
        raise SystemExit(f"unknown system {selection[0]}")
    if "--tex" in flags and selected != list(SYSTEMS):
        raise SystemExit("--tex needs --all")
    outputs = {}
    aggregate = [AGGREGATE_HEADER]
    for system in selected:
        summary, report = render(system)
        root = Path(os.environ["ASE_PANEL_OUT"]) if os.environ.get("ASE_PANEL_OUT") else ROOT / "bench"
        directory = root / f"ase-{system}"
        outputs[directory / "summary.csv"] = summary
        outputs[directory / "README.md"] = report
        aggregate.extend(aggregate_rows(system, summary))
    if selected == list(SYSTEMS):
        recommended = root / "ase-recommended" / "summary.csv"
        outputs[recommended] = "\n".join(aggregate) + "\n"
    if "--tex" in flags:
        document = Path(os.environ["ASE_PANEL_TEX"]) if os.environ.get("ASE_PANEL_TEX") else DOCUMENT
        source = document.read_text() if document.exists() else DOCUMENT.read_text()
        outputs[document] = splice(source, tex_table())
    if "--check" in flags:
        def generated(text: str) -> str:
            # A panel mark names a commit this repository does not hold. It is not part of the generated block.
            return re.sub(r"\\StalePanel\{[^{}]*\}", "", text)

        stale = [
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            for path, text in outputs.items()
            if not path.exists() or generated(path.read_text()) != generated(text)
        ]
        if stale:
            raise SystemExit("stale generated report: " + ", ".join(stale))
    else:
        present = [str(path) for path in outputs if path.exists()]
        if present:
            raise SystemExit("refusing to overwrite " + ", ".join(present))
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


if __name__ == "__main__":
    main()
