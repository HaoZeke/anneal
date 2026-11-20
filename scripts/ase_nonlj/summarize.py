#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Generate the non-LJ (Morse) ASE panel report from its committed rows.

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
from machine_provenance import csv_fields  # noqa: E402

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
ROW_FIELDS = (
    "system", "arm", "seed", "budget", "hit", "evals_to_hit", "charged", "best",
    "wall_s", "inside_s", "outside_s", "code_revision", "host", "stop", "cap_s",
    "harness_revision",
)
CURRENT_ROW_FIELDS = (*ROW_FIELDS, "machine_id", "cpu_model")
HIT_TOL = DEFAULT_HIT_TOL
TEX_LABELS = {
    "m38r6": "M\\(_{38}\\), \\(\\rho=6\\)",
    "m38r14": "M\\(_{38}\\), \\(\\rho=14\\)",
    "m55r6": "M\\(_{55}\\), \\(\\rho=6\\)",
    "m55r14": "M\\(_{55}\\), \\(\\rho=14\\)",
}
DOCUMENT = ROOT / "paper" / "jctc" / "si.tex"
TEX_MARKS = ("% ase-nonlj table: start", "% ase-nonlj table: end")
ASE_VERSION = "3.29.0"
REVISION = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"


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
    path = ROOT / "bench" / "ase-nonlj" / "rerun-20261119" / "rows.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) not in (ROW_FIELDS, CURRENT_ROW_FIELDS):
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
        key = (row["code_revision"], system, arm, row["seed"])
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
            if not energy_hits(float(row["best"]), SPEC[system][3], HIT_TOL + 5.0e-9):
                raise SystemExit(f"hit above threshold {row}")
        elif row["stop"] == "budget":
            if charged != budget:
                raise SystemExit(f"invalid budget stop {row}")
        elif row["stop"] == "wall":
            if charged >= budget or not row["cap_s"]:
                raise SystemExit(f"invalid wall stop {row}")
        else:
            raise SystemExit(f"invalid miss stop {row}")
        if row["code_revision"] == REVISION:
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


def show_p(p: float) -> str:
    """Two significant figures; 1 only when p is exactly 1."""
    if p >= 1.0:
        return "1"
    if p >= 0.01:
        return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def show_budget(budget: str) -> str:
    mantissa, exponent = f"{int(budget):.0e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def show_wall(value: float) -> str:
    return f"{value:.3f}" if value < 100.0 else f"{value:.1f}"


def tex_table(grouped: dict[str, dict[str, list[dict[str, str]]]]) -> str:
    """Table tab:ase-morse: recommended against both ASE arms, Fisher on hits only."""
    body = []
    revisions = {row["code_revision"] for s in SYSTEMS for row in grouped[s]["recommended"]}
    if len(revisions) != 1:
        raise SystemExit(f"recommended rows carry more than one revision: {sorted(revisions)}")
    for system in SYSTEMS:
        _, _, budget, _, _ = SPEC[system]
        arms = grouped[system]
        if any(row["stop"] == "wall" for row in arms["recommended"]):
            raise SystemExit(f"{system}: a recommended row stopped at the wall cap")
        if not all(row["stop"] == "wall" for arm in ASE_ARMS for row in arms[arm]):
            raise SystemExit(f"{system}: an ASE row did not stop at the wall cap")
        caps = {row["cap_s"] for arm in ASE_ARMS for row in arms[arm]}
        slowest = max(float(row["wall_s"]) for row in arms["for_cluster"])
        if len(caps) != 1 or abs(float(next(iter(caps))) - 3.0 * slowest) > 1.0e-6:
            raise SystemExit(f"{system}: the ASE wall cap is not three times the slowest for_cluster wall")

        def hits(arm: str) -> int:
            return sum(row["hit"] == "1" for row in arms[arm])

        body.append(" & ".join([
            TEX_LABELS[system], show_budget(budget), f"{float(caps.pop()):.3f}",
            f"{hits('recommended')}/12", f"{hits('ase_bh')}/12", f"{hits('ase_mh')}/12",
            show_wall(median([float(row["wall_s"]) for row in arms["recommended"]])),
            show_wall(median([float(row["wall_s"]) for row in arms["ase_bh"]])),
            show_wall(median([float(row["wall_s"]) for row in arms["ase_mh"]])),
            show_p(fisher(hits("recommended"), hits("ase_bh"))),
            show_p(fisher(hits("recommended"), hits("ase_mh"))),
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This work against the basin hopping (BH) and minima hopping",
        f"(MH) of ASE {ASE_VERSION} on Morse clusters.",
        f"This work is anneal \\texttt{{{revisions.pop()[:8]}}} at the length",
        "scale of the Morse pair.",
        "Seeds 2000--2011, one core, budget in force calls.",
        "Cap is the wall in seconds at which a BH or MH run stops, three times",
        "the slowest wall of the cluster-preset arm of the panel.",
        "Every BH and MH run ended at the cap, as a miss with fewer calls than",
        "the budget, and no run of this work did.",
        "Wall is the median over the 12 runs of the seconds each run took;",
        "a BH or MH run is counted at its wall, which lies at the cap.",
        "\\(p_\\mathrm{BH}\\) and \\(p_\\mathrm{MH}\\) are the two-sided Fisher",
        "exact test on hits against each ASE arm.}",
        "\\label{tab:ase-morse}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrrrrr@{}}",
        "\\toprule",
        " & & & \\multicolumn{3}{c}{Hits} & \\multicolumn{3}{c}{Wall (s)} & & \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
        "Cluster & Budget & Cap & This work & BH & MH & This work & BH & MH & "
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


def group_revision(group: list[dict[str, str]]) -> str:
    found = {row["code_revision"] for row in group}
    if len(found) != 1:
        raise SystemExit(f"a summary row needs one code revision, found {sorted(found)}")
    return found.pop()


def retained_summary_lines(
    existing: list[str], summary_header: str, revision: str
) -> list[str]:
    """Keep prior revisions while parsing quoted CSV fields correctly."""
    if existing and existing[0] != summary_header:
        raise SystemExit("summary has unexpected columns")
    parsed = list(csv.DictReader(existing))
    return [
        line
        for line, row in zip(existing[1:], parsed)
        if row["code_revision"] != revision
    ]


def main() -> None:
    flags = sys.argv[1:]
    if any(flag not in ("--check", "--tex") for flag in flags) or len(set(flags)) != len(flags):
        raise SystemExit("usage: summarize.py [--check] [--tex]")
    grouped = load_rows()
    identity = load_identity()
    revisions = sorted({row["code_revision"] for s in SYSTEMS for a in ARMS for row in grouped[s][a]})
    current_schema = all(
        row.get("machine_id") and row.get("cpu_model")
        for system in SYSTEMS
        for arm in ARMS
        for row in grouped[system][arm]
    )
    summary_header = (
        "system,arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,median_outside_s,"
        "fisher_p_bh,mann_whitney_U_bh,mann_whitney_p_bh,fisher_p_mh,mann_whitney_U_mh,mann_whitney_p_mh,"
        "wall_over_bh,wall_over_mh,wall_stops,wall_cap_s,code_revision,host"
    )
    if current_schema:
        summary_header += ",machine_id,cpu_model"
    summary_root = Path(os.environ["ASE_NONLJ_OUT"]) if os.environ.get("ASE_NONLJ_OUT") else ROOT / "bench" / "ase-nonlj"
    summary_path = summary_root / "summary.csv"
    summary_lines = [summary_header]
    if summary_path.exists():
        existing = summary_path.read_text().splitlines()
        try:
            summary_lines.extend(
                retained_summary_lines(existing, summary_header, REVISION)
            )
        except SystemExit as error:
            raise SystemExit(f"{summary_path}: unexpected columns") from error
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
            w_bh, inv_bh = wall_ratio(group, grouped[system]["ase_bh"])
            w_mh, inv_mh = wall_ratio(group, grouped[system]["ase_mh"])
            arm_cap = "" if arm == "for_cluster" else cap
            summary_lines.append(
                f"{system},{arm},{hits},12,{best:.8f},{best - reference:.8f},"
                f"{wall:.6f},{inside:.6f},{outside:.6f},"
                f"{f_bh},{u_bh},{mw_bh},{f_mh},{u_mh},{mw_mh},"
                f"{w_bh:.6f},{w_mh:.6f},{wall_stops},{arm_cap},{group_revision(group)},bench"
                + (
                    ","
                    + csv_fields(
                        (
                            ";".join(sorted({row["machine_id"] for row in group})),
                            ";".join(sorted({row["cpu_model"] for row in group})),
                        )
                    )
                    if current_schema
                    else ""
                )
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
    outputs = {
        summary_path: "\n".join(summary_lines) + "\n",
        summary_root / "README.md": "\n".join(readme),
    }
    if "--tex" in flags:
        document = Path(os.environ["ASE_NONLJ_TEX"]) if os.environ.get("ASE_NONLJ_TEX") else DOCUMENT
        source = document.read_text() if document.exists() else DOCUMENT.read_text()
        outputs[document] = splice(source, tex_table(grouped))
    if "--check" in flags:
        stale = [
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            for path, text in outputs.items()
            if not path.exists() or path.read_text() != text
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
