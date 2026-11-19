#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the head-to-head tables of the manuscript and the SI from the committed summaries.

Five blocks, each between a pair of marker comments:

    tab:gofee       si.tex          bench/ase-gofee-h2h/summary.csv
    tab:tip4p-gmin  si.tex          bench/tip4p-h2h/rows-h2h-r2.csv, wall-r2.csv
    tab:gofee-si    si.tex          bench/ase-gofee-h2h/asg2-summary.csv, arms measured there
    tab:sc          manuscript.tex  bench/metal-cluster-h2h/sc-n12/results.csv, pvalues.csv
    tab:sc-si       si.tex          bench/metal-cluster-h2h/sc-n12/results.csv

    python3 paper/jctc/scripts/h2h_tables.py            # rewrite the blocks
    python3 paper/jctc/scripts/h2h_tables.py --check    # exit 1 if one is stale
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "tip4p_h2h"))
sys.path.insert(0, str(ROOT / "bench" / "tip4p-h2h"))
from run_h2h_r2 import mann_whitney_u_two_sided_exact  # noqa: E402

MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
SI = ROOT / "paper" / "jctc" / "si.tex"
GOFEE = ROOT / "bench" / "ase-gofee-h2h"
TIP4P = ROOT / "bench" / "tip4p-h2h" / "rows-h2h-r2.csv"
TIP4P_WALL = ROOT / "bench" / "tip4p-h2h" / "wall-r2.csv"
SC = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12" / "results.csv"
SC_SEEDS = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12" / "seeds.csv"
SC_P = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12" / "pvalues.csv"
CREST_EQUAL = ROOT / "bench" / "crest-equal"
CREST_ARMS = ("anneal", "crest", "crest-wall")
# The call tolerance of the panel: run_panel.py counts a CREST run as at the hop arm's calls
# when it lies within 5 percent of them.
CREST_CALL_TOLERANCE = 0.05

MARKS = {
    "gofee": ("% gofee table: start", "% gofee table: end"),
    "tip4p": ("% tip4p-gmin table: start", "% tip4p-gmin table: end"),
    "tip4p-prose": ("% tip4p-gmin prose: start", "% tip4p-gmin prose: end"),
    "gofee-si": ("% gofee-si table: start", "% gofee-si table: end"),
    "sc": ("% sc table: start", "% sc table: end"),
    "sc-si": ("% sc-si table: start", "% sc-si table: end"),
}
SC_ORDER = ("sc126-38", "sc108-38", "sc96-38", "sc126-55", "sc108-55", "sc96-55", "sc126-80", "sc108-80", "sc96-80")
SC_FORM = {"126": "12-6", "108": "10-8", "96": "9-6"}
ARM_LABEL = {"paper": "This work", "ase_bh": "ASE BH", "ase_mh": "ASE MH", "agox_gofee": "GOFEE"}
LJ_LABEL = {"lj38": "LJ\\(_{38}\\)", "lj55": "LJ\\(_{55}\\)", "lj75": "LJ\\(_{75}\\)"}
OTHER_LABEL = {"au55": "Au\\(_{55}\\)", "cu38": "Cu\\(_{38}\\)", "cu12au26": "Cu\\(_{12}\\)Au\\(_{26}\\)",
               "sw10": "Si\\(_{10}\\)", "tc10": "C\\(_{10}\\)", "tip4p8": "(H\\(_2\\)O)\\(_8\\)"}
TIP4P_LABEL = {"tip4p8": "(H\\(_2\\)O)\\(_8\\)", "tip4p15": "(H\\(_2\\)O)\\(_{15}\\)",
               "tip4p20": "(H\\(_2\\)O)\\(_{20}\\)", "tip4p21": "(H\\(_2\\)O)\\(_{21}\\)",
               "tip4p25": "(H\\(_2\\)O)\\(_{25}\\)"}


def commit_held(revision: str) -> bool:
    """The revision is a commit object in this repository."""
    if not revision or revision in {"unknown", "latest"}:
        return False
    return subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT, capture_output=True,
    ).returncode == 0


def show_s(value: float) -> str:
    if value < 100.0:
        return f"{value:.3f}"
    return f"{value:.1f}"


def show_calls(value: str) -> str:
    return f"{float(value):.0f}" if value else "--"


def type7_median(values: list[float]) -> float:
    ordered = sorted(values)
    h = 0.5 * (len(ordered) - 1)
    lo = int(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def show_calls_median(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def show_p(value: float) -> str:
    """Two significant figures; 1 only when p is exactly 1."""
    if value >= 1.0:
        return "1"
    if value >= 0.01:
        return f"{value:.2f}" if value >= 0.1 else f"{value:.3f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def budget_tex(budget: str) -> str:
    mantissa, exponent = f"{int(budget):.0e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def gofee_block() -> str:
    """The Lennard-Jones GOFEE table, or nothing when a quoted row cites an absent commit."""
    seeds = [row for row in read(GOFEE / "results.csv") if row["system"] in LJ_LABEL]
    if not seeds or any(not commit_held(row["head"]) for row in seeds):
        return ""
    rows = read(GOFEE / "summary.csv")
    body = []
    for system in LJ_LABEL:
        for arm in ARM_LABEL:
            row = next(r for r in rows if r["system"] == system and r["arm"] == arm)
            body.append(" & ".join([
                LJ_LABEL[system] if arm == "paper" else "", ARM_LABEL[arm],
                budget_tex(row["budget"]) if arm == "paper" else "",
                f"{row['hits']}/{row['n']}", row["capped"], show_calls(row["calls_to_hit_median"]),
                show_s(float(row["wall_median_s"])),
            ]) + " \\\\")
        if system != "lj75":
            body.append("\\addlinespace")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This work against ASE basin hopping (BH), ASE minima hopping (MH)",
        "and AGOX GOFEE on the same three Lennard-Jones clusters.",
        "Seeds 2000--2005, one core, budget in force calls.",
        "Capped counts the runs stopped by a wall cap of 2 hours, which are misses.",
        "Calls is the median force-call count of the runs that hit, and Wall is the median",
        "wall in seconds of all six runs.}",
        "\\label{tab:gofee}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "Cluster & Arm & Budget & Hits & Capped & Calls & Wall (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def tip4p_groups() -> dict[int, tuple[list[dict[str, str]], list[dict[str, str]]]]:
    rows = read(TIP4P)
    sizes = sorted({int(row["molecules"]) for row in rows})
    groups = {}
    for size in sizes:
        anneal = [row for row in rows if row["molecules"] == str(size) and row["arm"] == "anneal"]
        gmin = [row for row in rows if row["molecules"] == str(size) and row["arm"] == "gmin"]
        if len(anneal) != 12 or len(gmin) != 12:
            raise ValueError(f"TIP4P {size}: expected 12 rows per arm")
        a_starts = {row["seed"]: row["start_id"] for row in anneal}
        g_starts = {row["seed"]: row["start_id"] for row in gmin}
        if a_starts != g_starts:
            raise ValueError(f"TIP4P {size}: arm starts differ")
        groups[size] = (anneal, gmin)
    return groups


def tip4p_wall() -> dict[int, dict[str, str]]:
    rows = {int(row["molecules"]): row for row in read(TIP4P_WALL)}
    if set(rows) != {8, 15, 20, 21, 25}:
        raise ValueError("TIP4P wall summary has the wrong sizes")
    if any(row["mwu_method"] != "exact-rank-enumeration"
           for row in rows.values()):
        raise ValueError("TIP4P wall summary is not exact")
    raw = read(TIP4P)
    for size, row in rows.items():
        anneal = [
            float(item["wall_s"]) for item in raw
            if item["molecules"] == str(size) and item["arm"] == "anneal"
        ]
        gmin = [
            float(item["wall_s"]) for item in raw
            if item["molecules"] == str(size) and item["arm"] == "gmin"
        ]
        p_value, method = mann_whitney_u_two_sided_exact(anneal, gmin)
        if method != row["mwu_method"] or abs(
            p_value - float(row["wall_mwu_p"])
        ) > 5.0e-13:
            raise ValueError(f"TIP4P {size}: stale wall summary")
    return rows


def tip4p_prose() -> str:
    groups = tip4p_groups()
    wall = tip4p_wall()
    seeds = sorted({int(row["seed"]) for arms in groups.values() for arm in arms for row in arm})
    hit = {
        size: tuple(sum(int(row["hit"]) for row in arm) for arm in arms)
        for size, arms in groups.items()
    }
    all_rows = [row for arms in groups.values() for arm in arms for row in arm]
    if len({row["machine_id"] for row in all_rows}) != 1:
        raise ValueError("the TIP4P ABBA comparison used more than one machine")
    if {row["core"] for row in all_rows} != {"0"}:
        raise ValueError("the TIP4P ABBA comparison was not pinned to core 0")
    expected_order = ("anneal,gmin", "gmin,anneal")
    for arms in groups.values():
        rows_by_seed = {row["seed"]: row for row in arms[0]}
        orders = [
            rows_by_seed[str(seed)]["arm_order"]
            for seed in seeds
        ]
        if any(order != expected_order[index % 2]
               for index, order in enumerate(orders)):
            raise ValueError("the TIP4P comparison does not use ABBA arm order")
    shown = {
        size: show_p(float(row["wall_mwu_p"]))
        .removeprefix("\\(").removesuffix("\\)")
        for size, row in wall.items() if size != 25
    }
    repeated = set(shown.values())
    if len(repeated) != 1:
        raise ValueError("TIP4P scored sizes do not share one wall-time p")
    exhausted_cost = {
        size: (
            float(wall[size]["anneal_wall_median_s"]) * 1.0e6
            / float(wall[size]["anneal_calls_median"]),
            float(wall[size]["gmin_wall_median_s"]) * 1.0e6
            / float(wall[size]["gmin_calls_median"]),
        )
        for size in (21,)
    }
    unscored = wall[25]
    return "\n".join([
        "Table~\\ref{tab:tip4p-gmin} puts this setting against GMIN on rigid TIP4P",
        "clusters of 8, 15, 20, 21, and 25 molecules.",
        "The hit tolerance for the scored sizes is 0.001~kJ/mol.",
        f"The 120-run ABBA comparison uses seeds {seeds[0]}--{seeds[-1]} on one machine pinned to core 0,",
        "with the arm order alternating across seeds.",
        f"This setting hits \\((\\mathrm{{H_2O}})_{{8}}\\) on {hit[8][0]} of 12 against GMIN on {hit[8][1]} of 12,",
        f"\\((\\mathrm{{H_2O}})_{{15}}\\) on {hit[15][0]} of 12 against {hit[15][1]} of 12, and",
        f"\\((\\mathrm{{H_2O}})_{{20}}\\) on {hit[20][0]} of 12 against {hit[20][1]} of 12.",
        "Neither arm hits \\((\\mathrm{H_2O})_{21}\\).",
        "Size 25 has no independent reference and is unscored.",
        "This setting's best and median final energies are "
        f"{unscored['anneal_energy_best_kj_mol']} and "
        f"{unscored['anneal_energy_median_kj_mol']}~kJ/mol, and its median wall "
        f"is {unscored['anneal_wall_median_s']}~s.",
        "GMIN's corresponding values are "
        f"{unscored['gmin_energy_best_kj_mol']} and "
        f"{unscored['gmin_energy_median_kj_mol']}~kJ/mol and "
        f"{unscored['gmin_wall_median_s']}~s.",
        "Median walls for this setting and GMIN are",
        f"{float(wall[8]['anneal_wall_median_s']):.6f} and "
        f"{float(wall[8]['gmin_wall_median_s']):.6f}~s at 8 molecules,",
        f"{float(wall[15]['anneal_wall_median_s']):.6f} and "
        f"{float(wall[15]['gmin_wall_median_s']):.6f}~s at 15 molecules, and",
        f"{float(wall[20]['anneal_wall_median_s']):.6f} and "
        f"{float(wall[20]['gmin_wall_median_s']):.6f}~s at 20 molecules.",
        "At 21 molecules every run exhausts the two-million-call budget;",
        "the median costs per call for this setting and GMIN are",
        f"{exhausted_cost[21][0]:.6f} and {exhausted_cost[21][1]:.6f}~"
        "\\(\\mu\\)s at 21 molecules.",
        "The two-sided exact rank-enumeration Mann--Whitney test on wall time",
        f"gives \\(p={next(iter(repeated))}\\) at sizes 8, 15, 20, and 21.",
    ])


def tip4p_block() -> str:
    body = []
    wall = tip4p_wall()
    for size, (anneal, gmin) in tip4p_groups().items():
        if size == 25:
            continue
        a_best = [float(row["best_energy"]) for row in anneal]
        g_best = [float(row["best_energy"]) for row in gmin]
        hits = (
            f"{sum(int(row['hit']) for row in anneal)}/12",
            f"{sum(int(row['hit']) for row in gmin)}/12",
        )
        body.append(" & ".join([
            TIP4P_LABEL[f"tip4p{size}"],
            *hits,
            show_calls_median(type7_median([float(row["calls"]) for row in anneal])),
            show_calls_median(type7_median([float(row["calls"]) for row in gmin])),
            f"{type7_median(a_best):.6f}",
            f"{type7_median(g_best):.6f}",
            show_p(float(wall[size]["wall_mwu_p"])),
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This work against GMIN on rigid TIP4P water clusters.",
        "The 120-run ABBA comparison uses seeds 2000--2011 on one machine pinned to core 0,",
        "with the arm order alternating across seeds. Calls and Best are the medians over the 12 runs.",
        "\\(p_\\mathrm{wall}\\) is the two-sided exact rank-enumeration",
        "Mann--Whitney test on the per-run wall times.}",
        "\\label{tab:tip4p-gmin}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Hits} & \\multicolumn{2}{c}{Calls} & "
        "\\multicolumn{2}{c}{Best energy (kJ/mol)} & \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Cluster & This work & GMIN & This work & GMIN & This work & GMIN & "
        "\\(p_\\mathrm{wall}\\) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def gofee_si_block() -> str:
    rows = [r for r in read(GOFEE / "asg2-summary.csv") if r["source"] == "measured"]
    body = []
    for system in OTHER_LABEL:
        first = True
        for arm in ("ase_bh", "ase_mh", "agox_gofee"):
            row = next(r for r in rows if r["system"] == system and r["arm"] == arm)
            body.append(" & ".join([
                OTHER_LABEL[system] if first else "", ARM_LABEL[arm],
                budget_tex(row["budget"]) if first else "",
                f"{row['hits']}/{row['n']}", row["capped"], show_calls(row["calls_to_hit_median"]),
                show_s(float(row["wall_median_s"])),
            ]) + " \\\\")
            first = False
        if system != "tip4p8":
            body.append("\\addlinespace")
    return "\n".join([
        "\\begin{table}",
        "\\caption{ASE basin hopping (BH), ASE minima hopping (MH) and AGOX GOFEE on the",
        "non-Lennard-Jones panel systems, on the potential of this code.",
        "Six seeds, one core, budget in force calls, and a wall cap of 2 hours per run.",
        "Capped counts the runs stopped by the cap, which are misses.",
        "Calls is the median force-call count of the runs that hit, and Wall is the median",
        "wall in seconds of all six runs.}",
        "\\label{tab:gofee-si}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "System & Arm & Budget & Hits & Capped & Calls & Wall (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def sc_label(system: str) -> str:
    form, size = system.removeprefix("sc").split("-")
    return f"{SC_FORM[form]}, {size}"


def sc_pairs() -> list[tuple[dict[str, str], dict[str, str]]]:
    """The anneal row of each system and the GMIN row of the corrected data file.

    The corrected file is the GMIN harness that differs from the anneal one;
    the anneal harness's own GMIN rows ran with Lennard-Jones step and
    temperature and are not shown."""
    rows = read(SC)
    pairs = []
    for system in SC_ORDER:
        anneal = next(r for r in rows if r["system"] == system and r["arm"] == "anneal")
        gmin = [r for r in rows if r["system"] == system and r["arm"] == "gmin"
                and r["harness_revision"] != anneal["harness_revision"]]
        if len(gmin) != 1:
            raise SystemExit(f"{system}: expected one corrected GMIN row, found {len(gmin)}")
        pairs.append((anneal, gmin[0]))
    return pairs


def sc_tests() -> dict[str, dict[str, str]]:
    """The p-values of each system against its corrected GMIN rows, from pvalues.csv."""
    tests: dict[str, dict[str, str]] = {}
    for anneal, gmin in sc_pairs():
        rows = [r for r in read(SC_P) if r["system"] == anneal["system"]
                and r["harness_revision"] == gmin["harness_revision"]]
        if len(rows) != 1:
            raise SystemExit(f"{anneal['system']}: expected one pvalues.csv row, found {len(rows)}")
        tests[anneal["system"]] = rows[0]
    return tests


def sc_gmin_quotable() -> bool:
    """The corrected GMIN harness of every Sutton-Chen system is a commit this repository holds."""
    return all(
        commit_held(anneal["harness_revision"]) and commit_held(gmin["harness_revision"])
        for anneal, gmin in sc_pairs()
    )


def sc_block() -> str:
    if not sc_gmin_quotable():
        return _sc_anneal_block()
    body = []
    tests = sc_tests()
    seed_rows = read(SC_SEEDS)
    for anneal, gmin in sc_pairs():
        system = anneal["system"]
        test = tests[system]
        a_seeds = [float(r["evals_to_hit"]) for r in seed_rows
                   if r["system"] == system and r["arm"] == "anneal"
                   and r["harness_revision"] == anneal["harness_revision"]]
        g_seeds = [float(r["evals_to_hit"]) for r in seed_rows
                   if r["system"] == system and r["arm"] == "gmin"
                   and r["harness_revision"] == gmin["harness_revision"]]
        if len(a_seeds) != 12 or len(g_seeds) != 12:
            raise SystemExit(f"{system}: expected 12 seed rows per arm, found {len(a_seeds)} and {len(g_seeds)}")
        a_med = type7_median(a_seeds)
        g_med = type7_median(g_seeds)
        body.append(" & ".join([
            sc_label(system), budget_tex(anneal["budget"]),
            anneal["hits_over_n"], gmin["hits_over_n"],
            show_calls_median(a_med), show_calls_median(g_med),
            show_s(float(anneal["wall_median_s"])), show_s(float(gmin["wall_median_s"])),
            show_p(float(test["fisher_p"])), show_p(float(test["mannwhitney_p_evals"])),
            show_p(float(test["mannwhitney_p_wall"])),
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This work against GMIN on reduced Sutton--Chen clusters at equal workload.",
        "Form \\(n\\)-\\(m\\) and size \\(N\\); seeds 2000--2011; the same start file and",
        "the same force-call budget for both arms; one run at a time on one core.",
        "Calls is the median force-call count with a miss at the budget, and Wall the",
        "median wall in seconds over the 12 runs.",
        "\\(p_\\mathrm{F}\\) is the two-sided Fisher exact test on hits, and \\(p_\\mathrm{U}\\)",
        "the two-sided continuity-corrected asymptotic Mann--Whitney test on calls and on wall",
        "against the Doye--Wales Sutton--Chen GMIN rows, as",
        "\\texttt{scripts/metal\\_h2h/sc\\_n12\\_pvalues.py} writes them to",
        "\\texttt{bench/metal-cluster-h2h/sc-n12/pvalues.csv}.}",
        "\\label{tab:sc}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}%",
        "\\begin{tabular}{@{}lrrrrrrrrrr@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Hits} & \\multicolumn{2}{c}{Calls} & \\multicolumn{2}{c}{Wall (s)} & & "
        "\\multicolumn{2}{c}{\\(p_\\mathrm{U}\\)} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){10-11}",
        "Form, \\(N\\) & Budget & This work & GMIN & This work & GMIN & This work & GMIN & "
        "\\(p_\\mathrm{F}\\) & Calls & Wall \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def _sc_anneal_block() -> str:
    """This setting only. The GMIN arm of the comparison cites a commit this repository does not hold."""
    body = []
    seed_rows = read(SC_SEEDS)
    harnesses = set()
    for anneal, _gmin in sc_pairs():
        if not commit_held(anneal["harness_revision"]):
            continue
        system = anneal["system"]
        harnesses.add(anneal["harness_revision"][:8])
        calls = [float(row["evals_to_hit"]) for row in seed_rows
                 if row["system"] == system and row["arm"] == "anneal"
                 and row["harness_revision"] == anneal["harness_revision"]]
        if len(calls) != 12:
            raise SystemExit(f"{system}: expected 12 held anneal seed rows, found {len(calls)}")
        body.append(" & ".join([
            sc_label(system), budget_tex(anneal["budget"]), anneal["hits_over_n"],
            show_calls_median(type7_median(calls)), show_s(float(anneal["wall_median_s"])),
        ]) + " \\\\")
    shown = ", ".join(f"\\texttt{{{item}}}" for item in sorted(harnesses))
    return "\n".join([
        "\\begin{table}",
        "\\caption{This setting on reduced Sutton--Chen clusters.",
        "Form \\(n\\)-\\(m\\) and size \\(N\\); seeds 2000--2011; one run at a time on one core.",
        "Calls is the median force-call count with a miss at the budget, and Wall the",
        f"median wall in seconds over the 12 runs. Every row is harness {shown}.}}",
        "\\label{tab:sc}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}%",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Form, \\(N\\) & Budget & Hits & Calls & Wall (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def _sc_anneal_si_block() -> str:
    body = []
    harnesses = set()
    for anneal, _gmin in sc_pairs():
        if not commit_held(anneal["harness_revision"]):
            continue
        harnesses.add(anneal["harness_revision"][:8])
        body.append(" & ".join([
            sc_label(anneal["system"]), f"\\({anneal['reference']}\\)",
            f"{1e6 * float(anneal['pot_per_eval_s']):.2f}",
            f"{1e6 * float(anneal['outside_per_eval_s']):.2f}",
        ]) + " \\\\")
    shown = ", ".join(f"\\texttt{{{item}}}" for item in sorted(harnesses))
    return "\n".join([
        "\\begin{table}",
        "\\caption{Reference energies and the cost per force call of this setting on Sutton--Chen.",
        "The reference is the Doye--Wales global minimum in reduced units.",
        "Inside is the median time per call inside the energy and gradient, and",
        "Outside the median time per call spent elsewhere, in microseconds.",
        f"Every row is harness {shown}.}}",
        "\\label{tab:sc-si}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Form, \\(N\\) & Reference & Inside (\\(\\mu\\)s) & Outside (\\(\\mu\\)s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def sc_si_block() -> str:
    if not sc_gmin_quotable():
        return _sc_anneal_si_block()
    body = []
    for anneal, gmin in sc_pairs():
        body.append(" & ".join([
            sc_label(anneal["system"]), f"\\({anneal['reference']}\\)",
            f"{1e6 * float(anneal['pot_per_eval_s']):.2f}", f"{1e6 * float(gmin['pot_per_eval_s']):.2f}",
            f"{1e6 * float(anneal['outside_per_eval_s']):.2f}", f"{1e6 * float(gmin['outside_per_eval_s']):.2f}",
            gmin["harness_revision"][:8],
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table}",
        "\\caption{Reference energies and the cost per force call of the Sutton--Chen panel.",
        "The reference is the Doye--Wales global minimum in reduced units.",
        "Inside is the median time per call inside the energy and gradient, and",
        "Outside the median time per call spent elsewhere, in microseconds.",
        "Harness is the revision of the GMIN Sutton--Chen data file; every anneal row is",
        "harness \\texttt{e3ed7dad}.}",
        "\\label{tab:sc-si}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrrrrl@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Inside (\\(\\mu\\)s)} & \\multicolumn{2}{c}{Outside (\\(\\mu\\)s)} & \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Form, \\(N\\) & Reference & This work & GMIN & This work & GMIN & Harness \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def crest_equal_summary() -> dict[str, dict[str, str]]:
    """The summary row of each arm at the revision of the latest rows, checked against the rows."""
    rows = read(CREST_EQUAL / "rows.csv")
    revision = rows[-1]["code_revision"]
    rows = [r for r in rows if r["code_revision"] == revision]
    summary = {r["arm"]: r for r in read(CREST_EQUAL / "summary.csv") if r["code_revision"] == revision}
    if set(summary) != set(CREST_ARMS):
        raise SystemExit(f"crest-equal: summary arms at {revision[:8]} are {sorted(summary)}")
    for arm, row in summary.items():
        runs = [r for r in rows if r["arm"] == arm]
        hits = sum(r["hit"] == "true" for r in runs)
        if (len(runs), hits) != (int(row["n"]), int(row["hits"])):
            raise SystemExit(f"crest-equal {arm}: rows give {hits}/{len(runs)}, summary {row['hits']}/{row['n']}")
    return summary


def crest_p(value: str) -> str:
    """Three decimals down to 0.001, then two significant figures."""
    if not value:
        return "--"
    number = float(value)
    if number >= 0.999:
        return "1"
    if number >= 0.001:
        return f"{number:.3f}"
    return show_p(number)


def crest_equal_block() -> str:
    summary = crest_equal_summary()
    label = {"anneal": "This work", "crest": "CREST", "crest-wall": "CREST"}
    body = []
    for arm in CREST_ARMS:
        row = summary[arm]
        body.append(" & ".join([
            label[arm], row["crest_len_ps"] or "--", f"{row['hits']}/{row['n']}",
            f"{float(row['calls_median']):.1f}".removesuffix(".0"), f"{row['calls_min']}--{row['calls_max']}",
            row["within_5pct"], f"{float(row['wall_median_s']):.1f}", f"{float(row['pot_median_s']):.2f}",
            f"{float(row['outside_median_s']):.2f}", crest_p(row["fisher_p"]),
            crest_p(row["mannwhitney_calls_p"]), crest_p(row["mannwhitney_inside_p"]),
            crest_p(row["mannwhitney_wall_p"]),
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table*}",
        "\\caption{This work against CREST 3.0.2 on the water hexamer at equal GFN2 calls.",
        "Both arms evaluate GFN2-xTB through xtb 6.7.1 at accuracy 0.01 plus the one-sided O--H",
        "wall; seeds 2000--2011; one run at a time on one core.",
        "Length is the \\texttt{-mquick} length of CREST in picoseconds.",
        "Calls is the GFN2 energy and gradient count served by the driver, the median",
        "and the range; the CREST rows had a driver ceiling of 50000 calls, and their",
        "4000 budget field is a label rather than a cap.",
        "\\(\\pm5\\%\\)",
        f"counts the runs within {100 * CREST_CALL_TOLERANCE:.0f}\\% of the {float(summary['anneal']['calls_median']):.0f} "
        "calls of this work.",
        "Wall, Inside, and Outside are medians in seconds of the whole run, of the time inside the",
        "potential, and of the rest.",
        "A hit is a vtight polish of the best structure within \\(10^{-7}\\)~Eh of the prism.",
        "\\(p_\\mathrm{F}\\) is the two-sided Fisher exact test on hits and \\(p_\\mathrm{U}\\) the",
        "two-sided Mann--Whitney test, each against this work, as",
        "\\texttt{scripts/crest\\_equal/run\\_panel.py} writes them to",
        "\\texttt{bench/crest-equal/summary.csv}.}",
        "\\label{tab:crest-equal}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}%",
        "\\begin{tabular}{@{}lrrrrrrrrrrrr@{}}",
        "\\toprule",
        " & & & \\multicolumn{3}{c}{Calls} & & & & & \\multicolumn{3}{c}{\\(p_\\mathrm{U}\\)} \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){11-13}",
        "Arm & Length & Hits & Median & Range & \\(\\pm5\\%\\) & Wall & Inside & Outside & "
        "\\(p_\\mathrm{F}\\) & Calls & Inside & Wall \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])


BLOCKS = {
    "gofee": (SI, gofee_block),
    "tip4p-prose": (SI, tip4p_prose),
    "tip4p": (SI, tip4p_block),
    "gofee-si": (SI, gofee_si_block),
    "sc": (MANUSCRIPT, sc_block),
    "sc-si": (SI, sc_si_block),
}


def splice(text: str, marks: tuple[str, str], block: str) -> str:
    start, end = marks
    if not block:
        if start in text or end in text:
            raise SystemExit(f"{start!r}: the file still marks an empty block")
        return text
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{start!r}: the file needs exactly one block")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{block}\n{end}{tail}"


def main(argv: list[str]) -> int:
    stale = False
    override = os.environ.get("H2H_TABLES_OUTPUT", "")
    checking = "--check" in argv
    for name, (path, make) in BLOCKS.items():
        target = Path(override) / path.name if override else path
        if not checking and target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        source = target if target.exists() else path
        text = source.read_text()
        updated = splice(text, MARKS[name], make())
        if checking:
            if updated != text:
                print(f"the {name} table in {target.name} is stale", file=sys.stderr)
                stale = True
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(updated)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
