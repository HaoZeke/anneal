#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the head-to-head tables of the manuscript and the SI from the committed summaries.

Three blocks, each between a pair of marker comments:

    tab:gofee       manuscript.tex  bench/ase-gofee-h2h/summary.csv
    tab:tip4p-gmin  manuscript.tex  bench/tip4p-h2h/h2h-summary.csv
    tab:gofee-si    si.tex          bench/ase-gofee-h2h/asg2-summary.csv, arms measured there

    python3 paper/jctc/scripts/h2h_tables.py            # rewrite the blocks
    python3 paper/jctc/scripts/h2h_tables.py --check    # exit 1 if one is stale
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
SI = ROOT / "paper" / "jctc" / "si.tex"
GOFEE = ROOT / "bench" / "ase-gofee-h2h"
TIP4P = ROOT / "bench" / "tip4p-h2h" / "h2h-summary.csv"

MARKS = {
    "gofee": ("% gofee table: start", "% gofee table: end"),
    "tip4p": ("% tip4p-gmin table: start", "% tip4p-gmin table: end"),
    "gofee-si": ("% gofee-si table: start", "% gofee-si table: end"),
}
ARM_LABEL = {"paper": "This work", "ase_bh": "ASE BH", "ase_mh": "ASE MH", "agox_gofee": "GOFEE"}
LJ_LABEL = {"lj38": "LJ\\(_{38}\\)", "lj55": "LJ\\(_{55}\\)", "lj75": "LJ\\(_{75}\\)"}
OTHER_LABEL = {"au55": "Au\\(_{55}\\)", "cu38": "Cu\\(_{38}\\)", "cu12au26": "Cu\\(_{12}\\)Au\\(_{26}\\)",
               "sw10": "Si\\(_{10}\\)", "tc10": "C\\(_{10}\\)", "tip4p8": "(H\\(_2\\)O)\\(_8\\)"}
TIP4P_LABEL = {"tip4p8": "(H\\(_2\\)O)\\(_8\\)", "tip4p15": "(H\\(_2\\)O)\\(_{15}\\)",
               "tip4p20": "(H\\(_2\\)O)\\(_{20}\\)", "tip4p21": "(H\\(_2\\)O)\\(_{21}\\)",
               "tip4p25": "(H\\(_2\\)O)\\(_{25}\\)"}


def show_s(value: float) -> str:
    if value < 100.0:
        return f"{value:.3f}"
    return f"{value:.1f}"


def show_calls(value: str) -> str:
    return f"{float(value):.0f}" if value else "--"


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
        "wall in seconds of all six runs.~\\rerun{5}}",
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


def tip4p_block() -> str:
    body = []
    for row in read(TIP4P):
        body.append(" & ".join([
            TIP4P_LABEL[row["system"]], budget_tex(row["budget"]),
            row["anneal_hits"], row["gmin_hits"],
            show_s(float(row["anneal_wall_median_s"])), show_s(float(row["gmin_wall_median_s"])),
            show_p(float(row["fisher_p"])), show_p(float(row["mwu_p"])),
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This work against GMIN on rigid TIP4P water clusters.",
        "Seeds 2000--2011, one core, budget in force calls.",
        "Wall is the median wall in seconds over the 12 runs.",
        "\\(p_\\mathrm{F}\\) is the two-sided Fisher exact test on hits and \\(p_\\mathrm{U}\\)",
        "the two-sided Mann--Whitney test on the best energy of each run.}",
        "\\label{tab:tip4p-gmin}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Hits} & \\multicolumn{2}{c}{Wall (s)} & & \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Cluster & Budget & This work & GMIN & This work & GMIN & "
        "\\(p_\\mathrm{F}\\) & \\(p_\\mathrm{U}\\) \\\\",
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


BLOCKS = {
    "gofee": (MANUSCRIPT, gofee_block),
    "tip4p": (MANUSCRIPT, tip4p_block),
    "gofee-si": (SI, gofee_si_block),
}


def splice(text: str, marks: tuple[str, str], block: str) -> str:
    start, end = marks
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{start!r}: the file needs exactly one block")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{block}\n{end}{tail}"


def main(argv: list[str]) -> int:
    stale = False
    for name, (path, make) in BLOCKS.items():
        text = path.read_text()
        updated = splice(text, MARKS[name], make())
        if "--check" in argv:
            if updated != text:
                print(f"the {name} table in {path.name} is stale", file=sys.stderr)
                stale = True
        else:
            path.write_text(updated)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
