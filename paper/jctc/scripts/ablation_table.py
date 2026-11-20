#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the posterior-against-uniform table of the manuscript from the result files.

The rows are bench/depth-posterior-ablation/results.csv (seeds 2000-2005) and
results/size-tiered-selection.csv (seeds 7000-7005). LJ38 and LJ55 use the first
file; LJ75, LJ98 and LJ104 pool both. Each hit flag is checked against its energy
by selector_fisher.check_hit. The two arms of a seed start from the same geometry
(examples/depth_ablation.rs draws the start from the seed before the arm acts), so
the table pairs them by seed: the paired outcome table, the exact McNemar test, and
the 95% interval of method 10 of Newcombe for paired data, from scripts/paired_stats.py.

    python3 paper/jctc/scripts/ablation_table.py            # rewrite the block
    python3 paper/jctc/scripts/ablation_table.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import paired_stats  # noqa: E402

MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
TEX_MARKS = ("% ablation table: start", "% ablation table: end")
BUDGET = {38: "4\\times 10^{5}", 55: "1\\times 10^{6}", 75: "4\\times 10^{6}", 98: "4\\times 10^{6}", 104: "4\\times 10^{6}"}

SPEC = importlib.util.spec_from_file_location("selector_fisher", HERE / "selector_fisher.py")
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


def show_interval(low: float, high: float) -> str:
    def one(v: float) -> str:
        text = f"{v:.2f}"
        return "0.00" if text == "-0.00" else text.replace("-", "\\(-\\)")
    return f"{one(low)} to {one(high)}"


def show_p(p: float) -> str:
    if p >= 1.0:
        return "1"
    return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"


def seed_hits() -> dict[int, dict[str, dict[int, bool]]]:
    """Per size and arm, the checked hit flag of each seed of the sizes the table pools."""
    out: dict[int, dict[str, dict[int, bool]]] = {}
    with SELECTOR.FIRST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            size = int(row["system"].removeprefix("lj"))
            out.setdefault(size, {}).setdefault(row["arm"], {})[int(row["seed"])] = \
                bool(SELECTOR.check_hit(size, row["hit"], row["best"]))
    with SELECTOR.SECOND.open(newline="") as handle:
        for row in csv.DictReader(handle):
            size = int(row["n"])
            if row["arm"] == "tiered" or size not in SELECTOR.POOLED:
                continue
            out.setdefault(size, {}).setdefault(row["arm"], {})[int(row["seed"])] = \
                bool(SELECTOR.check_hit(size, row["hit"], row["best"]))
    return out


def tables() -> list[tuple[int, tuple[int, int, int, int]]]:
    """The paired outcome table, posterior first, of each size in table order."""
    hits = seed_hits()
    out = []
    for size in SELECTOR.FIRST_ONLY + SELECTOR.POOLED:
        posterior, uniform = hits[size]["posterior"], hits[size]["uniform"]
        want = 6 if size in SELECTOR.FIRST_ONLY else 12
        if sorted(posterior) != sorted(uniform) or len(posterior) != want:
            raise SystemExit(f"LJ{size}: the arms do not share {want} seeds")
        seeds = sorted(posterior)
        out.append((size, paired_stats.outcome_table([posterior[s] for s in seeds], [uniform[s] for s in seeds])))
    return out


def pooled() -> tuple[int, int, int, int]:
    return tuple(sum(t[i] for _, t in tables()) for i in range(4))


def line(label: str, budget: str, table: tuple[int, int, int, int]) -> str:
    both, only_post, only_unif, neither = table
    n = sum(table)
    p = paired_stats.mcnemar_exact(only_post, only_unif)
    interval = show_interval(*paired_stats.newcombe_paired(*table))
    return (f"{label} & {budget} & {both + only_post}/{n} & {both + only_unif}/{n} & {both} & {only_post} & "
            f"{only_unif} & {neither} & {show_p(p)} & {interval} \\\\")


def tex_table() -> str:
    lines = [
        "\\begin{table}",
        "\\caption{Hits for the depth posterior and a uniform draw, paired by seed.",
        "LJ\\(_{38}\\) and LJ\\(_{55}\\) are seeds 2000--2005.",
        "The other rows pool those seeds with 7000--7005.",
        "Both arms of a seed start from the same geometry.",
        "Both, Post., Unif., and Neither count the seeds both arms, the posterior alone, the uniform",
        "draw alone, and neither arm hit.",
        "The column \\(p\\) is the two-sided exact McNemar test, and the last column the 95\\%",
        "interval of Newcombe's method 10 for paired data, for the posterior hit rate less the",
        "uniform one.",
        "A hit satisfies equation~\\ref{eq:hit}.}",
        "\\label{tab:ablation}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}",
        "\\toprule",
        "Cluster & Budget & Posterior & Uniform & Both & Post. & Unif. & Neither & \\(p\\) & "
        "95\\% interval \\\\",
        "\\midrule",
    ]
    for size, table in tables():
        lines.append(line(f"LJ\\(_{{{size}}}\\)", f"\\({BUDGET[size]}\\)", table))
    lines.append("\\midrule")
    lines.append(line("Pooled", "", pooled()))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def splice(text: str, block: str) -> str:
    start, end = TEX_MARKS
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit("the manuscript needs exactly one ablation table block")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{block}\n{end}{tail}"


def main(argv: list[str]) -> int:
    paired_stats.self_test()
    text = MANUSCRIPT.read_text()
    updated = splice(text, tex_table())
    if "--check" in argv:
        if updated != text:
            print("the ablation table in manuscript.tex is stale", file=sys.stderr)
            return 1
        return 0
    MANUSCRIPT.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
