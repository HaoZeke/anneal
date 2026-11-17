#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the posterior-against-uniform table of the manuscript from the result files.

The rows are bench/depth-posterior-ablation/results.csv (seeds 2000-2005) and
results/size-tiered-selection.csv (seeds 7000-7005). LJ38 and LJ55 use the first
file; LJ75, LJ98 and LJ104 pool both. Hits, flags and the Fisher probability
come from selector_fisher.py, so the table and that script cannot disagree.

    python3 paper/jctc/scripts/ablation_table.py            # rewrite the block
    python3 paper/jctc/scripts/ablation_table.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
TEX_MARKS = ("% ablation table: start", "% ablation table: end")
BUDGET = {38: "4\\times 10^{5}", 55: "1\\times 10^{6}", 75: "4\\times 10^{6}", 98: "4\\times 10^{6}", 104: "4\\times 10^{6}"}

SPEC = importlib.util.spec_from_file_location("selector_fisher", HERE / "selector_fisher.py")
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


def rows() -> list[tuple[int, str, str, str]]:
    first = SELECTOR.load_first()
    second = SELECTOR.load_second()
    out = []
    for size in SELECTOR.FIRST_ONLY + SELECTOR.POOLED:
        arms = {}
        for arm in ("posterior", "uniform"):
            hits, n = SELECTOR.counts(first[(size, arm)])
            if size in SELECTOR.POOLED:
                more, extra = SELECTOR.counts(second[(size, arm)])
                hits, n = hits + more, n + extra
            arms[arm] = (hits, n)
        p = SELECTOR.fisher_two_sided(*arms["posterior"], *arms["uniform"])
        out.append((size, "{}/{}".format(*arms["posterior"]), "{}/{}".format(*arms["uniform"]),
                    SELECTOR.format_probability(p)))
    return out


def tex_table() -> str:
    lines = [
        "\\begin{table}",
        "\\caption{Hits for the depth posterior and a uniform draw.",
        "LJ\\(_{38}\\) and LJ\\(_{55}\\) are seeds 2000--2005.",
        "The other rows pool those seeds with 7000--7005.",
        "The column \\(p\\) is the two-sided Fisher exact probability.",
        "A hit satisfies equation~\\ref{eq:hit}.}",
        "\\label{tab:ablation}",
        "\\centering",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Cluster & Budget & Posterior & Uniform & \\(p\\) \\\\",
        "\\midrule",
    ]
    for size, posterior, uniform, p in rows():
        lines.append(f"LJ\\(_{{{size}}}\\) & \\({BUDGET[size]}\\) & {posterior} & {uniform} & {p} \\\\")
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
