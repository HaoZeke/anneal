#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write tab:si-template-era of the SI: the GMIN panel of c2de652d against the rows of tab:gmin.

c2de652d precedes the removal of the structure templates in 1c1b1238 and of their
switches in 5499be31; its panel is paper/jctc/data/template_era/rows.csv, copied
unchanged from the commit its PROVENANCE names. The other side is the rows of
tab:gmin, read and checked by gmin_table.py. Both sessions run each seed from the
start the driver writes for it, and the GMIN arm of each session takes the same calls
to the same minimum on every seed, which this script requires; Au55, whose GMIN
build changed between the sessions, is left out. The paired tests are
those of scripts/paired_stats.py, for the later revision less c2de652d.

    python3 paper/jctc/scripts/template_era_table.py            # rewrite the block
    python3 paper/jctc/scripts/template_era_table.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("gmin_table", HERE / "gmin_table.py")
TABLE = importlib.util.module_from_spec(_SPEC)
sys.modules["gmin_table"] = TABLE
_SPEC.loader.exec_module(TABLE)
PANEL = TABLE.PANEL
STATS = PANEL.paired_stats

DATA = TABLE.ROOT / "paper" / "jctc" / "data" / "template_era"
TEMPLATES = "c2de652dca0bc75e2b0cb4afffac91fd656404e4"
TEMPLATE_GMIN = "aac1548a8aec13b53ad12a8d6904625081685d2e"
SI = TABLE.ROOT / "paper" / "jctc" / "si.tex"
MARKS = ("% template-era table: start", "% template-era table: end")
LEFT_OUT = ("au55",)


def template_rows() -> list[dict[str, str]]:
    path = DATA / "rows.csv"
    fields = TABLE.provenance("", DATA)
    if TABLE.git("hash-object", str(path)) != fields["blob"] or fields["ancestor"] != TEMPLATES:
        raise SystemExit(f"template_era_table: {path} is not the blob its PROVENANCE names")
    if not TABLE.is_ancestor(fields["commit"]) or not TABLE.is_ancestor(TEMPLATES):
        raise SystemExit("template_era_table: the panel commit or its revision is not an ancestor of HEAD")
    return list(csv.DictReader(path.open()))


def groups(rows: list[dict[str, str]], revision: str, gmin: str,
           harness: str | None = None) -> dict[tuple[str, str], list[dict[str, str]]]:
    PANEL.GMIN_REV = gmin
    out = PANEL.panel_groups(rows, revision, harness)
    for (system, code), group in out.items():
        PANEL.validate_group(group, system, code)
    return out


def blocks() -> list[tuple[str, str, dict[tuple[str, str], list[dict[str, str]]]]]:
    later = [(TABLE.REVISION, TABLE.rows()), (TABLE.SUBMITTED, TABLE.develop_rows())]
    return [(revision, harness, groups(rows, revision, TABLE.GMIN_SNAPSHOT, harness))
            for (revision, rows), harness in zip(later, (TABLE.REVISION, TABLE.DEVELOP_HARNESS))]


def compared() -> list[tuple[str, list[dict[str, object]]]]:
    old = groups(template_rows(), TEMPLATES, TEMPLATE_GMIN)
    out = []
    for revision, _, new in blocks():
        lines = []
        for system in PANEL.PAPER_BUDGETS:
            if (system, "anneal") not in new or system in LEFT_OUT:
                continue
            before, after = old[(system, "anneal")], new[(system, "anneal")]
            if [r["seed"] for r in before] != [r["seed"] for r in after]:
                raise SystemExit(f"template_era_table: {system} does not share seeds")
            same = [(a["evals_to_hit"], a["best"]) == (b["evals_to_hit"], b["best"])
                    for a, b in zip(old[(system, "gmin")], new[(system, "gmin")])]
            if len(same) != len(after) or not all(same):
                raise SystemExit(f"template_era_table: GMIN differs on {system}, so the starts may differ")
            table = STATS.outcome_table([r["hit"] == "1" for r in after], [r["hit"] == "1" for r in before])
            calls = [float(a["evals_to_hit"]) - float(b["evals_to_hit"]) for a, b in zip(after, before)]
            lines.append({"system": system, "revision": revision, "before": before, "after": after, "table": table,
                          "mcnemar_p": STATS.mcnemar_exact(table[1], table[2]),
                          "newcombe": STATS.newcombe_paired(*table),
                          "wilcoxon_calls": STATS.wilcoxon_exact(calls)[2]})
        out.append((revision, lines))
    return out


def calls(group: list[dict[str, str]], p: float) -> float:
    return PANEL.type7([float(r["evals_to_hit"]) for r in group], p)


def line_cells(line: dict[str, object]) -> list[list[str]]:
    both, only_after, only_before, neither = line["table"]
    out = []
    for first, group, only, revision in ((True, line["before"], only_before, TEMPLATES),
                                         (False, line["after"], only_after, line["revision"])):
        hits = sum(r["hit"] == "1" for r in group)
        out.append([
            PANEL.LABELS[line["system"]][1] if first else "",
            PANEL.show_budget(int(group[0]["budget"]), True) if first else "",
            f"\\texttt{{{revision[:8]}}}",
            f"{hits}/{len(group)}",
            str(only),
            str(both) if first else "",
            str(neither) if first else "",
            PANEL.show_calls(calls(group, 0.5)),
            PANEL.show_calls(calls(group, 0.75) - calls(group, 0.25)),
            "" if first else PANEL.show_exact_p(line["mcnemar_p"], True),
            "" if first else PANEL.show_interval(*line["newcombe"]),
            "" if first else PANEL.show_p(line["wilcoxon_calls"], True),
        ])
    return out


def table() -> str:
    body = []
    for index, (revision, lines) in enumerate(compared()):
        if index:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\emph{{\\texttt{{c2de652d}} against "
                    f"\\texttt{{{revision[:8]}}}}}}} \\\\")
        body += [" & ".join(cells) + " \\\\" for line in lines for cells in line_cells(line)]
    return "\n".join(
        [
            "\\begin{table}",
            "\\caption{This code at the earlier template-assisted revision \\texttt{c2de652d} (structure",
            "templates since removed in \\texttt{1c1b1238}) against the rows of this code in the GMIN",
            "table of the main text, paired by seed.",
            "Each revision runs \\texttt{Config::recommended}\\((N)\\) through the drivers of that",
            "table; at \\texttt{c2de652d}, LJ\\(_{103}\\) runs a build with",
            "\\texttt{bench/paper-0120/lj103-table.patch}, which adds its reference energy and budget.",
            "Both sessions start each seed from the geometry the driver writes for it, and the GMIN arm",
            "of each takes the same calls to the same minimum on every seed of every cluster here.",
            "Au\\(_{55}\\) is left out, since the GMIN build of the main text sets the run index",
            "before its first use and its GMIN arm differs from the earlier session.",
            "Only counts the seeds that revision alone hits; Calls and IQR are as in the GMIN table",
            "of the main text.",
            "\\(p_\\mathrm{M}\\), \\(\\Delta\\) and \\(p_\\mathrm{calls}\\) are the tests of that",
            "table, for the later revision less \\texttt{c2de652d}.",
            "The comparison spans every change between the revisions, not the templates alone.",
            "The rows of \\texttt{c2de652d} are \\texttt{paper/jctc/data/template\\_era/rows.csv}.}",
            "\\label{tab:si-template-era}",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{@{}lrlrrrrrrrcr@{}}",
            "\\toprule",
            "Cluster & Budget & Revision & Hits & Only & Both & Neither & Calls & IQR & "
            "\\(p_\\mathrm{M}\\) & \\(\\Delta\\), 95\\% & \\(p_\\mathrm{calls}\\) \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def block(text: str) -> str:
    start, end = MARKS
    return text.split(start, 1)[1].split(end, 1)[0].strip("\n")


def main(argv: list[str]) -> int:
    text = SI.read_text()
    fresh = table()
    if "--check" in argv:
        if MARKS[0] not in text or block(text) != fresh:
            print("template_era_table: the template-era block of the SI is stale", file=sys.stderr)
            return 1
        return 0
    if "--print" in argv:
        print(fresh)
        return 0
    PANEL.splice(SI, MARKS, fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
