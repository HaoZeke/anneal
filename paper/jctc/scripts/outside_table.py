#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write tab:si-outside of the SI: the outside time of the submitted revision, two builds.

The rows are paper/jctc/data/outside_perf/rows.csv, copied unchanged from the commit its
PROVENANCE names. Only the develop and develop_prime arms are read: both are builds of the
submitted revision 61fa67cf, and this script requires that they take the same calls to the
same minimum on every seed. The third arm of the record is a commit outside the history of
the submitted revision and is not printed, nor are the BLJ systems, which the article does
not run. Every number is recomputed from the per-seed rows; the paired test is the exact
Wilcoxon signed-rank test of scripts/paired_stats.py on the per-seed outside seconds.

    python3 paper/jctc/scripts/outside_table.py            # rewrite the block
    python3 paper/jctc/scripts/outside_table.py --check    # exit 1 if it is stale
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

SI = TABLE.ROOT / "paper" / "jctc" / "si.tex"
DATA = TABLE.ROOT / "paper" / "jctc" / "data" / "outside_perf"
ARMS = ("develop", "develop_prime")
SYSTEMS = ("lj104", "lj150", "m80r6", "m80r14", "au55")
MARKS = ("% outside table: start", "% outside table: end")


def rows() -> dict[tuple[str, str], list[dict[str, str]]]:
    path = DATA / "rows.csv"
    fields = TABLE.provenance("", DATA)
    if TABLE.git("hash-object", str(path)) != fields["blob"]:
        raise SystemExit(f"outside_table: {path} is not the blob its PROVENANCE names")
    if TABLE.git("rev-parse", f"{fields['commit']}^{{tree}}") != fields["tree"]:
        raise SystemExit("outside_table: the record commit is not the tree its PROVENANCE names")
    if fields["ancestor"] != TABLE.SUBMITTED or not TABLE.is_ancestor(TABLE.SUBMITTED):
        raise SystemExit("outside_table: the arms are not the submitted revision")
    groups: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for row in csv.DictReader(path.open()):
        if row["arm"] in ARMS and row["system"] in SYSTEMS:
            if row["code_revision"] != TABLE.SUBMITTED:
                raise SystemExit(f"outside_table: {row['system']} {row['arm']} is not the submitted revision")
            groups.setdefault((row["system"], row["arm"]), {})[int(row["seed"])] = row
    out = {key: [group[s] for s in sorted(group)] for key, group in groups.items()}
    for system in SYSTEMS:
        first, second = (out[(system, arm)] for arm in ARMS)
        if [r["seed"] for r in first] != [str(s) for s in PANEL.SEEDS] or [r["seed"] for r in second] != [
                r["seed"] for r in first]:
            raise SystemExit(f"outside_table: {system} does not cover seeds 2000-2011 in both arms")
        for a, b in zip(first, second):
            if (a["hit"], a["evals_to_hit"], a["best"]) != (b["hit"], b["evals_to_hit"], b["best"]):
                raise SystemExit(f"outside_table: {system} seed {a['seed']} differs between the builds")
    return out


def median(group: list[dict[str, str]]) -> float:
    return PANEL.type7([float(r["outside_s"]) for r in group], 0.5)


def lines() -> list[dict[str, object]]:
    groups = rows()
    out = []
    for system in SYSTEMS:
        first, second = (groups[(system, arm)] for arm in ARMS)
        differences = [float(b["outside_s"]) - float(a["outside_s"]) for a, b in zip(first, second)]
        out.append({
            "system": system,
            "budget": int(first[0]["budget"]),
            "hits": sum(r["hit"] == "true" for r in first),
            "n": len(first),
            "first": median(first),
            "second": median(second),
            "change": median(second) / median(first) - 1.0,
            "p": STATS.wilcoxon_exact(differences)[2],
        })
    return out


def table() -> str:
    body = [" & ".join([
        PANEL.LABELS[line["system"]][1],
        PANEL.show_budget(line["budget"], True),
        f"{line['hits']}/{line['n']}",
        PANEL.show_s(line["first"]),
        PANEL.show_s(line["second"]),
        f"{100.0 * line['change']:+.1f}".replace("-", "\\(-\\)"),
        PANEL.show_p(line["p"], True),
    ]) + " \\\\" for line in lines()]
    return "\n".join(
        [
            "\\begin{table}",
            "\\caption{Outside time of the submitted revision \\texttt{61fa67cf} in one session of",
            "\\texttt{record/outside-perf3-r2} (commit \\texttt{9178884e}): the arms \\texttt{develop} and",
            "\\texttt{develop\\_prime} are two builds of that source, with the same binary digest, run on",
            "seeds 2000--2011 on core 0 of one host, alternating in blocks of four seeds.",
            "Both arms take the same calls to the same minimum on every seed.",
            "Hits are those of either arm; the next two columns are the median seconds outside the",
            "potential, Change is the second median over the first less one, and \\(p\\) is the",
            "two-sided exact Wilcoxon signed-rank test on the per-seed outside seconds.",
            "The rows are \\texttt{paper/jctc/data/outside\\_perf/rows.csv}.}",
            "\\label{tab:si-outside}",
            "\\centering",
            "\\begin{tabular}{@{}lrrrrrr@{}}",
            "\\toprule",
            "Cluster & Budget & Hits & \\texttt{develop} (s) & \\texttt{develop\\_prime} (s) & Change (\\%) & "
            "\\(p\\) \\\\",
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
            print("outside_table: the outside block of the SI is stale", file=sys.stderr)
            return 1
        return 0
    if "--print" in argv:
        print(fresh)
        return 0
    PANEL.splice(SI, MARKS, fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
