#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write tab:si-morse-g1 of the SI: the submitted revision alone on ten more Morse clusters.

The rows are paper/jctc/data/gmin_g1/rows.csv, copied unchanged from the commit its
PROVENANCE names: Config::recommended(N) of 61fa67cf through h2h_morse, seeds
2000-2011, with the systems and budgets of scripts/morse_gmin/run_panel.py and
scripts/morse_panel/run_panel.py. The record also pairs these rows with stored GMIN rows
of earlier sessions; those are not read here, because they name no machine for any
cluster and no GMIN revision for half of them.
Every hit is checked against the reference energy of the run script.

    python3 paper/jctc/scripts/morse_g1_table.py            # rewrite the block
    python3 paper/jctc/scripts/morse_g1_table.py --check    # exit 1 if it is stale
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
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SI = TABLE.ROOT / "paper" / "jctc" / "si.tex"
DATA = TABLE.ROOT / "paper" / "jctc" / "data" / "gmin_g1"
SYSTEMS = ("m30r6", "m30r14", "m38r6", "m38r14", "m70r3", "m70r6", "m70r10", "m70r14", "m80r3", "m80r10")
RUN_SCRIPTS = ("scripts/morse_gmin/run_panel.py", "scripts/morse_panel/run_panel.py")
MARKS = ("% morse-g1 table: start", "% morse-g1 table: end")


def specs() -> dict[str, tuple[int, float, int, float]]:
    out: dict[str, tuple[int, float, int, float]] = {}
    for index, path in enumerate(RUN_SCRIPTS):
        spec = importlib.util.spec_from_file_location(f"morse_run_{index}", TABLE.ROOT / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, n, rho, budget, reference in module.SYSTEMS:
            if out.setdefault(name, (n, rho, budget, reference)) != (n, rho, budget, reference):
                raise SystemExit(f"morse_g1_table: the run scripts disagree on {name}")
    return out


def label(name: str) -> str:
    n, rho, _budget, _reference = specs()[name]
    return f"M\\(_{{{n}}}\\), \\(\\rho={rho:g}\\)"


def rows() -> dict[str, list[dict[str, str]]]:
    path = DATA / "rows.csv"
    fields = TABLE.provenance("", DATA)
    if TABLE.git("hash-object", str(path)) != fields["blob"]:
        raise SystemExit(f"morse_g1_table: {path} is not the blob its PROVENANCE names")
    if TABLE.git("rev-parse", f"{fields['commit']}^{{tree}}") != fields["tree"]:
        raise SystemExit("morse_g1_table: the record commit is not the tree its PROVENANCE names")
    if fields["ancestor"] != TABLE.SUBMITTED or not TABLE.is_ancestor(TABLE.SUBMITTED):
        raise SystemExit("morse_g1_table: the rows are not of the submitted revision")
    table = specs()
    groups: dict[str, dict[int, dict[str, str]]] = {}
    hosts = set()
    for row in csv.DictReader(path.open()):
        name = row["system"]
        _n, _rho, budget, reference = table[name]
        if (row["code"], row["code_revision"], row["config"]) != ("anneal", TABLE.SUBMITTED, "recommended"):
            raise SystemExit(f"morse_g1_table: {name} {row['seed']} is not the stock preset of the submitted revision")
        if int(row["budget"]) != budget or int(row["charged"]) > budget:
            raise SystemExit(f"morse_g1_table: {name} {row['seed']} has another budget than the run script")
        hit = row["hit"] == "1"
        if row["hit"] not in ("0", "1") or hit != energy_hits(float(row["best"]), reference, DEFAULT_HIT_TOL):
            raise SystemExit(f"morse_g1_table: {name} {row['seed']} scores against another reference")
        if not hit and int(row["evals_to_hit"]) != budget:
            raise SystemExit(f"morse_g1_table: {name} {row['seed']} is a miss below the budget")
        hosts.add(row["host"])
        groups.setdefault(name, {})[int(row["seed"])] = row
    if len(hosts) != 1:
        raise SystemExit("morse_g1_table: the rows come from more than one host")
    if sorted(groups) != sorted(SYSTEMS) or any(sorted(g) != list(PANEL.SEEDS) for g in groups.values()):
        raise SystemExit("morse_g1_table: the rows are not seeds 2000-2011 on each of the ten clusters")
    return {name: [groups[name][s] for s in PANEL.SEEDS] for name in SYSTEMS}


def lines() -> list[dict[str, object]]:
    out = []
    for name, group in rows().items():
        calls = [float(r["evals_to_hit"]) for r in group]
        out.append({
            "system": name,
            "budget": int(group[0]["budget"]),
            "hits": sum(r["hit"] == "1" for r in group),
            "n": len(group),
            "calls": PANEL.type7(calls, 0.5),
            "iqr": PANEL.type7(calls, 0.75) - PANEL.type7(calls, 0.25),
            "wall": PANEL.type7([float(r["wall_s"]) for r in group], 0.5),
            "inside": PANEL.type7([float(r["pot_s"]) for r in group], 0.5),
            "outside": PANEL.type7([float(r["outside_s"]) for r in group], 0.5),
        })
    return out


def table() -> str:
    body = [" & ".join([
        label(line["system"]),
        PANEL.show_budget(line["budget"], True),
        f"{line['hits']}/{line['n']}",
        PANEL.show_calls(line["calls"]),
        PANEL.show_calls(line["iqr"]),
        PANEL.show_s(line["wall"]),
        PANEL.show_s(line["inside"]),
        PANEL.show_s(line["outside"]),
    ]) + " \\\\" for line in lines()]
    return "\n".join(
        [
            "\\begin{table}",
            "\\caption{The submitted revision \\texttt{61fa67cf} alone on ten Morse clusters that",
            "the GMIN table of the main text does not run, from commit \\texttt{42562693} of",
            "\\texttt{record/g1-develop-stock}: \\texttt{Config::recommended}\\((N)\\) through",
            "\\texttt{h2h\\_morse} with no driver variable but the budget, in the Lennard-Jones length",
            "unit, seeds 2000--2011.",
            "A hit is a lowest energy within \\(10^{-3}\\) of the reference of the run script.",
            "Calls and IQR are the median and the interquartile range of the calls to the minimum,",
            "with a miss at the full budget; Wall, In and Out are the median seconds of the run,",
            "inside the potential and outside it.",
            "The rows are \\texttt{paper/jctc/data/gmin\\_g1/rows.csv}.}",
            "\\label{tab:si-morse-g1}",
            "\\centering",
            "\\begin{tabular}{@{}lrrrrrrr@{}}",
            "\\toprule",
            "Cluster & Budget & Hits & Calls & IQR & Wall & In & Out \\\\",
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
            print("morse_g1_table: the Morse block of the SI is stale", file=sys.stderr)
            return 1
        return 0
    if "--print" in argv:
        print(fresh)
        return 0
    PANEL.splice(SI, MARKS, fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
