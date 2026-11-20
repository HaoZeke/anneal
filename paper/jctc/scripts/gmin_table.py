#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the GMIN table of the manuscript from the imported rerun rows.

The rows are the four shards of the 2026-11-19 GMIN panel rerun, copied unchanged
from the records their PROVENANCE files name. Every row of this code is revision
eeec8e2c, an ancestor of develop, and every GMIN row is the pinned snapshot. The
summary and the paired tests are those of scripts/gmin_panel/run_panel.py, which
validates each row against its reference and budget and takes the paired outcome
table, McNemar, Newcombe and Wilcoxon from scripts/paired_stats.py.

    python3 paper/jctc/scripts/gmin_table.py            # rewrite the block
    python3 paper/jctc/scripts/gmin_table.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("run_panel", ROOT / "scripts" / "gmin_panel" / "run_panel.py")
PANEL = importlib.util.module_from_spec(_SPEC)
sys.modules["run_panel"] = PANEL
_SPEC.loader.exec_module(PANEL)

MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
DATA = ROOT / "paper" / "jctc" / "data" / "gmin_rerun"
SHARDS = ("a", "b", "c", "d")
REVISION = "eeec8e2c216ae1506768bca8e29846c1c039ca3b"
GMIN_SNAPSHOT = "9cbec1a446352bc9ef064bd2b1a5767194655919"
SYSTEMS = ("lj38", "lj55", "lj75", "lj98", "lj102", "lj103", "lj104", "lj150", "m80r6")
MARKS = PANEL.TEX_MARKS


def provenance(shard: str) -> dict[str, str]:
    fields = {}
    for line in (DATA / shard / "PROVENANCE").read_text().splitlines():
        key, value = line.split(" ", 1)
        fields[key] = value
    return fields


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
                          capture_output=True).returncode == 0


def rows() -> list[dict[str, str]]:
    out = []
    for shard in SHARDS:
        path = DATA / shard / "rows.csv"
        if git("hash-object", str(path)) != provenance(shard)["blob"]:
            raise SystemExit(f"gmin_table: {path} is not the blob its PROVENANCE names")
        out += list(csv.DictReader(path.open()))
    if not is_ancestor(REVISION):
        raise SystemExit(f"gmin_table: {REVISION[:8]} is not an ancestor of HEAD")
    for row in out:
        want = REVISION if row["code"] == "anneal" else GMIN_SNAPSHOT
        if row["code_revision"] != want or row["harness_revision"] != REVISION:
            raise SystemExit(f"gmin_table: {row['system']} {row['code']} {row['seed']} is not at the pinned revisions")
    keys = [(r["system"], r["code"], r["seed"]) for r in out]
    if len(keys) != len(set(keys)) or sorted({r["system"] for r in out}) != sorted(SYSTEMS):
        raise SystemExit("gmin_table: the shards do not cover each cell once")
    return out


def summary() -> tuple[list[dict[str, str]], dict[str, dict[str, object]]]:
    PANEL.GMIN_REV = GMIN_SNAPSHOT
    data = rows()
    return PANEL.summarise(data, REVISION), PANEL.paired_summary(data, REVISION)


def table() -> str:
    lines, paired = summary()
    body = [" & ".join(cells) + " \\\\" for cells in PANEL.paired_lines(lines, paired)]
    return "\n".join(
        [
            "\\begin{table}",
            "\\caption{This code at revision \\texttt{eeec8e2c}, an ancestor of the submitted revision, against",
            "GMIN with the inputs of Section~\\ref{sec:protocol}, both arms of each cluster on one host,",
            "in alternating order by seed.",
            "GMIN is the snapshot of SVNREV \\texttt{9cbec1a4} built with three patches,",
            "\\texttt{bench/lj104-lattice/v2/h2h-gmin.patch}, \\texttt{scripts/morse\\_h2h/morse-timer.patch},",
            "and \\texttt{bench/metal-cluster-h2h/gmin-bgupta.patch}, which cap the force calls, time the",
            "potential, and, in the serial build, set the run index (\\texttt{JP=1}) before its first use.",
            "Seeds 2000--2011, one core, budget in force calls; both arms of a seed start from the",
            "same geometry.",
            "The column Only counts the seeds that arm alone hits, and Both and Neither the seeds",
            "both arms and neither arm hit.",
            "Calls and IQR are the median and the interquartile range of the",
            "calls to the minimum, with a miss at the full budget.",
            "Wall, In and Out are the median seconds of the run, inside the",
            "potential and outside it.",
            "\\(p_\\mathrm{M}\\) is the two-sided exact McNemar test on the seeds only one arm hits,",
            "and \\(\\Delta\\) the 95\\% interval of Newcombe's method 10 for paired data, for the hit rate",
            "of this code less that of GMIN;",
            "\\(p_\\mathrm{calls}\\) and \\(p_\\mathrm{wall}\\) are the two-sided exact Wilcoxon",
            "signed-rank test on the per-seed differences in calls, with a miss at the budget, and in",
            "the wall of the run.",
            "The rows are \\texttt{paper/jctc/data/gmin\\_rerun/*/rows.csv}.}",
            "\\label{tab:gmin}",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{@{}lrlrrrrrrrrrrcrr@{}}",
            "\\toprule",
            "Cluster & Budget & Arm & Hits & Only & Both & Neither & Calls & IQR & Wall & In & Out & "
            "\\(p_\\mathrm{M}\\) & \\(\\Delta\\), 95\\% & \\(p_\\mathrm{calls}\\) & \\(p_\\mathrm{wall}\\) \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
        ]
    )


def block(text: str) -> str:
    start, end = MARKS
    return text.split(start, 1)[1].split(end, 1)[0].strip("\n")


def main(argv: list[str]) -> int:
    text = MANUSCRIPT.read_text()
    fresh = table()
    if "--check" in argv:
        if block(text) != fresh:
            print("gmin_table: the GMIN block of the manuscript is stale", file=sys.stderr)
            return 1
        return 0
    if "--print" in argv:
        print(fresh)
        return 0
    PANEL.splice(MANUSCRIPT, MARKS, fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
