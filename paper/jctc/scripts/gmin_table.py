#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the GMIN table of the manuscript from the imported rerun rows.

The first block is the four shards of the 2026-11-19 GMIN panel rerun, whose rows of
this code are revision eeec8e2c, an ancestor of develop. The second block is the
2026-11-20 session at the submitted revision 61fa67cf on gold and Morse. Every row is
copied unchanged from the record its PROVENANCE file names, and every GMIN row is the
pinned snapshot. The summary and the paired tests are those of
scripts/gmin_panel/run_panel.py, which validates each row against its reference and
budget and takes the paired outcome table, McNemar, Newcombe and Wilcoxon from
scripts/paired_stats.py.

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
DATA = ROOT / "paper" / "jctc" / "data" / "gmin_eeec8e2c"
DEVELOP = ROOT / "paper" / "jctc" / "data" / "gmin_develop"
SHARDS = ("a", "b", "c", "d")
REVISION = "eeec8e2c216ae1506768bca8e29846c1c039ca3b"
SUBMITTED = "61fa67cf380e46876b569da45e7cc091647a775b"
DEVELOP_HARNESS = "f009acc96bb95023eeb6e394e863630c752a5d05"
DEVELOP_SYSTEMS = ("au55", "m55r6", "m55r14", "m80r6", "m80r14")
GMIN_SNAPSHOT = "9cbec1a446352bc9ef064bd2b1a5767194655919"
SYSTEMS = ("lj38", "lj55", "lj75", "lj98", "lj102", "lj103", "lj104", "lj150", "m80r6")
MARKS = PANEL.TEX_MARKS


def provenance(shard: str, folder: Path = DATA) -> dict[str, str]:
    fields = {}
    for line in (folder / shard / "PROVENANCE").read_text().splitlines():
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


def develop_rows() -> list[dict[str, str]]:
    path = DEVELOP / "rows.csv"
    if git("hash-object", str(path)) != provenance("", DEVELOP)["blob"]:
        raise SystemExit(f"gmin_table: {path} is not the blob its PROVENANCE names")
    out = list(csv.DictReader(path.open()))
    if not is_ancestor(SUBMITTED):
        raise SystemExit(f"gmin_table: {SUBMITTED[:8]} is not an ancestor of HEAD")
    for row in out:
        want = SUBMITTED if row["code"] == "anneal" else GMIN_SNAPSHOT
        if row["code_revision"] != want or row["harness_revision"] != DEVELOP_HARNESS:
            raise SystemExit(f"gmin_table: {row['system']} {row['code']} {row['seed']} is not at the pinned revisions")
    keys = [(r["system"], r["code"], r["seed"]) for r in out]
    if len(keys) != len(set(keys)) or sorted({r["system"] for r in out}) != sorted(DEVELOP_SYSTEMS):
        raise SystemExit("gmin_table: the develop session does not cover each cell once")
    return out


def summary() -> tuple[list[dict[str, str]], dict[str, dict[str, object]]]:
    PANEL.GMIN_REV = GMIN_SNAPSHOT
    data = rows()
    return PANEL.summarise(data, REVISION), PANEL.paired_summary(data, REVISION)


def develop_summary() -> tuple[list[dict[str, str]], dict[str, dict[str, object]]]:
    PANEL.GMIN_REV = GMIN_SNAPSHOT
    data = develop_rows()
    return (PANEL.summarise(data, SUBMITTED, DEVELOP_HARNESS),
            PANEL.paired_summary(data, SUBMITTED, DEVELOP_HARNESS))


def heading(text: str) -> str:
    return f"\\multicolumn{{16}}{{@{{}}l}}{{\\emph{{{text}}}}} \\\\"


def table() -> str:
    lines, paired = summary()
    later, later_paired = develop_summary()
    body = [heading("Revision \\texttt{eeec8e2c}, an ancestor of the submitted revision")]
    body += [" & ".join(cells) + " \\\\" for cells in PANEL.paired_lines(lines, paired)]
    body += ["\\midrule", heading("The submitted revision \\texttt{61fa67cf}")]
    body += [" & ".join(cells) + " \\\\" for cells in PANEL.paired_lines(later, later_paired)]
    return "\n".join(
        [
            "\\begin{table}",
            "\\caption{This code at the revision each block names against GMIN with the inputs of",
            "Section~\\ref{sec:protocol}, both arms of each cluster on one host, in alternating order by seed.",
            "On every Lennard-Jones size, LJ\\(_{75}\\) included, GMIN runs the Wales-group LJ\\(_{38}\\)",
            "introduction file with no symmetrization keyword, not the published LJ\\(_{75}\\) input.",
            "The harness clears every driver variable, so each driver runs its own preset:",
            "\\texttt{Config::recommended}\\((N)\\) through \\texttt{h2h\\_gmin\\_proof} on Lennard-Jones and",
            "through \\texttt{h2h\\_morse} on Morse, in the Lennard-Jones length unit, and on Au\\(_{55}\\)",
            "\\texttt{Config::recommended}\\((55)\\) with every atom labeled gold through \\texttt{h2h\\_metal},",
            "which applies the basin bias of equation~\\ref{eq:delta} on a stride of 8.",
            "Both revisions follow the removal of the structure templates in \\texttt{1c1b1238} and of",
            "their switches in \\texttt{5499be31}.",
            "Their search code differs only by \\texttt{plateau\\_heat}, which the preset leaves off, and",
            "M\\(_{80}\\) at \\(\\rho=6\\) takes the same calls to the same minimum on every seed at both.",
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
            "potential and outside it; the Supporting Information times the outside work of the",
            "submitted revision again on LJ\\(_{104}\\), LJ\\(_{150}\\), M\\(_{80}\\) at both \\(\\rho\\), and",
            "Au\\(_{55}\\) in a separate session with two builds of that source.",
            "\\(p_\\mathrm{M}\\) is the two-sided exact McNemar test on the seeds only one arm hits,",
            "and \\(\\Delta\\) the 95\\% interval of Newcombe's method 10 for paired data, for the hit rate",
            "of this code less that of GMIN;",
            "\\(p_\\mathrm{calls}\\) and \\(p_\\mathrm{wall}\\) are the two-sided exact Wilcoxon",
            "signed-rank test on the per-seed differences in calls, with a miss at the budget, and in",
            "the wall of the run.",
            "The rows are \\texttt{paper/jctc/data/gmin\\_eeec8e2c/*/rows.csv} and",
            "\\texttt{paper/jctc/data/gmin\\_develop/rows.csv}.}",
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
