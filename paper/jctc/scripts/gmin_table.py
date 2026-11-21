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
GMIN_TARBALL_DATE = "2026-11-19"
GMIN_TARBALL_SHA256 = "ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776"
RAW_SYSTEMS = ("lj38", "lj55", "lj75", "lj98", "lj102", "lj103", "lj104", "lj150", "m80r6")
SYSTEMS = tuple(system for system in RAW_SYSTEMS if system != "lj75")
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


def gmin_row_id(data: list[dict[str, str]]) -> str:
    """The historical identifier stored in imported GMIN rows, not the source pin."""
    identifiers = {row["code_revision"] for row in data if row["code"] == "gmin"}
    if len(identifiers) != 1:
        raise SystemExit(f"gmin_table: imported GMIN rows have {len(identifiers)} source identifiers")
    return identifiers.pop()


def rows() -> list[dict[str, str]]:
    out = []
    for shard in SHARDS:
        path = DATA / shard / "rows.csv"
        if git("hash-object", str(path)) != provenance(shard)["blob"]:
            raise SystemExit(f"gmin_table: {path} is not the blob its PROVENANCE names")
        out += list(csv.DictReader(path.open()))
    if not is_ancestor(REVISION):
        raise SystemExit(f"gmin_table: {REVISION[:8]} is not an ancestor of HEAD")
    gmin_row_id(out)
    for row in out:
        if ((row["code"] == "anneal" and row["code_revision"] != REVISION)
                or row["harness_revision"] != REVISION):
            raise SystemExit(f"gmin_table: {row['system']} {row['code']} {row['seed']} is not at the recorded revisions")
        has_tarball_fields = "gmin_snapshot" in row or "tarball_sha256" in row
        if row["code"] == "gmin" and has_tarball_fields and (
                row.get("gmin_snapshot") != GMIN_TARBALL_DATE
                or row.get("tarball_sha256") != GMIN_TARBALL_SHA256):
            raise SystemExit(f"gmin_table: {row['system']} {row['seed']} is not from the dated tarball")
    host_rows = list(csv.DictReader((DATA / "b" / "host.csv").open()))
    if any(row["snapshot"] != GMIN_TARBALL_DATE or row["tarball_sha256"] != GMIN_TARBALL_SHA256
           for row in host_rows if row["code"] == "gmin"):
        raise SystemExit("gmin_table: shard b is not from the dated tarball")
    keys = [(r["system"], r["code"], r["seed"]) for r in out]
    if len(keys) != len(set(keys)) or sorted({r["system"] for r in out}) != sorted(RAW_SYSTEMS):
        raise SystemExit("gmin_table: the shards do not cover each cell once")
    return out


def develop_rows() -> list[dict[str, str]]:
    path = DEVELOP / "rows.csv"
    if git("hash-object", str(path)) != provenance("", DEVELOP)["blob"]:
        raise SystemExit(f"gmin_table: {path} is not the blob its PROVENANCE names")
    out = list(csv.DictReader(path.open()))
    if not is_ancestor(SUBMITTED):
        raise SystemExit(f"gmin_table: {SUBMITTED[:8]} is not an ancestor of HEAD")
    if gmin_row_id(out) != gmin_row_id(rows()):
        raise SystemExit("gmin_table: the two imported blocks name different GMIN source identifiers")
    for row in out:
        if ((row["code"] == "anneal" and row["code_revision"] != SUBMITTED)
                or row["harness_revision"] != DEVELOP_HARNESS):
            raise SystemExit(f"gmin_table: {row['system']} {row['code']} {row['seed']} is not at the recorded revisions")
    keys = [(r["system"], r["code"], r["seed"]) for r in out]
    if len(keys) != len(set(keys)) or sorted({r["system"] for r in out}) != sorted(DEVELOP_SYSTEMS):
        raise SystemExit("gmin_table: the develop session does not cover each cell once")
    return out


def summary() -> tuple[list[dict[str, str]], dict[str, dict[str, object]]]:
    data = rows()
    PANEL.GMIN_REV = gmin_row_id(data)
    printed = [row for row in data if row["system"] in SYSTEMS]
    return PANEL.summarise(printed, REVISION), PANEL.paired_summary(printed, REVISION)


def develop_summary() -> tuple[list[dict[str, str]], dict[str, dict[str, object]]]:
    data = develop_rows()
    PANEL.GMIN_REV = gmin_row_id(data)
    return (PANEL.summarise(data, SUBMITTED, DEVELOP_HARNESS),
            PANEL.paired_summary(data, SUBMITTED, DEVELOP_HARNESS))


def heading(text: str) -> str:
    return f"\\multicolumn{{14}}{{@{{}}l}}{{\\emph{{{text}}}}} \\\\"


def printed_lines(lines: list[dict[str, str]], paired: dict[str, dict[str, object]]) -> list[list[str]]:
    """Rows without cross-arm tests of unlike evaluation counts or capped work."""
    return [cells[:-2] for cells in PANEL.paired_lines(lines, paired)]


def table() -> str:
    lines, paired = summary()
    later, later_paired = develop_summary()
    body = [heading("Revision \\texttt{eeec8e2c}, an ancestor of the submitted revision")]
    body += [" & ".join(cells) + " \\\\" for cells in printed_lines(lines, paired)]
    body += ["\\midrule", heading("The submitted revision \\texttt{61fa67cf}")]
    body += [" & ".join(cells) + " \\\\" for cells in printed_lines(later, later_paired)]
    return "\n".join(
        [
            "\\begin{table}",
            "\\caption{Paired outcomes and reported work for this code against GMIN.",
            "The Supporting Information gives the configurations, inputs, revisions, build recipe,",
            "and row provenance. Both arms start from the same geometry and run in alternating",
            "order on seeds 2000--2011.",
            "This code uses \\texttt{Config::recommended}\\((N)\\) through \\texttt{h2h\\_gmin\\_proof}",
            "on Lennard-Jones, through \\texttt{h2h\\_morse} on Morse, and through",
            "\\texttt{h2h\\_metal} with every atom labeled gold on Au\\(_{55}\\).",
            "Every printed Lennard-Jones row runs the Wales-group LJ\\(_{38}\\)",
            "introduction file with no symmetrization keyword.",
            "Eval. is a combined energy-and-gradient pass for this code and every",
            "\\texttt{POTENTIAL} entry for GMIN, including \\texttt{GRADT=false}; these counts are",
            "different operations and are not tested across arms.",
            "The column Only counts the seeds that arm alone hits, and Both and Neither the seeds",
            "both arms and neither arm hit.",
            "Eval. and IQR are the median and interquartile range of the arm's reported count to",
            "the minimum, with a miss at its full budget.",
            "Wall, In and Out are the median seconds of the run, inside the",
            "potential and outside it.",
            "\\(p_\\mathrm{M}\\) is the two-sided exact McNemar test on the seeds only one arm hits,",
            "and \\(\\Delta\\) the 95\\% interval of Newcombe's method 10 for paired data, for the hit rate",
            "of this code less that of GMIN. No cross-arm work or wall test is reported.",
            "The rows are \\texttt{paper/jctc/data/gmin\\_eeec8e2c/*/rows.csv} and",
            "\\texttt{paper/jctc/data/gmin\\_develop/rows.csv}.}",
            "\\label{tab:gmin}",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{@{}lrlrrrrrrrrrrc@{}}",
            "\\toprule",
            "Cluster & Budget & Arm & Hits & Only & Both & Neither & Eval. & IQR & Wall & In & Out & "
            "\\(p_\\mathrm{M}\\) & \\(\\Delta\\), 95\\% \\\\",
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
