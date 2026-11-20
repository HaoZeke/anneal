#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the knockout table and the TIP4P table of the manuscript from imported rows.

Each knockout is one switch of the search, run with and without it on the same seeds
from the same starts. Its rows are copied unchanged from the record named in the
PROVENANCE file beside them, and the blob of the copy is that of the record. The arms
pair by seed: the paired outcome table, the exact McNemar test and the 95% interval of
method 10 of Newcombe, all from scripts/paired_stats.py. The TIP4P table pairs this
code with GMIN by seed in the same way and adds the exact Wilcoxon test on calls and wall.
Both arms of the template block are revisions that are ancestors of HEAD. Both arms of the
length-unit block ran the tree its rows name. That tree is the submitted revision's search
source plus the knockout switches, and the Morse driver is that tree's.

    python3 paper/jctc/scripts/knockout_tables.py            # rewrite both blocks
    python3 paper/jctc/scripts/knockout_tables.py --check    # exit 1 if either is stale
"""

from __future__ import annotations

import csv
import importlib.util
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import paired_stats  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("run_panel", ROOT / "scripts" / "gmin_panel" / "run_panel.py")
PANEL = importlib.util.module_from_spec(_SPEC)
sys.modules["run_panel"] = PANEL
_SPEC.loader.exec_module(PANEL)

MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
DATA = ROOT / "paper" / "jctc" / "data"
KNOCKOUT_MARKS = ("% knockout table: start", "% knockout table: end")
TIP4P_MARKS = ("% tip4p-gmin table: start", "% tip4p-gmin table: end")
TRUE = {"1", "true", "True"}


@dataclass(frozen=True)
class Knockout:
    key: str
    title: str
    column: str
    with_arm: str
    without_arm: str
    keep: Callable[[dict], bool]
    hit: str = "hit"
    systems: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()
    ancestors: tuple[str, ...] = ()
    tree: str = ""
    driver: str = ""


KNOCKOUTS = (
    Knockout("templates", "Structure templates, \\texttt{628773d6} against \\texttt{1c1b1238}", "revision",
             "628773d6837ea2d959a27256ff4163b21162fd4d", "1c1b12384683485cf820d3a244151b48973dc0f7",
             lambda r: r["pair"] == "release", "hit", ("m38r6", "m38r14", "sc126-38", "sc108-38"),
             ancestors=("628773d6837ea2d959a27256ff4163b21162fd4d", "1c1b12384683485cf820d3a244151b48973dc0f7")),
    Knockout("pairscale", "Length unit, the Lennard-Jones diameter against the Morse minimum, tree \\texttt{b12293b9}",
             "arm", "stock", "H2H_PAIR_SCALE=1", lambda r: r["pair"] == "pairscale-length-unit", "hit",
             ("m55r6", "m55r14", "m80r6", "m80r14"), ancestors=("dd6ff9273be30ab317bfd5508596f4beb19cb3a4",),
             tree="b12293b9d637dd32417be3aa482302e0f30dd594", driver="examples/h2h_morse.rs"),
)
# The search source of the tree the rows name, compared with the submitted revision.
SEARCH_SOURCE = ("src", "examples", "Cargo.toml", "Cargo.lock")
SUBMITTED = "61fa67cf380e46876b569da45e7cc091647a775b"
KNOCKOUT_SWITCH_FILES = (
    "examples/h2h_gmin_proof.rs",
    "examples/h2h_metal.rs",
    "examples/h2h_morse.rs",
    "src/methods/cluster_hopping.rs",
    "src/methods/cluster_hopping/config.rs",
)

NAMES = {
    "au55": "Au\\(_{55}\\)", "cu6au32": "Cu\\(_{6}\\)Au\\(_{32}\\)", "cu19au19": "Cu\\(_{19}\\)Au\\(_{19}\\)",
    "cu27au28": "Cu\\(_{27}\\)Au\\(_{28}\\)", "cu10au45": "Cu\\(_{10}\\)Au\\(_{45}\\)",
    "cu30au8": "Cu\\(_{30}\\)Au\\(_{8}\\)", "cu12au26": "Cu\\(_{12}\\)Au\\(_{26}\\)",
    "pt12cu11": "Pt\\(_{12}\\)Cu\\(_{11}\\)", "sc126-38": "SC 12--6, \\(N=38\\)", "sc108-38": "SC 10--8, \\(N=38\\)",
}


def name(system: str) -> str:
    if system in NAMES:
        return NAMES[system]
    if system.startswith("lj"):
        return f"LJ\\(_{{{system[2:]}}}\\)"
    if system.startswith("tip4p"):
        return f"(H\\(_2\\)O)\\(_{{{system[5:]}}}\\)"
    size, rho = system[1:].split("r")
    return f"M\\(_{{{size}}}\\), \\(\\rho={rho}\\)"


def budget(value: str) -> str:
    number = int(float(value))
    exponent = len(str(number)) - 1
    mantissa = number // 10**exponent
    if mantissa * 10**exponent != number:
        raise SystemExit(f"knockout_tables: budget {value}")
    return f"\\({mantissa}\\times 10^{{{exponent}}}\\)"


def show_p(p: float) -> str:
    return PANEL.show_exact_p(p, True)


def show_interval(low: float, high: float) -> str:
    return PANEL.show_interval(low, high)


def energy(value: float) -> str:
    return f"\\({value:.3f}\\)"


def read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def provenance(folder: Path) -> dict[str, str]:
    return dict(line.split(" ", 1) for line in (folder / "PROVENANCE").read_text().splitlines())


def check_blob(folder: Path) -> None:
    blob = subprocess.run(["git", "hash-object", str(folder / "rows.csv")], capture_output=True, text=True,
                          check=True, cwd=ROOT).stdout.strip()
    if blob != provenance(folder)["blob"]:
        raise SystemExit(f"knockout_tables: {folder} is not the record's blob")


def is_ancestor(revision: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"], capture_output=True,
                          cwd=ROOT).returncode == 0


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT)


def check_source(knockout: Knockout, rows: list[dict]) -> None:
    """The rows name the block's tree.

    That tree is the submitted revision's search source plus the knockout
    switches, and the driver blob is that tree's. Later commits may change
    the search source of HEAD. The rows still name the tree they ran.
    """
    if {r["revision"] for r in rows} != {knockout.tree}:
        raise SystemExit(f"knockout_tables: {knockout.key} rows do not all name {knockout.tree[:8]}")
    changed = git("diff", "--name-only", SUBMITTED, knockout.tree, "--", *SEARCH_SOURCE).stdout.split()
    if sorted(changed) != sorted(KNOCKOUT_SWITCH_FILES):
        raise SystemExit(f"knockout_tables: {knockout.tree[:8]} is not the submitted revision's "
                         f"search source plus the knockout switches")
    blob = git("rev-parse", f"{knockout.tree}:{knockout.driver}").stdout.strip()
    if {r["driver_blob"] for r in rows} != {blob}:
        raise SystemExit(f"knockout_tables: {knockout.key} rows ran another {knockout.driver}")


def check_order(knockout: Knockout, rows: list[dict]) -> None:
    """Even seeds run the With arm first and odd seeds the Without arm first."""
    first: dict[tuple[str, str], str] = {}
    for row in rows:
        first.setdefault((row["system"], row["seed"]), row[knockout.column])
    for (system, seed), arm in first.items():
        if arm != (knockout.with_arm if int(seed) % 2 == 0 else knockout.without_arm):
            raise SystemExit(f"knockout_tables: {knockout.key} {system} seed {seed} out of alternating order")


def pairs(knockout: Knockout) -> dict[str, list[tuple[dict, dict]]]:
    folder = DATA / "knockouts" / knockout.key
    check_blob(folder)
    for revision in knockout.ancestors:
        if not is_ancestor(revision):
            raise SystemExit(f"knockout_tables: {revision[:8]} is not an ancestor of HEAD")
    kept = [row for row in read(folder / "rows.csv") if knockout.keep(row)]
    if knockout.tree:
        check_source(knockout, kept)
        check_order(knockout, kept)
    arms: dict[tuple[str, str], dict[str, dict]] = {}
    for row in kept:
        if row[knockout.column] not in (knockout.with_arm, knockout.without_arm):
            continue
        seeds = arms.setdefault((row["system"], row[knockout.column]), {})
        if row["seed"] in seeds:
            raise SystemExit(f"knockout_tables: {knockout.key} {row['system']} seed {row['seed']} twice")
        seeds[row["seed"]] = row
    out = {}
    for system in knockout.systems:
        first, second = arms[(system, knockout.with_arm)], arms[(system, knockout.without_arm)]
        if sorted(first) != sorted(second):
            raise SystemExit(f"knockout_tables: {knockout.key} {system} arms ran different seeds")
        out[system] = [(first[s], second[s]) for s in sorted(first, key=int)]
    both = {s for s, arm in arms if arm == knockout.with_arm} & {s for s, arm in arms if arm == knockout.without_arm}
    if both != set(knockout.systems) | set(knockout.omitted):
        raise SystemExit(f"knockout_tables: {knockout.key} systems {sorted(both)}")
    return out


def knockout_summary(knockout: Knockout) -> dict[str, dict]:
    out = {}
    for system, rows in pairs(knockout).items():
        table = paired_stats.outcome_table([a[knockout.hit] in TRUE for a, _ in rows],
                                           [b[knockout.hit] in TRUE for _, b in rows])
        out[system] = {
            "budget": rows[0][0]["budget"], "n": len(rows), "table": table,
            "p": paired_stats.mcnemar_exact(table[1], table[2]),
            "interval": paired_stats.newcombe_paired(*table),
        }
    return out


def knockout_table() -> str:
    lines = [
        "\\begin{table*}",
        "\\caption{Two parts of the search run with and without them on the same seeds.",
        "The structure templates, from the same starts: the earlier template-assisted revision",
        "\\texttt{628773d6} (structure templates since removed in \\texttt{1c1b1238}) against its child",
        "\\texttt{1c1b1238}, both ancestors of the submitted revision.",
        "Their With arm is not the method of this article; it is the search before the removal.",
        "Each arm is \\texttt{Config::recommended}\\((38)\\) of its revision through \\texttt{h2h\\_morse} or",
        "\\texttt{h2h\\_sc}; at \\(\\rho=6\\) the Morse driver of \\texttt{628773d6} also quenches one",
        "icosahedral template before the walk, and \\texttt{1c1b1238} removes that line.",
        "The length unit: both arms run tree \\texttt{b12293b9}, the search source of the submitted revision",
        "with the default-off knockout switches of the drivers, through \\texttt{h2h\\_morse}, in alternating",
        "order by seed.",
        "With is \\texttt{Config::recommended}\\((N)\\) in the Lennard-Jones diameter it was measured in, the",
        "method of this article, and Without is \\texttt{Config::recommended\\_for\\_pair}\\((N)\\), every preset",
        "length rescaled to the Morse pair minimum.",
        "With and Without are the hits of the two arms, Only with and Only without the seeds that",
        "one arm alone hits.",
        "\\(p_\\mathrm{M}\\) is the two-sided exact McNemar test on those seeds, and \\(\\Delta\\) the 95\\%",
        "interval of Newcombe's method 10 for paired data, for the hit rate of With less that of Without.",
        "A budget counts one call per combined energy and gradient pass.",
        "The rows are \\texttt{paper/jctc/data/knockouts/templates/rows.csv} and",
        "\\texttt{paper/jctc/data/knockouts/pairscale/rows.csv}, copied from the records their",
        "\\texttt{PROVENANCE} files name.}",
        "\\label{tab:knockouts}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{@{}lrrrrrrrc@{}}",
        "\\toprule",
        "System & Budget & Seeds & With & Without & Only with & Only without & \\(p_\\mathrm{M}\\) & "
        "\\(\\Delta\\), 95\\% \\\\",
    ]
    for knockout in KNOCKOUTS:
        lines += ["\\midrule", f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{knockout.title}}}}} \\\\"]
        for system, cell in knockout_summary(knockout).items():
            both, only_with, only_without, _ = cell["table"]
            n = cell["n"]
            lines.append(
                f"{name(system)} & {budget(cell['budget'])} & {n} & {both + only_with}/{n} & "
                f"{both + only_without}/{n} & {only_with} & {only_without} & {show_p(cell['p'])} & "
                f"{show_interval(*cell['interval'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


TIP4P_SIZES = (8, 15, 20, 21)


def tip4p_pairs() -> dict[int, list[tuple[dict, dict]]]:
    folder = DATA / "tip4p_gmin"
    check_blob(folder)
    arms: dict[tuple[int, str], dict[str, dict]] = {}
    for row in read(folder / "rows.csv"):
        seeds = arms.setdefault((int(row["molecules"]), row["arm"]), {})
        if row["seed"] in seeds:
            raise SystemExit(f"knockout_tables: TIP4P {row['molecules']} seed {row['seed']} twice")
        seeds[row["seed"]] = row
    out = {}
    for size in sorted({s for s, _ in arms}):
        a, g = arms[(size, "anneal")], arms[(size, "gmin")]
        if sorted(a) != sorted(g) or any(a[s]["start_id"] != g[s]["start_id"] for s in a):
            raise SystemExit(f"knockout_tables: TIP4P {size} arms do not share starts")
        out[size] = [(a[s], g[s]) for s in sorted(a, key=int)]
    return out


def tip4p_summary() -> dict[int, dict]:
    out = {}
    for size, rows in tip4p_pairs().items():
        cell = {"n": len(rows),
                "best": [statistics.median(float(r[i]["best_energy"]) for r in rows) for i in (0, 1)],
                "low": [min(float(r[i]["best_energy"]) for r in rows) for i in (0, 1)],
                "wall": [statistics.median(float(r[i]["wall_s"]) for r in rows) for i in (0, 1)]}
        if size in TIP4P_SIZES:
            table = paired_stats.outcome_table([a["hit"] == "1" for a, _ in rows], [g["hit"] == "1" for _, g in rows])
            cell.update(table=table, p=paired_stats.mcnemar_exact(table[1], table[2]),
                        interval=paired_stats.newcombe_paired(*table),
                        calls=paired_stats.wilcoxon_exact([float(a["calls"]) - float(g["calls"]) for a, g in rows]),
                        walls=paired_stats.wilcoxon_exact([float(a["wall_s"]) - float(g["wall_s"]) for a, g in rows]))
        out[size] = cell
    return out


def tip4p_table() -> str:
    summary = tip4p_summary()
    lines = [
        "\\begin{table}",
        "\\caption{This code against GMIN on rigid TIP4P water clusters, paired by seed.",
        "This code is \\texttt{Config::recommended\\_tip4p}\\((N)\\) of revision \\texttt{e1a69a57}, after the",
        "removal of the structure templates, through \\texttt{h2h\\_seed} with the rigid water geometry of GMIN.",
        "Seeds 2000--2011, \\(2\\times 10^{6}\\) calls, both arms from the same start on each",
        "seed, in alternating order.",
        "The budget of this code counts one call per combined energy and gradient pass, and GMIN's budget counts",
        "every \\texttt{POTENTIAL} entry, including \\texttt{GRADT=false}.",
        "Hits, Only, \\(p_\\mathrm{M}\\) and \\(\\Delta\\) are as in Table~\\ref{tab:gmin};",
        "\\(p_\\mathrm{calls}\\) and \\(p_\\mathrm{wall}\\) are the two-sided exact Wilcoxon signed-rank test on",
        "the per-seed differences in calls, with a miss at the budget, and in wall.",
        "(H\\(_2\\)O)\\(_{25}\\) has no independent reference and is not scored; its row gives the lowest and",
        "the median best energy of each arm in kJ/mol.",
        "The rows are \\texttt{paper/jctc/data/tip4p\\_gmin/rows.csv}.}",
        "\\label{tab:tip4p-gmin}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrcrr@{}}",
        "\\toprule",
        "Cluster & This work & GMIN & Only, this work & Only, GMIN & \\(p_\\mathrm{M}\\) & \\(\\Delta\\), 95\\% & "
        "\\(p_\\mathrm{calls}\\) & \\(p_\\mathrm{wall}\\) \\\\",
        "\\midrule",
    ]
    for size, cell in summary.items():
        if size not in TIP4P_SIZES:
            continue
        both, only_a, only_g, _ = cell["table"]
        n = cell["n"]
        lines.append(
            f"(H\\(_2\\)O)\\(_{{{size}}}\\) & {both + only_a}/{n} & {both + only_g}/{n} & {only_a} & {only_g} & "
            f"{show_p(cell['p'])} & {show_interval(*cell['interval'])} & {show_p(cell['calls'][2])} & "
            f"{show_p(cell['walls'][2])} \\\\")
    for size, cell in summary.items():
        if size in TIP4P_SIZES:
            continue
        lines.append(
            f"(H\\(_2\\)O)\\(_{{{size}}}\\) & \\multicolumn{{8}}{{l}}{{this work {energy(cell['low'][0])}, median "
            f"{energy(cell['best'][0])}; GMIN {energy(cell['low'][1])}, median {energy(cell['best'][1])}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def splice(text: str, marks: tuple[str, str], block: str) -> str:
    start, end = text.index(marks[0]), text.index(marks[1])
    return text[: start + len(marks[0])] + "\n" + block + "\n" + text[end:]


def main(argv: list[str]) -> int:
    text = MANUSCRIPT.read_text()
    new = splice(splice(text, KNOCKOUT_MARKS, knockout_table()), TIP4P_MARKS, tip4p_table())
    if argv == ["--check"]:
        if new != text:
            print("knockout_tables: manuscript tables are stale", file=sys.stderr)
            return 1
        return 0
    MANUSCRIPT.write_text(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
