#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the two ASE tables of the manuscript from the imported rerun rows.

Every row file under paper/jctc/data/ase_rerun is copied unchanged from the
record its PROVENANCE file names, and every row of this code is a revision
that is an ancestor of HEAD. Hits, stops, caps, median walls and the Fisher
tests are recomputed here from the rows; nothing is read from a summary.

    python3 paper/jctc/scripts/ase_table.py            # rewrite both blocks
    python3 paper/jctc/scripts/ase_table.py --check    # exit 1 if either is stale
"""

from __future__ import annotations

import csv
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from paired_stats import fisher_exact  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
DATA = ROOT / "paper" / "jctc" / "data" / "ase_rerun"
SEEDS = tuple(range(2000, 2012))
ASE_VERSION = "3.29.0"
ASE_ARMS = ("ase_bh", "ase_mh")
LJ_REVISION = "61fa67cf380e46876b569da45e7cc091647a775b"
MORSE_REVISION = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
LJ = {
    "lj38": ("lj38", 400_000, -173.928427, "LJ\\(_{38}\\)"),
    "lj55": ("lj55", 1_000_000, -279.248470, "LJ\\(_{55}\\)"),
    "lj98": ("lj98", 4_000_000, -543.665361, "LJ\\(_{98}\\)"),
    "lj104": ("lj104", 4_000_000, -582.086642, "LJ\\(_{104}\\)"),
    "au55": ("au55", 1_000_000, -191.03085280, "Au\\(_{55}\\)"),
}
MORSE = {
    "m38r6": ("morse", 400_000, -157.477108, "M\\(_{38}\\), \\(\\rho=6\\)"),
    "m38r14": ("morse", 400_000, -144.321054, "M\\(_{38}\\), \\(\\rho=14\\)"),
    "m55r6": ("morse", 1_000_000, -250.286609, "M\\(_{55}\\), \\(\\rho=6\\)"),
    "m55r14": ("morse", 1_000_000, -220.646208, "M\\(_{55}\\), \\(\\rho=14\\)"),
}
LJ_ARMS = ("anneal", "recommended", *ASE_ARMS)
MORSE_ARMS = ("for_cluster", "recommended", *ASE_ARMS)
MARKS = {
    "ase": ("% ase-panel table: start", "% ase-panel table: end"),
    "ase-morse": ("% ase-nonlj table: start", "% ase-nonlj table: end"),
}


def provenance(folder: str) -> dict[str, str]:
    fields = {}
    for line in (DATA / folder / "PROVENANCE").read_text().splitlines():
        key, value = line.split(" ", 1)
        fields[key] = value
    return fields


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def rows_of(folder: str, revision: str) -> list[dict[str, str]]:
    path = DATA / folder / "rows.csv"
    stated = provenance(folder)
    if git("hash-object", str(path)).stdout.strip() != stated["blob"]:
        raise SystemExit(f"ase_table: {path} is not the blob its PROVENANCE names")
    if stated["ancestor"] != revision or git("merge-base", "--is-ancestor", revision, "HEAD").returncode:
        raise SystemExit(f"ase_table: {revision[:8]} is not an ancestor of HEAD")
    rows = list(csv.DictReader(path.open(newline="")))
    if any(row["code_revision"] != revision for row in rows):
        raise SystemExit(f"ase_table: {path} has a row of another revision")
    return rows


def group(rows: list[dict[str, str]], system: str, spec: tuple, arms: tuple[str, ...]) -> dict[str, list[dict[str, str]]]:
    _, budget, reference, _ = spec
    out = {}
    for arm in arms:
        picked = sorted((r for r in rows if r["system"] == system and r["arm"] == arm), key=lambda r: int(r["seed"]))
        if tuple(int(r["seed"]) for r in picked) != SEEDS:
            raise SystemExit(f"ase_table: {system} {arm} does not hold seeds 2000-2011 once each")
        for row in picked:
            charged = int(row["charged"])
            hit = row["hit"] == "1"
            if row["budget"] != str(budget) or not 0 < charged <= budget or hit != bool(row["evals_to_hit"]):
                raise SystemExit(f"ase_table: invalid row {row}")
            if hit != energy_hits(float(row["best"]), reference, DEFAULT_HIT_TOL + 5.0e-9):
                raise SystemExit(f"ase_table: hit flag disagrees with the reference {row}")
            stop = {"1": "hit"}.get(row["hit"], "budget" if charged == budget else "wall")
            if row["stop"] != stop or (stop == "wall" and not row["cap_s"]):
                raise SystemExit(f"ase_table: stop disagrees with the calls {row}")
            if arm in ASE_ARMS and row.get("ase_version", ASE_VERSION) != ASE_VERSION:
                raise SystemExit(f"ase_table: ASE version {row}")
        out[arm] = picked
    return out


def summary(system: str, arms: dict[str, list[dict[str, str]]], preset: str) -> dict:
    caps = {r["cap_s"] for arm in ("recommended", *ASE_ARMS) for r in arms[arm]}
    slowest = max(float(r["wall_s"]) for r in arms[preset])
    if len(caps) != 1 or abs(float(next(iter(caps))) - 3.0 * slowest) > 1.0e-5:
        raise SystemExit(f"ase_table: {system} cap is not three times the slowest {preset} wall")
    hits = {arm: sum(r["hit"] == "1" for r in rows) for arm, rows in arms.items()}
    stops = {arm: sum(r["stop"] == "wall" for r in rows) for arm, rows in arms.items()}
    if stops["recommended"]:
        raise SystemExit(f"ase_table: {system} a run of this code stopped at the cap")
    return {
        "cap": float(next(iter(caps))),
        "hits": hits,
        "stops": stops,
        "wall": {arm: statistics.median(float(r["wall_s"]) for r in rows) for arm, rows in arms.items()},
        "calls": {arm: (min(int(r["charged"]) for r in rows), max(int(r["charged"]) for r in rows))
                  for arm, rows in arms.items()},
        "p": {arm: fisher_exact(hits["recommended"], 12, hits[arm], 12) for arm in ASE_ARMS},
    }


def lj_panel() -> dict[str, dict]:
    out = {}
    for system, spec in LJ.items():
        out[system] = summary(system, group(rows_of(spec[0], LJ_REVISION), system, spec, LJ_ARMS), "anneal")
    return out


def morse_panel() -> dict[str, dict]:
    rows = rows_of("morse", MORSE_REVISION)
    if {r["system"] for r in rows} != set(MORSE):
        raise SystemExit("ase_table: the Morse rows do not cover the four clusters")
    return {system: summary(system, group(rows, system, spec, MORSE_ARMS), "for_cluster")
            for system, spec in MORSE.items()}


def show_p(p: float) -> str:
    if p >= 1.0:
        return "1"
    if p >= 0.01:
        return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def show_budget(budget: int) -> str:
    mantissa, exponent = f"{budget:.0e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def show_wall(value: float) -> str:
    return f"{value:.3f}" if value < 100.0 else f"{value:.1f}"


def body(specs: dict[str, tuple], panel: dict[str, dict]) -> list[str]:
    lines = []
    for system, (_, budget, _, label) in specs.items():
        s = panel[system]
        lines.append(" & ".join([
            label, show_budget(budget), show_wall(s["cap"]),
            *(f"{s['hits'][arm]}/12" for arm in ("recommended", *ASE_ARMS)),
            *(str(s["stops"][arm]) for arm in ASE_ARMS),
            *(show_wall(s["wall"][arm]) for arm in ("recommended", *ASE_ARMS)),
            *(show_p(s["p"][arm]) for arm in ASE_ARMS),
        ]) + " \\\\")
    return lines


def table(caption: list[str], label: str, rows: list[str]) -> str:
    return "\n".join([
        "\\begin{table}",
        *caption,
        f"\\label{{{label}}}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrrrrrrr@{}}",
        "\\toprule",
        " & & & \\multicolumn{3}{c}{Hits} & \\multicolumn{2}{c}{Stops} & \\multicolumn{3}{c}{Wall (s)} & & \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}",
        "Cluster & Budget & Cap & This code & BH & MH & BH & MH & This code & BH & MH & "
        "\\(p_\\mathrm{BH}\\) & \\(p_\\mathrm{MH}\\) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


COMMON = [
    "Seeds 2000--2011, one core of one host, budget in force calls.",
    "Cap is the wall in seconds at which a run of this code or of BH or MH stops, three times",
    "the slowest wall of the cluster preset of this code on that cluster.",
    "Stops counts the BH and MH runs that ended at the cap; those runs",
    "are misses with fewer calls than the budget, and no run of this code ended at the cap.",
    "Wall is the median over the 12 runs of the seconds each run took;",
    "a BH or MH run that ended at the cap is counted at its wall.",
    "\\(p_\\mathrm{BH}\\) and \\(p_\\mathrm{MH}\\) are the two-sided Fisher",
    "exact test on hits against each ASE arm; the test is unpaired because the random",
    "generators of the two codes do not make a shared start from one seed.",
]


def ase_block() -> str:
    caption = [
        "\\caption{This code at revision",
        f"\\texttt{{{LJ_REVISION[:8]}}}, against the basin hopping (BH) and minima hopping",
        f"(MH) of ASE {ASE_VERSION} at its constructor defaults, at the effort each run reports.",
        "This code is \\texttt{Config::recommended}\\((N)\\) through \\texttt{examples/ase\\_panel.rs} on",
        "Lennard-Jones and \\texttt{Config::recommended\\_with\\_scales} at the Gupta length and well depth",
        "on Au\\(_{55}\\), with no species labels, which is not the Au\\(_{55}\\) preset of Table~\\ref{tab:gmin}.",
        *COMMON[:-1],
        COMMON[-1],
        "The rows are \\texttt{paper/jctc/data/ase\\_rerun/*/rows.csv}.}",
    ]
    return table(caption, "tab:ase", body(LJ, lj_panel()))


def morse_block() -> str:
    caption = [
        "\\caption{This code, \\texttt{Config::recommended\\_with\\_scales} at the length scale",
        f"of the Morse pair at revision \\texttt{{{MORSE_REVISION[:8]}}} through \\texttt{{examples/ase\\_nonlj.rs}},",
        "against the basin hopping",
        f"(BH) and minima hopping (MH) of ASE {ASE_VERSION} on Morse clusters, at the effort each run reports.",
        *COMMON,
        "The rows are \\texttt{paper/jctc/data/ase\\_rerun/morse/rows.csv}.}",
    ]
    return table(caption, "tab:ase-morse", body(MORSE, morse_panel()))


def splice(text: str, marks: tuple[str, str], block: str) -> str:
    start, end = marks
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"ase_table: {MANUSCRIPT} needs one '{start}' and one '{end}'")
    head, rest = text.split(start, 1)
    return f"{head}{start}\n{block}\n{end}{rest.split(end, 1)[1]}"


def main() -> None:
    flags = sys.argv[1:]
    if flags not in ([], ["--check"]):
        raise SystemExit("usage: ase_table.py [--check]")
    text = MANUSCRIPT.read_text()
    new = splice(splice(text, MARKS["ase"], ase_block()), MARKS["ase-morse"], morse_block())
    if flags:
        if new != text:
            raise SystemExit("ase_table: the ASE tables are stale")
    else:
        MANUSCRIPT.write_text(new)


if __name__ == "__main__":
    main()
