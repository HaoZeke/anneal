# Copyright (c) anneal developers
"""Write the CREST table of the manuscript from the committed GFN2-xTB rows.

CREST runs to its own end, so a pair is kept only when the call budget of this code is
at least the median calls of CREST's complete runs. An aborted CREST run, charged no
call, is dropped from its pair. Hits are compared with the two-sided Fisher exact test
of scripts/paired_stats.py, because CREST is given no seed and its dynamics are not
paired with those of this code.

    python3 paper/jctc/scripts/crest_table.py            # rewrite the block
    python3 paper/jctc/scripts/crest_table.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from paired_stats import fisher_exact

MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
BENCH = ROOT / "bench"
MARKS = ("% crest table: start", "% crest table: end")
NH3_REFERENCE_EH = -35.470769324714
NH3_TOLERANCE_EH = 1e-6
SI10_ANNEAL_REVISION = "f3d6a050"
C20_REVISION = "5351f272"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(line for line in handle if line.strip() and not line.startswith("#")))


def hits(rows: list[dict[str, str]]) -> int:
    return sum(r["hit"] in ("true", "yes") for r in rows)


def complete(rows: list[dict[str, str]], column: str = "charged") -> list[dict[str, str]]:
    return [r for r in rows if float(r[column]) > 0]


def water_hexamer() -> dict[str, object]:
    rows = read(BENCH / "crest-equal" / "rows.csv")
    revision = next(r["code_revision"] for r in reversed(rows) if r["arm"] == "anneal")
    anneal = [r for r in rows if r["arm"] == "anneal" and r["code_revision"] == revision]
    crest = complete([r for r in rows if r["arm"] == "crest"])
    budgets = {int(r["budget"]) for r in anneal}
    if len(budgets) != 1:
        raise SystemExit(f"crest-equal: anneal budgets {sorted(budgets)}")
    return {"system": "(H\\(_2\\)O)\\(_6\\)", "budget": budgets.pop(), "anneal": anneal, "crest": crest,
            "crest_calls": [float(r["charged"]) for r in crest]}


def ammonia_octamer() -> dict[str, object]:
    results = read(BENCH / "nh3-octamer" / "results.csv")
    crest = [r for r in results if r["arm"] == "crest" and r["row"] == "v4"]
    calls = {r["seed"]: int(r["calls"]) for r in crest}
    ladder = {}
    for row in read(BENCH / "nh3-octamer" / "ladder.csv"):
        if row["seed"] in calls and int(row["budget"]) == calls[row["seed"]]:
            ladder[row["seed"]] = row
    if set(ladder) != set(calls):
        raise SystemExit(f"nh3-octamer: ladder seeds {sorted(ladder)} against CREST {sorted(calls)}")
    anneal = [dict(r, hit="true" if float(r["energy_Eh"]) <= NH3_REFERENCE_EH + NH3_TOLERANCE_EH else "false")
              for r in ladder.values()]
    return {"system": "(NH\\(_3\\))\\(_8\\)", "budget": None, "anneal": anneal, "crest": crest,
            "crest_calls": [float(r["calls"]) for r in crest]}


def covalent() -> list[dict[str, object]]:
    folder = BENCH / "covalent-h2h" / "gfn2"
    seeds = read(folder / "seeds.csv")
    rows = read(folder / "rows.csv")
    false_hits = {(r["system"], r["code"], r["code_revision"], r["seed"]) for r in read(folder / "false-hits.csv")}

    def pick(table: list[dict[str, str]], system: str, code: str, revision: str = "") -> list[dict[str, str]]:
        return [r for r in table if r["system"] == system and r["code"] == code
                and r.get("code_revision", "").startswith(revision)]

    def counted(table: list[dict[str, str]]) -> list[dict[str, str]]:
        return [dict(r, hit="false") if (r["system"], r["code"], r.get("code_revision", ""), r["seed"]) in false_hits
                else r for r in table]

    out = []
    for system, label, anneal, crest in (
            ("c10", "C\\(_{10}\\)", pick(seeds, "c10", "anneal-general"),
             pick(seeds, "c10", "crest") + pick(rows, "c10", "crest")),
            ("si10", "Si\\(_{10}\\)", pick(rows, "si10", "anneal-general", SI10_ANNEAL_REVISION),
             pick(seeds, "si10", "crest") + pick(rows, "si10", "crest")),
            ("c20", "C\\(_{20}\\)", pick(rows, "c20", "anneal-general", C20_REVISION),
             pick(rows, "c20", "crest", C20_REVISION))):
        budgets = {int(r["budget"]) for r in anneal}
        if len(budgets) != 1:
            raise SystemExit(f"{system}: anneal budgets {sorted(budgets)}")
        crest = complete(crest)
        out.append({"system": label, "budget": budgets.pop(), "anneal": counted(anneal), "crest": crest,
                    "crest_calls": [float(r["charged"]) for r in crest]})
    return out


def pairs() -> list[dict[str, object]]:
    kept = []
    for pair in (water_hexamer(), ammonia_octamer(), *covalent()):
        median = statistics.median(pair["crest_calls"])
        if pair["budget"] is not None and pair["budget"] < median:
            continue
        a, c = pair["anneal"], pair["crest"]
        pair.update(median=median, p=fisher_exact(hits(a), len(a), hits(c), len(c)))
        kept.append(pair)
    return kept


def number(value: float) -> str:
    return f"{value:.1f}" if value != int(value) else f"{int(value)}"


def budget(value: int | None) -> str:
    if value is None:
        return "CREST's"
    if value % 10000 == 0:
        mantissa, exponent = value, 0
        while mantissa % 10 == 0:
            mantissa //= 10
            exponent += 1
        return f"\\({mantissa}\\times 10^{{{exponent}}}\\)"
    return str(value)


def pvalue(p: float) -> str:
    if p >= 0.995:
        return "1"
    if p >= 0.01:
        return f"{p:.2f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def table() -> str:
    lines = [
        "\\begin{table}",
        "\\caption{This code against CREST 3.0.2 on GFN2-xTB clusters.",
        "Both arms count GFN2-xTB energy and gradient calls.",
        "CREST calls is the median calls of the complete CREST runs, and a pair is kept only",
        "when the budget of this code is at least that median; an aborted CREST run is dropped",
        "from its pair.",
        "On (NH\\(_3\\))\\(_8\\) the budget of each seed is the calls of its CREST run.",
        "\\(p\\) is the two-sided Fisher exact test on the hits.",
        "The Supporting Information gives the settings, revisions, records, and the arms that",
        "the budget rule removes.}",
        "\\label{tab:crest}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        " & & & \\multicolumn{2}{c}{Hits} & \\\\",
        "\\cmidrule(lr){4-5}",
        "Cluster & Budget & CREST calls & This work & CREST & \\(p\\) \\\\",
        "\\midrule",
    ]
    for pair in pairs():
        a, c = pair["anneal"], pair["crest"]
        lines.append(f"{pair['system']} & {budget(pair['budget'])} & {number(pair['median'])} & "
                     f"{hits(a)}/{len(a)} & {hits(c)}/{len(c)} & {pvalue(pair['p'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def block(text: str) -> str:
    start, end = MARKS
    return text.split(start, 1)[1].split(end, 1)[0].strip("\n")


def main(argv: list[str]) -> int:
    text = MANUSCRIPT.read_text()
    fresh = table()
    if "--check" in argv:
        if block(text) != fresh:
            print("crest_table: the CREST block of the manuscript is stale", file=sys.stderr)
            return 1
        return 0
    if "--print" in argv:
        print(fresh)
        return 0
    start, end = MARKS
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    MANUSCRIPT.write_text(f"{head}{start}\n{fresh}\n{end}{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
