# Copyright (c) anneal developers
"""Write the CREST table of the manuscript from the committed GFN2-xTB rows.

The printed row is the water octamer. anneal's file does not record the CREST
target or budget, so the table names each arm's own stop and prints no hit test.

    python3 paper/jctc/scripts/crest_table.py            # rewrite the block
    python3 paper/jctc/scripts/crest_table.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import csv
import io
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
BENCH = ROOT / "bench"
MARKS = ("% crest table: start", "% crest table: end")
SI10_ANNEAL_REVISION = "f3d6a050"
C20_REVISION = "5351f272"
# Twelve-seed ammonia record. The printed row is the hit rate against the six
# CREST runs already in bench/nh3-octamer/results.csv. No call column is printed.
NH3_RECORD = "d3c3e351e2bfd8c26a279943a26e5a8230467284"
NH3_RECORD_PATH = "bench/octamer-matched/nh3x8/results.csv"
# First-hit (H2O)8 records. The anneal file has no target, tolerance, or budget.
# The table prints each arm's own stop and no hit test.
H2O8_ANNEAL_COMMIT = "42e708289a718d3d916b46e5ae43f26cf0a677c1"
H2O8_ANNEAL_PATH = "bench/octamer-matched/h2o8/poi/results.csv"
H2O8_CREST_COMMIT = "7ad8eab67d61f1d5835a9ce7b67e7b2cb2b4ede9"
H2O8_CREST_PATH = "bench/h2o8-crest-firsthit/rows.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(line for line in handle if line.strip() and not line.startswith("#")))


def show_csv(commit: str, path: str) -> list[dict[str, str]]:
    """Rows of one file on a record commit. The audit checks the same object."""
    proc = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"crest_table: {commit}:{path} is not in this clone")
    return list(csv.DictReader(io.StringIO(proc.stdout)))


def hits(rows: list[dict[str, str]]) -> int:
    return sum(r["hit"] in ("true", "yes", "1") for r in rows)


def complete(rows: list[dict[str, str]], column: str = "charged") -> list[dict[str, str]]:
    return [r for r in rows if float(r[column]) > 0]


def water_hexamer() -> dict[str, object]:
    rows = read(BENCH / "crest-equal" / "rows.csv")
    revision = next(r["code_revision"] for r in reversed(rows) if r["arm"] == "anneal")
    anneal = [r for r in rows if r["arm"] == "anneal" and r["code_revision"] == revision]
    crest = complete([r for r in rows if r["arm"] == "crest"])
    budgets = {int(r["budget"]) for r in anneal}
    charged = {int(r["charged"]) for r in anneal}
    lengths = {r["crest_len_ps"] for r in crest}
    if len(budgets) != 1 or len(charged) != 1 or len(lengths) != 1:
        raise SystemExit(f"crest-equal: budgets {sorted(budgets)} charges {sorted(charged)} lengths {sorted(lengths)}")
    return {"system": "(H\\(_2\\)O)\\(_6\\)", "budget": budgets.pop(), "anneal": anneal, "crest": crest,
            "crest_calls": [float(r["charged"]) for r in crest], "charge": charged.pop(),
            "length_ps": float(lengths.pop()), "separate": True}


def water_octamer() -> dict[str, object]:
    """Hits and first-hit medians. The comparison is not a budget-rule row."""
    anneal = show_csv(H2O8_ANNEAL_COMMIT, H2O8_ANNEAL_PATH)
    crest = show_csv(H2O8_CREST_COMMIT, H2O8_CREST_PATH)
    if len(anneal) != 12 or len(crest) != 12:
        raise SystemExit(f"h2o8 first-hit: {len(anneal)} anneal rows and {len(crest)} CREST rows")
    return {
        "system": "(H\\(_2\\)O)\\(_8\\)",
        "budget": None,
        "anneal": anneal,
        "crest": crest,
        "crest_calls": [],
        "first_hit": True,
        "anneal_median": statistics.median(float(r["calls_to_hit"]) for r in anneal),
        "crest_median": statistics.median(float(r["first_hit_call"]) for r in crest),
    }


def ammonia_octamer() -> dict[str, object]:
    """Hit rate only: twelve seeds of this code against the six recorded CREST runs."""
    anneal = show_csv(NH3_RECORD, NH3_RECORD_PATH)
    crest = [r for r in read(BENCH / "nh3-octamer" / "results.csv") if r["arm"] == "crest" and r["row"] == "v4"]
    if len(anneal) != 12 or len(crest) != 6:
        raise SystemExit(f"nh3-octamer: {len(anneal)} anneal rows and {len(crest)} CREST rows")
    targets = {int(r["calls_to_hit"]) for r in anneal if r["stop_reason"] == "target"}
    misses = {int(r["calls_to_hit"]) for r in anneal if r["stop_reason"] == "budget"}
    if len(targets) != 1 or misses != {next(iter(targets)) + 1}:
        raise SystemExit(f"nh3-octamer: target counts {sorted(targets)} budget counts {sorted(misses)}")
    return {"system": "(NH\\(_3\\))\\(_8\\)", "budget": None, "anneal": anneal, "crest": crest,
            "crest_calls": [], "hits_only": True, "separate": True, "stop_calls": next(iter(targets))}


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
                    "crest_calls": [float(r["charged"]) for r in crest], "separate": True})
    return out


def pairs() -> list[dict[str, object]]:
    """The water octamer row. The arms do not share a recorded stop."""
    return [water_octamer()]


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
        "\\caption{anneal and CREST 3.0.2 on the GFN2-xTB water octamer.",
        "Each arm stays under its own recorded stop.",
        "anneal's rows do not record the CREST target or budget.",
        "There is no paired test.",
        "The Holm family for CREST is empty.",
        "The Supporting Information reports the other clusters under their own stops.}",
        "\\label{tab:crest}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}lll@{}}",
        "\\toprule",
        "Cluster & anneal stop & CREST stop \\\\",
        "\\midrule",
        "(H\\(_2\\)O)\\(_8\\) & not recorded & end of its run \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
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
