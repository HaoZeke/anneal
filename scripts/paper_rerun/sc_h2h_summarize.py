#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write summary.csv and README.md from the Sutton-Chen rerun rows.

Fisher is the two-sided exact test on hits. Mann-Whitney is the two-sided
normal approximation with the tie correction and the continuity correction,
on calls, with a miss ranked after every hit. Medians are Hyndman-Fan type 7.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "metal_h2h"))

import run_sc as sc  # noqa: E402

ROWS = ROOT / "bench" / "paper-rerun" / "sc-h2h" / "rows.csv"
SUMMARY = ROOT / "bench" / "paper-rerun" / "sc-h2h" / "summary.csv"
README = ROOT / "bench" / "paper-rerun" / "sc-h2h" / "README.md"
PRINTED = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12" / "README.md"
FIELDS = (
    "system",
    "arm",
    "n",
    "hits",
    "hits_over_n",
    "budget",
    "reference",
    "calls_median",
    "fisher_p_vs_gmin",
    "mannwhitney_p_calls",
)


def type7(sorted_values: list[float], p: float) -> float:
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    from math import comb

    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b

    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        return comb(aa + bb, aa) * comb(cc + dd, cc) / comb(aa + bb + cc + dd, aa + cc)

    total = a + b + c + d
    row = a + b
    col = a + c
    observed = prob(a, b, c, d)
    score = 0.0
    for aa in range(max(0, row + col - total), min(row, col) + 1):
        bb = row - aa
        cc = col - aa
        dd = total - row - cc
        value = prob(aa, bb, cc, dd)
        if value <= observed + 1e-15:
            score += value
    return score


def mann_whitney(left: list[float], right: list[float]) -> float:
    """Two-sided Mann-Whitney, tie correction, continuity correction."""
    n1 = len(left)
    n2 = len(right)
    if n1 == 0 or n2 == 0:
        return float("nan")
    combined = [(value, 0) for value in left] + [(value, 1) for value in right]
    combined.sort(key=lambda item: item[0])
    ranks = [0.0] * len(combined)
    tie_term = 0.0
    index = 0
    while index < len(combined):
        end = index
        while end < len(combined) and combined[end][0] == combined[index][0]:
            end += 1
        average = 0.5 * ((index + 1) + end)
        for slot in range(index, end):
            ranks[slot] = average
        width = end - index
        if width > 1:
            tie_term += width**3 - width
        index = end
    rank_sum = sum(rank for rank, (_, group) in zip(ranks, combined) if group == 0)
    u1 = rank_sum - n1 * (n1 + 1) / 2.0
    total = n1 + n2
    var = n1 * n2 / 12.0 * ((total + 1) - tie_term / (total * (total - 1)))
    if var <= 0.0:
        return 1.0
    z = (abs(u1 - n1 * n2 / 2.0) - 0.5) / math.sqrt(var)
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def ranked(group: list[dict[str, str]]) -> list[float]:
    """Calls, with a miss strictly worse than a hit at the full budget."""
    out = []
    for row in group:
        calls = float(row["evals_to_hit"])
        if row["hit"] != "1":
            calls = float(row["budget"]) + 1.0
        out.append(calls)
    return out


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def printed_table(path: Path) -> list[dict[str, str]]:
    found = []
    for line in path.read_text().splitlines():
        if not line.startswith("| sc"):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[1] not in {"anneal", "gmin"} or "/" not in cells[3]:
            continue
        found.append(
            {
                "system": cells[0],
                "arm": cells[1],
                "harness": cells[2],
                "hits": cells[3],
                "calls": cells[4],
            }
        )
    return found


def calls_text(value: str) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def show(value: float) -> str:
    if math.isnan(value):
        return ""
    text = f"{value:.6g}"
    return text


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        grouped.setdefault(row["system"], {}).setdefault(row["arm"], []).append(row)
    out = []
    for system in sc.PAPER_BUDGETS:
        arms = grouped.get(system, {})
        anneal = arms.get("anneal", [])
        gmin = arms.get("gmin", [])
        tests = {"fisher": "", "calls": ""}
        if anneal and gmin and len(anneal) == len(gmin):
            tests = {
                "fisher": show(
                    fisher(
                        sum(row["hit"] == "1" for row in anneal),
                        len(anneal),
                        sum(row["hit"] == "1" for row in gmin),
                        len(gmin),
                    )
                ),
                "calls": show(mann_whitney(ranked(anneal), ranked(gmin))),
            }
        for arm in ("anneal", "gmin"):
            group = arms.get(arm, [])
            if not group:
                continue
            hits = sum(row["hit"] == "1" for row in group)
            out.append(
                {
                    "system": system,
                    "arm": arm,
                    "n": str(len(group)),
                    "hits": str(hits),
                    "hits_over_n": f"{hits}/{len(group)}",
                    "budget": str(sc.PAPER_BUDGETS[system]),
                    "reference": f"{sc.REFERENCES[system]:.4f}",
                    "calls_median": f"{median([float(row['evals_to_hit']) for row in group]):.6f}",
                    "fisher_p_vs_gmin": tests["fisher"] if arm == "anneal" else "",
                    "mannwhitney_p_calls": tests["calls"] if arm == "anneal" else "",
                }
            )
    return out


def readme(rows: list[dict[str, str]], summary: list[dict[str, str]], printed: list[dict[str, str]]) -> str:
    by_key = {(row["system"], row["arm"]): row for row in summary}
    lines = [
        "# Sutton-Chen head-to-head rerun",
        "",
        "Anneal against GMIN on the printed cells: reduced Sutton-Chen 12-6, 10-8 and 9-6",
        "at 38, 55 and 80 atoms, seeds 2000-2011, at `PAPER_BUDGETS` in",
        "`scripts/metal_h2h/run_sc.py`. One session, serial, ABBA by seed, cpu 0.",
        "",
        "A hit is best <= reference + 0.001 (`scripts/target_tol.py`, `DEFAULT_HIT_TOL`).",
        "Fisher is two-sided exact on hits. Mann-Whitney is two-sided on calls with the",
        "tie correction and the continuity correction, and a miss ranked worse than a hit",
        "at the full budget. The calls median is type 7 of the stored call counts.",
        "",
        "The GMIN data file is `scripts/metal_h2h/run_sc.py` `data_text`: the container",
        "left to GMIN, the Doye-Wales temperature, and the step 0.40 sigma in units of a.",
        "",
        "| System | Arm | Hits | Calls median | Fisher p | Mann-Whitney p calls |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            "| {system} | {arm} | {hits_over_n} | {calls_median} | {fisher_p_vs_gmin} | {mannwhitney_p_calls} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Printed counts are the cells in `bench/metal-cluster-h2h/sc-n12/README.md`.",
            "Two GMIN rows are printed for each system: the earlier data file and the corrected file.",
            "",
            "| System | Arm | Printed harness | Printed hits | Printed calls | New hits | New calls |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for cell in printed:
        new = by_key.get((cell["system"], cell["arm"]))
        lines.append(
            "| {system} | {arm} | {harness} | {hits} | {calls} | {new_hits} | {new_calls} |".format(
                system=cell["system"],
                arm=cell["arm"],
                harness=cell["harness"],
                hits=cell["hits"],
                calls=cell["calls"],
                new_hits="" if new is None else new["hits_over_n"],
                new_calls="" if new is None else calls_text(new["calls_median"]),
            )
        )
    lines.append("")
    if rows:
        lines.append(f"Rows: {len(rows)}.")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    rows = load(ROWS)
    summary = summarise(rows)
    printed = printed_table(PRINTED)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    README.write_text(readme(rows, summary, printed))
    print(f"wrote {SUMMARY}")
    print(f"wrote {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
