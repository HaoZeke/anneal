# Copyright (c) anneal developers
"""Summarize an nm3 gate CSV.

Hits use a two-sided Fisher exact test. Calls to the minimum use the
two-sided Mann-Whitney test, normal approximation, with the tie
correction and the continuity correction. A miss is already stored at
the budget. Medians are Hyndman-Fan type 7.

B loses when any system has fewer hits at Fisher p < 0.05, or a higher
median call count at Mann-Whitney p < 0.05.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
SUMMARY = HERE / "summary.csv"
IDENTITY = HERE / "identity.csv"
VERDICT = HERE / "verdict.txt"
README = HERE / "README.md"
README_MARKS = ("<!-- results: start -->", "<!-- results: end -->")

SUMMARY_FIELDS = [
    "system",
    "n",
    "budget",
    "hits_a",
    "hits_b",
    "fisher_p",
    "median_calls_a",
    "median_calls_b",
    "mannwhitney_u",
    "mannwhitney_p",
    "median_pot_a_s",
    "median_pot_b_s",
    "median_outside_a_s",
    "median_outside_b_s",
    "identical_seeds",
    "hit_drop",
    "call_increase",
    "code_revision_a",
    "code_revision_b",
    "host",
]

IDENTITY_FIELDS = [
    "system",
    "seed",
    "hit_a",
    "hit_b",
    "evals_a",
    "evals_b",
    "charged_a",
    "charged_b",
    "best_a",
    "best_b",
    "bit_identical",
]


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def fisher_exact_two_sided(h1: int, n1: int, h2: int, n2: int) -> float:
    a, b = h1, n1 - h1
    c, d = h2, n2 - h2
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c
    row2 = c + d

    def table_p(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = table_p(a)
    total = 0.0
    for x in range(max(0, col1 - row2), min(row1, col1) + 1):
        px = table_p(x)
        if px <= observed * (1.0 + 1e-9):
            total += px
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """U of `left`, and a two-sided tie- and continuity-corrected p."""
    n1, n2 = len(left), len(right)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    seq = sorted(left + right)
    rank: dict[float, float] = {}
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        rank[seq[i]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r1 = sum(rank[value] for value in left)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    if len(rank) == 1:
        return u1, 1.0
    n = n1 + n2
    tie = 0
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        t = j - i + 1
        tie += t**3 - t
        i = j + 1
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0.0:
        return u1, 1.0
    numer = u1 - n1 * n2 / 2.0
    if numer != 0.0:
        numer -= math.copysign(0.5, numer)
    z = numer / math.sqrt(var)
    return u1, math.erfc(abs(z) / math.sqrt(2.0))


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group(rows: list[dict[str, str]], system: str, arm: str) -> list[dict[str, str]]:
    chosen = [row for row in rows if row["system"] == system and row["arm"] == arm]
    return sorted(chosen, key=lambda row: int(row["seed"]))


def num(text: str) -> str:
    value = float(text)
    if math.isnan(value):
        return ""
    return format(value, ".10g")


def analyse(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    systems = list(dict.fromkeys(row["system"] for row in rows))
    summary: list[dict[str, str]] = []
    identity: list[dict[str, str]] = []
    drops: list[str] = []
    increases: list[str] = []
    loses = False
    incomplete: list[str] = []
    for system in systems:
        left = group(rows, system, "A")
        right = group(rows, system, "B")
        if len(left) != len(right) or not left:
            incomplete.append(system)
            continue
        seeds_l = [row["seed"] for row in left]
        seeds_r = [row["seed"] for row in right]
        if seeds_l != seeds_r:
            incomplete.append(system)
            continue
        hits_a = sum(row["hit"] == "1" for row in left)
        hits_b = sum(row["hit"] == "1" for row in right)
        n = len(left)
        calls_a = [float(row["evals_to_hit"]) for row in left]
        calls_b = [float(row["evals_to_hit"]) for row in right]
        fisher_p = fisher_exact_two_sided(hits_a, n, hits_b, n)
        u, mw_p = mann_whitney(calls_a, calls_b)
        med_a = type7(calls_a)
        med_b = type7(calls_b)
        hit_drop = int(hits_b < hits_a)
        call_increase = int(med_b > med_a)
        if hit_drop:
            drops.append(f"{system} hits {hits_b}/{n} against {hits_a}/{n} Fisher p {fisher_p:.6g}")
            if fisher_p < 0.05:
                loses = True
        if call_increase:
            increases.append(
                f"{system} median calls {med_b:.6g} against {med_a:.6g} Mann-Whitney p {mw_p:.6g}"
            )
            if mw_p < 0.05:
                loses = True
        identical = 0
        for a, b in zip(left, right):
            same = int(
                a["hit"] == b["hit"]
                and a["evals_to_hit"] == b["evals_to_hit"]
                and a["charged"] == b["charged"]
                and a["best"] == b["best"]
            )
            identical += same
            identity.append(
                {
                    "system": system,
                    "seed": a["seed"],
                    "hit_a": a["hit"],
                    "hit_b": b["hit"],
                    "evals_a": a["evals_to_hit"],
                    "evals_b": b["evals_to_hit"],
                    "charged_a": a["charged"],
                    "charged_b": b["charged"],
                    "best_a": a["best"],
                    "best_b": b["best"],
                    "bit_identical": str(same),
                }
            )
        hosts = {row["host"] for row in left + right}
        summary.append(
            {
                "system": system,
                "n": str(n),
                "budget": left[0]["budget"],
                "hits_a": str(hits_a),
                "hits_b": str(hits_b),
                "fisher_p": num(fisher_p),
                "median_calls_a": num(med_a),
                "median_calls_b": num(med_b),
                "mannwhitney_u": num(u),
                "mannwhitney_p": num(mw_p),
                "median_pot_a_s": num(type7([float(row["pot_s"]) for row in left])),
                "median_pot_b_s": num(type7([float(row["pot_s"]) for row in right])),
                "median_outside_a_s": num(type7([float(row["outside_s"]) for row in left])),
                "median_outside_b_s": num(type7([float(row["outside_s"]) for row in right])),
                "identical_seeds": str(identical),
                "hit_drop": str(hit_drop),
                "call_increase": str(call_increase),
                "code_revision_a": left[0]["code_revision"],
                "code_revision_b": right[0]["code_revision"],
                "host": hosts.pop() if len(hosts) == 1 else " ".join(sorted(hosts)),
            }
        )
    if incomplete:
        verdict = "INCOMPLETE " + " ".join(incomplete)
    elif loses:
        verdict = "B LOSES"
    else:
        verdict = "PASS"
    lines = [verdict]
    if drops:
        lines.append("hit drops:")
        lines.extend(drops)
    else:
        lines.append("hit drops: none")
    if increases:
        lines.append("call increases:")
        lines.extend(increases)
    else:
        lines.append("call increases: none")
    return summary, identity, "\n".join(lines) + "\n"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def splice_readme(text: str) -> None:
    if not README.is_file():
        return
    body = README.read_text()
    start, end = README_MARKS
    if start not in body or end not in body:
        return
    head, rest = body.split(start, 1)
    _old, tail = rest.split(end, 1)
    README.write_text(f"{head}{start}\n{text.rstrip()}\n{end}{tail}")


def self_check() -> None:
    fisher = fisher_exact_two_sided(12, 12, 7, 12)
    if abs(fisher - 0.037267080745341616) > 1e-15:
        raise SystemExit(f"fisher self-check {fisher}")
    if abs(fisher_exact_two_sided(12, 12, 0, 12) - 7.39602301050679e-07) > 1e-18:
        raise SystemExit("fisher extreme self-check")
    u, p = mann_whitney([10.0, 20.0, 30.0], [11.0, 21.0, 31.0])
    if u != 3.0 or abs(p - 0.6625205835400574) > 1e-12:
        raise SystemExit(f"mann-whitney self-check {u} {p}")
    u, p = mann_whitney([1.0, 1.0, 2.0, 3.0, 5.0, 8.0], [1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    if u != 11.0 or abs(p - 0.29283365264601635) > 1e-12:
        raise SystemExit(f"mann-whitney tie self-check {u} {p}")
    _u, p = mann_whitney([4.0] * 6, [4.0] * 6)
    if p != 1.0:
        raise SystemExit(f"all-tie self-check {p}")
    if type7([1, 2, 3, 4]) != 2.5:
        raise SystemExit("type7 self-check")


def main() -> None:
    self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return
    rows = load(args.rows)
    summary, identity, verdict = analyse(rows)
    write_csv(SUMMARY, SUMMARY_FIELDS, summary)
    write_csv(IDENTITY, IDENTITY_FIELDS, identity)
    VERDICT.write_text(verdict)
    splice_readme(verdict + "\n" + _table(summary))
    sys.stdout.write(verdict)


def _table(summary: list[dict[str, str]]) -> str:
    lines = [
        "system hits_A hits_B fisher median_calls_A median_calls_B mannwhitney "
        "pot_A pot_B outside_A outside_B identical",
    ]
    for row in summary:
        lines.append(
            " ".join(
                [
                    row["system"],
                    f"{row['hits_a']}/{row['n']}",
                    f"{row['hits_b']}/{row['n']}",
                    row["fisher_p"],
                    row["median_calls_a"],
                    row["median_calls_b"],
                    row["mannwhitney_p"],
                    row["median_pot_a_s"],
                    row["median_pot_b_s"],
                    row["median_outside_a_s"],
                    row["median_outside_b_s"],
                    f"{row['identical_seeds']}/{row['n']}",
                ]
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
