# Copyright (c) anneal developers
"""Tables and the verdict of the A/B gate, from the stored rows.

Each system has twelve seeds on arm A and on arm B. Fisher is the two-sided
exact test on hits: the probability of every table with the observed margins
that is no more likely than the observed one. Mann-Whitney is two-sided on
the calls to the minimum, a miss entered at its budget: the normal
approximation with the tie and continuity corrections, and every value tied
gives p = 1. Medians are Hyndman-Fan type 7.

A crashed run is a miss at its budget and is left out of the times. A
system's seeds are bit-identical when both arms report the same hit, calls,
charged calls and best energy for every seed; the binary Lennard-Jones
driver also reports the IEEE-754 bits of the best energy and the other
drivers report it to eight decimals.

B loses when any system has fewer hits with Fisher p < 0.05 or a higher
median of calls with Mann-Whitney p < 0.05. Otherwise it passes.

    summarize_panel.py --rows rows.csv --summary summary.csv --pairs pairs.csv
        [--readme README.md]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import run_panel

ALPHA = 0.05
BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"


def type7(values: list[float], p: float) -> float:
    """Hyndman-Fan type 7."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided exact p, summing the tables no more likely than the observed."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """U of ``left`` and the two-sided p, normal approximation with the tie
    and continuity corrections. Every value tied gives p = 1."""
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def pp(value: float) -> str:
    if value >= 0.995:
        return "1"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[int, dict[str, str]]]]:
    """System, arm, seed to row. A repeated key is a corrupt table."""
    table: dict[str, dict[str, dict[int, dict[str, str]]]] = {}
    for row in rows:
        seed = int(row["seed"])
        slot = table.setdefault(row["system"], {}).setdefault(row["arm"], {})
        if seed in slot:
            raise SystemExit(f"{row['system']} arm {row['arm']} seed {seed} is stored twice")
        slot[seed] = row
    return table


def identity(row: dict[str, str]) -> tuple[str, ...]:
    return (row["status"], row["hit"], row["evals_to_hit"], row["charged"], row["best"], row.get("best_bits", ""))


def medians(rows: list[dict[str, str]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r["status"] == "ok"]
    return type7(values, 0.5) if values else None


def ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0.0:
        return None
    return num / den


def fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def summarise(rows: list[dict[str, str]], seeds: tuple[int, ...] = run_panel.SEEDS) -> list[dict[str, str]]:
    """One record per system with every seed stored on both arms."""
    table = grouped(rows)
    out = []
    for system, driver, reference in run_panel.PANEL:
        arms = table.get(system, {})
        if any(set(arms.get(arm, {})) != set(seeds) for arm in run_panel.ARMS):
            continue
        a = [arms["A"][s] for s in seeds]
        b = [arms["B"][s] for s in seeds]
        n = len(seeds)
        a_hits = sum(r["hit"] == "1" for r in a)
        b_hits = sum(r["hit"] == "1" for r in b)
        p_hits = fisher(a_hits, n, b_hits, n)
        a_calls = [float(r["evals_to_hit"]) for r in a]
        b_calls = [float(r["evals_to_hit"]) for r in b]
        u_b, p_calls = mann_whitney(b_calls, a_calls)
        a_med = type7(a_calls, 0.5)
        b_med = type7(b_calls, 0.5)
        centre = n * n / 2.0
        if u_b > centre:
            rank_side = "B higher"
        elif u_b < centre:
            rank_side = "B lower"
        else:
            rank_side = "even"
        same = sum(identity(x) == identity(y) for x, y in zip(a, b, strict=True))
        a_pot, b_pot = medians(a, "pot_s"), medians(b, "pot_s")
        a_out, b_out = medians(a, "outside_s"), medians(b, "outside_s")
        a_wall, b_wall = medians(a, "wall_s"), medians(b, "wall_s")
        lose_hits = b_hits < a_hits and p_hits < ALPHA
        lose_calls = b_med > a_med and p_calls < ALPHA
        out.append(
            {
                "system": system,
                "driver": driver,
                "budget": a[0]["budget"],
                "reference": f"{reference}",
                "seeds": str(n),
                "a_hits": str(a_hits),
                "b_hits": str(b_hits),
                "fisher_p": f"{p_hits:.6g}",
                "a_calls_median": f"{a_med:.1f}",
                "b_calls_median": f"{b_med:.1f}",
                "mw_p_calls": f"{p_calls:.6g}",
                "calls_rank_side": rank_side,
                "a_pot_median_s": fmt(a_pot, ".3f"),
                "b_pot_median_s": fmt(b_pot, ".3f"),
                "pot_ratio_b_over_a": fmt(ratio(b_pot, a_pot), ".3f"),
                "a_outside_median_s": fmt(a_out, ".3f"),
                "b_outside_median_s": fmt(b_out, ".3f"),
                "outside_ratio_b_over_a": fmt(ratio(b_out, a_out), ".3f"),
                "a_wall_median_s": fmt(a_wall, ".3f"),
                "b_wall_median_s": fmt(b_wall, ".3f"),
                "wall_ratio_b_over_a": fmt(ratio(b_wall, a_wall), ".3f"),
                "identical_seeds": str(same),
                "bit_identical": "yes" if same == n else "no",
                "a_crashes": str(sum(r["status"] != "ok" for r in a)),
                "b_crashes": str(sum(r["status"] != "ok" for r in b)),
                "hit_drop": "yes" if b_hits < a_hits else "no",
                "lose_hits": "yes" if lose_hits else "no",
                "lose_calls": "yes" if lose_calls else "no",
            }
        )
    return out


def pairs(rows: list[dict[str, str]], seeds: tuple[int, ...] = run_panel.SEEDS) -> list[dict[str, str]]:
    table = grouped(rows)
    out = []
    for system, _driver, _reference in run_panel.PANEL:
        arms = table.get(system, {})
        for seed in seeds:
            if seed not in arms.get("A", {}) or seed not in arms.get("B", {}):
                continue
            a, b = arms["A"][seed], arms["B"][seed]
            out.append(
                {
                    "system": system,
                    "seed": str(seed),
                    "a_hit": a["hit"],
                    "b_hit": b["hit"],
                    "a_calls": a["evals_to_hit"],
                    "b_calls": b["evals_to_hit"],
                    "a_charged": a["charged"],
                    "b_charged": b["charged"],
                    "a_best": a["best"],
                    "b_best": b["best"],
                    "a_best_bits": a.get("best_bits", ""),
                    "b_best_bits": b.get("best_bits", ""),
                    "identical": "yes" if identity(a) == identity(b) else "no",
                }
            )
    return out


def verdict(table: list[dict[str, str]]) -> tuple[str, list[str]]:
    """The verdict line and the systems behind it."""
    expected = [name for name, _driver, _reference in run_panel.PANEL]
    missing = [name for name in expected if name not in {row["system"] for row in table}]
    if missing:
        return f"VERDICT: INCOMPLETE ({len(missing)} of {len(expected)} systems lack a full seed set)", missing
    reasons = []
    for row in table:
        if row["lose_hits"] == "yes":
            reasons.append(
                f"{row['system']}: hits {row['a_hits']} to {row['b_hits']}, Fisher p {pp(float(row['fisher_p']))}"
            )
        if row["lose_calls"] == "yes":
            reasons.append(
                f"{row['system']}: median calls {row['a_calls_median']} to {row['b_calls_median']}, "
                f"Mann-Whitney p {pp(float(row['mw_p_calls']))}"
            )
    if reasons:
        return "VERDICT: B LOSES (" + "; ".join(reasons) + ")", reasons
    return "VERDICT: PASS", []


def hit_drops(table: list[dict[str, str]]) -> list[str]:
    return [
        f"{row['system']} {row['a_hits']}/{row['seeds']} to {row['b_hits']}/{row['seeds']} "
        f"(Fisher p {pp(float(row['fisher_p']))})"
        for row in table
        if row["hit_drop"] == "yes"
    ]


def watch(table: list[dict[str, str]]) -> list[str]:
    """Systems whose calls differ at p < 0.05 without being a loss."""
    lines = []
    for row in table:
        if row["lose_calls"] == "yes" or float(row["mw_p_calls"]) >= ALPHA:
            continue
        lines.append(
            f"{row['system']}: calls median {row['a_calls_median']} to {row['b_calls_median']}, "
            f"{row['calls_rank_side']} by rank, Mann-Whitney p {pp(float(row['mw_p_calls']))}"
        )
    return lines


def write_csv(path: Path, records: list[dict[str, str]], fields: list[str] | None = None) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def crash_lines(rows: list[dict[str, str]]) -> list[str]:
    return [
        f"{row['system']} arm {row['arm']} seed {row['seed']} (status {row['status']})"
        for row in rows
        if row["status"] != "ok"
    ]


def render(table: list[dict[str, str]], rows: list[dict[str, str]]) -> str:
    """Markdown for the README: hits and calls, times, then the verdict."""
    lines = [
        "| System | Budget | A hits | B hits | Fisher p | A calls median | B calls median "
        "| Mann-Whitney p | Identical seeds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['system']} | {int(r['budget']):,} | {r['a_hits']}/{r['seeds']} | {r['b_hits']}/{r['seeds']} "
            f"| {pp(float(r['fisher_p']))} | {r['a_calls_median']} | {r['b_calls_median']} "
            f"| {pp(float(r['mw_p_calls']))} "
            f"| {r['identical_seeds']}/{r['seeds']} |"
        )
    lines += [
        "",
        "| System | A in potential (s) | B in potential (s) | A outside (s) | B outside (s) "
        "| A wall (s) | B wall (s) | B/A wall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['system']} | {r['a_pot_median_s']} | {r['b_pot_median_s']} | {r['a_outside_median_s']} "
            f"| {r['b_outside_median_s']} | {r['a_wall_median_s']} | {r['b_wall_median_s']} "
            f"| {r['wall_ratio_b_over_a']} |"
        )
    line, _reasons = verdict(table)
    drops = hit_drops(table)
    lines += ["", "Systems with any hit drop, significant or not: " + ("; ".join(drops) if drops else "none") + "."]
    notes = watch(table)
    if notes:
        lines += ["", "Calls differing at p < 0.05 without a loss: " + "; ".join(notes) + "."]
    bad = crash_lines(rows)
    if bad:
        lines += ["", "Crashed runs, counted as misses: " + "; ".join(bad) + "."]
    lines += ["", f"`{line}`"]
    return "\n".join(lines)


def splice(readme: Path, block: str) -> None:
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{readme} has no results markers")
    head, rest = text.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    readme.write_text(f"{head}{BEGIN}\n{block}\n{END}{tail}")


def report(table: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
    if not table:
        print("summary: no system has every seed on both arms yet", flush=True)
        return
    print(
        "system hitsA hitsB fisher callsA callsB mw inA inB outA outB identical lose",
        flush=True,
    )
    for r in table:
        lose = (
            ",".join(name for name, flag in (("hits", r["lose_hits"]), ("calls", r["lose_calls"])) if flag == "yes")
            or "no"
        )
        print(
            f"{r['system']} {r['a_hits']} {r['b_hits']} {pp(float(r['fisher_p']))} {r['a_calls_median']} "
            f"{r['b_calls_median']} {pp(float(r['mw_p_calls']))} {r['a_pot_median_s']} {r['b_pot_median_s']} "
            f"{r['a_outside_median_s']} {r['b_outside_median_s']} {r['identical_seeds']}/{r['seeds']} {lose}",
            flush=True,
        )
    line, _reasons = verdict(table)
    drops = hit_drops(table)
    print("hit drops: " + ("; ".join(drops) if drops else "none"), flush=True)
    print(line, flush=True)


def write_all(rows_path: Path, summary: Path, pairs_path: Path, readme: Path | None) -> None:
    rows = read_rows(rows_path)
    table = summarise(rows)
    write_csv(summary, table)
    write_csv(pairs_path, pairs(rows))
    if readme is not None and table:
        splice(readme, render(table, rows))
    report(table, rows)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args(argv[1:])
    write_all(args.rows, args.summary, args.pairs, args.readme)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
