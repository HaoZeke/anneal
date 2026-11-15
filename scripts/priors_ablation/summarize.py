#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Priors off against priors on and against GMIN, all from one session.

Per system and arm: hits out of n, the median calls to hit with a miss
counted as the full budget, and the median wall, inside (potential) and
outside seconds. Medians are Hyndman-Fan type 7. Priors off is compared
with priors on and with GMIN twice: the two-sided Fisher exact test of the
hit counts, and the two-sided exact Mann-Whitney test of the calls to hit,
where a miss is the full budget. The Mann-Whitney null distribution is the
exact permutation distribution of the rank sum given the observed ties,
so a censored block of misses is one tie, not a run of distinct values.
A blank time is a GMIN build that does not time the potential (NEWTIP).
Rows are only ever appended. A (system, arm, seed) run again at a later
revision is scored from its last row, and each arm lists the revisions
and hosts its rows ran at.
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

ORDER = (
    "lj38", "lj75", "lj98", "lj104", "lj150",
    "m55r6", "m80r3", "m80r6", "m55r14", "m80r14",
    "au55", "cu6au32", "cu27au28",
    "cu19au19", "cu10au45", "cu30au8", "cu12au26", "pt12cu11",
    "tip4p4", "tip4p8", "tip4p15",
)


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided Fisher exact p for a/n1 against b/n2 hits."""
    k = a + b
    total = comb(n1 + n2, k)

    def prob(x: int) -> float:
        return comb(n1, x) * comb(n2, k - x) / total

    observed = prob(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def mann_whitney(x: list[float], y: list[float]) -> float:
    """Two-sided exact Mann-Whitney p of x against y, ties included.

    Ranks are midranks, doubled so they are integers. The null counts every
    way of choosing len(x) of the pooled values, so its rank sums carry the
    observed ties.
    """
    pooled = sorted(x + y)
    doubled = {}
    i = 0
    while i < len(pooled):
        j = i
        while j < len(pooled) and pooled[j] == pooled[i]:
            j += 1
        doubled[pooled[i]] = i + 1 + j
        i = j
    ranks = [doubled[v] for v in pooled]
    n1, total = len(x), len(pooled)
    observed = sum(doubled[v] for v in x)
    centre = n1 * (total + 1)
    # ways[k][s]: subsets of k pooled values whose doubled ranks sum to s.
    top = sum(ranks)
    ways = [[0] * (top + 1) for _ in range(n1 + 1)]
    ways[0][0] = 1
    for rank in ranks:
        for k in range(n1, 0, -1):
            row, below = ways[k], ways[k - 1]
            for s in range(top, rank - 1, -1):
                if below[s - rank]:
                    row[s] += below[s - rank]
    extreme = abs(observed - centre)
    tail = sum(count for s, count in enumerate(ways[n1]) if abs(s - centre) >= extreme)
    return min(1.0, tail / comb(total, n1))


def read(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def is_hit(value: str) -> bool:
    return value.strip().lower() in ("1", "true")


def calls(rows: list[dict[str, str]]) -> list[float]:
    return [float(r["evals_to_hit"]) if is_hit(r["hit"]) else float(r["budget"]) for r in rows]


def seed_cells(rows: list[dict[str, str]]) -> dict:
    walls = [float(r["wall_s"]) for r in rows]
    inside, outside = [], []
    for r, wall in zip(rows, walls):
        if r.get("pot_s"):
            pot = float(r["pot_s"])
            inside.append(pot)
            outside.append(float(r["outside_s"]) if r.get("outside_s") else max(wall - pot, 0.0))
    return {
        "hits": sum(is_hit(r["hit"]) for r in rows),
        "n": len(rows),
        "calls": calls(rows),
        "evals": type7(calls(rows)),
        "wall": type7(walls),
        "inside": type7(inside) if inside else None,
        "outside": type7(outside) if outside else None,
    }


def fmt_time(value) -> str:
    return "–" if value is None else f"{value:.3f}"


def cell(s: dict | None) -> str:
    if s is None:
        return "no row"
    return (
        f"{s['hits']}/{s['n']} · {s['evals']:.0f} · {s['wall']:.3f} s · "
        f"{fmt_time(s['inside'])} / {fmt_time(s['outside'])}"
    )


def tests(a: dict | None, b: dict | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return None
    return fisher(a["hits"], a["n"], b["hits"], b["n"]), mann_whitney(a["calls"], b["calls"])


def fmt_tests(t: tuple[float, float] | None) -> str:
    return "–" if t is None else f"{t[0]:.3f} / {t[1]:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--gmin-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    latest = {}
    for r in read(args.rows):
        r["hit"] = r["hit_binary"]
        latest[(r["system"], r["arm"], r["seed"])] = r
    rows = list(latest.values())
    gmin_rows = read(args.gmin_rows) if args.gmin_rows.is_file() else []
    for r in gmin_rows:
        # The GMIN build times the LJ, Morse and Gupta kernels but not NEWTIP.
        if r["system"].startswith("tip4p"):
            r["pot_s"] = r["outside_s"] = ""
    lines = [
        "| system | GMIN | anneal priors on | anneal priors off "
        "| off vs on: Fisher / MW | off vs GMIN: Fisher / MW |",
        "|---|---|---|---|---|---|",
    ]
    out_rows = []
    for system in ORDER:
        arms = {
            "gmin": [r for r in gmin_rows if r["system"] == system],
            "priors_on": [r for r in rows if r["system"] == system and r["arm"] == "default"],
            "priors_off": [r for r in rows if r["system"] == system and r["arm"] == "priors_off"],
        }
        cells = {arm: seed_cells(found) if found else None for arm, found in arms.items()}
        if not any(cells.values()):
            continue
        off_on = tests(cells["priors_off"], cells["priors_on"])
        off_gmin = tests(cells["priors_off"], cells["gmin"])
        lines.append(
            f"| {system} | {cell(cells['gmin'])} | {cell(cells['priors_on'])} | "
            f"{cell(cells['priors_off'])} | {fmt_tests(off_on)} | {fmt_tests(off_gmin)} |"
        )
        for arm, s in cells.items():
            if s is None:
                continue
            against = {"priors_on": off_on, "gmin": off_gmin}.get(arm)
            out_rows.append(
                {
                    "system": system,
                    "arm": arm,
                    "hits": s["hits"],
                    "n": s["n"],
                    "evals_median": f"{s['evals']:.1f}",
                    "wall_median_s": f"{s['wall']:.6f}",
                    "inside_median_s": "" if s["inside"] is None else f"{s['inside']:.6f}",
                    "outside_median_s": "" if s["outside"] is None else f"{s['outside']:.6f}",
                    "fisher_p_vs_priors_off": "" if against is None else f"{against[0]:.6f}",
                    "mann_whitney_p_vs_priors_off": "" if against is None else f"{against[1]:.6f}",
                    "code_revision": " ".join(sorted({r["code_revision"] for r in arms[arm]})),
                    "host": " ".join(sorted({r["host"] for r in arms[arm]})),
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
