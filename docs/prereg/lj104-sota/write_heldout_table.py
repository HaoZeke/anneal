#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rebuild the held-out table from the raw seed logs.

The numbers in the written note are computed here. A row that
disagrees with its log is a failure.
"""

from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "heldout"
E_KNOWN = -582.086642
HIT_LINE = E_KNOWN + 1e-6
Z = 1.96


def kv(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key.strip()] = value.strip()
    return fields


def type7(values: list[float], p: float) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise SystemExit("quantile of an empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = math.floor(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0 or index + 1 >= n:
        return ordered[index]
    return ordered[index] + frac * (ordered[index + 1] - ordered[index])


def wilson(hits: int, n: int) -> tuple[float, float, float]:
    phat = hits / n
    z2 = Z * Z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    half = Z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n)) / denom
    return phat, max(0.0, centre - half), min(1.0, centre + half)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the table [[a, b], [c, d]]."""

    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        return (
            math.comb(aa + bb, aa)
            * math.comb(cc + dd, cc)
            / math.comb(aa + bb + cc + dd, aa + cc)
        )

    observed = prob(a, b, c, d)
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = 0.0
    low = max(0, col1 - row2)
    high = min(col1, row1)
    for aa in range(low, high + 1):
        bb = row1 - aa
        cc = col1 - aa
        dd = row2 - cc
        pp = prob(aa, bb, cc, dd)
        if pp <= observed + 1e-15:
            total += pp
    return min(1.0, total)


def load_arm(code: str) -> list[dict[str, str]]:
    rows = []
    table = (ROOT / "rows.tsv").read_text(encoding="utf-8").splitlines()
    header = table[0].split("\t")
    for line in table[1:]:
        if not line.strip():
            continue
        row = dict(zip(header, line.split("\t")))
        if row["code"] != code:
            continue
        seed = row["seed"]
        if code == "dls":
            log = kv((ROOT / "dls" / f"{seed}.log").read_text(encoding="utf-8"))
            requench = kv((ROOT / "requench" / f"dls-{seed}.txt").read_text(encoding="utf-8"))
            if log.get("charged") != row["calls"]:
                raise SystemExit(f"dls {seed} charged disagrees with the log")
            if requench.get("energy") != row["energy"]:
                raise SystemExit(f"dls {seed} energy disagrees with the requench log")
        else:
            stdout = (ROOT / "gmin" / seed / "stdout.txt").read_text(encoding="utf-8")
            charged = ""
            for item in stdout.splitlines():
                if item.startswith("h2h_row "):
                    parts = dict(piece.split("=", 1) for piece in item.split()[1:] if "=" in piece)
                    charged = parts.get("charged", "")
            requench = kv((ROOT / "requench" / f"gmin-{seed}.txt").read_text(encoding="utf-8"))
            if charged != row["calls"]:
                raise SystemExit(f"gmin {seed} charged disagrees with the log")
            if requench.get("energy") != row["energy"]:
                raise SystemExit(f"gmin {seed} energy disagrees with the requench log")
        hit = float(row["energy"]) <= HIT_LINE
        if (hit and row["hit"] != "true") or ((not hit) and row["hit"] != "false"):
            raise SystemExit(f"{code} {seed} hit flag disagrees with the energy")
        rows.append(row)
    if len(rows) != 30:
        raise SystemExit(f"{code} does not have 30 rows")
    return rows


def summarise(rows: list[dict[str, str]]) -> dict[str, str]:
    hits = [row for row in rows if row["hit"] == "true"]
    walls = [float(row["wall_seconds"]) for row in hits]
    costs = [float(row["calls"]) for row in hits]
    energies = [float(row["energy"]) for row in rows if row["energy"]]
    rate, low, high = wilson(len(hits), len(rows))
    return {
        "hits": f"{len(hits)}/30",
        "rate": f"{rate:.6f}",
        "wilson_low": f"{low:.6f}",
        "wilson_high": f"{high:.6f}",
        "time_median": f"{type7(walls, 0.5):.3f}" if walls else "none",
        "time_q1": f"{type7(walls, 0.25):.3f}" if walls else "none",
        "time_q3": f"{type7(walls, 0.75):.3f}" if walls else "none",
        "cost_median": f"{type7(costs, 0.5):.1f}" if costs else "none",
        "cost_q1": f"{type7(costs, 0.25):.1f}" if costs else "none",
        "cost_q3": f"{type7(costs, 0.75):.1f}" if costs else "none",
        "best": f"{min(energies):.10f}" if energies else "none",
    }


def main() -> None:
    dls = load_arm("dls")
    gmin = load_arm("gmin")
    left = summarise(dls)
    right = summarise(gmin)
    a = sum(row["hit"] == "true" for row in dls)
    b = 30 - a
    c = sum(row["hit"] == "true" for row in gmin)
    d = 30 - c
    fisher = fisher_two_sided(a, b, c, d)
    lines = [
        "Copyright (c) anneal developers",
        "",
        "The held-out seeds are 5000 through 5029.",
        "Arm D is lowest-index lattice v3.2.",
        "Arm B stays UNFILLED.",
        "The call budget is 4000000.",
        "A hit is re-quenched E <= -582.086641.",
        "Wilson intervals use z = 1.96.",
        "Quantiles are Hyndman-Fan type 7.",
        "Time to hit and charged cost are the hit seeds only.",
        "",
        "Arm D hits " + left["hits"] + ".",
        "The Wilson interval is " + left["wilson_low"] + " to " + left["wilson_high"] + ".",
        "The median time to hit is " + left["time_median"] + " seconds.",
        "The time quartiles are " + left["time_q1"] + " and " + left["time_q3"] + ".",
        "The median charged cost is " + left["cost_median"] + ".",
        "The cost quartiles are " + left["cost_q1"] + " and " + left["cost_q3"] + ".",
        "The best energy is " + left["best"] + ".",
        "",
        "GMIN hits " + right["hits"] + ".",
        "The Wilson interval is " + right["wilson_low"] + " to " + right["wilson_high"] + ".",
        "The median time to hit is " + right["time_median"] + " seconds.",
        "The time quartiles are " + right["time_q1"] + " and " + right["time_q3"] + ".",
        "The median charged cost is " + right["cost_median"] + ".",
        "The cost quartiles are " + right["cost_q1"] + " and " + right["cost_q3"] + ".",
        "The best energy is " + right["best"] + ".",
        "",
        f"The Fisher table is {a} {b} {c} {d}.",
        f"The two-sided Fisher p-value is {fisher:.6f}.",
        "",
    ]
    text = "\n".join(lines)
    for line in text.splitlines():
        if len(line) > 72:
            raise SystemExit(f"line longer than 72: {line}")
    (HERE / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
