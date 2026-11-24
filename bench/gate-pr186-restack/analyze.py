#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Statistics for the restack gate, as fixed in PROTOCOL.md.

Reads rows.csv written by run_gate.py. Writes analysis.csv and
analysis.md next to it. Arm Ap in the rows is arm A2 in the protocol.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

BOOT_SEED = 20261123
BOOT_DRAWS = 10_000
CAP = 1.10
ALPHA = 0.05


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def by_seed(rows: list[dict], cell: str, arm: str) -> dict[int, dict]:
    return {int(r["seed"]): r for r in rows if r["cell"] == cell and r["arm"] == arm}


def geomean_ci(ratios: np.ndarray) -> tuple[float, float, float]:
    """Seed-paired geometric mean and 95 percent bootstrap interval."""
    logs = np.log(ratios)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(logs), size=(BOOT_DRAWS, len(logs)))
    means = np.exp(logs[idx].mean(axis=1))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.exp(logs.mean())), float(lo), float(hi)


def paired_ratio(num: dict[int, dict], den: dict[int, dict], key) -> np.ndarray:
    seeds = sorted(set(num) & set(den))
    out = []
    for seed in seeds:
        n, d = key(num[seed]), key(den[seed])
        if not (n > 0 and d > 0 and math.isfinite(n) and math.isfinite(d)):
            raise ValueError(f"non-positive or non-finite value at seed {seed}")
        out.append(n / d)
    return np.asarray(out)


def per_call(row: dict) -> float:
    return float(row["outside_per_call_s"])


def total_outside(row: dict) -> float:
    return float(row["wall_s"]) - float(row["inside_s"])


def rss(row: dict) -> float:
    return float(row["rss_anon_kb"])


def calls_scored(row: dict) -> float:
    hit = int(row["hit"])
    return float(row["calls"]) if hit else float(row["budget"]) + 1.0


def hit_tests(a: dict[int, dict], b: dict[int, dict]) -> tuple[float, float, int, int]:
    seeds = sorted(set(a) & set(b))
    ha = sum(int(a[s]["hit"]) for s in seeds)
    hb = sum(int(b[s]["hit"]) for s in seeds)
    n = len(seeds)
    p_fisher = float(fisher_exact([[hb, n - hb], [ha, n - ha]], alternative="less")[1])
    xa = [calls_scored(a[s]) for s in seeds]
    xb = [calls_scored(b[s]) for s in seeds]
    if len(set(xa + xb)) == 1:
        p_mwu = 1.0
    else:
        p_mwu = float(mannwhitneyu(xb, xa, alternative="greater").pvalue)
    return p_fisher, p_mwu, ha, hb


def identical(a: dict[int, dict], b: dict[int, dict]) -> tuple[bool, list[int]]:
    bad = []
    for seed in sorted(a):
        ra, rb = a[seed], b.get(seed)
        if rb is None or any(ra[k] != rb[k] for k in ("trajectory", "energy_bits", "calls", "charged")) or ra["trajectory"] == "":
            bad.append(seed)
    return (not bad), bad


def judge(rows: list[dict], cell: str) -> dict:
    a, b, a2 = (by_seed(rows, cell, arm) for arm in ("A", "B", "Ap"))
    res: dict = {"cell": cell, "n": len(a)}
    seeds_ok = set(a) == set(b) == set(a2) and len(a) > 0
    errors = [r for r in rows if r["cell"] == cell and r["error"] != ""]
    res["complete"] = seeds_ok and not errors
    if not res["complete"]:
        res["verdict"] = "INCONCLUSIVE"
        res["notes"] = f"incomplete or error rows: {len(errors)}"
        return res
    p_f, p_m, ha, hb = hit_tests(a, b)
    p_f2, p_m2, _, ha2 = hit_tests(a, a2)
    res.update(hits_a=ha, hits_b=hb, hits_a2=ha2, p_fisher=p_f, p_mwu=p_m, p_fisher_a2=p_f2, p_mwu_a2=p_m2)
    res["hit_loss"] = p_f < ALPHA or p_m < ALPHA
    notes = []
    for tag, key in (("opc", per_call), ("tot", total_outside)):
        gb = geomean_ci(paired_ratio(b, a, key))
        ga = geomean_ci(paired_ratio(a2, a, key))
        res[f"{tag}_ratio"], res[f"{tag}_lo"], res[f"{tag}_hi"] = gb
        res[f"{tag}_aa"], res[f"{tag}_aa_lo"], res[f"{tag}_aa_hi"] = ga
        res[f"{tag}_strict"] = gb[2] <= CAP
        res[f"{tag}_band"] = gb[2] <= max(CAP, ga[2])
        res[f"{tag}_regression"] = gb[1] > 1.0 and gb[1] > ga[2]
    rb_, ra_ = paired_ratio(b, a, rss), paired_ratio(a2, a, rss)
    gb, ga = geomean_ci(rb_), geomean_ci(ra_)
    res["rss_ratio"], res["rss_lo"], res["rss_hi"] = gb
    res["rss_aa_hi"] = ga[2]
    res["rss_strict"] = gb[2] <= CAP
    res["rss_fail"] = gb[1] > max(CAP, ga[2])
    early = [r for r in list(a.values()) + list(b.values()) if int(r["hit"]) and float(r["calls"]) < float(r["budget"])]
    res["early_stop_runs"] = len(early)
    res["calls_ratio"] = geomean_ci(paired_ratio(b, a, lambda r: float(r["calls"])))[0]
    if early:
        notes.append("early-stop per-call artifact possible: read totals beside per-call")
    if (res["opc_band"] != res["tot_band"]) or (res["opc_strict"] != res["tot_strict"]):
        notes.append("per-call and total outside time disagree")
    if res["calls_ratio"] and abs(res["calls_ratio"] - 1.0) > 0.02:
        notes.append("calls differ between A and B")
    if a and next(iter(a.values()))["kind"] == "guard":
        ok, bad = identical(a, b)
        res["identical"] = ok
        res["mismatch"] = " ".join(map(str, bad))
        if not ok:
            notes.append("default-parity fields differ on seeds " + res["mismatch"])
    failures = []
    if res["hit_loss"]:
        failures.append("hit loss")
    if res["opc_regression"] or not res["opc_band"]:
        failures.append("outside time (per call)")
    if res["rss_fail"]:
        failures.append("memory")
    if res.get("identical") is False:
        failures.append("default parity")
    res["failures"] = "; ".join(failures)
    res["notes"] = "; ".join(notes)
    if failures:
        res["verdict"] = "FIX"
    elif res["opc_strict"] and res["rss_strict"]:
        res["verdict"] = "PASS"
    else:
        res["verdict"] = "PASS-BAND"
    return res


FIELDS = [
    "cell", "n", "verdict", "failures", "notes", "hits_a", "hits_b", "hits_a2", "p_fisher", "p_mwu",
    "opc_ratio", "opc_lo", "opc_hi", "opc_aa_hi", "tot_ratio", "tot_lo", "tot_hi", "tot_aa_hi",
    "rss_ratio", "rss_lo", "rss_hi", "rss_aa_hi", "calls_ratio", "early_stop_runs", "identical", "mismatch",
]


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.rows)
    cells = list(dict.fromkeys(r["cell"] for r in rows))
    results = [judge(rows, cell) for cell in cells]
    out = args.rows.parent
    with (out / "analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = [
        "| cell | n | verdict | hits A/B/A2 | p Fisher | p MWU | out/call B/A [lo,hi] | A2/A hi | out total B/A hi | RSS B/A [lo,hi] | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if not r["complete"]:
            lines.append(f"| {r['cell']} | {r['n']} | {r['verdict']} | | | | | | | | {r['notes']} |")
            continue
        lines.append(
            f"| {r['cell']} | {r['n']} | {r['verdict']} | {r['hits_a']}/{r['hits_b']}/{r['hits_a2']} "
            f"| {fmt(r['p_fisher'])} | {fmt(r['p_mwu'])} "
            f"| {fmt(r['opc_ratio'])} [{fmt(r['opc_lo'])},{fmt(r['opc_hi'])}] | {fmt(r['opc_aa_hi'])} "
            f"| {fmt(r['tot_ratio'])} {fmt(r['tot_hi'])} "
            f"| {fmt(r['rss_ratio'])} [{fmt(r['rss_lo'])},{fmt(r['rss_hi'])}] | {r['failures'] or r['notes']} |"
        )
    worst = "FIX" if any(r["verdict"] in ("FIX", "INCONCLUSIVE") for r in results) else "PASS"
    lines.append("")
    lines.append(f"overall: {worst}")
    text = "\n".join(lines) + "\n"
    (out / "analysis.md").write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
