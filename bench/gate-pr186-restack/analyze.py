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
import struct
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


def peak_rss(row: dict) -> float:
    val = row.get("peak_rss_anon_kb")
    return float(val) if val not in (None, "") else float(row["rss_anon_kb"])


def bits_to_float(text: str) -> float:
    return struct.unpack("<d", int(text, 16).to_bytes(8, "little"))[0]


def calls_scored(row: dict) -> float:
    hit = int(row["hit"])
    return float(row["calls_to_hit"]) if hit else float(row["budget"]) + 1.0


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
        if (
            rb is None
            or any(
                ra[k] != rb[k]
                for k in (
                    "trajectory",
                    "energy_bits",
                    "callback_calls",
                    "calls_to_hit",
                    "charged",
                )
            )
            or ra["trajectory"] == ""
        ):
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
    over_budget = [
        r
        for r in list(a.values()) + list(b.values()) + list(a2.values())
        if float(r["charged"]) > float(r["budget"])
        or (int(r["hit"]) and r["calls_to_hit"] == "")
        or (
            r["calls_to_hit"] != ""
            and (
                float(r["calls_to_hit"]) > float(r["budget"])
                or float(r["calls_to_hit"]) > float(r["charged"])
            )
        )
    ]
    if over_budget:
        res["verdict"] = "FIX"
        res["failures"] = "call count exceeds budget"
        res["notes"] = f"invalid call counts: {len(over_budget)}"
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
    res["rss_band"] = gb[2] <= max(CAP, ga[2])
    res["rss_fail"] = gb[1] > max(CAP, ga[2])
    prb_, pra_ = paired_ratio(b, a, peak_rss), paired_ratio(a2, a, peak_rss)
    pgb, pga = geomean_ci(prb_), geomean_ci(pra_)
    res["peak_rss_ratio"], res["peak_rss_lo"], res["peak_rss_hi"] = pgb
    res["peak_rss_aa_hi"] = pga[2]
    res["peak_rss_strict"] = pgb[2] <= CAP
    res["peak_rss_band"] = pgb[2] <= max(CAP, pga[2])
    res["peak_rss_fail"] = pgb[1] > max(CAP, pga[2])
    path_checks = []
    path_enabled = 0
    path_unavailable = 0
    for row in b.values():
        flag = row.get("branch_on_stall")
        draws_text = row.get("b_path_draws")
        spent_text = row.get("b_path_spent")
        if (
            flag not in ("0", "1")
            or draws_text in ("", None)
            or spent_text in ("", None)
        ):
            path_unavailable += 1
            continue
        draws = int(draws_text)
        spent = int(spent_text)
        if flag == "1":
            path_enabled += 1
            path_checks.append(draws > 0 and spent > 0)
        elif flag == "0":
            path_checks.append(draws == 0 and spent == 0)
        else:
            path_checks.append(False)
    res["b_path_matches"] = sum(path_checks)
    res["b_path_checkable"] = len(path_checks)
    res["b_path_enabled"] = path_enabled
    res["b_path_draws_sum"] = sum(
        int(r["b_path_draws"]) for r in b.values()
        if r.get("b_path_draws") not in ("", None)
    )
    res["b_path_spent_sum"] = sum(
        int(r["b_path_spent"]) for r in b.values()
        if r.get("b_path_spent") not in ("", None)
    )
    early = [
        r
        for r in list(a.values()) + list(b.values())
        if int(r["hit"]) and float(r["calls_to_hit"]) < float(r["budget"])
    ]
    res["early_stop_runs"] = len(early)
    res["calls_ratio"] = geomean_ci(
        paired_ratio(b, a, lambda r: float(r["callback_calls"]))
    )[0]
    if early:
        notes.append("early-stop per-call artifact possible: read totals beside per-call")
    if (res["opc_band"] != res["tot_band"]) or (res["opc_strict"] != res["tot_strict"]):
        notes.append("per-call and total outside time disagree")
    if res["calls_ratio"] and abs(res["calls_ratio"] - 1.0) > 0.02:
        notes.append("calls differ between A and B")
    kind = next(iter(a.values()))["kind"] if a else ""
    if path_unavailable:
        notes.append(f"B conf-miss counters unavailable in {path_unavailable} rows")
    elif res["b_path_matches"] != res["n"]:
        notes.append("B conf-miss counters disagree with the resolved configuration")
    if kind == "guard":
        ok, bad = identical(a, b)
        res["identical"] = ok
        res["mismatch"] = " ".join(map(str, bad))
        diffs = [
            abs(bits_to_float(b[s]["energy_bits"]) - bits_to_float(a[s]["energy_bits"]))
            for s in sorted(a)
            if a[s].get("energy_bits") and b.get(s, {}).get("energy_bits")
        ]
        res["max_abs_dE"] = max(diffs) if diffs else None
        if not ok:
            notes.append("default-parity fields differ on seeds " + res["mismatch"])
    failures = []
    if res["hit_loss"]:
        failures.append("hit loss")
    if res["opc_regression"] or not res["opc_band"]:
        failures.append("outside time (per call)")
    if res["rss_fail"] or res["peak_rss_fail"]:
        failures.append("memory")
    if res.get("identical") is False:
        failures.append("default parity")
    if not path_unavailable and res["b_path_matches"] != res["n"]:
        failures.append("conf-miss path")
    res["failures"] = "; ".join(failures)
    res["notes"] = "; ".join(notes)
    if failures:
        res["verdict"] = "FIX"
    elif path_unavailable:
        res["verdict"] = "INCONCLUSIVE"
    elif res["opc_strict"] and res["rss_strict"] and res["peak_rss_strict"]:
        res["verdict"] = "PASS"
    else:
        res["verdict"] = "PASS-BAND"
    return res


FIELDS = [
    "cell", "n", "verdict", "failures", "notes", "hits_a", "hits_b", "hits_a2", "p_fisher", "p_mwu",
    "opc_ratio", "opc_lo", "opc_hi", "opc_aa_hi", "opc_strict", "opc_band", "opc_regression",
    "tot_ratio", "tot_lo", "tot_hi", "tot_aa_hi", "tot_strict", "tot_band", "tot_regression",
    "rss_ratio", "rss_lo", "rss_hi", "rss_aa_hi", "rss_strict", "rss_fail",
    "peak_rss_ratio", "peak_rss_lo", "peak_rss_hi", "peak_rss_aa_hi", "peak_rss_strict", "peak_rss_fail",
    "b_path_matches", "b_path_checkable", "b_path_enabled", "b_path_draws_sum", "b_path_spent_sum",
    "calls_ratio", "early_stop_runs", "identical", "max_abs_dE", "mismatch",
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
        "| cell | n | verdict | hits A/B/A2 | p Fisher | p MWU | out/call B/A [lo,hi] | A2/A hi | out total B/A [lo,hi] | end RSS B/A [lo,hi] | peak RSS B/A [lo,hi] | B path | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if not r["complete"]:
            lines.append(f"| {r['cell']} | {r['n']} | {r['verdict']} | | | | | | | | | | {r['notes']} |")
            continue
        bpath = (
            f"{r['b_path_matches']}/{r['b_path_checkable']} "
            f"({r['b_path_enabled']} enabled)"
        )
        lines.append(
            f"| {r['cell']} | {r['n']} | {r['verdict']} | {r['hits_a']}/{r['hits_b']}/{r['hits_a2']} "
            f"| {fmt(r['p_fisher'])} | {fmt(r['p_mwu'])} "
            f"| {fmt(r['opc_ratio'])} [{fmt(r['opc_lo'])},{fmt(r['opc_hi'])}] | {fmt(r['opc_aa_hi'])} "
            f"| {fmt(r['tot_ratio'])} [{fmt(r['tot_lo'])},{fmt(r['tot_hi'])}] "
            f"| {fmt(r['rss_ratio'])} [{fmt(r['rss_lo'])},{fmt(r['rss_hi'])}] "
            f"| {fmt(r['peak_rss_ratio'])} [{fmt(r['peak_rss_lo'])},{fmt(r['peak_rss_hi'])}] "
            f"| {bpath} | {r['failures'] or r['notes']} |"
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
