#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Checks for the ledger gate. The rules are fixed in PROTOCOL.md."""

from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path

import numpy as np

BOOT_SEED = 20261123
BOOT_DRAWS = 10_000
ALPHA = 0.05
CAP = 1.10
ENERGY_ABS = 1e-6
TERMS = (
    "screen",
    "full",
    "check",
    "closure",
    "quench",
    "settle",
    "lattice",
    "reoccupy",
    "probe",
    "probe_unbacked",
)
INTERNAL = ("lattice", "reoccupy", "quench", "settle")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def by_seed(rows: list[dict], cell: str, arm: str, phase: str | None = None) -> dict[int, dict]:
    out = {}
    for row in rows:
        if row["cell"] != cell or row["arm"] != arm:
            continue
        if phase is not None and row.get("phase") not in (None, "", phase):
            continue
        out[int(row["seed"])] = row
    return out


def as_int(text: str) -> int | None:
    if text in ("", "na", None):
        return None
    return int(float(text))


def as_float(text: str) -> float | None:
    if text in ("", "na", None):
        return None
    return float(text)


def energy_of(row: dict) -> float | None:
    bits = row.get("energy_bits") or ""
    if bits and bits != "na":
        return struct.unpack("<d", int(bits, 16).to_bytes(8, "little"))[0]
    return as_float(row.get("energy") or "")


def calls_scored(row: dict) -> float:
    if int(row["hit"]):
        return float(row["calls_to_hit"])
    return float(row["budget"]) + 1.0


def geomean_ci(ratios: np.ndarray) -> tuple[float, float, float]:
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
            raise ValueError(f"non-positive value at seed {seed}: {n} / {d}")
        out.append(n / d)
    return np.asarray(out, dtype=float)


def outside(row: dict) -> float:
    return float(row["wall_s"]) - float(row["inside_s"])


def peak(row: dict) -> float:
    text = row.get("peak_rss_anon_kb")
    if text in ("", None):
        text = row.get("rss_anon_kb")
    return float(text)


def ledger_failures(rows: list[dict], cell: str) -> list[str]:
    bad = []
    for row in rows:
        if row["cell"] != cell or row["arm"] != "B":
            continue
        seed = row["seed"]
        if row.get("error"):
            bad.append(f"seed {seed} error {row['error']}")
        parts = []
        for name in TERMS:
            value = as_int(row.get(name) or "")
            if value is None:
                bad.append(f"seed {seed} missing {name}")
                parts = None
                break
            parts.append(value)
        if parts is None:
            continue
        charged = as_int(row.get("charged") or "")
        total = sum(parts)
        if charged != total:
            bad.append(f"seed {seed} charged {charged} sum {total} " + " ".join(f"{n}={v}" for n, v in zip(TERMS, parts)))
        unnamed = row.get("unnamed") or ""
        if unnamed not in ("0", ""):
            bad.append(f"seed {seed} unnamed {unnamed}")
        oracle = as_int(row.get("oracle_calls") or "")
        spent = as_int(row.get("spent") or "")
        if oracle is None or spent is None or oracle != spent:
            bad.append(f"seed {seed} oracle_calls {row.get('oracle_calls')} spent {row.get('spent')}")
        call_budget = as_int(row.get("call_budget") or "")
        search_budget = as_int(row.get("search_budget") or "")
        if call_budget is None or search_budget is None or call_budget != search_budget:
            bad.append(
                f"seed {seed} call_budget {row.get('call_budget')} search_budget {row.get('search_budget')}"
            )
        if row.get("kind") == "molecule":
            callbacks = as_int(row.get("callbacks") or "")
            internal = as_int(row.get("internal") or "")
            extras = as_int(row.get("split_extras") or "")
            probe_unbacked = parts[-1]
            if None in (callbacks, internal, extras, spent):
                bad.append(f"seed {seed} python identity missing a term")
            elif callbacks + internal + probe_unbacked != spent + extras:
                bad.append(
                    f"seed {seed} callbacks {callbacks} internal {internal} "
                    f"probe_unbacked {probe_unbacked} spent {spent} split_extras {extras}"
                )
        sample = as_float(row.get("sample_s") or "")
        if sample is None or sample < 0.05:
            bad.append(f"seed {seed} sample_s {row.get('sample_s')}")
    return bad


def moved(left: dict[int, dict], right: dict[int, dict]) -> list[str]:
    notes = []
    for seed in sorted(set(left) | set(right)):
        a = left.get(seed)
        b = right.get(seed)
        if a is None or b is None:
            notes.append(f"seed {seed} missing")
            continue
        reasons = []
        ea, eb = energy_of(a), energy_of(b)
        if ea is None or eb is None or abs(ea - eb) > ENERGY_ABS:
            reasons.append(f"energy {ea} {eb}")
        if str(a.get("hit")) != str(b.get("hit")):
            reasons.append(f"hit {a.get('hit')} {b.get('hit')}")
        if str(a.get("calls_to_hit") or "") != str(b.get("calls_to_hit") or ""):
            reasons.append(f"calls {a.get('calls_to_hit')} {b.get('calls_to_hit')}")
        if reasons:
            notes.append(f"seed {seed} " + "; ".join(reasons))
    return notes


def hit_tests(a: dict[int, dict], b: dict[int, dict]) -> tuple[float, float, int, int]:
    from scipy.stats import fisher_exact, mannwhitneyu

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


def judge(rows: list[dict], cell: str) -> dict:
    a = by_seed(rows, cell, "A", "session")
    b = by_seed(rows, cell, "B", "session")
    a2 = by_seed(rows, cell, "Ap", "session")
    p = by_seed(rows, cell, "P", "reference")
    res: dict = {"cell": cell, "n": len(b)}
    ledger = ledger_failures(rows, cell)
    res["ledger"] = ledger
    res["moved_a"] = moved(b, a)
    res["moved_p"] = moved(b, p) if p else ["P absent"]
    complete = set(a) == set(b) == set(a2) and len(a) == 12
    errors = [
        r for r in rows if r["cell"] == cell and r["arm"] in ("A", "B", "Ap") and r.get("error")
    ]
    res["complete"] = complete and not errors
    if not res["complete"]:
        res["verdict"] = "FAIL"
        res["note"] = f"incomplete errors {len(errors)} nA {len(a)} nB {len(b)} nA2 {len(a2)}"
        return res
    p_f, p_m, ha, hb = hit_tests(a, b)
    res.update(p_fisher=p_f, p_mwu=p_m, hits_a=ha, hits_b=hb, hit_loss=p_f < ALPHA or p_m < ALPHA)
    for tag, key in (("out", outside), ("rss", peak)):
        gb = geomean_ci(paired_ratio(b, a, key))
        ga = geomean_ci(paired_ratio(a2, a, key))
        res[f"{tag}_ratio"], res[f"{tag}_lo"], res[f"{tag}_hi"] = gb
        res[f"{tag}_aa"], res[f"{tag}_aa_lo"], res[f"{tag}_aa_hi"] = ga
        res[f"{tag}_cap"] = gb[2] <= CAP
        res[f"{tag}_real"] = gb[1] > 1.0 and gb[1] > ga[2]
    fail = bool(ledger or res["moved_a"] or res["moved_p"] or res["hit_loss"])
    fail = fail or res["out_real"] or res["rss_real"]
    res["verdict"] = "FAIL" if fail else "PASS"
    return res


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.rows)
    cells = []
    seen = []
    for row in rows:
        if row["cell"] not in seen:
            seen.append(row["cell"])
    lines = ["# Ledger gate report", ""]
    table = ["cell verdict ledger traj hit cth out_lo out_hi out_cap rss_lo rss_hi rss_cap"]
    overall = "PASS"
    for cell in seen:
        res = judge(rows, cell)
        cells.append(res)
        if res["verdict"] != "PASS":
            overall = "FAIL"
        ledger = "PASS" if not res.get("ledger") else "FAIL"
        traj = "PASS" if not res.get("moved_a") and not res.get("moved_p") else "FAIL"
        if "p_fisher" not in res:
            hit = cth = out_lo = out_hi = out_cap = rss_lo = rss_hi = rss_cap = "NA"
        else:
            hit = "FAIL" if res["p_fisher"] < ALPHA else "PASS"
            cth = "FAIL" if res["p_mwu"] < ALPHA else "PASS"
            out_lo = fmt(res["out_lo"])
            out_hi = fmt(res["out_hi"])
            out_cap = "yes" if res["out_cap"] else "no"
            rss_lo = fmt(res["rss_lo"])
            rss_hi = fmt(res["rss_hi"])
            rss_cap = "yes" if res["rss_cap"] else "no"
        table.append(
            f"{cell} {res['verdict']} {ledger} {traj} {hit} {cth} {out_lo} {out_hi} {out_cap} {rss_lo} {rss_hi} {rss_cap}"
        )
        if res.get("ledger"):
            lines.append(f"{cell} ledger:")
            for item in res["ledger"]:
                lines.append(f"- {item}")
        if res.get("moved_a"):
            lines.append(f"{cell} B vs A:")
            for item in res["moved_a"]:
                lines.append(f"- {item}")
        if res.get("moved_p"):
            lines.append(f"{cell} B vs P:")
            for item in res["moved_p"]:
                lines.append(f"- {item}")
        if res.get("hit_loss"):
            lines.append(
                f"{cell} hits A {res['hits_a']} B {res['hits_b']} "
                f"fisher {res['p_fisher']:.4g} mwu {res['p_mwu']:.4g}"
            )
        if res.get("out_real") or res.get("rss_real"):
            lines.append(
                f"{cell} real out {res.get('out_real')} rss {res.get('rss_real')} "
                f"out [{res.get('out_lo')} {res.get('out_hi')}] aa {res.get('out_aa_hi')} "
                f"rss [{res.get('rss_lo')} {res.get('rss_hi')}] aa {res.get('rss_aa_hi')}"
            )
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.extend(table)
    lines.append("")
    lines.append(f"OVERALL {overall}")
    lines.append("")
    text = "\n".join(lines)
    args.report.write_text(text)
    print(text)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
