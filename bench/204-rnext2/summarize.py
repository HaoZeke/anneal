#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize and evaluate the anneal #204 r-next2 gate."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

ARMS = ("A", "B", "AA", "AP")
DEFAULT_SEEDS = tuple(range(2000, 2012))
CELL_ORDER = (
    ("lj38", "lj", 400_000, -173.928427),
    ("lj75", "lj", 4_000_000, -397.492331),
    ("lj98", "lj", 4_000_000, -543.665361),
    ("lj102", "lj", 4_000_000, -569.363652),
    ("lj104", "lj", 4_000_000, -582.086642),
    ("m80r14", "morse", 4_000_000, -340.811371),
    ("sc108-80", "sc", 1_000_000, -21819.9952),
    ("blj130n38", "blj", 400_000, -191.1840192602),
)

OUTSIDE_RATIO_LIMIT = 1.10
HIT_TOL = 1.0e-3

SUMMARY_FIELDS = [
    "cell",
    "family",
    "n",
    "budget",
    "hits_A",
    "hits_B",
    "hits_AA",
    "hits_AP",
    "fisher_p_BA",
    "no_hit_loss",
    "med_outside_us_A",
    "med_outside_us_B",
    "med_outside_us_AA",
    "med_outside_us_AP",
    "outside_ratio_BA",
    "outside_ratio_AA_A",
    "outside_ratio_AP_A",
    "mw_p_outside_BA",
    "mw_p_outside_AA_A",
    "mw_p_outside_AP_A",
    "outside_ok",
    "bit_id_BA",
    "bit_id_AA_A",
    "bit_id_AP_A",
    "med_rss_A_kb",
    "med_rss_B_kb",
    "med_rss_AA_kb",
    "med_rss_AP_kb",
    "rss_band_upper_kb",
    "rss_ratio_BA",
    "rss_ratio_AA_A",
    "rss_ratio_AP_A",
    "mw_p_rss_gt_BA",
    "mem_ok",
    "cell_verdict",
]

BIT_ID_FIELDS = [
    "panel",
    "cell",
    "seed",
    "hit_A",
    "hit_B",
    "hit_eq",
    "evals_A",
    "evals_B",
    "evals_eq",
    "charged_A",
    "charged_B",
    "charged_eq",
    "best_A",
    "best_B",
    "best_bits_A",
    "best_bits_B",
    "best_eq",
    "all_eq_BA",
    "all_eq_AA_A",
    "all_eq_AP_A",
]

EXPLICIT_FIELDS = [
    "cell",
    "seed",
    "arm",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "best_bits",
    "outside_us",
    "rss_anon_kb",
    "eq_A_hit",
    "eq_A_evals",
    "eq_A_charged",
    "eq_A_best_bits",
]


def type7_quantile(values: list[float], p: float) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return s[0]
    h = 1.0 + (n - 1) * p
    lo = int(math.floor(h))
    frac = h - lo
    idx = max(0, lo - 1)
    if frac == 0.0:
        return s[idx]
    return s[idx] * (1.0 - frac) + s[idx + 1] * frac


def type7_median(values: list[float]) -> float:
    return type7_quantile(values, 0.5)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(row[k]) for k in fields})


def rederive_hit(row: dict[str, str], ref: float) -> int:
    try:
        best = float(row["best"])
    except ValueError:
        return 0
    if not math.isfinite(best):
        return 0
    return 1 if best <= ref + HIT_TOL else 0


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=here / "rows.csv")
    parser.add_argument("--summary", type=Path, default=here / "summary.csv")
    parser.add_argument("--bit-identity", type=Path, default=here / "bit_identity.csv")
    parser.add_argument("--explicit-csv", type=Path, default=here / "explicit_lj104.csv")
    parser.add_argument("--report", type=Path, default=here / "REPORT.md")
    parser.add_argument("--b-commit", default="")
    parser.add_argument("--b-tree", default="")
    args = parser.parse_args()

    with args.rows.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))

    by_key: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for row in all_rows:
        key = (row["panel"], row["cell"], row["arm"], int(row["seed"]))
        by_key[key] = row

    summary_rows: list[dict[str, str]] = []
    bit_id_rows: list[dict[str, str]] = []
    explicit_rows: list[dict[str, str]] = []

    incomplete_cells: list[str] = []
    failed_cells: list[str] = []

    for cell, family, budget, ref in CELL_ORDER:
        cell_missing = False
        per_arm: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARMS}
        for seed in DEFAULT_SEEDS:
            for arm in ARMS:
                row = by_key.get(("default", cell, arm, seed))
                if row is None:
                    cell_missing = True
                else:
                    per_arm[arm].append(row)

            ra = by_key.get(("default", cell, "A", seed))
            rb = by_key.get(("default", cell, "B", seed))
            raa = by_key.get(("default", cell, "AA", seed))
            rap = by_key.get(("default", cell, "AP", seed))
            if ra and rb and raa and rap:
                ha, hb = rederive_hit(ra, ref), rederive_hit(rb, ref)
                haa, hap = rederive_hit(raa, ref), rederive_hit(rap, ref)
                hit_eq = int(ha == hb)
                evals_eq = int(ra["evals_to_hit"] == rb["evals_to_hit"])
                charged_eq = int(ra["charged"] == rb["charged"])
                best_eq = int(ra["best_bits"] == rb["best_bits"] and ra["best"] == rb["best"])
                all_ba = int(hit_eq and evals_eq and charged_eq and best_eq)
                all_aa = int(
                    ha == haa
                    and ra["evals_to_hit"] == raa["evals_to_hit"]
                    and ra["charged"] == raa["charged"]
                    and ra["best_bits"] == raa["best_bits"]
                )
                all_ap = int(
                    ha == hap
                    and ra["evals_to_hit"] == rap["evals_to_hit"]
                    and ra["charged"] == rap["charged"]
                    and ra["best_bits"] == rap["best_bits"]
                )
                bit_id_rows.append(
                    {
                        "panel": "default",
                        "cell": cell,
                        "seed": str(seed),
                        "hit_A": str(ha),
                        "hit_B": str(hb),
                        "hit_eq": str(hit_eq),
                        "evals_A": ra["evals_to_hit"],
                        "evals_B": rb["evals_to_hit"],
                        "evals_eq": str(evals_eq),
                        "charged_A": ra["charged"],
                        "charged_B": rb["charged"],
                        "charged_eq": str(charged_eq),
                        "best_A": ra["best"],
                        "best_B": rb["best"],
                        "best_bits_A": ra["best_bits"],
                        "best_bits_B": rb["best_bits"],
                        "best_eq": str(best_eq),
                        "all_eq_BA": str(all_ba),
                        "all_eq_AA_A": str(all_aa),
                        "all_eq_AP_A": str(all_ap),
                    }
                )

        if cell_missing or any(len(per_arm[a]) != len(DEFAULT_SEEDS) for a in ARMS):
            incomplete_cells.append(cell)
            continue

        n = len(DEFAULT_SEEDS)
        hits = {a: sum(rederive_hit(r, ref) for r in per_arm[a]) for a in ARMS}
        _, fisher_p = fisher_exact(
            [[hits["B"], n - hits["B"]], [hits["A"], n - hits["A"]]],
            alternative="two-sided",
        )
        no_hit_loss = hits["B"] >= hits["A"]

        out_us = {
            a: [float(r["outside_s"]) / max(1, int(r["charged"])) * 1.0e6 for r in per_arm[a]]
            for a in ARMS
        }
        med_out = {a: type7_median(out_us[a]) for a in ARMS}
        out_ratio_ba = med_out["B"] / med_out["A"]
        out_ratio_aa = med_out["AA"] / med_out["A"]
        out_ratio_ap = med_out["AP"] / med_out["A"]
        _, mw_out_ba = mannwhitneyu(out_us["B"], out_us["A"])
        _, mw_out_aa = mannwhitneyu(out_us["AA"], out_us["A"])
        _, mw_out_ap = mannwhitneyu(out_us["AP"], out_us["A"])
        outside_ok = out_ratio_ba <= OUTSIDE_RATIO_LIMIT

        cell_bit_rows = [r for r in bit_id_rows if r["panel"] == "default" and r["cell"] == cell]
        bit_id_ba = sum(int(r["all_eq_BA"]) for r in cell_bit_rows)
        bit_id_aa = sum(int(r["all_eq_AA_A"]) for r in cell_bit_rows)
        bit_id_ap = sum(int(r["all_eq_AP_A"]) for r in cell_bit_rows)

        rss = {a: [float(r["rss_anon_kb"]) for r in per_arm[a]] for a in ARMS}
        med_rss = {a: type7_median(rss[a]) for a in ARMS}
        ctrl_shift = max(abs(med_rss["AA"] - med_rss["A"]), abs(med_rss["AP"] - med_rss["A"]), 4.0)
        band_upper_kb = max(
            med_rss["A"],
            med_rss["AA"],
            med_rss["AP"],
            med_rss["A"] + ctrl_shift,
        )
        _, mw_rss_gt = mannwhitneyu(rss["B"], rss["A"], alternative="greater")
        mem_ok = med_rss["B"] <= band_upper_kb

        cell_pass = no_hit_loss and outside_ok and mem_ok
        if not cell_pass:
            failed_cells.append(cell)

        summary_rows.append(
            {
                "cell": cell,
                "family": family,
                "n": str(n),
                "budget": str(budget),
                "hits_A": str(hits["A"]),
                "hits_B": str(hits["B"]),
                "hits_AA": str(hits["AA"]),
                "hits_AP": str(hits["AP"]),
                "fisher_p_BA": f"{fisher_p:.6g}",
                "no_hit_loss": "1" if no_hit_loss else "0",
                "med_outside_us_A": f"{med_out['A']:.4f}",
                "med_outside_us_B": f"{med_out['B']:.4f}",
                "med_outside_us_AA": f"{med_out['AA']:.4f}",
                "med_outside_us_AP": f"{med_out['AP']:.4f}",
                "outside_ratio_BA": f"{out_ratio_ba:.4f}",
                "outside_ratio_AA_A": f"{out_ratio_aa:.4f}",
                "outside_ratio_AP_A": f"{out_ratio_ap:.4f}",
                "mw_p_outside_BA": f"{mw_out_ba:.6g}",
                "mw_p_outside_AA_A": f"{mw_out_aa:.6g}",
                "mw_p_outside_AP_A": f"{mw_out_ap:.6g}",
                "outside_ok": "1" if outside_ok else "0",
                "bit_id_BA": f"{bit_id_ba}/{n}",
                "bit_id_AA_A": f"{bit_id_aa}/{n}",
                "bit_id_AP_A": f"{bit_id_ap}/{n}",
                "med_rss_A_kb": f"{med_rss['A']:.1f}",
                "med_rss_B_kb": f"{med_rss['B']:.1f}",
                "med_rss_AA_kb": f"{med_rss['AA']:.1f}",
                "med_rss_AP_kb": f"{med_rss['AP']:.1f}",
                "rss_band_upper_kb": f"{band_upper_kb:.1f}",
                "rss_ratio_BA": f"{med_rss['B'] / med_rss['A']:.4f}",
                "rss_ratio_AA_A": f"{med_rss['AA'] / med_rss['A']:.4f}",
                "rss_ratio_AP_A": f"{med_rss['AP'] / med_rss['A']:.4f}",
                "mw_p_rss_gt_BA": f"{mw_rss_gt:.6g}",
                "mem_ok": "1" if mem_ok else "0",
                "cell_verdict": "PASS" if cell_pass else "FIX",
            }
        )

    lj104_ref = -582.086642
    explicit_missing = False
    explicit_bit_ok = True
    for seed in (2001, 2021):
        ra = by_key.get(("explicit", "lj104", "A", seed))
        rb = by_key.get(("explicit", "lj104", "B", seed))
        raa = by_key.get(("explicit", "lj104", "AA", seed))
        rap = by_key.get(("explicit", "lj104", "AP", seed))
        if not (ra and rb and raa and rap):
            explicit_missing = True
        if ra and rb:
            ha, hb = rederive_hit(ra, lj104_ref), rederive_hit(rb, lj104_ref)
            haa, hap = rederive_hit(raa, lj104_ref), rederive_hit(rap, lj104_ref)
            bit_id_rows.append(
                {
                    "panel": "explicit",
                    "cell": "lj104",
                    "seed": str(seed),
                    "hit_A": str(ha),
                    "hit_B": str(hb),
                    "hit_eq": str(int(ha == hb)),
                    "evals_A": ra["evals_to_hit"],
                    "evals_B": rb["evals_to_hit"],
                    "evals_eq": str(int(ra["evals_to_hit"] == rb["evals_to_hit"])),
                    "charged_A": ra["charged"],
                    "charged_B": rb["charged"],
                    "charged_eq": str(int(ra["charged"] == rb["charged"])),
                    "best_A": ra["best"],
                    "best_B": rb["best"],
                    "best_bits_A": ra["best_bits"],
                    "best_bits_B": rb["best_bits"],
                    "best_eq": str(int(ra["best_bits"] == rb["best_bits"])),
                    "all_eq_BA": str(
                        int(
                            ha == hb
                            and ra["evals_to_hit"] == rb["evals_to_hit"]
                            and ra["charged"] == rb["charged"]
                            and ra["best_bits"] == rb["best_bits"]
                        )
                    ),
                    "all_eq_AA_A": str(
                        int(
                            ha == haa
                            and ra["evals_to_hit"] == raa["evals_to_hit"]
                            and ra["charged"] == raa["charged"]
                            and ra["best_bits"] == raa["best_bits"]
                        )
                    )
                    if raa
                    else "0",
                    "all_eq_AP_A": str(
                        int(
                            ha == hap
                            and ra["evals_to_hit"] == rap["evals_to_hit"]
                            and ra["charged"] == rap["charged"]
                            and ra["best_bits"] == rap["best_bits"]
                        )
                    )
                    if rap
                    else "0",
                }
            )
        for arm in ARMS:
            row = by_key.get(("explicit", "lj104", arm, seed))
            if row is None:
                continue
            h_arm = rederive_hit(row, lj104_ref)
            if ra is not None:
                ha = rederive_hit(ra, lj104_ref)
                eq_h = int(h_arm == ha)
                eq_e = int(row["evals_to_hit"] == ra["evals_to_hit"])
                eq_c = int(row["charged"] == ra["charged"])
                eq_b = int(row["best_bits"] == ra["best_bits"])
                if not (eq_h and eq_e and eq_c and eq_b):
                    explicit_bit_ok = False
            else:
                eq_h = eq_e = eq_c = eq_b = 0
            explicit_rows.append(
                {
                    "cell": "lj104",
                    "seed": str(seed),
                    "arm": arm,
                    "hit": str(h_arm),
                    "evals_to_hit": row["evals_to_hit"],
                    "charged": row["charged"],
                    "best": row["best"],
                    "best_bits": row["best_bits"],
                    "outside_us": row["outside_us"],
                    "rss_anon_kb": row["rss_anon_kb"],
                    "eq_A_hit": str(eq_h),
                    "eq_A_evals": str(eq_e),
                    "eq_A_charged": str(eq_c),
                    "eq_A_best_bits": str(eq_b),
                }
            )

    write_csv(args.summary, SUMMARY_FIELDS, summary_rows)
    write_csv(args.bit_identity, BIT_ID_FIELDS, bit_id_rows)
    write_csv(args.explicit_csv, EXPLICIT_FIELDS, explicit_rows)

    if incomplete_cells or explicit_missing:
        verdict = "PARTIAL"
    elif failed_cells or not explicit_bit_ok:
        verdict = "FIX"
    else:
        verdict = "PASS"

    lines = [
        "# anneal #204 r-next2 Default-Panel Gate Report",
        "",
        f"**Verdict: {verdict}**",
        "",
    ]
    if args.b_commit or args.b_tree:
        lines.append(f"- Arm B commit: `{args.b_commit}`, tree: `{args.b_tree}`")
    lines.append(f"- Default-panel cells completed: {len(summary_rows)}/{len(CELL_ORDER)}")
    if incomplete_cells:
        lines.append(f"- Incomplete cells: {', '.join(incomplete_cells)}")
    if failed_cells:
        lines.append(f"- Failed cells: {', '.join(failed_cells)}")
    lines.append("")
    lines.append("## Per-Cell Summary (`panel=default`, `n=12`, seeds 2000..2011)")
    lines.append("")
    lines.append(
        "| Cell | Hits A/B/AA/AP | Fisher p (B vs A) | Outside us A/B/AA/AP | "
        "B/A (MW p) | AA/A (MW p) | AP/A (MW p) | Bit-ID B=A | "
        "RssAnon KiB A/B/AA/AP (band) | Verdict |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in summary_rows:
        hits_str = f"{s['hits_A']}/{s['hits_B']}/{s['hits_AA']}/{s['hits_AP']}"
        out_str = (
            f"{s['med_outside_us_A']}/{s['med_outside_us_B']}/"
            f"{s['med_outside_us_AA']}/{s['med_outside_us_AP']}"
        )
        ba_str = f"{s['outside_ratio_BA']} ({s['mw_p_outside_BA']})"
        aa_str = f"{s['outside_ratio_AA_A']} ({s['mw_p_outside_AA_A']})"
        ap_str = f"{s['outside_ratio_AP_A']} ({s['mw_p_outside_AP_A']})"
        rss_str = (
            f"{s['med_rss_A_kb']}/{s['med_rss_B_kb']}/"
            f"{s['med_rss_AA_kb']}/{s['med_rss_AP_kb']} (<={s['rss_band_upper_kb']})"
        )
        lines.append(
            f"| {s['cell']} | {hits_str} | {s['fisher_p_BA']} | {out_str} | "
            f"{ba_str} | {aa_str} | {ap_str} | {s['bit_id_BA']} | "
            f"{rss_str} | {s['cell_verdict']} |"
        )

    lines.extend(["", "## Explicit `lj104` Seed Verification (`2001`, `2021`)", ""])
    lines.append(
        "| Seed | Arm | Hit | Evals to hit | Charged | Best | best_bits | Eq A (hit/evals/charged/bits) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for er in explicit_rows:
        eq_str = (
            f"{er['eq_A_hit']}/{er['eq_A_evals']}/"
            f"{er['eq_A_charged']}/{er['eq_A_best_bits']}"
        )
        lines.append(
            f"| {er['seed']} | {er['arm']} | {er['hit']} | {er['evals_to_hit']} | "
            f"{er['charged']} | {er['best']} | `{er['best_bits']}` | {eq_str} |"
        )

    mismatches = [r for r in bit_id_rows if r["all_eq_BA"] != "1"]
    lines.extend(["", "## Bit-Identity Check (`B` vs `A`)", ""])
    lines.append(
        f"- Total paired rows checked: {len(bit_id_rows)}; exact matches: "
        f"{len(bit_id_rows) - len(mismatches)}; mismatches: {len(mismatches)}."
    )
    if mismatches:
        lines.append("")
        lines.append("| Panel | Cell | Seed | A (hit/evals/charged/best) | B (hit/evals/charged/best) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for m in mismatches:
            a_str = f"{m['hit_A']}/{m['evals_A']}/{m['charged_A']}/{m['best_A']} ({m['best_bits_A']})"
            b_str = f"{m['hit_B']}/{m['evals_B']}/{m['charged_B']}/{m['best_B']} ({m['best_bits_B']})"
            lines.append(f"| {m['panel']} | {m['cell']} | {m['seed']} | {a_str} | {b_str} |")

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verdict: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
