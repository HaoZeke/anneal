#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize tip4p-hits-b2 gate rows, exact tests, A/A calibration, placement band, and verdict."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ALPHA = 0.05
MAX_OVERALL_OUTSIDE_RATIO = 1.10
PLACEMENT_VARIANT_BANDS = ("PL1", "PL2", "PL3", "PL4", "PL5")
# The water_hexamer A/A series is the second release build, not a separate rerun.
PLACEMENT_BANDS = ("AA",) + PLACEMENT_VARIANT_BANDS
# One comparison per layout variant, applied to the single pooled test.
PLACEMENT_BONFERRONI_M = len(PLACEMENT_VARIANT_BANDS)

EXPECTED_AB_CELLS = [
    ("tip4p", "tip4p8"),
    ("tip4p", "tip4p15"),
    ("tip4p", "tip4p21"),
    ("tip4p", "tip4p25"),
    ("molecular", "water_hexamer"),
    ("conformer", "rq3"),
    ("conformer", "xld"),
    ("lj", "lj38"),
    ("lj", "lj75"),
    ("lj", "lj98"),
]

EXPECTED_AA_CELLS = [
    ("tip4p", "tip4p8"),
    ("tip4p", "tip4p15"),
    ("conformer", "rq3"),
    ("lj", "lj38"),
    ("molecular", "water_hexamer"),
]

# Rigid-body search is the armed path. These cells must stay on A's trajectories.
UNTOUCHED_SYSTEMS = {"rq3", "xld", "lj38", "lj75", "lj98"}

SUMMARY_FIELDS = [
    "band",
    "panel",
    "system",
    "status",
    "n_a",
    "n_b",
    "budget",
    "hits_a",
    "hits_b",
    "fisher_p",
    "calls_median_a",
    "calls_median_b",
    "mannwhitney_u_calls",
    "mannwhitney_p_calls",
    "pot_median_s_a",
    "pot_median_s_b",
    "outside_median_s_a",
    "outside_median_s_b",
    "outside_ratio_b_over_a",
    "mannwhitney_u_outside",
    "mannwhitney_p_outside",
    "wall_median_s_a",
    "wall_median_s_b",
    "bit_identical_seeds",
    "hit_drop",
    "call_increase",
    "outside_increase",
    "gate_fail",
    "fail_reasons",
    "base_ref",
    "base_tree",
    "harness_run_gate",
    "harness_summarize",
    "harness_snakefile",
]

IDENTITY_FIELDS = [
    "band",
    "panel",
    "system",
    "seed",
    "bit_identical",
    "hit_a",
    "hit_b",
    "calls_a",
    "calls_b",
    "charged_a",
    "charged_b",
    "best_a",
    "best_b",
    "energy_bits_a",
    "energy_bits_b",
    "trace_sha256_a",
    "trace_sha256_b",
]


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return float("nan")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total_hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, total_hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, total_hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, total_hits - n_b), min(n_a, total_hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1.0 + 1e-12)))


def mann_whitney_exact(left: list[float], right: list[float]) -> tuple[float, float]:
    """Exact two-sided Mann-Whitney U test via integer DP on doubled midranks (handles ties)."""
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann-whitney needs non-empty samples")
    n = n1 + n2
    pooled = sorted(left + right)
    doubled_ranks: dict[float, int] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        # Midrank is (i + 1 + j + 1) / 2; doubled midrank is i + j + 2 (always an integer).
        doubled_ranks[pooled[i]] = i + j + 2
        i = j + 1

    r2_pool = [doubled_ranks[v] for v in pooled]
    s2_obs = sum(doubled_ranks[v] for v in left)
    u_left = (s2_obs - n1 * (n1 + 1)) / 2.0

    mean_s2 = n1 * (n + 1)
    dist_obs = abs(s2_obs - mean_s2)
    max_s2 = sum(sorted(r2_pool, reverse=True)[:n1])

    dp = [[0] * (max_s2 + 1) for _ in range(n1 + 1)]
    dp[0][0] = 1
    for idx, w in enumerate(r2_pool):
        max_k = min(n1, idx + 1)
        for k in range(max_k, 0, -1):
            prev = dp[k - 1]
            cur = dp[k]
            for s in range(max_s2, w - 1, -1):
                if prev[s - w]:
                    cur[s] += prev[s - w]

    total = math.comb(n, n1)
    if sum(dp[n1]) != total:
        raise RuntimeError("mann-whitney DP count mismatch")
    extreme = sum(cnt for s, cnt in enumerate(dp[n1]) if cnt and abs(s - mean_s2) >= dist_obs)
    return u_left, min(1.0, extreme / total)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_tests_file(path: Path | None) -> tuple[bool, list[str]]:
    if path is None or not path.is_file():
        return True, ["tests.txt not supplied"]
    lines = path.read_text(encoding="utf-8").splitlines()
    b_only_fail = False
    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        summary_lines.append(stripped)
        if "b_only_failures=" in stripped:
            val = int(stripped.split("b_only_failures=", 1)[1].split()[0])
            if val > 0:
                b_only_fail = True
        if "verdict=FAIL" in stripped:
            b_only_fail = True
    return (not b_only_fail), summary_lines


def fmt_p(p: float) -> str:
    return f"{p:.6g}"


def placement_cell_fails(ratio: float, lo: float, hi: float, p_adj: float) -> bool:
    """Fail an untouched cell when its outside ratio is past the placement band and the corrected test is significant."""
    if not (math.isfinite(ratio) and math.isfinite(lo) and math.isfinite(hi) and math.isfinite(p_adj)):
        return False
    outside = ratio < lo or ratio > hi
    return outside and p_adj < ALPHA


def _hex_rows(raw_summary: list[dict[str, object]], band: str) -> dict[str, object] | None:
    for item in raw_summary:
        if item["band"] == band and item["system"] == "water_hexamer":
            return item
    return None


def analyze_placement(raw_summary: list[dict[str, object]]) -> dict[str, object]:
    """Median outside ratios of the five layout variants and the hexamer A/A, plus B versus the pooled variants."""
    found: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for band in PLACEMENT_BANDS:
        item = _hex_rows(raw_summary, band)
        if item is None or item["status"] != "ok":
            missing.append(band)
        elif item is not None:
            found[band] = item
    ab = _hex_rows(raw_summary, "AB")
    if ab is None or ab["status"] != "ok":
        missing.append("AB")
    if missing:
        return {"ready": False, "missing": missing}

    series: list[dict[str, object]] = []
    ratios: list[float] = []
    for band in PLACEMENT_BANDS:
        item = found[band]
        ratio = float(item["outside_ratio"])
        ratios.append(ratio)
        right_rows = item["rows_r"]
        if band == "AA":
            kind = "second_build"
        else:
            kind = str(right_rows[0]["driver"]) if right_rows else ""
        series.append(
            {
                "band": band,
                "kind": kind,
                "ratio": ratio,
                "med_a": float(item["med_out_a"]),
                "med_b": float(item["med_out_b"]),
                "bit_identical": int(item["bit_identical"]),
                "n_common": int(item["n_common"]),
                "p_out": float(item["p_out"]),
            }
        )
    lo, hi = min(ratios), max(ratios)
    pooled: list[float] = []
    for band in PLACEMENT_VARIANT_BANDS:
        pooled.extend(float(row["outside_s"]) for row in found[band]["rows_r"])
    b_times = [float(row["outside_s"]) for row in ab["rows_r"]]
    u_stat, p_value = mann_whitney_exact(b_times, pooled)
    p_adj = min(1.0, p_value * PLACEMENT_BONFERRONI_M)

    def rows_match(left: dict[str, str], right: dict[str, str]) -> bool:
        return (
            left["calls"] == right["calls"]
            and left["charged"] == right["charged"]
            and left["hit"] == right["hit"]
            and left["best"] == right["best"]
            and left["final_energy_bits"] == right["final_energy_bits"]
            and bool(left["trace_sha256"])
            and left["trace_sha256"] == right["trace_sha256"]
        )

    archived = {int(row["seed"]): row for row in ab["rows_l"]}
    cross_ok = True
    for band in PLACEMENT_BANDS:
        item = found[band]
        arms = [item["rows_l"], item["rows_r"]]
        # AA is the second build versus the first. Variant bands compare the right arm to archived A.
        check_rows = arms[0] + arms[1] if band == "AA" else arms[1]
        for row in check_rows:
            base = archived.get(int(row["seed"]))
            if base is None or not rows_match(base, row):
                cross_ok = False
    within_ok = all(int(found[band]["bit_identical"]) == int(found[band]["n_common"]) for band in PLACEMENT_BANDS)
    return {
        "ready": True,
        "missing": [],
        "series": series,
        "lo": lo,
        "hi": hi,
        "u": u_stat,
        "p": p_value,
        "p_adj": p_adj,
        "n_b": len(b_times),
        "n_pooled": len(pooled),
        "within_ok": within_ok,
        "cross_ok": cross_ok,
    }


def abba_problems(rows: list[dict[str, str]]) -> list[str]:
    """Finished cells must split twelve seeds into two opposite orders of six."""
    orders: dict[tuple[str, str], dict[str, int]] = {}
    seen: dict[tuple[str, str, str], str] = {}
    problems: list[str] = []
    for row in rows:
        cell = (row["band"], row["system"])
        key = (row["band"], row["system"], row["seed"])
        order = row["arm_order"]
        previous = seen.get(key)
        if previous is None:
            seen[key] = order
            bucket = orders.setdefault(cell, {})
            bucket[order] = bucket.get(order, 0) + 1
        elif previous != order:
            problems.append(f"ABBA/{cell[0]}/{cell[1]} seed {row['seed']} mixed orders")
    for band, system in sorted(orders):
        counts = orders[(band, system)]
        if sum(counts.values()) != 12:
            continue
        if sorted(counts.values()) != [6, 6]:
            problems.append(f"ABBA/{band}/{system} unbalanced {counts}")
    return problems


def summarize(rows: list[dict[str, str]], tests_path: Path | None = None, kit: dict[str, str] | None = None) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str]:
    by_key: dict[tuple[str, str, str, str], dict[int, dict[str, str]]] = {}
    for row in rows:
        band = row["band"]
        panel = row["panel"]
        system = row["system"]
        arm = row["arm"]
        seed = int(row["seed"])
        by_key.setdefault((band, panel, system, arm), {})[seed] = row

    identity_rows: list[dict[str, str]] = []
    raw_summary: list[dict[str, object]] = []

    def pair_names(band: str) -> tuple[str, str]:
        if band == "AB":
            return ("A", "B")
        if band.startswith("PL") and band != "PLAA":
            return ("A", "V")
        return ("A1", "A2")

    cells: list[tuple[str, str, str]] = []
    for panel, system in EXPECTED_AA_CELLS:
        cells.append(("AA", panel, system))
    for panel, system in EXPECTED_AB_CELLS:
        cells.append(("AB", panel, system))

    # Also include any unexpected extra cells present in rows.csv.
    seen_cells = set(cells)
    for band, panel, system, _arm in sorted(by_key):
        if (band, panel, system) not in seen_cells:
            cells.append((band, panel, system))
            seen_cells.add((band, panel, system))

    for band, panel, system in cells:
        left_arm, right_arm = pair_names(band)
        arm_l = by_key.get((band, panel, system, left_arm), {})
        arm_r = by_key.get((band, panel, system, right_arm), {})
        seeds = sorted(set(arm_l) & set(arm_r))
        n_l, n_r = len(arm_l), len(arm_r)
        if not seeds or n_l != 12 or n_r != 12 or sorted(arm_l) != sorted(arm_r):
            status = f"incomplete:l={n_l},r={n_r}"
        else:
            status = "ok"

        rows_l = [arm_l[s] for s in sorted(arm_l)]
        rows_r = [arm_r[s] for s in sorted(arm_r)]
        n = len(seeds)

        bit_id_count = 0
        for s in seeds:
            rl, rr = arm_l[s], arm_r[s]
            bit_same = (
                rl["hit"] == rr["hit"]
                and rl["calls"] == rr["calls"]
                and rl["charged"] == rr["charged"]
                and rl["best"] == rr["best"]
                and rl["final_energy_bits"] == rr["final_energy_bits"]
                and bool(rl["trace_sha256"])
                and rl["trace_sha256"] == rr["trace_sha256"]
            )
            bit_id_count += int(bit_same)
            identity_rows.append(
                {
                    "band": band,
                    "panel": panel,
                    "system": system,
                    "seed": str(s),
                    "bit_identical": str(int(bit_same)),
                    "hit_a": rl["hit"],
                    "hit_b": rr["hit"],
                    "calls_a": rl["calls"],
                    "calls_b": rr["calls"],
                    "charged_a": rl["charged"],
                    "charged_b": rr["charged"],
                    "best_a": rl["best"],
                    "best_b": rr["best"],
                    "energy_bits_a": rl["final_energy_bits"],
                    "energy_bits_b": rr["final_energy_bits"],
                    "trace_sha256_a": rl["trace_sha256"],
                    "trace_sha256_b": rr["trace_sha256"],
                }
            )

        if not rows_l or not rows_r:
            raw_summary.append(
                {
                    "band": band,
                    "panel": panel,
                    "system": system,
                    "status": status,
                    "n_a": n_l,
                    "n_b": n_r,
                    "budget": "",
                    "hits_a": 0,
                    "hits_b": 0,
                    "fisher_p": 1.0,
                    "med_calls_a": float("nan"),
                    "med_calls_b": float("nan"),
                    "u_calls": float("nan"),
                    "p_calls": 1.0,
                    "med_pot_a": float("nan"),
                    "med_pot_b": float("nan"),
                    "med_out_a": float("nan"),
                    "med_out_b": float("nan"),
                    "outside_ratio": float("nan"),
                    "u_out": float("nan"),
                    "p_out": 1.0,
                    "med_wall_a": float("nan"),
                    "med_wall_b": float("nan"),
                    "bit_identical": 0,
                    "n_common": 0,
                    "hit_drop": False,
                    "call_inc": False,
                    "out_inc": False,
                    "base_ref": "",
                    "base_tree": "",
                    "harness_run_gate": "",
                    "harness_summarize": "",
                    "harness_snakefile": "",
                    "rows_l": rows_l,
                    "rows_r": rows_r,
                }
            )
            continue

        hits_l = sum(int(r["hit"]) for r in rows_l)
        hits_r = sum(int(r["hit"]) for r in rows_r)
        calls_l = [float(r["calls"]) for r in rows_l]
        calls_r = [float(r["calls"]) for r in rows_r]
        pot_l = [float(r["pot_s"]) for r in rows_l]
        pot_r = [float(r["pot_s"]) for r in rows_r]
        out_l = [float(r["outside_s"]) for r in rows_l]
        out_r = [float(r["outside_s"]) for r in rows_r]
        wall_l = [float(r["wall_s"]) for r in rows_l]
        wall_r = [float(r["wall_s"]) for r in rows_r]

        f_p = fisher_two_sided(hits_l, n_l, hits_r, n_r)
        u_calls, p_calls = mann_whitney_exact(calls_r, calls_l)
        u_out, p_out = mann_whitney_exact(out_r, out_l)

        med_calls_l = type7(calls_l, 0.5)
        med_calls_r = type7(calls_r, 0.5)
        med_pot_l = type7(pot_l, 0.5)
        med_pot_r = type7(pot_r, 0.5)
        med_out_l = type7(out_l, 0.5)
        med_out_r = type7(out_r, 0.5)
        med_wall_l = type7(wall_l, 0.5)
        med_wall_r = type7(wall_r, 0.5)
        outside_ratio = (med_out_r / med_out_l) if med_out_l > 0.0 else float("nan")

        hit_drop = hits_r < hits_l
        call_inc = (med_calls_r > med_calls_l) or (u_calls > (n_l * n_r) / 2.0)
        out_inc = (med_out_r > med_out_l) or (u_out > (n_l * n_r) / 2.0)

        raw_summary.append(
            {
                "band": band,
                "panel": panel,
                "system": system,
                "status": status,
                "n_a": n_l,
                "n_b": n_r,
                "budget": rows_l[0]["budget"],
                "hits_a": hits_l,
                "hits_b": hits_r,
                "fisher_p": f_p,
                "med_calls_a": med_calls_l,
                "med_calls_b": med_calls_r,
                "u_calls": u_calls,
                "p_calls": p_calls,
                "med_pot_a": med_pot_l,
                "med_pot_b": med_pot_r,
                "med_out_a": med_out_l,
                "med_out_b": med_out_r,
                "outside_ratio": outside_ratio,
                "u_out": u_out,
                "p_out": p_out,
                "med_wall_a": med_wall_l,
                "med_wall_b": med_wall_r,
                "bit_identical": bit_id_count,
                "n_common": n,
                "hit_drop": hit_drop,
                "call_inc": call_inc,
                "out_inc": out_inc,
                "base_ref": " ".join(sorted({r["base_ref"] for r in rows_l + rows_r})),
                "base_tree": " ".join(sorted({r["base_tree"] for r in rows_l + rows_r})),
                "harness_run_gate": " ".join(sorted({r["harness_run_gate"] for r in rows_l + rows_r})),
                "harness_summarize": " ".join(sorted({r["harness_summarize"] for r in rows_l + rows_r})),
                "harness_snakefile": " ".join(sorted({r["harness_snakefile"] for r in rows_l + rows_r})),
                "rows_l": rows_l,
                "rows_r": rows_r,
            }
        )

    aa_items = [item for item in raw_summary if item["band"] == "AA"]
    ab_items = [item for item in raw_summary if item["band"] == "AB"]

    aa_complete = [item for item in aa_items if item["status"] == "ok"]
    aa_ratios = [
        float(item["outside_ratio"])
        for item in aa_complete
        if math.isfinite(float(item["outside_ratio"]))
    ]
    aa_max_ratio = max(aa_ratios) if aa_ratios else 1.0
    aa_min_outside_p = min((float(item["p_out"]) for item in aa_complete), default=1.0)
    aa_total_out_l = sum(float(r["outside_s"]) for item in aa_complete for r in item["rows_l"])
    aa_total_out_r = sum(float(r["outside_s"]) for item in aa_complete for r in item["rows_r"])
    aa_overall_ratio = (aa_total_out_r / aa_total_out_l) if aa_total_out_l > 0.0 else 1.0

    ab_total_out_a = sum(float(r["outside_s"]) for item in ab_items for r in item["rows_l"])
    ab_total_out_b = sum(float(r["outside_s"]) for item in ab_items for r in item["rows_r"])
    overall_outside_ratio = (ab_total_out_b / ab_total_out_a) if ab_total_out_a > 0.0 else float("nan")

    tests_ok, test_lines = parse_tests_file(tests_path)
    gate_failures: list[str] = []
    incomplete_cells: list[str] = []
    placement = analyze_placement(raw_summary)

    if not tests_ok:
        gate_failures.append("B-only test failure in tests.txt")

    aa_names = {item["system"] for item in aa_complete}
    if len(aa_complete) < 5 or "tip4p15" not in aa_names or "water_hexamer" not in aa_names:
        incomplete_cells.append(
            "A/A band missing >=5 complete cells including tip4p15 and water_hexamer"
        )

    for item in aa_complete:
        if int(item["bit_identical"]) != int(item["n_common"]):
            gate_failures.append(
                f"AA/{item['system']} non-deterministic ({item['bit_identical']}/{item['n_common']} bit-identical)"
            )

    summary_csv_rows: list[dict[str, str]] = []
    for item in raw_summary:
        band = str(item["band"])
        panel = str(item["panel"])
        system = str(item["system"])
        status = str(item["status"])
        reasons: list[str] = []

        if status != "ok":
            incomplete_cells.append(f"{band}/{system} ({status})")

        if str(band).startswith("PL") and status == "ok":
            if int(item["bit_identical"]) != int(item["n_common"]):
                reasons.append(f"placement_not_bit_identical:{item['bit_identical']}/{item['n_common']}")

        if band == "AB" and status == "ok":
            n_common = int(item["n_common"])
            bit_id = int(item["bit_identical"])
            if panel == "lj" and bit_id != n_common:
                reasons.append(f"lj_divergence:{bit_id}/{n_common}")
            if bool(item["hit_drop"]) and float(item["fisher_p"]) < ALPHA:
                reasons.append(f"hit_loss:p={fmt_p(float(item['fisher_p']))}")
            if bool(item["call_inc"]) and float(item["p_calls"]) < ALPHA:
                reasons.append(f"call_regression:p={fmt_p(float(item['p_calls']))}")
            out_ratio = float(item["outside_ratio"])
            p_out = float(item["p_out"])
            hexamer = panel == "molecular" and system == "water_hexamer"
            untouched = system in UNTOUCHED_SYSTEMS
            if hexamer or untouched:
                if not placement["ready"]:
                    incomplete_cells.append(
                        "placement band for "
                        + system
                        + " ("
                        + ",".join(str(b) for b in placement["missing"])
                        + ")"
                    )
                elif hexamer:
                    if not bool(placement["within_ok"]) or not bool(placement["cross_ok"]):
                        reasons.append("placement_trajectory_mismatch")
                    # A loss is a slower outside time. A faster ratio is not a loss.
                    if (
                        math.isfinite(out_ratio)
                        and out_ratio > float(placement["hi"])
                        and float(placement["p_adj"]) < ALPHA
                    ):
                        reasons.append(
                            "outside_placement_band:"
                            f"ratio={out_ratio:.4f},"
                            f"band={float(placement['lo']):.4f}-{float(placement['hi']):.4f},"
                            f"p={fmt_p(float(placement['p']))},"
                            f"p_adj={fmt_p(float(placement['p_adj']))}"
                        )
                else:
                    if panel == "conformer" and bit_id != n_common:
                        reasons.append(f"untouched_divergence:{bit_id}/{n_common}")
                    if (
                        bool(item["out_inc"])
                        and math.isfinite(out_ratio)
                        and out_ratio > float(placement["hi"])
                        and p_out < ALPHA
                    ):
                        reasons.append(
                            "outside_placement_band:"
                            f"ratio={out_ratio:.4f},"
                            f"band={float(placement['lo']):.4f}-{float(placement['hi']):.4f},"
                            f"p={fmt_p(p_out)}"
                        )
            elif bool(item["out_inc"]) and p_out < ALPHA:
                if out_ratio > aa_max_ratio or p_out < aa_min_outside_p:
                    reasons.append(f"outside_regression_beyond_AA:ratio={out_ratio:.3f},p={fmt_p(p_out)}")

        if reasons:
            gate_failures.extend(f"{band}/{system}: {r}" for r in reasons)

        n_a = int(item["n_a"])
        n_b = int(item["n_b"])
        n_c = int(item["n_common"])
        summary_csv_rows.append(
            {
                "band": band,
                "panel": panel,
                "system": system,
                "status": status,
                "n_a": str(n_a),
                "n_b": str(n_b),
                "budget": str(item["budget"]),
                "hits_a": f"{item['hits_a']}/{n_a}" if n_a else "",
                "hits_b": f"{item['hits_b']}/{n_b}" if n_b else "",
                "fisher_p": fmt_p(float(item["fisher_p"])) if n_a and n_b else "",
                "calls_median_a": f"{float(item['med_calls_a']):.1f}" if n_a else "",
                "calls_median_b": f"{float(item['med_calls_b']):.1f}" if n_b else "",
                "mannwhitney_u_calls": f"{float(item['u_calls']):.1f}" if n_a and n_b else "",
                "mannwhitney_p_calls": fmt_p(float(item["p_calls"])) if n_a and n_b else "",
                "pot_median_s_a": f"{float(item['med_pot_a']):.6f}" if n_a else "",
                "pot_median_s_b": f"{float(item['med_pot_b']):.6f}" if n_b else "",
                "outside_median_s_a": f"{float(item['med_out_a']):.6f}" if n_a else "",
                "outside_median_s_b": f"{float(item['med_out_b']):.6f}" if n_b else "",
                "outside_ratio_b_over_a": f"{float(item['outside_ratio']):.4f}" if math.isfinite(float(item["outside_ratio"])) else "",
                "mannwhitney_u_outside": f"{float(item['u_out']):.1f}" if n_a and n_b else "",
                "mannwhitney_p_outside": fmt_p(float(item["p_out"])) if n_a and n_b else "",
                "wall_median_s_a": f"{float(item['med_wall_a']):.6f}" if n_a else "",
                "wall_median_s_b": f"{float(item['med_wall_b']):.6f}" if n_b else "",
                "bit_identical_seeds": f"{item['bit_identical']}/{n_c}" if n_c else "",
                "hit_drop": str(int(bool(item["hit_drop"]))),
                "call_increase": str(int(bool(item["call_inc"]))),
                "outside_increase": str(int(bool(item["out_inc"]))),
                "gate_fail": str(int(bool(reasons))),
                "fail_reasons": ";".join(reasons),
                "base_ref": str(item["base_ref"]),
                "base_tree": str(item["base_tree"]),
                "harness_run_gate": str(item["harness_run_gate"]),
                "harness_summarize": str(item["harness_summarize"]),
                "harness_snakefile": str(item["harness_snakefile"]),
            }
        )

    if math.isfinite(overall_outside_ratio) and overall_outside_ratio > MAX_OVERALL_OUTSIDE_RATIO:
        gate_failures.append(
            f"overall outside-potential ratio {overall_outside_ratio:.4f} > {MAX_OVERALL_OUTSIDE_RATIO:.2f}"
        )

    order_problems = abba_problems(rows)
    gate_failures.extend(order_problems)

    if gate_failures:
        verdict = "FAIL"
    elif incomplete_cells:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"

    hex_ab = _hex_rows(raw_summary, "AB")
    hex_ratio = float(hex_ab["outside_ratio"]) if hex_ab and hex_ab["status"] == "ok" else float("nan")

    def ab_ratio(system: str) -> float:
        for item in ab_items:
            if item["system"] == system and item["status"] == "ok":
                return float(item["outside_ratio"])
        return float("nan")

    tip4p21_ratio = ab_ratio("tip4p21")
    tip4p25_ratio = ab_ratio("tip4p25")
    tip4p15_hits = next(
        (
            (int(item["hits_a"]), int(item["hits_b"]), int(item["n_a"]))
            for item in ab_items
            if item["system"] == "tip4p15" and item["status"] == "ok"
        ),
        None,
    )

    def inside_band(ratio: float) -> str:
        if not placement["ready"] or not math.isfinite(ratio):
            return "unknown"
        return "inside" if float(placement["lo"]) <= ratio <= float(placement["hi"]) else "outside"

    v_lines = [
        f"verdict: {verdict}",
        f"alpha: {ALPHA}",
        f"overall_outside_s_a: {ab_total_out_a:.6f}",
        f"overall_outside_s_b: {ab_total_out_b:.6f}",
        f"overall_outside_ratio_b_over_a: {overall_outside_ratio:.6f}" if math.isfinite(overall_outside_ratio) else "overall_outside_ratio_b_over_a: nan",
        f"aa_complete_cells: {len(aa_complete)}",
        f"aa_overall_outside_ratio: {aa_overall_ratio:.6f}",
        f"aa_max_cell_outside_ratio: {aa_max_ratio:.6f}",
        f"aa_min_outside_p: {fmt_p(aa_min_outside_p)}",
        f"tests_ok: {int(tests_ok)}",
        f"placement_ready: {int(bool(placement['ready']))}",
        f"placement_bonferroni_m: {PLACEMENT_BONFERRONI_M}",
        f"abba_balanced: {int(not order_problems)}",
    ]
    if kit:
        for key in ("base_ref", "base_tree", "b_kit_sha256", "b_kit_tree", "harness_run_gate", "harness_summarize", "harness_snakefile"):
            if kit.get(key):
                v_lines.append(f"{key}: {kit[key]}")
    if placement["ready"]:
        v_lines.extend(
            [
                f"placement_ratio_min: {float(placement['lo']):.6f}",
                f"placement_ratio_max: {float(placement['hi']):.6f}",
                f"placement_hexamer_ratio: {hex_ratio:.6f}" if math.isfinite(hex_ratio) else "placement_hexamer_ratio: nan",
                f"placement_hexamer_vs_band: {inside_band(hex_ratio)}",
                f"placement_mw_u: {float(placement['u']):.1f}",
                f"placement_mw_p: {fmt_p(float(placement['p']))}",
                f"placement_mw_p_adj: {fmt_p(float(placement['p_adj']))}",
                f"placement_n_b: {int(placement['n_b'])}",
                f"placement_n_pooled: {int(placement['n_pooled'])}",
                f"placement_bit_identical: {int(bool(placement['within_ok']) and bool(placement['cross_ok']))}",
                f"tip4p21_outside_ratio: {tip4p21_ratio:.6f}",
                f"tip4p21_vs_placement_band: {inside_band(tip4p21_ratio)}",
                f"tip4p25_outside_ratio: {tip4p25_ratio:.6f}",
                f"tip4p25_vs_placement_band: {inside_band(tip4p25_ratio)}",
            ]
        )
    else:
        v_lines.append("placement_missing: " + ",".join(str(b) for b in placement["missing"]))
    if tip4p15_hits is not None:
        hits_a, hits_b, n_hits = tip4p15_hits
        v_lines.extend(
            [
                f"tip4p15_hits_a: {hits_a}/{n_hits}",
                f"tip4p15_hits_b: {hits_b}/{n_hits}",
                f"tip4p15_prior_gain_holds: {int(hits_a == 7 and hits_b == 12 and n_hits == 12)}",
            ]
        )
    if gate_failures:
        v_lines.append("failures:")
        v_lines.extend(f"- {msg}" for msg in gate_failures)
    if incomplete_cells:
        v_lines.append("incomplete:")
        v_lines.extend(f"- {msg}" for msg in incomplete_cells)
    verdict_text = "\n".join(v_lines) + "\n"

    ab_rows = [r for r in summary_csv_rows if r["band"] == "AB"]
    aa_rows = [r for r in summary_csv_rows if r["band"] == "AA"]

    def md_table(entries: list[dict[str, str]], label_a: str, label_b: str) -> list[str]:
        out = [
            f"| Panel | System | Budget | Hits {label_a}/{label_b} | Fisher p | Med Calls {label_a}/{label_b} | MW p (calls) | Out Med {label_a}/{label_b} (s) | Ratio {label_b}/{label_a} | MW p (out) | Bit-ID | Gate |",
            "| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
        for r in entries:
            gate_col = "FAIL" if r["gate_fail"] == "1" else ("ok" if r["status"] == "ok" else r["status"])
            out.append(
                f"| {r['panel']} | {r['system']} | {r['budget']} | {r['hits_a']} vs {r['hits_b']} | {r['fisher_p']} | "
                f"{r['calls_median_a']} / {r['calls_median_b']} | {r['mannwhitney_p_calls']} | "
                f"{r['outside_median_s_a']} / {r['outside_median_s_b']} | {r['outside_ratio_b_over_a']} | "
                f"{r['mannwhitney_p_outside']} | {r['bit_identical_seeds']} | {gate_col} |"
            )
        return out

    rep = [
        "# tip4p-hits-b2 Regression and Performance Gate",
        "",
        f"## Verdict: `{verdict}`",
        "",
    ]
    if kit:
        rep.extend(
            [
                f"- Base ref: `{kit.get('base_ref', '')}`",
                f"- Base tree: `{kit.get('base_tree', '')}`",
                f"- B kit sha256: `{kit.get('b_kit_sha256', '')}`",
                f"- B kit tree: `{kit.get('b_kit_tree', '')}`",
                "",
            ]
        )
    rep.extend(
        [
        f"- Overall outside-potential wall ratio (B / A): `{overall_outside_ratio:.4f}` (`{ab_total_out_b:.3f} s` vs `{ab_total_out_a:.3f} s`, limit `<= {MAX_OVERALL_OUTSIDE_RATIO:.2f}x`)" if math.isfinite(overall_outside_ratio) else "- Overall outside-potential wall ratio: incomplete",
        f"- A/A calibration band ({len(aa_complete)} cells, including `tip4p15` and `water_hexamer`, second release build): max cell outside ratio `{aa_max_ratio:.4f}`, overall A/A outside ratio `{aa_overall_ratio:.4f}`, min exact Mann-Whitney outside p `{fmt_p(aa_min_outside_p)}`",
        f"- Test suite parity (`tests.txt`): `{'PASS (zero B-only failures)' if tests_ok else 'FAIL'}`",
        "",
        "## Test Suite Results",
        "",
        ]
    )
    if test_lines:
        rep.append("```text")
        rep.extend(test_lines)
        rep.append("```")
    else:
        rep.append("No test records found.")
    rep.extend(["", "## A/B Panel (12 seeds, 2000-2011, serial core 0, ABBA per-seed order)", ""])
    rep.extend(md_table(ab_rows, "A", "B"))
    rep.extend(["", "## A/A Calibration Band (12 seeds, 2000-2011, serial core 0, A1 vs A2)", ""])
    rep.extend(md_table(aa_rows, "A1", "A2"))
    rep.extend(["", "## Placement Band (water_hexamer)", ""])
    rep.append(
        "Five arm A layout variants change only object size or section alignment of `anneal._core`. "
        "Each variant is run against arm A. Their median outside ratios are pooled with the "
        "second-build A/A on `water_hexamer`. Seeds 2000-2011, serial core 0, ABBA order, one thread."
    )
    rep.append("")
    if not placement["ready"]:
        rep.append(
            "Placement band incomplete: " + ", ".join(str(b) for b in placement["missing"]) + "."
        )
    else:
        rep.extend(
            [
                "| Series | Kind | Out med left/right (s) | Ratio | Exact MW p | Bit-ID |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for series in placement["series"]:
            rep.append(
                f"| {series['band']} | {series['kind']} | {float(series['med_a']):.6f} / {float(series['med_b']):.6f} | "
                f"{float(series['ratio']):.4f} | {fmt_p(float(series['p_out']))} | "
                f"{series['bit_identical']}/{series['n_common']} |"
            )
        hex_vs = inside_band(hex_ratio)
        rep.extend(
            [
                "",
                f"- Pooled placement band (min and max of the six median ratios): `{float(placement['lo']):.4f}` to `{float(placement['hi']):.4f}`",
                f"- `water_hexamer` B/A median outside ratio `{hex_ratio:.4f}` is {hex_vs} the placement band"
                if math.isfinite(hex_ratio)
                else "- `water_hexamer` B/A median outside ratio: incomplete",
                f"- Exact Mann-Whitney of arm B outside times (n={int(placement['n_b'])}) against the pooled variant outside times (n={int(placement['n_pooled'])}): U=`{float(placement['u']):.1f}`, p=`{fmt_p(float(placement['p']))}`, Bonferroni m=`{PLACEMENT_BONFERRONI_M}`, p_adj=`{fmt_p(float(placement['p_adj']))}`",
                "- `water_hexamer` loses on outside time only when its ratio is above the band and p_adj < 0.05",
                "- Untouched cells (`rq3`, `xld`, `lj38`, `lj75`, `lj98`) lose on outside time only when their ratio is above that band and their exact Mann-Whitney p < 0.05",
                f"- `tip4p21` outside ratio `{tip4p21_ratio:.4f}` is {inside_band(tip4p21_ratio)} the placement band",
                f"- `tip4p25` outside ratio `{tip4p25_ratio:.4f}` is {inside_band(tip4p25_ratio)} the placement band",
                f"- Variant and A/A trajectories match arm A: `{int(bool(placement['within_ok']) and bool(placement['cross_ok']))}`",
            ]
        )
    if gate_failures:
        rep.extend(["", "## Gate Failure Details", ""])
        rep.extend(f"- {msg}" for msg in gate_failures)
    if incomplete_cells:
        rep.extend(["", "## Incomplete Cells", ""])
        rep.extend(f"- {msg}" for msg in incomplete_cells)
    rep.append("")
    report_text = "\n".join(rep)

    return summary_csv_rows, identity_rows, verdict_text, report_text


def self_test() -> None:
    import itertools

    assert abs(type7([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.5) - 3.5) < 1e-12
    assert abs(fisher_two_sided(12, 12, 0, 12) - 2.0 / math.comb(24, 12)) < 1e-15
    assert fisher_two_sided(6, 12, 6, 12) == 1.0

    # All-tied returns p=1.0
    u, p = mann_whitney_exact([10.0] * 12, [10.0] * 12)
    assert u == 72.0 and p == 1.0

    # Brute-force check on small tied sample (n1=4, n2=4)
    a = [1.0, 2.0, 2.0, 5.0]
    b = [2.0, 3.0, 4.0, 4.0]
    u_got, p_got = mann_whitney_exact(a, b)
    pooled = a + b
    n1 = len(a)

    def u_stat(xs: list[float], ys: list[float]) -> float:
        return sum(1.0 for x in xs for y in ys if x > y) + 0.5 * sum(1.0 for x in xs for y in ys if x == y)

    obs_dev = abs(u_got - n1 * len(b) / 2.0)
    extreme = 0
    total = 0
    for idx in itertools.combinations(range(len(pooled)), n1):
        iset = set(idx)
        xs = [pooled[i] for i in range(len(pooled)) if i in iset]
        ys = [pooled[i] for i in range(len(pooled)) if i not in iset]
        if abs(u_stat(xs, ys) - n1 * len(b) / 2.0) >= obs_dev - 1e-12:
            extreme += 1
        total += 1
    assert abs(p_got - extreme / total) < 1e-12, (p_got, extreme / total)

    try:
        from scipy.stats import fisher_exact, mannwhitneyu
    except ImportError:
        pass
    else:
        left = [10.0, 11.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 27.0, 30.0, 31.0, 33.0]
        right = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 13.0, 14.0, 16.0]
        want = mannwhitneyu(left, right, alternative="two-sided", method="exact")
        u, p = mann_whitney_exact(left, right)
        assert abs(u - want.statistic) < 1e-9 and abs(p - want.pvalue) < 1e-12
        assert abs(fisher_two_sided(9, 12, 2, 12) - fisher_exact([[9, 3], [2, 10]])[1]) < 1e-12
    assert placement_cell_fails(1.1242, 0.90, 1.05, 0.01)
    assert not placement_cell_fails(1.1242, 0.90, 1.20, 0.01)
    assert not placement_cell_fails(1.1242, 0.90, 1.05, 0.05)
    assert not placement_cell_fails(1.1242, 0.90, 1.05, 0.20)
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--rows", type=Path, default=here / "rows.csv")
    parser.add_argument("--sessions", type=Path, default=here / "sessions.jsonl")
    parser.add_argument("--tests", type=Path, default=here / "tests.txt")
    parser.add_argument("--summary", type=Path, default=here / "summary.csv")
    parser.add_argument("--identity", type=Path, default=here / "identity.csv")
    parser.add_argument("--verdict", type=Path, default=here / "verdict.txt")
    parser.add_argument("--report", type=Path, default=here / "README.md")
    args = parser.parse_args(argv[1:])

    rows = read_rows(args.rows)
    kit: dict[str, str] = {}
    if args.sessions.is_file():
        for line in args.sessions.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in (
                "base_ref",
                "base_tree",
                "b_kit_sha256",
                "b_kit_tree",
                "harness_run_gate",
                "harness_summarize",
                "harness_snakefile",
            ):
                if rec.get(key):
                    kit[key] = str(rec[key])
    summary_rows, identity_rows, verdict_text, report_text = summarize(rows, args.tests, kit)
    write_csv(args.summary, SUMMARY_FIELDS, summary_rows)
    write_csv(args.identity, IDENTITY_FIELDS, identity_rows)
    args.verdict.parent.mkdir(parents=True, exist_ok=True)
    args.verdict.write_text(verdict_text, encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    print(verdict_text.splitlines()[0], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
