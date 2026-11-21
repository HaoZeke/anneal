#!/usr/bin/env python3
"""Pre-registered gate analysis for Config::cap_twist default-on."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

from cells import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CANDIDATE_COMMIT,
    CANDIDATE_TREE,
    DEVELOP_COMMIT,
    DEVELOP_TREE,
    HIT_TOL,
    MAIN_ARMS,
    OUTSIDE_RATIO_CAP,
    PANEL_CELLS,
    PLACE_ARMS,
    PLACEMENT_CELLS,
    REPLACEMENT_CAP,
    RSS_ANON_ABS_TOL_KB,
    SEEDS,
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "cell",
    "family",
    "n",
    "rigid_body",
    "budget",
    "reference",
    "seeds_n",
    "replacements_used",
    "hits_a",
    "hits_b",
    "hits_d",
    "delta_hits_ba",
    "delta_hits_bd",
    "fisher_p_ba",
    "mcnemar_p_ba",
    "evals_med_a",
    "evals_med_b",
    "evals_med_d",
    "charged_med_a",
    "charged_med_b",
    "charged_med_d",
    "outside_ns_med_a",
    "outside_ns_med_b",
    "outside_ns_med_d",
    "outside_ns_ratio_ba",
    "outside_ns_ci95_lo",
    "outside_ns_ci95_hi",
    "outside_s_med_a",
    "outside_s_med_b",
    "outside_s_ratio_ba",
    "rss_anon_kb_med_a",
    "rss_anon_kb_med_b",
    "rss_anon_kb_med_d",
    "rss_anon_ratio_ba",
    "rss_anon_ci95_lo",
    "rss_anon_ci95_hi",
    "rss_anon_abs_diff_kb",
    "rss_anon_in_band",
    "vm_hwm_kb_med_a",
    "vm_hwm_kb_med_b",
    "vm_hwm_ratio_ba",
    "ru_maxrss_kb_med_a",
    "ru_maxrss_kb_med_b",
    "ru_maxrss_ratio_ba",
    "cap_flag_sum_a",
    "cap_flag_sum_b",
    "cap_escapes_sum_a",
    "cap_escapes_sum_b",
    "cap_draws_sum_b",
    "cap_accepts_sum_b",
    "water_traj_identical",
    "pass_hits",
    "pass_outside",
    "pass_rss_anon",
    "pass_cap_counter",
    "verdict",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    digest = sha256_file(path)
    if digest != hashlib.sha256(data).hexdigest():
        raise RuntimeError(f"sha256 mismatch writing {path.name}")
    return digest


def atomic_write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return sha256_file(path)


def type7_quantile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    h = 1.0 + (n - 1) * p
    lo = int(math.floor(h))
    frac = h - lo
    idx = lo - 1
    if frac == 0.0:
        return s[idx]
    return s[idx] * (1.0 - frac) + s[idx + 1] * frac


def median(values: list[float]) -> float:
    return type7_quantile(values, 0.5)


def fisher_two_sided_equal_n(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total_hits = hits_a + hits_b
    total_n = n_a + n_b
    denom = math.comb(total_n, total_hits)
    if denom == 0:
        return 1.0

    def hypergeom_prob(k_a: int) -> float:
        k_b = total_hits - k_a
        if k_a < 0 or k_a > n_a or k_b < 0 or k_b > n_b:
            return 0.0
        return (math.comb(n_a, k_a) * math.comb(n_b, k_b)) / denom

    p_obs = hypergeom_prob(hits_a)
    lo = max(0, total_hits - n_b)
    hi = min(n_a, total_hits)
    total_p = 0.0
    for k in range(lo, hi + 1):
        pk = hypergeom_prob(k)
        if pk <= p_obs + 1e-15:
            total_p += pk
    return min(1.0, total_p)


def mcnemar_exact_two_sided(hits_by_seed_a: dict[int, int], hits_by_seed_b: dict[int, int]) -> float:
    b01 = 0
    b10 = 0
    for seed in SEEDS:
        ha = hits_by_seed_a[seed]
        hb = hits_by_seed_b[seed]
        if ha == 0 and hb == 1:
            b01 += 1
        elif ha == 1 and hb == 0:
            b10 += 1
    m = b01 + b10
    if m == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(m, i) for i in range(0, k + 1)) / (2**m)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_median_ratio_ci(
    by_seed_a: dict[int, float],
    by_seed_b: dict[int, float],
    rng: random.Random,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float]:
    seeds = list(SEEDS)
    n = len(seeds)
    ratios: list[float] = []
    for _ in range(draws):
        sampled = [seeds[rng.randrange(n)] for _ in range(n)]
        med_a = median([by_seed_a[s] for s in sampled])
        med_b = median([by_seed_b[s] for s in sampled])
        ratios.append(med_b / med_a if med_a > 0.0 else 1.0)
    return (type7_quantile(ratios, 0.025), type7_quantile(ratios, 0.975))


def compute_pooled_bands(
    active_main: dict[tuple[str, str], dict[int, dict[str, Any]]],
    active_place: dict[tuple[str, str], dict[int, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    metrics = (
        "outside_per_call_ns",
        "outside_s",
        "rss_anon_kb",
        "vm_hwm_kb",
        "ru_maxrss_kb",
    )
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        pooled_ratios: list[float] = []
        aa_ratios: list[float] = []
        pad_ratios: list[float] = []
        for cell_name in PLACEMENT_CELLS:
            if ("place", cell_name) not in active_place and (
                cell_name,
                "PA",
            ) not in active_place:
                pass
            pa_map = active_place.get((cell_name, "PA"), {})
            if len(pa_map) != len(SEEDS):
                continue
            med_pa = median([float(pa_map[s][metric]) for s in SEEDS])
            if med_pa <= 0.0:
                continue

            # Same-binary A/A across main vs place
            a_main = active_main.get((cell_name, "A"), {})
            if len(a_main) == len(SEEDS):
                med_a = median([float(a_main[s][metric]) for s in SEEDS])
                if med_a > 0.0:
                    r_same = med_pa / med_a
                    pooled_ratios.append(r_same)
                    aa_ratios.append(r_same)

            # Independent rebuild A2/PA
            a2_map = active_place.get((cell_name, "A2"), {})
            if len(a2_map) == len(SEEDS):
                med_a2 = median([float(a2_map[s][metric]) for s in SEEDS])
                r_a2 = med_a2 / med_pa
                pooled_ratios.append(r_a2)
                aa_ratios.append(r_a2)

            # 5 padded rebuilds P1..P5 vs PA
            for pad_arm in ("P1", "P2", "P3", "P4", "P5"):
                pmap = active_place.get((cell_name, pad_arm), {})
                if len(pmap) == len(SEEDS):
                    med_p = median([float(pmap[s][metric]) for s in SEEDS])
                    r_p = med_p / med_pa
                    pooled_ratios.append(r_p)
                    pad_ratios.append(r_p)

        if pooled_ratios:
            out[metric] = {
                "count": float(len(pooled_ratios)),
                "lo": min(pooled_ratios),
                "hi": max(pooled_ratios),
                "q025": type7_quantile(pooled_ratios, 0.025),
                "med": median(pooled_ratios),
                "q975": type7_quantile(pooled_ratios, 0.975),
                "aa_lo": min(aa_ratios) if aa_ratios else float("nan"),
                "aa_hi": max(aa_ratios) if aa_ratios else float("nan"),
                "pad_lo": min(pad_ratios) if pad_ratios else float("nan"),
                "pad_hi": max(pad_ratios) if pad_ratios else float("nan"),
            }
        else:
            out[metric] = {
                "count": 0.0,
                "lo": float("nan"),
                "hi": float("nan"),
                "q025": float("nan"),
                "med": float("nan"),
                "q975": float("nan"),
                "aa_lo": float("nan"),
                "aa_hi": float("nan"),
                "pad_lo": float("nan"),
                "pad_hi": float("nan"),
            }
    return out


def verify_abba_and_invariants(all_rows: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    active = [r for r in all_rows if int(r["superseded"]) == 0]
    binary_shas: dict[str, set[str]] = {}
    for r in active:
        binary_shas.setdefault(r["arm"], set()).add(r["binary_sha256"])
        if int(r["threads_max"]) != 1:
            notes.append(f"threads_max != 1 on row {r['row_id']}")
        if r["arm"] == "D":
            if r["commit_sha"] != DEVELOP_COMMIT or r["tree_sha"] != DEVELOP_TREE:
                notes.append(f"develop ref mismatch on row {r['row_id']}")
        else:
            if (
                r["commit_sha"] != CANDIDATE_COMMIT
                or r["tree_sha"] != CANDIDATE_TREE
            ):
                notes.append(f"candidate sha/tree mismatch on row {r['row_id']}")

    if "A" in binary_shas and "B" in binary_shas:
        if binary_shas["A"] != binary_shas["B"] or len(binary_shas["A"]) != 1:
            notes.append("Arm A and Arm B binary_sha256 sets do not match 1-to-1")

    # Verify exact ABBA pair order on main cells with complete active rows
    by_cell_seed: dict[tuple[str, int], dict[str, int]] = {}
    for r in active:
        if r["band"] == "main":
            by_cell_seed.setdefault((r["cell"], int(r["seed"])), {})[
                r["arm"]
            ] = int(r["slot"])
    for spec in PANEL_CELLS:
        orders: list[str] = []
        complete = True
        for s in SEEDS:
            slots = by_cell_seed.get((spec.name, s), {})
            if "A" not in slots or "B" not in slots:
                complete = False
                break
            orders.append("A" if slots["A"] < slots["B"] else "B")
        if complete:
            for blk in range(0, len(orders), 4):
                chunk = "".join(orders[blk : blk + 4])
                if chunk not in ("ABBA", "BAAB"):
                    notes.append(
                        f"ABBA violation in {spec.name} block {blk}: {chunk}"
                    )
    return notes


def analyze_campaign(rows_path: Path, summary_path: Path, report_path: Path) -> int:
    with rows_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    invariant_errors = verify_abba_and_invariants(all_rows)
    active_main: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    active_place: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    replacements_main: dict[str, int] = {}
    replacements_place: dict[str, int] = {}

    for r in all_rows:
        sup = int(r["superseded"])
        cell = r["cell"]
        arm = r["arm"]
        seed = int(r["seed"])
        if r["band"] == "main":
            if sup == 1:
                replacements_main[cell] = replacements_main.get(cell, 0) + 1
            else:
                active_main.setdefault((cell, arm), {})[seed] = r
        elif r["band"] == "place":
            if sup == 1:
                replacements_place[cell] = replacements_place.get(cell, 0) + 1
            else:
                active_place.setdefault((cell, arm), {})[seed] = r

    bands = compute_pooled_bands(active_main, active_place)
    rss_band = bands["rss_anon_kb"]
    outside_band = bands["outside_per_call_ns"]
    rng = random.Random(BOOTSTRAP_SEED)

    summary_rows: list[dict[str, Any]] = []
    verdict_counts = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}

    for spec in PANEL_CELLS:
        c = spec.name
        rep_used = replacements_main.get(c, 0)
        map_a = active_main.get((c, "A"), {})
        map_b = active_main.get((c, "B"), {})
        map_d = active_main.get((c, "D"), {})

        complete = (
            len(map_a) == len(SEEDS)
            and len(map_b) == len(SEEDS)
            and len(map_d) == len(SEEDS)
            and math.isfinite(rss_band["lo"])
            and math.isfinite(rss_band["hi"])
        )

        if not complete or rep_used > REPLACEMENT_CAP or invariant_errors:
            verdict_counts["INCONCLUSIVE"] += 1
            summary_rows.append(
                {
                    "cell": c,
                    "family": spec.family,
                    "n": spec.n,
                    "rigid_body": int(spec.rigid_body),
                    "budget": spec.budget,
                    "reference": f"{spec.reference:.10f}",
                    "seeds_n": min(len(map_a), len(map_b), len(map_d)),
                    "replacements_used": rep_used,
                    "hits_a": 0,
                    "hits_b": 0,
                    "hits_d": 0,
                    "delta_hits_ba": 0,
                    "delta_hits_bd": 0,
                    "fisher_p_ba": "nan",
                    "mcnemar_p_ba": "nan",
                    "evals_med_a": "nan",
                    "evals_med_b": "nan",
                    "evals_med_d": "nan",
                    "charged_med_a": "nan",
                    "charged_med_b": "nan",
                    "charged_med_d": "nan",
                    "outside_ns_med_a": "nan",
                    "outside_ns_med_b": "nan",
                    "outside_ns_med_d": "nan",
                    "outside_ns_ratio_ba": "nan",
                    "outside_ns_ci95_lo": "nan",
                    "outside_ns_ci95_hi": "nan",
                    "outside_s_med_a": "nan",
                    "outside_s_med_b": "nan",
                    "outside_s_ratio_ba": "nan",
                    "rss_anon_kb_med_a": "nan",
                    "rss_anon_kb_med_b": "nan",
                    "rss_anon_kb_med_d": "nan",
                    "rss_anon_ratio_ba": "nan",
                    "rss_anon_ci95_lo": "nan",
                    "rss_anon_ci95_hi": "nan",
                    "rss_anon_abs_diff_kb": "nan",
                    "rss_anon_in_band": 0,
                    "vm_hwm_kb_med_a": "nan",
                    "vm_hwm_kb_med_b": "nan",
                    "vm_hwm_ratio_ba": "nan",
                    "ru_maxrss_kb_med_a": "nan",
                    "ru_maxrss_kb_med_b": "nan",
                    "ru_maxrss_ratio_ba": "nan",
                    "cap_flag_sum_a": 0,
                    "cap_flag_sum_b": 0,
                    "cap_escapes_sum_a": 0,
                    "cap_escapes_sum_b": 0,
                    "cap_draws_sum_b": 0,
                    "cap_accepts_sum_b": 0,
                    "water_traj_identical": 0,
                    "pass_hits": 0,
                    "pass_outside": 0,
                    "pass_rss_anon": 0,
                    "pass_cap_counter": 0,
                    "verdict": "INCONCLUSIVE",
                }
            )
            continue

        hits_seed_a = {
            s: int(float(map_a[s]["best"]) <= spec.reference + HIT_TOL)
            for s in SEEDS
        }
        hits_seed_b = {
            s: int(float(map_b[s]["best"]) <= spec.reference + HIT_TOL)
            for s in SEEDS
        }
        hits_seed_d = {
            s: int(float(map_d[s]["best"]) <= spec.reference + HIT_TOL)
            for s in SEEDS
        }
        hits_a = sum(hits_seed_a.values())
        hits_b = sum(hits_seed_b.values())
        hits_d = sum(hits_seed_d.values())
        fisher_p = fisher_two_sided_equal_n(hits_a, len(SEEDS), hits_b, len(SEEDS))
        mcnemar_p = mcnemar_exact_two_sided(hits_seed_a, hits_seed_b)

        evals_med_a = median([float(map_a[s]["evals_to_hit"]) for s in SEEDS])
        evals_med_b = median([float(map_b[s]["evals_to_hit"]) for s in SEEDS])
        evals_med_d = median([float(map_d[s]["evals_to_hit"]) for s in SEEDS])

        charged_med_a = median([float(map_a[s]["charged"]) for s in SEEDS])
        charged_med_b = median([float(map_b[s]["charged"]) for s in SEEDS])
        charged_med_d = median([float(map_d[s]["charged"]) for s in SEEDS])

        out_ns_a = {s: float(map_a[s]["outside_per_call_ns"]) for s in SEEDS}
        out_ns_b = {s: float(map_b[s]["outside_per_call_ns"]) for s in SEEDS}
        out_ns_d = {s: float(map_d[s]["outside_per_call_ns"]) for s in SEEDS}
        out_ns_med_a = median(list(out_ns_a.values()))
        out_ns_med_b = median(list(out_ns_b.values()))
        out_ns_med_d = median(list(out_ns_d.values()))
        out_ns_ratio = out_ns_med_b / out_ns_med_a
        out_ci_lo, out_ci_hi = paired_bootstrap_median_ratio_ci(
            out_ns_a, out_ns_b, rng
        )

        out_s_med_a = median([float(map_a[s]["outside_s"]) for s in SEEDS])
        out_s_med_b = median([float(map_b[s]["outside_s"]) for s in SEEDS])
        out_s_ratio = out_s_med_b / out_s_med_a if out_s_med_a > 0.0 else 1.0

        rss_a = {s: float(map_a[s]["rss_anon_kb"]) for s in SEEDS}
        rss_b = {s: float(map_b[s]["rss_anon_kb"]) for s in SEEDS}
        rss_d = {s: float(map_d[s]["rss_anon_kb"]) for s in SEEDS}
        rss_med_a = median(list(rss_a.values()))
        rss_med_b = median(list(rss_b.values()))
        rss_med_d = median(list(rss_d.values()))
        rss_ratio = rss_med_b / rss_med_a
        rss_ci_lo, rss_ci_hi = paired_bootstrap_median_ratio_ci(rss_a, rss_b, rng)
        rss_abs_diff = abs(rss_med_b - rss_med_a)
        rss_in_band = int(rss_band["lo"] <= rss_ratio <= rss_band["hi"])

        hwm_med_a = median([float(map_a[s]["vm_hwm_kb"]) for s in SEEDS])
        hwm_med_b = median([float(map_b[s]["vm_hwm_kb"]) for s in SEEDS])
        hwm_ratio = hwm_med_b / hwm_med_a if hwm_med_a > 0.0 else 1.0

        ru_med_a = median([float(map_a[s]["ru_maxrss_kb"]) for s in SEEDS])
        ru_med_b = median([float(map_b[s]["ru_maxrss_kb"]) for s in SEEDS])
        ru_ratio = ru_med_b / ru_med_a if ru_med_a > 0.0 else 1.0

        cap_flag_a = sum(int(map_a[s]["cap_twist_flag"]) for s in SEEDS)
        cap_flag_b = sum(int(map_b[s]["cap_twist_flag"]) for s in SEEDS)
        cap_flag_d = sum(int(map_d[s]["cap_twist_flag"]) for s in SEEDS)
        cap_esc_a = sum(int(map_a[s]["cap_escapes"]) for s in SEEDS)
        cap_esc_b = sum(int(map_b[s]["cap_escapes"]) for s in SEEDS)
        cap_esc_d = sum(int(map_d[s]["cap_escapes"]) for s in SEEDS)
        cap_draws_b = sum(int(map_b[s]["cap_arm_draws"]) for s in SEEDS)
        cap_acc_b = sum(int(map_b[s]["cap_arm_accepts"]) for s in SEEDS)

        water_identical = int(
            all(
                int(map_a[s]["charged"]) == int(map_b[s]["charged"])
                and map_a[s]["best_bits"] == map_b[s]["best_bits"]
                for s in SEEDS
            )
        )

        # Pre-registered gate criteria
        pass_hits = int(hits_b >= hits_a)
        pass_outside = int(out_ns_ratio <= OUTSIDE_RATIO_CAP)
        early_stop_saving = (rss_med_b < rss_med_a) and (
            charged_med_b < charged_med_a
        )
        pass_rss_anon = int(
            (rss_abs_diff <= RSS_ANON_ABS_TOL_KB)
            or bool(rss_in_band)
            or early_stop_saving
        )
        if spec.rigid_body:
            pass_cap_counter = int(
                cap_flag_a == 0
                and cap_flag_d == 0
                and cap_esc_a == 0
                and cap_esc_d == 0
                and cap_flag_b == len(SEEDS)
                and cap_esc_b == 0
                and water_identical == 1
            )
        else:
            pass_cap_counter = int(
                cap_flag_a == 0
                and cap_flag_d == 0
                and cap_esc_a == 0
                and cap_esc_d == 0
                and cap_flag_b == len(SEEDS)
                and cap_esc_b > 0
                and cap_draws_b > 0
            )

        verdict = (
            "PASS"
            if (
                pass_hits
                and pass_outside
                and pass_rss_anon
                and pass_cap_counter
            )
            else "FAIL"
        )
        verdict_counts[verdict] += 1

        summary_rows.append(
            {
                "cell": c,
                "family": spec.family,
                "n": spec.n,
                "rigid_body": int(spec.rigid_body),
                "budget": spec.budget,
                "reference": f"{spec.reference:.10f}",
                "seeds_n": len(SEEDS),
                "replacements_used": rep_used,
                "hits_a": hits_a,
                "hits_b": hits_b,
                "hits_d": hits_d,
                "delta_hits_ba": hits_b - hits_a,
                "delta_hits_bd": hits_b - hits_d,
                "fisher_p_ba": f"{fisher_p:.6f}",
                "mcnemar_p_ba": f"{mcnemar_p:.6f}",
                "evals_med_a": f"{evals_med_a:.1f}",
                "evals_med_b": f"{evals_med_b:.1f}",
                "evals_med_d": f"{evals_med_d:.1f}",
                "charged_med_a": f"{charged_med_a:.1f}",
                "charged_med_b": f"{charged_med_b:.1f}",
                "charged_med_d": f"{charged_med_d:.1f}",
                "outside_ns_med_a": f"{out_ns_med_a:.2f}",
                "outside_ns_med_b": f"{out_ns_med_b:.2f}",
                "outside_ns_med_d": f"{out_ns_med_d:.2f}",
                "outside_ns_ratio_ba": f"{out_ns_ratio:.4f}",
                "outside_ns_ci95_lo": f"{out_ci_lo:.4f}",
                "outside_ns_ci95_hi": f"{out_ci_hi:.4f}",
                "outside_s_med_a": f"{out_s_med_a:.6f}",
                "outside_s_med_b": f"{out_s_med_b:.6f}",
                "outside_s_ratio_ba": f"{out_s_ratio:.4f}",
                "rss_anon_kb_med_a": f"{rss_med_a:.1f}",
                "rss_anon_kb_med_b": f"{rss_med_b:.1f}",
                "rss_anon_kb_med_d": f"{rss_med_d:.1f}",
                "rss_anon_ratio_ba": f"{rss_ratio:.4f}",
                "rss_anon_ci95_lo": f"{rss_ci_lo:.4f}",
                "rss_anon_ci95_hi": f"{rss_ci_hi:.4f}",
                "rss_anon_abs_diff_kb": f"{rss_abs_diff:.1f}",
                "rss_anon_in_band": rss_in_band,
                "vm_hwm_kb_med_a": f"{hwm_med_a:.1f}",
                "vm_hwm_kb_med_b": f"{hwm_med_b:.1f}",
                "vm_hwm_ratio_ba": f"{hwm_ratio:.4f}",
                "ru_maxrss_kb_med_a": f"{ru_med_a:.1f}",
                "ru_maxrss_kb_med_b": f"{ru_med_b:.1f}",
                "ru_maxrss_ratio_ba": f"{ru_ratio:.4f}",
                "cap_flag_sum_a": cap_flag_a,
                "cap_flag_sum_b": cap_flag_b,
                "cap_escapes_sum_a": cap_esc_a,
                "cap_escapes_sum_b": cap_esc_b,
                "cap_draws_sum_b": cap_draws_b,
                "cap_accepts_sum_b": cap_acc_b,
                "water_traj_identical": water_identical,
                "pass_hits": pass_hits,
                "pass_outside": pass_outside,
                "pass_rss_anon": pass_rss_anon,
                "pass_cap_counter": pass_cap_counter,
                "verdict": verdict,
            }
        )

    overall = (
        "PASS"
        if verdict_counts["FAIL"] == 0 and verdict_counts["INCONCLUSIVE"] == 0
        else (
            "FAIL" if verdict_counts["FAIL"] > 0 else "INCONCLUSIVE"
        )
    )

    summary_sha = atomic_write_csv(summary_path, SUMMARY_FIELDS, summary_rows)

    superseded_rows = [r for r in all_rows if int(r["superseded"]) == 1]
    total_hits_a = sum(int(r["hits_a"]) for r in summary_rows if r["verdict"] != "INCONCLUSIVE")
    total_hits_b = sum(int(r["hits_b"]) for r in summary_rows if r["verdict"] != "INCONCLUSIVE")
    total_hits_d = sum(int(r["hits_d"]) for r in summary_rows if r["verdict"] != "INCONCLUSIVE")

    lines: list[str] = [
        "# Cap-Turn Default-On Gate Report",
        "",
        f"- Overall verdict: **{overall}** (`PASS={verdict_counts['PASS']}`, `FAIL={verdict_counts['FAIL']}`, `INCONCLUSIVE={verdict_counts['INCONCLUSIVE']}`)",
        f"- Candidate commit (`A`/`B`): `{CANDIDATE_COMMIT}` (tree `{CANDIDATE_TREE}`)",
        f"- Reference commit (`D`, `develop`): `{DEVELOP_COMMIT}` (tree `{DEVELOP_TREE}`)",
        f"- Active + superseded rows in `rows.csv`: `{len(all_rows)}` (`{len(superseded_rows)}` superseded)",
        f"- Total panel hits (`34` cells x `12` seeds = `408` runs/arm): `A={total_hits_a}/408`, `B={total_hits_b}/408`, `D={total_hits_d}/408`",
        "",
        "## Pooled `A/A + Placement` Calibration Bands (`12` cells, `5` padded rebuilds + `A2` + `PA/A`)",
        "",
        "| Metric | N ratios | Empirical `[lo, hi]` | `2.5%..97.5%` | `A/A` `[lo, hi]` | Padding `[lo, hi]` |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for metric_key, label in (
        ("rss_anon_kb", "In-process peak `RssAnon`"),
        ("outside_per_call_ns", "Outside time / call (`ns`)"),
        ("outside_s", "Outside time / run (`s`)"),
        ("vm_hwm_kb", "In-process `VmHWM` (diag)"),
        ("ru_maxrss_kb", "Child `ru_maxrss` (diag)"),
    ):
        b = bands[metric_key]
        lines.append(
            f"| {label} | {int(b['count'])} | `[{b['lo']:.4f}, {b['hi']:.4f}]` | `[{b['q025']:.4f}, {b['q975']:.4f}]` | `[{b['aa_lo']:.4f}, {b['aa_hi']:.4f}]` | `[{b['pad_lo']:.4f}, {b['pad_hi']:.4f}]` |"
        )

    if invariant_errors:
        lines.extend(["", "## Invariant Warnings", ""])
        for err in invariant_errors:
            lines.append(f"- {err}")

    lines.extend(
        [
            "",
            "## Per-Cell Gate Summary",
            "",
            "| Cell | Hits A / B / D | Fisher p | McNemar p | Outside/call B/A `[95% CI]` | `RssAnon` B/A (`|B-A|` kB) | `VmHWM` B/A | `ru_maxrss` B/A | `cap_flag` B | `cap_escapes` B | Verdict |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in summary_rows:
        if r["verdict"] == "INCONCLUSIVE":
            lines.append(
                f"| `{r['cell']}` | inc | nan | nan | nan | nan | nan | nan | 0 | 0 | **INCONCLUSIVE** |"
            )
            continue
        lines.append(
            f"| `{r['cell']}` | `{r['hits_a']}/{r['hits_b']}/{r['hits_d']}` | `{r['fisher_p_ba']}` | `{r['mcnemar_p_ba']}` | `{r['outside_ns_ratio_ba']}` `[{r['outside_ns_ci95_lo']}, {r['outside_ns_ci95_hi']}]` | `{r['rss_anon_ratio_ba']}` (`{r['rss_anon_abs_diff_kb']}`) | `{r['vm_hwm_ratio_ba']}` | `{r['ru_maxrss_ratio_ba']}` | `{r['cap_flag_sum_b']}` | `{r['cap_escapes_sum_b']}` | **{r['verdict']}** |"
        )

    lines.append("")
    report_sha = atomic_write_text(report_path, "\n".join(lines))
    print(
        f"ANALYZED overall={overall} pass={verdict_counts['PASS']} fail={verdict_counts['FAIL']} inc={verdict_counts['INCONCLUSIVE']} summary_sha256={summary_sha} report_sha256={report_sha}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze cap-turn gate rows.")
    parser.add_argument("--rows", required=True, help="Input rows.csv path.")
    parser.add_argument("--summary", required=True, help="Output summary.csv path.")
    parser.add_argument("--report", required=True, help="Output REPORT.md path.")
    args = parser.parse_args()
    return analyze_campaign(
        Path(args.rows), Path(args.summary), Path(args.report)
    )


if __name__ == "__main__":
    sys.exit(main())
