#!/usr/bin/env python3
"""Statistical analysis (Fisher, MWU, 10k bootstrap CIs) for LJ75 benches."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from pathlib import Path
from statistics import NormalDist

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import paired_stats  # noqa: E402

BOOT_RESAMPLES = 10_000
BOOT_SEED = 20261123
MISS_CODE = 4_000_001


def mann_whitney_u(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided tie-corrected Mann-Whitney U test (normal approximation)."""
    n1, n2 = len(left), len(right)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    combined = left + right
    if len(set(combined)) == 1:
        return n1 * n2 / 2.0, 1.0
    ranks = paired_stats.midranks(combined)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)

    counts: dict[float, int] = {}
    for v in combined:
        counts[v] = counts.get(v, 0) + 1
    n = n1 + n2
    tie_sum = sum(t * t * t - t for t in counts.values() if t > 1)
    var_u = (n1 * n2 / 12.0) * ((n + 1) - tie_sum / (n * (n - 1)))
    if var_u <= 0.0:
        return u1, 1.0
    mean_u = n1 * n2 / 2.0
    z = (abs(u1 - mean_u) - 0.5) / math.sqrt(var_u)
    z = max(0.0, z)
    p = 2.0 * (1.0 - NormalDist().cdf(z))
    return u1, min(1.0, max(0.0, p))


def bootstrap_ci(
    values: list[float],
    stat_fn,
    *,
    resamples: int = BOOT_RESAMPLES,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    est = float(stat_fn(values))
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return est, est, est
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(float(stat_fn(sample)))
    boots.sort()
    lo_idx = int(0.025 * (resamples - 1))
    hi_idx = int(0.975 * (resamples - 1))
    return est, boots[lo_idx], boots[hi_idx]


def bootstrap_ratio_ci(
    t1_vals: list[float],
    tp_vals: list[float],
    workers: int,
    *,
    resamples: int = BOOT_RESAMPLES,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float] | None:
    if not t1_vals or not tp_vals:
        return None
    est = statistics.median(t1_vals) / (workers * statistics.median(tp_vals))
    rng = random.Random(seed)
    n1, np_ = len(t1_vals), len(tp_vals)
    boots = []
    for _ in range(resamples):
        s1 = [t1_vals[rng.randrange(n1)] for _ in range(n1)]
        sp = [tp_vals[rng.randrange(np_)] for _ in range(np_)]
        boots.append(statistics.median(s1) / (workers * statistics.median(sp)))
    boots.sort()
    lo_idx = int(0.025 * (resamples - 1))
    hi_idx = int(0.975 * (resamples - 1))
    return est, boots[lo_idx], boots[hi_idx]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_scaling(path: Path) -> None:
    # Blank session is the original ladder. Later sessions are separate.
    rows = [r for r in load_rows(path) if not (r.get("session") or "")]
    by_workers: dict[int, list[dict[str, str]]] = {}
    for r in rows:
        by_workers.setdefault(int(r["workers"]), []).append(r)

    t1_hits_wall = [
        float(r["wall_to_hit_s"])
        for r in by_workers.get(1, [])
        if r["hit"] == "1" and r["wall_to_hit_s"]
    ]
    base_rung = min(by_workers) if 1 not in by_workers else 1
    base_rows = by_workers[base_rung]
    base_hits = sum(r["hit"] == "1" for r in base_rows)
    base_calls = [float(r["calls_to_hit"]) for r in base_rows]
    base_wall = [float(r["wall_s"]) for r in base_rows]

    print(
        f"workers,n,hits,hit_rate,hit_ci95,wall_to_hit_med_s,wall_to_hit_ci95,"
        f"calls_to_hit_med,calls_to_hit_ci95,wall_med_s,cpu_med_s,peak_rss_med_kib,"
        f"efficiency,efficiency_ci95,rel_eff_vs_p{base_rung},fisher_p_vs_p{base_rung},mwu_p_vs_p{base_rung},oversubscribed"
    )
    for w in sorted(by_workers):
        grp = by_workers[w]
        n = len(grp)
        hits_vec = [1.0 if r["hit"] == "1" else 0.0 for r in grp]
        hits = int(sum(hits_vec))
        hr_est, hr_lo, hr_hi = bootstrap_ci(hits_vec, statistics.mean)
        calls_vec = [float(r["calls_to_hit"]) for r in grp]
        c_est, c_lo, c_hi = bootstrap_ci(calls_vec, statistics.median)
        wall_hit_vec = [
            float(r["wall_to_hit_s"])
            for r in grp
            if r["hit"] == "1" and r["wall_to_hit_s"]
        ]
        if wall_hit_vec:
            wh_est, wh_lo, wh_hi = bootstrap_ci(wall_hit_vec, statistics.median)
            wh_str = f"{wh_est:.3f}"
            wh_ci = f"[{wh_lo:.3f}; {wh_hi:.3f}]"
        else:
            wh_str, wh_ci = "NA", "NA"

        over = grp[0]["oversubscribed"]
        if over == "true" or not t1_hits_wall or not wall_hit_vec:
            eff_str, eff_ci = "NA", "NA"
        else:
            eff = bootstrap_ratio_ci(t1_hits_wall, wall_hit_vec, w)
            assert eff is not None
            eff_str = f"{eff[0]:.3f}"
            eff_ci = f"[{eff[1]:.3f}; {eff[2]:.3f}]"

        wall_vec = [float(r["wall_s"]) for r in grp]
        rel_eff = bootstrap_ratio_ci(base_wall, wall_vec, w)
        rel_eff_str = (
            "NA"
            if over == "true" or rel_eff is None
            else f"{base_rung * rel_eff[0]:.3f}"
        )

        fp = paired_stats.fisher_exact(hits, n, base_hits, len(base_rows))
        _, mp = mann_whitney_u(calls_vec, base_calls)
        wall_med = statistics.median(wall_vec)
        cpu_med = statistics.median(float(r["cpu_s"]) for r in grp)
        rss_med = statistics.median(float(r["peak_rss_anon_kib"]) for r in grp)
        print(
            f"{w},{n},{hits}/{n},{hr_est:.3f},[{hr_lo:.3f}; {hr_hi:.3f}],"
            f"{wh_str},{wh_ci},{c_est:.0f},[{c_lo:.0f}; {c_hi:.0f}],"
            f"{wall_med:.3f},{cpu_med:.2f},{rss_med:.0f},{eff_str},{eff_ci},"
            f"{rel_eff_str},{fp:.6g},{mp:.6g},{over}"
        )


def summarize_warmstart(path: Path) -> None:
    rows = load_rows(path)
    arms: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)

    cold = sorted(arms.get("cold", []), key=lambda r: int(r["seed"]))
    warm = sorted(arms.get("warm", []), key=lambda r: int(r["seed"]))
    if not cold or not warm:
        raise SystemExit("missing cold or warm rows")

    cold_hits = [r["hit"] == "1" for r in cold]
    warm_hits = [r["hit"] == "1" for r in warm]
    both, only_warm, only_cold, neither = paired_stats.outcome_table(warm_hits, cold_hits)
    fp = paired_stats.fisher_exact(
        sum(warm_hits), len(warm_hits), sum(cold_hits), len(cold_hits)
    )
    mp_paired = paired_stats.mcnemar_exact(only_warm, only_cold)
    warm_calls = [float(r["calls_to_hit"]) for r in warm]
    cold_calls = [float(r["calls_to_hit"]) for r in cold]
    u_stat, mwu_p = mann_whitney_u(warm_calls, cold_calls)

    for name, grp in [("cold", cold), ("warm", warm)]:
        hits_vec = [1.0 if r["hit"] == "1" else 0.0 for r in grp]
        hr_est, hr_lo, hr_hi = bootstrap_ci(hits_vec, statistics.mean)
        calls_vec = [float(r["calls_to_hit"]) for r in grp]
        c_est, c_lo, c_hi = bootstrap_ci(calls_vec, statistics.median)
        wall_hit_vec = [
            float(r["wall_to_hit_s"])
            for r in grp
            if r["hit"] == "1" and r["wall_to_hit_s"]
        ]
        if wall_hit_vec:
            wh_est, wh_lo, wh_hi = bootstrap_ci(wall_hit_vec, statistics.median)
            wh_s = f"{wh_est:.3f} [{wh_lo:.3f}, {wh_hi:.3f}]"
        else:
            wh_s = "NA"
        wall_med = statistics.median(float(r["wall_s"]) for r in grp)
        cpu_med = statistics.median(float(r["cpu_s"]) for r in grp)
        rss_med = statistics.median(float(r["peak_rss_anon_kib"]) for r in grp)
        print(
            f"{name}: n={len(grp)} hits={sum(r['hit']=='1' for r in grp)}/{len(grp)} "
            f"rate={hr_est:.3f} [{hr_lo:.3f}, {hr_hi:.3f}] "
            f"calls_med={c_est:.0f} [{c_lo:.0f}, {c_hi:.0f}] "
            f"wall_to_hit={wh_s} wall_med={wall_med:.3f}s cpu_med={cpu_med:.2f}s rss_med={rss_med:.0f}KiB"
        )
    print(
        f"paired: both={both} warm_only={only_warm} cold_only={only_cold} neither={neither} "
        f"fisher_p={fp:.6g} mcnemar_p={mp_paired:.6g} mwu_U={u_stat:.1f} mwu_p={mwu_p:.6g}"
    )


def session_of(row: dict[str, str]) -> str:
    return row.get("session") or ""


def hit_walls(rows: list[dict[str, str]]) -> list[float]:
    return [
        float(r["wall_to_hit_s"])
        for r in rows
        if r["hit"] == "1" and r["wall_to_hit_s"]
    ]


def calls_of(rows: list[dict[str, str]]) -> list[float]:
    return [float(r["calls_to_hit"]) for r in rows]


def wall_coded(rows: list[dict[str, str]]) -> list[float]:
    """Wall-to-hit with a miss coded as budget + 1."""
    coded: list[float] = []
    for row in rows:
        if row.get("hit") == "1" and row.get("wall_to_hit_s"):
            coded.append(float(row["wall_to_hit_s"]))
        else:
            coded.append(float(MISS_CODE))
    return coded


def hit_median_ci(rows: list[dict[str, str]]) -> tuple[str, str]:
    walls = hit_walls(rows)
    if not walls:
        return "NA", "NA"
    est, lo, hi = bootstrap_ci(walls, statistics.median)
    return f"{est:.3f}", f"[{lo:.3f}; {hi:.3f}]"


def arm_provenance(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "nproc= affinity= machine_id="

    def uniq(key: str) -> str:
        return "|".join(sorted({r.get(key) or "" for r in rows}))

    return (
        f"nproc={uniq('nproc')} affinity={uniq('pinned_cores')} "
        f"machine_id={uniq('machine_id')}"
    )


def print_arm(label: str, rows: list[dict[str, str]]) -> None:
    n = len(rows)
    hits = sum(r.get("hit") == "1" for r in rows)
    med, ci = hit_median_ci(rows)
    print(
        f"ARM {label} n={n} hits={hits}/{n} "
        f"wall_to_hit_med_s={med} wall_ci95={ci} {arm_provenance(rows)}"
    )


def print_agreement(
    tag: str,
    left_name: str,
    right_name: str,
    left: list[dict[str, str]],
    right: list[dict[str, str]],
) -> None:
    if not left or not right:
        print(
            f"INCONCLUSIVE: {tag} missing rows "
            f"n_left={len(left)} n_right={len(right)}"
        )
        return
    left_hits = sum(r.get("hit") == "1" for r in left)
    right_hits = sum(r.get("hit") == "1" for r in right)
    fisher_p = paired_stats.fisher_exact(
        left_hits, len(left), right_hits, len(right)
    )
    u_stat, mwu_p = mann_whitney_u(wall_coded(left), wall_coded(right))
    left_med, left_ci = hit_median_ci(left)
    right_med, right_ci = hit_median_ci(right)
    print(
        f"COMPARE {tag} hits={left_hits}/{len(left)} vs "
        f"{right_hits}/{len(right)} fisher_p={fisher_p:.6g} "
        f"mwu_U={u_stat:.1f} mwu_p={mwu_p:.6g} "
        f"wall_med={left_med} {left_ci} vs {right_med} {right_ci}"
    )
    if len(left) != 12 or len(right) != 12:
        print(f"INCONCLUSIVE: {tag} n={len(left)} vs {len(right)}")
        return
    hits_state = "same" if fisher_p > 0.05 else "different"
    wall_state = "same" if mwu_p > 0.05 else "different"
    if hits_state == "same" and wall_state == "same":
        print(
            f"AGREE: {left_name} and {right_name} agree on hits "
            "and wall-to-hit"
        )
        return
    parts = []
    if hits_state != "same":
        parts.append("hits")
    if wall_state != "same":
        parts.append("wall-to-hit")
    print(
        f"DISAGREE: {left_name} and {right_name} differ on "
        + ", ".join(parts)
    )


def provenance(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "empty"
    def uniq(key: str) -> str:
        return "|".join(sorted({r.get(key) or "" for r in rows}))

    return (
        f"rev={uniq('revision')} nproc={uniq('nproc')} cpu={uniq('cpu_model')} "
        f"pin={uniq('pinned_cores')} over={uniq('oversubscribed')}"
    )


def summarize_followup(path: Path) -> None:
    """p=1 ladder plus the same-VM p=2 control.

    Efficiency is T1/(p*Tp) on hit-conditioned wall-to-hit medians.
    Oversubscribed rungs (p > nproc) are excluded. Misses stay at
    budget+1 for the calls-to-hit median and the Mann-Whitney test.

    Sessions p1-control and p2-control are a later same-VM rerun.
    Those comparisons use Fisher's exact test on hits and a
    Mann-Whitney U test on wall-to-hit, with a miss coded as
    budget+1. Medians stay hit-conditioned, with a 10,000-resample
    percentile bootstrap 95% CI at seed 20261123.
    """
    rows = load_rows(path)
    known = {"", "primary", "control", "p1-control", "p2-control"}
    bad = sorted({session_of(r) for r in rows} - known)
    if bad:
        raise SystemExit(f"unexpected session values: {bad}")

    original: dict[int, list[dict[str, str]]] = {}
    for r in rows:
        if session_of(r) == "":
            original.setdefault(int(r["workers"]), []).append(r)
    primary = [r for r in rows if session_of(r) == "primary"]
    control = [r for r in rows if session_of(r) == "control"]
    if any(int(r["workers"]) != 1 for r in primary):
        raise SystemExit("primary session is only the p=1 rung")
    if any(int(r["workers"]) != 2 for r in control):
        raise SystemExit("control session is only the p=2 rung")

    t1 = hit_walls(primary)
    base_hits = sum(r["hit"] == "1" for r in primary)
    base_n = len(primary)
    base_calls = calls_of(primary)

    groups: list[tuple[str, int, list[dict[str, str]], str]] = [
        ("primary", 1, primary, "definition")
    ]
    for workers in sorted(original):
        groups.append(("original", workers, original[workers], "original"))
    if control:
        groups.append(("control", 2, control, "control"))

    print(
        "label,workers,n,hits,hit_rate,hit_ci95,wall_to_hit_med_s,wall_to_hit_ci95,"
        "calls_to_hit_med,calls_to_hit_ci95,fisher_p_vs_p1,mwu_U_vs_p1,mwu_p_vs_p1,"
        "eff,eff_ci95,eff_basis,oversubscribed,provenance"
    )
    for label, workers, grp, basis in groups:
        if not grp:
            continue
        n = len(grp)
        hits_vec = [1.0 if r["hit"] == "1" else 0.0 for r in grp]
        hits = int(sum(hits_vec))
        hr_est, hr_lo, hr_hi = bootstrap_ci(hits_vec, statistics.mean)
        calls_vec = calls_of(grp)
        c_est, c_lo, c_hi = bootstrap_ci(calls_vec, statistics.median)
        walls = hit_walls(grp)
        if walls:
            wh_est, wh_lo, wh_hi = bootstrap_ci(walls, statistics.median)
            wh_s = f"{wh_est:.3f}"
            wh_ci = f"[{wh_lo:.3f}; {wh_hi:.3f}]"
        else:
            wh_s, wh_ci = "NA", "NA"
        over = grp[0]["oversubscribed"]
        if not primary:
            fisher_s, u_s, mwu_s = "NA", "NA", "NA"
        elif label == "primary":
            fisher_s, u_s, mwu_s = "1", "NA", "1"
        else:
            fisher_s = f"{paired_stats.fisher_exact(hits, n, base_hits, base_n):.6g}"
            u_stat, mwu_p = mann_whitney_u(calls_vec, base_calls)
            u_s = f"{u_stat:.1f}"
            mwu_s = f"{mwu_p:.6g}"

        excluded = over == "true"
        if label == "primary" and walls:
            eff_s, eff_ci, eff_basis = "1.000", "[1.000; 1.000]", "definition"
        elif excluded or not t1 or not walls:
            eff_s, eff_ci, eff_basis = "NA", "NA", "excluded" if excluded else basis
        else:
            eff = bootstrap_ratio_ci(t1, walls, workers)
            assert eff is not None
            eff_s = f"{eff[0]:.3f}"
            eff_ci = f"[{eff[1]:.3f}; {eff[2]:.3f}]"
            eff_basis = basis
        print(
            f"{label},{workers},{n},{hits}/{n},{hr_est:.3f},"
            f"[{hr_lo:.3f}; {hr_hi:.3f}],{wh_s},{wh_ci},"
            f"{c_est:.0f},[{c_lo:.0f}; {c_hi:.0f}],{fisher_s},{u_s},{mwu_s},"
            f"{eff_s},{eff_ci},{eff_basis},{over},{provenance(grp)}"
        )

    print("EXCLUDED p=16 from efficiency because p > nproc (oversubscribed)")
    if not control or 2 not in original or not primary:
        print("AGREEMENT NA: need primary p=1, original p=2, and control p=2")
        return

    orig = original[2]
    c_hits = sum(r["hit"] == "1" for r in control)
    o_hits = sum(r["hit"] == "1" for r in orig)
    fisher_p = paired_stats.fisher_exact(c_hits, len(control), o_hits, len(orig))
    u_stat, mwu_p = mann_whitney_u(calls_of(control), calls_of(orig))
    c_walls = hit_walls(control)
    o_walls = hit_walls(orig)
    ratio = bootstrap_ratio_ci(c_walls, o_walls, 1) if c_walls and o_walls else None
    if ratio is None:
        ratio_s = "NA"
        wall_state = "missing"
    else:
        ratio_s = f"{ratio[0]:.3f} [{ratio[1]:.3f}; {ratio[2]:.3f}]"
        wall_state = "covers-1" if ratio[1] <= 1.0 <= ratio[2] else "excludes-1"
    hits_state = "same" if fisher_p > 0.05 else "different"
    calls_state = "same" if mwu_p > 0.05 else "different"
    print(
        f"COMPARE control_vs_original_p2 hits={c_hits}/{len(control)} vs "
        f"{o_hits}/{len(orig)} fisher_p={fisher_p:.6g} mwu_U={u_stat:.1f} "
        f"mwu_p={mwu_p:.6g} wall_median_ratio={ratio_s} "
        f"hits={hits_state} calls={calls_state} wall={wall_state}"
    )
    if hits_state == "same" and calls_state == "same" and wall_state == "covers-1":
        print(
            "AGREE: original-session p=2 and control p=2 agree on hits, "
            "calls-to-hit, and the hit-conditioned wall median"
        )
    elif wall_state == "missing":
        print("INCONCLUSIVE: control or original p=2 has no hit-conditioned wall")
    else:
        parts = []
        if hits_state != "same":
            parts.append("hits")
        if calls_state != "same":
            parts.append("calls-to-hit")
        if wall_state != "covers-1":
            parts.append("hit-conditioned wall median")
        print(
            "DISAGREE: original-session p=2 and control p=2 differ on "
            + ", ".join(parts)
        )

    p1_control = [r for r in rows if session_of(r) == "p1-control"]
    p2_control = [r for r in rows if session_of(r) == "p2-control"]
    if any(int(r["workers"]) != 1 for r in p1_control):
        raise SystemExit("p1-control session is only the p=1 rung")
    if any(int(r["workers"]) != 2 for r in p2_control):
        raise SystemExit("p2-control session is only the p=2 rung")
    print_arm("p1-control", p1_control)
    print_arm("recorded-p1", primary)
    print_arm("p2-control", p2_control)
    print_arm("original-p2", original.get(2, []))
    print_agreement(
        "p2_control_vs_original_p2",
        "same-VM p=2",
        "original p=2",
        p2_control,
        original.get(2, []),
    )
    print_agreement(
        "p1_control_vs_recorded_p1",
        "p=1 control",
        "recorded p=1",
        p1_control,
        primary,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("scaling", "warmstart", "followup"))
    parser.add_argument("rows", type=Path)
    args = parser.parse_args()
    if args.mode == "scaling":
        summarize_scaling(args.rows)
    elif args.mode == "warmstart":
        summarize_warmstart(args.rows)
    else:
        summarize_followup(args.rows)


if __name__ == "__main__":
    main()
