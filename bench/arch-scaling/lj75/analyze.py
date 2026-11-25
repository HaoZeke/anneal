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
    rows = load_rows(path)
    by_workers: dict[int, list[dict[str, str]]] = {}
    for r in rows:
        by_workers.setdefault(int(r["workers"]), []).append(r)

    t1_hits_wall = [
        float(r["wall_to_hit_s"])
        for r in by_workers.get(1, [])
        if r["hit"] == "1" and r["wall_to_hit_s"]
    ]
    base_rung = max(by_workers) if 1 not in by_workers else 1
    base_rows = by_workers[base_rung]
    base_hits = sum(r["hit"] == "1" for r in base_rows)
    base_calls = [float(r["calls_to_hit"]) for r in base_rows]

    print(
        f"workers,n,hits,hit_rate,hit_ci95,wall_to_hit_med_s,wall_to_hit_ci95,"
        f"calls_to_hit_med,calls_to_hit_ci95,wall_med_s,cpu_med_s,peak_rss_med_kib,"
        f"efficiency,efficiency_ci95,fisher_p_vs_p{base_rung},mwu_p_vs_p{base_rung},oversubscribed"
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
            wh_ci = f"[{wh_lo:.3f}, {wh_hi:.3f}]"
        else:
            wh_str, wh_ci = "NA", "NA"

        over = grp[0]["oversubscribed"]
        if over == "true" or not t1_hits_wall or not wall_hit_vec:
            eff_str, eff_ci = "NA", "NA"
        else:
            eff = bootstrap_ratio_ci(t1_hits_wall, wall_hit_vec, w)
            assert eff is not None
            eff_str = f"{eff[0]:.3f}"
            eff_ci = f"[{eff[1]:.3f}, {eff[2]:.3f}]"

        fp = paired_stats.fisher_exact(hits, n, base_hits, len(base_rows))
        _, mp = mann_whitney_u(calls_vec, base_calls)
        wall_med = statistics.median(float(r["wall_s"]) for r in grp)
        cpu_med = statistics.median(float(r["cpu_s"]) for r in grp)
        rss_med = statistics.median(float(r["peak_rss_anon_kib"]) for r in grp)
        print(
            f"{w},{n},{hits}/{n},{hr_est:.3f},[{hr_lo:.3f}; {hr_hi:.3f}],"
            f"{wh_str},{wh_ci},{c_est:.0f},[{c_lo:.0f}; {c_hi:.0f}],"
            f"{wall_med:.3f},{cpu_med:.2f},{rss_med:.0f},{eff_str},{eff_ci},"
            f"{fp:.6g},{mp:.6g},{over}"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("scaling", "warmstart"))
    parser.add_argument("rows", type=Path)
    args = parser.parse_args()
    if args.mode == "scaling":
        summarize_scaling(args.rows)
    else:
        summarize_warmstart(args.rows)


if __name__ == "__main__":
    main()
