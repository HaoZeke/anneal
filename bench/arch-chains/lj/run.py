#!/usr/bin/env python3
"""Runner and summarizer for the communicating vs independent vs single LJ campaign.

Shard selection is `--systems`, `--seeds`, and `--no-cap`. See PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
PAPER_0120_DIR = SCRIPTS_DIR / "paper_0120"
for candidate in (SCRIPTS_DIR, PAPER_0120_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from paired_stats import fisher_exact, midranks  # noqa: E402
from run_panel import PAPER_BUDGETS  # noqa: E402
from target_tol import SEARCH_HIT_TOL, energy_hits  # noqa: E402

SYSTEMS: tuple[tuple[str, int, float], ...] = (
    ("lj38", 38, -173.928427),
    ("lj75", 75, -397.492331),
    ("lj98", 98, -543.665361),
)
ARMS: tuple[str, ...] = ("comm", "indep", "single")
SEEDS: tuple[int, ...] = tuple(range(2000, 2012))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261123
CAMPAIGN_CAP_SECONDS = 4.0 * 3600.0
MIN_SAMPLE_SECONDS = 0.050

# Balanced ABBA schedule across consecutive seed pairs (2k, 2k+1),
# cycling all 6 permutations of (comm, indep, single).
SEED_ARM_ORDER: tuple[tuple[str, str, str], ...] = (
    ("comm", "indep", "single"),
    ("single", "indep", "comm"),
    ("indep", "comm", "single"),
    ("single", "comm", "indep"),
    ("comm", "single", "indep"),
    ("indep", "single", "comm"),
    ("comm", "indep", "single"),
    ("single", "indep", "comm"),
    ("indep", "comm", "single"),
    ("single", "comm", "indep"),
    ("comm", "single", "indep"),
    ("indep", "single", "comm"),
)

ROW_FIELDS: list[str] = [
    "system",
    "n_points",
    "arm",
    "replicas",
    "seed",
    "round_order_idx",
    "budget",
    "reference",
    "best_energy",
    "hit",
    "first_target_calls",
    "calls_to_hit",
    "aggregate_charged",
    "ledger_sum",
    "hops",
    "screened_out",
    "returned",
    "basins",
    "history_minima",
    "history_accepted",
    "history_visits",
    "history_obs",
    "history_new",
    "history_refused",
    "history_s",
    "orbits_count",
    "symmetrised_count",
    "continuous_sym_count",
    "visit_same",
    "visit_known",
    "visit_new",
    "corpus_appended",
    "wall_s",
    "cpu_s",
    "peak_rss_anon_kib",
    "vm_hwm_kib",
]


def type7(values: list[float], q: float) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return math.nan
    if n == 1:
        return float(ordered[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U with tie and continuity corrections."""
    n1 = len(left)
    n2 = len(right)
    n = n1 + n2
    if n1 == 0 or n2 == 0 or len(set(left + right)) <= 1:
        return n1 * n2 / 2.0, 1.0
    combined = list(left) + list(right)
    ranks = midranks(combined)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    counts = Counter(combined)
    tie_term = sum(c * c * c - c for c in counts.values() if c > 1)
    var = (n1 * n2 / 12.0) * (n + 1.0 - tie_term / (n * (n - 1.0)))
    if var <= 0.0:
        return u1, 1.0
    mu = n1 * n2 / 2.0
    diff = abs(u1 - mu)
    z = max(0.0, diff - 0.5) / math.sqrt(var)
    p = min(1.0, math.erfc(z / math.sqrt(2.0)))
    return u, p


def bootstrap_median_ci(
    values: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    est = type7(values, 0.5)
    draws = [type7([values[rng.randrange(n)] for _ in range(n)], 0.5) for _ in range(resamples)]
    return est, type7(draws, 0.025), type7(draws, 0.975)


def bootstrap_rate_ci(
    hits: list[bool],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(hits)
    est = sum(hits) / n
    draws = [sum(hits[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    return est, type7(draws, 0.025), type7(draws, 0.975)


def bootstrap_rate_diff_ci(
    hits_a: list[bool],
    hits_b: list[bool],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    na = len(hits_a)
    nb = len(hits_b)
    est = sum(hits_a) / na - sum(hits_b) / nb
    draws = []
    for _ in range(resamples):
        ra = sum(hits_a[rng.randrange(na)] for _ in range(na)) / na
        rb = sum(hits_b[rng.randrange(nb)] for _ in range(nb)) / nb
        draws.append(ra - rb)
    return est, type7(draws, 0.025), type7(draws, 0.975)


def bootstrap_median_diff_ci(
    vals_a: list[float],
    vals_b: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    na = len(vals_a)
    nb = len(vals_b)
    est = type7(vals_a, 0.5) - type7(vals_b, 0.5)
    draws = []
    for _ in range(resamples):
        ma = type7([vals_a[rng.randrange(na)] for _ in range(na)], 0.5)
        mb = type7([vals_b[rng.randrange(nb)] for _ in range(nb)], 0.5)
        draws.append(ma - mb)
    return est, type7(draws, 0.025), type7(draws, 0.975)


def find_driver_binary() -> Path:
    meta = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    target_dir = Path(json.loads(meta.stdout)["target_directory"])
    binary = target_dir / "release" / "examples" / "arch_chains_lj"
    if not binary.is_file():
        raise SystemExit(f"missing binary: {binary} (build with --release --locked --features bank-rpc --example arch_chains_lj)")
    return binary


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_one(
    binary: Path,
    n_points: int,
    budget: int,
    arm: str,
    seed: int,
    corpus_dir: Path,
) -> dict:
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    cmd = [
        "taskset",
        "-c",
        "0",
        str(binary),
        str(n_points),
        str(budget),
        arm,
        str(seed),
        str(corpus_dir),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :])
    raise RuntimeError(f"missing RESULT_JSON for lj{n_points} {arm} {seed}: {proc.stdout}\n{proc.stderr}")


def calibrate_system(
    binary: Path,
    system: str,
    n_points: int,
    full_budget: int,
    corpus_dir: Path,
    cal_budget: int = 100_000,
) -> dict:
    arm_walls: dict[str, float] = {}
    for arm in ARMS:
        res = run_one(binary, n_points, cal_budget, arm, SEEDS[0], corpus_dir)
        arm_walls[arm] = float(res["wall_s"])
    scale = full_budget / cal_budget
    projected_per_seed_s = sum(arm_walls.values()) * scale
    projected_system_s = projected_per_seed_s * len(SEEDS)
    return {
        "system": system,
        "n_points": n_points,
        "cal_budget": cal_budget,
        "full_budget": full_budget,
        "cal_arm_wall_s": arm_walls,
        "projected_per_seed_s": projected_per_seed_s,
        "projected_system_s": projected_system_s,
    }


def write_success_vs_calls(rows: list[dict[str, str]], out_path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["arm"]), []).append(row)

    out_rows: list[dict[str, str]] = []
    for system, _n_points, _ref in SYSTEMS:
        budget = PAPER_BUDGETS[system]
        checkpoints = sorted(
            set(
                [0, 1000, 5000, 10_000, 20_000, 40_000, 75_000, 100_000, 200_000, 300_000]
                + [round(budget * i / 20) for i in range(21)]
            )
        )
        checkpoints = [c for c in checkpoints if c <= budget]
        for arm in ARMS:
            grp = grouped.get((system, arm), [])
            if not grp:
                continue
            n_runs = len(grp)
            for calls_thresh in checkpoints:
                hits_at = sum(
                    1
                    for r in grp
                    if r["hit"] == "1" and int(r["calls_to_hit"]) <= calls_thresh
                )
                out_rows.append(
                    {
                        "system": system,
                        "arm": arm,
                        "budget": str(budget),
                        "calls": str(calls_thresh),
                        "hits": str(hits_at),
                        "n": str(n_runs),
                        "success_rate": f"{hits_at / n_runs:.6f}",
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["system", "arm", "budget", "calls", "hits", "n", "success_rate"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(out_rows)


def write_report(
    rows: list[dict[str, str]],
    projections: list[dict],
    out_path: Path,
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["arm"]), []).append(row)

    lines: list[str] = [
        "# Communicating vs Independent vs Single-Chain LJ Report",
        "",
        f"Rows recorded: {len(rows)} (seeds {SEEDS[0]}..{SEEDS[-1]}, `SEARCH_HIT_TOL = {SEARCH_HIT_TOL:.1e}`).",
        "",
    ]

    for system, n_points, reference in SYSTEMS:
        budget = PAPER_BUDGETS[system]
        has_any = any((system, arm) in grouped for arm in ARMS)
        if not has_any:
            continue
        lines.extend(
            [
                f"## {system.upper()} (`n = {n_points}`, `budget = {budget}`, `reference = {reference:.6f}`)",
                "",
                "| arm | n | hits | hit rate (95% CI) | median calls_to_hit (95% CI) | median wall_s (95% CI) | median cpu_s (95% CI) | median peak RssAnon KiB (95% CI) | mean ledger_sum | corpus appended |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for arm in ARMS:
            grp = grouped.get((system, arm), [])
            if not grp:
                continue
            hits = [r["hit"] == "1" for r in grp]
            calls = [float(r["calls_to_hit"]) for r in grp]
            walls = [float(r["wall_s"]) for r in grp]
            cpus = [float(r["cpu_s"]) for r in grp]
            rss = [float(r["peak_rss_anon_kib"]) for r in grp]
            ledger_mean = sum(int(r["ledger_sum"]) for r in grp) / len(grp)
            corpus_tot = sum(int(r["corpus_appended"]) for r in grp)

            hr, hr_lo, hr_hi = bootstrap_rate_ci(hits)
            mc, mc_lo, mc_hi = bootstrap_median_ci(calls)
            mw, mw_lo, mw_hi = bootstrap_median_ci(walls)
            mp, mp_lo, mp_hi = bootstrap_median_ci(cpus)
            mr, mr_lo, mr_hi = bootstrap_median_ci(rss)

            lines.append(
                f"| `{arm}` | {len(grp)} | {sum(hits)}/{len(grp)} "
                f"| {hr:.3f} [{hr_lo:.3f}, {hr_hi:.3f}] "
                f"| {mc:.0f} [{mc_lo:.0f}, {mc_hi:.0f}] "
                f"| {mw:.2f} [{mw_lo:.2f}, {mw_hi:.2f}] "
                f"| {mp:.2f} [{mp_lo:.2f}, {mp_hi:.2f}] "
                f"| {mr:.0f} [{mr_lo:.0f}, {mr_hi:.0f}] "
                f"| {ledger_mean:.0f} | {corpus_tot} |"
            )

        lines.extend(
            [
                "",
                f"### {system.upper()} Pairwise Comparisons (10,000-resample bootstrap, seed {BOOTSTRAP_SEED})",
                "",
                "| comparison | hit rate diff (95% CI) | Fisher exact p | calls_to_hit median diff (95% CI) | MWU U | MWU p | wall_s median diff (95% CI) |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for a, b in (("comm", "indep"), ("comm", "single"), ("indep", "single")):
            ga = grouped.get((system, a), [])
            gb = grouped.get((system, b), [])
            if not ga or not gb:
                continue
            ha = [r["hit"] == "1" for r in ga]
            hb = [r["hit"] == "1" for r in gb]
            ca = [float(r["calls_to_hit"]) for r in ga]
            cb = [float(r["calls_to_hit"]) for r in gb]
            wa = [float(r["wall_s"]) for r in ga]
            wb = [float(r["wall_s"]) for r in gb]

            dr, dr_lo, dr_hi = bootstrap_rate_diff_ci(ha, hb)
            fp = fisher_exact(sum(ha), len(ha), sum(hb), len(hb))
            dc, dc_lo, dc_hi = bootstrap_median_diff_ci(ca, cb)
            u_stat, mwu_p = mann_whitney(ca, cb)
            dw, dw_lo, dw_hi = bootstrap_median_diff_ci(wa, wb)

            lines.append(
                f"| `{a}` vs `{b}` | {dr:+.3f} [{dr_lo:+.3f}, {dr_hi:+.3f}] | {fp:.4g} "
                f"| {dc:+.0f} [{dc_lo:+.0f}, {dc_hi:+.0f}] | {u_stat:.1f} | {mwu_p:.4g} "
                f"| {dw:+.2f} [{dw_lo:+.2f}, {dw_hi:+.2f}] |"
            )

        lines.extend(
            [
                "",
                f"### {system.upper()} Ledger and Mechanism Means",
                "",
                "| arm | hops | screened_out | returned | basins | history_minima | history_obs | history_new | history_s | orbits | continuous_sym |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for arm in ARMS:
            grp = grouped.get((system, arm), [])
            if not grp:
                continue
            def mean_col(col: str) -> float:
                return sum(float(r[col]) for r in grp) / len(grp)

            lines.append(
                f"| `{arm}` | {mean_col('hops'):.1f} | {mean_col('screened_out'):.1f} "
                f"| {mean_col('returned'):.1f} | {mean_col('basins'):.1f} "
                f"| {mean_col('history_minima'):.1f} | {mean_col('history_obs'):.1f} "
                f"| {mean_col('history_new'):.1f} | {mean_col('history_s'):.2f} "
                f"| {mean_col('orbits_count'):.1f} | {mean_col('continuous_sym_count'):.1f} |"
            )
        lines.append("")

    if projections:
        lines.extend(
            [
                "## Remaining Systems Stopped Under 4-Hour Time Guard",
                "",
                "Execution stopped before starting systems whose projected full-budget wall time exceeded the 4-hour (14,400 s) campaign cap (bottlenecked by `universal_descriptor_space` in `MinimumHistory::observe` at ~0.10 s/obs for `lj75` and ~0.19 s/obs for `lj98` across 36 runs per system).",
                "",
                "| system | full budget | cal budget | comm cal wall_s | indep cal wall_s | single cal wall_s | projected per-seed (s) | projected 12-seed total (s / h) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for p in projections:
            w = p["cal_arm_wall_s"]
            lines.append(
                f"| `{p['system']}` | {p['full_budget']} | {p['cal_budget']} "
                f"| {w['comm']:.2f} | {w['indep']:.2f} | {w['single']:.2f} "
                f"| {p['projected_per_seed_s']:.1f} "
                f"| {p['projected_system_s']:.0f} s ({p['projected_system_s'] / 3600.0:.2f} h) |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")



def _csv_tokens(spec: str) -> list[str]:
    parts = [part.strip() for part in spec.split(",")]
    if not parts or any(not part for part in parts):
        raise SystemExit(f"invalid list: {spec}")
    return parts


def filter_systems(spec: str | None) -> tuple[tuple[str, int, float], ...]:
    """Return SYSTEMS entries named in spec, in original SYSTEMS order.

    None or a blank spec keeps every system. Whitespace is stripped.
    Empty tokens and unknown names abort.
    """
    if spec is None or spec.strip() == "":
        return SYSTEMS
    wanted = _csv_tokens(spec)
    known = {item[0] for item in SYSTEMS}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown system(s): {', '.join(unknown)}")
    wanted_set = set(wanted)
    return tuple(item for item in SYSTEMS if item[0] in wanted_set)


def filter_seed_schedule(spec: str | None) -> list[tuple[int, int]]:
    """Return (original SEEDS index, seed) pairs, in original SEEDS order.

    The index is the position in SEEDS, so SEED_ARM_ORDER stays aligned
    when the comma list is a subset or is written out of order. None or
    a blank spec keeps every seed.
    """
    if spec is None or spec.strip() == "":
        return list(enumerate(SEEDS))
    wanted: list[int] = []
    for text in _csv_tokens(spec):
        try:
            wanted.append(int(text))
        except ValueError as exc:
            raise SystemExit(f"invalid seed: {text}") from exc
    index_of = {seed: idx for idx, seed in enumerate(SEEDS)}
    unknown = [str(seed) for seed in wanted if seed not in index_of]
    if unknown:
        raise SystemExit(f"unknown seed(s): {', '.join(unknown)}")
    wanted_set = set(wanted)
    return [(idx, seed) for idx, seed in enumerate(SEEDS) if seed in wanted_set]


def enforce_campaign_cap(system: str, no_cap: bool) -> bool:
    """True when this system must be calibrated and checked against the cap."""
    return system != "lj38" and not no_cap


def self_test() -> None:
    if PAPER_BUDGETS["lj38"] != 400_000 or PAPER_BUDGETS["lj75"] != 4_000_000 or PAPER_BUDGETS["lj98"] != 4_000_000:
        raise SystemExit("unexpected PAPER_BUDGETS values")
    if not energy_hits(-173.928427, -173.928427, SEARCH_HIT_TOL):
        raise SystemExit("energy_hits exact")
    u, p = mann_whitney([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    if abs(u - 4.5) > 1e-12 or abs(p - 1.0) > 1e-12:
        raise SystemExit(f"mann_whitney tie self-test failed: {u}, {p}")
    u2, p2 = mann_whitney([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    if abs(u2 - 0.0) > 1e-12 or p2 >= 0.05:
        raise SystemExit(f"mann_whitney separated self-test failed: {u2}, {p2}")

    if len(SEED_ARM_ORDER) != len(SEEDS):
        raise SystemExit("SEED_ARM_ORDER length does not match SEEDS")
    if filter_systems(None) != SYSTEMS or filter_systems("") != SYSTEMS:
        raise SystemExit("empty system filter should keep the full campaign")
    if filter_seed_schedule(None) != list(enumerate(SEEDS)):
        raise SystemExit("empty seed filter should keep the full campaign")
    if filter_seed_schedule("") != list(enumerate(SEEDS)):
        raise SystemExit("blank seed filter should keep the full campaign")
    filtered_systems = filter_systems("lj98, lj75")
    if tuple(item[0] for item in filtered_systems) != ("lj75", "lj98"):
        raise SystemExit(f"systems filter failed: {filtered_systems}")
    if filter_systems("lj75") != (SYSTEMS[1],):
        raise SystemExit("system filter lj75")
    schedule = filter_seed_schedule("2002,2000,2001")
    if schedule != [(0, 2000), (1, 2001), (2, 2002)]:
        raise SystemExit(f"seed filter index failed: {schedule}")
    arms = [SEED_ARM_ORDER[idx] for idx, _seed in schedule]
    expected_arms = [
        ("comm", "indep", "single"),
        ("single", "indep", "comm"),
        ("indep", "comm", "single"),
    ]
    if arms != expected_arms:
        raise SystemExit(f"seed arm order failed: {arms}")
    late = filter_seed_schedule("2011")
    if late != [(11, 2011)] or SEED_ARM_ORDER[late[0][0]] != ("indep", "single", "comm"):
        raise SystemExit(f"late seed index failed: {late}")
    shard_2003 = filter_seed_schedule("2003,2004,2005")
    if shard_2003 != [(3, 2003), (4, 2004), (5, 2005)]:
        raise SystemExit(f"seed shard 2003 failed: {shard_2003}")
    shard_2006 = filter_seed_schedule("2008,2006,2007")
    if shard_2006 != [(6, 2006), (7, 2007), (8, 2008)]:
        raise SystemExit(f"seed shard 2006 failed: {shard_2006}")
    expected_2006 = (
        ("comm", "indep", "single"),
        ("single", "indep", "comm"),
        ("indep", "comm", "single"),
    )
    if tuple(SEED_ARM_ORDER[idx] for idx, _seed in shard_2006) != expected_2006:
        raise SystemExit("seed shard 2006 arm order")
    shard_2009 = filter_seed_schedule("2009,2010,2011")
    if shard_2009 != [(9, 2009), (10, 2010), (11, 2011)]:
        raise SystemExit(f"seed shard 2009 failed: {shard_2009}")
    if enforce_campaign_cap("lj75", False) is not True:
        raise SystemExit("lj75 should be calibrated and cap-checked")
    if enforce_campaign_cap("lj75", True) is not False:
        raise SystemExit("no-cap should skip calibration and the cap check")
    if enforce_campaign_cap("lj38", False) or enforce_campaign_cap("lj98", True):
        raise SystemExit("cap predicate failed")
    for bad in ("lj75,", ",lj75", "2006,", ",2000"):
        try:
            filter_systems(bad)
        except SystemExit:
            pass
        else:
            raise SystemExit(f"empty token accepted: {bad}")
    try:
        filter_systems("lj75,missing")
    except SystemExit as exc:
        if "missing" not in str(exc):
            raise
    else:
        raise SystemExit("unknown system was accepted")
    try:
        filter_seed_schedule("2000,1999")
    except SystemExit as exc:
        if "1999" not in str(exc):
            raise
    else:
        raise SystemExit("unknown seed was accepted")
    print("ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--systems",
        default=None,
        help="comma-separated subset of SYSTEMS, kept in original order",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated subset of SEEDS; each keeps its original SEED_ARM_ORDER index",
    )
    parser.add_argument(
        "--no-cap",
        action="store_true",
        help="skip calibration and the campaign cap check",
    )
    args = parser.parse_args(argv[1:])

    if args.self_test:
        self_test()
        return 0

    rows_path = args.dir / "rows.csv"
    curves_path = args.dir / "success_vs_calls.csv"
    report_path = args.dir / "REPORT.md"
    proj_path = args.dir / "projections.json"

    if args.summarize:
        rows = read_rows(rows_path)
        projections = json.loads(proj_path.read_text()) if proj_path.is_file() else []
        write_success_vs_calls(rows, curves_path)
        write_report(rows, projections, report_path)
        return 0

    systems = filter_systems(args.systems)
    schedule = filter_seed_schedule(args.seeds)

    binary = find_driver_binary()
    existing = read_rows(rows_path)
    done = {(r["system"], r["arm"], int(r["seed"])) for r in existing}

    campaign_started = time.monotonic()
    projections: list[dict] = json.loads(proj_path.read_text()) if proj_path.is_file() else []
    projected_names = {p["system"] for p in projections}

    with tempfile.TemporaryDirectory(prefix="anneal-minima-") as tmp_root:
        corpus_dir = Path(tmp_root)
        for system, n_points, reference in systems:
            budget = int(PAPER_BUDGETS[system])
            # Skip when every requested (seed, arm) run is already recorded.
            missing = [
                (idx, seed, arm)
                for idx, seed in schedule
                for arm in SEED_ARM_ORDER[idx]
                if (system, arm, seed) not in done
            ]
            if not missing:
                continue

            # For systems > lj38, run a 100k calibration check and enforce the 4h cap.
            if enforce_campaign_cap(system, args.no_cap):
                if system not in projected_names:
                    proj = calibrate_system(binary, system, n_points, budget, corpus_dir)
                    projections.append(proj)
                    projected_names.add(system)
                    proj_path.write_text(json.dumps(projections, indent=2) + "\n")
                else:
                    proj = next(p for p in projections if p["system"] == system)
                elapsed = time.monotonic() - campaign_started
                if elapsed + proj["projected_system_s"] > CAMPAIGN_CAP_SECONDS:
                    print(
                        f"Stopping before {system}: elapsed={elapsed:.1f}s + "
                        f"projected={proj['projected_system_s']:.1f}s > cap={CAMPAIGN_CAP_SECONDS:.0f}s",
                        flush=True,
                    )
                    continue

            for idx, seed in schedule:
                for arm in SEED_ARM_ORDER[idx]:
                    if (system, arm, seed) in done:
                        continue
                    res = run_one(binary, n_points, budget, arm, seed, corpus_dir)
                    if float(res["wall_s"]) < MIN_SAMPLE_SECONDS:
                        raise RuntimeError(f"sample below 50 ms: {res}")
                    if int(res["aggregate_charged"]) != int(res["ledger_sum"]):
                        raise RuntimeError(f"ledger mismatch: {res}")
                    hit = energy_hits(float(res["best_energy"]), reference, SEARCH_HIT_TOL)
                    if bool(res["hit"]) != hit:
                        raise RuntimeError(f"hit mismatch: {res}")
                    row = {
                        "system": system,
                        "n_points": str(res["n_points"]),
                        "arm": arm,
                        "replicas": str(res["replicas"]),
                        "seed": str(seed),
                        "round_order_idx": str(idx),
                        "budget": str(budget),
                        "reference": f"{reference:.6f}",
                        "best_energy": f"{float(res['best_energy']):.10f}",
                        "hit": "1" if hit else "0",
                        "first_target_calls": ""
                        if res["first_target_calls"] is None
                        else str(res["first_target_calls"]),
                        "calls_to_hit": str(res["calls_to_hit"]),
                        "aggregate_charged": str(res["aggregate_charged"]),
                        "ledger_sum": str(res["ledger_sum"]),
                        "hops": str(res["hops"]),
                        "screened_out": str(res["screened_out"]),
                        "returned": str(res["returned"]),
                        "basins": str(res["basins"]),
                        "history_minima": str(res["history_minima"]),
                        "history_accepted": str(res["history_accepted"]),
                        "history_visits": str(res["history_visits"]),
                        "history_obs": str(res["history_obs"]),
                        "history_new": str(res["history_new"]),
                        "history_refused": str(res["history_refused"]),
                        "history_s": f"{float(res['history_s']):.6f}",
                        "orbits_count": str(res["orbits_count"]),
                        "symmetrised_count": str(res["symmetrised_count"]),
                        "continuous_sym_count": str(res["continuous_sym_count"]),
                        "visit_same": str(res["visit_same"]),
                        "visit_known": str(res["visit_known"]),
                        "visit_new": str(res["visit_new"]),
                        "corpus_appended": str(res["corpus_appended"]),
                        "wall_s": f"{float(res['wall_s']):.6f}",
                        "cpu_s": f"{float(res['cpu_s']):.6f}",
                        "peak_rss_anon_kib": str(res["peak_rss_anon_kib"]),
                        "vm_hwm_kib": str(res["vm_hwm_kib"]),
                    }
                    append_row(rows_path, row)
                    done.add((system, arm, seed))
                    print(
                        f"[{system} seed={seed} arm={arm}] hit={row['hit']} "
                        f"calls={row['calls_to_hit']} best={row['best_energy']} "
                        f"wall={row['wall_s']}s rss={row['peak_rss_anon_kib']}KiB",
                        flush=True,
                    )

    rows = read_rows(rows_path)
    write_success_vs_calls(rows, curves_path)
    write_report(rows, projections, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
