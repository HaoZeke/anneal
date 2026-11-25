#!/usr/bin/env python3
"""Score PR 182 r-next A/B rows. Time and RSS stay context."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20261123
RATIO_NOISE = 1e-9

SHA_A = "d41f0ae293884d1173d6ab1a7201e66749d3848a"
SHA_B = "af23f1f7f977d5b4a42e50c2a8993233b40853f4"

CELL_ORDER = (
    ("lj98_short_d0", 2500, 0),
    ("lj98_short_d4", 2500, 4),
    ("lj98_traj_d0", 5000, 0),
    ("lj98_traj_d4", 5000, 4),
    ("lj38_short_d0", 2500, 0),
    ("lj38_short_d4", 2500, 4),
    ("lj38_d0", 400_000, 0),
    ("lj38_d4", 400_000, 4),
    ("lj55", 1_000_000, 4),
    ("lj75", 4_000_000, 4),
    ("lj98", 4_000_000, 4),
    ("lj102", 4_000_000, 4),
    ("lj103", 4_000_000, 4),
    ("lj104", 4_000_000, 4),
    ("lj150", 4_000_000, 4),
)

SUMMARY_FIELDS = (
    "cell",
    "n",
    "budget",
    "deferral",
    "hits_a",
    "hits_b",
    "hits_a2",
    "fisher_two_sided_p",
    "fisher_a_greater_p",
    "med_calls_a",
    "med_calls_b",
    "calls_med_ba",
    "calls_ci_lo_ba",
    "calls_ci_hi_ba",
    "mwu_u_b",
    "mwu_two_sided_p",
    "mwu_b_slower_p",
    "med_best_a",
    "med_best_b",
    "max_abs_de",
    "med_hops_a",
    "med_hops_b",
    "outside_med_ba",
    "outside_ci_lo_ba",
    "outside_ci_hi_ba",
    "outside_med_a2a",
    "outside_ci_lo_a2a",
    "outside_ci_hi_a2a",
    "rss_med_ba",
    "rss_ci_lo_ba",
    "rss_ci_hi_ba",
    "rss_med_a2a",
    "rss_ci_lo_a2a",
    "rss_ci_hi_a2a",
    "aa_hit_mismatch",
    "aa_calls_mismatch",
    "hit_loss",
    "calls_loss",
    "verdict",
)


def type7(values: list[float], prob: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n < 1:
        raise ValueError("type7 needs a non-empty sample")
    if n == 1:
        return ordered[0]
    pos = (n - 1) * prob
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def bootstrap_median_ci(values: list[float], rng: random.Random) -> tuple[float, float, float]:
    est = type7(values, 0.5)
    n = len(values)
    boots = [
        type7([values[rng.randrange(n)] for _ in range(n)], 0.5) for _ in range(BOOTSTRAPS)
    ]
    return est, type7(boots, 0.025), type7(boots, 0.975)


def fisher_exact(hits_a: int, hits_b: int, n: int) -> tuple[float, float]:
    total_hits = hits_a + hits_b
    denom = math.comb(2 * n, total_hits)

    def prob(k: int) -> float:
        other = total_hits - k
        if not 0 <= other <= n:
            return 0.0
        return math.comb(n, k) * math.comb(n, other) / denom

    p_obs = prob(hits_a)
    p_two = min(1.0, sum(prob(k) for k in range(n + 1) if prob(k) <= p_obs + 1e-12))
    p_a_gt = min(1.0, sum(prob(k) for k in range(hits_a, n + 1)))
    return p_two, p_a_gt


def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float, float]:
    pooled = sorted(a + b)
    ranks: dict[float, float] = {}
    tie_term = 0
    idx = 0
    while idx < len(pooled):
        end = idx
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[idx]:
            end += 1
        ranks[pooled[idx]] = (idx + end + 2) / 2.0
        w = end - idx + 1
        tie_term += w**3 - w
        idx = end + 1
    n_a, n_b = len(a), len(b)
    u_b = sum(ranks[v] for v in b) - n_b * (n_b + 1) / 2.0
    mean = n_a * n_b / 2.0
    total = n_a + n_b
    var = n_a * n_b / 12.0 * (total + 1 - tie_term / (total * (total - 1)))
    if var <= 0.0:
        return u_b, 1.0, 1.0
    sd = math.sqrt(var)
    diff = abs(u_b - mean) - 0.5
    z_two = max(0.0, diff) / sd
    p_two = min(1.0, math.erfc(z_two / math.sqrt(2.0)))
    z_b_gt = (u_b - mean - 0.5) / sd
    p_b_gt = 0.5 * math.erfc(z_b_gt / math.sqrt(2.0))
    return u_b, p_two, p_b_gt


def fmt(val: float, digits: int = 8) -> str:
    return f"{val:.{digits}g}"


def fmt_calls(val: float) -> str:
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.1f}"


def load_builds(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError("builds.csv is missing")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    arms = [row["arm"] for row in rows]
    if arms != ["A", "A2", "B"]:
        raise RuntimeError(f"builds.csv arms {arms}")
    for row in rows:
        if row["arm"] in ("A", "A2") and row["commit"] != SHA_A:
            raise RuntimeError("A build commit mismatch")
        if row["arm"] == "B" and row["commit"] != SHA_B:
            raise RuntimeError("B build commit mismatch")
        if row["profile"] != "release":
            raise RuntimeError("build profile is not release")
    if rows[0]["tree"] != rows[1]["tree"]:
        raise RuntimeError("A and A2 trees differ")
    return rows


def summarize(rows_path: Path) -> list[dict[str, str]]:
    with rows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_cell_arm: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_cell_arm.setdefault((row["cell"], row["arm"]), []).append(row)

    missing = [cell for cell, _, _ in CELL_ORDER if (cell, "A") not in by_cell_arm]
    if missing:
        raise RuntimeError(f"missing cells: {missing}")

    rng = random.Random(BOOTSTRAP_SEED)
    out: list[dict[str, str]] = []

    for cell, budget, deferral in CELL_ORDER:
        arms = {arm: by_cell_arm[(cell, arm)] for arm in ("A", "B", "A2")}
        n = len(arms["A"])
        if n != 12 or any(len(arms[a]) != n for a in arms):
            raise RuntimeError(f"{cell}: arm counts {[len(arms[a]) for a in arms]}")

        for i in range(n):
            seeds = {arms[a][i]["seed"] for a in arms}
            reps = {arms[a][i]["inner_reps"] for a in arms}
            defs = {arms[a][i]["deferral"] for a in arms}
            if len(seeds) != 1 or len(reps) != 1 or defs != {str(deferral)}:
                raise RuntimeError(f"{cell} round {i}: seed/reps/deferral mismatch")
            if arms["A"][i]["sha"] != SHA_A or arms["A2"][i]["sha"] != SHA_A:
                raise RuntimeError(f"{cell} round {i}: A sha mismatch")
            if arms["B"][i]["sha"] != SHA_B:
                raise RuntimeError(f"{cell} round {i}: B sha mismatch")
            if any(arms[a][i]["core"] != "0" for a in arms):
                raise RuntimeError(f"{cell} round {i}: core is not 0")

        hits = {a: sum(int(r["hit"]) for r in arms[a]) for a in arms}
        calls = {
            a: [
                float(r["calls_to_hit"]) if int(r["hit"]) == 1 else float(budget + 1)
                for r in arms[a]
            ]
            for a in arms
        }
        if any(v <= 0.0 for series in calls.values() for v in series):
            raise RuntimeError(f"{cell}: non-positive calls_to_hit")
        best = {a: [float(r["best_energy"]) for r in arms[a]] for a in arms}
        hops = {a: [float(r["hops"]) for r in arms[a]] for a in arms}
        calls_ba = [b / a for a, b in zip(calls["A"], calls["B"], strict=True)]
        outside_ba = [
            float(b["outside_s"]) / float(a["outside_s"])
            for a, b in zip(arms["A"], arms["B"], strict=True)
        ]
        outside_a2a = [
            float(a2["outside_s"]) / float(a["outside_s"])
            for a, a2 in zip(arms["A"], arms["A2"], strict=True)
        ]
        rss_ba = [
            float(b["rss_anon_kib"]) / float(a["rss_anon_kib"])
            for a, b in zip(arms["A"], arms["B"], strict=True)
        ]
        rss_a2a = [
            float(a2["rss_anon_kib"]) / float(a["rss_anon_kib"])
            for a, a2 in zip(arms["A"], arms["A2"], strict=True)
        ]

        calls_med, calls_lo, calls_hi = bootstrap_median_ci(calls_ba, rng)
        out_ba_med, out_ba_lo, out_ba_hi = bootstrap_median_ci(outside_ba, rng)
        out_a2a_med, out_a2a_lo, out_a2a_hi = bootstrap_median_ci(outside_a2a, rng)
        rss_ba_med, rss_ba_lo, rss_ba_hi = bootstrap_median_ci(rss_ba, rng)
        rss_a2a_med, rss_a2a_lo, rss_a2a_hi = bootstrap_median_ci(rss_a2a, rng)

        f_two, f_a_gt = fisher_exact(hits["A"], hits["B"], n)
        u_b, mwu_two, mwu_b_gt = mann_whitney(calls["A"], calls["B"])
        med_calls_a = type7(calls["A"], 0.5)
        med_calls_b = type7(calls["B"], 0.5)
        max_abs_de = max(abs(b - a) for a, b in zip(best["A"], best["B"], strict=True))

        aa_hit_mismatch = sum(
            int(a["hit"]) != int(a2["hit"]) for a, a2 in zip(arms["A"], arms["A2"], strict=True)
        )
        aa_calls_mismatch = sum(
            float(a["calls_to_hit"]) != float(a2["calls_to_hit"])
            for a, a2 in zip(arms["A"], arms["A2"], strict=True)
        )
        hit_loss = hits["B"] < hits["A"]
        calls_loss = (
            med_calls_b > med_calls_a or mwu_b_gt < 0.05 or calls_lo > 1.0 + RATIO_NOISE
        )
        verdict = "FAIL" if hit_loss or calls_loss else "PASS"

        out.append(
            {
                "cell": cell,
                "n": str(n),
                "budget": str(budget),
                "deferral": str(deferral),
                "hits_a": str(hits["A"]),
                "hits_b": str(hits["B"]),
                "hits_a2": str(hits["A2"]),
                "fisher_two_sided_p": fmt(f_two),
                "fisher_a_greater_p": fmt(f_a_gt),
                "med_calls_a": fmt_calls(med_calls_a),
                "med_calls_b": fmt_calls(med_calls_b),
                "calls_med_ba": fmt(calls_med),
                "calls_ci_lo_ba": fmt(calls_lo),
                "calls_ci_hi_ba": fmt(calls_hi),
                "mwu_u_b": fmt(u_b),
                "mwu_two_sided_p": fmt(mwu_two),
                "mwu_b_slower_p": fmt(mwu_b_gt),
                "med_best_a": f"{type7(best['A'], 0.5):.12g}",
                "med_best_b": f"{type7(best['B'], 0.5):.12g}",
                "max_abs_de": f"{max_abs_de:.3e}",
                "med_hops_a": fmt(type7(hops["A"], 0.5)),
                "med_hops_b": fmt(type7(hops["B"], 0.5)),
                "outside_med_ba": fmt(out_ba_med),
                "outside_ci_lo_ba": fmt(out_ba_lo),
                "outside_ci_hi_ba": fmt(out_ba_hi),
                "outside_med_a2a": fmt(out_a2a_med),
                "outside_ci_lo_a2a": fmt(out_a2a_lo),
                "outside_ci_hi_a2a": fmt(out_a2a_hi),
                "rss_med_ba": fmt(rss_ba_med),
                "rss_ci_lo_ba": fmt(rss_ba_lo),
                "rss_ci_hi_ba": fmt(rss_ba_hi),
                "rss_med_a2a": fmt(rss_a2a_med),
                "rss_ci_lo_a2a": fmt(rss_a2a_lo),
                "rss_ci_hi_a2a": fmt(rss_a2a_hi),
                "aa_hit_mismatch": str(aa_hit_mismatch),
                "aa_calls_mismatch": str(aa_calls_mismatch),
                "hit_loss": "1" if hit_loss else "0",
                "calls_loss": "1" if calls_loss else "0",
                "verdict": verdict,
            }
        )
    return out


def write_summary(summaries: list[dict[str, str]], builds: list[dict[str, str]]) -> None:
    with (HERE / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    overall = "PASS" if summaries and all(r["verdict"] == "PASS" for r in summaries) else "FAIL"
    lines = [
        "# PR 182 r-next A/B Gate Summary",
        "",
        f"Overall verdict: **{overall}**",
        "",
        "A cell fails only on a hit loss or a calls-to-hit loss.",
        "Outside time and RSS are context.",
        "",
        "| Arm | Commit | Tree | Binary sha256 |",
        "| --- | --- | --- | --- |",
    ]
    for row in builds:
        lines.append(
            f"| {row['arm']} | `{row['commit']}` | `{row['tree']}` | `{row['binary_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "| Cell | Hits A/B | Fisher 2s / A>B | Med calls A/B | "
            "Calls B/A [95% CI] | MWU 2s / B>A | "
            "Outside B/A [95% CI] (A2/A) | RSS B/A [95% CI] (A2/A) | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['cell']} | {row['hits_a']}/{row['hits_b']} | "
            f"{row['fisher_two_sided_p']} / {row['fisher_a_greater_p']} | "
            f"{row['med_calls_a']} / {row['med_calls_b']} | "
            f"{row['calls_med_ba']} [{row['calls_ci_lo_ba']}, {row['calls_ci_hi_ba']}] | "
            f"{row['mwu_two_sided_p']} / {row['mwu_b_slower_p']} | "
            f"{row['outside_med_ba']} [{row['outside_ci_lo_ba']}, {row['outside_ci_hi_ba']}] "
            f"({row['outside_ci_lo_a2a']}..{row['outside_ci_hi_a2a']}) | "
            f"{row['rss_med_ba']} [{row['rss_ci_lo_ba']}, {row['rss_ci_hi_ba']}] "
            f"({row['rss_ci_lo_a2a']}..{row['rss_ci_hi_a2a']}) | {row['verdict']} |"
        )
    lines.append("")
    (HERE / "SUMMARY.md").write_text("\n".join(lines), newline="\n")


def main() -> int:
    builds = load_builds(HERE / "builds.csv")
    summaries = summarize(HERE / "rows.csv")
    write_summary(summaries, builds)
    overall = "PASS" if summaries and all(r["verdict"] == "PASS" for r in summaries) else "FAIL"
    print(f"overall={overall} cells={len(summaries)}")
    return 0 if len(summaries) == len(CELL_ORDER) else 1


if __name__ == "__main__":
    raise SystemExit(main())
