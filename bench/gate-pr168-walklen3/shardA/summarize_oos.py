#!/usr/bin/env python3
"""Summarize the out-of-sample A/M/O walk-length gate."""

from __future__ import annotations

import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20_261_123
CELLS = ("lj98", "lj102", "lj103", "lj104", "lj150", "tip4p15", "tip4p20")
EXPECTED_N = {
    "lj98": 36,
    "lj102": 12,
    "lj103": 12,
    "lj104": 12,
    "lj150": 12,
    "tip4p15": 12,
    "tip4p20": 12,
}
ORDERS = ("AMO", "OMA", "MAO", "OAM", "MOA", "AOM")
COMPARISONS = (("O", "M"), ("O", "A"), ("M", "A"))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * probability
    low = math.floor(at)
    high = math.ceil(at)
    if low == high:
        return ordered[low]
    fraction = at - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def median_ratio(candidate: list[float], base: list[float]) -> float:
    return statistics.median(candidate) / statistics.median(base)


def geomean_ratio(candidate: list[float], base: list[float]) -> float:
    return math.exp(
        statistics.fmean(math.log(x / y) for x, y in zip(candidate, base, strict=True))
    )


def bootstrap_ratio(
    candidate: list[float], base: list[float]
) -> tuple[float, float, float]:
    point = median_ratio(candidate, base)
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(candidate)
    estimates = []
    for _ in range(BOOTSTRAPS):
        indices = [rng.randrange(n) for _ in range(n)]
        estimates.append(
            median_ratio(
                [candidate[index] for index in indices],
                [base[index] for index in indices],
            )
        )
    return point, percentile(estimates, 0.025), percentile(estimates, 0.975)


def wilson(hits: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = hits / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return centre - half, centre + half


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    def probability(x: int) -> float:
        return math.comb(col1, x) * math.comb(total - col1, row1 - x) / math.comb(
            total, row1
        )

    low = max(0, row1 - (total - col1))
    high = min(row1, col1)
    observed = probability(a)
    return min(
        1.0,
        sum(
            probability(x)
            for x in range(low, high + 1)
            if probability(x) <= observed + 1.0e-15
        ),
    )


def mann_whitney_two_sided(left: list[float], right: list[float]) -> tuple[float, float]:
    combined = [(value, 0) for value in left] + [(value, 1) for value in right]
    combined.sort()
    rank_sum = [0.0, 0.0]
    ties: list[int] = []
    index = 0
    while index < len(combined):
        end = index + 1
        while end < len(combined) and combined[end][0] == combined[index][0]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for _, group in combined[index:end]:
            rank_sum[group] += rank
        ties.append(end - index)
        index = end
    n1 = len(left)
    n2 = len(right)
    u1 = rank_sum[0] - n1 * (n1 + 1) / 2.0
    mean = n1 * n2 / 2.0
    total = n1 + n2
    tie_term = sum(size**3 - size for size in ties)
    variance = n1 * n2 / 12.0 * (
        total + 1 - tie_term / (total * (total - 1))
    )
    if variance == 0.0:
        return u1, 1.0
    distance = max(0.0, abs(u1 - mean) - 0.5)
    p = math.erfc(distance / math.sqrt(2.0 * variance))
    return u1, min(1.0, p)


def load(path: Path) -> dict[str, dict[int, dict[str, dict[str, str]]]]:
    rows: dict[str, dict[int, dict[str, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cell = row["cell"]
            seed = int(row["seed"])
            arm = row["arm"]
            if arm in rows[cell][seed]:
                raise ValueError(f"duplicate {cell} {seed} {arm}")
            rows[cell][seed][arm] = row
    return rows


def validate(
    rows: dict[str, dict[int, dict[str, dict[str, str]]]]
) -> dict[str, bool]:
    identity: dict[str, bool] = {}
    for cell in CELLS:
        count = len(rows[cell])
        if not 0 < count <= EXPECTED_N[cell]:
            raise ValueError(f"{cell} has {count}, target {EXPECTED_N[cell]}")
        cell_identity = True
        for seed, arms in rows[cell].items():
            if set(arms) != {"A", "M", "O"}:
                raise ValueError(f"{cell} {seed} arms are {sorted(arms)}")
            expected = ORDERS[(seed - 2012) % len(ORDERS)]
            if {row["order"] for row in arms.values()} != {expected}:
                raise ValueError(f"{cell} {seed} order is not {expected}")
            positions = {
                arm: int(row["run_position"]) for arm, row in arms.items()
            }
            if "".join(sorted(positions, key=positions.get)) != expected:
                raise ValueError(f"{cell} {seed} positions do not encode {expected}")
            hashes = {
                (row["trace_sha256"], row["geometry_sha256"])
                for row in arms.values()
            }
            cell_identity &= len(hashes) == 1
        identity[cell] = cell_identity
    return identity


def values(
    rows: dict[str, dict[int, dict[str, dict[str, str]]]],
    cell: str,
    arm: str,
    field: str,
) -> list[float]:
    return [float(rows[cell][seed][arm][field]) for seed in sorted(rows[cell])]


def arm_stats(
    rows: dict[str, dict[int, dict[str, dict[str, str]]]], cell: str, arm: str
) -> dict[str, object]:
    sample = [rows[cell][seed][arm] for seed in sorted(rows[cell])]
    hits = sum(int(row["hit"]) for row in sample)
    return {
        "n": len(sample),
        "hits": hits,
        "wilson": wilson(hits, len(sample)),
        "calls": statistics.median(float(row["calls_to_hit"]) for row in sample),
        "outside": statistics.median(float(row["outside_s"]) for row in sample),
        "rss": statistics.median(float(row["rss_anon_kb"]) for row in sample),
    }


def comparison(
    rows: dict[str, dict[int, dict[str, dict[str, str]]]],
    cell: str,
    candidate: str,
    base: str,
) -> dict[str, object]:
    n = len(rows[cell])
    candidate_hits = sum(int(rows[cell][seed][candidate]["hit"]) for seed in rows[cell])
    base_hits = sum(int(rows[cell][seed][base]["hit"]) for seed in rows[cell])
    paired_losses = sum(
        rows[cell][seed][base]["hit"] == "1"
        and rows[cell][seed][candidate]["hit"] == "0"
        for seed in rows[cell]
    )
    calls_candidate = values(rows, cell, candidate, "calls_to_hit")
    calls_base = values(rows, cell, base, "calls_to_hit")
    outside_candidate = values(rows, cell, candidate, "outside_s")
    outside_base = values(rows, cell, base, "outside_s")
    rss_candidate = values(rows, cell, candidate, "rss_anon_kb")
    rss_base = values(rows, cell, base, "rss_anon_kb")
    calls_ratio = bootstrap_ratio(calls_candidate, calls_base)
    outside_ratio = bootstrap_ratio(outside_candidate, outside_base)
    rss_ratio = bootstrap_ratio(rss_candidate, rss_base)
    fisher = fisher_two_sided(
        candidate_hits,
        n - candidate_hits,
        base_hits,
        n - base_hits,
    )
    mwu_u, mwu_p = mann_whitney_two_sided(calls_candidate, calls_base)
    bars = {
        "hits": candidate_hits >= base_hits,
        "calls": calls_ratio[0] <= 1.0,
        "outside": outside_ratio[2] <= 1.10,
        "rss": rss_ratio[2] <= 1.10,
    }
    clear_failure = (
        candidate_hits < base_hits
        or calls_ratio[0] > 1.0
        or outside_ratio[1] > 1.10
        or rss_ratio[1] > 1.10
    )
    complete = n == EXPECTED_N[cell]
    if clear_failure:
        verdict = "FIX"
    elif complete and all(bars.values()):
        verdict = "PASS"
    else:
        verdict = "UNRESOLVED"
    return {
        "hits": (candidate_hits, base_hits),
        "paired_losses": paired_losses,
        "fisher": fisher,
        "mwu_u": mwu_u,
        "mwu_p": mwu_p,
        "calls": calls_ratio,
        "outside": outside_ratio,
        "rss": rss_ratio,
        "verdict": verdict,
    }


def fmt_ci(result: tuple[float, float, float]) -> str:
    return f"{result[0]:.3f} [{result[1]:.3f}, {result[2]:.3f}]"


def arm_verdicts(
    comparisons: dict[tuple[str, str, str], dict[str, object]], cell: str
) -> dict[str, str]:
    m = comparisons[(cell, "M", "A")]["verdict"]
    o_results = {
        comparisons[(cell, "O", "A")]["verdict"],
        comparisons[(cell, "O", "M")]["verdict"],
    }
    if "FIX" in o_results:
        o = "FIX"
    elif o_results == {"PASS"}:
        o = "PASS"
    else:
        o = "UNRESOLVED"
    return {"A": "PASS", "M": m, "O": o}


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = load(root / "raw_rows.csv")
    identity = validate(rows)
    stats = {
        (cell, arm): arm_stats(rows, cell, arm)
        for cell in CELLS
        for arm in ("A", "M", "O")
    }
    comparisons = {
        (cell, candidate, base): comparison(rows, cell, candidate, base)
        for cell in CELLS
        for candidate, base in COMPARISONS
    }
    verdicts = {cell: arm_verdicts(comparisons, cell) for cell in CELLS}
    incomplete = [
        f"{cell} {stats[(cell, 'A')]['n']}/{EXPECTED_N[cell]}"
        for cell in CELLS
        if stats[(cell, "A")]["n"] < EXPECTED_N[cell]
    ]
    completion = (
        "Every cell reached target n."
        if not incomplete
        else "The four-hour cutoff left incomplete cells: " + ", ".join(incomplete) + "."
    )
    with (root / "in_sample_rows.csv").open(newline="") as handle:
        in_sample = list(csv.DictReader(handle))

    lines = [
        "# PR 168 walklen3 out-of-sample gate — shard A",
        "",
        "## Measurement",
        "",
        "- A: revision `7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec`, tree "
        "`e5ebda63f9f0117226d50c13abf33243ccf452cc`.",
        "- M: candidate `4ab2681ae6bf72d61c633197e88047f00cdd7efa`, merged tree "
        "`991d5afc6d7b7f7e6c392cb68aa351af1fc50179`.",
        "- O: candidate `fd5449766b4f1f693de2245933afa609a93a5c6b`, original tree "
        "`491819895e46a8fccbfc2449c18868780d1a090b`, merged tree "
        "`7f2625597a167d3c7841b2266eec844997e53266`. The bundle SHA-256 was "
        "`c20089cff9ebd73d9237ca6c4cf7a270e99404863addd9a6555d5f7cb80092e3`; "
        "its prerequisite check passed and it contained three commits after `52af8e47`.",
        "- Both merge-tree operations exited 0. M and O were built from scratch merge "
        "commits outside the delivered history.",
        "- Runs were serial on one core. Order rotated AMO, OMA, MAO, OAM, MOA, AOM by seed.",
        "- Every arm used its own `h2h_seed`, recommended configuration, and the same "
        "release build flags. LJ used 4,000,000 calls and TIP4P used 2,000,000; stops "
        "were reference + 0.001.",
        "- The same RSS-only in-process `RssAnon` reader sampled every 4,096 potential "
        "calls and at process end in all arms. No GMIN executable or data were used.",
        f"- {completion}",
        "- Bootstrap intervals use 10,000 paired resamples and seed 20261123. Calls misses "
        "are budget + 1. Fisher and MWU tests are two-sided.",
        "",
        "## Per-arm results",
        "",
        "| cell | arm | hits/n (Wilson 95% CI) | median calls | median outside s | median RssAnon KiB | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for cell in CELLS:
        for arm in ("A", "M", "O"):
            result = stats[(cell, arm)]
            low, high = result["wilson"]
            lines.append(
                f"| {cell} | {arm} | {result['hits']}/{result['n']} "
                f"[{low:.3f}, {high:.3f}] | {result['calls']:.1f} | "
                f"{result['outside']:.6f} | {result['rss']:.1f} | "
                f"**{verdicts[cell][arm]}** |"
            )
    lines.extend(
        [
            "",
            "A is the reference arm and is labelled PASS. M is judged against A. O must "
            "clear both O/A and O/M to PASS; a FIX in either comparison makes O FIX.",
            "",
            "## Pairwise results",
            "",
            "| cell | comparison | hits candidate/base | paired hit losses | Fisher p | MWU U, p | median calls ratio (95% CI) | median outside ratio (95% CI) | median RSS ratio (95% CI) | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cell in CELLS:
        for candidate, base in COMPARISONS:
            result = comparisons[(cell, candidate, base)]
            lines.append(
                f"| {cell} | {candidate}/{base} | {result['hits'][0]}/{result['hits'][1]} | "
                f"{result['paired_losses']} | {result['fisher']:.4g} | "
                f"{result['mwu_u']:.1f}, {result['mwu_p']:.4g} | "
                f"{fmt_ci(result['calls'])} | {fmt_ci(result['outside'])} | "
                f"{fmt_ci(result['rss'])} | **{result['verdict']}** |"
            )
    lines.extend(
        [
            "",
            "The comparison bar is no lower hit count, median calls ratio at most 1.00, "
            "and upper confidence bounds at most 1.10 for outside time and RSS. A directly "
            "observed hit-count or median-calls loss is FIX; a lower confidence bound above "
            "a resource bar is FIX; other unmet bars are UNRESOLVED. Verdicts are the direct "
            "first assignment.",
            "",
            "## TIP4P identity",
            "",
        ]
    )
    for cell in ("tip4p15", "tip4p20"):
        state = "bit-identical" if identity[cell] else "not bit-identical"
        lines.append(f"- {cell}: **{state}** across A, M and O for every measured seed.")

    lj98_calls_m = values(rows, "lj98", "M", "calls_to_hit")
    lj98_calls_a = values(rows, "lj98", "A", "calls_to_hit")
    large_geo = [
        geomean_ratio(
            values(rows, cell, "M", "calls_to_hit"),
            values(rows, cell, "A", "calls_to_hit"),
        )
        for cell in ("lj102", "lj103", "lj104", "lj150")
    ]
    walk_ratio = geomean_ratio(
        values(rows, "lj150", "M", "calls_per_walk"),
        values(rows, "lj150", "A", "calls_per_walk"),
    )
    lines.extend(
        [
            "",
            "## In-sample record",
            "",
            f"`in_sample_rows.csv` retains {len(in_sample)} prior A-vs-M rows on seeds "
            "2000–2011 with `sample_scope=in-sample-a-vs-m`. They are excluded from every "
            "out-of-sample statistic and verdict above.",
            "",
            "## Builder claims for M",
            "",
            f"- LJ98 geometric 0.972 and median 1.102: measured "
            f"{geomean_ratio(lj98_calls_m, lj98_calls_a):.3f} and "
            f"{median_ratio(lj98_calls_m, lj98_calls_a):.3f}.",
            f"- LJ102–LJ150 geometric 0.027–0.080: measured "
            f"{min(large_geo):.3f}–{max(large_geo):.3f}.",
            f"- Calls per walk up to 4.5× on LJ150: measured {walk_ratio:.3f}×.",
            "",
            "Builder family: `gpt`.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines))
    counts = Counter(
        verdicts[cell][arm] for cell in CELLS for arm in ("A", "M", "O")
    )
    print(" ".join(f"{key}={counts[key]}" for key in sorted(counts)))


if __name__ == "__main__":
    main()
