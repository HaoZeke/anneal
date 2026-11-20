# Copyright (c) anneal developers
"""Seed-paired comparison of one arm against a base arm of ``rows.csv``.

Both arms run each seed at the same budget, so a seed is a pair. Per
molecule the table gives:

- hits of each arm, the seeds only the arm hit and the seeds only the base
  hit, and the exact two-sided McNemar test on those discordant seeds
  (the binomial test of the split at one half; 1 when none is discordant)
- the stereo-check failures (``invalid``) of each arm summed over seeds
- the median per-call time outside the potential, ``outside_s`` over
  ``calls`` in microseconds, of each arm, the median over seeds of the
  arm-to-base ratio, and the two-sided Wilcoxon signed-rank test of the
  paired per-call times, exact with equal pairs dropped (see ``wilcoxon_p``;
  1 when every pair is equal)
- the local step each arm resolved to, as ``step:count`` pairs

``-`` marks a value that does not exist, such as a ratio with no base time.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest, wilcoxon

FIELDS = [
    "molecule", "arm", "base", "n", "hits_arm", "hits_base", "only_arm", "only_base", "mcnemar_p",
    "invalid_arm", "invalid_base", "outside_us_per_call_arm", "outside_us_per_call_base",
    "outside_ratio_median", "wilcoxon_p", "local_step_arm", "local_step_base",
]


def mcnemar_exact(only_arm: int, only_base: int) -> float:
    """Two-sided exact McNemar p from the two discordant counts."""
    discordant = only_arm + only_base
    if discordant == 0:
        return 1.0
    return float(binomtest(only_arm, discordant, 0.5).pvalue)


def per_call_us(row: dict) -> float | None:
    calls = float(row["calls"]) if row.get("calls") else 0.0
    outside = row.get("outside_s")
    if calls <= 0 or outside in (None, ""):
        return None
    return 1e6 * float(outside) / calls


EXACT_WILCOXON_LIMIT = 50


def wilcoxon_p(arm: list[float], base: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p of the paired values ``arm`` and ``base``.

    Pairs whose values are equal carry no sign and are dropped before the test
    (the ``wilcox`` zero method), so the result does not depend on which zero
    method a scipy release defaults to. The p is the exact one, from the
    distribution of the signed-rank statistic over all sign assignments, while
    at most ``EXACT_WILCOXON_LIMIT`` pairs remain; with more it is the normal
    approximation. scipy's ``auto`` would switch to the normal approximation
    as soon as one pair was equal, which at 24 seeds shifts the p by a few
    per cent for no reason in the data. The exact distribution takes the
    differences as untied; equal absolute differences are rare in measured
    times and are ranked by their mean rank. 1 when no pair is left.
    """
    if not arm:
        return None
    diffs = [a - b for a, b in zip(arm, base, strict=True) if a != b]
    if not diffs:
        return 1.0
    method = "exact" if len(diffs) <= EXACT_WILCOXON_LIMIT else "approx"
    return float(wilcoxon(diffs, zero_method="wilcox", method=method).pvalue)


def steps(rows: list[dict], local_steps: dict) -> str:
    count = Counter(local_steps.get((r["molecule"], r["arm"], r["seed"]), "?") for r in rows)
    return " ".join(f"{k}:{v}" for k, v in sorted(count.items()))


def compare(rows: list[dict], arm: str, base: str, local_steps: dict | None = None) -> list[dict]:
    local_steps = local_steps or {}
    molecules: list[str] = []
    for r in rows:
        if r["molecule"] not in molecules:
            molecules.append(r["molecule"])
    table = []
    for name in molecules:
        mine = {r["seed"]: r for r in rows if r["molecule"] == name and r["arm"] == arm}
        theirs = {r["seed"]: r for r in rows if r["molecule"] == name and r["arm"] == base}
        seeds = sorted(set(mine) & set(theirs), key=int)
        a = [mine[s] for s in seeds]
        b = [theirs[s] for s in seeds]
        hit = lambda r: int(r["hit"])  # noqa: E731
        only_arm = sum(1 for x, y in zip(a, b) if hit(x) and not hit(y))
        only_base = sum(1 for x, y in zip(a, b) if hit(y) and not hit(x))
        pairs = [(per_call_us(x), per_call_us(y)) for x, y in zip(a, b)]
        pairs = [(p, q) for p, q in pairs if p is not None and q is not None]
        ratios = [p / q for p, q in pairs if q > 0]
        times_a = [p for p, _ in pairs]
        times_b = [q for _, q in pairs]
        p_w = wilcoxon_p(times_a, times_b)
        table.append(
            {
                "molecule": name,
                "arm": arm,
                "base": base,
                "n": len(seeds),
                "hits_arm": sum(hit(x) for x in a),
                "hits_base": sum(hit(y) for y in b),
                "only_arm": only_arm,
                "only_base": only_base,
                "mcnemar_p": f"{mcnemar_exact(only_arm, only_base):.4f}",
                "invalid_arm": sum(int(x["invalid"]) for x in a),
                "invalid_base": sum(int(y["invalid"]) for y in b),
                "outside_us_per_call_arm": f"{statistics.median(times_a):.2f}" if times_a else "-",
                "outside_us_per_call_base": f"{statistics.median(times_b):.2f}" if times_b else "-",
                "outside_ratio_median": f"{statistics.median(ratios):.4f}" if ratios else "-",
                "wilcoxon_p": f"{p_w:.4f}" if p_w is not None else "-",
                "local_step_arm": steps(a, local_steps),
                "local_step_base": steps(b, local_steps),
            }
        )
    return table


def read_local_steps(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        return {(r["molecule"], r["arm"], r["seed"]): r["local_step"] for r in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--arm", default="anneal")
    parser.add_argument("--base", default="develop")
    parser.add_argument("--local-steps", type=Path, help="csv with molecule, arm, seed and local_step")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.rows.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    table = compare(rows, args.arm, args.base, read_local_steps(args.local_steps))
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
