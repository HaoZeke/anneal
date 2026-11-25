#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Matched-budget (H2O)8 anneal rerun against the recorded CREST rows.

The search is ``examples/gfn2_mol_h2h`` at ``Config::recommended_molecular``.
The budget is the largest recorded CREST call count. The ledger stops at
the CREST reference plus the CREST tolerance. Seeds run one at a time on
one core. A finished seed is not run again.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import time
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
REPO = HERE.parents[2]
CREST_TABLE = BENCH / "h2o-octamer" / "results.csv"
REFERENCE_EH = -40.681336013150
TOL_EH = 1.0e-6
# rgpot XTBPot and anneal's molecular catalog use this hartree.
HARTREE_EV = 27.211386245988
REFERENCE_EV = REFERENCE_EH * HARTREE_EV
TOL_EV = TOL_EH * HARTREE_EV
THRESHOLD_EV = REFERENCE_EV + TOL_EV
SEEDS = tuple(range(2000, 2012))
BINARY_NAME = "gfn2_mol_h2h"
FEATURES = "rgpot-ex"
BOOT_DRAWS = 10000
BOOT_SEED = 20261123
Z95 = NormalDist().inv_cdf(0.975)

ROW_FIELDS = [
    "arm",
    "seed",
    "budget",
    "hit",
    "calls_to_hit",
    "stop",
    "best_Eh",
    "best_eV",
    "wall_s",
    "crest_calls",
    "charged",
    "intact",
    "binary",
    "features",
    "revision",
]


def energy_hits(energy: float, target: float, tol: float) -> bool:
    if not (math.isfinite(energy) and math.isfinite(target) and math.isfinite(tol)):
        return False
    return tol >= 0.0 and energy <= target + tol


def read_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def load_crest(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="") as handle:
        for item in csv.DictReader(handle):
            if item["arm"] != "crest":
                continue
            rows.append(item)
    if len(rows) != 6:
        raise SystemExit(f"expected 6 CREST rows in {path.name}, found {len(rows)}")
    return rows


def wilson(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    p = hits / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def fisher_exact(hits_first: int, n_first: int, hits_second: int, n_second: int) -> float:
    hits = hits_first + hits_second
    total = n_first + n_second

    def probability(value: int) -> float:
        return math.comb(n_first, value) * math.comb(n_second, hits - value) / math.comb(total, hits)

    observed = probability(hits_first)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(max(0, hits - n_second), min(n_first, hits) + 1)
            if probability(value) <= observed * (1.0 + 1.0e-12)
        ),
    )


def _midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j + 2) / 2
        i = j + 1
    return ranks


def mann_whitney_exact(left: list[float], right: list[float]) -> float:
    """Two-sided exact p of the rank sum, midranks, all labelings.

    A labeling counts when its rank sum is at least as far from the null
    mean as the observed one.
    """
    if not left or not right:
        return float("nan")
    values = list(left) + list(right)
    ranks = _midranks(values)
    n1 = len(left)
    n = len(values)
    observed = sum(ranks[:n1])
    mean = n1 * (n + 1) / 2.0
    seen = abs(observed - mean)
    if seen == 0.0 and len(set(values)) == 1:
        return 1.0
    extreme = 0
    total = 0
    for choice in combinations(range(n), n1):
        total += 1
        rank_sum = sum(ranks[i] for i in choice)
        if abs(rank_sum - mean) + 1.0e-9 >= seen:
            extreme += 1
    return extreme / total


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def bootstrap_median_ci(values: list[float], draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(draws):
        samples.append(median([values[rng.randrange(n)] for _ in range(n)]))
    samples.sort()
    return samples[int(0.025 * draws)], samples[min(draws - 1, int(0.975 * draws))]


def self_test() -> None:
    if abs(NormalDist().cdf(Z95) - 0.975) > 1.0e-12:
        raise SystemExit("z")
    if fisher_exact(12, 12, 0, 12) != math.comb(24, 12) ** -1 * 2:
        raise SystemExit("fisher")
    # Three low against three high: the two extreme assignments out of 20.
    if abs(mann_whitney_exact([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) - 0.1) > 1.0e-12:
        raise SystemExit("mann-whitney")
    if mann_whitney_exact([1.0, 1.0], [1.0, 1.0]) != 1.0:
        raise SystemExit("mann-whitney ties")
    low, high = wilson(0, 12)
    if low != 0.0 or not (0.2 < high < 0.3):
        raise SystemExit(f"wilson {low} {high}")
    if abs((REFERENCE_EH + TOL_EH) * HARTREE_EV - THRESHOLD_EV) > 0.0:
        raise SystemExit("threshold")


def git_revision() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def finished(run_txt: Path) -> bool:
    if not run_txt.is_file():
        return False
    fields = read_fields(run_txt)
    return "wall_s" in fields and "stop" in fields and "objective_best_eV" in fields


def run_seed(binary: Path, seed: int, budget: int, out: Path, core: str, revision: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["ANNEAL_SOURCE_COMMIT"] = revision
    env["ANNEAL_SOURCE_TREE"] = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    command = [
        "taskset",
        "-c",
        core,
        str(binary),
        "h2o-8",
        str(budget),
        str(seed),
        str(out),
        f"{THRESHOLD_EV:.10f}",
    ]
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        started = time.time()
        proc = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
        elapsed = time.time() - started
    if proc.returncode != 0 or not finished(out / "run.txt"):
        raise SystemExit(f"seed {seed} exited {proc.returncode} after {elapsed:.1f}s")


def anneal_row(seed: int, budget: int, out: Path, revision: str) -> dict[str, str]:
    fields = read_fields(out / "run.txt")
    best_eV = float(fields["objective_best_eV"])
    intact = fields.get("best_intact") == "true"
    first = fields.get("first_hit_call", "none")
    stop = fields.get("stop", "missing")
    crossed = first != "none" and energy_hits(best_eV, REFERENCE_EV, TOL_EV)
    hit = intact and (stop == "target" or crossed)
    if hit:
        calls = int(first) if first != "none" else int(fields["energy_calls"])
    else:
        calls = budget + 1
    return {
        "arm": "anneal",
        "seed": str(seed),
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "calls_to_hit": str(calls),
        "stop": stop,
        "best_Eh": f"{best_eV / HARTREE_EV:.12f}",
        "best_eV": f"{best_eV:.10f}",
        "wall_s": f"{float(fields['wall_s']):.6f}",
        "crest_calls": "",
        "charged": fields.get("energy_calls", ""),
        "intact": "intact" if intact else "broken",
        "binary": BINARY_NAME,
        "features": FEATURES,
        "revision": revision,
    }


def crest_row(item: dict[str, str], budget: int) -> dict[str, str]:
    calls = int(item["calls"])
    hit = item["hit"] == "yes"
    # A recorded CREST hit keeps its own call count. A miss would be budget + 1.
    calls_to_hit = calls if hit else budget + 1
    return {
        "arm": "crest",
        "seed": item["seed"],
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "calls_to_hit": str(calls_to_hit),
        "stop": "recorded",
        "best_Eh": item["energy_Eh"],
        "best_eV": f"{float(item['energy_Eh']) * HARTREE_EV:.10f}",
        "wall_s": item["wall_s"],
        "crest_calls": str(calls),
        "charged": str(calls),
        "intact": item["intact"],
        "binary": "crest",
        "features": "",
        "revision": "",
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def calls_of(rows: list[dict[str, str]]) -> list[float]:
    return [float(row["calls_to_hit"]) for row in rows]


def write_report(
    path: Path,
    budget: int,
    crest: list[dict[str, str]],
    anneal: list[dict[str, str]],
    revision: str,
) -> None:
    crest_calls = [int(row["crest_calls"]) for row in crest]
    anneal_hits = sum(row["hit"] == "1" for row in anneal)
    crest_hits = sum(row["hit"] == "1" for row in crest)
    n_a, n_c = len(anneal), len(crest)
    low, high = wilson(anneal_hits, n_a)
    anneal_calls = calls_of(anneal)
    crest_call_values = calls_of(crest)
    med = median(anneal_calls)
    blo, bhi = bootstrap_median_ci(anneal_calls)
    fisher = fisher_exact(anneal_hits, n_a, crest_hits, n_c)
    mann = mann_whitney_exact(anneal_calls, crest_call_values)
    lines = [
        "# (H2O)8 matched budget",
        "",
        "Anneal was rerun at one shared budget, the largest recorded CREST",
        f"call count, {budget}. The recorded CREST rows were not rerun.",
        f"Source revision {revision}.",
        "",
        "## Recorded CREST call counts",
        "",
        "| seed | calls | energy / Eh | hit |",
        "| --- | --- | --- | --- |",
    ]
    for row in sorted(crest, key=lambda item: int(item["seed"])):
        lines.append(
            f"| {row['seed']} | {row['crest_calls']} | {row['best_Eh']} | {row['hit']} |"
        )
    lines += [
        "",
        f"Median of the six CREST call counts: {median([float(v) for v in crest_calls]):.1f}.",
        f"Maximum, used as the anneal budget: {budget}.",
        "",
        "## Anneal seeds 2000-2011",
        "",
        "| seed | hit | calls_to_hit | stop | best / Eh | wall / s | crest_calls |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in anneal:
        lines.append(
            f"| {row['seed']} | {row['hit']} | {row['calls_to_hit']} | {row['stop']} "
            f"| {row['best_Eh']} | {row['wall_s']} | {row['crest_calls'] or 'unpaired'} |"
        )
    lines += [
        "",
        "## Comparison",
        "",
        f"Anneal hits {anneal_hits}/{n_a}. Wilson 95% interval "
        f"[{low:.4f}, {high:.4f}].",
        f"Median calls to the hit, with a miss stored as {budget + 1}: {med:.1f}.",
        f"Bootstrap 95% interval of that median, {BOOT_DRAWS} resamples, "
        f"seed {BOOT_SEED}: [{blo:.1f}, {bhi:.1f}].",
        f"CREST hits {crest_hits}/{n_c}.",
        f"Fisher exact two-sided p on the hit counts: {fisher:.6g}.",
        f"Mann-Whitney exact two-sided p on calls to the hit: {mann:.6g}.",
        "",
        "The anneal seeds and the CREST seeds are different samples.",
        "The tests are unpaired. Each CREST row keeps that seed's own call count.",
        "",
    ]
    path.write_text("\n".join(lines))


def campaign(binary: Path, core: str, rows_path: Path, report_path: Path) -> None:
    self_test()
    crest_src = load_crest(CREST_TABLE)
    budget = max(int(item["calls"]) for item in crest_src)
    revision = git_revision()
    crest = [crest_row(item, budget) for item in crest_src]
    anneal: list[dict[str, str]] = []
    started = time.time()
    for index, seed in enumerate(SEEDS, start=1):
        out = HERE / "runs" / str(seed)
        if not finished(out / "run.txt"):
            print(f"SEED {seed} start budget {budget}", flush=True)
            run_seed(binary, seed, budget, out, core, revision)
        row = anneal_row(seed, budget, out, revision)
        anneal.append(row)
        write_rows(rows_path, crest + anneal)
        elapsed = time.time() - started
        print(
            f"SEED {seed} hit={row['hit']} calls={row['calls_to_hit']} "
            f"stop={row['stop']} wall={row['wall_s']} elapsed={elapsed:.1f}",
            flush=True,
        )
        if index < len(SEEDS):
            per = elapsed / index
            print(f"PROJECTION remaining {len(SEEDS) - index} seeds at {per:.1f}s each", flush=True)
    write_report(report_path, budget, crest, anneal, revision)
    print(f"WROTE {rows_path.name} {report_path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--core", default=os.environ.get("H2O8_CORE", "0"))
    parser.add_argument("--rows", type=Path, default=HERE / "rows.csv")
    parser.add_argument("--report", type=Path, default=HERE / "REPORT.md")
    args = parser.parse_args()
    campaign(args.binary, args.core, args.rows, args.report)


if __name__ == "__main__":
    main()
