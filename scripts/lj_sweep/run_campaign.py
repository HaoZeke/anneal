#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Anneal arm of the twelve-seed Lennard-Jones panel.

GMIN rows for LJ38, 55, 75, 76, 77, 98, 102, 103, 104 and 150 are the
summary committed in ``bench/lj-n12-500aaf65``. That binary did not
change, so those rows are copied and not rerun. Anneal is
``Config::recommended`` on this tree. Seeds are 2000 through 2011, one
process at a time. A miss is the paper budget. Fisher is the two-sided
exact test on the hit counts. This process refuses to run unless the
UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

SIZES = (38, 55, 75, 150, 76, 77, 98, 102, 103, 104)
SEEDS = tuple(range(2000, 2012))
BUDGET = {
    38: 400_000,
    55: 1_000_000,
    75: 4_000_000,
    76: 4_000_000,
    77: 4_000_000,
    98: 4_000_000,
    102: 4_000_000,
    103: 4_000_000,
    104: 4_000_000,
    150: 4_000_000,
}
GMIN_REVISION = "aac1548a8aec13b53ad12a8d6904625081685d2e"
PANEL = "bench/lj-n12-500aaf65/summary.csv"

SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "hits",
    "hits_over_n",
    "budget",
    "evals_median",
    "evals_iqr",
    "wall_median_s",
    "inside_us",
    "outside_us",
    "fisher_p",
    "mannwhitney_p",
    "code_revision",
    "anneal_head",
    "seeds",
]


def type7(ordered: list[float], p: float) -> float:
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0], 0.0
    return type7(ordered, 0.5), type7(ordered, 0.75) - type7(ordered, 0.25)


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def fisher_exact_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b

    def probability(aa: int, bb: int, cc: int, dd: int) -> float:
        total = aa + bb + cc + dd
        return comb(aa + bb, aa) * comb(cc + dd, cc) / comb(total, aa + cc)

    observed = probability(a, b, c, d)
    total = 0.0
    for aa in range(a + b + 1):
        cc = a + c - aa
        bb = a + b - aa
        dd = c + d - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        mass = probability(aa, bb, cc, dd)
        if mass <= observed + 1e-15:
            total += mass
    return min(1.0, total)


def load_gmin(root: Path) -> dict[int, dict[str, str]]:
    path = root / PANEL
    found: dict[int, dict[str, str]] = {}
    for row in csv.DictReader(path.open()):
        if row["arm"] != "gmin":
            continue
        n = int(row["system"].removeprefix("lj"))
        if row["gmin_revision"] != GMIN_REVISION:
            raise SystemExit(f"GMIN revision changed for LJ{n}")
        if int(row["n"]) != 12 or int(row["budget"]) != BUDGET[n]:
            raise SystemExit(f"GMIN row for LJ{n} is not the twelve-seed paper budget")
        found[n] = row
    missing = [n for n in SIZES if n not in found]
    if missing:
        raise SystemExit(f"no twelve-seed GMIN row for {missing}")
    return found


def parse_anneal(text: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if ",anneal," in line]
    if len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s"
    )
    row = next(csv.DictReader([header, data[0]]))
    if row["hit"] not in ("0", "1", "true", "false"):
        raise ValueError(row["hit"])
    row["hit"] = "1" if row["hit"] in ("1", "true") else "0"
    return row


def run_anneal(binary: Path, n: int, seed: int, log: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(BUDGET[n])
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "run", f"lj{n}", str(seed)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"anneal LJ{n} seed {seed} exited {proc.returncode}")
    return parse_anneal(log.read_text())


def summarise(anneal: list[dict], gmin: dict[int, dict[str, str]], head: str) -> list[dict[str, str]]:
    by: dict[int, list[dict]] = {}
    for row in anneal:
        by.setdefault(int(row["system"].removeprefix("lj")), []).append(row)
    seed_text = " ".join(str(seed) for seed in SEEDS)
    out = []
    for n in SIZES:
        group = by.get(n, [])
        if len(group) != len(SEEDS):
            continue
        evals = [float(row["evals_to_hit"]) for row in group]
        walls = [float(row["wall_s"]) for row in group]
        insides = []
        outsides = []
        for row in group:
            charged = int(row["charged"])
            pot = float(row["pot_s"])
            wall = float(row["wall_s"])
            insides.append(pot / charged * 1e6)
            outsides.append(max(wall - pot, 0.0) / charged * 1e6)
        evals_median, evals_iqr = median_iqr(evals)
        wall_median, _wall_iqr = median_iqr(walls)
        hits = sum(row["hit"] == "1" for row in group)
        gmin_hits = int(gmin[n]["hits"])
        fisher = fisher_exact_two_sided(hits, len(group), gmin_hits, 12)
        out.append(
            {
                "system": f"lj{n}",
                "arm": "anneal",
                "n": str(len(group)),
                "hits": str(hits),
                "hits_over_n": f"{hits}/{len(group)}",
                "budget": str(BUDGET[n]),
                "evals_median": f"{evals_median:.6f}",
                "evals_iqr": f"{evals_iqr:.6f}",
                "wall_median_s": f"{wall_median:.6f}",
                "inside_us": f"{median_iqr(insides)[0]:.4f}",
                "outside_us": f"{median_iqr(outsides)[0]:.4f}",
                "fisher_p": f"{fisher:.6g}",
                "mannwhitney_p": "",
                "code_revision": head,
                "anneal_head": head,
                "seeds": seed_text,
            }
        )
        out.append(
            {
                "system": f"lj{n}",
                "arm": "gmin",
                "n": gmin[n]["n"],
                "hits": gmin[n]["hits"],
                "hits_over_n": gmin[n]["hits_over_n"],
                "budget": gmin[n]["budget"],
                "evals_median": gmin[n]["evals_median"],
                "evals_iqr": gmin[n]["evals_iqr"],
                "wall_median_s": gmin[n]["wall_median_s"],
                "inside_us": "",
                "outside_us": "",
                "fisher_p": f"{fisher:.6g}",
                "mannwhitney_p": "",
                "code_revision": GMIN_REVISION,
                "anneal_head": gmin[n]["anneal_head"],
                "seeds": gmin[n]["seeds"],
            }
        )
    return out


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args(argv[1:])
    sizes = tuple(args.size) if args.size else SIZES
    seeds = tuple(args.seed) if args.seed else SEEDS
    head = args.head_file.read_text().strip()
    gmin = load_gmin(args.root)
    args.runs.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for n in sizes:
        if n not in BUDGET:
            raise SystemExit(f"no paper budget for LJ{n}")
        for seed in seeds:
            saved = args.runs / f"lj{n}" / f"anneal-{seed}.row"
            saved.parent.mkdir(parents=True, exist_ok=True)
            if saved.is_file() and saved.stat().st_size > 0:
                row = parse_anneal(saved.read_text())
                rows.append(row)
                print(f"skip lj{n} seed {seed} hit={row['hit']} evals={row['evals_to_hit']}", flush=True)
                continue
            print(f"start lj{n} seed {seed} budget={BUDGET[n]}", flush=True)
            log = args.runs / f"lj{n}" / f"anneal-{seed}.log"
            row = run_anneal(args.binary, n, seed, log)
            saved.write_text(log.read_text())
            rows.append(row)
            print(
                f"done lj{n} seed {seed} hit={row['hit']} evals={row['evals_to_hit']} best={row['best']}",
                flush=True,
            )
            write_summary(args.summary, summarise(rows, gmin, head))
    write_summary(args.summary, summarise(rows, gmin, head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
