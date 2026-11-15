#!/usr/bin/env python3
# Copyright (c) anneal developers
#
# M80 ρ=6 rows for the structure-prior bisect. One process, one core, the
# host the caller already set. Appends. Each row carries the code revision,
# the host, and whether the structure priors were on.
#
#   rows.py --binary h2h_morse --revision SHA --priors on|off --host bench --out rows.csv
#
# `--summarize rows.csv` prints hits, Hyndman-Fan type-7 medians, and the
# Fisher and Mann-Whitney comparisons of each priors=off arm against the
# priors=on arm of the same revision.

import argparse
import csv
import math
import os
import socket
import subprocess
import sys

FIELDS = (
    "system",
    "seed",
    "code_revision",
    "host",
    "priors",
    "hit",
    "calls",
    "wall_s",
    "pot_s",
    "outside_s",
)


def seeds_of(text):
    if "-" in text:
        start, stop = text.split("-", 1)
        return range(int(start), int(stop) + 1)
    return [int(part) for part in text.split(",") if part]


def parse_row(text):
    lines = [line for line in text.splitlines() if line and not line.startswith("arm ")]
    header = lines[-2].split(",")
    values = lines[-1].split(",")
    row = dict(zip(header, values))
    hit = row["hit"].strip().lower() in ("1", "true", "yes")
    calls = int(row["evals_to_hit"])
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = float(row["outside_s"]) if "outside_s" in row else max(0.0, wall - pot)
    return hit, calls, wall, pot, outside


def run_one(binary, seed, priors):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    if priors == "off":
        env["H2H_PRIORS"] = "0"
    else:
        env.pop("H2H_PRIORS", None)
    completed = subprocess.run(
        [binary, "run", "m80r6", str(seed)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return parse_row(completed.stdout)


def append_rows(path, rows):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def type7(values, p):
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    position = 1.0 + (n - 1) * p
    low = int(math.floor(position))
    high = int(math.ceil(position))
    weight = position - low
    return ordered[low - 1] * (1.0 - weight) + ordered[high - 1] * weight


def log_factorial(n):
    return math.lgamma(n + 1)


def fisher_two_sided(hits_a, n_a, hits_b, n_b):
    a = hits_a
    b = n_a - hits_a
    c = hits_b
    d = n_b - hits_b
    row1 = a + b
    row2 = c + d
    col1 = a + c
    n = row1 + row2

    def probability(aa, bb, cc, dd):
        return math.exp(
            log_factorial(aa + bb)
            + log_factorial(cc + dd)
            + log_factorial(aa + cc)
            + log_factorial(bb + dd)
            - (
                log_factorial(aa)
                + log_factorial(bb)
                + log_factorial(cc)
                + log_factorial(dd)
                + log_factorial(n)
            )
        )

    observed = probability(a, b, c, d)
    total = 0.0
    for aa in range(max(0, col1 - row2), min(row1, col1) + 1):
        bb = row1 - aa
        cc = col1 - aa
        dd = row2 - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        mass = probability(aa, bb, cc, dd)
        if mass <= observed * (1.0 + 1e-9):
            total += mass
    return min(1.0, total)


def mann_whitney(left, right):
    n1 = len(left)
    n2 = len(right)
    pooled = [(value, 0) for value in left] + [(value, 1) for value in right]
    pooled.sort(key=lambda item: item[0])
    ranks = [0.0] * len(pooled)
    i = 0
    tie_term = 0.0
    while i < len(pooled):
        j = i
        while j < len(pooled) and pooled[j][0] == pooled[i][0]:
            j += 1
        average = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[k] = average
        tie = j - i
        if tie > 1:
            tie_term += tie * (tie * tie - 1)
        i = j
    rank_left = sum(rank for rank, (_, arm) in zip(ranks, pooled) if arm == 0)
    u = n1 * n2 + n1 * (n1 + 1) / 2.0 - rank_left
    mean = n1 * n2 / 2.0
    n = n1 + n2
    variance = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if variance <= 0.0:
        return u, 1.0
    z = (abs(u - mean) - 0.5) / math.sqrt(variance)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return u, min(1.0, 2.0 * phi * poly)


def summarize(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    print("revision priors host hits n calls_median wall_median pot_median outside_median")
    groups = {}
    order = []
    for row in rows:
        key = (row["code_revision"], row["priors"])
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(row)
    for revision, priors in order:
        chosen = groups[(revision, priors)]
        hits = sum(1 for row in chosen if row["hit"] in ("1", "true", "True"))
        calls = [float(row["calls"]) for row in chosen]
        wall = [float(row["wall_s"]) for row in chosen]
        pot = [float(row["pot_s"]) for row in chosen]
        outside = [float(row["outside_s"]) for row in chosen]
        print(
            f"{revision[:12]} {priors} {chosen[0]['host']} {hits} {len(chosen)} "
            f"{type7(calls, 0.5):.1f} {type7(wall, 0.5):.3f} "
            f"{type7(pot, 0.5):.3f} {type7(outside, 0.5):.3f}"
        )
    print("revision fisher_p mann_whitney_p")
    revisions = []
    for revision, _priors in order:
        if revision not in revisions:
            revisions.append(revision)
    for revision in revisions:
        if (revision, "on") not in groups or (revision, "off") not in groups:
            continue
        on = groups[(revision, "on")]
        off = groups[(revision, "off")]
        hits_on = sum(1 for row in on if row["hit"] in ("1", "true", "True"))
        hits_off = sum(1 for row in off if row["hit"] in ("1", "true", "True"))
        fisher = fisher_two_sided(hits_on, len(on), hits_off, len(off))
        _, mw = mann_whitney(
            [float(row["calls"]) for row in on],
            [float(row["calls"]) for row in off],
        )
        print(f"{revision[:12]} {fisher:.4f} {mw:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary")
    parser.add_argument("--revision")
    parser.add_argument("--priors", choices=("on", "off"))
    parser.add_argument("--host", default="")
    parser.add_argument("--out")
    parser.add_argument("--seeds", default="2000-2011")
    parser.add_argument("--summarize")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if not args.binary or not args.revision or not args.priors or not args.out:
        parser.error("--binary, --revision, --priors and --out are required")
    host = args.host or socket.gethostname()
    written = []
    for seed in seeds_of(args.seeds):
        hit, calls, wall, pot, outside = run_one(args.binary, seed, args.priors)
        row = {
            "system": "m80r6",
            "seed": str(seed),
            "code_revision": args.revision,
            "host": host,
            "priors": args.priors,
            "hit": "1" if hit else "0",
            "calls": str(calls),
            "wall_s": f"{wall:.6f}",
            "pot_s": f"{pot:.6f}",
            "outside_s": f"{outside:.6f}",
        }
        written.append(row)
        print(
            f"m80r6 {args.revision[:12]} priors={args.priors} seed {seed} "
            f"hit {row['hit']} calls {calls} wall {wall:.3f} pot {pot:.3f}",
            flush=True,
        )
    append_rows(args.out, written)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or "")
        sys.stderr.write(exc.stdout or "")
        raise
