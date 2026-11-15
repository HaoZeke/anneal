#!/usr/bin/env python3
# Copyright (c) anneal developers
#
# Seed rows for the descent-lemma screen. One process, one core, the host
# name the caller already set. Appends, so a second revision lands in the
# same file. Every row carries the code revision and the host.
#
#   rows.py --examples DIR --revision REV --host HOST --out rows.csv
#
# A second run with the same --out appends. `--summarize rows.csv` prints
# hits, Hyndman-Fan type-7 medians, and the Fisher and Mann-Whitney
# comparisons between the two revisions in the file.

import argparse
import csv
import math
import os
import socket
import subprocess
import sys

SYSTEMS = (
    ("m80r3", "h2h_morse", ("run", "m80r3")),
    ("m80r6", "h2h_morse", ("run", "m80r6")),
    ("lj38", "h2h_gmin_proof", ("run", "lj38")),
    ("lj75", "h2h_gmin_proof", ("run", "lj75")),
    ("lj98", "h2h_gmin_proof", ("run", "lj98")),
    ("au55", "h2h_metal", ("run", "au55")),
)

FIELDS = (
    "system",
    "seed",
    "code_revision",
    "host",
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
    if len(lines) < 2:
        raise RuntimeError("no csv row in:\n" + text)
    header = lines[-2].split(",")
    values = lines[-1].split(",")
    row = dict(zip(header, values))
    hit = row["hit"].strip().lower() in ("1", "true", "yes")
    calls = int(row["evals_to_hit"])
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    if "outside_s" in row:
        outside = float(row["outside_s"])
    else:
        outside = max(0.0, wall - pot)
    return hit, calls, wall, pot, outside


def run_one(binary, args, seed):
    command = [binary, *args, str(seed)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
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
    """Hyndman-Fan type 7 quantile. The median is p = 1/2."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise RuntimeError("empty sample")
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
    """Two-sided Fisher exact on hit counts, same seeds per arm."""
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
    """Two-sided Mann-Whitney on the call counts, normal approximation
    with the tie correction. Returns (U, p). U is the left-arm statistic."""
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
    # Two-sided tail of the standard normal, Abramowitz-Stegun 7.1.26.
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.319381530
        + t
        * (
            -0.356563782
            + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
        )
    )
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    one_tail = phi * poly
    return u, min(1.0, 2.0 * one_tail)


def summarize(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    revisions = []
    for row in rows:
        if row["code_revision"] not in revisions:
            revisions.append(row["code_revision"])
    systems = []
    for row in rows:
        if row["system"] not in systems:
            systems.append(row["system"])
    print(
        "system revision host hits n calls_median wall_median pot_median outside_median"
    )
    grouped = {}
    for system in systems:
        for revision in revisions:
            chosen = [
                row
                for row in rows
                if row["system"] == system and row["code_revision"] == revision
            ]
            if not chosen:
                continue
            hits = sum(1 for row in chosen if row["hit"] in ("1", "true", "True"))
            calls = [float(row["calls"]) for row in chosen]
            wall = [float(row["wall_s"]) for row in chosen]
            pot = [float(row["pot_s"]) for row in chosen]
            outside = [float(row["outside_s"]) for row in chosen]
            host = chosen[0]["host"]
            grouped[(system, revision)] = (hits, len(chosen), calls)
            print(
                f"{system} {revision} {host} {hits} {len(chosen)} "
                f"{type7(calls, 0.5):.1f} {type7(wall, 0.5):.3f} "
                f"{type7(pot, 0.5):.3f} {type7(outside, 0.5):.3f}"
            )
    if len(revisions) == 2:
        base, other = revisions
        print("system fisher_p mann_whitney_p")
        for system in systems:
            if (system, base) not in grouped or (system, other) not in grouped:
                continue
            hits_a, n_a, calls_a = grouped[(system, base)]
            hits_b, n_b, calls_b = grouped[(system, other)]
            fisher = fisher_two_sided(hits_a, n_a, hits_b, n_b)
            _, mw = mann_whitney(calls_a, calls_b)
            print(f"{system} {fisher:.4f} {mw:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", help="directory of h2h_morse, h2h_metal, h2h_gmin_proof")
    parser.add_argument("--revision", help="code revision recorded on every row")
    parser.add_argument("--host", default="", help="host name recorded on every row")
    parser.add_argument("--out", help="CSV to append")
    parser.add_argument("--seeds", default="2000-2011")
    parser.add_argument("--summarize", help="CSV to summarize instead of running")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if not args.examples or not args.revision or not args.out:
        parser.error("--examples, --revision and --out are required")
    host = args.host or socket.gethostname()
    written = []
    for seed in seeds_of(args.seeds):
        for system, binary, prefix in SYSTEMS:
            path = os.path.join(args.examples, binary)
            hit, calls, wall, pot, outside = run_one(path, prefix, seed)
            row = {
                "system": system,
                "seed": str(seed),
                "code_revision": args.revision,
                "host": host,
                "hit": "1" if hit else "0",
                "calls": str(calls),
                "wall_s": f"{wall:.6f}",
                "pot_s": f"{pot:.6f}",
                "outside_s": f"{outside:.6f}",
            }
            written.append(row)
            print(
                f"{system} {args.revision} seed {seed} hit {row['hit']} "
                f"calls {calls} wall {wall:.3f} pot {pot:.3f} outside {outside:.3f}",
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
