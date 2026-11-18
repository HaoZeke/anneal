#!/usr/bin/env python3
# Copyright (c) anneal developers
#
# Serial gate for the detected-symmetry series: one process at a time, the
# base build and the new build interleaved per system and seed, in one
# session. Rows are appended as they finish, each with the code revision and
# the host.
#
#   gate.py run --out rows.csv \
#       --base-bin DIR --base-tree DIR --new-bin DIR --new-tree DIR
#   gate.py summarize rows.csv
#
# `DIR` of a build is where `h2h_gmin_proof` and `h2h_morse` are. The tree is
# the working directory the build is run from, which is where the example
# reads its revision from.

import argparse
import csv
import io
import math
import os
import socket
import subprocess
import sys

LJ = ("lj38", "lj55", "lj75", "lj98", "lj103", "lj104")
MORSE = ("m38r10", "m80r6")
SYSTEMS = LJ + MORSE
FIELDS = (
    "system",
    "build",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "code_revision",
    "tree",
    "host",
)


def seeds_of(text):
    if "-" in text:
        start, stop = text.split("-", 1)
        return range(int(start), int(stop) + 1)
    return [int(part) for part in text.split(",") if part]


def parse_run(text):
    lines = [line for line in text.splitlines() if line and not line.startswith("arm ")]
    header, values = lines[-2], lines[-1]
    return next(csv.DictReader(io.StringIO(header + "\n" + values)))


def run_one(bin_dir, tree, system, seed):
    example = "h2h_gmin_proof" if system in LJ else "h2h_morse"
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[name] = "1"
    for name in ("H2H_BUDGET", "H2H_PRIORS", "H2H_ABL"):
        env.pop(name, None)
    done = subprocess.run(
        [os.path.join(bin_dir, example), "run", system, str(seed)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tree,
        env=env,
    )
    return parse_run(done.stdout)


def append_row(path, row):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run(args):
    host = args.host or socket.gethostname()
    builds = (("base", args.base_bin, args.base_tree), ("new", args.new_bin, args.new_tree))
    systems = args.systems.split(",") if args.systems else SYSTEMS
    for system in systems:
        for seed in seeds_of(args.seeds):
            for label, bin_dir, tree in builds:
                got = run_one(bin_dir, tree, system, seed)
                head = got.get("head") or got.get("code_revision") or ""
                row = {
                    "system": system,
                    "build": label,
                    "seed": str(seed),
                    "budget": got["budget"],
                    "hit": "1" if got["hit"].strip().lower() in ("1", "true") else "0",
                    "evals_to_hit": got["evals_to_hit"],
                    "charged": got["charged"],
                    "wall_s": got["wall_s"],
                    "best": got["best"],
                    "pot_s": got.get("pot_s", ""),
                    "code_revision": head,
                    "tree": got.get("tree", ""),
                    "host": host,
                }
                append_row(args.out, row)
                print(
                    f"{system} {label} {head[:12]} seed {seed} hit {row['hit']} "
                    f"evals {row['evals_to_hit']} wall {float(row['wall_s']):.1f}",
                    flush=True,
                )


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
    a, b, c, d = hits_a, n_a - hits_a, hits_b, n_b - hits_b
    row1, row2, col1 = a + b, c + d, a + c
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
        bb, cc = row1 - aa, col1 - aa
        dd = row2 - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        mass = probability(aa, bb, cc, dd)
        if mass <= observed * (1.0 + 1e-9):
            total += mass
    return min(1.0, total)


def mann_whitney(left, right):
    n1, n2 = len(left), len(right)
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
    return u, min(1.0, math.erfc(z / math.sqrt(2.0)))


def summarize(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    revisions = {}
    for row in rows:
        revisions.setdefault(row["build"], set()).add(row["code_revision"])
    for build, found in sorted(revisions.items()):
        print(f"# {build}: {' '.join(sorted(found))}")
    print(
        "system budget base_hits new_hits n base_median new_median "
        "fisher_p mann_whitney_p verdict"
    )
    seen = []
    for row in rows:
        if row["system"] not in seen:
            seen.append(row["system"])
    losses = 0
    for system in seen:
        base = [r for r in rows if r["system"] == system and r["build"] == "base"]
        new = [r for r in rows if r["system"] == system and r["build"] == "new"]
        if not base or not new:
            continue
        hits_base = sum(r["hit"] == "1" for r in base)
        hits_new = sum(r["hit"] == "1" for r in new)
        calls_base = [float(r["evals_to_hit"]) for r in base]
        calls_new = [float(r["evals_to_hit"]) for r in new]
        fisher = fisher_two_sided(hits_base, len(base), hits_new, len(new))
        _, mw = mann_whitney(calls_base, calls_new)
        median_base, median_new = type7(calls_base, 0.5), type7(calls_new, 0.5)
        worse_hits = hits_new < hits_base and fisher < 0.05
        worse_calls = median_new > median_base and mw < 0.05
        verdict = "LOSS" if worse_hits or worse_calls else "ok"
        losses += verdict == "LOSS"
        print(
            f"{system} {base[0]['budget']} {hits_base}/{len(base)} {hits_new}/{len(new)} "
            f"{len(new)} {median_base:.1f} {median_new:.1f} {fisher:.4g} {mw:.4g} {verdict}"
        )
    print(f"losses {losses}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--out", required=True)
    runner.add_argument("--base-bin", required=True)
    runner.add_argument("--base-tree", required=True)
    runner.add_argument("--new-bin", required=True)
    runner.add_argument("--new-tree", required=True)
    runner.add_argument("--seeds", default="2000-2011")
    runner.add_argument("--systems", default="")
    runner.add_argument("--host", default="")
    summary = sub.add_parser("summarize")
    summary.add_argument("path")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args.path)


if __name__ == "__main__":
    sys.exit(main())
