#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One serial session of two revisions on one core.

Even seed offsets run the first binary, then the second. Odd offsets run
the second, then the first. The command is ``run <system> <seed>`` with
the budget in the environment. No quick flag and no warm start.

A hit is a finite best energy at or below the reference plus 0.001.
The median is Hyndman-Fan type 7 of evals_to_hit. Mann-Whitney ranks a
miss worse than every hit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

REFERENCES = {
    "m38r6": -157.477108,
    "m38r14": -144.321054,
    "sc126-38": -35419.9804,
    "sc108-38": -10117.2454,
}
HIT_TOL = DEFAULT_HIT_TOL
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_DOMAIN_NUM_THREADS",
)
CLEAR_KEYS = (
    "H2H_GENERAL",
    "H2H_BASIN_STRIDE",
    "H2H_BEST",
    "H2H_REFERENCE",
    "H2H_PRIORS",
    "H2H_REPLICAS",
    "H2H_TARGET",
    "H2H_TOL",
    "H2H_SEED",
    "H2H_SYSTEM",
    "H2H_ROW",
    "H2H_BUDGET_RESTART",
    "H2H_PAIR_SCALE",
    "H2H_PLATEAU_HEAT",
    "H2H_BRANCH",
    "H2H_QUICK",
    "H2H_WARM",
    "H2H_FINAL",
)
FIELDS = (
    "pair",
    "revision",
    "driver_blob",
    "system",
    "seed",
    "budget",
    "hit",
    "best",
    "evals_to_hit",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "machine_id",
)


def pin(env: dict[str, str], budget: int) -> None:
    for key in CLEAR_KEYS:
        env.pop(key, None)
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    env["H2H_BUDGET"] = str(budget)


def type7(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * 0.5
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    from math import comb

    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b
    if min(a, b, c, d) < 0:
        return float("nan")
    tables = []
    for aa in range(0, a + b + 1):
        bb = a + b - aa
        cc = a + c - aa
        dd = c + d - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        tables.append((aa, bb, cc, dd))
    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        return (
            comb(aa + bb, aa)
            * comb(cc + dd, cc)
            / comb(aa + bb + cc + dd, aa + cc)
        )
    base = prob(a, b, c, d)
    total = 0.0
    for aa, bb, cc, dd in tables:
        p = prob(aa, bb, cc, dd)
        if p <= base * (1.0 + 1e-9):
            total += p
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> float:
    n1, n2 = len(left), len(right)
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = sorted(left + right)
    ranks: dict[float, float] = {}
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        ranks[pooled[i]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r1 = sum(ranks[value] for value in left)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie = 0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        t = j - i + 1
        tie += t**3 - t
        i = j + 1
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if sigma2 <= 0.0:
        return 1.0
    z = (abs(u - mu) - 0.5) / math.sqrt(sigma2)
    if z < 0.0:
        z = 0.0
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2.0))))


def parse_anneal(text: str, system: str) -> dict[str, str]:
    rows = []
    for line in text.splitlines():
        if ",anneal," not in line:
            continue
        parts = line.split(",")
        if len(parts) < 11 or parts[0] != system or parts[1] != "anneal":
            continue
        rows.append(parts)
    if len(rows) != 1:
        raise RuntimeError(f"expected one anneal row for {system}, saw {len(rows)}")
    parts = rows[0]
    return {
        "seed": parts[2],
        "budget": parts[3],
        "driver_hit": parts[4],
        "evals_to_hit": parts[5],
        "charged": parts[6],
        "wall_s": parts[7],
        "best": parts[8],
        "pot_s": parts[9],
        "outside_s": parts[10],
    }


def is_hit(best: float, system: str) -> bool:
    return energy_hits(best, REFERENCES[system], HIT_TOL)


def run_one(binary: Path, system: str, seed: int, budget: int, log: Path) -> dict[str, str]:
    env = os.environ.copy()
    pin(env, budget)
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", system, str(seed)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary} {system} {seed} exited {proc.returncode}: {proc.stderr[-400:]}"
        )
    return parse_anneal(proc.stdout, system)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarise(rows: list[dict[str, str]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["pair"], row["system"]), []).append(row)
    print(
        "pair,system,revision,driver_blob,hits,n,median_evals_to_hit,"
        "fisher_p,mannwhitney_p"
    )
    for key in sorted(groups):
        pair, system = key
        block = groups[key]
        revisions = []
        for row in block:
            if row["revision"] not in revisions:
                revisions.append(row["revision"])
        samples = []
        for revision in revisions:
            owned = [row for row in block if row["revision"] == revision]
            hits = sum(row["hit"] == "1" for row in owned)
            med = type7([float(row["evals_to_hit"]) for row in owned])
            ranked = [
                float(row["evals_to_hit"]) if row["hit"] == "1" else math.inf
                for row in owned
            ]
            blob = owned[0]["driver_blob"]
            samples.append((revision, blob, hits, len(owned), med, ranked))
        fisher_p = ""
        mw_p = ""
        if len(samples) == 2:
            fisher_p = f"{fisher(samples[0][2], samples[0][3], samples[1][2], samples[1][3]):.6g}"
            mw_p = f"{mann_whitney(samples[0][5], samples[1][5]):.6g}"
        for revision, blob, hits, n, med, _ranked in samples:
            print(
                f"{pair},{system},{revision},{blob},{hits},{n},{med:.4g},{fisher_p},{mw_p}"
            )


def order_for(seed: int, first: int) -> tuple[int, int]:
    if (seed - first) % 2 == 0:
        return (0, 1)
    return (1, 0)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarise", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--pair-name", default="release")
    parser.add_argument("--budget", type=int, default=400_000)
    parser.add_argument("--seed-first", type=int, default=2000)
    parser.add_argument("--seed-last", type=int, default=2011)
    parser.add_argument("--cells", nargs="+", required=False)
    parser.add_argument("--revision", action="append", default=[])
    parser.add_argument("--blob", action="append", default=[])
    parser.add_argument("--binary", action="append", default=[])
    args = parser.parse_args(argv)
    if args.summarise:
        summarise(load_rows(args.summarise))
        return 0
    if args.out is None or args.logs is None or not args.cells:
        raise SystemExit("need --out, --logs and --cells")
    nbin = len(args.binary)
    if nbin < 1 or nbin > 2 or not (nbin == len(args.revision) == len(args.blob)):
        raise SystemExit("pass one or two revision/blob/binary triples")
    for system in args.cells:
        if system not in REFERENCES:
            raise SystemExit(f"unknown system {system}")
    machine = "unrecorded"
    binaries = [Path(item) for item in args.binary]
    for binary in binaries:
        if not binary.is_file():
            raise SystemExit(f"missing {binary}")
    rows = load_rows(args.out)
    seen = {(row["pair"], row["revision"], row["system"], row["seed"]) for row in rows}
    seeds = range(args.seed_first, args.seed_last + 1)
    for system in args.cells:
        for seed in seeds:
            slots = order_for(seed, args.seed_first) if nbin == 2 else (0,)
            for slot in slots:
                revision = args.revision[slot]
                blob = args.blob[slot]
                key = (args.pair_name, revision, system, str(seed))
                if key in seen:
                    print(f"keep {args.pair_name} {revision[:12]} {system} {seed}", flush=True)
                    continue
                print(
                    f"run {args.pair_name} {revision[:12]} {blob[:12]} {system} {seed}",
                    flush=True,
                )
                parsed = run_one(binaries[slot], system, seed, args.budget, args.logs / f"{args.pair_name}-{revision[:12]}-{system}-{seed}.log")
                best = float(parsed["best"])
                hit = "1" if is_hit(best, system) else "0"
                row = {
                    "pair": args.pair_name,
                    "revision": revision,
                    "driver_blob": blob,
                    "system": system,
                    "seed": str(seed),
                    "budget": str(args.budget),
                    "hit": hit,
                    "best": f"{best:.8f}",
                    "evals_to_hit": parsed["evals_to_hit"],
                    "charged": parsed["charged"],
                    "wall_s": parsed["wall_s"],
                    "pot_s": parsed["pot_s"],
                    "outside_s": parsed["outside_s"],
                    "machine_id": machine,
                }
                rows.append(row)
                seen.add(key)
                write_rows(args.out, rows)
                print(
                    f"done {system} {seed} hit={hit} best={row['best']} calls={row['evals_to_hit']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
