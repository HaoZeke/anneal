#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append-only rows for the labelled funnel leave.

One job at a time. The process refuses to run unless the UTS hostname is
bench and the working directory is /work. A row already stored for the
same system, priors, seed, arm and code revision is left in place.

Au55 is Gupta at 1_000_000 calls, seeds 2000–2035, the recommended
preset. LJ98 is the 4_000_000-call paper budget. The seven binaries use
their paper budgets with priors on and off. The SC80 trio is sc96-80,
sc108-80 and sc126-80 at 1_000_000 calls, launched the same way as
``scripts/metal_h2h/run_sc.py``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "bench" / "au55-funnel2" / "seeds.csv"
DEFAULT_LOGS = ROOT / "bench" / "au55-funnel2" / "logs"
HIT_TOL = 0.001

BINARIES = (
    ("cu6au32", 400_000, -130.79994836),
    ("cu19au19", 400_000, -122.99148391),
    ("cu10au45", 1_000_000, -191.30598075),
    ("cu27au28", 1_000_000, -182.93136675),
    ("cu30au8", 400_000, -115.53258986),
    ("cu12au26", 400_000, -127.21076866),
    ("pt12cu11", 200_000, -96.03443577),
)
LJ98 = (98, 4_000_000, -543.665361)
SC80 = ("sc96-80", "sc108-80", "sc126-80")
SC_BUDGET = 1_000_000

FIELDS = (
    "system",
    "arm",
    "priors",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
)


def require_bench() -> str:
    host = os.uname().nodename
    if host != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return ""
    if Path.cwd() != Path("/work"):
        print(f"working directory is {Path.cwd()}, expected /work", file=sys.stderr)
        return ""
    return host


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def key_of(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["system"],
        row["arm"],
        row.get("priors", ""),
        row["seed"],
        row["code_revision"],
    )


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(env)
    merged["OMP_NUM_THREADS"] = "1"
    merged["RAYON_NUM_THREADS"] = "1"
    return subprocess.run(
        ["taskset", "-c", "0", *cmd],
        check=False,
        text=True,
        capture_output=True,
        env=merged,
    )


def from_metal(text: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if ",anneal," in line]
    if len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}\n{text[-500:]}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,"
        "pot_s,outside_s,pot_per_eval_s,outside_per_eval_s"
    )
    return next(csv.DictReader([header, data[0]]))


def from_seed(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            fields = dict(tok.split("=", 1) for tok in line.split()[1:])
            hit = fields.get("hit", "0") in {"1", "true", "True"}
            return {
                "hit": "true" if hit else "false",
                "evals_to_hit": fields["evals_to_hit"],
                "charged": fields["charged"],
                "best": fields["best"],
                "wall_s": fields["wall_s"],
                "pot_s": fields["pot_s"],
                "outside_s": fields["outside_s"],
                "budget": fields["budget"],
            }
    raise ValueError(f"no h2h_row in\n{text[-500:]}")


def store(
    path: Path,
    existing: set[tuple[str, str, str, str, str]],
    row: dict[str, str],
) -> None:
    identity = key_of(row)
    if identity in existing:
        print(f"keep {identity}", flush=True)
        return
    append_row(path, row)
    existing.add(identity)
    print(
        f"{row['arm']} {row['system']} priors={row['priors']} seed={row['seed']} "
        f"hit={row['hit']} best={row['best']} pot={row['pot_s']} outside={row['outside_s']}",
        flush=True,
    )


def metal_job(
    binary: Path,
    system: str,
    seed: int,
    budget: int,
    reference: float,
    priors: str,
    arm: str,
    revision: str,
    host: str,
) -> dict[str, str]:
    env = {"H2H_BUDGET": str(budget), "H2H_REFERENCE": f"{reference:.8f}"}
    if priors != "":
        env["H2H_PRIORS"] = priors
    proc = run([str(binary), "run", system, str(seed)], env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1000:] + proc.stdout[-1000:])
    parsed = from_metal(proc.stdout)
    return {
        "system": system,
        "arm": arm,
        "priors": priors,
        "seed": str(seed),
        "budget": parsed["budget"],
        "hit": parsed["hit"],
        "evals_to_hit": parsed["evals_to_hit"],
        "charged": parsed["charged"],
        "best": f"{float(parsed['best']):.8f}",
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
        "code_revision": revision,
        "host": host,
    }


def lj_job(
    binary: Path,
    seed: int,
    arm: str,
    revision: str,
    host: str,
    logs: Path,
) -> dict[str, str]:
    n, budget, reference = LJ98
    stop = f"{reference + HIT_TOL:.6f}"
    prefix = logs / f"lj98-{arm}-{seed}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    proc = run(
        [
            str(binary),
            "lj",
            str(n),
            "0",
            str(budget),
            str(seed),
            "recommended",
            stop,
            str(prefix),
        ],
        {},
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1000:] + proc.stdout[-1000:])
    parsed = from_seed(proc.stdout)
    return {
        "system": "lj98",
        "arm": arm,
        "priors": "",
        "seed": str(seed),
        "budget": parsed["budget"],
        "hit": parsed["hit"],
        "evals_to_hit": parsed["evals_to_hit"],
        "charged": parsed["charged"],
        "best": f"{float(parsed['best']):.10f}",
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
        "code_revision": revision,
        "host": host,
    }


def sc_job(
    binary: Path,
    system: str,
    seed: int,
    arm: str,
    revision: str,
    host: str,
    logs: Path,
) -> dict[str, str]:
    log = logs / f"{system}-{arm}-{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {"H2H_BUDGET": str(SC_BUDGET), "OMP_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}
    proc = run([str(binary), "run", system, str(seed), str(log.with_suffix(".xyz"))], env)
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} seed {seed} exited {proc.returncode}\n{proc.stderr[-800:]}")
    parsed = from_metal(proc.stdout)
    return {
        "system": system,
        "arm": arm,
        "priors": "",
        "seed": str(seed),
        "budget": parsed["budget"],
        "hit": parsed["hit"],
        "evals_to_hit": parsed["evals_to_hit"],
        "charged": parsed["charged"],
        "best": f"{float(parsed['best']):.4f}",
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
        "code_revision": revision,
        "host": host,
    }


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    from math import comb

    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b

    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        return comb(aa + bb, aa) * comb(cc + dd, cc) / comb(aa + bb + cc + dd, aa + cc)

    total = a + b + c + d
    row = a + b
    col = a + c
    observed = prob(a, b, c, d)
    score = 0.0
    for aa in range(max(0, row + col - total), min(row, col) + 1):
        bb = row - aa
        cc = col - aa
        dd = total - row - cc
        value = prob(aa, bb, cc, dd)
        if value <= observed + 1e-15:
            score += value
    return score


def mann_whitney(left: list[float], right: list[float]) -> float:
    n1 = len(left)
    n2 = len(right)
    if n1 == 0 or n2 == 0:
        return float("nan")
    combined = [(value, 0) for value in left] + [(value, 1) for value in right]
    combined.sort(key=lambda item: item[0])
    ranks = [0.0] * len(combined)
    tie_term = 0.0
    index = 0
    while index < len(combined):
        end = index
        while end < len(combined) and combined[end][0] == combined[index][0]:
            end += 1
        average = 0.5 * ((index + 1) + end)
        for slot in range(index, end):
            ranks[slot] = average
        width = end - index
        if width > 1:
            tie_term += width**3 - width
        index = end
    rank_sum = sum(rank for rank, (_, group) in zip(ranks, combined) if group == 0)
    u1 = rank_sum - n1 * (n1 + 1) / 2.0
    total = n1 + n2
    var = n1 * n2 / 12.0 * ((total + 1) - tie_term / (total * (total - 1)))
    if var <= 0.0:
        return 1.0
    z = (abs(u1 - n1 * n2 / 2.0) - 0.5) / math.sqrt(var)
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def summarise(rows: list[dict[str, str]], system: str, priors: str) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["system"] == system and row.get("priors", "") == priors:
            grouped.setdefault(row["arm"], []).append(row)
    if len(grouped) < 2:
        return
    names = sorted(grouped)
    left, right = names[0], names[1]
    a, b = grouped[left], grouped[right]
    ha = sum(row["hit"] == "true" for row in a)
    hb = sum(row["hit"] == "true" for row in b)
    print(
        f"summary {system} priors={priors} {left} {ha}/{len(a)} {right} {hb}/{len(b)} "
        f"fisher={fisher(hb, len(b), ha, len(a)):.4g} "
        f"mw_calls={mann_whitney([float(r['evals_to_hit']) for r in b], [float(r['evals_to_hit']) for r in a]):.4g}",
        flush=True,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--suite", choices=("au55", "lj98", "binaries", "sc80"), required=True)
    parser.add_argument("--seeds-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--sc-lib", default="")
    args = parser.parse_args(argv)
    host = require_bench()
    if not host:
        return 2
    rows = load_rows(args.seeds_out)
    existing = {key_of(row) for row in rows}
    if args.suite == "au55":
        for seed in range(2000, 2036):
            if ( "au55", args.arm, "", str(seed), args.revision) in existing:
                print(f"keep au55 {args.arm} {seed}", flush=True)
                continue
            store(
                args.seeds_out,
                existing,
                metal_job(
                    args.binary,
                    "au55",
                    seed,
                    1_000_000,
                    -191.03085280,
                    "",
                    args.arm,
                    args.revision,
                    host,
                ),
            )
        summarise(load_rows(args.seeds_out), "au55", "")
    elif args.suite == "lj98":
        for seed in range(2000, 2012):
            if ("lj98", args.arm, "", str(seed), args.revision) in existing:
                print(f"keep lj98 {args.arm} {seed}", flush=True)
                continue
            store(
                args.seeds_out,
                existing,
                lj_job(args.binary, seed, args.arm, args.revision, host, args.logs),
            )
    elif args.suite == "binaries":
        for system, budget, reference in BINARIES:
            for priors in ("1", "0"):
                for seed in range(2000, 2012):
                    if (system, args.arm, priors, str(seed), args.revision) in existing:
                        print(f"keep {system} {args.arm} priors={priors} {seed}", flush=True)
                        continue
                    store(
                        args.seeds_out,
                        existing,
                        metal_job(
                            args.binary,
                            system,
                            seed,
                            budget,
                            reference,
                            priors,
                            args.arm,
                            args.revision,
                            host,
                        ),
                    )
    elif args.suite == "sc80":
        if args.sc_lib:
            os.environ["RGPOT_SC_LIB"] = args.sc_lib
        for system in SC80:
            for seed in range(2000, 2012):
                if (system, args.arm, "", str(seed), args.revision) in existing:
                    print(f"keep {system} {args.arm} {seed}", flush=True)
                    continue
                store(
                    args.seeds_out,
                    existing,
                    sc_job(
                        args.binary,
                        system,
                        seed,
                        args.arm,
                        args.revision,
                        host,
                        args.logs,
                    ),
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
