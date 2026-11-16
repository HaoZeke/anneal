#!/usr/bin/env python3
# Copyright (c) anneal developers
"""GMIN alone on Sutton-Chen, at N=38 and on 9-6 at N=78.

N=38 is seeds 2000-2011 at 400000 calls, from the sc-n12 starts, on 9-6
unless --n38-system names other rows of that panel. Its rows go to
bench/metal-cluster-h2h/sc-n12/seeds.csv. N=78 is 9-6, seeds 9532-9563 at
2000000 calls, from the starts of scripts/sc96_78_common.py, unless --no-n78.
Its rows go to data/sc96_78_hits.csv. The anneal rows of both files are kept
as they are.

Each row records the GMIN revision, the host and the harness revision, the
anneal commit whose run_sc.py wrote the GMIN data file. A seed already
stored for this harness revision is skipped. One job at a time, on core 0.
The process refuses to run unless the UTS hostname is bench and the
working directory is /work.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sc as sc  # noqa: E402
import run_sc_n12 as n12  # noqa: E402

N38_SYSTEMS = ("sc96-38", "sc126-38", "sc108-38")
N78 = "sc96-78"
N78_SEEDS = tuple(range(9532, 9564))
N78_BUDGET = 2_000_000
N78_FIELDS = (
    "arm",
    "seed",
    "hit",
    "hit_calls",
    "spent",
    "best",
    "reference",
    "code_revision",
    "host",
    "harness_revision",
    "wall_s",
    "pot_s",
)


def load(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def save(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def run_n38(args, system: str, seed: int, head: str, revision: str) -> dict[str, str]:
    coords = args.runs / system / f"start-{seed}.xyz"
    coords.parent.mkdir(parents=True, exist_ok=True)
    if not coords.is_file():
        subprocess.run([str(args.binary), "coords", system, str(seed), str(coords)], check=True)
    log = args.logs / system / f"gmin-{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    row = sc.run_gmin(args.gmin, system, seed, sc.PAPER_BUDGETS[system], coords, args.runs / system / f"gmin-{seed}", log)
    seed_row = n12.as_seed(system, "gmin", row, {})
    seed_row.update({"code_revision": revision, "host": os.uname().nodename, "harness_revision": head})
    return seed_row


def run_n78(args, seed: int, reference: str, head: str, revision: str) -> dict[str, str]:
    coords = args.starts78 / f"seed{seed}.xyz"
    log = args.logs / N78 / f"gmin-{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    row = sc.run_gmin(args.gmin, N78, seed, N78_BUDGET, coords, args.runs / N78 / f"gmin-{seed}", log,
                      target=float(reference))
    hit = row["hit"] == "1"
    return {
        "arm": "gmin",
        "seed": str(seed),
        "hit": "1" if hit else "0",
        "hit_calls": row["evals_to_hit"] if hit else "",
        "spent": row["charged"],
        "best": row["best"],
        "reference": reference,
        "code_revision": revision,
        "host": os.uname().nodename,
        "harness_revision": head,
        "wall_s": f"{float(row['wall_s']):.6f}",
        "pot_s": f"{float(row['pot_s']):.6f}",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--n38-seeds", type=Path, required=True)
    parser.add_argument("--n38-summary", type=Path, required=True)
    parser.add_argument("--n78-rows", type=Path, required=True)
    parser.add_argument("--starts78", type=Path, required=True)
    parser.add_argument("--reference78", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--n38-system", action="append", choices=N38_SYSTEMS)
    parser.add_argument("--no-n78", action="store_true")
    args = parser.parse_args(argv)
    systems = tuple(args.n38_system) if args.n38_system else ("sc96-38",)
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    if Path.cwd() != Path("/work"):
        print(f"working directory is {Path.cwd()}, expected /work", file=sys.stderr)
        return 2
    head = args.head_file.read_text().strip()
    revision = sc.load_source(args.source)
    reference = args.reference78.read_text().strip()

    n38_fields, n38_rows = load(args.n38_seeds)
    if list(n38_fields) != list(n12.SEED_FIELDS):
        raise SystemExit(f"{args.n38_seeds} columns are not run_sc_n12.SEED_FIELDS")
    n78_fields, n78_rows = load(args.n78_rows)
    n78_fields = n78_fields + [field for field in N78_FIELDS if field not in n78_fields]

    def stored(rows: list[dict[str, str]], seed: int, **match: str) -> bool:
        return any(
            r["arm"] == "gmin" and r["seed"] == str(seed) and r.get("harness_revision") == head
            and all(r.get(k) == v for k, v in match.items())
            for r in rows
        )

    for system in systems:
        for seed in n12.SEEDS:
            if stored(n38_rows, seed, system=system):
                print(f"skip {system} gmin seed {seed}", flush=True)
                continue
            print(f"run {system} gmin seed {seed}", flush=True)
            n38_rows.append(run_n38(args, system, seed, head, revision))
            save(args.n38_seeds, list(n12.SEED_FIELDS), n38_rows)
            print(f"done {system} gmin seed {seed} best={n38_rows[-1]['best']} hit={n38_rows[-1]['hit']} "
                  f"wall={n38_rows[-1]['wall_s']}", flush=True)
    summary = n12.summarise(n38_rows, head, revision)
    save(args.n38_summary, list(n12.SUMMARY_FIELDS), summary)

    if args.no_n78:
        return 0
    for seed in N78_SEEDS:
        if stored(n78_rows, seed):
            print(f"skip {N78} gmin seed {seed}", flush=True)
            continue
        print(f"run {N78} gmin seed {seed}", flush=True)
        n78_rows.append(run_n78(args, seed, reference, head, revision))
        save(args.n78_rows, n78_fields, n78_rows)
        print(f"done {N78} gmin seed {seed} best={n78_rows[-1]['best']} hit={n78_rows[-1]['hit']} "
              f"wall={n78_rows[-1]['wall_s']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
