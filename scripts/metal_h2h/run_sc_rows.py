#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the Sutton–Chen binary and append the raw rows.

`restart`, `pop`, `chain`, `portfolio`, and `session` are the tables
under bench/metal-cluster-h2h. A finished seed is left as it was written.
`session` appends a same-session baseline and portfolio for seeds
2000-2023 and does not touch the other tables.
`code_revision` is the git revision the binary was built from, and
`host` is the UTS hostname. The campaign runs on hostname bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ("sc126-80", "sc96-80", "sc108-80", "sc108-55")
SEEDS = tuple(range(2000, 2012))
SESSION_SEEDS = tuple(range(2000, 2024))
PROBE_SEEDS = (2000, 2007, 2003)

RESTART_FIELDS = (
    "variant",
    "system",
    "seed",
    "hit",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
)
POP_FIELDS = (
    "system",
    "code",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
)


def jobs(
    campaign: str, root: Path
) -> tuple[Path, tuple[str, ...], list[tuple[str, str, int, dict[str, str]]]]:
    """Output path, columns, and (variant, system, seed, env) jobs."""
    if campaign == "restart":
        path = root / "bench/metal-cluster-h2h/sc-restart/rows.csv"
        planned: list[tuple[str, str, int, dict[str, str]]] = []
        for system in SYSTEMS:
            for seed in SEEDS:
                planned.append(
                    ("stuck_k1", system, seed, {"H2H_RESTART": "stuck", "H2H_RESTART_K": "1"})
                )
        for variant, env in (
            ("luby", {"H2H_RESTART": "luby"}),
            ("archive", {"H2H_RESTART": "archive"}),
            ("stuck2", {"H2H_RESTART": "stuck", "H2H_RESTART_K": "2"}),
            ("stuck3", {"H2H_RESTART": "stuck", "H2H_RESTART_K": "3"}),
        ):
            for seed in PROBE_SEEDS:
                planned.append((variant, "sc126-80", seed, env))
        return path, RESTART_FIELDS, planned
    if campaign == "pop":
        path = root / "bench/metal-cluster-h2h/sc-pop/rows.csv"
        planned = [
            ("pop2", system, seed, {"H2H_POP": "2"}) for system in SYSTEMS for seed in SEEDS
        ]
        return path, POP_FIELDS, planned
    if campaign == "chain":
        path = root / "bench/metal-cluster-h2h/sc-chain/rows.csv"
        planned = [
            ("chain", system, seed, {"H2H_POP": "2"}) for system in SYSTEMS for seed in SEEDS
        ]
        return path, POP_FIELDS, planned
    if campaign == "portfolio":
        path = root / "bench/metal-cluster-h2h/sc-portfolio/rows.csv"
        planned = [
            ("portfolio", system, seed, {"H2H_PORTFOLIO": "1"})
            for system in SYSTEMS
            for seed in SEEDS
        ]
        return path, POP_FIELDS, planned
    if campaign == "session":
        path = root / "bench/metal-cluster-h2h/sc-session/rows.csv"
        planned = []
        for variant, env in (
            ("baseline", {}),
            ("portfolio", {"H2H_PORTFOLIO": "1"}),
        ):
            for system in SYSTEMS:
                for seed in SESSION_SEEDS:
                    planned.append((variant, system, seed, env))
        return path, ("variant",) + POP_FIELDS, planned
    raise SystemExit(f"unknown campaign {campaign}")


def load_done(path: Path, campaign: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    for row in csv.DictReader(path.open()):
        if campaign in ("restart", "session"):
            done.add((row["variant"], f"{row['system']}:{row['seed']}"))
        else:
            done.add((row["system"], row["seed"]))
    return done


def parse_stdout(text: str, system: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith(system + ","):
            header = (
                "system,code,seed,budget,hit,evals_to_hit,charged,"
                "wall_s,best,pot_s,outside_s"
            ).split(",")
            values = line.split(",")
            if len(values) < len(header):
                break
            return dict(zip(header, values))
    raise RuntimeError(f"no result row for {system}\n{text[-1500:]}")


def append_row(path: Path, fields: tuple[str, ...], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n", extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fields})


def run_one(
    binary: Path,
    system: str,
    seed: int,
    extra: dict[str, str],
) -> dict[str, str]:
    env = os.environ.copy()
    if "H2H_PORTFOLIO" not in extra:
        env.pop("H2H_PORTFOLIO", None)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    env.update(extra)
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", system, str(seed)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{system} seed {seed} exited {proc.returncode}\n{proc.stderr[-1500:]}"
        )
    return parse_stdout(proc.stdout, system)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaign", choices=("restart", "pop", "chain", "portfolio", "session")
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root the row tables are written under",
    )
    args = parser.parse_args(argv)
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    revision = args.head_file.read_text().strip()
    host = os.uname().nodename
    path, fields, planned = jobs(args.campaign, args.root.resolve())
    done = load_done(path, args.campaign)
    for variant, system, seed, extra in planned:
        key = (
            (variant, f"{system}:{seed}")
            if args.campaign in ("restart", "session")
            else (system, str(seed))
        )
        if key in done:
            print(f"skip {args.campaign} {variant} {system} {seed}", flush=True)
            continue
        print(f"run {args.campaign} {variant} {system} {seed}", flush=True)
        parsed = run_one(args.binary, system, seed, extra)
        hit = parsed["hit"]
        if args.campaign == "restart":
            hit = "1" if hit in ("true", "1") else "0"
        row = {
            **parsed,
            "variant": variant,
            "hit": hit,
            "code_revision": revision,
            "host": host,
        }
        append_row(path, fields, row)
        done.add(key)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
