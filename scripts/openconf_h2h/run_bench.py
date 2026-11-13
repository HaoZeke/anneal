#!/usr/bin/env python3
# Copyright (c) anneal developers
"""anneal, openconf and CREST on drug-like molecules, one serial session.

``pilot`` runs openconf on seeds 900-902 of every molecule. The anneal budget
of a molecule is the median MMFF gradient calls of those three runs, so
anneal is charged what openconf spends. Pilot seeds are not reported.

``campaign`` shuffles every (molecule, seed, arm) job with the schedule
seed and runs them one at a time on core 0. A job whose ``job.json``
exists is skipped, so an interrupted session resumes.

``score`` polishes and scores every finished job (``h2h_score.py``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h2h_common as hc  # noqa: E402

ARMS = ("anneal", "openconf", "crest")
SEEDS = tuple(range(2000, 2006))
PILOT_SEEDS = (900, 901, 902)
SCHEDULE_SEED = 20261113
BENCH = hc.ROOT / "bench" / "openconf-h2h"


def job(arm: str, molecule: str, seed: int, out: Path, args, budget: int = 0) -> dict:
    env = dict(os.environ)
    if arm != "crest":
        env["LD_PRELOAD"] = args.counter
    cmd = [
        "taskset", "-c", str(args.core), sys.executable, "-u", str(HERE / "h2h_job.py"),
        arm, molecule, str(seed), "--out", str(out), "--crest", args.crest,
    ]
    if arm == "anneal":
        cmd += ["--budget", str(budget)]
        if args.local_step:
            cmd += ["--local-step", args.local_step]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "job.log").open("w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{arm} {molecule} {seed} exited {proc.returncode}; see {out / 'job.log'}")
    info = json.loads((out / "job.json").read_text())
    if info["hostname"] != "bench" and not args.any_host:
        raise RuntimeError(f"hostname was {info['hostname']}")
    return info


def budgets(path: Path) -> dict[str, int]:
    with path.open(newline="") as handle:
        return {row["molecule"]: int(row["anneal_budget"]) for row in csv.DictReader(handle)}


def pilot(args) -> None:
    rows = []
    for name in hc.molecules():
        calls = []
        for seed in PILOT_SEEDS:
            info = job("openconf", name, seed, args.work / "pilot" / name / str(seed), args)
            calls.append(info["grad_calls"])
            print(f"PILOT {name} {seed} grad_calls {info['grad_calls']} wall {info['wall_s']:.2f}", flush=True)
        rows.append(
            {
                "molecule": name,
                "pilot_seeds": " ".join(map(str, PILOT_SEEDS)),
                "openconf_grad_calls": " ".join(map(str, calls)),
                "anneal_budget": int(statistics.median(calls)),
            }
        )
    with args.budgets.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def schedule(names, arms, seeds) -> list[tuple[str, int, str]]:
    jobs = [(name, seed, arm) for name in names for seed in seeds for arm in arms]
    random.Random(SCHEDULE_SEED).shuffle(jobs)
    return jobs


def campaign(args) -> None:
    table = budgets(args.budgets)
    names = args.molecules or list(hc.molecules())
    extra = sorted(set(args.seeds) - set(SEEDS))
    jobs = schedule(list(hc.molecules()), ARMS, SEEDS) + schedule(list(hc.molecules()), ARMS, extra)
    jobs = [j for j in jobs if j[0] in names and j[2] in args.arms and j[1] in args.seeds]
    if args.only:
        name, seed, arm = args.only[0], int(args.only[1]), args.only[2]
        jobs = [(name, seed, arm)]
    tree = args.work / ("asv" if args.only else "runs")
    for position, (name, seed, arm) in enumerate(jobs, 1):
        out = tree / name / str(seed) / arm
        if (out / "job.json").is_file() and not args.only:
            continue
        info = job(arm, name, seed, out, args, budget=table[name])
        print(
            f"JOB {position}/{len(jobs)} {name} {seed} {arm} wall {info['wall_s']:.2f} "
            f"grad {info.get('grad_calls')} candidates {info.get('candidates')}",
            flush=True,
        )


def score(args) -> None:
    subprocess.run(
        [
            sys.executable, str(HERE / "h2h_score.py"),
            "--runs", str(args.work / "runs"),
            "--seeds-out", str(BENCH / "seeds.csv"),
            "--summary", str(BENCH / "results.csv"),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["pilot", "campaign", "score"])
    parser.add_argument("--work", type=Path, default=Path("/work/openconf-h2h"))
    parser.add_argument("--budgets", type=Path, default=BENCH / "budgets.csv")
    parser.add_argument("--counter", default="/work/openconf-h2h/bin/librdff_count.so")
    parser.add_argument("--crest", default="/work/openconf-h2h/bin/crest")
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--molecules", nargs="*")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--local-step", choices=["lbfgs", "tric", "auto"], help="anneal quench; default the preset's")
    parser.add_argument("--only", nargs=3, metavar=("MOLECULE", "SEED", "ARM"))
    parser.add_argument("--any-host", action="store_true", help="allow a hostname other than bench")
    args = parser.parse_args()
    {"pilot": pilot, "campaign": campaign, "score": score}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
