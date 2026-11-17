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

``paired`` is the two-arm run on seeds 2000-2011. Each (molecule, seed) pair
is drawn in a shuffled order; openconf runs first, and anneal then gets the
energy plus gradient calls openconf spent on that seed. anneal charges one
budget unit per point and makes one energy and one gradient call there, so
its budget is half that count.
``paired-score`` appends the scored rows (``paired_score.py``).

With ``--baseline DIR`` a pair also runs the ``develop`` arm: anneal imported
from ``DIR`` (``build.sh`` with ``OPENCONF_H2H_BASELINE`` installs it there),
at the same budget as the ``anneal`` arm. The two anneal arms of a pair run
in an order drawn from the schedule seed.
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
PAIRED_SEEDS = tuple(range(2000, 2012))
PAIRED_SCHEDULE_SEED = 20261115
OPENCONF_REV = "e32c5fa7b6e6aac944bda31c851be42e14340235"


def job(arm: str, molecule: str, seed: int, out: Path, args, budget: int = 0) -> dict:
    env = dict(os.environ)
    if arm != "crest":
        env["LD_PRELOAD"] = args.counter
    cmd = [
        "taskset", "-c", str(args.core), sys.executable, "-u", str(HERE / "h2h_job.py"),
        arm, molecule, str(seed), "--out", str(out), "--crest", args.crest,
    ]
    if arm in ("anneal", "develop"):
        cmd += ["--budget", str(budget)]
        if args.local_step:
            cmd += ["--local-step", args.local_step]
    if arm == "anneal" and args.torsion_walk:
        cmd.append("--torsion-walk")
    if arm == "develop":
        cmd += ["--anneal-path", str(args.baseline)]
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
    revision = (args.work / "bin" / "anneal-head").read_text().strip()
    for position, (name, seed, arm) in enumerate(jobs, 1):
        out = tree / name / str(seed) / arm
        if (out / "job.json").is_file() and not args.only:
            continue
        info = stamp(out, job(arm, name, seed, out, args, budget=table[name]), revision)
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


def search_calls(info: dict) -> int:
    return int(info["energy_calls"]) + int(info["grad_calls"])


def stamp(out: Path, info: dict, revision: str) -> dict:
    info.update(code_revision=revision, openconf_revision=OPENCONF_REV)
    (out / "job.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    return info


def paired(args) -> None:
    revisions = {"anneal": (args.work / "bin" / "anneal-head").read_text().strip()}
    revisions["openconf"] = revisions["anneal"]
    if args.baseline:
        revisions["develop"] = (args.work / "bin" / "baseline-head").read_text().strip()
    names = args.molecules or list(hc.molecules())
    pairs = [(name, seed) for name in hc.molecules() for seed in PAIRED_SEEDS]
    rng = random.Random(PAIRED_SCHEDULE_SEED)
    rng.shuffle(pairs)
    orders = {pair: rng.random() < 0.5 for pair in pairs}
    # Seeds past the paired set follow it, so its schedule is unchanged.
    beyond = sorted(set(args.seeds) - set(PAIRED_SEEDS))
    extra = [(name, seed) for name in hc.molecules() for seed in beyond]
    rng.shuffle(extra)
    orders.update({pair: rng.random() < 0.5 for pair in extra})
    pairs = [p for p in pairs + extra if p[0] in names and p[1] in args.seeds]
    tree = args.work / args.tree
    if args.pair:
        pairs = [(args.pair[0], int(args.pair[1]))]
        tree = args.work / "asv-paired"
    for position, (name, seed) in enumerate(pairs, 1):
        base = tree / name / str(seed)
        arms = ["anneal", "develop"] if args.baseline else ["anneal"]
        if args.baseline and orders.get((name, seed), False):
            arms.reverse()
        done = {arm: (base / arm / "job.json").is_file() and not args.pair for arm in ["openconf", *arms]}
        if done["openconf"]:
            first = json.loads((base / "openconf" / "job.json").read_text())
        else:
            first = stamp(base / "openconf", job("openconf", name, seed, base / "openconf", args), revisions["openconf"])
        budget = search_calls(first) // 2
        walls = []
        for arm in arms:
            if done[arm]:
                continue
            info = stamp(base / arm, job(arm, name, seed, base / arm, args, budget=budget), revisions[arm])
            walls.append(f"{arm} {info['wall_s']:.2f} s calls {search_calls(info)}")
        if walls:
            print(
                f"PAIR {position}/{len(pairs)} {name} {seed} budget {budget} "
                f"openconf {first['wall_s']:.2f} s " + " ".join(walls),
                flush=True,
            )


def paired_score(args) -> None:
    out = args.results if args.results.is_absolute() else hc.ROOT / args.results
    cmd = [
        sys.executable, str(HERE / "paired_score.py"),
        "--runs", str(args.work / args.tree),
        "--rows", str(out / "seeds.csv"),
        "--summary", str(out / "summary.csv"),
        "--stats", str(out / "stats.csv"),
    ]
    if args.baseline:
        cmd += ["--versus-develop", str(out / "stats-develop.csv")]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["pilot", "campaign", "score", "paired", "paired-score"])
    parser.add_argument("--work", type=Path, default=Path("/work/openconf-h2h"))
    parser.add_argument("--budgets", type=Path, default=BENCH / "budgets.csv")
    parser.add_argument("--counter", default="/work/openconf-h2h/bin/librdff_count.so")
    parser.add_argument("--crest", default="/work/openconf-h2h/bin/crest")
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--molecules", nargs="*")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--local-step", choices=["lbfgs", "tric", "auto"], help="anneal quench; default the preset's")
    parser.add_argument("--torsion-walk", action="store_true", help="the anneal arm, not develop, with torsion_walk on")
    parser.add_argument("--only", nargs=3, metavar=("MOLECULE", "SEED", "ARM"))
    parser.add_argument("--pair", nargs=2, metavar=("MOLECULE", "SEED"), help="one paired job into asv-paired")
    parser.add_argument("--any-host", action="store_true", help="allow a hostname other than bench")
    parser.add_argument("--baseline", type=Path, help="anneal install of the develop arm (paired)")
    parser.add_argument("--results", type=Path, default=BENCH / "paired", help="paired-score output directory")
    parser.add_argument("--tree", default="paired", help="paired runs directory under --work")
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = list(PAIRED_SEEDS if args.command.startswith("paired") else SEEDS)
    commands = {"pilot": pilot, "campaign": campaign, "score": score, "paired": paired, "paired-score": paired_score}
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
