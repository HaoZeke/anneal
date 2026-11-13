#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial Gupta head-to-head: anneal against the ASE genetic algorithm.

Systems are the Cambridge zinc minima at 38, 55, 75 and 80, and Cu19Au19
from the Cambridge CuAu_38 archive. Six seeds, the same list on both arms.
Within a system the (seed, arm) pairs are shuffled. A hit is a finished
relaxation within 0.001 of the Cambridge energy. A miss is stored as the
full budget. The median of calls to hit uses only the hits. The wall ratio
is the ASE median wall divided by the anneal median wall, over every seed.

This process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

SYSTEMS = ("zn38", "zn55", "zn75", "zn80", "cu19au19")
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("anneal", "ase")
SCHEDULE_SEED = 20261112
HIT_TOL = 0.001

SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "hits",
    "hits_over_n",
    "budget",
    "evals_median_to_hit",
    "evals_iqr_to_hit",
    "wall_median_s",
    "wall_iqr_s",
    "wall_ratio",
    "pot_per_eval_s",
    "outside_per_eval_s",
    "seeds",
    "schedule_seed",
    "anneal_head",
]

ROW_FIELDS = [
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
    "pot_per_eval_s",
    "outside_per_eval_s",
]


def type7(sorted_values: list[float], p: float) -> float:
    """Hyndman-Fan type 7. ``sorted_values`` is ascending."""
    count = len(sorted_values)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return sorted_values[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return type7(ordered, 0.5), type7(ordered, 0.75) - type7(ordered, 0.25)


def schedule(
    systems: tuple[str, ...],
    seeds: tuple[int, ...],
    arms: tuple[str, ...],
    rng_seed: int,
) -> list[tuple[str, int, str]]:
    """Systems in the given order. Pairs inside a system are shuffled."""
    rng = random.Random(rng_seed)
    order: list[tuple[str, int, str]] = []
    for system in systems:
        jobs = [(system, seed, arm) for seed in seeds for arm in arms]
        rng.shuffle(jobs)
        order.extend(jobs)
    return order


def binary_budget(binary: Path, system: str) -> tuple[int, float]:
    proc = subprocess.run(
        [str(binary), "budget", system],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"budget for {system} returned {proc.stdout!r}")
    return int(lines[0]), float(lines[1])


def parse_row(text: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if ",anneal," in line or ",ase," in line]
    if len(data) != 1:
        raise ValueError(f"expected one result row, saw {len(data)} in {text!r}")
    return next(csv.DictReader([",".join(ROW_FIELDS), data[0]]))


def _env(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMBA_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    return env


def run_arm(
    binary: Path,
    ga: Path,
    system: str,
    seed: int,
    budget: int,
    arm: str,
    log: Path,
    work: Path,
) -> dict[str, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    if arm == "anneal":
        command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
    else:
        command = [
            "taskset",
            "-c",
            "0",
            sys.executable,
            str(ga),
            "--binary",
            str(binary),
            "--system",
            system,
            "--seed",
            str(seed),
            "--budget",
            str(budget),
            "--work",
            str(work),
        ]
    with log.open("w") as handle:
        proc = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=_env(budget),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"{arm} {system} seed {seed} exited {proc.returncode}\n{text[-2000:]}")
    row = parse_row(text)
    if int(row["budget"]) != budget or row["code"] != arm:
        raise RuntimeError(f"{arm} row {row}")
    return row


def as_result(row: dict[str, str]) -> dict:
    return {
        "system": row["system"],
        "arm": row["code"],
        "seed": int(row["seed"]),
        "budget": int(row["budget"]),
        "hit": row["hit"] in ("1", "true", "True"),
        "evals": float(row["evals_to_hit"]),
        "wall_s": float(row["wall_s"]),
        "best": float(row["best"]),
        "pot_per": float(row["pot_per_eval_s"]),
        "outside_per": float(row["outside_per_eval_s"]),
        "charged": float(row["charged"]),
    }


def summarise(rows: list[dict], seeds: tuple[int, ...], head: str) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        grouped.setdefault(row["system"], {}).setdefault(row["arm"], []).append(row)
    seed_text = " ".join(str(seed) for seed in seeds)
    out: list[dict[str, str]] = []
    for system in SYSTEMS:
        arms = grouped.get(system, {})
        if not arms:
            continue
        walls = {
            arm: median_iqr([item["wall_s"] for item in group])[0]
            for arm, group in arms.items()
            if group
        }
        ratio = ""
        if "anneal" in walls and "ase" in walls and walls["anneal"] > 0.0:
            ratio = f"{walls['ase'] / walls['anneal']:.6f}"
        for arm in ARMS:
            group = arms.get(arm, [])
            if not group:
                continue
            hits = [item for item in group if item["hit"]]
            if hits:
                evals_median, evals_iqr = median_iqr([item["evals"] for item in hits])
                evals_median_text = f"{evals_median:.6f}"
                evals_iqr_text = f"{evals_iqr:.6f}"
            else:
                evals_median_text = ""
                evals_iqr_text = ""
            wall_median, wall_iqr = median_iqr([item["wall_s"] for item in group])
            weight = sum(item["charged"] for item in group)
            pot = sum(item["pot_per"] * item["charged"] for item in group) / weight
            outside = sum(item["outside_per"] * item["charged"] for item in group) / weight
            hit_count = len(hits)
            out.append(
                {
                    "system": system,
                    "arm": arm,
                    "n": str(len(group)),
                    "hits": str(hit_count),
                    "hits_over_n": f"{hit_count}/{len(group)}",
                    "budget": str(group[0]["budget"]),
                    "evals_median_to_hit": evals_median_text,
                    "evals_iqr_to_hit": evals_iqr_text,
                    "wall_median_s": f"{wall_median:.6f}",
                    "wall_iqr_s": f"{wall_iqr:.6f}",
                    "wall_ratio": ratio,
                    "pot_per_eval_s": f"{pot:.9f}",
                    "outside_per_eval_s": f"{outside:.9f}",
                    "seeds": seed_text,
                    "schedule_seed": str(SCHEDULE_SEED),
                    "anneal_head": head,
                }
            )
    return out


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def note(progress: Path, line: str) -> None:
    print(line, flush=True)
    with progress.open("a") as handle:
        handle.write(line + "\n")


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    order = schedule(SYSTEMS, SEEDS, ARMS, SCHEDULE_SEED)
    seen: list[str] = []
    for system, _seed, _arm in order:
        if not seen or seen[-1] != system:
            seen.append(system)
    if seen != list(SYSTEMS):
        raise SystemExit(f"system order {seen}")
    for system in SYSTEMS:
        jobs = [(seed, arm) for name, seed, arm in order if name == system]
        expected = [(seed, arm) for seed in SEEDS for arm in ARMS]
        if sorted(jobs) != sorted(expected):
            raise SystemExit(f"{system} is not the shared seed list")
        arms = [arm for _seed, arm in jobs]
        if arms == ["anneal"] * len(SEEDS) + ["ase"] * len(SEEDS):
            raise SystemExit(f"{system} is not interleaved")
    rows = [
        {
            "system": "zn38",
            "arm": "anneal",
            "hit": True,
            "evals": 10.0,
            "wall_s": 1.0,
            "pot_per": 0.2,
            "outside_per": 0.8,
            "charged": 10.0,
            "budget": 50,
            "best": -50.0,
            "seed": 1,
        },
        {
            "system": "zn38",
            "arm": "anneal",
            "hit": False,
            "evals": 50.0,
            "wall_s": 3.0,
            "pot_per": 0.2,
            "outside_per": 0.8,
            "charged": 50.0,
            "budget": 50,
            "best": -40.0,
            "seed": 2,
        },
        {
            "system": "zn38",
            "arm": "ase",
            "hit": True,
            "evals": 30.0,
            "wall_s": 4.0,
            "pot_per": 0.5,
            "outside_per": 1.5,
            "charged": 30.0,
            "budget": 50,
            "best": -50.0,
            "seed": 1,
        },
    ]
    summary = summarise(rows, (1, 2), "abc")
    anneal = summary[0]
    if anneal["hits"] != "1" or anneal["evals_median_to_hit"] != "10.000000":
        raise SystemExit(f"hit median {anneal}")
    if summary[1]["wall_ratio"] != "2.000000":
        raise SystemExit(f"wall ratio {summary[1]}")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--ga", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", choices=("anneal", "ase", "both"), default="both")
    args = parser.parse_args(argv[1:])
    systems = tuple(args.system) if args.system else SYSTEMS
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = ARMS if args.arm == "both" else (args.arm,)
    for system in systems:
        if system not in SYSTEMS:
            raise SystemExit(f"unknown system {system}")
    head = args.head_file.read_text().strip()
    progress = args.runs.parent / "progress.txt"
    progress.parent.mkdir(parents=True, exist_ok=True)
    order = schedule(systems, seeds, arms, SCHEDULE_SEED)
    note(progress, "schedule " + " ".join(f"{system}:{arm}:{seed}" for system, seed, arm in order))
    rows: list[dict] = []
    for index, (system, seed, arm) in enumerate(order, start=1):
        budget, _reference = binary_budget(args.binary, system)
        log = args.logs / f"{index:02d}-{system}-seed{seed}-{arm}.log"
        saved = args.runs / system / f"{arm}-{seed}.row"
        saved.parent.mkdir(parents=True, exist_ok=True)
        if saved.is_file() and saved.stat().st_size > 0:
            row = parse_row(saved.read_text())
            result = as_result(row)
            rows.append(result)
            note(progress, f"skip {system} {arm} seed {seed} hit {int(result['hit'])}")
            continue
        note(progress, f"start {system} {arm} seed {seed} budget {budget}")
        raw = run_arm(
            args.binary,
            args.ga,
            system,
            seed,
            budget,
            arm,
            log,
            args.runs / system / f"{arm}-{seed}",
        )
        saved.write_text(log.read_text())
        result = as_result(raw)
        rows.append(result)
        note(
            progress,
            f"done {system} {arm} seed {seed} hit {int(result['hit'])} "
            f"evals {result['evals']:.0f} wall_s {result['wall_s']:.3f} "
            f"best {result['best']:.6f} pot_per {result['pot_per']:.6e} "
            f"outside_per {result['outside_per']:.6e}",
        )
        if args.summary:
            write_summary(args.summary, summarise(rows, seeds, head))
    text_rows = summarise(rows, seeds, head)
    if args.summary:
        write_summary(args.summary, text_rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(text_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
