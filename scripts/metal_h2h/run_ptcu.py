#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Anneal against GMIN on Cleri–Rosato Pt–Cu.

Sizes are 13, 23 and 38, the sizes in Wu, Lin, He, Chen and Shi,
Nanoscale 18, 14001 (2026). Three compositions at each size. Six seeds,
2000–2005, one job at a time, in one shuffled order. The process refuses
to run unless the UTS hostname is bench and the working directory is /work.

Wu et al. do not publish an open table of Gupta total energies. A hit is
a quenched energy within 0.001 eV of the lowest energy either arm recorded
for that system in this session. Every seed spends its budget. The wall
ratio is the GMIN median wall divided by the anneal median wall. Inside
time is potential time per charged call. Outside time is the rest of the
wall on the same denominator.

38 atoms uses the Lennard-Jones paper budget, 400000 calls. 13 and 23 are
not in that map; they use 100000 and 200000. Those three numbers are this
panel's PAPER_BUDGETS and are not the LJ campaign's dict.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

# N=38 is the LJ38 entry of the paper map. N=13 and N=23 are not in it.
PAPER_BUDGETS = {
    "pt13": 100_000,
    "cu13": 100_000,
    "pt7cu6": 100_000,
    "pt12cu11": 200_000,
    "pt11cu12": 200_000,
    "pt6cu17": 200_000,
    "pt38": 400_000,
    "pt19cu19": 400_000,
    "cu38": 400_000,
}
# First species count is platinum. The rest are copper.
N_PLATINUM = {
    "pt13": 13,
    "cu13": 0,
    "pt7cu6": 7,
    "pt12cu11": 12,
    "pt11cu12": 11,
    "pt6cu17": 6,
    "pt38": 38,
    "pt19cu19": 19,
    "cu38": 0,
}
N_ATOMS = {name: 13 if name.endswith("13") or name == "pt7cu6" else 23 if "11" in name or "12" in name or name == "pt6cu17" else 38 for name in PAPER_BUDGETS}
# A, p, q, xi, r0. Cleri and Rosato, Phys. Rev. B 48, 22 (1993),
# as listed by Guedes-Sobrinho et al., Eur. Phys. J. B 91, 170 (2018), Table 1.
# The cross term is the arithmetic mean, their equation (4) at w = 0.5.
PLATINUM = (0.2975, 10.612, 4.004, 2.695, 2.775)
COPPER = (0.0855, 10.960, 2.278, 1.224, 2.556)
CROSS = (0.1915, 10.786, 3.141, 1.9595, 2.6655)
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("anneal", "gmin")
SCHEDULE_SEED = 20261113
HIT_TOL = 0.001

SUMMARY_FIELDS = (
    "system",
    "arm",
    "n",
    "hits",
    "hits_over_n",
    "budget",
    "reference_eV",
    "evals_median",
    "wall_median_s",
    "wall_ratio",
    "tts_median_s",
    "pot_per_eval_s",
    "outside_per_eval_s",
    "seeds",
    "schedule_seed",
    "anneal_head",
    "gmin_revision",
)
SEED_FIELDS = (
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "best",
    "charged",
    "wall_s",
    "pot_per_eval_s",
    "outside_per_eval_s",
)


def type7(sorted_values: list[float], p: float) -> float:
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def schedule() -> list[tuple[str, int, str]]:
    jobs = [(system, seed, arm) for system in PAPER_BUDGETS for seed in SEEDS for arm in ARMS]
    random.Random(SCHEDULE_SEED).shuffle(jobs)
    return jobs


def fmt_pair(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def data_text(system: str, steps: int, swaps: bool) -> str:
    n_pt = N_PLATINUM[system]
    n_atoms = N_ATOMS[system]
    if n_pt == 0:
        n_type, aa, ab, bb = n_atoms, COPPER, COPPER, COPPER
        swaps = False
    elif n_pt == n_atoms:
        n_type, aa, ab, bb = n_atoms, PLATINUM, PLATINUM, PLATINUM
        swaps = False
    else:
        n_type, aa, ab, bb = n_pt, PLATINUM, CROSS, COPPER
    lines = [
        f"BGUPTAT {n_type} {fmt_pair(aa)}",
        f"BGUPTATAB {fmt_pair(ab)}",
        f"BGUPTATBB {fmt_pair(bb)}",
        "SLOPPYCONV 5.0D-3",
        "TIGHTCONV 1.0D-3",
        "EDIFF 0.003",
        "MAXIT 600 500",
        "SAVE 2",
        "CENTRE",
        "RADIUS 50.0",
        "TEMPERATURE 1.0",
        f"STEPS {steps} 1.0",
        "STEP 0.40 0.51",
        "NEWRESTART 40",
    ]
    if swaps:
        lines.append("LSWAPS 8 1 1.0")
    lines.append("")
    return "\n".join(lines)


def run_gmin(
    gmin: Path,
    system: str,
    seed: int,
    budget: int,
    coords: Path,
    work: Path,
    log: Path,
) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    swaps = 0 < N_PLATINUM[system] < N_ATOMS[system]
    (work / "data").write_text(data_text(system, 90_000_000, swaps) + f"RANSEED {seed}\n")
    (work / "coords").write_bytes(coords.read_bytes())
    env = os.environ.copy()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "H2H_SYSTEM": system,
            "H2H_SEED": str(seed),
            "H2H_TARGET": "-1.0e99",
            "H2H_TOL": f"{HIT_TOL}",
            "H2H_ROW": str(work / "row.txt"),
            "OMP_NUM_THREADS": "1",
        }
    )
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(gmin)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=work,
        )
    row_path = work / "row.txt"
    if not row_path.is_file():
        tail = log.read_text()[-2000:]
        raise RuntimeError(f"gmin {system} seed {seed} exited {proc.returncode}\n{tail}")
    return parse_gmin(row_path.read_text())


def run_anneal(binary: Path, system: str, seed: int, budget: int, coords: Path, log: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "OMP_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
        }
    )
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "run", system, str(seed), str(coords)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal {system} seed {seed} exited {proc.returncode}\n{text[-2000:]}")
    return parse_anneal(text)


def parse_anneal(text: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if ",anneal," in line]
    if len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,"
        "pot_s,outside_s,pot_per_eval_s,outside_per_eval_s"
    )
    return next(csv.DictReader([header, data[0]]))


def parse_gmin(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in text.split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    if "best" not in fields:
        raise ValueError(f"no gmin row in {text!r}")
    return fields


def as_result(system: str, arm: str, row: dict[str, str]) -> dict:
    charged = max(int(float(row["charged"])), 1)
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = max(wall - pot, 0.0)
    return {
        "system": system,
        "arm": arm,
        "seed": int(float(row["seed"])),
        "budget": int(float(row["budget"])),
        "best": float(row["best"]),
        "charged": charged,
        "wall_s": wall,
        "pot_per": pot / charged,
        "out_per": outside / charged,
    }


def mark_hits(rows: list[dict]) -> dict[str, float]:
    lowest: dict[str, float] = {}
    for row in rows:
        lowest[row["system"]] = min(lowest.get(row["system"], row["best"]), row["best"])
    for row in rows:
        row["hit"] = row["best"] <= lowest[row["system"]] + HIT_TOL
    return lowest


def summarise(rows: list[dict], lowest: dict[str, float], head: str, revision: str) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        grouped.setdefault(row["system"], {}).setdefault(row["arm"], []).append(row)
    seed_text = " ".join(str(seed) for seed in SEEDS)
    out: list[dict[str, str]] = []
    for system in PAPER_BUDGETS:
        arms = grouped.get(system, {})
        walls = {arm: median([item["wall_s"] for item in group]) for arm, group in arms.items()}
        ratio = ""
        if "anneal" in walls and "gmin" in walls and walls["anneal"] > 0.0:
            ratio = f"{walls['gmin'] / walls['anneal']:.6f}"
        for arm in ARMS:
            group = arms.get(arm, [])
            if not group:
                continue
            hits = [item for item in group if item["hit"]]
            tts = f"{median([item['wall_s'] for item in hits]):.6f}" if hits else ""
            out.append(
                {
                    "system": system,
                    "arm": arm,
                    "n": str(len(group)),
                    "hits": str(len(hits)),
                    "hits_over_n": f"{len(hits)}/{len(group)}",
                    "budget": str(PAPER_BUDGETS[system]),
                    "reference_eV": f"{lowest[system]:.8f}",
                    "evals_median": f"{median([float(item['charged']) for item in group]):.6f}",
                    "wall_median_s": f"{median([item['wall_s'] for item in group]):.6f}",
                    "wall_ratio": ratio,
                    "tts_median_s": tts,
                    "pot_per_eval_s": f"{median([item['pot_per'] for item in group]):.9f}",
                    "outside_per_eval_s": f"{median([item['out_per'] for item in group]):.9f}",
                    "seeds": seed_text,
                    "schedule_seed": str(SCHEDULE_SEED),
                    "anneal_head": head,
                    "gmin_revision": revision,
                }
            )
    return out


def load_source(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("revision="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{path} has no revision")


def one_job(args, system: str, seed: int, arm: str) -> dict:
    budget = PAPER_BUDGETS[system]
    log = args.logs / system / f"{arm}-{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    coords = args.runs / system / f"start-{seed}.xyz"
    coords.parent.mkdir(parents=True, exist_ok=True)
    if not coords.is_file():
        subprocess.run(
            [str(args.binary), "coords", system, str(seed), str(coords)],
            check=True,
        )
    print(f"run {system} {arm} seed {seed}", flush=True)
    if arm == "anneal":
        row = run_anneal(args.binary, system, seed, budget, coords, log)
    else:
        row = run_gmin(
            args.gmin,
            system,
            seed,
            budget,
            coords,
            args.runs / system / f"gmin-{seed}",
            log,
        )
    return as_result(system, arm, row)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seeds-out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--only", nargs=3, metavar=("SYSTEM", "SEED", "ARM"))
    args = parser.parse_args(argv)
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    if Path.cwd() != Path("/work"):
        print(f"working directory is {Path.cwd()}, expected /work", file=sys.stderr)
        return 2
    if set(PAPER_BUDGETS) != set(N_PLATINUM):
        raise SystemExit("budgets and compositions differ")
    head = args.head_file.read_text().strip()
    revision = load_source(args.source)
    jobs = schedule()
    if args.only:
        system, seed_text, arm = args.only
        if system not in PAPER_BUDGETS or arm not in ARMS:
            raise SystemExit(f"unknown job {system} {arm}")
        one_job(args, system, int(seed_text), arm)
        return 0
    rows: list[dict] = []
    for system, seed, arm in jobs:
        rows.append(one_job(args, system, seed, arm))
        print(
            f"done {system} {arm} seed {seed} best={rows[-1]['best']:.6f} wall={rows[-1]['wall_s']:.3f}",
            flush=True,
        )
    lowest = mark_hits(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summarise(rows, lowest, head, revision))
    with args.seeds_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "system": row["system"],
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "budget": row["budget"],
                    "hit": int(row["hit"]),
                    "best": f"{row['best']:.8f}",
                    "charged": row["charged"],
                    "wall_s": f"{row['wall_s']:.6f}",
                    "pot_per_eval_s": f"{row['pot_per']:.9f}",
                    "outside_per_eval_s": f"{row['out_per']:.9f}",
                }
            )
    print(f"wrote {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
