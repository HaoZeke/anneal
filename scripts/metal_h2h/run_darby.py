#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Anneal against GMIN on Darby Cu-Au, at the paper call budgets.

GMIN's binary Gupta keyword is BGUPTAT. The parameter order on that
keyword is A, p, q, xi, r0. Copper, gold and the cross term are the
Darby, Mortimer-Jones, Johnston and Roberts, J. Chem. Phys. 116, 1536
(2002), rows. A hit is a quenched energy within 0.001 eV of the
Cambridge energy. The size budgets are the Lennard-Jones paper map:
400000 calls at 38 atoms and 1000000 at 55. Six seeds, one job at a
time, in one shuffled order. The process refuses to run unless the
UTS hostname is bench and the working directory is /work.

Time to solution is the median wall time of the seeds that hit. The
wall ratio is the GMIN median wall divided by the anneal median wall.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

PAPER_BUDGETS = {
    "au38": 400_000,
    "cu6au32": 400_000,
    "cu19au19": 400_000,
    "au55": 1_000_000,
    "cu27au28": 1_000_000,
}
REFERENCES = {
    "au38": -130.54001708,
    "cu6au32": -130.79994836,
    "cu19au19": -122.99148391,
    "au55": -191.03085280,
    "cu27au28": -182.93136675,
}
COPPER_COUNT = {
    "au38": 0,
    "cu6au32": 6,
    "cu19au19": 19,
    "au55": 0,
    "cu27au28": 27,
}
N_ATOMS = {
    "au38": 38,
    "cu6au32": 38,
    "cu19au19": 38,
    "au55": 55,
    "cu27au28": 55,
}
FIXTURE = {
    "au38": ("cuau38", "Cu0Au38.txt"),
    "cu6au32": ("cuau38", "Cu6Au32.txt"),
    "cu19au19": ("cuau38", "Cu19Au19.txt"),
    "au55": ("cuau55", "Cu0Au55.txt"),
    "cu27au28": ("cuau55", "Cu27Au28.txt"),
}
# A, p, q, xi, r0. Darby Cu-Au.
COPPER = (0.0855, 10.960, 2.278, 1.224, 2.556)
GOLD = (0.2061, 10.229, 4.036, 1.790, 2.884)
CROSS = (0.1539, 11.05, 3.0475, 1.5605, 2.556)
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("anneal", "gmin")
SCHEDULE_SEED = 20261113
HIT_TOL = 0.001
# A quench of a Cambridge minimum may move by the optimiser tolerance.
# A wrong Gupta formula misses by electronvolts.
GMIN_AGREE = 1e-4

SUMMARY_FIELDS = (
    "system",
    "arm",
    "n",
    "hits",
    "hits_over_n",
    "budget",
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
    n_copper = COPPER_COUNT[system]
    n_atoms = N_ATOMS[system]
    if n_copper == 0:
        n_type, aa, ab, bb = n_atoms, GOLD, GOLD, GOLD
        swaps = False
    elif n_copper == n_atoms:
        n_type, aa, ab, bb = n_atoms, COPPER, COPPER, COPPER
        swaps = False
    else:
        n_type, aa, ab, bb = n_copper, COPPER, CROSS, GOLD
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
        # The default container is sized for Lennard-Jones. It radially
        # resets any atom outside it, which moves a Gupta geometry and
        # changes the energy. 50 A leaves the 38- and 55-atom clusters,
        # and ordinary basin-hopping steps, inside the container.
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


def reorder_copper_then_gold(path: Path) -> str:
    copper: list[str] = []
    gold: list[str] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 4 or fields[0] not in {"0", "1"}:
            continue
        row = f"{float(fields[1]):.10f} {float(fields[2]):.10f} {float(fields[3]):.10f}\n"
        (copper if fields[0] == "0" else gold).append(row)
    if not copper and not gold:
        raise SystemExit(f"{path} has no typed coordinates")
    return "".join(copper + gold)


def anneal_energy(binary: Path, system: str, coords: Path) -> float:
    proc = subprocess.run(
        [str(binary), "energy", system, str(coords)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip().splitlines()[-1])


def check_potentials(binary: Path, gmin: Path, fixtures: Path, runs: Path) -> None:
    for system, reference in REFERENCES.items():
        source = fixtures.joinpath(*FIXTURE[system])
        work = runs / "agree" / system
        work.mkdir(parents=True, exist_ok=True)
        coords = work / "coords"
        coords.write_text(reorder_copper_then_gold(source))
        got = anneal_energy(binary, system, coords)
        if abs(got - reference) > 1e-6:
            raise SystemExit(
                f"{system} anneal energy {got:.10f} is not the Cambridge {reference:.8f}"
            )
        print(f"agree anneal {system} {got:.10f} d {got - reference:.3e}", flush=True)
        row = run_gmin(
            gmin,
            system,
            seed=1,
            budget=100_000,
            target=reference,
            coords=coords,
            work=work,
            log=work / "gmin.log",
            steps=1,
            swaps=False,
        )
        best = float(row["best"])
        gap = abs(best - reference)
        print(f"agree gmin {system} {best:.10f} d {best - reference:.3e}", flush=True)
        if gap > GMIN_AGREE:
            raise SystemExit(
                f"{system} GMIN energy {best:.10f} disagrees with Cambridge {reference:.8f} by {gap:.3e}"
            )


def run_gmin(
    gmin: Path,
    system: str,
    seed: int,
    budget: int,
    target: float,
    coords: Path,
    work: Path,
    log: Path,
    steps: int,
    swaps: bool,
) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    (work / "data").write_text(data_text(system, steps, swaps) + f"RANSEED {seed}\n")
    (work / "coords").write_bytes(coords.read_bytes())
    env = os.environ.copy()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "H2H_SYSTEM": system,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{target:.10f}",
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


def run_anneal(
    binary: Path,
    system: str,
    seed: int,
    budget: int,
    coords: Path,
    log: Path,
) -> dict[str, str]:
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
        raise RuntimeError(f"anneal {system} seed {seed} exited {proc.returncode}")
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
    if "hit" not in fields or "best" not in fields:
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
        "hit": row["hit"] in ("1", "true"),
        "evals": float(row["evals_to_hit"]),
        "wall_s": wall,
        "pot_per": pot / charged,
        "out_per": outside / charged,
    }


def summarise(rows: list[dict], head: str, revision: str) -> list[dict[str, str]]:
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
                    "budget": str(group[0]["budget"]),
                    "evals_median": f"{median([item['evals'] for item in group]):.6f}",
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    if Path.cwd() != Path("/work"):
        print(f"working directory is {Path.cwd()}, expected /work", file=sys.stderr)
        return 2
    if set(PAPER_BUDGETS) != set(REFERENCES):
        raise SystemExit("budgets and references differ")
    head = args.head_file.read_text().strip()
    revision = load_source(args.source)
    check_potentials(args.binary, args.gmin, args.fixtures, args.runs)
    rows: list[dict] = []
    for system, seed, arm in schedule():
        budget = PAPER_BUDGETS[system]
        log = args.logs / system / f"{arm}-{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        saved = args.runs / system / f"{arm}-{seed}.row"
        coords = args.runs / system / f"start-{seed}.xyz"
        coords.parent.mkdir(parents=True, exist_ok=True)
        if not coords.is_file():
            subprocess.run(
                [str(args.binary), "coords", system, str(seed), str(coords)],
                check=True,
            )
        if saved.is_file() and saved.stat().st_size > 0:
            text = saved.read_text()
            row = parse_gmin(text) if arm == "gmin" else parse_anneal(text)
        else:
            print(f"run {system} {arm} seed {seed}", flush=True)
            if arm == "anneal":
                row = run_anneal(args.binary, system, seed, budget, coords, log)
                saved.write_text(log.read_text())
            else:
                row = run_gmin(
                    args.gmin,
                    system,
                    seed,
                    budget,
                    REFERENCES[system],
                    coords,
                    args.runs / system / f"gmin-{seed}",
                    log,
                    steps=90_000_000,
                    swaps=0 < COPPER_COUNT[system] < N_ATOMS[system],
                )
                saved.write_text((args.runs / system / f"gmin-{seed}" / "row.txt").read_text())
        result = as_result(system, arm, row)
        if result["budget"] != budget:
            raise RuntimeError(f"{system} {arm} budget {result['budget']} != {budget}")
        rows.append(result)
        print(
            f"done {system} {arm} seed {seed} hit={int(result['hit'])} "
            f"wall={result['wall_s']:.3f}",
            flush=True,
        )
    summary = summarise(rows, head, revision)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
