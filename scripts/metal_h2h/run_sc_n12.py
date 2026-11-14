#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Twelve-seed Sutton–Chen head-to-head.

The systems are the ones already in the six-seed table: 12-6, 10-8 and
9-6 at 38, 55 and 80 atoms. Seeds are 2000–2011. A finished seed is
left in place. The six-seed files are not opened.

80-atom anneal runs first, then 80-atom GMIN, then the smaller sizes.
A hit is a quenched energy within 0.001 of the Cambridge table. Fisher
exact compares hit counts. Mann–Whitney compares calls to the hit and
wall time. Inside and outside time are the medians of those clocks.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sc as sc  # noqa: E402

SEEDS = tuple(range(2000, 2012))
EIGHTY = ("sc126-80", "sc96-80", "sc108-80")
SMALLER = tuple(name for name in sc.PAPER_BUDGETS if name not in EIGHTY)
SCHEDULE_SEED = "ordered-80-first"

SEED_FIELDS = (
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "best",
    "charged",
    "evals_to_hit",
    "wall_s",
    "pot_s",
    "outside_s",
    "pot_per_eval_s",
    "outside_per_eval_s",
    "final_e",
    "hops",
    "stall_escapes",
    "trail_escapes",
    "restarts",
    "paths",
    "path_improvements",
)
SUMMARY_FIELDS = (
    "system",
    "arm",
    "n",
    "hits",
    "hits_over_n",
    "budget",
    "reference",
    "evals_median",
    "wall_median_s",
    "pot_median_s",
    "outside_median_s",
    "wall_ratio",
    "tts_median_s",
    "pot_per_eval_s",
    "outside_per_eval_s",
    "fisher_p_vs_gmin",
    "mannwhitney_p_evals",
    "mannwhitney_p_wall",
    "seeds",
    "schedule_seed",
    "anneal_head",
    "gmin_revision",
)
ARM_FIELDS = (
    "system",
    "seed",
    "kind",
    "name",
    "draws",
    "accepts",
    "improved",
    "best",
)


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
    """Two-sided Mann–Whitney, normal approximation with the tie correction."""
    import math

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


def jobs() -> list[tuple[str, int, str]]:
    ordered: list[tuple[str, int, str]] = []
    for system in EIGHTY:
        for seed in SEEDS:
            ordered.append((system, seed, "anneal"))
    for system in EIGHTY:
        for seed in SEEDS:
            ordered.append((system, seed, "gmin"))
    for system in SMALLER:
        for seed in SEEDS:
            ordered.append((system, seed, "anneal"))
            ordered.append((system, seed, "gmin"))
    return ordered


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_diag(text: str) -> dict[str, str]:
    found = {
        "final_e": "",
        "hops": "",
        "stall_escapes": "",
        "trail_escapes": "",
        "restarts": "",
        "paths": "",
        "path_improvements": "",
    }
    arms: list[dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith("diag "):
            for part in line.split()[1:]:
                key, value = part.split("=", 1)
                if key == "final":
                    found["final_e"] = value
                elif key in found:
                    found[key] = value
        elif line.startswith("arm "):
            parts = line.split()
            # arm NAME draws N accepts N best ENERGY
            if len(parts) >= 8:
                arms.append(
                    {
                        "kind": "arm",
                        "name": parts[1],
                        "draws": parts[3],
                        "accepts": parts[5],
                        "improved": "",
                        "best": parts[7],
                    }
                )
        elif line.startswith("move "):
            parts = line.split()
            # move NAME proposed N accepted N improved N
            if len(parts) >= 8:
                arms.append(
                    {
                        "kind": "move",
                        "name": parts[1],
                        "draws": parts[3],
                        "accepts": parts[5],
                        "improved": parts[7],
                        "best": "",
                    }
                )
    found["_arms"] = arms  # type: ignore[assignment]
    return found


def as_seed(system: str, arm: str, row: dict, diag: dict[str, str]) -> dict[str, str]:
    charged = max(int(float(row["charged"])), 1)
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = max(wall - pot, 0.0)
    hit = row.get("hit", "0") in {"1", "1.0", "true", "True"}
    return {
        "system": system,
        "arm": arm,
        "seed": str(int(float(row["seed"]))),
        "budget": str(int(float(row["budget"]))),
        "hit": "1" if hit else "0",
        "best": f"{float(row['best']):.8f}",
        "charged": str(charged),
        "evals_to_hit": str(int(float(row.get("evals_to_hit", charged)))),
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "pot_per_eval_s": f"{pot / charged:.9f}",
        "outside_per_eval_s": f"{outside / charged:.9f}",
        "final_e": diag.get("final_e", ""),
        "hops": diag.get("hops", ""),
        "stall_escapes": diag.get("stall_escapes", ""),
        "trail_escapes": diag.get("trail_escapes", ""),
        "restarts": diag.get("restarts", ""),
        "paths": diag.get("paths", ""),
        "path_improvements": diag.get("path_improvements", ""),
    }


def run_anneal(binary: Path, system: str, seed: int, budget: int, coords: Path, log: Path, runs: Path) -> tuple[dict, dict]:
    env = os.environ.copy()
    best = runs / system / f"best-{seed}.xyz"
    final = runs / system / f"final-{seed}.xyz"
    best.parent.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "H2H_BEST": str(best),
            "H2H_FINAL": str(final),
            "OMP_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    import subprocess

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
    parsed = sc.parse_anneal(text)
    diag = parse_diag(text)
    return parsed, diag


def summarise(rows: list[dict[str, str]], head: str, revision: str) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        grouped.setdefault(row["system"], {}).setdefault(row["arm"], []).append(row)
    seed_text = " ".join(str(seed) for seed in SEEDS)
    out: list[dict[str, str]] = []
    for system in list(EIGHTY) + list(SMALLER):
        arms = grouped.get(system, {})
        walls = {
            arm: sc.median([float(item["wall_s"]) for item in group]) for arm, group in arms.items()
        }
        ratio = ""
        if "anneal" in walls and "gmin" in walls and walls["anneal"] > 0.0:
            ratio = f"{walls['gmin'] / walls['anneal']:.6f}"
        anneal = arms.get("anneal", [])
        gmin = arms.get("gmin", [])
        fisher_p = ""
        mw_evals = ""
        mw_wall = ""
        if anneal and gmin:
            fisher_p = f"{fisher(sum(r['hit'] == '1' for r in anneal), len(anneal), sum(r['hit'] == '1' for r in gmin), len(gmin)):.6g}"
            mw_evals = f"{mann_whitney([float(r['evals_to_hit']) for r in anneal], [float(r['evals_to_hit']) for r in gmin]):.6g}"
            mw_wall = f"{mann_whitney([float(r['wall_s']) for r in anneal], [float(r['wall_s']) for r in gmin]):.6g}"
        for arm in ("anneal", "gmin"):
            group = arms.get(arm, [])
            if not group:
                continue
            hits = [item for item in group if item["hit"] == "1"]
            tts = f"{sc.median([float(item['wall_s']) for item in hits]):.6f}" if hits else ""
            out.append(
                {
                    "system": system,
                    "arm": arm,
                    "n": str(len(group)),
                    "hits": str(len(hits)),
                    "hits_over_n": f"{len(hits)}/{len(group)}",
                    "budget": str(sc.PAPER_BUDGETS[system]),
                    "reference": f"{sc.REFERENCES[system]:.4f}",
                    "evals_median": f"{sc.median([float(item['evals_to_hit']) for item in group]):.6f}",
                    "wall_median_s": f"{sc.median([float(item['wall_s']) for item in group]):.6f}",
                    "pot_median_s": f"{sc.median([float(item['pot_s']) for item in group]):.6f}",
                    "outside_median_s": f"{sc.median([float(item['outside_s']) for item in group]):.6f}",
                    "wall_ratio": ratio if arm == "anneal" else ratio,
                    "tts_median_s": tts,
                    "pot_per_eval_s": f"{sc.median([float(item['pot_per_eval_s']) for item in group]):.9f}",
                    "outside_per_eval_s": f"{sc.median([float(item['outside_per_eval_s']) for item in group]):.9f}",
                    "fisher_p_vs_gmin": fisher_p if arm == "anneal" else "",
                    "mannwhitney_p_evals": mw_evals if arm == "anneal" else "",
                    "mannwhitney_p_wall": mw_wall if arm == "anneal" else "",
                    "seeds": seed_text,
                    "schedule_seed": SCHEDULE_SEED,
                    "anneal_head": head,
                    "gmin_revision": revision,
                }
            )
    return out


def one_job(args, system: str, seed: int, arm: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    budget = sc.PAPER_BUDGETS[system]
    log = args.logs / system / f"{arm}-{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    coords = args.runs / system / f"start-{seed}.xyz"
    coords.parent.mkdir(parents=True, exist_ok=True)
    if not coords.is_file():
        import subprocess

        subprocess.run(
            [str(args.binary), "coords", system, str(seed), str(coords)],
            check=True,
        )
    print(f"run {system} {arm} seed {seed}", flush=True)
    if arm == "anneal":
        row, diag = run_anneal(args.binary, system, seed, budget, coords, log, args.runs)
        arms = diag.pop("_arms", [])
    else:
        row = sc.run_gmin(
            args.gmin,
            system,
            seed,
            budget,
            coords,
            args.runs / system / f"gmin-{seed}",
            log,
        )
        diag = {}
        arms = []
    seed_row = as_seed(system, arm, row, diag)
    arm_rows = [
        {
            "system": system,
            "seed": str(seed),
            **item,
        }
        for item in arms
    ]
    return seed_row, arm_rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seeds-out", type=Path, required=True)
    parser.add_argument("--arms-out", type=Path)
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
    head = args.head_file.read_text().strip()
    revision = sc.load_source(args.source)
    rows = load_rows(args.seeds_out)
    arm_rows = load_rows(args.arms_out) if args.arms_out else []
    done = {(row["system"], row["seed"], row["arm"]) for row in rows}

    def store(seed_row: dict[str, str], extra_arms: list[dict[str, str]]) -> None:
        key = (seed_row["system"], seed_row["seed"], seed_row["arm"])
        if key in done:
            return
        rows.append(seed_row)
        done.add(key)
        arm_rows.extend(extra_arms)
        write_rows(args.seeds_out, rows, SEED_FIELDS)
        if args.arms_out:
            write_rows(args.arms_out, arm_rows, ARM_FIELDS)
        write_rows(args.summary, summarise(rows, head, revision), SUMMARY_FIELDS)

    if args.only:
        system, seed_text, arm = args.only
        if system not in sc.PAPER_BUDGETS or arm not in sc.ARMS:
            raise SystemExit(f"unknown job {system} {arm}")
        seed_row, extra = one_job(args, system, int(seed_text), arm)
        # A timing run does not enter the stored table.
        print(
            f"done {system} {arm} seed {seed_text} best={seed_row['best']} hit={seed_row['hit']}",
            flush=True,
        )
        return 0
    for system, seed, arm in jobs():
        if (system, str(seed), arm) in done:
            print(f"skip {system} {arm} seed {seed}", flush=True)
            continue
        seed_row, extra = one_job(args, system, seed, arm)
        store(seed_row, extra)
        print(
            f"done {system} {arm} seed {seed} best={seed_row['best']} hit={seed_row['hit']} wall={seed_row['wall_s']}",
            flush=True,
        )
    print(f"wrote {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
