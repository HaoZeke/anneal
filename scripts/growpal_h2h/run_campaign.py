#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GrowPAL against the paper preset, at the paper evaluation budgets.

Sizes are LJ38, LJ75, LJ98, and LJ104. Six seeds, the same list on both
arms. Within a size the (seed, arm) pairs are shuffled. A hit is a
quenched energy within 0.001 of the Cambridge minimum. A miss is stored
as the full budget, and the median and interquartile range use
Hyndman-Fan type 7 on that column. The wall ratio is the GrowPAL median
wall divided by the anneal median wall.

Inside microseconds are the potential time divided by the charged
evaluations. Outside microseconds are the remainder of the wall, on the
same denominator. This process refuses to run unless the UTS hostname
is bench.
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
    38: 400_000,
    75: 4_000_000,
    98: 4_000_000,
    104: 4_000_000,
}
SIZES = (38, 75, 98, 104)
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("anneal", "growpal")
SCHEDULE_SEED = 20261112
HIT_TOL = 0.001

SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "planned",
    "hits",
    "hits_over_n",
    "budget",
    "evals_median",
    "evals_iqr",
    "wall_median_s",
    "wall_iqr_s",
    "wall_ratio",
    "inside_us",
    "outside_us",
    "seeds",
    "schedule_seed",
    "growpal_version",
    "aegon_version",
    "growpal_source",
    "anneal_head",
]

SEED_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "pot_s",
    "inside_us",
    "outside_us",
    "best",
    "stop",
    "reached",
]


def type7(sorted_values: list[float], p: float) -> float:
    """Hyndman-Fan type 7. ``sorted_values`` is ascending."""
    n = len(sorted_values)
    if n < 1:
        raise ValueError("type 7 needs a sample")
    if n == 1:
        return sorted_values[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    q1 = type7(ordered, 0.25)
    q3 = type7(ordered, 0.75)
    return type7(ordered, 0.5), q3 - q1


def schedule(
    sizes: tuple[int, ...],
    seeds: tuple[int, ...],
    arms: tuple[str, ...],
    rng_seed: int,
) -> list[tuple[int, int, str]]:
    """Sizes in the given order. Pairs inside a size are shuffled."""
    rng = random.Random(rng_seed)
    order: list[tuple[int, int, str]] = []
    for n in sizes:
        jobs = [(n, seed, arm) for seed in seeds for arm in arms]
        rng.shuffle(jobs)
        order.extend(jobs)
    return order


def system_name(n: int) -> str:
    return f"lj{n}"


def load_source(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    for key in ("growpal_version", "aegon_version", "growpal_source"):
        if not fields.get(key):
            raise SystemExit(f"{path} has no {key}")
    return fields


def binary_budget(binary: Path, n: int) -> tuple[int, float]:
    proc = subprocess.run(
        [str(binary), "budget", system_name(n)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"budget for LJ{n} returned {proc.stdout!r}")
    return int(lines[0]), float(lines[1])


def _env(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMBA_NUM_THREADS"] = "1"
    return env


def parse_anneal(text: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if ",anneal," in line]
    if len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,"
        "cpu,pins,pot_s"
    )
    row = next(csv.DictReader([header, data[0]]))
    hit = "1" if row["hit"] in ("1", "true") else "0"
    if row["hit"] not in ("0", "1", "true", "false"):
        raise ValueError(f"anneal hit {row['hit']}")
    stop = "hit" if hit == "1" else "budget"
    return {
        "system": row["system"],
        "arm": "anneal",
        "seed": row["seed"],
        "budget": row["budget"],
        "hit": hit,
        "evals_to_hit": row["evals_to_hit"],
        "charged": row["charged"],
        "wall_s": row["wall_s"],
        "pot_s": row["pot_s"],
        "best": row["best"],
        "stop": stop,
        "reached": row["system"][2:],
    }


def with_rates(row: dict[str, str]) -> dict[str, str]:
    from growpal_arm import per_eval

    inside, outside = per_eval(float(row["wall_s"]), float(row["pot_s"]), int(row["charged"]))
    out = dict(row)
    out["inside_us"] = f"{inside:.4f}"
    out["outside_us"] = f"{outside:.4f}"
    return out


def write_seed(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_FIELDS)
        writer.writeheader()
        writer.writerow({key: row[key] for key in SEED_FIELDS})


def load_seed(path: Path) -> dict[str, str] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path} has {len(rows)} rows")
    return rows[0]


def run_anneal(binary: Path, n: int, seed: int, budget: int, log: Path) -> dict[str, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "run", system_name(n), str(seed)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=_env(budget),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal LJ{n} seed {seed} exited {proc.returncode}\n{text[-500:]}")
    row = parse_anneal(text)
    if int(row["budget"]) != budget:
        raise RuntimeError(f"anneal budget {row['budget']} != {budget}")
    return with_rates(row)


def run_growpal_seed(
    n: int, seed: int, budget: int, reference: float, seeds_path: str, log: Path
) -> dict[str, str]:
    import growpal_arm

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", buffering=1) as handle:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = handle
        sys.stderr = handle
        try:
            row = growpal_arm.run_growpal(n, seed, budget, reference, seeds_path)
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
    row["seed"] = str(seed)
    row["system"] = system_name(n)
    if int(row["charged"]) > budget:
        raise RuntimeError(f"growpal charged {row['charged']} past {budget}")
    return with_rates(row)


def summarise(rows: list[dict[str, str]], seeds: tuple[int, ...], source: dict[str, str], head: str):
    by_n: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        by_n.setdefault(row["system"], {}).setdefault(row["arm"], []).append(row)
    seed_text = " ".join(str(seed) for seed in seeds)
    out: list[dict[str, str]] = []
    for system in sorted(by_n, key=lambda name: int(name[2:])):
        arms = by_n[system]
        walls = {
            arm: median_iqr([float(item["wall_s"]) for item in group])[0]
            for arm, group in arms.items()
            if group
        }
        ratio = ""
        if "anneal" in walls and "growpal" in walls and walls["anneal"] > 0.0:
            ratio = f"{walls['growpal'] / walls['anneal']:.6f}"
        for arm in ARMS:
            group = arms.get(arm, [])
            if not group:
                continue
            evals_median, evals_iqr = median_iqr(
                [float(item["evals_to_hit"]) for item in group]
            )
            wall_median, wall_iqr = median_iqr([float(item["wall_s"]) for item in group])
            inside_median, _inside_iqr = median_iqr(
                [float(item["inside_us"]) for item in group]
            )
            outside_median, _outside_iqr = median_iqr(
                [float(item["outside_us"]) for item in group]
            )
            hits = sum(item["hit"] == "1" for item in group)
            out.append(
                {
                    "system": system,
                    "arm": arm,
                    "n": str(len(group)),
                    "planned": str(len(seeds)),
                    "hits": str(hits),
                    "hits_over_n": f"{hits}/{len(group)}",
                    "budget": group[0]["budget"],
                    "evals_median": f"{evals_median:.6f}",
                    "evals_iqr": f"{evals_iqr:.6f}",
                    "wall_median_s": f"{wall_median:.6f}",
                    "wall_iqr_s": f"{wall_iqr:.6f}",
                    "wall_ratio": ratio,
                    "inside_us": f"{inside_median:.4f}",
                    "outside_us": f"{outside_median:.4f}",
                    "seeds": seed_text,
                    "schedule_seed": str(SCHEDULE_SEED),
                    "growpal_version": source["growpal_version"],
                    "aegon_version": source["aegon_version"],
                    "growpal_source": source["growpal_source"],
                    "anneal_head": head,
                }
            )
    return out


def write_table(path: Path, fields: list[str], rows: list[dict[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def note(progress: Path, line: str) -> None:
    print(line, flush=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a") as handle:
        handle.write(line + "\n")


def self_test() -> None:
    from growpal_arm import per_eval

    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12:
        raise SystemExit("type 7 lower quartile")
    if abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7 median")
    median, iqr = median_iqr(sample)
    if abs(median - 3.5) > 1e-12 or abs(iqr - 2.5) > 1e-12:
        raise SystemExit("median iqr")
    if PAPER_BUDGETS != {38: 400_000, 75: 4_000_000, 98: 4_000_000, 104: 4_000_000}:
        raise SystemExit("paper budgets")
    if abs(HIT_TOL - 0.001) > 1e-15:
        raise SystemExit("hit tolerance")
    inside, outside = per_eval(1.0, 0.25, 1000)
    if abs(inside - 250.0) > 1e-9 or abs(outside - 750.0) > 1e-9:
        raise SystemExit("per-evaluation split")
    order = schedule(SIZES, SEEDS, ARMS, SCHEDULE_SEED)
    seen: list[int] = []
    for n, _seed, _arm in order:
        if not seen or seen[-1] != n:
            seen.append(n)
    if seen != list(SIZES):
        raise SystemExit(f"size order {seen}")
    for n in SIZES:
        jobs = [(seed, arm) for size, seed, arm in order if size == n]
        expected = [(seed, arm) for seed in SEEDS for arm in ARMS]
        if sorted(jobs) != sorted(expected):
            raise SystemExit(f"LJ{n} is not the shared seed list")
        arms = [arm for _seed, arm in jobs]
        if arms == ["anneal"] * len(SEEDS) + ["growpal"] * len(SEEDS):
            raise SystemExit(f"LJ{n} is not interleaved")
        if arms == ["growpal"] * len(SEEDS) + ["anneal"] * len(SEEDS):
            raise SystemExit(f"LJ{n} is not interleaved")
    reference = -173.928427
    level = reference + HIT_TOL
    if abs(level - (reference + 0.001)) > 1e-15:
        raise SystemExit("hit level")
    if reference + 0.002 <= level:
        raise SystemExit("hit level is wider than 0.001")
    print("ok")


def seed_path(rows: Path, n: int, seed: int, arm: str) -> Path:
    return rows / system_name(n) / f"{arm}-{seed}.row"


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seeds-xyz", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--seeds-out", type=Path)
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", choices=("anneal", "growpal", "both"), default="both")
    parser.add_argument("--budget", type=int)
    args = parser.parse_args(argv[1:])
    sizes = tuple(args.size) if args.size else SIZES
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = ARMS if args.arm == "both" else (args.arm,)
    for n in sizes:
        if n not in PAPER_BUDGETS:
            raise SystemExit(f"no paper budget for LJ{n}")
        if args.budget is not None and args.budget != PAPER_BUDGETS[n]:
            raise SystemExit(
                f"LJ{n} budget {args.budget} is not the paper budget {PAPER_BUDGETS[n]}"
            )
    source = load_source(args.source)
    head = args.head_file.read_text().strip()
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import growpal_arm

    growpal_arm.assert_budget_stops_lbfgs()
    progress = args.runs.parent / "progress.txt"
    order = schedule(sizes, seeds, arms, SCHEDULE_SEED)
    note(progress, "schedule " + " ".join(f"lj{n}:{arm}:{seed}" for n, seed, arm in order))
    rows: list[dict[str, str]] = []
    produced = False
    for n, seed, arm in order:
        budget = PAPER_BUDGETS[n]
        binary_calls, reference = binary_budget(args.binary, n)
        if binary_calls != budget:
            raise SystemExit(f"binary budget {binary_calls} != paper budget {budget}")
        saved = seed_path(args.rows, n, seed, arm)
        log = args.runs / system_name(n) / f"{arm}-{seed}.log"
        existing = load_seed(saved)
        if existing is not None:
            rows.append(existing)
            note(
                progress,
                f"skip n={n} arm={arm} seed={seed} hit={existing['hit']} "
                f"evals={existing['evals_to_hit']}",
            )
            continue
        note(progress, f"start n={n} arm={arm} seed={seed} budget={budget}")
        if arm == "anneal":
            row = run_anneal(args.binary, n, seed, budget, log)
        else:
            row = run_growpal_seed(n, seed, budget, reference, str(args.seeds_xyz), log)
        write_seed(saved, row)
        produced = True
        rows.append(row)
        note(
            progress,
            f"done n={n} arm={arm} seed={seed} hit={row['hit']} "
            f"evals={row['evals_to_hit']} charged={row['charged']} "
            f"wall_s={row['wall_s']} pot_s={row['pot_s']} best={row['best']} "
            f"stop={row['stop']} inside_us={row['inside_us']} outside_us={row['outside_us']}",
        )
        text_rows = summarise(rows, seeds, source, head)
        if args.seeds_out:
            write_table(args.seeds_out, SEED_FIELDS, rows, overwrite=True)
        if args.summary and not (args.summary.exists() and not produced):
            write_table(args.summary, SUMMARY_FIELDS, text_rows, overwrite=True)
    text_rows = summarise(rows, seeds, source, head)
    if args.seeds_out and (produced or not args.seeds_out.exists()):
        write_table(args.seeds_out, SEED_FIELDS, rows, overwrite=True)
    if args.summary and produced:
        write_table(args.summary, SUMMARY_FIELDS, text_rows, overwrite=True)
    elif args.summary and not args.summary.exists():
        write_table(args.summary, SUMMARY_FIELDS, text_rows, overwrite=True)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(text_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
