#!/usr/bin/env python3
"""Run the PR 182 escape-4 performance gate serially on core 0."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = "0"
MIN_SAMPLE_S = 0.050
TARGET_CALIBRATE_S = 0.075

SHA_A = "cd6995e53ffe0e065fe3e4e6b63d2a9c481b904f"
SHA_B = "d41f0ae293884d1173d6ab1a7201e66749d3848a"

ARM_SHAS = {
    "A": SHA_A,
    "A2": SHA_A,
    "B": SHA_B,
    "B2": SHA_B,
}

ABBA_ORDERS = (
    ("A", "B", "B2", "A2"),
    ("B", "A", "A2", "B2"),
    ("A2", "B2", "B", "A"),
    ("B2", "A2", "A", "B"),
)

CELLS = (
    {"name": "lj98_short_d0", "seeds": (7,) * 12, "deferral": 0, "budget": 2500},
    {"name": "lj98_short_d4", "seeds": (7,) * 12, "deferral": 4, "budget": 2500},
    {"name": "lj98_traj_d0", "seeds": tuple(range(2000, 2012)), "deferral": 0, "budget": 5000},
    {"name": "lj98_traj_d4", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 5000},
    {"name": "lj38_short_d0", "seeds": tuple(range(2000, 2012)), "deferral": 0, "budget": 2500},
    {"name": "lj38_short_d4", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 2500},
    {"name": "lj38_d0", "seeds": tuple(range(2000, 2012)), "deferral": 0, "budget": 400000},
    {"name": "lj38_d4", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 400000},
    {"name": "lj55", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 1000000},
    {"name": "lj75", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 4000000},
    {"name": "lj98", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 4000000},
    {"name": "lj102", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 4000000},
    {"name": "lj103", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 4000000},
    {"name": "lj104", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 4000000},
    {"name": "lj150", "seeds": tuple(range(2000, 2012)), "deferral": 4, "budget": 4000000},
)

ROW_FIELDS = (
    "arm",
    "sha",
    "cell",
    "seed",
    "deferral",
    "hit",
    "calls_to_hit",
    "best_energy",
    "hops",
    "inner_reps",
    "outside_s",
    "rss_anon_kib",
    "core",
)

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENMP_NUM_THREADS",
)

ROW_RE = re.compile(
    r"^gate_row cell=(\S+) seed=(\d+) deferral=(\d+) hit=(\d+) "
    r"calls_to_hit=(\d+) best_energy=(\S+) hops=(\d+) inner_reps=(\d+) "
    r"wall_total_s=(\S+) pot_total_s=(\S+) outside_s=(\S+) rss_anon_kib=(\d+)$",
    re.M,
)


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in THREAD_VARS:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    return env


def run_driver(binary: Path, cell: str, seed: int, inner_reps: int, env: dict[str, str]) -> dict[str, str]:
    cmd = ["taskset", "-c", CORE, str(binary), cell, str(seed), str(inner_reps)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary} {cell} {seed} {inner_reps} failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    match = ROW_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"missing gate_row in output:\n{proc.stdout}")
    (
        out_cell,
        out_seed,
        deferral,
        hit,
        calls_to_hit,
        best_energy,
        hops,
        out_reps,
        wall_total_s,
        pot_total_s,
        outside_s,
        rss_anon_kib,
    ) = match.groups()
    if out_cell != cell or int(out_seed) != seed or int(out_reps) != inner_reps:
        raise RuntimeError(f"unexpected output echo: {match.group(0)}")
    return {
        "deferral": deferral,
        "hit": hit,
        "calls_to_hit": calls_to_hit,
        "best_energy": best_energy,
        "hops": hops,
        "inner_reps": out_reps,
        "wall_total_s": wall_total_s,
        "pot_total_s": pot_total_s,
        "outside_s": outside_s,
        "rss_anon_kib": rss_anon_kib,
    }


def calibrate_reps(binary_a: Path, cell: str, seed: int, budget: int, env: dict[str, str]) -> int:
    if budget >= 4_000_000:
        return 1
    reps = 1
    while True:
        res = run_driver(binary_a, cell, seed, reps, env)
        wall = float(res["wall_total_s"])
        if wall >= TARGET_CALIBRATE_S:
            return reps
        if wall <= 0.0:
            reps *= 8
        else:
            factor = max(2, int((TARGET_CALIBRATE_S / wall) + 1.0))
            reps *= factor
        if reps > 1024:
            return reps


def load_existing(path: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    by_cell_arm: dict[tuple[str, str], int] = {}
    round_arms: dict[tuple[str, int], dict[str, str]] = {}
    if not path.exists():
        return round_arms
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        cell = row["cell"]
        arm = row["arm"]
        idx = by_cell_arm.get((cell, arm), 0)
        by_cell_arm[(cell, arm)] = idx + 1
        round_arms.setdefault((cell, idx), {})[arm] = row["inner_reps"]
    return round_arms


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-a", type=Path, required=True)
    parser.add_argument("--bin-b", type=Path, required=True)
    parser.add_argument("--bin-a2", type=Path, required=True)
    parser.add_argument("--bin-b2", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=HERE / "rows.csv")
    parser.add_argument("--cells", default="")
    args = parser.parse_args(argv[1:])

    bins = {
        "A": args.bin_a.resolve(),
        "B": args.bin_b.resolve(),
        "A2": args.bin_a2.resolve(),
        "B2": args.bin_b2.resolve(),
    }
    if len(set(bins.values())) != 4:
        raise SystemExit("All four arm binaries must reside at distinct paths")
    for arm, path in bins.items():
        if not path.is_file():
            raise SystemExit(f"Missing binary for {arm}: {path}")

    wanted = set(args.cells.split(",")) if args.cells else {c["name"] for c in CELLS}
    selected = [c for c in CELLS if c["name"] in wanted]
    if len(selected) != len(wanted):
        raise SystemExit(f"Unknown cell in {args.cells}")

    if os.sched_getaffinity(0) != {0}:
        os.sched_setaffinity(0, {0})

    env = single_thread_env()
    done_rounds = load_existing(args.rows)

    for spec in selected:
        cell = str(spec["name"])
        seeds = spec["seeds"]
        budget = int(spec["budget"])
        for r, seed in enumerate(seeds):
            existing = done_rounds.get((cell, r), {})
            if len(existing) == 4:
                continue
            if existing:
                reps = int(next(iter(existing.values())))
            else:
                reps = calibrate_reps(bins["A"], cell, int(seed), budget, env)
            order = ABBA_ORDERS[r % 4]
            for arm in order:
                if arm in existing:
                    continue
                res = run_driver(bins[arm], cell, int(seed), reps, env)
                wall_total = float(res["wall_total_s"])
                if wall_total < MIN_SAMPLE_S:
                    raise RuntimeError(
                        f"{cell} r={r} arm={arm} wall_total_s={wall_total:.6f} < {MIN_SAMPLE_S}"
                    )
                row = {
                    "arm": arm,
                    "sha": ARM_SHAS[arm],
                    "cell": cell,
                    "seed": str(seed),
                    "deferral": res["deferral"],
                    "hit": res["hit"],
                    "calls_to_hit": res["calls_to_hit"],
                    "best_energy": res["best_energy"],
                    "hops": res["hops"],
                    "inner_reps": res["inner_reps"],
                    "outside_s": res["outside_s"],
                    "rss_anon_kib": res["rss_anon_kib"],
                    "core": CORE,
                }
                append_row(args.rows, row)
                existing[arm] = res["inner_reps"]
                print(
                    f"cell={cell} r={r:02d} seed={seed} arm={arm} hit={row['hit']} "
                    f"calls={row['calls_to_hit']} hops={row['hops']} reps={reps} "
                    f"wall={wall_total:.4f}s out={row['outside_s']}s rss={row['rss_anon_kib']}KiB",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
