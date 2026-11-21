#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One pinned seed of a head-to-head binary.

The binary is the gate driver. Thread counts are one. No quick flag.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

THREAD_KEYS = (
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
    "JULIA_NUM_THREADS",
    "TBB_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENMP_NUM_THREADS",
)


def child_env(trace: str | None, sc_lib: str | None) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("H2H_") or key.startswith("ANNEAL_"):
            del env[key]
        elif key.endswith(("_NUM_THREADS", "_MAX_THREADS", "_MAXIMUM_THREADS")):
            env[key] = "1"
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["ANNEAL_MOVE_COUNT"] = "1"
    if trace:
        env["ANNEAL_STEP_TRACE"] = trace
    # Filled by the caller when a short identity run is requested.
    budget = os.environ.get("PROFILE_BUDGET")
    if budget:
        env["H2H_BUDGET"] = budget
    if sc_lib:
        env["RGPOT_SC_LIB"] = sc_lib
        lib_dir = os.path.dirname(os.path.abspath(sc_lib))
        env["LD_LIBRARY_PATH"] = lib_dir + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    return env


def parse(text: str, system: str) -> tuple[dict[str, str], list[list[str]], str]:
    row = None
    tallies: list[list[str]] = []
    best_bits = ""
    for line in text.splitlines():
        if line.startswith("tally,"):
            tallies.append(line.split(","))
            if line.startswith("tally,best,"):
                best_bits = line.split(",")[2]
        elif line.startswith("best_bits,"):
            best_bits = line.split(",", 1)[1].strip()
        elif line.startswith(system + ","):
            row = line
    if row is None:
        raise SystemExit("no result row")
    values = row.split(",")
    # Lennard-Jones field 9 is a commit id. Sutton-Chen field 9 is pot_s.
    if len(values) >= 14 and "." not in values[9]:
        pot_s = values[13]
    else:
        pot_s = values[9]
    parsed = {
        "budget": values[3],
        "hit": values[4],
        "evals_to_hit": values[5],
        "charged": values[6],
        "wall_s": values[7],
        "best": values[8],
        "pot_s": pot_s,
    }
    return parsed, tallies, best_bits


def append_csv(path: str, fields: list[str], row: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sc-lib")
    parser.add_argument("--trace")
    parser.add_argument("--perf")
    parser.add_argument("--perf-data")
    parser.add_argument("--cpu", default="0")
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    if args.system == "lj104":
        command = [args.binary, "run", "lj104", str(args.seed)]
    elif args.system == "sc108-80":
        command = [args.binary, "run", "sc108-80", str(args.seed)]
    else:
        raise SystemExit("unknown system")
    if args.budget:
        os.environ["PROFILE_BUDGET"] = str(args.budget)
    else:
        os.environ.pop("PROFILE_BUDGET", None)
    if args.perf:
        if not args.perf_data:
            raise SystemExit("perf data path required")
        command = [
            args.perf,
            "record",
            "-g",
            "--call-graph",
            "fp",
            "-e",
            "cpu-clock",
            "-F",
            "399",
            "-o",
            args.perf_data,
            "--",
            *command,
        ]
    command = ["taskset", "-c", args.cpu, *command]
    proc = subprocess.run(
        command,
        env=child_env(args.trace, args.sc_lib),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(proc.returncode)
    parsed, tallies, best_bits = parse(proc.stdout, args.system)
    charged = int(parsed["charged"])
    wall = float(parsed["wall_s"])
    pot = float(parsed["pot_s"])
    outside = wall - pot
    if outside < 0.0:
        outside = 0.0
    per_call = outside / charged if charged else 0.0
    run_fields = [
        "arm",
        "system",
        "seed",
        "budget",
        "hit",
        "evals_to_hit",
        "charged",
        "wall_s",
        "pot_s",
        "outside_s",
        "outside_per_call",
        "best",
        "best_bits",
    ]
    append_csv(
        os.path.join(args.out_dir, "runs.csv"),
        run_fields,
        {
            "arm": args.arm,
            "system": args.system,
            "seed": str(args.seed),
            "budget": parsed["budget"],
            "hit": parsed["hit"],
            "evals_to_hit": parsed["evals_to_hit"],
            "charged": str(charged),
            "wall_s": f"{wall:.6f}",
            "pot_s": f"{pot:.6f}",
            "outside_s": f"{outside:.6f}",
            "outside_per_call": f"{per_call:.9e}",
            "best": parsed["best"],
            "best_bits": best_bits,
        },
    )
    move_fields = [
        "arm",
        "system",
        "seed",
        "kind",
        "name",
        "proposed",
        "accepted",
        "charged",
    ]
    for parts in tallies:
        if len(parts) < 6 or parts[1] not in ("move", "gap", "total"):
            continue
        append_csv(
            os.path.join(args.out_dir, "move_counts.csv"),
            move_fields,
            {
                "arm": args.arm,
                "system": args.system,
                "seed": str(args.seed),
                "kind": parts[1],
                "name": parts[2],
                "proposed": parts[3],
                "accepted": parts[4],
                "charged": parts[5],
            },
        )


if __name__ == "__main__":
    main()
