#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial paired gate runner for Arm A (develop) vs Arm B (fix5 patch)."""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import time
from pathlib import Path

HIT_TOL = 0.001
SEEDS_12 = tuple(range(2000, 2012))

# Order keeps shorter panels before blj130n98.
SYSTEMS = (
    # panel, system, driver, budget, reference, seeds
    ("blj", "blj105n38", "h2h_blj", 400_000, -177.2090846748, SEEDS_12),
    ("blj", "blj105n55", "h2h_blj", 1_000_000, -283.1545313810, SEEDS_12),
    ("blj", "blj115n38", "h2h_blj", 400_000, -187.3676867453, SEEDS_12),
    ("blj", "blj130n38", "h2h_blj", 400_000, -191.1840192602, SEEDS_12),
    ("blj", "blj130n55", "h2h_blj", 1_000_000, -306.7337643721, SEEDS_12),
    ("blj", "blj115n55", "h2h_blj", 1_000_000, -289.2931853058, SEEDS_12),
    ("binary_water", "lj100-binary", "search_binary", 2_000_000, -604.7963065717, SEEDS_12),
    ("water_gmin", "tip4p8", "h2h_seed", 2_000_000, -305.5183221676, SEEDS_12),
    ("water_gmin", "tip4p12", "h2h_seed", 2_000_000, -492.9082357122, SEEDS_12),
    ("water_gmin", "tip4p16", "h2h_seed", 2_000_000, -681.1925945898, SEEDS_12),
    ("water_gmin", "tip4p20", "h2h_seed", 2_000_000, -872.9887544952, SEEDS_12),
    ("water_gmin", "tip4p21", "h2h_seed", 2_000_000, -916.7074302940, SEEDS_12),
    ("blj", "blj130n98", "h2h_blj", 4_000_000, -590.7874132760, SEEDS_12),
)

ROW_FIELDS = [
    "panel",
    "system",
    "arm",
    "driver",
    "seed",
    "order_pos",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
    "pot_source",
    "code_revision",
    "host",
    "cpu",
    "session",
    "utc",
]


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip().replace(",", " ")
    return ""


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("H2H_"):
            del env[key]
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def git_rev(tree: Path, expr: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", expr],
        cwd=tree,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({k: row[k] for k in ROW_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def parse_kv(text: str, prefix: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith(prefix):
            fields: dict[str, str] = {}
            for part in line.split()[1:]:
                key, value = part.split("=", 1)
                fields[key] = value
            return fields
    raise RuntimeError(f"missing {prefix!r} line in output:\n{text[-2000:]}")


def check_row(system: str, seed: int, budget: int, reference: float, hit: int, evals: int, best: float) -> None:
    expected_hit = int(best <= reference + HIT_TOL + 1e-12)
    if hit != expected_hit:
        raise RuntimeError(f"{system} seed {seed}: hit={hit} inconsistent with best={best} ref={reference}")
    if hit == 0 and evals != budget:
        raise RuntimeError(f"{system} seed {seed}: miss has evals_to_hit={evals} != budget={budget}")
    if hit == 1 and not (0 < evals <= budget):
        raise RuntimeError(f"{system} seed {seed}: hit has invalid evals_to_hit={evals}")


def run_blj(bin_dir: Path, tree: Path, system: str, seed: int, budget: int, ref: float, log: Path) -> dict[str, str]:
    cmd = ["taskset", "-c", "0", str(bin_dir / "gate_driver"), "blj", system, str(seed)]
    proc = subprocess.run(cmd, cwd=tree, env=single_thread_env(), capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(cmd) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} seed {seed} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    fields = parse_kv(proc.stdout, "gate_row ")
    if fields["system"] != system or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"unexpected row {fields}")
    hit = int(fields["hit"])
    evals = int(fields["evals_to_hit"])
    best = float(fields["best"])
    check_row(system, seed, budget, ref, hit, evals, best)
    return {
        "hit": str(hit),
        "evals_to_hit": str(evals),
        "charged": str(int(fields["charged"])),
        "wall_s": f"{float(fields['wall_s']):.6f}",
        "best": f"{best:.10f}",
        "pot_s": f"{float(fields['pot_s']):.6f}",
        "outside_s": f"{float(fields['outside_s']):.6f}",
        "pot_source": fields["pot_source"],
    }


def run_search_binary(
    bin_dir: Path,
    tree: Path,
    lj100_dir: Path,
    seed: int,
    budget: int,
    ref: float,
    log: Path,
) -> dict[str, str]:
    start_idx = seed - 2000
    coords_file = lj100_dir / f"coords.{start_idx}"
    if not coords_file.is_file():
        raise RuntimeError(f"missing {coords_file}")
    target = ref + HIT_TOL
    cmd = [
        "taskset", "-c", "0", str(bin_dir / "gate_driver"),
        "search_binary", str(coords_file), str(seed), str(budget), repr(target),
    ]
    proc = subprocess.run(cmd, cwd=tree, env=single_thread_env(), capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(cmd) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"lj100-binary seed {seed} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    fields = parse_kv(proc.stdout, "gate_row ")
    if fields["system"] != "lj100-binary" or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"unexpected row {fields}")
    hit = int(fields["hit"])
    evals = int(fields["evals_to_hit"])
    best = float(fields["best"])
    check_row("lj100-binary", seed, budget, ref, hit, evals, best)
    return {
        "hit": str(hit),
        "evals_to_hit": str(evals),
        "charged": str(int(fields["charged"])),
        "wall_s": f"{float(fields['wall_s']):.6f}",
        "best": f"{best:.10f}",
        "pot_s": f"{float(fields['pot_s']):.6f}",
        "outside_s": f"{float(fields['outside_s']):.6f}",
        "pot_source": fields["pot_source"],
    }


def run_water_gmin(
    bin_dir: Path,
    tree: Path,
    system: str,
    seed: int,
    budget: int,
    ref: float,
    scratch: Path,
    log: Path,
) -> dict[str, str]:
    n = int(system.removeprefix("tip4p"))
    prefix = scratch / f"{system}-s{seed}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stop = repr(ref + HIT_TOL)
    cmd = [
        "taskset", "-c", "0", str(bin_dir / "h2h_seed"),
        "tip4p", str(n), "0", str(budget), str(seed), "recommended", stop, str(prefix),
    ]
    proc = subprocess.run(cmd, cwd=tree, env=single_thread_env(), capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(cmd) + "\n" + proc.stdout + proc.stderr)
    for ext in (".trace.csv", ".xyz", ".start.gmin", ".gmin"):
        (prefix.parent / f"{prefix.name}{ext}").unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} seed {seed} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    fields = parse_kv(proc.stdout, "h2h_row ")
    if fields["system"] != system or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"unexpected row {fields}")
    hit = int(fields["hit"])
    evals = int(fields["evals_to_hit"])
    best = float(fields["best"])
    check_row(system, seed, budget, ref, hit, evals, best)
    return {
        "hit": str(hit),
        "evals_to_hit": str(evals),
        "charged": str(int(fields["charged"])),
        "wall_s": f"{float(fields['wall_s']):.6f}",
        "best": f"{best:.10f}",
        "pot_s": f"{float(fields['pot_s']):.6f}",
        "outside_s": f"{float(fields['outside_s']):.6f}",
        "pot_source": "direct",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--arm-a-tree", type=Path, required=True)
    parser.add_argument("--arm-b-tree", type=Path, required=True)
    parser.add_argument("--arm-a-bin", type=Path, required=True)
    parser.add_argument("--arm-b-bin", type=Path, required=True)
    parser.add_argument("--lj100-dir", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--allow-rerun-seed", type=int, action="append")
    parser.add_argument("--session")
    args = parser.parse_args()

    host = socket.gethostname()
    if host != "bench":
        raise SystemExit(f"hostname is {host!r}, expected 'bench'")
    if os.sched_getaffinity(0) != {0}:
        raise SystemExit(f"cpu affinity is {sorted(os.sched_getaffinity(0))}, expected [0]")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise SystemExit(f"{key}={os.environ.get(key)!r}, expected '1'")

    rev_a = git_rev(args.arm_a_tree, "HEAD")
    rev_b = git_rev(args.arm_b_tree, "HEAD")
    tree_b = git_rev(args.arm_b_tree, "HEAD^{tree}")
    if not rev_a.startswith("21cd57b6"):
        raise SystemExit(f"Arm A revision {rev_a} is not develop 21cd57b6")
    if not tree_b.startswith("208b1b1d"):
        raise SystemExit(f"Arm B tree {tree_b} does not match 208b1b1d")

    arms = {
        "A": (args.arm_a_bin, args.arm_a_tree, rev_a),
        "B": (args.arm_b_bin, args.arm_b_tree, rev_b),
    }

    stored = read_rows(args.rows)
    session = args.session or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    only_seeds = set(args.seed or [])
    rerun_seeds = set(args.allow_rerun_seed or [])
    done = {(row["panel"], row["system"], row["arm"], row["seed"], row["code_revision"]) for row in stored}
    cpu = cpu_model()

    print(
        f"session={session} host={host} arm_a={rev_a[:8]} arm_b={rev_b[:8]} arm_b_tree={tree_b[:8]}",
        flush=True,
    )

    selected = [row for row in SYSTEMS if not args.system or row[1] in args.system]
    for panel, system, driver, budget, ref, seeds in selected:
        for seed in seeds:
            if only_seeds and seed not in only_seeds:
                continue
            order = ("A", "B") if (seed - seeds[0]) % 2 == 0 else ("B", "A")
            for order_pos, arm in enumerate(order):
                bin_dir, tree, revision = arms[arm]
                key = (panel, system, arm, str(seed), revision)
                if key in done and seed not in rerun_seeds:
                    print(f"skip {panel}/{system} arm={arm} seed={seed}", flush=True)
                    continue
                print(f"start {panel}/{system} arm={arm} seed={seed} order={order_pos}", flush=True)
                log = args.logs / system / f"{arm}-{seed}.log"
                if panel == "blj":
                    res = run_blj(bin_dir, tree, system, seed, budget, ref, log)
                elif panel == "binary_water":
                    res = run_search_binary(bin_dir, tree, args.lj100_dir, seed, budget, ref, log)
                elif panel == "water_gmin":
                    res = run_water_gmin(bin_dir, tree, system, seed, budget, ref, args.scratch / arm, log)
                else:
                    raise RuntimeError(f"unknown panel {panel}")
                row = {
                    "panel": panel,
                    "system": system,
                    "arm": arm,
                    "driver": driver,
                    "seed": str(seed),
                    "order_pos": str(order_pos),
                    "budget": str(budget),
                    **res,
                    "code_revision": revision,
                    "host": host,
                    "cpu": cpu,
                    "session": session,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                append_row(args.rows, row)
                done.add(key)
                print(
                    f"done {panel}/{system} arm={arm} seed={seed} hit={row['hit']} "
                    f"calls={row['evals_to_hit']} wall_s={row['wall_s']} "
                    f"pot_s={row['pot_s']} outside_s={row['outside_s']} best={row['best']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
