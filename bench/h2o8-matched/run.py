#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Matched anneal arm of the (H2O)8 GFN2 head-to-head.

The search is Config.recommended_molecular. Seeds run one at a time.
A hit is one sided. A miss is stored as budget + 1.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# The threshold is one sided. A geometry hits when its energy is at or
# below reference + tolerance. That sum is written -40.681335013150 Eh.
# A miss is stored as budget + 1.
REFERENCE_EH = -40.681336013150
TOLERANCE_EH = 1e-6
THRESHOLD_EH = REFERENCE_EH + TOLERANCE_EH
BUDGET = 151738
MISS = BUDGET + 1
SEEDS = tuple(range(2000, 2012))
PRESET = "recommended_molecular"
COLUMNS = (
    "seed",
    "target_eh",
    "tol_eh",
    "threshold_eh",
    "budget",
    "stop",
    "hit",
    "calls_to_hit",
    "charged",
    "best_eh",
    "engine_sha",
    "preset",
    "wall_s",
)


def load_phases():
    import importlib.util
    import sys

    name = "h2o8_matched_phases"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "phases.py")
    if spec is None or spec.loader is None:
        raise SystemExit("missing phases.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fields_of(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        found[key] = value.strip()
    return found


def require(record: dict[str, str], key: str) -> str:
    if key not in record or record[key] == "":
        raise SystemExit(f"run.txt is missing {key}")
    return record[key]


def row_of(seed: int, record: dict[str, str]) -> dict[str, str]:
    if require(record, "consistent") != "yes":
        raise SystemExit(f"seed {seed} is not consistent")
    if require(record, "preset") != PRESET:
        raise SystemExit(f"seed {seed} preset {record['preset']}")
    charged = int(require(record, "charged"))
    timed = int(require(record, "timed_calls"))
    spent = int(require(record, "energy_calls"))
    if not charged == timed == spent:
        raise SystemExit(f"seed {seed} charged {charged} timed {timed} spent {spent}")
    hit = int(require(record, "hit"))
    calls = int(require(record, "calls_to_hit"))
    stop = require(record, "stop")
    if hit == 1:
        if stop != "target" or calls != charged:
            raise SystemExit(f"seed {seed} hit stop={stop} calls={calls} charged={charged}")
    elif hit == 0:
        if stop != "budget" or calls != MISS:
            raise SystemExit(f"seed {seed} miss stop={stop} calls={calls}")
    else:
        raise SystemExit(f"seed {seed} hit {hit}")
    return {
        "seed": str(seed),
        "target_eh": f"{REFERENCE_EH:.12f}",
        "tol_eh": "1e-6",
        "threshold_eh": f"{THRESHOLD_EH:.12f}",
        "budget": str(BUDGET),
        "stop": stop,
        "hit": str(hit),
        "calls_to_hit": str(calls),
        "charged": str(charged),
        "best_eh": f"{float(require(record, 'lowest_Eh')):.12f}",
        "engine_sha": require(record, "source_commit"),
        "preset": PRESET,
        "wall_s": f"{float(require(record, 'wall_s')):.6f}",
    }


def write_rows(path: Path) -> None:
    rows = []
    for seed in SEEDS:
        record = fields_of(HERE / "runs" / str(seed) / "run.txt")
        rows.append(row_of(seed, record))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def release_example() -> Path:
    name = Path("release") / "examples" / "gfn2_mol_h2h"
    direct = ROOT / "target" / name
    if direct.is_file():
        return direct
    config = ROOT / ".cargo" / "config.toml"
    for line in config.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line.startswith("target-dir"):
            continue
        value = line.split("=", 1)[1].strip().strip("\"'")
        found = (ROOT / value / name).resolve()
        if found.is_file():
            return found
    raise SystemExit("missing the release gfn2_mol_h2h example")


def run_seed(seed: int) -> None:
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 2000-2011")
    out = HERE / "runs" / str(seed)
    done = out / "done.txt"
    if done.is_file():
        return
    engine = os.environ.get("RGPOT_XTB_ENGINE", "")
    if engine == "" or not Path(engine).is_file():
        raise SystemExit("RGPOT_XTB_ENGINE must name libxtb_engine.so")
    binary = release_example()
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "run.txt"
    if partial.is_file():
        partial.unlink()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
    ).strip()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["ANNEAL_SOURCE_COMMIT"] = commit
    env["ANNEAL_SOURCE_TREE"] = tree
    command = [
        str(binary),
        "h2o-8",
        str(BUDGET),
        str(seed),
        str(out),
        "--eh",
        "-40.681336013150",
        "1e-6",
    ]
    taskset = shutil.which("taskset")
    if taskset:
        command = [taskset, "-c", "0", *command]
    lock_path = HERE / ".serial.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.time()
        print(f"SEED {seed} start", flush=True)
        proc = subprocess.run(command, cwd=out, env=env)
        elapsed = time.time() - started
        if proc.returncode != 0:
            raise SystemExit(f"seed {seed} exited {proc.returncode} after {elapsed:.1f}s")
        record = fields_of(out / "run.txt")
        built = row_of(seed, record)
        load_phases().append_phase(out / "run.txt", seed)
        done.write_text("ok\n")
        print(
            f"SEED {seed} hit={built['hit']} calls={built['calls_to_hit']} "
            f"charged={built['charged']} stop={built['stop']} wall={built['wall_s']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rows", type=Path)
    args = parser.parse_args()
    if args.seed is None and args.rows is None:
        raise SystemExit("pass --seed or --rows")
    if args.seed is not None:
        run_seed(args.seed)
    if args.rows is not None:
        write_rows(args.rows)


if __name__ == "__main__":
    main()
