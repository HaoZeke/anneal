#!/usr/bin/env python3
# Copyright (c) anneal developers
"""ABBA wall times for (H2O)8 on one core.

Each seed is anneal, CREST, CREST, anneal. Anneal is the full polish-on-improve
budget with no hit stop. CREST is the stored full run from crest_run.py:
``--gfn2 -T 1 --legacy``, xtb behind ``-xnam``, stop at the call budget rather
than at the first hit. The stored first-hit table is not rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import run

HERE = Path(__file__).resolve().parent
COLUMNS = [
    "arm",
    "seed",
    "repeat",
    "wall_s",
    "cpu_s",
    "best_Eh",
    "host_id",
    "core",
]


def steps() -> list[tuple[str, int, int]]:
    order = []
    for seed in run.SEEDS:
        order.append(("anneal", seed, 1))
        order.append(("crest", seed, 1))
        order.append(("crest", seed, 2))
        order.append(("anneal", seed, 2))
    return order


def step_dir(arm: str, seed: int, repeat: int) -> Path:
    return HERE / "runs-abba" / f"{arm}-{seed}-r{repeat}"


def anneal_ready(out: Path) -> bool:
    return run.finished(out / "run.txt") and (out / "time.txt").is_file()


def crest_ready(out: Path, seed: int) -> bool:
    return (out / str(seed) / "row.json").is_file()


def anneal_once(binary: Path, xtb_engine: Path, xtb_lib: Path, seed: int, out: Path) -> dict[str, str]:
    if not anneal_ready(out):
        env = run.base_env(xtb_engine, xtb_lib)
        code, _, _ = run.run_process(binary, seed, out, "0", env)
        if code != 0 or not anneal_ready(out):
            raise SystemExit(f"anneal {seed} exited {code}")
    fields = run.read_fields(out / "run.txt")
    if fields.get("config_sha256") != run.CONFIG_SHA or fields.get("polish_on_improve") != "true":
        raise SystemExit(f"anneal {seed} config {fields.get('config_sha256')}")
    if int(fields["energy_calls"]) != run.BUDGET:
        raise SystemExit(f"anneal {seed} charged {fields['energy_calls']}")
    wall_s, cpu_s = run.load_time(out)
    best_eV = float(fields["objective_best_eV"])
    return {
        "wall_s": f"{wall_s:.6f}",
        "cpu_s": f"{cpu_s:.6f}",
        "best_Eh": f"{best_eV / run.HARTREE_EV:.12f}",
    }


def crest_once(binary: Path, crest: Path, xtb: Path, xtb_lib: Path, crest_lib: Path, seed: int, out: Path) -> dict[str, str]:
    row_path = out / str(seed) / "row.json"
    if not row_path.is_file():
        env = os.environ.copy()
        env["H2H_BIN"] = str(binary)
        env["CREST_BIN"] = str(crest)
        env["XTB_BIN"] = str(xtb)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(str(path) for path in (crest_lib, xtb_lib))
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_DYNAMIC"] = "FALSE"
        env.pop("H2H_START_ONLY", None)
        env.pop("H2H_POLISH_ON_IMPROVE", None)
        env.pop("H2O_HIT_EV", None)
        env.pop("H2O8_CREST_WRAP", None)
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(HERE / "crest_run.py"),
            "--seeds",
            str(seed),
            "--out",
            str(out),
            "--rows",
            str(out / "harness-rows.csv"),
            "--hours",
            "48",
        ]
        with (out / "driver.out").open("w") as stdout, (out / "driver.err").open("w") as stderr:
            proc = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
        if not row_path.is_file():
            raise SystemExit(f"crest {seed} exited {proc.returncode} without a row")
    row = json.loads(row_path.read_text())
    if not row.get("best_Eh") or not row.get("wall_s"):
        raise SystemExit(f"crest {seed} row missing energy or time")
    return {
        "wall_s": f"{float(row['wall_s']):.6f}",
        "cpu_s": f"{float(row['cpu_s']):.6f}",
        "best_Eh": f"{float(row['best_Eh']):.12f}",
    }


def write_table(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def campaign(binary: Path, crest: Path, xtb: Path, xtb_engine: Path, xtb_lib: Path, crest_lib: Path, table: Path, sessions: Path) -> None:
    host = run.host_id()
    done: list[dict[str, str]] = []
    if sessions.is_file():
        done = [json.loads(line) for line in sessions.read_text().splitlines() if line.strip()]
    have = {(row["arm"], int(row["seed"]), int(row["repeat"])) for row in done}
    for arm, seed, repeat in steps():
        key = (arm, seed, repeat)
        if key in have:
            print(f"{arm} {seed} r{repeat} already finished", flush=True)
            continue
        print(f"{arm} {seed} r{repeat} start", flush=True)
        out = step_dir(arm, seed, repeat)
        if arm == "anneal":
            measured = anneal_once(binary, xtb_engine, xtb_lib, seed, out)
        else:
            measured = crest_once(binary, crest, xtb, xtb_lib, crest_lib, seed, out)
        row = {
            "arm": arm,
            "seed": str(seed),
            "repeat": str(repeat),
            "wall_s": measured["wall_s"],
            "cpu_s": measured["cpu_s"],
            "best_Eh": measured["best_Eh"],
            "host_id": host,
            "core": "0",
        }
        done.append(row)
        have.add(key)
        write_table(done, table)
        with sessions.open("w") as handle:
            for item in done:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        print(
            f"{arm} {seed} r{repeat} wall_s {row['wall_s']} cpu_s {row['cpu_s']} best_Eh {row['best_Eh']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--xtb-engine", type=Path)
    parser.add_argument("--xtb", type=Path)
    parser.add_argument("--xtb-lib", type=Path)
    parser.add_argument("--crest", type=Path)
    parser.add_argument("--crest-lib", type=Path)
    parser.add_argument("--rows", type=Path, default=HERE / "rows-abba.csv")
    parser.add_argument("--sessions", type=Path, default=HERE / "sessions-abba.jsonl")
    args = parser.parse_args()
    needed = (args.binary, args.xtb_engine, args.xtb, args.xtb_lib, args.crest, args.crest_lib)
    if any(path is None for path in needed):
        raise SystemExit("binary, xtb-engine, xtb, xtb-lib, crest, and crest-lib are required")
    campaign(
        args.binary,
        args.crest,
        args.xtb,
        args.xtb_engine,
        args.xtb_lib,
        args.crest_lib,
        args.rows,
        args.sessions,
    )


if __name__ == "__main__":
    main()
