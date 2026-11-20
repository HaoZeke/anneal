#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Non-water panel at PAPER_BUDGETS: DEVELOP, TIP and a second DEVELOP build.

Cells are LJ (``h2h_gmin_proof``), binary LJ (``h2h_blj``), Morse
(``h2h_morse``), Sutton-Chen (``h2h_sc``) and Gupta metals (``h2h_metal``),
each through the driver from ``gate_examples.py``. Budgets are the drivers'
own paper budgets; no H2H_* variable is set. Seeds 2000-2002. Even seeds
run A, B, A2 and odd seeds A2, B, A: A is DEVELOP, B is TIP, A2 is a second
fresh release build of DEVELOP for the A/A ratio. One process at a time,
``taskset -c 0``, one thread.

A hit is a finite best at or below the cell's reference plus 1e-3.
``calls`` is the driver's calls to the hit, or the budget on a miss. The
status is ``ok``, or ``driver_hit_differs`` when the driver's own hit flag
disagrees with that rule.
"""

from __future__ import annotations

import argparse
import datetime
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

ROWS = HERE / "panel-rows.csv"
SEEDS = (2000, 2001, 2002)
HIT_TOL = 1.0e-3
TREES = {
    "develop": "cff796c507a8fa1d0bfb28888c3570b61988055e",
    "tip": "b04621fd81776cfb78626162c5b3e746acb9859f",
    "develop-aa": "cff796c507a8fa1d0bfb28888c3570b61988055e",
}
FIELDS = [*common.ROW_FIELDS, "family", "reference", "driver_hit"]

# family, system, driver, reference. References are the drivers' tables.
CELLS = (
    ("lj", "lj38", "gate_h2h_gmin_proof", -173.928427),
    ("lj", "lj55", "gate_h2h_gmin_proof", -279.248470),
    ("lj", "lj75", "gate_h2h_gmin_proof", -397.492331),
    ("lj", "lj98", "gate_h2h_gmin_proof", -543.665361),
    ("blj", "blj105n38", "gate_h2h_blj", -177.2090846748),
    ("blj", "blj115n38", "gate_h2h_blj", -187.3676867453),
    ("blj", "blj130n38", "gate_h2h_blj", -191.1840192602),
    ("blj", "blj105n55", "gate_h2h_blj", -283.1545313810),
    ("blj", "blj115n55", "gate_h2h_blj", -289.2931853058),
    ("blj", "blj130n55", "gate_h2h_blj", -306.7337643721),
    ("blj", "blj130n98", "gate_h2h_blj", -590.7874132760),
    ("morse", "m55r6", "gate_h2h_morse", -250.286609),
    ("morse", "m55r14", "gate_h2h_morse", -220.646208),
    ("morse", "m80r6", "gate_h2h_morse", -378.333471),
    ("morse", "m80r14", "gate_h2h_morse", -340.811371),
    ("sc", "sc126-38", "gate_h2h_sc", -35419.9804),
    ("sc", "sc108-38", "gate_h2h_sc", -10117.2454),
    ("sc", "sc96-38", "gate_h2h_sc", -8917.7056),
    ("sc", "sc126-55", "gate_h2h_sc", -52884.6806),
    ("sc", "sc108-55", "gate_h2h_sc", -14814.5225),
    ("sc", "sc96-55", "gate_h2h_sc", -13217.8963),
    ("sc", "sc126-80", "gate_h2h_sc", -78414.6271),
    ("sc", "sc108-80", "gate_h2h_sc", -21819.9952),
    ("sc", "sc96-80", "gate_h2h_sc", -19560.8966),
    ("mw", "zn38", "gate_h2h_metal", -50.499983),
    ("mw", "zn55", "gate_h2h_metal", -73.310032),
    ("mw", "zn75", "gate_h2h_metal", -100.203580),
    ("mw", "zn80", "gate_h2h_metal", -106.927940),
    ("mw", "cu19au19", "gate_h2h_metal", -122.99148391),
    ("mw", "au38", "gate_h2h_metal", -130.54001708),
    ("mw", "cu6au32", "gate_h2h_metal", -130.79994836),
    ("mw", "au55", "gate_h2h_metal", -191.03085280),
    ("mw", "cu27au28", "gate_h2h_metal", -182.93136675),
)


def order(seed: int) -> tuple[str, ...]:
    return ("develop", "tip", "develop-aa") if seed % 2 == 0 else ("develop-aa", "tip", "develop")


def parse(stdout: str) -> tuple[dict[str, str], float, int]:
    lines = stdout.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("system,code,seed,"))
    row = dict(zip(lines[header].split(","), lines[header + 1].split(",")))
    full = next(line for line in lines if line.startswith("best_full "))
    _, text, raw = full.split()
    value = float(text)
    if common.bits(value) != f"{int(raw):016x}":
        raise RuntimeError(f"best_full {text} does not round-trip to bits {raw}")
    return row, value, int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="append", required=True, help="arm=target-dir")
    parser.add_argument("--sc-lib", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--core", default="0")
    args = parser.parse_args()

    builds = dict(item.split("=", 1) for item in args.build)
    if set(builds) != set(TREES):
        raise SystemExit(f"builds must be {sorted(TREES)}")
    for arm, target in builds.items():
        stamp = (Path(target) / "SOURCE_TREE").read_text().strip()
        if stamp != TREES[arm]:
            raise SystemExit(f"{arm} was built from tree {stamp}, not {TREES[arm]}")
    if Path(builds["develop"]).resolve() == Path(builds["develop-aa"]).resolve():
        raise SystemExit("the A/A arm needs its own build")

    env = common.run_env({"RGPOT_SC_LIB": str(args.sc_lib)})
    machine = common.machine_id()
    harness = common.blob_of(Path(__file__))
    done = {(r["system"], r["seed"], r["arm"]) for r in common.read_rows(args.rows)}
    for family, system, driver, reference in CELLS:
        for seed in SEEDS:
            for slot, arm in enumerate(order(seed)):
                if (system, str(seed), arm) in done:
                    continue
                target = Path(builds[arm])
                binary = target / "release" / "examples" / driver
                command = ["taskset", "-c", args.core, str(binary), "run", system, str(seed)]
                now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
                print(f"start {system} {seed} {arm} slot {slot} {now}", flush=True)
                proc = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
                log = args.logs / system / f"{seed}-{slot}-{arm}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(" ".join(command) + "\n" + proc.stdout + "\n--- stderr\n" + proc.stderr)
                if proc.returncode != 0:
                    raise SystemExit(f"{system} {seed} {arm} exited {proc.returncode}; see {log}")
                printed, best, _ = parse(proc.stdout)
                budget = int(printed["budget"])
                hit = math.isfinite(best) and best <= reference + HIT_TOL
                driver_hit = printed["hit"] in ("true", "1")
                calls = int(printed["evals_to_hit"]) if hit else budget
                pot_s = float(printed["pot_s"])
                wall = float(printed["wall_s"])
                outside = printed.get("outside_s") or f"{max(wall - pot_s, 0.0):.6f}"
                row = {
                    "system": system,
                    "arm": arm,
                    "seed": str(seed),
                    "slot": str(slot),
                    "core": args.core,
                    "hit": "1" if hit else "0",
                    "calls": str(calls),
                    "charged": printed["charged"],
                    "wall": printed["wall_s"],
                    "inside_s": printed["pot_s"],
                    "outside_s": outside,
                    "best": repr(best),
                    "best_bits": common.bits(best),
                    "budget": str(budget),
                    "machine_id": machine,
                    "library_tree": TREES[arm],
                    "harness_blob": harness,
                    "driver_blob": common.blob_of(target / "drivers" / f"{driver}.rs"),
                    "status": "ok" if hit == driver_hit else "driver_hit_differs",
                    "family": family,
                    "reference": repr(reference),
                    "driver_hit": "1" if driver_hit else "0",
                }
                common.append_row(args.rows, row, FIELDS)
                print(",".join(row[key] for key in FIELDS), flush=True)


if __name__ == "__main__":
    main()
