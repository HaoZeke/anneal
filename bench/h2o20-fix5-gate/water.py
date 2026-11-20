#!/usr/bin/env python3
# Copyright (c) anneal developers
"""GFN2 (H2O)20 at TIP's default config against CREST, one session.

The searches are the existing harness, ``scripts/h2o20_crest/run_onehost.py``:
``run_anneal`` for the default anneal arm (TIP4P surrogate stage, then the
flexible GFN2 search, no capped Lindh), ``run_crest`` for CREST 3.0.2 through
xtb ``--driver``, and its vtight rescore. This file only fixes the order and
writes the rows. Seeds are the harness's ``SEEDS``, the budget its
``PAPER_BUDGETS``. Even seeds run anneal then CREST, odd seeds CREST then
anneal. One process at a time on one core, one thread.

A hit is a rescore at or below the reference of the 131b411a panel,
-101.749249454294 Eh, by 1e-6 Eh, which is the harness's rule. Neither
driver records the call that found its best. For an anneal hit ``calls`` is
the charged count at the last improvement in ``improvements.txt``; for a CREST hit
it is the charged count. A miss has ``calls`` equal to the budget.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

ROOT = common.ROOT
ONEHOST = ROOT / "scripts" / "h2o20_crest" / "run_onehost.py"
TIP_TREE = "b04621fd81776cfb78626162c5b3e746acb9859f"
ROWS = ROOT / "bench" / "h2o20-crest" / f"rows-{TIP_TREE}.csv"
REFERENCE_EH = -101.749249454294
FIELDS = [*common.ROW_FIELDS, "objective_eV", "onehost_blob"]


def load_onehost():
    spec = importlib.util.spec_from_file_location("run_onehost", ONEHOST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule(seeds: tuple[int, ...]) -> list[tuple[int, int, str]]:
    out = []
    for seed in seeds:
        order = ("anneal", "crest") if seed % 2 == 0 else ("crest", "anneal")
        out.extend((seed, slot, arm) for slot, arm in enumerate(order))
    return out


def fields_of(path: Path) -> dict[str, str]:
    found = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            found[key.strip()] = value.strip()
    return found


def anneal_counts(where: Path, hit: bool, budget: int) -> tuple[int, int, str]:
    run = fields_of(where / "run.txt")
    charged = int(run["energy_calls"])
    if not hit:
        return budget, charged, run.get("status", "unknown")
    last = (where / "improvements.txt").read_text().strip().splitlines()[-1].split()
    return int(last[1]), charged, run.get("status", "unknown")


def crest_counts(where: Path, hit: bool, budget: int) -> tuple[int, int, str]:
    served = fields_of(where / "driver" / "driver.txt")
    charged = int(served["calls"])
    status = (where / "exit.txt").read_text().strip()
    return (charged if hit else budget), charged, status


def failed_anneal(where: Path) -> dict[str, str]:
    """A search that exited with the failed status. It has no geometry to rescore."""
    run = fields_of(where / "run.txt")
    return {
        "rescored_Eh": "nan",
        "objective_eV": run.get("objective_best_eV", "nan"),
        "wall_s": run.get("wall_s", "nan"),
        "pot_s": run.get("potential_s", "nan"),
        "outside_s": run.get("outside_s", "nan"),
        "status": "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, required=True, help="clean TIP worktree")
    parser.add_argument("--anneal", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--crest", type=Path, required=True)
    parser.add_argument("--xtb", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--core", default="0")
    args = parser.parse_args()

    tree = common.tree_of(args.tree)
    if tree != TIP_TREE or not common.clean(args.tree):
        raise SystemExit(f"arm tree {tree} is not the clean TIP tree {TIP_TREE}")
    onehost = load_onehost()
    budget = onehost.BUDGET
    seeds = onehost.SEEDS

    os.environ.clear()
    os.environ.update(
        common.run_env(
            {
                "RGPOT_XTB_ENGINE": str(args.engine),
                "XTB_BIN": str(args.xtb),
                "ANNEAL_SOURCE_TREE": tree,
                "ANNEAL_SOURCE_COMMIT": common.git(args.tree, "rev-parse", "HEAD"),
            }
        )
    )
    machine = common.machine_id()
    harness = common.blob_of(Path(__file__))
    onehost_blob = common.blob_of(ONEHOST)
    drivers = {
        "anneal": common.blob_of(args.tree / "examples" / "gfn2_water_h2h.rs"),
        "crest": common.blob_of(args.tree / "examples" / "gfn2_water_crest_driver.rs"),
    }
    starts = args.out / "starts"
    starts.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        source = (args.starts / f"{seed}.xyz").read_bytes()
        path = starts / f"{seed}.xyz"
        if path.is_file() and path.read_bytes() != source:
            raise SystemExit(f"{path} differs from {args.starts / f'{seed}.xyz'}")
        path.write_bytes(source)

    done = {(row["seed"], row["arm"]) for row in common.read_rows(args.rows)}
    for seed, slot, arm in schedule(seeds):
        if (str(seed), arm) in done:
            print(f"skip {arm} {seed}", flush=True)
            continue
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        print(f"start {arm} {seed} slot {slot} {now}", flush=True)
        start = starts / f"{seed}.xyz"
        where = args.out / arm / str(seed)
        if arm == "anneal":
            try:
                measured = onehost.run_anneal(args.anneal, seed, start, where, args.core, False)
            except subprocess.CalledProcessError:
                measured = failed_anneal(where)
        else:
            measured = onehost.run_crest(args.crest, args.xtb, args.driver, start, where, args.core)
        best = float(measured["rescored_Eh"])
        status_override = measured.pop("status", None)
        hit = onehost.energy_hits(best, REFERENCE_EH, onehost.HIT_TOL_EH)
        counts = anneal_counts if arm == "anneal" else crest_counts
        calls, charged, status = counts(where, hit, budget)
        status = status_override or status
        row = {
            "system": onehost.SYSTEM,
            "arm": arm,
            "seed": str(seed),
            "slot": str(slot),
            "core": args.core,
            "hit": "1" if hit else "0",
            "calls": str(calls),
            "charged": str(charged),
            "wall": measured["wall_s"],
            "inside_s": measured["pot_s"],
            "outside_s": measured["outside_s"],
            "best": measured["rescored_Eh"],
            "best_bits": common.bits(best),
            "budget": str(budget),
            "machine_id": machine,
            "library_tree": tree,
            "harness_blob": harness,
            "driver_blob": drivers[arm],
            "status": status,
            "objective_eV": measured["objective_eV"],
            "onehost_blob": onehost_blob,
        }
        common.append_row(args.rows, row, FIELDS)
        print(",".join(row[key] for key in FIELDS), flush=True)


if __name__ == "__main__":
    main()
