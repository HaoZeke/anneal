#!/usr/bin/env python3
"""First-encounter comparison of a bank arm against its no-bank control.

Reads the improve / seed lines written by molecular_cluster and
slab_adsorption. Reports hits, Kaplan-Meier median first encounter,
and mean best energy. The target is the deeper of the two cohort
bests unless --target is set.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

SEED = re.compile(
    r"seed\s+(\d+):\s+best\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+eV"
    r".*?charged\s+(\d+)"
)
IMPROVE = re.compile(
    r"improve hops=(\d+) charged=(\d+) basins=(\d+) e=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def parse_dir(path: Path) -> list[dict]:
    runs = []
    for fp in sorted(path.glob("*.out")):
        text = fp.read_text(errors="replace")
        seeds = list(SEED.finditer(text))
        if not seeds:
            continue
        improves = [
            (int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)))
            for m in IMPROVE.finditer(text)
        ]
        last = seeds[-1]
        runs.append(
            {
                "file": fp.name,
                "seed": int(last.group(1)),
                "best": float(last.group(2)),
                "charged": int(last.group(3)),
                "improvements": improves,
            }
        )
    return runs


def first_encounter(run: dict, target: float, tol: float) -> tuple[bool, int]:
    for _h, charged, _b, e in run["improvements"]:
        if e < target + tol:
            return True, charged
    if run["best"] < target + tol:
        return True, run["charged"]
    return False, run["charged"]


def km_median(events: list[tuple[int, bool]]) -> int | None:
    if not events:
        return None
    events = sorted(events, key=lambda t: t[0])
    at_risk = float(len(events))
    survival = 1.0
    for charged, found in events:
        if found:
            survival *= 1.0 - 1.0 / at_risk
            if survival <= 0.5:
                return charged
        at_risk -= 1.0
        if at_risk <= 0.0:
            break
    return None


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def report(name: str, runs: list[dict], target: float, tol: float) -> dict:
    enc = [first_encounter(r, target, tol) for r in runs]
    hits = sum(1 for f, _ in enc if f)
    med = km_median([(c, f) for f, c in enc])
    bests = [r["best"] for r in runs]
    spent = [r["charged"] for r in runs]
    print(f"{name}: n={len(runs)} hits={hits}/{len(runs)} "
          f"median_encounter={med if med is not None else 'censored'} "
          f"mean_best={mean(bests):.6f} mean_charged={mean([float(s) for s in spent]):.1f}")
    for r, (found, charged) in zip(runs, enc):
        tag = "found" if found else "censored"
        print(f"  seed {r['seed']}: best {r['best']:.6f}  {tag} at {charged}")
    return {"hits": hits, "n": len(runs), "median": med, "mean_best": mean(bests)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ctrl", type=Path, help="no-bank output directory")
    ap.add_argument("bank", type=Path, help="bank output directory")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    ctrl = parse_dir(args.ctrl)
    bank = parse_dir(args.bank)
    if not ctrl and not bank:
        print("no seed lines in either directory", file=sys.stderr)
        return 2
    pool = [r["best"] for r in ctrl + bank if math.isfinite(r["best"])]
    target = args.target if args.target is not None else (min(pool) if pool else 0.0)
    print(f"target={target:.8f}  tol={args.tol}  (deeper of both arms unless --target)")
    c = report("nobank", ctrl, target, args.tol)
    b = report("bank", bank, target, args.tol)
    if c["n"] and b["n"]:
        if b["hits"] > c["hits"]:
            print("bank has more hits at this budget")
        elif b["hits"] < c["hits"]:
            print("bank has fewer hits at this budget")
        if b["median"] is not None and c["median"] is not None:
            if b["median"] < c["median"]:
                print(f"bank median first encounter is lower ({b['median']} < {c['median']})")
            elif b["median"] > c["median"]:
                print(f"bank median first encounter is higher ({b['median']} > {c['median']})")
            else:
                print("median first encounter is the same")
        if b["mean_best"] < c["mean_best"] - args.tol:
            print("bank mean best is deeper")
        elif b["mean_best"] > c["mean_best"] + args.tol:
            print("bank mean best is shallower")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
