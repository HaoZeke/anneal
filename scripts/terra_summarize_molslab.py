#!/usr/bin/env python3
"""Summarize terra_measure_molslab.sh seed logs into the campaign tables."""

from __future__ import annotations

import pathlib
import re
import statistics
import sys

seed_re = re.compile(
    r"seed (?P<seed>\d+): best (?P<best>[-+0-9.eE]+) eV\s+hops (?P<hops>\d+)\s+"
    r"charged (?P<charged>\d+)\s+basins (?P<basins>\d+)"
)
improve_re = re.compile(
    r"improve hops=\d+ charged=(?P<charged>\d+) basins=\d+ e=(?P<energy>[-+0-9.eE]+)"
)
eval_re = re.compile(
    r"eval_wall engine=(?P<engine>\S+) repeats=(?P<repeats>\d+) mean_us=(?P<us>\d+)"
)


def parse_arm(directory: pathlib.Path) -> tuple[list[dict], list[tuple[str, int]]]:
    rows: list[dict] = []
    evals: list[tuple[str, int]] = []
    for path in sorted(directory.glob("seed_*.out")):
        text = path.read_text(errors="replace")
        seed_m = seed_re.search(text)
        if not seed_m:
            continue
        best = float(seed_m.group("best"))
        charged_to_best = None
        for match in improve_re.finditer(text):
            energy = float(match.group("energy"))
            if abs(energy - best) <= 1e-6:
                charged_to_best = int(match.group("charged"))
        rows.append(
            {
                "seed": int(seed_m.group("seed")),
                "best": best,
                "charged": int(seed_m.group("charged")),
                "basins": int(seed_m.group("basins")),
                "charged_to_best": charged_to_best,
            }
        )
        eval_m = eval_re.search(text)
        if eval_m:
            evals.append((eval_m.group("engine"), int(eval_m.group("us"))))
    return rows, evals


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: terra_summarize_molslab.py OUT_DIR", file=sys.stderr)
        return 2
    out = pathlib.Path(sys.argv[1])
    print("=== eval wall ===")
    print("engine  mean_us")
    seen: dict[str, list[int]] = {}
    for directory in sorted(p for p in out.iterdir() if p.is_dir()):
        _, evals = parse_arm(directory)
        for engine, us in evals:
            seen.setdefault(engine, []).append(us)
    for engine, values in seen.items():
        print(f"{engine:6}  {statistics.median(values)}")

    print()
    print("=== water baseline (plain) ===")
    print("m  seed  best_eV  charged  basins")
    for m in range(6, 11):
        directory = out / f"h2o{m}_plain"
        if not directory.is_dir():
            continue
        rows, _ = parse_arm(directory)
        for row in rows:
            print(
                f"{m}  {row['seed']}  {row['best']:.6f}  {row['charged']}  {row['basins']}"
            )
        if rows:
            distinct = len({round(row["best"], 5) for row in rows})
            median_basins = statistics.median(row["basins"] for row in rows)
            print(f"{m}  distinct_bests {distinct}  median_basins {median_basins:.1f}")

    print()
    print("=== two-phase vs plain ===")
    print("m  arm  n_at_global_best  median_charged_to_it  best_seen")
    for m in range(6, 11):
        arms = ["plain"] + [
            f"k{kappa}_m{mu}" for kappa in ("0.7", "0.8") for mu in ("2.5", "5")
        ]
        parsed: dict[str, list[dict]] = {}
        for arm in arms:
            name = f"h2o{m}_plain" if arm == "plain" else f"h2o{m}_{arm}"
            directory = out / name
            if directory.is_dir():
                parsed[arm], _ = parse_arm(directory)
        all_bests = [row["best"] for rows in parsed.values() for row in rows]
        if not all_bests:
            continue
        global_best = min(all_bests)
        for arm in arms:
            rows = parsed.get(arm, [])
            hits = [row for row in rows if abs(row["best"] - global_best) <= 1e-3]
            charges = [row["charged_to_best"] or row["charged"] for row in hits]
            median = statistics.median(charges) if charges else float("nan")
            arm_best = min((row["best"] for row in rows), default=float("nan"))
            print(f"{m}  {arm:12}  {len(hits)}/{len(rows)}  {median}  {arm_best:.6f}")
        print(f"{m}  global_best {global_best:.6f}")

    print()
    print("=== CuH2 slab ===")
    print("seed  best_eV  charged  basins")
    directory = out / "cuh2_plain"
    rows, _ = parse_arm(directory)
    for row in rows:
        print(f"{row['seed']}  {row['best']:.6f}  {row['charged']}  {row['basins']}")
    if rows:
        distinct = len({round(row["best"], 5) for row in rows})
        median_basins = statistics.median(row["basins"] for row in rows)
        print(f"distinct_bests {distinct}  median_basins {median_basins:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
