#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the current-code searches needed to replace the six sealed raster panels.

The run is resumable under ``--work``. Source books are kept outside Git until
every declared seed has finished; only then are landscape.ld and fes.csv
published into the paper data tree.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "paper/jctc/data/landfold_maps_v7"


@dataclass(frozen=True)
class Campaign:
    name: str
    family: str
    size: int
    config: str
    budget: int
    first: int
    last: int
    panels: tuple[str, ...]


CAMPAIGNS = (
    Campaign("lj_csm/lj75/recommended", "lj", 75, "recommended", 4_000_000, 2000, 2199,
             ("representative", "aggregate")),
    Campaign("lj_csm/lj98/derived", "lj", 98, "derived", 4_000_000, 2000, 2047,
             ("representative",)),
    Campaign("lj_csm_c26/lj98/recommended", "lj", 98, "recommended", 4_000_000, 2000, 2047,
             ("representative",)),
    Campaign("tip4p/hexamer/for_tip4p", "tip4p", 6, "for_tip4p", 20_000, 0, 3,
             ("representative", "aggregate")),
)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def source_rows(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def fold(binary: Path, books: list[Path], family: str, folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    folded = folder / "fold.csv"
    command = [str(binary), "landfold", "--seed", "0", "--threads", "1", "--out", str(folded)]
    descriptor = None
    if family == "tip4p":
        descriptor = folder / "rigid-state.csv"
        with descriptor.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for book in books:
                for line in book.read_text().splitlines():
                    if line.strip():
                        writer.writerow(line.split()[1:])
        command.extend(["--descriptors", str(descriptor)])
    else:
        command.extend(str(book) for book in books)
    run(command)
    points = []
    with folded.open(newline="") as handle:
        for row in csv.DictReader(handle):
            points.append((float(row["x"]), float(row["y"])))
    if not points:
        raise SystemExit(f"{folded}: current campaign produced no fold points")
    with (folder / "landscape.ld").open("w") as handle:
        handle.write("# current-code landfold coordinate 1  coordinate 2\n")
        for x, y in points:
            handle.write(f"{x:.12e} {y:.12e}\n")
    write_fes(points, folder / "fes.csv")
    return len(points)


def write_fes(points: list[tuple[float, float]], path: Path, grid: int = 48) -> None:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = xmax - xmin or 1.0, ymax - ymin or 1.0
    xmin, xmax = xmin - 0.05 * xspan, xmax + 0.05 * xspan
    ymin, ymax = ymin - 0.05 * yspan, ymax + 0.05 * yspan
    counts = [[0 for _ in range(grid)] for _ in range(grid)]
    for x, y in points:
        ix = min(grid - 1, max(0, int((x - xmin) / (xmax - xmin) * grid)))
        iy = min(grid - 1, max(0, int((y - ymin) / (ymax - ymin) * grid)))
        counts[iy][ix] += 1
    peak = max(max(row) for row in counts)
    floor = max(peak * 0.05, 1e-12)
    with path.open("w") as handle:
        handle.write("# x y relative_free_energy\n")
        for iy, row in enumerate(counts):
            y = ymin + (iy + 0.5) * (ymax - ymin) / grid
            for ix, count in enumerate(row):
                x = xmin + (ix + 0.5) * (xmax - xmin) / grid
                density = max(count, floor)
                energy = -0.2 * math.log(density / peak)
                handle.write(f"{x:.12e} {y:.12e} {energy:.12e}\n")


def revision() -> tuple[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
    return commit, tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/work/anneal-landfold-current"))
    parser.add_argument("--publish", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit-seeds", type=int, help="run at most this many seeds per campaign; never publish")
    parser.add_argument("--budget-cap", type=int, help="cap each budget for a smoke run; never publish")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    run(["cargo", "build", "--release", "--example", "landfold_campaign", "--bin", "anneal"])
    search = ROOT.parent / "tmp/anneal-target/release/examples/landfold_campaign"
    landfold = ROOT.parent / "tmp/anneal-target/release/anneal"
    if not search.is_file() or not landfold.is_file():
        raise SystemExit("the current-code campaign binaries were not built at the configured target")

    complete = True
    for campaign in CAMPAIGNS:
        source = args.work / campaign.name / "sources"
        source.mkdir(parents=True, exist_ok=True)
        seeds = list(range(campaign.first, campaign.last + 1))
        selected = seeds if args.limit_seeds is None else seeds[: args.limit_seeds]
        for seed in selected:
            output = source / f"seed-{seed:04d}.min"
            if output.is_file() and source_rows(output):
                continue
            pending = output.with_suffix(".min.tmp")
            run([
                "taskset", "-c", "0", str(search), campaign.family, str(campaign.size),
                campaign.config, str(min(campaign.budget, args.budget_cap or campaign.budget)), str(seed), str(pending),
            ])
            if not pending.is_file() or not source_rows(pending):
                raise SystemExit(f"{campaign.name} seed {seed}: no validated minimum was exported")
            pending.replace(output)
        books = [source / f"seed-{seed:04d}.min" for seed in seeds]
        if not all(path.is_file() and source_rows(path) for path in books):
            complete = False
            continue
        products = args.work / campaign.name / "products"
        for panel in campaign.panels:
            chosen = books[:1] if panel == "representative" else books
            rows = fold(landfold, chosen, campaign.family, products / panel)
            if rows != sum(source_rows(path) for path in chosen):
                raise SystemExit(f"{campaign.name}/{panel}: fold dropped source rows")

    if not complete:
        raise SystemExit("campaign incomplete: resumable source books remain under --work; no paper table was published")
    if args.limit_seeds is not None or args.budget_cap is not None:
        raise SystemExit("seed or budget limits make this a smoke run; paper tables were not published")

    commit, tree = revision()
    receipt = {
        "code_revision": commit,
        "code_tree": tree,
        "host": socket.gethostname(),
        "campaigns": [campaign.__dict__ for campaign in CAMPAIGNS],
    }
    for campaign in CAMPAIGNS:
        for panel in campaign.panels:
            src = args.work / campaign.name / "products" / panel
            dest = args.publish / campaign.name / panel
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("landscape.ld", "fes.csv"):
                shutil.copyfile(src / name, dest / name)
    (args.publish / "current-rerun.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.work / "complete.stamp").write_text(f"{commit} {tree} {socket.gethostname()}\n")


if __name__ == "__main__":
    main()
