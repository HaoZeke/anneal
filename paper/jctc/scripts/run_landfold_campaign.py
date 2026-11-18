#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the current-code searches and publish the six replacement vector panels.

The run is resumable under ``--work``. Source books are kept outside Git until
every declared seed has finished; only then are landscape.ld and fes.csv
published into the paper data tree.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
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
CAMPAIGN_IDS = {
    "lj75-recommended": CAMPAIGNS[0],
    "lj98-derived": CAMPAIGNS[1],
    "lj98-recommended": CAMPAIGNS[2],
    "tip4p-hexamer": CAMPAIGNS[3],
}


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def source_rows(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def deduplicate_book(source: Path, output: Path) -> int:
    energies: list[float] = []
    kept = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        energy = float(line.split(maxsplit=1)[0])
        if any(abs(energy - held) < 1e-6 for held in energies):
            continue
        energies.append(energy)
        kept.append(line)
    output.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(kept)


def run_shard(campaign_id: str, seed: int, output: Path, receipt: Path) -> None:
    campaign = CAMPAIGN_IDS[campaign_id]
    if seed < campaign.first or seed > campaign.last:
        raise SystemExit(f"{campaign_id}: seed {seed} is outside {campaign.first}..{campaign.last}")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and source_rows(output) and receipt.is_file():
        return
    legacy = Path("/work/anneal-landfold-current") / campaign.name / "sources" / f"seed-{seed:04d}.min"
    elapsed = None
    stderr = ""
    if legacy.is_file() and source_rows(legacy):
        if output.exists():
            output.unlink()
        rows = deduplicate_book(legacy, output)
        charged, hops, best = campaign.budget, None, None
    else:
        pending = output.with_suffix(".min.tmp")
        command = [
            str(ROOT.parent / "tmp/anneal-target/release/examples/landfold_campaign"),
            campaign.family, str(campaign.size), campaign.config, str(campaign.budget),
            str(seed), str(pending),
        ]
        started = time.monotonic()
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        elapsed = time.monotonic() - started
        stderr = completed.stderr.strip()
        if completed.returncode:
            raise SystemExit(stderr or f"{campaign_id} seed {seed}: search exited {completed.returncode}")
        match = re.search(
            r"charged=(\d+) hops=(\d+) rows=(\d+) best=([-+0-9.eE]+)", stderr
        )
        if match is None:
            raise SystemExit(f"{campaign_id} seed {seed}: cannot parse campaign summary: {stderr}")
        charged, hops, rows, best = (
            int(match.group(1)), int(match.group(2)), int(match.group(3)), float(match.group(4))
        )
        if charged != campaign.budget:
            raise SystemExit(f"{campaign_id} seed {seed}: charged {charged}, expected {campaign.budget}")
        if not pending.is_file() or source_rows(pending) != rows or rows == 0:
            raise SystemExit(f"{campaign_id} seed {seed}: invalid source book")
        pending.replace(output)
    commit, tree = revision()
    record = {
        "campaign": campaign.name,
        "seed": seed,
        "budget": campaign.budget,
        "charged": charged,
        "hops": hops,
        "minima": rows,
        "best": best,
        "seconds": elapsed,
        "calls_per_second": charged / elapsed if elapsed else None,
        "sha256": digest(output),
        "code_revision": commit,
        "code_tree": tree,
        "host": os.environ.get("LANDFOLD_HOST", "cloud-run"),
        "summary": stderr,
    }
    receipt.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


def fold(
    binary: Path,
    books: list[Path],
    family: str,
    folder: Path,
    threads: int | None = None,
) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    folded = folder / "fold.csv"
    expected = sum(source_rows(book) for book in books)
    if not folded.is_file() or source_rows(folded) != expected + 1:
        command = [
            str(binary), "landfold", "--seed", "0", "--threads",
            str(max(1, threads or os.cpu_count() or 1)), "--out", str(folded),
        ]
        if family == "tip4p":
            descriptor = folder / "rigid-state.csv"
            with descriptor.open("w", newline="") as handle:
                writer = csv.writer(handle)
                for book in books:
                    with book.open() as rows:
                        for line in rows:
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
    commit, tree = revision()
    (folder / "receipt.json").write_text(
        json.dumps(
            {
                "rows": len(points),
                "landscape_sha256": digest(folder / "landscape.ld"),
                "fes_sha256": digest(folder / "fes.csv"),
                "code_revision": commit,
                "code_tree": tree,
                "host": os.environ.get("LANDFOLD_HOST", "cloud-run"),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return len(points)


def validate_book(path: Path, campaign: Campaign) -> tuple[int, int]:
    expected = campaign.size * (6 if campaign.family == "tip4p" else 3)
    rows = 0
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != expected + 1:
            raise SystemExit(
                f"{path}:{number}: {len(values) - 1} coordinates, expected {expected}"
            )
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"{path}:{number}: minimum is not finite")
        rows += 1
    if rows == 0:
        raise SystemExit(f"{path}: source book has no minima")
    return rows, expected


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


def publish_shard_status(work: Path, publish: Path) -> None:
    records = []
    for campaign_id, campaign in CAMPAIGN_IDS.items():
        source = work / "shards" / campaign_id
        for seed in range(campaign.first, campaign.last + 1):
            book = source / f"seed-{seed:04d}.min"
            receipt = source / f"seed-{seed:04d}.json"
            if not book.is_file() or not receipt.is_file():
                raise SystemExit(f"{campaign_id} seed {seed}: shard is incomplete")
            row = json.loads(receipt.read_text())
            rows, dimension = validate_book(book, campaign)
            if row["sha256"] != digest(book) or row["minima"] != rows:
                raise SystemExit(f"{receipt}: receipt does not authenticate its source book")
            if row["charged"] != campaign.budget or not row["code_revision"] or not row["host"]:
                raise SystemExit(f"{receipt}: incomplete campaign receipt")
            row["dimension"] = dimension
            records.append(row)
    publish.mkdir(parents=True, exist_ok=True)
    columns = (
        "campaign", "seed", "budget", "charged", "hops", "minima", "dimension", "best", "seconds",
        "calls_per_second", "sha256", "code_revision", "code_tree", "host",
    )
    pending = publish / "campaign-shards.csv.tmp"
    with pending.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["campaign"], row["seed"])))
    pending.replace(publish / "campaign-shards.csv")
    commit, tree = revision()
    status = {
        "status": "shards-complete",
        "code_revision": commit,
        "code_tree": tree,
        "host": os.environ.get("LANDFOLD_HOST", "cloud-run"),
        "shards": len(records),
        "charged": sum(row["charged"] for row in records),
        "minima": sum(row["minima"] for row in records),
        "remaining": [
            "lj_csm/lj75/recommended/representative",
            "lj_csm/lj75/recommended/aggregate",
            "lj_csm/lj98/derived/representative",
            "lj_csm_c26/lj98/recommended/representative",
            "tip4p/hexamer/for_tip4p/representative",
            "tip4p/hexamer/for_tip4p/aggregate",
        ],
        "blocker": (
            "the LJ75 aggregate has 1124747 minima; current landfold descriptor "
            "classification remained serial for 68 minutes before this checkpoint"
        ),
    }
    (publish / "current-rerun-status.json").write_text(
        json.dumps(status, sort_keys=True, indent=2) + "\n"
    )


def finalize(work: Path, publish: Path, threads: int | None = None) -> None:
    landfold = ROOT.parent / "tmp/anneal-target/release/anneal"
    if not landfold.is_file():
        raise SystemExit("the current-code landfold binary is not built")
    records = []
    panels = {}
    panel_jobs = []
    for campaign_id, campaign in CAMPAIGN_IDS.items():
        source = work / "shards" / campaign_id
        books = [source / f"seed-{seed:04d}.min" for seed in range(campaign.first, campaign.last + 1)]
        receipts = [source / f"seed-{seed:04d}.json" for seed in range(campaign.first, campaign.last + 1)]
        if not all(path.is_file() and source_rows(path) for path in books):
            raise SystemExit(f"{campaign_id}: source books are incomplete")
        if not all(path.is_file() for path in receipts):
            raise SystemExit(f"{campaign_id}: receipts are incomplete")
        for book, receipt in zip(books, receipts, strict=True):
            row = json.loads(receipt.read_text())
            rows, dimension = validate_book(book, campaign)
            if row["sha256"] != digest(book) or row["minima"] != rows:
                raise SystemExit(f"{receipt}: receipt does not authenticate its source book")
            if row["charged"] != campaign.budget or not row["code_revision"] or not row["host"]:
                raise SystemExit(f"{receipt}: incomplete campaign receipt")
            row["dimension"] = dimension
            records.append(row)
        for panel in campaign.panels:
            chosen = books[:1] if panel == "representative" else books
            panel_jobs.append(
                (
                    sum(source_rows(path) for path in chosen),
                    campaign,
                    panel,
                    chosen,
                    work / campaign.name / "products" / panel,
                )
            )
    cores = max(1, threads or os.cpu_count() or 1)
    workers = min(2, len(panel_jobs))
    per_job_threads = max(1, cores // workers)

    def run_panel(job: tuple[int, Campaign, str, list[Path], Path]):
        expected, campaign, panel, chosen, folder = job
        rows = fold(
            landfold,
            chosen,
            campaign.family,
            folder,
            threads=per_job_threads,
        )
        if rows != expected:
            raise SystemExit(f"{campaign.name}/{panel}: fold dropped source rows")
        return campaign, panel, folder, json.loads((folder / "receipt.json").read_text())

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        completed = pool.map(run_panel, sorted(panel_jobs, key=lambda job: -job[0]))
        for campaign, panel, folder, receipt in completed:
            panels[f"{campaign.name}/{panel}"] = receipt
    publish.mkdir(parents=True, exist_ok=True)
    for campaign in CAMPAIGNS:
        for panel in campaign.panels:
            src = work / campaign.name / "products" / panel
            dest = publish / campaign.name / panel
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("landscape.ld", "fes.csv"):
                shutil.copyfile(src / name, dest / name)
    columns = (
        "campaign", "seed", "budget", "charged", "hops", "minima", "dimension", "best", "seconds",
        "calls_per_second", "sha256", "code_revision", "code_tree", "host",
    )
    with (publish / "campaign-shards.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["campaign"], row["seed"])))
    commit, tree = revision()
    campaign_receipt = {
        "status": "complete",
        "code_revision": commit,
        "code_tree": tree,
        "host": os.environ.get("LANDFOLD_HOST", "cloud-run"),
        "campaigns": [campaign.__dict__ for campaign in CAMPAIGNS],
        "shards": len(records),
        "charged": sum(row["charged"] for row in records),
        "minima": sum(row["minima"] for row in records),
        "panels": panels,
    }
    (publish / "current-rerun.json").write_text(json.dumps(campaign_receipt, indent=2) + "\n")
    (work / "complete.stamp").write_text(f"{commit} {tree} {campaign_receipt['host']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/work/anneal-landfold-current"))
    parser.add_argument("--publish", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--campaign-id", choices=CAMPAIGN_IDS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--threads", type=int, help="total worker threads for panel folding")
    parser.add_argument("--limit-seeds", type=int, help="run at most this many seeds per campaign; never publish")
    parser.add_argument("--budget-cap", type=int, help="cap each budget for a smoke run; never publish")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.campaign_id is not None:
        if args.seed is None or args.output is None or args.receipt is None:
            parser.error("--campaign-id requires --seed, --output and --receipt")
        run_shard(args.campaign_id, args.seed, args.output, args.receipt)
        return
    if args.finalize_only:
        finalize(args.work, args.publish, args.threads)
        return
    if args.status_only:
        publish_shard_status(args.work, args.publish)
        return
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
        "host": os.environ.get("LANDFOLD_HOST", "cloud-run"),
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
    (args.work / "complete.stamp").write_text(
        f"{commit} {tree} {os.environ.get('LANDFOLD_HOST', 'cloud-run')}\n"
    )


if __name__ == "__main__":
    main()
