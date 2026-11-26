#!/usr/bin/env python3
"""Run the PR 168 walk-length gate on one core with rotated arm order."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

BASE = "7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec"
BASE_TREE = "e5ebda63f9f0117226d50c13abf33243ccf452cc"
M_BUILD_REF = "refs/scratch/gate-pr168-walklen3-sharda-m"
M_TREE = "991d5afc6d7b7f7e6c392cb68aa351af1fc50179"
O_BUILD_REF = "refs/scratch/gate-pr168-walklen3-sharda-o"
O_TREE = "7f2625597a167d3c7841b2266eec844997e53266"
BUDGET_LJ = 4_000_000
BUDGET_WATER = 2_000_000
REFERENCES = {
    "lj98": -543.665361,
    "lj102": -569.363652,
    "lj103": -575.766131,
    "lj104": -582.086642,
    "lj150": -893.310258,
    "tip4p15": -628.3722525077,
    "tip4p20": -872.9887544952,
}
CELLS = {
    "lj98": range(2012, 2048),
    "lj102": range(2012, 2024),
    "lj103": range(2012, 2024),
    "lj104": range(2012, 2024),
    "lj150": range(2012, 2024),
    "tip4p15": range(2012, 2024),
    "tip4p20": range(2012, 2024),
}
FIELDS = [
    "cell",
    "seed",
    "arm",
    "order",
    "run_position",
    "revision",
    "tree",
    "budget",
    "reference",
    "stop",
    "hit",
    "calls_to_hit",
    "charged",
    "best",
    "hops",
    "basins",
    "wall_s",
    "pot_s",
    "outside_s",
    "rss_anon_kb",
    "calls_per_walk",
    "trace_sha256",
    "geometry_sha256",
]
ORDERS = ("AMO", "OMA", "MAO", "OAM", "MOA", "AOM")


def run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdout: int | io.TextIOBase | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=stdout,
        stderr=subprocess.PIPE,
        check=True,
    )


def archive(repo: Path, tree: str, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    data = subprocess.check_output(["git", "archive", "--format=tar", tree], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as bundle:
        bundle.extractall(destination, filter="data")


def add_rss_reader(driver: Path) -> None:
    text = driver.read_text()
    if "gate_peak_rss_anon_kb" in text:
        return
    text = text.replace(
        "use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};",
        "use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};",
    )
    marker = "enum Model {"
    reader = """\
fn gate_rss_anon_kb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("RssAnon:")?
                    .trim()
                    .strip_suffix("kB")?
                    .trim()
                    .parse()
                    .ok()
            })
        })
        .unwrap_or(0)
}

"""
    if marker not in text:
        raise RuntimeError(f"cannot locate model in {driver}")
    text = text.replace(marker, reader + marker, 1)
    text = text.replace(
        "    pot_ns: AtomicU64,\n}",
        "    pot_ns: AtomicU64,\n    gate_peak_rss_anon_kb: AtomicUsize,\n}",
        1,
    )
    text = text.replace(
        "        seen.calls += 1;\n",
        """\
        seen.calls += 1;
        if seen.calls & 4095 == 0 {
            self.gate_peak_rss_anon_kb
                .fetch_max(gate_rss_anon_kb(), Ordering::Relaxed);
        }
""",
        1,
    )
    text = text.replace(
        "        pot_ns: AtomicU64::new(0),\n",
        "        pot_ns: AtomicU64::new(0),\n        gate_peak_rss_anon_kb: AtomicUsize::new(gate_rss_anon_kb()),\n",
        1,
    )
    anchor = '    println!("outside_s: {outside_s:.6}");\n'
    addition = """\
    let gate_rss_end = gate_rss_anon_kb();
    let gate_rss_peak = objective
        .gate_peak_rss_anon_kb
        .load(Ordering::Relaxed)
        .max(gate_rss_end);
    println!("gate_peak_rss_anon_kb: {gate_rss_peak}");
"""
    if anchor not in text:
        raise RuntimeError(f"cannot locate output anchor in {driver}")
    text = text.replace(anchor, anchor + addition, 1)
    driver.write_text(text)


def prepare(repo: Path, work: Path) -> dict[str, Path]:
    sources = work / "sources"
    bins: dict[str, Path] = {}
    for arm, tree in (("A", BASE), ("M", M_BUILD_REF), ("O", O_BUILD_REF)):
        source = sources / arm
        binary = source / "target" / "release" / "examples" / "h2h_seed"
        if binary.is_file():
            bins[arm] = binary
            continue
        archive(repo, tree, source)
        add_rss_reader(source / "examples" / "h2h_seed.rs")
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(source / "target")
        run(
            ["cargo", "build", "--release", "--example", "h2h_seed"],
            cwd=source,
            env=env,
        )
        if not binary.is_file():
            raise RuntimeError(f"missing binary for {arm}")
        bins[arm] = binary
    return bins


def parse_output(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key] = value
        if line.startswith("h2h_row "):
            values.update(part.split("=", 1) for part in line.split()[1:])
    required = {
        "calls",
        "charged",
        "lowest",
        "hops",
        "basins",
        "wall_seconds",
        "pot_s",
        "outside_s",
        "gate_peak_rss_anon_kb",
    }
    missing = required - values.keys()
    if missing:
        raise RuntimeError(f"driver output lacks {sorted(missing)}")
    return values


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_csv(path: Path, row: dict[str, str]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def existing_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["cell"], int(row["seed"]), row["arm"])
            for row in csv.DictReader(handle)
        }


def command_for(
    binary: Path, cell: str, seed: int, stop: float, prefix: Path
) -> list[str]:
    if cell.startswith("lj"):
        kind = "lj"
        n = cell.removeprefix("lj")
        budget = BUDGET_LJ
    else:
        kind = "tip4p"
        n = cell.removeprefix("tip4p")
        budget = BUDGET_WATER
    return [
        str(binary),
        kind,
        n,
        "0",
        str(budget),
        str(seed),
        "recommended",
        repr(stop),
        str(prefix),
    ]


def run_one(
    *,
    cell: str,
    seed: int,
    arm: str,
    order: str,
    position: int,
    binary: Path,
    work: Path,
    core: int,
) -> tuple[dict[str, str], dict[str, object]]:
    budget = BUDGET_LJ if cell.startswith("lj") else BUDGET_WATER
    reference = REFERENCES[cell]
    stop = reference + 1.0e-3
    run_dir = work / "runs" / cell / str(seed) / str(position)
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_dir / "result"
    log = run_dir / "stdout.log"
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
        }
    )
    command = ["taskset", "-c", str(core), *command_for(binary, cell, seed, stop, prefix)]
    started = time.monotonic()
    with log.open("w") as handle:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    text = log.read_text()
    if completed.returncode != 0:
        raise RuntimeError(
            f"{cell} seed {seed} position {position} exited "
            f"{completed.returncode}\n{text[-2000:]}"
        )
    parsed = parse_output(text)
    calls = int(parsed["calls"])
    best = float(parsed["lowest"])
    hit = best <= stop
    hops = int(parsed["hops"])
    revisions = {
        "A": BASE,
        "M": "4ab2681ae6bf72d61c633197e88047f00cdd7efa",
        "O": "fd5449766b4f1f693de2245933afa609a93a5c6b",
    }
    trees = {"A": BASE_TREE, "M": M_TREE, "O": O_TREE}
    row = {
        "cell": cell,
        "seed": str(seed),
        "arm": arm,
        "order": order,
        "run_position": str(position),
        "revision": revisions[arm],
        "tree": trees[arm],
        "budget": str(budget),
        "reference": repr(reference),
        "stop": repr(stop),
        "hit": "1" if hit else "0",
        "calls_to_hit": str(calls if hit else budget + 1),
        "charged": parsed["charged"],
        "best": parsed["lowest"],
        "hops": str(hops),
        "basins": parsed["basins"],
        "wall_s": parsed["wall_seconds"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
        "rss_anon_kb": parsed["gate_peak_rss_anon_kb"],
        "calls_per_walk": f"{calls / max(hops, 1):.12g}",
        "trace_sha256": digest(Path(f"{prefix}.trace.csv")),
        "geometry_sha256": digest(Path(f"{prefix}.xyz")),
    }
    recorded = command[2:-1] + ["<run-prefix>"]
    recorded[1] = f"{arm}:target/release/examples/h2h_seed"
    session = {
        "arm": arm,
        "cell": cell,
        "command": recorded,
        "elapsed_s": round(elapsed, 6),
        "order": order,
        "position": position,
        "returncode": completed.returncode,
        "seed": seed,
    }
    return row, session


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    output = Path(__file__).resolve().parent
    rows_path = output / "raw_rows.csv"
    sessions_path = output / "sessions.jsonl"
    bins = prepare(repo, args.work)
    if args.prepare_only:
        return 0
    core = min(os.sched_getaffinity(0))
    have = existing_keys(rows_path)
    deadline = time.monotonic() + args.max_hours * 3600.0
    panel = [
        (cell, seed)
        for cell, seeds in CELLS.items()
        for seed in seeds
        if seed < 2024
    ]
    lj98_extension = [("lj98", seed) for seed in CELLS["lj98"] if seed >= 2024]
    for cell, seed in panel + lj98_extension:
        order = ORDERS[(seed - 2012) % len(ORDERS)]
        missing = [arm for arm in order if (cell, seed, arm) not in have]
        if missing and time.monotonic() >= deadline:
            print("campaign deadline reached", flush=True)
            return 3
        for position, arm in enumerate(order, 1):
            key = (cell, seed, arm)
            if key in have:
                continue
            row, session = run_one(
                cell=cell,
                seed=seed,
                arm=arm,
                order=order,
                position=position,
                binary=bins[arm],
                work=args.work,
                core=core,
            )
            append_csv(rows_path, row)
            append_jsonl(sessions_path, session)
            have.add(key)
            print(
                f"{cell} {seed} {arm} order={order} position={position} "
                f"hit={row['hit']} calls={row['calls_to_hit']} "
                f"outside={row['outside_s']} rss={row['rss_anon_kb']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
