#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GMIN against the paper preset, at the paper force-call budgets.

The default sizes are LJ98 and LJ104, the panel in
``bench/gmin-h2h/results.csv``. ``--size`` picks any other size in
``PAPER_BUDGETS``; LJ102, LJ150 and LJ200 are ``bench/lj-large``. Six
seeds, the same list on both arms. Within a size the (seed, arm) pairs
are shuffled, and sizes run in the order given. A hit is a quenched energy within
0.001 of the Cambridge minimum. A miss is stored as the full budget, and the median and
interquartile range use Hyndman-Fan type 7 on that column. The wall
ratio is the GMIN median wall divided by the anneal median wall.

The search binaries are the existing ``h2h_gmin_proof`` example and a
GMIN built from Wales's public tarball. This process refuses to run
unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from target_tol import DEFAULT_HIT_TOL  # noqa: E402
from machine_provenance import (  # noqa: E402
    machine_provenance,
    read_built_gmin_source,
    require_gmin_build_key,
    validate_saved_gmin_provenance,
    validate_saved_row_identity,
)
from gmin_pin import (  # noqa: E402
    TARBALL_SHA256,
    read_svnrev,
)

PAPER_BUDGETS = {
    98: 4_000_000,
    102: 4_000_000,
    104: 4_000_000,
    150: 4_000_000,
    200: 4_000_000,
}
SIZES = (98, 104)
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("anneal", "gmin")
SCHEDULE_SEED = 20261111
HIT_TOL = DEFAULT_HIT_TOL

SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "planned",
    "hits",
    "hits_over_n",
    "budget",
    "evals_median",
    "evals_iqr",
    "wall_median_s",
    "wall_iqr_s",
    "wall_ratio",
    "seeds",
    "schedule_seed",
    "gmin_url",
    "gmin_revision",
    "anneal_head",
    "code_revision",
    "host",
    "machine_id",
    "cpu_model",
]
PER_SEED_FIELDS = [
    "system",
    "code",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "gradient_passes",
    "energy_only_entries",
    "wall_s",
    "best",
    "pot_s",
    "head",
    "tree",
    "gmin_url",
    "gmin_revision",
    "gmin_build_key",
    "anneal_head",
    "code_revision",
    "host",
    "machine_id",
    "cpu_model",
]


def type7(sorted_values: list[float], p: float) -> float:
    """Hyndman-Fan type 7. ``sorted_values`` is ascending."""
    n = len(sorted_values)
    if n < 1:
        raise ValueError("type 7 needs a sample")
    if n == 1:
        return sorted_values[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    q1 = type7(ordered, 0.25)
    q3 = type7(ordered, 0.75)
    return type7(ordered, 0.5), q3 - q1


def schedule(
    sizes: tuple[int, ...],
    seeds: tuple[int, ...],
    arms: tuple[str, ...],
    rng_seed: int,
) -> list[tuple[int, int, str]]:
    """Sizes in the given order. Pairs inside a size are shuffled."""
    rng = random.Random(rng_seed)
    order: list[tuple[int, int, str]] = []
    for n in sizes:
        jobs = [(n, seed, arm) for seed in seeds for arm in arms]
        rng.shuffle(jobs)
        order.extend(jobs)
    return order


def system_name(n: int) -> str:
    return f"lj{n}"


def system_size(system: str) -> int:
    if not system.startswith("lj") or not system[2:].isdigit():
        raise ValueError(f"invalid LJ system name {system!r}")
    return int(system[2:])


def load_source(path: Path) -> tuple[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    url = fields.get("url", "")
    revision = fields.get("revision", "")
    if not url or not revision:
        raise SystemExit(f"{path} has no url and revision")
    return url, revision


def binary_budget(binary: Path, n: int) -> tuple[int, float]:
    proc = subprocess.run(
        [str(binary), "budget", system_name(n)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"budget for LJ{n} returned {proc.stdout!r}")
    return int(lines[0]), float(lines[1])


def write_start(binary: Path, n: int, seed: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return
    subprocess.run(
        [str(binary), "coords", system_name(n), str(seed), str(path)],
        check=True,
    )


def parse_anneal(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines[:-1]):
        if line.startswith("system,code,seed,budget,"):
            return next(csv.DictReader(lines[index : index + 2]))
    data = [line for line in text.splitlines() if ",anneal," in line]
    if len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,"
        "head,tree,cpu,pins"
    )
    row = next(csv.DictReader([header, data[0]]))
    return row


def parse_gmin(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0].startswith("system,code,seed,budget,"):
        return next(csv.DictReader(lines[:2]))
    fields: dict[str, str] = {}
    for part in text.split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    if "hit" not in fields or "evals_to_hit" not in fields:
        raise ValueError(f"no gmin row in {text!r}")
    return fields


def write_per_seed_row(
    path: Path,
    row: dict[str, str],
    machine_id: str,
    cpu_model: str,
    *,
    code_revision: str = "",
    anneal_head: str = "",
    gmin_url: str = "",
    gmin_revision: str = "",
    gmin_build_key: str = "",
) -> dict[str, str]:
    values = {field: row.get(field, "") for field in PER_SEED_FIELDS}
    resolved_head = anneal_head or row.get("anneal_head") or row.get("head", "")
    resolved_gmin = gmin_revision or row.get("gmin_revision", "")
    resolved_code = (
        code_revision or row.get("code_revision") or resolved_head or resolved_gmin
    )
    values.update(
        {
            "gmin_url": gmin_url or row.get("gmin_url", ""),
            "gmin_revision": resolved_gmin,
            "gmin_build_key": gmin_build_key or row.get("gmin_build_key", ""),
            "anneal_head": resolved_head,
            "code_revision": resolved_code,
            "host": row.get("host") or os.uname().nodename,
            "machine_id": machine_id,
            "cpu_model": cpu_model,
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SEED_FIELDS)
        writer.writeheader()
        writer.writerow(values)
    return values


def _env(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    return env


def run_anneal(
    binary: Path,
    n: int,
    seed: int,
    budget: int,
    coords: Path,
    log: Path,
    saved: Path,
    machine_id: str,
    cpu_model: str,
    head: str = "",
) -> dict[str, str]:
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "run", system_name(n), str(seed), str(coords)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=_env(budget),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal LJ{n} seed {seed} exited {proc.returncode}")
    row = parse_anneal(text)
    if int(row["budget"]) != budget or row["hit"] not in ("0", "1", "true", "false"):
        raise RuntimeError(f"anneal row {row}")
    reported_head = row.get("head", "").strip()
    if head and reported_head and reported_head != head:
        raise RuntimeError(f"anneal binary reported {reported_head}, expected {head}")
    effective_head = head or reported_head
    return write_per_seed_row(
        saved,
        row,
        machine_id,
        cpu_model,
        code_revision=effective_head,
        anneal_head=effective_head,
    )


def run_gmin(
    gmin: Path,
    data: Path,
    n: int,
    seed: int,
    budget: int,
    target: float,
    coords: Path,
    work: Path,
    log: Path,
    saved: Path,
    machine_id: str,
    cpu_model: str,
    *,
    url: str = "",
    revision: str = "",
    build_key: str = "",
) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    (work / "data").write_text(data.read_text() + f"RANSEED {seed}\n")
    (work / "coords").write_bytes(coords.read_bytes())
    env = _env(budget)
    env.update(
        {
            "H2H_SYSTEM": system_name(n),
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{target:.10f}",
            "H2H_TOL": f"{HIT_TOL}",
            "H2H_ROW": str(work / "row.txt"),
        }
    )
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(gmin)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=work,
        )
    row_path = work / "row.txt"
    if not row_path.is_file():
        raise RuntimeError(
            f"gmin LJ{n} seed {seed} exited {proc.returncode} with no row"
        )
    text = row_path.read_text()
    row = parse_gmin(text)
    if int(row["budget"]) != budget:
        raise RuntimeError(f"gmin budget {row['budget']} != {budget}")
    required = ("gradient_passes", "energy_only_entries")
    if any(field not in row for field in required):
        raise RuntimeError(f"gmin row has no parity counters: {row}")
    if (
        int(row["gradient_passes"]) != int(row["charged"])
        or not 0 <= int(row["gradient_passes"]) <= budget
        or int(row["energy_only_entries"]) < 0
    ):
        raise RuntimeError(f"invalid gmin counters: {row}")
    verified_key = require_gmin_build_key(build_key, "gmin build_key")
    return write_per_seed_row(
        saved,
        row,
        machine_id,
        cpu_model,
        code_revision=revision,
        gmin_url=url,
        gmin_revision=revision,
        gmin_build_key=verified_key,
    )


def as_result(n: int, arm: str, row: dict[str, str]) -> dict:
    hit = row["hit"] in ("1", "true")
    return {
        "n": n,
        "arm": arm,
        "seed": int(row["seed"]),
        "budget": int(row["budget"]),
        "hit": hit,
        "evals": float(row["evals_to_hit"]),
        "wall_s": float(row["wall_s"]),
        "best": float(row["best"]),
        "gmin_url": row.get("gmin_url", ""),
        "gmin_revision": row.get("gmin_revision", ""),
        "gmin_build_key": row.get("gmin_build_key", ""),
        "anneal_head": row.get("anneal_head") or row.get("head", ""),
        "code_revision": row.get("code_revision", ""),
        "host": row.get("host", ""),
        "machine_id": row.get("machine_id", ""),
        "cpu_model": row.get("cpu_model", ""),
    }


def _provenance_value(items: list[dict], key: str, label: str, fallback: str) -> str:
    values = {
        str(item.get(key, "")).strip()
        for item in items
        if str(item.get(key, "")).strip()
    }
    if len(values) > 1:
        raise ValueError(f"mixed {label} values on rows: {sorted(values)}")
    if values:
        actual = next(iter(values))
        if fallback and actual != fallback:
            raise ValueError(
                f"refusing to relabel {label} {actual!r} with {fallback!r}"
            )
        return actual
    return fallback


def summarise(
    rows: list[dict],
    seeds: tuple[int, ...],
    url: str = "",
    revision: str = "",
    head: str = "",
    machine_id: str = "",
    cpu_model: str = "",
) -> list[dict[str, str]]:
    by_n: dict[int, dict[str, list[dict]]] = {}
    for row in rows:
        by_n.setdefault(row["n"], {}).setdefault(row["arm"], []).append(row)
    seed_text = " ".join(str(seed) for seed in seeds)
    out: list[dict[str, str]] = []
    for n in sorted(by_n):
        arms = by_n[n]
        anneal_group = arms.get("anneal", [])
        gmin_group = arms.get("gmin", [])
        resolved_head = _provenance_value(
            anneal_group, "code_revision", "anneal code_revision", head
        )
        resolved_head = _provenance_value(
            anneal_group, "anneal_head", "anneal_head", resolved_head
        )
        resolved_revision = _provenance_value(
            gmin_group, "code_revision", "gmin code_revision", revision
        )
        resolved_revision = _provenance_value(
            gmin_group, "gmin_revision", "gmin_revision", resolved_revision
        )
        resolved_url = _provenance_value(gmin_group, "gmin_url", "gmin_url", url)
        walls = {
            arm: median_iqr([item["wall_s"] for item in group])[0]
            for arm, group in arms.items()
            if group
        }
        ratio = ""
        if "anneal" in walls and "gmin" in walls and walls["anneal"] > 0.0:
            ratio = f"{walls['gmin'] / walls['anneal']:.6f}"
        for arm in ARMS:
            group = arms.get(arm, [])
            if not group:
                continue
            evals_median, evals_iqr = median_iqr([item["evals"] for item in group])
            wall_median, wall_iqr = median_iqr([item["wall_s"] for item in group])
            hits = sum(bool(item["hit"]) for item in group)
            arm_code_rev = _provenance_value(
                group,
                "code_revision",
                f"{arm} code_revision",
                resolved_head if arm == "anneal" else resolved_revision,
            )
            hosts = sorted(
                {str(item.get("host", "")).strip() for item in group if str(item.get("host", "")).strip()}
            )
            mids = sorted(
                {str(item.get("machine_id", "")).strip() for item in group if str(item.get("machine_id", "")).strip()}
            )
            cpus = sorted(
                {str(item.get("cpu_model", "")).strip() for item in group if str(item.get("cpu_model", "")).strip()}
            )
            out.append(
                {
                    "system": system_name(n),
                    "arm": arm,
                    "n": str(len(group)),
                    "planned": str(len(seeds)),
                    "hits": str(hits),
                    "hits_over_n": f"{hits}/{len(group)}",
                    "budget": str(group[0]["budget"]),
                    "evals_median": f"{evals_median:.6f}",
                    "evals_iqr": f"{evals_iqr:.6f}",
                    "wall_median_s": f"{wall_median:.6f}",
                    "wall_iqr_s": f"{wall_iqr:.6f}",
                    "wall_ratio": ratio,
                    "seeds": seed_text,
                    "schedule_seed": str(SCHEDULE_SEED),
                    "gmin_url": resolved_url,
                    "gmin_revision": resolved_revision,
                    "anneal_head": resolved_head,
                    "code_revision": arm_code_rev,
                    "host": ";".join(hosts) if hosts else os.uname().nodename,
                    "machine_id": ";".join(mids) if mids else machine_id,
                    "cpu_model": ";".join(cpus) if cpus else cpu_model,
                }
            )
    return out


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def job_paths(runs: Path, n: int, seed: int, arm: str) -> tuple[Path, Path]:
    directory = runs / system_name(n)
    return directory / f"{arm}-{seed}.row", directory / f"{arm}-{seed}.log"


def _parse_saved_row(text: str, arm: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines[:-1]):
        cols = next(csv.reader([line]), [])
        if "system" in cols and ("code" in cols or "arm" in cols) and "seed" in cols:
            return next(csv.DictReader(lines[index : index + 2]))
    if any(line.startswith("h2h_row ") for line in lines):
        return parse_gmin(text)
    if any(",anneal," in line for line in lines):
        return parse_anneal(text)
    anneal_header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,"
        "head,tree,cpu,pins"
    )
    anneal_cols = anneal_header.split(",")
    candidates = [
        line
        for line in lines
        if len(next(csv.reader([line]), [])) == len(anneal_cols)
    ]
    if len(candidates) == 1:
        return next(csv.DictReader([anneal_header, candidates[0]]))
    return parse_anneal(text) if arm == "anneal" else parse_gmin(text)


def load_saved(
    path: Path,
    system: str,
    seed: int,
    arm: str,
    budget: int,
    machine_id: str,
    cpu_model: str,
    *,
    code_revision: str,
    gmin_url: str = "",
    gmin_build_key: str = "",
) -> dict | None:
    n = system_size(system)
    if arm == "gmin":
        require_gmin_build_key(gmin_build_key, "expected gmin_build_key")
    if not path.is_file():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    row = _parse_saved_row(text, arm)
    validate_saved_row_identity(
        row,
        path=path,
        system=system,
        arm=arm,
        seed=seed,
        budget=budget,
    )
    if arm == "gmin":
        validate_saved_gmin_provenance(
            row,
            path=path,
            revision=code_revision,
            build_key=gmin_build_key,
            url=gmin_url,
            revision_fields=("code_revision", "gmin_revision"),
        )
        if (
            not str(row.get("gradient_passes", "")).strip()
            or not str(row.get("energy_only_entries", "")).strip()
        ):
            return None
        if (
            int(row["gradient_passes"]) != int(row["charged"])
            or not 0 <= int(row["gradient_passes"]) <= budget
            or int(row["energy_only_entries"]) < 0
        ):
            raise ValueError(f"{path}: invalid gmin counters: {row}")
    if row.get("machine_id") != machine_id or row.get("cpu_model") != cpu_model:
        return None
    if not code_revision:
        return None
    if arm == "anneal":
        if row.get("code_revision") != code_revision:
            return None
        if row.get("anneal_head") != code_revision:
            return None
        if row.get("head") and row["head"] != code_revision:
            return None
    return as_result(n, arm, row)


def note(progress: Path, line: str) -> None:
    print(line, flush=True)
    with progress.open("a") as handle:
        handle.write(line + "\n")


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12:
        raise SystemExit("type 7 lower quartile")
    if abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7 median")
    median, iqr = median_iqr(sample)
    if abs(median - 3.5) > 1e-12 or abs(iqr - 2.5) > 1e-12:
        raise SystemExit("median iqr")
    if any(budget != 4_000_000 for budget in PAPER_BUDGETS.values()):
        raise SystemExit("paper budgets")
    if set(PAPER_BUDGETS) != {98, 102, 104, 150, 200}:
        raise SystemExit("sizes")
    if SIZES != (98, 104):
        raise SystemExit("default sizes")
    order = schedule(SIZES, SEEDS, ARMS, SCHEDULE_SEED)
    seen: list[int] = []
    for n, _seed, _arm in order:
        if not seen or seen[-1] != n:
            seen.append(n)
    if seen != list(SIZES):
        raise SystemExit(f"size order {seen}")
    for n in SIZES:
        jobs = [(seed, arm) for size, seed, arm in order if size == n]
        expected = [(seed, arm) for seed in SEEDS for arm in ARMS]
        if sorted(jobs) != sorted(expected):
            raise SystemExit(f"LJ{n} is not the shared seed list")
        arms = [arm for _seed, arm in jobs]
        if arms == ["anneal"] * len(SEEDS) + ["gmin"] * len(SEEDS):
            raise SystemExit(f"LJ{n} is not interleaved")
        if arms == ["gmin"] * len(SEEDS) + ["anneal"] * len(SEEDS):
            raise SystemExit(f"LJ{n} is not interleaved")
    print("ok")


def load_source_metadata(source_path: Path, svnrev_path: Path) -> tuple[str, str, str]:
    load_source(source_path)
    return read_built_gmin_source(source_path, svnrev_path)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--svnrev", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", choices=("anneal", "gmin", "both"), default="both")
    parser.add_argument("--budget", type=int)
    args = parser.parse_args(argv[1:])
    sizes = tuple(args.size) if args.size else SIZES
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = ARMS if args.arm == "both" else (args.arm,)
    for n in sizes:
        if n not in PAPER_BUDGETS:
            raise SystemExit(f"no paper budget for LJ{n}")
        if args.budget is not None and args.budget != PAPER_BUDGETS[n]:
            raise SystemExit(
                f"LJ{n} budget {args.budget} is not the paper budget {PAPER_BUDGETS[n]}"
            )
    try:
        url, revision, build_key = load_source_metadata(args.source, args.svnrev)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    head = args.head_file.read_text().strip()
    machine_id, cpu_model = machine_provenance()
    progress = args.runs.parent / "progress.txt"
    progress.parent.mkdir(parents=True, exist_ok=True)
    order = schedule(sizes, seeds, arms, SCHEDULE_SEED)
    note(progress, "schedule " + " ".join(f"lj{n}:{arm}:{seed}" for n, seed, arm in order))
    rows: list[dict] = []
    for n, seed, arm in order:
        budget = PAPER_BUDGETS[n]
        _budget, target = binary_budget(args.binary, n)
        if _budget != budget:
            raise SystemExit(f"binary budget {_budget} != paper budget {budget}")
        coords = args.starts / system_name(n) / f"seed-{seed}.xyz"
        write_start(args.binary, n, seed, coords)
        saved, log = job_paths(args.runs, n, seed, arm)
        saved.parent.mkdir(parents=True, exist_ok=True)
        expected_revision = head if arm == "anneal" else revision
        existing = (
            load_saved(
                saved,
                system_name(n),
                seed,
                arm,
                budget,
                machine_id,
                cpu_model,
                code_revision=expected_revision,
                gmin_url=url,
                gmin_build_key=build_key,
            )
            if saved.is_file()
            else None
        )
        if existing is not None:
            rows.append(existing)
            note(
                progress,
                f"skip n={n} arm={arm} seed={seed} hit={int(existing['hit'])} "
                f"evals={existing['evals']:.0f}",
            )
            continue
        note(progress, f"start n={n} arm={arm} seed={seed} budget={budget}")
        if arm == "anneal":
            raw = run_anneal(
                args.binary,
                n,
                seed,
                budget,
                coords,
                log,
                saved,
                machine_id,
                cpu_model,
                head,
            )
        else:
            raw = run_gmin(
                args.gmin,
                args.data,
                n,
                seed,
                budget,
                target,
                coords,
                args.runs / system_name(n) / f"gmin-{seed}",
                log,
                saved,
                machine_id,
                cpu_model,
                url=url,
                revision=revision,
                build_key=build_key,
            )
        result = as_result(n, arm, raw)
        rows.append(result)
        note(
            progress,
            f"done n={n} arm={arm} seed={seed} hit={int(result['hit'])} "
            f"evals={result['evals']:.0f} wall_s={result['wall_s']:.3f} "
            f"best={result['best']:.6f}",
        )
        if args.summary:
            write_summary(
                args.summary,
                summarise(rows, seeds, url, revision, head, machine_id, cpu_model),
            )
    text_rows = summarise(rows, seeds, url, revision, head, machine_id, cpu_model)
    if args.summary:
        write_summary(args.summary, text_rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(text_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
