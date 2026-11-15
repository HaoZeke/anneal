#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append-only rows for the replace-chain restart against develop.

One job at a time. The process refuses to run unless the UTS hostname is
bench and the working directory is /work. A row already stored for the
same system, arm, seed and code revision is left in place.

The thirteen systems use the paper budgets. Seeds 2000–2011 are the
twelve-seed panel. Seeds 2012–2023 extend LJ76, LJ98, LJ102, LJ150 and
Morse M55 at rho 14 to twenty-four. Develop is 7192e9f8. The restart
arm is the binary built with the plateau restart on.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "bench" / "lj150d" / "seeds.csv"
HIT_TOL = 0.001
SCHEDULE_SEED = 20261111
N12 = tuple(range(2000, 2012))
N24 = tuple(range(2012, 2024))

# name, kind, reference
SYSTEMS = (
    ("lj38", "lj", -173.928427),
    ("lj55", "lj", -279.248470),
    ("lj75", "lj", -397.492331),
    ("lj76", "lj", -402.894866),
    ("lj77", "lj", -409.083517),
    ("lj98", "lj", -543.665361),
    ("lj102", "lj", -569.363652),
    ("lj103", "lj", -575.766131),
    ("lj104", "lj", -582.086642),
    ("lj150", "lj", -893.310258),
    ("au55", "au", -191.03085280),
    ("tip4p15", "tip4p", -628.3722525077),
    ("m55r14", "morse", -220.646208),
)
EXTEND = ("lj76", "lj98", "lj102", "lj150", "m55r14")
FIELDS = (
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
)


def require_bench() -> str:
    host = os.uname().nodename
    if host != "bench":
        print(f"uts hostname is {host}, expected bench", file=sys.stderr)
        return ""
    if Path.cwd() != Path("/work"):
        print(f"working directory is {Path.cwd()}, expected /work", file=sys.stderr)
        return ""
    return host


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def key_of(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["system"], row["arm"], row["seed"], row["code_revision"])


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "H2H_BUDGET",
        "H2H_SYSTEM",
        "H2H_SEED",
        "H2H_TARGET",
        "H2H_ROW",
        "H2H_TOL",
        "ANNEAL_DISC",
        "ANNEAL_BASIN",
        "ANNEAL_ACC",
        "ANNEAL_PLATEAU",
        "ANNEAL_BEST",
    ):
        env.pop(name, None)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    return subprocess.run(
        ["taskset", "-c", "0", *cmd],
        check=False,
        text=True,
        capture_output=True,
        env=env,
        cwd="/work",
    )


def from_csv(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line and not line.startswith("arm ")]
    header = None
    data = None
    for line in lines:
        if line.startswith("system,"):
            header = line
            continue
        if header and "," in line:
            data = line
            break
    if header is None or data is None:
        raise ValueError(f"no csv row in\n{text[-500:]}")
    return next(csv.DictReader([header, data]))


def from_seed(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            fields = dict(tok.split("=", 1) for tok in line.split()[1:])
            hit = fields.get("hit", "0") in {"1", "true", "True"}
            wall = float(fields["wall_s"])
            pot = float(fields.get("pot_s", "nan"))
            outside = wall - pot
            return {
                "hit": "true" if hit else "false",
                "evals_to_hit": fields["evals_to_hit"],
                "charged": fields.get("charged", fields["evals_to_hit"]),
                "best": fields["best"],
                "wall_s": f"{wall:.6f}",
                "pot_s": f"{pot:.6f}",
                "outside_s": f"{outside:.6f}",
                "budget": fields["budget"],
            }
    raise ValueError(f"no h2h_row in\n{text[-500:]}")


def normalise(raw: dict[str, str]) -> dict[str, str]:
    hit = raw.get("hit", "false")
    hit = "true" if hit in {"1", "true", "True"} else "false"
    wall = float(raw["wall_s"])
    pot = float(raw["pot_s"]) if raw.get("pot_s") else float("nan")
    if "outside_s" in raw and raw["outside_s"] != "":
        outside = float(raw["outside_s"])
    else:
        outside = wall - pot
    return {
        "budget": raw["budget"],
        "hit": hit,
        "evals_to_hit": raw["evals_to_hit"],
        "charged": raw.get("charged", raw["evals_to_hit"]),
        "best": raw["best"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
    }


def one_job(binary: Path, kind: str, system: str, seed: int, reference: float) -> dict[str, str]:
    if kind == "lj":
        proc = run([str(binary / "h2h_gmin_proof"), "run", system, str(seed)])
        parsed = from_csv(proc.stdout)
    elif kind == "au":
        proc = run([str(binary / "h2h_metal"), "run", system, str(seed)])
        parsed = from_csv(proc.stdout)
    elif kind == "morse":
        proc = run([str(binary / "h2h_morse"), "run", system, str(seed)])
        parsed = from_csv(proc.stdout)
    elif kind == "tip4p":
        stop = reference + HIT_TOL
        prefix_dir = ROOT / "runs" / "lj150d" / system
        prefix_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(prefix_dir / f"seed{seed}")
        proc = run(
            [
                str(binary / "h2h_seed"),
                "tip4p",
                "15",
                "0",
                "2000000",
                str(seed),
                "recommended",
                f"{stop:.10f}",
                prefix,
            ]
        )
        parsed = from_seed(proc.stdout)
    else:
        raise RuntimeError(kind)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-800:] + proc.stdout[-800:])
    return normalise(parsed)


def jobs() -> list[tuple[str, str, float, int, str]]:
    """System, kind, reference, seed, arm. Shuffled inside each system."""
    planned: list[tuple[str, str, float, int, str]] = []
    arms = ("develop", "v1")

    def extend(names: tuple[str, ...], seeds: tuple[int, ...]) -> None:
        for name, kind, reference in SYSTEMS:
            if name not in names:
                continue
            group = [
                (name, kind, reference, seed, arm) for seed in seeds for arm in arms
            ]
            random.Random(SCHEDULE_SEED).shuffle(group)
            planned.extend(group)

    extend(tuple(name for name, _kind, _reference in SYSTEMS), N12)
    extend(EXTEND, N24)
    return planned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--develop", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--develop-rev", required=True)
    parser.add_argument("--restart-rev", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    host = require_bench()
    if not host:
        return 2
    existing = {key_of(row) for row in load_rows(args.out)}
    arms = {
        "develop": (args.develop, args.develop_rev),
        "v1": (args.restart, args.restart_rev),
    }
    planned = jobs()
    print(f"jobs {len(planned)} stored {len(existing)}", flush=True)
    for name, kind, reference, seed, arm in planned:
        binary, revision = arms[arm]
        row = {
            "system": name,
            "arm": arm,
            "seed": str(seed),
            "code_revision": revision,
            "host": host,
        }
        if key_of(row) in existing:
            continue
        print(f"START {name} {arm} {seed}", flush=True)
        measured = one_job(binary, kind, name, seed, reference)
        row.update(measured)
        append_row(args.out, row)
        existing.add(key_of(row))
        print(
            f"DONE {name} {arm} {seed} hit={row['hit']} evals={row['evals_to_hit']} "
            f"wall={row['wall_s']} pot={row['pot_s']}",
            flush=True,
        )
    print("PANEL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
