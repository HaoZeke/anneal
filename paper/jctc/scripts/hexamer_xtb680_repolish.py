#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Re-polish the water-hexamer finals on a newer xtb.

The CREST and anneal searches are left as they stand. The eighteen raw
finals of the six-seed GFN2 comparison, three arms by seeds 0 to 5, are
relaxed and single-pointed with the flags in logs/water/VERSIONS:

    xtb --gfn 2 --opt vtight --acc 0.01 --iterations 500
    xtb --gfn 2 --sp --acc 0.01 --iterations 500

The reference of a seed is its polished CREST structure when that
optimization converges and the waters stay one cluster. Otherwise it is
the lowest intact converged polish, the rule written in NUMBERS.tsv.
The gap is the new arm energy minus that reference. The CSV has one
row per final: structure, seed, the 6.7.1 energy, the new energy, and
that gap. Raw xtb output stays under /work.

Run from a /work directory, on a host named bench:

    sudo unshare --uts bash -c 'hostname bench; python3 paper/jctc/scripts/hexamer_xtb680_repolish.py'
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import socket
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import fetch_paper_data

SEEDS = range(6)
ARMS = ("crest", "fair", "paper4000")
OH_BOND_A = 1.15
HBOND_OO_A = 3.5
POLISH = ["--gfn", "2", "--opt", "vtight", "--acc", "0.01", "--iterations", "500"]
SINGLE_POINT = ["--gfn", "2", "--sp", "--acc", "0.01", "--iterations", "500"]
ENERGY = re.compile(
    r"TOTAL ENERGY\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+Eh"
)


def fail(message: str) -> None:
    raise SystemExit(f"hexamer_xtb680_repolish: {message}")


def require_bench(work: Path) -> None:
    if socket.gethostname() != "bench":
        fail(f"hostname is {socket.gethostname()}, want bench")
    if not str(work).startswith("/work/") and work != Path("/work"):
        fail(f"work directory {work} is not under /work")


def raw_final(logs: Path, seed: int, arm: str) -> Path:
    if arm == "crest":
        return logs / f"anneal-seed{seed}" / "crest_best.xyz"
    if arm == "fair":
        return logs / f"anneal-seed{seed}" / "best.xyz"
    if arm == "paper4000":
        return logs / f"anneal-paper4000-seed{seed}" / "best.xyz"
    fail(f"unknown arm {arm}")
    raise AssertionError


def read_xyz(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text().splitlines()
    count = int(lines[0].split()[0])
    symbols: list[str] = []
    rows: list[list[float]] = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        symbols.append(parts[0])
        rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(rows) != count:
        fail(f"{path} claimed {count} atoms and held {len(rows)}")
    return symbols, rows


def distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right)))


def cluster_intact(path: Path) -> bool:
    """True when every oxygen keeps two hydrogens and the oxygens are one cluster."""
    symbols, rows = read_xyz(path)
    oxygens = [i for i, symbol in enumerate(symbols) if symbol.upper().startswith("O")]
    hydrogens = [i for i, symbol in enumerate(symbols) if symbol.upper().startswith("H")]
    if len(oxygens) != 6 or len(hydrogens) != 12:
        return False
    owned: dict[int, list[int]] = {oxygen: [] for oxygen in oxygens}
    for hydrogen in hydrogens:
        nearest = min(oxygens, key=lambda oxygen: distance(rows[hydrogen], rows[oxygen]))
        owned[nearest].append(hydrogen)
    for oxygen, pair in owned.items():
        if len(pair) != 2:
            return False
        for hydrogen in pair:
            if distance(rows[hydrogen], rows[oxygen]) > OH_BOND_A:
                return False
    seen = {oxygens[0]}
    frontier = [oxygens[0]]
    while frontier:
        here = frontier.pop()
        for there in oxygens:
            if there in seen:
                continue
            if distance(rows[here], rows[there]) < HBOND_OO_A:
                seen.add(there)
                frontier.append(there)
    return len(seen) == len(oxygens)


def xtb_env(source: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["XTBPATH"] = str(source)
    env["TMPDIR"] = "."
    return env


def run_xtb(xtb: Path, source: Path, work: Path, xyz_name: str, args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [str(xtb), xyz_name, *args],
        cwd=work,
        env=xtb_env(source),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def parse_energy(text: str) -> float | None:
    found = ENERGY.findall(text)
    if not found:
        return None
    value = float(found[-1])
    if not math.isfinite(value):
        return None
    return value


def polish_one(task: tuple[str, str, int, str, str, str]) -> dict[str, object]:
    xtb, source, seed, arm, raw, work_root = task
    work = Path(work_root) / f"s{seed}-{arm}"
    work.mkdir(parents=True, exist_ok=True)
    dest = work / "in.xyz"
    dest.write_bytes(Path(raw).read_bytes())
    code, opt_text = run_xtb(Path(xtb), Path(source), work, "in.xyz", POLISH)
    (work / "opt.out").write_text(opt_text)
    polished = work / "xtbopt.xyz"
    converged = (
        code == 0
        and "GEOMETRY OPTIMIZATION CONVERGED" in opt_text
        and polished.is_file()
    )
    energy = None
    sp_text = ""
    if polished.is_file():
        sp_code, sp_text = run_xtb(Path(xtb), Path(source), work, "xtbopt.xyz", SINGLE_POINT)
        (work / "sp.out").write_text(sp_text)
        if sp_code == 0:
            energy = parse_energy(sp_text)
    intact = cluster_intact(polished) if polished.is_file() else False
    return {
        "seed": seed,
        "arm": arm,
        "converged": converged,
        "intact": intact,
        "energy": energy,
    }


def load_old(numbers: Path) -> dict[tuple[int, str], dict[str, float]]:
    header = None
    found: dict[tuple[int, str], dict[str, float]] = {}
    for line in numbers.read_text().splitlines():
        if line.startswith("# seed\t") or line.startswith("seed\t"):
            header = line.lstrip("# ").split("\t")
            continue
        if header is None or not line or line.startswith("#") or line.startswith("summary"):
            continue
        record = dict(zip(header, line.split("\t")))
        found[(int(record["seed"]), record["arm"])] = {
            "energy": float(record["polished_energy_Eh"]),
        }
    return found


def fmt(value: float) -> str:
    return f"{value:.12e}"


def choose_reference(by_arm: dict[str, dict[str, object]]) -> dict[str, object] | None:
    crest = by_arm.get("crest")
    if crest and crest["converged"] and crest["intact"] and crest["energy"] is not None:
        return crest
    pool = [
        found
        for found in by_arm.values()
        if found["converged"] and found["intact"] and found["energy"] is not None
    ]
    if not pool:
        return None
    return min(pool, key=lambda found: float(found["energy"]))


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["structure", "seed", "old_energy_Eh", "new_energy_Eh", "gap_Eh"])
        writer.writerows(rows)


def main() -> None:
    here = Path(__file__).resolve()
    paper = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xtb", type=Path, default=Path("/work/xtb-6.8.0/build/xtb"))
    parser.add_argument("--source", type=Path, default=Path("/work/xtb-6.8.0"))
    parser.add_argument("--work", type=Path, default=Path("/work/hexamer-xtb680"))
    parser.add_argument(
        "--csv",
        type=Path,
        default=paper / "data" / "hexamer-xtb680-repolish.csv",
    )
    args = parser.parse_args()
    require_bench(args.work)
    if not args.xtb.is_file():
        fail(f"missing xtb binary {args.xtb}")
    fetch_paper_data()
    logs = paper / "logs" / "water"
    numbers = logs / "NUMBERS.tsv"
    if not numbers.is_file():
        fail(f"missing {numbers}")
    old = load_old(numbers)
    tasks = []
    for seed in SEEDS:
        for arm in ARMS:
            raw = raw_final(logs, seed, arm)
            if not raw.is_file():
                fail(f"missing final {raw}")
            if (seed, arm) not in old:
                fail(f"NUMBERS.tsv has no {arm} seed {seed}")
            tasks.append((str(args.xtb), str(args.source), seed, arm, str(raw), str(args.work)))
    args.work.mkdir(parents=True, exist_ok=True)
    workers = min(8, len(tasks))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        polished = list(pool.map(polish_one, tasks))
    by_seed: dict[int, dict[str, dict[str, object]]] = {seed: {} for seed in SEEDS}
    for found in polished:
        by_seed[int(found["seed"])][str(found["arm"])] = found

    rows: list[list[object]] = []
    for seed in SEEDS:
        reference = choose_reference(by_seed[seed])
        if reference is None or reference["energy"] is None:
            fail(f"seed {seed} has no reference energy")
        reference_energy = float(reference["energy"])
        for arm in ARMS:
            found = by_seed[seed][arm]
            if not found["converged"] or not found["intact"] or found["energy"] is None:
                fail(f"{arm} seed {seed} did not converge to an intact cluster")
            new_energy = float(found["energy"])
            if arm == str(reference["arm"]):
                gap = 0.0
            else:
                gap = new_energy - reference_energy
            rows.append(
                [arm, seed, fmt(old[(seed, arm)]["energy"]), fmt(new_energy), fmt(gap)]
            )
    write_csv(args.csv, rows)


if __name__ == "__main__":
    sys.exit(main())
