#!/usr/bin/env python3
"""Build and run the matched-budget ammonia-octamer anneal arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

BUDGET = 332_675
MISS_CALLS = BUDGET + 1
REFERENCE_EH = -35.470769324714
TOLERANCE_EH = 1.0e-6
AUTOEV = 27.21138505
TARGET_EV = (REFERENCE_EH + TOLERANCE_EH) * AUTOEV
SEEDS = tuple(range(2000, 2012))
FEATURES = "rgpot-ex"
EXAMPLE = "gfn2_mol_h2h_matched"
FIELDS = (
    "seed",
    "hit",
    "calls_to_hit",
    "stop_reason",
    "best_energy_Eh",
    "wall_s",
    "binary",
    "features",
    "revision",
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", root, *args], text=True).strip()


def field(path: Path, name: str) -> str | None:
    prefix = name + ":"
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def generate_example(root: Path) -> Path:
    source = (root / "examples/gfn2_mol_h2h.rs").read_text()
    old = (
        "    let cfg = Config::recommended_molecular("
        "built.species.clone(), built.groups.clone(), 1.0);"
    )
    new = (
        "    let mut cfg = Config::recommended_molecular("
        "built.species.clone(), built.groups.clone(), 1.0);\n"
        f"    cfg.target_energy = Some({TARGET_EV:.15});"
    )
    if source.count(old) != 1:
        raise RuntimeError("recommended-molecular construction changed")
    generated = source.replace(old, new)
    path = root / "examples" / f"{EXAMPLE}.rs"
    path.write_text(generated)
    return path


def build(root: Path, target: Path) -> Path:
    generated = generate_example(root)
    try:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--features",
                FEATURES,
                "--example",
                EXAMPLE,
            ],
            cwd=root,
            env={**os.environ, "CARGO_TARGET_DIR": str(target)},
            check=True,
        )
    finally:
        generated.unlink(missing_ok=True)
    return target / "release" / "examples" / EXAMPLE


def binary_identity(binary: Path) -> str:
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    return f"{EXAMPLE}@sha256:{digest}"


def load_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def write_rows(path: Path, rows: dict[int, dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[seed] for seed in sorted(rows))
    temporary.replace(path)


def run_seed(
    root: Path,
    binary: Path,
    identity: str,
    revision: str,
    tree: str,
    seed: int,
    runs: Path,
) -> dict[str, str]:
    rundir = runs / str(seed)
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)
    env = {
        **os.environ,
        "ANNEAL_SOURCE_COMMIT": revision,
        "ANNEAL_SOURCE_TREE": tree,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
    }
    with (rundir / "stdout.txt").open("w") as stdout, (
        rundir / "stderr.txt"
    ).open("w") as stderr:
        completed = subprocess.run(
            [
                "taskset",
                "-c",
                "0",
                str(binary),
                "nh3-8",
                str(BUDGET),
                str(seed),
                str(rundir),
            ],
            cwd=root,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    meta = rundir / "run.txt"
    if not meta.is_file():
        raise RuntimeError(f"seed {seed} produced no run metadata (exit {completed.returncode})")
    calls = int(field(meta, "energy_calls") or "0")
    wall = float(field(meta, "wall_s") or "nan")
    best_ev = float(field(meta, "objective_best_eV") or "nan")
    best_eh = best_ev / AUTOEV
    hit = best_ev <= TARGET_EV
    if hit:
        stop = "target"
        calls_to_hit = calls
    elif calls >= BUDGET:
        stop = "budget"
        calls_to_hit = MISS_CALLS
    elif field(meta, "search_error"):
        stop = "error"
        calls_to_hit = MISS_CALLS
    else:
        stop = f"exit_{completed.returncode}"
        calls_to_hit = MISS_CALLS
    return {
        "seed": str(seed),
        "hit": "yes" if hit else "no",
        "calls_to_hit": str(calls_to_hit),
        "stop_reason": stop,
        "best_energy_Eh": f"{best_eh:.12f}",
        "wall_s": f"{wall:.6f}",
        "binary": identity,
        "features": FEATURES,
        "revision": revision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    engine = os.environ.get("RGPOT_XTB_ENGINE")
    if not engine or not Path(engine).is_file():
        raise SystemExit("RGPOT_XTB_ENGINE must name the pinned engine library")
    revision = git(root, "rev-parse", "HEAD")
    tree = git(root, "rev-parse", "HEAD^{tree}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve() if args.binary else build(root, args.target_dir.resolve())
    identity = binary_identity(binary)
    rows_path = out / "results.csv"
    rows = load_rows(rows_path)
    runs = out / "runs"
    runs.mkdir(exist_ok=True)
    for seed in args.seeds:
        if seed in rows:
            print(f"seed {seed}: already complete", flush=True)
            continue
        print(f"seed {seed}: start", flush=True)
        rows[seed] = run_seed(root, binary, identity, revision, tree, seed, runs)
        write_rows(rows_path, rows)
        print(
            f"seed {seed}: {rows[seed]['stop_reason']} "
            f"at {rows[seed]['calls_to_hit']} calls",
            flush=True,
        )
    if set(rows) != set(SEEDS):
        raise SystemExit("campaign is incomplete")


if __name__ == "__main__":
    main()
