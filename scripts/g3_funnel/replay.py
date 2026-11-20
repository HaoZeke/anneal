#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Replay every G3 row, check it against the row, and keep its final best.

Usage: replay.py BIN_DIR ROWS.csv OUT_DIR MISSES_DIR [JOBS]

anneal rows rerun ``h2h_gmin_proof`` or ``h2h_morse`` with ``--trace`` and
``--incumbents``; minima hopping rows rerun ``run_mh.py --calls <charged>``.
Replays are deterministic and are not timed, so they may run in parallel.
Writes OUT_DIR/replay.csv with, per row, whether the replay's trace sha256
and printed best match the row, the best energy bits, and
``hop_first_best`` (``bench/g3-funnel/misses-rules.md``). For every miss the
final best goes to MISSES_DIR/<system>-<method>-<seed>.con via ``xyz_to_con``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SAME_STRUCTURE = 0.01
FIELDS = [
    "system",
    "method",
    "seed",
    "trace_match",
    "best_match",
    "best",
    "best_bits",
    "incumbents",
    "hop_first_best",
    "con",
]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session = load("g3_session", HERE / "run_session.py")


def frames(path: Path):
    lines = path.read_text().splitlines()
    at = 0
    while at < len(lines) and lines[at].strip():
        n = int(lines[at])
        meta = dict(field.split("=", 1) for field in lines[at + 1].split())
        xyz = np.array([[float(v) for v in line.split()[1:4]] for line in lines[at + 2 : at + 2 + n]])
        yield meta, xyz
        at += 2 + n


def distances(xyz: np.ndarray) -> np.ndarray:
    diff = xyz[:, None, :] - xyz[None, :, :]
    upper = np.triu_indices(len(xyz), 1)
    return np.sort(np.sqrt((diff**2).sum(-1))[upper])


def same_structure(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and float(np.max(np.abs(a - b))) <= SAME_STRUCTURE


def replay(bin_dir: Path, out: Path, misses: Path, row: dict) -> dict:
    system, method, seed = row["system"], row["method"], int(row["seed"])
    stem = f"{system}-{method}-{seed}"
    trace = out / "traces" / f"{stem}.xyz"
    incumbents = out / "incumbents" / f"{stem}.xyz"
    env = session.environment(out)
    if method == "anneal":
        binary = "h2h_morse" if system.startswith("m") else "h2h_gmin_proof"
        command = [str(bin_dir / binary), "run", system, str(seed), "--trace", str(trace), "--incumbents", str(incumbents)]
    else:
        command = [
            sys.executable,
            str(HERE / "run_mh.py"),
            system,
            str(seed),
            row["cap_s"],
            str(out / "work" / stem),
            str(trace),
            "--calls",
            row["charged"],
            "--incumbents",
            str(incumbents),
        ]
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    (out / "logs" / f"{stem}.log").write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    shutil.rmtree(out / "work" / stem, ignore_errors=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{stem}: exit {proc.returncode}")
    if method == "anneal":
        best = session.gmin_panel.anneal_result(
            proc.stdout, system, seed, session.gmin_panel.PAPER_BUDGETS[system]
        )["best"]
    else:
        line = next(line for line in proc.stdout.splitlines() if line.startswith("g3_row "))
        best = dict(part.split("=", 1) for part in line.split()[1:])["best"]
    sequence = list(frames(incumbents))
    if not sequence:
        raise RuntimeError(f"{stem}: no incumbent")
    final_meta, final_xyz = sequence[-1]
    final = distances(final_xyz)
    first = next(meta for meta, xyz in sequence if same_structure(distances(xyz), final))
    con = ""
    if row["hit"] != "1":
        target = misses / f"{stem}.con"
        subprocess.run([str(bin_dir / "xyz_to_con"), str(incumbents), str(target)], check=True)
        con = target.name
    return {
        "system": system,
        "method": method,
        "seed": seed,
        "trace_match": int(hashlib.sha256(trace.read_bytes()).hexdigest() == row["trace_sha256"]),
        "best_match": int(best == row["best"]),
        "best": best,
        "best_bits": final_meta["energy_bits"],
        "incumbents": len(sequence),
        "hop_first_best": first["hop"],
        "con": con,
    }


def guarded(bin_dir: Path, out: Path, misses: Path, row: dict) -> dict:
    try:
        return replay(bin_dir, out, misses, row)
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", flush=True)
        empty = dict.fromkeys(FIELDS, "")
        empty.update(system=row["system"], method=row["method"], seed=row["seed"], trace_match=0, best_match=0)
        return empty


def main() -> int:
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    bin_dir, rows_path, out, misses = (Path(v).resolve() for v in sys.argv[1:5])
    jobs = int(sys.argv[5]) if len(sys.argv) == 6 else 1
    for sub in ("logs", "traces", "incumbents", "work"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    misses.mkdir(parents=True, exist_ok=True)
    with rows_path.open() as handle:
        rows = list(csv.DictReader(handle))
    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(lambda row: guarded(bin_dir, out, misses, row), rows))
    with (out / "replay.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print("REPLAY-DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
