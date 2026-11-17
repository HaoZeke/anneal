# Copyright (c) anneal developers
"""LJ98 GMIN at the 64-seed campaign ceiling, on the published symmetrise block.

The campaign in paper/jctc/data/lj55_lj98/seeds.csv records 64 hits on
seeds 0-63 at a ceiling of 1_000_000 calls. The coordinate files of that
campaign are not in this tree, so this check cannot replay those starts.
It uses the same ``coords`` writer as the panel, the published OptBench
LJ98 gmin-symmetrise block, and the same call ceiling. A hit is a
quenched energy at or below the reference plus 0.001.

Usage, under hostname bench and taskset -c 0:

    python3 lj98_check.py --bin /work/gmin-panel/bin --rows bench/gmin-panel/lj98-campaign-check.csv --svnrev aac1548a8aec13b53ad12a8d6904625081685d2e
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inputs import body  # noqa: E402

SEEDS = tuple(range(64))
BUDGET = 1_000_000
REFERENCE = -543.665361
TOL = 0.001
FIELDS = [
    "system", "code", "driver", "seed", "budget", "hit", "evals_to_hit", "charged",
    "wall_s", "best", "pot_s", "outside_s", "code_revision", "host", "cpu", "utc",
    "start",
]
DRIVER = "GMIN fair lj98-campaign"


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip().replace(",", " ")
    return ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/work/gmin-panel/lj98-check"))
    parser.add_argument("--svnrev", required=True)
    args = parser.parse_args(argv[1:])
    if socket.gethostname() != "bench":
        print(f"uts hostname is {socket.gethostname()}, expected bench", file=sys.stderr)
        return 2
    if os.sched_getaffinity(0) != {0}:
        print(f"pinned to {sorted(os.sched_getaffinity(0))}, expected core 0", file=sys.stderr)
        return 2
    done = set()
    if args.rows.exists():
        with args.rows.open(newline="") as handle:
            done = {int(row["seed"]) for row in csv.DictReader(handle) if row["driver"] == DRIVER}
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    if not args.rows.exists() or args.rows.stat().st_size == 0:
        with args.rows.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n").writeheader()
    host = socket.gethostname()
    cpu = cpu_model()
    for seed in SEEDS:
        if seed in done:
            print(f"skip {seed}", flush=True)
            continue
        work = args.work / str(seed)
        work.mkdir(parents=True, exist_ok=True)
        for stale in ("row.txt", "coords", "data"):
            (work / stale).unlink(missing_ok=True)
        subprocess.run(
            [str(args.bin / "h2h_gmin_proof"), "coords", "lj98", str(seed), str(work / "coords")],
            check=True,
            capture_output=True,
            text=True,
        )
        (work / "data").write_text(body("lj98") + f"RANSEED {seed}\n")
        env = os.environ.copy()
        env.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            H2H_BUDGET=str(BUDGET),
            H2H_SYSTEM="lj98",
            H2H_SEED=str(seed),
            H2H_TARGET=f"{REFERENCE:.10f}",
            H2H_TOL=str(TOL),
            H2H_ROW=str(work / "row.txt"),
        )
        proc = subprocess.run(
            ["taskset", "-c", "0", str(args.bin / "GMIN")],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        (work / "stdout.txt").write_text(proc.stdout)
        (work / "stderr.txt").write_text(proc.stderr)
        if proc.returncode != 0 or not (work / "row.txt").exists():
            raise SystemExit(f"seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
        fields = dict(part.split("=", 1) for part in (work / "row.txt").read_text().split() if "=" in part)
        wall = float(fields["wall_s"])
        pot = float(fields["pot_s"])
        row = {
            "system": "lj98",
            "code": "gmin",
            "driver": DRIVER,
            "seed": str(seed),
            "budget": str(BUDGET),
            "hit": fields["hit"],
            "evals_to_hit": fields["evals_to_hit"],
            "charged": fields["charged"],
            "wall_s": f"{wall:.6f}",
            "best": fields["best"],
            "pot_s": f"{pot:.6f}",
            "outside_s": f"{max(0.0, wall - pot):.6f}",
            "code_revision": args.svnrev,
            "host": host,
            "cpu": cpu,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "start": "panel coords writer; the campaign coordinate files are not in this tree",
        }
        with args.rows.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n").writerow(row)
        print(f"seed {seed} hit={row['hit']} calls={row['evals_to_hit']} wall={row['wall_s']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
