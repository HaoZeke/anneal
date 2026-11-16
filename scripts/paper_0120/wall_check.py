# Copyright (c) anneal developers
"""Same-session wall check: two h2h_gmin_proof builds, seeds interleaved.

Used where the release rows repeat a develop revision's calls seed for
seed, so a change in wall is the build or the session and not the walk.
For each system and seed the base build runs, then the release build,
``taskset -c 0``, one thread. Rows are appended with the code revision
and the host. Refuses to run unless the UTS hostname is bench.

    wall_check.py --rows wall-check.csv \\
        --build REV=path/to/h2h_gmin_proof --build REV=... \\
        --tree REV=path/to/worktree --tree REV=... \\
        --run lj75:2000-2005 --run lj98:2000-2002
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
from pathlib import Path

FIELDS = ["system", "seed", "order", "hit", "evals_to_hit", "charged", "wall_s", "pot_s",
          "outside_s", "code_revision", "host"]


def pairs(values: list[str]) -> list[tuple[str, str]]:
    return [tuple(value.split("=", 1)) for value in values]


def seeds_of(text: str) -> list[int]:
    start, _, stop = text.partition("-")
    return list(range(int(start), int(stop or start) + 1))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--build", action="append", required=True)
    parser.add_argument("--tree", action="append", required=True)
    parser.add_argument("--run", action="append", required=True)
    args = parser.parse_args(argv[1:])
    if socket.gethostname() != "bench":
        print(f"uts hostname is {socket.gethostname()}, expected bench", file=sys.stderr)
        return 2
    builds = pairs(args.build)
    trees = dict(pairs(args.tree))
    env = {k: v for k, v in os.environ.items() if not k.startswith("H2H_")}
    env.update(OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    fresh = not args.rows.is_file() or args.rows.stat().st_size == 0
    with args.rows.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        for spec in args.run:
            system, _, seeds = spec.partition(":")
            for seed in seeds_of(seeds):
                for order, (revision, binary) in enumerate(builds):
                    proc = subprocess.run(
                        ["taskset", "-c", "0", binary, "run", system, str(seed)],
                        cwd=trees[revision], env=env, capture_output=True, text=True, check=True,
                    )
                    lines = proc.stdout.strip().splitlines()
                    row = dict(zip(lines[-2].split(","), lines[-1].split(",")))
                    wall, pot = float(row["wall_s"]), float(row["pot_s"])
                    out = {
                        "system": system,
                        "seed": str(seed),
                        "order": str(order),
                        "hit": "1" if row["hit"] in ("1", "true") else "0",
                        "evals_to_hit": row["evals_to_hit"],
                        "charged": row["charged"],
                        "wall_s": f"{wall:.6f}",
                        "pot_s": f"{pot:.6f}",
                        "outside_s": f"{wall - pot:.6f}",
                        "code_revision": revision,
                        "host": socket.gethostname(),
                    }
                    writer.writerow(out)
                    handle.flush()
                    print(" ".join(f"{k}={v}" for k, v in out.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
