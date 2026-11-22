#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Check the r-next4 TIP4P17 claim on seeds 9200--9211."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import run_gate as gate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = tuple(range(9200, 9212))
BUDGET = 2_000_000
TOL = 1.0e-3
CORE = "0"
TREE = "81fa4c3de8722fd35eab7311c7e365c2294e5fb6"
FIELDS = (
    "seed", "tree", "exit_code", "best", "hit", "calls", "charged",
    "hops", "rss_anon_end_kb", *gate.DIAG_FIELDS, "binary_sha256",
)


def reference() -> float:
    with (ROOT / "data" / "tip4p_ccd.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["n"]) == 17:
                return float(row["energy"])
    raise SystemExit("TIP4P17 reference missing")


def append(path: Path, row: dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--rss-probe", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    binary = args.binary.resolve()
    probe = args.rss_probe.resolve()
    rows = args.out.resolve() / "tip4p17-claim-rows.csv"
    if rows.exists():
        raise SystemExit("claim rows exist: one session only")
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    ref = reference()
    for seed in SEEDS:
        prefix = work / f"tip4p17-s{seed}-b"
        command = [
            "taskset", "-c", CORE, str(binary), "tip4p", "17", "0",
            str(BUDGET), str(seed), "recommended", repr(ref + TOL),
            str(prefix),
        ]
        result = gate.run_measured(
            command,
            work,
            gate.child_env(probe),
            Path(f"{prefix}.out"),
            Path(f"{prefix}.err"),
        )
        rc, stdout, stderr, affinity, rss, _, _, image = result
        if rc != 0 or affinity != CORE or image != "h2h_seed":
            raise SystemExit(f"invalid claim run for seed {seed}")
        diags = gate.rigid_diag(stderr)
        if len(diags) != 1:
            raise SystemExit(f"missing rigid diagnostics for seed {seed}")
        parsed = gate.parse("tip4p17", stdout)
        best = float(parsed["best"])
        hops = re.search(r"^hops: (\d+)$", stdout, re.M)
        row = {
            "seed": seed,
            "tree": TREE,
            "exit_code": rc,
            "best": parsed["best"],
            "hit": int(best <= ref + TOL),
            "calls": (
                parsed["evals"] if best <= ref + TOL else BUDGET
            ),
            "charged": parsed["charged"],
            "hops": hops.group(1) if hops else "",
            "rss_anon_end_kb": rss,
            "binary_sha256": gate.sha256_file(binary),
        }
        row.update(diags[0])
        append(rows, row)


if __name__ == "__main__":
    main()
