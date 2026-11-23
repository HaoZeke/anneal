#!/usr/bin/env python3
"""Run the 24-seed focused TIP4P10 timing gate."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / "160-rnext6" / "run_gate.py"
BASE_SPEC = importlib.util.spec_from_file_location("rnext6_gate_base", BASE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("unable to load the prior gate harness")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)

CELL = "tip4p10"
SEEDS = tuple(range(2000, 2024))
ORDER = ("A", "B", "A2", "B")
ROW_FIELDS = (
    "cell",
    "arm",
    "tree",
    "seed",
    "position",
    "arm_repeat",
    "slot",
    "exit_code",
    "energy",
    "energy_bits",
    "hit",
    "calls",
    "charged",
    "outside_s",
    "per_call_s",
    "rss_anon_end_kb",
    "exec_image",
    "affinity",
    *base.DIAG_FIELDS,
    "budget",
    "reference",
    "binary_sha256",
)


def append_row(path: Path, row: dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ROW_FIELDS,
            lineterminator="\n",
        )
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_session(path: Path, event: dict[str, object]) -> None:
    with path.open("a", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--rss-probe", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    if os.sched_getaffinity(0) != {0}:
        os.sched_setaffinity(0, {0})
    arms = base.parse_arms(args.arm)
    output = args.out.resolve()
    rows_path = output / "rows-tip4p10-n24.csv"
    session_path = output / "sessions-tip4p10-n24.jsonl"
    if rows_path.exists() or session_path.exists():
        raise RuntimeError("focused output already exists")
    work = args.work.resolve()
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    probe = args.rss_probe.resolve()
    if not probe.is_file():
        raise RuntimeError("RSS probe missing")
    environment = base.child_env(probe)
    reference = base.references()[CELL]
    append_session(session_path, {"event": "begin", "rows": 96})
    for arm in arms.values():
        append_session(
            session_path,
            {
                "arm": arm.name,
                "binary_sha256": arm.binary_sha,
                "event": "binary",
                "tree": base.ARM_TREES[arm.name],
            },
        )
    slot = 0
    for seed in SEEDS:
        repeats = {"A": 0, "A2": 0, "B": 0}
        append_session(
            session_path,
            {"cell": CELL, "event": "seed_begin", "seed": seed},
        )
        for position, arm_name in enumerate(ORDER, 1):
            repeats[arm_name] += 1
            arm = arms[arm_name]
            slot += 1
            tag = (
                f"{CELL}-s{seed}-p{position}-"
                f"{arm_name.lower()}-{repeats[arm_name]}"
            )
            prefix = logs / tag
            stdout_path = Path(f"{prefix}.out")
            stderr_path = Path(f"{prefix}.err")
            command = base.command(arm, CELL, seed, reference, prefix)
            rc, stdout, stderr, rss, image, affinity = base.run_measured(
                command,
                logs,
                environment,
                stdout_path,
                stderr_path,
            )
            Path(f"{prefix}.rc").write_text(f"{rc}\n")
            if rc != 0:
                raise RuntimeError(f"{tag} exited {rc}")
            if image != "h2h_seed" or affinity != base.CORE:
                raise RuntimeError(f"{tag} exec verification failed")
            parsed = base.parse_output(stdout)
            if int(parsed["budget"]) != base.BUDGET:
                raise RuntimeError(f"{tag} budget mismatch")
            diagnostics = base.parse_diag(stderr)
            expected = 1 if arm_name == "B" else 0
            if len(diagnostics) != expected:
                raise RuntimeError(f"{tag} rigid diagnostics mismatch")
            energy = float(parsed["best"])
            hit = energy <= reference + base.HIT_TOL
            calls = int(parsed["calls"]) if hit else base.BUDGET
            charged = int(parsed["charged"])
            outside = float(parsed["outside"])
            row: dict[str, object] = {
                "cell": CELL,
                "arm": arm_name,
                "tree": base.ARM_TREES[arm_name],
                "seed": seed,
                "position": position,
                "arm_repeat": repeats[arm_name],
                "slot": slot,
                "exit_code": rc,
                "energy": parsed["best"],
                "energy_bits": struct.pack(">d", energy).hex(),
                "hit": int(hit),
                "calls": calls,
                "charged": charged,
                "outside_s": parsed["outside"],
                "per_call_s": f"{outside / charged:.12g}",
                "rss_anon_end_kb": rss,
                "exec_image": image,
                "affinity": affinity,
                "budget": base.BUDGET,
                "reference": repr(reference),
                "binary_sha256": arm.binary_sha,
            }
            row.update({field: "" for field in base.DIAG_FIELDS})
            if diagnostics:
                row.update(diagnostics[0])
            append_row(rows_path, row)
        append_session(
            session_path,
            {"cell": CELL, "event": "seed_end", "seed": seed},
        )
    for arm in arms.values():
        arm.verify()
    append_session(session_path, {"event": "complete", "rows": slot})


if __name__ == "__main__":
    main(sys.argv)
