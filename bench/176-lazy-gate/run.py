#!/usr/bin/env python3
"""Run the lazy-basin panel and append one row per launch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_io import (
    child_env,
    digest_bit_hex,
    durable_write_text,
    launch_timed,
    load_rows,
    machine_id_sha256,
    parse_driver,
    parse_final,
    parse_telemetry,
    scored_stdout,
    sha256_file,
    write_rows,
)
from panel import (
    ARM_ROTATION,
    B0_TREE,
    B_TREE,
    BASE_TREE,
    CELLS,
    DRIVERS,
    HIT_TOL,
    IDENTITY_ARM,
    PANEL_ARMS,
    SESSION,
    TIMED_PAD_BYTES,
)


def git_rev(worktree: Path, rev: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", rev], text=True
    ).strip()


class Arm:
    def __init__(self, name: str, worktree: Path, target: Path, expected_tree: str) -> None:
        self.name = name
        self.commit = git_rev(worktree, "HEAD")
        self.tree = git_rev(worktree, "HEAD^{tree}")
        if self.tree != expected_tree:
            raise RuntimeError(f"arm {name} tree mismatch")
        self.binaries = {
            driver: target / "release" / "examples" / driver for driver in DRIVERS
        }
        for driver, path in self.binaries.items():
            if not path.is_file():
                raise RuntimeError(f"missing binary for {name} {driver}")
        self.hashes = {driver: sha256_file(path) for driver, path in self.binaries.items()}


def blank_row(spec, arm: Arm, seed: int, slot: int, core: str, machine: str) -> dict[str, str]:
    return {
        "cell": spec.cell,
        "arm": arm.name,
        "seed": str(seed),
        "slot": str(slot),
        "budget": str(spec.budget),
        "core": core,
        "machine_id_sha256": machine,
        "arm_commit": arm.commit,
        "arm_tree": arm.tree,
        "binary_sha256": arm.hashes[spec.driver],
        "session_id": SESSION,
        "status": "failed",
        "attempt": "1",
    }


def log_name(cell: str, arm: str, seed: int, slot: int) -> str:
    return f"{SESSION}_{cell}_{arm.lower()}_s{seed}_slot{slot}.log"


def run_one(spec, arm, seed, slot, core, machine, env, logs, scratch) -> dict[str, str]:
    row = blank_row(spec, arm, seed, slot, core, machine)
    binary = arm.binaries[spec.driver]
    run_scratch = scratch / f"{spec.cell}_{arm.name.lower()}_s{seed}"
    run_scratch.mkdir(parents=True, exist_ok=True)
    if spec.driver == "h2h_seed":
        stop = spec.reference + HIT_TOL
        cmd = [
            "taskset", "-c", core, str(binary), "tip4p",
            str(spec.n_molecules), "0", str(spec.budget), str(seed),
            "recommended", repr(stop), str(run_scratch / "trace"),
        ]
        seeded = True
    else:
        cmd = ["taskset", "-c", core, str(binary), "run", spec.cell, str(seed)]
        seeded = False
    exit_code, stdout, ru_kb = launch_timed(cmd, env)
    body = scored_stdout(stdout)
    body += f"exit_code={exit_code} ru_maxrss_kb={ru_kb}\n"
    path = logs / log_name(spec.cell, arm.name, seed, slot)
    row["log_sha256"] = durable_write_text(path, body)
    row["exit_code"] = str(exit_code)
    row["ru_maxrss_kb"] = str(ru_kb)
    if exit_code != 0:
        return row
    try:
        parsed = parse_driver(stdout, seeded)
        telemetry = parse_telemetry(stdout)
        final = parse_final(stdout)
    except RuntimeError:
        return row
    if int(parsed["budget"]) != spec.budget:
        return row
    if int(telemetry["pad_bytes"]) != TIMED_PAD_BYTES:
        return row
    best = float(parsed["best"])
    charged = int(parsed["charged"])
    wall_s = float(parsed["wall_s"])
    pot_s = float(parsed["pot_s"])
    outside_s = float(parsed.get("outside_s", max(0.0, wall_s - pot_s)))
    hit = int(best == best and best <= spec.reference + HIT_TOL)
    row.update({
        "hit": str(hit),
        "best": f"{best:.10f}",
        "best_bits": "0x" + final["energy_bits"],
        "charged": str(charged),
        "evals_to_hit": str(int(parsed["evals_to_hit"])),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{outside_s:.6f}",
        "outside_per_charged_s": f"{outside_s / max(1, charged):.12e}",
        "rss_anon_kb": str(int(telemetry["rss_anon_kb"])),
        "vm_hwm_kb": str(int(telemetry["vm_hwm_kb"])),
        "basis_walks": str(int(telemetry["basis_walks"])),
        "basis_evidence": str(int(telemetry["basis_evidence"])),
        "basis_claims": str(int(telemetry["basis_claims"])),
        "basis_probes": str(int(telemetry["basis_probes"])),
        "basis_candidates": str(int(telemetry["basis_candidates"])),
        "basis_spent": str(int(telemetry["basis_spent"])),
        "pad_bytes": str(int(telemetry["pad_bytes"])),
        "struct_sha256": digest_bit_hex(final["struct_bits"]),
        "final_struct_sha256": digest_bit_hex(final["final_struct_bits"]),
        "final_energy_bits": "0x" + final["final_energy_bits"],
        "status": "ok",
    })
    return row


def warm(spec, arms, env, core: str) -> None:
    for name in (*PANEL_ARMS, IDENTITY_ARM):
        binary = arms[name].binaries[spec.driver]
        if spec.driver == "h2h_seed":
            cmd = ["taskset", "-c", core, str(binary)]
        else:
            cmd = ["taskset", "-c", core, str(binary), "budget", spec.cell]
        launch_timed(cmd, env)


def launch_row(spec, arm, seed, slot, core, machine, env, logs, scratch, rows, rows_path, seen):
    key = (spec.cell, arm.name, str(seed))
    if key in seen:
        return
    row = run_one(spec, arm, seed, slot, core, machine, env, logs, scratch)
    rows.append(row)
    write_rows(rows_path, rows)
    seen.add(key)
    print(
        f"{spec.cell} arm={arm.name} seed={seed} status={row['status']} "
        f"hit={row.get('hit', '')} wall={row.get('wall_s', '')} "
        f"charged={row.get('charged', '')}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--ld-library-path", default="")
    args = parser.parse_args()

    stage = Path(args.stage)
    arms = {
        "A": Arm("A", stage / "wt-a", stage / "target-a", BASE_TREE),
        "A2": Arm("A2", stage / "wt-a", stage / "target-a2", BASE_TREE),
        "B": Arm("B", stage / "wt-b", stage / "target-b", B_TREE),
        "B0": Arm("B0", stage / "wt-b0", stage / "target-b0", B0_TREE),
    }
    rows_path = Path(args.rows)
    logs = Path(args.logs)
    scratch = Path(args.scratch)
    logs.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    rows = load_rows(rows_path)
    if not rows_path.is_file():
        write_rows(rows_path, rows)
    env = child_env(args.ld_library_path)
    machine = machine_id_sha256()
    seen = {
        (row.get("cell"), row.get("arm"), row.get("seed"))
        for row in rows
        if row.get("session_id") == SESSION
    }
    for spec in CELLS:
        warm(spec, arms, env, args.core)
        for seed in spec.seeds:
            order = ARM_ROTATION[(seed - spec.seeds[0]) % len(ARM_ROTATION)]
            for slot, arm_name in enumerate(order, 1):
                launch_row(
                    spec, arms[arm_name], seed, slot, args.core, machine,
                    env, logs, scratch, rows, rows_path, seen,
                )
        for seed in spec.seeds:
            launch_row(
                spec, arms[IDENTITY_ARM], seed, 4, args.core, machine,
                env, logs, scratch, rows, rows_path, seen,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
