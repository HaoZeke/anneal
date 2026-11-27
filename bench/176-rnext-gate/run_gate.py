#!/usr/bin/env python3
"""Round-2 three-arm runner for the 176 r-next gate.

Rows are append-only. A failed attempt is kept and marked superseded
when a replacement is started. Logs are one file per attempt.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_gate import summarize
from gate_io import (
    child_env,
    double_bits_hex,
    durable_write_text,
    launch_timed,
    load_rows,
    machine_id_sha256,
    parse_driver,
    parse_telemetry,
    scored_stdout,
    sha256_file,
    write_rows,
)
from gate_panel import (
    ARM_ROTATION,
    BASE_TREE,
    CELL_BY_NAME,
    DRIVERS,
    EXPECTED_B_TREE,
    HIT_TOL,
    PLACEMENT_PADS,
    PLACEMENT_SEED,
    REPLACEMENT_CAP,
    RUN_ORDER,
    SEEDS,
    SESSION_R2,
    pad_bytes_for,
)

def git_rev(worktree: Path, rev: str) -> str:
    out = subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", rev],
        text=True,
    )
    return out.strip()


class Arm:
    def __init__(self, name: str, worktree: Path, binary_dir: Path, expected_tree: str) -> None:
        self.name = name
        self.commit = git_rev(worktree, "HEAD")
        self.tree = git_rev(worktree, "HEAD^{tree}")
        if self.tree != expected_tree:
            raise RuntimeError(f"arm {name} tree mismatch")
        self.binaries: dict[str, Path] = {}
        self.hashes: dict[str, str] = {}
        for driver in DRIVERS:
            path = binary_dir / driver
            if not path.is_file():
                raise RuntimeError(f"missing binary for {name} {driver}")
            self.binaries[driver] = path
            self.hashes[driver] = sha256_file(path)


def arm_tag(arm: str) -> str:
    if arm == "A'":
        return "ap"
    if arm == "A":
        return "a"
    if arm == "B":
        return "b"
    return arm.lower()


def log_name(session: str, cell: str, arm: str, seed: int, slot: int, attempt: int) -> str:
    return f"{session}_{cell}_{arm_tag(arm)}_s{seed}_slot{slot}_att{attempt}.log"



def blank_row(spec, arm: Arm, seed: int, slot: int, attempt: int, session: str, core: str, mach: str) -> dict[str, str]:
    return {
        "cell": spec.cell,
        "arm": arm.name,
        "seed": str(seed),
        "slot": str(slot),
        "hit": "",
        "best": "",
        "best_bits": "",
        "charged": "",
        "evals_to_hit": "",
        "budget": str(spec.budget),
        "wall_s": "",
        "pot_s": "",
        "outside_s": "",
        "outside_per_charged_s": "",
        "core": core,
        "machine_id_sha256": mach,
        "arm_commit": arm.commit,
        "arm_tree": arm.tree,
        "binary_sha256": arm.hashes[spec.driver],
        "session_id": session,
        "status": "failed",
        "superseded": "",
        "attempt": str(attempt),
        "exit_code": "",
        "rss_anon_kb": "",
        "vm_hwm_kb": "",
        "ru_maxrss_kb": "",
        "basis_walks": "",
        "basis_evidence": "",
        "basis_claims": "",
        "basis_probes": "",
        "basis_candidates": "",
        "basis_spent": "",
        "pad_bytes": "",
        "log_sha256": "",
    }


def run_attempt(spec, arm: Arm, seed: int, slot: int, attempt: int, core: str, mach: str, session: str, env: dict[str, str], log_dir: Path, scratch_root: Path) -> dict[str, str]:
    row = blank_row(spec, arm, seed, slot, attempt, session, core, mach)
    binary = arm.binaries[spec.driver]
    scratch = scratch_root / f"{session}_{spec.cell}_{arm_tag(arm.name)}_s{seed}_att{attempt}"
    scratch.mkdir(parents=True, exist_ok=True)
    if spec.driver == "h2h_seed":
        stop = spec.reference + HIT_TOL
        prefix = scratch / "trace"
        cmd = [
            "taskset", "-c", core, str(binary), "tip4p", str(spec.n_molecules),
            "0", str(spec.budget), str(seed), "recommended", repr(stop), str(prefix),
        ]
        seeded = True
    else:
        cmd = ["taskset", "-c", core, str(binary), "run", spec.cell, str(seed)]
        seeded = False
    exit_code, stdout, ru_kb = launch_timed(cmd, env)
    log_body = scored_stdout(stdout)
    log_body += f"exit_code={exit_code} ru_maxrss_kb={ru_kb}\n"
    log_path = log_dir / log_name(session, spec.cell, arm.name, seed, slot, attempt)
    row["log_sha256"] = durable_write_text(log_path, log_body)
    row["exit_code"] = str(exit_code)
    row["ru_maxrss_kb"] = str(ru_kb)
    if exit_code != 0:
        return row
    try:
        parsed = parse_driver(stdout, seeded)
        telem = parse_telemetry(stdout)
    except RuntimeError:
        return row
    if int(parsed["budget"]) != spec.budget:
        return row
    if int(telem["pad_bytes"]) != pad_bytes_for(arm.name):
        return row
    best = float(parsed["best"])
    charged = int(parsed["charged"])
    wall_s = float(parsed["wall_s"])
    pot_s = float(parsed["pot_s"])
    outside_s = float(parsed["outside_s"]) if "outside_s" in parsed else max(0.0, wall_s - pot_s)
    hit = 1 if best <= spec.reference + HIT_TOL and best == best else 0
    row.update({
        "hit": str(hit),
        "best": f"{best:.10f}",
        "best_bits": double_bits_hex(best),
        "charged": str(charged),
        "evals_to_hit": str(int(parsed["evals_to_hit"])),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{outside_s:.6f}",
        "outside_per_charged_s": f"{outside_s / max(1, charged):.12e}",
        "rss_anon_kb": str(int(telem["rss_anon_kb"])),
        "vm_hwm_kb": str(int(telem["vm_hwm_kb"])),
        "basis_walks": str(int(telem["basis_walks"])),
        "basis_evidence": str(int(telem["basis_evidence"])),
        "basis_claims": str(int(telem["basis_claims"])),
        "basis_probes": str(int(telem["basis_probes"])),
        "basis_candidates": str(int(telem["basis_candidates"])),
        "basis_spent": str(int(telem["basis_spent"])),
        "pad_bytes": str(int(telem["pad_bytes"])),
        "status": "ok",
    })
    return row


def is_ok(row: dict[str, str], cell: str, arm: str, seed: int) -> bool:
    return (
        row.get("session_id") == SESSION_R2
        and row.get("cell") == cell
        and row.get("arm") == arm
        and int(row.get("seed") or -1) == seed
        and row.get("status") == "ok"
        and not row.get("superseded")
    )


def replacements_of(rows: list[dict[str, str]], cell: str) -> int:
    """Extra attempts that reported telemetry and were not kept."""
    return sum(
        1
        for row in rows
        if row.get("session_id") == SESSION_R2
        and row.get("cell") == cell
        and int(row.get("attempt") or "0") >= 2
        and row.get("rss_anon_kb")
        and row.get("status") != "ok"
    )


def next_attempt(rows, log_dir: Path, session: str, cell: str, arm: str, seed: int, slot: int) -> int:
    used = {
        int(row["attempt"])
        for row in rows
        if row.get("session_id") == SESSION_R2
        and row.get("cell") == cell
        and row.get("arm") == arm
        and int(row.get("seed") or -1) == seed
    }
    number = 1
    while number in used or (log_dir / log_name(session, cell, arm, seed, slot, number)).exists():
        number += 1
    return number


def persist(rows_path: Path, rows: list[dict[str, str]]) -> None:
    write_rows(rows_path, rows)


def run_slot(spec, arm: Arm, seed: int, slot: int, rows: list[dict[str, str]], rows_path: Path, core: str, mach: str, session: str, env, log_dir: Path, scratch_root: Path) -> None:
    if any(is_ok(row, spec.cell, arm.name, seed) for row in rows):
        return
    attempt = next_attempt(rows, log_dir, session, spec.cell, arm.name, seed, slot)
    while True:
        if attempt >= 2 and replacements_of(rows, spec.cell) >= REPLACEMENT_CAP:
            return
        if attempt >= 2:
            changed = False
            for prior in rows:
                if (
                    prior.get("session_id") == SESSION_R2
                    and prior.get("cell") == spec.cell
                    and prior.get("arm") == arm.name
                    and int(prior.get("seed") or -1) == seed
                    and prior.get("status") != "ok"
                    and not prior.get("superseded")
                ):
                    prior["superseded"] = "replaced"
                    changed = True
            if changed:
                persist(rows_path, rows)
        row = run_attempt(
            spec, arm, seed, slot, attempt, core, mach, session, env, log_dir, scratch_root
        )
        rows.append(row)
        persist(rows_path, rows)
        print(
            f"{spec.cell} arm={arm.name} seed={seed} slot={slot} att={attempt} "
            f"status={row['status']} hit={row['hit']} charged={row['charged']} "
            f"rss={row['rss_anon_kb']} walks={row['basis_walks']} evidence={row['basis_evidence']}",
            flush=True,
        )
        if row["status"] == "ok":
            return
        if replacements_of(rows, spec.cell) >= REPLACEMENT_CAP:
            return
        row["superseded"] = "replaced"
        persist(rows_path, rows)
        attempt = next_attempt(rows, log_dir, session, spec.cell, arm.name, seed, slot)


def warm(spec, arms: dict[str, Arm], env, core: str) -> None:
    for name in ("A", "A'", "B"):
        binary = arms[name].binaries[spec.driver]
        if spec.driver == "h2h_seed":
            cmd = ["taskset", "-c", core, str(binary)]
        else:
            cmd = ["taskset", "-c", core, str(binary), "budget", spec.cell]
        launch_timed(cmd, env)


def selected_names(arg: str) -> list[str]:
    if not arg.strip():
        return list(RUN_ORDER)
    wanted = {name.strip() for name in arg.split(",") if name.strip()}
    return [name for name in RUN_ORDER if name in wanted]


def load_arms(args) -> dict[str, Arm]:
    arms = {
        "A": Arm("A", Path(args.worktree_a), Path(args.target_a) / "release" / "examples", BASE_TREE),
        "A'": Arm("A'", Path(args.worktree_ap), Path(args.target_ap) / "release" / "examples", BASE_TREE),
        "B": Arm("B", Path(args.worktree_b), Path(args.target_b) / "release" / "examples", EXPECTED_B_TREE),
    }
    pad_root = Path(args.pad_root)
    for name, _size in PLACEMENT_PADS:
        arms[name] = Arm(name, Path(args.worktree_a), pad_root / name, BASE_TREE)
    return arms


def cmd_run(args) -> int:
    arms = load_arms(args)
    rows_path = Path(args.rows)
    log_dir = Path(args.logs)
    scratch_root = Path(args.scratch)
    log_dir.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(rows_path)
    if not rows_path.is_file():
        persist(rows_path, rows)
    env = child_env(args.ld_library_path)
    mach = machine_id_sha256()
    names = selected_names(args.cells)
    for name in names:
        spec = CELL_BY_NAME[name]
        warm(spec, arms, env, args.core)
        for seed in SEEDS:
            order = ARM_ROTATION[(seed - SEEDS[0]) % len(ARM_ROTATION)]
            for slot, arm_name in enumerate(order, start=1):
                run_slot(
                    spec, arms[arm_name], seed, slot, rows, rows_path,
                    args.core, mach, args.session_id, env, log_dir, scratch_root,
                )
        for arm_name, _size in PLACEMENT_PADS:
            run_slot(
                spec, arms[arm_name], PLACEMENT_SEED, 0, rows, rows_path,
                args.core, mach, args.session_id, env, log_dir, scratch_root,
            )
        done = sum(
            1 for row in rows
            if row.get("session_id") == args.session_id and row.get("cell") == name and row.get("status") == "ok"
        )
        print(f"cell {name} ok_rows={done}", flush=True)
    return 0


def cmd_summarize(args) -> int:
    label = summarize(Path(args.rows), Path(args.summary), Path(args.verdict))
    print(label, flush=True)
    return 0 if label == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--worktree-a", required=True)
    run_p.add_argument("--target-a", required=True)
    run_p.add_argument("--worktree-ap", required=True)
    run_p.add_argument("--target-ap", required=True)
    run_p.add_argument("--worktree-b", required=True)
    run_p.add_argument("--target-b", required=True)
    run_p.add_argument("--pad-root", required=True)
    run_p.add_argument("--rows", required=True)
    run_p.add_argument("--logs", required=True)
    run_p.add_argument("--scratch", required=True)
    run_p.add_argument("--core", default="0")
    run_p.add_argument("--session-id", default=SESSION_R2)
    run_p.add_argument("--ld-library-path", default="")
    run_p.add_argument("--cells", default="")
    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("--rows", required=True)
    sum_p.add_argument("--summary", required=True)
    sum_p.add_argument("--verdict", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.subcmd == "run":
        return cmd_run(args)
    if args.subcmd == "summarize":
        return cmd_summarize(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
