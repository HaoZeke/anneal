#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial three-arm timing recheck of outside-potential time per call.

Runs blj115n38, blj105n38, and sc126-55 at the part B gate budgets and
references, seeds 2100-2123. Arms A, B, and A2 run back to back for each
seed on CPU 0. No quick mode, no warm-up search, and no discarded search.
The per-job timeout is the part B gate timeout. The pass rule lives in
stats.py and is not read from the command line.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE_DIR = HERE.parent / "nm3-land-gate-b"
for _path in (str(HERE), str(GATE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_gate  # noqa: E402
import stats  # noqa: E402

ARM_A_COMMIT = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
ARM_A_TREE = "7a2ba49b37241a8fe42b2c45c226374de6d92274"
ARM_B_TREE = "2ba43832b540c6e9005cfb6be31243b36ab086a7"
ARM_A_LIBRARY = f"commit:{ARM_A_COMMIT}"
ARM_B_LIBRARY = f"tree:{ARM_B_TREE}"

DRIVER_KIND = {
    "blj115n38": "blj",
    "blj105n38": "blj",
    "sc126-55": "sc",
}

COMMITTED = (
    "bench/nm3-land-gate-b-timing/run_timing.py",
    "bench/nm3-land-gate-b-timing/stats.py",
    "bench/nm3-land-gate-b-timing/summarize.py",
    "bench/nm3-land-gate-b/driver/Cargo.toml",
    "bench/nm3-land-gate-b/driver/src/main.rs",
    "bench/nm3-land-gate-b/panels.csv",
)

# Pools named by the part B gate, plus the other common single-process limits.
EXTRA_THREAD_VARS = (
    "NUMEXPR_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "JULIA_NUM_THREADS",
    "TBB_NUM_THREADS",
)


def head_blob(repo: Path, relpath: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"HEAD:{relpath}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{relpath} is not in HEAD")
    work = subprocess.run(
        ["git", "-C", str(repo), "hash-object", str(repo / relpath)],
        capture_output=True,
        text=True,
        check=True,
    )
    if proc.stdout.strip() != work.stdout.strip():
        raise SystemExit(f"{relpath} differs from HEAD")
    return proc.stdout.strip()


def tree_sha(path: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"cannot read tree of {path.name}")
    return proc.stdout.strip()


def content_blob(path: Path) -> str:
    proc = subprocess.run(
        ["git", "hash-object", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def assert_committed(repo: Path) -> None:
    for relpath in COMMITTED:
        head_blob(repo, relpath)


def harness_blob_text(repo: Path, arm: run_gate.ArmSpec, driver_kind: str) -> str:
    runner = content_blob(HERE / "run_timing.py")
    rule = content_blob(HERE / "stats.py")
    if driver_kind == "sc":
        source = run_gate.arm_blob_id(arm, "examples/h2h_sc.rs")
        return (
            f"examples/h2h_sc.rs:{source};"
            f"bench/nm3-land-gate-b-timing/run_timing.py:{runner};"
            f"bench/nm3-land-gate-b-timing/stats.py:{rule}"
        )
    cargo_toml = content_blob(repo / "bench/nm3-land-gate-b/driver/Cargo.toml")
    driver_rs = content_blob(repo / "bench/nm3-land-gate-b/driver/src/main.rs")
    return (
        f"bench/nm3-land-gate-b/driver/Cargo.toml:{cargo_toml};"
        f"bench/nm3-land-gate-b/driver/src/main.rs:{driver_rs};"
        f"bench/nm3-land-gate-b-timing/run_timing.py:{runner};"
        f"bench/nm3-land-gate-b-timing/stats.py:{rule}"
    )


def load_panels(path: Path) -> dict[str, dict[str, str]]:
    rows = run_gate.read_csv(path)
    found = {row["system"]: row for row in rows}
    missing = [cell for cell in stats.CELLS if cell not in found]
    if missing:
        raise SystemExit(f"panels file is missing {missing}")
    return found


def check_budgets(
    arms: dict[str, run_gate.ArmSpec],
    panels: dict[str, dict[str, str]],
    env: dict[str, str],
) -> dict[str, tuple[int, float]]:
    checked: dict[str, tuple[int, float]] = {}
    for cell in stats.CELLS:
        binary = run_gate.DRIVERS[DRIVER_KIND[cell]]
        seen: set[tuple[int, str]] = set()
        value: tuple[int, float] | None = None
        for arm in arms.values():
            budget, reference = run_gate.query_budget(arm.bin_dir / binary, cell, env)
            seen.add((budget, f"{reference:.10f}"))
            value = (budget, reference)
        if len(seen) != 1 or value is None:
            raise SystemExit(f"{cell}: arms disagree on budget/reference: {seen}")
        budget, reference = value
        panel = panels[cell]
        if budget != int(panel["budget"]) or f"{reference:.10f}" != f"{float(panel['reference']):.10f}":
            raise SystemExit(
                f"{cell}: budget {budget} reference {reference:.10f} "
                f"!= panel {panel['budget']} {panel['reference']}"
            )
        checked[cell] = (budget, reference)
    return checked


def finish_row(
    base: dict[str, str],
    status: str,
    out: str,
    elapsed: float,
    budget: int,
    reference: float,
    driver_kind: str,
    system: str,
    seed: int,
) -> dict[str, str]:
    if status != "ok":
        return run_gate.failed_row(base, status, budget, reference, elapsed)
    try:
        parsed = run_gate.parse_output(out, driver_kind, system, seed, budget)
    except (KeyError, ValueError) as err:
        return run_gate.failed_row(base, f"parse:{err}", budget, reference, elapsed)
    wall_s = float(parsed["wall_s"])
    pot_s = float(parsed["pot_s"])
    outside_s = float(parsed["outside_s"]) if "outside_s" in parsed else max(wall_s - pot_s, 0.0)
    outside_s = max(outside_s, 0.0)
    charged = int(float(parsed["charged"]))
    best_val = float(parsed["best"])
    row = dict(base)
    row.update(
        {
            "budget": str(budget),
            "reference": f"{reference:.10f}",
            "hit": parsed["hit"],
            "evals_to_hit": str(int(float(parsed["evals_to_hit"]))),
            "charged": str(charged),
            "best": parsed["best"],
            "best_bits_hex": run_gate.float_bits_hex(best_val),
            "wall_s": f"{wall_s:.6f}",
            "pot_s": f"{pot_s:.6f}",
            "outside_s": f"{outside_s:.6f}",
            "outside_per_call_s": f"{outside_s / max(charged, 1):.12f}",
            "status": "ok",
        }
    )
    return row


def require_distinct(label: str, paths: list[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise SystemExit(f"{label} paths must be distinct")


def cmd_run(args: argparse.Namespace) -> int:
    aff_text = run_gate.verify_pinned(int(args.cpu))
    repo = Path(args.repo).resolve()
    work = Path(args.work).resolve()
    logs_dir = Path(args.logs).resolve()
    for sub in ("home", "tmp", "scratch"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    arms = {
        "A": run_gate.ArmSpec(
            "A",
            Path(args.arm_a_bin).resolve(),
            Path(args.arm_a_tree).resolve(),
            args.arm_a_library,
            args.arm_a_tree_sha,
        ),
        "B": run_gate.ArmSpec(
            "B",
            Path(args.arm_b_bin).resolve(),
            Path(args.arm_b_tree).resolve(),
            args.arm_b_library,
            args.arm_b_tree_sha,
        ),
        "A2": run_gate.ArmSpec(
            "A2",
            Path(args.arm_a2_bin).resolve(),
            Path(args.arm_a2_tree).resolve(),
            args.arm_a2_library,
            args.arm_a2_tree_sha,
        ),
    }
    if args.arm_a_library != ARM_A_LIBRARY or args.arm_a2_library != ARM_A_LIBRARY:
        raise SystemExit("arm A library id is not the declared commit")
    if args.arm_a_tree_sha != ARM_A_TREE or args.arm_a2_tree_sha != ARM_A_TREE:
        raise SystemExit("arm A tree is not the declared tree")
    if args.arm_b_library != ARM_B_LIBRARY or args.arm_b_tree_sha != ARM_B_TREE:
        raise SystemExit("arm B library id is not the declared tree")
    for name, expected in (("A", ARM_A_TREE), ("A2", ARM_A_TREE), ("B", ARM_B_TREE)):
        got = tree_sha(arms[name].tree_dir)
        if got != expected:
            raise SystemExit(f"arm {name} checkout tree does not match the declaration")
    require_distinct("build", [arm.bin_dir for arm in arms.values()])
    require_distinct("source", [arm.tree_dir for arm in arms.values()])

    extra = {name: "1" for name in EXTRA_THREAD_VARS}
    rgpot_sha = ""
    if "sc126-55" in stats.CELLS:
        if not args.rgpot_lib:
            raise SystemExit("sc126-55 requires the Sutton-Chen library")
        lib = Path(args.rgpot_lib).resolve()
        if not lib.is_file():
            raise SystemExit("Sutton-Chen library is not a file")
        extra["RGPOT_SC_LIB"] = str(lib)
        extra["LD_LIBRARY_PATH"] = str(lib.parent)
        rgpot_sha = run_gate.sha256_file(lib)
    env = run_gate.scrubbed_env(work, extra)
    for key in (*run_gate.THREAD_ENV_VARS, *EXTRA_THREAD_VARS):
        if env.get(key) != "1":
            raise SystemExit(f"{key} is not pinned to 1")

    panels = load_panels(Path(args.panels) if args.panels else repo / "bench/nm3-land-gate-b/panels.csv")
    budgets = check_budgets(arms, panels, env)
    if "sc126-55" in stats.CELLS:
        scratch = work / "scratch"
        for arm in arms.values():
            run_gate.preflight_sc_energy(
                arm.bin_dir / run_gate.DRIVERS["sc"],
                "sc126-55",
                env,
                scratch,
            )

    blobs = {
        (arm, kind): harness_blob_text(repo, arms[arm], kind)
        for arm in stats.ARMS
        for kind in ("blj", "sc")
    }
    bin_rows: list[dict[str, str]] = []
    binaries = sorted({run_gate.DRIVERS[DRIVER_KIND[cell]] for cell in stats.CELLS})
    for arm in stats.ARMS:
        for name in binaries:
            digest = run_gate.sha256_file(arms[arm].bin_dir / name)
            if len(digest) != 64:
                raise SystemExit(f"missing {name} for arm {arm}")
            bin_rows.append({"arm": arm, "binary": name, "sha256": digest})
    bin_path = Path(args.bin_sha256)
    if bin_path.is_file() and bin_path.stat().st_size:
        previous = run_gate.read_csv(bin_path)
        if previous != bin_rows:
            raise SystemExit("binary digests changed; refusing to mix builds")
    else:
        run_gate.write_csv(bin_path, run_gate.BIN_FIELDS, bin_rows)

    mid_sha = run_gate.hashed_machine_id()
    rows_path = Path(args.rows)
    existing = run_gate.read_csv(rows_path)
    for row in existing:
        seed = int(row["seed"])
        if row["system"] not in stats.CELLS or row["arm"] not in stats.ARMS or seed not in stats.SEEDS:
            raise SystemExit("existing rows are outside the declared design")
    session_ids = {row["session_id"] for row in existing}
    machines = {row["machine_id_sha256"] for row in existing}
    if existing and (len(session_ids) != 1 or machines != {mid_sha}):
        raise SystemExit("existing rows are not one session on this machine")
    if existing:
        session_id = session_ids.pop()
        known = {row["session_id"] for row in run_gate.read_csv(Path(args.sessions))}
        if session_id not in known:
            raise SystemExit("existing rows have no session record")
    else:
        session_id = run_gate.now_session_id()
        run_gate.append_csv(
            Path(args.sessions),
            [
                "session_id",
                "utc",
                "machine_id_sha256",
                "arch",
                "kernel",
                "cpu_model",
                "cpu_count",
                "affinity",
                "arm_a_library",
                "arm_a_tree",
                "arm_b_library",
                "arm_b_tree",
                "arm_a2_library",
                "arm_a2_tree",
                "rustc_version",
                "cargo_version",
                "python_version",
                "seeds",
                "cells",
                "arm_order_protocol",
                "job_timeout_s",
                "ljos_seat",
                "n_resamples",
                "bootstrap_rng_seed",
                "pass_threshold",
                "rgpot_sha256",
            ],
            {
                "session_id": session_id,
                "utc": run_gate.utc_now(),
                "machine_id_sha256": mid_sha,
                "arch": platform.machine(),
                "kernel": platform.release(),
                "cpu_model": run_gate.cpu_model(),
                "cpu_count": str(os.cpu_count() or 1),
                "affinity": aff_text,
                "arm_a_library": ARM_A_LIBRARY,
                "arm_a_tree": ARM_A_TREE,
                "arm_b_library": ARM_B_LIBRARY,
                "arm_b_tree": ARM_B_TREE,
                "arm_a2_library": ARM_A_LIBRARY,
                "arm_a2_tree": ARM_A_TREE,
                "rustc_version": run_gate.first_line(["rustc", "+1.88.0", "-V"]),
                "cargo_version": run_gate.first_line(["cargo", "+1.88.0", "-V"]),
                "python_version": platform.python_version(),
                "seeds": f"{stats.SEEDS[0]}-{stats.SEEDS[-1]}",
                "cells": " ".join(stats.CELLS),
                "arm_order_protocol": "six_permutation_williams_rotation",
                "job_timeout_s": f"{run_gate.JOB_TIMEOUT_S:.1f}",
                "ljos_seat": env["LJOS_SEAT"],
                "n_resamples": str(stats.N_RESAMPLES),
                "bootstrap_rng_seed": str(stats.RNG_SEED),
                "pass_threshold": f"{stats.THRESHOLD:.2f}",
                "rgpot_sha256": rgpot_sha,
            },
        )

    done = {(row["system"], row["arm"], int(row["seed"])) for row in existing}
    run_seq = len(existing)
    total = len(stats.CELLS) * len(stats.SEEDS) * len(stats.ARMS)
    print(
        f"session={session_id} cells={' '.join(stats.CELLS)} "
        f"seeds={stats.SEEDS[0]}-{stats.SEEDS[-1]} total_jobs={total} already_done={len(done)}",
        flush=True,
    )
    scratch = work / "scratch"
    for cell in stats.CELLS:
        driver_kind = DRIVER_KIND[cell]
        budget, reference = budgets[cell]
        panel_name = run_gate.PANEL_BY_DRIVER[driver_kind]
        driver_name = run_gate.DRIVERS[driver_kind]
        for seed in stats.SEEDS:
            order = stats.order_for_seed(seed)
            tag = stats.order_tag(order)
            for slot, arm_name in enumerate(order, start=1):
                if (cell, arm_name, seed) in done:
                    continue
                arm = arms[arm_name]
                run_seq += 1
                argv = run_gate.command_of(cell, driver_kind, arm, seed, work / "unused", 0, 0.0, {})
                log = logs_dir / f"{cell}-{arm_name}-s{seed}.log"
                base = {
                    "session_id": session_id,
                    "run_seq": str(run_seq),
                    "panel": panel_name,
                    "system": cell,
                    "driver": driver_name,
                    "arm": arm_name,
                    "seed": str(seed),
                    "slot": str(slot),
                    "arm_order": tag,
                    "library": arm.library,
                    "library_tree": arm.library_tree,
                    "harness_blobs": blobs[(arm_name, driver_kind)],
                    "trace_sha256": "",
                    "xyz_sha256": "",
                    "gmin_sha256": "",
                    "start_sha256": "",
                    "machine_id_sha256": mid_sha,
                    "utc": run_gate.utc_now(),
                }
                status, out, elapsed = run_gate.run_job(
                    argv, args.cpu, env, scratch, log, run_gate.JOB_TIMEOUT_S
                )
                row = finish_row(base, status, out, elapsed, budget, reference, driver_kind, cell, seed)
                run_gate.append_csv(rows_path, run_gate.ROW_FIELDS, row)
                done.add((cell, arm_name, seed))
                print(
                    f"[{len(done)}/{total}] {cell} seed={seed} slot={slot} arm={arm_name} "
                    f"order={tag} status={row['status']} hit={row['hit']} "
                    f"calls={row['evals_to_hit']} charged={row['charged']} "
                    f"wall={row['wall_s']} outside={row['outside_s']}",
                    flush=True,
                )
    return 0


def main(argv: list[str]) -> int:
    if "--quick" in argv:
        raise SystemExit("--quick is prohibited by the timing protocol")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the three cells serially on core 0")
    run.add_argument("--repo", required=True)
    run.add_argument("--arm-a-bin", required=True)
    run.add_argument("--arm-b-bin", required=True)
    run.add_argument("--arm-a2-bin", required=True)
    run.add_argument("--arm-a-tree", required=True)
    run.add_argument("--arm-b-tree", required=True)
    run.add_argument("--arm-a2-tree", required=True)
    run.add_argument("--arm-a-library", required=True)
    run.add_argument("--arm-b-library", required=True)
    run.add_argument("--arm-a2-library", required=True)
    run.add_argument("--arm-a-tree-sha", required=True)
    run.add_argument("--arm-b-tree-sha", required=True)
    run.add_argument("--arm-a2-tree-sha", required=True)
    run.add_argument("--work", required=True)
    run.add_argument("--logs", required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--sessions", required=True)
    run.add_argument("--bin-sha256", required=True)
    run.add_argument("--panels")
    run.add_argument("--rgpot-lib")
    run.add_argument("--cpu", default="0")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    if args.command == "run":
        assert_committed(Path(args.repo).resolve())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
