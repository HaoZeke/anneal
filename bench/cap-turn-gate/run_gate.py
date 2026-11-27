#!/usr/bin/env python3
"""Serial single-core ABBA gate runner for Config::cap_twist."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cells import (
    CANDIDATE_COMMIT,
    CANDIDATE_TREE,
    CELL_BY_NAME,
    DEVELOP_COMMIT,
    DEVELOP_TREE,
    HIT_TOL,
    MAIN_ARMS,
    MAIN_EXECUTION_ORDER,
    PADDING_KB_BY_ARM,
    PLACE_ARMS,
    PLACEMENT_CELLS,
    REPLACEMENT_CAP,
    SEEDS,
    main_arm_order,
    place_arm_order,
)

ROW_FIELDS: tuple[str, ...] = (
    "row_id",
    "band",
    "cell",
    "family",
    "n",
    "seed",
    "slot",
    "arm",
    "attempt",
    "superseded",
    "superseded_reason",
    "commit_sha",
    "tree_sha",
    "binary_sha256",
    "pad_kb",
    "budget",
    "reference",
    "hit",
    "evals_to_hit",
    "charged",
    "energy_calls",
    "grad_calls",
    "best",
    "best_bits",
    "hops",
    "basins",
    "wall_s",
    "pot_s",
    "status_s",
    "outside_s",
    "outside_per_call_ns",
    "rss_anon_kb",
    "vm_hwm_kb",
    "ru_maxrss_kb",
    "ru_utime_s",
    "ru_stime_s",
    "status_samples",
    "threads_max",
    "cpus_allowed",
    "cap_twist_flag",
    "cap_escapes",
    "cap_arm_draws",
    "cap_arm_accepts",
    "loadavg_1m",
    "exit_code",
    "machine_hash",
    "started_utc",
)

THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

LOADAVG_LIMIT: float = 2.50


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    digest = sha256_file(path)
    expected = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise RuntimeError(f"sha256 mismatch on {path.name}")
    return digest


def atomic_write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return sha256_file(path)


def machine_id_hash() -> str:
    return "unrecorded"


def read_loadavg_1m() -> float:
    text = Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    return float(text.split()[0])


def append_journal(journal_path: Path, existing_lines: list[str], event: dict[str, Any]) -> str:
    line = json.dumps(event, sort_keys=True, separators=(",", ":"))
    existing_lines.append(line)
    payload = ("\n".join(existing_lines) + "\n").encode("utf-8")
    return atomic_write_bytes(journal_path, payload)


def build_child_env(arm: str) -> dict[str, str]:
    # Clean, deterministic child environment: never forward parent secrets or
    # extraneous variables.
    env: dict[str, str] = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    env.update(THREAD_ENV)
    if arm == "B":
        env["H2H_CAP_TWIST"] = "1"
    return env


def parse_gate_row(stdout_text: str) -> dict[str, str] | None:
    for line in stdout_text.splitlines():
        if line.startswith("gate_row "):
            parsed: dict[str, str] = {}
            for token in line[len("gate_row ") :].split():
                if "=" in token:
                    key, val = token.split("=", 1)
                    parsed[key] = val
            return parsed
    return None


def spawn_and_wait4(
    binary_path: Path,
    cell_name: str,
    seed: int,
    core: int,
    child_env: dict[str, str],
    raw_log_path: Path,
) -> tuple[int, str, float, float, int]:
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = raw_log_path.with_name(raw_log_path.name + ".out.tmp")
    cmd = [
        "taskset",
        "-c",
        str(core),
        str(binary_path),
        cell_name,
        str(seed),
    ]
    with tmp_out.open("wb") as out_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=out_fh,
            stderr=subprocess.STDOUT,
            env=child_env,
            close_fds=True,
        )
        _, status, rusage = os.wait4(proc.pid, 0)
        out_fh.flush()
        os.fsync(out_fh.fileno())
    exit_code = os.waitstatus_to_exitcode(status)
    raw_bytes = tmp_out.read_bytes()
    os.replace(tmp_out, raw_log_path)
    _ = sha256_file(raw_log_path)
    return (
        exit_code,
        raw_bytes.decode("utf-8", errors="replace"),
        float(rusage.ru_utime),
        float(rusage.ru_stime),
        int(rusage.ru_maxrss),
    )


def load_existing_rows(rows_path: Path) -> list[dict[str, Any]]:
    if not rows_path.exists():
        return []
    with rows_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_existing_journal(journal_path: Path) -> list[str]:
    if not journal_path.exists():
        return []
    return [
        line
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def arm_commit_and_tree(arm: str) -> tuple[str, str]:
    if arm == "D":
        return DEVELOP_COMMIT, DEVELOP_TREE
    return CANDIDATE_COMMIT, CANDIDATE_TREE


def validate_parsed(
    parsed: dict[str, str] | None,
    exit_code: int,
    loadavg_1m: float,
    arm: str,
    expected_pad_kb: int,
    expected_budget: int,
    core: int,
) -> str:
    if exit_code != 0:
        return f"exit_{exit_code}"
    if parsed is None:
        return "missing_gate_row"
    if int(parsed.get("budget", "0")) != expected_budget:
        return "budget_mismatch"
    if int(parsed.get("pad_kb", "-1")) != expected_pad_kb:
        return "pad_kb_mismatch"
    if int(parsed.get("threads_max", "0")) != 1:
        return "threads_not_1"
    if parsed.get("cpus_allowed", "") != str(core):
        return f"cpus_allowed_{parsed.get('cpus_allowed', 'bad')}"
    expected_flag = 1 if arm == "B" else 0
    if int(parsed.get("cap_twist_flag", "-1")) != expected_flag:
        return "cap_twist_flag_mismatch"
    if int(parsed.get("rss_anon_kb", "0")) <= 0:
        return "rss_anon_zero"
    if loadavg_1m > LOADAVG_LIMIT:
        return f"loadavg_{loadavg_1m:.2f}"
    return ""


def make_row(
    row_id: int,
    band: str,
    cell_name: str,
    seed: int,
    slot: int,
    arm: str,
    attempt: int,
    superseded: int,
    superseded_reason: str,
    binary_sha: str,
    parsed: dict[str, str] | None,
    ru_utime_s: float,
    ru_stime_s: float,
    ru_maxrss_kb: int,
    loadavg_1m: float,
    exit_code: int,
    mhash: str,
    started_utc: str,
) -> dict[str, Any]:
    spec = CELL_BY_NAME[cell_name]
    commit_sha, tree_sha = arm_commit_and_tree(arm)
    pad_kb = PADDING_KB_BY_ARM[arm]
    if parsed is None:
        return {
            "row_id": row_id,
            "band": band,
            "cell": cell_name,
            "family": spec.family,
            "n": spec.n,
            "seed": seed,
            "slot": slot,
            "arm": arm,
            "attempt": attempt,
            "superseded": superseded,
            "superseded_reason": superseded_reason,
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "binary_sha256": binary_sha,
            "pad_kb": pad_kb,
            "budget": spec.budget,
            "reference": f"{spec.reference:.10f}",
            "hit": 0,
            "evals_to_hit": spec.budget,
            "charged": 0,
            "energy_calls": 0,
            "grad_calls": 0,
            "best": "nan",
            "best_bits": "0x0000000000000000",
            "hops": 0,
            "basins": 0,
            "wall_s": "0.000000",
            "pot_s": "0.000000",
            "status_s": "0.000000",
            "outside_s": "0.000000",
            "outside_per_call_ns": "0.000000",
            "rss_anon_kb": 0,
            "vm_hwm_kb": 0,
            "ru_maxrss_kb": ru_maxrss_kb,
            "ru_utime_s": f"{ru_utime_s:.6f}",
            "ru_stime_s": f"{ru_stime_s:.6f}",
            "status_samples": 0,
            "threads_max": 0,
            "cpus_allowed": "none",
            "cap_twist_flag": 0,
            "cap_escapes": 0,
            "cap_arm_draws": 0,
            "cap_arm_accepts": 0,
            "loadavg_1m": f"{loadavg_1m:.2f}",
            "exit_code": exit_code,
            "machine_hash": mhash,
            "started_utc": started_utc,
        }

    charged = int(parsed["charged"])
    outside_s = float(parsed["outside_s"])
    outside_per_call_ns = (outside_s * 1.0e9) / max(1, charged)
    best_val = float(parsed["best"])
    rederived_hit = int(best_val <= spec.reference + HIT_TOL)
    return {
        "row_id": row_id,
        "band": band,
        "cell": cell_name,
        "family": spec.family,
        "n": spec.n,
        "seed": seed,
        "slot": slot,
        "arm": arm,
        "attempt": attempt,
        "superseded": superseded,
        "superseded_reason": superseded_reason,
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "binary_sha256": binary_sha,
        "pad_kb": pad_kb,
        "budget": int(parsed["budget"]),
        "reference": f"{spec.reference:.10f}",
        "hit": rederived_hit,
        "evals_to_hit": int(parsed["evals_to_hit"]),
        "charged": charged,
        "energy_calls": int(parsed["energy_calls"]),
        "grad_calls": int(parsed["grad_calls"]),
        "best": parsed["best"],
        "best_bits": parsed["best_bits"],
        "hops": int(parsed["hops"]),
        "basins": int(parsed["basins"]),
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "status_s": parsed["status_s"],
        "outside_s": parsed["outside_s"],
        "outside_per_call_ns": f"{outside_per_call_ns:.6f}",
        "rss_anon_kb": int(parsed["rss_anon_kb"]),
        "vm_hwm_kb": int(parsed["vm_hwm_kb"]),
        "ru_maxrss_kb": ru_maxrss_kb,
        "ru_utime_s": f"{ru_utime_s:.6f}",
        "ru_stime_s": f"{ru_stime_s:.6f}",
        "status_samples": int(parsed["status_samples"]),
        "threads_max": int(parsed["threads_max"]),
        "cpus_allowed": parsed["cpus_allowed"],
        "cap_twist_flag": int(parsed["cap_twist_flag"]),
        "cap_escapes": int(parsed["cap_escapes"]),
        "cap_arm_draws": int(parsed["cap_arm_draws"]),
        "cap_arm_accepts": int(parsed["cap_arm_accepts"]),
        "loadavg_1m": f"{loadavg_1m:.2f}",
        "exit_code": exit_code,
        "machine_hash": mhash,
        "started_utc": started_utc,
    }


def pin_orchestrator_away_from_core(bench_core: int) -> list[int]:
    avail = sorted(os.sched_getaffinity(0))
    other = [c for c in avail if c != bench_core]
    if other:
        os.sched_setaffinity(0, set(other))
        return other
    return avail


def run_band_schedule(
    band: str,
    cells_order: tuple[str, ...],
    arm_order_fn: Any,
    binaries: dict[str, Path],
    binary_hashes: dict[str, str],
    rows: list[dict[str, Any]],
    rows_path: Path,
    raw_dir: Path,
    journal_path: Path,
    journal_lines: list[str],
    bench_core: int,
    mhash: str,
) -> None:
    active_done: set[tuple[str, str, int, str]] = {
        (r["band"], r["cell"], int(r["seed"]), r["arm"])
        for r in rows
        if int(r["superseded"]) == 0
    }
    replacements_by_cell: dict[tuple[str, str], int] = {}
    attempts_by_key: dict[tuple[str, str, int, str], int] = {}
    for r in rows:
        ckey = (r["band"], r["cell"])
        if int(r["superseded"]) == 1:
            replacements_by_cell[ckey] = replacements_by_cell.get(ckey, 0) + 1
        rkey = (r["band"], r["cell"], int(r["seed"]), r["arm"])
        attempts_by_key[rkey] = max(attempts_by_key.get(rkey, 0), int(r["attempt"]))

    next_row_id = len(rows) + 1
    t_band = time.monotonic()

    for cell_idx, cell_name in enumerate(cells_order):
        spec = CELL_BY_NAME[cell_name]
        ckey = (band, cell_name)
        t_cell = time.monotonic()
        print(
            f"[{band}] start cell={cell_name} ({cell_idx + 1}/{len(cells_order)}) budget={spec.budget}",
            flush=True,
        )
        for seed_idx, seed in enumerate(SEEDS):
            arms_this_seed = arm_order_fn(cell_idx, seed_idx)
            for slot, arm in enumerate(arms_this_seed):
                rkey = (band, cell_name, seed, arm)
                if rkey in active_done:
                    continue
                if replacements_by_cell.get(ckey, 0) > REPLACEMENT_CAP:
                    continue

                while True:
                    attempt = attempts_by_key.get(rkey, 0) + 1
                    attempts_by_key[rkey] = attempt
                    started_utc = (
                        dt.datetime.now(dt.timezone.utc)
                        .replace(microsecond=0)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
                    raw_log_name = (
                        f"{band}-{cell_name}-s{seed}-{arm}-a{attempt}.log"
                    )
                    raw_log_path = raw_dir / raw_log_name
                    child_env = build_child_env(arm)
                    exit_code, stdout_text, ru_u, ru_s, ru_rss = spawn_and_wait4(
                        binaries[arm],
                        cell_name,
                        seed,
                        bench_core,
                        child_env,
                        raw_log_path,
                    )
                    loadavg_1m = read_loadavg_1m()
                    parsed = parse_gate_row(stdout_text)
                    reason = validate_parsed(
                        parsed,
                        exit_code,
                        loadavg_1m,
                        arm,
                        PADDING_KB_BY_ARM[arm],
                        spec.budget,
                        bench_core,
                    )
                    is_bad = 1 if reason else 0
                    row = make_row(
                        row_id=next_row_id,
                        band=band,
                        cell_name=cell_name,
                        seed=seed,
                        slot=slot,
                        arm=arm,
                        attempt=attempt,
                        superseded=is_bad,
                        superseded_reason=reason,
                        binary_sha=binary_hashes[arm],
                        parsed=parsed,
                        ru_utime_s=ru_u,
                        ru_stime_s=ru_s,
                        ru_maxrss_kb=ru_rss,
                        loadavg_1m=loadavg_1m,
                        exit_code=exit_code,
                        mhash=mhash,
                        started_utc=started_utc,
                    )
                    rows.append(row)
                    next_row_id += 1
                    rows_sha = atomic_write_csv(rows_path, ROW_FIELDS, rows)

                    if not is_bad:
                        active_done.add(rkey)
                        break

                    replacements_by_cell[ckey] = (
                        replacements_by_cell.get(ckey, 0) + 1
                    )
                    append_journal(
                        journal_path,
                        journal_lines,
                        {
                            "event": "row_superseded",
                            "band": band,
                            "cell": cell_name,
                            "seed": seed,
                            "arm": arm,
                            "attempt": attempt,
                            "reason": reason,
                            "replacements_used": replacements_by_cell[ckey],
                            "rows_sha256": rows_sha,
                            "utc": started_utc,
                        },
                    )
                    if replacements_by_cell[ckey] > REPLACEMENT_CAP:
                        print(
                            f"[{band}] cell={cell_name} exceeded replacement cap ({replacements_by_cell[ckey]})",
                            flush=True,
                        )
                        break

        elapsed_cell = time.monotonic() - t_cell
        total_elapsed = time.monotonic() - t_band
        print(
            f"[{band}] done cell={cell_name} cell_s={elapsed_cell:.1f} band_s={total_elapsed:.1f} total_rows={len(rows)}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cap-turn default-on gate.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSON mapping each arm (A, B, D, PA, A2, P1..P5) to binary path.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for rows.csv, raw logs, and journal.jsonl.",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=0,
        help="Single CPU core for benchmark children (default: 0).",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "place", "main"),
        default="all",
        help="Which schedule band(s) to execute.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binaries: dict[str, Path] = {k: Path(v) for k, v in manifest["binaries"].items()}

    needed_arms: set[str] = set()
    if args.mode in ("all", "place"):
        needed_arms.update(PLACE_ARMS)
    if args.mode in ("all", "main"):
        needed_arms.update(MAIN_ARMS)

    binary_hashes: dict[str, str] = {}
    for arm in sorted(needed_arms):
        if arm not in binaries or not binaries[arm].exists():
            raise RuntimeError(f"missing binary for arm {arm}")
        binary_hashes[arm] = sha256_file(binaries[arm])

    if "A" in binary_hashes and "B" in binary_hashes:
        if binary_hashes["A"] != binary_hashes["B"]:
            raise RuntimeError("Arm A and Arm B must use the exact same binary")
    if "A" in binary_hashes and "PA" in binary_hashes:
        if binary_hashes["A"] != binary_hashes["PA"]:
            raise RuntimeError("Arm PA and Arm A must use the exact same binary")

    orch_cores = pin_orchestrator_away_from_core(args.core)
    mhash = machine_id_hash()

    out_dir = Path(args.out_dir)
    rows_path = out_dir / "rows.csv"
    raw_dir = out_dir / "raw"
    journal_path = out_dir / "journal.jsonl"
    rows = load_existing_rows(rows_path)
    journal_lines = load_existing_journal(journal_path)

    now_utc = (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    append_journal(
        journal_path,
        journal_lines,
        {
            "event": "session_start",
            "mode": args.mode,
            "bench_core": args.core,
            "orchestrator_cores": orch_cores,
            "binary_sha256": binary_hashes,
            "machine_hash": mhash,
            "existing_rows": len(rows),
            "thread_env": THREAD_ENV,
            "utc": now_utc,
        },
    )

    if args.mode in ("all", "place"):
        run_band_schedule(
            band="place",
            cells_order=PLACEMENT_CELLS,
            arm_order_fn=place_arm_order,
            binaries=binaries,
            binary_hashes=binary_hashes,
            rows=rows,
            rows_path=rows_path,
            raw_dir=raw_dir,
            journal_path=journal_path,
            journal_lines=journal_lines,
            bench_core=args.core,
            mhash=mhash,
        )

    if args.mode in ("all", "main"):
        run_band_schedule(
            band="main",
            cells_order=MAIN_EXECUTION_ORDER,
            arm_order_fn=main_arm_order,
            binaries=binaries,
            binary_hashes=binary_hashes,
            rows=rows,
            rows_path=rows_path,
            raw_dir=raw_dir,
            journal_path=journal_path,
            journal_lines=journal_lines,
            bench_core=args.core,
            mhash=mhash,
        )

    end_utc = (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    rows_sha = sha256_file(rows_path) if rows_path.exists() else "empty"
    append_journal(
        journal_path,
        journal_lines,
        {
            "event": "session_end",
            "mode": args.mode,
            "total_rows": len(rows),
            "rows_sha256": rows_sha,
            "utc": end_utc,
        },
    )
    print(f"COMPLETE total_rows={len(rows)} rows_sha256={rows_sha}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
