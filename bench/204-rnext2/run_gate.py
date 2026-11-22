#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial four-arm default-panel gate runner for anneal #204 r-next2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HIT_TOL = 1.0e-3
DEFAULT_TIMEOUT_S = 21600
ARMS = ("A", "B", "AA", "AP")
DEFAULT_SEEDS = tuple(range(2000, 2012))
EXPLICIT_SPECS = (
    ("lj104", 2001, ("A", "B", "AA", "AP")),
    ("lj104", 2021, ("AP", "AA", "B", "A")),
)

# Balanced 4x4 Williams square on (A, B, AA, AP).
WILLIAMS_ORDER = (
    ("A", "B", "AP", "AA"),
    ("B", "AA", "A", "AP"),
    ("AA", "AP", "B", "A"),
    ("AP", "A", "AA", "B"),
)

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
    "POLARS_MAX_THREADS",
    "TBB_NUM_THREADS",
)

ROW_FIELDS = [
    "panel",
    "cell",
    "family",
    "arm",
    "seed",
    "slot",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "best_bits",
    "wall_s",
    "inside_s",
    "outside_s",
    "outside_us",
    "rss_anon_kb",
    "end_rss_anon_kb",
    "vm_hwm_kb",
    "ru_maxrss_kb",
    "arm_rev",
    "arm_tree",
    "bin_sha256",
    "core",
    "cpus_allowed",
    "taskset_mask",
    "elapsed_s",
]


@dataclass(frozen=True)
class CellSpec:
    system: str
    family: str
    driver: str
    budget: int
    reference: float


PANEL_CELLS = (
    CellSpec("lj38", "lj", "h2h_lj", 400_000, -173.928427),
    CellSpec("lj75", "lj", "h2h_lj", 4_000_000, -397.492331),
    CellSpec("lj98", "lj", "h2h_lj", 4_000_000, -543.665361),
    CellSpec("lj102", "lj", "h2h_lj", 4_000_000, -569.363652),
    CellSpec("lj104", "lj", "h2h_lj", 4_000_000, -582.086642),
    CellSpec("m80r14", "morse", "h2h_morse", 4_000_000, -340.811371),
    CellSpec("sc108-80", "sc", "h2h_sc", 1_000_000, -21819.9952),
    CellSpec("blj130n38", "blj", "h2h_blj", 400_000, -191.1840192602),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_bits_hex(value: float) -> str:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{raw:016x}"


def is_hit(best: float, reference: float) -> bool:
    return math.isfinite(best) and best <= reference + HIT_TOL


def build_env(sc_lib: Path | None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "MKL_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
        "PYTHONHASHSEED": "0",
    }
    for key in ("LANG", "LC_ALL", "LD_LIBRARY_PATH", "TMPDIR", "TERM"):
        if key in os.environ:
            env[key] = os.environ[key]
    for key in THREAD_ENV_KEYS:
        env[key] = "1"
    if sc_lib is not None:
        env["RGPOT_SC_LIB"] = str(sc_lib)
    return env


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle))
        if existing_header != ROW_FIELDS:
            raise RuntimeError(f"{path} header mismatch")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({k: str(row[k]) for k in ROW_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def read_existing_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (r["panel"], r["cell"], r["arm"], r["seed"])
            for r in csv.DictReader(handle)
        }


def verify_budgets(
    cells: tuple[CellSpec, ...],
    bins: dict[str, Path],
    env: dict[str, str],
) -> None:
    for cell in cells:
        for arm in ARMS:
            exe = bins[arm] / cell.driver
            proc = subprocess.run(
                ["taskset", "-c", "0", str(exe), "budget", cell.system],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"budget probe failed for {arm} {cell.system}: {proc.stderr.strip()}"
                )
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            got_budget = int(lines[0])
            got_ref = float(lines[1])
            if got_budget != cell.budget:
                raise RuntimeError(f"{arm} {cell.system}: budget {got_budget} != {cell.budget}")
            if abs(got_ref - cell.reference) > 1e-6:
                raise RuntimeError(f"{arm} {cell.system}: ref {got_ref} != {cell.reference}")


def parse_driver_output(
    stdout: str,
    system: str,
    seed: int,
    budget: int,
) -> dict[str, str]:
    headers = [ln for ln in stdout.splitlines() if ln.startswith("system,")]
    if not headers:
        raise RuntimeError(f"{system} seed {seed}: missing CSV header")
    header = headers[-1].split(",")
    prefix = f"{system},anneal,{seed},"
    matches = [ln for ln in stdout.splitlines() if ln.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected 1 CSV row, got {len(matches)}")
    row = dict(zip(header, matches[0].split(","), strict=True))
    if int(float(row["budget"])) != budget:
        raise RuntimeError(f"{system} seed {seed}: budget {row['budget']} != {budget}")

    mem_lines = [ln for ln in stdout.splitlines() if ln.startswith("h2h_mem ")]
    mem_kv: dict[str, str] = {}
    if mem_lines:
        for tok in mem_lines[-1].split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                mem_kv[k] = v

    return {
        "best": row["best"].strip(),
        "driver_hit": "1" if row["hit"].strip().lower() in ("1", "true") else "0",
        "evals_to_hit": str(int(float(row["evals_to_hit"]))),
        "charged": str(int(float(row["charged"]))),
        "wall_s": row["wall_s"].strip(),
        "inside_s": row["pot_s"].strip(),
        "outside_s": row["outside_s"].strip(),
        "end_rss_anon_kb": mem_kv.get("rss_anon_kb", "0"),
        "vm_hwm_kb": mem_kv.get("vm_hwm_kb", "0"),
        "best_bits": mem_kv.get("best_bits", ""),
    }


def run_one(
    *,
    panel: str,
    cell: CellSpec,
    arm: str,
    seed: int,
    slot: int,
    exe: Path,
    bin_sha256: str,
    arm_rev: str,
    arm_tree: str,
    work_dir: Path,
    logs_dir: Path,
    env: dict[str, str],
    timeout_s: int,
) -> dict[str, str]:
    run_dir = work_dir / panel / cell.system / f"s{seed}-{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{panel}-{cell.system}-s{seed}-{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    out_file = run_dir / "stdout.txt"
    err_file = run_dir / "stderr.txt"
    cmd = ["taskset", "-c", "0", str(exe), "run", cell.system, str(seed)]

    polled_rss_anon_kb = 0
    cpus_allowed = ""
    taskset_p_line = ""
    taskset_mask = ""
    timed_out = False
    rc = -1

    t0 = time.monotonic()
    with out_file.open("w", encoding="utf-8") as out_f, err_file.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=run_dir,
            env=env,
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        status_path = Path(f"/proc/{proc.pid}/status")
        while True:
            state = ""
            try:
                st_text = status_path.read_text(encoding="utf-8")
                for ln in st_text.splitlines():
                    if ln.startswith("State:"):
                        state = ln.split(":", 1)[1].strip()
                    elif ln.startswith("RssAnon:"):
                        val = int(ln.split()[1])
                        if val > polled_rss_anon_kb:
                            polled_rss_anon_kb = val
                    elif ln.startswith("Cpus_allowed_list:"):
                        cpus_allowed = ln.split(":", 1)[1].strip()
            except FileNotFoundError:
                pass
            if state.startswith("Z"):
                break
            elapsed_now = time.monotonic() - t0
            if elapsed_now > timeout_s:
                timed_out = True
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.0005 if elapsed_now < 0.05 else 0.015)

        tp = subprocess.run(
            ["taskset", "-p", str(proc.pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        taskset_p_line = tp.stdout.strip()
        if ":" in taskset_p_line:
            taskset_mask = taskset_p_line.rsplit(":", 1)[1].strip()

        _pid_done, wstatus, rusage = os.wait4(proc.pid, 0)
        ru_maxrss_kb = int(rusage.ru_maxrss)
        if not timed_out:
            rc = os.waitstatus_to_exitcode(wstatus)
    elapsed = time.monotonic() - t0

    stdout = out_file.read_text(encoding="utf-8", errors="replace")
    stderr = err_file.read_text(encoding="utf-8", errors="replace")

    if cpus_allowed != "0":
        raise RuntimeError(f"{panel}/{cell.system} {arm} s{seed}: Cpus_allowed_list={cpus_allowed!r}")
    if taskset_mask != "1":
        raise RuntimeError(f"{panel}/{cell.system} {arm} s{seed}: taskset -p={taskset_p_line!r}")

    if timed_out or rc != 0:
        log_path.write_text(
            f"cmd=taskset -c 0 {exe.name} run {cell.system} {seed}\n"
            f"arm={arm} rev={arm_rev} tree={arm_tree}\n"
            f"taskset_p={taskset_p_line}\n"
            f"Cpus_allowed_list={cpus_allowed}\n"
            f"polled_rss_anon_kb={polled_rss_anon_kb}\n"
            f"ru_maxrss_kb={ru_maxrss_kb}\n"
            f"rc={rc} timed_out={timed_out} elapsed_s={elapsed:.6f}\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
            encoding="utf-8",
        )
        return {
            "panel": panel,
            "cell": cell.system,
            "family": cell.family,
            "arm": arm,
            "seed": str(seed),
            "slot": str(slot),
            "budget": str(cell.budget),
            "hit": "0",
            "evals_to_hit": str(cell.budget),
            "charged": "0",
            "best": "nan",
            "best_bits": float_bits_hex(float("nan")),
            "wall_s": f"{elapsed:.6f}",
            "inside_s": "0.000000",
            "outside_s": f"{elapsed:.6f}",
            "outside_us": "inf",
            "rss_anon_kb": str(polled_rss_anon_kb),
            "end_rss_anon_kb": "0",
            "vm_hwm_kb": "0",
            "ru_maxrss_kb": str(ru_maxrss_kb),
            "arm_rev": arm_rev,
            "arm_tree": arm_tree,
            "bin_sha256": bin_sha256,
            "core": "0",
            "cpus_allowed": cpus_allowed,
            "taskset_mask": taskset_mask,
            "elapsed_s": f"{elapsed:.6f}",
        }

    parsed = parse_driver_output(stdout, cell.system, seed, cell.budget)
    end_rss_anon_kb = int(parsed["end_rss_anon_kb"])
    peak_rss_anon_kb = max(polled_rss_anon_kb, end_rss_anon_kb)
    if peak_rss_anon_kb <= 0:
        raise RuntimeError(f"{panel}/{cell.system} {arm} s{seed}: peak RssAnon not recorded")

    best_val = float(parsed["best"])
    hit_val = 1 if is_hit(best_val, cell.reference) else 0
    if str(hit_val) != parsed["driver_hit"] and math.isfinite(best_val):
        if abs(best_val - (cell.reference + HIT_TOL)) <= 1e-7:
            hit_val = int(parsed["driver_hit"])
        else:
            raise RuntimeError(
                f"{panel}/{cell.system} {arm} s{seed}: hit mismatch best={parsed['best']} "
                f"vs driver_hit={parsed['driver_hit']}"
            )

    evals = parsed["evals_to_hit"] if hit_val else str(cell.budget)
    charged = int(parsed["charged"])
    outside_s = float(parsed["outside_s"])
    outside_us = outside_s / max(1, charged) * 1.0e6
    best_bits = parsed["best_bits"] or float_bits_hex(best_val)

    log_path.write_text(
        f"cmd=taskset -c 0 {exe.name} run {cell.system} {seed}\n"
        f"arm={arm} rev={arm_rev} tree={arm_tree}\n"
        f"taskset_p={taskset_p_line}\n"
        f"Cpus_allowed_list={cpus_allowed}\n"
        f"polled_rss_anon_kb={polled_rss_anon_kb} end_rss_anon_kb={end_rss_anon_kb} "
        f"peak_rss_anon_kb={peak_rss_anon_kb} vm_hwm_kb={parsed['vm_hwm_kb']} "
        f"ru_maxrss_kb={ru_maxrss_kb}\n"
        f"rc={rc} timed_out={timed_out} elapsed_s={elapsed:.6f}\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )

    return {
        "panel": panel,
        "cell": cell.system,
        "family": cell.family,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "budget": str(cell.budget),
        "hit": str(hit_val),
        "evals_to_hit": evals,
        "charged": str(charged),
        "best": parsed["best"],
        "best_bits": best_bits,
        "wall_s": parsed["wall_s"],
        "inside_s": parsed["inside_s"],
        "outside_s": f"{outside_s:.6f}",
        "outside_us": f"{outside_us:.6f}",
        "rss_anon_kb": str(peak_rss_anon_kb),
        "end_rss_anon_kb": str(end_rss_anon_kb),
        "vm_hwm_kb": parsed["vm_hwm_kb"],
        "ru_maxrss_kb": str(ru_maxrss_kb),
        "arm_rev": arm_rev,
        "arm_tree": arm_tree,
        "bin_sha256": bin_sha256,
        "core": "0",
        "cpus_allowed": cpus_allowed,
        "taskset_mask": taskset_mask,
        "elapsed_s": f"{elapsed:.6f}",
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=here / "rows.csv")
    parser.add_argument("--work-dir", type=Path, default=here / "work")
    parser.add_argument("--logs-dir", type=Path, default=here / "logs")
    parser.add_argument("--bin-a", type=Path, required=True)
    parser.add_argument("--bin-b", type=Path, required=True)
    parser.add_argument("--bin-aa", type=Path, required=True)
    parser.add_argument("--bin-ap", type=Path, required=True)
    parser.add_argument("--rev-a", required=True)
    parser.add_argument("--rev-b", required=True)
    parser.add_argument("--rev-aa", required=True)
    parser.add_argument("--rev-ap", required=True)
    parser.add_argument("--tree-a", required=True)
    parser.add_argument("--tree-b", required=True)
    parser.add_argument("--tree-aa", required=True)
    parser.add_argument("--tree-ap", required=True)
    parser.add_argument("--sc-lib", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args()

    bins = {"A": args.bin_a, "B": args.bin_b, "AA": args.bin_aa, "AP": args.bin_ap}
    revs = {"A": args.rev_a, "B": args.rev_b, "AA": args.rev_aa, "AP": args.rev_ap}
    trees = {"A": args.tree_a, "B": args.tree_b, "AA": args.tree_aa, "AP": args.tree_ap}

    if not args.sc_lib.is_file():
        raise FileNotFoundError(f"sc-lib not found: {args.sc_lib}")

    env = build_env(args.sc_lib)
    verify_budgets(PANEL_CELLS, bins, env)

    sha_cache: dict[tuple[str, str], str] = {}
    for cell in PANEL_CELLS:
        for arm in ARMS:
            exe = bins[arm] / cell.driver
            if not exe.is_file():
                raise FileNotFoundError(f"missing binary: {exe}")
            sha_cache[(arm, cell.driver)] = sha256_file(exe)

    done = read_existing_keys(args.rows)

    for cell in PANEL_CELLS:
        for seed in DEFAULT_SEEDS:
            order = WILLIAMS_ORDER[(seed - 2000) % len(WILLIAMS_ORDER)]
            for slot, arm in enumerate(order, 1):
                key = ("default", cell.system, arm, str(seed))
                if key in done:
                    continue
                row = run_one(
                    panel="default",
                    cell=cell,
                    arm=arm,
                    seed=seed,
                    slot=slot,
                    exe=bins[arm] / cell.driver,
                    bin_sha256=sha_cache[(arm, cell.driver)],
                    arm_rev=revs[arm],
                    arm_tree=trees[arm],
                    work_dir=args.work_dir,
                    logs_dir=args.logs_dir,
                    env=env,
                    timeout_s=args.timeout_s,
                )
                append_row(args.rows, row)
                done.add(key)
                print(
                    f"[default] {cell.system} s{seed} {arm} slot={slot} "
                    f"hit={row['hit']} evals={row['evals_to_hit']} "
                    f"out_us={float(row['outside_us']):.3f} rss={row['rss_anon_kb']}KB",
                    flush=True,
                )

    lj104_cell = next(c for c in PANEL_CELLS if c.system == "lj104")
    for system, seed, order in EXPLICIT_SPECS:
        assert system == lj104_cell.system
        for slot, arm in enumerate(order, 1):
            key = ("explicit", system, arm, str(seed))
            if key in done:
                continue
            row = run_one(
                panel="explicit",
                cell=lj104_cell,
                arm=arm,
                seed=seed,
                slot=slot,
                exe=bins[arm] / lj104_cell.driver,
                bin_sha256=sha_cache[(arm, lj104_cell.driver)],
                arm_rev=revs[arm],
                arm_tree=trees[arm],
                work_dir=args.work_dir,
                logs_dir=args.logs_dir,
                env=env,
                timeout_s=args.timeout_s,
            )
            append_row(args.rows, row)
            done.add(key)
            print(
                f"[explicit] {system} s{seed} {arm} slot={slot} "
                f"hit={row['hit']} best={row['best']} evals={row['evals_to_hit']} "
                f"out_us={float(row['outside_us']):.3f} rss={row['rss_anon_kb']}KB",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
