#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Three-arm serial no-regression gate for the basin-bound speedup.

Arms A and A2 are separate release builds of one develop tree. Arm B
is that tree plus the block-sum basin bound and the one-turn
symmetrise rotation. Drivers are the same source on every arm. The
failure rule is the one declared in the README beside this file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HIT_TOL = 1.0e-3
OUTSIDE_RATIO_LIMIT = 1.10
RSS_RATIO_LIMIT = 1.05
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261121
SEEDS = tuple(range(2000, 2024))
DEFAULT_TIMEOUT_S = 21600
ARMS = ("A", "A2", "B")
OBSERVATIONS = (("A", "A"), ("B1", "B"), ("B2", "B"), ("A2", "A2"))
OBSERVATION_ARMS = dict(OBSERVATIONS)
DRIVERS = ("h2h_gmin_proof", "h2h_seed", "h2h_morse", "h2h_blj", "h2h_sc")
PREFLIGHT_CELLS = {
    "h2h_gmin_proof": "lj38",
    "h2h_blj": "blj115n38",
    "h2h_morse": "m55r6",
    "h2h_sc": "sc126-55",
    "h2h_seed": "tip4p8",
}
PREFLIGHT_SEED = SEEDS[0]

ROW_FIELDS = [
    "cell",
    "arm",
    "observation",
    "seed",
    "slot",
    "rotation",
    "superseded",
    "status",
    "exit_code",
    "hit",
    "evals",
    "calls_to_hit",
    "charged",
    "best",
    "best_bits",
    "wall_s",
    "pot_s",
    "outside_s",
    "elapsed_s",
    "ru_utime_s",
    "ru_stime_s",
    "rss_anon_kb",
    "external_rss_anon_kb",
    "vm_hwm_kb",
    "rss_disagreement_kb",
    "ru_maxrss_kb",
    "fixed_ring_bound",
    "library_tree",
    "binary_sha256",
    "log_sha256",
    "core",
    "affinity",
    "taskset_mask",
    "machine_id_sha256",
    "session",
    "sequence",
    "started_unix",
    "note",
]

SUMMARY_FIELDS = [
    "cell",
    "family",
    "budget",
    "reference",
    "rows",
    "inconclusive",
    "traj_match_ba",
    "traj_match_a2a",
    "hits_a",
    "hits_a2",
    "hits_b1",
    "hits_b2",
    "hit_loss_seeds",
    "med_calls_a",
    "med_calls_b",
    "outside_a_us",
    "outside_a2_us",
    "outside_b_us",
    "outside_ba",
    "outside_ba_ci_lo",
    "outside_ba_ci_hi",
    "outside_a2a",
    "outside_a2a_ci_lo",
    "outside_a2a_ci_hi",
    "outside_limit",
    "rss_anon_a_kb",
    "rss_anon_a2_kb",
    "rss_anon_b_kb",
    "rss_anon_ba",
    "rss_anon_ba_ci_lo",
    "rss_anon_ba_ci_hi",
    "rss_anon_a2a",
    "rss_anon_a2a_ci_lo",
    "rss_anon_a2a_ci_hi",
    "rss_anon_limit",
    "fixed_ring_a",
    "fixed_ring_a2",
    "fixed_ring_b",
    "ru_maxrss_a_kb",
    "ru_maxrss_a2_kb",
    "ru_maxrss_b_kb",
    "ru_maxrss_ba",
    "ru_maxrss_a2a",
    "wall_a_s",
    "wall_a2_s",
    "wall_b_s",
    "wall_ba",
    "wall_a2a",
    "fail_a",
    "fail_b",
    "fail_c",
    "fail_d",
    "fail_e",
    "verdict",
]

HOST_FIELDS = [
    "machine_id_sha256",
    "logical_cpus",
    "os",
    "arch",
    "kernel_release",
    "core",
    "affinity",
    "compiler",
]

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TBB_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
    "POLARS_MAX_THREADS",
)

ALLOWED_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "TMPDIR",
    "TERM",
)

TIP4P_BUDGET = 2_000_000

# Lennard-Jones sizes and Cambridge energies from examples/h2h_gmin_proof.rs
# and scripts/paper_0120/run_panel.py. Budgets are the paper list.
LJ_SPECS = (
    ("lj38", 400_000, -173.928427),
    ("lj75", 4_000_000, -397.492331),
    ("lj98", 4_000_000, -543.665361),
    ("lj102", 4_000_000, -569.363652),
    ("lj104", 4_000_000, -582.086642),
)

# Budgets and Cambridge headers from the sc-shelf-land4 gate-b harness.
# m55r6 is the basin-bound author's cell that the panel lacked.
MORSE_SPECS = (
    ("m55r6", 1_000_000, -250.286609),
    ("m55r14", 1_000_000, -220.646208),
    ("m80r6", 4_000_000, -378.333471),
    ("m80r14", 4_000_000, -340.811371),
)

BLJ_SPECS = (
    ("blj105n38", 400_000, -177.2090846748),
    ("blj105n55", 1_000_000, -283.1545313810),
    ("blj115n38", 400_000, -187.3676867453),
    ("blj115n55", 1_000_000, -289.2931853058),
)

# Sutton-Chen 12-6 from examples/h2h_sc.rs and scripts/metal_h2h/run_sc.py.
SC_SPECS = (
    ("sc126-38", 400_000, -35419.9804),
    ("sc126-55", 1_000_000, -52884.6806),
)


@dataclass(frozen=True)
class CellSpec:
    system: str
    family: str
    driver: str
    budget: int
    reference: float
    ref_text: str
    n: int = 0


def load_tip4p_refs(repo_root: Path) -> dict[int, tuple[float, str]]:
    path = repo_root / "data" / "tip4p_ccd.csv"
    out: dict[int, tuple[float, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            txt = row["energy"].strip()
            out[n] = (float(txt), txt)
    return out


def build_cells(repo_root: Path) -> list[CellSpec]:
    tip4p_refs = load_tip4p_refs(repo_root)
    cells: list[CellSpec] = []
    for system, budget, ref in LJ_SPECS:
        cells.append(CellSpec(system, "lj", "h2h_gmin_proof", budget, ref, f"{ref:.6f}"))
    for system, budget, ref in BLJ_SPECS:
        cells.append(CellSpec(system, "blj", "h2h_blj", budget, ref, f"{ref:.10f}"))
    for system, budget, ref in MORSE_SPECS:
        cells.append(CellSpec(system, "morse", "h2h_morse", budget, ref, f"{ref:.6f}"))
    for system, budget, ref in SC_SPECS:
        cells.append(CellSpec(system, "sc", "h2h_sc", budget, ref, f"{ref:.4f}"))
    missing = [n for n in range(8, 22) if n not in tip4p_refs]
    if missing:
        raise RuntimeError(f"missing TIP4P n={missing} in data/tip4p_ccd.csv")
    for n in range(8, 22):
        ref_val, ref_txt = tip4p_refs[n]
        cells.append(
            CellSpec(f"tip4p{n}", "tip4p", "h2h_seed", TIP4P_BUDGET, ref_val, ref_txt, n=n)
        )
    return cells


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return sha256_bytes(raw.encode("utf-8"))
    raise RuntimeError("cannot read a machine id")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_session(machine_id: str) -> str:
    return sha256_bytes(f"{machine_id}\n{utc_now()}".encode("utf-8"))


def float_bits_hex(value: float) -> str:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{raw:016x}"


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def is_hit(best: float, reference: float, tol: float = HIT_TOL) -> bool:
    return math.isfinite(best) and best <= reference + tol


def observation_order(seed: int) -> tuple[str, str, str, str]:
    """Rotate the labelled ABBA observations by the seed offset."""
    offset = seed - SEEDS[0]
    base = [name for name, _arm in OBSERVATIONS]
    shift = offset % 4
    rotated = base[shift:] + base[:shift]
    return (rotated[0], rotated[1], rotated[2], rotated[3])


def child_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in ALLOWED_ENV_KEYS if k in os.environ}
    for key in THREAD_ENV_KEYS:
        env[key] = "1"
    for key in os.environ:
        if key.endswith("_NUM_THREADS"):
            env[key] = "1"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    env["PYTHONHASHSEED"] = "0"
    return env


def verify_core0() -> str:
    aff = sorted(os.sched_getaffinity(0))
    if aff != [0]:
        raise RuntimeError(f"process affinity is {aff}, expected [0]")
    return "0"


def type7_quantile(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lo = int(h)
    frac = h - lo
    idx = lo - 1
    if frac == 0.0:
        return ordered[idx]
    left = ordered[idx]
    right = ordered[idx + 1]
    if math.isinf(left) or math.isinf(right):
        if math.isinf(left) and math.isinf(right):
            return left
        if math.isinf(right) and not math.isinf(left):
            return math.inf
        return left
    return left * (1.0 - frac) + right * frac


def finite_ratio(num: float, den: float) -> float:
    """Ratio of two medians. Two infinities count as 1."""
    if math.isnan(num) or math.isnan(den):
        return math.nan
    if math.isinf(num) and math.isinf(den):
        return 1.0
    if math.isinf(den) and not math.isinf(num):
        return 0.0
    if den == 0.0 or math.isinf(num):
        return math.inf
    return num / den


def rederived_hit(row: dict[str, str], reference: float) -> bool:
    if row["status"] != "ok":
        return False
    return is_hit(parse_float(row["best"]), reference, HIT_TOL)


def calls_for_test(row: dict[str, str], reference: float) -> float:
    if not rederived_hit(row, reference):
        return math.inf
    return float(row["calls_to_hit"])


def bootstrap_median_ratio_ci(
    nums: list[float],
    dens: list[float],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Paired seed resamples of the ratio of type-7 medians."""
    n = len(nums)
    if n == 0 or len(dens) != n:
        return math.nan, math.nan
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(draws):
        idxs = [rng.randrange(n) for _ in range(n)]
        num_sample = [nums[i] for i in idxs]
        den_sample = [dens[i] for i in idxs]
        replicates.append(
            finite_ratio(type7_quantile(num_sample, 0.5), type7_quantile(den_sample, 0.5))
        )
    return type7_quantile(replicates, 0.025), type7_quantile(replicates, 0.975)


def bootstrap_cluster_median_ratio_ci(
    nums: list[list[float]],
    dens: list[list[float]],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Seed-cluster bootstrap of a ratio of type-7 medians."""
    n = len(nums)
    if n == 0 or len(dens) != n:
        return math.nan, math.nan
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(draws):
        idxs = [rng.randrange(n) for _ in range(n)]
        num_sample = [value for i in idxs for value in nums[i]]
        den_sample = [value for i in idxs for value in dens[i]]
        replicates.append(
            finite_ratio(type7_quantile(num_sample, 0.5), type7_quantile(den_sample, 0.5))
        )
    return type7_quantile(replicates, 0.025), type7_quantile(replicates, 0.975)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_durable(path: Path, text: str) -> str:
    """Write text as path.tmp, fsync, rename onto path, return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = text.encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)
    got = sha256_file(path)
    if got != sha256_bytes(data):
        raise RuntimeError(f"read-back sha256 mismatch on {path.name}")
    return got


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle))
        if existing_header != fields:
            raise RuntimeError(f"{path.name} header mismatch")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({k: str(row[k]) for k in fields})
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_csv_driver_row(stdout: str, system: str, seed: int, budget: int) -> dict[str, str]:
    headers = [line for line in stdout.splitlines() if line.startswith("system,")]
    if not headers:
        raise ValueError("missing CSV header in stdout")
    header = headers[-1].split(",")
    prefix = f"{system},anneal,{seed},"
    matches = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected 1 CSV row, got {len(matches)}")
    row = dict(zip(header, matches[0].split(","), strict=True))
    if int(float(row["budget"])) != budget:
        raise ValueError(f"budget {row['budget']} != {budget}")
    wall_s = float(row["wall_s"])
    pot_s = float(row["pot_s"])
    outside_s = float(row["outside_s"]) if "outside_s" in row else max(wall_s - pot_s, 0.0)
    return {
        "best": row["best"].strip(),
        "evals": str(int(float(row["evals_to_hit"]))),
        "charged": str(int(float(row["charged"]))),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{outside_s:.6f}",
    }


def parse_h2h_seed_stdout(stdout: str, system: str, seed: int, budget: int) -> dict[str, str]:
    rows = [line for line in stdout.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise ValueError(f"expected 1 h2h_row, got {len(rows)}")
    kv = dict(tok.split("=", 1) for tok in rows[0].split()[1:])
    if kv["system"] != system or int(kv["seed"]) != seed or int(kv["budget"]) != budget:
        raise ValueError("malformed h2h_row")
    return {
        "best": kv["best"].strip(),
        "evals": str(int(kv["evals_to_hit"])),
        "charged": str(int(kv["charged"])),
        "wall_s": f"{float(kv['wall_s']):.6f}",
        "pot_s": f"{float(kv['pot_s']):.6f}",
        "outside_s": f"{float(kv['outside_s']):.6f}",
    }


def verify_driver_budgets(cells: list[CellSpec], bins: dict[str, Path], env: dict[str, str]) -> None:
    checked: set[tuple[str, str]] = set()
    for cell in cells:
        if cell.driver == "h2h_seed":
            continue
        for arm in ARMS:
            key = (arm, cell.system)
            if key in checked:
                continue
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
            checked.add(key)


def driver_command(cell: CellSpec, exe: Path, seed: int, arm: str) -> tuple[list[str], list[str]]:
    if cell.driver == "h2h_seed":
        rel_prefix = f"{cell.system}-{arm}-{seed}"
        tail = [
            "tip4p",
            str(cell.n),
            "0",
            str(cell.budget),
            str(seed),
            "recommended",
            repr(cell.reference + HIT_TOL),
            rel_prefix,
        ]
    else:
        tail = ["run", cell.system, str(seed)]
    return (
        ["taskset", "-c", "0", str(exe), *tail],
        ["taskset", "-c", "0", exe.name, *tail],
    )


def run_one(
    *,
    cell: CellSpec,
    arm: str,
    observation: str,
    seed: int,
    slot: int,
    rotation: int,
    exe: Path,
    bin_sha256: str,
    lib_tree: str,
    work_dir: Path,
    log_path: Path,
    machine_id: str,
    session: str,
    sequence: int,
    env: dict[str, str],
    timeout_s: int,
) -> dict[str, str]:
    """Run one cell, seed and arm, and return its row.

    Any non-zero exit, timeout, unparsable output or pinning fault makes
    the row inconclusive. It is still recorded and is never rerun.
    """
    run_dir = work_dir / cell.family / cell.system / f"{seed}-{observation}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd, log_cmd = driver_command(cell, exe, seed, arm)

    out_file = run_dir / "stdout.tmp"
    err_file = run_dir / "stderr.tmp"
    external_peak_rss_anon_kb = 0
    exec_confirmed = False
    cpus_allowed = ""
    taskset_p_line = ""
    taskset_mask = ""
    timed_out = False
    rc: int | None = None
    started_unix = time.time()
    t0 = time.monotonic()
    with out_file.open("w", encoding="utf-8") as out_f, err_file.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=out_f, stderr=err_f)
        status_path = Path(f"/proc/{proc.pid}/status")
        exe_path = Path(f"/proc/{proc.pid}/exe")
        cmdline_path = Path(f"/proc/{proc.pid}/cmdline")
        expected_exe = exe.resolve()
        while True:
            state = ""
            try:
                st_text = status_path.read_text(encoding="utf-8")
                for ln in st_text.splitlines():
                    if ln.startswith("State:"):
                        state = ln.split(":", 1)[1].strip()
                    elif exec_confirmed and ln.startswith("RssAnon:"):
                        val = int(ln.split()[1])
                        if val > external_peak_rss_anon_kb:
                            external_peak_rss_anon_kb = val
                    elif ln.startswith("Cpus_allowed_list:"):
                        cpus_allowed = ln.split(":", 1)[1].strip()
            except (FileNotFoundError, ProcessLookupError):
                pass
            if not exec_confirmed:
                try:
                    proc_exe = exe_path.resolve()
                    cmd0 = cmdline_path.read_bytes().split(b"\0", 1)[0]
                    exec_confirmed = proc_exe == expected_exe or Path(
                        os.fsdecode(cmd0)
                    ).resolve() == expected_exe
                except (FileNotFoundError, ProcessLookupError, OSError):
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
            time.sleep(0.005)

        tp = subprocess.run(["taskset", "-p", str(proc.pid)], capture_output=True, text=True, check=False)
        taskset_p_line = tp.stdout.strip()
        if ":" in taskset_p_line:
            taskset_mask = taskset_p_line.rsplit(":", 1)[1].strip()

        _pid_done, wstatus, rusage = os.wait4(proc.pid, 0)
        elapsed = time.monotonic() - t0
        rc = os.waitstatus_to_exitcode(wstatus)
        proc.returncode = rc
    ru_maxrss_kb = int(rusage.ru_maxrss)
    ru_utime_s = float(rusage.ru_utime)
    ru_stime_s = float(rusage.ru_stime)

    stdout = out_file.read_text(encoding="utf-8", errors="replace")
    stderr = err_file.read_text(encoding="utf-8", errors="replace")

    notes: list[str] = []
    memory_lines = [line for line in stderr.splitlines() if line.startswith("gate_memory ")]
    memory: dict[str, str] = {}
    if len(memory_lines) == 1:
        try:
            memory = dict(token.split("=", 1) for token in memory_lines[0].split()[1:])
        except ValueError:
            memory = {}
    in_process_peak_rss_anon_kb = int(memory.get("rss_anon_kb", "0"))
    vm_hwm_kb = int(memory.get("vm_hwm_kb", "0"))
    fixed_ring_bound = int(memory.get("fixed_ring_bound", "0"))
    rss_disagreement_kb = abs(in_process_peak_rss_anon_kb - external_peak_rss_anon_kb)
    if timed_out:
        notes.append("timeout")
    if rc != 0:
        notes.append(f"exit {rc}")
    if cpus_allowed != "0":
        notes.append(f"Cpus_allowed_list {cpus_allowed!r}")
    if taskset_mask != "1":
        notes.append(f"taskset mask {taskset_mask!r}")
    if not exec_confirmed:
        notes.append("exec not confirmed")
    if in_process_peak_rss_anon_kb <= 0:
        notes.append("in-process RssAnon not sampled")
    if external_peak_rss_anon_kb <= 0:
        notes.append("external RssAnon not sampled")
    if vm_hwm_kb <= 0:
        notes.append("VmHWM not sampled")

    parsed: dict[str, str] | None = None
    if not notes:
        try:
            if cell.driver == "h2h_seed":
                parsed = parse_h2h_seed_stdout(stdout, cell.system, seed, cell.budget)
            else:
                parsed = parse_csv_driver_row(stdout, cell.system, seed, cell.budget)
        except (ValueError, KeyError) as err:
            notes.append(f"parse: {err}")
            parsed = None

    log_sha = write_durable(
        log_path,
        f"cmd={' '.join(log_cmd)}\n"
        f"taskset_p={taskset_p_line}\n"
        f"Cpus_allowed_list={cpus_allowed}\n"
        f"in_process_peak_rss_anon_kb={in_process_peak_rss_anon_kb}\n"
        f"external_peak_rss_anon_kb={external_peak_rss_anon_kb}\n"
        f"vm_hwm_kb={vm_hwm_kb}\n"
        f"fixed_ring_bound={fixed_ring_bound}\n"
        f"ru_maxrss_kb={ru_maxrss_kb}\n"
        f"ru_utime_s={ru_utime_s:.6f} ru_stime_s={ru_stime_s:.6f}\n"
        f"rc={rc} timed_out={timed_out} elapsed_s={elapsed:.6f}\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
    )

    row = {
        "cell": cell.system,
        "arm": arm,
        "observation": observation,
        "seed": str(seed),
        "slot": str(slot),
        "rotation": str(rotation),
        "superseded": "0",
        "status": "ok",
        "exit_code": str(rc),
        "hit": "0",
        "evals": "",
        "calls_to_hit": "inf",
        "charged": "",
        "best": "nan",
        "best_bits": float_bits_hex(float("nan")),
        "wall_s": "",
        "pot_s": "",
        "outside_s": "",
        "elapsed_s": f"{elapsed:.6f}",
        "ru_utime_s": f"{ru_utime_s:.6f}",
        "ru_stime_s": f"{ru_stime_s:.6f}",
        "rss_anon_kb": str(in_process_peak_rss_anon_kb),
        "external_rss_anon_kb": str(external_peak_rss_anon_kb),
        "vm_hwm_kb": str(vm_hwm_kb),
        "rss_disagreement_kb": str(rss_disagreement_kb),
        "ru_maxrss_kb": str(ru_maxrss_kb),
        "fixed_ring_bound": str(fixed_ring_bound),
        "library_tree": lib_tree,
        "binary_sha256": bin_sha256,
        "log_sha256": log_sha,
        "core": "0",
        "affinity": cpus_allowed,
        "taskset_mask": taskset_mask,
        "machine_id_sha256": machine_id,
        "session": session,
        "sequence": str(sequence),
        "started_unix": f"{started_unix:.3f}",
        "note": "",
    }
    if parsed is None:
        row["status"] = "inconclusive"
        row["note"] = "; ".join(notes)
        return row

    best_val = parse_float(parsed["best"])
    hit_val = 1 if is_hit(best_val, cell.reference, HIT_TOL) else 0
    row.update(
        {
            "hit": str(hit_val),
            "evals": parsed["evals"],
            "calls_to_hit": parsed["evals"] if hit_val else "inf",
            "charged": parsed["charged"],
            "best": parsed["best"],
            "best_bits": float_bits_hex(best_val),
            "wall_s": parsed["wall_s"],
            "pot_s": parsed["pot_s"],
            "outside_s": parsed["outside_s"],
        }
    )
    return row


def fmt_num(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def median_or_nan(values: list[float]) -> float:
    return type7_quantile(values, 0.5) if values else math.nan


def score_cell(
    cell: CellSpec,
    by_observation: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, str], list[str]]:
    rows: dict[str, dict[int, dict[str, str]]] = {}
    for observation, _arm in OBSERVATIONS:
        key = observation.lower()
        got = by_observation.get(observation, [])
        seeds = sorted(int(r["seed"]) for r in got)
        if seeds != list(SEEDS):
            raise RuntimeError(
                f"{cell.system} {observation}: expected one row per seed "
                f"{SEEDS[0]}..{SEEDS[-1]}"
            )
        rows[key] = {int(r["seed"]): r for r in got}

    inconclusive = sum(1 for got in rows.values() for r in got.values() if r["status"] != "ok")
    hits = {
        key: sum(1 for r in got.values() if rederived_hit(r, cell.reference))
        for key, got in rows.items()
    }
    med_calls_a = type7_quantile(
        [calls_for_test(r, cell.reference) for r in rows["a"].values()], 0.5
    )
    med_calls_b = type7_quantile(
        [
            calls_for_test(rows[key][seed], cell.reference)
            for seed in SEEDS
            for key in ("b1", "b2")
        ],
        0.5,
    )

    traj_fields = ("status", "evals", "charged", "calls_to_hit", "best_bits")
    traj_match = {"b": 0, "a2": 0}
    traj_diffs: list[str] = []
    hit_loss_seeds: list[int] = []
    for seed in SEEDS:
        ra = rows["a"][seed]
        for key, label in (("b1", "B1"), ("b2", "B2"), ("a2", "A2")):
            other = rows[key][seed]
            diffs = [f for f in traj_fields if other[f] != ra[f]]
            if not diffs and ra["status"] == "ok":
                traj_match["b" if key.startswith("b") else "a2"] += 1
            else:
                detail = ", ".join(
                    f"{field} {other[field]} vs {ra[field]}" for field in diffs
                )
                traj_diffs.append(f"{cell.system} seed {seed} {label} vs A: {detail}")
        if rederived_hit(ra, cell.reference) and any(
            not rederived_hit(rows[key][seed], cell.reference) for key in ("b1", "b2")
        ):
            hit_loss_seeds.append(seed)

    ok_seeds = [s for s in SEEDS if all(rows[k][s]["status"] == "ok" for k in rows)]

    def value(key: str, seed: int, field: str) -> float:
        return float(rows[key][seed][field])

    def per_call(key: str, seed: int) -> float:
        row = rows[key][seed]
        return float(row["outside_s"]) / max(1, int(row["charged"]))

    opc_a = [[per_call("a", seed)] for seed in ok_seeds]
    opc_a2 = [[per_call("a2", seed)] for seed in ok_seeds]
    opc_b = [[per_call("b1", seed), per_call("b2", seed)] for seed in ok_seeds]
    flat_opc_a = [item for cluster in opc_a for item in cluster]
    flat_opc_a2 = [item for cluster in opc_a2 for item in cluster]
    flat_opc_b = [item for cluster in opc_b for item in cluster]
    med_opc_a = median_or_nan(flat_opc_a)
    med_opc_a2 = median_or_nan(flat_opc_a2)
    med_opc_b = median_or_nan(flat_opc_b)
    outside_ba = finite_ratio(med_opc_b, med_opc_a)
    outside_a2a = finite_ratio(med_opc_a2, med_opc_a)
    ba_lo, ba_hi = bootstrap_cluster_median_ratio_ci(opc_b, opc_a)
    a2a_lo, a2a_hi = bootstrap_cluster_median_ratio_ci(opc_a2, opc_a)
    outside_limit = max(OUTSIDE_RATIO_LIMIT, a2a_hi) if not math.isnan(a2a_hi) else OUTSIDE_RATIO_LIMIT

    rss_a = [[value("a", seed, "rss_anon_kb")] for seed in ok_seeds]
    rss_a2 = [[value("a2", seed, "rss_anon_kb")] for seed in ok_seeds]
    rss_b = [
        [value("b1", seed, "rss_anon_kb"), value("b2", seed, "rss_anon_kb")]
        for seed in ok_seeds
    ]
    flat_rss_a = [item for cluster in rss_a for item in cluster]
    flat_rss_a2 = [item for cluster in rss_a2 for item in cluster]
    flat_rss_b = [item for cluster in rss_b for item in cluster]
    med_rss_a = median_or_nan(flat_rss_a)
    med_rss_a2 = median_or_nan(flat_rss_a2)
    med_rss_b = median_or_nan(flat_rss_b)
    rss_ba = finite_ratio(med_rss_b, med_rss_a)
    rss_a2a = finite_ratio(med_rss_a2, med_rss_a)
    rss_ba_lo, rss_ba_hi = bootstrap_cluster_median_ratio_ci(rss_b, rss_a)
    rss_a2a_lo, rss_a2a_hi = bootstrap_cluster_median_ratio_ci(rss_a2, rss_a)
    rss_limit = (
        max(RSS_RATIO_LIMIT, rss_a2a_hi) if not math.isnan(rss_a2a_hi) else RSS_RATIO_LIMIT
    )

    def flat_field(keys: tuple[str, ...], field: str) -> list[float]:
        return [value(key, seed, field) for seed in ok_seeds for key in keys]

    med_ru_a = median_or_nan(flat_field(("a",), "ru_maxrss_kb"))
    med_ru_a2 = median_or_nan(flat_field(("a2",), "ru_maxrss_kb"))
    med_ru_b = median_or_nan(flat_field(("b1", "b2"), "ru_maxrss_kb"))
    med_wall_a = median_or_nan(flat_field(("a",), "elapsed_s"))
    med_wall_a2 = median_or_nan(flat_field(("a2",), "elapsed_s"))
    med_wall_b = median_or_nan(flat_field(("b1", "b2"), "elapsed_s"))
    fixed_a = [int(rows["a"][seed]["fixed_ring_bound"]) for seed in SEEDS]
    fixed_a2 = [int(rows["a2"][seed]["fixed_ring_bound"]) for seed in SEEDS]
    fixed_b = [
        int(rows[key][seed]["fixed_ring_bound"])
        for seed in SEEDS
        for key in ("b1", "b2")
    ]

    fail_a = traj_match["b"] < 2 * len(SEEDS) or traj_match["a2"] < len(SEEDS)
    fail_b = bool(hit_loss_seeds) or min(hits["b1"], hits["b2"]) < hits["a"]
    fail_c = not (ba_hi <= outside_limit)
    fail_d = not (rss_ba <= rss_limit)
    fail_e = any(fixed_a) or any(fixed_a2) or any(value <= 0 for value in fixed_b)
    if inconclusive:
        verdict = "INCONCLUSIVE"
    elif fail_a or fail_b or fail_c or fail_d or fail_e:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    return (
        {
            "cell": cell.system,
            "family": cell.family,
            "budget": str(cell.budget),
            "reference": cell.ref_text,
            "rows": str(sum(len(got) for got in rows.values())),
            "inconclusive": str(inconclusive),
            "traj_match_ba": str(traj_match["b"]),
            "traj_match_a2a": str(traj_match["a2"]),
            "hits_a": str(hits["a"]),
            "hits_a2": str(hits["a2"]),
            "hits_b1": str(hits["b1"]),
            "hits_b2": str(hits["b2"]),
            "hit_loss_seeds": " ".join(str(s) for s in hit_loss_seeds),
            "med_calls_a": fmt_num(med_calls_a),
            "med_calls_b": fmt_num(med_calls_b),
            "outside_a_us": fmt_num(med_opc_a * 1e6),
            "outside_a2_us": fmt_num(med_opc_a2 * 1e6),
            "outside_b_us": fmt_num(med_opc_b * 1e6),
            "outside_ba": fmt_num(outside_ba),
            "outside_ba_ci_lo": fmt_num(ba_lo),
            "outside_ba_ci_hi": fmt_num(ba_hi),
            "outside_a2a": fmt_num(outside_a2a),
            "outside_a2a_ci_lo": fmt_num(a2a_lo),
            "outside_a2a_ci_hi": fmt_num(a2a_hi),
            "outside_limit": fmt_num(outside_limit),
            "rss_anon_a_kb": fmt_num(med_rss_a),
            "rss_anon_a2_kb": fmt_num(med_rss_a2),
            "rss_anon_b_kb": fmt_num(med_rss_b),
            "rss_anon_ba": fmt_num(rss_ba),
            "rss_anon_ba_ci_lo": fmt_num(rss_ba_lo),
            "rss_anon_ba_ci_hi": fmt_num(rss_ba_hi),
            "rss_anon_a2a": fmt_num(rss_a2a),
            "rss_anon_a2a_ci_lo": fmt_num(rss_a2a_lo),
            "rss_anon_a2a_ci_hi": fmt_num(rss_a2a_hi),
            "rss_anon_limit": fmt_num(rss_limit),
            "fixed_ring_a": str(min(fixed_a)),
            "fixed_ring_a2": str(min(fixed_a2)),
            "fixed_ring_b": str(min(fixed_b)),
            "ru_maxrss_a_kb": fmt_num(med_ru_a),
            "ru_maxrss_a2_kb": fmt_num(med_ru_a2),
            "ru_maxrss_b_kb": fmt_num(med_ru_b),
            "ru_maxrss_ba": fmt_num(finite_ratio(med_ru_b, med_ru_a)),
            "ru_maxrss_a2a": fmt_num(finite_ratio(med_ru_a2, med_ru_a)),
            "wall_a_s": fmt_num(med_wall_a),
            "wall_a2_s": fmt_num(med_wall_a2),
            "wall_b_s": fmt_num(med_wall_b),
            "wall_ba": fmt_num(finite_ratio(med_wall_b, med_wall_a)),
            "wall_a2a": fmt_num(finite_ratio(med_wall_a2, med_wall_a)),
            "fail_a": "1" if fail_a else "0",
            "fail_b": "1" if fail_b else "0",
            "fail_c": "1" if fail_c else "0",
            "fail_d": "1" if fail_d else "0",
            "fail_e": "1" if fail_e else "0",
            "verdict": verdict,
        },
        traj_diffs,
    )


def write_csv_durable(path: Path, fields: list[str], rows: list[dict[str, str]]) -> str:
    lines = [",".join(fields)]
    for row in rows:
        for field in fields:
            if any(ch in row[field] for ch in ',"\n'):
                raise RuntimeError(f"{path.name}: {field} needs quoting")
        lines.append(",".join(row[field] for field in fields))
    return write_durable(path, "\n".join(lines) + "\n")


def write_hosts_csv(path: Path, machine_ids: list[str], compiler: str) -> None:
    rows = [
        {
            "machine_id_sha256": mid,
            "logical_cpus": str(os.cpu_count() or 1),
            "os": platform.system(),
            "arch": platform.machine(),
            "kernel_release": platform.release(),
            "core": "0",
            "affinity": "0",
            "compiler": compiler,
        }
        for mid in machine_ids
    ]
    write_csv_durable(path, HOST_FIELDS, rows)


def render_readme_results(
    scored: list[dict[str, str]],
    traj_diffs: list[str],
    rows: list[dict[str, str]],
    total_rows: int,
    inconclusive_rows: list[str],
) -> str:
    verdicts = {v: [r["cell"] for r in scored if r["verdict"] == v] for v in ("PASS", "FAIL", "INCONCLUSIVE")}
    if verdicts["FAIL"]:
        gate_verdict = "FAIL"
    elif verdicts["INCONCLUSIVE"]:
        gate_verdict = "INCONCLUSIVE"
    else:
        gate_verdict = "PASS"

    def cells_failing(flag: str) -> str:
        names = [r["cell"] for r in scored if r[flag] == "1"]
        return ", ".join(names) if names else "none"

    lines = [
        "## Results",
        "",
        "This section is generated from `rows.csv` by `run_gate.py summarize`.",
        "",
        f"{len(scored)} cells, twenty-four seeds, four observations, {total_rows} rows, "
        f"{len(inconclusive_rows)} inconclusive. Gate verdict: **{gate_verdict}**. "
        f"{len(verdicts['PASS'])} cells pass, {len(verdicts['FAIL'])} fail and "
        f"{len(verdicts['INCONCLUSIVE'])} are inconclusive.",
        "",
        f"- Clause (a), trajectory identity, fails on: {cells_failing('fail_a')}.",
        f"- Clause (b), hit loss, fails on: {cells_failing('fail_b')}.",
        f"- Clause (c), outside time per call, fails on: {cells_failing('fail_c')}.",
        f"- Clause (d), peak RssAnon, fails on: {cells_failing('fail_d')}.",
        f"- Clause (e), fixed-ring execution, fails on: {cells_failing('fail_e')}.",
        "",
        "`Traj` counts B observations and A2 observations that match A.",
        "Hits are A, A2, B1 and B2, re-derived from `best`. Outside is the median time outside",
        "energy calls per charged call. Its B/A ratio carries the bootstrap",
        "95% interval, and the limit is the clause (c) bound. RssAnon carries",
        "the seed-cluster bootstrap interval. VmHWM and `ru_maxrss` never gate.",
        "",
        "| Cell | Identity B, A2 | Hits A, A2, B1, B2 | Outside B/A [95% CI] | Outside A2/A [95% CI] | Outside limit | RssAnon B/A [95% CI] | RssAnon A2/A [95% CI] | RSS limit | Bound A, A2, B min | Verdict |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | --- | --- |",
    ]
    for row in scored:
        lines.append(
            f"| {row['cell']} | {row['traj_match_ba']}, {row['traj_match_a2a']} | "
            f"{row['hits_a']}, {row['hits_a2']}, {row['hits_b1']}, {row['hits_b2']} | "
            f"{row['outside_ba']} [{row['outside_ba_ci_lo']}, {row['outside_ba_ci_hi']}] | "
            f"{row['outside_a2a']} [{row['outside_a2a_ci_lo']}, {row['outside_a2a_ci_hi']}] | "
            f"{row['outside_limit']} | "
            f"{row['rss_anon_ba']} [{row['rss_anon_ba_ci_lo']}, {row['rss_anon_ba_ci_hi']}] | "
            f"{row['rss_anon_a2a']} [{row['rss_anon_a2a_ci_lo']}, {row['rss_anon_a2a_ci_hi']}] | "
            f"{row['rss_anon_limit']} | {row['fixed_ring_a']}, {row['fixed_ring_a2']}, "
            f"{row['fixed_ring_b']} | {row['verdict']} |"
        )
    lines.extend(["", "### Trajectory differences", ""])
    if traj_diffs:
        lines.extend(f"- {item}" for item in traj_diffs)
    else:
        lines.append(
            f"None. On all {len(scored)} cells and {len(SEEDS)} seeds, both B "
            "observations and A2 match A."
        )
    lines.extend(["", "### Inconclusive rows", ""])
    if inconclusive_rows:
        lines.extend(f"- {item}" for item in inconclusive_rows)
    else:
        lines.append("None. Every run exited 0 with timing and RssAnon recorded.")
    disagreements = [
        row for row in rows if int(row["rss_disagreement_kb"]) > 2 * 1024
    ]
    lines.extend(["", "### Internal and external peak disagreements", ""])
    if disagreements:
        for row in disagreements:
            lines.append(
                f"- {row['cell']} seed {row['seed']} {row['observation']}: "
                f"{float(row['rss_anon_kb']) / 1024:.1f} MiB internal, "
                f"{float(row['external_rss_anon_kb']) / 1024:.1f} MiB external."
            )
    else:
        lines.append("None over 2 MiB.")

    mode_rows = [row for row in rows if row["cell"] == "blj105n38"]

    def mode(row: dict[str, str]) -> str:
        mib = float(row["rss_anon_kb"]) / 1024
        if 26.0 <= mib < 30.0:
            return "~28"
        if 30.0 <= mib <= 34.0:
            return "~32"
        return "other"

    lines.extend(
        [
            "",
            "### blj105n38 RssAnon modes",
            "",
            "| Seed | A | B1 | B2 | A2 |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    indexed = {(row["seed"], row["observation"]): row for row in mode_rows}
    for seed in SEEDS:
        cells: list[str] = []
        for observation in ("A", "B1", "B2", "A2"):
            row = indexed[(str(seed), observation)]
            cells.append(f"{mode(row)} ({float(row['rss_anon_kb']) / 1024:.1f})")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "| Arm | ~28 MiB | ~32 MiB | Other |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm, observations in (("A", ("A",)), ("A2", ("A2",)), ("B", ("B1", "B2"))):
        labels = [
            mode(row) for row in mode_rows if row["observation"] in observations
        ]
        lines.append(
            f"| {arm} | {labels.count('~28')} | {labels.count('~32')} | "
            f"{labels.count('other')} |"
        )
    return "\n".join(lines) + "\n"


def update_readme(readme_path: Path, results_block: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    marker = "\n## Results\n"
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(f"missing '## Results' section in {readme_path.name}")
    write_durable(readme_path, text[: idx + 1] + results_block)


def summarize(
    rows_path: Path,
    summary_path: Path,
    repo_root: Path,
    hosts_path: Path | None,
    readme_path: Path | None,
    compiler: str,
) -> int:
    cells = build_cells(repo_root)
    rows = read_rows(rows_path)
    active_rows = [row for row in rows if row["superseded"] == "0"]
    keys = [(r["cell"], r["observation"], r["seed"]) for r in active_rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("rows.csv holds a repeated active cell, observation and seed")
    by_cell_observation: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in active_rows:
        by_cell_observation.setdefault((row["cell"], row["observation"]), []).append(row)
    scored: list[dict[str, str]] = []
    all_traj_diffs: list[str] = []
    for cell in cells:
        cell_score, cell_diffs = score_cell(
            cell,
            {
                observation: by_cell_observation.get((cell.system, observation), [])
                for observation, _arm in OBSERVATIONS
            },
        )
        scored.append(cell_score)
        all_traj_diffs.extend(cell_diffs)
    inconclusive_rows = [
        f"{r['cell']} seed {r['seed']} {r['observation']}: {r['note']}"
        for r in active_rows
        if r["status"] != "ok"
    ]
    write_csv_durable(summary_path, SUMMARY_FIELDS, scored)
    if hosts_path is not None:
        mids = sorted({row["machine_id_sha256"] for row in active_rows}) or [hashed_machine_id()]
        write_hosts_csv(hosts_path, mids, compiler)
    if readme_path is not None:
        update_readme(
            readme_path,
            render_readme_results(
                scored,
                all_traj_diffs,
                active_rows,
                len(active_rows),
                inconclusive_rows,
            ),
        )
    for row in scored:
        print(
            f"{row['cell']} traj {row['traj_match_ba']}/{row['traj_match_a2a']} "
            f"hits {row['hits_a']}/{row['hits_a2']}/{row['hits_b1']}/{row['hits_b2']} "
            f"outside {row['outside_ba']} [{row['outside_ba_ci_lo']},{row['outside_ba_ci_hi']}] "
            f"limit {row['outside_limit']} rss {row['rss_anon_ba']} wall {row['wall_ba']} "
            f"{row['verdict']}"
        )
    print(f"traj_diffs: {len(all_traj_diffs)} inconclusive_rows: {len(inconclusive_rows)}")
    return 0


def preflight(
    cells: list[CellSpec],
    bins: dict[str, Path],
    trees: dict[str, str],
    bin_shas: dict[tuple[str, str], str],
    work_dir: Path,
    log_dir: Path,
    machine_id: str,
    session: str,
    env: dict[str, str],
    timeout_s: int,
    journal: Path,
) -> None:
    """Run one real cell and seed per arm and driver through run_one.

    These runs are not rows. Any non-zero exit or empty timing or
    RssAnon field aborts the campaign before the first row.
    """
    by_name = {cell.system: cell for cell in cells}
    for driver in DRIVERS:
        cell = by_name[PREFLIGHT_CELLS[driver]]
        for arm in ARMS:
            row = run_one(
                cell=cell,
                arm=arm,
                observation=arm,
                seed=PREFLIGHT_SEED,
                slot=0,
                rotation=0,
                exe=bins[arm] / driver,
                bin_sha256=bin_shas[(arm, driver)],
                lib_tree=trees[arm],
                work_dir=work_dir / "preflight",
                log_path=log_dir / "preflight" / f"{cell.system}-{PREFLIGHT_SEED}-{arm}.log",
                machine_id=machine_id,
                session=session,
                sequence=0,
                env=env,
                timeout_s=timeout_s,
            )
            empty = [
                f
                for f in (
                    "wall_s",
                    "pot_s",
                    "outside_s",
                    "elapsed_s",
                    "ru_utime_s",
                    "rss_anon_kb",
                    "external_rss_anon_kb",
                    "vm_hwm_kb",
                    "ru_maxrss_kb",
                )
                if row[f] in ("", "nan") or (f == "rss_anon_kb" and int(row[f]) <= 0)
            ]
            ok = row["status"] == "ok" and row["exit_code"] == "0" and not empty
            append_jsonl(
                journal,
                {
                    "event": "preflight",
                    "utc": utc_now(),
                    "session": session,
                    "driver": driver,
                    "cell": cell.system,
                    "seed": PREFLIGHT_SEED,
                    "arm": arm,
                    "exit_code": row["exit_code"],
                    "status": row["status"],
                    "empty_fields": empty,
                    "charged": row["charged"],
                    "best_bits": row["best_bits"],
                    "elapsed_s": row["elapsed_s"],
                    "rss_anon_kb": row["rss_anon_kb"],
                    "log_sha256": row["log_sha256"],
                    "ok": ok,
                },
            )
            print(f"[{utc_now()}] preflight {driver} {cell.system} {arm} ok={ok}", flush=True)
            if not ok:
                raise SystemExit(f"preflight failed: {driver} {arm} {row['note']} empty={empty}")


def self_test() -> None:
    assert abs(type7_quantile([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) < 1e-12
    assert math.isinf(type7_quantile([1.0, 2.0] + [math.inf] * 6, 0.5))
    assert is_hit(-305.5174, -305.5183221676)
    assert not is_hit(float("nan"), -1.0)
    assert rederived_hit({"status": "ok", "best": "-305.5174", "hit": "0"}, -305.5183221676)
    assert not rederived_hit({"status": "ok", "best": "nan", "hit": "1"}, -305.5183221676)
    assert not rederived_hit({"status": "inconclusive", "best": "-400", "hit": "1"}, -305.5)
    ci_lo, ci_hi = bootstrap_median_ratio_ci([2.0] * 24, [2.0] * 24, draws=200)
    assert abs(ci_lo - 1.0) < 1e-12 and abs(ci_hi - 1.0) < 1e-12
    cluster_lo, cluster_hi = bootstrap_cluster_median_ratio_ci(
        [[2.0, 2.0]] * 24, [[2.0]] * 24, draws=200
    )
    assert abs(cluster_lo - 1.0) < 1e-12 and abs(cluster_hi - 1.0) < 1e-12
    observations = tuple(name for name, _arm in OBSERVATIONS)
    counts = {(slot, observation): 0 for slot in (1, 2, 3, 4) for observation in observations}
    for seed in SEEDS:
        order = observation_order(seed)
        assert len(order) == len(set(order)) == 4
        for slot, observation in enumerate(order, start=1):
            counts[(slot, observation)] += 1
    assert all(value == 6 for value in counts.values()), counts
    assert observation_order(2000) == ("A", "B1", "B2", "A2")
    assert observation_order(2001) == ("B1", "B2", "A2", "A")
    repo_root = Path(__file__).resolve().parents[2]
    cells = build_cells(repo_root)
    names = [cell.system for cell in cells]
    assert names[:5] == ["lj38", "lj75", "lj98", "lj102", "lj104"]
    for name in ("m55r6", "m55r14", "m80r6", "m80r14", "blj105n38", "sc126-38", "tip4p8", "tip4p21"):
        assert name in names, name
    assert len([cell for cell in cells if cell.family == "tip4p"]) == 14
    assert len(cells) == 5 + 4 + 4 + 2 + 14
    assert set(PREFLIGHT_CELLS) == set(DRIVERS)
    print("self-test ok")


def main(argv: list[str]) -> int:
    if "--quick" in argv:
        raise SystemExit("--quick is prohibited")
    if len(argv) >= 2 and argv[1] == "--self-test":
        self_test()
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    for arm in ARMS:
        run_p.add_argument(f"--{arm.lower()}-bin", type=Path, required=True)
        run_p.add_argument(f"--{arm.lower()}-tree", required=True)
    run_p.add_argument("--compiler", required=True)
    run_p.add_argument("--build-flags", required=True)
    run_p.add_argument("--rows", type=Path, required=True)
    run_p.add_argument("--journal", type=Path, required=True)
    run_p.add_argument("--work", type=Path, required=True)
    run_p.add_argument("--logs", type=Path, required=True)
    run_p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)

    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("--rows", type=Path, required=True)
    sum_p.add_argument("--summary", type=Path, required=True)
    sum_p.add_argument("--hosts", type=Path, default=None)
    sum_p.add_argument("--readme", type=Path, default=None)
    sum_p.add_argument("--compiler", required=True)

    args = parser.parse_args(argv[1:])
    repo_root = Path(__file__).resolve().parents[2]

    if args.cmd == "summarize":
        return summarize(args.rows, args.summary, repo_root, args.hosts, args.readme, args.compiler)

    if read_rows(args.rows):
        raise SystemExit("rows.csv already holds rows; the journal is append-only and never resumed")
    verify_core0()
    env = child_env()
    cells = build_cells(repo_root)
    bins = {arm: getattr(args, f"{arm.lower()}_bin").resolve() for arm in ARMS}
    trees = {arm: getattr(args, f"{arm.lower()}_tree") for arm in ARMS}
    verify_driver_budgets(cells, bins, env)

    machine_id = hashed_machine_id()
    session = new_session(machine_id)
    bin_shas = {(arm, drv): sha256_file(bins[arm] / drv) for arm in ARMS for drv in DRIVERS}
    append_jsonl(
        args.journal,
        {
            "event": "session_start",
            "utc": utc_now(),
            "session": session,
            "machine_id_sha256": machine_id,
            "compiler": args.compiler,
            "build_flags": args.build_flags,
            "arms": {
                arm: {
                    "tree": trees[arm],
                    "binaries": {drv: bin_shas[(arm, drv)] for drv in DRIVERS},
                }
                for arm in ARMS
            },
            "cells": [cell.system for cell in cells],
            "seeds": list(SEEDS),
            "core": 0,
            "thread_env": {k: env[k] for k in sorted(env) if k.endswith("_THREADS")},
            "timeout_s": args.timeout,
        },
    )

    preflight(
        cells, bins, trees, bin_shas, args.work, args.logs, machine_id, session, env, args.timeout, args.journal
    )

    total_jobs = len(cells) * len(SEEDS) * len(OBSERVATIONS)
    sequence = 0
    for cell in cells:
        for seed in SEEDS:
            rotation = (seed - SEEDS[0]) % 4
            for slot, observation in enumerate(observation_order(seed), start=1):
                arm = OBSERVATION_ARMS[observation]
                sequence += 1
                row = run_one(
                    cell=cell,
                    arm=arm,
                    observation=observation,
                    seed=seed,
                    slot=slot,
                    rotation=rotation,
                    exe=bins[arm] / cell.driver,
                    bin_sha256=bin_shas[(arm, cell.driver)],
                    lib_tree=trees[arm],
                    work_dir=args.work,
                    log_path=args.logs / cell.family / cell.system / f"{seed}-{observation}.log",
                    machine_id=machine_id,
                    session=session,
                    sequence=sequence,
                    env=env,
                    timeout_s=args.timeout,
                )
                append_csv(args.rows, ROW_FIELDS, row)
                print(
                    f"[{utc_now()}] {sequence}/{total_jobs} {cell.system} seed={seed} slot={slot} "
                    f"observation={observation} arm={arm} status={row['status']} "
                    f"hit={row['hit']} calls={row['calls_to_hit']} "
                    f"best={row['best']} elapsed={row['elapsed_s']}",
                    flush=True,
                )

    append_jsonl(
        args.journal,
        {"event": "session_end", "utc": utc_now(), "session": session, "rows": sequence},
    )
    print(f"campaign complete: {sequence}/{total_jobs} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
