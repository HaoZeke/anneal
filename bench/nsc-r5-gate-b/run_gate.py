#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial two-arm no-loss gate runner (part B) for no-size-constants r5.

Runs every Sutton-Chen cell in scripts/metal_h2h/run_sc.py::PAPER_BUDGETS
(sc126-38, sc108-38, sc96-38, sc126-55, sc108-55, sc96-55, sc126-80,
sc108-80, sc96-80), the seven Doye-Meyer BLJ clusters, and rigid TIP4P
(H2O)n for n = 8..21 at PAPER_BUDGETS with n = 12 seeds (2000-2011).

Protocol invariants enforced by this runner:
- Whole process tree pinned to CPU core 0 (verified via sched_getaffinity).
- Every thread pool forced to 1 thread; H2H_BUDGET and ANNEAL_* scrubbed.
- Balanced ABBA interleaving per (system, seed): even offset runs (A, B),
  odd offset runs (B, A).
- Separate release builds per arm; zero pre-warm, smoke, quick, or rescue runs.
- Generous 12 h per-job timeout; failed runs are recorded once as a miss.
- Records library commit/tree, harness git blob ids, binary sha256, and hashed
  machine id per row (no hostnames, VM/store paths, or tool/model names).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

SEEDS = tuple(range(2000, 2012))
JOB_TIMEOUT_S = 43200.0  # 12 hours
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
)

SC_SYSTEMS = (
    "sc126-38",
    "sc108-38",
    "sc96-38",
    "sc126-55",
    "sc108-55",
    "sc96-55",
    "sc126-80",
    "sc108-80",
    "sc96-80",
)
BLJ_SYSTEMS = (
    "blj105n38",
    "blj115n38",
    "blj130n38",
    "blj105n55",
    "blj115n55",
    "blj130n55",
    "blj130n98",
)
TIP4P_MIN_N = 8
TIP4P_MAX_N = 21

PANEL_BY_DRIVER = {
    "sc": "Sutton-Chen",
    "blj": "BLJ",
    "tip4p": "water/TIP4P rigid",
}

DRIVERS = {
    "sc": "h2h_sc",
    "blj": "nsc-r5-gate-driver",
    "tip4p": "h2h_seed",
}

ROW_FIELDS = [
    "session_id",
    "run_seq",
    "panel",
    "system",
    "driver",
    "arm",
    "seed",
    "slot",
    "arm_order",
    "budget",
    "reference",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "best_bits_hex",
    "wall_s",
    "pot_s",
    "outside_s",
    "outside_per_call_s",
    "status",
    "library",
    "library_tree",
    "harness_blobs",
    "binary_sha256",
    "trace_sha256",
    "xyz_sha256",
    "gmin_sha256",
    "start_sha256",
    "machine_id_sha256",
    "utc",
]

PANEL_FIELDS = ["panel", "system", "driver", "budget", "reference"]

SESSION_FIELDS = [
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
    "rustc_version",
    "cargo_version",
    "python_version",
    "seeds",
    "arm_order_protocol",
    "job_timeout_s",
]

BIN_FIELDS = ["arm", "binary", "sha256"]


class ArmSpec(NamedTuple):
    name: str
    bin_dir: Path
    tree_dir: Path
    library: str
    library_tree: str


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_session_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes()) if path.is_file() else ""


def float_bits_hex(value: float) -> str:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{raw:016x}"


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return sha256_bytes(raw.encode("utf-8"))
    raise RuntimeError("cannot read machine-id")


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return " ".join(line.split(":", 1)[1].split()).replace(",", " ")
    return platform.processor() or "unknown"


def first_line(cmd: list[str]) -> str:
    exe = shutil.which(cmd[0])
    if exe is None:
        return "absent"
    proc = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, check=False)
    text = (proc.stdout or proc.stderr).strip()
    lines = text.splitlines()
    return lines[0].strip() if proc.returncode == 0 and lines else "absent"


def read_tip4p_references(repo: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with (repo / "data" / "tip4p_ccd.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            if TIP4P_MIN_N <= n <= TIP4P_MAX_N:
                out[n] = float(row["energy"])
    return out


def tip4p_paper_constants(repo: Path) -> tuple[int, float]:
    """Budget and hit tolerance from the repository TIP4P paper definitions."""
    scripts_dir = repo / "scripts"
    tip4p_dir = scripts_dir / "tip4p_h2h"
    for p in (str(scripts_dir), str(tip4p_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_h2h  # noqa: PLC0415
    from target_tol import DEFAULT_HIT_TOL  # noqa: PLC0415

    return int(run_h2h.BUDGET), float(DEFAULT_HIT_TOL)


def sc_paper_constants(repo: Path) -> tuple[dict[str, int], dict[str, float]]:
    scripts_dir = repo / "scripts"
    metal_dir = scripts_dir / "metal_h2h"
    for p in (str(scripts_dir), str(metal_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_sc  # noqa: PLC0415

    return (
        {s: int(run_sc.PAPER_BUDGETS[s]) for s in SC_SYSTEMS},
        {s: float(run_sc.REFERENCES[s]) for s in SC_SYSTEMS},
    )


def git_blob_id(repo: Path, relpath: str) -> str:
    target = repo / relpath
    if not target.is_file():
        raise FileNotFoundError(f"missing file for git blob hash: {target}")
    proc = subprocess.run(
        ["git", "hash-object", str(target)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def arm_blob_id(arm: ArmSpec, relpath: str) -> str:
    if (arm.tree_dir / ".git").exists():
        proc = subprocess.run(
            ["git", "-C", str(arm.tree_dir), "rev-parse", f"HEAD:{relpath}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    return git_blob_id(arm.tree_dir, relpath)


def verify_pinned(expected_cpu: int) -> str:
    aff = sorted(os.sched_getaffinity(0))
    if aff != [expected_cpu]:
        raise SystemExit(f"cpu affinity is {aff}, expected [{expected_cpu}]")
    return ",".join(str(c) for c in aff)


def scrubbed_env(work: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(work / "home"),
        "TMPDIR": str(work / "tmp"),
        "LANG": "C",
        "LC_ALL": "C",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
        "PYTHONHASHSEED": "0",
    }
    for key in THREAD_ENV_VARS:
        env[key] = "1"
    if extra:
        for key, value in extra.items():
            if key.startswith(("H2H_", "ANNEAL_", "TWO_PHASE_", "BANK_")):
                raise ValueError(f"forbidden environment variable {key}")
            env[key] = value
    assert "H2H_BUDGET" not in env
    return env


def query_budget(binary: Path, system: str, env: dict[str, str]) -> tuple[int, float]:
    out = subprocess.run(
        [str(binary), "budget", system],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.split()
    return int(out[0]), float(out[1])


def verify_rgpot_sc_symbol(lib_path: Path) -> None:
    """Confirm librgpot.so loads and exports rgpot_sutton_chen_energy_forces without running any search."""
    if not lib_path.is_file():
        raise SystemExit(f"missing --rgpot-lib: {lib_path}")
    handle = ctypes.CDLL(str(lib_path))
    if not hasattr(handle, "rgpot_sutton_chen_energy_forces"):
        raise SystemExit(f"{lib_path} lacks symbol rgpot_sutton_chen_energy_forces")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="", encoding="utf-8") as handle:
            existing = next(csv.reader(handle))
        if existing != fields:
            raise SystemExit(f"{path} header mismatch; append-only invariant violated")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({k: str(row.get(k, "")) for k in fields})
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tip4p_size(system: str) -> int:
    return int(system.removeprefix("tip4p"))


def command_of(
    system: str,
    driver_kind: str,
    arm: ArmSpec,
    seed: int,
    prefix: Path,
    tip4p_budget: int,
    tip4p_tol: float,
    tip4p_refs: dict[int, float],
) -> list[str]:
    binary = str(arm.bin_dir / DRIVERS[driver_kind])
    if driver_kind != "tip4p":
        return [binary, "run", system, str(seed)]
    n = tip4p_size(system)
    stop = repr(tip4p_refs[n] + tip4p_tol)
    return [
        binary,
        "tip4p",
        str(n),
        "0",
        str(tip4p_budget),
        str(seed),
        "recommended",
        stop,
        str(prefix),
    ]


def parse_output(text: str, driver_kind: str, system: str, seed: int, budget: int) -> dict[str, str]:
    if driver_kind == "tip4p":
        for line in text.splitlines():
            if line.startswith("h2h_row "):
                parsed = dict(part.split("=", 1) for part in line.split()[1:])
                if (
                    parsed.get("system") != system
                    or int(parsed.get("seed", "-1")) != seed
                    or int(parsed.get("budget", "-1")) != budget
                ):
                    raise ValueError(f"bad h2h_row {line}")
                parsed["hit"] = "1" if parsed["hit"] in ("1", "true") else "0"
                return parsed
        raise ValueError("no h2h_row line")

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not line.startswith("system,code,seed,"):
            continue
        header = line.split(",")
        for candidate in lines[idx + 1 :]:
            values = candidate.split(",")
            if len(values) == len(header) and values[0] == system and values[2] == str(seed):
                row = dict(zip(header, values, strict=True))
                if int(float(row["budget"])) != budget:
                    raise ValueError(f"budget mismatch {row['budget']} != {budget}")
                row["hit"] = "1" if row["hit"].strip().lower() in ("1", "true") else "0"
                return row
        break
    raise ValueError("no result row")


def run_job(
    argv: list[str],
    cpu: str,
    env: dict[str, str],
    cwd: Path,
    log: Path,
    timeout_s: float,
) -> tuple[str, str, float]:
    pinned = [shutil.which("taskset") or "taskset", "-c", cpu, *argv]
    started = time.monotonic()
    proc = subprocess.Popen(
        pinned,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        status = "timeout"
    elapsed = time.monotonic() - started
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"# argv={' '.join(pinned)}\n# status={status}\n" + (out or ""), encoding="utf-8")
    return status, out or "", elapsed


def failed_row(base: dict[str, str], status: str, budget: int, reference: float, elapsed: float) -> dict[str, str]:
    row = dict(base)
    row.update(
        {
            "budget": str(budget),
            "reference": f"{reference:.10f}",
            "hit": "0",
            "evals_to_hit": str(budget),
            "charged": "0",
            "best": "nan",
            "best_bits_hex": float_bits_hex(float("nan")),
            "wall_s": f"{elapsed:.6f}",
            "pot_s": "0.000000",
            "outside_s": f"{elapsed:.6f}",
            "outside_per_call_s": f"{elapsed:.12f}",
            "status": status,
        }
    )
    return row


def build_panel_list(tip4p_sizes: tuple[int, ...]) -> tuple[tuple[str, str], ...]:
    return (
        *((name, "sc") for name in SC_SYSTEMS),
        *((name, "blj") for name in BLJ_SYSTEMS),
        *((f"tip4p{n}", "tip4p") for n in tip4p_sizes),
    )


def cmd_run(args: argparse.Namespace) -> int:
    expected_cpu = int(args.cpu)
    aff_text = verify_pinned(expected_cpu)
    repo = Path(args.repo).resolve()
    work = Path(args.work).resolve()
    logs_dir = Path(args.logs).resolve()
    for sub in ("home", "tmp", "scratch", "tip4p_artifacts"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    scratch = work / "scratch"

    tip4p_refs = read_tip4p_references(repo)
    tip4p_sizes = tuple(sorted(tip4p_refs))
    if tip4p_sizes != tuple(range(TIP4P_MIN_N, TIP4P_MAX_N + 1)):
        raise SystemExit(f"unexpected TIP4P sizes in data/tip4p_ccd.csv: {tip4p_sizes}")
    tip4p_budget, tip4p_tol = tip4p_paper_constants(repo)
    sc_budgets, sc_refs = sc_paper_constants(repo)

    arms = (
        ArmSpec(
            "A",
            Path(args.arm_a_bin).resolve(),
            Path(args.arm_a_tree).resolve(),
            args.arm_a_library,
            args.arm_a_tree_sha,
        ),
        ArmSpec(
            "B",
            Path(args.arm_b_bin).resolve(),
            Path(args.arm_b_tree).resolve(),
            args.arm_b_library,
            args.arm_b_tree_sha,
        ),
    )

    extra: dict[str, str] = {}
    if args.rgpot_lib:
        lib = Path(args.rgpot_lib).resolve()
        verify_rgpot_sc_symbol(lib)
        extra["RGPOT_SC_LIB"] = str(lib)
        extra["LD_LIBRARY_PATH"] = str(lib.parent)
    env = scrubbed_env(work, extra)

    all_panels = build_panel_list(tip4p_sizes)
    selected = [p for p in all_panels if not args.only or p[0] in args.only]
    if not selected:
        raise SystemExit("no systems selected")

    if any(dk == "sc" for _s, dk in selected) and "RGPOT_SC_LIB" not in extra:
        raise SystemExit("Sutton-Chen selected but --rgpot-lib was not provided")

    runner_blob = git_blob_id(repo, "bench/nsc-r5-gate-b/run_gate.py")
    summarizer_blob = git_blob_id(repo, "bench/nsc-r5-gate-b/summarize.py")
    driver_toml_blob = git_blob_id(repo, "bench/nsc-r5-gate-b/driver/Cargo.toml")
    driver_rs_blob = git_blob_id(repo, "bench/nsc-r5-gate-b/driver/src/main.rs")

    harness_blobs_by_arm_driver: dict[tuple[str, str], str] = {}
    for arm in arms:
        h2h_seed_blob = arm_blob_id(arm, "examples/h2h_seed.rs")
        h2h_sc_blob = arm_blob_id(arm, "examples/h2h_sc.rs")
        harness_blobs_by_arm_driver[(arm.name, "tip4p")] = (
            f"examples/h2h_seed.rs:{h2h_seed_blob};"
            f"bench/nsc-r5-gate-b/run_gate.py:{runner_blob};"
            f"bench/nsc-r5-gate-b/summarize.py:{summarizer_blob}"
        )
        harness_blobs_by_arm_driver[(arm.name, "sc")] = (
            f"examples/h2h_sc.rs:{h2h_sc_blob};"
            f"bench/nsc-r5-gate-b/run_gate.py:{runner_blob};"
            f"bench/nsc-r5-gate-b/summarize.py:{summarizer_blob}"
        )
        harness_blobs_by_arm_driver[(arm.name, "blj")] = (
            f"bench/nsc-r5-gate-b/driver/Cargo.toml:{driver_toml_blob};"
            f"bench/nsc-r5-gate-b/driver/src/main.rs:{driver_rs_blob};"
            f"bench/nsc-r5-gate-b/run_gate.py:{runner_blob};"
            f"bench/nsc-r5-gate-b/summarize.py:{summarizer_blob}"
        )

    panel_rows: list[dict[str, str]] = []
    for system, driver_kind in selected:
        if driver_kind == "tip4p":
            panel_rows.append(
                {
                    "panel": PANEL_BY_DRIVER[driver_kind],
                    "system": system,
                    "driver": DRIVERS[driver_kind],
                    "budget": str(tip4p_budget),
                    "reference": f"{tip4p_refs[tip4p_size(system)]:.10f}",
                }
            )
            continue
        seen = {query_budget(arm.bin_dir / DRIVERS[driver_kind], system, env) for arm in arms}
        if len(seen) != 1:
            raise SystemExit(f"{system}: Arm A and Arm B disagree on budget/reference: {seen}")
        budget, reference = seen.pop()
        if driver_kind == "sc":
            if budget != sc_budgets[system]:
                raise SystemExit(f"{system}: budget {budget} != PAPER_BUDGETS {sc_budgets[system]}")
            if abs(reference - sc_refs[system]) > 1e-6:
                raise SystemExit(f"{system}: reference {reference} != {sc_refs[system]}")
        panel_rows.append(
            {
                "panel": PANEL_BY_DRIVER[driver_kind],
                "system": system,
                "driver": DRIVERS[driver_kind],
                "budget": str(budget),
                "reference": f"{reference:.10f}",
            }
        )

    write_csv(Path(args.panels), PANEL_FIELDS, panel_rows)
    budgets = {r["system"]: int(r["budget"]) for r in panel_rows}
    references = {r["system"]: float(r["reference"]) for r in panel_rows}

    bin_rows: list[dict[str, str]] = []
    bin_sha_by_arm_name: dict[tuple[str, str], str] = {}
    seen_bins = sorted({DRIVERS[dk] for _s, dk in selected})
    for arm in arms:
        for bin_name in seen_bins:
            bin_path = arm.bin_dir / bin_name
            digest = sha256_file(bin_path)
            if not digest:
                raise SystemExit(f"missing binary: {bin_path}")
            bin_sha_by_arm_name[(arm.name, bin_name)] = digest
            bin_rows.append(
                {
                    "arm": arm.name,
                    "binary": bin_name,
                    "sha256": digest,
                }
            )
    if args.bin_sha256:
        write_csv(Path(args.bin_sha256), BIN_FIELDS, bin_rows)

    mid_sha = hashed_machine_id()
    session_id = now_session_id()
    if args.sessions:
        append_csv(
            Path(args.sessions),
            SESSION_FIELDS,
            {
                "session_id": session_id,
                "utc": utc_now(),
                "machine_id_sha256": mid_sha,
                "arch": platform.machine(),
                "kernel": platform.release(),
                "cpu_model": cpu_model(),
                "cpu_count": str(os.cpu_count() or 1),
                "affinity": aff_text,
                "arm_a_library": args.arm_a_library,
                "arm_a_tree": args.arm_a_tree_sha,
                "arm_b_library": args.arm_b_library,
                "arm_b_tree": args.arm_b_tree_sha,
                "rustc_version": first_line(["rustc", "+1.88.0", "-V"]),
                "cargo_version": first_line(["cargo", "+1.88.0", "-V"]),
                "python_version": platform.python_version(),
                "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
                "arm_order_protocol": "ABBA_balanced_per_seed",
                "job_timeout_s": f"{args.job_timeout_s:.1f}",
            },
        )

    rows_path = Path(args.rows)
    existing = read_csv(rows_path)
    done = {(r["system"], r["arm"], int(r["seed"])) for r in existing}
    run_seq = len(existing)
    total = len(selected) * len(SEEDS) * len(arms)
    print(f"session={session_id} total_jobs={total} already_done={len(done)}", flush=True)

    for system, driver_kind in selected:
        budget = budgets[system]
        ref = references[system]
        panel_name = PANEL_BY_DRIVER[driver_kind]
        driver_name = DRIVERS[driver_kind]
        for offset, seed in enumerate(SEEDS):
            order = arms if offset % 2 == 0 else (arms[1], arms[0])
            order_tag = "".join(a.name for a in order)
            for slot, arm in enumerate(order, start=1):
                if (system, arm.name, seed) in done:
                    continue
                run_seq += 1
                prefix = work / "tip4p_artifacts" / f"{system}-{arm.name}-s{seed}"
                argv = command_of(
                    system,
                    driver_kind,
                    arm,
                    seed,
                    prefix,
                    tip4p_budget,
                    tip4p_tol,
                    tip4p_refs,
                )
                log = logs_dir / f"{system}-{arm.name}-s{seed}.log"
                base = {
                    "session_id": session_id,
                    "run_seq": str(run_seq),
                    "panel": panel_name,
                    "system": system,
                    "driver": driver_name,
                    "arm": arm.name,
                    "seed": str(seed),
                    "slot": str(slot),
                    "arm_order": order_tag,
                    "library": arm.library,
                    "library_tree": arm.library_tree,
                    "harness_blobs": harness_blobs_by_arm_driver[(arm.name, driver_kind)],
                    "binary_sha256": bin_sha_by_arm_name[(arm.name, driver_name)],
                    "trace_sha256": "",
                    "xyz_sha256": "",
                    "gmin_sha256": "",
                    "start_sha256": "",
                    "machine_id_sha256": mid_sha,
                    "utc": utc_now(),
                }
                status, out, elapsed = run_job(
                    argv, args.cpu, env, scratch, log, args.job_timeout_s
                )
                if driver_kind == "tip4p":
                    base["trace_sha256"] = sha256_file(Path(f"{prefix}.trace.csv"))
                    base["xyz_sha256"] = sha256_file(Path(f"{prefix}.xyz"))
                    base["gmin_sha256"] = sha256_file(Path(f"{prefix}.gmin"))
                    base["start_sha256"] = sha256_file(Path(f"{prefix}.start.gmin"))
                if status != "ok":
                    row = failed_row(base, status, budget, ref, elapsed)
                else:
                    try:
                        parsed = parse_output(out, driver_kind, system, seed, budget)
                        wall_s = float(parsed["wall_s"])
                        pot_s = float(parsed["pot_s"])
                        outside_s = (
                            float(parsed["outside_s"])
                            if "outside_s" in parsed
                            else max(wall_s - pot_s, 0.0)
                        )
                        outside_s = max(outside_s, 0.0)
                        charged = int(float(parsed["charged"]))
                        best_val = float(parsed["best"])
                        out_per_call = outside_s / max(charged, 1)
                        row = dict(base)
                        row.update(
                            {
                                "budget": str(budget),
                                "reference": f"{ref:.10f}",
                                "hit": parsed["hit"],
                                "evals_to_hit": str(int(float(parsed["evals_to_hit"]))),
                                "charged": str(charged),
                                "best": parsed["best"],
                                "best_bits_hex": float_bits_hex(best_val),
                                "wall_s": f"{wall_s:.6f}",
                                "pot_s": f"{pot_s:.6f}",
                                "outside_s": f"{outside_s:.6f}",
                                "outside_per_call_s": f"{out_per_call:.12f}",
                                "status": "ok",
                            }
                        )
                    except (KeyError, ValueError) as err:
                        row = failed_row(base, f"parse:{err}", budget, ref, elapsed)
                append_csv(rows_path, ROW_FIELDS, row)
                done.add((system, arm.name, seed))
                print(
                    f"[{len(done)}/{total}] {system} seed={seed} slot={slot} arm={arm.name} "
                    f"status={row['status']} hit={row['hit']} calls={row['evals_to_hit']} "
                    f"wall={row['wall_s']} out_per_call={row['outside_per_call_s']}",
                    flush=True,
                )
    return 0


def main(argv: list[str]) -> int:
    if "--quick" in argv:
        raise SystemExit("--quick is prohibited by the gate protocol")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run every cell serially on core 0")
    run.add_argument("--repo", required=True)
    run.add_argument("--arm-a-bin", required=True)
    run.add_argument("--arm-b-bin", required=True)
    run.add_argument("--arm-a-tree", required=True)
    run.add_argument("--arm-b-tree", required=True)
    run.add_argument("--arm-a-library", required=True)
    run.add_argument("--arm-b-library", required=True)
    run.add_argument("--arm-a-tree-sha", required=True)
    run.add_argument("--arm-b-tree-sha", required=True)
    run.add_argument("--work", required=True)
    run.add_argument("--logs", required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--panels", required=True)
    run.add_argument("--sessions")
    run.add_argument("--bin-sha256")
    run.add_argument("--rgpot-lib")
    run.add_argument("--cpu", default="0")
    run.add_argument("--job-timeout-s", type=float, default=JOB_TIMEOUT_S)
    run.add_argument("--only", nargs="*")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
