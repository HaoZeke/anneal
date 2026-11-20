#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session gate for the no-size-constants tree.

Stock examples are the unreached guards. h2h_seed recommended is the
reached set. The pass rule is the one in README.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import os
import random
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

A_TREE = "cff796c507a8fa1d0bfb28888c3570b61988055e"
B_TREE = "113a3420d7770508519da9b5257d519d3b0fb829"
SEEDS = list(range(2000, 2024))
HIT_TOL = 0.001
JOB_TIMEOUT_S = 7200
BOOT_SEED = 20261120
BOOT_REPS = 10000
ALPHA = 0.05
OUTSIDE_CAP = 1.10
ROW_FIELDS = [
    "cell",
    "arm",
    "seed",
    "slot",
    "core",
    "affinity",
    "hit",
    "calls",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "best",
    "best_bits",
    "budget",
    "binary_sha256",
    "revision",
    "machine_id_sha256",
    "session",
    "sequence",
    "rss_peak_kb",
    "exit_code",
]
ARMS = ("A", "Ap", "B")
THREAD_COUNTS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "JULIA_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
)


@dataclass(frozen=True)
class Cell:
    name: str
    role: str
    unreached: int
    driver: str
    system: str
    budget: int
    reference: float
    kind: str
    n: int
    rho: float


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hexamer_constants(root: Path) -> tuple[int, float]:
    text = (root / "bench/gate-pr168-no-size-constants-r5/guard_trace.rs").read_text()
    if "let budget = 500_000usize;" not in text or "let reference = -197.7805260034;" not in text:
        raise SystemExit("hexamer helper does not carry the expected literals")
    return 500_000, -197.7805260034


def collect_tip4p(root: Path) -> tuple[list[Cell], list[tuple[str, str]]]:
    folder = root / "scripts" / "tip4p_h2h"
    radius = load_module(folder / "run_radius.py", "tip4p_radius")
    campaign = load_module(folder / "run_campaign.py", "tip4p_campaign")
    h2h = load_module(folder / "run_h2h.py", "tip4p_h2h_mod")
    water = load_module(folder / "run_water_gmin.py", "tip4p_water")
    refs: dict[int, float] = {}
    budgets: dict[int, int] = {}

    def add_ref(n: int, ref: float) -> None:
        ref = float(ref)
        if n in refs and abs(refs[n] - ref) > 1e-9:
            raise SystemExit(f"tip4p{n} references disagree")
        refs[n] = ref

    def add_budget(n: int, budget: int) -> None:
        budget = int(budget)
        if n in budgets and budgets[n] != budget:
            raise SystemExit(f"tip4p{n} budgets disagree")
        budgets[n] = budget

    for n, budget in radius.PAPER_BUDGETS.items():
        add_budget(int(n), budget)
        add_ref(int(n), radius.REFERENCES[n])
    for n, budget in campaign.PAPER_BUDGETS.items():
        add_budget(int(n), budget)
        add_ref(int(n), campaign.REFERENCES[n])
    for n, ref in h2h.CAMBRIDGE.items():
        add_budget(int(n), h2h.BUDGET)
        add_ref(int(n), ref)
    for n, pair in water.CAMBRIDGE.items():
        add_budget(int(n), h2h.BUDGET)
        add_ref(int(n), pair[0])
    missing = []
    for n in h2h.SIZES:
        if int(n) not in refs:
            missing.append(
                (
                    f"tip4p{n}",
                    "listed without a published reference; the hit is the session minimum",
                )
            )
    cells = []
    for n in sorted(refs):
        cells.append(
            Cell(
                name=f"tip4p{n}",
                role="guard",
                unreached=0,
                driver="h2h_seed",
                system=f"tip4p{n}",
                budget=budgets[n],
                reference=refs[n],
                kind="tip4p",
                n=n,
                rho=0.0,
            )
        )
    return cells, missing


def collect_cells(root: Path) -> tuple[list[Cell], list[tuple[str, str]]]:
    gmin = load_module(root / "scripts" / "gmin_panel" / "run_panel.py", "gmin_panel_mod")
    morse = load_module(root / "scripts" / "morse_gmin" / "run_panel.py", "morse_gmin_mod")
    paper = load_module(root / "scripts" / "paper_0120" / "run_panel.py", "paper_0120_mod")
    cells: list[Cell] = []
    for name, kind, n, reference, _rho, driver, binary in gmin.SYSTEMS:
        if kind not in ("lj", "gupta"):
            continue
        budget = int(gmin.PAPER_BUDGETS[name])
        # The panel still names the historical lj103 patch binary. The
        # example at this tree already has that system.
        stock = "h2h_gmin_proof" if binary == "h2h_gmin_proof_lj103" else binary
        cells.append(
            Cell(
                name=name,
                role="guard",
                unreached=1,
                driver=stock,
                system=name,
                budget=budget,
                reference=float(reference),
                kind="",
                n=int(n),
                rho=0.0,
            )
        )
    for name, n, rho, budget, reference in morse.SYSTEMS:
        cells.append(
            Cell(
                name=name,
                role="guard",
                unreached=1,
                driver="h2h_morse",
                system=name,
                budget=int(budget),
                reference=float(reference),
                kind="",
                n=int(n),
                rho=float(rho),
            )
        )
    budget, reference = hexamer_constants(root)
    cells.append(
        Cell(
            name="hexamer",
            role="guard",
            unreached=0,
            driver="h2h_seed",
            system="hexamer",
            budget=budget,
            reference=reference,
            kind="tip4p",
            n=6,
            rho=0.0,
        )
    )
    tip4p, missing = collect_tip4p(root)
    cells.extend(tip4p)
    seed_plan = (
        ("lj75-seed", "lj75", "lj", 75, 0.0),
        ("lj103-seed", "lj103", "lj", 103, 0.0),
        ("m38r6-seed", "m38r6", "morse", 38, 6.0),
        ("m55r14-seed", "m55r14", "morse", 55, 14.0),
    )
    lookup = {cell.name: cell for cell in cells}
    if ("lj75", "h2h_seed", "h2h_seed", ()) not in paper.RUNS:
        raise SystemExit("paper panel has no lj75 h2h_seed run")
    if int(paper.PAPER_BUDGETS["lj75"]) != lookup["lj75"].budget:
        raise SystemExit("lj75 budgets disagree")
    if abs(float(paper.LJ75_REFERENCE) - lookup["lj75"].reference) > 1e-9:
        raise SystemExit("lj75 references disagree")
    for name, source, kind, n, rho in seed_plan:
        base = lookup[source]
        cells.append(
            Cell(
                name=name,
                role="reached",
                unreached=0,
                driver="h2h_seed",
                system=source,
                budget=base.budget,
                reference=base.reference,
                kind=kind,
                n=n,
                rho=rho,
            )
        )
    missing.append(
        (
            "au55-seed",
            "h2h_seed accepts lj, morse or tip4p and has no gold potential",
        )
    )
    return cells, missing


def arm_order(seed: int) -> list[str]:
    order = ["A", "B", "Ap"]
    if (seed - 2000) % 2 == 1:
        order = list(reversed(order))
    return order


def build_schedule(cells: list[Cell]) -> list[dict]:
    jobs = []
    sequence = 0
    for seed in SEEDS:
        order = arm_order(seed)
        for cell in cells:
            for slot, arm in enumerate(order):
                jobs.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "arm": arm,
                        "slot": slot,
                        "sequence": sequence,
                    }
                )
                sequence += 1
    return jobs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_bytes().strip()
            if raw:
                return hashlib.sha256(raw).hexdigest()
    raise SystemExit("machine id is unavailable")


def tree_of(src: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD^{tree}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit("tree check failed")
    return proc.stdout.strip()


def available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise SystemExit("available memory is unknown")


def lane_cap() -> int:
    kib = available_kib()
    if kib < 4 * 1024 * 1024:
        return 1
    if kib < 8 * 1024 * 1024:
        return 2
    return 3


def thread_env() -> dict[str, str]:
    kept = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "LC_CTYPE")
    env = {key: os.environ[key] for key in kept if key in os.environ}
    for key in THREAD_COUNTS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def bits_of(token: str) -> str:
    try:
        value = float(token)
    except ValueError:
        return ""
    if not math.isfinite(value):
        return ""
    raw = struct.unpack(">Q", struct.pack(">d", value))[0]
    return f"{raw:016x}"


def as_hit(token: str) -> int:
    return int(token.strip().lower() in {"1", "true"})


def parse_stock(text: str, system: str, seed: int, budget: int) -> dict[str, str] | None:
    header = ""
    data = ""
    for line in text.splitlines():
        if line.startswith("system,") and not header:
            header = line
        elif line.startswith(f"{system},anneal,{seed},"):
            data = line
    if not header or not data:
        return None
    row = dict(zip(header.split(","), data.split(","), strict=False))
    try:
        if int(row["seed"]) != seed or int(row["budget"]) != budget:
            return None
        wall = row["wall_s"]
        pot = row["pot_s"]
        if row.get("outside_s"):
            outside = row["outside_s"]
        else:
            outside = f"{max(float(wall) - float(pot), 0.0):.6f}"
    except (KeyError, ValueError):
        return None
    return {
        "hit": str(as_hit(row["hit"])),
        "calls": str(int(row["evals_to_hit"])),
        "charged": str(int(row["charged"])),
        "wall_s": wall,
        "pot_s": pot,
        "outside_s": outside,
        "best": row["best"],
    }


def parse_seed(text: str, seed: int, budget: int) -> dict[str, str] | None:
    rows = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        return None
    fields = {}
    for part in rows[0].split()[1:]:
        if "=" not in part:
            return None
        key, value = part.split("=", 1)
        fields[key] = value
    try:
        if int(fields["seed"]) != seed or int(fields["budget"]) != budget:
            return None
    except (KeyError, ValueError):
        return None
    return {
        "hit": str(as_hit(fields.get("hit", "0"))),
        "calls": str(int(fields["evals_to_hit"])),
        "charged": str(int(fields["charged"])),
        "wall_s": fields["wall_s"],
        "pot_s": fields["pot_s"],
        "outside_s": fields["outside_s"],
        "best": fields["best"],
    }


def vmhwm(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return None


def affinity_of(pid: int) -> str:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def command_for(cell: Cell, seed: int, binary: Path, core: int, prefix: Path) -> list[str]:
    taskset = shutil.which("taskset")
    if taskset is None:
        raise SystemExit("taskset is unavailable")
    if cell.driver == "h2h_seed":
        rho = "0" if cell.kind != "morse" else str(int(cell.rho) if float(cell.rho).is_integer() else cell.rho)
        stop = repr(cell.reference + HIT_TOL)
        args = [
            cell.kind,
            str(cell.n),
            rho,
            str(cell.budget),
            str(seed),
            "recommended",
            stop,
            str(prefix),
        ]
    else:
        args = ["run", cell.system, str(seed)]
    return [taskset, "-c", str(core), str(binary), *args]


class Running:
    def __init__(self, proc: subprocess.Popen, started: float, out_path: Path, err_path: Path, directory: tempfile.TemporaryDirectory):
        self.proc = proc
        self.started = started
        self.out_path = out_path
        self.err_path = err_path
        self.directory = directory
        self.peak = 0
        self.affinity = ""
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self) -> None:
        while not self.stop.is_set():
            value = vmhwm(self.proc.pid)
            if value is not None:
                self.peak = max(self.peak, value)
            if not self.affinity:
                self.affinity = affinity_of(self.proc.pid)
            self.stop.wait(0.02)
        value = vmhwm(self.proc.pid)
        if value is not None:
            self.peak = max(self.peak, value)

    def finish(self, timed_out: bool) -> tuple[str, str, int, int, str]:
        if timed_out:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=10)
        self.stop.set()
        self.thread.join(timeout=2)
        stdout = self.out_path.read_text(errors="replace") if self.out_path.is_file() else ""
        stderr = self.err_path.read_text(errors="replace") if self.err_path.is_file() else ""
        code = self.proc.returncode
        if timed_out:
            exit_code = 124
        elif code is None:
            exit_code = 1
        elif code < 0:
            exit_code = 128 + (-code)
        else:
            exit_code = code
        self.directory.cleanup()
        return stdout, stderr, exit_code, self.peak, self.affinity


def launch(job: dict, core: int, bins: dict[str, Path], hashes: dict[tuple[str, str], str]) -> Running:
    cell: Cell = job["cell"]
    directory = tempfile.TemporaryDirectory(prefix="gate-")
    folder = Path(directory.name)
    out_path = folder / "out.txt"
    err_path = folder / "err.txt"
    binary = bins[job["arm"]] / cell.driver
    cmd = command_for(cell, job["seed"], binary, core, folder / "run")
    out_handle = out_path.open("w")
    err_handle = err_path.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=folder,
        env=thread_env(),
        stdout=out_handle,
        stderr=err_handle,
        start_new_session=True,
    )
    out_handle.close()
    err_handle.close()
    running = Running(proc, time.monotonic(), out_path, err_path, directory)
    running.binary_sha = hashes[(job["arm"], cell.driver)]
    return running


def blank_row(job: dict, core: str, affinity: str, session: str, machine: str, binary_sha: str, revision: str, rss: int, exit_code: int) -> dict[str, str]:
    cell: Cell = job["cell"]
    return {
        "cell": cell.name,
        "arm": job["arm"],
        "seed": str(job["seed"]),
        "slot": str(job["slot"]),
        "core": core,
        "affinity": affinity,
        "hit": "0",
        "calls": "0",
        "charged": "0",
        "wall_s": "",
        "pot_s": "",
        "outside_s": "",
        "best": "",
        "best_bits": "",
        "budget": str(cell.budget),
        "binary_sha256": binary_sha,
        "revision": revision,
        "machine_id_sha256": machine,
        "session": session,
        "sequence": str(job["sequence"]),
        "rss_peak_kb": str(rss),
        "exit_code": str(exit_code),
    }


def row_from_parse(job: dict, parsed: dict[str, str], core: str, affinity: str, session: str, machine: str, binary_sha: str, revision: str, rss: int, exit_code: int) -> dict[str, str]:
    row = blank_row(job, core, affinity, session, machine, binary_sha, revision, rss, exit_code)
    row.update(parsed)
    row["best_bits"] = bits_of(parsed["best"])
    if exit_code != 0:
        row["hit"] = "0"
    return row


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def revisions_for(args) -> dict[str, str]:
    found = {
        "A": tree_of(args.src_a),
        "Ap": tree_of(args.src_ap),
        "B": tree_of(args.src_b),
    }
    if found["A"] != A_TREE or found["Ap"] != A_TREE or found["B"] != B_TREE:
        raise SystemExit(
            f"tree check failed A={found['A']} Ap={found['Ap']} B={found['B']}"
        )
    return found


def hashes_for(bins: dict[str, Path], cells: list[Cell]) -> dict[tuple[str, str], str]:
    found = {}
    drivers = sorted({cell.driver for cell in cells})
    for arm, folder in bins.items():
        for driver in drivers:
            path = folder / driver
            if not path.is_file():
                raise SystemExit(f"missing binary {driver} for {arm}")
            found[(arm, driver)] = sha256_file(path)
    return found


def check_budgets(bins: dict[str, Path], cells: list[Cell]) -> list[tuple[str, str]]:
    env = thread_env()
    missing = []
    binary = bins["A"]
    for cell in cells:
        if cell.driver == "h2h_seed":
            continue
        proc = subprocess.run(
            [str(binary / cell.driver), "budget", cell.system],
            cwd=Path("/"),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode != 0 or len(lines) < 2:
            missing.append((cell.name, "budget command failed on the base build"))
            continue
        got_budget = int(lines[0])
        got_ref = float(lines[1])
        if got_budget != cell.budget or abs(got_ref - cell.reference) > 1e-6:
            missing.append(
                (
                    cell.name,
                    f"example budget {got_budget} reference {got_ref} does not match the panel",
                )
            )
    return missing


def pin_probe() -> None:
    taskset = shutil.which("taskset")
    if taskset is None:
        raise SystemExit("taskset is unavailable")
    for core in (0, 1, 2):
        proc = subprocess.run(
            [taskset, "-c", str(core), "cat", "/proc/self/status"],
            check=False,
            capture_output=True,
            text=True,
        )
        affinity = ""
        for line in proc.stdout.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                affinity = line.split(":", 1)[1].strip()
        if affinity != str(core):
            raise SystemExit(f"pin probe for core {core} saw {affinity or 'nothing'}")


def require_toolchain() -> None:
    proc = subprocess.run(
        ["rustc", "+1.88.0", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or proc.stderr).strip()
    if proc.returncode != 0 or "1.88.0" not in text.split()[1:2] and "1.88.0" not in text:
        raise SystemExit("toolchain 1.88.0 is not available")


def write_session(path: Path, session: str, machine: str, hashes: dict[tuple[str, str], str], cap: int) -> None:
    fresh = not path.is_file()
    binaries = " ".join(f"{arm}:{driver}:{digest}" for (arm, driver), digest in sorted(hashes.items()))
    row = {
        "session": session,
        "machine_id_sha256": machine,
        "a_tree": A_TREE,
        "ap_tree": A_TREE,
        "b_tree": B_TREE,
        "seeds": "2000-2023",
        "n": "24",
        "lane_cap": str(cap),
        "cores": "0,1,2",
        "timeout_s": str(JOB_TIMEOUT_S),
        "mem_available_kib": str(available_kib()),
        "binaries": binaries,
    }
    fields = list(row)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_campaign(args, cells: list[Cell], jobs: list[dict]) -> list[tuple[str, str]]:
    require_toolchain()
    pin_probe()
    revisions = revisions_for(args)
    bins = {"A": args.bin_a, "Ap": args.bin_ap, "B": args.bin_b}
    hashes = hashes_for(bins, cells)
    budget_missing = check_budgets(bins, cells)
    blocked = {name for name, _reason in budget_missing}
    if blocked:
        for name, reason in budget_missing:
            print(f"MISSING {name} {reason}", flush=True)
    machine = machine_id()
    rows_path = args.out / "rows.csv"
    done = {(row["cell"], row["arm"], int(row["seed"])) for row in read_rows(rows_path)}
    existing = read_rows(rows_path)
    session = existing[0]["session"] if existing else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not existing:
        write_session(args.out / "sessions.csv", session, machine, hashes, lane_cap())
    pending = [
        job
        for job in jobs
        if job["cell"].name not in blocked
        and (job["cell"].name, job["arm"], job["seed"]) not in done
    ]
    ready = [job for job in pending if job["slot"] == 0 or (job["cell"].name, arm_order(job["seed"])[job["slot"] - 1], job["seed"]) in done]
    waiting = [job for job in pending if job not in ready]
    free = [0, 1, 2]
    running: dict[int, tuple[dict, Running]] = {}
    total = len(jobs)
    finished = len(done)
    print(f"START session {session} rows {finished} pending {len(pending)}", flush=True)
    while ready or waiting or running:
        cap = min(3, lane_cap())
        busy = set(running)
        if set(free) | busy != {0, 1, 2}:
            free = sorted({0, 1, 2} - busy)
        while ready and len(running) < cap and free:
            core = free.pop(0)
            if core >= cap:
                free.append(core)
                break
            job = ready.pop(0)
            state = launch(job, core, bins, hashes)
            running[core] = (job, state)
        if not running and not ready:
            for job in waiting:
                cell: Cell = job["cell"]
                row = blank_row(
                    job,
                    "",
                    "",
                    session,
                    machine,
                    hashes[(job["arm"], cell.driver)],
                    revisions[job["arm"]],
                    0,
                    1,
                )
                append_row(rows_path, row)
                print(f"ROW dropped {cell.name} {job['arm']} {job['seed']}", flush=True)
            waiting.clear()
            break
        time.sleep(0.05)
        for core, (job, state) in list(running.items()):
            timed_out = time.monotonic() - state.started > JOB_TIMEOUT_S
            if state.proc.poll() is None and not timed_out:
                continue
            cell: Cell = job["cell"]
            stdout, _stderr, exit_code, peak, affinity = state.finish(timed_out)
            if cell.driver == "h2h_seed":
                parsed = parse_seed(stdout, job["seed"], cell.budget)
            else:
                parsed = parse_stock(stdout, cell.system, job["seed"], cell.budget)
            if parsed is None:
                exit_code = exit_code or 1
                row = blank_row(
                    job,
                    str(core),
                    affinity,
                    session,
                    machine,
                    hashes[(job["arm"], cell.driver)],
                    revisions[job["arm"]],
                    peak,
                    exit_code,
                )
            else:
                row = row_from_parse(
                    job,
                    parsed,
                    str(core),
                    affinity,
                    session,
                    machine,
                    hashes[(job["arm"], cell.driver)],
                    revisions[job["arm"]],
                    peak,
                    exit_code,
                )
            append_row(rows_path, row)
            done.add((cell.name, job["arm"], job["seed"]))
            finished += 1
            print(
                f"ROW {finished}/{total} {cell.name} {job['arm']} {job['seed']} exit {exit_code} rss {peak} hit {row['hit']}",
                flush=True,
            )
            del running[core]
            free.append(core)
            free.sort()
            nxt = next(
                (
                    item
                    for item in waiting
                    if item["cell"].name == cell.name
                    and item["seed"] == job["seed"]
                    and item["slot"] == job["slot"] + 1
                ),
                None,
            )
            if nxt is not None:
                waiting.remove(nxt)
                ready.append(nxt)
    return budget_missing


def ratio(num: float, den: float) -> float:
    if den == 0:
        return math.inf if num > 0 else 1.0
    return num / den


def median(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(statistics.median(values))


def mcnemar_p(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    observed = math.comb(n, a_only)
    total = sum(math.comb(n, k) for k in range(n + 1) if math.comb(n, k) <= observed)
    return min(1.0, total / 2**n)


def wilcoxon_p(left: list[float], right: list[float]) -> float:
    diffs = [b - a for a, b in zip(left, right, strict=True)]
    nonzero = [diff for diff in diffs if diff != 0.0]
    if not nonzero:
        return 1.0
    ordered = sorted(abs(diff) for diff in nonzero)
    ranks: dict[float, float] = {}
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[index]:
            end += 1
        ranks[ordered[index]] = (index + 1 + end + 1) / 2.0
        index = end + 1
    items = [int(round(ranks[abs(diff)] * 2)) for diff in nonzero]
    positive = sum(int(round(ranks[abs(diff)] * 2)) for diff in nonzero if diff > 0)
    total = sum(items)
    ways = [0] * (total + 1)
    ways[0] = 1
    for item in items:
        for score in range(total, item - 1, -1):
            ways[score] += ways[score - item]
    denom = sum(ways)
    low = sum(ways[: positive + 1])
    high = sum(ways[positive:])
    return min(1.0, 2 * min(low, high) / denom)


def bootstrap(pairs: list[tuple[float, float]], rng: random.Random) -> tuple[float, float, float]:
    if not pairs:
        return math.nan, math.nan, math.nan
    point = ratio(median([b for _a, b in pairs]), median([a for a, _b in pairs]))
    draws = []
    count = len(pairs)
    for _ in range(BOOT_REPS):
        sample = [pairs[rng.randrange(count)] for _ in range(count)]
        draws.append(ratio(median([b for _a, b in sample]), median([a for a, _b in sample])))
    draws.sort()
    return point, draws[int(9999 * 0.025)], draws[int(9999 * 0.975)]


def fmt(value: float, digits: int = 6) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "nan"
    if digits == 0:
        return f"{value:.0f}"
    return f"{value:.{digits}g}"


def group(rows: list[dict[str, str]], cell: str, arm: str) -> dict[int, dict[str, str]]:
    found = {}
    for row in rows:
        if row["cell"] == cell and row["arm"] == arm:
            found[int(row["seed"])] = row
    return found


def completed(found: dict[int, dict[str, str]]) -> dict[int, dict[str, str]]:
    return {
        seed: row
        for seed, row in found.items()
        if int(row["exit_code"]) == 0 and row["best"] != ""
    }


def hit_of(row: dict[str, str]) -> int:
    return int(row["hit"])


def charged_rank(row: dict[str, str], budget: int) -> float:
    if hit_of(row) == 0:
        return float(budget) + 1.0
    return float(row["charged"])


def per_call(row: dict[str, str]) -> float | None:
    charged = float(row["charged"])
    if charged <= 0 or row["outside_s"] == "":
        return None
    return float(row["outside_s"]) / charged


def summarize(args, cells: list[Cell], missing: list[tuple[str, str]]) -> int:
    rows = read_rows(args.out / "rows.csv")
    hit_diffs = []
    spreads = []
    prepared = []
    for cell in cells:
        arms = {arm: group(rows, cell.name, arm) for arm in ARMS}
        if any(len(arms[arm]) != len(SEEDS) for arm in ARMS):
            prepared.append((cell, arms, None))
            continue
        prepared.append((cell, arms, completed(arms["A"])))
        hits_a = sum(hit_of(arms["A"][seed]) for seed in SEEDS)
        hits_ap = sum(hit_of(arms["Ap"][seed]) for seed in SEEDS)
        hit_diffs.append(abs(hits_ap - hits_a))
        done_a = completed(arms["A"])
        done_ap = completed(arms["Ap"])
        if done_a and done_ap:
            charged_a = median([float(done_a[seed]["charged"]) for seed in done_a])
            charged_ap = median([float(done_ap[seed]["charged"]) for seed in done_ap])
            if math.isfinite(charged_a) and math.isfinite(charged_ap):
                value = ratio(charged_ap, charged_a)
                spreads.append(max(value, 1.0 / value) if math.isfinite(value) and value > 0 else math.inf)
    hit_band = max(hit_diffs) if hit_diffs else 0
    charged_band = max(spreads) if spreads else math.inf
    rng = random.Random(BOOT_SEED)
    summary_rows = []
    band_rows = []
    failing = []
    for cell, arms, _done in prepared:
        failures = []
        if any(len(arms[arm]) != len(SEEDS) for arm in ARMS):
            failures.append("incomplete")
        else:
            for arm in ARMS:
                for seed in SEEDS:
                    row = arms[arm][seed]
                    if int(row["exit_code"]) != 0 or row["best"] == "":
                        failures.append(f"{arm}:{seed}:exit{row['exit_code']}")
                    elif row["affinity"] != row["core"] or row["core"] not in {"0", "1", "2"}:
                        failures.append(f"{arm}:{seed}:affinity")
        ok = {arm: completed(arms[arm]) for arm in ARMS} if not any(len(arms[arm]) != len(SEEDS) for arm in ARMS) else {arm: {} for arm in ARMS}
        both = a_only = b_only = neither = 0
        hits = {arm: 0 for arm in ARMS}
        strings = {arm: "" for arm in ARMS}
        if all(len(arms[arm]) == len(SEEDS) for arm in ARMS):
            for seed in SEEDS:
                ha = hit_of(arms["A"][seed])
                hb = hit_of(arms["B"][seed])
                hits["A"] += ha
                hits["Ap"] += hit_of(arms["Ap"][seed])
                hits["B"] += hb
                strings["A"] += str(ha)
                strings["B"] += str(hb)
                if ha and hb:
                    both += 1
                elif ha:
                    a_only += 1
                elif hb:
                    b_only += 1
                else:
                    neither += 1
        p_hits = mcnemar_p(a_only, b_only) if strings["A"] else math.nan
        worse_hits = bool(strings["A"]) and a_only > b_only and p_hits < ALPHA
        deficit = hits["A"] - hits["B"]
        fail_hits = worse_hits or deficit > hit_band
        def series(arm: str, key: str) -> list[float]:
            return [float(ok[arm][seed][key]) for seed in SEEDS if seed in ok[arm] and ok[arm][seed][key] != ""]

        def pairs(left: str, right: str, key: str) -> list[tuple[float, float]]:
            found = []
            for seed in SEEDS:
                if seed in ok[left] and seed in ok[right]:
                    if ok[left][seed][key] == "" or ok[right][seed][key] == "":
                        continue
                    found.append((float(ok[left][seed][key]), float(ok[right][seed][key])))
            return found

        def outside_pairs(left: str, right: str) -> list[tuple[float, float]]:
            found = []
            for seed in SEEDS:
                if seed not in ok[left] or seed not in ok[right]:
                    continue
                a = per_call(ok[left][seed])
                b = per_call(ok[right][seed])
                if a is None or b is None:
                    continue
                found.append((a, b))
            return found

        charged_a = median(series("A", "charged"))
        charged_ap = median(series("Ap", "charged"))
        charged_b = median(series("B", "charged"))
        charged_ba = ratio(charged_b, charged_a) if math.isfinite(charged_a) and math.isfinite(charged_b) else math.nan
        charged_apa = ratio(charged_ap, charged_a) if math.isfinite(charged_a) and math.isfinite(charged_ap) else math.nan
        hits_improve = hits["B"] > hits["A"]
        fail_calls = (not math.isfinite(charged_ba)) or (charged_ba > charged_band and not hits_improve)
        call_ba = bootstrap(pairs("A", "B", "calls"), rng)
        call_apa = bootstrap(pairs("A", "Ap", "calls"), rng)
        wall_ba = bootstrap(pairs("A", "B", "wall_s"), rng)
        wall_apa = bootstrap(pairs("A", "Ap", "wall_s"), rng)
        out_ba = bootstrap(outside_pairs("A", "B"), rng)
        out_apa = bootstrap(outside_pairs("A", "Ap"), rng)
        fail_outside = (not math.isfinite(out_ba[0])) or out_ba[0] > OUTSIDE_CAP
        rank_a = []
        rank_b = []
        if all(len(arms[arm]) == len(SEEDS) for arm in ("A", "B")):
            for seed in SEEDS:
                rank_a.append(charged_rank(arms["A"][seed], cell.budget))
                rank_b.append(charged_rank(arms["B"][seed], cell.budget))
        w_p = wilcoxon_p(rank_a, rank_b) if rank_a else math.nan
        bits = 0
        if all(len(arms[arm]) == len(SEEDS) for arm in ("A", "B")):
            for seed in SEEDS:
                a = arms["A"][seed]
                b = arms["B"][seed]
                if a["best"] and b["best"] and a["best"] == b["best"] and a["calls"] == b["calls"] and a["charged"] == b["charged"]:
                    bits += 1
        fail_rows = bool(failures)
        verdict = "FAIL" if fail_rows or fail_hits or fail_calls or fail_outside else "PASS"
        if verdict == "FAIL":
            failing.append(cell.name)
        band_rows.append(
            {
                "cell": cell.name,
                "hit_abs_diff": str(abs(hits["Ap"] - hits["A"])) if strings["A"] else "",
                "charged_ratio": fmt(charged_apa),
                "charged_spread": fmt(max(charged_apa, 1.0 / charged_apa) if math.isfinite(charged_apa) and charged_apa > 0 else math.nan),
                "outside_ratio": fmt(out_apa[0]),
                "wall_ratio": fmt(wall_apa[0]),
                "calls_ratio": fmt(call_apa[0]),
            }
        )
        summary_rows.append(
            {
                "cell": cell.name,
                "role": cell.role,
                "unreached": str(cell.unreached),
                "budget": str(cell.budget),
                "n": str(len(SEEDS)),
                "hits_a": str(hits["A"]),
                "hits_ap": str(hits["Ap"]),
                "hits_b": str(hits["B"]),
                "both": str(both),
                "a_only": str(a_only),
                "b_only": str(b_only),
                "neither": str(neither),
                "mcnemar_p": fmt(p_hits),
                "hit_deficit": str(deficit),
                "hit_band": str(hit_band),
                "charged_a": fmt(charged_a, 8),
                "charged_ap": fmt(charged_ap, 8),
                "charged_b": fmt(charged_b, 8),
                "charged_ba": fmt(charged_ba),
                "charged_apa": fmt(charged_apa),
                "charged_band": fmt(charged_band),
                "calls_a": fmt(median(series("A", "calls")), 8),
                "calls_b": fmt(median(series("B", "calls")), 8),
                "calls_ba": fmt(call_ba[0]),
                "calls_ba_lo": fmt(call_ba[1]),
                "calls_ba_hi": fmt(call_ba[2]),
                "calls_apa": fmt(call_apa[0]),
                "calls_apa_lo": fmt(call_apa[1]),
                "calls_apa_hi": fmt(call_apa[2]),
                "wall_a": fmt(median(series("A", "wall_s"))),
                "wall_b": fmt(median(series("B", "wall_s"))),
                "wall_ba": fmt(wall_ba[0]),
                "wall_ba_lo": fmt(wall_ba[1]),
                "wall_ba_hi": fmt(wall_ba[2]),
                "wall_apa": fmt(wall_apa[0]),
                "wall_apa_lo": fmt(wall_apa[1]),
                "wall_apa_hi": fmt(wall_apa[2]),
                "outside_a": fmt(median([item for item in (per_call(ok["A"][seed]) for seed in ok["A"]) if item is not None])),
                "outside_b": fmt(median([item for item in (per_call(ok["B"][seed]) for seed in ok["B"]) if item is not None])),
                "outside_ba": fmt(out_ba[0]),
                "outside_ba_lo": fmt(out_ba[1]),
                "outside_ba_hi": fmt(out_ba[2]),
                "outside_apa": fmt(out_apa[0]),
                "outside_apa_lo": fmt(out_apa[1]),
                "outside_apa_hi": fmt(out_apa[2]),
                "wilcoxon_p": fmt(w_p),
                "bits_same": str(bits),
                "bits_n": str(len(SEEDS)),
                "failures": str(len(failures)),
                "fail_hits": str(int(fail_hits)),
                "fail_calls": str(int(fail_calls)),
                "fail_outside": str(int(fail_outside)),
                "fail_rows": str(int(fail_rows)),
                "verdict": verdict,
                "hit_a": strings["A"],
                "hit_b": strings["B"],
            }
        )
    for name, reason in missing:
        summary_rows.append(
            {
                "cell": name,
                "role": "missing",
                "unreached": "",
                "budget": "",
                "n": "0",
                "hits_a": "",
                "hits_ap": "",
                "hits_b": "",
                "both": "",
                "a_only": "",
                "b_only": "",
                "neither": "",
                "mcnemar_p": "",
                "hit_deficit": "",
                "hit_band": str(hit_band),
                "charged_a": "",
                "charged_ap": "",
                "charged_b": "",
                "charged_ba": "",
                "charged_apa": "",
                "charged_band": fmt(charged_band),
                "calls_a": "",
                "calls_b": "",
                "calls_ba": "",
                "calls_ba_lo": "",
                "calls_ba_hi": "",
                "calls_apa": "",
                "calls_apa_lo": "",
                "calls_apa_hi": "",
                "wall_a": "",
                "wall_b": "",
                "wall_ba": "",
                "wall_ba_lo": "",
                "wall_ba_hi": "",
                "wall_apa": "",
                "wall_apa_lo": "",
                "wall_apa_hi": "",
                "outside_a": "",
                "outside_b": "",
                "outside_ba": "",
                "outside_ba_lo": "",
                "outside_ba_hi": "",
                "outside_apa": "",
                "outside_apa_lo": "",
                "outside_apa_hi": "",
                "wilcoxon_p": "",
                "bits_same": "",
                "bits_n": "0",
                "failures": "0",
                "fail_hits": "0",
                "fail_calls": "0",
                "fail_outside": "0",
                "fail_rows": "0",
                "verdict": "MISSING",
                "hit_a": reason,
                "hit_b": "",
            }
        )
    band_rows.append(
        {
            "cell": "band",
            "hit_abs_diff": str(hit_band),
            "charged_ratio": "",
            "charged_spread": fmt(charged_band),
            "outside_ratio": "",
            "wall_ratio": "",
            "calls_ratio": "",
        }
    )
    write_table(args.out / "summary.csv", summary_rows)
    write_table(args.out / "aa_band.csv", band_rows)
    measured_fail = [row["cell"] for row in summary_rows if row["verdict"] == "FAIL"]
    print(f"HIT_BAND {hit_band}", flush=True)
    print(f"CHARGED_BAND {fmt(charged_band)}", flush=True)
    print(f"FAIL {' '.join(measured_fail) if measured_fail else 'none'}", flush=True)
    print(f"MISSING {' '.join(name for name, _reason in missing) if missing else 'none'}", flush=True)
    for row in summary_rows:
        if row["verdict"] == "MISSING":
            print(f"CELL {row['cell']} MISSING {row['hit_a']}", flush=True)
            continue
        print(
            f"CELL {row['cell']} {row['verdict']} hits {row['hits_a']}/{row['hits_b']}/{row['hits_ap']} "
            f"p {row['mcnemar_p']} charged {row['charged_ba']} outside {row['outside_ba']} "
            f"bits {row['bits_same']}/{row['bits_n']} disc {row['both']},{row['a_only']},{row['b_only']},{row['neither']}",
            flush=True,
        )
    return 1 if measured_fail else 0


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def selftest() -> None:
    if mcnemar_p(0, 0) != 1.0:
        raise SystemExit("mcnemar empty")
    if abs(mcnemar_p(8, 0) - 2 / 256) > 1e-15:
        raise SystemExit("mcnemar tail")
    if abs(mcnemar_p(6, 1) - 16 / 128) > 1e-15:
        raise SystemExit("mcnemar body")
    if wilcoxon_p([1, 1, 1], [1, 1, 1]) != 1.0:
        raise SystemExit("wilcoxon tie")
    if abs(wilcoxon_p([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]) - 2 / 64) > 1e-12:
        raise SystemExit("wilcoxon shift")
    if bits_of("1.0") != "3ff0000000000000":
        raise SystemExit("bits")
    if ratio(0.0, 0.0) != 1.0 or ratio(1.0, 0.0) != math.inf:
        raise SystemExit("ratio")
    order = arm_order(2000)
    if order != ["A", "B", "Ap"] or arm_order(2001) != ["Ap", "B", "A"]:
        raise SystemExit("order")
    stock = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s\n"
        "m38r6,anneal,2000,400000,true,12,34,1.500000,-157.47710800,1.250000,0.250000\n"
    )
    parsed = parse_stock(stock, "m38r6", 2000, 400000)
    if parsed is None or parsed["hit"] != "1" or parsed["outside_s"] != "0.250000":
        raise SystemExit("stock parse")
    gmin = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s\n"
        "lj38,anneal,2000,400000,false,400000,400000,2.000000,-173.90000000,unknown,unknown,cpu,0,1.500000\n"
    )
    parsed = parse_stock(gmin, "lj38", 2000, 400000)
    if parsed is None or parsed["hit"] != "0" or parsed["outside_s"] != "0.500000":
        raise SystemExit("gmin parse")
    seed_text = (
        "arm recommended draws 1 accepts 1 best -1.000000\n"
        "h2h_row system=lj75 code=anneal seed=2000 budget=4000000 hit=1 "
        "evals_to_hit=10 charged=20 best=-397.4923310000 wall_s=1.000000 "
        "pot_s=0.500000 outside_s=0.500000 basins=3\n"
    )
    parsed = parse_seed(seed_text, 2000, 4000000)
    if parsed is None or parsed["calls"] != "10" or parsed["best"] != "-397.4923310000":
        raise SystemExit("seed parse")
    print("SELFTEST ok", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("cells", "selftest", "run", "summarize"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--src-a", type=Path)
    parser.add_argument("--src-ap", type=Path)
    parser.add_argument("--src-b", type=Path)
    parser.add_argument("--bin-a", type=Path)
    parser.add_argument("--bin-ap", type=Path)
    parser.add_argument("--bin-b", type=Path)
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
        return 0
    cells, missing = collect_cells(args.root)
    if args.command == "cells":
        for cell in cells:
            print(
                f"{cell.name} role {cell.role} unreached {cell.unreached} driver {cell.driver} "
                f"budget {cell.budget} reference {cell.reference}",
                flush=True,
            )
        for name, reason in missing:
            print(f"MISSING {name} {reason}", flush=True)
        print(f"JOBS {len(cells) * len(SEEDS) * 3}", flush=True)
        return 0
    if args.command == "summarize":
        return summarize(args, cells, missing)
    for required in (args.src_a, args.src_ap, args.src_b, args.bin_a, args.bin_ap, args.bin_b):
        if required is None:
            raise SystemExit("run needs sources and binaries")
    jobs = build_schedule(cells)
    extra = run_campaign(args, cells, jobs)
    blocked = {name for name, _reason in extra}
    measured = [cell for cell in cells if cell.name not in blocked]
    return summarize(args, measured, missing + extra)


if __name__ == "__main__":
    sys.exit(main())
