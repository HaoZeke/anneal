# Copyright (c) anneal developers
"""Timed rows for the morse m38 rho-14 round-four gate.

One seed's arms run on one core, in the order from cells.arm_order.
Lanes are separate cores. A row already stored is not run again.
An unfinished marker from an earlier process is stored as a miss.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cells as gate

ROW_FIELDS = [
    "cell",
    "driver",
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
    "rss_anon_peak",
    "ru_maxrss",
    "vmhwm_peak",
    "printed_hit",
    "conclusive",
    "log_sha256",
    "binary_sha256",
    "revision",
    "machine_id_sha256",
    "session",
    "sequence",
    "exit_code",
]

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "JULIA_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}


def child_env(tmpdir: str) -> dict[str, str]:
    """Whitelist passed to a timed process. Nothing else is copied."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": tmpdir,
    }
    env.update(THREAD_ENV)
    return env


def machine_id_sha256() -> str:
    for path in (Path("/etc/machine-id"), Path("/proc/sys/kernel/random/boot_id")):
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    raise SystemExit("machine id file is missing")


def session_of(machine: str) -> str:
    text = machine + "\ngate-pr171-m38r14d-r4\n"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_id(src: Path) -> tuple[str, str]:
    head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(src), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    return head, tree


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise SystemExit("MemAvailable missing")


def choose_lanes(requested: int) -> tuple[int, int]:
    available = mem_available()
    if requested != 1:
        raise SystemExit("round four uses one lane")
    return 1, available


def binary_name(cell: gate.Cell) -> str:
    if cell.invoke == "seed":
        return "h2h_seed"
    return cell.driver


def argv_for(cell: gate.Cell, binary: Path, seed: int, prefix: Path) -> list[str]:
    if cell.invoke == "panel":
        return [str(binary), "run", cell.name, str(seed)]
    rho = str(int(cell.rho)) if cell.rho == int(cell.rho) else repr(cell.rho)
    return [
        str(binary),
        cell.model,
        str(cell.n),
        rho,
        str(cell.budget),
        str(seed),
        "recommended",
        gate.stop_token(cell.reference),
        str(prefix),
    ]


def best_bits(token: str) -> str:
    if not token:
        return ""
    value = float(token)
    if not math.isfinite(value):
        return ""
    return format(struct.unpack(">Q", struct.pack(">d", value))[0], "016x")


def energy_hits(energy: float, target: float) -> bool:
    return (
        math.isfinite(energy)
        and math.isfinite(target)
        and energy <= target + gate.HIT_TOL
    )


def truthy(token: str) -> bool:
    return token in ("1", "true", "True")


def parse_panel(text: str, name: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if not line.startswith("system,code,seed"):
            continue
        header = line.split(",")
        for nxt in lines[index + 1 :]:
            if nxt.startswith(name + ","):
                data = nxt.split(",")
                row = dict(zip(header, data))
                wall = row["wall_s"]
                pot = row["pot_s"]
                outside = row.get("outside_s") or ""
                if not outside:
                    outside = f"{max(float(wall) - float(pot), 0.0):.6f}"
                return {
                    "hit": "1" if truthy(row["hit"]) else "0",
                    "printed_hit": "1" if truthy(row["hit"]) else "0",
                    "calls": row["evals_to_hit"],
                    "charged": row["charged"],
                    "wall_s": wall,
                    "pot_s": pot,
                    "outside_s": outside,
                    "best": row["best"],
                }
    raise ValueError("panel row missing")


def parse_seed(text: str, reference: float) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        if ": " in line and not line.startswith("h2h_row"):
            key, value = line.split(": ", 1)
            found[key.strip()] = value.strip()
    best = found["best"]
    lowest = found["lowest"]
    hit = energy_hits(float(lowest), reference)
    printed_hit = energy_hits(float(best), reference)
    return {
        "hit": "1" if hit else "0",
        "printed_hit": "1" if printed_hit else "0",
        "calls": found["calls"],
        "charged": found["charged"],
        "wall_s": found["wall_seconds"],
        "pot_s": found["pot_s"],
        "outside_s": found["outside_s"],
        "best": best,
    }


def parse_output(cell: gate.Cell, text: str) -> dict[str, str]:
    if cell.invoke == "seed":
        return parse_seed(text, cell.reference)
    return parse_panel(text, cell.name)


def key_of(row: dict[str, str]) -> tuple[str, str, str, int]:
    return (row["cell"], row["driver"], row["arm"], int(row["seed"]))


class Store:
    def __init__(self, rows_path: Path, sessions_path: Path, logs: Path):
        self.rows_path = rows_path
        self.sessions_path = sessions_path
        self.logs = logs
        self.pending = logs / "pending"
        self.completed = logs / "completed"
        self.lock = threading.Lock()
        self.rows_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending.mkdir(parents=True, exist_ok=True)
        self.completed.mkdir(parents=True, exist_ok=True)
        self.done: set[tuple[str, str, str, int]] = set()
        self.sequence = 0
        if self.rows_path.is_file() and self.rows_path.stat().st_size:
            with self.rows_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    self.done.add(key_of(row))
                    self.sequence = max(self.sequence, int(row["sequence"]))

    def append(self, path: Path, fields: list[str], row: dict[str, str]) -> None:
        fresh = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            if fresh:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    def add_row(self, row: dict[str, str]) -> None:
        with self.lock:
            self.sequence += 1
            row["sequence"] = str(self.sequence)
            self.append(self.rows_path, ROW_FIELDS, row)
            self.done.add(key_of(row))

    def add_session(self, row: dict[str, str]) -> None:
        fields = [
            "session",
            "event",
            "machine_id_sha256",
            "lanes",
            "cores",
            "mem_available",
            "revision",
            "tree",
            "arm",
            "binary",
            "binary_sha256",
            "detail",
        ]
        with self.lock:
            self.append(self.sessions_path, fields, row)


def marker_path(store: Store, cell: gate.Cell, arm: str, seed: int) -> Path:
    return store.pending / f"{cell.name}__{cell.driver}__{arm}__{seed}"


def finish_marker(store: Store, marker: Path) -> None:
    os.replace(marker, store.completed / marker.name)


def recover_markers(store: Store, session: str, machine: str, revisions: dict[str, str]) -> None:
    """A marker with no row means the process ended before the row."""
    for path in sorted(store.pending.iterdir()):
        name, driver, arm, seed_text = path.name.split("__")
        cell = next(item for item in gate.CELLS if item.name == name and item.driver == driver)
        if (name, driver, arm, int(seed_text)) in store.done:
            finish_marker(store, path)
            continue
        order = gate.arm_order(int(seed_text))
        slot = order.index(arm) + 1
        text = path.read_text()
        core = text.strip() or "0"
        store.add_row(
            {
                "cell": name,
                "driver": driver,
                "arm": arm,
                "seed": seed_text,
                "slot": str(slot),
                "core": core,
                "affinity": core,
                "hit": "0",
                "calls": str(cell.budget),
                "charged": "",
                "wall_s": "",
                "pot_s": "",
                "outside_s": "",
                "best": "",
                "best_bits": "",
                "budget": str(cell.budget),
                "rss_anon_peak": "",
                "ru_maxrss": "",
                "vmhwm_peak": "",
                "printed_hit": "",
                "conclusive": "0",
                "log_sha256": "",
                "binary_sha256": "",
                "revision": revisions.get(arm, ""),
                "machine_id_sha256": machine,
                "session": session,
                "sequence": "",
                "exit_code": "-9",
            }
        )
        finish_marker(store, path)


def read_rss(path: Path) -> str:
    if not path.is_file():
        return ""
    match = re.search(r"MAXRSS_KB\s+(\d+)", path.read_text())
    if not match:
        return ""
    return str(int(match.group(1)) * 1024)


def process_ids(root: int) -> set[int]:
    found = {root}
    pending = [root]
    while pending:
        pid = pending.pop()
        child_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = [int(token) for token in child_path.read_text().split()]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for child in children:
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def status_kib(pid: int, field: str) -> int:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    match = re.search(rf"^{field}:\s+(\d+)\s+kB$", text, re.MULTILINE)
    return int(match.group(1)) if match else 0


def monitor_memory(pid: int, stop: threading.Event, peaks: dict[str, int]) -> None:
    while not stop.wait(0.01):
        pids = process_ids(pid)
        rss_anon = sum(status_kib(item, "RssAnon") for item in pids)
        vmhwm = max((status_kib(item, "VmHWM") for item in pids), default=0)
        peaks["rss_anon"] = max(peaks["rss_anon"], rss_anon)
        peaks["vmhwm"] = max(peaks["vmhwm"], vmhwm)


def write_log(path: Path, text: str) -> str:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.sync()
    os.replace(tmp, path)
    os.sync()
    return sha256_file(path)


def run_arm(
    store: Store,
    cell: gate.Cell,
    arm: str,
    seed: int,
    slot: int,
    core: int,
    paths: dict,
    session: str,
    machine: str,
) -> None:
    key = (cell.name, cell.driver, arm, seed)
    with store.lock:
        if key in store.done:
            return
    binary = paths[arm]["bin"] / binary_name(cell)
    src = paths[arm]["src"]
    scratch = Path(tempfile.mkdtemp(prefix="gate-", dir=store.logs))
    prefix = scratch / "out"
    command = argv_for(cell, binary, seed, prefix)
    time_path = scratch / "time.txt"
    log_path = store.logs / f"{cell.name}__{cell.driver}__{arm}__{seed}.log"
    marker = marker_path(store, cell, arm, seed)
    marker.write_text(str(core) + "\n")
    wrapped = [
        "timeout",
        "--foreground",
        str(gate.HANG_S),
        "taskset",
        "-c",
        str(core),
        "/usr/bin/time",
        "-f",
        "MAXRSS_KB %M",
        "-o",
        str(time_path),
        "--",
        *command,
    ]
    env = child_env(str(scratch))
    proc = subprocess.Popen(
        wrapped,
        cwd=src,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stop = threading.Event()
    peaks = {"rss_anon": 0, "vmhwm": 0}
    monitor = threading.Thread(
        target=monitor_memory,
        args=(proc.pid, stop, peaks),
        daemon=False,
    )
    monitor.start()
    stdout, stderr = proc.communicate()
    stop.set()
    monitor.join()
    log_digest = write_log(
        log_path,
        "exit " + str(proc.returncode) + "\n" + stdout + "\n" + stderr,
    )
    parsed = None
    error = ""
    if proc.returncode == 0:
        try:
            parsed = parse_output(cell, stdout)
        except (ValueError, KeyError, IndexError) as exc:
            error = str(exc)
    exit_code = proc.returncode if proc.returncode != 0 or parsed is not None else 2
    if parsed is None:
        parsed = {
            "hit": "0",
            "printed_hit": "0",
            "calls": str(cell.budget),
            "charged": "",
            "wall_s": "",
            "pot_s": "",
            "outside_s": "",
            "best": "",
        }
    row = {
        "cell": cell.name,
        "driver": cell.driver,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "core": str(core),
        "affinity": str(core),
        "hit": parsed["hit"] if exit_code == 0 else "0",
        "calls": parsed["calls"],
        "charged": parsed["charged"],
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
        "best": parsed["best"],
        "best_bits": best_bits(parsed["best"]) if exit_code == 0 else "",
        "budget": str(cell.budget),
        "rss_anon_peak": str(peaks["rss_anon"] * 1024),
        "ru_maxrss": read_rss(time_path),
        "vmhwm_peak": str(peaks["vmhwm"] * 1024),
        "printed_hit": parsed["printed_hit"] if exit_code == 0 else "",
        "conclusive": "1" if exit_code == 0 else "0",
        "log_sha256": log_digest,
        "binary_sha256": paths[arm]["sha"][binary_name(cell)],
        "revision": paths[arm]["head"],
        "machine_id_sha256": machine,
        "session": session,
        "sequence": "",
        "exit_code": str(exit_code),
    }
    store.add_row(row)
    finish_marker(store, marker)
    print(
        f"done {cell.name} {cell.driver} {arm} {seed} exit={exit_code} hit={row['hit']} {error}",
        flush=True,
    )


def worker(core: int, jobs: queue.Queue, store: Store, paths, session: str, machine: str) -> None:
    while True:
        item = jobs.get()
        if item is None:
            return
        cell, seed = item
        for slot, arm in enumerate(gate.arm_order(seed), start=1):
            run_arm(store, cell, arm, seed, slot, core, paths, session, machine)


def check_trees(paths: dict) -> None:
    for arm in ("A", "A2"):
        if paths[arm]["head"] != gate.BASE or paths[arm]["tree"] != gate.BASE_TREE:
            raise SystemExit(f"{arm} is not BASE")
    for arm in ("B", "B2"):
        if paths[arm]["head"] != gate.PR_HEAD or paths[arm]["tree"] != gate.PR_TREE:
            raise SystemExit(f"{arm} tree mismatch")


def load_paths(args) -> dict:
    spec = {
        "A": (args.bin_a, args.src_a),
        "A2": (args.bin_a2, args.src_a2),
        "B": (args.bin_b, args.src_b),
        "B2": (args.bin_b, args.src_b),
    }
    paths = {}
    names = ("h2h_gmin_proof", "h2h_morse", "h2h_metal", "h2h_seed")
    for arm, (bin_dir, src) in spec.items():
        head, tree = git_id(src)
        digest = {}
        for name in names:
            binary = bin_dir / name
            if not binary.is_file():
                raise SystemExit(f"missing binary {name} for {arm}")
            digest[name] = sha256_file(binary)
        paths[arm] = {
            "bin": bin_dir,
            "src": src,
            "head": head,
            "tree": tree,
            "sha": digest,
        }
    return paths


def self_test() -> None:
    env = child_env("/tmp")
    if set(env) - set(THREAD_ENV) - {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}:
        raise SystemExit("env whitelist")
    for key, value in THREAD_ENV.items():
        if env[key] != value:
            raise SystemExit(key)
    panel = """\
system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s
lj38,anneal,2000,400000,true,10615,11000,0.075305,-173.92842659,abc,def,cpu,0,0.029236
"""
    parsed = parse_panel(panel, "lj38")
    if parsed["hit"] != "1" or parsed["calls"] != "10615":
        raise SystemExit("panel parse")
    if abs(float(parsed["outside_s"]) - (0.075305 - 0.029236)) > 1e-9:
        raise SystemExit("outside")
    if best_bits("-173.92842659") == "":
        raise SystemExit("bits")
    seed_text = """\
calls: 12000
charged: 11800
lowest: -397.49233100
best: -397.49233098
wall_seconds: 1.500000
pot_s: 1.000000
outside_s: 0.500000
h2h_row system=lj75 code=anneal seed=2000 budget=4000000 hit=1 evals_to_hit=12000 charged=11800 best=-397.49233100 wall_s=1.500000 pot_s=1.000000 outside_s=0.500000 basins=3
"""
    cell = next(item for item in gate.CELLS if item.name == "lj75" and item.driver == "h2h_seed")
    parsed = parse_seed(seed_text, cell.reference)
    if parsed["best"] != "-397.49233098" or parsed["hit"] != "1":
        raise SystemExit(f"seed parse {parsed}")
    if parsed["charged"] != "11800" or parsed["wall_s"] != "1.500000":
        raise SystemExit("seed fields")
    command = argv_for(cell, Path("h2h_seed"), 2000, Path("out"))
    if command[1:7] != ["lj", "75", "0", "4000000", "2000", "recommended"]:
        raise SystemExit(f"argv {command}")
    morse = next(item for item in gate.CELLS if item.name == "m38r14" and item.invoke == "panel")
    if argv_for(morse, Path("h2h_morse"), 2001, Path("out"))[-2:] != ["m38r14", "2001"]:
        raise SystemExit("panel argv")
    lanes, _available = choose_lanes(1)
    if lanes != 1:
        raise SystemExit("lanes")
    print("harness ok")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--bin-a", type=Path, required=True)
    parser.add_argument("--src-a", type=Path, required=True)
    parser.add_argument("--bin-a2", type=Path, required=True)
    parser.add_argument("--src-a2", type=Path, required=True)
    parser.add_argument("--bin-b", type=Path, required=True)
    parser.add_argument("--src-b", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    paths = load_paths(args)
    check_trees(paths)
    preflight_path = args.logs / "preflight.log"
    if args.preflight:
        gate.self_test()
        self_test()
        subprocess.run(
            ["taskset", "-c", "0", "true"],
            check=True,
            capture_output=True,
            text=True,
        )
        args.logs.mkdir(parents=True, exist_ok=True)
        lines = ["preflight ok"]
        for arm in ("A", "B", "B2", "A2"):
            lines.append(f"{arm} {paths[arm]['head']} {paths[arm]['tree']}")
        digest = write_log(preflight_path, "\n".join(lines) + "\n")
        print(f"preflight {digest}")
        return 0
    if not preflight_path.is_file():
        raise SystemExit("preflight is missing")
    preflight_digest = sha256_file(preflight_path)
    lanes, available = choose_lanes(args.lanes)
    cores = [0]
    machine = machine_id_sha256()
    session = session_of(machine)
    store = Store(args.rows, args.sessions, args.logs)
    recover_markers(store, session, machine, {arm: paths[arm]["head"] for arm in paths})
    store.add_session(
        {
            "session": session,
            "event": "start",
            "machine_id_sha256": machine,
            "lanes": str(lanes),
            "cores": " ".join(str(core) for core in cores),
            "mem_available": str(available),
            "revision": gate.BASE,
            "tree": gate.BASE_TREE,
            "arm": "",
            "binary": "",
            "binary_sha256": "",
            "detail": (
                f"b {gate.PR_HEAD} tree {gate.PR_TREE} "
                f"preflight {preflight_digest}"
            ),
        }
    )
    for arm in ("A", "B", "B2", "A2"):
        for name, digest in sorted(paths[arm]["sha"].items()):
            store.add_session(
                {
                    "session": session,
                    "event": "binary",
                    "machine_id_sha256": machine,
                    "lanes": str(lanes),
                    "cores": " ".join(str(core) for core in cores),
                    "mem_available": str(available),
                    "revision": paths[arm]["head"],
                    "tree": paths[arm]["tree"],
                    "arm": arm,
                    "binary": name,
                    "binary_sha256": digest,
                    "detail": "",
                }
            )
    jobs: queue.Queue = queue.Queue()
    pending_jobs = 0
    for cell in gate.CELLS:
        for seed in gate.SEEDS:
            needed = False
            for arm in gate.arm_order(seed):
                if (cell.name, cell.driver, arm, seed) not in store.done:
                    needed = True
            if needed:
                jobs.put((cell, seed))
                pending_jobs += 1
    print(f"session {session} lanes {lanes} jobs {pending_jobs}", flush=True)
    for _ in cores:
        jobs.put(None)
    threads = [
        threading.Thread(
            target=worker,
            args=(core, jobs, store, paths, session, machine),
            daemon=False,
        )
        for core in cores
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    store.add_session(
        {
            "session": session,
            "event": "end",
            "machine_id_sha256": machine,
            "lanes": str(lanes),
            "cores": " ".join(str(core) for core in cores),
            "mem_available": str(mem_available()),
            "revision": gate.BASE,
            "tree": gate.BASE_TREE,
            "arm": "",
            "binary": "",
            "binary_sha256": "",
            "detail": f"rows {store.sequence}",
        }
    )
    print(f"finished sequence {store.sequence}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
