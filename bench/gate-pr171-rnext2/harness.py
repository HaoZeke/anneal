#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial same-session runner for the PR 171 restack gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import resource
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cells as gate

ROW_FIELDS = [
    "cell",
    "driver",
    "arm",
    "seed",
    "slot",
    "hit",
    "calls",
    "charged",
    "outside_s",
    "outside_per_call_s",
    "child_user_s",
    "child_system_s",
    "pot_s",
    "seen_lowest",
    "out_best",
    "geometry_energy",
    "stop_run",
    "best_is_lowest",
    "seen_bits",
    "out_bits",
    "budget",
    "rss_anon_peak",
    "probe_offered",
    "probe_hit",
    "probe_calls",
    "binary_sha256",
    "revision",
    "tree",
    "machine_id_sha256",
    "session",
    "sequence",
    "exit_code",
    "log_sha256",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def machine_hash() -> str:
    for path in (Path("/etc/machine-id"), Path("/proc/sys/kernel/random/boot_id")):
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    raise SystemExit("machine identifier is unavailable")


def git_id(path: Path) -> tuple[str, str]:
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    return head, tree


def bits(token: str) -> str:
    value = float(token)
    if not math.isfinite(value):
        return ""
    return format(struct.unpack(">Q", struct.pack(">d", value))[0], "016x")


def child_env(scratch: Path, counter: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": str(scratch),
        "ANNEAL_GATE_OBSERVED_BASIS_COUNTER": str(counter),
    }
    env.update(THREAD_ENV)
    return env


def binary_name(cell: gate.Cell) -> str:
    return cell.driver


def command_for(cell: gate.Cell, binary: Path, seed: int, prefix: Path) -> list[str]:
    if cell.model == "panel":
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
        f"{gate.stop_level(cell):.10f}",
        str(prefix),
    ]


def parse_panel(text: str, name: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if not line.startswith("system,code,seed"):
            continue
        header = line.split(",")
        for candidate in lines[index + 1 :]:
            if candidate.startswith(name + ","):
                row = dict(zip(header, candidate.split(",")))
                return {
                    "calls": row["evals_to_hit"],
                    "charged": row["charged"],
                    "pot_s": row["pot_s"],
                    "seen_lowest": row["best"],
                    "out_best": row["best"],
                    "geometry_energy": "",
                }
    raise ValueError("panel row is missing")


def parse_seed(text: str) -> dict[str, str]:
    found = {}
    for line in text.splitlines():
        if ": " in line and not line.startswith("h2h_row"):
            key, value = line.split(": ", 1)
            found[key.strip()] = value.strip()
    return {
        "calls": found["calls"],
        "charged": found["charged"],
        "pot_s": found["pot_s"],
        "seen_lowest": found["lowest"],
        "out_best": found["best"],
        "geometry_energy": found["geometry_energy"],
    }


def parse_counter(path: Path) -> tuple[int, int, int]:
    found = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        found[key] = int(value)
    return found["offered"], found["hit"], found["calls"]


def status_rss(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    match = re.search(r"^RssAnon:\s+(\d+)\s+kB$", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else 0


def execute(
    command: list[str], source: Path, env: dict[str, str], scratch: Path
) -> tuple[int, str, str, resource.struct_rusage, int, float]:
    stdout_path = scratch / "stdout"
    stderr_path = scratch / "stderr"
    peak = 0
    started = time.monotonic()
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        proc = subprocess.Popen(
            ["taskset", "-c", "0", *command],
            cwd=source,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        while True:
            waited, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if waited:
                proc.returncode = os.waitstatus_to_exitcode(status)
                break
            peak = max(peak, status_rss(proc.pid))
            if time.monotonic() - started >= gate.HANG_S:
                proc.kill()
                _waited, _status, usage = os.wait4(proc.pid, 0)
                proc.returncode = 124
                break
            time.sleep(0.005)
    peak = max(peak, int(usage.ru_maxrss) * 1024)
    wall = time.monotonic() - started
    return (
        proc.returncode,
        stdout_path.read_text(),
        stderr_path.read_text(),
        usage,
        peak,
        wall,
    )


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    fresh = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_paths(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    specs = {
        "A": (args.src_a, args.bin_a, gate.BASE, gate.BASE_TREE),
        "A2": (args.src_a, args.bin_a2, gate.BASE, gate.BASE_TREE),
        "B": (args.src_b, args.bin_b, None, gate.PR_TREE),
        "B2": (args.src_b, args.bin_b, None, gate.PR_TREE),
    }
    paths = {}
    names = {binary_name(cell) for cell in gate.CELLS}
    for arm, (source, bin_dir, expected_head, expected_tree) in specs.items():
        head, tree = git_id(source)
        if tree != expected_tree or expected_head not in (None, head):
            raise SystemExit(f"{arm} revision or tree mismatch")
        digests = {}
        for name in names:
            binary = bin_dir / name
            if not binary.is_file():
                raise SystemExit(f"{arm} binary is missing: {name}")
            digests[name] = sha256_file(binary)
        paths[arm] = {
            "source": source,
            "bin": bin_dir,
            "head": head,
            "tree": tree,
            "sha": digests,
        }
    return paths


def existing_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["cell"], row["arm"], int(row["seed"]))
            for row in csv.DictReader(handle)
        }


def run_one(
    cell: gate.Cell,
    arm: str,
    seed: int,
    slot: int,
    paths: dict[str, dict[str, object]],
    logs: Path,
    machine: str,
    session: str,
    sequence: int,
) -> dict[str, str]:
    info = paths[arm]
    scratch = Path(tempfile.mkdtemp(prefix="run-", dir=logs))
    counter = scratch / "counter"
    binary = info["bin"] / binary_name(cell)
    command = command_for(cell, binary, seed, scratch / "result")
    code, stdout, stderr, usage, rss, wall = execute(
        command,
        info["source"],
        child_env(scratch, counter),
        scratch,
    )
    parsed = {}
    error = ""
    if code == 0:
        try:
            parsed = parse_panel(stdout, cell.name) if cell.model == "panel" else parse_seed(stdout)
            offered, probe_hit, probe_calls = parse_counter(counter)
        except (KeyError, ValueError, OSError) as exc:
            code = 2
            error = str(exc)
    if code != 0:
        parsed = {
            "calls": str(cell.budget),
            "charged": "",
            "pot_s": "",
            "seen_lowest": "",
            "out_best": "",
            "geometry_energy": "",
        }
        offered = probe_hit = probe_calls = 0
    log = logs / f"{cell.name}__{arm}__{seed}.log"
    log.write_text(
        f"exit {code}\nerror {error}\n"
        f"user_s {usage.ru_utime:.9f}\nsystem_s {usage.ru_stime:.9f}\n"
        + stdout
        + "\n"
        + stderr,
        newline="\n",
    )
    pot = float(parsed["pot_s"]) if parsed["pot_s"] else 0.0
    outside = max(wall - pot, 0.0)
    call_count = int(parsed["calls"]) if parsed["calls"] else 0
    outside_per_call = outside / call_count if call_count else 0.0
    seen = parsed["seen_lowest"]
    out = parsed["out_best"]
    hit = (
        code == 0
        and float(seen) <= gate.stop_level(cell)
        and float(out) <= gate.stop_level(cell)
    )
    # A stop run reached the stop level before the budget ran out. On such
    # a run the reported best must be the lowest evaluated energy, and the
    # returned geometry must evaluate to it.
    geometry = parsed["geometry_energy"]
    stop_run = (
        code == 0
        and cell.driver == "h2h_seed"
        and float(seen) <= gate.stop_level(cell)
        and int(parsed["calls"]) < cell.budget
    )
    if cell.driver != "h2h_seed" or code != 0:
        best_is_lowest = ""
    else:
        best_is_lowest = "1" if out == seen == geometry else "0"
    return {
        "cell": cell.name,
        "driver": cell.driver,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "hit": "1" if hit else "0",
        "calls": parsed["calls"],
        "charged": parsed["charged"],
        "outside_s": f"{outside:.9f}" if code == 0 else "",
        "outside_per_call_s": (
            f"{outside_per_call:.12g}" if code == 0 else ""
        ),
        "child_user_s": f"{usage.ru_utime:.9f}",
        "child_system_s": f"{usage.ru_stime:.9f}",
        "pot_s": parsed["pot_s"],
        "seen_lowest": seen,
        "out_best": out,
        "geometry_energy": geometry,
        "stop_run": "1" if stop_run else "0",
        "best_is_lowest": best_is_lowest,
        "seen_bits": bits(seen) if seen else "",
        "out_bits": bits(out) if out else "",
        "budget": str(cell.budget),
        "rss_anon_peak": str(rss),
        "probe_offered": str(offered),
        "probe_hit": str(probe_hit),
        "probe_calls": str(probe_calls),
        "binary_sha256": info["sha"][binary_name(cell)],
        "revision": info["head"],
        "tree": info["tree"],
        "machine_id_sha256": machine,
        "session": session,
        "sequence": str(sequence),
        "exit_code": str(code),
        "log_sha256": sha256_file(log),
    }


def self_test() -> None:
    gate.self_test()
    env = child_env(Path("/tmp"), Path("/tmp/counter"))
    assert all(env[key] == "1" for key in env if key.endswith("_NUM_THREADS"))
    sample = (
        "calls: 12\ncharged: 13\nlowest: -2\ngeometry_energy: -2\nbest: -2\n"
        "pot_s: 0.5\n"
    )
    assert parse_seed(sample)["calls"] == "12"
    assert parse_seed(sample)["geometry_energy"] == "-2"
    print("harness ok")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--hosts", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--src-a", type=Path, required=True)
    parser.add_argument("--src-b", type=Path, required=True)
    parser.add_argument("--bin-a", type=Path, required=True)
    parser.add_argument("--bin-a2", type=Path, required=True)
    parser.add_argument("--bin-b", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.rows.parent, args.sessions.parent, args.hosts.parent, args.logs):
        path.mkdir(parents=True, exist_ok=True)
    paths = load_paths(args)
    keys = existing_keys(args.rows)
    machine = machine_hash()
    session = hashlib.sha256(
        (machine + "\ngate-pr171-rnext2\n").encode()
    ).hexdigest()[:16]
    append_csv(
        args.hosts,
        ["session", "machine_id_sha256"],
        {"session": session, "machine_id_sha256": machine},
    )
    append_jsonl(
        args.sessions,
        {
            "event": "start",
            "session": session,
            "machine_id_sha256": machine,
            "base": gate.BASE,
            "candidate": paths["B"]["head"],
            "candidate_tree": gate.PR_TREE,
            "core": 0,
            "seeds": {cell.name: list(cell.seeds) for cell in gate.CELLS},
        },
    )
    sequence = len(keys)
    for cell in gate.CELLS:
        for seed in cell.seeds:
            for slot, arm in enumerate(gate.arm_order(seed), start=1):
                key = (cell.name, arm, seed)
                if key in keys:
                    continue
                sequence += 1
                row = run_one(
                    cell,
                    arm,
                    seed,
                    slot,
                    paths,
                    args.logs,
                    machine,
                    session,
                    sequence,
                )
                append_csv(args.rows, ROW_FIELDS, row)
                keys.add(key)
                print(
                    f"done {cell.name} {arm} {seed} "
                    f"exit={row['exit_code']} hit={row['hit']}",
                    flush=True,
                )
    append_jsonl(
        args.sessions,
        {"event": "end", "session": session, "rows": sequence},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
