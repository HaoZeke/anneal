#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the fixed G1 Morse comparison in one serial session."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import time
import uuid
from pathlib import Path

ANNEAL_REVISION = "61fa67cf380e46876b569da45e7cc091647a775b"
GMIN_REVISION = "9cbec1a446352bc9ef064bd2b1a5767194655919"
GMIN_ARCHIVE_SHA256 = (
    "ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776"
)
SEEDS = tuple(range(2000, 2012))
CELLS = (
    ("m30r6", 30, 6.0, 400_000, -118.432844),
    ("m30r14", 30, 14.0, 400_000, -106.835790),
    ("m38r6", 38, 6.0, 400_000, -157.477108),
    ("m38r14", 38, 14.0, 400_000, -144.321054),
    ("m55r6", 55, 6.0, 1_000_000, -250.286609),
    ("m55r14", 55, 14.0, 1_000_000, -220.646208),
    ("m70r3", 70, 3.0, 1_000_000, -577.739782),
    ("m70r6", 70, 6.0, 1_000_000, -325.887749),
    ("m70r10", 70, 10.0, 1_000_000, -296.412149),
    ("m70r14", 70, 14.0, 1_000_000, -292.462856),
    ("m80r3", 80, 3.0, 4_000_000, -690.577890),
    ("m80r10", 80, 10.0, 4_000_000, -344.911381),
)
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
FIELDS = (
    "system",
    "rho",
    "arm",
    "seed",
    "budget",
    "reference_energy",
    "best_energy",
    "best_bits",
    "calls",
    "charged",
    "wall_s",
    "hit",
    "exit_code",
    "gmin_revision",
    "anneal_commit",
    "data_file_sha256",
    "machine_id",
    "cpus_allowed_list",
    "thread_variables",
    "peak_rssanon_kb",
    "session",
    "sequence",
)


def cell(name: str) -> tuple:
    for item in CELLS:
        if item[0] == name:
            return item
    raise ValueError(name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def machine_id() -> str:
    return sha256(Path("/etc/machine-id"))


def run_env() -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    keys = set(THREAD_KEYS)
    keys.update(key for key in env if key.endswith("_NUM_THREADS"))
    for key in keys:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    shown = {key: env[key] for key in sorted(keys)}
    shown["MKL_DYNAMIC"] = env["MKL_DYNAMIC"]
    shown["OMP_DYNAMIC"] = env["OMP_DYNAMIC"]
    return env, json.dumps(shown, sort_keys=True, separators=(",", ":"))


def status_value(pid: int, key: str) -> str:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return ""


def run_process(
    command: list[str], cwd: Path, env: dict[str, str], log: Path
) -> tuple[int, float, str, int, str, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log.with_suffix(".stdout")
    stderr_path = log.with_suffix(".stderr")
    started = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        allowed = ""
        peak = 0
        while proc.poll() is None:
            allowed = status_value(proc.pid, "Cpus_allowed_list") or allowed
            value = status_value(proc.pid, "RssAnon").split()
            if value:
                peak = max(peak, int(value[0]))
            time.sleep(0.002)
        code = proc.wait()
    wall = time.perf_counter() - started
    stdout_text = stdout_path.read_text()
    stderr_text = stderr_path.read_text()
    log.write_text(
        f"exit_code={code}\n"
        f"wall_s={wall:.9f}\n"
        f"cpus_allowed_list={allowed}\n"
        f"peak_rssanon_kb={peak}\n"
    )
    return code, wall, allowed, peak, stdout_text, stderr_text


def anneal_result(text: str, system: str, seed: int) -> tuple[float, int, int]:
    rows = [
        line
        for line in text.splitlines()
        if line.startswith(f"{system},anneal,{seed},")
    ]
    if len(rows) != 1:
        raise RuntimeError(f"{system} {seed}: missing anneal result")
    header = (
        "system,arm,seed,budget,hit,evals_to_hit,charged,wall_s,best,"
        "pot_s,outside_s"
    )
    row = next(csv.DictReader([header, rows[0]]))
    return float(row["best"]), int(row["evals_to_hit"]), int(row["charged"])


def gmin_result(path: Path, system: str, seed: int) -> tuple[float, int, int]:
    lines = [line for line in path.read_text().splitlines() if line.startswith("h2h_row ")]
    if len(lines) != 1:
        raise RuntimeError(f"{system} {seed}: missing GMIN result")
    values = dict(part.split("=", 1) for part in lines[0].split()[1:] if "=" in part)
    if values["system"] != system or int(values["seed"]) != seed:
        raise RuntimeError(f"{system} {seed}: mismatched GMIN result")
    return (
        float(values["best"]),
        int(values["evals_to_hit"]),
        int(values["charged"]),
    )


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def arm_order(seed: int) -> tuple[str, str]:
    return ("anneal", "gmin") if (seed - SEEDS[0]) % 2 == 0 else ("gmin", "anneal")


def verify_binary(binary: Path, root: Path) -> None:
    env, _shown = run_env()
    for name, _n, _rho, budget, reference in CELLS:
        proc = subprocess.run(
            [str(binary), "budget", name],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        values = [line for line in proc.stdout.splitlines() if line]
        if int(values[0]) != budget or abs(float(values[1]) - reference) > 1e-12:
            raise RuntimeError(f"{name}: binary specification differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--anneal-bin", type=Path, required=True)
    parser.add_argument("--gmin-bin", type=Path, required=True)
    parser.add_argument("--gmin-source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()

    if args.rows.exists():
        raise SystemExit("rows output already exists; refusing a partial rerun")
    source = dict(
        line.split("=", 1)
        for line in args.gmin_source.read_text().splitlines()
        if "=" in line
    )
    if source.get("svnrev") != GMIN_REVISION:
        raise SystemExit("GMIN revision differs")
    if source.get("tarball_sha256") != GMIN_ARCHIVE_SHA256:
        raise SystemExit("GMIN archive differs")
    verify_binary(args.anneal_bin, args.root)

    env, shown_threads = run_env()
    machine = machine_id()
    session = uuid.uuid4().hex
    sequence = 0
    for system, _n, rho, budget, reference in CELLS:
        data_path = args.data_dir / f"{system}.data"
        data_hash = sha256(data_path)
        for seed in SEEDS:
            for arm in arm_order(seed):
                sequence += 1
                run_dir = args.work / system / str(seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                log = args.logs / system / f"{arm}-{seed}.log"
                if arm == "anneal":
                    command = [
                        "taskset",
                        "-c",
                        "0",
                        str(args.anneal_bin),
                        "run",
                        system,
                        str(seed),
                    ]
                    code, wall, allowed, peak, stdout, stderr = run_process(
                        command, args.root, env, log
                    )
                    if code != 0:
                        raise RuntimeError(f"{system} anneal {seed} exited {code}: {stderr}")
                    best, calls, charged = anneal_result(stdout, system, seed)
                else:
                    coords = run_dir / "coords"
                    coord_proc = subprocess.run(
                        [
                            str(args.anneal_bin),
                            "coords",
                            system,
                            str(seed),
                            str(coords),
                        ],
                        cwd=args.root,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    if coord_proc.stdout or coord_proc.stderr:
                        raise RuntimeError(f"{system} {seed}: unexpected coordinate output")
                    data = data_path.read_text()
                    if not data.endswith("\n"):
                        data += "\n"
                    (run_dir / "data").write_text(data + f"RANSEED {seed}\n")
                    row_path = run_dir / "row.txt"
                    row_path.unlink(missing_ok=True)
                    child_env = env.copy()
                    child_env.update(
                        {
                            "H2H_BUDGET": str(budget),
                            "H2H_SYSTEM": system,
                            "H2H_SEED": str(seed),
                            "H2H_TARGET": f"{reference:.10f}",
                            "H2H_TOL": "0.001",
                            "H2H_ROW": str(row_path),
                        }
                    )
                    command = ["taskset", "-c", "0", str(args.gmin_bin)]
                    code, wall, allowed, peak, _stdout, stderr = run_process(
                        command, run_dir, child_env, log
                    )
                    if code != 0 or not row_path.exists():
                        raise RuntimeError(f"{system} GMIN {seed} exited {code}: {stderr}")
                    best, calls, charged = gmin_result(row_path, system, seed)

                hit = math.isfinite(best) and best <= reference + 0.001
                if not hit:
                    calls = budget
                row = {
                    "system": system,
                    "rho": f"{rho:g}",
                    "arm": arm,
                    "seed": str(seed),
                    "budget": str(budget),
                    "reference_energy": f"{reference:.6f}",
                    "best_energy": f"{best:.17g}",
                    "best_bits": struct.pack(">d", best).hex(),
                    "calls": str(calls),
                    "charged": str(charged),
                    "wall_s": f"{wall:.9f}",
                    "hit": "1" if hit else "0",
                    "exit_code": str(code),
                    "gmin_revision": GMIN_REVISION,
                    "anneal_commit": ANNEAL_REVISION,
                    "data_file_sha256": data_hash,
                    "machine_id": machine,
                    "cpus_allowed_list": allowed,
                    "thread_variables": shown_threads,
                    "peak_rssanon_kb": str(peak),
                    "session": session,
                    "sequence": str(sequence),
                }
                append_row(args.rows, row)
                print(
                    f"{sequence}/288 {system} {arm} {seed} "
                    f"hit={row['hit']} calls={calls} wall={wall:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
