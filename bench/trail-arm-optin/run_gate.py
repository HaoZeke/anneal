#!/usr/bin/env python3
"""Build and run the serial three-arm opt-in gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

FIELDS = [
    "session",
    "cell",
    "system",
    "seed",
    "arm",
    "arm_order",
    "budget",
    "commit_sha",
    "tree",
    "binary_sha256",
    "machine_id",
    "core",
    "thread_count",
    "hit",
    "calls_to_hit",
    "charged",
    "best",
    "final_energy",
    "final_energy_bits",
    "redraws",
    "stall_triggers",
    "outside_s",
    "rss_anon_kib",
]

PANELS = [
    ("lj13-panel", "lj13", 80000, range(2000, 2012)),
    ("lj38-panel", "lj38", 80000, range(2000, 2012)),
    ("lj13-reach", "lj13", 80000, range(1, 4)),
]


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed with {proc.returncode}\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return proc


def output(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> str:
    return run(command, cwd=cwd, env=env).stdout.strip()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fixed_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def anonymous_machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            lineterminator="\n",
        )
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def parse_fields(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise RuntimeError(f"missing {prefix}: {line}")
    return dict(field.split("=", 1) for field in line.split()[1:])


def prepare_arm(
    name: str,
    tree: Path,
    work: Path,
    bench: Path,
    env: dict[str, str],
) -> dict[str, str | Path]:
    source = tree / "examples" / "stall_arm_probe.rs"
    shutil.copyfile(bench / "stall_arm_probe.rs", source)
    run(
        [
            "python3",
            str(bench / "instrument.py"),
            str(tree / "src" / "methods" / "cluster_hopping.rs"),
        ],
        cwd=tree,
        env=env,
    )
    target = work / name / "target"
    tmp = work / name / "tmp"
    target.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    build_env = env.copy()
    build_env["CARGO_TARGET_DIR"] = str(target)
    build_env["TMPDIR"] = str(tmp)
    build_env["RUSTFLAGS"] = (
        f"--remap-path-prefix={tree}=/source "
        f"--remap-path-prefix={Path.home()}=/user"
    )
    run(
        [
            "cargo",
            "+1.88.0",
            "build",
            "--release",
            "--locked",
            "-j",
            "1",
            "--example",
            "stall_arm_probe",
        ],
        cwd=tree,
        env=build_env,
    )
    binary = target / "release" / "examples" / "stall_arm_probe"
    return {
        "name": name,
        "tree_path": tree,
        "binary": binary,
        "commit": output(
            ["git", "rev-parse", "HEAD"],
            cwd=tree,
            env=env,
        ),
        "tree": output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=tree,
            env=env,
        ),
        "binary_sha256": digest(binary),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--a2-tree", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    bench = Path(__file__).resolve().parent
    env = fixed_env()
    args.work.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    core = min(os.sched_getaffinity(0))
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    machine = anonymous_machine_id()
    arms = {
        "A": prepare_arm(
            "A",
            args.a_tree.resolve(),
            args.work.resolve(),
            bench,
            env,
        ),
        "B": prepare_arm(
            "B",
            args.b_tree.resolve(),
            args.work.resolve(),
            bench,
            env,
        ),
        "A2": prepare_arm(
            "A2",
            args.a2_tree.resolve(),
            args.work.resolve(),
            bench,
            env,
        ),
    }
    done: set[tuple[str, int, str]] = set()
    if args.rows.exists():
        with args.rows.open(newline="") as handle:
            done = {
                (row["cell"], int(row["seed"]), row["arm"])
                for row in csv.DictReader(handle)
            }
    total = sum(len(list(seeds)) * 3 for _, _, _, seeds in PANELS)
    completed = len(done)
    for cell, system, budget, seeds in PANELS:
        for offset, seed in enumerate(seeds):
            order = (
                ("A", "B", "A2")
                if offset % 2 == 0
                else ("A2", "B", "A")
            )
            for arm_name in order:
                key = (cell, seed, arm_name)
                if key in done:
                    continue
                arm = arms[arm_name]
                command = [
                    "taskset",
                    "-c",
                    str(core),
                    str(arm["binary"]),
                    system,
                    str(budget),
                    str(seed),
                ]
                started = time.monotonic_ns()
                proc = run(
                    command,
                    cwd=Path(arm["tree_path"]),
                    env=env,
                    check=False,
                )
                elapsed = (time.monotonic_ns() - started) / 1e9
                log = args.logs / cell / f"{seed}-{arm_name}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(
                    f"exit={proc.returncode}\n"
                    f"outside_s={elapsed:.9f}\n"
                    f"{proc.stdout}{proc.stderr}"
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"failed run; inspect {log}")
                result_line = next(
                    line
                    for line in proc.stdout.splitlines()
                    if line.startswith("RESULT ")
                )
                trace_line = next(
                    line
                    for line in proc.stderr.splitlines()
                    if line.startswith("BENCH_STALL ")
                )
                result = parse_fields(result_line, "RESULT ")
                trace = parse_fields(trace_line, "BENCH_STALL ")
                append_row(
                    args.rows,
                    {
                        "session": session,
                        "cell": cell,
                        "system": system,
                        "seed": str(seed),
                        "arm": arm_name,
                        "arm_order": ",".join(order),
                        "budget": str(budget),
                        "commit_sha": str(arm["commit"]),
                        "tree": str(arm["tree"]),
                        "binary_sha256": str(
                            arm["binary_sha256"]
                        ),
                        "machine_id": machine,
                        "core": str(core),
                        "thread_count": "1",
                        "hit": result["hit"],
                        "calls_to_hit": result["calls"],
                        "charged": result["charged"],
                        "best": result["best"],
                        "final_energy": result["final"],
                        "final_energy_bits": result["final_bits"],
                        "redraws": trace["redraws"],
                        "stall_triggers": trace["triggers"],
                        "outside_s": f"{elapsed:.9f}",
                        "rss_anon_kib": result["rss_anon_kib"],
                    },
                )
                completed += 1
                done.add(key)
                print(
                    f"{completed}/{total} {cell} seed={seed} "
                    f"arm={arm_name} hit={result['hit']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
