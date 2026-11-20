#!/usr/bin/env python3
"""Run the no-size-constants r5 A/B gate serially."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

A_REVISION = "61fa67cf380e46876b569da45e7cc091647a775b"
B_TREE = "64b12739c8707669e6ae203d8f8b99a160609b55"
SEEDS = tuple(range(2000, 2012))
TIMEOUT_SECONDS = 12 * 60 * 60
SYSTEMS = (
    ("lj", "lj38", "h2h_gmin_proof", 400_000),
    ("lj", "lj75", "h2h_gmin_proof", 4_000_000),
    ("lj", "lj98", "h2h_gmin_proof", 4_000_000),
    ("lj", "lj103", "h2h_gmin_proof", 4_000_000),
    ("lj", "lj104", "h2h_gmin_proof", 4_000_000),
    ("morse", "m38r6", "h2h_morse", 400_000),
    ("morse", "m38r14", "h2h_morse", 400_000),
    ("morse", "m55r6", "h2h_morse", 1_000_000),
    ("morse", "m55r14", "h2h_morse", 1_000_000),
    ("morse", "m80r6", "h2h_morse", 4_000_000),
    ("morse", "m80r14", "h2h_morse", 4_000_000),
    ("au", "au38", "h2h_metal", 400_000),
)
ROW_FIELDS = (
    "session",
    "panel",
    "system",
    "seed",
    "slot",
    "arm",
    "order",
    "budget",
    "reference",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "outside_per_call_s",
    "code_revision",
    "code_tree",
    "binary_sha256",
    "runner_blob",
    "summarizer_blob",
    "machine_id_sha256",
)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
)


def checked(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"command failed with {proc.returncode}: {' '.join(command)}\n{proc.stderr[-2000:]}")
    return proc.stdout.strip()


def git_value(tree: Path, expression: str) -> str:
    return checked(["git", "rev-parse", expression], cwd=tree)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def machine_id_hash() -> str:
    value = Path("/etc/machine-id").read_bytes().strip()
    if not value:
        raise RuntimeError("empty machine identity")
    return hashlib.sha256(value).hexdigest()


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("H2H_"):
            del env[key]
    for key in THREAD_VARS:
        env[key] = "1"
    env["CARGO_BUILD_JOBS"] = "1"
    if extra:
        env.update(extra)
    return env


def build(tree: Path, destination: Path) -> dict[str, object]:
    target = destination / "target"
    binaries = destination / "bin"
    target.mkdir(parents=True)
    binaries.mkdir(parents=True)
    env = clean_env({"CARGO_TARGET_DIR": str(target)})
    checked(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "-j",
            "1",
            "--example",
            "h2h_gmin_proof",
            "--example",
            "h2h_morse",
            "--example",
            "h2h_metal",
        ],
        cwd=tree,
        env=env,
    )
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name in ("h2h_gmin_proof", "h2h_morse", "h2h_metal"):
        source = target / "release" / "examples" / name
        output = binaries / name
        shutil.copyfile(source, output)
        output.chmod(0o755)
        paths[name] = output
        hashes[name] = sha256(output)
    return {
        "revision": git_value(tree, "HEAD"),
        "tree": git_value(tree, "HEAD^{tree}"),
        "root": tree,
        "paths": paths,
        "hashes": hashes,
    }


def budget_and_reference(binary: Path, system: str, cwd: Path) -> tuple[int, float]:
    output = checked([str(binary), "budget", system], cwd=cwd, env=clean_env())
    lines = output.splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"{system}: malformed budget response")
    return int(lines[0]), float(lines[1])


def result_row(output: str, system: str, seed: int, budget: int) -> dict[str, str]:
    headers = [line for line in output.splitlines() if line.startswith("system,")]
    records = [line for line in output.splitlines() if line.startswith(f"{system},anneal,{seed},")]
    if len(headers) != 1 or len(records) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one result row")
    row = dict(zip(headers[0].split(","), records[0].split(",")))
    if int(row["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: budget {row['budget']} != {budget}")
    wall = float(row["wall_s"])
    potential = float(row["pot_s"])
    outside = float(row.get("outside_s", wall - potential))
    charged = int(row["charged"])
    return {
        "hit": "1" if row["hit"].lower() in ("1", "true") else "0",
        "evals_to_hit": str(int(row["evals_to_hit"])),
        "charged": str(charged),
        "best": row["best"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{potential:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call_s": f"{outside / max(charged, 1):.12g}",
    }


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def verify_schedule(rows: list[dict[str, str]]) -> None:
    expected = {(slot, arm): 6 for slot in (1, 2) for arm in ("A", "B")}
    for _panel, system, _binary, _budget in SYSTEMS:
        selected = [row for row in rows if row["system"] == system]
        counts = {
            (slot, arm): sum(int(row["slot"]) == slot and row["arm"] == arm for row in selected)
            for slot, arm in expected
        }
        if counts != expected:
            raise RuntimeError(f"{system}: unbalanced slot/arm schedule {counts}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--runner-blob", required=True)
    parser.add_argument("--summarizer-blob", required=True)
    args = parser.parse_args(argv)

    if args.rows.exists():
        raise SystemExit("rows already exist; this campaign cannot be resumed or rerun")
    if git_value(args.a_tree, "HEAD") != A_REVISION:
        raise SystemExit("arm A revision mismatch")
    if git_value(args.b_tree, "HEAD^{tree}") != B_TREE:
        raise SystemExit("arm B tree mismatch")

    arms = {
        "A": build(args.a_tree.resolve(), args.work / "A"),
        "B": build(args.b_tree.resolve(), args.work / "B"),
    }
    identity = machine_id_hash()
    references: dict[str, float] = {}
    for _panel, system, binary_name, expected_budget in SYSTEMS:
        arm_values = []
        for arm in ("A", "B"):
            info = arms[arm]
            arm_values.append(
                budget_and_reference(info["paths"][binary_name], system, info["root"])  # type: ignore[index]
            )
        if any(value[0] != expected_budget for value in arm_values):
            raise SystemExit(f"{system}: paper budget mismatch {arm_values}")
        if arm_values[0] != arm_values[1]:
            raise SystemExit(f"{system}: arm budget/reference mismatch {arm_values}")
        references[system] = arm_values[0][1]

    rows: list[dict[str, str]] = []
    args.logs.mkdir(parents=True, exist_ok=True)
    for panel, system, binary_name, budget in SYSTEMS:
        for offset, seed in enumerate(SEEDS):
            order = ("A", "B") if offset % 2 == 0 else ("B", "A")
            for slot, arm in enumerate(order, 1):
                info = arms[arm]
                binary = info["paths"][binary_name]  # type: ignore[index]
                command = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=60",
                    str(TIMEOUT_SECONDS),
                    "taskset",
                    "-c",
                    "0",
                    str(binary),
                    "run",
                    system,
                    str(seed),
                ]
                proc = subprocess.run(
                    command,
                    cwd=info["root"],  # type: ignore[arg-type]
                    env=clean_env(),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                log = args.logs / panel / system / f"{seed}-{arm}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(
                    f"exit={proc.returncode}\narm={arm}\nsystem={system}\nseed={seed}\n"
                    + proc.stdout
                    + proc.stderr
                )
                if proc.returncode:
                    raise SystemExit(f"{system} seed {seed} arm {arm} exited {proc.returncode}")
                parsed = result_row(proc.stdout, system, seed, budget)
                row = {
                    "session": "nsc-r5-gate-a",
                    "panel": panel,
                    "system": system,
                    "seed": str(seed),
                    "slot": str(slot),
                    "arm": arm,
                    "order": "".join(order),
                    "budget": str(budget),
                    "reference": f"{references[system]:.10f}",
                    **parsed,
                    "code_revision": str(info["revision"]),
                    "code_tree": str(info["tree"]),
                    "binary_sha256": info["hashes"][binary_name],  # type: ignore[index]
                    "runner_blob": args.runner_blob,
                    "summarizer_blob": args.summarizer_blob,
                    "machine_id_sha256": identity,
                }
                append_row(args.rows, row)
                rows.append(row)
                print(
                    f"{system} seed={seed} arm={arm} hit={row['hit']} "
                    f"calls={row['evals_to_hit']} wall={row['wall_s']}",
                    flush=True,
                )
    verify_schedule(rows)
    print(f"completed rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
