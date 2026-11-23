#!/usr/bin/env python3
"""Run the focused same-session TIP4P timing gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = tuple(range(2000, 2012))
CELLS = ("tip4p9", "tip4p10", "tip4p14", "tip4p16", "tip4p17")
BUDGET = 2_000_000
HIT_TOL = 1.0e-3
CORE = "0"
ARM_HEADS = {
    "A": "cb0cfbfa0cfae21b62cc32037defc5327dc52a51",
    "A2": "cb0cfbfa0cfae21b62cc32037defc5327dc52a51",
    "B": "752b938f4c680b72c9785516a2694676e2d7c93b",
}
ARM_TREES = {
    "A": "65972b37b798ebacd9d99d789df761099eefea1f",
    "A2": "65972b37b798ebacd9d99d789df761099eefea1f",
    "B": "618d1d0e981d54e2a5a46c9e5b9bb956fe30533b",
}
ORDERS = (
    ("A", "B", "A2", "B"),
    ("B", "A2", "B", "A"),
)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_DOMAIN_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "TBB_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENMP_NUM_THREADS",
)
DIAG_FIELDS = (
    "rigid_shell",
    "rigid_archive",
    "rigid_recorded",
    "rigid_withhold_nonfinite",
    "rigid_withhold_insane",
    "rigid_withhold_overlap",
    "rigid_withhold_gradient",
)
ROW_FIELDS = (
    "cell",
    "arm",
    "tree",
    "seed",
    "position",
    "arm_repeat",
    "slot",
    "exit_code",
    "energy",
    "energy_bits",
    "hit",
    "calls",
    "charged",
    "outside_s",
    "per_call_s",
    "rss_anon_end_kb",
    "exec_image",
    "affinity",
    *DIAG_FIELDS,
    "budget",
    "reference",
    "binary_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def references() -> dict[str, float]:
    with (ROOT / "data" / "tip4p_ccd.csv").open(newline="") as handle:
        source = {
            int(row["n"]): float(row["energy"])
            for row in csv.DictReader(handle)
        }
    return {cell: source[int(cell.removeprefix("tip4p"))] for cell in CELLS}


def child_env(probe: Path) -> dict[str, str]:
    env = {
        "HOME": os.environ["HOME"],
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LD_PRELOAD": str(probe),
    }
    for name in THREAD_VARS:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


def append(path: Path, fields: tuple[str, ...], row: dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def session(path: Path, *fields: object) -> None:
    with path.open("a", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(fields)
        handle.flush()
        os.fsync(handle.fileno())


class Arm:
    def __init__(self, name: str, spec: str) -> None:
        bin_dir, source_dir = spec.split(":", 1)
        self.name = name
        self.binary = Path(bin_dir).resolve() / "h2h_seed"
        self.source = Path(source_dir).resolve()
        self.binary_sha = sha256_file(self.binary)
        self.driver_sha = sha256_file(self.source / "examples" / "h2h_seed.rs")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.source), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def verify(self) -> None:
        dirty = self.git("status", "--porcelain")
        if dirty:
            raise RuntimeError(f"{self.name} source is dirty")
        if self.git("rev-parse", "HEAD") != ARM_HEADS[self.name]:
            raise RuntimeError(f"{self.name} head mismatch")
        if self.git("rev-parse", "HEAD^{tree}") != ARM_TREES[self.name]:
            raise RuntimeError(f"{self.name} tree mismatch")
        if sha256_file(self.binary) != self.binary_sha:
            raise RuntimeError(f"{self.name} binary changed")


def parse_arms(specs: list[str]) -> dict[str, Arm]:
    arms: dict[str, Arm] = {}
    for spec in specs:
        name, value = spec.split("=", 1)
        arms[name] = Arm(name, value)
    if set(arms) != {"A", "A2", "B"}:
        raise RuntimeError("A, A2, and B arms are required")
    if len({arm.driver_sha for arm in arms.values()}) != 1:
        raise RuntimeError("driver sources differ")
    for arm in arms.values():
        arm.verify()
    return arms


def run_measured(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, str, str, int, str, str]:
    pid = os.fork()
    if pid == 0:
        try:
            os.chdir(cwd)
            out_fd = os.open(
                stdout_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o644,
            )
            err_fd = os.open(
                stderr_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o644,
            )
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            os.close(out_fd)
            os.close(err_fd)
            os.execvpe(command[0], command, env)
        except BaseException as error:  # noqa: BLE001
            os.write(2, f"exec failed: {error}\n".encode())
            os._exit(127)
    _, status = os.waitpid(pid, 0)
    stdout = stdout_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace")
    reports = re.findall(
        r"^rss_anon_self_kb=(\d+) exec_image=(\S+) affinity=(\S+)$",
        stderr,
        re.M,
    )
    if len(reports) != 1:
        raise RuntimeError(f"self RSS reports: {len(reports)}")
    rss, image, affinity = reports[0]
    return os.waitstatus_to_exitcode(status), stdout, stderr, int(rss), image, affinity


def parse_output(text: str) -> dict[str, str]:
    line = next(value for value in text.splitlines() if value.startswith("h2h_row "))
    row = dict(token.split("=", 1) for token in line.split()[1:])
    calls = re.search(r"^calls: (\d+)$", text, re.M)
    if calls is None:
        raise RuntimeError("calls missing")
    return {
        "calls": calls.group(1),
        "charged": row["charged"],
        "outside": row["outside_s"],
        "best": row["best"],
        "budget": row["budget"],
    }


def parse_diag(stderr: str) -> list[dict[str, str]]:
    pattern = (
        r"^rigid_diag shell=(\d+) archive=(\d+) recorded=(\d+) "
        r"withhold_nonfinite=(\d+) withhold_insane=(\d+) "
        r"withhold_overlap=(\d+) withhold_gradient=(\d+)$"
    )
    return [
        dict(zip(DIAG_FIELDS, values, strict=True))
        for values in re.findall(pattern, stderr, re.M)
    ]


def command(
    arm: Arm,
    cell: str,
    seed: int,
    reference: float,
    prefix: Path,
) -> list[str]:
    size = cell.removeprefix("tip4p")
    return [
        "taskset",
        "-c",
        CORE,
        str(arm.binary),
        "tip4p",
        size,
        "0",
        str(BUDGET),
        str(seed),
        "recommended",
        repr(reference + HIT_TOL),
        str(prefix),
    ]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--rss-probe", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    if os.sched_getaffinity(0) != {0}:
        os.sched_setaffinity(0, {0})
    arms = parse_arms(args.arm)
    output = args.out.resolve()
    rows_path = output / "rows.csv"
    sessions_path = output / "sessions.csv"
    if rows_path.exists():
        raise RuntimeError("rows already exist")
    work = args.work.resolve()
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    probe = args.rss_probe.resolve()
    if not probe.is_file():
        raise RuntimeError("RSS probe missing")
    env = child_env(probe)
    refs = references()
    session(sessions_path, "event", "cell", "seed", "value")
    session(sessions_path, "begin", "", "", "")
    for arm in arms.values():
        session(sessions_path, "binary", "", "", f"{arm.name}:{arm.binary_sha}")
    slot = 0
    for cell in CELLS:
        session(sessions_path, "cell_begin", cell, "", "")
        for seed_index, seed in enumerate(SEEDS):
            repeats = {"A": 0, "A2": 0, "B": 0}
            for position, arm_name in enumerate(ORDERS[seed_index % 2], 1):
                repeats[arm_name] += 1
                arm = arms[arm_name]
                slot += 1
                tag = (
                    f"{cell}-s{seed}-p{position}-"
                    f"{arm_name.lower()}-{repeats[arm_name]}"
                )
                prefix = logs / tag
                stdout_path = Path(f"{prefix}.out")
                stderr_path = Path(f"{prefix}.err")
                run = command(arm, cell, seed, refs[cell], prefix)
                rc, stdout, stderr, rss, image, affinity = run_measured(
                    run,
                    logs,
                    env,
                    stdout_path,
                    stderr_path,
                )
                Path(f"{prefix}.rc").write_text(f"{rc}\n")
                if rc != 0:
                    raise RuntimeError(f"{tag} exited {rc}")
                if image != "h2h_seed" or affinity != CORE:
                    raise RuntimeError(f"{tag} exec verification failed")
                parsed = parse_output(stdout)
                if int(parsed["budget"]) != BUDGET:
                    raise RuntimeError(f"{tag} budget mismatch")
                diagnostics = parse_diag(stderr)
                expected = 1 if arm_name == "B" else 0
                if len(diagnostics) != expected:
                    raise RuntimeError(f"{tag} rigid diagnostics mismatch")
                energy = float(parsed["best"])
                hit = energy <= refs[cell] + HIT_TOL
                calls = int(parsed["calls"]) if hit else BUDGET
                charged = int(parsed["charged"])
                outside = float(parsed["outside"])
                row: dict[str, object] = {
                    "cell": cell,
                    "arm": arm_name,
                    "tree": ARM_TREES[arm_name],
                    "seed": seed,
                    "position": position,
                    "arm_repeat": repeats[arm_name],
                    "slot": slot,
                    "exit_code": rc,
                    "energy": parsed["best"],
                    "energy_bits": struct.pack(">d", energy).hex(),
                    "hit": int(hit),
                    "calls": calls,
                    "charged": charged,
                    "outside_s": parsed["outside"],
                    "per_call_s": f"{outside / charged:.12g}",
                    "rss_anon_end_kb": rss,
                    "exec_image": image,
                    "affinity": affinity,
                    "budget": BUDGET,
                    "reference": repr(refs[cell]),
                    "binary_sha256": arm.binary_sha,
                }
                row.update({field: "" for field in DIAG_FIELDS})
                if diagnostics:
                    row.update(diagnostics[0])
                append(rows_path, ROW_FIELDS, row)
        session(sessions_path, "cell_end", cell, "", "")
    for arm in arms.values():
        arm.verify()
    session(sessions_path, "complete", "", "", slot)


if __name__ == "__main__":
    main(sys.argv)
