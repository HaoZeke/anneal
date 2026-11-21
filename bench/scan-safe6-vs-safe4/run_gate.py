#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial orchestrator for the scan-safe6 against scan-safe4 gate.

The schedule, the cells and the refusal rules are the ones named in
README.md. A run that has a start line is never launched again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS = (
    ("lj38", 400_000, -173.928427),
    ("lj55", 1_000_000, -279.248470),
    ("lj75", 4_000_000, -397.492331),
    ("lj98", 4_000_000, -543.665361),
    ("lj102", 4_000_000, -569.363652),
    ("lj103", 4_000_000, -575.766131),
    ("lj104", 4_000_000, -582.086642),
    ("lj150", 4_000_000, -893.310258),
    ("au55", 1_000_000, -191.03085280),
    ("m80r6", 4_000_000, -378.333471),
    ("m80r14", 4_000_000, -340.811371),
)

SEEDS = tuple(range(2000, 2006))
ROUNDS = ("A", "B", "B", "A", "A", "A2", "A2", "A")

EXPECTED_COMMIT = {
    "A": "7d5bd6924075fbc66c33da3632d86f2e7eb40b76",
    "A2": "7d5bd6924075fbc66c33da3632d86f2e7eb40b76",
    "B": "7965b1873421dfbd812f4143ae273e20f17b429c",
}
EXPECTED_TREE = {
    "A": "578a12095ab5e0e6ac89c805173db78a9d1578b7",
    "A2": "578a12095ab5e0e6ac89c805173db78a9d1578b7",
    "B": "68b35a35c13fd4c5ce683594c744f3b5cb2cad8d",
}

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "KMP_NUM_THREADS",
)

FIELDS = [
    "cell",
    "arm",
    "seed",
    "round",
    "exit",
    "parse_ok",
    "hit",
    "calls",
    "charged",
    "best",
    "best_bits",
    "final",
    "final_bits",
    "trajectory_sha256",
    "nearest_index_calls",
    "wall",
    "inside_s",
    "outside_s",
    "budget",
    "reference",
    "rss_anon_peak_kb",
    "vm_hwm_kb",
    "ru_maxrss_kb",
    "rss_full",
    "core",
    "machine_id",
    "library_tree",
    "binary_sha256",
]

METRIC_FIELDS = [
    "cell",
    "seed",
    "budget",
    "reference",
    "hit",
    "calls",
    "charged",
    "wall",
    "inside_s",
    "outside_s",
    "best",
    "best_bits",
    "final",
    "final_bits",
    "trajectory_sha256",
    "nearest_index_calls",
    "rss_anon_peak_kb",
    "vm_hwm_kb",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_text(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def journal_line(path: Path, **items: str) -> None:
    parts = [f"ts={utc_now()}"]
    for key, value in items.items():
        if any(ch.isspace() for ch in str(value)):
            raise ValueError(f"journal value for {key} contains whitespace")
        parts.append(f"{key}={value}")
    append_text(path, " ".join(parts))


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    raise SystemExit("machine id is not readable")


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def measured_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("H2H_", "ANNEAL_"))
    }
    for key in list(env):
        if key.endswith("_NUM_THREADS"):
            env[key] = "1"
    for key in THREAD_KEYS:
        env[key] = "1"
    return env


def pin_orchestrator() -> None:
    cores = set(range(os.cpu_count() or 1))
    cores.discard(0)
    if not cores:
        raise SystemExit("the orchestrator needs a core other than 0")
    os.sched_setaffinity(0, cores)


def schedule() -> list[dict[str, str]]:
    runs = []
    for index, arm in enumerate(ROUNDS, start=1):
        for cell, _budget, _reference in CELLS:
            for seed in SEEDS:
                runs.append(
                    {
                        "round": str(index),
                        "arm": arm,
                        "cell": cell,
                        "seed": str(seed),
                        "run": f"r{index}-{cell}-s{seed}-{arm}",
                    }
                )
    return runs


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_header(path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())


def append_row(path: Path, row: dict[str, str]) -> None:
    ensure_header(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writerow({key: row.get(key, "") for key in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def parse_journal(path: Path) -> tuple[dict[str, dict[str, str]], set[str]]:
    starts: dict[str, dict[str, str]] = {}
    pids: dict[str, str] = {}
    ends: set[str] = set()
    if not path.is_file():
        return starts, ends
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = {}
        for part in line.split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            item[key] = value
        run = item.get("run", "")
        event = item.get("event", "")
        if event == "start" and run:
            starts[run] = item
        elif event == "pid" and run and item.get("pid", "").isdigit():
            pids[run] = item["pid"]
        elif event == "end" and run:
            ends.add(run)
    for run, pid in pids.items():
        if run in starts:
            starts[run]["pid"] = pid
    return starts, ends


def process_matches(pid: int, run: dict[str, str]) -> bool:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    parts = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    if "run" not in parts:
        return False
    try:
        at = parts.index("run")
    except ValueError:
        return False
    tail = parts[at + 1 :]
    return len(tail) >= 2 and tail[0] == run["cell"] and tail[1] == run["seed"]


def parse_metrics(stdout_path: Path, cell: str, seed: str) -> dict[str, str] | None:
    if not stdout_path.is_file():
        return None
    lines = [
        line.strip()
        for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        return None
    try:
        rows = list(csv.DictReader(lines[-2:]))
    except csv.Error:
        return None
    if len(rows) != 1:
        return None
    parsed = rows[0]
    if any(key not in parsed or parsed[key] is None for key in METRIC_FIELDS):
        return None
    if parsed["cell"] != cell or parsed["seed"] != seed:
        return None
    return {key: parsed[key] for key in METRIC_FIELDS}


def blank_metrics() -> dict[str, str]:
    return {key: "" for key in METRIC_FIELDS}


def reap(pid: int) -> tuple[int, int]:
    waited, status, usage = os.wait4(pid, 0)
    if waited != pid:
        raise RuntimeError(f"waited on {waited} while sampling {pid}")
    exit_code = os.waitstatus_to_exitcode(status)
    return exit_code, int(usage.ru_maxrss)


def launch(
    binary: Path,
    run: dict[str, str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("w", encoding="utf-8", newline="\n")
    stderr_handle = stderr_path.open("w", encoding="utf-8", newline="\n")
    proc = subprocess.Popen(
        ["taskset", "-c", "0", str(binary), "run", run["cell"], run["seed"]],
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        stdin=subprocess.DEVNULL,
    )
    stdout_handle.close()
    stderr_handle.close()
    return proc.pid


def verify_arms(dirs: dict[str, Path], bins: dict[str, Path]) -> tuple[dict[str, str], dict[str, str]]:
    trees: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for arm, repo in dirs.items():
        commit = git_output(repo, "rev-parse", "HEAD")
        tree = git_output(repo, "rev-parse", "HEAD^{tree}")
        if commit != EXPECTED_COMMIT[arm] or tree != EXPECTED_TREE[arm]:
            raise SystemExit(
                f"arm {arm} is {commit} tree {tree}, "
                f"expected {EXPECTED_COMMIT[arm]} tree {EXPECTED_TREE[arm]}"
            )
        trees[arm] = tree
        binary = bins[arm]
        if not binary.is_file():
            raise SystemExit(f"arm {arm} binary is missing")
        hashes[arm] = sha256_file(binary)
    return trees, hashes


def check_budgets(bins: dict[str, Path], env: dict[str, str]) -> None:
    for cell, budget, reference in CELLS:
        for arm, binary in bins.items():
            proc = subprocess.run(
                [str(binary), "budget", cell],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise SystemExit(f"budget check failed for {arm} {cell}: {proc.stderr}")
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if len(lines) < 2 or int(lines[0]) != budget:
                raise SystemExit(f"budget mismatch for {arm} {cell}: {lines!r}")
            if abs(float(lines[1]) - reference) > 1e-9:
                raise SystemExit(f"reference mismatch for {arm} {cell}: {lines[1]!r}")


def write_provenance(
    path: Path,
    trees: dict[str, str],
    hashes: dict[str, str],
    machine_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["arm", "commit", "tree", "binary_sha256", "machine_id"],
            lineterminator="\n",
        )
        writer.writeheader()
        for arm in ("A", "A2", "B"):
            writer.writerow(
                {
                    "arm": arm,
                    "commit": EXPECTED_COMMIT[arm],
                    "tree": trees[arm],
                    "binary_sha256": hashes[arm],
                    "machine_id": machine_id,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())


def finish_run(
    run: dict[str, str],
    exit_code: int,
    max_rss: int,
    stdout_path: Path,
    rows_path: Path,
    journal_path: Path,
    machine_id: str,
    trees: dict[str, str],
    hashes: dict[str, str],
) -> None:
    parsed = parse_metrics(stdout_path, run["cell"], run["seed"])
    row = {key: "" for key in FIELDS}
    row.update(parsed if parsed is not None else {})
    row.update(
        {
            "cell": run["cell"],
            "arm": run["arm"],
            "seed": run["seed"],
            "round": run["round"],
            "exit": str(exit_code),
            "parse_ok": "1" if parsed is not None else "0",
            "ru_maxrss_kb": str(max_rss),
            "rss_full": "1" if parsed is not None else "0",
            "core": "0",
            "machine_id": machine_id,
            "library_tree": trees[run["arm"]],
            "binary_sha256": hashes[run["arm"]],
        }
    )
    journal_line(journal_path, event="end", run=run["run"], exit=str(exit_code))
    append_row(rows_path, row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the registered scan gate.")
    parser.add_argument("--arm-a-dir", type=Path, required=True)
    parser.add_argument("--arm-a2-dir", type=Path, required=True)
    parser.add_argument("--arm-b-dir", type=Path, required=True)
    parser.add_argument("--arm-a-bin", type=Path, required=True)
    parser.add_argument("--arm-a2-bin", type=Path, required=True)
    parser.add_argument("--arm-b-bin", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=HERE / "rows.csv")
    parser.add_argument("--journal", type=Path, default=HERE / "journal.log")
    parser.add_argument("--provenance", type=Path, default=HERE / "provenance.csv")
    parser.add_argument("--print-schedule", action="store_true")
    args = parser.parse_args(argv)

    runs = schedule()
    if args.print_schedule:
        print(f"runs {len(runs)}")
        return 0

    pin_orchestrator()
    dirs = {"A": args.arm_a_dir, "A2": args.arm_a2_dir, "B": args.arm_b_dir}
    bins = {"A": args.arm_a_bin, "A2": args.arm_a2_bin, "B": args.arm_b_bin}
    trees, hashes = verify_arms(dirs, bins)
    env = measured_env()
    check_budgets(bins, env)
    machine_id = hashed_machine_id()
    write_provenance(args.provenance, trees, hashes, machine_id)
    ensure_header(args.rows)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    restarting = args.journal.is_file() and args.journal.stat().st_size > 0
    if restarting:
        journal_line(args.journal, event="orchestrator_restart")
    else:
        journal_line(args.journal, event="orchestrator_start")

    done_rows = {
        (row["cell"], row["arm"], row["seed"], row["round"]) for row in read_rows(args.rows)
    }
    starts, ends = parse_journal(args.journal)

    for ordinal, run in enumerate(runs, start=1):
        key = (run["cell"], run["arm"], run["seed"], run["round"])
        stdout_path = args.logs_dir / f"{run['run']}.out"
        stderr_path = args.logs_dir / f"{run['run']}.err"
        if key in done_rows:
            continue
        started = starts.get(run["run"])
        if started is not None:
            pid_text = started.get("pid", "")
            if pid_text.isdigit() and process_matches(int(pid_text), run):
                exit_code, max_rss = reap(int(pid_text))
                finish_run(
                    run,
                    exit_code,
                    max_rss,
                    stdout_path,
                    args.rows,
                    args.journal,
                    machine_id,
                    trees,
                    hashes,
                )
                done_rows.add(key)
                print(f"[{ordinal}/{len(runs)}] adopted {run['run']} exit={exit_code}", flush=True)
            continue
        if run["run"] in ends:
            continue
        journal_line(
            args.journal,
            event="start",
            run=run["run"],
            cell=run["cell"],
            seed=run["seed"],
            round=run["round"],
            arm=run["arm"],
        )
        pid = launch(bins[run["arm"]], run, env, stdout_path, stderr_path)
        journal_line(args.journal, event="pid", run=run["run"], pid=str(pid))
        exit_code, max_rss = reap(pid)
        finish_run(
            run,
            exit_code,
            max_rss,
            stdout_path,
            args.rows,
            args.journal,
            machine_id,
            trees,
            hashes,
        )
        done_rows.add(key)
        print(
            f"[{ordinal}/{len(runs)}] {run['run']} exit={exit_code}",
            flush=True,
        )
        append_text(
            args.logs_dir / "progress.txt",
            f"{utc_now()} {ordinal}/{len(runs)} {run['run']} exit={exit_code}",
        )

    import analyze

    analyze.write_report(args.rows, args.journal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
