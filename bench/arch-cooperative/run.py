#!/usr/bin/env python3
"""Run pinned four-chain CooperativeRun ensembles for LJ75."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "bench/arch-cooperative/lj75/rows.csv"
BASE = "3ccb2ad4fb930270f0fe95568ff09a5d0127d81b"
TARGET = -397.492331
TOLERANCE = 1.0e-4
REPLICAS = 4
KIND_NAMES = (
    "local_proposal",
    "remote_proposal",
    "descriptor_evaluation",
    "accepted_quench",
    "rejected_quench",
    "fresh_validation",
    "retry",
    "rpc_fallback",
    "auxiliary_evaluation",
    "basin_escape",
    "saddle_ride",
)
FIELDS = (
    "system",
    "arm",
    "seed",
    "chains",
    "workers",
    "total_oracle_calls",
    "call_budget",
    "hit",
    "calls_to_hit",
    "best_energy",
    "wall_s",
    "cpu_s",
    "peak_rssanon_kib",
    *(field for name in KIND_NAMES for field in (f"{name}_events", f"{name}_calls")),
    "ledger_named_sum",
    "ensemble_total",
    "coordinator_total",
    "revision",
    "nproc",
    "cpu_model",
)


def paper_budget() -> int:
    import sys

    panel = ROOT / "scripts/paper_0120"
    sys.path.insert(0, str(panel))
    import run_panel

    budget = int(run_panel.PAPER_BUDGETS["lj75"])
    if budget != 4_000_000:
        raise RuntimeError(f"unexpected lj75 paper budget {budget}")
    return budget


def cargo_target_directory() -> Path:
    output = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(json.loads(output.stdout)["target_directory"])


def expanded_seed(seed: int, replica: int) -> int:
    return (seed + replica * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)


def direct_children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, ProcessLookupError):
        return []


def process_family(roots: list[subprocess.Popen[bytes]]) -> set[int]:
    pending = [process.pid for process in roots if process.poll() is None]
    family: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in family:
            continue
        family.add(pid)
        pending.extend(direct_children(pid))
    return family


def rssanon_kib(roots: list[subprocess.Popen[bytes]]) -> int:
    total = 0
    for pid in process_family(roots):
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        for line in status.splitlines():
            if line.startswith("RssAnon:"):
                total += int(line.split()[1])
                break
    return total


def timed_command(cpu: int, timing: Path, command: list[str]) -> list[str]:
    return [
        "taskset",
        "-c",
        str(cpu),
        "/usr/bin/time",
        "-f",
        "%U,%S",
        "-o",
        str(timing),
        *command,
    ]


def await_endpoint(
    server: subprocess.Popen[bytes], output: Path, error: Path
) -> str:
    for _ in range(200):
        if server.poll() is not None:
            raise RuntimeError(
                "catalog coordinator exited during startup:\n"
                + error.read_text(errors="replace")
            )
        if output.exists():
            for line in output.read_text(errors="replace").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get("addr"):
                    return str(value["addr"])
        time.sleep(0.05)
    raise RuntimeError("catalog coordinator did not publish an endpoint")


def parse_timing(path: Path) -> float:
    line = path.read_text().strip().splitlines()[-1]
    user, system = line.split(",")
    return float(user) + float(system)


def parse_ledger(path: Path) -> dict[str, int]:
    lines = [
        line
        for line in path.read_text(errors="replace").splitlines()
        if line.startswith("COOPERATIVE_LEDGER ")
    ]
    if len(lines) != 1:
        raise RuntimeError(f"{path.name}: expected one cooperative ledger line")
    values: dict[str, int] = {}
    for token in lines[0].split()[1:]:
        key, value = token.split("=", 1)
        values[key] = int(value)
    expected = {
        field
        for name in KIND_NAMES
        for field in (f"{name}_events", f"{name}_calls")
    } | {"named_sum", "ensemble_total"}
    if set(values) != expected:
        raise RuntimeError(f"{path.name}: incomplete ledger terms")
    return values


def parse_best(path: Path) -> float:
    matches = re.findall(r"^  seed \d+: best (-?\d+\.\d+)", path.read_text(), re.M)
    if len(matches) != 1:
        raise RuntimeError(f"{path.name}: expected one final seed result")
    return float(matches[0])


def trace_totals(paths: list[Path]) -> tuple[int, int | None]:
    coordinator_total = 0
    calls_to_hit: list[int] = []
    for path in paths:
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if "aggregate_charged" not in event:
                continue
            charged = int(event["aggregate_charged"])
            coordinator_total = max(coordinator_total, charged)
            energy = event.get("slice_energy")
            if energy is not None and float(energy) <= TARGET + TOLERANCE:
                calls_to_hit.append(charged)
    return coordinator_total, min(calls_to_hit) if calls_to_hit else None


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def append_row(row: dict[str, object]) -> None:
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    if ROWS.exists():
        with ROWS.open(newline="") as handle:
            for previous in csv.DictReader(handle):
                if int(previous["seed"]) == int(row["seed"]):
                    raise RuntimeError(f"seed {row['seed']} is already recorded")
    new_file = not ROWS.exists()
    with ROWS.open("a", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_seed(seed: int) -> dict[str, object]:
    budget = paper_budget()
    per_replica = budget // REPLICAS
    examples = cargo_target_directory() / "release/examples"
    binary = examples / "lj_cluster_search"
    server_binary = examples / "catalog_server"
    for executable in (binary, server_binary):
        if not executable.is_file():
            raise RuntimeError(f"missing release executable {executable.name}")

    with tempfile.TemporaryDirectory(prefix=f"arch-coop-{seed}-") as directory:
        work = Path(directory)
        state = work / "state"
        state.mkdir()
        campaign = "arch-cooperative"
        ensemble = f"lj75-{seed}"
        server_out = work / "coordinator.jsonl"
        server_err = work / "coordinator.err"
        server_time = work / "coordinator.time"
        server_stdout = server_out.open("wb")
        server_stderr = server_err.open("wb")
        start = time.monotonic()
        server = subprocess.Popen(
            timed_command(
                4,
                server_time,
                [
                    str(server_binary),
                    "127.0.0.1:0",
                    "75",
                    "30",
                    "0.1",
                    str(budget),
                    campaign,
                    ensemble,
                    "0,1,2,3",
                    str(state),
                ],
            ),
            stdout=server_stdout,
            stderr=server_stderr,
        )
        workers: list[subprocess.Popen[bytes]] = []
        worker_files: list[tuple[object, object]] = []
        try:
            endpoint = await_endpoint(server, server_out, server_err)
            for replica in range(REPLICAS):
                output = work / f"worker-{replica}.out"
                error = work / f"worker-{replica}.err"
                timing = work / f"worker-{replica}.time"
                trace = work / f"trace-{replica}.jsonl"
                resolved = work / f"resolved-{replica}.json"
                stdout = output.open("wb")
                stderr = error.open("wb")
                worker_files.append((stdout, stderr))
                environment = os.environ.copy()
                environment.update(
                    {
                        "CATALOG_CAMPAIGN": campaign,
                        "CATALOG_ENSEMBLE": ensemble,
                        "CATALOG_REPLICA": str(replica),
                        "CATALOG_REPLICAS": str(REPLICAS),
                        "CATALOG_SLICE": "500",
                        "CATALOG_TRANSPORT_NOISE": "0.05",
                        "CATALOG_TRANSPORT_RADIUS": str(math.sqrt(75)),
                        "CATALOG_POPULATION_INTERVAL": "50000",
                        "CATALOG_TRACE": str(trace),
                        "CATALOG_RPC": endpoint,
                        "CATALOG_SHARING": "shared",
                        "ANNEAL_RESOLVED_CONFIG": str(resolved),
                        "SEED_OFFSET": str(expanded_seed(seed, replica)),
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "BLIS_NUM_THREADS": "1",
                        "RAYON_NUM_THREADS": "1",
                    }
                )
                workers.append(
                    subprocess.Popen(
                        timed_command(
                            replica,
                            timing,
                            [
                                str(binary),
                                "75",
                                str(per_replica),
                                "1",
                                "communicating",
                            ],
                        ),
                        stdout=stdout,
                        stderr=stderr,
                        env=environment,
                    )
                )
            peak_rss = 0
            while any(worker.poll() is None for worker in workers):
                peak_rss = max(peak_rss, rssanon_kib([server, *workers]))
                time.sleep(0.01)
            peak_rss = max(peak_rss, rssanon_kib([server, *workers]))
            failures = [
                (replica, worker.returncode)
                for replica, worker in enumerate(workers)
                if worker.returncode != 0
            ]
            if failures:
                detail = "\n".join(
                    (work / f"worker-{replica}.err").read_text(errors="replace")
                    for replica, _ in failures
                )
                raise RuntimeError(f"worker failures {failures}:\n{detail}")
        finally:
            if server.poll() is None:
                children = direct_children(server.pid)
                if children:
                    for child in children:
                        os.kill(child, signal.SIGTERM)
                else:
                    server.send_signal(signal.SIGTERM)
            server.wait()
            server_stdout.close()
            server_stderr.close()
            for stdout, stderr in worker_files:
                stdout.close()
                stderr.close()
        wall = time.monotonic() - start
        if wall < 0.05:
            raise RuntimeError(f"timed sample was only {wall:.6f} seconds")

        ledgers = [parse_ledger(work / f"worker-{replica}.out") for replica in range(REPLICAS)]
        best = min(parse_best(work / f"worker-{replica}.out") for replica in range(REPLICAS))
        coordinator_total, calls_to_hit = trace_totals(
            [work / f"trace-{replica}.jsonl" for replica in range(REPLICAS)]
        )
        row: dict[str, object] = {
            "system": "lj75",
            "arm": "CooperativeRun",
            "seed": seed,
            "chains": REPLICAS,
            "workers": REPLICAS,
            "total_oracle_calls": sum(value["ensemble_total"] for value in ledgers),
            "call_budget": budget,
            "hit": int(best <= TARGET + TOLERANCE),
            "calls_to_hit": calls_to_hit if calls_to_hit is not None else "",
            "best_energy": f"{best:.6f}",
            "wall_s": f"{wall:.6f}",
            "cpu_s": f"{sum(parse_timing(work / f'worker-{replica}.time') for replica in range(REPLICAS)) + parse_timing(server_time):.6f}",
            "peak_rssanon_kib": peak_rss,
            "revision": BASE,
            "nproc": len(os.sched_getaffinity(0)),
            "cpu_model": cpu_model(),
        }
        named_sum = 0
        for name in KIND_NAMES:
            events = sum(value[f"{name}_events"] for value in ledgers)
            calls = sum(value[f"{name}_calls"] for value in ledgers)
            row[f"{name}_events"] = events
            row[f"{name}_calls"] = calls
            named_sum += calls
        ensemble_total = sum(value["ensemble_total"] for value in ledgers)
        row["ledger_named_sum"] = named_sum
        row["ensemble_total"] = ensemble_total
        row["coordinator_total"] = coordinator_total
        if not named_sum == ensemble_total == coordinator_total == budget:
            raise RuntimeError(
                "ledger mismatch: "
                f"named={named_sum} workers={ensemble_total} "
                f"coordinator={coordinator_total} budget={budget}"
            )
        if bool(row["hit"]) != (calls_to_hit is not None):
            raise RuntimeError(
                f"hit/trace mismatch: best={best} calls_to_hit={calls_to_hit}"
            )
        return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", nargs="*", type=int)
    arguments = parser.parse_args()
    seeds = arguments.seeds or list(range(2000, 2012))
    for seed in seeds:
        if seed not in range(2000, 2012):
            raise SystemExit(f"seed {seed} is outside 2000-2011")
        row = run_seed(seed)
        append_row(row)
        print(
            f"seed {seed}: hit={row['hit']} best={row['best_energy']} "
            f"wall={row['wall_s']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
