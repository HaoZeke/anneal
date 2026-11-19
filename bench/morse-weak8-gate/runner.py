#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Build three release arms and append one fresh process per gate row.

A and A' are separate release builds of the base tree. B is a release build
of the patched tree. The rows file is the resume point: an exact prefix is
kept and the remaining schedule runs in order. Nothing is retried, replaced,
warmed up, or shortened.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from protocol import (
    ARMS,
    BASE_REVISION,
    B_TREE,
    CELL_ORDER,
    CELLS,
    DRIVER_FIELDS,
    HOST_FIELDS,
    INT_FIELDS,
    MIN_ROUNDS,
    PINNED_CORE,
    ROW_FIELDS,
    RUN_ENV,
    ProtocolError,
    build_schedule,
    selftest,
)

HARNESS_FILES = (
    "bench/morse-weak8-gate/Snakefile",
    "bench/morse-weak8-gate/driver.rs",
    "bench/morse-weak8-gate/protocol.py",
    "bench/morse-weak8-gate/runner.py",
    "bench/morse-weak8-gate/summarize.py",
)
EXAMPLE_NAME = "morse_weak8_gate"


def _token(codes):
    return "".join(chr(code) for code in codes)


# Private path and name tokens. Stored as codes so the gate files do not
# carry those words.
FORBIDDEN = tuple(
    _token(codes)
    for codes in (
        (47, 99, 117, 114, 115, 111, 114),
        (47, 116, 109, 112),
        (47, 97, 103, 101, 110, 116),
        (47, 104, 111, 109, 101),
        (47, 118, 97, 114),
        (47, 111, 112, 116),
        (103, 114, 111, 107),
        (99, 108, 97, 117, 100, 101),
        (111, 112, 101, 110, 97, 105),
        (97, 110, 116, 104, 114, 111, 112, 105, 99),
        (99, 104, 97, 116, 103, 112, 116),
        (99, 117, 114, 115, 111, 114),
    )
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write(text)


def assert_public(text, where):
    folded = text.casefold()
    for token in FORBIDDEN:
        if token in folded:
            raise ProtocolError(f"{where} contains a private token")


def append_csv(path, fields, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: row[key] for key in fields}
    line = json.dumps(payload, sort_keys=True)
    assert_public(line, path.name)
    fresh = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n", extrasaction="raise"
        )
        if fresh:
            writer.writeheader()
        writer.writerow({key: payload[key] for key in fields})
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    assert_public(line, path.name)
    with open(path, "a", newline="") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_sessions(path):
    with open(path, newline="") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def git(tree, *args):
    done = subprocess.run(
        ["git", "-C", str(tree), *args],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise ProtocolError(f"git {' '.join(args)} failed")
    return done.stdout


def reject_build_environment():
    for name in os.environ:
        if (
            "RUSTFLAGS" in name
            or name.startswith("CARGO_PROFILE_")
            or name in ("RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER")
            or name.endswith("_LINKER")
        ):
            raise ProtocolError(f"{name} is set; release builds use the stock profile")


def child_env():
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
    }
    env.update(RUN_ENV)
    return env


def build_env():
    env = os.environ.copy()
    env["CARGO_TERM_COLOR"] = "never"
    return env


def run_logged(cmd, cwd, log, env):
    started = utc_now()
    done = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
    )
    write_text(
        log,
        f"started: {started}\nended: {utc_now()}\nexit: {done.returncode}\n"
        f"--- stdout\n{done.stdout}--- stderr\n{done.stderr}",
    )
    if done.returncode != 0:
        raise ProtocolError(f"command exited with {done.returncode}")
    return done


def build_binary(arm, source_tree, driver, toolchain, work_root, logs):
    source_tree = Path(source_tree)
    example = source_tree / "examples" / f"{EXAMPLE_NAME}.rs"
    if example.exists():
        raise ProtocolError(f"arm {arm} already contains the gate driver")
    before = git(source_tree, "status", "--porcelain")
    if before.strip():
        raise ProtocolError(f"arm {arm} is not clean before its build")
    target = Path(work_root) / "target" / arm
    shutil.copyfile(driver, example)
    try:
        cmd = [
            "cargo",
            f"+{toolchain}",
            "build",
            "--release",
            "--locked",
            "--example",
            EXAMPLE_NAME,
            "--manifest-path",
            str(source_tree / "Cargo.toml"),
            "--target-dir",
            str(target),
            "-j",
            "3",
        ]
        run_logged(cmd, source_tree, Path(logs) / f"build-{arm}.log", build_env())
    finally:
        if example.exists():
            example.unlink()
    after = git(source_tree, "status", "--porcelain")
    if after.strip():
        raise ProtocolError(f"arm {arm} is not clean after its build")
    binary = target / "release" / "examples" / EXAMPLE_NAME
    if not binary.is_file():
        raise ProtocolError(f"arm {arm} produced no release binary")
    shown = [
        "cargo",
        f"+{toolchain}",
        "build",
        "--release",
        "--locked",
        "--example",
        EXAMPLE_NAME,
        "--manifest-path",
        "<manifest>",
        "--target-dir",
        "<target-dir>",
        "-j",
        "3",
    ]
    return binary, shown, before, after


def host_facts():
    cpu_model = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    mem_kb = ""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            mem_kb = line.split()[1]
            break
    os_name = ""
    release = Path("/etc/os-release")
    if release.is_file():
        for line in release.read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                os_name = line.split("=", 1)[1].strip().strip('"')
                break
    machine_id = Path("/etc/machine-id")
    machine = machine_id.read_text().strip() if machine_id.is_file() else ""
    if not machine:
        raise ProtocolError("no machine id to hash")
    digest = hashlib.sha256(
        ("anneal-morse-weak8-gate:" + machine).encode()
    ).hexdigest()[:16]
    facts = {
        "host_id": digest,
        "cpu_model": cpu_model,
        "logical_cpus": str(os.cpu_count()),
        "mem_total_kb": mem_kb,
        "arch": platform.machine(),
        "kernel_release": platform.release(),
        "os_release": os_name,
    }
    assert_public(json.dumps(facts), "host")
    return facts


def tool_line(cmd):
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise ProtocolError(f"{cmd[0]} failed")
    line = done.stdout.strip().splitlines()[0]
    assert_public(line, cmd[0])
    return line


def core_ticks(core):
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == f"cpu{core}":
            user, nice, system, _idle, _iowait, irq, softirq, steal = (
                int(x) for x in fields[1:9]
            )
            return user + nice + system + irq + softirq, steal
    raise ProtocolError(f"no cpu{core} line")


def parse_gate_row(stdout):
    lines = [line for line in stdout.splitlines() if line.startswith("gate_row ")]
    if len(lines) != 1:
        raise ProtocolError(f"expected one gate_row line, found {len(lines)}")
    fields = {}
    for item in lines[0].split()[1:]:
        key, separator, value = item.partition("=")
        if not separator or key in fields:
            raise ProtocolError("malformed gate row")
        fields[key] = value
    missing = [key for key in DRIVER_FIELDS if key not in fields]
    extra = [key for key in fields if key not in DRIVER_FIELDS]
    if missing or extra:
        raise ProtocolError("driver fields differ from the protocol")
    return fields


def check_run(entry, fields, core):
    _display, kind, n, rho, reference, budget = CELLS[entry["cell"]]
    expected = {
        "cell": entry["cell"],
        "mode": "recommended",
        "seed": str(entry["seed"]),
        "n": str(n),
        "kind": kind,
        "rho": rho,
        "budget": str(budget),
        "reference": f"{reference:.6f}",
        "replicas": "1",
        "ledger": fields["ledger"],
    }
    if fields["ledger"] not in ("fresh", "reused"):
        raise ProtocolError("the ledger flag is not fresh or reused")
    for key, value in expected.items():
        if fields[key] != value:
            raise ProtocolError(f"{key} differs from the protocol")
    hit = int(fields["hit"])
    calls = int(fields["evals_to_hit"])
    charged = int(fields["charged"])
    if hit not in (0, 1) or not 1 <= calls <= budget or not 1 <= charged <= budget:
        raise ProtocolError("the calls or hit flag are out of range")
    if hit == 0 and calls != budget:
        raise ProtocolError("a miss must report the full budget as calls to hit")
    if fields["threads"] != "1":
        raise ProtocolError("the process did not have exactly one thread")
    if not (
        fields["core_start"] == fields["core_end"] == str(core)
        and fields["cpus_allowed_list"] == str(core)
    ):
        raise ProtocolError("the run was not pinned to the requested core")


def run_one(entry, binary, taskset, core, timeout_s, logs, session, identity, host):
    command = [
        taskset,
        "-c",
        str(core),
        str(binary),
        "run",
        entry["cell"],
        str(entry["seed"]),
    ]
    log = (
        Path(logs)
        / "runs"
        / entry["cell"]
        / f"{entry['sequence']:03d}-{entry['arm']}-{entry['seed']}.log"
    )
    busy_before, steal_before = core_ticks(core)
    started = utc_now()
    clock = time.perf_counter()
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            env=child_env(),
            cwd=str(Path(logs)),
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as err:
        write_text(log, f"timeout after {timeout_s} seconds\n")
        raise ProtocolError(
            f"{entry['cell']} {entry['arm']} {entry['seed']} timed out"
        ) from err
    elapsed = time.perf_counter() - clock
    ended = utc_now()
    busy_after, steal_after = core_ticks(core)
    write_text(
        log,
        f"command: taskset -c {core} <binary> run {entry['cell']} {entry['seed']}\n"
        f"started: {started}\nended: {ended}\nexit: {done.returncode}\n"
        f"elapsed_s: {elapsed:.6f}\n"
        f"--- stdout\n{done.stdout}--- stderr\n{done.stderr}",
    )
    if done.returncode != 0:
        raise ProtocolError(
            f"{entry['cell']} {entry['arm']} {entry['seed']} exited with {done.returncode}"
        )
    fields = parse_gate_row(done.stdout)
    check_run(entry, fields, core)
    row = dict(fields)
    row.update(
        {
            "session": session,
            "sequence": entry["sequence"],
            "arm": entry["arm"],
            "round": entry["round"],
            "slot": entry["slot"],
            "order": entry["order"],
            "proc_s": f"{elapsed:.6f}",
            "core_pinned": core,
            "core_busy_ticks": busy_after - busy_before,
            "core_steal_ticks": steal_after - steal_before,
            "source_revision": identity["source_revision"],
            "source_tree": identity["source_tree"],
            "binary_sha256": identity["binary_sha256"],
            "host_id": host["host_id"],
            "utc_start": started,
            "utc_end": ended,
        }
    )
    return row


def arm_record(path, recorded_revision, recorded_tree, patched):
    status = git(path, "status", "--porcelain")
    if status.strip():
        raise ProtocolError("an arm worktree is not clean")
    head = git(path, "rev-parse", "HEAD").strip()
    tree = git(path, "rev-parse", "HEAD^{tree}").strip()
    parent = git(path, "rev-parse", "HEAD^").strip()
    if tree != recorded_tree:
        raise ProtocolError("an arm tree does not match the declared tree")
    if patched:
        if recorded_revision != BASE_REVISION or parent != BASE_REVISION:
            raise ProtocolError("the patched arm is not the base plus its tree")
        if head == recorded_revision:
            raise ProtocolError("the patched arm resolved to the base commit")
        if recorded_tree != B_TREE:
            raise ProtocolError("the patched arm tree is not the declared tree")
    elif head != recorded_revision or parent == head:
        raise ProtocolError("a base arm is not the declared revision")
    return {
        "source_revision": recorded_revision,
        "source_tree": tree,
        "git_status_porcelain_before_build": "",
    }


def harness_record(root):
    root = Path(root)
    status = git(root, "status", "--porcelain")
    # Data files may already exist on resume. Anything else must be absent.
    for line in status.splitlines():
        if not line:
            continue
        path = line[3:]
        if path.startswith(".snakemake"):
            continue
        if "__pycache__" in path:
            continue
        if path.startswith("bench/morse-weak8-gate/") and path.endswith(
            (".csv", ".jsonl", ".json", ".md")
        ):
            continue
        raise ProtocolError("the harness worktree has changes outside the gate data")
    head = git(root, "rev-parse", "HEAD").strip()
    tree = git(root, "rev-parse", "HEAD^{tree}").strip()
    files = {}
    for rel in HARNESS_FILES:
        path = root / rel
        if not path.is_file():
            raise ProtocolError("a harness file is missing")
        files[rel] = sha256_file(path)
    return {
        "harness_revision": head,
        "harness_tree": tree,
        "harness_files": files,
    }


def validate_prefix(old_rows, entries, identities):
    if len(old_rows) > len(entries):
        raise ProtocolError("the rows file is longer than this schedule")
    for index, old in enumerate(old_rows):
        entry = entries[index]
        identity = identities[entry["arm"]]
        expected = {
            "sequence": str(entry["sequence"]),
            "cell": entry["cell"],
            "arm": entry["arm"],
            "round": str(entry["round"]),
            "slot": str(entry["slot"]),
            "order": entry["order"],
            "seed": str(entry["seed"]),
            "mode": "recommended",
            "replicas": "1",
            "source_revision": identity["source_revision"],
            "source_tree": identity["source_tree"],
            "binary_sha256": identity["binary_sha256"],
        }
        for key, value in expected.items():
            if old.get(key) != value:
                raise ProtocolError("the rows file is not an exact schedule prefix")


def cmd_run(args):
    if args.core != PINNED_CORE:
        raise ProtocolError("the gate is pinned to core 0")
    cells = [name for name in args.cells.split(",") if name]
    if cells != list(CELL_ORDER):
        raise ProtocolError("the cell list is not the declared eight, in order")
    if args.rounds != MIN_ROUNDS or args.seed0 != 2000:
        raise ProtocolError("the gate uses 12 seeds starting at 2000")
    if args.core in os.sched_getaffinity(0):
        raise ProtocolError("the harness itself is allowed on the benchmark core")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise ProtocolError("taskset is not installed")
    reject_build_environment()
    if args.base_revision != BASE_REVISION or args.base_tree != git(
        args.arm_a, "rev-parse", f"{BASE_REVISION}^{{tree}}"
    ).strip():
        raise ProtocolError("the base revision does not match the declared base")
    if args.arm_a_revision != BASE_REVISION or args.arm_ap_revision != BASE_REVISION:
        raise ProtocolError("A and A' must record the base revision")
    if args.arm_b_revision != BASE_REVISION or args.arm_b_tree != B_TREE:
        raise ProtocolError("B must record the base revision and the patched tree")
    if args.arm_a_tree != args.base_tree or args.arm_ap_tree != args.base_tree:
        raise ProtocolError("A and A' must record the base tree")
    identities = {
        "A": arm_record(args.arm_a, args.arm_a_revision, args.arm_a_tree, False),
        "AP": arm_record(args.arm_ap, args.arm_ap_revision, args.arm_ap_tree, False),
        "B": arm_record(args.arm_b, args.arm_b_revision, args.arm_b_tree, True),
    }
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    rows_path = Path(args.rows)
    old_rows = read_csv(rows_path) if rows_path.is_file() else []
    wanted_binary = {}
    for row in old_rows:
        wanted_binary.setdefault(row["arm"], row["binary_sha256"])
    driver = Path(args.harness_root) / "bench/morse-weak8-gate/driver.rs"
    cache_root = Path(args.work_root) / "bin"
    cache_root.mkdir(parents=True, exist_ok=True)
    builds = {}
    for arm, path in ("A", args.arm_a), ("AP", args.arm_ap), ("B", args.arm_b):
        cached = cache_root / arm
        if (
            arm in wanted_binary
            and cached.is_file()
            and sha256_file(cached) == wanted_binary[arm]
        ):
            status = git(path, "status", "--porcelain").strip()
            if status:
                raise ProtocolError(f"arm {arm} is not clean")
            identities[arm]["git_status_porcelain_before_build"] = status
            identities[arm]["git_status_porcelain_after_build"] = status
            identities[arm]["binary_sha256"] = wanted_binary[arm]
            identities[arm]["build_command"] = ["reused cached release binary"]
            builds[arm] = cached
            continue
        binary, command, before, after = build_binary(
            arm, path, driver, args.toolchain, args.work_root, logs
        )
        digest = sha256_file(binary)
        if arm in wanted_binary and digest != wanted_binary[arm]:
            raise ProtocolError("a rebuilt arm does not match the rows file")
        shutil.copy2(binary, cached)
        identities[arm]["git_status_porcelain_before_build"] = before.strip()
        identities[arm]["git_status_porcelain_after_build"] = after.strip()
        identities[arm]["binary_sha256"] = digest
        identities[arm]["build_command"] = command
        builds[arm] = cached
    host = host_facts()
    sessions_path = Path(args.sessions)
    hosts_path = Path(args.hosts)
    known = hosts_path.is_file() and any(
        row["host_id"] == host["host_id"] for row in read_csv(hosts_path)
    )
    if not known:
        append_csv(hosts_path, HOST_FIELDS, host)
    entries = build_schedule(cells, args.rounds, args.seed0)
    validate_prefix(old_rows, entries, identities)
    provenance = harness_record(args.harness_root)
    session = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    serializable = {
        arm: {
            "source_revision": identity["source_revision"],
            "source_tree": identity["source_tree"],
            "binary_sha256": identity["binary_sha256"],
            "build_command": identity["build_command"],
            "git_status_porcelain_before_build": identity["git_status_porcelain_before_build"],
            "git_status_porcelain_after_build": identity["git_status_porcelain_after_build"],
        }
        for arm, identity in identities.items()
    }
    previous = load_sessions(sessions_path) if sessions_path.is_file() else []
    resumed = []
    for row in old_rows:
        if row["session"] not in resumed:
            resumed.append(row["session"])
    start = {
        "record": "start",
        "session": session,
        "utc": utc_now(),
        "host_id": host["host_id"],
        "toolchain": args.toolchain,
        "rustc": tool_line(["rustc", f"+{args.toolchain}", "--version"]),
        "cargo": tool_line(["cargo", f"+{args.toolchain}", "--version"]),
        "python": platform.python_version(),
        "taskset": tool_line([taskset, "--version"]),
        "run_environment": RUN_ENV,
        "core": args.core,
        "cells": cells,
        "rounds": args.rounds,
        "seed0": args.seed0,
        "order_rule": "six-order block: rotations of A A' B, then those orders reversed, repeated",
        "timeout_s": args.timeout_s,
        "planned": len(entries),
        "resumed_rows": len(old_rows),
        "resume_of": resumed,
        "prior_session_records": len(previous),
        "base_revision": BASE_REVISION,
        "base_tree": args.base_tree,
        "arms": serializable,
        **provenance,
    }
    append_jsonl(sessions_path, start)
    pending = entries[len(old_rows) :]
    print(f"session {session}: {len(pending)} runs to go of {len(entries)}", flush=True)
    completed = 0
    try:
        for entry in pending:
            row = run_one(
                entry,
                builds[entry["arm"]],
                taskset,
                args.core,
                args.timeout_s,
                logs,
                session,
                identities[entry["arm"]],
                host,
            )
            append_csv(rows_path, ROW_FIELDS, row)
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {entry['cell']} {entry['arm']} "
                f"seed {entry['seed']} hit={row['hit']} calls={row['evals_to_hit']} "
                f"wall={row['wall_s']}s",
                flush=True,
            )
    except ProtocolError as err:
        append_jsonl(
            sessions_path,
            {
                "record": "abort",
                "session": session,
                "utc": utc_now(),
                "reason": str(err),
                "completed": completed,
            },
        )
        raise
    final_rows = read_csv(rows_path)
    append_jsonl(
        sessions_path,
        {
            "record": "end",
            "session": session,
            "utc": utc_now(),
            "completed": completed,
            "rows_total": len(final_rows),
            "rows_sha256": sha256_file(rows_path),
            "arm_git_status_porcelain": {
                arm: git(path, "status", "--porcelain").strip()
                for arm, path in (
                    ("A", args.arm_a),
                    ("AP", args.arm_ap),
                    ("B", args.arm_b),
                )
            },
        },
    )
    print("session complete", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--harness-root", required=True)
    run.add_argument("--base-revision", required=True)
    run.add_argument("--base-tree", required=True)
    run.add_argument("--arm-a", required=True)
    run.add_argument("--arm-a-revision", required=True)
    run.add_argument("--arm-a-tree", required=True)
    run.add_argument("--arm-ap", required=True)
    run.add_argument("--arm-ap-revision", required=True)
    run.add_argument("--arm-ap-tree", required=True)
    run.add_argument("--arm-b", required=True)
    run.add_argument("--arm-b-revision", required=True)
    run.add_argument("--arm-b-tree", required=True)
    run.add_argument("--toolchain", required=True)
    run.add_argument("--work-root", required=True)
    run.add_argument("--core", type=int, required=True)
    run.add_argument("--cells", required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--seed0", type=int, required=True)
    run.add_argument("--timeout-s", type=float, required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--sessions", required=True)
    run.add_argument("--hosts", required=True)
    run.add_argument("--logs", required=True)
    run.set_defaults(func=cmd_run)
    test = sub.add_parser("selftest")
    test.set_defaults(func=lambda _args: selftest())
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except ProtocolError as err:
        print(f"runner: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
