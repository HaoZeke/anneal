#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run and report the paired weak6 source-tree gate.

The three arms are separate release builds. A and A' are independent builds of
the base source tree; B is a build of the source tree after the two weak6
patches. Every arm runs the same seed and start for each cell. Across successive
seeds the relative A/B order is ABBA, with A' interleaved.

The runner is resumable only from an exact prefix of its rows file. It never
retries, replaces, warms up, or reduces a run.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

EXAMPLE = "morse_weak6_gate"
DRIVER_PATH = "examples/morse_weak6_gate.rs"
HARNESS_PATH = "scripts/morse_weak6_gate/harness.py"
ALLOWED_CHANGES = (DRIVER_PATH, HARNESS_PATH)

ARMS = ("A", "AP", "B")
ARM_LABEL = {"A": "A", "AP": "A'", "B": "B"}
ARM_MODE = {"A": "base", "AP": "base", "B": "weak6"}

# name -> display name, kind, atoms, rho, Cambridge minimum, paper budget.
CELLS = {
    "lj75": ("LJ75", "lj", 75, "na", -397.492331, 4_000_000),
    "lj98": ("LJ98", "lj", 98, "na", -543.665361, 4_000_000),
    "m75r6": ("M75 rho 6", "morse", 75, "6.0", -351.472365, 4_000_000),
    "m75r14": ("M75 rho 14", "morse", 75, "14.0", -318.407330, 4_000_000),
    "m80r6": ("M80 rho 6", "morse", 80, "6.0", -378.333471, 4_000_000),
    "m80r14": ("M80 rho 14", "morse", 80, "14.0", -340.811371, 4_000_000),
    "lj38": ("LJ38", "lj", 38, "na", -173.928427, 400_000),
    "m38r10": ("m38r10", "morse", 38, "10.0", -145.849817, 400_000),
}

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}
RUN_ENV = {**THREAD_ENV, "LJOS_SEAT": "benchbot"}
BAD_BUILD_ENV = (
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_RUSTC_WRAPPER",
)
MIN_ROUNDS = 12
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261119
MASK64 = (1 << 64) - 1

DRIVER_FIELDS = (
    "cell",
    "mode",
    "seed",
    "n",
    "kind",
    "rho",
    "budget",
    "reference",
    "replicas",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "best_bits",
    "wall_s",
    "pot_s",
    "outside_s",
    "cpu_ns",
    "wait_ns",
    "vol_ctxt",
    "invol_ctxt",
    "threads",
    "core_start",
    "core_end",
    "cpus_allowed",
    "cpus_allowed_list",
    "start_sha256",
    "config_sha256",
)
ROW_FIELDS = (
    "session",
    "sequence",
    "cell",
    "arm",
    "mode",
    "round",
    "slot",
    "order",
    "seed",
    "n",
    "kind",
    "rho",
    "budget",
    "reference",
    "replicas",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "best_bits",
    "wall_s",
    "pot_s",
    "outside_s",
    "proc_s",
    "cpu_ns",
    "wait_ns",
    "vol_ctxt",
    "invol_ctxt",
    "threads",
    "core_pinned",
    "core_busy_ticks",
    "core_steal_ticks",
    "core_start",
    "core_end",
    "cpus_allowed",
    "cpus_allowed_list",
    "start_sha256",
    "config_sha256",
    "source_revision",
    "source_tree",
    "binary_sha256",
    "host_id",
    "utc_start",
    "utc_end",
)
INT_FIELDS = (
    "sequence",
    "round",
    "slot",
    "seed",
    "n",
    "budget",
    "replicas",
    "hit",
    "evals_to_hit",
    "charged",
    "cpu_ns",
    "wait_ns",
    "vol_ctxt",
    "invol_ctxt",
    "threads",
    "core_pinned",
    "core_busy_ticks",
    "core_steal_ticks",
    "core_start",
    "core_end",
)
FLOAT_FIELDS = ("reference", "wall_s", "pot_s", "outside_s", "proc_s")
HOST_FIELDS = (
    "host_id",
    "cpu_model",
    "logical_cpus",
    "mem_total_kb",
    "arch",
    "kernel_release",
    "os_release",
)


class Abort(Exception):
    """The declared protocol cannot continue."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
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


def append_csv(path, fields, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n", extrasaction="raise"
        )
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_sessions(path):
    with open(path, newline="") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --- statistics ---------------------------------------------------------------


def type7(sorted_values, p):
    if not sorted_values:
        raise ValueError("a quantile needs a sample")
    h = (len(sorted_values) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def median(values):
    return type7(sorted(values), 0.5)


def fisher_two_sided(hits_a, n_a, hits_b, n_b):
    hits = hits_a + hits_b

    def weight(x):
        return math.comb(n_a, x) * math.comb(n_b, hits - x)

    observed = weight(hits_a)
    mass = sum(
        weight(x)
        for x in range(max(0, hits - n_b), min(n_a, hits) + 1)
        if weight(x) <= observed
    )
    return mass / math.comb(n_a + n_b, hits)


def mann_whitney_two_sided(x, y):
    if not x or not y:
        return None
    n1, n2 = len(x), len(y)
    n = n1 + n2
    pooled = list(x) + list(y)
    order = sorted(range(n), key=lambda i: pooled[i])
    ranks = [0.0] * n
    tie_term = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[order[j + 1]] == pooled[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        size = j - i + 1
        tie_term += size**3 - size
        i = j + 1
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2
    u = max(u1, n1 * n2 - u1)
    variance = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if variance <= 0:
        return 1.0
    z = (u - n1 * n2 / 2 - 0.5) / math.sqrt(variance)
    return min(1.0, math.erfc(z / math.sqrt(2)))


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, bound):
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def bootstrap_medians(values, draws, rng):
    if not values:
        return []
    n = len(values)
    result = []
    for _ in range(draws):
        result.append(median([values[rng.below(n)] for _ in range(n)]))
    result.sort()
    return result


def median_ci(values, seed):
    if not values:
        return None
    draws = bootstrap_medians(values, BOOTSTRAP_DRAWS, SplitMix64(seed))
    return {
        "median": median(values),
        "bootstrap_95_ci": [type7(draws, 0.025), type7(draws, 0.975)],
        "n": len(values),
    }


def aa_ratio_band(values_a, values_ap, seed):
    if not values_a or not values_ap:
        return None
    rng = SplitMix64(seed)
    ratios = []
    for _ in range(BOOTSTRAP_DRAWS):
        ma = median([values_a[rng.below(len(values_a))] for _ in values_a])
        map_ = median([values_ap[rng.below(len(values_ap))] for _ in values_ap])
        if ma > 0:
            ratios.append(map_ / ma)
    ratios.sort()
    return [type7(ratios, 0.025), type7(ratios, 0.975)]


# --- schedule -----------------------------------------------------------------


def arm_order(round_index):
    orders = (
        ("A", "AP", "B"),
        ("B", "AP", "A"),
        ("AP", "B", "A"),
        ("A", "B", "AP"),
    )
    return orders[round_index % len(orders)]


def build_schedule(cells, rounds, seed0):
    entries = []
    sequence = 0
    for cell in cells:
        for round_index in range(rounds):
            order = arm_order(round_index)
            for slot, arm in enumerate(order):
                entries.append(
                    {
                        "sequence": sequence,
                        "cell": cell,
                        "round": round_index,
                        "slot": slot,
                        "arm": arm,
                        "seed": seed0 + round_index,
                        "order": ">".join(order),
                    }
                )
                sequence += 1
    return entries


# --- provenance and builds ----------------------------------------------------


def git(tree, *args):
    done = subprocess.run(
        ["git", "-C", str(tree), *args], capture_output=True, text=True
    )
    if done.returncode != 0:
        raise Abort(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def harness_provenance(tree, base_rev, base_tree):
    status = git(tree, "status", "--porcelain")
    if status:
        raise Abort("the harness worktree is not clean")
    base = git(tree, "rev-parse", "--verify", f"{base_rev}^{{commit}}").strip()
    found_tree = git(tree, "rev-parse", f"{base}^{{tree}}").strip()
    if found_tree != base_tree:
        raise Abort("the base commit does not carry the expected tree")
    head = git(tree, "rev-parse", "HEAD").strip()
    head_tree = git(tree, "rev-parse", "HEAD^{tree}").strip()
    changed = tuple(
        line
        for line in git(tree, "diff", "--name-only", base, "HEAD").splitlines()
        if line
    )
    if changed != tuple(sorted(ALLOWED_CHANGES)):
        raise Abort("the harness commit changes paths outside the two harness files")
    for path in ALLOWED_CHANGES:
        blob = git(tree, "show", f"HEAD:{path}").encode()
        if hashlib.sha256(blob).hexdigest() != sha256_file(Path(tree) / path):
            raise Abort(f"{path} is not the committed copy")
    return {
        "base_revision": base,
        "base_tree": found_tree,
        "harness_revision": head,
        "harness_tree": head_tree,
        "harness_sha256": sha256_file(Path(tree) / HARNESS_PATH),
        "driver_sha256": sha256_file(Path(tree) / DRIVER_PATH),
        "changed_paths": list(changed),
        "git_status_porcelain": status,
    }


def arm_provenance(path, expected_revision, expected_tree):
    status = git(path, "status", "--porcelain")
    revision = git(path, "rev-parse", "HEAD").strip()
    tree = git(path, "rev-parse", "HEAD^{tree}").strip()
    if status:
        raise Abort("an arm worktree is not clean")
    if revision != expected_revision or tree != expected_tree:
        raise Abort("an arm does not have its declared revision and tree")
    return {
        "source_revision": revision,
        "source_tree": tree,
        "git_status_porcelain_before_build": status,
    }


def reject_build_environment():
    for name in os.environ:
        if (
            "RUSTFLAGS" in name
            or name.startswith("CARGO_PROFILE_")
            or name in BAD_BUILD_ENV
            or name.endswith("_LINKER")
        ):
            raise Abort(f"{name} is set; release builds must use the stock profile")


def runner_manifest(source_tree):
    source_tree = str(Path(source_tree).resolve()).replace("\\", "\\\\").replace('"', '\\"')
    return f"""[package]
name = "morse-weak6-gate-runner"
version = "0.0.0"
edition = "2024"

[features]
default = []
weak6 = []

[dependencies]
anneal-core = {{ path = "{source_tree}" }}
ndarray = "0.17"
rand = "0.9"
sha2 = "0.10"
rgpot-pair = {{ git = "https://github.com/OmniPotentRPC/rgpot", rev = "8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e" }}
"""


def run_logged(cmd, cwd, log):
    started = utc_now()
    done = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, errors="replace")
    write_text(
        log,
        f"started: {started}\nended: {utc_now()}\nexit: {done.returncode}\n"
        f"--- stdout\n{done.stdout}--- stderr\n{done.stderr}",
    )
    if done.returncode != 0:
        raise Abort(f"release build command exited with {done.returncode}")


def build_binary(arm, source_tree, driver, toolchain, work_root, logs):
    runner = Path(work_root) / "build" / arm
    source = runner / "src" / "main.rs"
    write_text(runner / "Cargo.toml", runner_manifest(source_tree))
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(driver, source)
    lock_cmd = [
        "cargo",
        f"+{toolchain}",
        "generate-lockfile",
        "--manifest-path",
        str(runner / "Cargo.toml"),
    ]
    run_logged(lock_cmd, runner, Path(logs) / f"build-{arm}-lock.log")
    build_cmd = [
        "cargo",
        f"+{toolchain}",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        str(runner / "Cargo.toml"),
        "--target-dir",
        str(runner / "target"),
    ]
    if arm == "B":
        build_cmd.extend(["--features", "weak6"])
    run_logged(build_cmd, runner, Path(logs) / f"build-{arm}.log")
    binary = runner / "target" / "release" / "morse-weak6-gate-runner"
    if not binary.is_file():
        raise Abort(f"arm {arm} produced no release binary")
    shown = [
        "cargo",
        f"+{toolchain}",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        "<runner-manifest>",
        "--target-dir",
        "<target-dir>",
    ]
    if arm == "B":
        shown.extend(["--features", "weak6"])
    return binary, shown


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
    machine_id = ""
    for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        if Path(candidate).is_file():
            machine_id = Path(candidate).read_text().strip()
            break
    if not machine_id:
        raise Abort("no machine id to hash")
    digest = hashlib.sha256(
        ("anneal-morse-weak6-gate:" + machine_id).encode()
    ).hexdigest()[:16]
    return {
        "host_id": digest,
        "cpu_model": cpu_model,
        "logical_cpus": str(os.cpu_count()),
        "mem_total_kb": mem_kb,
        "arch": platform.machine(),
        "kernel_release": platform.release(),
        "os_release": os_name,
    }


def tool_lines(*cmd):
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise Abort(f"{' '.join(cmd)}: {done.stderr.strip()}")
    return done.stdout.strip().splitlines()


# --- runs ---------------------------------------------------------------------


def core_ticks(core):
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == f"cpu{core}":
            user, nice, system, _idle, _iowait, irq, softirq, steal = (
                int(x) for x in fields[1:9]
            )
            return user + nice + system + irq + softirq, steal
    raise Abort(f"no cpu{core} line in /proc/stat")


def parse_gate_row(stdout):
    lines = [line for line in stdout.splitlines() if line.startswith("gate_row ")]
    if len(lines) != 1:
        raise Abort(f"expected one gate_row line, found {len(lines)}")
    fields = {}
    for item in lines[0].split()[1:]:
        key, separator, value = item.partition("=")
        if not separator or key in fields:
            raise Abort(f"malformed field {item!r}")
        fields[key] = value
    missing = [key for key in DRIVER_FIELDS if key not in fields]
    extra = [key for key in fields if key not in DRIVER_FIELDS]
    if missing or extra:
        raise Abort(f"driver fields differ: missing {missing} extra {extra}")
    return fields


def check_run(entry, fields, core):
    _display, kind, n, rho, reference, budget = CELLS[entry["cell"]]
    expected = {
        "cell": entry["cell"],
        "mode": ARM_MODE[entry["arm"]],
        "seed": str(entry["seed"]),
        "n": str(n),
        "kind": kind,
        "rho": rho,
        "budget": str(budget),
        "reference": f"{reference:.6f}",
    }
    for key, value in expected.items():
        if fields[key] != value:
            raise Abort(f"{key} is {fields[key]!r}, expected {value!r}")
    hit = int(fields["hit"])
    calls = int(fields["evals_to_hit"])
    charged = int(fields["charged"])
    if hit not in (0, 1) or not 1 <= calls <= budget or not 1 <= charged <= budget:
        raise Abort("the calls or hit flag are out of range")
    if hit == 0 and calls != budget:
        raise Abort("a miss must report the full budget as calls to hit")
    if fields["threads"] != "1":
        raise Abort("the process did not have exactly one thread")
    if not (
        fields["core_start"] == fields["core_end"] == str(core)
        and fields["cpus_allowed_list"] == str(core)
    ):
        raise Abort("the run was not pinned to the requested core")


def run_one(entry, binary, taskset, core, timeout_s, logs, session, identity, host_id):
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
    mark = time.perf_counter()
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            env=RUN_ENV,
            cwd=str(Path(logs)),
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as err:
        write_text(log, f"timeout after {timeout_s} seconds\n")
        raise Abort(
            f"{entry['cell']} {entry['arm']} {entry['seed']} timed out"
        ) from err
    proc_s = time.perf_counter() - mark
    ended = utc_now()
    busy_after, steal_after = core_ticks(core)
    write_text(
        log,
        f"command: taskset -c {core} <binary> run {entry['cell']} {entry['seed']}\n"
        f"started: {started}\nended: {ended}\nexit: {done.returncode}\n"
        f"proc_s: {proc_s:.6f}\n--- stdout\n{done.stdout}--- stderr\n{done.stderr}",
    )
    if done.returncode != 0:
        raise Abort(
            f"{entry['cell']} {entry['arm']} {entry['seed']} exited with "
            f"{done.returncode}"
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
            "proc_s": f"{proc_s:.6f}",
            "core_pinned": core,
            "core_busy_ticks": busy_after - busy_before,
            "core_steal_ticks": steal_after - steal_before,
            "source_revision": identity["source_revision"],
            "source_tree": identity["source_tree"],
            "binary_sha256": identity["binary_sha256"],
            "host_id": host_id,
            "utc_start": started,
            "utc_end": ended,
        }
    )
    return row


def validate_prefix(old_rows, entries, identities):
    if len(old_rows) > len(entries):
        raise Abort("the rows file is longer than this schedule")
    for index, old in enumerate(old_rows):
        entry = entries[index]
        expected = {
            "sequence": str(entry["sequence"]),
            "cell": entry["cell"],
            "arm": entry["arm"],
            "round": str(entry["round"]),
            "slot": str(entry["slot"]),
            "order": entry["order"],
            "seed": str(entry["seed"]),
            "source_revision": identities[entry["arm"]]["source_revision"],
            "source_tree": identities[entry["arm"]]["source_tree"],
            "binary_sha256": identities[entry["arm"]]["binary_sha256"],
        }
        for key, value in expected.items():
            if old.get(key) != value:
                raise Abort(f"the rows file is not an exact schedule prefix at row {index}")


def cmd_run(args):
    cells = [name for name in args.cells.split(",") if name]
    unknown = [name for name in cells if name not in CELLS]
    if unknown or not cells or len(cells) != len(set(cells)):
        raise Abort(f"cells must be distinct names from {sorted(CELLS)}")
    if args.rounds < MIN_ROUNDS:
        raise Abort(f"the gate requires at least {MIN_ROUNDS} seeds per cell")
    if args.core in os.sched_getaffinity(0):
        raise Abort("the harness itself is allowed on the benchmark core")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise Abort("taskset is not installed")
    reject_build_environment()
    harness_tree = Path(args.harness_tree).resolve()
    work_root = Path(args.work_root).resolve()
    logs = Path(args.logs).resolve()
    rows_path = Path(args.rows).resolve()
    sessions_path = Path(args.sessions).resolve()
    hosts_path = Path(args.hosts).resolve()
    for path in (work_root, logs, rows_path, sessions_path, hosts_path):
        if path == harness_tree or harness_tree in path.parents:
            raise Abort("run output must be outside the harness worktree")
    logs.mkdir(parents=True, exist_ok=True)
    provenance = harness_provenance(
        harness_tree, args.base_revision, args.base_tree
    )
    arm_args = {
        "A": (args.arm_a, args.arm_a_revision, args.arm_a_tree),
        "AP": (args.arm_ap, args.arm_ap_revision, args.arm_ap_tree),
        "B": (args.arm_b, args.arm_b_revision, args.arm_b_tree),
    }
    identities = {
        arm: arm_provenance(path, revision, tree)
        for arm, (path, revision, tree) in arm_args.items()
    }
    builds = {}
    for arm in ARMS:
        path = arm_args[arm][0]
        binary, command = build_binary(
            arm,
            path,
            harness_tree / DRIVER_PATH,
            args.toolchain,
            work_root,
            logs,
        )
        status = git(path, "status", "--porcelain")
        if status:
            raise Abort(f"the arm {arm} worktree is unclean after its build")
        identities[arm]["git_status_porcelain_after_build"] = status
        identities[arm]["binary_sha256"] = sha256_file(binary)
        identities[arm]["build_command"] = command
        identities[arm]["binary"] = binary
        builds[arm] = binary
    host = host_facts()
    known_host = (
        hosts_path.is_file()
        and any(row["host_id"] == host["host_id"] for row in read_csv(hosts_path))
    )
    if not known_host:
        append_csv(hosts_path, HOST_FIELDS, host)
    entries = build_schedule(cells, args.rounds, args.seed0)
    old_rows = read_csv(rows_path) if rows_path.is_file() else []
    validate_prefix(old_rows, entries, identities)
    previous_sessions = load_sessions(sessions_path) if sessions_path.is_file() else []
    resumed = []
    for row in old_rows:
        if row["session"] not in resumed:
            resumed.append(row["session"])
    session = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    serializable_arms = {
        arm: {key: value for key, value in identity.items() if key != "binary"}
        for arm, identity in identities.items()
    }
    start = {
        "record": "start",
        "session": session,
        "utc": utc_now(),
        "host_id": host["host_id"],
        "toolchain": args.toolchain,
        "rustc": tool_lines("rustc", f"+{args.toolchain}", "--version", "--verbose"),
        "cargo": tool_lines("cargo", f"+{args.toolchain}", "--version")[0],
        "python": platform.python_version(),
        "taskset": tool_lines(taskset, "--version")[0],
        "run_environment": RUN_ENV,
        "core": args.core,
        "harness_affinity": sorted(os.sched_getaffinity(0)),
        "cells": cells,
        "rounds": args.rounds,
        "seed0": args.seed0,
        "order_rule": "successive seeds use AB, BA, BA, AB for A/B; A' is interleaved",
        "timeout_s": args.timeout_s,
        "planned": len(entries),
        "resumed_rows": len(old_rows),
        "resume_of": resumed,
        "prior_session_records": len(previous_sessions),
        "arms": serializable_arms,
        **provenance,
    }
    append_jsonl(sessions_path, start)
    pending = entries[len(old_rows) :]
    print(f"session {session}: {len(pending)} runs to go of {len(entries)}", flush=True)
    completed = 0
    try:
        for entry in pending:
            arm = entry["arm"]
            row = run_one(
                entry,
                builds[arm],
                taskset,
                args.core,
                args.timeout_s,
                logs,
                session,
                identities[arm],
                host["host_id"],
            )
            append_csv(rows_path, ROW_FIELDS, row)
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {entry['cell']} {arm} "
                f"seed {entry['seed']} hit={row['hit']} "
                f"calls={row['evals_to_hit']} wall={row['wall_s']}s",
                flush=True,
            )
    except Abort as err:
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
                arm: git(arm_args[arm][0], "status", "--porcelain")
                for arm in ARMS
            },
        },
    )
    print("session complete", flush=True)


# --- report -------------------------------------------------------------------


def load_rows(path):
    rows = read_csv(path)
    for row in rows:
        for key in INT_FIELDS:
            row[key] = int(row[key])
        for key in FLOAT_FIELDS:
            row[key] = float(row[key])
    return rows


def group_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["cell"], row["arm"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["seed"])
    return groups


def arm_numbers(rows, seed):
    hit_calls = [row["evals_to_hit"] for row in rows if row["hit"]]
    return {
        "n": len(rows),
        "hits": sum(row["hit"] for row in rows),
        "hit_calls": median_ci(hit_calls, seed),
        "total_calls": sum(row["charged"] for row in rows),
        "wall_median_s": median([row["wall_s"] for row in rows]),
    }


def cell_numbers(cell, groups):
    by_arm = {arm: groups[(cell, arm)] for arm in ARMS}
    seed = (BOOTSTRAP_SEED * 1_000_003 + zlib.crc32(cell.encode())) & MASK64
    arms = {
        arm: arm_numbers(by_arm[arm], seed + index * 100_003)
        for index, arm in enumerate(ARMS)
    }
    calls = {
        arm: [row["evals_to_hit"] for row in by_arm[arm] if row["hit"]]
        for arm in ARMS
    }
    n = arms["A"]["n"]
    fisher_ba = fisher_two_sided(arms["A"]["hits"], n, arms["B"]["hits"], n)
    fisher_apa = fisher_two_sided(arms["A"]["hits"], n, arms["AP"]["hits"], n)
    mw_ba = mann_whitney_two_sided(calls["A"], calls["B"])
    mw_apa = mann_whitney_two_sided(calls["A"], calls["AP"])
    band = aa_ratio_band(calls["A"], calls["AP"], seed + 700_021)
    median_a = arms["A"]["hit_calls"]["median"] if arms["A"]["hit_calls"] else None
    median_b = arms["B"]["hit_calls"]["median"] if arms["B"]["hit_calls"] else None
    ratio_ba = (
        median_b / median_a
        if median_a is not None and median_b is not None and median_a > 0
        else None
    )
    by_seed_b = {row["seed"]: row for row in by_arm["B"]}
    paired = [(row, by_seed_b[row["seed"]]) for row in by_arm["A"]]
    identity = [
        left["best_bits"] == right["best_bits"]
        and left["charged"] == right["charged"]
        for left, right in paired
    ]
    return {
        "display": CELLS[cell][0],
        "arms": arms,
        "fisher_p_b_vs_a": fisher_ba,
        "fisher_p_ap_vs_a": fisher_apa,
        "mann_whitney_p_hit_calls_b_vs_a": mw_ba,
        "mann_whitney_p_hit_calls_ap_vs_a": mw_apa,
        "aa_bootstrap_95_band_hit_call_median_ratio_ap_over_a": band,
        "hit_call_median_ratio_b_over_a": ratio_ba,
        "ab_final_energy_and_charged_calls_identical": {
            "count": sum(identity),
            "n": len(identity),
            "all": all(identity),
        },
    }


def data_checks(rows, sessions, groups, cells, rounds):
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    starts = [record for record in sessions if record["record"] == "start"]
    ends = {record["session"] for record in sessions if record["record"] == "end"}
    aborts = [record for record in sessions if record["record"] == "abort"]
    check(
        "every session ended without abort",
        bool(starts)
        and all(record["session"] in ends for record in starts)
        and not aborts,
        f"{len(starts)} start, {len(ends)} end, {len(aborts)} abort",
    )
    expected = len(cells) * rounds * len(ARMS)
    check("complete schedule", len(rows) == expected, f"{len(rows)} of {expected} rows")
    check(
        "one row per sequence",
        [row["sequence"] for row in rows] == list(range(len(rows))),
    )
    check(
        "same run count in every cell and arm",
        all(len(groups.get((cell, arm), [])) == rounds for cell in cells for arm in ARMS),
    )
    check(
        "same seeds and starts in every arm",
        all(
            {
                (row["seed"], row["start_sha256"])
                for row in groups.get((cell, "A"), [])
            }
            == {
                (row["seed"], row["start_sha256"])
                for row in groups.get((cell, arm), [])
            }
            for cell in cells
            for arm in ("AP", "B")
        ),
    )
    check(
        "ABBA order with A' interleaved",
        all(row["order"] == ">".join(arm_order(row["round"])) for row in rows),
    )
    check(
        "one pinned core and one thread",
        all(
            row["core_pinned"]
            == row["core_start"]
            == row["core_end"]
            == starts[0]["core"]
            and row["cpus_allowed_list"] == str(starts[0]["core"])
            and row["threads"] == 1
            for row in rows
        )
        if starts
        else False,
    )
    check(
        "arm source identities and binaries are stable",
        all(
            len(
                {
                    (
                        row["source_revision"],
                        row["source_tree"],
                        row["binary_sha256"],
                    )
                    for row in rows
                    if row["arm"] == arm
                }
            )
            == 1
            for arm in ARMS
        ),
    )
    check(
        "all arm worktree status records are empty",
        all(
            not identity["git_status_porcelain_before_build"]
            and not identity["git_status_porcelain_after_build"]
            for start in starts
            for identity in start["arms"].values()
        )
        and all(
            not status
            for end in sessions
            if end["record"] == "end"
            for status in end["arm_git_status_porcelain"].values()
        ),
    )
    return checks


def compute_summary(rows, sessions):
    starts = [record for record in sessions if record["record"] == "start"]
    if not starts:
        raise Abort("no start session")
    cells = starts[-1]["cells"]
    rounds = starts[-1]["rounds"]
    groups = group_rows(rows)
    missing = [
        (cell, arm)
        for cell in cells
        for arm in ARMS
        if (cell, arm) not in groups
    ]
    if missing:
        raise Abort(f"missing cell-arm groups: {missing}")
    per_cell = {cell: cell_numbers(cell, groups) for cell in cells}
    checks = data_checks(rows, sessions, groups, cells, rounds)
    loss_hits = []
    loss_calls = []
    wins = []
    marginal_losses = []
    bonferroni = 0.05 / len(cells)
    for cell, stats in per_cell.items():
        a = stats["arms"]["A"]
        b = stats["arms"]["B"]
        fisher = stats["fisher_p_b_vs_a"]
        mw = stats["mann_whitney_p_hit_calls_b_vs_a"]
        ratio = stats["hit_call_median_ratio_b_over_a"]
        band = stats["aa_bootstrap_95_band_hit_call_median_ratio_ap_over_a"]
        if b["hits"] < a["hits"] and fisher < 0.05:
            loss_hits.append({"cell": cell, "p": fisher})
        if (
            ratio is not None
            and band is not None
            and ratio > 1.0
            and ratio > band[1]
            and mw is not None
            and mw < 0.05
        ):
            loss_calls.append(
                {"cell": cell, "p": mw, "ratio": ratio, "aa_band": band}
            )
        if b["hits"] > a["hits"] and fisher < bonferroni:
            wins.append(
                {
                    "cell": cell,
                    "metric": "hits",
                    "p": fisher,
                    "bonferroni_p": min(1.0, fisher * len(cells)),
                }
            )
        if ratio is not None and ratio < 1.0 and mw is not None and mw < bonferroni:
            wins.append(
                {
                    "cell": cell,
                    "metric": "hit calls",
                    "p": mw,
                    "bonferroni_p": min(1.0, mw * len(cells)),
                }
            )
        if b["hits"] < a["hits"] and 0.05 <= fisher < 0.1:
            marginal_losses.append({"cell": cell, "metric": "hits", "p": fisher})
        if ratio is not None and ratio > 1.0 and mw is not None and 0.05 <= mw < 0.1:
            marginal_losses.append(
                {
                    "cell": cell,
                    "metric": "hit calls",
                    "p": mw,
                    "outside_aa_band": band is not None and ratio > band[1],
                }
            )
    integrity = all(check["pass"] for check in checks)
    verdict = "PASS" if integrity and not loss_hits and not loss_calls else "FAIL"
    resumed_rows = sum(start.get("resumed_rows", 0) for start in starts)
    return {
        "verdict": verdict,
        "rule": (
            "PASS only when B has no cell with fewer hits at two-sided Fisher "
            "p<0.05 and no cell with a higher hit-call median at two-sided "
            "Mann-Whitney p<0.05 above the A/A' bootstrap band"
        ),
        "cells": per_cell,
        "gate_losses": {"hits": loss_hits, "hit_calls": loss_calls},
        "wins_bonferroni_8_cells": wins,
        "marginal_losses_0.05_to_0.1": marginal_losses,
        "bonferroni_threshold": bonferroni,
        "checks": checks,
        "resume": {
            "resumed": resumed_rows > 0,
            "resumed_rows": resumed_rows,
            "session_count": len(starts),
        },
    }


def fmt_p(value):
    if value is None:
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def fmt_int(value):
    return f"{value:.0f}"


def fmt_calls(summary):
    if summary is None:
        return "none"
    lo, hi = summary["bootstrap_95_ci"]
    return (
        f"{fmt_int(summary['median'])} "
        f"[{fmt_int(lo)}, {fmt_int(hi)}] (n={summary['n']})"
    )


def md_table(header, body):
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def render_report(summary, sessions, hosts):
    starts = [record for record in sessions if record["record"] == "start"]
    last = starts[-1]
    host = next(row for row in hosts if row["host_id"] == last["host_id"])
    lines = [
        "# Morse weak6 paired gate",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        summary["rule"] + ".",
        "",
        "- A and A' are separate release builds of the base source tree. B is the two-patch weak6 tree.",
        "- All arms use the same 12 seeds and starts per cell. The A/B order follows ABBA across successive seeds, with A' interleaved.",
        "- Runs are serial, one thread, pinned to one core, with no warm-up, retry, replacement, quick mode, or reduced budget.",
        "- Calls-to-hit statistics include hitting runs only. Confidence intervals are 95% percentile bootstrap intervals with 20000 fixed-stream draws.",
        "- The calls gate compares B/A's hit-call median ratio with the 95% bootstrap band of A'/A.",
        f"- Resume: {'yes' if summary['resume']['resumed'] else 'no'}"
        + (
            f" ({summary['resume']['resumed_rows']} rows)"
            if summary["resume"]["resumed"]
            else ""
        )
        + ".",
        "",
        "## Provenance",
        "",
    ]
    provenance = [
        ("base revision", last["base_revision"]),
        ("base tree", last["base_tree"]),
        ("harness revision", last["harness_revision"]),
        ("harness tree", last["harness_tree"]),
        ("harness sha256", last["harness_sha256"]),
        ("driver sha256", last["driver_sha256"]),
        ("host id (machine-id hash)", last["host_id"]),
        ("cpu", host["cpu_model"]),
        ("logical cpus", host["logical_cpus"]),
        ("memory kB", host["mem_total_kb"]),
        ("kernel", host["kernel_release"]),
        ("os", host["os_release"]),
        ("arch", host["arch"]),
        ("rustc", last["rustc"][0]),
        ("cargo", last["cargo"]),
        ("python", last["python"]),
        ("taskset", last["taskset"]),
        ("pinned core", str(last["core"])),
        (
            "run environment",
            " ".join(f"{key}={value}" for key, value in sorted(RUN_ENV.items())),
        ),
    ]
    lines += md_table(
        ["field", "value"],
        [[key, f"`{value}`" if value else "empty"] for key, value in provenance],
    )
    lines += ["", "Arm builds:", ""]
    build_rows = []
    for arm in ARMS:
        identity = last["arms"][arm]
        build_rows.append(
            [
                ARM_LABEL[arm],
                identity["source_revision"],
                identity["source_tree"],
                identity["binary_sha256"],
                "empty"
                if not identity["git_status_porcelain_before_build"]
                else "nonempty",
                "empty"
                if not identity["git_status_porcelain_after_build"]
                else "nonempty",
            ]
        )
    lines += md_table(
        [
            "arm",
            "source revision",
            "source tree",
            "binary sha256",
            "porcelain before",
            "porcelain after",
        ],
        build_rows,
    )
    lines += ["", "## Per-cell results", ""]
    body = []
    for _cell, stats in summary["cells"].items():
        arms = stats["arms"]
        body.append(
            [
                stats["display"],
                f"{arms['A']['hits']}/{arms['A']['n']}",
                f"{arms['AP']['hits']}/{arms['AP']['n']}",
                f"{arms['B']['hits']}/{arms['B']['n']}",
                fmt_p(stats["fisher_p_b_vs_a"]),
                fmt_p(stats["fisher_p_ap_vs_a"]),
                fmt_calls(arms["A"]["hit_calls"]),
                fmt_calls(arms["AP"]["hit_calls"]),
                fmt_calls(arms["B"]["hit_calls"]),
                fmt_p(stats["mann_whitney_p_hit_calls_b_vs_a"]),
                fmt_p(stats["mann_whitney_p_hit_calls_ap_vs_a"]),
            ]
        )
    lines += md_table(
        [
            "cell",
            "A hits",
            "A' hits",
            "B hits",
            "Fisher B/A",
            "Fisher A'/A",
            "A hit calls median [boot CI]",
            "A' hit calls median [boot CI]",
            "B hit calls median [boot CI]",
            "MW B/A",
            "MW A'/A",
        ],
        body,
    )
    lines += ["", "Totals, wall medians, A/A' bands, and bit identity:", ""]
    body = []
    for _cell, stats in summary["cells"].items():
        arms = stats["arms"]
        band = stats["aa_bootstrap_95_band_hit_call_median_ratio_ap_over_a"]
        identity = stats["ab_final_energy_and_charged_calls_identical"]
        body.append(
            [
                stats["display"],
                f"{arms['A']['total_calls']} / {arms['AP']['total_calls']} / {arms['B']['total_calls']}",
                f"{arms['A']['wall_median_s']:.2f} / {arms['AP']['wall_median_s']:.2f} / {arms['B']['wall_median_s']:.2f}",
                (
                    f"{stats['hit_call_median_ratio_b_over_a']:.3f}"
                    if stats["hit_call_median_ratio_b_over_a"] is not None
                    else "n/a"
                ),
                f"[{band[0]:.3f}, {band[1]:.3f}]" if band else "n/a",
                f"{identity['count']}/{identity['n']}",
            ]
        )
    lines += md_table(
        [
            "cell",
            "total charged calls A / A' / B",
            "wall median s A / A' / B",
            "hit-call median B/A",
            "A/A' bootstrap band",
            "A/B final-energy bits + calls identical",
        ],
        body,
    )
    all_identical = [
        stats["display"]
        for stats in summary["cells"].values()
        if stats["ab_final_energy_and_charged_calls_identical"]["all"]
    ]
    lines += [
        "",
        "Cells bit-identical in final energy and charged calls for every A/B seed: "
        + (", ".join(all_identical) if all_identical else "none")
        + ".",
        "",
        "## Gate decision",
        "",
        "Significant hit losses: "
        + (json.dumps(summary["gate_losses"]["hits"], sort_keys=True) if summary["gate_losses"]["hits"] else "none")
        + ".",
        "",
        "Significant hit-call losses outside the A/A' band: "
        + (json.dumps(summary["gate_losses"]["hit_calls"], sort_keys=True) if summary["gate_losses"]["hit_calls"] else "none")
        + ".",
        "",
        "Bonferroni wins over eight cells: "
        + (json.dumps(summary["wins_bonferroni_8_cells"], sort_keys=True) if summary["wins_bonferroni_8_cells"] else "none")
        + ".",
        "",
        "Marginal losses (0.05 <= p < 0.1): "
        + (json.dumps(summary["marginal_losses_0.05_to_0.1"], sort_keys=True) if summary["marginal_losses_0.05_to_0.1"] else "none")
        + ".",
        "",
        "## Checks",
        "",
    ]
    for check in summary["checks"]:
        suffix = f" ({check['detail']})" if check["detail"] else ""
        lines.append(f"- {'PASS' if check['pass'] else 'FAIL'}: {check['name']}{suffix}")
    return "\n".join(lines) + "\n"


def cmd_report(args):
    rows = load_rows(args.rows)
    if not rows:
        raise Abort("no rows")
    sessions = load_sessions(args.sessions)
    hosts = read_csv(args.hosts)
    summary = compute_summary(rows, sessions)
    write_text(args.summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_text(args.out, render_report(summary, sessions, hosts))


def cmd_schedule(args):
    cells = [name for name in args.cells.split(",") if name]
    for entry in build_schedule(cells, args.rounds, args.seed0):
        print(
            entry["sequence"],
            entry["cell"],
            entry["round"],
            entry["slot"],
            entry["arm"],
            entry["seed"],
            entry["order"],
        )


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def cmd_selftest(_args):
    ok = True

    def expect(name, condition):
        nonlocal ok
        print(("pass " if condition else "FAIL ") + name)
        ok = ok and condition

    expect("type7", median([1, 2, 3, 4, 5, 6]) == 3.5)
    expect("balanced Fisher", close(fisher_two_sided(5, 10, 5, 10), 1.0))
    expect(
        "complete-split Fisher",
        close(fisher_two_sided(0, 12, 12, 12), 2 / math.comb(24, 12)),
    )
    expect(
        "Mann-Whitney separated",
        mann_whitney_two_sided([1, 2, 3, 4], [5, 6, 7, 8]) < 0.05,
    )
    rng = SplitMix64(0)
    expect(
        "SplitMix64 stream",
        [rng.next_u64() for _ in range(3)]
        == [0xE220A8397B1DCDaf, 0x6E789E6AA1B965F4, 0x06C45D188009454F],
    )
    schedule = build_schedule(["lj75"], 12, 2000)
    pair_orders = []
    for round_index in range(12):
        order = [entry["arm"] for entry in schedule if entry["round"] == round_index]
        pair_orders.append("".join(arm for arm in order if arm != "AP"))
        expect(
            f"round {round_index} shares one seed",
            len({entry["seed"] for entry in schedule if entry["round"] == round_index})
            == 1,
        )
    expect("ABBA order", pair_orders == ["AB", "BA", "BA", "AB"] * 3)
    expect("schedule length", len(schedule) == 36)
    sample = "gate_row " + " ".join(f"{key}=1" for key in DRIVER_FIELDS) + "\n"
    expect("driver row parser", set(parse_gate_row(sample)) == set(DRIVER_FIELDS))
    expect(
        "row fields cover driver fields",
        set(DRIVER_FIELDS) <= set(ROW_FIELDS)
        and len(set(ROW_FIELDS)) == len(ROW_FIELDS),
    )
    expect(
        "paper budgets",
        all(
            value[5]
            == (400_000 if value[2] <= 38 else 1_000_000 if value[2] < 75 else 4_000_000)
            for value in CELLS.values()
        ),
    )
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="build three arms and append the run rows")
    run.add_argument("--harness-tree", required=True)
    run.add_argument("--base-revision", required=True)
    run.add_argument("--base-tree", required=True)
    for name in ("a", "ap", "b"):
        run.add_argument(f"--arm-{name}", required=True)
        run.add_argument(f"--arm-{name}-revision", required=True)
        run.add_argument(f"--arm-{name}-tree", required=True)
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
    report = sub.add_parser("report", help="generate summary JSON and verdict report")
    report.add_argument("--rows", required=True)
    report.add_argument("--sessions", required=True)
    report.add_argument("--hosts", required=True)
    report.add_argument("--summary", required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)
    schedule = sub.add_parser("schedule", help="print the serial run order")
    schedule.add_argument("--cells", required=True)
    schedule.add_argument("--rounds", type=int, required=True)
    schedule.add_argument("--seed0", type=int, required=True)
    schedule.set_defaults(func=cmd_schedule)
    selftest = sub.add_parser("selftest", help="check statistics and protocol")
    selftest.set_defaults(func=cmd_selftest)
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except Abort as err:
        print(f"harness: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
