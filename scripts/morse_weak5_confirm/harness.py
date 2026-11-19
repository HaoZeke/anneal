#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Opt-in confirmation run of the held-stream preset.

Arm A is ``Config::recommended(n)``. Arm B is ``Config::recommended_adaptive(n)``,
the held-stream fork. Arm AA is a second arm of A on an independent seed set; it
gives the band that A against A reaches on seed luck alone. All three run the one
release binary of ``examples/morse_weak5_confirm.rs``, built here from a clean
worktree, so the arms differ in the preset and nothing else.

Subcommands, every path and every number a required argument::

    harness.py run --tree T --base-rev R --base-tree X --toolchain V --target-dir D
                   --core C --cells c1,c2 --rounds N --seed0 S --aa-seed0 S2
                   --timeout-s SEC --settle-s SEC --rows F --sessions F --hosts F
                   --logs DIR [--smoke]
    harness.py report --rows F --sessions F --hosts F --out F
    harness.py schedule --cells c1,c2 --rounds N --seed0 S --aa-seed0 S2
    harness.py selftest

``run`` protocol. One process at a time, one thread, started under ``taskset``
on one core that the harness itself does not use, with a fixed environment of
thread-count variables and nothing else. Each run is a fresh process. Round ``r``
runs A, B, AA when ``r`` is even and AA, B, A when it is odd, so a drift in
machine speed is spread over the three arms alike. A and B take the seed
``seed0 + r`` and the same start geometry. AA takes ``aa-seed0 + r``. There is
no warm-up run, no retry and no replacement: a run that fails, times out or
reports something other than what was asked ends the session, and a rerun of
the command resumes after the last row written.

The worktree must be clean before the build and after the last run, and it must
differ from the base commit only in this harness and its driver. Outputs go
outside the worktree. After the build the harness idles for ``--settle-s`` seconds
and refuses to start if a build tool is still running. ``--smoke`` allows fewer than
twelve rounds for a dry run and marks the session so that the report fails its
check on it. ``report`` reads only the three data files.
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

EXAMPLE = "morse_weak5_confirm"
DRIVER_PATH = "examples/morse_weak5_confirm.rs"
HARNESS_PATH = "scripts/morse_weak5_confirm/harness.py"
ALLOWED_CHANGES = (DRIVER_PATH, "scripts/morse_weak5_confirm/")

ARMS = ("A", "B", "AA")
ARM_PRESET = {"A": "recommended", "B": "adaptive", "AA": "recommended"}
ARM_SEED_SET = {"A": "main", "B": "main", "AA": "independent"}
ARM_LABEL = {"A": "A", "B": "B", "AA": "A'"}
ARM_FLAGS = {"A": (0, 0, 1), "B": (1, 1, 2), "AA": (0, 0, 1)}

# name -> kind, atoms, rho as the driver prints it, Cambridge minimum, paper budget.
CELLS = {
    "lj38": ("lj", 38, "na", -173.928427, 400_000),
    "m38r10": ("morse", 38, "10.0", -145.849817, 400_000),
    "lj75": ("lj", 75, "na", -397.492331, 4_000_000),
    "lj98": ("lj", 98, "na", -543.665361, 4_000_000),
    "m75r6": ("morse", 75, "6.0", -351.472365, 4_000_000),
    "m75r14": ("morse", 75, "14.0", -318.407330, 4_000_000),
    "m80r6": ("morse", 80, "6.0", -378.333471, 4_000_000),
    "m80r14": ("morse", 80, "14.0", -340.811371, 4_000_000),
}

# The held-stream preset forks after this many charged calls; a budget at or
# below it never forks.
HOLD_ALLOWANCE = 1 << 19

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}
BUILD_TOOLS = ("cargo", "rustc", "rustup", "cc", "cc1", "gcc", "ld", "collect2")
BAD_BUILD_ENV = ("RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER")

BAND_DRAWS = 20_000
BAND_SEED = 20261119
MASK64 = (1 << 64) - 1
MIN_ROUNDS = 12

DRIVER_FIELDS = (
    "cell", "preset", "seed", "n", "kind", "rho", "budget", "reference",
    "adaptive_search", "stream_hold", "replicas", "hit", "evals_to_hit", "charged",
    "best", "best_bits", "wall_s", "pot_s", "outside_s", "cpu_ns", "wait_ns",
    "vol_ctxt", "invol_ctxt", "threads", "core_start", "core_end", "cpus_allowed",
    "cpus_allowed_list", "start_sha256", "config_sha256",
)
ROW_FIELDS = (
    "session", "cell", "arm", "preset", "round", "slot", "order", "seed_set", "seed",
    "n", "kind", "rho", "budget", "reference", "adaptive_search", "stream_hold", "replicas",
    "hit", "evals_to_hit", "charged", "best", "best_bits", "wall_s", "pot_s", "outside_s",
    "proc_s", "cpu_ns", "wait_ns", "vol_ctxt", "invol_ctxt", "threads", "core_pinned",
    "core_busy_ticks", "core_steal_ticks",
    "core_start", "core_end", "cpus_allowed", "cpus_allowed_list", "start_sha256",
    "config_sha256", "code_revision", "code_tree", "binary_sha256", "host_id",
    "utc_start", "utc_end",
)
INT_FIELDS = (
    "round", "slot", "seed", "n", "budget", "adaptive_search", "stream_hold", "replicas",
    "hit", "evals_to_hit", "charged", "cpu_ns", "wait_ns", "vol_ctxt", "invol_ctxt",
    "threads", "core_pinned", "core_busy_ticks", "core_steal_ticks", "core_start", "core_end",
)
FLOAT_FIELDS = ("reference", "wall_s", "pot_s", "outside_s", "proc_s")
HOST_FIELDS = ("host_id", "cpu_model", "logical_cpus", "mem_total_kb", "arch", "kernel_release", "os_release")


class Abort(Exception):
    """The session cannot continue and nothing may be retried or replaced."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- statistics ---------------------------------------------------------------


def type7(sorted_values, p):
    """Hyndman-Fan type 7 quantile of an ascending sample."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("a quantile needs a sample")
    h = (n - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def median(values):
    return type7(sorted(values), 0.5)


def quartiles(values):
    ordered = sorted(values)
    return type7(ordered, 0.25), type7(ordered, 0.5), type7(ordered, 0.75)


def fisher_two_sided(hits_a, n_a, hits_b, n_b):
    """Two-sided Fisher exact test on two hit counts, by exact integer weights."""
    hits = hits_a + hits_b

    def weight(x):
        return math.comb(n_a, x) * math.comb(n_b, hits - x)

    observed = weight(hits_a)
    mass = sum(weight(x) for x in range(max(0, hits - n_b), min(n_a, hits) + 1) if weight(x) <= observed)
    return mass / math.comb(n_a + n_b, hits)


def mcnemar_exact(only_a, only_b):
    """Two-sided exact test on the runs where exactly one arm hit."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def mann_whitney_two_sided(x, y):
    """Normal approximation with the tie and continuity corrections."""
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
    """A small generator whose stream does not depend on the interpreter version."""

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


def hits_band(hits_a, hits_aa, n):
    """Exact 95% band of the hit-count difference between two halves of A's pool.

    The pool is the ``n`` runs of A and the ``n`` runs of A'. The halves are drawn
    without replacement, so the difference is hypergeometric. The band runs from the
    smallest difference whose lower tail reaches 2.5% to the largest whose upper
    tail does.
    """
    pooled = hits_a + hits_aa
    weights = {
        k: math.comb(pooled, k) * math.comb(2 * n - pooled, n - k)
        for k in range(max(0, pooled - n), min(n, pooled) + 1)
    }
    denominator = math.comb(2 * n, n)
    diffs = sorted((2 * k - pooled, w) for k, w in weights.items())
    cumulative = 0
    lower = None
    for diff, weight in diffs:
        cumulative += weight
        if cumulative * 40 >= denominator:
            lower = diff
            break
    cumulative = 0
    upper = None
    for diff, weight in reversed(diffs):
        cumulative += weight
        if cumulative * 40 >= denominator:
            upper = diff
            break
    return lower, upper


def split_ratios(pool, n, draws, seed):
    """Median ratios between random halves of a pool of ``(calls, cost)`` pairs.

    Returns the ascending ratios of the first half's median to the second half's,
    for calls and for cost, from the same draws.
    """
    rng = SplitMix64(seed)
    index = list(range(len(pool)))
    calls_ratio = []
    cost_ratio = []
    for _ in range(draws):
        for i in range(len(index) - 1, 0, -1):
            j = rng.below(i + 1)
            index[i], index[j] = index[j], index[i]
        first, second = index[:n], index[n:]
        calls_ratio.append(median([pool[i][0] for i in first]) / median([pool[i][0] for i in second]))
        cost_ratio.append(median([pool[i][1] for i in first]) / median([pool[i][1] for i in second]))
    calls_ratio.sort()
    cost_ratio.sort()
    return calls_ratio, cost_ratio


# --- schedule -----------------------------------------------------------------


def arm_order(round_index):
    return ARMS if round_index % 2 == 0 else ARMS[::-1]


def build_schedule(cells, rounds, seed0, aa_seed0):
    """Every run in execution order. A and B share a seed; AA has its own set."""
    if not (aa_seed0 >= seed0 + rounds or aa_seed0 + rounds <= seed0):
        raise Abort("the two seed sets overlap")
    entries = []
    for cell in cells:
        for r in range(rounds):
            order = arm_order(r)
            for slot, arm in enumerate(order):
                seed = (aa_seed0 if arm == "AA" else seed0) + r
                entries.append(
                    {"cell": cell, "round": r, "slot": slot, "arm": arm, "seed": seed, "order": ">".join(order)}
                )
    return entries


# --- the run ------------------------------------------------------------------


def git(tree, *args):
    done = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise Abort(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def inside(path, tree):
    path = Path(path).resolve()
    tree = Path(tree).resolve()
    return path == tree or tree in path.parents


def provenance(tree, base_rev, base_tree):
    porcelain = git(tree, "status", "--porcelain")
    if porcelain != "":
        raise Abort("the worktree is not clean")
    head = git(tree, "rev-parse", "HEAD").strip()
    head_tree = git(tree, "rev-parse", "HEAD^{tree}").strip()
    base = git(tree, "rev-parse", "--verify", f"{base_rev}^{{commit}}").strip()
    found = git(tree, "rev-parse", f"{base}^{{tree}}").strip()
    if found != base_tree:
        raise Abort("the base commit does not carry the expected tree")
    if subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor", base, "HEAD"]).returncode != 0:
        raise Abort("the base commit is not an ancestor of HEAD")
    changed = [line for line in git(tree, "diff", "--name-only", base, "HEAD").splitlines() if line]
    stray = [path for path in changed if not path.startswith(ALLOWED_CHANGES)]
    if stray:
        raise Abort("the tested tree differs from the base outside the harness: " + " ".join(stray))
    blob = git(tree, "rev-parse", f"HEAD:{HARNESS_PATH}").strip()
    shown = subprocess.run(["git", "-C", str(tree), "cat-file", "blob", blob], capture_output=True).stdout
    running = Path(__file__).resolve()
    if not inside(running, tree) or hashlib.sha256(shown).hexdigest() != sha256_file(running):
        raise Abort("this harness is not the copy committed in the worktree")
    return {
        "base_revision": base,
        "base_tree": found,
        "code_revision": head,
        "code_tree": head_tree,
        "changed_paths": changed,
        "git_status_porcelain": porcelain,
        "harness_sha256": sha256_file(running),
        "harness_blob": blob,
        "driver_blob": git(tree, "rev-parse", f"HEAD:{DRIVER_PATH}").strip(),
    }


def tool_lines(*cmd):
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise Abort(f"{' '.join(cmd)}: {done.stderr.strip()}")
    return done.stdout.strip().splitlines()


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
    digest = hashlib.sha256(("anneal-morse-weak5-confirm:" + machine_id).encode()).hexdigest()[:16]
    return {
        "host_id": digest,
        "cpu_model": cpu_model,
        "logical_cpus": str(os.cpu_count()),
        "mem_total_kb": mem_kb,
        "arch": platform.machine(),
        "kernel_release": platform.release(),
        "os_release": os_name,
    }


def write_text(path, text):
    with open(path, "w", newline="\n") as handle:
        handle.write(text)


def core_ticks(core):
    """Busy and stolen clock ticks of one core since boot, from ``/proc/stat``.

    Busy is user, nice, system, irq and softirq time; steal is time the
    hypervisor took from the core. A tick is 10 ms.
    """
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == f"cpu{core}":
            user, nice, system, _idle, _iowait, irq, softirq, steal = (int(x) for x in fields[1:9])
            return user + nice + system + irq + softirq, steal
    raise Abort(f"no cpu{core} line in /proc/stat")


def running_build_tools():
    names = []
    for comm in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = comm.read_text().strip()
        except OSError:
            continue
        if name in BUILD_TOOLS:
            names.append(name)
    return sorted(names)


def append_csv(path, fields, row):
    path = Path(path)
    fresh = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path, record):
    with open(path, "a", newline="") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def parse_confirm_row(stdout):
    lines = [line for line in stdout.splitlines() if line.startswith("confirm_row ")]
    if len(lines) != 1:
        raise Abort(f"expected one confirm_row line, found {len(lines)}")
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
    kind, n, rho, reference, budget = CELLS[entry["cell"]]
    expected = {
        "cell": entry["cell"],
        "preset": ARM_PRESET[entry["arm"]],
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
    flags = tuple(int(fields[key]) for key in ("adaptive_search", "stream_hold", "replicas"))
    if flags != ARM_FLAGS[entry["arm"]]:
        raise Abort(f"arm {entry['arm']} ran with flags {flags}")
    hit, evals, charged = int(fields["hit"]), int(fields["evals_to_hit"]), int(fields["charged"])
    if hit not in (0, 1) or not 1 <= evals <= budget or not 1 <= charged <= budget:
        raise Abort("the calls or the hit flag are out of range")
    if hit == 0 and evals != budget:
        raise Abort("a miss must be charged the budget")
    if fields["threads"] != "1":
        raise Abort(f"the process had {fields['threads']} threads")
    if not (fields["core_start"] == fields["core_end"] == str(core) and fields["cpus_allowed_list"] == str(core)):
        raise Abort("the run was not pinned to the requested core")


def run_one(entry, binary, taskset, core, timeout_s, logs, session, identity, host_id):
    preset = ARM_PRESET[entry["arm"]]
    cmd = [taskset, "-c", str(core), str(binary), "run", entry["cell"], preset, str(entry["seed"])]
    log = Path(logs) / entry["cell"] / f"{entry['arm']}-{entry['seed']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    busy_before, steal_before = core_ticks(core)
    started = utc_now()
    mark = time.perf_counter()
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", env=dict(THREAD_ENV), cwd=str(logs), timeout=timeout_s
        )
    except subprocess.TimeoutExpired as err:
        write_text(log, f"timeout after {timeout_s} s\n")
        raise Abort(f"{entry['cell']} {entry['arm']} {entry['seed']} timed out") from err
    proc_s = time.perf_counter() - mark
    ended = utc_now()
    busy_after, steal_after = core_ticks(core)
    write_text(
        log,
        f"command: taskset -c {core} <binary> run {entry['cell']} {preset} {entry['seed']}\n"
        f"started: {started}\nended: {ended}\nexit: {done.returncode}\nproc_s: {proc_s:.6f}\n"
        f"--- stdout\n{done.stdout}--- stderr\n{done.stderr}",
    )
    if done.returncode != 0:
        raise Abort(f"{entry['cell']} {entry['arm']} {entry['seed']} exited with {done.returncode}")
    fields = parse_confirm_row(done.stdout)
    check_run(entry, fields, core)
    row = dict(fields)
    row.update(
        {
            "session": session,
            "arm": entry["arm"],
            "round": entry["round"],
            "slot": entry["slot"],
            "order": entry["order"],
            "seed_set": ARM_SEED_SET[entry["arm"]],
            "proc_s": f"{proc_s:.6f}",
            "core_pinned": core,
            "core_busy_ticks": busy_after - busy_before,
            "core_steal_ticks": steal_after - steal_before,
            "code_revision": identity["code_revision"],
            "code_tree": identity["code_tree"],
            "binary_sha256": identity["binary_sha256"],
            "host_id": host_id,
            "utc_start": started,
            "utc_end": ended,
        }
    )
    return row


def build_binary(tree, toolchain, target_dir):
    for name in os.environ:
        if "RUSTFLAGS" in name or name.startswith("CARGO_PROFILE_") or name in BAD_BUILD_ENV or name.endswith("_LINKER"):
            raise Abort(f"{name} is set; the build must use the stock release profile")
    cmd = ["cargo", f"+{toolchain}", "build", "--release", "--locked", "--example", EXAMPLE, "--target-dir"]
    done = subprocess.run([*cmd, str(target_dir)], cwd=str(tree))
    if done.returncode != 0:
        raise Abort("the release build failed")
    binary = Path(target_dir) / "release" / "examples" / EXAMPLE
    if not binary.is_file():
        raise Abort("the build produced no binary")
    return binary, [*cmd, "<target-dir>"]


def cmd_run(args):
    cells = [name for name in args.cells.split(",") if name]
    unknown = [name for name in cells if name not in CELLS]
    if unknown or not cells or len(set(cells)) != len(cells):
        raise Abort(f"cells must be distinct names from {sorted(CELLS)}")
    if args.rounds < MIN_ROUNDS and not args.smoke:
        raise Abort(f"fewer than {MIN_ROUNDS} rounds needs --smoke, and a smoke session is not a result")
    tree = Path(args.tree).resolve()
    for label, path in (("rows", args.rows), ("sessions", args.sessions), ("hosts", args.hosts), ("logs", args.logs), ("target-dir", args.target_dir)):
        if inside(path, tree):
            raise Abort(f"--{label} lies inside the worktree")
    if args.core in os.sched_getaffinity(0):
        raise Abort("the harness runs on the benchmark core; start it under taskset on other cores")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise Abort("taskset is not installed")
    Path(args.logs).mkdir(parents=True, exist_ok=True)
    session = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    identity = provenance(tree, args.base_rev, args.base_tree)
    binary, build_command = build_binary(tree, args.toolchain, args.target_dir)
    after_build = git(tree, "status", "--porcelain")
    if after_build != "":
        raise Abort("the build left the worktree unclean")
    identity["git_status_porcelain_after_build"] = after_build
    identity["binary_sha256"] = sha256_file(binary)
    host = host_facts()
    print(f"built; idle for {args.settle_s:.0f} s before the first run", flush=True)
    time.sleep(args.settle_s)
    leftover = running_build_tools()
    if leftover:
        raise Abort("build tools are still running: " + " ".join(leftover))
    entries = build_schedule(cells, args.rounds, args.seed0, args.aa_seed0)
    done_keys = set()
    resumed = []
    if Path(args.rows).is_file():
        for old in read_csv(args.rows):
            if old["code_tree"] != identity["code_tree"] or old["binary_sha256"] != identity["binary_sha256"]:
                raise Abort("the rows file holds rows of another build")
            done_keys.add((old["cell"], old["arm"], old["seed"]))
            if old["session"] not in resumed:
                resumed.append(old["session"])
    pending = [entry for entry in entries if (entry["cell"], entry["arm"], str(entry["seed"])) not in done_keys]
    host_known = Path(args.hosts).is_file() and any(r["host_id"] == host["host_id"] for r in read_csv(args.hosts))
    if not host_known:
        append_csv(args.hosts, HOST_FIELDS, host)
    start = {
        "record": "start",
        "session": session,
        "utc": utc_now(),
        "smoke": bool(args.smoke),
        "host_id": host["host_id"],
        "toolchain": args.toolchain,
        "rustc": tool_lines("rustc", f"+{args.toolchain}", "--version", "--verbose"),
        "cargo": tool_lines("cargo", f"+{args.toolchain}", "--version")[0],
        "python": platform.python_version(),
        "taskset": tool_lines(taskset, "--version")[0],
        "build_command": build_command,
        "env": dict(THREAD_ENV),
        "core": args.core,
        "harness_affinity": sorted(os.sched_getaffinity(0)),
        "loadavg": Path("/proc/loadavg").read_text().split()[:3],
        "cells": cells,
        "rounds": args.rounds,
        "seed0": args.seed0,
        "aa_seed0": args.aa_seed0,
        "order_rule": "even rounds A>B>AA, odd rounds AA>B>A",
        "timeout_s": args.timeout_s,
        "settle_s": args.settle_s,
        "planned": len(entries),
        "resumed_rows": len(entries) - len(pending),
        "resume_of": resumed,
    }
    start.update(identity)
    append_jsonl(args.sessions, start)
    print(f"session {session}: {len(pending)} runs to go of {len(entries)}", flush=True)
    completed = 0
    try:
        for entry in pending:
            row = run_one(entry, binary, taskset, args.core, args.timeout_s, args.logs, session, identity, host["host_id"])
            append_csv(args.rows, ROW_FIELDS, row)
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {entry['cell']} {entry['arm']} seed {entry['seed']} "
                f"hit={row['hit']} calls={row['evals_to_hit']} wall={row['wall_s']}s",
                flush=True,
            )
    except Abort as err:
        append_jsonl(args.sessions, {"record": "abort", "session": session, "utc": utc_now(), "reason": str(err), "completed": completed})
        raise
    append_jsonl(
        args.sessions,
        {
            "record": "end",
            "session": session,
            "utc": utc_now(),
            "completed": completed,
            "git_status_porcelain": git(tree, "status", "--porcelain"),
            "rows_total": len(read_csv(args.rows)),
            "rows_sha256": sha256_file(Path(args.rows)),
        },
    )
    if git(tree, "status", "--porcelain") != "":
        raise Abort("the worktree is unclean after the session")
    print("session complete", flush=True)


# --- report -------------------------------------------------------------------


def load_rows(path):
    rows = read_csv(path)
    for row in rows:
        for key in INT_FIELDS:
            row[key] = int(row[key])
        for key in FLOAT_FIELDS:
            row[key] = float(row[key])
        row["calls"] = row["evals_to_hit"]
        row["cost"] = row["wall_s"] * 1.0e6 / row["charged"]
    return rows


def load_sessions(path):
    with open(path, newline="") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_n(x):
    return f"{x:.0f}"


def fmt_s(x):
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def fmt_ratio(x):
    return f"{x:.2f}"


def fmt_spread(values, fmt):
    q1, q2, q3 = quartiles(values)
    return f"{fmt(q2)} [{fmt(q1)}, {fmt(q3)}]"


def md_table(header, body):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def group_rows(rows):
    cells = []
    groups = {}
    for row in rows:
        if row["cell"] not in cells:
            cells.append(row["cell"])
        groups.setdefault((row["cell"], row["arm"]), []).append(row)
    for key in groups:
        groups[key].sort(key=lambda r: r["seed"])
    return cells, groups


def cell_numbers(cell, groups):
    a, b, aa = (groups.get((cell, arm), []) for arm in ARMS)
    n = len(a)
    out = {"n": n, "a": a, "b": b, "aa": aa}
    out["hits"] = {arm: sum(r["hit"] for r in groups.get((cell, arm), [])) for arm in ARMS}
    if not (len(a) == len(b) == len(aa) and n > 0):
        return out
    out["fisher_ab"] = fisher_two_sided(out["hits"]["A"], n, out["hits"]["B"], n)
    out["fisher_aa"] = fisher_two_sided(out["hits"]["A"], n, out["hits"]["AA"], n)
    out["mw_ab"] = mann_whitney_two_sided([r["calls"] for r in a], [r["calls"] for r in b])
    out["mw_aa"] = mann_whitney_two_sided([r["calls"] for r in a], [r["calls"] for r in aa])
    by_seed_b = {r["seed"]: r for r in b}
    paired = [(r, by_seed_b[r["seed"]]) for r in a if r["seed"] in by_seed_b]
    out["paired"] = paired
    out["only_a"] = sum(1 for x, y in paired if x["hit"] and not y["hit"])
    out["only_b"] = sum(1 for x, y in paired if y["hit"] and not x["hit"])
    out["mcnemar"] = mcnemar_exact(out["only_a"], out["only_b"])
    out["same_walk"] = [
        x["charged"] == y["charged"] and x["best_bits"] == y["best_bits"] and x["evals_to_hit"] == y["evals_to_hit"]
        and x["hit"] == y["hit"]
        for x, y in paired
    ]
    pool = [(r["calls"], r["cost"]) for r in a + aa]
    seed = (BAND_SEED * 1_000_003 + zlib.crc32(cell.encode())) & MASK64
    calls_ratios, cost_ratios = split_ratios(pool, n, BAND_DRAWS, seed)
    out["band_hits"] = hits_band(out["hits"]["A"], out["hits"]["AA"], n)
    out["band_calls"] = (type7(calls_ratios, 0.025), type7(calls_ratios, 0.975))
    out["band_cost"] = (type7(cost_ratios, 0.025), type7(cost_ratios, 0.975))
    out["median_calls"] = {arm: median([r["calls"] for r in out[key]]) for arm, key in (("A", "a"), ("B", "b"), ("AA", "aa"))}
    out["median_cost"] = {arm: median([r["cost"] for r in out[key]]) for arm, key in (("A", "a"), ("B", "b"), ("AA", "aa"))}
    return out


def verdict(inside_band):
    return "inside" if inside_band else "OUTSIDE"


def session_checks(rows, sessions, groups, cells, stats):
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))

    starts = [s for s in sessions if s["record"] == "start"]
    ends = {s["session"] for s in sessions if s["record"] == "end"}
    aborts = [s for s in sessions if s["record"] == "abort"]
    check("every session started and ended", bool(starts) and all(s["session"] in ends for s in starts) and not aborts,
          f"{len(starts)} start, {len(ends)} end, {len(aborts)} abort")
    check("no smoke session in the data", not any(s.get("smoke") for s in starts))
    check("worktree clean at every start and end",
          all(s["git_status_porcelain"] == "" for s in starts) and all(s["git_status_porcelain"] == "" for s in sessions if s["record"] == "end"))
    for key in ("code_revision", "code_tree", "binary_sha256", "host_id"):
        values = sorted({r[key] for r in rows})
        check(f"one {key} in the rows", len(values) == 1, " ".join(values))
    for key in ("base_revision", "base_tree", "toolchain", "core", "harness_sha256"):
        values = sorted({str(s[key]) for s in starts})
        check(f"one {key} across sessions", len(values) == 1, " ".join(values))
    core = starts[0]["core"] if starts else None
    check("every run pinned to the one core, one thread",
          all(r["core_pinned"] == r["core_start"] == r["core_end"] == core and r["cpus_allowed_list"] == str(core) and r["threads"] == 1 for r in rows))
    check("one row per cell, arm and seed", len({(r["cell"], r["arm"], r["seed"]) for r in rows}) == len(rows))
    rounds = sorted({stats[c]["n"] for c in cells})
    check("the same number of runs in every arm and cell", len(rounds) == 1 and all(len(groups.get((c, a), [])) == rounds[0] for c in cells for a in ARMS),
          " ".join(str(x) for x in rounds))
    def arm_rows(c, arm):
        return groups.get((c, arm), [])

    check("A and B share seeds and starts",
          all({(r["seed"], r["start_sha256"]) for r in arm_rows(c, "A")} == {(r["seed"], r["start_sha256"]) for r in arm_rows(c, "B")} for c in cells))
    check("A' seeds are disjoint from A seeds",
          all(not ({r["seed"] for r in arm_rows(c, "A")} & {r["seed"] for r in arm_rows(c, "AA")}) for c in cells))
    same_config = []
    for c in cells:
        mine = {r["config_sha256"] for r in arm_rows(c, "A")} | {r["config_sha256"] for r in arm_rows(c, "AA")}
        held = {r["config_sha256"] for r in arm_rows(c, "B")}
        same_config.append(len(mine) == 1 and len(held) == 1 and mine != held)
    check("A and A' share one config hash per cell and B has another", all(same_config))
    for arm in ARMS:
        flags = {(r["adaptive_search"], r["stream_hold"], r["replicas"]) for r in rows if r["arm"] == arm}
        check(f"arm {ARM_LABEL[arm]} ran the preset it names", flags == {ARM_FLAGS[arm]}, str(sorted(flags)))
    check("interleave: even rounds A>B>AA, odd rounds AA>B>A",
          all(r["order"] == ">".join(arm_order(r["round"])) for r in rows))
    check("the table of cells matches the driver's report",
          all((r["kind"], r["n"], r["rho"], r["budget"]) == (CELLS[r["cell"]][0], CELLS[r["cell"]][1], CELLS[r["cell"]][2], CELLS[r["cell"]][4])
              and f"{r['reference']:.6f}" == f"{CELLS[r['cell']][3]:.6f}" for r in rows))
    inside_allowance = [c for c in cells if CELLS[c][4] <= HOLD_ALLOWANCE]
    check("below the fork allowance B is the same walk as A, seed by seed",
          all(all(stats[c].get("same_walk", [False])) for c in inside_allowance),
          " ".join(inside_allowance) or "no such cell")
    early = []
    for c in cells:
        for x, y in stats[c].get("paired", []):
            if x["hit"] and x["evals_to_hit"] <= HOLD_ALLOWANCE:
                early.append((x["charged"] == y["charged"] and x["best_bits"] == y["best_bits"] and x["evals_to_hit"] == y["evals_to_hit"]))
    check("where A hit inside the fork allowance, B is the same walk", all(early), f"{len(early)} seeds")
    return checks


def render_report(rows, sessions, hosts):
    seen_cells, groups = group_rows(rows)
    stats = {cell: cell_numbers(cell, groups) for cell in seen_cells}
    cells = [c for c in seen_cells if all(groups.get((c, arm)) for arm in ARMS)]
    starts = [s for s in sessions if s["record"] == "start"]
    runs_per_arm = stats[cells[0]]["n"] if cells else 0
    lines = ["# Held-stream confirmation run", ""]
    lines += [
        "Every number below is computed by `scripts/morse_weak5_confirm/harness.py report` from",
        "`rows.csv`, `sessions.jsonl` and `hosts.csv` in this directory.",
        "",
        "- Arm A: `Config::recommended(n)`. Arm B: `Config::recommended_adaptive(n)`, the held-stream fork.",
        "  Arm A': `Config::recommended(n)` again on an independent seed set, the A/A control.",
        "- A and B share the seed and the start geometry of each run. A' uses seeds of its own.",
        "- One release binary, one process per run, one thread, pinned to one core, one run at a time.",
        "  No warm-up, no retry, no replacement run, no reduced budget.",
        "- Round `r` runs A, B, A' when `r` is even and A', B, A when `r` is odd.",
        "- A hit is a quenched energy at or below the Cambridge minimum plus 0.001. A miss is charged the full budget.",
        "- Median and quartiles are Hyndman-Fan type 7. Fisher is the two-sided exact test on hit counts.",
        "  Mann-Whitney is two-sided, normal approximation with tie and continuity corrections, on calls with misses at the budget.",
        "  McNemar is the exact test on the seeds where exactly one of A and B hit.",
        f"- The A/A band pools the {runs_per_arm} runs of A and the {runs_per_arm} of A' and splits them at random into halves of that size.",
        "  The hit-count band is exact. The median-ratio bands are from "
        f"{BAND_DRAWS} draws of a fixed generator. A band is the 2.5th to 97.5th percentile of the first half against the second.",
        "",
    ]
    lines += ["## Provenance", ""]
    last = starts[-1]
    host = next((h for h in hosts if h["host_id"] == last["host_id"]), None)
    prov = [
        ("base commit (patch under test)", last["base_revision"]),
        ("base tree", last["base_tree"]),
        ("measured commit", last["code_revision"]),
        ("measured tree", last["code_tree"]),
        ("paths changed from base", ", ".join(last["changed_paths"])),
        ("git status --porcelain at start", "empty" if last["git_status_porcelain"] == "" else last["git_status_porcelain"]),
        ("binary sha256", last["binary_sha256"]),
        ("harness sha256", last["harness_sha256"]),
        ("rustc", last["rustc"][0]),
        ("cargo", last["cargo"]),
        ("build", " ".join(last["build_command"])),
        ("taskset", last["taskset"]),
        ("python", last["python"]),
        ("environment given to each run", " ".join(f"{k}={v}" for k, v in sorted(last["env"].items()))),
        ("pinned core", str(last["core"])),
        ("harness affinity", ",".join(str(c) for c in last["harness_affinity"])),
        ("host id (hash of the machine id)", last["host_id"]),
    ]
    if host:
        prov += [("cpu", host["cpu_model"]), ("logical cpus", host["logical_cpus"]), ("memory kB", host["mem_total_kb"]),
                 ("kernel", host["kernel_release"]), ("os", host["os_release"]), ("arch", host["arch"])]
    lines += md_table(["field", "value"], [[k, f"`{v}`" if v else ""] for k, v in prov])
    lines += ["", "Sessions:", ""]
    lines += md_table(
        ["session", "start", "end", "rows added", "load average at start", "smoke"],
        [
            [
                s["session"],
                s["utc"],
                next((e["utc"] for e in sessions if e["record"] == "end" and e["session"] == s["session"]), "none"),
                str(next((e["completed"] for e in sessions if e["record"] == "end" and e["session"] == s["session"]), "none")),
                " ".join(s["loadavg"]),
                "yes" if s.get("smoke") else "no",
            ]
            for s in starts
        ],
    )
    lines += ["", "## Per cell and arm", ""]
    body = []
    for c in cells:
        n = stats[c]["n"]
        for arm in ARMS:
            g = groups[(c, arm)]
            hit_calls = [r["calls"] for r in g if r["hit"]]
            body.append(
                [
                    c if arm == "A" else "",
                    ARM_LABEL[arm],
                    f"{stats[c]['hits'][arm]}/{len(g)}",
                    fmt_spread([r["calls"] for r in g], fmt_n),
                    (fmt_spread(hit_calls, fmt_n) + f" (n={len(hit_calls)})") if hit_calls else "none",
                    fmt_spread([r["wall_s"] for r in g], fmt_s),
                    fmt_spread([r["cost"] for r in g], fmt_s),
                ]
            )
    lines += md_table(
        ["cell", "arm", "hits/n", "calls, median [Q1, Q3]", "calls to hit, hit runs only", "wall s, median [Q1, Q3]", "us per call, median [Q1, Q3]"],
        body,
    )
    lines += ["", "Calls count every run, a miss at its budget. Calls to hit count the runs that hit.", ""]
    lines += ["## B against A", ""]
    body = []
    for c in cells:
        s = stats[c]
        if "fisher_ab" not in s:
            continue
        body.append(
            [
                c,
                f"{s['hits']['A']}/{s['n']}",
                f"{s['hits']['B']}/{s['n']}",
                fmt_p(s["fisher_ab"]),
                f"{s['only_a']} / {s['only_b']}",
                fmt_p(s["mcnemar"]),
                fmt_p(s["mw_ab"]),
                fmt_ratio(s["median_calls"]["B"] / s["median_calls"]["A"]),
                fmt_ratio(s["median_cost"]["B"] / s["median_cost"]["A"]),
                f"{sum(s['same_walk'])}/{len(s['same_walk'])}",
            ]
        )
    lines += md_table(
        ["cell", "A hits", "B hits", "Fisher p", "hit only A / only B", "McNemar p", "Mann-Whitney p, calls", "median calls B/A", "median us per call B/A", "same walk"],
        body,
    )
    lines += ["", "Same walk counts the seeds on which A and B charged the same calls and ended on the same energy bits.", ""]
    lines += ["## A against A', the A/A control", ""]
    body = []
    for c in cells:
        s = stats[c]
        if "fisher_ab" not in s:
            continue
        d_aa = s["hits"]["AA"] - s["hits"]["A"]
        d_b = s["hits"]["B"] - s["hits"]["A"]
        lo, hi = s["band_hits"]
        body.append(
            [
                c,
                f"{s['hits']['A']}/{s['n']}",
                f"{s['hits']['AA']}/{s['n']}",
                fmt_p(s["fisher_aa"]),
                fmt_p(s["mw_aa"]),
                f"{d_aa:+d}",
                f"{d_b:+d}",
                f"[{lo:+d}, {hi:+d}]",
                verdict(lo <= d_b <= hi),
            ]
        )
    lines += md_table(
        ["cell", "A hits", "A' hits", "Fisher p, A vs A'", "Mann-Whitney p, A vs A'", "hits A' - A", "hits B - A",
         "A/A band of the hit difference", "B - A against the band"],
        body,
    )
    lines += ["", "## Median ratios against the A/A band", ""]
    body = []
    for c in cells:
        s = stats[c]
        if "fisher_ab" not in s:
            continue
        mc, mk = s["median_calls"], s["median_cost"]
        clo, chi = s["band_calls"]
        klo, khi = s["band_cost"]
        calls_b = mc["B"] / mc["A"]
        cost_b = mk["B"] / mk["A"]
        body.append(
            [
                c,
                fmt_ratio(mc["AA"] / mc["A"]),
                fmt_ratio(calls_b),
                f"[{fmt_ratio(clo)}, {fmt_ratio(chi)}]",
                verdict(clo <= calls_b <= chi),
                fmt_ratio(mk["AA"] / mk["A"]),
                fmt_ratio(cost_b),
                f"[{fmt_ratio(klo)}, {fmt_ratio(khi)}]",
                verdict(klo <= cost_b <= khi),
            ]
        )
    lines += md_table(
        ["cell", "calls A'/A", "calls B/A", "A/A band", "B/A against the band", "us per call A'/A", "us per call B/A", "A/A band", "B/A against the band"],
        body,
    )
    lines += ["", "Ratios are of medians. A band is what a split of A's own runs does by chance. A difference inside it is not separated from seed luck.", ""]
    lines += ["## Per seed", ""]
    for c in cells:
        s = stats[c]
        lines += [f"### {c}", ""]
        body = []
        for x, y in s.get("paired", []):
            same = x["charged"] == y["charged"] and x["best_bits"] == y["best_bits"] and x["evals_to_hit"] == y["evals_to_hit"] and x["hit"] == y["hit"]
            body.append(
                [str(x["seed"]), "hit" if x["hit"] else "miss", str(x["calls"]), "hit" if y["hit"] else "miss", str(y["calls"]), "yes" if same else "no"]
            )
        lines += md_table(["seed", "A", "A calls", "B", "B calls", "same walk"], body)
        lines += [""]
        body = [[str(r["seed"]), "hit" if r["hit"] else "miss", str(r["calls"])] for r in groups.get((c, "AA"), [])]
        lines += md_table(["A' seed", "A'", "A' calls"], body)
        lines += [""]
    lines += [
        "## Noise diagnostics",
        "",
        "Switches, wait time and cpu time are the process's own account over the search. Core busy time is what the",
        "pinned core did while the process ran, from `/proc/stat` in 10 ms ticks, and steal is what the hypervisor took.",
        "A core busy time near the process cpu time means nothing else ran on the core.",
        "",
    ]
    body = []
    for c in cells:
        for arm in ARMS:
            g = groups[(c, arm)]
            cpu_s = sum(r["cpu_ns"] for r in g) / 1.0e9
            busy_s = sum(r["core_busy_ticks"] for r in g) / 100.0
            body.append(
                [
                    c if arm == "A" else "",
                    ARM_LABEL[arm],
                    str(sum(r["invol_ctxt"] for r in g)),
                    str(max(r["invol_ctxt"] for r in g)),
                    f"{sum(r['wait_ns'] for r in g) / max(sum(r['cpu_ns'] for r in g), 1) * 100:.4f}",
                    f"{cpu_s / sum(r['wall_s'] for r in g):.4f}",
                    f"{busy_s / cpu_s:.3f}",
                    str(sum(r["core_steal_ticks"] for r in g)),
                ]
            )
    lines += md_table(
        ["cell", "arm", "involuntary switches, total", "involuntary switches, worst run", "wait / cpu time, total, %",
         "cpu time / wall, total", "core busy time / cpu time, total", "steal ticks, total"],
        body,
    )
    lines += [""]
    lines += ["## Checks", ""]
    for name, ok, detail in session_checks(rows, sessions, groups, seen_cells, stats):
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))
    return "\n".join(lines) + "\n"


def cmd_report(args):
    rows = load_rows(args.rows)
    if not rows:
        raise Abort("no rows")
    sessions = load_sessions(args.sessions)
    hosts = read_csv(args.hosts)
    text = render_report(rows, sessions, hosts)
    with open(args.out, "w", newline="\n") as handle:
        handle.write(text)


def cmd_schedule(args):
    cells = [name for name in args.cells.split(",") if name]
    for index, entry in enumerate(build_schedule(cells, args.rounds, args.seed0, args.aa_seed0), 1):
        print(index, entry["cell"], entry["round"], entry["slot"], entry["arm"], entry["seed"], entry["order"])


# --- selftest -----------------------------------------------------------------


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def cmd_selftest(_args):
    ok = True

    def expect(name, condition):
        nonlocal ok
        print(("pass " if condition else "FAIL ") + name)
        ok = ok and condition

    expect("type7 on 1..6", quartiles([1, 2, 3, 4, 5, 6]) == (2.25, 3.5, 4.75))
    expect("fisher on a balanced table", close(fisher_two_sided(5, 10, 5, 10), 1.0))
    expect("fisher on a complete split", close(fisher_two_sided(0, 12, 12, 12), 2 / math.comb(24, 12)))
    expect(
        "fisher against reference values",
        all(
            abs(fisher_two_sided(a, 12, b, 12) - want) < 5e-7
            for a, b, want in ((7, 9, 0.666846), (8, 11, 0.316770), (9, 12, 0.217391), (12, 12, 1.0))
        ),
    )
    expect(
        "mann-whitney against reference values",
        close(mann_whitney_two_sided([1, 2, 2, 3, 10, 4_000_000], [2, 4, 4, 5, 6, 7, 4_000_000, 4_000_000]), 0.2677055732456023)
        and close(
            mann_whitney_two_sided(
                [4_000_000] * 7 + [120_000, 340_000, 560_000, 900_000, 1_200_000],
                [4_000_000] * 3 + [80_000, 150_000, 210_000, 330_000, 410_000, 800_000, 990_000, 1_300_000, 2_100_000],
            ),
            0.15038078305185046,
        ),
    )
    expect("mcnemar on one side", close(mcnemar_exact(0, 5), 2 / 32))
    expect("mcnemar on no discordance", mcnemar_exact(0, 0) == 1.0)
    expect("mann-whitney on identical samples", mann_whitney_two_sided([3, 3, 3], [3, 3, 3]) == 1.0)
    expect("mann-whitney on separated samples", mann_whitney_two_sided([1, 2, 3, 4], [5, 6, 7, 8]) < 0.05)
    rng = SplitMix64(0)
    expect("splitmix64 stream", [rng.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F])
    lo, hi = hits_band(6, 6, 12)
    expect("hit band is symmetric about zero", lo == -hi and lo < 0)
    lo, hi = hits_band(0, 0, 12)
    expect("hit band of an empty pool is zero", (lo, hi) == (0, 0))
    pool = [(i + 1.0, 2.0 * (i + 1)) for i in range(24)]
    first = split_ratios(pool, 12, 200, 7)
    expect("split ratios are reproducible", first == split_ratios(pool, 12, 200, 7))
    expect("split ratios do not depend on the unit of the cost", all(close(a, b) for a, b in zip(*first)))
    expect("split ratios are ascending", first[0] == sorted(first[0]) and first[0][0] < 1.0 < first[0][-1])
    entries = build_schedule(["lj38", "lj75"], 12, 2000, 6000)
    expect("schedule length", len(entries) == 2 * 12 * 3)
    for cell in ("lj38", "lj75"):
        mine = [e for e in entries if e["cell"] == cell]
        positions = {arm: [e["slot"] for e in mine if e["arm"] == arm] for arm in ARMS}
        expect(f"{cell}: each arm sits in the end slots equally and B in the middle",
               positions["B"] == [1] * 12 and sorted(positions["A"]) == [0] * 6 + [2] * 6 and sorted(positions["AA"]) == [0] * 6 + [2] * 6)
        expect(f"{cell}: A and B share seeds, A' does not",
               [e["seed"] for e in mine if e["arm"] == "A"] == [e["seed"] for e in mine if e["arm"] == "B"] == list(range(2000, 2012))
               and [e["seed"] for e in mine if e["arm"] == "AA"] == list(range(6000, 6012)))
    try:
        build_schedule(["lj38"], 12, 2000, 2005)
        expect("overlapping seed sets are refused", False)
    except Abort:
        expect("overlapping seed sets are refused", True)
    sample = (
        "noise\nconfirm_row " + " ".join(f"{k}=1" for k in DRIVER_FIELDS) + "\n"
    )
    expect("a driver line parses", set(parse_confirm_row(sample)) == set(DRIVER_FIELDS))
    try:
        parse_confirm_row("confirm_row cell=lj38\n")
        expect("a short driver line is refused", False)
    except Abort:
        expect("a short driver line is refused", True)
    expect("row fields cover the driver fields", set(DRIVER_FIELDS) <= set(ROW_FIELDS) and len(set(ROW_FIELDS)) == len(ROW_FIELDS))
    expect("every cell budget follows the paper list", all(v[4] == (400_000 if v[1] <= 38 else 1_000_000 if v[1] < 75 else 4_000_000) for v in CELLS.values()))
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="build, run the schedule, append rows")
    run.add_argument("--tree", required=True)
    run.add_argument("--base-rev", required=True)
    run.add_argument("--base-tree", required=True)
    run.add_argument("--toolchain", required=True)
    run.add_argument("--target-dir", required=True)
    run.add_argument("--core", type=int, required=True)
    run.add_argument("--cells", required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--seed0", type=int, required=True)
    run.add_argument("--aa-seed0", type=int, required=True)
    run.add_argument("--timeout-s", type=float, required=True)
    run.add_argument("--settle-s", type=float, required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--sessions", required=True)
    run.add_argument("--hosts", required=True)
    run.add_argument("--logs", required=True)
    run.add_argument("--smoke", action="store_true")
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report", help="write the report from the data files")
    report.add_argument("--rows", required=True)
    report.add_argument("--sessions", required=True)
    report.add_argument("--hosts", required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)
    schedule = sub.add_parser("schedule", help="print the run order")
    schedule.add_argument("--cells", required=True)
    schedule.add_argument("--rounds", type=int, required=True)
    schedule.add_argument("--seed0", type=int, required=True)
    schedule.add_argument("--aa-seed0", type=int, required=True)
    schedule.set_defaults(func=cmd_schedule)
    selftest = sub.add_parser("selftest", help="check the statistics, the schedule and the parsers")
    selftest.set_defaults(func=cmd_selftest)
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except Abort as err:
        print(f"harness: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
