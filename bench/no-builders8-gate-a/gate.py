#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial no-loss gate for no-builders8, part a.

Lennard-Jones lj83, lj87 and lj94, seeds 2000-2071, both arms, one
session. The search is Config::recommended. No H2H variable is set.
Budgets come from the repository paper-budget maps. A hit is a finite
best energy at or below the reference plus 1e-3.

Even offsets from 2000 run A then B. Odd offsets run B then A, so the
arm sequence on each system is ABBA. Slot 0 is the first arm of a seed.
Starts are drawn once, on arm A, and both arms quench that geometry.

The verdict is FIX when B has fewer hits than A on any system, when B is
worse on calls-to-hit at Mann-Whitney p < 0.05, or when B's outside-potential
time per charged call exceeds A's by more than a factor 1.1. Otherwise PASS.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import os
import re
import struct
import subprocess
import time
import traceback
from pathlib import Path

BASE_COMMIT = "61fa67cf380e46876b569da45e7cc091647a775b"
BASE_TREE = "cff796c507a8fa1d0bfb28888c3570b61988055e"
ARM_B_TREE = "39ac5e0642beb3e7874a340d2767aa151fa95eac"
SYSTEMS = ("lj83", "lj87", "lj94")
SEEDS = tuple(range(2000, 2072))
HIT_TOL = 1e-3
JOB_TIMEOUT_S = 8 * 60 * 60
ALPHA = 0.05
OUTSIDE_RATIO_LIMIT = 1.1
CORE = "0"

# Cambridge pair-well minima at the six-decimal precision used by
# examples/h2h_gmin_proof.rs. Sizes that file names are checked against
# it. These three sizes are the same table; that file does not list them.
REFERENCES = {
    "lj83": -446.924094,
    "lj87": -472.098165,
    "lj94": -517.264131,
}
REFERENCE_TEXT = {name: f"{value:.6f}" for name, value in REFERENCES.items()}

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "OMP_PROC_BIND": "TRUE",
    "OMP_PLACES": "cores",
}

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "slot",
    "core",
    "hit",
    "calls",
    "charged",
    "wall_s",
    "inside_s",
    "outside_s",
    "best",
    "best_bits",
    "budget",
    "machine_id",
    "base_commit",
    "base_tree",
    "library_commit",
    "library_tree",
    "harness_gate_blob",
    "harness_runner_blob",
    "harness_build_blob",
    "start_sha256",
    "session",
    "reference",
]

HARNESS_PATHS = {
    "harness_gate_blob": "bench/no-builders8-gate-a/gate.py",
    "harness_runner_blob": "bench/no-builders8-gate-a/lj_gate.rs",
    "harness_build_blob": "bench/no-builders8-gate-a/build.sh",
}


def git(tree: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(tree), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def extract_braced(text: str, marker: str) -> str:
    start = text.index(marker)
    brace = text.index("{", start)
    depth = 0
    for index, char in enumerate(text[brace:], brace):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    raise RuntimeError(f"unclosed block at {marker}")


def paper_budgets(path: Path) -> dict[str, int]:
    block = extract_braced(path.read_text(encoding="utf-8"), "PAPER_BUDGETS =")
    parsed = ast.literal_eval(block)
    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError(f"empty PAPER_BUDGETS in {path.name}")
    out: dict[str, int] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise RuntimeError(f"bad PAPER_BUDGETS entry {key!r}")
        out[key] = value
    return out


def shell_budgets(path: Path) -> dict[int, int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"declare -Ar PAPER_BUDGETS=\((.*?)\)", text, re.S)
    if match is None:
        raise RuntimeError(f"shell PAPER_BUDGETS missing in {path.name}")
    found: dict[int, int] = {}
    for size, budget in re.findall(r"\[(\d+)\]=(\d+)", match.group(1)):
        found[int(size)] = int(budget)
    if not found:
        raise RuntimeError(f"shell PAPER_BUDGETS empty in {path.name}")
    return found


def proof_table(path: Path) -> tuple[dict[int, float], dict[int, int]]:
    text = path.read_text(encoding="utf-8")
    references: dict[int, float] = {}
    for name, count, energy in re.findall(r'"(lj\d+)" => \((\d+), (-?\d+\.\d+)\)', text):
        size = int(name[2:])
        if size != int(count):
            raise RuntimeError(f"{name} does not match atom count {count}")
        references[size] = float(energy)
    budgets: dict[int, int] = {}
    for raw_sizes, budget in re.findall(
        r"(?m)^\s+([0-9 |]+)=> Some\(([0-9_]+)\)",
        text,
    ):
        for size in raw_sizes.split("|"):
            budgets[int(size.strip())] = int(budget.replace("_", ""))
    if not references or not budgets:
        raise RuntimeError(f"paper table missing in {path.name}")
    return references, budgets


def hit_tolerance(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"pub const DEFAULT_HIT_TOL: f64 = ([0-9.eE+-]+);", text)
    if match is None:
        raise RuntimeError("DEFAULT_HIT_TOL missing")
    return float(match.group(1))


def load_budget(repo: Path) -> int:
    panel = paper_budgets(repo / "scripts" / "gmin_panel" / "run_panel.py")
    other = paper_budgets(repo / "scripts" / "paper_0120" / "run_panel.py")
    shell = shell_budgets(repo / "scripts" / "elja_submit_jcc_lj.sh")
    proof_refs, proof_budgets = proof_table(repo / "examples" / "h2h_gmin_proof.rs")
    tol = hit_tolerance(repo / "src" / "target_tol.rs")
    if abs(tol - HIT_TOL) > 0.0:
        raise RuntimeError(f"DEFAULT_HIT_TOL is {tol}")
    span = []
    for size in range(75, 105):
        key = f"lj{size}"
        values = []
        if key in panel:
            values.append(panel[key])
        if key in other:
            values.append(other[key])
        if size in shell:
            values.append(shell[size])
        if size in proof_budgets:
            values.append(proof_budgets[size])
        if not values:
            continue
        if len(set(values)) != 1:
            raise RuntimeError(f"lj{size} paper budgets disagree: {values}")
        span.append(values[0])
    if len(set(span)) != 1:
        raise RuntimeError(f"75-104 paper budgets are not one value: {sorted(set(span))}")
    for size, energy in proof_refs.items():
        if size in (83, 87, 94):
            raise RuntimeError(f"lj{size} is already in the proof table")
    known = {
        38: -173.928427,
        55: -279.248470,
        75: -397.492331,
        98: -543.665361,
        102: -569.363652,
        103: -575.766131,
        104: -582.086642,
        150: -893.310258,
        200: -1229.184776,
    }
    if proof_refs != known:
        raise RuntimeError("h2h_gmin_proof references are not the expected table")
    return span[0]


def schedule() -> list[tuple[str, int, str, int]]:
    jobs = []
    for seed in SEEDS:
        arms = ("A", "B") if (seed - SEEDS[0]) % 2 == 0 else ("B", "A")
        for system in SYSTEMS:
            for slot, arm in enumerate(arms):
                jobs.append((system, seed, arm, slot))
    return jobs


def assert_schedule(jobs: list[tuple[str, int, str, int]]) -> None:
    if len(jobs) != len(SYSTEMS) * len(SEEDS) * 2:
        raise RuntimeError(f"schedule length {len(jobs)}")
    if len(SEEDS) != 72 or SEEDS[0] != 2000 or SEEDS[-1] != 2071:
        raise RuntimeError("seed span is not 2000-2071")
    counts: dict[tuple[int, str], int] = {}
    per_system: dict[str, dict[tuple[int, str], int]] = {name: {} for name in SYSTEMS}
    for system, seed, arm, slot in jobs:
        counts[(slot, arm)] = counts.get((slot, arm), 0) + 1
        per_system[system][(slot, arm)] = per_system[system].get((slot, arm), 0) + 1
        if (seed - 2000) % 2 == 0 and (arm, slot) not in {("A", 0), ("B", 1)}:
            raise RuntimeError(f"even seed {seed} is not AB")
        if (seed - 2000) % 2 == 1 and (arm, slot) not in {("B", 0), ("A", 1)}:
            raise RuntimeError(f"odd seed {seed} is not BA")
    expected = len(SYSTEMS) * len(SEEDS) // 2
    for key in ((0, "A"), (0, "B"), (1, "A"), (1, "B")):
        if counts.get(key) != expected:
            raise RuntimeError(f"(slot, arm) {key} count {counts.get(key)}")
        for system in SYSTEMS:
            if per_system[system].get(key) != len(SEEDS) // 2:
                raise RuntimeError(f"{system} (slot, arm) {key} is unbalanced")


def f64_from_bits(text: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(text, 16)))[0]


def f64_bits(value: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def full_precision(value: float) -> str:
    if not math.isfinite(value):
        raise RuntimeError("non-finite value")
    for digits in range(1, 18):
        text = format(value, f".{digits}f")
        if float(text) == value:
            return text
    text = repr(value)
    if float(text) != value:
        raise RuntimeError("value does not round-trip")
    return text


def is_hit(best: float, reference: float) -> bool:
    return (
        math.isfinite(best)
        and math.isfinite(reference)
        and best <= reference + HIT_TOL
    )


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise RuntimeError("empty sample")
    if count == 1:
        return float(ordered[0])
    position = 1.0 + (count - 1) * probability
    lower = math.floor(position)
    frac = position - lower
    index = int(lower) - 1
    if frac == 0.0:
        return float(ordered[index])
    return float(ordered[index] * (1.0 - frac) + ordered[index + 1] * frac)


def log_factorial(number: int) -> float:
    return math.lgamma(number + 1)


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    left_hit, left_miss = hits_a, n_a - hits_a
    right_hit, right_miss = hits_b, n_b - hits_b
    row1 = left_hit + left_miss
    row2 = right_hit + right_miss
    col1 = left_hit + right_hit
    total_n = row1 + row2

    def probability(aa: int, bb: int, cc: int, dd: int) -> float:
        return math.exp(
            log_factorial(aa + bb)
            + log_factorial(cc + dd)
            + log_factorial(aa + cc)
            + log_factorial(bb + dd)
            - (
                log_factorial(aa)
                + log_factorial(bb)
                + log_factorial(cc)
                + log_factorial(dd)
                + log_factorial(total_n)
            )
        )

    observed = probability(left_hit, left_miss, right_hit, right_miss)
    total = 0.0
    low = max(0, col1 - row2)
    high = min(row1, col1)
    for aa in range(low, high + 1):
        bb = row1 - aa
        cc = col1 - aa
        dd = row2 - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        mass = probability(aa, bb, cc, dd)
        if mass <= observed * (1.0 + 1e-9):
            total += mass
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """Tie-corrected normal approximation with continuity correction.

    ``u_left`` is large when ``left`` tends to exceed ``right``.
    """
    n1, n2 = len(left), len(right)
    pooled = [(value, 0) for value in left] + [(value, 1) for value in right]
    pooled.sort(key=lambda item: item[0])
    ranks = [0.0] * len(pooled)
    tie_term = 0.0
    index = 0
    while index < len(pooled):
        end = index
        while end < len(pooled) and pooled[end][0] == pooled[index][0]:
            end += 1
        average = 0.5 * ((index + 1) + end)
        for rank_at in range(index, end):
            ranks[rank_at] = average
        tie = end - index
        if tie > 1:
            tie_term += tie * (tie * tie - 1)
        index = end
    rank_left = sum(rank for rank, (_, arm) in zip(ranks, pooled) if arm == 0)
    u_left = rank_left - n1 * (n1 + 1) / 2.0
    mean = n1 * n2 / 2.0
    total_n = n1 + n2
    variance = n1 * n2 / 12.0 * ((total_n + 1) - tie_term / (total_n * (total_n - 1)))
    if variance <= 0.0:
        return u_left, 1.0
    distance = abs(u_left - mean) - 0.5
    if distance <= 0.0:
        return u_left, 1.0
    zed = distance / math.sqrt(variance)
    return u_left, min(1.0, math.erfc(zed / math.sqrt(2.0)))


def call_rank(row: dict[str, str]) -> float:
    calls = float(row["calls"])
    if row["hit"] == "1":
        return calls
    return math.inf


def self_check() -> None:
    jobs = schedule()
    assert_schedule(jobs)
    if abs(fisher_two_sided(9, 12, 6, 12) - 0.40032305828509895) > 1e-12:
        raise RuntimeError("Fisher check failed")
    if abs(fisher_two_sided(6, 6, 6, 6) - 1.0) > 1e-12:
        raise RuntimeError("Fisher tie check failed")
    tied = mann_whitney([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    if abs(tied[1] - 1.0) > 1e-12:
        raise RuntimeError("Mann-Whitney tie check failed")
    shift = mann_whitney([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
    if abs(shift[0] - 9.0) > 1e-12 or not (shift[1] < 0.2):
        raise RuntimeError("Mann-Whitney shift check failed")
    if abs(type7([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) > 1e-12:
        raise RuntimeError("type 7 check failed")
    sample = -446.924094
    if float(full_precision(sample)) != sample:
        raise RuntimeError("full precision check failed")
    if not is_hit(sample, sample) or is_hit(sample + 1e-3 + 1e-12, sample):
        raise RuntimeError("hit rule check failed")
    if is_hit(float("nan"), sample) or is_hit(float("-inf"), sample):
        raise RuntimeError("non-finite hit check failed")


def machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    raise RuntimeError("machine id file is missing")


def affinity() -> str:
    text = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Cpus_allowed_list is missing")


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in THREAD_ENV.items():
        env[key] = value
    env["LJOS_SEAT"] = "benchbot"
    for key in list(env):
        if key.startswith("H2H_"):
            del env[key]
    return env


def require_environment() -> None:
    if os.environ.get("LJOS_SEAT") != "benchbot":
        raise RuntimeError("LJOS_SEAT is not benchbot")
    stray = sorted(key for key in os.environ if key.startswith("H2H_"))
    if stray:
        raise RuntimeError(f"H2H variables are set: {', '.join(stray)}")
    for key, value in THREAD_ENV.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"{key} is not {value}")
    if affinity() != CORE:
        raise RuntimeError(f"process affinity is {affinity()}, not {CORE}")


def harness_blobs(repo: Path) -> dict[str, str]:
    found = {}
    for field, rel in HARNESS_PATHS.items():
        committed = git(repo, "rev-parse", f"HEAD:{rel}")
        work = subprocess.run(
            ["git", "-C", str(repo), "hash-object", rel],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if committed != work:
            raise RuntimeError(f"{rel} does not match HEAD")
        found[field] = committed
    return found


def identity(tree: Path) -> tuple[str, str]:
    head = git(tree, "rev-parse", "HEAD")
    tree_id = git(tree, "rev-parse", "HEAD^{tree}")
    porcelain = git(tree, "status", "--porcelain")
    if porcelain:
        raise RuntimeError(f"worktree {tree} is dirty")
    return head, tree_id


def binary(bin_dir: Path) -> Path:
    path = bin_dir / "release" / "examples" / "lj_gate"
    if not path.is_file():
        raise RuntimeError(f"missing driver under {bin_dir}")
    head = (bin_dir / "release" / "SOURCE_HEAD").read_text(encoding="utf-8").strip()
    tree = (bin_dir / "release" / "SOURCE_TREE").read_text(encoding="utf-8").strip()
    return path


def require_build(bin_dir: Path, head: str, tree_id: str) -> Path:
    path = binary(bin_dir)
    got_head = (bin_dir / "release" / "SOURCE_HEAD").read_text(encoding="utf-8").strip()
    got_tree = (bin_dir / "release" / "SOURCE_TREE").read_text(encoding="utf-8").strip()
    if got_head != head or got_tree != tree_id:
        raise RuntimeError("build provenance does not match the worktree")
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_driver(cmd: list[str], log_path: Path, env: dict[str, str], cwd: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        done = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=JOB_TIMEOUT_S,
            check=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    text = log_path.read_text(encoding="utf-8")
    if done.returncode != 0:
        raise RuntimeError(f"driver exit {done.returncode}: {text[-1000:]}")
    return text


def parse_row_line(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("gate_row ")]
    if len(lines) != 1:
        raise RuntimeError(f"expected one gate_row, saw {len(lines)}")
    fields = {}
    for token in lines[0].split()[1:]:
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def write_status(path: Path, done: int, total: int, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(f"{done}/{total} {detail}\n", encoding="utf-8")
    tmp.replace(path)


def cmd_check(args: argparse.Namespace) -> int:
    self_check()
    repo = Path(args.repo)
    budget = load_budget(repo)
    print(f"budget {budget}")
    print(f"systems {len(SYSTEMS)} seeds {len(SEEDS)} jobs {len(schedule())}")
    counts: dict[tuple[int, str], int] = {}
    for _system, _seed, arm, slot in schedule():
        counts[(slot, arm)] = counts.get((slot, arm), 0) + 1
    for key in sorted(counts):
        print(f"slot {key[0]} arm {key[1]} {counts[key]}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    self_check()
    require_environment()
    repo = Path(args.repo)
    harness = Path(args.harness)
    arm_a = Path(args.arm_a)
    arm_b = Path(args.arm_b)
    out = Path(args.out)
    log_dir = Path(args.log_dir)
    budget = load_budget(repo)
    if git(repo, "rev-parse", f"{BASE_COMMIT}^{{tree}}") != BASE_TREE:
        raise RuntimeError("base tree does not match the pinned commit")
    blobs = harness_blobs(harness)
    head_a, tree_a = identity(arm_a)
    head_b, tree_b = identity(arm_b)
    if head_a != BASE_COMMIT or tree_a != BASE_TREE:
        raise RuntimeError("arm A is not the pinned base")
    if tree_b != ARM_B_TREE:
        raise RuntimeError(f"arm B tree {tree_b} is not {ARM_B_TREE}")
    bins = {
        "A": require_build(Path(args.arm_a_bin), head_a, tree_a),
        "B": require_build(Path(args.arm_b_bin), head_b, tree_b),
    }
    trees = {"A": arm_a, "B": arm_b}
    libraries = {
        "A": (head_a, tree_a),
        "B": (head_b, tree_b),
    }
    jobs = schedule()
    assert_schedule(jobs)
    existing = read_rows(out)
    done_keys = {(row["system"], row["seed"], row["arm"]) for row in existing}
    if existing:
        session = existing[0]["session"]
        if any(row["session"] != session for row in existing):
            raise RuntimeError("existing rows do not share one session")
    else:
        session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    env = child_env()
    host_id = machine_id()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "session").write_text(session + "\n", encoding="utf-8")
    total = len(jobs)
    finished = len(done_keys)
    write_status(log_dir / "status", finished, total, "start")
    for system, seed, arm, slot in jobs:
        key = (system, str(seed), arm)
        if key in done_keys:
            continue
        start_path = log_dir / "starts" / f"{system}-{seed}.txt"
        start_path.parent.mkdir(parents=True, exist_ok=True)
        if not start_path.is_file():
            start_log = log_dir / "logs" / f"{system}-s{seed}-start.log"
            text = run_driver(
                [
                    "taskset",
                    "-c",
                    CORE,
                    str(bins["A"]),
                    "coords",
                    system[2:],
                    str(seed),
                    str(start_path),
                ],
                start_log,
                env,
                arm_a,
            )
            if f"n={system[2:]}" not in text or f"seed={seed}" not in text:
                raise RuntimeError(f"start for {system} seed {seed} was not written")
        log_path = log_dir / "logs" / f"{system}-s{seed}-{arm}.log"
        text = run_driver(
            [
                "taskset",
                "-c",
                CORE,
                str(bins[arm]),
                "run",
                system[2:],
                str(seed),
                str(budget),
                REFERENCE_TEXT[system],
                str(start_path),
            ],
            log_path,
            env,
            trees[arm],
        )
        got = parse_row_line(text)
        if got["n"] != system[2:] or int(got["seed"]) != seed or int(got["budget"]) != budget:
            raise RuntimeError(f"{system} seed {seed} arm {arm} identity mismatch")
        if got["affinity"] != CORE:
            raise RuntimeError(f"{system} seed {seed} arm {arm} affinity {got['affinity']}")
        best = f64_from_bits(got["best_bits"])
        reference = f64_from_bits(got["reference_bits"])
        if f64_bits(float(REFERENCE_TEXT[system])) != got["reference_bits"]:
            raise RuntimeError(f"{system} reference bits differ from the harness")
        if abs(reference - REFERENCES[system]) > 0.0 and reference != REFERENCES[system]:
            if f64_bits(REFERENCES[system]) != got["reference_bits"]:
                raise RuntimeError(f"{system} reference does not match")
        hit = is_hit(best, REFERENCES[system])
        if str(int(hit)) != got["hit"]:
            raise RuntimeError(f"{system} seed {seed} arm {arm} hit flag disagrees")
        wall_s = f64_from_bits(got["wall_bits"])
        inside_s = f64_from_bits(got["inside_bits"])
        outside_s = f64_from_bits(got["outside_bits"])
        if outside_s != wall_s - inside_s:
            raise RuntimeError(f"{system} seed {seed} arm {arm} outside clock disagrees")
        calls = int(got["calls"])
        charged = int(got["charged"])
        if calls < 0 or charged < 0 or calls > budget or charged > budget:
            raise RuntimeError(f"{system} seed {seed} arm {arm} counts exceed the budget")
        if hit and calls > charged:
            raise RuntimeError(f"{system} seed {seed} arm {arm} hit calls exceed charged")
        if not hit and calls != budget:
            raise RuntimeError(f"{system} seed {seed} arm {arm} miss is not the budget")
        library_commit, library_tree = libraries[arm]
        row = {
            "system": system,
            "arm": arm,
            "seed": str(seed),
            "slot": str(slot),
            "core": CORE,
            "hit": got["hit"],
            "calls": str(calls),
            "charged": str(charged),
            "wall_s": full_precision(wall_s),
            "inside_s": full_precision(inside_s),
            "outside_s": full_precision(outside_s),
            "best": full_precision(best),
            "best_bits": got["best_bits"],
            "budget": str(budget),
            "machine_id": host_id,
            "base_commit": BASE_COMMIT,
            "base_tree": BASE_TREE,
            "library_commit": library_commit,
            "library_tree": library_tree,
            "harness_gate_blob": blobs["harness_gate_blob"],
            "harness_runner_blob": blobs["harness_runner_blob"],
            "harness_build_blob": blobs["harness_build_blob"],
            "start_sha256": got["start_sha256"],
            "session": session,
            "reference": REFERENCE_TEXT[system],
        }
        if float(row["best"]) != best or f64_bits(float(row["best"])) != row["best_bits"]:
            raise RuntimeError("recorded best does not round-trip")
        append_row(out, row)
        finished += 1
        detail = (
            f"{system} {arm} seed {seed} slot {slot} hit {row['hit']} "
            f"calls {calls} wall {wall_s:.3f}"
        )
        write_status(log_dir / "status", finished, total, detail)
        print(detail, flush=True)
    rows = read_rows(out)
    if len(rows) != total:
        raise RuntimeError(f"finished with {len(rows)} rows, expected {total}")
    write_status(log_dir / "status", total, total, "complete")
    print(f"complete {total}", flush=True)
    return 0


def grouped(rows: list[dict[str, str]], system: str, arm: str) -> list[dict[str, str]]:
    found = [row for row in rows if row["system"] == system and row["arm"] == arm]
    return sorted(found, key=lambda row: int(row["seed"]))


def validate_rows(rows: list[dict[str, str]], budget: int) -> None:
    jobs = schedule()
    assert_schedule(jobs)
    if len(rows) != len(jobs):
        raise RuntimeError(f"{len(rows)} rows, expected {len(jobs)}")
    seen = {(row["system"], int(row["seed"]), row["arm"], int(row["slot"])) for row in rows}
    if seen != set(jobs):
        raise RuntimeError("rows do not match the schedule")
    sessions = {row["session"] for row in rows}
    if len(sessions) != 1:
        raise RuntimeError("rows do not share one session")
    machines = {row["machine_id"] for row in rows}
    if len(machines) != 1 or len(next(iter(machines))) != 64:
        raise RuntimeError("machine id is missing")
    for row in rows:
        if row["core"] != CORE or row["base_commit"] != BASE_COMMIT or row["base_tree"] != BASE_TREE:
            raise RuntimeError("row provenance does not match the base")
        if int(row["budget"]) != budget or row["reference"] != REFERENCE_TEXT[row["system"]]:
            raise RuntimeError("row budget or reference does not match")
        if row["arm"] == "A" and row["library_tree"] != BASE_TREE:
            raise RuntimeError("arm A library tree is not the base")
        if row["arm"] == "B" and row["library_tree"] != ARM_B_TREE:
            raise RuntimeError("arm B library tree is not the kit tree")
        best = f64_from_bits(row["best_bits"])
        if float(row["best"]) != best:
            raise RuntimeError("best text does not match best_bits")
        hit = is_hit(best, REFERENCES[row["system"]])
        if str(int(hit)) != row["hit"]:
            raise RuntimeError("hit does not match best and reference")
        for field in ("wall_s", "inside_s", "outside_s"):
            if float(full_precision(float(row[field]))) != float(row[field]):
                raise RuntimeError(f"{field} does not round-trip")
        if abs((float(row["wall_s"]) - float(row["inside_s"])) - float(row["outside_s"])) > 0.0:
            if float(row["outside_s"]) != float(row["wall_s"]) - float(row["inside_s"]):
                raise RuntimeError("outside time is not wall minus inside")
    for system in SYSTEMS:
        for seed in SEEDS:
            pair = [row for row in rows if row["system"] == system and int(row["seed"]) == seed]
            if len({row["start_sha256"] for row in pair}) != 1:
                raise RuntimeError(f"{system} seed {seed} starts differ")
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["slot"], row["arm"])
        counts[key] = counts.get(key, 0) + 1
    for key in (("0", "A"), ("0", "B"), ("1", "A"), ("1", "B")):
        if counts.get(key) != len(SYSTEMS) * len(SEEDS) // 2:
            raise RuntimeError(f"recorded (slot, arm) {key} is unbalanced")


def fmt(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6g}"


def verdict_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[str]]:
    report = []
    failing = []
    for system in SYSTEMS:
        left = grouped(rows, system, "A")
        right = grouped(rows, system, "B")
        hits_a = sum(row["hit"] == "1" for row in left)
        hits_b = sum(row["hit"] == "1" for row in right)
        fisher = fisher_two_sided(hits_a, len(left), hits_b, len(right))
        ranks_a = [call_rank(row) for row in left]
        ranks_b = [call_rank(row) for row in right]
        u_b, mw = mann_whitney(ranks_b, ranks_a)
        calls_worse = u_b > (len(ranks_b) * len(ranks_a) / 2.0) and mw < ALPHA
        wall_a = type7([float(row["wall_s"]) for row in left], 0.5)
        wall_b = type7([float(row["wall_s"]) for row in right], 0.5)
        charged_a = sum(int(row["charged"]) for row in left)
        charged_b = sum(int(row["charged"]) for row in right)
        outside_a = sum(float(row["outside_s"]) for row in left) / charged_a
        outside_b = sum(float(row["outside_s"]) for row in right) / charged_b
        ratio = outside_b / outside_a
        calls_a = type7([float(row["calls"]) for row in left], 0.5)
        calls_b = type7([float(row["calls"]) for row in right], 0.5)
        reasons = []
        if hits_b < hits_a:
            reasons.append("fewer hits")
        if calls_worse:
            reasons.append("calls")
        if ratio > OUTSIDE_RATIO_LIMIT:
            reasons.append("outside time per call")
        if reasons:
            failing.append(f"{system} ({', '.join(reasons)})")
        report.append(
            {
                "system": system,
                "hits_a": hits_a,
                "hits_b": hits_b,
                "n": len(left),
                "fisher": fisher,
                "calls_a": calls_a,
                "calls_b": calls_b,
                "u_b": u_b,
                "mw": mw,
                "calls_worse": calls_worse,
                "wall_a": wall_a,
                "wall_b": wall_b,
                "outside_a": outside_a,
                "outside_b": outside_b,
                "ratio": ratio,
                "reasons": reasons,
                "left": left,
                "right": right,
            }
        )
    return report, failing


def render_readme(rows: list[dict[str, str]], report: list[dict[str, object]], failing: list[str]) -> str:
    verdict = "FIX" if failing else "PASS"
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["slot"], row["arm"])
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# no-builders8 gate, part a",
        "",
        f"Verdict: {verdict}",
        "",
        "Cells are lj83, lj87 and lj94, 72 seeds (2000-2071) on each arm.",
        "Even offsets from 2000 run A then B and odd offsets run B then A.",
        "Slot 0 is the first arm of the seed. The search is the recommended preset.",
        "A hit is a finite best at or below the reference plus 1e-3.",
        "Calls-to-hit ranks a miss worse than every hit.",
        "Outside time per call is the sum of outside time divided by the sum of charged calls.",
        "FIX if B has fewer hits than A on any system, if B is worse on calls at p < 0.05,",
        "or if the outside-time-per-call ratio B/A is above 1.1. Otherwise PASS.",
        "",
        "## Comparison",
        "",
        "| system | hits A | hits B | Fisher p | calls median A | calls median B | Mann-Whitney p | calls loss | wall median A | wall median B | outside/call A | outside/call B | B/A |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report:
        lines.append(
            "| {system} | {hits_a}/{n} | {hits_b}/{n} | {fisher:.6g} | {ca} | {cb} | {mw:.6g} | {loss} | {wa:.6g} | {wb:.6g} | {oa:.6g} | {ob:.6g} | {ratio:.6g} |".format(
                system=item["system"],
                hits_a=item["hits_a"],
                hits_b=item["hits_b"],
                n=item["n"],
                fisher=item["fisher"],
                ca=fmt(item["calls_a"]),
                cb=fmt(item["calls_b"]),
                mw=item["mw"],
                loss="yes" if item["calls_worse"] else "no",
                wa=item["wall_a"],
                wb=item["wall_b"],
                oa=item["outside_a"],
                ob=item["outside_b"],
                ratio=item["ratio"],
            )
        )
    lines.extend(["", "## Failing cells", ""])
    if failing:
        for name in failing:
            lines.append(f"- {name}")
    else:
        lines.append("None.")
    lines.extend(["", "## Slot balance", ""])
    for slot, arm in (("0", "A"), ("0", "B"), ("1", "A"), ("1", "B")):
        lines.append(f"- slot {slot} arm {arm}: {counts[(slot, arm)]}")
    lines.extend(["", "## Per-seed hits", ""])
    for item in report:
        lines.extend(
            [
                f"### {item['system']}",
                "",
                "| seed | hit A | hit B | calls A | calls B |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        by_seed_a = {int(row["seed"]): row for row in item["left"]}
        by_seed_b = {int(row["seed"]): row for row in item["right"]}
        for seed in SEEDS:
            left = by_seed_a[seed]
            right = by_seed_b[seed]
            lines.append(
                f"| {seed} | {left['hit']} | {right['hit']} | {left['calls']} | {right['calls']} |"
            )
        lines.append("")
    sample = rows[0]
    lines.extend(
        [
            "## Provenance",
            "",
            f"- session: {sample['session']}",
            f"- base commit: {sample['base_commit']}",
            f"- base tree: {sample['base_tree']}",
            f"- arm A library commit: {next(row['library_commit'] for row in rows if row['arm'] == 'A')}",
            f"- arm A library tree: {next(row['library_tree'] for row in rows if row['arm'] == 'A')}",
            f"- arm B library commit: {next(row['library_commit'] for row in rows if row['arm'] == 'B')}",
            f"- arm B library tree: {next(row['library_tree'] for row in rows if row['arm'] == 'B')}",
            f"- machine id: {sample['machine_id']}",
            f"- gate blob: {sample['harness_gate_blob']}",
            f"- runner blob: {sample['harness_runner_blob']}",
            f"- build blob: {sample['harness_build_blob']}",
            f"- budget: {sample['budget']}",
            f"- core: {sample['core']}",
            "",
        ]
    )
    return "\n".join(lines)


def cmd_summarize(args: argparse.Namespace) -> int:
    self_check()
    repo = Path(args.repo)
    budget = load_budget(repo)
    rows = read_rows(Path(args.rows))
    validate_rows(rows, budget)
    report, failing = verdict_rows(rows)
    text = render_readme(rows, report, failing)
    path = Path(args.readme)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(text.splitlines()[2])
    if failing:
        for name in failing:
            print(name)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--repo", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo", required=True)
    run.add_argument("--harness", required=True)
    run.add_argument("--arm-a", required=True)
    run.add_argument("--arm-b", required=True)
    run.add_argument("--arm-a-bin", required=True)
    run.add_argument("--arm-b-bin", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--log-dir", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--repo", required=True)
    summary.add_argument("--rows", required=True)
    summary.add_argument("--readme", required=True)
    args = parser.parse_args()
    if args.command == "check":
        return cmd_check(args)
    if args.command == "run":
        return cmd_run(args)
    return cmd_summarize(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as err:
        traceback.print_exc()
        print(f"gate: {err}", flush=True)
        raise SystemExit(1)
