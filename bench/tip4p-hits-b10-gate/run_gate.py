#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Independent four-arm gate for rigid TIP4P quench-check reuse (b10).

Arms, built as separate release binaries from separate worktrees:
- ``D``: develop baseline (commit ``61fa67cf380e46876b569da45e7cc091647a775b``)
- ``B9``: b9 baseline (commit ``b58dd00910d10cc6cc75ac1a61427730234cac0d``)
- ``B10``: b10 candidate (tree ``27132421b07aa2d58c88f13ff9a9c94bb58380a3``)
- ``B9'``: second independent release build of b9 (A/A control band)

Panel: every rigid TIP4P cell ``n=8..21`` present in ``data/tip4p_ccd.csv``
at ``PAPER_BUDGETS`` (2_000_000 evaluations, never setting any ``H2H_*``
variable), seeds 2000-2011 (n=12 per arm), budget-only stop, no warm-up,
serial on core 0 with every thread pool pinned to 1, one session, balanced
Williams ABBA arm rotation, identical starting geometries across all arms.

Rules, declared before execution:
1. Exact ``to_bits`` identity of ``best`` (and ``outcome_best``), and equal
   ``calls`` and ``charged``, for ``B10`` vs ``B9`` on every row.
2. Outside time per charged call (per-row ``outside_s / charged``, then
   Hyndman-Fan type-7 median per cell): ``B10 / B9 <= 1.10`` or within the
   ``B9 / B9'`` A/A spread, and ``B10 / D <= 1.10`` or within the A/A spread,
   in every cell.

Informational statistics per cell: hits per arm, two-sided exact Fisher test
(``B10`` vs ``D``), and two-sided tie- and continuity-corrected Mann-Whitney U
test on calls-to-hit with misses ranked worst (``budget + 1``).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER_SRC = Path(__file__).resolve().parent / "driver" / "main.rs"
sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

ARMS = ("D", "B9", "B10", "B9'")
ARM_SLUGS = {"D": "d", "B9": "b9", "B10": "b10", "B9'": "b9-prime"}

# Williams-balanced ABBA design for 4 arms across 4-seed blocks:
# - Every consecutive pair of seeds is an exact palindrome (ABBA mirror).
# - Every (slot, arm) pair appears once per 4 seeds (3 per cell, 42 total).
# - Every ordered adjacent transition (X -> Y, X != Y) appears once per 4 seeds.
ABBA_CYCLE = (
    ("D", "B9", "B9'", "B10"),
    ("B10", "B9'", "B9", "D"),
    ("B9", "B10", "D", "B9'"),
    ("B9'", "D", "B10", "B9"),
)

SEEDS = tuple(range(2000, 2012))
TIP4P_N_MIN = 8
TIP4P_N_MAX = 21
TIP4P_BUDGET = 2_000_000
RATIO_LIMIT = 1.10
CORE = "0"
JOB_TIMEOUT_S = 4 * 60 * 60

EXPECT_DEVELOP_COMMIT = "61fa67cf380e46876b569da45e7cc091647a775b"
EXPECT_DEVELOP_TREE = "cff796c507a8fa1d0bfb28888c3570b61988055e"
EXPECT_B9_COMMIT = "b58dd00910d10cc6cc75ac1a61427730234cac0d"
EXPECT_B9_TREE = "40238b525d795bf99fdf22a6ee66a4e238e4782e"
EXPECT_B10_TREE = "27132421b07aa2d58c88f13ff9a9c94bb58380a3"

PAPER_BUDGET_CHECKS = (
    ("scripts/tip4p_h2h/run_campaign.py", "PAPER_BUDGETS = {15: 2_000_000, 20: 2_000_000}"),
    (
        "scripts/tip4p_h2h/run_radius.py",
        "PAPER_BUDGETS = {4: 2_000_000, 8: 2_000_000, 15: 2_000_000, 20: 2_000_000}",
    ),
    ("bench/tip4p-h2h/benchmarks/tip4p_h2h.py", "PAPER_BUDGETS: ClassVar[dict[int, int]] = {n: 2_000_000 for n in SIZES}"),
)

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "slot",
    "core",
    "hit",
    "calls",
    "calls_to_hit",
    "charged",
    "wall_s",
    "inside_s",
    "outside_s",
    "best",
    "best_bits",
    "outcome_best",
    "outcome_best_bits",
    "budget",
    "machine_id",
    "library_commit",
    "library_tree",
    "session",
    "start_sha256",
]

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
    "GOTO_NUM_THREADS",
    "TBB_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "POLARS_MAX_THREADS",
)


def type7(values: list[float], p: float = 0.5) -> float:
    """Hyndman-Fan type-7 quantile."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type7 requires at least one value")
    if count == 1:
        return ordered[0]
    pos = 1.0 + (count - 1) * p
    lo = int(pos)
    frac = pos - lo
    if frac == 0.0:
        return ordered[lo - 1]
    return ordered[lo - 1] * (1.0 - frac) + ordered[lo] * frac


def fisher_exact_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided exact Fisher p-value summing tables no more likely than observed."""
    total_hits = hits_a + hits_b
    denom = math.comb(n_a + n_b, total_hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, total_hits - k) / denom

    observed = prob(hits_a)
    lo = max(0, total_hits - n_b)
    hi = min(n_a, total_hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1.0 + 1e-9)))


def mann_whitney_two_sided(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U and asymptotic p with tie and continuity corrections."""
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann-whitney requires non-empty samples")
    pooled = sorted(left + right)
    n = n1 + n2
    ranks: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        ranks[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(ranks[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    p = min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))
    return u, p


def bits_of(token: str) -> str:
    """IEEE-754 binary64 hex representation of a finite float token."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value {token!r}")
    return f"{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def arm_sequence(seed: int) -> tuple[str, ...]:
    return ABBA_CYCLE[(seed - SEEDS[0]) % len(ABBA_CYCLE)]


def planned_jobs(systems: list[str]) -> list[tuple[str, int, int, str]]:
    jobs: list[tuple[str, int, int, str]] = []
    for system in systems:
        for seed in SEEDS:
            for slot, arm in enumerate(arm_sequence(seed)):
                jobs.append((system, seed, slot, arm))
    return jobs


def pair_counts(jobs: list[tuple[str, int, int, str]]) -> dict[tuple[int, str], int]:
    counts = {(slot, arm): 0 for slot in range(len(ARMS)) for arm in ARMS}
    for _system, _seed, slot, arm in jobs:
        counts[(slot, arm)] += 1
    return counts


def assert_balanced(jobs: list[tuple[str, int, int, str]], systems: list[str]) -> None:
    per_cell = len(SEEDS) // len(ARMS)
    if len(SEEDS) % len(ARMS) != 0:
        raise SystemExit("seed count must be divisible by arm count")
    expected_total = per_cell * len(systems)
    total_counts = pair_counts(jobs)
    for key, count in total_counts.items():
        if count != expected_total:
            raise SystemExit(f"unbalanced global (slot, arm) {key}: {count} != {expected_total}")
    for system in systems:
        cell_jobs = [job for job in jobs if job[0] == system]
        cell_counts = pair_counts(cell_jobs)
        for key, count in cell_counts.items():
            if count != per_cell:
                raise SystemExit(f"unbalanced {system} (slot, arm) {key}: {count} != {per_cell}")
        for idx in range(0, len(SEEDS), 2):
            s0 = arm_sequence(SEEDS[idx])
            s1 = arm_sequence(SEEDS[idx + 1])
            if s1 != tuple(reversed(s0)):
                raise SystemExit(f"not ABBA mirror for seeds {SEEDS[idx]}, {SEEDS[idx + 1]}: {s0} vs {s1}")
    transitions: dict[tuple[str, str], int] = {
        (a, b): 0 for a in ARMS for b in ARMS if a != b
    }
    for seed in SEEDS:
        seq = arm_sequence(seed)
        for a, b in zip(seq, seq[1:]):
            transitions[(a, b)] += 1
    expected_trans = len(SEEDS) // len(ARMS)
    for pair, count in transitions.items():
        if count != expected_trans:
            raise SystemExit(f"unbalanced adjacency {pair}: {count} != {expected_trans}")
    if len(jobs) != len(systems) * len(SEEDS) * len(ARMS):
        raise SystemExit("unexpected total job count")


def tip4p_cells(text: str) -> list[tuple[str, int, float]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    found: list[tuple[str, int, float]] = []
    for row in rows:
        n = int(row["n"])
        if TIP4P_N_MIN <= n <= TIP4P_N_MAX:
            found.append((f"tip4p{n}", n, float(row["energy"])))
    found.sort(key=lambda item: item[1])
    seen = [n for _name, n, _energy in found]
    wanted = list(range(TIP4P_N_MIN, TIP4P_N_MAX + 1))
    if seen != wanted:
        raise SystemExit(f"TIP4P n {seen} does not cover {wanted}")
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    if not raw:
        raise SystemExit("empty /etc/machine-id")
    return hashlib.sha256(raw).hexdigest()


def affinity_list() -> str:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list"):
            return line.split(":", 1)[1].strip()
    return ""


def git_out(tree: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", "-C", str(tree), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


def assert_paper_scripts(tree: Path) -> None:
    for relative, needle in PAPER_BUDGET_CHECKS:
        text = (tree / relative).read_text()
        if needle not in text:
            raise SystemExit(f"{relative} missing expected PAPER_BUDGETS declaration")


def assert_arm_tree(arm: str, tree: Path) -> tuple[str, str]:
    status = git_out(tree, ["status", "--porcelain"])
    if status:
        raise SystemExit(f"arm {arm} worktree is dirty:\n{status}")
    commit = git_out(tree, ["rev-parse", "HEAD"])
    tree_hash = git_out(tree, ["rev-parse", "HEAD^{tree}"])
    if arm == "D":
        if commit != EXPECT_DEVELOP_COMMIT or tree_hash != EXPECT_DEVELOP_TREE:
            raise SystemExit(f"arm D expected {EXPECT_DEVELOP_COMMIT}/{EXPECT_DEVELOP_TREE}, got {commit}/{tree_hash}")
    elif arm in ("B9", "B9'"):
        if commit != EXPECT_B9_COMMIT or tree_hash != EXPECT_B9_TREE:
            raise SystemExit(f"arm {arm} expected {EXPECT_B9_COMMIT}/{EXPECT_B9_TREE}, got {commit}/{tree_hash}")
    elif arm == "B10":
        if tree_hash != EXPECT_B10_TREE:
            raise SystemExit(f"arm B10 expected tree {EXPECT_B10_TREE}, got {tree_hash}")
    else:
        raise SystemExit(f"unknown arm {arm}")
    return commit, tree_hash


def driver_cargo_toml(worktree: Path) -> str:
    return (
        "[workspace]\n\n"
        "[package]\n"
        'name = "tip4p-gate-driver"\n'
        'version = "0.1.0"\n'
        'edition = "2024"\n\n'
        "[[bin]]\n"
        'name = "tip4p_gate_driver"\n'
        'path = "src/main.rs"\n\n'
        "[dependencies]\n"
        f'anneal-core = {{ path = "{worktree.resolve()}" }}\n'
        'eindir-core = { version = "=0.6.0", features = ["capi"] }\n'
        'ndarray = "=0.17.2"\n'
        'rand = "=0.9.5"\n'
        'sha2 = "=0.10.9"\n'
    )


def build_arms(args: argparse.Namespace) -> None:
    arm_trees: dict[str, Path] = {}
    for spec in args.arm:
        name, path_str = spec.split("=", 1)
        if name not in ARMS:
            raise SystemExit(f"unknown arm {name}")
        arm_trees[name] = Path(path_str).resolve()
    if set(arm_trees) != set(ARMS):
        raise SystemExit(f"must provide all arms {ARMS}")
    build_root = Path(args.build_root).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    cargo_home = os.environ.get("CARGO_HOME", "")
    for arm in ARMS:
        tree = arm_trees[arm]
        commit, tree_hash = assert_arm_tree(arm, tree)
        assert_paper_scripts(tree)
        slug = ARM_SLUGS[arm]
        crate_dir = build_root / slug / "crate"
        target_dir = build_root / slug / "target"
        if crate_dir.exists():
            shutil.rmtree(crate_dir)
        (crate_dir / "src").mkdir(parents=True, exist_ok=True)
        (crate_dir / "Cargo.toml").write_text(driver_cargo_toml(tree), encoding="utf-8", newline="\n")
        shutil.copy2(DRIVER_SRC, crate_dir / "src" / "main.rs")
        shutil.copy2(tree / "Cargo.lock", crate_dir / "Cargo.lock")
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(target_dir)
        remap = [
            f"--remap-path-prefix={tree}=/src",
            f"--remap-path-prefix={crate_dir}=/driver",
        ]
        if cargo_home:
            remap.append(f"--remap-path-prefix={cargo_home}=/cargo")
        env["RUSTFLAGS"] = " ".join(remap)
        cmd = ["cargo", "+1.88.0", "build", "--release", "--offline", "-j", str(args.jobs)]
        proc = subprocess.run(cmd, cwd=crate_dir, env=env, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            # Retry without --offline only if a crate is not yet in the local cargo cache.
            cmd_online = ["cargo", "+1.88.0", "build", "--release", "-j", str(args.jobs)]
            proc = subprocess.run(cmd_online, cwd=crate_dir, env=env, text=True, capture_output=True, check=False)
            if proc.returncode != 0:
                raise SystemExit(f"build failed for arm {arm}:\n{proc.stdout}\n{proc.stderr}")
        binary = target_dir / "release" / "tip4p_gate_driver"
        if not binary.is_file():
            raise SystemExit(f"missing built binary for {arm}: {binary}")
        print(
            f"built arm={arm} commit={commit} tree={tree_hash} bin_sha256={sha256_file(binary)}",
            flush=True,
        )


def parse_gate_row(text: str, system: str, seed: int, budget: int, reference: float) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("gate_row ")]
    if len(lines) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one gate_row\n{text[-2000:]}")
    fields = dict(part.split("=", 1) for part in lines[0].split()[1:])
    if fields.get("system") != system or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: mismatched header {lines[0]}")
    best_token = fields["best"]
    best_bits = fields["best_bits"]
    if bits_of(best_token) != best_bits:
        raise RuntimeError(f"{system} seed {seed}: best_bits mismatch {best_token} vs {best_bits}")
    outcome_token = fields["outcome_best"]
    outcome_bits = fields["outcome_best_bits"]
    if bits_of(outcome_token) != outcome_bits:
        raise RuntimeError(f"{system} seed {seed}: outcome_best_bits mismatch {outcome_token} vs {outcome_bits}")
    hit = int(fields["hit"])
    if hit not in (0, 1):
        raise RuntimeError(f"{system} seed {seed}: invalid hit {hit}")
    expected_hit = 1 if energy_hits(float(best_token), reference, DEFAULT_HIT_TOL) else 0
    if hit != expected_hit:
        raise RuntimeError(f"{system} seed {seed}: hit {hit} != expected {expected_hit}")
    calls = int(fields["calls"])
    charged = int(fields["charged"])
    if max(calls, charged) < budget or calls < 1 or charged < 1:
        raise RuntimeError(f"{system} seed {seed}: calls={calls} charged={charged} budget={budget}")
    calls_to_hit = "" if fields["calls_to_hit"] == "-" else str(int(fields["calls_to_hit"]))
    if bool(hit) != bool(calls_to_hit):
        raise RuntimeError(f"{system} seed {seed}: hit={hit} inconsistent with calls_to_hit={calls_to_hit!r}")
    if calls_to_hit and not (1 <= int(calls_to_hit) <= budget):
        raise RuntimeError(f"{system} seed {seed}: calls_to_hit out of range {calls_to_hit}")
    wall_s = float(fields["wall_s"])
    inside_s = float(fields["inside_s"])
    outside_s = float(fields["outside_s"])
    if wall_s < 0.0 or inside_s < 0.0 or outside_s < 0.0:
        raise RuntimeError(f"{system} seed {seed}: negative timing")
    if abs(outside_s - max(wall_s - inside_s, 0.0)) > 1e-6:
        raise RuntimeError(f"{system} seed {seed}: outside_s {outside_s} != wall_s - inside_s")
    return {
        "hit": str(hit),
        "calls": str(calls),
        "calls_to_hit": calls_to_hit,
        "charged": str(charged),
        "wall_s": f"{wall_s:.9f}",
        "inside_s": f"{inside_s:.9f}",
        "outside_s": f"{outside_s:.9f}",
        "best": best_token,
        "best_bits": best_bits,
        "outcome_best": outcome_token,
        "outcome_best_bits": outcome_bits,
        "budget": str(budget),
        "start_sha256": fields["start_sha256"],
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise SystemExit(f"unexpected row header: {reader.fieldnames}")
        return list(reader)


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def parse_arm_run_spec(text: str) -> tuple[str, Path, Path]:
    name, rest = text.split("=", 1)
    bin_str, tree_str = rest.rsplit("@", 1)
    if name not in ARMS:
        raise SystemExit(f"unknown arm {name}")
    return name, Path(bin_str).resolve(), Path(tree_str).resolve()


def execute(args: argparse.Namespace) -> None:
    if affinity_list() != CORE:
        raise SystemExit(f"process must be pinned to core {CORE}, got {affinity_list()!r}")
    if shutil.which("taskset") is None:
        raise SystemExit("taskset is required")
    arms: dict[str, tuple[Path, Path]] = {}
    for spec in args.arm:
        name, binary, tree = parse_arm_run_spec(spec)
        arms[name] = (binary, tree)
    if set(arms) != set(ARMS):
        raise SystemExit(f"must specify all four arms {ARMS}")
    arm_meta: dict[str, tuple[str, str]] = {}
    csv_hashes: set[str] = set()
    for arm in ARMS:
        binary, tree = arms[arm]
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise SystemExit(f"arm {arm} binary missing or not executable: {binary}")
        commit, tree_hash = assert_arm_tree(arm, tree)
        assert_paper_scripts(tree)
        arm_meta[arm] = (commit, tree_hash)
        csv_hashes.add(sha256_file(tree / "data" / "tip4p_ccd.csv"))
    if len(csv_hashes) != 1:
        raise SystemExit("data/tip4p_ccd.csv differs across arm worktrees")

    cells = tip4p_cells((arms["B10"][1] / "data" / "tip4p_ccd.csv").read_text())
    systems = [name for name, _n, _energy in cells]
    sizes = {name: n for name, n, _energy in cells}
    references = {name: energy for name, _n, energy in cells}
    jobs = planned_jobs(systems)
    assert_balanced(jobs, systems)

    machine_id = hashed_machine_id()
    session = args.session or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ-tip4p-b10-gate")
    rows_path = Path(args.rows).resolve()
    logs_dir = Path(args.logs).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    existing = read_rows(rows_path)
    if existing:
        if len(existing) > len(jobs):
            raise SystemExit("stored rows exceed schedule length")
        session = existing[0]["session"]
        for idx, row in enumerate(existing):
            system, seed, slot, arm = jobs[idx]
            if (row["system"], int(row["seed"]), int(row["slot"]), row["arm"]) != (system, seed, slot, arm):
                raise SystemExit(f"stored row {idx} does not match schedule")
            commit, tree_hash = arm_meta[arm]
            if (
                row["library_commit"] != commit
                or row["library_tree"] != tree_hash
                or row["machine_id"] != machine_id
                or row["session"] != session
                or row["core"] != CORE
                or int(row["budget"]) != TIP4P_BUDGET
            ):
                raise SystemExit(f"stored row {idx} metadata mismatch")

    print(f"session={session} machine_id={machine_id} jobs={len(jobs)} resuming_from={len(existing)}", flush=True)
    for arm in ARMS:
        commit, tree_hash = arm_meta[arm]
        print(f"arm={arm} commit={commit} tree={tree_hash}", flush=True)
    counts = pair_counts(jobs)
    print(
        "abba_counts " + " ".join(f"s{slot}:{arm}={counts[(slot, arm)]}" for slot in range(len(ARMS)) for arm in ARMS),
        flush=True,
    )

    env = single_thread_env()
    starts_by_seed: dict[tuple[str, int], str] = {}
    for row in existing:
        key = (row["system"], int(row["seed"]))
        prev = starts_by_seed.get(key)
        if prev is not None and prev != row["start_sha256"]:
            raise SystemExit(f"start mismatch in stored rows for {key}")
        starts_by_seed[key] = row["start_sha256"]

    for idx, (system, seed, slot, arm) in enumerate(jobs):
        if idx < len(existing):
            continue
        binary, tree = arms[arm]
        n = sizes[system]
        ref = references[system]
        command = [
            "taskset",
            "-c",
            CORE,
            str(binary),
            str(n),
            str(TIP4P_BUDGET),
            str(seed),
            repr(ref),
        ]
        proc = subprocess.run(
            command,
            cwd=tree,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=JOB_TIMEOUT_S,
        )
        slug = ARM_SLUGS[arm]
        log_path = logs_dir / f"{idx:04d}-{system}-{slug}-s{seed}.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"job_index={idx} system={system} arm={arm} seed={seed} slot={slot} core={CORE}\n")
            handle.write(proc.stdout)
            if proc.stdout and not proc.stdout.endswith("\n"):
                handle.write("\n")
            if proc.stderr:
                handle.write(proc.stderr)
                if not proc.stderr.endswith("\n"):
                    handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if proc.returncode != 0:
            raise SystemExit(f"job {idx} {system} {arm} s{seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
        parsed = parse_gate_row(proc.stdout, system, seed, TIP4P_BUDGET, ref)
        key = (system, seed)
        prev_start = starts_by_seed.get(key)
        if prev_start is not None and prev_start != parsed["start_sha256"]:
            raise SystemExit(f"start mismatch for {system} seed {seed}: {prev_start} != {parsed['start_sha256']}")
        starts_by_seed[key] = parsed["start_sha256"]
        commit, tree_hash = arm_meta[arm]
        row = {
            "system": system,
            "arm": arm,
            "seed": str(seed),
            "slot": str(slot),
            "core": CORE,
            **parsed,
            "machine_id": machine_id,
            "library_commit": commit,
            "library_tree": tree_hash,
            "session": session,
        }
        append_row(rows_path, row)
        print(
            f"[{idx + 1:03d}/{len(jobs)}] {system} {arm:3s} s{seed} slot={slot} "
            f"hit={row['hit']} calls={row['calls']} cth={row['calls_to_hit'] or '-'} "
            f"charged={row['charged']} wall={float(row['wall_s']):.3f}s "
            f"out/call={float(row['outside_s']) / int(row['charged']) * 1e6:.3f}us "
            f"bits={row['best_bits']}",
            flush=True,
        )


def evaluate_rows(rows: list[dict[str, str]], cells: list[tuple[str, int, float]]) -> dict[str, object]:
    systems = [name for name, _n, _energy in cells]
    references = {name: energy for name, _n, energy in cells}
    jobs = planned_jobs(systems)
    assert_balanced(jobs, systems)
    if len(rows) != len(jobs):
        raise SystemExit(f"expected {len(jobs)} rows, found {len(rows)}")

    sessions = {r["session"] for r in rows}
    machines = {r["machine_id"] for r in rows}
    cores = {r["core"] for r in rows}
    budgets = {int(r["budget"]) for r in rows}
    if len(sessions) != 1 or len(machines) != 1 or cores != {CORE} or budgets != {TIP4P_BUDGET}:
        raise SystemExit(f"session/machine/core/budget invariant failed: {sessions} {machines} {cores} {budgets}")

    # Verify exact schedule, bit round-trip, and identical starts across all 4 arms.
    by_key: dict[tuple[str, int, str], dict[str, str]] = {}
    starts: dict[tuple[str, int], set[str]] = {}
    arm_commits: dict[str, set[str]] = {arm: set() for arm in ARMS}
    arm_trees: dict[str, set[str]] = {arm: set() for arm in ARMS}
    for idx, row in enumerate(rows):
        system, seed, slot, arm = jobs[idx]
        if (row["system"], int(row["seed"]), int(row["slot"]), row["arm"]) != (system, seed, slot, arm):
            raise SystemExit(f"row {idx} breaks schedule")
        if bits_of(row["best"]) != row["best_bits"]:
            raise SystemExit(f"row {idx} best_bits mismatch")
        if bits_of(row["outcome_best"]) != row["outcome_best_bits"]:
            raise SystemExit(f"row {idx} outcome_best_bits mismatch")
        expected_hit = 1 if energy_hits(float(row["best"]), references[system], DEFAULT_HIT_TOL) else 0
        if int(row["hit"]) != expected_hit:
            raise SystemExit(f"row {idx} hit flag inconsistent with reference")
        by_key[(system, seed, arm)] = row
        starts.setdefault((system, seed), set()).add(row["start_sha256"])
        arm_commits[arm].add(row["library_commit"])
        arm_trees[arm].add(row["library_tree"])

    for key, hashes in starts.items():
        if len(hashes) != 1:
            raise SystemExit(f"starts differ across arms for {key}: {hashes}")
    for arm in ARMS:
        if len(arm_commits[arm]) != 1 or len(arm_trees[arm]) != 1:
            raise SystemExit(f"arm {arm} mixed commits/trees")

    # Rule (1): B10 vs B9 bit-identity of best, outcome_best, calls, charged.
    mismatches_b10_b9: list[dict[str, str]] = []
    mismatches_aa: list[dict[str, str]] = []
    for system in systems:
        for seed in SEEDS:
            r9 = by_key[(system, seed, "B9")]
            r10 = by_key[(system, seed, "B10")]
            r9p = by_key[(system, seed, "B9'")]
            diffs = []
            for field in ("best_bits", "outcome_best_bits", "calls", "charged", "calls_to_hit", "hit"):
                if r10[field] != r9[field]:
                    diffs.append(f"{field}: B9={r9[field]} vs B10={r10[field]}")
            if diffs:
                mismatches_b10_b9.append({"system": system, "seed": str(seed), "detail": "; ".join(diffs)})
            aa_diffs = []
            for field in ("best_bits", "outcome_best_bits", "calls", "charged", "calls_to_hit", "hit"):
                if r9p[field] != r9[field]:
                    aa_diffs.append(f"{field}: B9={r9[field]} vs B9'={r9p[field]}")
            if aa_diffs:
                mismatches_aa.append({"system": system, "seed": str(seed), "detail": "; ".join(aa_diffs)})

    # Rule (2) and informational stats per cell.
    cell_summaries: list[dict[str, object]] = []
    timing_failures: list[str] = []
    for system, n, reference in cells:
        med_opc: dict[str, float] = {}
        hits: dict[str, int] = {}
        ranked_cth: dict[str, list[float]] = {}
        for arm in ARMS:
            group = [by_key[(system, seed, arm)] for seed in SEEDS]
            opcs = [float(r["outside_s"]) / int(r["charged"]) for r in group]
            med_opc[arm] = type7(opcs, 0.5)
            hits[arm] = sum(int(r["hit"]) for r in group)
            ranked_cth[arm] = [
                float(int(r["calls_to_hit"])) if int(r["hit"]) == 1 else float(TIP4P_BUDGET + 1)
                for r in group
            ]
        ratio_b10_b9 = med_opc["B10"] / med_opc["B9"]
        ratio_b10_d = med_opc["B10"] / med_opc["D"]
        ratio_aa = med_opc["B9"] / med_opc["B9'"]
        aa_spread = max(ratio_aa, 1.0 / ratio_aa)
        pass_b10_b9 = (ratio_b10_b9 <= RATIO_LIMIT) or (ratio_b10_b9 <= aa_spread)
        pass_b10_d = (ratio_b10_d <= RATIO_LIMIT) or (ratio_b10_d <= aa_spread)
        cell_pass = pass_b10_b9 and pass_b10_d
        reasons = []
        if not pass_b10_b9:
            reasons.append(f"B10/B9={ratio_b10_b9:.4f}>max(1.10,{aa_spread:.4f})")
        if not pass_b10_d:
            reasons.append(f"B10/D={ratio_b10_d:.4f}>max(1.10,{aa_spread:.4f})")
        if not cell_pass:
            timing_failures.append(f"{system}: {', '.join(reasons)}")

        fisher_p = fisher_exact_two_sided(hits["B10"], len(SEEDS), hits["D"], len(SEEDS))
        mwu_u, mwu_p = mann_whitney_two_sided(ranked_cth["B10"], ranked_cth["D"])
        cell_summaries.append(
            {
                "system": system,
                "n": n,
                "reference": reference,
                "med_opc": med_opc,
                "ratio_b10_b9": ratio_b10_b9,
                "ratio_b10_d": ratio_b10_d,
                "ratio_aa": ratio_aa,
                "aa_spread": aa_spread,
                "pass_b10_b9": pass_b10_b9,
                "pass_b10_d": pass_b10_d,
                "timing_pass": cell_pass,
                "fail_reason": "; ".join(reasons) if reasons else "none",
                "hits": hits,
                "fisher_b10_d_p": fisher_p,
                "mwu_b10_d_u": mwu_u,
                "mwu_b10_d_p": mwu_p,
                "med_cth_d": type7(ranked_cth["D"], 0.5),
                "med_cth_b10": type7(ranked_cth["B10"], 0.5),
            }
        )

    rule1_pass = len(mismatches_b10_b9) == 0
    rule2_pass = len(timing_failures) == 0
    verdict = "PASS" if (rule1_pass and rule2_pass) else "FIX"
    return {
        "verdict": verdict,
        "rule1_pass": rule1_pass,
        "rule2_pass": rule2_pass,
        "mismatches_b10_b9": mismatches_b10_b9,
        "mismatches_aa": mismatches_aa,
        "timing_failures": timing_failures,
        "cells": cell_summaries,
        "session": next(iter(sessions)),
        "machine_id": next(iter(machines)),
        "arm_commits": {arm: next(iter(vals)) for arm, vals in arm_commits.items()},
        "arm_trees": {arm: next(iter(vals)) for arm, vals in arm_trees.items()},
        "abba_counts": pair_counts(jobs),
        "total_rows": len(rows),
    }


def render_readme(result: dict[str, object], rows_sha256: str) -> str:
    verdict = result["verdict"]
    arm_commits = result["arm_commits"]
    arm_trees = result["arm_trees"]
    cells = result["cells"]
    mismatches = result["mismatches_b10_b9"]
    mismatches_aa = result["mismatches_aa"]
    timing_failures = result["timing_failures"]
    abba_counts = result["abba_counts"]

    b9p = "B9'"
    lines: list[str] = [
        "# tip4p-hits-b10 independent gate",
        "",
        f"Verdict: **{verdict}**.",
        "",
        "## Arms",
        "",
        "| Arm | Commit | Tree | Role |",
        "| --- | --- | --- | --- |",
        f"| `D` | `{arm_commits['D']}` | `{arm_trees['D']}` | `develop` baseline |",
        f"| `B9` | `{arm_commits['B9']}` | `{arm_trees['B9']}` | b9 baseline |",
        f"| `B10` | `{arm_commits['B10']}` | `{arm_trees['B10']}` | b10 candidate (kit tip) |",
        f"| `B9'` | `{arm_commits[b9p]}` | `{arm_trees[b9p]}` | second independent release build of b9 (A/A band) |",
        "",
        "## Protocol",
        "",
        "- Panel: all 14 rigid TIP4P cells `n=8..21` with Cambridge Cluster Database references in `data/tip4p_ccd.csv`.",
        f"- Budget: `{TIP4P_BUDGET}` evaluations per run (`PAPER_BUDGETS`, no `H2H_*` environment variables, budget-only stop, no warm-up).",
        f"- Seeds: `{SEEDS[0]}..{SEEDS[-1]}` (`n={len(SEEDS)}` per arm per cell, `{result['total_rows']}` rows total).",
        f"- Execution: single serial session (`{result['session']}`), whole process tree pinned via `taskset -c {CORE}`, all thread environment variables set to `1`, identical starting geometries verified by SHA-256 across all four arms for every `(system, seed)`.",
        f"- Hashed machine id: `{result['machine_id']}`.",
        f"- `rows.csv` sha256: `{rows_sha256}`.",
        "",
        "### Balanced ABBA rotation check",
        "",
        "Williams-balanced 4-arm ABBA cycle across 4-seed blocks (`(D, B9, B9', B10)`, `(B10, B9', B9, D)`, `(B9, B10, D, B9')`, `(B9', D, B10, B9)`). Global `(slot, arm)` counts:",
        "",
        "| Slot | `D` | `B9` | `B10` | `B9'` |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for slot in range(len(ARMS)):
        lines.append(
            f"| {slot} | {abba_counts[(slot, 'D')]} | {abba_counts[(slot, 'B9')]} | "
            f"{abba_counts[(slot, 'B10')]} | {abba_counts[(slot, b9p)]} |"
        )

    lines.extend(
        [
            "",
            "## Declared gate rules",
            "",
            "1. **Bit-identity (`B10` vs `B9`)**: identical `best_bits` (`f64::to_bits`), `outcome_best_bits`, `calls`, and `charged` on every row (`168/168` pairs).",
            f"   - Result: **{'PASS' if result['rule1_pass'] else 'FIX'}** (`{len(mismatches)}` mismatches in `B10` vs `B9`; `{len(mismatches_aa)}` mismatches in `B9` vs `B9'`).",
            "2. **Outside time per charged call**: per-row `outside_s / charged`, Hyndman-Fan type-7 median per cell; require `B10/B9 <= 1.10` or `<= A/A spread`, and `B10/D <= 1.10` or `<= A/A spread` in every cell, where `A/A spread = max(B9/B9', B9'/B9)`.",
            f"   - Result: **{'PASS' if result['rule2_pass'] else 'FIX'}** (`{len(timing_failures)}` failing cells).",
            "",
        ]
    )

    if mismatches:
        lines.extend(["### Rule (1) mismatches (`B10` vs `B9`)", ""])
        for item in mismatches:
            lines.append(f"- `{item['system']}` seed `{item['seed']}`: {item['detail']}")
        lines.append("")

    if timing_failures:
        lines.extend(["### Rule (2) failing cells", ""])
        for item in timing_failures:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend(
        [
            "## Per-cell timing (outside time per charged call)",
            "",
            "Medians are Hyndman-Fan type-7 in microseconds per charged call (`outside_s / charged * 1e6`).",
            "",
            "| Cell | `D` (us/call) | `B9` (us/call) | `B10` (us/call) | `B9'` (us/call) | `B10/B9` | `B10/D` | `B9/B9'` | A/A spread | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for cell in cells:
        m = cell["med_opc"]
        status = "PASS" if cell["timing_pass"] else f"FIX ({cell['fail_reason']})"
        lines.append(
            f"| {cell['system']} | {m['D'] * 1e6:.4f} | {m['B9'] * 1e6:.4f} | "
            f"{m['B10'] * 1e6:.4f} | {m[b9p] * 1e6:.4f} | "
            f"{cell['ratio_b10_b9']:.4f} | {cell['ratio_b10_d']:.4f} | "
            f"{cell['ratio_aa']:.4f} | {cell['aa_spread']:.4f} | {status} |"
        )

    # Summary ratios across cells
    r_b9_vals = [c["ratio_b10_b9"] for c in cells]
    r_d_vals = [c["ratio_b10_d"] for c in cells]
    r_aa_vals = [c["aa_spread"] for c in cells]
    lines.extend(
        [
            "",
            f"Across all 14 cells: `B10/B9` median `{type7(r_b9_vals):.4f}` (range `{min(r_b9_vals):.4f}`..`{max(r_b9_vals):.4f}`), "
            f"`B10/D` median `{type7(r_d_vals):.4f}` (range `{min(r_d_vals):.4f}`..`{max(r_d_vals):.4f}`), "
            f"A/A spread median `{type7(r_aa_vals):.4f}` (max `{max(r_aa_vals):.4f}`).",
            "",
            "## Informational hit counts and calls-to-hit (`B10` vs `D`)",
            "",
            "Hits are scored at `best <= reference + 0.001` kJ/mol within the 2,000,000-call budget. "
            "Fisher p is two-sided exact on `B10` vs `D`. Mann-Whitney U and two-sided asymptotic p "
            "(tie- and continuity-corrected) compare calls-to-hit for `B10` vs `D` with misses ranked worst at `2_000_001`.",
            "",
            "| Cell | Ref (kJ/mol) | Hits `D` | Hits `B9` | Hits `B10` | Hits `B9'` | Fisher p (`B10` vs `D`) | Med calls `D` | Med calls `B10` | MWU U | MWU p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    total_hits = {arm: 0 for arm in ARMS}
    for cell in cells:
        h = cell["hits"]
        for arm in ARMS:
            total_hits[arm] += h[arm]
        lines.append(
            f"| {cell['system']} | {cell['reference']:.10f} | "
            f"{h['D']}/12 | {h['B9']}/12 | {h['B10']}/12 | {h[b9p]}/12 | "
            f"{cell['fisher_b10_d_p']:.4f} | {cell['med_cth_d']:.1f} | {cell['med_cth_b10']:.1f} | "
            f"{cell['mwu_b10_d_u']:.1f} | {cell['mwu_b10_d_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Total hits across all 14 cells (168 runs/arm): `D` `{total_hits['D']}/168`, "
            f"`B9` `{total_hits['B9']}/168`, `B10` `{total_hits['B10']}/168`, `B9'` `{total_hits[b9p]}/168`.",
            "",
        ]
    )
    return "\n".join(lines)


def report(args: argparse.Namespace) -> None:
    rows_path = Path(args.rows).resolve()
    rows = read_rows(rows_path)
    cells = tip4p_cells((ROOT / "data" / "tip4p_ccd.csv").read_text())
    result = evaluate_rows(rows, cells)
    rows_sha256 = sha256_file(rows_path)
    readme_text = render_readme(result, rows_sha256)
    readme_path = Path(args.readme).resolve()
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_text, encoding="utf-8", newline="\n")
    print(f"verdict={result['verdict']} rule1={result['rule1_pass']} rule2={result['rule2_pass']} wrote={readme_path}", flush=True)


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert abs(type7(sample, 0.5) - 3.5) < 1e-12
    assert abs(type7(sample, 0.25) - 2.25) < 1e-12
    assert abs(fisher_exact_two_sided(6, 12, 6, 12) - 1.0) < 1e-12
    assert fisher_exact_two_sided(12, 12, 0, 12) < 1e-5
    u, p = mann_whitney_two_sided([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(u - 4.5) < 1e-12 and abs(p - 1.0) < 1e-12
    assert bits_of("-305.5183221676") == f"{struct.unpack('<Q', struct.pack('<d', -305.5183221676))[0]:016x}"
    cells = tip4p_cells((ROOT / "data" / "tip4p_ccd.csv").read_text())
    systems = [name for name, _n, _e in cells]
    assert len(systems) == 14 and systems[0] == "tip4p8" and systems[-1] == "tip4p21"
    jobs = planned_jobs(systems)
    assert_balanced(jobs, systems)
    print("self-test ok", flush=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Independent four-arm gate for tip4p-hits-b10")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-test", help="Run harness self-tests")

    p_build = sub.add_parser("build", help="Build independent release drivers for all four arms")
    p_build.add_argument("--arm", action="append", required=True, help="NAME=WORKTREE")
    p_build.add_argument("--build-root", required=True, help="Directory for isolated per-arm builds")
    p_build.add_argument("--jobs", type=int, default=4, help="Cargo build parallelism")

    p_run = sub.add_parser("run", help="Run the full serial gate session on core 0")
    p_run.add_argument("--arm", action="append", required=True, help="NAME=BINARY@WORKTREE")
    p_run.add_argument("--rows", required=True, help="Output CSV path")
    p_run.add_argument("--logs", required=True, help="Output log directory")
    p_run.add_argument("--session", default="", help="Session identifier")

    p_rep = sub.add_parser("report", help="Verify rows.csv and write README verdict")
    p_rep.add_argument("--rows", required=True, help="Input CSV path")
    p_rep.add_argument("--readme", required=True, help="Output README path")

    args = parser.parse_args(argv[1:])
    if args.cmd == "self-test":
        self_test()
    elif args.cmd == "build":
        build_arms(args)
    elif args.cmd == "run":
        execute(args)
    elif args.cmd == "report":
        report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
