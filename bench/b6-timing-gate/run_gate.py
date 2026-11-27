#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session timing gate for rigid TIP4P and one non-rigid spot check.

Arm A and arm B run in one serial session, pinned to core 0, with every
thread pool at one thread. Seeds 2000-2011. Within each cell an even
offset from 2000 runs A then B and an odd offset runs B then A, so the
arm order is ABBA repeated. Slot 0 is the first arm of a seed and slot 1
is the second. The gate counts ``(slot, arm)`` pairs and refuses a schedule
that is not balanced.

Cells are rigid TIP4P ``(H2O)n`` for every n from 8 to 21 that has an
energy in ``data/tip4p_ccd.csv``, at 20000 charged calls, then lj38 at
the budget ``h2h_gmin_proof`` reports for that system. TIP4P is
``h2h_seed`` on the recommended rigid preset and stops at the reference
plus the default hit tolerance. lj38 is ``h2h_gmin_proof run lj38``.

A hit is the driver's flag. Calls to a hit keep that call count; a miss
is ranked at ``budget + 1``, after every hit. The calls test is the
two-sided Mann-Whitney U test, normal approximation, with the tie
correction and the continuity correction. Hits use the two-sided Fisher
exact test. The pooled TIP4P test is the exact two-sided test of the sum
of arm-A hits, conditional on each size's margins.

Outside time per call is ``outside_s / charged``. The cell ratio is the
type-7 median on B divided by the type-7 median on A. Each 95% interval is the type-7 0.025 and 0.975 quantiles of 10000
paired bootstrap replicates from ``random.Random(20261120)``. The pooled
TIP4P ratio pools every TIP4P row the same way; its bootstrap resamples
seeds inside each size and starts from the same generator seed.

Verdict, fixed before the runs: PASS if no cell loses hits (B below A)
and every cell's outside-per-call ratio is at most 1.10. Otherwise FIX,
naming the cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SEEDS = tuple(range(2000, 2012))
TIP4P_BUDGET = 20_000
TIP4P_N_MIN = 8
TIP4P_N_MAX = 21
RATIO_LIMIT = 1.10
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261120
RULE = (
    "PASS if no cell loses hits (B below A) and every cell's "
    "outside-per-call ratio is at most 1.10. Otherwise FIX, naming the cells."
)
ROW_FIELDS = [
    "run_index",
    "system",
    "arm",
    "slot",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "outside_per_call",
    "code_tree",
    "binary_sha256",
    "machine_id",
]
SUMMARY_FIELDS = [
    "system",
    "test",
    "budget",
    "hits_a",
    "hits_b",
    "n",
    "fisher_p",
    "mw_u",
    "mw_p",
    "calls_median_a",
    "calls_median_b",
    "wall_median_a",
    "wall_median_b",
    "pot_median_a",
    "pot_median_b",
    "outside_median_a",
    "outside_median_b",
    "opc_median_a",
    "opc_median_b",
    "opc_ratio",
    "opc_ratio_ci_low",
    "opc_ratio_ci_high",
    "loses_hits",
    "ratio_over_limit",
]
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "GOTO_NUM_THREADS",
)


def type7(values: list[float], p: float) -> float:
    """Hyndman-Fan type 7."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided exact p, summing the tables no more likely than the observed."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """U of ``left`` and the two-sided p, normal approximation with the tie
    and continuity corrections. Every value tied gives p = 1."""
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann-whitney needs two non-empty samples")
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def arm_sequence(seed: int) -> tuple[str, str]:
    """A then B on an even offset from 2000, B then A on an odd offset."""
    if (seed - SEEDS[0]) % 2 == 0:
        return ("A", "B")
    return ("B", "A")


def planned_jobs(systems: list[str]) -> list[tuple[str, int, int, str]]:
    jobs = []
    for system in systems:
        for seed in SEEDS:
            for slot, arm in enumerate(arm_sequence(seed)):
                jobs.append((system, seed, slot, arm))
    return jobs


def pair_counts(jobs: list[tuple[str, int, int, str]]) -> dict[tuple[int, str], int]:
    counts = {(0, "A"): 0, (0, "B"): 0, (1, "A"): 0, (1, "B"): 0}
    for _system, _seed, slot, arm in jobs:
        counts[(slot, arm)] += 1
    return counts


def assert_balanced(jobs: list[tuple[str, int, int, str]], systems: list[str]) -> None:
    expect = len(SEEDS) // 2 * len(systems)
    counts = pair_counts(jobs)
    for key, count in counts.items():
        if count != expect:
            raise SystemExit(f"unbalanced (slot, arm) {key}: {count} != {expect}")
    per = len(SEEDS) // 2
    for system in systems:
        cell = [job for job in jobs if job[0] == system]
        cell_counts = pair_counts(cell)
        for key, count in cell_counts.items():
            if count != per:
                raise SystemExit(f"{system} unbalanced {key}: {count} != {per}")
    if len(jobs) != len(systems) * len(SEEDS) * 2:
        raise SystemExit("job count")
    first = [arm for system, _seed, _slot, arm in jobs if system == systems[0]][:4]
    if first != ["A", "B", "B", "A"]:
        raise SystemExit(f"ABBA opening {first}")


def tip4p_cells(text: str) -> list[tuple[str, int, float]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    found = []
    for row in rows:
        n = int(row["n"])
        if TIP4P_N_MIN <= n <= TIP4P_N_MAX:
            found.append((f"tip4p{n}", n, float(row["energy"])))
    found.sort(key=lambda item: item[1])
    seen = [n for _name, n, _energy in found]
    if seen != sorted(set(seen)):
        raise SystemExit(f"repeated TIP4P n in {seen}")
    return found


def miss_rank(budget: int) -> float:
    """A miss ranks one call worse than a hit on the final charged call."""
    return float(budget + 1)


def ranked_calls(hit: bool, evals: int, budget: int) -> float:
    if hit:
        if not 0 < evals <= budget:
            raise ValueError(f"hit evals {evals} budget {budget}")
        return float(evals)
    return miss_rank(budget)


def ratio_of_medians(left: list[float], right: list[float]) -> float:
    """Median of ``right`` over median of ``left``. Both zero is 1."""
    median_left = type7(left, 0.5)
    median_right = type7(right, 0.5)
    if median_left == 0.0:
        return 1.0 if median_right == 0.0 else math.inf
    return median_right / median_left


def bootstrap_ratio(pairs: list[tuple[float, float]], rng: random.Random) -> tuple[float, float]:
    """Type-7 95% interval for the paired ratio of medians."""
    stats = []
    n = len(pairs)
    for _ in range(BOOTSTRAP_REPS):
        draw = [pairs[rng.randrange(n)] for _ in range(n)]
        stats.append(ratio_of_medians([item[0] for item in draw], [item[1] for item in draw]))
    return type7(stats, 0.025), type7(stats, 0.975)


def bootstrap_pooled(strata: list[list[tuple[float, float]]], rng: random.Random) -> tuple[float, float]:
    """Type-7 95% interval, resampling seeds inside each stratum."""
    stats = []
    for _ in range(BOOTSTRAP_REPS):
        left: list[float] = []
        right: list[float] = []
        for stratum in strata:
            n = len(stratum)
            for _draw in range(n):
                a, b = stratum[rng.randrange(n)]
                left.append(a)
                right.append(b)
        stats.append(ratio_of_medians(left, right))
    return type7(stats, 0.025), type7(stats, 0.975)


def hypergeometric(n_a: int, n_b: int, hits_a: int, hits_b: int) -> list[tuple[int, float]]:
    """Arm-A hit counts given both margins, and their conditional probabilities."""
    hits = hits_a + hits_b
    total = n_a + n_b
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    ways = math.comb(total, hits)
    return [
        (k, math.comb(n_a, k) * math.comb(n_b, hits - k) / ways) for k in range(lo, hi + 1)
    ]


def stratified_fisher(tables: list[tuple[int, int, int, int]]) -> float:
    """Exact two-sided p for the sum of arm-A hits, given each stratum's margins.

    ``tables`` entries are ``(hits_a, n_a, hits_b, n_b)``. A stratum with a
    single admissible count does not move the sum. With no free stratum the
    p-value is 1.
    """
    dist = {0: 1.0}
    observed = 0
    expected = 0.0
    for hits_a, n_a, hits_b, n_b in tables:
        observed += hits_a
        hits = hits_a + hits_b
        total = n_a + n_b
        expected += n_a * hits / total
        folded: dict[int, float] = {}
        for total_hits, mass in dist.items():
            for k, prob in hypergeometric(n_a, n_b, hits_a, hits_b):
                folded[total_hits + k] = folded.get(total_hits + k, 0.0) + mass * prob
        dist = folded
    mass = math.fsum(dist.values())
    if abs(mass - 1.0) > 1e-6:
        raise RuntimeError(f"stratified Fisher mass {mass}")
    dev = abs(observed - expected)
    p = math.fsum(prob for value, prob in dist.items() if abs(value - expected) >= dev - 1e-9)
    return min(1.0, p)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    if not raw:
        raise SystemExit("machine id is empty")
    return hashlib.sha256(raw).hexdigest()


def affinity_list() -> str:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list"):
            return line.split(":", 1)[1].strip()
    return ""


def git_tree(tree: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD^{tree}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "git rev-parse failed")
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


def parse_arm(text: str) -> tuple[str, Path, Path]:
    name, rest = text.split("=", 1)
    binary, tree = rest.rsplit("@", 1)
    if name not in ("A", "B"):
        raise SystemExit(f"arm name {name}")
    return name, Path(binary), Path(tree)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def driver_for(system: str) -> str:
    if system.startswith("tip4p"):
        return "h2h_seed"
    if system == "lj38":
        return "h2h_gmin_proof"
    raise SystemExit(f"no driver for {system}")


def paper_lj38(binary: Path, tree: Path) -> tuple[int, float]:
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary / "h2h_gmin_proof"), "budget", "lj38"],
        cwd=tree,
        env=single_thread_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"lj38 budget exited {proc.returncode}\n{proc.stderr}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"lj38 budget output {proc.stdout!r}")
    return int(lines[0]), float(lines[1])


def run_tip4p(
    binary: Path,
    tree: Path,
    n: int,
    reference: float,
    seed: int,
    scratch: Path,
    log: Path,
) -> dict[str, str]:
    stop = repr(reference + DEFAULT_HIT_TOL)
    prefix = scratch / f"tip4p{n}-{seed}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset",
        "-c",
        "0",
        str(binary / "h2h_seed"),
        "tip4p",
        str(n),
        "0",
        str(TIP4P_BUDGET),
        str(seed),
        "recommended",
        stop,
        str(prefix),
    ]
    return launch(command, tree, log, f"tip4p{n}", seed, TIP4P_BUDGET, reference, kind="h2h_row")


def run_lj38(binary: Path, tree: Path, seed: int, budget: int, reference: float, log: Path) -> dict[str, str]:
    command = ["taskset", "-c", "0", str(binary / "h2h_gmin_proof"), "run", "lj38", str(seed)]
    return launch(command, tree, log, "lj38", seed, budget, reference, kind="csv")


def launch(
    command: list[str],
    tree: Path,
    log: Path,
    system: str,
    seed: int,
    budget: int,
    reference: float,
    kind: str,
) -> dict[str, str]:
    proc = subprocess.run(
        command,
        cwd=tree,
        env=single_thread_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", newline="\n") as handle:
        handle.write(" ".join(command) + "\n")
        handle.write(proc.stdout)
        if not proc.stdout.endswith("\n"):
            handle.write("\n")
        handle.write(proc.stderr)
        handle.flush()
        os.fsync(handle.fileno())
    if proc.returncode != 0:
        raise RuntimeError(f"{system} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    if kind == "h2h_row":
        return parse_h2h_row(proc.stdout, system, seed, budget, reference)
    return parse_csv_row(proc.stdout, system, seed, budget, reference)


def parse_h2h_row(text: str, system: str, seed: int, budget: int, reference: float) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one h2h_row\n{text[-2000:]}")
    fields = dict(part.split("=", 1) for part in rows[0].split()[1:])
    if fields.get("system") != system or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: row {rows[0]}")
    return checked_result(
        system,
        seed,
        budget,
        reference,
        fields["hit"] == "1",
        fields["best"],
        int(fields["evals_to_hit"]),
        int(fields["charged"]),
        float(fields["wall_s"]),
        float(fields["pot_s"]),
        float(fields["outside_s"]),
    )


def parse_csv_row(text: str, system: str, seed: int, budget: int, reference: float) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("system,") or f",anneal,{seed}," in line]
    headers = [line for line in lines if line.startswith("system,")]
    data = [line for line in lines if not line.startswith("system,")]
    if len(headers) != 1 or len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one result row\n{text[-2000:]}")
    row = dict(zip(headers[0].split(","), data[0].split(",")))
    if row["system"] != system or int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: row {data[0]}")
    hit = row["hit"].strip().lower() in ("1", "true")
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = float(row["outside_s"]) if "outside_s" in row else max(wall - pot, 0.0)
    return checked_result(
        system,
        seed,
        budget,
        reference,
        hit,
        row["best"],
        int(row["evals_to_hit"]),
        int(row["charged"]),
        wall,
        pot,
        outside,
    )


def checked_result(
    system: str,
    seed: int,
    budget: int,
    reference: float,
    hit: bool,
    best_text: str,
    evals: int,
    charged: int,
    wall: float,
    pot: float,
    outside: float,
) -> dict[str, str]:
    best = float(best_text)
    label = f"{system} seed {seed}"
    if hit != energy_hits(best, reference, DEFAULT_HIT_TOL):
        raise RuntimeError(f"{label}: hit {int(hit)} best {best_text} reference {reference}")
    if charged < 1:
        raise RuntimeError(f"{label}: charged {charged}")
    if hit and not 0 < evals <= budget:
        raise RuntimeError(f"{label}: hit evals {evals} budget {budget}")
    if not hit and evals != budget:
        raise RuntimeError(f"{label}: miss evals {evals} budget {budget}")
    if wall < 0.0 or pot < 0.0 or not math.isfinite(best):
        raise RuntimeError(f"{label}: wall {wall} pot {pot} best {best}")
    if outside < 0.0 or abs(outside - max(wall - pot, 0.0)) > 5e-6:
        raise RuntimeError(f"{label}: outside {outside} wall {wall} pot {pot}")
    per_call = outside / charged
    return {
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(charged),
        "best": best_text,
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call": f"{per_call:.12g}",
    }


def have(rows: list[dict[str, str]], system: str, arm: str, seed: int) -> bool:
    return any(row["system"] == system and row["arm"] == arm and int(row["seed"]) == seed for row in rows)


def execute(args: argparse.Namespace) -> None:
    if affinity_list() != "0":
        raise SystemExit(f"this process is not pinned to core 0 (allowed {affinity_list()!r})")
    if shutil_which("taskset") is None:
        raise SystemExit("taskset is required")
    arms = {}
    for text in args.arm:
        name, binary, tree = parse_arm(text)
        arms[name] = (binary, tree)
    if set(arms) != {"A", "B"}:
        raise SystemExit("name both arms A and B")
    trees = {name: git_tree(tree) for name, (_binary, tree) in arms.items()}
    csv_hash = {}
    for name, (_binary, tree) in arms.items():
        path = tree / "data" / "tip4p_ccd.csv"
        csv_hash[name] = sha256_file(path)
    if csv_hash["A"] != csv_hash["B"]:
        raise SystemExit("TIP4P reference files differ")
    cells = tip4p_cells((arms["A"][1] / "data" / "tip4p_ccd.csv").read_text())
    if not cells:
        raise SystemExit("no TIP4P reference from 8 to 21")
    lj_budget = {}
    lj_reference = {}
    for name, (binary, tree) in arms.items():
        lj_budget[name], lj_reference[name] = paper_lj38(binary, tree)
    if lj_budget["A"] != lj_budget["B"] or lj_reference["A"] != lj_reference["B"]:
        raise SystemExit(f"lj38 budget mismatch {lj_budget} {lj_reference}")
    budget_lj = lj_budget["A"]
    reference_lj = lj_reference["A"]
    systems = [name for name, _n, _energy in cells] + ["lj38"]
    jobs = planned_jobs(systems)
    assert_balanced(jobs, systems)
    references = {name: energy for name, _n, energy in cells}
    references["lj38"] = reference_lj
    budgets = {name: TIP4P_BUDGET for name, _n, _energy in cells}
    budgets["lj38"] = budget_lj
    sizes = {name: n for name, n, _energy in cells}
    binaries = {}
    for name, (binary, _tree) in arms.items():
        binaries[(name, "h2h_seed")] = sha256_file(binary / "h2h_seed")
        binaries[(name, "h2h_gmin_proof")] = sha256_file(binary / "h2h_gmin_proof")
    machine = hashed_machine_id()
    print(RULE, flush=True)
    print(
        "trees",
        trees["A"],
        trees["B"],
        "lj38_budget",
        budget_lj,
        "tip4p_budget",
        TIP4P_BUDGET,
        "machine",
        machine,
        flush=True,
    )
    print("abba", " ".join(f"{slot}:{arm}={count}" for (slot, arm), count in sorted(pair_counts(jobs).items())), flush=True)
    rows = read_rows(args.rows)
    logs = args.logs
    scratch = args.scratch
    for index, (system, seed, slot, arm) in enumerate(jobs):
        if have(rows, system, arm, seed):
            stored = next(
                row for row in rows if row["system"] == system and row["arm"] == arm and int(row["seed"]) == seed
            )
            if (
                stored["code_tree"] != trees[arm]
                or stored["binary_sha256"] != binaries[(arm, driver_for(system))]
                or stored["machine_id"] != machine
                or int(stored["run_index"]) != index
                or int(stored["slot"]) != slot
            ):
                raise SystemExit(f"stored {system} {arm} seed {seed} does not match this build")
            continue
        binary, tree = arms[arm]
        log = logs / f"{index:04d}-{system}-{arm}-s{seed}.log"
        if system == "lj38":
            result = run_lj38(binary, tree, seed, budgets[system], references[system], log)
        else:
            result = run_tip4p(
                binary,
                tree,
                sizes[system],
                references[system],
                seed,
                scratch / f"{system}-{arm}-s{seed}",
                log,
            )
        row = {
            "run_index": str(index),
            "system": system,
            "arm": arm,
            "slot": str(slot),
            "seed": str(seed),
            "code_tree": trees[arm],
            "binary_sha256": binaries[(arm, driver_for(system))],
            "machine_id": machine,
            **result,
        }
        append_row(args.rows, row)
        rows.append(row)
        print(",".join(row[field] for field in ROW_FIELDS), flush=True)


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(":"):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sorted(rows, key=lambda item: int(item["run_index"])):
        grouped.setdefault((row["system"], row["arm"]), []).append(row)
    return grouped


def require_schedule(rows: list[dict[str, str]]) -> list[str]:
    systems: list[str] = []
    for row in sorted(rows, key=lambda item: int(item["run_index"])):
        if row["system"] not in systems:
            systems.append(row["system"])
    jobs = planned_jobs(systems)
    assert_balanced(jobs, systems)
    if len(rows) != len(jobs):
        raise SystemExit(f"row count {len(rows)} != jobs {len(jobs)}")
    by_index = {int(row["run_index"]): row for row in rows}
    if sorted(by_index) != list(range(len(jobs))):
        raise SystemExit("run_index is not 0..N-1")
    for index, (system, seed, slot, arm) in enumerate(jobs):
        row = by_index[index]
        if (row["system"], int(row["seed"]), int(row["slot"]), row["arm"]) != (system, seed, slot, arm):
            raise SystemExit(f"row {index} does not match the ABBA schedule")
    return systems


def cell_summary(system: str, left: list[dict[str, str]], right: list[dict[str, str]]) -> dict[str, str]:
    budget = int(left[0]["budget"])
    if any(int(row["budget"]) != budget for row in left + right):
        raise SystemExit(f"{system}: mixed budgets")
    hits_a = sum(row["hit"] == "1" for row in left)
    hits_b = sum(row["hit"] == "1" for row in right)
    calls_a = [ranked_calls(row["hit"] == "1", int(row["evals_to_hit"]), budget) for row in left]
    calls_b = [ranked_calls(row["hit"] == "1", int(row["evals_to_hit"]), budget) for row in right]
    u, mw_p = mann_whitney(calls_a, calls_b)
    opc_a = [float(row["outside_s"]) / int(row["charged"]) for row in left]
    opc_b = [float(row["outside_s"]) / int(row["charged"]) for row in right]
    ratio = ratio_of_medians(opc_a, opc_b)
    pairs = list(zip(opc_a, opc_b, strict=True))
    low, high = bootstrap_ratio(pairs, random.Random(BOOTSTRAP_SEED))
    loses = hits_b < hits_a
    over = not (ratio <= RATIO_LIMIT)

    def med(rows: list[dict[str, str]], key: str) -> float:
        return type7([float(row[key]) for row in rows], 0.5)

    return {
        "system": system,
        "test": "fisher",
        "budget": str(budget),
        "hits_a": str(hits_a),
        "hits_b": str(hits_b),
        "n": str(len(left)),
        "fisher_p": f"{fisher(hits_a, len(left), hits_b, len(right)):.6g}",
        "mw_u": f"{u:.6g}",
        "mw_p": f"{mw_p:.6g}",
        "calls_median_a": f"{type7(calls_a, 0.5):.6g}",
        "calls_median_b": f"{type7(calls_b, 0.5):.6g}",
        "wall_median_a": f"{med(left, 'wall_s'):.6f}",
        "wall_median_b": f"{med(right, 'wall_s'):.6f}",
        "pot_median_a": f"{med(left, 'pot_s'):.6f}",
        "pot_median_b": f"{med(right, 'pot_s'):.6f}",
        "outside_median_a": f"{med(left, 'outside_s'):.6f}",
        "outside_median_b": f"{med(right, 'outside_s'):.6f}",
        "opc_median_a": f"{type7(opc_a, 0.5):.6g}",
        "opc_median_b": f"{type7(opc_b, 0.5):.6g}",
        "opc_ratio": format_ratio(ratio),
        "opc_ratio_ci_low": format_ratio(low),
        "opc_ratio_ci_high": format_ratio(high),
        "loses_hits": "1" if loses else "0",
        "ratio_over_limit": "1" if over else "0",
    }


def format_ratio(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6g}"


def pooled_summary(cells: list[dict[str, str]], strata: list[list[tuple[float, float]]], tables: list[tuple[int, int, int, int]]) -> dict[str, str]:
    left = [pair[0] for stratum in strata for pair in stratum]
    right = [pair[1] for stratum in strata for pair in stratum]
    ratio = ratio_of_medians(left, right)
    low, high = bootstrap_pooled(strata, random.Random(BOOTSTRAP_SEED))
    hits_a = sum(int(row["hits_a"]) for row in cells)
    hits_b = sum(int(row["hits_b"]) for row in cells)
    n = sum(int(row["n"]) for row in cells)
    return {
        "system": "tip4p-pooled",
        "test": "stratified-fisher",
        "budget": str(TIP4P_BUDGET),
        "hits_a": str(hits_a),
        "hits_b": str(hits_b),
        "n": str(n),
        "fisher_p": f"{stratified_fisher(tables):.6g}",
        "mw_u": "",
        "mw_p": "",
        "calls_median_a": "",
        "calls_median_b": "",
        "wall_median_a": "",
        "wall_median_b": "",
        "pot_median_a": "",
        "pot_median_b": "",
        "outside_median_a": "",
        "outside_median_b": "",
        "opc_median_a": f"{type7(left, 0.5):.6g}",
        "opc_median_b": f"{type7(right, 0.5):.6g}",
        "opc_ratio": format_ratio(ratio),
        "opc_ratio_ci_low": format_ratio(low),
        "opc_ratio_ci_high": format_ratio(high),
        "loses_hits": "",
        "ratio_over_limit": "",
    }


def changed_seeds(grouped: dict[tuple[str, str], list[dict[str, str]]], systems: list[str]) -> list[str]:
    lines = []
    for system in systems:
        left = {int(row["seed"]): row for row in grouped[(system, "A")]}
        right = {int(row["seed"]): row for row in grouped[(system, "B")]}
        for seed in SEEDS:
            hit_a = left[seed]["hit"] == "1"
            hit_b = right[seed]["hit"] == "1"
            if hit_a == hit_b:
                continue
            if hit_a and not hit_b:
                lines.append(f"{system} seed {seed}: A hit, B miss")
            else:
                lines.append(f"{system} seed {seed}: A miss, B hit")
    return lines


def lj38_lines(grouped: dict[tuple[str, str], list[dict[str, str]]]) -> list[str]:
    left = {int(row["seed"]): row for row in grouped[("lj38", "A")]}
    right = {int(row["seed"]): row for row in grouped[("lj38", "B")]}
    lines = []
    best_same = charged_same = evals_same = True
    for seed in SEEDS:
        a, b = left[seed], right[seed]
        best = a["best"] == b["best"]
        charged = a["charged"] == b["charged"]
        evals = a["evals_to_hit"] == b["evals_to_hit"]
        best_same &= best
        charged_same &= charged
        evals_same &= evals
        lines.append(
            f"seed {seed}: best {a['best']} vs {b['best']} ({'identical' if best else 'differs'}), "
            f"charged {a['charged']} vs {b['charged']} ({'identical' if charged else 'differs'}), "
            f"evals_to_hit {a['evals_to_hit']} vs {b['evals_to_hit']} ({'identical' if evals else 'differs'})"
        )
    head = []
    head.append(
        "Best energy is identical on every lj38 seed."
        if best_same
        else "Best energy differs on at least one lj38 seed."
    )
    head.append(
        "Charged call count is identical on every lj38 seed."
        if charged_same
        else "Charged call count differs on at least one lj38 seed."
    )
    head.append(
        "Calls to the hit are identical on every lj38 seed."
        if evals_same
        else "Calls to the hit differ on at least one lj38 seed."
    )
    return head + lines


def write_table(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def verdict_of(cells: list[dict[str, str]]) -> tuple[str, list[str]]:
    named = []
    for row in cells:
        reasons = []
        if row["loses_hits"] == "1":
            reasons.append("hits")
        if row["ratio_over_limit"] == "1":
            reasons.append("outside-per-call")
        if reasons:
            named.append(f"{row['system']} ({', '.join(reasons)})")
    if named:
        return "FIX", named
    return "PASS", []


def render_readme(
    rows: list[dict[str, str]],
    cells: list[dict[str, str]],
    pooled: dict[str, str],
    systems: list[str],
) -> str:
    verdict, named = verdict_of(cells)
    trees = sorted({(row["arm"], row["code_tree"]) for row in rows})
    binaries = sorted({(row["arm"], row["system"], row["binary_sha256"]) for row in rows})
    machines = sorted({row["machine_id"] for row in rows})
    counts = pair_counts(
        [(row["system"], int(row["seed"]), int(row["slot"]), row["arm"]) for row in rows]
    )
    grouped = group_rows(rows)
    changes = changed_seeds(grouped, systems)
    identity = lj38_lines(grouped)
    lines = [
        "# b6 timing gate",
        "",
        RULE,
        "",
        f"Verdict: {verdict}",
        "",
    ]
    if named:
        lines.append("Cells: " + "; ".join(named) + ".")
        lines.append("")
    lines.extend(
        [
            "Seeds 2000-2011. TIP4P budget 20000 calls. lj38 budget "
            f"{next(row['budget'] for row in cells if row['system'] == 'lj38')} calls. "
            "One serial session, core 0, one thread in each pool. "
            "Arms alternate ABBA. Misses rank at budget + 1 for the calls test. "
            "Outside time per call is outside seconds divided by charged calls. "
            "The ratio is the type-7 median on B over the type-7 median on A. "
            f"The interval is a paired bootstrap of {BOOTSTRAP_REPS} replicates, "
            f"seed {BOOTSTRAP_SEED}, type-7 quantiles at 0.025 and 0.975. "
            "The pooled TIP4P p-value is the exact two-sided test of the sum of "
            "arm-A hits conditional on each size's margins. The pooled ratio "
            "uses every TIP4P row, and its bootstrap resamples inside each size.",
            "",
            "## Verdict",
            "",
            "| cell | hits A | hits B | Fisher p | calls A | calls B | MW U | MW p | outside/call A | outside/call B | ratio | CI low | CI high |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in cells:
        lines.append(
            "| {system} | {hits_a}/{n} | {hits_b}/{n} | {fisher_p} | {calls_median_a} | {calls_median_b} | {mw_u} | {mw_p} | {opc_median_a} | {opc_median_b} | {opc_ratio} | {opc_ratio_ci_low} | {opc_ratio_ci_high} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Time",
            "",
            "| cell | wall A | wall B | pot A | pot B | outside A | outside B |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in cells:
        lines.append(
            "| {system} | {wall_median_a} | {wall_median_b} | {pot_median_a} | {pot_median_b} | {outside_median_a} | {outside_median_b} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Pooled TIP4P",
            "",
            f"Stratified Fisher p {pooled['fisher_p']}. "
            f"Hits {pooled['hits_a']}/{pooled['n']} on A and {pooled['hits_b']}/{pooled['n']} on B. "
            f"Outside per call median {pooled['opc_median_a']} on A and {pooled['opc_median_b']} on B. "
            f"Ratio {pooled['opc_ratio']} with 95% interval {pooled['opc_ratio_ci_low']} to {pooled['opc_ratio_ci_high']}.",
            "",
            "## Hit status changes",
            "",
        ]
    )
    if changes:
        lines.extend(f"- {line}" for line in changes)
    else:
        lines.append("No seed changed hit status in either direction.")
    lines.extend(["", "## lj38", ""])
    lines.extend(identity)
    lines.extend(["", "## Schedule and build", ""])
    lines.append(
        "Counted (slot, arm) pairs: "
        + ", ".join(f"({slot}, {arm}) = {count}" for (slot, arm), count in sorted(counts.items()))
        + "."
    )
    lines.append("")
    lines.append("Code trees:")
    for arm, tree in trees:
        lines.append(f"- {arm} `{tree}`")
    lines.append("")
    lines.append("Binary sha256:")
    seen_bins: list[tuple[str, str, str]] = []
    for arm, system, digest in binaries:
        driver = driver_for(system)
        key = (arm, driver, digest)
        if key not in seen_bins:
            seen_bins.append(key)
    for arm, driver, digest in seen_bins:
        lines.append(f"- {arm} {driver} `{digest}`")
    lines.append("")
    lines.append("Hashed machine id: `" + machines[0] + "`.")
    if len(machines) != 1:
        lines.append("Machine ids differ across rows.")
    lines.append("")
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> None:
    rows = read_rows(args.rows)
    if not rows:
        raise SystemExit("no rows")
    systems = require_schedule(rows)
    grouped = group_rows(rows)
    for system in systems:
        for arm in ("A", "B"):
            seeds = [int(row["seed"]) for row in grouped[(system, arm)]]
            if seeds != list(SEEDS):
                raise SystemExit(f"{system} {arm} seeds {seeds}")
    machines = {row["machine_id"] for row in rows}
    if len(machines) != 1:
        raise SystemExit("more than one machine id")
    for arm in ("A", "B"):
        trees = {row["code_tree"] for row in rows if row["arm"] == arm}
        if len(trees) != 1:
            raise SystemExit(f"arm {arm} has more than one tree")
    cells = []
    strata = []
    tables = []
    for system in systems:
        left = grouped[(system, "A")]
        right = grouped[(system, "B")]
        cells.append(cell_summary(system, left, right))
        if system.startswith("tip4p"):
            strata.append(
                [
                    (float(a["outside_s"]) / int(a["charged"]), float(b["outside_s"]) / int(b["charged"]))
                    for a, b in zip(left, right)
                ]
            )
            tables.append(
                (
                    sum(row["hit"] == "1" for row in left),
                    len(left),
                    sum(row["hit"] == "1" for row in right),
                    len(right),
                )
            )
    pooled = pooled_summary([row for row in cells if row["system"].startswith("tip4p")], strata, tables)
    write_table(args.summary, SUMMARY_FIELDS, cells + [pooled])
    text = render_readme(rows, cells, pooled, systems)
    args.readme.parent.mkdir(parents=True, exist_ok=True)
    with args.readme.open("w", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    verdict, named = verdict_of(cells)
    print(verdict, flush=True)
    if named:
        print(" ".join(named), flush=True)


def self_test() -> None:
    systems = [f"tip4p{n}" for n in range(8, 22)] + ["lj38"]
    jobs = planned_jobs(systems)
    assert_balanced(jobs, systems)
    if pair_counts([job for job in jobs if job[0] == "lj38"]) != {(0, "A"): 6, (0, "B"): 6, (1, "A"): 6, (1, "B"): 6}:
        raise SystemExit("lj38 pair counts")
    sample = "n,energy,source,sha256\n7,-1,a,b\n8,-305.5,a,b\n21,-916.7,a,b\n22,-1,a,b\n"
    parsed = tip4p_cells(sample)
    if [item[1] for item in parsed] != [8, 21]:
        raise SystemExit(f"tip4p filter {parsed}")
    if abs(fisher(2, 2, 0, 2) - (1.0 / 3.0)) > 1e-12:
        raise SystemExit("fisher 2x2")
    if fisher(12, 12, 12, 12) != 1.0 or fisher(0, 12, 0, 12) != 1.0:
        raise SystemExit("fisher ties")
    u, p = mann_whitney([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    if p != 1.0 or u != 4.5:
        raise SystemExit(f"mann whitney ties {u} {p}")
    u_miss, p_sep = mann_whitney([1.0, 2.0, 3.0], [miss_rank(3), miss_rank(3), miss_rank(3)])
    if u_miss != 0.0 or not p_sep < 0.1:
        raise SystemExit(f"misses did not rank worse {u_miss} {p_sep}")
    if abs(stratified_fisher([(2, 2, 0, 2)]) - fisher(2, 2, 0, 2)) > 1e-12:
        raise SystemExit("stratified single table")
    if stratified_fisher([(0, 12, 0, 12), (12, 12, 12, 12)]) != 1.0:
        raise SystemExit("stratified degenerate")
    brute_tables = [(1, 2, 0, 2), (2, 3, 1, 3)]
    if abs(stratified_fisher(brute_tables) - brute_stratified(brute_tables)) > 1e-12:
        raise SystemExit("stratified brute force")
    ratio = ratio_of_medians([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.4])
    if ratio > RATIO_LIMIT:
        raise SystemExit("ratio limit fixture")
    pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    a = bootstrap_ratio(pairs, random.Random(BOOTSTRAP_SEED))
    b = bootstrap_ratio(pairs, random.Random(BOOTSTRAP_SEED))
    if a != b or a[0] > 1.0 or a[1] < 1.0:
        raise SystemExit(f"bootstrap {a} {b}")
    if float(repr(DEFAULT_HIT_TOL)) != DEFAULT_HIT_TOL:
        raise SystemExit("tolerance round trip")
    if "PASS if no cell loses hits" not in RULE:
        raise SystemExit("rule text")
    if RATIO_LIMIT != 1.10:
        raise SystemExit("ratio limit")
    over = ratio_of_medians([1.0, 1.0], [1.2, 1.2])
    if over <= RATIO_LIMIT:
        raise SystemExit("ratio over the limit was accepted")
    if "\r" in Path(__file__).read_text():
        raise SystemExit("source has CR")
    print("ok")


def brute_stratified(tables: list[tuple[int, int, int, int]]) -> float:
    expected = 0.0
    observed = 0
    factors = []
    for hits_a, n_a, hits_b, n_b in tables:
        observed += hits_a
        total = n_a + n_b
        expected += n_a * (hits_a + hits_b) / total
        factors.append(hypergeometric(n_a, n_b, hits_a, hits_b))
    joint: dict[int, float] = {0: 1.0}
    for factor in factors:
        folded: dict[int, float] = {}
        for value, mass in joint.items():
            for k, prob in factor:
                folded[value + k] = folded.get(value + k, 0.0) + mass * prob
        joint = folded
    dev = abs(observed - expected)
    return min(1.0, math.fsum(prob for value, prob in joint.items() if abs(value - expected) >= dev - 1e-9))


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser(description="TIP4P b6 timing gate")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--rows", type=Path, required=True)
    run_p.add_argument("--logs", type=Path, required=True)
    run_p.add_argument("--scratch", type=Path, required=True)
    run_p.add_argument("--arm", action="append", required=True)
    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("--rows", type=Path, required=True)
    sum_p.add_argument("--summary", type=Path, required=True)
    sum_p.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    if args.command == "run":
        execute(args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
