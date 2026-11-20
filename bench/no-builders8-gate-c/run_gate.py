#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial paired gate, part c, for the no-builders series.

One process at a time. Arms alternate ABBA across seeds 2000-2011: an even
seed runs A then B, an odd seed runs B then A. Both arms read the start
that arm A wrote for that seed. Budgets are the paper budgets. A hit is a
finite best at or below the reference plus 1e-3. Misses rank as worse than
every hit in the calls comparison.

    run_gate.py run --bin-a BIN --bin-b BIN --tree-a DIR --tree-b DIR --out CSV --log-dir DIR --repo DIR
    run_gate.py summarize --rows CSV --readme README
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
HIT_TOL = 1.0e-3
ARM_A = "61fa67cf380e46876b569da45e7cc091647a775b"
TREE_A = "cff796c507a8fa1d0bfb28888c3570b61988055e"
TREE_B = "39ac5e0642beb3e7874a340d2767aa151fa95eac"
TIP4P_BUDGET = 2_000_000
JOB_TIMEOUT_S = 12 * 60 * 60

# Paper budgets. Sutton-Chen sizes follow scripts/metal_h2h/run_sc.py.
# Binary Lennard-Jones sizes follow the Lennard-Jones map in
# scripts/gmin_panel/run_panel.py. Rigid TIP4P uses the 2_000_000 ceiling
# in scripts/tip4p_h2h/run_h2h.py for every listed size.
SYSTEMS = (
    ("sc126-55", 1_000_000, -52884.6806),
    ("sc96-38", 400_000, -8917.7056),
    ("blj105n38", 400_000, -177.2090846748),
    ("blj105n55", 1_000_000, -283.1545313810),
    ("blj115n38", 400_000, -187.3676867453),
    ("blj115n55", 1_000_000, -289.2931853058),
    ("blj130n38", 400_000, -191.1840192602),
    ("blj130n55", 1_000_000, -306.7337643721),
    ("blj130n98", 4_000_000, -590.7874132760),
    ("tip4p8", TIP4P_BUDGET, -305.5183221676),
    ("tip4p9", TIP4P_BUDGET, -344.4352092012),
    ("tip4p10", TIP4P_BUDGET, -391.0226842125),
    ("tip4p11", TIP4P_BUDGET, -431.4890723318),
    ("tip4p12", TIP4P_BUDGET, -492.9082357122),
    ("tip4p13", TIP4P_BUDGET, -532.9705363445),
    ("tip4p14", TIP4P_BUDGET, -582.9911036211),
    ("tip4p15", TIP4P_BUDGET, -628.3722525077),
    ("tip4p16", TIP4P_BUDGET, -681.1925945898),
    ("tip4p17", TIP4P_BUDGET, -723.8077531390),
    ("tip4p18", TIP4P_BUDGET, -773.2318693143),
    ("tip4p19", TIP4P_BUDGET, -821.0369191145),
    ("tip4p20", TIP4P_BUDGET, -872.9887544952),
    ("tip4p21", TIP4P_BUDGET, -916.7074302940),
)

FIELDS = (
    "system",
    "arm",
    "seed",
    "slot",
    "core",
    "hit",
    "calls",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "best",
    "best_bits",
    "budget",
    "machine_id",
    "commit",
    "tree",
    "harness_blob",
    "driver_blob",
)

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "TBB_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "TOKIO_WORKER_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENMP_NUM_THREADS",
)


def bits_to_float(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFFFFFFFFFFFFFF))[0]


def float_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def full_precision(value: float) -> str:
    return repr(value)


def type7(values: list[float], p: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower - 1] * (1.0 - weight) + ordered[upper - 1] * weight


def log_factorial(n: int) -> float:
    return math.lgamma(n + 1)


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b
    row1, row2, col1 = a + b, c + d, a + c
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

    observed = probability(a, b, c, d)
    total = 0.0
    for aa in range(max(0, col1 - row2), min(row1, col1) + 1):
        bb, cc = row1 - aa, col1 - aa
        dd = row2 - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        mass = probability(aa, bb, cc, dd)
        if mass <= observed * (1.0 + 1e-9):
            total += mass
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney. The returned U belongs to `right`."""
    n1, n2 = len(left), len(right)
    pooled = [(value, 0) for value in left] + [(value, 1) for value in right]
    pooled.sort(key=lambda item: item[0])
    ranks = [0.0] * len(pooled)
    index = 0
    tie_term = 0.0
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
    u_right = n1 * n2 + n1 * (n1 + 1) / 2.0 - rank_left
    mean = n1 * n2 / 2.0
    total_n = n1 + n2
    variance = n1 * n2 / 12.0 * ((total_n + 1) - tie_term / (total_n * (total_n - 1)))
    if variance <= 0.0:
        return u_right, 1.0
    z = (abs(u_right - mean) - 0.5) / math.sqrt(variance)
    return u_right, min(1.0, math.erfc(z / math.sqrt(2.0)))


def ranked_calls(rows: list[dict[str, str]]) -> list[float]:
    ranked = []
    for row in rows:
        if row["hit"] == "1":
            ranked.append(float(row["calls"]))
        else:
            ranked.append(math.inf)
    return ranked


def self_check() -> None:
    if abs(fisher_two_sided(12, 12, 12, 12) - 1.0) > 1e-12:
        raise SystemExit("fisher self-check failed on a tie")
    if fisher_two_sided(12, 12, 0, 12) > 1e-6:
        raise SystemExit("fisher self-check failed on a separation")
    _, tied = mann_whitney([1.0, 2.0, 3.0, 4.0] * 3, [1.0, 2.0, 3.0, 4.0] * 3)
    if tied < 0.99:
        raise SystemExit("mann-whitney self-check failed on a tie")
    u_right, separated = mann_whitney([1.0] * 12, [100.0] * 12)
    if separated >= 0.05 or u_right <= 72.0:
        raise SystemExit("mann-whitney self-check failed on a shift")
    if type7([1.0, 2.0, 3.0, 4.0], 0.5) != 2.5:
        raise SystemExit("type 7 self-check failed")


def machine_id() -> str:
    path = Path("/etc/machine-id")
    if not path.is_file():
        path = Path("/proc/sys/kernel/random/boot_id")
    digest = hashlib.sha256(path.read_bytes().strip()).hexdigest()
    return digest


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def blob(repo: Path, path: str) -> str:
    return git(repo, "rev-parse", f"HEAD:{path}")


def affinity_is_core_0() -> bool:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip() == "0"
    return False


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("H2H_"):
            del env[name]
    for name in THREAD_VARS:
        env[name] = "1"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["LJOS_SEAT"] = "benchbot"
    return env


def run_driver(binary: Path, args: list[str], log_path: Path, env: dict[str, str]) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        done = subprocess.run(
            ["taskset", "-c", "0", "timeout", "--foreground", "-k", "120", str(JOB_TIMEOUT_S), str(binary), *args],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    text = log_path.read_text(encoding="utf-8")
    if done.returncode != 0:
        raise SystemExit(f"{binary} {' '.join(args)} exited {done.returncode}; log {log_path}")
    return text


def parse_nb8(text: str) -> dict[str, str]:
    found = [line for line in text.splitlines() if line.startswith("nb8 ")]
    if len(found) != 1:
        raise SystemExit(f"expected one nb8 line, found {len(found)}")
    row: dict[str, str] = {}
    for token in found[0].split()[1:]:
        key, value = token.split("=", 1)
        row[key] = value
    return row


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow({key: row[key] for key in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def schedule() -> list[tuple[str, int, int, str]]:
    jobs = []
    for system, _budget, _reference in SYSTEMS:
        for seed in SEEDS:
            arms = ("A", "B") if (seed - SEEDS[0]) % 2 == 0 else ("B", "A")
            for slot, arm in enumerate(arms):
                jobs.append((system, seed, slot, arm))
    counts: dict[tuple[int, str], int] = {}
    for _system, _seed, slot, arm in jobs:
        counts[(slot, arm)] = counts.get((slot, arm), 0) + 1
    expect = len(SYSTEMS) * len(SEEDS) // 2
    if counts != {(0, "A"): expect, (1, "B"): expect, (0, "B"): expect, (1, "A"): expect}:
        raise SystemExit(f"ABBA schedule is not balanced: {counts}")
    return jobs


def have(rows: list[dict[str, str]], system: str, seed: int, arm: str) -> bool:
    return any(row["system"] == system and row["seed"] == str(seed) and row["arm"] == arm for row in rows)


def reference_of(system: str) -> tuple[int, float]:
    for name, budget, reference in SYSTEMS:
        if name == system:
            return budget, reference
    raise SystemExit(f"unknown system {system}")


def check_spec(parsed: dict[str, str], system: str) -> None:
    budget, reference = reference_of(system)
    if int(parsed["budget"]) != budget:
        raise SystemExit(f"{system}: binary budget {parsed['budget']} is not {budget}")
    got = bits_to_float(int(parsed["reference_bits"]))
    if got != reference and abs(got - reference) > 0.0:
        # Decimal literals must be the same value the binary compiled.
        if abs(got - reference) > 1e-12:
            raise SystemExit(f"{system}: reference {got!r} is not {reference!r}")


def run(args: argparse.Namespace) -> None:
    self_check()
    if not affinity_is_core_0():
        raise SystemExit("this process is not pinned to core 0")
    repo = Path(args.repo)
    tree_a = Path(args.tree_a)
    tree_b = Path(args.tree_b)
    log_dir = Path(args.log_dir).resolve()
    out = Path(args.out)
    if log_dir == Path(git(repo, "rev-parse", "--show-toplevel")).resolve() or str(log_dir).startswith(
        str(Path(git(repo, "rev-parse", "--show-toplevel")).resolve()) + os.sep
    ):
        raise SystemExit("logs must stay outside the repository")
    if git(tree_a, "rev-parse", "HEAD") != ARM_A or git(tree_a, "rev-parse", "HEAD^{tree}") != TREE_A:
        raise SystemExit("arm A is not the pinned develop commit")
    if git(tree_b, "rev-parse", "HEAD^{tree}") != TREE_B:
        raise SystemExit("arm B tree is not the pinned kit tree")
    harness_blob = blob(repo, "bench/no-builders8-gate-c/run_gate.py")
    driver_blob = blob(repo, "bench/no-builders8-gate-c/driver.rs")
    for name in (harness_blob, driver_blob, ARM_A, TREE_A, TREE_B):
        subprocess.run(["git", "-C", str(repo), "cat-file", "-e", name], check=True)
    ident = {
        "A": (git(tree_a, "rev-parse", "HEAD"), git(tree_a, "rev-parse", "HEAD^{tree}")),
        "B": (git(tree_b, "rev-parse", "HEAD"), git(tree_b, "rev-parse", "HEAD^{tree}")),
    }
    for commit, tree in ident.values():
        subprocess.run(["git", "-C", str(repo), "cat-file", "-e", commit], check=True)
        subprocess.run(["git", "-C", str(repo), "cat-file", "-e", tree], check=True)
    env = child_env()
    bins = {"A": Path(args.bin_a), "B": Path(args.bin_b)}
    for system, _budget, _reference in SYSTEMS:
        for arm in ("A", "B"):
            text = run_driver(bins[arm], ["spec", system], log_dir / f"spec-{system}-{arm}.log", env)
            check_spec(parse_nb8(text), system)
    starts = log_dir / "starts"
    starts.mkdir(parents=True, exist_ok=True)
    for system, _budget, _reference in SYSTEMS:
        for seed in SEEDS:
            path = starts / f"{system}-{seed}.f64bits"
            text = run_driver(
                bins["A"],
                ["start", system, str(seed), str(path)],
                log_dir / f"start-{system}-{seed}-A.log",
                env,
            )
            parse_nb8(text)
            other = starts / f"{system}-{seed}-B.f64bits"
            text = run_driver(
                bins["B"],
                ["start", system, str(seed), str(other)],
                log_dir / f"start-{system}-{seed}-B.log",
                env,
            )
            parse_nb8(text)
            if path.read_bytes() != other.read_bytes():
                (log_dir / "start-mismatch.txt").open("a").write(f"{system} {seed}\n")
            other.unlink()
    rows = load_rows(out)
    host_hash = machine_id()
    jobs = schedule()
    for system, seed, slot, arm in jobs:
        if have(rows, system, seed, arm):
            continue
        start = starts / f"{system}-{seed}.f64bits"
        log_path = log_dir / f"run-{system}-s{seed}-{arm}.log"
        text = run_driver(bins[arm], ["run", system, str(seed), str(start)], log_path, env)
        parsed = parse_nb8(text)
        budget, reference = reference_of(system)
        if parsed["system"] != system or int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
            raise SystemExit(f"row identity mismatch for {system} {arm} {seed}")
        best = bits_to_float(int(parsed["best_bits"]))
        hit = best == best and math.isfinite(best) and best <= reference + HIT_TOL
        if hit != (parsed["hit"] == "1"):
            raise SystemExit(f"hit flag mismatch for {system} {arm} {seed}")
        commit, tree = ident[arm]
        row = {
            "system": system,
            "arm": arm,
            "seed": str(seed),
            "slot": str(slot),
            "core": "0",
            "hit": "1" if hit else "0",
            "calls": parsed["calls"],
            "charged": parsed["charged"],
            "wall_s": parsed["wall_s"],
            "pot_s": parsed["pot_s"],
            "outside_s": parsed["outside_s"],
            "best": full_precision(best),
            "best_bits": parsed["best_bits"],
            "budget": parsed["budget"],
            "machine_id": host_hash,
            "commit": commit,
            "tree": tree,
            "harness_blob": harness_blob,
            "driver_blob": driver_blob,
        }
        if float(row["best"]) != best:
            raise SystemExit(f"best text does not round-trip for {system} {arm} {seed}")
        append_row(out, row)
        rows.append(row)
        print(
            f"{system} {arm} slot {slot} seed {seed} hit {row['hit']} calls {row['calls']} wall {row['wall_s']}",
            flush=True,
        )
    print(f"rows {len(rows)}", flush=True)


def outside_per_call(rows: list[dict[str, str]]) -> float:
    outside = sum(float(row["outside_s"]) for row in rows)
    charged = sum(int(row["charged"]) for row in rows)
    if charged <= 0:
        return math.inf if outside > 0.0 else 0.0
    return outside / charged


def verdict_of(hits_a: int, hits_b: int, calls_p: float, u_b: float, n: int, ratio: float) -> str:
    fails = []
    if hits_b < hits_a:
        fails.append("hits")
    if calls_p < 0.05 and u_b > (n * n / 2.0):
        fails.append("calls")
    if ratio > 1.1:
        fails.append("outside")
    return ",".join(fails) if fails else "ok"


def summarize(args: argparse.Namespace) -> None:
    self_check()
    rows = load_rows(Path(args.rows))
    by_system: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_system.setdefault(row["system"], []).append(row)
    lines = [
        "# no-builders8 gate, part c",
        "",
        "Arm A is develop `61fa67cf380e46876b569da45e7cc091647a775b`.",
        "Arm B is that commit plus the no-builders8 kit, tree `39ac5e0642beb3e7874a340d2767aa151fa95eac`.",
        "Twelve seeds, 2000-2011, one core, paper budgets, default search, no budget override.",
        "A hit is a finite best at or below the reference plus 1e-3.",
        "Calls use the two-sided Mann-Whitney test with the tie correction and the continuity correction.",
        "A miss ranks worse than every hit. Wall medians are Hyndman-Fan type 7.",
        "Outside time per call is the total time outside the potential divided by the total charged calls.",
        "",
        "| system | hits A | hits B | fisher p | calls p | wall A | wall B | outside/call A | outside/call B | B/A | cell |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    failing = []
    seed_lines = ["", "## Per-seed hits", ""]
    for system, budget, reference in SYSTEMS:
        group = by_system.get(system, [])
        arm_a = [row for row in group if row["arm"] == "A"]
        arm_b = [row for row in group if row["arm"] == "B"]
        if len(arm_a) != 12 or len(arm_b) != 12:
            raise SystemExit(f"{system}: expected 12 rows on each arm, found {len(arm_a)} and {len(arm_b)}")
        arm_a.sort(key=lambda row: int(row["seed"]))
        arm_b.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in arm_a] != list(SEEDS) or [row["seed"] for row in arm_a] != [row["seed"] for row in arm_b]:
            raise SystemExit(f"{system}: seed lists differ")
        hits_a = sum(row["hit"] == "1" for row in arm_a)
        hits_b = sum(row["hit"] == "1" for row in arm_b)
        fisher = fisher_two_sided(hits_a, 12, hits_b, 12)
        u_b, calls_p = mann_whitney(ranked_calls(arm_a), ranked_calls(arm_b))
        wall_a = type7([float(row["wall_s"]) for row in arm_a], 0.5)
        wall_b = type7([float(row["wall_s"]) for row in arm_b], 0.5)
        rate_a = outside_per_call(arm_a)
        rate_b = outside_per_call(arm_b)
        ratio = math.inf if rate_a == 0.0 and rate_b > 0.0 else (rate_b / rate_a if rate_a != 0.0 else 1.0)
        cell = verdict_of(hits_a, hits_b, calls_p, u_b, 12, ratio)
        if cell != "ok":
            failing.append(f"{system} ({cell})")
        lines.append(
            f"| {system} | {hits_a}/12 | {hits_b}/12 | {fisher:.6g} | {calls_p:.6g} | {wall_a:.6g} | {wall_b:.6g} | {rate_a:.6g} | {rate_b:.6g} | {ratio:.6g} | {cell} |"
        )
        seed_lines.append(f"### {system}")
        seed_lines.append("")
        seed_lines.append(f"Reference {reference:.10g}, budget {budget}.")
        seed_lines.append("")
        seed_lines.append("| seed | hit A | hit B | calls A | calls B | best A | best B |")
        seed_lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for left, right in zip(arm_a, arm_b):
            seed_lines.append(
                f"| {left['seed']} | {left['hit']} | {right['hit']} | {left['calls']} | {right['calls']} | {left['best']} | {right['best']} |"
            )
        seed_lines.append("")
    overall = "FIX" if failing else "PASS"
    named = ", ".join(failing) if failing else "none"
    lines.extend(
        [
            "",
            f"Verdict: {overall}",
            "",
            f"Failing cells: {named}",
            "",
        ]
    )
    lines.extend(seed_lines)
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    Path(args.readme).write_text(text, encoding="utf-8", newline="\n")
    print(overall)
    print(named)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--bin-a", required=True)
    runner.add_argument("--bin-b", required=True)
    runner.add_argument("--tree-a", required=True)
    runner.add_argument("--tree-b", required=True)
    runner.add_argument("--out", required=True)
    runner.add_argument("--log-dir", required=True)
    runner.add_argument("--repo", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--rows", required=True)
    summary.add_argument("--readme", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
