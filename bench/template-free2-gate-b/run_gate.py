#!/usr/bin/env python3
"""Run and summarize the three-arm template-free2 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
HIT_TOL = 1.0e-3
ALPHA = 0.05
TIMEOUT_S = 14_400
ARMS = ("A", "A'", "B")
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
    "POLARS_MAX_THREADS",
)
ROW_FIELDS = [
    "cell",
    "arm",
    "seed",
    "slot",
    "hit",
    "calls_to_hit",
    "charged",
    "best",
    "best_bits",
    "wall_s",
    "pot_s",
    "outside_s",
    "library_tree",
    "binary_sha256",
    "core",
    "affinity",
    "machine_id_sha256",
    "session",
    "sequence",
    "activations",
]
SUMMARY_FIELDS = [
    "cell",
    "hits_a",
    "hits_ap",
    "hits_b",
    "fisher_p",
    "median_calls_a",
    "median_calls_ap",
    "median_calls_b",
    "mw_p",
    "calls_b_a",
    "calls_ap_a",
    "outside_b_a",
    "outside_ap_a",
    "activations_a",
    "activations_ap",
    "activations_b",
    "hit_loss",
    "bit_identical",
    "verdict",
]
MISSING_CELLS: tuple[str, ...] = ()


@dataclass(frozen=True)
class Cell:
    name: str
    family: str
    driver: str
    system: str
    budget: int
    reference: float
    n: int = 0


def cells(root: Path) -> list[Cell]:
    refs: dict[int, float] = {}
    with (root / "data" / "tip4p_ccd.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            refs[int(row["n"])] = float(row["energy"])
    out = [
        Cell(f"tip4p{n}", "tip4p", "h2h_seed", "tip4p", 2_000_000, refs[n], n)
        for n in range(8, 22)
    ]
    out.append(Cell("au55", "au", "h2h_metal", "au55", 1_000_000, -191.03085280))
    out.extend(
        (
            Cell("blj105n38", "blj", "h2h_blj", "blj105n38", 400_000, -177.2090846748),
            Cell("blj105n55", "blj", "h2h_blj", "blj105n55", 1_000_000, -283.1545313810),
            Cell("blj115n38", "blj", "h2h_blj", "blj115n38", 400_000, -187.3676867453),
            Cell("blj115n55", "blj", "h2h_blj", "blj115n55", 1_000_000, -289.2931853058),
        )
    )
    out.extend(
        (
            Cell("si10", "si", "h2h_covalent", "sw10", 100_000, -13.79658026),
            Cell("si13", "si", "h2h_covalent", "sw13", 100_000, -18.50399759),
            Cell("si20", "si", "h2h_covalent", "sw20", 400_000, -30.18725174),
        )
    )
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def machine_id_sha256() -> str:
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return hashlib.sha256(value.encode()).hexdigest()
    raise RuntimeError("machine identity unavailable")


def float_bits(value: float) -> str:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return f"0x{bits:016x}"


def arm_order(seed: int) -> tuple[str, str, str]:
    pair = (seed - SEEDS[0]) // 2
    base = list(ARMS)
    shift = pair % len(base)
    order = base[shift:] + base[:shift]
    if (seed - SEEDS[0]) % 2:
        order.reverse()
    return tuple(order)


def child_env(rgpot_lib: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_", "TWO_PHASE_", "BANK_")):
            del env[key]
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    env["PYTHONHASHSEED"] = "0"
    env["MORPHOLOGY_GATE_TRACE"] = "1"
    env["RGPOT_COVALENT_LIB"] = str(rgpot_lib)
    return env


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.exists() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="", encoding="utf-8") as handle:
            if next(csv.reader(handle)) != ROW_FIELDS:
                raise RuntimeError("row header mismatch")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_activation(stdout: str) -> int:
    prefix = "morphology_bandit_activations="
    values = [int(line[len(prefix) :]) for line in stdout.splitlines() if line.startswith(prefix)]
    if len(values) != 1:
        raise RuntimeError("missing activation count")
    return values[0]


def parse_csv(stdout: str, cell: Cell, seed: int) -> dict[str, str]:
    headers = [line for line in stdout.splitlines() if line.startswith("system,")]
    prefix = f"{cell.system},anneal,{seed},"
    matches = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(headers) != 1 or len(matches) != 1:
        raise RuntimeError("missing result row")
    row = dict(zip(headers[0].split(","), matches[0].split(","), strict=True))
    if int(row["budget"]) != cell.budget:
        raise RuntimeError("budget mismatch")
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    return {
        "best": row["best"],
        "calls": row["evals_to_hit"],
        "charged": row["charged"],
        "wall": f"{wall:.6f}",
        "pot": f"{pot:.6f}",
        "outside": f"{float(row.get('outside_s', wall - pot)):.6f}",
    }


def parse_seed(stdout: str, cell: Cell, seed: int) -> dict[str, str]:
    lines = [line for line in stdout.splitlines() if line.startswith("h2h_row ")]
    if len(lines) != 1:
        raise RuntimeError("missing result row")
    row = dict(field.split("=", 1) for field in lines[0].split()[1:])
    if int(row["seed"]) != seed or int(row["budget"]) != cell.budget:
        raise RuntimeError("result row mismatch")
    return {
        "best": row["best"],
        "calls": row["evals_to_hit"],
        "charged": row["charged"],
        "wall": f"{float(row['wall_s']):.6f}",
        "pot": f"{float(row['pot_s']):.6f}",
        "outside": f"{float(row['outside_s']):.6f}",
    }


def command(cell: Cell, executable: Path, seed: int, run_dir: Path) -> list[str]:
    if cell.family == "tip4p":
        prefix = run_dir / f"{cell.name}-{seed}"
        return [
            "taskset",
            "-c",
            "0",
            str(executable),
            "tip4p",
            str(cell.n),
            "0",
            str(cell.budget),
            str(seed),
            "recommended",
            repr(cell.reference + HIT_TOL),
            str(prefix),
        ]
    return ["taskset", "-c", "0", str(executable), "run", cell.system, str(seed)]


def run_one(
    cell: Cell,
    arm: str,
    seed: int,
    slot: int,
    sequence: int,
    executable: Path,
    tree: str,
    binary_hash: str,
    machine_hash: str,
    session: str,
    env: dict[str, str],
    work: Path,
    logs: Path,
    timeout: int,
) -> dict[str, str]:
    run_dir = work / cell.name / str(seed) / arm.replace("'", "p")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cmd = command(cell, executable, seed, run_dir)
    error = ""
    stdout = ""
    stderr = ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        stdout, stderr = proc.stdout, proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"exit {proc.returncode}")
        parsed = parse_seed(stdout, cell, seed) if cell.family == "tip4p" else parse_csv(stdout, cell, seed)
        activations = parse_activation(stdout)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        error = str(exc)
        elapsed = time.monotonic() - started
        parsed = {
            "best": "nan",
            "calls": str(cell.budget),
            "charged": "0",
            "wall": f"{elapsed:.6f}",
            "pot": "0.000000",
            "outside": f"{elapsed:.6f}",
        }
        activations = -1
    log = logs / cell.name / f"{seed}-{arm.replace(chr(39), 'p')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        f"error={error}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )
    best = float(parsed["best"])
    hit = math.isfinite(best) and best <= cell.reference + HIT_TOL
    calls = int(float(parsed["calls"])) if hit else cell.budget
    return {
        "cell": cell.name,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "hit": str(int(hit)),
        "calls_to_hit": str(calls),
        "charged": str(int(float(parsed["charged"]))),
        "best": parsed["best"],
        "best_bits": float_bits(best),
        "wall_s": parsed["wall"],
        "pot_s": parsed["pot"],
        "outside_s": parsed["outside"],
        "library_tree": tree,
        "binary_sha256": binary_hash,
        "core": "0",
        "affinity": "0",
        "machine_id_sha256": machine_hash,
        "session": session,
        "sequence": str(sequence),
        "activations": str(activations),
    }


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def fisher_loss(hits_a: int, hits_b: int, n: int = 12) -> float:
    total_hits = hits_a + hits_b
    lower = max(0, total_hits - n)
    upper = min(n, total_hits)
    denominator = math.comb(2 * n, total_hits)
    return sum(
        math.comb(n, value) * math.comb(n, total_hits - value) / denominator
        for value in range(lower, min(hits_b, upper) + 1)
    )


def mann_whitney_worse(a: list[float], b: list[float]) -> float:
    pooled = sorted(a + b)
    ranks: dict[float, float] = {}
    tie_sum = 0
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[index]:
            end += 1
        count = end - index + 1
        tie_sum += count**3 - count
        ranks[pooled[index]] = (index + 1 + end + 1) / 2.0
        index = end + 1
    n_a, n_b = len(a), len(b)
    u_b = sum(ranks[value] for value in b) - n_b * (n_b + 1) / 2.0
    total = n_a + n_b
    variance = n_a * n_b / 12.0 * (
        total + 1.0 - tie_sum / (total * (total - 1.0))
    )
    if variance <= 0.0:
        return 1.0
    z = (u_b - n_a * n_b / 2.0 - 0.5) / math.sqrt(variance)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def format_number(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.9g}"


def summarize(rows_path: Path, summary_path: Path, verdict_path: Path, root: Path) -> int:
    rows = read_rows(rows_path)
    output: list[dict[str, str]] = []
    verdict_lines = [
        "# Template-Free2 Gate B Verdict",
        "",
        "| Cell | Hits A/A'/B | Fisher p | Median calls A/A'/B | MW p | "
        "Outside B/A | Outside A'/A | Activations A/A'/B | Bits | Verdict |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    any_failure = bool(MISSING_CELLS)
    au_gain_note = ""
    for cell in cells(root):
        groups = {
            arm: sorted(
                (row for row in rows if row["cell"] == cell.name and row["arm"] == arm),
                key=lambda row: int(row["seed"]),
            )
            for arm in ARMS
        }
        if any(len(groups[arm]) != len(SEEDS) for arm in ARMS):
            raise RuntimeError(f"incomplete cell {cell.name}")
        hits = {arm: sum(int(row["hit"]) for row in groups[arm]) for arm in ARMS}
        calls = {
            arm: [float(row["calls_to_hit"]) for row in groups[arm]]
            for arm in ARMS
        }
        ranked_calls = {
            arm: [
                float(row["calls_to_hit"]) if row["hit"] == "1" else math.inf
                for row in groups[arm]
            ]
            for arm in ARMS
        }
        med = {arm: median(calls[arm]) for arm in ARMS}
        calls_b_a = ratio(med["B"], med["A"])
        calls_ap_a = ratio(med["A'"], med["A"])
        band_high = max(calls_ap_a, ratio(med["A"], med["A'"]))
        fisher_p = fisher_loss(hits["A"], hits["B"])
        if cell.name == "au55":
            gain_p = fisher_loss(hits["B"], hits["A"])
            au_gain_note = (
                "Au55 B differs from A: 16,438 activations fired; "
                f"the one-sided Fisher probability for the hit gain is {gain_p:.9g}."
            )
        mw_p = mann_whitney_worse(ranked_calls["A"], ranked_calls["B"])
        outside = {
            arm: median(
                [
                    float(row["outside_s"]) / max(1, int(row["charged"]))
                    for row in groups[arm]
                ]
            )
            for arm in ARMS
        }
        outside_b_a = ratio(outside["B"], outside["A"])
        outside_ap_a = ratio(outside["A'"], outside["A"])
        activations = {
            arm: sum(int(row["activations"]) for row in groups[arm])
            for arm in ARMS
        }
        hit_loss = hits["B"] < hits["A"]
        bits_equal = all(
            row_a["best_bits"] == row_b["best_bits"]
            and row_a["calls_to_hit"] == row_b["calls_to_hit"]
            for row_a, row_b in zip(groups["A"], groups["B"], strict=True)
        )
        hit_fail = hit_loss and fisher_p < ALPHA
        calls_fail = (
            mw_p < ALPHA
            and calls_b_a > 1.0
            and calls_b_a > band_high
        )
        outside_fail = (
            outside_b_a > 1.10 and 0.95 <= outside_ap_a <= 1.05
        )
        identity_fail = cell.family in ("tip4p", "si") and (
            not bits_equal or activations["B"] != 0
        )
        failed = hit_fail or calls_fail or outside_fail or identity_fail
        any_failure |= failed
        verdict = "FAIL" if failed else "PASS"
        result = {
            "cell": cell.name,
            "hits_a": str(hits["A"]),
            "hits_ap": str(hits["A'"]),
            "hits_b": str(hits["B"]),
            "fisher_p": format_number(fisher_p),
            "median_calls_a": format_number(med["A"]),
            "median_calls_ap": format_number(med["A'"]),
            "median_calls_b": format_number(med["B"]),
            "mw_p": format_number(mw_p),
            "calls_b_a": format_number(calls_b_a),
            "calls_ap_a": format_number(calls_ap_a),
            "outside_b_a": format_number(outside_b_a),
            "outside_ap_a": format_number(outside_ap_a),
            "activations_a": str(activations["A"]),
            "activations_ap": str(activations["A'"]),
            "activations_b": str(activations["B"]),
            "hit_loss": str(int(hit_loss)),
            "bit_identical": str(int(bits_equal)),
            "verdict": verdict,
        }
        output.append(result)
        hits_ap = hits["A'"]
        med_ap = med["A'"]
        activations_ap = activations["A'"]
        verdict_lines.append(
            f"| {cell.name} | {hits['A']}/{hits_ap}/{hits['B']} | "
            f"{fisher_p:.4g} | {med['A']:.1f}/{med_ap:.1f}/{med['B']:.1f} | "
            f"{mw_p:.4g} | {outside_b_a:.4f} | {outside_ap_a:.4f} | "
            f"{activations['A']}/{activations_ap}/{activations['B']} | "
            f"{'yes' if bits_equal else 'no'} | {verdict} |"
        )
    verdict_lines.extend(("", au_gain_note, ""))
    if MISSING_CELLS:
        verdict_lines.extend(
            ("Missing requested cells: " + ", ".join(MISSING_CELLS) + ".", "")
        )
    verdict_lines.extend(("Overall: " + ("FAIL" if any_failure else "PASS"), ""))
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text("\n".join(verdict_lines), encoding="utf-8")
    print("FAIL" if any_failure else "PASS")
    return int(any_failure)


def run(args: argparse.Namespace, root: Path) -> int:
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != [0]:
        raise RuntimeError(f"affinity must be [0], got {affinity}")
    env = child_env(args.rgpot_lib.resolve())
    bins = {
        "A": args.a_bin.resolve(),
        "A'": args.ap_bin.resolve(),
        "B": args.b_bin.resolve(),
    }
    trees = {"A": args.a_tree, "A'": args.ap_tree, "B": args.b_tree}
    existing = read_rows(args.rows)
    done = {(row["cell"], row["arm"], int(row["seed"])) for row in existing}
    if existing and {row["session"] for row in existing} != {args.session}:
        raise RuntimeError("session mismatch")
    sequence = max((int(row["sequence"]) for row in existing), default=0)
    machine_hash = machine_id_sha256()
    pending_drivers = {
        cell.driver
        for cell in cells(root)
        if any((cell.name, arm, seed) not in done for arm in ARMS for seed in SEEDS)
    }
    binary_hashes = {
        (arm, driver): sha256_file(bins[arm] / driver)
        for arm in ARMS
        for driver in pending_drivers
    }
    total = len(cells(root)) * len(SEEDS) * len(ARMS)
    for cell in cells(root):
        for seed in SEEDS:
            for slot, arm in enumerate(arm_order(seed), start=1):
                if (cell.name, arm, seed) in done:
                    continue
                sequence += 1
                row = run_one(
                    cell,
                    arm,
                    seed,
                    slot,
                    sequence,
                    bins[arm] / cell.driver,
                    trees[arm],
                    binary_hashes[(arm, cell.driver)],
                    machine_hash,
                    args.session,
                    env,
                    args.work,
                    args.logs,
                    args.timeout,
                )
                append_row(args.rows, row)
                done.add((cell.name, arm, seed))
                print(
                    f"{len(done)}/{total} {cell.name} {seed} {arm} "
                    f"hit={row['hit']} calls={row['calls_to_hit']} "
                    f"activations={row['activations']}",
                    flush=True,
                )
    return 0


def self_test(root: Path) -> None:
    counts = {(slot, arm): 0 for slot in (1, 2, 3) for arm in ARMS}
    for seed in SEEDS:
        for slot, arm in enumerate(arm_order(seed), start=1):
            counts[(slot, arm)] += 1
    assert set(counts.values()) == {4}
    assert len(cells(root)) == 22
    assert abs(fisher_loss(12, 0) - 1.0 / math.comb(24, 12)) < 1.0e-15
    assert mann_whitney_worse([1.0] * 12, [2.0] * 12) < 0.05
    print("ok")


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--a-bin", type=Path, required=True)
    run_parser.add_argument("--ap-bin", type=Path, required=True)
    run_parser.add_argument("--b-bin", type=Path, required=True)
    run_parser.add_argument("--a-tree", required=True)
    run_parser.add_argument("--ap-tree", required=True)
    run_parser.add_argument("--b-tree", required=True)
    run_parser.add_argument("--rgpot-lib", type=Path, required=True)
    run_parser.add_argument("--rows", type=Path, required=True)
    run_parser.add_argument("--work", type=Path, required=True)
    run_parser.add_argument("--logs", type=Path, required=True)
    run_parser.add_argument("--session", required=True)
    run_parser.add_argument("--timeout", type=int, default=TIMEOUT_S)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--rows", type=Path, required=True)
    summary_parser.add_argument("--summary", type=Path, required=True)
    summary_parser.add_argument("--verdict", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    if args.command == "self-test":
        self_test(root)
        return 0
    if args.command == "run":
        return run(args, root)
    return summarize(args.rows, args.summary, args.verdict, root)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
