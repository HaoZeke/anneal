#!/usr/bin/env python3
"""Same-session A/B gate for the xtb force-sign fix.

Arm A is the base tree. Arm B is that tree plus the force-sign fix.
The process tree is pinned to one core and every threaded runtime is
pinned to one thread. Seeds and starts are shared. Records are appended
and never rewritten.

The gold guard is Sutton-Chen family 108 through the registered h2h_sc
example, which calls rgpot_sutton_chen_energy_forces. The cell-list EAM
gold type in that library has no C entry this example can call.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
ISOMER_DIR = INPUTS
REFERENCE_PRISM = INPUTS / "reference-prism.xyz"

CELLS = ("h2o20", "hexamer", "crest-equal", "lj", "morse", "au")
GUARDS = ("lj", "morse", "au")
AA_CELLS = ("hexamer", "morse")
XTB_CELLS = ("h2o20", "hexamer", "crest-equal")
ALPHA = 0.05 / len(CELLS)
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261119
PRISM_CUTOFF_A = 0.598496
H2O20_REFERENCE_EH = -101.749249454294
H2O20_HIT_TOL_EH = 1.0e-6
CREST_EQUAL_REFERENCE_EH = -30.496506792187
CREST_EQUAL_HIT_TOL_EH = 1.0e-7
EQUAL_ABS = 1.0e-12

SEEDS = {
    "h2o20": tuple(range(2000, 2012)),
    "hexamer": tuple(range(0, 12)),
    "crest-equal": tuple(range(2000, 2012)),
    "lj": tuple(range(0, 12)),
    "morse": tuple(range(0, 12)),
    "au": tuple(range(0, 12)),
}

FIELDS = [
    "cell",
    "seed",
    "pair",
    "slot",
    "arm",
    "replica",
    "hit",
    "energy",
    "wall_s",
    "harness_wall_s",
    "evals",
    "engine_eV",
    "grad_energy",
    "forces",
    "status",
    "note",
]

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "XTB_NUM_THREADS": "1",
    "LJOS_SEAT": "benchbot",
}


def thread_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update(THREAD_ENV)
    for key in (
        "H2H_START_ONLY",
        "H2H_START_XYZ",
        "H2H_BEST",
        "H2H_FINAL",
        "H2O_SURROGATE",
        "WATER_POLISH_EXTERNAL",
        "WATER_FORCE_BUDGET",
        "WATER_SEED_FIRST",
        "WATER_SEED_COUNT",
        "WATER_ISOMER_DIR",
        "WATER_START_XYZ",
        "H2H_CAPPED_LINDH",
        "H2H_BUDGET",
        "PAPER_SMOKE",
        "SEED_OFFSET",
        "ANNEAL_MIN_DUMP",
    ):
        env.pop(key, None)
    return env


def abba(index: int) -> list[tuple[int, str, str]]:
    """Return (slot, arm, replica) for one seed. Even indexes run A then B."""
    if index % 2 == 0:
        return [(0, "A", "0"), (1, "B", "0")]
    return [(0, "B", "0"), (1, "A", "0")]


def aa_order(index: int) -> list[tuple[int, str, str]]:
    if index % 2 == 0:
        return [(0, "A", "0"), (1, "A", "1")]
    return [(0, "A", "1"), (1, "A", "0")]


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Exact two-sided Fisher from the hypergeometric tables no more likely than observed."""
    hits = hits_a + hits_b
    total = n_a + n_b
    if hits < 0 or hits > total or n_a < 0 or n_b < 0:
        raise ValueError("fisher counts are out of range")
    ways = math.comb(total, hits)

    def prob(k: int) -> float:
        if k < 0 or k > n_a or hits - k < 0 or hits - k > n_b:
            return 0.0
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    total_p = sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7))
    return min(1.0, total_p)


def mann_whitney(left: list[float], right: list[float]) -> float:
    """Exact two-sided Wilcoxon rank-sum test.

    Every way of assigning the pooled sample to the two arms is equally likely.
    Ties use midranks. The p-value counts assignments whose rank sum is at
    least as far from the null mean as the observed sum.
    """
    if not left or not right:
        return float("nan")
    values = [float(v) for v in left] + [float(v) for v in right]
    n1 = len(left)
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        mid = (i + 1 + j + 1) / 2.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = mid
        i = j + 1
    weights = [int(round(rank * 2)) for rank in ranks]
    observed = sum(weights[:n1])
    mean = n1 * (n + 1)
    deviation = abs(observed - mean)
    ways = [Counter() for _ in range(n1 + 1)]
    ways[0][0] = 1
    for weight in weights:
        for taken in range(n1, 0, -1):
            previous = ways[taken - 1]
            current = ways[taken]
            for score, count in previous.items():
                current[score + weight] += count
    total = math.comb(n, n1)
    extreme = sum(count for score, count in ways[n1].items() if abs(score - mean) >= deviation)
    return extreme / total


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def bootstrap_median_ci(values: list[float], draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, sample.size, size=(draws, sample.size))
    stats = np.median(sample[picks], axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    if not new:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle))
        if header != FIELDS:
            raise SystemExit(f"{path.name} header does not match this harness")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def done_keys(rows: list[dict[str, str]]) -> set[tuple[str, ...]]:
    return {
        (row["cell"], row["seed"], row["pair"], row["slot"], row["arm"], row["replica"])
        for row in rows
        if row.get("status") == "ok"
    }


def key_of(row: dict[str, str]) -> tuple[str, ...]:
    return (row["cell"], row["seed"], row["pair"], row["slot"], row["arm"], row["replica"])


def blank(cell: str, seed: int, pair: str, slot: int, arm: str, replica: str) -> dict[str, str]:
    return {
        "cell": cell,
        "seed": str(seed),
        "pair": pair,
        "slot": str(slot),
        "arm": arm,
        "replica": replica,
        "hit": "",
        "energy": "",
        "wall_s": "",
        "harness_wall_s": "",
        "evals": "",
        "engine_eV": "",
        "grad_energy": "",
        "forces": "",
        "status": "error",
        "note": "",
    }


def run_process(cmd: list[str], env: dict[str, str], cwd: Path, stdout_path: Path, stderr_path: Path) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(
            ["taskset", "-c", "0", *cmd],
            env=env,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
        )
    return completed.returncode, time.perf_counter() - started


def read_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not path.is_file():
        return fields
    for line in path.read_text(errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def last_energy_eh(path: Path) -> str:
    found = ""
    for line in path.read_text(errors="replace").splitlines():
        if "TOTAL ENERGY" not in line:
            continue
        parts = line.replace("|", " ").split()
        if "Eh" in parts:
            found = parts[parts.index("Eh") - 1]
    if not found:
        raise RuntimeError("rescore produced no energy")
    return found


def rescore(xtb: Path, xyz: Path, out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    (out / "in.xyz").write_bytes(xyz.read_bytes())
    stdout_path = out / "opt.out"
    stderr_path = out / "opt.err"
    code, _ = run_process(
        [str(xtb), "in.xyz", "--gfn", "2", "--opt", "vtight", "--acc", "0.01", "--iterations", "500"],
        thread_env(),
        out,
        stdout_path,
        stderr_path,
    )
    if code != 0:
        raise RuntimeError(f"rescore exit {code}")
    return last_energy_eh(stdout_path)


def finite(text: str) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    return math.isfinite(value)


def hit_below(energy: str, reference: float, tol: float) -> str:
    if not finite(energy):
        return "0"
    return "1" if float(energy) <= reference + tol else "0"


def example_dir(arm: str) -> Path:
    name = "ANNEAL_A_BIN" if arm == "A" else "ANNEAL_B_BIN"
    raw = os.environ.get(name, "")
    if not raw:
        raise SystemExit(f"{name} is required")
    path = Path(raw)
    if not path.is_dir():
        raise SystemExit(f"{name} is not a directory")
    return path


def binary(arm: str, name: str) -> str:
    path = example_dir(arm) / name
    if not path.is_file():
        raise SystemExit(f"missing example {name} for arm {arm}")
    return str(path)


def xtb_bin() -> Path:
    raw = os.environ.get("XTB_BIN", "")
    if not raw:
        raise SystemExit("XTB_BIN is required")
    path = Path(raw)
    if not path.is_file():
        raise SystemExit("XTB_BIN is not a file")
    return path


def require_runtime() -> None:
    for key in ("RGPOT_XTB_ENGINE", "RGPOT_SC_LIB"):
        raw = os.environ.get(key, "")
        if not raw or not Path(raw).is_file():
            raise SystemExit(f"{key} must name the engine file")
    if not REFERENCE_PRISM.is_file():
        raise SystemExit("reference prism geometry is missing")
    for name in ("prism.xyz", "cage.xyz", "book.xyz"):
        if not (ISOMER_DIR / name).is_file():
            raise SystemExit(f"isomer file {name} is missing")


def log_paths(work: Path, cell: str, pair: str, seed: int, slot: int, arm: str, replica: str) -> tuple[Path, Path, Path]:
    stem = work / "logs" / cell / pair / str(seed) / f"s{slot}-{arm}-r{replica}"
    return stem.parent, stem.with_suffix(".out"), stem.with_suffix(".err")


def generate_start(arm_bin: str, molecules: int, seed: int, dest: Path) -> None:
    if (dest / "start.xyz").is_file():
        return
    dest.mkdir(parents=True, exist_ok=True)
    env = thread_env()
    env["H2H_START_ONLY"] = "1"
    code, _ = run_process(
        [arm_bin, str(molecules), "1", str(seed), str(dest)],
        env,
        dest,
        dest / "start.out",
        dest / "start.err",
    )
    if code != 0 or not (dest / "start.xyz").is_file():
        raise RuntimeError(f"start generation failed for seed {seed}")


def run_h2h(
    cell: str,
    molecules: int,
    budget: int,
    seed: int,
    arm: str,
    work: Path,
    pair: str,
    slot: int,
    replica: str,
    start: Path,
    score_reference: float | None,
    score_tol: float | None,
) -> dict[str, str]:
    row = blank(cell, seed, pair, slot, arm, replica)
    out = work / "runs" / cell / pair / str(seed) / f"s{slot}-{arm}-r{replica}"
    out.mkdir(parents=True, exist_ok=True)
    _, stdout_path, stderr_path = log_paths(work, cell, pair, seed, slot, arm, replica)
    env = thread_env()
    env["H2H_START_XYZ"] = str(start)
    code, harness_wall = run_process(
        [binary(arm, "gfn2_water_h2h"), str(molecules), str(budget), str(seed), str(out)],
        env,
        out,
        stdout_path,
        stderr_path,
    )
    row["harness_wall_s"] = f"{harness_wall:.6f}"
    fields = read_fields(out / "run.txt")
    if code != 0 or "wall_s" not in fields or "energy_calls" not in fields:
        row["note"] = "search-failed"
        return row
    row["wall_s"] = fields["wall_s"]
    row["evals"] = fields["energy_calls"]
    row["engine_eV"] = fields.get("engine_best_eV", "")
    if score_reference is None or score_tol is None:
        row["status"] = "ok"
        row["note"] = "ok"
        return row
    best = out / "best.xyz"
    if not best.is_file():
        row["note"] = "no-best-geometry"
        return row
    try:
        energy = rescore(xtb_bin(), best, out / "score")
    except RuntimeError:
        row["energy"] = ""
        row["hit"] = "0"
        row["status"] = "ok"
        row["note"] = "rescore-failed"
        return row
    row["energy"] = energy
    row["hit"] = hit_below(energy, score_reference, score_tol)
    row["status"] = "ok"
    row["note"] = "search-error" if "search_error" in fields else "ok"
    return row


def run_hexamer(seed: int, arm: str, work: Path, pair: str, slot: int, replica: str) -> dict[str, str]:
    row = blank("hexamer", seed, pair, slot, arm, replica)
    _, stdout_path, stderr_path = log_paths(work, "hexamer", pair, seed, slot, arm, replica)
    env = thread_env()
    env["WATER_FORCE_BUDGET"] = "4000"
    env["WATER_SEED_FIRST"] = str(seed)
    env["WATER_SEED_COUNT"] = "1"
    env["WATER_ISOMER_DIR"] = str(ISOMER_DIR)
    code, harness_wall = run_process(
        [binary(arm, "water_hexamer"), "gfn2", "recommended"],
        env,
        work,
        stdout_path,
        stderr_path,
    )
    row["harness_wall_s"] = f"{harness_wall:.6f}"
    text = stdout_path.read_text(errors="replace") if stdout_path.is_file() else ""
    fields: dict[str, str] = {}
    started = False
    for line in text.splitlines():
        if line.strip() == f"seed: {seed}":
            fields = {}
            started = True
            continue
        if not started or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    if code != 0 or "force_calls" not in fields or "wall_seconds" not in fields:
        row["note"] = "search-failed"
        return row
    row["wall_s"] = fields["wall_seconds"]
    row["evals"] = fields["force_calls"]
    row["engine_eV"] = fields.get("best_energy", "")
    row["energy"] = fields.get("best_energy", "")
    rmsd = fields.get("prism_rmsd_A", "none")
    intact = fields.get("best_monomers_intact") == "true"
    row["hit"] = "1" if intact and finite(rmsd) and float(rmsd) <= PRISM_CUTOFF_A else "0"
    row["grad_energy"] = fields.get("isomer_gate_A", "")
    row["status"] = "ok"
    row["note"] = "ok"
    return row


def run_morse(seed: int, arm: str, work: Path, pair: str, slot: int, replica: str) -> dict[str, str]:
    row = blank("morse", seed, pair, slot, arm, replica)
    _, stdout_path, stderr_path = log_paths(work, "morse", pair, seed, slot, arm, replica)
    code, harness_wall = run_process(
        [binary(arm, "h2h_morse"), "run", "m38r6", str(seed)],
        thread_env(),
        work,
        stdout_path,
        stderr_path,
    )
    row["harness_wall_s"] = f"{harness_wall:.6f}"
    parsed = parse_csv_result(stdout_path)
    if code != 0 or parsed is None:
        row["note"] = "search-failed"
        return row
    row.update(parsed)
    row["status"] = "ok"
    row["note"] = "ok"
    return row


def run_lj(seed: int, arm: str, work: Path, pair: str, slot: int, replica: str) -> dict[str, str]:
    row = blank("lj", seed, pair, slot, arm, replica)
    _, stdout_path, stderr_path = log_paths(work, "lj", pair, seed, slot, arm, replica)
    env = thread_env()
    env["SEED_OFFSET"] = str(seed)
    code, harness_wall = run_process(
        [binary(arm, "lj_cluster_search"), "38", "400000", "1", "dls"],
        env,
        work,
        stdout_path,
        stderr_path,
    )
    row["harness_wall_s"] = f"{harness_wall:.6f}"
    row["wall_s"] = f"{harness_wall:.6f}"
    text = stdout_path.read_text(errors="replace") if stdout_path.is_file() else ""
    matched = None
    for line in text.splitlines():
        found = re.search(
            rf"seed {seed}: best ([-+]?\d+(?:\.\d+)?) .*charged (\d+)",
            line,
        )
        if found:
            matched = (found.group(1), found.group(2), "SOLVED" in line)
    if code != 0 or matched is None:
        row["note"] = "search-failed"
        return row
    energy, charged, solved = matched
    row["energy"] = energy
    row["evals"] = charged
    row["hit"] = "1" if solved else "0"
    row["status"] = "ok"
    row["note"] = "ok"
    return row


def run_au(seed: int, arm: str, work: Path, pair: str, slot: int, replica: str) -> dict[str, str]:
    row = blank("au", seed, pair, slot, arm, replica)
    out = work / "runs" / "au" / pair / str(seed) / f"s{slot}-{arm}-r{replica}"
    out.mkdir(parents=True, exist_ok=True)
    best = out / "best.xyz"
    _, stdout_path, stderr_path = log_paths(work, "au", pair, seed, slot, arm, replica)
    env = thread_env()
    env["H2H_BEST"] = str(best)
    env["RGPOT_SC_LIB"] = os.environ["RGPOT_SC_LIB"]
    env["RGPOT_LIBRARY"] = os.environ["RGPOT_SC_LIB"]
    code, harness_wall = run_process(
        [binary(arm, "h2h_sc"), "run", "sc108-38", str(seed)],
        env,
        out,
        stdout_path,
        stderr_path,
    )
    row["harness_wall_s"] = f"{harness_wall:.6f}"
    parsed = parse_csv_result(stdout_path)
    if code != 0 or parsed is None or not best.is_file():
        row["note"] = "search-failed"
        return row
    row.update(parsed)
    grad_out = stdout_path.with_name(stdout_path.stem + "-grad.out")
    grad_err = stderr_path.with_name(stderr_path.stem + "-grad.err")
    grad_code, _ = run_process(
        [binary(arm, "h2h_sc"), "grad", "sc108-38", str(best)],
        env,
        out,
        grad_out,
        grad_err,
    )
    if grad_code != 0:
        row["note"] = "grad-failed"
        row["status"] = "error"
        return row
    lines = [line.strip() for line in grad_out.read_text(errors="replace").splitlines() if line.strip()]
    if len(lines) < 2:
        row["note"] = "grad-failed"
        return row
    row["grad_energy"] = lines[0]
    components: list[str] = []
    for line in lines[1:]:
        components.extend(line.split())
    row["forces"] = " ".join(components)
    row["status"] = "ok"
    row["note"] = "ok"
    return row


def parse_csv_result(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split(",")
        if len(parts) < 10 or parts[1] != "anneal":
            continue
        hit = "1" if parts[4] == "true" else "0"
        return {
            "hit": hit,
            "evals": parts[6],
            "wall_s": parts[7],
            "energy": parts[8],
        }
    return None


def jobs_for(cell: str) -> list[tuple[str, int, int, str, str]]:
    planned: list[tuple[str, int, int, str, str]] = []
    for index, seed in enumerate(SEEDS[cell]):
        for slot, arm, replica in abba(index):
            planned.append(("ab", seed, slot, arm, replica))
        if cell in AA_CELLS:
            for slot, arm, replica in aa_order(index):
                planned.append(("aa", seed, slot, arm, replica))
    return planned


def dispatch(cell: str, seed: int, arm: str, work: Path, pair: str, slot: int, replica: str) -> dict[str, str]:
    if cell == "h2o20":
        start_dir = work / "starts" / "h2o20" / str(seed)
        generate_start(binary("A", "gfn2_water_h2h"), 20, seed, start_dir)
        return run_h2h(
            "h2o20",
            20,
            28000,
            seed,
            arm,
            work,
            pair,
            slot,
            replica,
            start_dir / "start.xyz",
            H2O20_REFERENCE_EH,
            H2O20_HIT_TOL_EH,
        )
    if cell == "crest-equal":
        start_dir = work / "starts" / "crest-equal" / str(seed)
        generate_start(binary("A", "gfn2_water_h2h"), 6, seed, start_dir)
        return run_h2h(
            "crest-equal",
            6,
            4000,
            seed,
            arm,
            work,
            pair,
            slot,
            replica,
            start_dir / "start.xyz",
            CREST_EQUAL_REFERENCE_EH,
            CREST_EQUAL_HIT_TOL_EH,
        )
    if cell == "hexamer":
        return run_hexamer(seed, arm, work, pair, slot, replica)
    if cell == "morse":
        return run_morse(seed, arm, work, pair, slot, replica)
    if cell == "lj":
        return run_lj(seed, arm, work, pair, slot, replica)
    if cell == "au":
        return run_au(seed, arm, work, pair, slot, replica)
    raise SystemExit(f"unknown cell {cell}")


def write_meta(work: Path) -> None:
    lines = [
        f"arm_a_sha {os.environ.get('ANNEAL_A_SHA', 'unset')}",
        f"arm_a_tree {os.environ.get('ANNEAL_A_TREE', 'unset')}",
        f"arm_b_sha {os.environ.get('ANNEAL_B_SHA', 'unset')}",
        f"arm_b_tree {os.environ.get('ANNEAL_B_TREE', 'unset')}",
        f"prism_cutoff_A {PRISM_CUTOFF_A}",
        f"h2o20_reference_Eh {H2O20_REFERENCE_EH:.12f}",
        f"crest_equal_reference_Eh {CREST_EQUAL_REFERENCE_EH:.12f}",
        f"bootstrap_draws {BOOTSTRAP_DRAWS}",
        f"bootstrap_seed {BOOTSTRAP_SEED}",
        f"bonferroni_alpha {ALPHA:.8f}",
    ]
    path = work / "meta.txt"
    path.write_text("\n".join(lines) + "\n")


def cmd_run(args: argparse.Namespace) -> int:
    require_runtime()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    write_meta(work)
    rows_path = work / "rows.csv"
    selected = tuple(part.strip() for part in args.cells.split(",") if part.strip())
    unknown = [cell for cell in selected if cell not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells: {', '.join(unknown)}")
    for cell in selected:
        for pair, seed, slot, arm, replica in jobs_for(cell):
            rows = load_rows(rows_path)
            if (cell, str(seed), pair, str(slot), arm, replica) in done_keys(rows):
                continue
            row = dispatch(cell, seed, arm, work, pair, slot, replica)
            append_row(rows_path, row)
            print(
                f"{cell} seed {seed} pair {pair} slot {slot} arm {arm} {row['status']} {row['note']}",
                flush=True,
            )
    rows = load_rows(rows_path)
    missing = 0
    for cell in selected:
        finished = done_keys(rows)
        for pair, seed, slot, arm, replica in jobs_for(cell):
            if (cell, str(seed), pair, str(slot), arm, replica) not in finished:
                missing += 1
    return 1 if missing else 0


def ok_rows(rows: list[dict[str, str]], cell: str, pair: str) -> list[dict[str, str]]:
    return [row for row in rows if row["cell"] == cell and row["pair"] == pair and row["status"] == "ok"]


def arm_rows(rows: list[dict[str, str]], arm: str) -> list[dict[str, str]]:
    return [row for row in rows if row["arm"] == arm]


def column(rows: list[dict[str, str]], name: str) -> list[float]:
    values = []
    for row in rows:
        text = row.get(name, "")
        if finite(text):
            values.append(float(text))
    return values


def hits_of(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if row.get("hit") == "1")


def fmt(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}g}"


def pair_by_seed(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    found: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        found.setdefault(row["seed"], {}).setdefault(row["arm"], row)
    return found


def energy_delta(left: str, right: str) -> float | None:
    if left == right:
        return 0.0
    if not finite(left) or not finite(right):
        return None
    return abs(float(left) - float(right))


def force_delta(left: str, right: str) -> float | None:
    if left == right:
        return 0.0
    a = left.split()
    b = right.split()
    if not a or len(a) != len(b):
        return None
    worst = 0.0
    for x, y in zip(a, b):
        if not finite(x) or not finite(y):
            return None
        worst = max(worst, abs(float(x) - float(y)))
    return worst


def aa_band(rows: list[dict[str, str]], name: str) -> tuple[float, float, float, float]:
    replica0 = column([row for row in rows if row["replica"] == "0"], name)
    replica1 = column([row for row in rows if row["replica"] == "1"], name)
    if not replica0 or not replica1:
        return float("nan"), float("nan"), float("nan"), float("nan")
    med0 = median(replica0)
    med1 = median(replica1)
    low, high = min(med0, med1), max(med0, med1)
    by_seed: dict[str, dict[str, float]] = {}
    for row in rows:
        if finite(row.get(name, "")):
            by_seed.setdefault(row["seed"], {})[row["replica"]] = float(row[name])
    gaps = [
        abs(item["0"] - item["1"])
        for item in by_seed.values()
        if "0" in item and "1" in item
    ]
    if high == low and gaps:
        pad = median(gaps)
        low -= pad
        high += pad
    return med0, med1, low, high


def relative_band(low: float, high: float, local: float) -> tuple[float, float]:
    if not math.isfinite(low) or not math.isfinite(high) or not math.isfinite(local):
        return float("nan"), float("nan")
    mid = (low + high) / 2.0
    if mid == 0.0:
        return local + (low - mid), local + (high - mid)
    return local * (low / mid), local * (high / mid)


def worse(left_median: float, right_median: float, p_value: float) -> bool:
    """True when B (right) is larger and the two-sided exact p is under the cell alpha."""
    return math.isfinite(p_value) and p_value < ALPHA and right_median > left_median


def outside(value: float, low: float, high: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(low) or not math.isfinite(high):
        return True
    return value < low or value > high


def cell_report(cell: str, rows: list[dict[str, str]], bands: dict[str, tuple[float, float]]) -> dict[str, str]:
    ab = ok_rows(rows, cell, "ab")
    a_rows = arm_rows(ab, "A")
    b_rows = arm_rows(ab, "B")
    expected = len(SEEDS[cell])
    result = {
        "cell": cell,
        "n_a": str(len(a_rows)),
        "n_b": str(len(b_rows)),
        "hits_a": str(hits_of(a_rows)),
        "hits_b": str(hits_of(b_rows)),
        "fisher_p": "nan",
        "energy_a": "nan",
        "energy_a_low": "nan",
        "energy_a_high": "nan",
        "energy_b": "nan",
        "energy_b_low": "nan",
        "energy_b_high": "nan",
        "energy_p": "nan",
        "wall_a": "nan",
        "wall_b": "nan",
        "wall_p": "nan",
        "evals_a": "nan",
        "evals_b": "nan",
        "evals_p": "nan",
        "band_low": "nan",
        "band_high": "nan",
        "guard_energy": "",
        "guard_force": "",
        "guard_evals": "",
        "verdict": "FAIL",
        "reason": "",
    }
    reasons: list[str] = []
    if len(a_rows) != expected or len(b_rows) != expected:
        reasons.append("incomplete")
    else:
        result["fisher_p"] = fmt(fisher(hits_of(a_rows), expected, hits_of(b_rows), expected), 4)
        if hits_of(b_rows) < hits_of(a_rows) and float(result["fisher_p"]) < ALPHA:
            reasons.append("hits")
        energy_a = column(a_rows, "energy")
        energy_b = column(b_rows, "energy")
        # A non-finite recorded energy is worse than every finite energy.
        energy_a_rank = energy_a + [math.inf] * (expected - len(energy_a))
        energy_b_rank = energy_b + [math.inf] * (expected - len(energy_b))
        med_a = median(energy_a_rank)
        med_b = median(energy_b_rank)
        low_a, high_a = bootstrap_median_ci(energy_a_rank)
        low_b, high_b = bootstrap_median_ci(energy_b_rank)
        result["energy_a"] = fmt(med_a, 12)
        result["energy_a_low"] = fmt(low_a, 12)
        result["energy_a_high"] = fmt(high_a, 12)
        result["energy_b"] = fmt(med_b, 12)
        result["energy_b_low"] = fmt(low_b, 12)
        result["energy_b_high"] = fmt(high_b, 12)
        p_energy = mann_whitney(energy_a_rank, energy_b_rank)
        result["energy_p"] = fmt(p_energy, 4)
        if worse(med_a, med_b, p_energy):
            reasons.append("energy")
        wall_a = column(a_rows, "wall_s")
        wall_b = column(b_rows, "wall_s")
        evals_a = column(a_rows, "evals")
        evals_b = column(b_rows, "evals")
        if wall_a and wall_b:
            result["wall_a"] = fmt(median(wall_a), 6)
            result["wall_b"] = fmt(median(wall_b), 6)
            p_wall = mann_whitney(wall_a, wall_b)
            result["wall_p"] = fmt(p_wall, 4)
        if evals_a and evals_b:
            result["evals_a"] = fmt(median(evals_a), 6)
            result["evals_b"] = fmt(median(evals_b), 6)
            p_evals = mann_whitney(evals_a, evals_b)
            result["evals_p"] = fmt(p_evals, 4)
        family = "hexamer" if cell in XTB_CELLS else "morse"
        source = bands.get(family, (float("nan"), float("nan")))
        local_wall = median(wall_a) if wall_a else float("nan")
        if cell in AA_CELLS:
            band_low, band_high = source
        else:
            band_low, band_high = relative_band(source[0], source[1], local_wall)
        result["band_low"] = fmt(band_low, 6)
        result["band_high"] = fmt(band_high, 6)
        if wall_a and wall_b:
            wall_worse = worse(median(wall_a), median(wall_b), mann_whitney(wall_a, wall_b))
            wall_out = outside(median(wall_b), band_low, band_high)
            if cell in GUARDS and wall_out:
                reasons.append("wall-band")
            elif cell not in GUARDS and wall_worse and wall_out:
                reasons.append("wall")
        if evals_a and evals_b:
            eval_worse = worse(median(evals_a), median(evals_b), mann_whitney(evals_a, evals_b))
            if cell not in GUARDS and eval_worse:
                reasons.append("evals")
        if cell in GUARDS:
            paired = pair_by_seed(ab)
            energy_worst = 0.0
            force_worst = 0.0
            eval_mismatch = 0
            hit_mismatch = 0
            missing_force = 0
            for seed, arms in paired.items():
                if "A" not in arms or "B" not in arms:
                    continue
                left, right = arms["A"], arms["B"]
                delta = energy_delta(left["energy"], right["energy"])
                if delta is None or delta > EQUAL_ABS:
                    energy_worst = math.inf
                else:
                    energy_worst = max(energy_worst, delta)
                if left["hit"] != right["hit"]:
                    hit_mismatch += 1
                if left["evals"] != right["evals"]:
                    eval_mismatch += 1
                if cell == "au":
                    fdelta = force_delta(left["forces"], right["forces"])
                    gdelta = energy_delta(left["grad_energy"], right["grad_energy"])
                    if fdelta is None or gdelta is None:
                        missing_force += 1
                    else:
                        force_worst = max(force_worst, fdelta, gdelta)
            result["guard_energy"] = fmt(energy_worst, 3)
            result["guard_force"] = fmt(force_worst, 3) if cell == "au" else "not-printed"
            result["guard_evals"] = str(eval_mismatch)
            if energy_worst > EQUAL_ABS:
                reasons.append("guard-energy")
            if hit_mismatch:
                reasons.append("guard-hits")
            if eval_mismatch:
                reasons.append("guard-evals")
            if cell == "au" and (missing_force or force_worst > EQUAL_ABS):
                reasons.append("guard-force")
        if cell == "hexamer":
            gates = {row.get("grad_energy", "") for row in ab}
            if gates != {"0.598496"}:
                reasons.append("gate")
    if not reasons:
        result["verdict"] = "PASS"
        result["reason"] = "none"
    else:
        result["verdict"] = "FAIL"
        result["reason"] = ",".join(reasons)
    return result


def slot_counts(rows: list[dict[str, str]]) -> list[str]:
    lines = ["slot arm pair count"]
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        if row["status"] != "ok" or row["pair"] != "ab":
            continue
        counts[(row["cell"], row["slot"], row["arm"], row["pair"])] += 1
    for cell in CELLS:
        for slot in ("0", "1"):
            for arm in ("A", "B"):
                lines.append(f"{cell} slot {slot} arm {arm} {counts[(cell, slot, arm, 'ab')]}")
    return lines


def read_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.is_file():
        return meta
    for line in path.read_text().splitlines():
        key, _, value = line.partition(" ")
        if key:
            meta[key] = value
    return meta


def render_report(rows: list[dict[str, str]], meta: dict[str, str]) -> str:
    bands: dict[str, tuple[float, float]] = {}
    for cell in AA_CELLS:
        _, _, low, high = aa_band(ok_rows(rows, cell, "aa"), "wall_s")
        bands[cell] = (low, high)
    reports = [cell_report(cell, rows, bands) for cell in CELLS]
    lines = [
        "# xtb force-sign gate",
        "",
        f"Arm A sha {meta.get('arm_a_sha', 'unset')} tree {meta.get('arm_a_tree', 'unset')}.",
        f"Arm B sha {meta.get('arm_b_sha', 'unset')} tree {meta.get('arm_b_tree', 'unset')}.",
        "",
        "Arm A publishes the xtb buffer after negating it. The engine already",
        "stores the physical force, and the potential negates that buffer once",
        "to form the objective gradient, so the extra negation reverses the force.",
        "Arm B publishes the engine force unchanged.",
        "",
        "Both arms ran natively. No gradient shim was applied.",
        "The H2O20 and crest-equal hit rules rescore the saved geometry with",
        "xtb GFN2 vtight. CREST itself is not executed; H2O20 hits at or below",
        "-101.749249454294 Eh by 1e-6 Eh, the campaign reference.",
        "The crest-equal rows hit at or below -30.496506792187 Eh by 1e-7 Eh.",
        "Hexamer prism hits are an intact cluster whose prism IRA RMSD is at most",
        "0.598496 A, the half-gap cutoff at 0.598 A.",
        "Lennard-Jones is dynamic lattice search at 400000 evaluations.",
        "Morse is m38r6 at 400000 evaluations.",
        "Gold is Sutton-Chen family 108 (sc108-38) at 400000 evaluations,",
        "with forces from a second evaluation of the saved geometry.",
        "That example does not print a force vector for Lennard-Jones or Morse;",
        "those guards compare the printed energy, the hit, and the charged count.",
        "",
        f"Two-sided exact tests use Bonferroni alpha {ALPHA:.6f} across the six cells.",
        f"Median intervals are percentile bootstrap, {BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED}.",
        "The Mann-Whitney p-value is the exact rank-sum test.",
        "A scoring cell fails when B is significantly worse on hits, energy, or cost.",
        "Cost is worse only when the exact test says so and the median wall is outside the A/A band.",
        "A guard fails when a printed energy differs by more than 1e-12, a hit or",
        "charged count differs, a gold force component differs by more than 1e-12,",
        "or the median wall is outside the A/A band.",
        "The hexamer A/A wall band is applied to the xtb cells.",
        "The Morse A/A wall band is applied to the guards.",
        "",
        "## Slot check",
        "",
        "Even seed indexes run A then B. Odd indexes run B then A.",
        "Each AB slot therefore holds six A runs and six B runs.",
        "",
        "```",
        *slot_counts(rows),
        "```",
        "",
        "## A/A wall band",
        "",
    ]
    for cell in AA_CELLS:
        med0, med1, low, high = aa_band(ok_rows(rows, cell, "aa"), "wall_s")
        lines.append(
            f"{cell}: replica medians {fmt(med0)} and {fmt(med1)}, band [{fmt(low)}, {fmt(high)}]"
        )
    lines.extend(
        [
            "",
            "## Cells",
            "",
            "| cell | hits A | hits B | Fisher p | median energy A (95% CI) | median energy B (95% CI) | energy p | median wall A | median wall B | wall p | A/A band | median evals A | median evals B | evals p | verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in reports:
        lines.append(
            "| {cell} | {hits_a}/{n_a} | {hits_b}/{n_b} | {fisher_p} | {energy_a} ({energy_a_low}, {energy_a_high}) | {energy_b} ({energy_b_low}, {energy_b_high}) | {energy_p} | {wall_a} | {wall_b} | {wall_p} | [{band_low}, {band_high}] | {evals_a} | {evals_b} | {evals_p} | {verdict} |".format(
                **item
            )
        )
    lines.extend(["", "## Verdicts", ""])
    for item in reports:
        lines.append(f"{item['cell']}: {item['verdict']} ({item['reason']})")
    failed = [item["cell"] for item in reports if item["verdict"] != "PASS"]
    lines.extend(
        [
            "",
            "GATE " + ("PASS" if not failed else "FAIL"),
            "",
        ]
    )
    if failed:
        lines.append("Failed cells: " + ", ".join(failed))
        lines.append("")
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    rows = load_rows(Path(args.rows))
    meta = read_meta(Path(args.meta)) if args.meta else {}
    text = render_report(rows, meta)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(text)
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    assert fisher(6, 12, 6, 12) == 1.0
    assert fisher(0, 12, 0, 12) == 1.0
    separated = fisher(12, 12, 0, 12)
    assert separated < 1e-6, separated
    tied = mann_whitney([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert tied == 1.0, tied
    ranked = mann_whitney([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert abs(ranked - 0.1) < 1e-12, ranked
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
    low, high = bootstrap_median_ci([1.0, 2.0, 3.0, 4.0], draws=200)
    assert low <= 2.5 <= high
    counts: Counter[tuple[int, str]] = Counter()
    for index in range(12):
        for slot, arm, _replica in abba(index):
            counts[(slot, arm)] += 1
    assert counts[(0, "A")] == 6 and counts[(0, "B")] == 6
    assert counts[(1, "A")] == 6 and counts[(1, "B")] == 6
    aa_counts: Counter[tuple[int, str]] = Counter()
    for index in range(12):
        for slot, arm, replica in aa_order(index):
            aa_counts[(slot, replica)] += 1
            assert arm == "A"
    assert aa_counts[(0, "0")] == 6 and aa_counts[(0, "1")] == 6
    assert energy_delta("1.00000000", "1.00000000") == 0.0
    assert force_delta("1.000000000000 2.0", "1.000000000000 2.0") == 0.0
    assert force_delta("0.0", "1e-12") == 1e-12
    assert abs(PRISM_CUTOFF_A - 0.598496) < 1e-12
    left = [float(i) for i in range(12)]
    right = [float(i) + 0.5 for i in range(12)]
    compared = mann_whitney(left, right)
    assert 0.0 < compared < 1.0, compared
    print("selftest ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--work", required=True)
    run.add_argument("--cells", default=",".join(CELLS))
    report = sub.add_parser("report")
    report.add_argument("--rows", required=True)
    report.add_argument("--meta", default="")
    report.add_argument("--output", required=True)
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)
    return cmd_selftest(args)


if __name__ == "__main__":
    raise SystemExit(main())
