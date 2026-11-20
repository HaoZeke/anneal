#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Extension of the sc-shelf-land4 gate A grid to n=24.

Four cells (lj75, m30r14, sc96-55, sc96-80), new seeds 2112..2123,
arms A and B, and a second build of A for the A/A band. The drivers,
budgets, references, hit rule, and release build are the gate A method.
The analysis below is fixed before the runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HIT_TOL = 0.001
NEW_SEEDS = tuple(range(2112, 2124))
PRIOR_SEEDS = tuple(range(2100, 2112))
POOLED_SEEDS = PRIOR_SEEDS + NEW_SEEDS
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261120
MISS_SHIFT = 1_000_000_000_000
GATE_BAND = 1.033093
PRIOR_ROWS_SHA256 = "03f37e1196b611c550f140455ac697addac5232f4abcb762213fafc69b9d1768"

A_COMMIT = "61fa67cf380e46876b569da45e7cc091647a775b"
A_TREE = "cff796c507a8fa1d0bfb28888c3570b61988055e"
B_COMMIT = "0f0f41b7541f0aaab3e21261e58380e1468f40b8"
B_TREE = "3381b56419aeeece37ff4b13d370967b0b2ab6d0"
RGPOT_REV = "672aef430855036125c902ace024756d682e180a"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DOMAIN_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "RAYON_NUM_THREADS": "1",
    "RUST_TEST_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
}

DRIVERS = ("h2h_gmin_proof", "h2h_morse", "h2h_sc")

ROW_FIELDS = [
    "session",
    "system",
    "family",
    "arm",
    "seed",
    "slot",
    "order",
    "budget",
    "reference",
    "best",
    "best_bits",
    "hit",
    "calls",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "core",
    "hashed_machine_id",
    "code_revision",
    "tree",
    "binary_sha256",
    "driver",
]

SESSION_FIELDS = [
    "session",
    "phase",
    "started_utc",
    "finished_utc",
    "core",
    "cpus_allowed",
    "hashed_machine_id",
    "rustc_version",
    "cargo_version",
    "arm",
    "code_revision",
    "tree",
    "driver",
    "binary_sha256",
    "rustflags",
    "rgpot_revision",
    "librgpot_sha256",
    "thread_env",
]

POOLED_FIELDS = [
    "system",
    "seed",
    "cohort",
    "hit_a",
    "hit_b",
    "calls_a",
    "calls_b",
    "charged_a",
    "charged_b",
    "outside_s_a",
    "outside_s_b",
    "opc_a",
    "opc_b",
    "best_a",
    "best_b",
    "pair",
]

SUMMARY_FIELDS = [
    "system",
    "cohort",
    "n",
    "both",
    "a_only",
    "b_only",
    "neither",
    "mcnemar_p",
    "wilcoxon_n",
    "wilcoxon_w_plus",
    "wilcoxon_p",
    "med_opc_a",
    "med_opc_b",
    "ratio_b_over_a",
    "ratio_ci_low",
    "ratio_ci_high",
    "new_band",
    "gate_band",
    "envelope",
    "point_gt_new_band",
    "point_gt_gate_band",
    "point_gt_envelope",
    "ci_gt_new_band",
    "ci_gt_gate_band",
    "ci_gt_envelope",
    "hit_str_a",
    "hit_str_b",
]

AA_FIELDS = [
    "system",
    "med_opc_a",
    "med_opc_a_prime",
    "ratio_a_prime_over_a",
    "sym_ratio",
    "new_band",
    "gate_band",
]


@dataclass(frozen=True)
class Cell:
    system: str
    family: str
    budget: int
    reference: float
    reference_str: str
    driver: str


CELLS: tuple[Cell, ...] = (
    Cell("lj75", "lj", 4_000_000, -397.492331, "-397.492331", "h2h_gmin_proof"),
    Cell("m30r14", "morse", 400_000, -106.835790, "-106.835790", "h2h_morse"),
    Cell("sc96-55", "sutton-chen", 1_000_000, -13217.8963, "-13217.8963", "h2h_sc"),
    Cell("sc96-80", "sutton-chen", 1_000_000, -19560.8966, "-19560.8966", "h2h_sc"),
)
CELL_BY_NAME = {cell.system: cell for cell in CELLS}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def session_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stamp_to_utc(stamp: str) -> str:
    return (
        f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"
        f"T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    raise RuntimeError("no machine-id file found")


def cpus_allowed() -> str:
    status = Path("/proc/self/status")
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def float_bits_hex(value: float) -> str:
    import struct

    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return f"0x{bits:016x}"


def rederived_hit(best_str: str, reference: float, tol: float = HIT_TOL) -> int:
    try:
        value = float(best_str)
    except ValueError:
        return 0
    if not math.isfinite(value):
        return 0
    return 1 if value <= reference + tol else 0


def type7(values: list[float], p: float) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n < 1:
        raise ValueError("type7 requires at least one value")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(math.floor(h))
    frac = h - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def abba_pair(seed_index: int, first: str, second: str) -> tuple[str, tuple[str, str]]:
    if seed_index % 4 in (0, 3):
        tag = "AB" if (first, second) == ("A", "B") else f"{first}/{second}"
        return tag, (first, second)
    tag = "BA" if (first, second) == ("A", "B") else f"{second}/{first}"
    return tag, (second, first)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar test, binomial on the discordant pairs.

    Under the null, the A-only count is Binomial(n, 1/2) given
    n = A-only + B-only. The p-value sums the point masses whose
    binomial coefficient is at most the observed coefficient.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    observed = math.comb(n, a_only)
    extreme = sum(math.comb(n, k) for k in range(n + 1) if math.comb(n, k) <= observed)
    return extreme / float(1 << n)


def wilcoxon_exact(scores_a: list[int], scores_b: list[int]) -> tuple[float, int, float]:
    """Exact two-sided Wilcoxon signed-rank test.

    Zero differences are dropped. Tied absolute differences take
    average ranks. Every sign assignment of the remaining pairs is
    equally likely. The p-value is the fraction whose absolute
    deviation from the null mean is at least the observed deviation.
    """
    diffs = [b - a for a, b in zip(scores_a, scores_b, strict=True)]
    nonzero = [d for d in diffs if d != 0]
    k = len(nonzero)
    if k == 0:
        return 0.0, 0, 1.0
    magnitudes = [abs(d) for d in nonzero]
    order = sorted(range(k), key=lambda i: magnitudes[i])
    ranks = [0.0] * k
    group = 0
    while group < k:
        end = group
        while end + 1 < k and magnitudes[order[end + 1]] == magnitudes[order[group]]:
            end += 1
        average = (group + 1 + end + 1) / 2.0
        for pos in range(group, end + 1):
            ranks[order[pos]] = average
        group = end + 1
    doubled = [int(round(rank * 2.0)) for rank in ranks]
    if any(abs(rank * 2.0 - value) > 1e-9 for rank, value in zip(ranks, doubled)):
        raise RuntimeError("rank doubling left a non-integer")
    positive = sum(value for value, diff in zip(doubled, nonzero) if diff > 0)
    total_doubled = sum(doubled)
    observed = abs(2 * positive - total_doubled)
    ways: dict[int, int] = {0: 1}
    for rank in doubled:
        nxt: dict[int, int] = {}
        for signed, count in ways.items():
            nxt[signed] = nxt.get(signed, 0) + count
            nxt[signed + rank] = nxt.get(signed + rank, 0) + count
        ways = nxt
    total = 1 << k
    if sum(ways.values()) != total:
        raise RuntimeError("sign enumeration did not cover 2^k")
    extreme = sum(count for signed, count in ways.items() if abs(2 * signed - total_doubled) >= observed)
    return positive / 2.0, k, extreme / float(total)


def paired_score(hit: int, charged: int) -> int:
    return charged + (0 if hit else MISS_SHIFT)


def bootstrap_ratio(opc_a: list[float], opc_b: list[float]) -> tuple[float, float]:
    """Percentile interval of the type-7 median ratio.

    One fresh Random(20261120) draws seeds with replacement, the pair
    staying together. Each draw recomputes the two medians and their
    ratio. Endpoints are type 7 at 0.025 and 0.975.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(opc_a)
    ratios: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        picked = [rng.randrange(n) for _ in range(n)]
        med_a = type7([opc_a[i] for i in picked], 0.5)
        med_b = type7([opc_b[i] for i in picked], 0.5)
        if med_a == 0.0:
            continue
        ratios.append(med_b / med_a)
    if len(ratios) < 100:
        raise RuntimeError("bootstrap produced too few finite ratios")
    return type7(ratios, 0.025), type7(ratios, 0.975)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def child_env(sc_lib: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("H2H_"):
            del env[key]
    env.update(THREAD_ENV)
    if sc_lib is not None:
        lib_dir = str(sc_lib.resolve().parent)
        env["RGPOT_SC_LIB"] = str(sc_lib.resolve())
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{prev}" if prev else lib_dir
    return env


def parse_csv_driver(stdout: str, system: str, seed: int, budget: int) -> dict[str, str]:
    headers = [line for line in stdout.splitlines() if line.startswith("system,")]
    data_lines = [line for line in stdout.splitlines() if line.startswith(f"{system},anneal,{seed},")]
    if len(headers) != 1 or len(data_lines) != 1:
        raise RuntimeError(f"{system} seed {seed}: malformed output\n{stdout[-2000:]}")
    row = dict(zip(headers[0].split(","), data_lines[0].split(","), strict=False))
    if int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: budget/seed mismatch")
    wall_s = float(row["wall_s"])
    pot_s = float(row["pot_s"])
    outside_s = float(row["outside_s"]) if "outside_s" in row else max(wall_s - pot_s, 0.0)
    return {
        "best": row["best"].strip(),
        "calls": str(int(row["evals_to_hit"])),
        "charged": str(int(row["charged"])),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{max(outside_s, 0.0):.6f}",
    }


def run_cell_seed(
    *,
    cell: Cell,
    seed: int,
    bin_dir: Path,
    work_cwd: Path,
    sc_lib: Path,
    log_path: Path,
) -> tuple[dict[str, str], str]:
    binary = bin_dir / cell.driver
    if not binary.is_file():
        raise RuntimeError(f"missing binary: {binary.name}")
    bin_sha = sha256_file(binary)
    env = child_env(sc_lib if cell.driver == "h2h_sc" else None)
    cmd = ["taskset", "-c", "0", str(binary), "run", cell.system, str(seed)]
    proc = subprocess.run(
        cmd,
        cwd=work_cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"cmd={' '.join(cmd)}\nexit={proc.returncode}\n{proc.stdout}\n{proc.stderr}",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{cell.system} {cell.driver} seed {seed} failed ({proc.returncode})")
    return parse_csv_driver(proc.stdout, cell.system, seed, cell.budget), bin_sha


def git_rev_and_tree(tree_path: Path) -> tuple[str, str]:
    rev = subprocess.check_output(["git", "-C", str(tree_path), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(tree_path), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    return rev, tree


def build_row(
    *,
    session: str,
    cell: Cell,
    arm: str,
    seed: int,
    slot: int,
    order: str,
    parsed: dict[str, str],
    machine_id: str,
    rev: str,
    tree: str,
    bin_sha: str,
) -> dict[str, str]:
    best_str = parsed["best"]
    best_val = float(best_str)
    hit = rederived_hit(best_str, cell.reference)
    calls = int(parsed["calls"]) if hit == 1 else cell.budget
    return {
        "session": session,
        "system": cell.system,
        "family": cell.family,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "order": order,
        "budget": str(cell.budget),
        "reference": cell.reference_str,
        "best": best_str,
        "best_bits": float_bits_hex(best_val) if math.isfinite(best_val) else "nan",
        "hit": str(hit),
        "calls": str(calls),
        "charged": parsed["charged"],
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
        "core": "0",
        "hashed_machine_id": machine_id,
        "code_revision": rev,
        "tree": tree,
        "binary_sha256": bin_sha,
        "driver": cell.driver,
    }


def check_budget(binary: Path, cell: Cell) -> None:
    proc = subprocess.run(
        [str(binary), "budget", cell.system],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"budget query failed for {cell.system}: {proc.stderr[-500:]}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2 or int(lines[0]) != cell.budget:
        raise RuntimeError(f"budget table mismatch for {cell.system}: {lines[:2]}")
    if abs(float(lines[1]) - cell.reference) > 1e-9:
        raise RuntimeError(f"reference mismatch for {cell.system}: {lines[1]}")


def prior_rows_path(out_dir: Path) -> Path:
    return out_dir.parent / "sc-shelf-land4-gate-a" / "rows.csv"


def require_prior_rows(path: Path) -> None:
    if sha256_file(path) != PRIOR_ROWS_SHA256:
        raise RuntimeError("gate A rows.csv hash changed; refusing to pool")


def hit_of(row: dict[str, str], cell: Cell) -> int:
    hit = rederived_hit(row["best"], cell.reference)
    if str(hit) != row["hit"]:
        raise RuntimeError(f"stored hit disagrees with the rule for {cell.system} seed {row['seed']}")
    return hit


def index_phase(
    rows: list[dict[str, str]],
    *,
    orders: set[str],
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, dict[str, dict[int, dict[str, str]]]]:
    found: dict[str, dict[str, dict[int, dict[str, str]]]] = {
        cell.system: {arm: {} for arm in arms} for cell in CELLS
    }
    seed_set = set(seeds)
    for row in rows:
        if row["system"] not in found or row["order"] not in orders or row["arm"] not in found[row["system"]]:
            continue
        seed = int(row["seed"])
        if seed not in seed_set:
            continue
        slot = found[row["system"]][row["arm"]]
        if seed in slot:
            raise RuntimeError(f"duplicate {row['system']} {row['arm']} {seed} {row['order']}")
        slot[seed] = row
    return found


def require_complete(
    indexed: dict[str, dict[str, dict[int, dict[str, str]]]],
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
    label: str,
) -> None:
    for cell in CELLS:
        for arm in arms:
            missing = [seed for seed in seeds if seed not in indexed[cell.system][arm]]
            if missing:
                raise RuntimeError(f"missing {label} {cell.system} {arm} {missing[:4]}")


def opc_of(row: dict[str, str]) -> float:
    charged = int(row["charged"])
    if charged <= 0:
        raise RuntimeError(f"non-positive charged count for {row['system']} seed {row['seed']}")
    return float(row["outside_s"]) / charged


def discordance(hits_a: list[int], hits_b: list[int]) -> dict[str, int]:
    both = a_only = b_only = neither = 0
    for left, right in zip(hits_a, hits_b, strict=True):
        if left and right:
            both += 1
        elif left and not right:
            a_only += 1
        elif right and not left:
            b_only += 1
        else:
            neither += 1
    return {"both": both, "a_only": a_only, "b_only": b_only, "neither": neither}


def fmt(value: float) -> str:
    return format(value, ".16g")


def flag(cond: bool) -> str:
    return "1" if cond else "0"


PROTOCOL = """\
# sc-shelf-land4 gate A extension

Paired extension of gate A on the four cells that failed that gate.
The rules in this file are the analysis. They are fixed before the
new seeds run and are not revised afterwards.

## Arms and cells

- Arm A: `61fa67cf380e46876b569da45e7cc091647a775b`, tree `cff796c507a8fa1d0bfb28888c3570b61988055e`.
- Arm B: `0f0f41b7541f0aaab3e21261e58380e1468f40b8`, tree `3381b56419aeeece37ff4b13d370967b0b2ab6d0`.
- Arm A': a second `cargo build --release --locked` of arm A, fresh target directory, same commit and tree.
- Cells, drivers, budgets, and references, unchanged from gate A:

| system | driver | budget | reference |
|---|---|---:|---:|
| lj75 | `h2h_gmin_proof` | 4000000 | -397.492331 |
| m30r14 | `h2h_morse` | 400000 | -106.835790 |
| sc96-55 | `h2h_sc` | 1000000 | -13217.8963 |
| sc96-80 | `h2h_sc` | 1000000 | -19560.8966 |

- Build: `rustc 1.88.0`, `cargo +1.88.0 build --release --locked --features rgpot-ex --example h2h_gmin_proof --example h2h_morse --example h2h_sc`. The feature is required to compile `h2h_sc`. Path prefixes in the compiler flags are remapped so a host path is not embedded.
- rgpot revision `672aef430855036125c902ace024756d682e180a`. Sutton-Chen loads that build through `RGPOT_SC_LIB`.
- Hit rule: a finite `best` is a hit when `best <= reference + 0.001`. The stored hit column is that rule, re-derived. A stored hit that disagrees stops the analysis.
- New seeds: `2112..2123` only (12 seeds). Seeds `2100..2111` are read from the gate A `rows.csv` and are not rerun. That file is checked by SHA-256 and is not rewritten.
- No warm-up search and no quick flag. A `budget` query checks the table in the binary before the grid. It does not search.

## Execution

Serial, one search at a time. The process tree is pinned with `taskset -c 0`. Each search is also started under `taskset -c 0`. Every thread variable below is `1`:

`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `MKL_DOMAIN_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS`, `GOTO_NUM_THREADS`, `OMP_THREAD_LIMIT`, `RAYON_NUM_THREADS`, `RUST_TEST_THREADS`, `TOKIO_WORKER_THREADS`, `POLARS_MAX_THREADS`.

Arm order is ABBA on the index in `2112..2123`. Index mod 4 in {0, 3} runs the first arm and then the second. The other indices run the second arm and then the first. The A/B grid uses A then B. The A/A' grid uses A then A', after the A/B grid has finished, in the same cell order. Six seeds in each grid have each arm in slot 0 and six have it in slot 1.

`rows.csv` holds the new runs only, both grids, with the gate A row columns and no seat column. A/B rows have order `AB` or `BA`. A/A' rows have order `A/A_prime` or `A_prime/A`. `sessions.csv` records the binaries and the pin, with no seat column. The machine is recorded as the SHA-256 of the operating system machine id.

## Analysis

The pooled sample is seeds `2100..2123` (n=24) for each cell. The new sample is seeds `2112..2123` (n=12). Both are reported for every test below.

Discordance counts, per cell, of the paired hits: both, A only, B only, neither.

Exact two-sided McNemar p, binomial on the discordant pairs. Let b be A only, c be B only, and n = b + c. If n is 0 the p-value is 1. Otherwise it is the sum of Binomial(n, 1/2) point masses whose binomial coefficient is at most C(n, b), divided by 2^n.

Wilcoxon signed-rank on charged calls, paired by seed, misses ranked worst. The paired value is the `charged` column. A miss adds 1000000000000, which is larger than any budget here, so every miss ranks strictly worse than every hit. Within a hit class, a larger charged count is worse. The signed difference is B minus A. Zero differences are dropped. Tied absolute differences take average ranks. The exact two-sided p-value is the fraction of the 2^k sign assignments whose absolute deviation from the null mean is at least the observed deviation. W+ is the sum of the positive ranks.

Outside time per charged call is `outside_s / charged` on that row. The arm summary is the type-7 median (the same quantile gate A uses), and the reported ratio is the median of B divided by the median of A. It is not the median of pairwise ratios and not a ratio of sums.

The bootstrap 95% interval resamples seeds with replacement, keeping each pair together, 10000 times. A new `random.Random(20261120)` is created for each system and cohort. Each draw recomputes the two type-7 medians and their ratio. The endpoints are type 7 at 0.025 and 0.975.

The new A/A band uses the A and A' rows of the new seeds only. For each cell, r is the type-7 median of A' divided by the type-7 median of A, and the cell contribution is max(r, 1/r). The new band is the maximum of those four contributions. The gate A band is the published value 1.033093. The envelope is the larger of the new band and 1.033093. A point exceedance is a B/A median ratio strictly greater than the band. A CI exceedance is a lower endpoint strictly greater than the band. Both comparisons are reported for the new band, the gate band, and the envelope.

`pooled.csv` is one row per cell and seed from 2100 through 2123. `summary.csv` is one row per cell and cohort (`pooled`, then `new`). `aa_band.csv` is one row per cell with the A/A' medians, the symmetric ratio, the new band, and the gate band. Hit strings are ordered by increasing seed, `1` for a hit and `0` for a miss.
"""


def render_readme(results: dict[str, object] | None) -> str:
    text = PROTOCOL
    if results is None:
        text += (
            "\n## Results\n\n"
            "The measurement has not been run. The tables below are filled\n"
            "from rows.csv after the grid finishes. The rules above are the\n"
            "analysis. They are not revised after the runs.\n"
        )
        return text
    sessions: list[dict[str, str]] = results["sessions"]  # type: ignore[assignment]
    summary: list[dict[str, str]] = results["summary"]  # type: ignore[assignment]
    band: list[dict[str, str]] = results["band"]  # type: ignore[assignment]
    machine_ids = sorted({row["hashed_machine_id"] for row in sessions})
    rustc = sorted({row["rustc_version"] for row in sessions})
    cargo = sorted({row["cargo_version"] for row in sessions})
    rgpot = sorted({row["rgpot_revision"] for row in sessions})
    libs = sorted({row["librgpot_sha256"] for row in sessions})
    lines = [
        "",
        "## Results",
        "",
        f"- Hashed machine id: `{', '.join(machine_ids)}`",
        f"- Toolchain: `{', '.join(rustc)}`; `{', '.join(cargo)}`",
        f"- rgpot: `{', '.join(rgpot)}`",
        f"- librgpot SHA-256: `{', '.join(libs)}`",
        f"- CPython: `{results['python']}`",
        "",
        "### Binary SHA-256",
        "",
        "| phase | arm | driver | code_revision | tree | binary_sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for row in sessions:
        lines.append(
            f"| {row['phase']} | {row['arm']} | `{row['driver']}` | `{row['code_revision']}` | "
            f"`{row['tree']}` | `{row['binary_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "### A/A band on the new seeds",
            "",
            "| system | med A | med A' | A'/A | max(r, 1/r) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in band:
        lines.append(
            f"| {row['system']} | {float(row['med_opc_a']):.6e} | "
            f"{float(row['med_opc_a_prime']):.6e} | {float(row['ratio_a_prime_over_a']):.6f} | "
            f"{float(row['sym_ratio']):.6f} |"
        )
    lines.append("")
    lines.append(
        f"New band `{float(band[0]['new_band']):.6f}`. "
        f"Gate A band `{band[0]['gate_band']}`. "
        f"Envelope `{max(float(band[0]['new_band']), GATE_BAND):.6f}`."
    )
    lines.extend(
        [
            "",
            "### Paired tests",
            "",
            "| system | cohort | n | both | A only | B only | neither | McNemar p | W+ | Wilcoxon n | Wilcoxon p | med opc B/A | 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['system']} | {row['cohort']} | {row['n']} | {row['both']} | {row['a_only']} | "
            f"{row['b_only']} | {row['neither']} | {float(row['mcnemar_p']):.6g} | "
            f"{float(row['wilcoxon_w_plus']):.4g} | {row['wilcoxon_n']} | {float(row['wilcoxon_p']):.6g} | "
            f"{float(row['ratio_b_over_a']):.6f} | "
            f"[{float(row['ratio_ci_low']):.6f}, {float(row['ratio_ci_high']):.6f}] |"
        )
    lines.extend(
        [
            "",
            "### Band comparisons",
            "",
            "A 1 means the B/A median ratio, or its lower CI endpoint, is strictly above that band.",
            "",
            "| system | cohort | ratio | CI low | point new | point gate | point envelope | CI new | CI gate | CI envelope |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['system']} | {row['cohort']} | {float(row['ratio_b_over_a']):.6f} | "
            f"{float(row['ratio_ci_low']):.6f} | {row['point_gt_new_band']} | {row['point_gt_gate_band']} | "
            f"{row['point_gt_envelope']} | {row['ci_gt_new_band']} | {row['ci_gt_gate_band']} | "
            f"{row['ci_gt_envelope']} |"
        )
    lines.extend(["", "### Per-seed hit strings", "", "| system | cohort | A | B |", "|---|---|---|---|"])
    for row in summary:
        lines.append(f"| {row['system']} | {row['cohort']} | `{row['hit_str_a']}` | `{row['hit_str_b']}` |")
    lines.append("")
    return text


def run_campaign(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    logs_dir: Path = args.logs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    allowed = cpus_allowed()
    if allowed != "0":
        raise RuntimeError(f"process must be pinned to core 0 (Cpus_allowed_list={allowed})")
    machine_id = hashed_machine_id()
    rustc_ver = subprocess.check_output(["rustc", "+1.88.0", "--version"], text=True).strip()
    cargo_ver = subprocess.check_output(["cargo", "+1.88.0", "--version"], text=True).strip()
    if not rustc_ver.startswith("rustc 1.88.0"):
        raise RuntimeError(f"unexpected rustc: {rustc_ver}")
    rgpot_rev = subprocess.check_output(["git", "-C", str(args.rgpot_root), "rev-parse", "HEAD"], text=True).strip()
    if rgpot_rev != RGPOT_REV:
        raise RuntimeError(f"rgpot revision {rgpot_rev} is not {RGPOT_REV}")
    librgpot_sha = sha256_file(args.sc_lib.resolve())
    a_rev, a_tree = git_rev_and_tree(args.a_tree)
    b_rev, b_tree = git_rev_and_tree(args.b_tree)
    ap_rev, ap_tree = git_rev_and_tree(args.a_prime_tree)
    if (a_rev, a_tree) != (A_COMMIT, A_TREE):
        raise RuntimeError(f"arm A mismatch {a_rev} {a_tree}")
    if (b_rev, b_tree) != (B_COMMIT, B_TREE):
        raise RuntimeError(f"arm B mismatch {b_rev} {b_tree}")
    if (ap_rev, ap_tree) != (A_COMMIT, A_TREE):
        raise RuntimeError(f"arm A' mismatch {ap_rev} {ap_tree}")
    for cell in CELLS:
        check_budget(args.a_bin / cell.driver, cell)
        check_budget(args.b_bin / cell.driver, cell)
        check_budget(args.a_prime_bin / cell.driver, cell)
    for driver in DRIVERS:
        for directory in (args.a_bin, args.b_bin, args.a_prime_bin):
            if not (directory / driver).is_file():
                raise RuntimeError(f"missing {driver} in {directory.name}")

    rows_path = out_dir / "rows.csv"
    existing_rows = read_csv(rows_path)
    session_id = existing_rows[0]["session"] if existing_rows else session_stamp()
    started = stamp_to_utc(session_id)
    done = {
        (row["system"], row["arm"], int(row["seed"]), row["order"])
        for row in existing_rows
    }
    phases = (
        (
            "ab",
            ("A", "B"),
            {
                "A": (args.a_bin, args.a_tree, a_rev, a_tree),
                "B": (args.b_bin, args.b_tree, b_rev, b_tree),
            },
        ),
        (
            "aa",
            ("A", "A_prime"),
            {
                "A": (args.a_bin, args.a_tree, a_rev, a_tree),
                "A_prime": (args.a_prime_bin, args.a_prime_tree, ap_rev, ap_tree),
            },
        ),
    )
    for phase, labels, meta in phases:
        for cell in CELLS:
            for index, seed in enumerate(NEW_SEEDS):
                order, seq = abba_pair(index, labels[0], labels[1])
                for slot, arm in enumerate(seq):
                    key = (cell.system, arm, seed, order)
                    if key in done:
                        continue
                    bin_dir, tree_cwd, rev, tree = meta[arm]
                    parsed, bin_sha = run_cell_seed(
                        cell=cell,
                        seed=seed,
                        bin_dir=bin_dir,
                        work_cwd=tree_cwd,
                        sc_lib=args.sc_lib,
                        log_path=logs_dir / phase / cell.system / f"seed_{seed}_{arm}.log",
                    )
                    row = build_row(
                        session=session_id,
                        cell=cell,
                        arm=arm,
                        seed=seed,
                        slot=slot,
                        order=order,
                        parsed=parsed,
                        machine_id=machine_id,
                        rev=rev,
                        tree=tree,
                        bin_sha=bin_sha,
                    )
                    append_csv(rows_path, ROW_FIELDS, row)
                    done.add(key)
                    print(
                        f"[{phase}] {cell.system} seed={seed} slot={slot} arm={arm} "
                        f"hit={row['hit']} calls={row['calls']} charged={row['charged']} "
                        f"wall={row['wall_s']}",
                        flush=True,
                    )
    finished = now_utc()
    sessions_path = out_dir / "sessions.csv"
    if len(read_csv(sessions_path)) != 4 * len(DRIVERS):
        thread_env = ";".join(f"{key}={value}" for key, value in THREAD_ENV.items())
        recorded = (
            ("ab", "A", a_rev, a_tree, args.a_bin),
            ("ab", "B", b_rev, b_tree, args.b_bin),
            ("aa", "A", a_rev, a_tree, args.a_bin),
            ("aa", "A_prime", ap_rev, ap_tree, args.a_prime_bin),
        )
        session_rows = []
        for phase, arm, rev, tree, bin_dir in recorded:
            for driver in DRIVERS:
                session_rows.append(
                    {
                        "session": session_id,
                        "phase": phase,
                        "started_utc": started,
                        "finished_utc": finished,
                        "core": "0",
                        "cpus_allowed": allowed,
                        "hashed_machine_id": machine_id,
                        "rustc_version": rustc_ver,
                        "cargo_version": cargo_ver,
                        "arm": arm,
                        "code_revision": rev,
                        "tree": tree,
                        "driver": driver,
                        "binary_sha256": sha256_file(bin_dir / driver),
                        "rustflags": args.rustflags_label,
                        "rgpot_revision": rgpot_rev,
                        "librgpot_sha256": librgpot_sha,
                        "thread_env": thread_env,
                    }
                )
        write_csv(sessions_path, SESSION_FIELDS, session_rows)
    print("CAMPAIGN COMPLETE", flush=True)
    return 0


def cohort_summary(
    *,
    system: str,
    cohort: str,
    seeds: tuple[int, ...],
    paired: dict[int, dict[str, dict[str, str]]],
    new_band: float,
) -> dict[str, str]:
    cell = CELL_BY_NAME[system]
    hits_a = [hit_of(paired[seed]["A"], cell) for seed in seeds]
    hits_b = [hit_of(paired[seed]["B"], cell) for seed in seeds]
    counts = discordance(hits_a, hits_b)
    scores_a = [paired_score(hit, int(paired[seed]["A"]["charged"])) for seed, hit in zip(seeds, hits_a)]
    scores_b = [paired_score(hit, int(paired[seed]["B"]["charged"])) for seed, hit in zip(seeds, hits_b)]
    w_plus, nonzero, wilcoxon_p = wilcoxon_exact(scores_a, scores_b)
    opc_a = [opc_of(paired[seed]["A"]) for seed in seeds]
    opc_b = [opc_of(paired[seed]["B"]) for seed in seeds]
    med_a = type7(opc_a, 0.5)
    med_b = type7(opc_b, 0.5)
    ratio = med_b / med_a
    ci_low, ci_high = bootstrap_ratio(opc_a, opc_b)
    envelope = max(new_band, GATE_BAND)
    return {
        "system": system,
        "cohort": cohort,
        "n": str(len(seeds)),
        "both": str(counts["both"]),
        "a_only": str(counts["a_only"]),
        "b_only": str(counts["b_only"]),
        "neither": str(counts["neither"]),
        "mcnemar_p": fmt(mcnemar_exact(counts["a_only"], counts["b_only"])),
        "wilcoxon_n": str(nonzero),
        "wilcoxon_w_plus": fmt(w_plus),
        "wilcoxon_p": fmt(wilcoxon_p),
        "med_opc_a": fmt(med_a),
        "med_opc_b": fmt(med_b),
        "ratio_b_over_a": fmt(ratio),
        "ratio_ci_low": fmt(ci_low),
        "ratio_ci_high": fmt(ci_high),
        "new_band": fmt(new_band),
        "gate_band": format(GATE_BAND, ".6f"),
        "envelope": fmt(envelope),
        "point_gt_new_band": flag(ratio > new_band),
        "point_gt_gate_band": flag(ratio > GATE_BAND),
        "point_gt_envelope": flag(ratio > envelope),
        "ci_gt_new_band": flag(ci_low > new_band),
        "ci_gt_gate_band": flag(ci_low > GATE_BAND),
        "ci_gt_envelope": flag(ci_low > envelope),
        "hit_str_a": "".join(str(hit) for hit in hits_a),
        "hit_str_b": "".join(str(hit) for hit in hits_b),
    }


def analyze(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    prior_path = prior_rows_path(out_dir)
    require_prior_rows(prior_path)
    prior = index_phase(
        read_csv(prior_path),
        orders={"AB", "BA"},
        arms=("A", "B"),
        seeds=PRIOR_SEEDS,
    )
    new_rows = read_csv(out_dir / "rows.csv")
    fresh = index_phase(
        new_rows,
        orders={"AB", "BA"},
        arms=("A", "B"),
        seeds=NEW_SEEDS,
    )
    aa = index_phase(
        new_rows,
        orders={"A/A_prime", "A_prime/A"},
        arms=("A", "A_prime"),
        seeds=NEW_SEEDS,
    )
    require_complete(prior, ("A", "B"), PRIOR_SEEDS, "prior")
    require_complete(fresh, ("A", "B"), NEW_SEEDS, "new")
    require_complete(aa, ("A", "A_prime"), NEW_SEEDS, "aa")
    band_rows: list[dict[str, str]] = []
    sym_ratios: list[float] = []
    for cell in CELLS:
        opc_a = [opc_of(aa[cell.system]["A"][seed]) for seed in NEW_SEEDS]
        opc_p = [opc_of(aa[cell.system]["A_prime"][seed]) for seed in NEW_SEEDS]
        med_a = type7(opc_a, 0.5)
        med_p = type7(opc_p, 0.5)
        ratio = med_p / med_a
        sym = max(ratio, 1.0 / ratio)
        sym_ratios.append(sym)
        band_rows.append(
            {
                "system": cell.system,
                "med_opc_a": fmt(med_a),
                "med_opc_a_prime": fmt(med_p),
                "ratio_a_prime_over_a": fmt(ratio),
                "sym_ratio": fmt(sym),
                "new_band": "",
                "gate_band": format(GATE_BAND, ".6f"),
            }
        )
    new_band = max(sym_ratios)
    for row in band_rows:
        row["new_band"] = fmt(new_band)
    pooled_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for cell in CELLS:
        paired: dict[int, dict[str, dict[str, str]]] = {}
        for seed in PRIOR_SEEDS:
            paired[seed] = {"A": prior[cell.system]["A"][seed], "B": prior[cell.system]["B"][seed]}
        for seed in NEW_SEEDS:
            paired[seed] = {"A": fresh[cell.system]["A"][seed], "B": fresh[cell.system]["B"][seed]}
        for seed in POOLED_SEEDS:
            left = paired[seed]["A"]
            right = paired[seed]["B"]
            hit_a = hit_of(left, cell)
            hit_b = hit_of(right, cell)
            if hit_a and hit_b:
                pair = "both"
            elif hit_a:
                pair = "a_only"
            elif hit_b:
                pair = "b_only"
            else:
                pair = "neither"
            pooled_rows.append(
                {
                    "system": cell.system,
                    "seed": str(seed),
                    "cohort": "prior" if seed in PRIOR_SEEDS else "new",
                    "hit_a": str(hit_a),
                    "hit_b": str(hit_b),
                    "calls_a": left["calls"],
                    "calls_b": right["calls"],
                    "charged_a": left["charged"],
                    "charged_b": right["charged"],
                    "outside_s_a": left["outside_s"],
                    "outside_s_b": right["outside_s"],
                    "opc_a": fmt(opc_of(left)),
                    "opc_b": fmt(opc_of(right)),
                    "best_a": left["best"],
                    "best_b": right["best"],
                    "pair": pair,
                }
            )
        summary_rows.append(
            cohort_summary(system=cell.system, cohort="pooled", seeds=POOLED_SEEDS, paired=paired, new_band=new_band)
        )
        summary_rows.append(
            cohort_summary(system=cell.system, cohort="new", seeds=NEW_SEEDS, paired=paired, new_band=new_band)
        )
    write_csv(out_dir / "pooled.csv", POOLED_FIELDS, pooled_rows)
    write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(out_dir / "aa_band.csv", AA_FIELDS, band_rows)
    sessions = read_csv(out_dir / "sessions.csv")
    if not sessions:
        raise RuntimeError("sessions.csv is empty")
    report = render_readme(
        {
            "sessions": sessions,
            "summary": summary_rows,
            "band": band_rows,
            "python": ".".join(map(str, sys.version_info[:3])),
        }
    )
    readme = out_dir / "README.md"
    readme.write_text(report, encoding="utf-8", newline="\n")
    readme_fd = readme.open("a", encoding="utf-8")
    try:
        readme_fd.flush()
        os.fsync(readme_fd.fileno())
    finally:
        readme_fd.close()
    print(f"ANALYZE COMPLETE new_band={new_band:.6f}", flush=True)
    return 0


def declare(out_dir: Path) -> None:
    path = out_dir / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_readme(None), encoding="utf-8", newline="\n")


def self_test() -> None:
    assert NEW_SEEDS[0] == 2112 and NEW_SEEDS[-1] == 2123 and len(NEW_SEEDS) == 12
    assert len(POOLED_SEEDS) == 24
    assert rederived_hit("-397.492331", -397.492331) == 1
    assert rederived_hit("-397.491331", -397.492331) == 1
    assert rederived_hit("-397.491330", -397.492331) == 0
    assert rederived_hit("nan", -397.492331) == 0
    counts = {(slot, arm): 0 for slot in (0, 1) for arm in ("A", "B")}
    for index in range(12):
        _, seq = abba_pair(index, "A", "B")
        for slot, arm in enumerate(seq):
            counts[(slot, arm)] += 1
    assert all(value == 6 for value in counts.values()), counts
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(2, 0) - 0.5) < 1e-15
    assert abs(mcnemar_exact(6, 2) - 74 / 256) < 1e-15
    w_plus, nonzero, p_value = wilcoxon_exact([1, 3, 2], [2, 3, 5])
    assert (w_plus, nonzero, p_value) == (3.0, 2, 0.5)
    assert paired_score(0, 10) > paired_score(1, 4_000_000)
    here = Path(__file__).resolve().parent
    prior = prior_rows_path(here)
    require_prior_rows(prior)
    indexed = index_phase(read_csv(prior), orders={"AB", "BA"}, arms=("A", "B"), seeds=PRIOR_SEEDS)
    expected = {
        "lj75": ("001110111110", "110010010000", 2, 6, 2, 2),
        "m30r14": ("110111101111", "100110111111", 8, 2, 1, 1),
        "sc96-55": ("111111111111", "111111111111", 12, 0, 0, 0),
        "sc96-80": ("000011110111", "000011100110", 5, 2, 0, 5),
    }
    for system, (text_a, text_b, both, a_only, b_only, neither) in expected.items():
        cell = CELL_BY_NAME[system]
        hits_a = [hit_of(indexed[system]["A"][seed], cell) for seed in PRIOR_SEEDS]
        hits_b = [hit_of(indexed[system]["B"][seed], cell) for seed in PRIOR_SEEDS]
        assert "".join(map(str, hits_a)) == text_a
        assert "".join(map(str, hits_b)) == text_b
        counts_d = discordance(hits_a, hits_b)
        assert (counts_d["both"], counts_d["a_only"], counts_d["b_only"], counts_d["neither"]) == (
            both,
            a_only,
            b_only,
            neither,
        )
    assert abs(mcnemar_exact(6, 2) - mcnemar_exact(2, 6)) < 1e-15
    low, high = bootstrap_ratio([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert abs(low - 2.0) < 1e-12 and abs(high - 2.0) < 1e-12
    print("self-test ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--declare", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--a-tree", type=Path)
    parser.add_argument("--b-tree", type=Path)
    parser.add_argument("--a-prime-tree", type=Path)
    parser.add_argument("--a-bin", type=Path)
    parser.add_argument("--b-bin", type=Path)
    parser.add_argument("--a-prime-bin", type=Path)
    parser.add_argument("--rgpot-root", type=Path)
    parser.add_argument("--sc-lib", type=Path)
    parser.add_argument("--rustflags-label", default="remap")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.declare:
        declare(args.out_dir)
        return 0
    if args.analyze:
        return analyze(args)
    required = (
        args.logs_dir,
        args.a_tree,
        args.b_tree,
        args.a_prime_tree,
        args.a_bin,
        args.b_bin,
        args.a_prime_bin,
        args.rgpot_root,
        args.sc_lib,
    )
    if any(item is None for item in required):
        parser.error("missing required campaign paths")
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
