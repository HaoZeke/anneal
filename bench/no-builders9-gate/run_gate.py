#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Three-arm serial gate for the no-builders9 series.

Arms A and A' are separate release builds of one develop tree. Arm B
is that tree plus the series. Drivers are the same source on every
arm. The failure rule is the one declared in the README beside this
file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import math
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HIT_TOL = 1.0e-3
OUTSIDE_RATIO_LIMIT = 1.10
AA_OUTSIDE_LO = 0.95
AA_OUTSIDE_HI = 1.05
ALPHA = 0.05
SEEDS = tuple(range(2000, 2012))
DEFAULT_TIMEOUT_S = 21600
ARMS = ("A", "A'", "B")

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
]

SUMMARY_FIELDS = [
    "cell",
    "family",
    "budget",
    "reference",
    "hits_a",
    "hits_ap",
    "hits_b",
    "fisher_p",
    "med_calls_a",
    "med_calls_ap",
    "med_calls_b",
    "mw_p",
    "calls_ratio_ba",
    "calls_ratio_apa",
    "outside_ba",
    "outside_apa",
    "hit_loss",
    "fail_a",
    "fail_b",
    "fail_c",
    "verdict",
]

THREAD_ENV_KEYS = (
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

SCRUB_PREFIXES = ("H2H_", "ANNEAL_", "TWO_PHASE_", "BANK_")

TIP4P_BUDGET = 2_000_000

# Lennard-Jones sizes and Cambridge energies from examples/h2h_gmin_proof.rs
# and scripts/paper_0120/run_panel.py. Budgets are the paper list.
LJ_SPECS = (
    ("lj38", 400_000, -173.928427),
    ("lj75", 4_000_000, -397.492331),
    ("lj98", 4_000_000, -543.665361),
    ("lj102", 4_000_000, -569.363652),
    ("lj104", 4_000_000, -582.086642),
)

# Budgets and Cambridge headers from the sc-shelf-land4 gate-b harness.
MORSE_SPECS = (
    ("m55r14", 1_000_000, -220.646208),
    ("m80r6", 4_000_000, -378.333471),
    ("m80r14", 4_000_000, -340.811371),
)

BLJ_SPECS = (
    ("blj105n38", 400_000, -177.2090846748),
    ("blj105n55", 1_000_000, -283.1545313810),
    ("blj115n38", 400_000, -187.3676867453),
    ("blj115n55", 1_000_000, -289.2931853058),
)

# Sutton-Chen 12-6 from examples/h2h_sc.rs and scripts/metal_h2h/run_sc.py.
SC_SPECS = (
    ("sc126-38", 400_000, -35419.9804),
    ("sc126-55", 1_000_000, -52884.6806),
)

# Author hit counts, unchecked until the rows exist.
CLAIMS = (
    ("blj115n38", 12, 9, "loss"),
    ("blj105n55", 8, 3, "loss"),
    ("m80r14", 6, 5, "loss"),
    ("lj75", 5, 10, "gain"),
    ("lj98", 7, 11, "gain"),
    ("blj105n38", 2, 9, "gain"),
)


@dataclass(frozen=True)
class CellSpec:
    system: str
    family: str
    driver: str
    budget: int
    reference: float
    ref_text: str
    n: int = 0


def load_tip4p_refs(repo_root: Path) -> dict[int, tuple[float, str]]:
    path = repo_root / "data" / "tip4p_ccd.csv"
    out: dict[int, tuple[float, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            txt = row["energy"].strip()
            out[n] = (float(txt), txt)
    return out


def build_cells(repo_root: Path) -> list[CellSpec]:
    tip4p_refs = load_tip4p_refs(repo_root)
    cells: list[CellSpec] = []
    for system, budget, ref in LJ_SPECS:
        cells.append(CellSpec(system, "lj", "h2h_gmin_proof", budget, ref, f"{ref:.6f}"))
    for system, budget, ref in BLJ_SPECS:
        cells.append(CellSpec(system, "blj", "h2h_blj", budget, ref, f"{ref:.10f}"))
    for system, budget, ref in MORSE_SPECS:
        cells.append(CellSpec(system, "morse", "h2h_morse", budget, ref, f"{ref:.6f}"))
    for system, budget, ref in SC_SPECS:
        cells.append(CellSpec(system, "sc", "h2h_sc", budget, ref, f"{ref:.4f}"))
    missing = [n for n in range(8, 22) if n not in tip4p_refs]
    if missing:
        raise RuntimeError(f"missing TIP4P n={missing} in data/tip4p_ccd.csv")
    for n in range(8, 22):
        ref_val, ref_txt = tip4p_refs[n]
        cells.append(
            CellSpec(f"tip4p{n}", "tip4p", "h2h_seed", TIP4P_BUDGET, ref_val, ref_txt, n=n)
        )
    return cells


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return sha256_bytes(raw.encode("utf-8"))
    raise RuntimeError("cannot read a machine id")


def new_session(machine_id: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return sha256_bytes(f"{machine_id}\n{stamp}".encode("utf-8"))


def float_bits_hex(value: float) -> str:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{raw:016x}"


def is_hit(best: float, reference: float, tol: float = HIT_TOL) -> bool:
    return math.isfinite(best) and best <= reference + tol


def arm_order(seed: int) -> tuple[str, str, str]:
    """Rotate (A, A', B) by the seed offset, and reverse odd offsets."""
    offset = seed - SEEDS[0]
    base = list(ARMS)
    shift = offset % 3
    rotated = base[shift:] + base[:shift]
    if offset % 2 == 1:
        rotated.reverse()
    return (rotated[0], rotated[1], rotated[2])


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(SCRUB_PREFIXES) or "SEAT" in key:
            del env[key]
    for key in THREAD_ENV_KEYS:
        env[key] = "1"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    env["PYTHONHASHSEED"] = "0"
    return env


def verify_core0() -> str:
    aff = sorted(os.sched_getaffinity(0))
    if aff != [0]:
        raise RuntimeError(f"process affinity is {aff}, expected [0]")
    return "0"


def type7_quantile(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lo = int(h)
    frac = h - lo
    idx = lo - 1
    if frac == 0.0:
        return ordered[idx]
    left = ordered[idx]
    right = ordered[idx + 1]
    if math.isinf(left) or math.isinf(right):
        if math.isinf(left) and math.isinf(right):
            return left
        if frac == 0.0:
            return left
        if math.isinf(right) and not math.isinf(left):
            return math.inf
        return left
    return left * (1.0 - frac) + right * frac


def finite_ratio(num: float, den: float) -> float:
    """Ratio of two medians. Two infinities count as 1."""
    if math.isinf(num) and math.isinf(den):
        return 1.0
    if math.isinf(den) and not math.isinf(num):
        return 0.0
    if den == 0.0 or math.isinf(num):
        return math.inf
    return num / den


def fisher_one_sided_fewer(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Exact one-sided Fisher p that B has fewer hits than the margin allows.

    Hypergeometric: population n_a + n_b, hits_a + hits_b marked, draw n_b.
    The p-value is P(X <= hits_b).
    """
    population = n_a + n_b
    marked = hits_a + hits_b
    draw = n_b
    if population == 0:
        return 1.0
    denom = math.comb(population, draw)

    def mass(k: int) -> float:
        other = draw - k
        if k < 0 or other < 0 or k > marked or other > population - marked:
            return 0.0
        return math.comb(marked, k) * math.comb(population - marked, other) / denom

    total = 0.0
    for k in range(0, hits_b + 1):
        total += mass(k)
    return min(1.0, total)


def mann_whitney_greater(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """One-sided Mann-Whitney p that ys are larger than xs.

    Tie-corrected normal approximation with the continuity correction.
    Returns (p, u_y). A larger u_y means ys tend to be larger.
    """
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    pooled = sorted(xs + ys)
    ranks: dict[float, float] = {}
    tie_sum = 0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        t = j - i + 1
        tie_sum += t**3 - t
        ranks[pooled[i]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_y = sum(ranks[v] for v in ys)
    u_y = r_y - n2 * (n2 + 1) / 2.0
    n = n1 + n2
    var = (n1 * n2 / 12.0) * (n + 1.0 - tie_sum / (n * (n - 1.0))) if n > 1 else 0.0
    if var <= 0.0:
        return 1.0, u_y
    mu = n1 * n2 / 2.0
    z = (u_y - mu - 0.5) / math.sqrt(var)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return min(1.0, max(0.0, p)), u_y


def calls_for_test(row: dict[str, str]) -> float:
    if row["hit"] != "1":
        return math.inf
    return float(row["calls_to_hit"])


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle))
        if existing_header != fields:
            raise RuntimeError(f"{path} header mismatch")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({k: str(row[k]) for k in fields})
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_csv_driver_row(stdout: str, system: str, seed: int, budget: int) -> dict[str, str]:
    headers = [line for line in stdout.splitlines() if line.startswith("system,")]
    if not headers:
        raise RuntimeError(f"{system} seed {seed}: missing CSV header in stdout")
    header = headers[-1].split(",")
    prefix = f"{system},anneal,{seed},"
    matches = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected 1 CSV row, got {len(matches)}")
    row = dict(zip(header, matches[0].split(","), strict=True))
    if int(float(row["budget"])) != budget:
        raise RuntimeError(f"{system} seed {seed}: budget {row['budget']} != {budget}")
    wall_s = float(row["wall_s"])
    pot_s = float(row["pot_s"])
    outside_s = float(row["outside_s"]) if "outside_s" in row else max(wall_s - pot_s, 0.0)
    return {
        "best": row["best"].strip(),
        "driver_hit": "1" if row["hit"].strip().lower() in ("1", "true") else "0",
        "calls": str(int(float(row["evals_to_hit"]))),
        "charged": str(int(float(row["charged"]))),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{outside_s:.6f}",
    }


def parse_h2h_seed_stdout(stdout: str, system: str, seed: int, budget: int) -> dict[str, str]:
    rows = [line for line in stdout.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected 1 h2h_row, got {len(rows)}")
    kv = dict(tok.split("=", 1) for tok in rows[0].split()[1:])
    if kv["system"] != system or int(kv["seed"]) != seed or int(kv["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: malformed h2h_row {rows[0]}")
    return {
        "best": kv["best"].strip(),
        "driver_hit": kv["hit"].strip(),
        "calls": str(int(kv["evals_to_hit"])),
        "charged": str(int(kv["charged"])),
        "wall_s": f"{float(kv['wall_s']):.6f}",
        "pot_s": f"{float(kv['pot_s']):.6f}",
        "outside_s": f"{float(kv['outside_s']):.6f}",
    }


def verify_driver_budgets(cells: list[CellSpec], bins: dict[str, Path], env: dict[str, str]) -> None:
    checked: set[tuple[str, str]] = set()
    for cell in cells:
        if cell.driver == "h2h_seed":
            continue
        for arm in ARMS:
            key = (arm, cell.system)
            if key in checked:
                continue
            exe = bins[arm] / cell.driver
            proc = subprocess.run(
                ["taskset", "-c", "0", str(exe), "budget", cell.system],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"budget probe failed for {arm} {cell.system}: {proc.stderr.strip()}"
                )
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            got_budget = int(lines[0])
            got_ref = float(lines[1])
            if got_budget != cell.budget:
                raise RuntimeError(f"{arm} {cell.system}: budget {got_budget} != {cell.budget}")
            if abs(got_ref - cell.reference) > 1e-6:
                raise RuntimeError(f"{arm} {cell.system}: ref {got_ref} != {cell.reference}")
            checked.add(key)


def miss_row(
    *,
    cell: CellSpec,
    arm: str,
    seed: int,
    slot: int,
    bin_sha256: str,
    lib_tree: str,
    machine_id: str,
    session: str,
    sequence: int,
    affinity: str,
    wall_s: float,
    charged: str = "0",
    best: str = "nan",
) -> dict[str, str]:
    best_val = float(best) if best != "nan" else float("nan")
    return {
        "cell": cell.system,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "hit": "0",
        "calls_to_hit": "inf",
        "charged": charged,
        "best": best,
        "best_bits": float_bits_hex(best_val),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": "0.000000",
        "outside_s": f"{wall_s:.6f}",
        "library_tree": lib_tree,
        "binary_sha256": bin_sha256,
        "core": "0",
        "affinity": affinity,
        "machine_id_sha256": machine_id,
        "session": session,
        "sequence": str(sequence),
    }


def run_one(
    *,
    cell: CellSpec,
    arm: str,
    seed: int,
    slot: int,
    exe: Path,
    bin_sha256: str,
    lib_tree: str,
    work_dir: Path,
    log_path: Path,
    machine_id: str,
    session: str,
    sequence: int,
    affinity: str,
    env: dict[str, str],
    timeout_s: int,
) -> tuple[dict[str, str], str]:
    run_dir = work_dir / cell.family / cell.system / f"{seed}-{arm.replace(chr(39), 'p')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if cell.driver == "h2h_seed":
        prefix = run_dir / f"{cell.system}-{arm.replace(chr(39), 'p')}-{seed}"
        stop_arg = repr(cell.reference + HIT_TOL)
        cmd = [
            "taskset",
            "-c",
            "0",
            str(exe),
            "tip4p",
            str(cell.n),
            "0",
            str(cell.budget),
            str(seed),
            "recommended",
            stop_arg,
            str(prefix),
        ]
    else:
        cmd = ["taskset", "-c", "0", str(exe), "run", cell.system, str(seed)]

    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = -1
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
    elapsed = time.monotonic() - t0

    log_path.write_text(
        f"cmd={' '.join(cmd)}\nrc={rc} timed_out={timed_out} elapsed_s={elapsed:.6f}\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )

    start_sha = ""
    if cell.driver == "h2h_seed":
        start_file = Path(str(prefix) + ".start.gmin")
        if start_file.is_file():
            start_sha = sha256_file(start_file)

    if timed_out or rc != 0:
        return (
            miss_row(
                cell=cell,
                arm=arm,
                seed=seed,
                slot=slot,
                bin_sha256=bin_sha256,
                lib_tree=lib_tree,
                machine_id=machine_id,
                session=session,
                sequence=sequence,
                affinity=affinity,
                wall_s=elapsed,
            ),
            start_sha,
        )

    if cell.driver == "h2h_seed":
        parsed = parse_h2h_seed_stdout(stdout, cell.system, seed, cell.budget)
    else:
        parsed = parse_csv_driver_row(stdout, cell.system, seed, cell.budget)

    best_val = float(parsed["best"])
    hit_val = 1 if is_hit(best_val, cell.reference, HIT_TOL) else 0
    if str(hit_val) != parsed["driver_hit"] and math.isfinite(best_val):
        if abs(best_val - (cell.reference + HIT_TOL)) <= 1e-7:
            hit_val = int(parsed["driver_hit"])
        else:
            raise RuntimeError(
                f"{cell.system} {arm} seed {seed}: hit mismatch best={parsed['best']} "
                f"vs driver_hit={parsed['driver_hit']} (ref={cell.ref_text})"
            )

    calls = parsed["calls"] if hit_val else "inf"
    row = {
        "cell": cell.system,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "hit": str(hit_val),
        "calls_to_hit": calls,
        "charged": parsed["charged"],
        "best": parsed["best"],
        "best_bits": float_bits_hex(best_val),
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
        "library_tree": lib_tree,
        "binary_sha256": bin_sha256,
        "core": "0",
        "affinity": affinity,
        "machine_id_sha256": machine_id,
        "session": session,
        "sequence": str(sequence),
    }
    return row, start_sha


def fmt_num(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def score_cell(cell: CellSpec, by_arm: dict[str, list[dict[str, str]]]) -> dict[str, str]:
    rows = {}
    for arm, key in (("A", "a"), ("A'", "ap"), ("B", "b")):
        got = sorted(by_arm.get(arm, []), key=lambda r: int(r["seed"]))
        if len(got) != len(SEEDS) or [int(r["seed"]) for r in got] != list(SEEDS):
            raise RuntimeError(f"{cell.system} {arm}: expected seeds {SEEDS[0]}..{SEEDS[-1]}")
        rows[key] = got
    hits = {key: sum(int(r["hit"]) for r in got) for key, got in rows.items()}
    fisher_p = fisher_one_sided_fewer(hits["a"], len(SEEDS), hits["b"], len(SEEDS))
    calls = {key: [calls_for_test(r) for r in got] for key, got in rows.items()}
    med_calls = {key: type7_quantile(values, 0.5) for key, values in calls.items()}
    mw_p, _u_b = mann_whitney_greater(calls["a"], calls["b"])
    ratio_ba = finite_ratio(med_calls["b"], med_calls["a"])
    ratio_apa = finite_ratio(med_calls["ap"], med_calls["a"])
    band = max(1.0, ratio_apa)

    def outside_ratios(got: list[dict[str, str]]) -> list[float]:
        out = []
        for row in got:
            charged = int(row["charged"])
            out.append(float(row["outside_s"]) / max(1, charged))
        return out

    opc = {key: type7_quantile(outside_ratios(got), 0.5) for key, got in rows.items()}
    outside_ba = finite_ratio(opc["b"], opc["a"])
    outside_apa = finite_ratio(opc["ap"], opc["a"])

    fail_a = hits["b"] < hits["a"] and fisher_p < ALPHA
    fail_b = mw_p < ALPHA and ratio_ba > band
    aa_outside_ok = AA_OUTSIDE_LO <= outside_apa <= AA_OUTSIDE_HI
    fail_c = outside_ba > OUTSIDE_RATIO_LIMIT and aa_outside_ok
    hit_loss = hits["b"] < hits["a"]
    verdict = "FAIL" if (fail_a or fail_b or fail_c) else "PASS"
    return {
        "cell": cell.system,
        "family": cell.family,
        "budget": str(cell.budget),
        "reference": cell.ref_text,
        "hits_a": str(hits["a"]),
        "hits_ap": str(hits["ap"]),
        "hits_b": str(hits["b"]),
        "fisher_p": f"{fisher_p:.6g}",
        "med_calls_a": fmt_num(med_calls["a"]),
        "med_calls_ap": fmt_num(med_calls["ap"]),
        "med_calls_b": fmt_num(med_calls["b"]),
        "mw_p": f"{mw_p:.6g}",
        "calls_ratio_ba": fmt_num(ratio_ba),
        "calls_ratio_apa": fmt_num(ratio_apa),
        "outside_ba": fmt_num(outside_ba),
        "outside_apa": fmt_num(outside_apa),
        "hit_loss": "1" if hit_loss else "0",
        "fail_a": "1" if fail_a else "0",
        "fail_b": "1" if fail_b else "0",
        "fail_c": "1" if fail_c else "0",
        "verdict": verdict,
    }


def claim_lines(scored: list[dict[str, str]]) -> list[str]:
    by_cell = {row["cell"]: row for row in scored}
    lines = []
    for cell, hits_a, hits_b, kind in CLAIMS:
        got = by_cell.get(cell)
        if got is None:
            lines.append(f"{cell}: missing")
            continue
        measured = (int(got["hits_a"]), int(got["hits_b"]))
        confirmed = measured == (hits_a, hits_b)
        lines.append(
            f"{cell} {kind} claimed {hits_a}->{hits_b} measured "
            f"{measured[0]}->{measured[1]} {'confirmed' if confirmed else 'not confirmed'}"
        )
    over = [row["cell"] for row in scored if float(row["outside_ba"]) > OUTSIDE_RATIO_LIMIT]
    most = len(over) > len(scored) / 2.0
    lines.append(
        f"outside B/A > 1.10 on {len(over)}/{len(scored)} cells "
        f"({'confirmed' if most else 'not confirmed'} as most): {', '.join(over) if over else 'none'}"
    )
    return lines


def summarize(rows_path: Path, summary_path: Path, repo_root: Path) -> int:
    cells = build_cells(repo_root)
    rows = read_rows(rows_path)
    by_cell_arm: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_cell_arm.setdefault((row["cell"], row["arm"]), []).append(row)
    scored = []
    for cell in cells:
        scored.append(
            score_cell(
                cell,
                {arm: by_cell_arm.get((cell.system, arm), []) for arm in ARMS},
            )
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(scored)
    fails = [row["cell"] for row in scored if row["verdict"] == "FAIL"]
    losses = [row["cell"] for row in scored if row["hit_loss"] == "1"]
    print(f"verdict={'FAIL' if fails else 'PASS'} failing={len(fails)} rows={len(rows)}")
    print("fails: " + (", ".join(fails) if fails else "none"))
    print("hit_losses: " + (", ".join(losses) if losses else "none"))
    for line in claim_lines(scored):
        print(line)
    for row in scored:
        print(
            f"{row['cell']} hits {row['hits_a']}/{row['hits_ap']}/{row['hits_b']} "
            f"fisher {row['fisher_p']} med {row['med_calls_a']}/{row['med_calls_ap']}/{row['med_calls_b']} "
            f"mw {row['mw_p']} outside {row['outside_ba']} {row['outside_apa']} {row['verdict']}"
        )
    return 0


def self_test() -> None:
    assert abs(type7_quantile([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) < 1e-12
    assert math.isinf(type7_quantile([1.0, 2.0] + [math.inf] * 6, 0.5))
    assert abs(fisher_one_sided_fewer(12, 12, 12, 12) - 1.0) < 1e-12
    # 12 vs 9 is one table only, C(21, 9) / C(24, 12).
    expected = math.comb(21, 9) / math.comb(24, 12)
    assert abs(fisher_one_sided_fewer(12, 12, 9, 12) - expected) < 1e-12
    assert fisher_one_sided_fewer(12, 12, 9, 12) > ALPHA
    p_eq, u_eq = mann_whitney_greater([100.0] * 12, [100.0] * 12)
    assert p_eq > 0.5 and abs(u_eq - 72.0) < 1e-9
    p_sep, u_sep = mann_whitney_greater([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
    assert u_sep == 9.0 and p_sep < 0.1
    p_inf, _ = mann_whitney_greater([1.0] * 12, [math.inf] * 12)
    assert p_inf < 0.01
    assert is_hit(-305.5174, -305.5183221676)
    assert not is_hit(float("nan"), -1.0)
    counts = {(slot, arm): 0 for slot in (1, 2, 3) for arm in ARMS}
    for seed in SEEDS:
        order = arm_order(seed)
        assert len(order) == len(set(order)) == 3
        for slot, arm in enumerate(order, start=1):
            counts[(slot, arm)] += 1
        if (seed - SEEDS[0]) % 2 == 1:
            shift = (seed - SEEDS[0]) % 3
            rotated = list(ARMS)[shift:] + list(ARMS)[:shift]
            assert list(order) == list(reversed(rotated))
    assert all(value == 4 for value in counts.values()), counts
    # Seed 2000 is A, A', B. Seed 2001 reverses the k=1 rotation.
    assert arm_order(2000) == ("A", "A'", "B")
    assert arm_order(2001) == ("A", "B", "A'")
    repo_root = Path(__file__).resolve().parents[2]
    cells = build_cells(repo_root)
    names = [cell.system for cell in cells]
    assert names[:5] == ["lj38", "lj75", "lj98", "lj102", "lj104"]
    assert "blj105n38" in names and "sc126-38" in names and "tip4p8" in names and "tip4p21" in names
    assert len([cell for cell in cells if cell.family == "tip4p"]) == 14
    assert len(cells) == 5 + 4 + 3 + 2 + 14
    print("self-test ok")


def main(argv: list[str]) -> int:
    if "--quick" in argv:
        raise SystemExit("--quick is prohibited")
    if len(argv) >= 2 and argv[1] == "--self-test":
        self_test()
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--a-bin", type=Path, required=True)
    run_p.add_argument("--ap-bin", type=Path, required=True)
    run_p.add_argument("--b-bin", type=Path, required=True)
    run_p.add_argument("--a-tree", required=True)
    run_p.add_argument("--ap-tree", required=True)
    run_p.add_argument("--b-tree", required=True)
    run_p.add_argument("--rows", type=Path, required=True)
    run_p.add_argument("--work", type=Path, required=True)
    run_p.add_argument("--logs", type=Path, required=True)
    run_p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    run_p.add_argument("--cells", default="", help="comma-separated subset, default all")

    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("--rows", type=Path, required=True)
    sum_p.add_argument("--summary", type=Path, required=True)

    args = parser.parse_args(argv[1:])
    repo_root = Path(__file__).resolve().parents[2]

    if args.cmd == "summarize":
        return summarize(args.rows, args.summary, repo_root)

    affinity = verify_core0()
    env = child_env()
    cells = build_cells(repo_root)
    if args.cells:
        wanted = {part.strip() for part in args.cells.split(",") if part.strip()}
        cells = [cell for cell in cells if cell.system in wanted]
        unknown = wanted - {cell.system for cell in cells}
        if unknown:
            raise SystemExit(f"unknown cells: {sorted(unknown)}")
    bins = {"A": args.a_bin.resolve(), "A'": args.ap_bin.resolve(), "B": args.b_bin.resolve()}
    trees = {"A": args.a_tree, "A'": args.ap_tree, "B": args.b_tree}
    verify_driver_budgets(cells, bins, env)

    machine_id = hashed_machine_id()
    existing = read_rows(args.rows)
    if existing:
        session = existing[0]["session"]
        sequence = max(int(row["sequence"]) for row in existing)
    else:
        session = new_session(machine_id)
        sequence = 0
    done = {(row["cell"], row["arm"], int(row["seed"])) for row in existing}

    drivers = ("h2h_gmin_proof", "h2h_seed", "h2h_morse", "h2h_blj", "h2h_sc")
    bin_shas: dict[tuple[str, str], str] = {}
    for arm in ARMS:
        for drv in drivers:
            bin_shas[(arm, drv)] = sha256_file(bins[arm] / drv)

    total_jobs = len(cells) * len(SEEDS) * len(ARMS)
    completed = len(done & {(cell.system, arm, seed) for cell in cells for arm in ARMS for seed in SEEDS})

    for cell in cells:
        for seed in SEEDS:
            order = arm_order(seed)
            start_shas: dict[str, str] = {}
            for slot, arm in enumerate(order, start=1):
                if (cell.system, arm, seed) in done:
                    continue
                sequence += 1
                exe = bins[arm] / cell.driver
                row, start_sha = run_one(
                    cell=cell,
                    arm=arm,
                    seed=seed,
                    slot=slot,
                    exe=exe,
                    bin_sha256=bin_shas[(arm, cell.driver)],
                    lib_tree=trees[arm],
                    work_dir=args.work,
                    log_path=args.logs / cell.family / cell.system / f"{seed}-{arm.replace(chr(39), 'p')}.log",
                    machine_id=machine_id,
                    session=session,
                    sequence=sequence,
                    affinity=affinity,
                    env=env,
                    timeout_s=args.timeout,
                )
                if start_sha:
                    start_shas[arm] = start_sha
                append_csv(args.rows, ROW_FIELDS, row)
                done.add((cell.system, arm, seed))
                completed += 1
                utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                print(
                    f"[{utc}] {completed}/{total_jobs} {cell.system} seed={seed} slot={slot} "
                    f"arm={arm} hit={row['hit']} calls={row['calls_to_hit']} "
                    f"best={row['best']} wall={row['wall_s']}",
                    flush=True,
                )
            if cell.driver == "h2h_seed" and len(start_shas) == 3:
                hashes = set(start_shas.values())
                if len(hashes) != 1:
                    raise RuntimeError(f"{cell.system} seed {seed}: tip4p start mismatch {start_shas}")

    print(f"campaign complete: {completed}/{total_jobs} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
