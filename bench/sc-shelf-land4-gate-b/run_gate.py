#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Paired serial A/B gate (part B) for sc-shelf-land4 against develop."""

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
ALPHA = 0.05
SEEDS = tuple(range(2000, 2012))
DEFAULT_TIMEOUT_S = 14400

ROW_FIELDS = [
    "system",
    "family",
    "arm",
    "seed",
    "slot",
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
    "machine_id_sha256",
    "library_tree",
    "harness_blob",
    "binary_sha256",
    "driver",
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

TIP4P_PAPER_BUDGET = 2_000_000

MORSE_SPECS = (
    ("m38r14", 400_000, -144.321054),
    ("m55r6", 1_000_000, -250.286609),
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

LJ_SPECS = (
    ("lj55", 1_000_000, -279.248470),
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
    for n in range(8, 22):
        if n not in tip4p_refs:
            raise RuntimeError(f"missing n={n} in data/tip4p_ccd.csv")
        ref_val, ref_txt = tip4p_refs[n]
        cells.append(CellSpec(f"tip4p{n}", "tip4p", "h2h_seed", TIP4P_PAPER_BUDGET, ref_val, ref_txt, n=n))
    for system, budget, ref in MORSE_SPECS:
        cells.append(CellSpec(system, "morse", "h2h_morse", budget, ref, f"{ref:.6f}"))
    for system, budget, ref in BLJ_SPECS:
        cells.append(CellSpec(system, "blj", "h2h_blj", budget, ref, f"{ref:.10f}"))
    return cells


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return sha256_bytes(raw.encode("utf-8"))
    raise RuntimeError("cannot read machine-id")


def float_bits_hex(value: float) -> str:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{raw:016x}"


def is_hit(best: float, reference: float, tol: float = HIT_TOL) -> bool:
    return math.isfinite(best) and best <= reference + tol


def abba_order(seed: int) -> tuple[str, str]:
    offset = seed - SEEDS[0]
    return ("A", "B") if (offset % 4 in (0, 3)) else ("B", "A")


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(SCRUB_PREFIXES):
            del env[key]
    for key in THREAD_ENV_KEYS:
        env[key] = "1"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    env["PYTHONHASHSEED"] = "0"
    env["LJOS_SEAT"] = "benchbot"
    return env


def verify_core0() -> None:
    aff = sorted(os.sched_getaffinity(0))
    if aff != [0]:
        raise RuntimeError(f"process affinity is {aff}, expected [0] (run under taskset -c 0)")


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
    return ordered[idx] * (1.0 - frac) + ordered[idx + 1] * frac


def fisher_exact_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b
    total_n = a + b + c + d
    if total_n == 0:
        return 1.0
    row1, col1 = a + b, a + c

    def prob(aa: int) -> float:
        bb = row1 - aa
        cc = col1 - aa
        dd = total_n - row1 - cc
        if min(aa, bb, cc, dd) < 0:
            return 0.0
        return math.comb(row1, aa) * math.comb(total_n - row1, cc) / math.comb(total_n, col1)

    p0 = prob(a)
    total = 0.0
    for aa in range(max(0, col1 - (total_n - row1)), min(row1, col1) + 1):
        pa = prob(aa)
        if pa <= p0 * (1.0 + 1e-12):
            total += pa
    return min(1.0, total)


def mann_whitney_two_sided(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Two-sided Mann-Whitney U test (tie- and continuity-corrected normal approx).

    Returns (p_value, u_a, u_b). Larger u_b means B tends to have larger values.
    """
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
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
    r_a = sum(ranks[v] for v in xs)
    r_b = sum(ranks[v] for v in ys)
    u_a = r_a - n1 * (n1 + 1) / 2.0
    u_b = r_b - n2 * (n2 + 1) / 2.0
    n = n1 + n2
    var = (n1 * n2 / 12.0) * (n + 1.0 - tie_sum / (n * (n - 1.0))) if n > 1 else 0.0
    if var <= 0.0:
        return 1.0, u_a, u_b
    mu = n1 * n2 / 2.0
    diff = abs(u_b - mu)
    if diff <= 0.5:
        return 1.0, u_a, u_b
    z = (diff - 0.5) / math.sqrt(var)
    p = math.erfc(z / math.sqrt(2.0))
    return min(1.0, max(0.0, p)), u_a, u_b


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
        for arm in ("A", "B"):
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
                raise RuntimeError(f"budget probe failed for {arm} {cell.system}: {proc.stderr}")
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            got_budget = int(lines[0])
            got_ref = float(lines[1])
            if got_budget != cell.budget:
                raise RuntimeError(f"{arm} {cell.system}: budget {got_budget} != {cell.budget}")
            if abs(got_ref - cell.reference) > 1e-10:
                raise RuntimeError(f"{arm} {cell.system}: ref {got_ref} != {cell.reference}")
            checked.add(key)


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
    harness_blob: str,
    env: dict[str, str],
    timeout_s: int,
) -> tuple[dict[str, str], str]:
    run_dir = work_dir / cell.family / cell.system / f"{seed}-{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if cell.driver == "h2h_seed":
        prefix = run_dir / f"{cell.system}-{arm}-{seed}"
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
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    elapsed = time.monotonic() - t0

    log_header = (
        f"cmd={' '.join(cmd)}\n"
        f"rc={rc} timed_out={timed_out} elapsed_s={elapsed:.6f}\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
    )
    log_path.write_text(log_header, encoding="utf-8")

    start_sha = ""
    if cell.driver == "h2h_seed":
        start_file = run_dir / f"{cell.system}-{arm}-{seed}.start.gmin"
        if start_file.is_file():
            start_sha = sha256_file(start_file)

    if timed_out or rc != 0:
        row = {
            "system": cell.system,
            "family": cell.family,
            "arm": arm,
            "seed": str(seed),
            "slot": str(slot),
            "budget": str(cell.budget),
            "reference": cell.ref_text,
            "best": "nan",
            "best_bits": float_bits_hex(float("nan")),
            "hit": "0",
            "calls": str(cell.budget),
            "charged": "0",
            "wall_s": f"{elapsed:.6f}",
            "pot_s": "0.000000",
            "outside_s": f"{elapsed:.6f}",
            "core": "0",
            "machine_id_sha256": machine_id,
            "library_tree": lib_tree,
            "harness_blob": harness_blob,
            "binary_sha256": bin_sha256,
            "driver": cell.driver,
        }
        return row, start_sha

    if cell.driver == "h2h_seed":
        parsed = parse_h2h_seed_stdout(stdout, cell.system, seed, cell.budget)
    else:
        parsed = parse_csv_driver_row(stdout, cell.system, seed, cell.budget)

    best_val = float(parsed["best"])
    hit_val = 1 if is_hit(best_val, cell.reference, HIT_TOL) else 0
    # Trust the driver's exact unrounded f64 hit check if formatting rounded at the boundary.
    if str(hit_val) != parsed["driver_hit"] and math.isfinite(best_val):
        if abs(best_val - (cell.reference + HIT_TOL)) <= 1e-7:
            hit_val = int(parsed["driver_hit"])
        else:
            raise RuntimeError(
                f"{cell.system} {arm} seed {seed}: hit mismatch best={parsed['best']} "
                f"vs driver_hit={parsed['driver_hit']} (ref={cell.ref_text})"
            )

    row = {
        "system": cell.system,
        "family": cell.family,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "budget": str(cell.budget),
        "reference": cell.ref_text,
        "best": parsed["best"],
        "best_bits": float_bits_hex(best_val),
        "hit": str(hit_val),
        "calls": parsed["calls"] if hit_val else str(cell.budget),
        "charged": parsed["charged"],
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
        "core": "0",
        "machine_id_sha256": machine_id,
        "library_tree": lib_tree,
        "harness_blob": harness_blob,
        "binary_sha256": bin_sha256,
        "driver": cell.driver,
    }
    return row, start_sha


def summarize(rows_path: Path, readme_path: Path, repo_root: Path) -> int:
    cells = build_cells(repo_root)
    rows = read_rows(rows_path)
    by_cell_arm: dict[tuple[str, str], list[dict[str, str]]] = {}
    slot_arm_counts = {(1, "A"): 0, (1, "B"): 0, (2, "A"): 0, (2, "B"): 0}
    trees: dict[str, set[str]] = {"A": set(), "B": set()}
    drivers: dict[tuple[str, str], set[str]] = {}
    machine_ids: set[str] = set()
    harness_blobs: set[str] = set()

    for row in rows:
        key = (row["system"], row["arm"])
        by_cell_arm.setdefault(key, []).append(row)
        slot_arm_counts[(int(row["slot"]), row["arm"])] += 1
        trees[row["arm"]].add(row["library_tree"])
        drivers.setdefault((row["arm"], row["driver"]), set()).add(row["binary_sha256"])
        machine_ids.add(row["machine_id_sha256"])
        harness_blobs.add(row["harness_blob"])

    cell_summaries: list[dict[str, object]] = []
    failing_cells: list[tuple[str, list[str]]] = []

    for cell in cells:
        rows_a = sorted(by_cell_arm.get((cell.system, "A"), []), key=lambda r: int(r["seed"]))
        rows_b = sorted(by_cell_arm.get((cell.system, "B"), []), key=lambda r: int(r["seed"]))
        if len(rows_a) != len(SEEDS) or len(rows_b) != len(SEEDS):
            raise RuntimeError(f"{cell.system}: expected 12 rows per arm, got A={len(rows_a)} B={len(rows_b)}")
        if [int(r["seed"]) for r in rows_a] != list(SEEDS) or [int(r["seed"]) for r in rows_b] != list(SEEDS):
            raise RuntimeError(f"{cell.system}: seed mismatch")

        hits_a_list = [int(r["hit"]) for r in rows_a]
        hits_b_list = [int(r["hit"]) for r in rows_b]
        hits_a = sum(hits_a_list)
        hits_b = sum(hits_b_list)
        hit_str_a = "".join(str(v) for v in hits_a_list)
        hit_str_b = "".join(str(v) for v in hits_b_list)
        fisher_p = fisher_exact_two_sided(hits_a, len(SEEDS), hits_b, len(SEEDS))

        calls_a = [float(r["calls"]) for r in rows_a]
        calls_b = [float(r["calls"]) for r in rows_b]
        calls_med_a = type7_quantile(calls_a, 0.5)
        calls_med_b = type7_quantile(calls_b, 0.5)
        mw_p, u_a, u_b = mann_whitney_two_sided(calls_a, calls_b)

        wall_med_a = type7_quantile([float(r["wall_s"]) for r in rows_a], 0.5)
        wall_med_b = type7_quantile([float(r["wall_s"]) for r in rows_b], 0.5)
        inside_med_a = type7_quantile([float(r["pot_s"]) for r in rows_a], 0.5)
        inside_med_b = type7_quantile([float(r["pot_s"]) for r in rows_b], 0.5)

        opc_a = [float(r["outside_s"]) / max(1, int(r["charged"])) for r in rows_a]
        opc_b = [float(r["outside_s"]) / max(1, int(r["charged"])) for r in rows_b]
        opc_med_a = type7_quantile(opc_a, 0.5)
        opc_med_b = type7_quantile(opc_b, 0.5)
        opc_ratio = opc_med_b / opc_med_a if opc_med_a > 0.0 else float("inf")

        reasons: list[str] = []
        if hits_b < hits_a:
            reasons.append(f"hits {hits_b}<{hits_a}")
        if mw_p < ALPHA and u_b > u_a:
            reasons.append(f"calls loss (MW p={mw_p:.4g})")
        if opc_ratio > OUTSIDE_RATIO_LIMIT:
            reasons.append(f"outside/call B/A={opc_ratio:.4f}>1.10")

        verdict = f"FIX ({'; '.join(reasons)})" if reasons else "PASS"
        if reasons:
            failing_cells.append((cell.system, reasons))

        cell_summaries.append(
            {
                "system": cell.system,
                "family": cell.family,
                "driver": cell.driver,
                "budget": cell.budget,
                "reference": cell.ref_text,
                "hits_a": hits_a,
                "hits_b": hits_b,
                "hit_str_a": hit_str_a,
                "hit_str_b": hit_str_b,
                "fisher_p": fisher_p,
                "calls_med_a": calls_med_a,
                "calls_med_b": calls_med_b,
                "mw_p": mw_p,
                "u_a": u_a,
                "u_b": u_b,
                "wall_med_a": wall_med_a,
                "wall_med_b": wall_med_b,
                "inside_med_a": inside_med_a,
                "inside_med_b": inside_med_b,
                "opc_med_a": opc_med_a,
                "opc_med_b": opc_med_b,
                "opc_ratio": opc_ratio,
                "verdict": verdict,
            }
        )

    overall_verdict = "FIX" if failing_cells else "PASS"
    tree_a = next(iter(trees["A"]))
    tree_b = next(iter(trees["B"]))

    lines: list[str] = [
        "# sc-shelf-land4 Gate (Part B)",
        "",
        f"**Verdict: {overall_verdict}**",
        "",
    ]
    if failing_cells:
        lines.append("Failing cells:")
        for sys_name, reasons in failing_cells:
            lines.append(f"- `{sys_name}`: {', '.join(reasons)}")
        lines.append("")
    else:
        lines.append("Failing cells: none.")
        lines.append("")

    lines.extend(
        [
            "## Protocol and Provenance",
            "",
            "- Arm A library tree: `" + tree_a + "` (`develop` `61fa67cf380e46876b569da45e7cc091647a775b`)",
            "- Arm B library tree: `" + tree_b + "` (Arm A + 9-commit `sc-shelf-land4` patch series)",
            f"- Cells: {len(cells)} (`tip4p8`..`tip4p21`, `blj105n38`, `blj105n55`, `blj115n38`, `blj115n55`, `m38r14`, `m55r6`, `m55r14`, `m80r6`, `m80r14`, `lj55`)",
            f"- Seeds: `{SEEDS[0]}`..`{SEEDS[-1]}` (`n = {len(SEEDS)}` per arm per cell), serial on `taskset -c 0`, single-threaded env, no `H2H_*` env set",
            "- Hit rule: finite `best <= reference + 1e-3`; misses enter calls-to-hit at `budget`",
            "- Outside time per call: computed per row as `outside_s / charged`, then Hyndman-Fan type 7 median per arm, then `B / A` ratio of medians",
            "- Gate rule: `FIX` if `hits_B < hits_A` on any cell (regardless of `p`), or `calls` loss at two-sided Mann-Whitney `p < 0.05` (misses ranked worst), or `B/A` outside-time-per-call ratio `> 1.10` on any cell; otherwise `PASS`",
            f"- Balanced ABBA `(slot, arm)` counts: `(1, A)={slot_arm_counts[(1, 'A')]}`, `(1, B)={slot_arm_counts[(1, 'B')]}`, `(2, A)={slot_arm_counts[(2, 'A')]}`, `(2, B)={slot_arm_counts[(2, 'B')]}`, total rows={len(rows)}",
            f"- Machine ID SHA-256: `{', '.join(sorted(machine_ids))}`",
            f"- Harness blob SHA-1: `{', '.join(sorted(harness_blobs))}`",
            "",
            "### Driver Binary SHA-256",
            "",
            "| Arm | Driver | SHA-256 |",
            "| --- | --- | --- |",
        ]
    )
    for (arm, drv), shas in sorted(drivers.items()):
        for sha in sorted(shas):
            lines.append(f"| {arm} | `{drv}` | `{sha}` |")

    lines.extend(
        [
            "",
            "## Per-Cell Results",
            "",
            "| System | Family | Budget | Hits A/B | Fisher p | Calls Med A/B | MW p | Wall Med A/B (s) | Inside Med A/B (s) | Out/Call Med A/B (s) | Out/Call B/A | Verdict |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for cs in cell_summaries:
        lines.append(
            f"| {cs['system']} | {cs['family']} | {cs['budget']:,} | "
            f"{cs['hits_a']}/12 vs {cs['hits_b']}/12 | {cs['fisher_p']:.4g} | "
            f"{cs['calls_med_a']:,.1f} / {cs['calls_med_b']:,.1f} | {cs['mw_p']:.4g} | "
            f"{cs['wall_med_a']:.4f} / {cs['wall_med_b']:.4f} | "
            f"{cs['inside_med_a']:.4f} / {cs['inside_med_b']:.4f} | "
            f"{cs['opc_med_a']:.3e} / {cs['opc_med_b']:.3e} | "
            f"{cs['opc_ratio']:.4f} | {cs['verdict']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Hit Strings (Seeds 2000..2011)",
            "",
            "| System | Arm A Hits (2000..2011) | Arm B Hits (2000..2011) |",
            "| --- | --- | --- |",
        ]
    )
    for cs in cell_summaries:
        lines.append(f"| {cs['system']} | `{cs['hit_str_a']}` | `{cs['hit_str_b']}` |")
    lines.append("")

    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"verdict={overall_verdict} failing={len(failing_cells)} rows={len(rows)}", flush=True)
    return 0


def self_test() -> None:
    assert abs(type7_quantile([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) < 1e-12
    assert abs(type7_quantile([10.0, 20.0, 30.0], 0.5) - 20.0) < 1e-12
    assert abs(fisher_exact_two_sided(12, 12, 12, 12) - 1.0) < 1e-12
    # 2x2 [[12, 0], [0, 12]] -> 2 * 1 / C(24, 12)
    expected_p = 2.0 / math.comb(24, 12)
    assert abs(fisher_exact_two_sided(12, 12, 0, 12) - expected_p) < 1e-15
    p_eq, ua_eq, ub_eq = mann_whitney_two_sided([100.0] * 12, [100.0] * 12)
    assert p_eq == 1.0 and ua_eq == 72.0 and ub_eq == 72.0
    p_sep, ua_sep, ub_sep = mann_whitney_two_sided([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
    assert ua_sep == 0.0 and ub_sep == 9.0 and p_sep < 0.1
    assert is_hit(-305.5174, -305.5183221676)
    assert not is_hit(float("nan"), -305.5183221676)
    assert not is_hit(float("inf"), -305.5183221676)
    assert not is_hit(-float("inf"), -305.5183221676)
    counts = {(1, "A"): 0, (1, "B"): 0, (2, "A"): 0, (2, "B"): 0}
    for seed in SEEDS:
        order = abba_order(seed)
        counts[(1, order[0])] += 1
        counts[(2, order[1])] += 1
    assert counts == {(1, "A"): 6, (1, "B"): 6, (2, "A"): 6, (2, "B"): 6}
    repo_root = Path(__file__).resolve().parents[2]
    cells = build_cells(repo_root)
    assert len(cells) == 24
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
    run_p.add_argument("--b-bin", type=Path, required=True)
    run_p.add_argument("--a-tree", required=True)
    run_p.add_argument("--b-tree", required=True)
    run_p.add_argument("--rows", type=Path, required=True)
    run_p.add_argument("--work", type=Path, required=True)
    run_p.add_argument("--logs", type=Path, required=True)
    run_p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)

    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("--rows", type=Path, required=True)
    sum_p.add_argument("--readme", type=Path, required=True)

    args = parser.parse_args(argv[1:])
    repo_root = Path(__file__).resolve().parents[2]

    if args.cmd == "summarize":
        return summarize(args.rows, args.readme, repo_root)

    verify_core0()
    env = child_env()
    cells = build_cells(repo_root)
    bins = {"A": args.a_bin.resolve(), "B": args.b_bin.resolve()}
    trees = {"A": args.a_tree, "B": args.b_tree}
    verify_driver_budgets(cells, bins, env)

    machine_id = hashed_machine_id()
    harness_blob = git_blob_sha1(Path(__file__).resolve())

    bin_shas: dict[tuple[str, str], str] = {}
    for arm in ("A", "B"):
        for drv in ("h2h_gmin_proof", "h2h_seed", "h2h_morse", "h2h_blj"):
            bin_shas[(arm, drv)] = sha256_file(bins[arm] / drv)

    existing = read_rows(args.rows)
    done = {(r["system"], r["arm"], int(r["seed"])) for r in existing}
    total_jobs = len(cells) * len(SEEDS) * 2
    completed = len(done)

    for cell in cells:
        for seed in SEEDS:
            order = abba_order(seed)
            start_shas: dict[str, str] = {}
            for slot, arm in enumerate(order, start=1):
                if (cell.system, arm, seed) in done:
                    continue
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
                    log_path=args.logs / cell.family / cell.system / f"{seed}-{arm}.log",
                    machine_id=machine_id,
                    harness_blob=harness_blob,
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
                    f"[{utc}] {completed}/{total_jobs} {cell.system} seed={seed} slot={slot} arm={arm} "
                    f"hit={row['hit']} calls={row['calls']} best={row['best']} wall={row['wall_s']}",
                    flush=True,
                )
            if len(start_shas) == 2 and start_shas["A"] != start_shas["B"]:
                raise RuntimeError(f"{cell.system} seed {seed}: tip4p start mismatch between A and B")

    print(f"campaign complete: {completed}/{total_jobs} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
