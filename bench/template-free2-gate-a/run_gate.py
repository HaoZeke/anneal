#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Three-arm serial gate for template-free morphology escapes, shard a.

Arms are A and A' (two clean release builds of one library tree) and B
(that tree plus the morphology-escape patch). One process, one core,
seeds 2000-2011. The declared fail rule is in the README beside this
file. This script records rows and, after every cell is complete,
writes the summary. It does not rewrite the README.
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
OUTSIDE_FAIL = 1.10
AA_OUTSIDE_LO = 0.95
AA_OUTSIDE_HI = 1.05
ALPHA = 0.05
SEEDS = tuple(range(2000, 2012))
ARMS = ("A", "A'", "B")
DEFAULT_TIMEOUT_S = 7200
TREE_A = "cff796c507a8fa1d0bfb28888c3570b61988055e"
TREE_B = "14b8458d118153d7a1179325058f8d09aea2d65d"

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
    "morph_activations",
]

SUMMARY_FIELDS = [
    "kind",
    "cell",
    "hits_a",
    "hits_ap",
    "hits_b",
    "fisher_p",
    "hit_loss",
    "calls_med_a",
    "calls_med_ap",
    "calls_med_b",
    "calls_ratio_ba",
    "calls_ratio_apa",
    "mw_p",
    "outside_med_a",
    "outside_med_ap",
    "outside_med_b",
    "outside_ratio_ba",
    "outside_ratio_apa",
    "morph_sum_a",
    "morph_sum_ap",
    "morph_sum_b",
    "morph_rows_b",
    "verdict",
    "detail",
]

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DOMAIN_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "TBB_NUM_THREADS": "1",
}

SCRUB_PREFIXES = ("H2H_", "ANNEAL_", "TWO_PHASE_", "BANK_", "GIT_")


@dataclass(frozen=True)
class Cell:
    name: str
    family: str
    driver: str
    budget: int
    reference: float
    ref_text: str


def cells() -> list[Cell]:
    morse = (
        ("m38r6", 400_000, -157.477108),
        ("m38r14", 400_000, -144.321054),
        ("m55r6", 1_000_000, -250.286609),
        ("m55r14", 1_000_000, -220.646208),
        ("m80r6", 4_000_000, -378.333471),
        ("m80r14", 4_000_000, -340.811371),
    )
    lj = (
        ("lj38", 400_000, -173.928427),
        ("lj55", 1_000_000, -279.248470),
        ("lj75", 4_000_000, -397.492331),
        ("lj98", 4_000_000, -543.665361),
        ("lj102", 4_000_000, -569.363652),
        ("lj103", 4_000_000, -575.766131),
        ("lj104", 4_000_000, -582.086642),
        ("lj150", 4_000_000, -893.310258),
    )
    sc = (
        ("sc126-38", 400_000, -35419.9804),
        ("sc126-55", 1_000_000, -52884.6806),
    )
    out: list[Cell] = []
    for name, budget, ref in morse:
        out.append(Cell(name, "morse", "h2h_morse", budget, ref, f"{ref:.6f}"))
    for name, budget, ref in sc:
        out.append(Cell(name, "sc", "h2h_sc", budget, ref, f"{ref:.4f}"))
    for name, budget, ref in lj:
        out.append(Cell(name, "lj", "h2h_gmin_proof", budget, ref, f"{ref:.6f}"))
    return out


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
    raise RuntimeError("machine id file is missing")


def float_bits_hex(value: float) -> str:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{raw:016x}"


def is_hit(best: float, reference: float) -> bool:
    return math.isfinite(best) and best <= reference + HIT_TOL


def arm_order(seed: int) -> tuple[str, str, str]:
    """Rotate (A, A', B) by pairs, and reverse the odd seed of each pair."""
    k = seed - SEEDS[0]
    rot = (k // 2) % 3
    order = ARMS[rot:] + ARMS[:rot]
    if k % 2 == 1:
        order = tuple(reversed(order))
    return order


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(SCRUB_PREFIXES):
            del env[key]
    env.pop("MORPH_ACTIVATIONS", None)
    for key, value in THREAD_ENV.items():
        env[key] = value
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    env["PYTHONHASHSEED"] = "0"
    # Opt-in post-search trace. The drivers read it only after the
    # search has returned and the clocks have been stored.
    env["MORPH_ACTIVATIONS"] = "1"
    return env


def verify_core0() -> str:
    aff = sorted(os.sched_getaffinity(0))
    if aff != [0]:
        raise RuntimeError(f"process affinity is {aff}, expected [0]")
    return "0"


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lo = math.floor(h)
    frac = h - lo
    idx = int(lo) - 1
    if frac == 0.0:
        return ordered[idx]
    left = ordered[idx]
    right = ordered[idx + 1]
    if math.isinf(left) or math.isinf(right):
        if math.isinf(left) and math.isinf(right):
            return left
        return math.inf
    return left * (1.0 - frac) + right * frac


def ratio(num: float, den: float) -> float:
    if math.isinf(num) and math.isinf(den):
        return 1.0
    if math.isinf(den):
        return 0.0
    if math.isinf(num):
        return math.inf
    if den == 0.0:
        return math.inf if num > 0.0 else 1.0
    return num / den


def fisher_b_fewer(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Exact one-sided Fisher: probability of hits_B this low or lower."""
    total_n = n_a + n_b
    successes = hits_a + hits_b
    draw = n_b
    if total_n == 0 or draw == 0:
        return 1.0
    denom = math.comb(total_n, draw)
    if denom == 0:
        return 1.0
    acc = 0.0
    for k in range(0, hits_b + 1):
        if k > successes or draw - k > total_n - successes:
            continue
        acc += math.comb(successes, k) * math.comb(total_n - successes, draw - k) / denom
    return min(1.0, acc)


def mann_whitney_b_larger(xs: list[float], ys: list[float]) -> float:
    """One-sided p that B (ys) is larger. Tie-corrected normal approximation."""
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return float("nan")
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
    r_b = sum(ranks[v] for v in ys)
    u_b = r_b - n2 * (n2 + 1) / 2.0
    n = n1 + n2
    var = (n1 * n2 / 12.0) * (n + 1.0 - tie_sum / (n * (n - 1.0))) if n > 1 else 0.0
    if var <= 0.0:
        return 1.0
    mu = n1 * n2 / 2.0
    z = (u_b - mu - 0.5) / math.sqrt(var)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return min(1.0, max(0.0, p))


def calls_for_test(row: dict[str, str]) -> float:
    if row["hit"] != "1":
        return math.inf
    return float(row["calls_to_hit"])


def morph_value(row: dict[str, str]) -> int | None:
    text = row.get("morph_activations", "").strip()
    if text == "":
        return None
    return int(text)


def fmt_num(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.10g}"


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="", encoding="utf-8") as handle:
            existing = next(csv.reader(handle))
        if existing != fields:
            raise RuntimeError(f"{path} header mismatch")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fields})
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise RuntimeError(f"{path} header mismatch")
        rows = []
        for row in reader:
            if any(row.get(key) in (None, "") for key in ("cell", "arm", "seed", "hit")):
                continue
            rows.append(row)
    return rows


def parse_driver_row(stdout: str, cell: Cell, seed: int) -> dict[str, str]:
    headers = [line for line in stdout.splitlines() if line.startswith("system,")]
    if not headers:
        raise RuntimeError(f"{cell.name} seed {seed}: missing CSV header")
    header = headers[-1].split(",")
    prefix = f"{cell.name},anneal,{seed},"
    matches = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"{cell.name} seed {seed}: expected 1 CSV row, got {len(matches)}")
    parts = matches[0].split(",")
    if len(parts) != len(header):
        raise RuntimeError(f"{cell.name} seed {seed}: CSV width {len(parts)} != {len(header)}")
    row = dict(zip(header, parts))
    if int(float(row["budget"])) != cell.budget:
        raise RuntimeError(f"{cell.name} seed {seed}: budget {row['budget']} != {cell.budget}")
    wall_s = float(row["wall_s"])
    pot_s = float(row["pot_s"])
    if "outside_s" in row and row["outside_s"] != "":
        outside_s = float(row["outside_s"])
    else:
        outside_s = max(wall_s - pot_s, 0.0)
    hit_text = row["hit"].strip().lower()
    return {
        "best": row["best"].strip(),
        "driver_hit": "1" if hit_text in ("1", "true") else "0",
        "calls": str(int(float(row["evals_to_hit"]))),
        "charged": str(int(float(row["charged"]))),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{outside_s:.6f}",
    }


def parse_morph(stderr: str, cell: Cell, seed: int) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.startswith("morph_activations ")]
    if len(lines) != 1:
        raise RuntimeError(f"{cell.name} seed {seed}: expected 1 morph line, got {len(lines)}")
    text = lines[0].split()
    if len(text) != 2:
        raise RuntimeError(f"{cell.name} seed {seed}: bad morph line {lines[0]}")
    return str(int(text[1]))


def verify_budgets(specs: list[Cell], bins: dict[str, Path], env: dict[str, str]) -> None:
    for cell in specs:
        for arm in ARMS:
            exe = bins[arm] / cell.driver
            proc = subprocess.run(
                ["taskset", "-c", "0", str(exe), "budget", cell.name],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"budget probe failed for {arm} {cell.name}: {proc.stderr}")
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if len(lines) < 2:
                raise RuntimeError(f"{arm} {cell.name}: budget probe returned {lines}")
            got_budget = int(lines[0])
            got_ref = float(lines[1])
            if got_budget != cell.budget:
                raise RuntimeError(f"{arm} {cell.name}: budget {got_budget} != {cell.budget}")
            if abs(got_ref - cell.reference) > 1e-6:
                raise RuntimeError(f"{arm} {cell.name}: ref {got_ref} != {cell.reference}")


def load_session(work: Path, machine_id: str) -> str:
    path = work / "session.txt"
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    token = sha256_bytes(f"{machine_id}|{stamp}|template-free2-gate-a".encode("utf-8"))[:16]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    return token


def blank_row(
    *,
    cell: Cell,
    arm: str,
    seed: int,
    slot: int,
    wall_s: float,
    lib_tree: str,
    bin_sha: str,
    machine_id: str,
    session: str,
    sequence: int,
    affinity: str,
) -> dict[str, str]:
    return {
        "cell": cell.name,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "hit": "0",
        "calls_to_hit": str(cell.budget),
        "charged": "0",
        "best": "nan",
        "best_bits": float_bits_hex(float("nan")),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": "0.000000",
        "outside_s": f"{wall_s:.6f}",
        "library_tree": lib_tree,
        "binary_sha256": bin_sha,
        "core": "0",
        "affinity": affinity,
        "machine_id_sha256": machine_id,
        "session": session,
        "sequence": str(sequence),
        "morph_activations": "",
    }


def run_one(
    *,
    cell: Cell,
    arm: str,
    seed: int,
    slot: int,
    exe: Path,
    bin_sha: str,
    lib_tree: str,
    work_dir: Path,
    log_path: Path,
    machine_id: str,
    session: str,
    sequence: int,
    affinity: str,
    env: dict[str, str],
    timeout_s: int,
) -> dict[str, str]:
    run_dir = work_dir / cell.family / cell.name / f"{seed}-{arm.replace(chr(39), 'p')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["taskset", "-c", "0", str(exe), "run", cell.name, str(seed)]
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
        f"rc={rc} timed_out={timed_out} elapsed_s={elapsed:.6f}\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )
    if timed_out or rc != 0:
        return blank_row(
            cell=cell,
            arm=arm,
            seed=seed,
            slot=slot,
            wall_s=elapsed,
            lib_tree=lib_tree,
            bin_sha=bin_sha,
            machine_id=machine_id,
            session=session,
            sequence=sequence,
            affinity=affinity,
        )
    parsed = parse_driver_row(stdout, cell, seed)
    morph = parse_morph(stderr, cell, seed)
    best_val = float(parsed["best"])
    hit_val = 1 if is_hit(best_val, cell.reference) else 0
    if str(hit_val) != parsed["driver_hit"] and math.isfinite(best_val):
        if abs(best_val - (cell.reference + HIT_TOL)) <= 1e-7:
            hit_val = int(parsed["driver_hit"])
        else:
            raise RuntimeError(
                f"{cell.name} {arm} seed {seed}: hit mismatch best={parsed['best']} "
                f"driver_hit={parsed['driver_hit']}"
            )
    return {
        "cell": cell.name,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "hit": str(hit_val),
        "calls_to_hit": parsed["calls"] if hit_val else str(cell.budget),
        "charged": parsed["charged"],
        "best": parsed["best"],
        "best_bits": float_bits_hex(best_val),
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
        "library_tree": lib_tree,
        "binary_sha256": bin_sha,
        "core": "0",
        "affinity": affinity,
        "machine_id_sha256": machine_id,
        "session": session,
        "sequence": str(sequence),
        "morph_activations": morph,
    }


def outside_median(rows: list[dict[str, str]]) -> float:
    values = []
    for row in rows:
        charged = int(row["charged"])
        if charged <= 0:
            continue
        values.append(float(row["outside_s"]) / charged)
    if not values:
        return float("nan")
    return type7(values, 0.5)


def judge_cell(name: str, grouped: dict[tuple[str, str], list[dict[str, str]]]) -> dict[str, str]:
    by_arm = {}
    for arm in ARMS:
        rows = sorted(grouped.get((name, arm), []), key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise RuntimeError(f"{name} {arm}: expected seeds {SEEDS[0]}-{SEEDS[-1]}")
        by_arm[arm] = rows
    hits = {arm: sum(int(row["hit"]) for row in by_arm[arm]) for arm in ARMS}
    fisher_p = fisher_b_fewer(hits["A"], len(SEEDS), hits["B"], len(SEEDS))
    hit_loss = hits["B"] < hits["A"]
    calls = {arm: [calls_for_test(row) for row in by_arm[arm]] for arm in ARMS}
    med = {arm: type7(calls[arm], 0.5) for arm in ARMS}
    mw_p = mann_whitney_b_larger(calls["A"], calls["B"])
    ratio_ba = ratio(med["B"], med["A"])
    ratio_apa = ratio(med["A'"], med["A"])
    band_hi = max(ratio_apa, 1.0)
    fail_b = mw_p < ALPHA and ratio_ba > 1.0 and ratio_ba > band_hi
    opc = {arm: outside_median(by_arm[arm]) for arm in ARMS}
    outside_ba = ratio(opc["B"], opc["A"])
    outside_apa = ratio(opc["A'"], opc["A"])
    fail_c = (
        outside_ba > OUTSIDE_FAIL
        and AA_OUTSIDE_LO <= outside_apa <= AA_OUTSIDE_HI
    )
    fail_a = hit_loss and fisher_p < ALPHA
    reasons = []
    if fail_a:
        reasons.append(f"hits {hits['B']}<{hits['A']} Fisher p={fisher_p:.4g}")
    elif hit_loss:
        reasons.append(f"hit loss {hits['B']}<{hits['A']} not significant p={fisher_p:.4g}")
    if fail_b:
        reasons.append(f"calls worse MW p={mw_p:.4g} ratio={ratio_ba:.4g} band={band_hi:.4g}")
    if fail_c:
        reasons.append(f"outside B/A={outside_ba:.4g} A'/A={outside_apa:.4g}")
    verdict = "FAIL" if (fail_a or fail_b or fail_c) else "PASS"
    morph_sums = {}
    morph_rows_b = 0
    for arm in ARMS:
        total = 0
        for row in by_arm[arm]:
            value = morph_value(row)
            if value is None:
                continue
            total += value
            if arm == "B" and value > 0:
                morph_rows_b += 1
        morph_sums[arm] = total
    return {
        "kind": "cell",
        "cell": name,
        "hits_a": str(hits["A"]),
        "hits_ap": str(hits["A'"]),
        "hits_b": str(hits["B"]),
        "fisher_p": fmt_num(fisher_p),
        "hit_loss": "1" if hit_loss else "0",
        "calls_med_a": fmt_num(med["A"]),
        "calls_med_ap": fmt_num(med["A'"]),
        "calls_med_b": fmt_num(med["B"]),
        "calls_ratio_ba": fmt_num(ratio_ba),
        "calls_ratio_apa": fmt_num(ratio_apa),
        "mw_p": fmt_num(mw_p),
        "outside_med_a": fmt_num(opc["A"]),
        "outside_med_ap": fmt_num(opc["A'"]),
        "outside_med_b": fmt_num(opc["B"]),
        "outside_ratio_ba": fmt_num(outside_ba),
        "outside_ratio_apa": fmt_num(outside_apa),
        "morph_sum_a": str(morph_sums["A"]),
        "morph_sum_ap": str(morph_sums["A'"]),
        "morph_sum_b": str(morph_sums["B"]),
        "morph_rows_b": str(morph_rows_b),
        "verdict": verdict,
        "detail": "; ".join(reasons),
    }


def claim_row(cell: str, confirmed: bool, detail: str) -> dict[str, str]:
    row = {key: "" for key in SUMMARY_FIELDS}
    row["kind"] = "claim"
    row["cell"] = cell
    row["verdict"] = "confirmed" if confirmed else "not-confirmed"
    row["detail"] = detail
    return row


def summarize(rows_path: Path, summary_path: Path) -> int:
    specs = cells()
    rows = read_rows(rows_path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["cell"], row["arm"]), []).append(row)
    cell_rows = [judge_cell(spec.name, grouped) for spec in specs]
    by_name = {row["cell"]: row for row in cell_rows}
    m38 = by_name["m38r14"]
    sc38 = by_name["sc126-38"]
    m80 = by_name["m80r6"]
    claim_hits = int(m38["hits_a"]) == 4 and int(m38["hits_b"]) == 12
    # Compare the gate medians. "inf" is worse than any finite median.
    def as_float(text: str) -> float:
        if text == "inf":
            return math.inf
        if text == "-inf":
            return -math.inf
        return float(text)

    sc_better = as_float(sc38["calls_med_b"]) < as_float(sc38["calls_med_a"])
    claim_m80 = int(m80["hits_a"]) == 7 and int(m80["hits_b"]) == 9
    losses = []
    for row in cell_rows:
        if row["hit_loss"] == "1":
            losses.append(f"{row['cell']} hits {row['hits_b']}<{row['hits_a']}")
        if as_float(row["calls_med_b"]) > as_float(row["calls_med_a"]):
            losses.append(f"{row['cell']} calls median")
        if row["verdict"] == "FAIL":
            losses.append(f"{row['cell']} {row['verdict']}")
    # A cell can be listed twice if it both loses hits and fails. Keep each once.
    seen = []
    for item in losses:
        if item not in seen:
            seen.append(item)
    nothing_loses = not any(row["hit_loss"] == "1" or row["verdict"] == "FAIL" for row in cell_rows)
    # The claim is about losses, not about a median that moves the wrong way
    # inside the noise band. A FAIL or a hit drop falsifies it.
    claims = [
        claim_row(
            "m38r14-4-of-12-to-12-of-12",
            claim_hits,
            f"A {m38['hits_a']}/12 A' {m38['hits_ap']}/12 B {m38['hits_b']}/12",
        ),
        claim_row(
            "sc126-38-median-improves",
            sc_better,
            f"median calls A {sc38['calls_med_a']} A' {sc38['calls_med_ap']} B {sc38['calls_med_b']}",
        ),
        claim_row(
            "m80r6-7-to-9",
            claim_m80,
            f"A {m80['hits_a']}/12 A' {m80['hits_ap']}/12 B {m80['hits_b']}/12",
        ),
        claim_row(
            "nothing-loses",
            nothing_loses,
            "no hit drop and no cell FAIL" if nothing_loses else "; ".join(seen),
        ),
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in cell_rows + claims:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    failed = [row["cell"] for row in cell_rows if row["verdict"] == "FAIL"]
    print(f"verdict={'FAIL' if failed else 'PASS'} failing={len(failed)}", flush=True)
    for row in cell_rows:
        print(
            f"{row['cell']} hits {row['hits_a']}/{row['hits_ap']}/{row['hits_b']} "
            f"fisher {row['fisher_p']} calls {row['calls_med_a']}/{row['calls_med_ap']}/{row['calls_med_b']} "
            f"mw {row['mw_p']} outside {row['outside_ratio_ba']}/{row['outside_ratio_apa']} "
            f"morph {row['morph_sum_b']} {row['verdict']}",
            flush=True,
        )
    for row in claims:
        print(f"claim {row['cell']} {row['verdict']} {row['detail']}", flush=True)
    return 0


def self_test() -> None:
    assert abs(type7([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) < 1e-12
    assert abs(type7([10.0, 20.0, 30.0], 0.5) - 20.0) < 1e-12
    assert math.isinf(type7([1.0, 2.0, math.inf, math.inf], 0.5))
    assert fisher_b_fewer(12, 12, 12, 12) == 1.0
    assert fisher_b_fewer(0, 12, 0, 12) == 1.0
    tiny = fisher_b_fewer(12, 12, 0, 12)
    assert tiny < 1e-3
    assert fisher_b_fewer(6, 12, 6, 12) > 0.5
    p_sep = mann_whitney_b_larger([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
    assert p_sep < 0.05
    assert mann_whitney_b_larger([1.0, 1.0], [1.0, 1.0]) == 1.0
    assert ratio(math.inf, math.inf) == 1.0
    assert ratio(math.inf, 2.0) == math.inf
    assert ratio(2.0, math.inf) == 0.0
    counts = {(slot, arm): 0 for slot in (1, 2, 3) for arm in ARMS}
    for seed in SEEDS:
        order = arm_order(seed)
        assert len(order) == 3 and set(order) == set(ARMS)
        if (seed - SEEDS[0]) % 2 == 1:
            prev = arm_order(seed - 1)
            assert order == tuple(reversed(prev))
        for slot, arm in enumerate(order, start=1):
            counts[(slot, arm)] += 1
    assert all(value == 4 for value in counts.values())
    specs = cells()
    assert [spec.name for spec in specs] == [
        "m38r6",
        "m38r14",
        "m55r6",
        "m55r14",
        "m80r6",
        "m80r14",
        "sc126-38",
        "sc126-55",
        "lj38",
        "lj55",
        "lj75",
        "lj98",
        "lj102",
        "lj103",
        "lj104",
        "lj150",
    ]
    assert len(SEEDS) == 12
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
    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("--rows", type=Path, required=True)
    sum_p.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    if args.cmd == "summarize":
        return summarize(args.rows, args.summary)
    if args.a_tree != TREE_A or args.ap_tree != TREE_A:
        raise SystemExit("A and A' must both be the develop library tree")
    if args.b_tree != TREE_B:
        raise SystemExit("B library tree does not match the morphology patch")
    affinity = verify_core0()
    env = child_env()
    specs = cells()
    bins = {"A": args.a_bin.resolve(), "A'": args.ap_bin.resolve(), "B": args.b_bin.resolve()}
    trees = {"A": args.a_tree, "A'": args.ap_tree, "B": args.b_tree}
    verify_budgets(specs, bins, env)
    machine_id = hashed_machine_id()
    session = load_session(args.work, machine_id)
    bin_shas: dict[tuple[str, str], str] = {}
    for arm in ARMS:
        for driver in ("h2h_morse", "h2h_sc", "h2h_gmin_proof"):
            bin_shas[(arm, driver)] = sha256_file(bins[arm] / driver)
    existing = read_rows(args.rows)
    done = {(row["cell"], row["arm"], int(row["seed"])) for row in existing}
    sequence = 1 + max((int(row["sequence"]) for row in existing), default=0)
    total = len(specs) * len(SEEDS) * len(ARMS)
    completed = len(done)
    for cell in specs:
        for seed in SEEDS:
            order = arm_order(seed)
            for slot, arm in enumerate(order, start=1):
                if (cell.name, arm, seed) in done:
                    continue
                exe = bins[arm] / cell.driver
                row = run_one(
                    cell=cell,
                    arm=arm,
                    seed=seed,
                    slot=slot,
                    exe=exe,
                    bin_sha=bin_shas[(arm, cell.driver)],
                    lib_tree=trees[arm],
                    work_dir=args.work,
                    log_path=args.logs / cell.family / cell.name / f"{seed}-{arm.replace(chr(39), 'p')}.log",
                    machine_id=machine_id,
                    session=session,
                    sequence=sequence,
                    affinity=affinity,
                    env=env,
                    timeout_s=args.timeout,
                )
                append_csv(args.rows, ROW_FIELDS, row)
                done.add((cell.name, arm, seed))
                completed += 1
                sequence += 1
                utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                print(
                    f"[{utc}] {completed}/{total} {cell.name} seed={seed} slot={slot} arm={arm} "
                    f"hit={row['hit']} calls={row['calls_to_hit']} best={row['best']} "
                    f"morph={row['morph_activations']} wall={row['wall_s']}",
                    flush=True,
                )
    print(f"campaign complete: {completed}/{total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
