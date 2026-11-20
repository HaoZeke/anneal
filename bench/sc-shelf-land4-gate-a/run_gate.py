#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Gate A harness for sc-shelf-land4 on Config::recommended.

Executes paired A/B runs (14 cells, seeds 2100-2111, ABBA order),
A/A noise-band runs (all 14 cells, seeds 2100-2111, ABBA order), and
3-variant link-placement noise-band runs on Arm A (sc108-55, sc96-38,
lj38, m30r14; default, -Wl,--sort-section=name, unused object).

All runs execute serially on core 0 with single-thread environment
variables, writing CSV rows incrementally with fsync.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HIT_TOL = 0.001
SEEDS = tuple(range(2100, 2112))
OUTSIDE_RATIO_CAP = 1.10

A_COMMIT = "61fa67cf380e46876b569da45e7cc091647a775b"
A_TREE = "cff796c507a8fa1d0bfb28888c3570b61988055e"
B_TREE = "3381b56419aeeece37ff4b13d370967b0b2ab6d0"

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

PLACEMENT_CELLS = ("sc108-55", "sc96-38", "lj38", "m30r14")
PLACEMENT_VARIANTS = ("default", "sort_section", "unused_obj")

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

BAND_ROW_FIELDS = [
    "band",
    *ROW_FIELDS,
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


@dataclass(frozen=True)
class Cell:
    system: str
    family: str
    budget: int
    reference: float
    reference_str: str
    driver: str


CELLS: tuple[Cell, ...] = (
    Cell("lj38", "lj", 400_000, -173.928427, "-173.928427", "h2h_gmin_proof"),
    Cell("lj75", "lj", 4_000_000, -397.492331, "-397.492331", "h2h_gmin_proof"),
    Cell("lj77", "lj", 4_000_000, -409.083517, "-409.083517", "h2h_seed"),
    Cell("lj98", "lj", 4_000_000, -543.665361, "-543.665361", "h2h_gmin_proof"),
    Cell("m30r14", "morse", 400_000, -106.835790, "-106.835790", "h2h_morse"),
    Cell("sc96-38", "sutton-chen", 400_000, -8917.7056, "-8917.7056", "h2h_sc"),
    Cell("sc96-55", "sutton-chen", 1_000_000, -13217.8963, "-13217.8963", "h2h_sc"),
    Cell("sc96-80", "sutton-chen", 1_000_000, -19560.8966, "-19560.8966", "h2h_sc"),
    Cell("sc108-38", "sutton-chen", 400_000, -10117.2454, "-10117.2454", "h2h_sc"),
    Cell("sc108-55", "sutton-chen", 1_000_000, -14814.5225, "-14814.5225", "h2h_sc"),
    Cell("sc108-80", "sutton-chen", 1_000_000, -21819.9952, "-21819.9952", "h2h_sc"),
    Cell("sc126-38", "sutton-chen", 400_000, -35419.9804, "-35419.9804", "h2h_sc"),
    Cell("sc126-55", "sutton-chen", 1_000_000, -52884.6806, "-52884.6806", "h2h_sc"),
    Cell("sc126-80", "sutton-chen", 1_000_000, -78414.6271, "-78414.6271", "h2h_sc"),
)

CELL_BY_NAME = {cell.system: cell for cell in CELLS}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    raise RuntimeError("no machine-id file found")


def cpus_allowed() -> str:
    status = Path("/proc/self/status")
    if not status.is_file():
        return "unknown"
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def float_bits_hex(value: float) -> str:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return f"0x{bits:016x}"


def rederived_hit(best_str: str, reference: float, tol: float = HIT_TOL) -> int:
    try:
        val = float(best_str)
    except ValueError:
        return 0
    if not math.isfinite(val):
        return 0
    return 1 if val <= reference + tol else 0


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


def abba_pair(seed_index: int, first_label: str, second_label: str) -> tuple[str, tuple[str, str]]:
    if seed_index % 4 in (0, 3):
        tag = "AB" if (first_label, second_label) == ("A", "B") else f"{first_label}/{second_label}"
        return tag, (first_label, second_label)
    tag = "BA" if (first_label, second_label) == ("A", "B") else f"{second_label}/{first_label}"
    return tag, (second_label, first_label)


def placement_triple(seed_index: int) -> tuple[str, tuple[str, str, str]]:
    perms = (
        ("default", "sort_section", "unused_obj"),
        ("sort_section", "unused_obj", "default"),
        ("unused_obj", "default", "sort_section"),
        ("default", "unused_obj", "sort_section"),
        ("unused_obj", "sort_section", "default"),
        ("sort_section", "default", "unused_obj"),
    )
    seq = perms[seed_index % len(perms)]
    return "|".join(seq), seq


def fisher_exact_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total_hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, total_hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, total_hits - k) / ways

    observed = prob(hits_a)
    lo = max(0, total_hits - n_b)
    hi = min(n_a, total_hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1.0 + 1e-12)))


def mann_whitney_exact_and_asymptotic(
    sample_a: list[tuple[int, float]],
    sample_b: list[tuple[int, float]],
) -> tuple[float, float, float]:
    """Two-sided exact and asymptotic Mann-Whitney U on (miss_flag, calls).

    ``miss_flag`` is 0 for hits and 1 for misses so misses rank strictly worst
    and tie with each other. Returns ``(u_b, exact_two_sided_p, asymp_two_sided_p)``.
    """
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a == 0 or n_b == 0:
        raise ValueError("empty sample")

    tagged = [(val, 0) for val in sample_a] + [(val, 1) for val in sample_b]
    tagged.sort(key=lambda item: item[0])
    total = n_a + n_b
    ranks_2: list[int] = []
    group_sizes: list[int] = []
    ties_sum = 0
    b_rank_sum_2 = 0
    idx = 0
    while idx < total:
        end = idx
        while end + 1 < total and tagged[end + 1][0] == tagged[idx][0]:
            end += 1
        # Average 1-based rank * 2 is (idx + 1 + end + 1) = idx + end + 2
        r2 = idx + end + 2
        g_size = end - idx + 1
        ranks_2.extend([r2] * g_size)
        group_sizes.append(g_size)
        ties_sum += g_size**3 - g_size
        for k in range(idx, end + 1):
            if tagged[k][1] == 1:
                b_rank_sum_2 += r2
        idx = end + 1

    # U_B * 2 = R_B*2 - n_B * (n_B + 1)
    u_b_2 = b_rank_sum_2 - n_b * (n_b + 1)
    u_b = u_b_2 / 2.0
    max_u_2 = 2 * n_a * n_b
    dist_from_mean = abs(u_b_2 - n_a * n_b)

    # Exact permutation distribution over all C(N, n_B) assignments of B items
    # using doubled rank sums so all DP keys are integers.
    dp: dict[tuple[int, int], int] = {(0, 0): 1}
    pos = 0
    for g_size in group_sizes:
        r2 = ranks_2[pos]
        pos += g_size
        next_dp: dict[tuple[int, int], int] = {}
        for (cnt, rsum), ways in dp.items():
            max_take = min(g_size, n_b - cnt)
            for take in range(max_take + 1):
                mult = math.comb(g_size, take)
                key = (cnt + take, rsum + take * r2)
                next_dp[key] = next_dp.get(key, 0) + ways * mult
        dp = next_dp

    total_ways = math.comb(total, n_b)
    offset = n_b * (n_b + 1)
    extreme_ways = 0
    for (cnt, rsum), ways in dp.items():
        if cnt != n_b:
            continue
        cur_u_2 = rsum - offset
        if abs(cur_u_2 - n_a * n_b) >= dist_from_mean:
            extreme_ways += ways
    exact_p = min(1.0, extreme_ways / total_ways)

    var = n_a * n_b / 12.0 * ((total + 1) - ties_sum / (total * (total - 1)))
    if var <= 0.0:
        asymp_p = 1.0
    else:
        dev = max(abs(u_b - n_a * n_b / 2.0) - 0.5, 0.0)
        asymp_p = min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))

    return u_b, exact_p, asymp_p


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
        lib_path = str(sc_lib.resolve())
        lib_dir = str(sc_lib.resolve().parent)
        env["RGPOT_SC_LIB"] = lib_path
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
        raise RuntimeError(f"{system} seed {seed}: budget/seed mismatch in {data_lines[0]}")
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


def parse_h2h_seed(stdout: str, system: str, seed: int, budget: int) -> dict[str, str]:
    rows = [line for line in stdout.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"{system} seed {seed}: missing h2h_row\n{stdout[-2000:]}")
    kv = dict(tok.split("=", 1) for tok in rows[0].split()[1:])
    if kv.get("system") != system or int(kv["seed"]) != seed or int(kv["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: bad h2h_row {rows[0]}")
    wall_s = float(kv["wall_s"])
    pot_s = float(kv["pot_s"])
    outside_s = max(float(kv["outside_s"]), 0.0)
    return {
        "best": kv["best"].strip(),
        "calls": str(int(kv["evals_to_hit"])),
        "charged": str(int(kv["charged"])),
        "wall_s": f"{wall_s:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "outside_s": f"{outside_s:.6f}",
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
        raise RuntimeError(f"missing binary: {binary}")
    bin_sha = sha256_file(binary)
    env = child_env(sc_lib if cell.driver == "h2h_sc" else None)

    with tempfile.TemporaryDirectory(prefix="gate-a-seed-") as scratch:
        if cell.driver == "h2h_seed":
            n = int(cell.system.removeprefix("lj"))
            stop = repr(cell.reference + HIT_TOL)
            cmd = [
                "taskset",
                "-c",
                "0",
                str(binary),
                "lj",
                str(n),
                "0",
                str(cell.budget),
                str(seed),
                "recommended",
                stop,
                str(Path(scratch) / "run"),
            ]
        else:
            cmd = [
                "taskset",
                "-c",
                "0",
                str(binary),
                "run",
                cell.system,
                str(seed),
            ]
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

    parsed = (
        parse_h2h_seed(proc.stdout, cell.system, seed, cell.budget)
        if cell.driver == "h2h_seed"
        else parse_csv_driver(proc.stdout, cell.system, seed, cell.budget)
    )
    return parsed, bin_sha


def git_rev_and_tree(tree_path: Path) -> tuple[str, str]:
    rev = subprocess.check_output(["git", "-C", str(tree_path), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(tree_path), "rev-parse", "HEAD^{tree}"], text=True).strip()
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
        "best_bits": float_bits_hex(best_val),
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


def summarize_results(
    rows: list[dict[str, str]],
    band_rows: list[dict[str, str]],
    sessions: list[dict[str, str]],
    out_dir: Path,
) -> dict[str, object]:
    ab_by_cell: dict[str, dict[str, dict[int, dict[str, str]]]] = {}
    for row in rows:
        ab_by_cell.setdefault(row["system"], {}).setdefault(row["arm"], {})[int(row["seed"])] = row

    aa_by_cell: dict[str, dict[str, dict[int, dict[str, str]]]] = {}
    pl_by_cell: dict[str, dict[str, dict[int, dict[str, str]]]] = {}
    for row in band_rows:
        if row["band"] == "aa":
            aa_by_cell.setdefault(row["system"], {}).setdefault(row["arm"], {})[int(row["seed"])] = row
        elif row["band"] == "placement":
            pl_by_cell.setdefault(row["system"], {}).setdefault(row["arm"], {})[int(row["seed"])] = row

    # Validate balanced (slot, arm) counts
    balance_ok = True
    balance_issues: list[str] = []
    for cell in CELLS:
        for name, group in (("ab", ab_by_cell), ("aa", aa_by_cell)):
            arm_labels = ("A", "B") if name == "ab" else ("A", "A_prime")
            counts = {(s, a): 0 for s in (0, 1) for a in arm_labels}
            for a in arm_labels:
                for seed in SEEDS:
                    r = group.get(cell.system, {}).get(a, {}).get(seed)
                    if r is None:
                        balance_ok = False
                        balance_issues.append(f"missing {name} {cell.system} {a} {seed}")
                    else:
                        counts[(int(r["slot"]), a)] += 1
            if any(v != 6 for v in counts.values()):
                balance_ok = False
                balance_issues.append(f"unbalanced {name} {cell.system}: {counts}")

    for sys_name in PLACEMENT_CELLS:
        counts = {(s, v): 0 for s in (0, 1, 2) for v in PLACEMENT_VARIANTS}
        for v in PLACEMENT_VARIANTS:
            for seed in SEEDS:
                r = pl_by_cell.get(sys_name, {}).get(v, {}).get(seed)
                if r is None:
                    balance_ok = False
                    balance_issues.append(f"missing placement {sys_name} {v} {seed}")
                else:
                    counts[(int(r["slot"]), v)] += 1
        if any(c != 4 for c in counts.values()):
            balance_ok = False
            balance_issues.append(f"unbalanced placement {sys_name}: {counts}")

    # Compute A/A ratios per cell (all 14 cells)
    aa_stats: list[dict[str, object]] = []
    aa_ratios_dir: list[float] = []
    aa_ratios_sym: list[float] = []
    for cell in CELLS:
        a_rows = [aa_by_cell[cell.system]["A"][s] for s in SEEDS]
        ap_rows = [aa_by_cell[cell.system]["A_prime"][s] for s in SEEDS]
        out_a = [float(r["outside_s"]) / int(r["charged"]) for r in a_rows]
        out_ap = [float(r["outside_s"]) / int(r["charged"]) for r in ap_rows]
        med_a = type7(out_a, 0.5)
        med_ap = type7(out_ap, 0.5)
        ratio = med_ap / med_a
        sym_ratio = max(ratio, 1.0 / ratio)
        aa_ratios_dir.append(ratio)
        aa_ratios_sym.append(sym_ratio)
        aa_stats.append(
            {
                "system": cell.system,
                "med_a": med_a,
                "med_ap": med_ap,
                "ratio_ap_over_a": ratio,
                "sym_ratio": sym_ratio,
            }
        )

    # Compute placement band medians & ratios per placement cell
    pl_stats: list[dict[str, object]] = []
    pl_ratios_by_sys: dict[str, dict[str, object]] = {}
    all_pl_ratios_vs_def: list[float] = []
    all_pl_spread_ratios: list[float] = []
    for sys_name in PLACEMENT_CELLS:
        meds = {}
        for v in PLACEMENT_VARIANTS:
            v_rows = [pl_by_cell[sys_name][v][s] for s in SEEDS]
            meds[v] = type7([float(r["outside_s"]) / int(r["charged"]) for r in v_rows], 0.5)
        r_sort = meds["sort_section"] / meds["default"]
        r_unused = meds["unused_obj"] / meds["default"]
        max_vs_def = max(1.0, r_sort, r_unused)
        spread = max(meds.values()) / min(meds.values())
        all_pl_ratios_vs_def.extend([r_sort, r_unused])
        all_pl_spread_ratios.append(spread)
        entry = {
            "system": sys_name,
            "med_default": meds["default"],
            "med_sort_section": meds["sort_section"],
            "med_unused_obj": meds["unused_obj"],
            "ratio_sort_vs_default": r_sort,
            "ratio_unused_vs_default": r_unused,
            "max_vs_default": max_vs_def,
            "spread_max_over_min": spread,
        }
        pl_stats.append(entry)
        pl_ratios_by_sys[sys_name] = entry

    aa_upper_dir = max(aa_ratios_dir)
    aa_upper_sym = max(aa_ratios_sym)
    pooled_band_no_pl = aa_upper_sym
    pooled_band_all = max(aa_upper_sym, max(all_pl_spread_ratios))

    # Per-cell A/B summary
    cell_summaries: list[dict[str, object]] = []
    any_fail = not balance_ok
    fail_reasons: list[str] = []
    if not balance_ok:
        fail_reasons.extend(balance_issues)

    for cell in CELLS:
        a_rows = [ab_by_cell[cell.system]["A"][s] for s in SEEDS]
        b_rows = [ab_by_cell[cell.system]["B"][s] for s in SEEDS]

        hits_a_list = [rederived_hit(r["best"], cell.reference) for r in a_rows]
        hits_b_list = [rederived_hit(r["best"], cell.reference) for r in b_rows]
        hits_a = sum(hits_a_list)
        hits_b = sum(hits_b_list)
        hit_str_a = "".join(str(h) for h in hits_a_list)
        hit_str_b = "".join(str(h) for h in hits_b_list)

        fisher_p = fisher_exact_two_sided(hits_a, len(a_rows), hits_b, len(b_rows))
        fisher_bonf = min(1.0, fisher_p * len(CELLS))

        calls_a = [int(r["calls"]) if h == 1 else cell.budget for r, h in zip(a_rows, hits_a_list)]
        calls_b = [int(r["calls"]) if h == 1 else cell.budget for r, h in zip(b_rows, hits_b_list)]
        mw_sample_a = [(1 - h, float(c)) for h, c in zip(hits_a_list, calls_a)]
        mw_sample_b = [(1 - h, float(c)) for h, c in zip(hits_b_list, calls_b)]
        mw_u, mw_exact_p, mw_asymp_p = mann_whitney_exact_and_asymptotic(mw_sample_a, mw_sample_b)
        mw_bonf = min(1.0, mw_exact_p * len(CELLS))

        med_calls_a = type7([float(c) for c in calls_a], 0.5)
        med_calls_b = type7([float(c) for c in calls_b], 0.5)
        med_wall_a = type7([float(r["wall_s"]) for r in a_rows], 0.5)
        med_wall_b = type7([float(r["wall_s"]) for r in b_rows], 0.5)
        med_pot_a = type7([float(r["pot_s"]) for r in a_rows], 0.5)
        med_pot_b = type7([float(r["pot_s"]) for r in b_rows], 0.5)
        med_out_a = type7([float(r["outside_s"]) for r in a_rows], 0.5)
        med_out_b = type7([float(r["outside_s"]) for r in b_rows], 0.5)

        opc_a = [float(r["outside_s"]) / int(r["charged"]) for r in a_rows]
        opc_b = [float(r["outside_s"]) / int(r["charged"]) for r in b_rows]
        med_opc_a = type7(opc_a, 0.5)
        med_opc_b = type7(opc_b, 0.5)
        opc_ratio = med_opc_b / med_opc_a

        # Band for this system: pooled A/A + placement for placement systems,
        # pooled band where system has no placement variants.
        if cell.system in pl_ratios_by_sys:
            sys_band_upper = max(aa_upper_sym, float(pl_ratios_by_sys[cell.system]["spread_max_over_min"]))
        else:
            sys_band_upper = pooled_band_all

        hit_ok = hits_b >= hits_a
        cap_ok = opc_ratio <= OUTSIDE_RATIO_CAP
        band_ok = opc_ratio <= sys_band_upper
        cell_pass = hit_ok and cap_ok and band_ok
        if not cell_pass:
            any_fail = True
            if not hit_ok:
                fail_reasons.append(f"{cell.system}: hit loss B={hits_b} < A={hits_a}")
            if not cap_ok:
                fail_reasons.append(f"{cell.system}: B/A outside/charged {opc_ratio:.6f} > 1.10")
            if not band_ok:
                fail_reasons.append(
                    f"{cell.system}: B/A outside/charged {opc_ratio:.6f} > band upper {sys_band_upper:.6f}"
                )

        cell_summaries.append(
            {
                "system": cell.system,
                "family": cell.family,
                "driver": cell.driver,
                "budget": cell.budget,
                "reference": cell.reference,
                "hits_a": hits_a,
                "hits_b": hits_b,
                "hit_str_a": hit_str_a,
                "hit_str_b": hit_str_b,
                "fisher_p": fisher_p,
                "fisher_bonf_14": fisher_bonf,
                "med_calls_a": med_calls_a,
                "med_calls_b": med_calls_b,
                "mw_u_b": mw_u,
                "mw_exact_p": mw_exact_p,
                "mw_asymp_p": mw_asymp_p,
                "mw_bonf_14": mw_bonf,
                "med_wall_a": med_wall_a,
                "med_wall_b": med_wall_b,
                "med_pot_a": med_pot_a,
                "med_pot_b": med_pot_b,
                "med_out_a": med_out_a,
                "med_out_b": med_out_b,
                "med_opc_a": med_opc_a,
                "med_opc_b": med_opc_b,
                "opc_ratio_b_over_a": opc_ratio,
                "sys_band_upper": sys_band_upper,
                "aa_upper_sym": pooled_band_no_pl,
                "pooled_band_all": pooled_band_all,
                "hit_ok": hit_ok,
                "cap_ok": cap_ok,
                "band_ok": band_ok,
                "verdict": "PASS" if cell_pass else "FAIL",
            }
        )

    # Write summary CSVs
    summary_fields = list(cell_summaries[0].keys())
    write_csv(
        out_dir / "summary.csv",
        summary_fields,
        [{k: str(v) for k, v in row.items()} for row in cell_summaries],
    )
    write_csv(
        out_dir / "aa_band.csv",
        list(aa_stats[0].keys()),
        [{k: str(v) for k, v in row.items()} for row in aa_stats],
    )
    write_csv(
        out_dir / "placement_band.csv",
        list(pl_stats[0].keys()),
        [{k: str(v) for k, v in row.items()} for row in pl_stats],
    )

    verdict = "PASS" if not any_fail else "FAIL"
    return {
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "balance_ok": balance_ok,
        "aa_upper_dir": aa_upper_dir,
        "aa_upper_sym": aa_upper_sym,
        "pooled_band_all": pooled_band_all,
        "cell_summaries": cell_summaries,
        "aa_stats": aa_stats,
        "pl_stats": pl_stats,
        "sessions": sessions,
    }


def write_report(summary: dict[str, object], report_path: Path) -> None:
    cells: list[dict[str, object]] = summary["cell_summaries"]  # type: ignore[assignment]
    aa_stats: list[dict[str, object]] = summary["aa_stats"]  # type: ignore[assignment]
    pl_stats: list[dict[str, object]] = summary["pl_stats"]  # type: ignore[assignment]
    sessions: list[dict[str, str]] = summary["sessions"]  # type: ignore[assignment]

    sc108_aa = next(row for row in aa_stats if row["system"] == "sc108-55")
    verdict = str(summary["verdict"])
    fail_reasons: list[str] = summary["fail_reasons"]  # type: ignore[assignment]

    lines: list[str] = [
        "# sc-shelf-land4 Gate A Report",
        "",
        f"**Verdict: {verdict}**",
        "",
    ]
    if fail_reasons:
        lines.append("## Failure Reasons")
        lines.append("")
        for reason in fail_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    # Provenance from sessions
    machine_ids = sorted({s["hashed_machine_id"] for s in sessions})
    rgpot_revs = sorted({s["rgpot_revision"] for s in sessions if s["rgpot_revision"]})
    rgpot_shas = sorted({s["librgpot_sha256"] for s in sessions if s["librgpot_sha256"]})
    rustcs = sorted({s["rustc_version"] for s in sessions})
    lines.extend(
        [
            "## Provenance & Protocol",
            "",
            f"- Arm A: `{A_COMMIT}` (tree `{A_TREE}`)",
            f"- Arm B: `{B_TREE}` (tree after `git am -3 --keep-cr` of 9-commit series on A)",
            f"- Hashed machine ID: `{', '.join(machine_ids)}`",
            f"- Toolchain: `{', '.join(rustcs)}` (`cargo build --release --locked`)",
            f"- rgpot revision: `{', '.join(rgpot_revs)}`",
            f"- librgpot SHA-256: `{', '.join(rgpot_shas)}`",
            f"- Core pinning: `taskset -c 0` across entire process tree; all thread env vars = 1",
            "- Seeds: `2100..2111` (`n=12`), balanced ABBA order (`6` `(slot=0, A)`, `6` `(slot=0, B)`, `6` `(slot=1, A)`, `6` `(slot=1, B)` per cell)",
            "- Hit rule: finite `best <= reference + 0.001` (re-derived from `best` column)",
            "- Outside time per call: `outside_s / charged` computed per row, median per arm, ratio `B/A` of medians",
            "",
            "### Binary SHA-256 Registry",
            "",
            "| phase | arm | driver | code_revision | tree | binary_sha256 |",
            "|---|---|---|---|---|---|",
        ]
    )
    seen_bins: set[tuple[str, str, str, str]] = set()
    for s in sessions:
        key = (s["phase"], s["arm"], s["driver"], s["binary_sha256"])
        if key in seen_bins:
            continue
        seen_bins.add(key)
        lines.append(
            f"| {s['phase']} | {s['arm']} | `{s['driver']}` | `{s['code_revision']}` | `{s['tree']}` | `{s['binary_sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Declared Rule Evaluation (14 Cells)",
            "",
            "| system | driver | hits A/B | Fisher p | Fisher Bonf(14) | med calls A/B | MW exact p | MW Bonf(14) | wall med s A/B | pot med s A/B | out/chg med us A/B | B/A out/chg | band upper | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in cells:
        opc_a_us = float(row["med_opc_a"]) * 1.0e6
        opc_b_us = float(row["med_opc_b"]) * 1.0e6
        lines.append(
            f"| {row['system']} | `{row['driver']}` | {row['hits_a']}/{row['hits_b']} | "
            f"{float(row['fisher_p']):.6g} | {float(row['fisher_bonf_14']):.6g} | "
            f"{float(row['med_calls_a']):.1f}/{float(row['med_calls_b']):.1f} | "
            f"{float(row['mw_exact_p']):.6g} | {float(row['mw_bonf_14']):.6g} | "
            f"{float(row['med_wall_a']):.4f}/{float(row['med_wall_b']):.4f} | "
            f"{float(row['med_pot_a']):.4f}/{float(row['med_pot_b']):.4f} | "
            f"{opc_a_us:.3f}/{opc_b_us:.3f} | "
            f"{float(row['opc_ratio_b_over_a']):.6f} | "
            f"{float(row['sys_band_upper']):.6f} | {row['verdict']} |"
        )

    lines.extend(
        [
            "",
            "### Per-Seed Hit Strings (seeds 2100..2111)",
            "",
            "| system | A hits | B hits |",
            "|---|---|---|",
        ]
    )
    for row in cells:
        lines.append(f"| {row['system']} | `{row['hit_str_a']}` ({row['hits_a']}/12) | `{row['hit_str_b']}` ({row['hits_b']}/12) |")

    lines.extend(
        [
            "",
            "## Noise Bands",
            "",
            f"- `sc108-55` A/A outside-time-per-charged-call ratio (`A'/A`): `{float(sc108_aa['ratio_ap_over_a']):.6f}` (symmetric `{float(sc108_aa['sym_ratio']):.6f}`)",
            f"- Pooled A/A upper edge across all 14 cells: `{float(summary['aa_upper_sym']):.6f}` (directional max `A'/A`: `{float(summary['aa_upper_dir']):.6f}`)",
            f"- Pooled A/A + placement band upper edge: `{float(summary['pooled_band_all']):.6f}`",
            "",
            "### A/A Band (All 14 Cells, A vs A')",
            "",
            "| system | med out/chg A (us) | med out/chg A' (us) | ratio A'/A | symmetric max(r, 1/r) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in aa_stats:
        lines.append(
            f"| {row['system']} | {float(row['med_a']) * 1e6:.3f} | {float(row['med_ap']) * 1e6:.3f} | "
            f"{float(row['ratio_ap_over_a']):.6f} | {float(row['sym_ratio']):.6f} |"
        )

    lines.extend(
        [
            "",
            "### Placement Band (Arm A Rebuilt in 3 Link Variants)",
            "",
            "| system | default (us) | sort_section (us) | unused_obj (us) | sort/default | unused/default | spread max/min |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pl_stats:
        lines.append(
            f"| {row['system']} | {float(row['med_default']) * 1e6:.3f} | "
            f"{float(row['med_sort_section']) * 1e6:.3f} | "
            f"{float(row['med_unused_obj']) * 1e6:.3f} | "
            f"{float(row['ratio_sort_vs_default']):.6f} | "
            f"{float(row['ratio_unused_vs_default']):.6f} | "
            f"{float(row['spread_max_over_min']):.6f} |"
        )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_campaign(args: argparse.Namespace) -> int:
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    machine_id = hashed_machine_id()
    allowed = cpus_allowed()
    if allowed != "0":
        raise RuntimeError(f"process tree must run pinned to core 0 (Cpus_allowed_list={allowed})")

    rustc_ver = subprocess.check_output(["rustc", "+1.88.0", "--version"], text=True).strip()
    cargo_ver = subprocess.check_output(["cargo", "+1.88.0", "--version"], text=True).strip()
    rgpot_rev = subprocess.check_output(["git", "-C", str(args.rgpot_root), "rev-parse", "HEAD"], text=True).strip()
    librgpot_sha = sha256_file(args.sc_lib.resolve())

    a_rev, a_tree = git_rev_and_tree(args.a_tree)
    b_rev, b_tree = git_rev_and_tree(args.b_tree)
    ap_rev, ap_tree = git_rev_and_tree(args.a_prime_tree)
    if a_rev != A_COMMIT or a_tree != A_TREE:
        raise RuntimeError(f"Arm A mismatch: {a_rev} / {a_tree}")
    if b_tree != B_TREE:
        raise RuntimeError(f"Arm B tree mismatch: {b_tree} != {B_TREE}")
    if ap_rev != A_COMMIT or ap_tree != A_TREE:
        raise RuntimeError(f"Arm A' mismatch: {ap_rev} / {ap_tree}")

    out_dir: Path = args.out_dir
    rows_path = out_dir / "rows.csv"
    band_rows_path = out_dir / "band_rows.csv"
    sessions_path = out_dir / "sessions.csv"
    logs_dir: Path = args.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_ab = {
        (r["system"], r["arm"], int(r["seed"]))
        for r in read_csv(rows_path)
    }
    existing_band = {
        (r["band"], r["system"], r["arm"], int(r["seed"]))
        for r in read_csv(band_rows_path)
    }

    thread_env_str = ";".join(f"{k}={v}" for k, v in THREAD_ENV.items())
    session_records: list[dict[str, str]] = []

    def record_session(
        *,
        phase: str,
        started: str,
        finished: str,
        arm: str,
        rev: str,
        tree: str,
        driver: str,
        bin_sha: str,
        rustflags: str,
    ) -> None:
        rec = {
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
            "binary_sha256": bin_sha,
            "rustflags": rustflags,
            "rgpot_revision": rgpot_rev,
            "librgpot_sha256": librgpot_sha,
            "thread_env": thread_env_str,
        }
        session_records.append(rec)
        append_csv(sessions_path, SESSION_FIELDS, rec)

    # Record binary metadata upfront for all phases
    phase_bins = [
        ("ab", "A", a_rev, a_tree, args.a_bin, "default"),
        ("ab", "B", b_rev, b_tree, args.b_bin, "default"),
        ("aa", "A", a_rev, a_tree, args.a_bin, "default"),
        ("aa", "A_prime", ap_rev, ap_tree, args.a_prime_bin, "default"),
        ("placement", "default", a_rev, a_tree, args.placement_default_bin, "default"),
        ("placement", "sort_section", a_rev, a_tree, args.placement_sort_bin, "-C link-arg=-Wl,--sort-section=name"),
        ("placement", "unused_obj", a_rev, a_tree, args.placement_unused_bin, "-C link-arg=unused_pad.o -C link-arg=-Wl,-u,anneal_placement_unused_fn"),
    ]
    started_all = now_utc()

    # 1. Main A/B campaign (14 cells x 12 seeds x 2 arms)
    arm_meta_ab = {
        "A": (args.a_bin, args.a_tree, a_rev, a_tree),
        "B": (args.b_bin, args.b_tree, b_rev, b_tree),
    }
    for cell in CELLS:
        for idx, seed in enumerate(SEEDS):
            order, seq = abba_pair(idx, "A", "B")
            for slot, arm in enumerate(seq):
                if (cell.system, arm, seed) in existing_ab:
                    continue
                bin_dir, tree_cwd, rev, tree = arm_meta_ab[arm]
                parsed, bin_sha = run_cell_seed(
                    cell=cell,
                    seed=seed,
                    bin_dir=bin_dir,
                    work_cwd=tree_cwd,
                    sc_lib=args.sc_lib,
                    log_path=logs_dir / "ab" / cell.system / f"seed_{seed}_{arm}.log",
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
                existing_ab.add((cell.system, arm, seed))
                print(
                    f"[AB] {cell.system} seed={seed} slot={slot} arm={arm} hit={row['hit']} "
                    f"calls={row['calls']} charged={row['charged']} wall={row['wall_s']} out={row['outside_s']}",
                    flush=True,
                )

    # 2. A/A band (all 14 cells x 12 seeds x 2 arms: A vs A_prime)
    arm_meta_aa = {
        "A": (args.a_bin, args.a_tree, a_rev, a_tree),
        "A_prime": (args.a_prime_bin, args.a_prime_tree, ap_rev, ap_tree),
    }
    for cell in CELLS:
        for idx, seed in enumerate(SEEDS):
            order, seq = abba_pair(idx, "A", "A_prime")
            for slot, arm in enumerate(seq):
                if ("aa", cell.system, arm, seed) in existing_band:
                    continue
                bin_dir, tree_cwd, rev, tree = arm_meta_aa[arm]
                parsed, bin_sha = run_cell_seed(
                    cell=cell,
                    seed=seed,
                    bin_dir=bin_dir,
                    work_cwd=tree_cwd,
                    sc_lib=args.sc_lib,
                    log_path=logs_dir / "aa" / cell.system / f"seed_{seed}_{arm}.log",
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
                append_csv(band_rows_path, BAND_ROW_FIELDS, {"band": "aa", **row})
                existing_band.add(("aa", cell.system, arm, seed))
                print(
                    f"[AA] {cell.system} seed={seed} slot={slot} arm={arm} hit={row['hit']} "
                    f"calls={row['calls']} charged={row['charged']} wall={row['wall_s']} out={row['outside_s']}",
                    flush=True,
                )

    # 3. Placement band (4 cells x 12 seeds x 3 link variants)
    arm_meta_pl = {
        "default": args.placement_default_bin,
        "sort_section": args.placement_sort_bin,
        "unused_obj": args.placement_unused_bin,
    }
    for sys_name in PLACEMENT_CELLS:
        cell = CELL_BY_NAME[sys_name]
        for idx, seed in enumerate(SEEDS):
            order, seq = placement_triple(idx)
            for slot, variant in enumerate(seq):
                if ("placement", cell.system, variant, seed) in existing_band:
                    continue
                bin_dir = arm_meta_pl[variant]
                parsed, bin_sha = run_cell_seed(
                    cell=cell,
                    seed=seed,
                    bin_dir=bin_dir,
                    work_cwd=args.a_tree,
                    sc_lib=args.sc_lib,
                    log_path=logs_dir / "placement" / cell.system / f"seed_{seed}_{variant}.log",
                )
                row = build_row(
                    session=session_id,
                    cell=cell,
                    arm=variant,
                    seed=seed,
                    slot=slot,
                    order=order,
                    parsed=parsed,
                    machine_id=machine_id,
                    rev=a_rev,
                    tree=a_tree,
                    bin_sha=bin_sha,
                )
                append_csv(band_rows_path, BAND_ROW_FIELDS, {"band": "placement", **row})
                existing_band.add(("placement", cell.system, variant, seed))
                print(
                    f"[PL] {cell.system} seed={seed} slot={slot} variant={variant} hit={row['hit']} "
                    f"calls={row['calls']} charged={row['charged']} wall={row['wall_s']} out={row['outside_s']}",
                    flush=True,
                )

    finished_all = now_utc()
    drivers_by_phase = {
        "ab": ("h2h_gmin_proof", "h2h_seed", "h2h_morse", "h2h_sc"),
        "aa": ("h2h_gmin_proof", "h2h_seed", "h2h_morse", "h2h_sc"),
        "placement": ("h2h_gmin_proof", "h2h_morse", "h2h_sc"),
    }
    for phase, arm, rev, tree, bin_dir, rustflags in phase_bins:
        for drv in drivers_by_phase[phase]:
            bin_path = bin_dir / drv
            record_session(
                phase=phase,
                started=started_all,
                finished=finished_all,
                arm=arm,
                rev=rev,
                tree=tree,
                driver=drv,
                bin_sha=sha256_file(bin_path),
                rustflags=rustflags,
            )

    all_rows = read_csv(rows_path)
    all_band = read_csv(band_rows_path)
    all_sessions = read_csv(sessions_path)
    summary = summarize_results(all_rows, all_band, all_sessions, out_dir)
    write_report(summary, out_dir / "README.md")
    print(f"CAMPAIGN COMPLETE verdict={summary['verdict']}", flush=True)
    return 0


def self_test() -> None:
    assert len(CELLS) == 14
    assert len(SEEDS) == 12 and SEEDS[0] == 2100 and SEEDS[-1] == 2111
    assert rederived_hit("-173.928427", -173.928427) == 1
    assert rederived_hit("-173.927427", -173.928427) == 1
    assert rederived_hit("-173.927426", -173.928427) == 0
    assert rederived_hit("nan", -173.928427) == 0
    assert rederived_hit("inf", -173.928427) == 0
    assert rederived_hit("-inf", -173.928427) == 0
    assert float_bits_hex(-1.0) == "0xbff0000000000000"

    # ABBA balance
    counts = {(s, a): 0 for s in (0, 1) for a in ("A", "B")}
    for i in range(12):
        _, seq = abba_pair(i, "A", "B")
        for s, a in enumerate(seq):
            counts[(s, a)] += 1
    assert all(v == 6 for v in counts.values()), counts

    # Placement 3-variant balance
    pcounts = {(s, v): 0 for s in (0, 1, 2) for v in PLACEMENT_VARIANTS}
    for i in range(12):
        _, seq = placement_triple(i)
        for s, v in enumerate(seq):
            pcounts[(s, v)] += 1
    assert all(v == 4 for v in pcounts.values()), pcounts

    # Fisher exact check: 12/12 vs 12/12 -> p=1.0
    assert abs(fisher_exact_two_sided(12, 12, 12, 12) - 1.0) < 1e-12
    # 12/12 vs 0/12 -> 2 / C(24, 12)
    expected_extreme = 2.0 / math.comb(24, 12)
    assert abs(fisher_exact_two_sided(12, 12, 0, 12) - expected_extreme) < 1e-15

    # Exact MW check with ties (misses ranked worst)
    a = [(0, 100.0), (0, 200.0), (1, 400.0)]
    b = [(0, 150.0), (1, 400.0), (1, 400.0)]
    u, ep, ap = mann_whitney_exact_and_asymptotic(a, b)
    assert 0.0 <= ep <= 1.0 and 0.0 <= ap <= 1.0 and u > 0.0
    print("self-test ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--a-tree", type=Path)
    parser.add_argument("--b-tree", type=Path)
    parser.add_argument("--a-prime-tree", type=Path)
    parser.add_argument("--a-bin", type=Path)
    parser.add_argument("--b-bin", type=Path)
    parser.add_argument("--a-prime-bin", type=Path)
    parser.add_argument("--placement-default-bin", type=Path)
    parser.add_argument("--placement-sort-bin", type=Path)
    parser.add_argument("--placement-unused-bin", type=Path)
    parser.add_argument("--rgpot-root", type=Path)
    parser.add_argument("--sc-lib", type=Path)
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if args.summarize:
        rows = read_csv(args.out_dir / "rows.csv")
        band_rows = read_csv(args.out_dir / "band_rows.csv")
        sessions = read_csv(args.out_dir / "sessions.csv")
        summary = summarize_results(rows, band_rows, sessions, args.out_dir)
        write_report(summary, args.out_dir / "README.md")
        print(f"verdict={summary['verdict']}")
        return 0
    required = (
        args.logs_dir,
        args.a_tree,
        args.b_tree,
        args.a_prime_tree,
        args.a_bin,
        args.b_bin,
        args.a_prime_bin,
        args.placement_default_bin,
        args.placement_sort_bin,
        args.placement_unused_bin,
        args.rgpot_root,
        args.sc_lib,
    )
    if any(item is None for item in required):
        parser.error("missing required campaign paths")
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
