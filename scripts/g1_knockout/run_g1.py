# Copyright (c) anneal developers
"""Single-knockout component ablation runner and analyzer (G1).

Runs the full ``recommended`` preset against eight single-knockout arms across
the nine-system Cambridge/Gupta panel (seeds 2000-2011) using the compiled
release ``g1_knockout`` example binary.

Conditions:
- Single serial session, pinned to core 0 via ``taskset -c 0``.
- Single-threaded numeric/runtime environment variables set to ``1``.
- All ``H2H_*`` environment variables stripped from the child process.
- Balanced ABBA arm rotation per ``(system, seed)`` block.
- ``PAPER_BUDGETS`` and published references imported from
  ``scripts/gmin_panel/run_panel.py``.
- Hit criterion delegated to ``scripts/target_tol.py``.
- No warm-up runs and no ``--quick`` mode.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (SCRIPTS_DIR, SCRIPTS_DIR / "gmin_panel"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_panel import PAPER_BUDGETS, SYSTEMS as PANEL_SYSTEMS  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SEEDS: tuple[int, ...] = tuple(range(2000, 2012))

SYSTEM_ORDER: tuple[str, ...] = (
    "lj38",
    "lj55",
    "lj75",
    "lj98",
    "lj150",
    "m55r14",
    "m80r6",
    "m80r14",
    "au55",
)

SYSTEM_LABELS: dict[str, str] = {
    "lj38": "LJ38",
    "lj55": "LJ55",
    "lj75": "LJ75",
    "lj98": "LJ98",
    "lj150": "LJ150",
    "m55r14": "M55 rho=14",
    "m80r6": "M80 rho=6",
    "m80r14": "M80 rho=14",
    "au55": "Au55",
}

ARMS: tuple[str, ...] = (
    "recommended",
    "a_restart_on_stall",
    "b_no_orbit",
    "c_no_symmetry",
    "d_no_screens",
    "e_no_surfaces",
    "f_no_energy_lattice",
    "g_no_priors",
    "h_no_basin_bias",
)

ARM_DECLARED_DIFFS: dict[str, frozenset[str]] = {
    "recommended": frozenset(),
    "a_restart_on_stall": frozenset(
        {"restart_on_stall", "escape_on_stall", "trail_on_stall", "tabu_on_stall"}
    ),
    "b_no_orbit": frozenset({"orbit_complete_on_new"}),
    "c_no_symmetry": frozenset({"continuous_symmetry", "symmetrise_on_stall"}),
    "d_no_screens": frozenset({"return_screen", "bayes_screen"}),
    "e_no_surfaces": frozenset({"surfaces"}),
    "f_no_energy_lattice": frozenset({"energy_lattice"}),
    "g_no_priors": frozenset({"prior_soap"}),
    "h_no_basin_bias": frozenset({"bias_height"}),
}

THREAD_ENV_VARS: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}

ROW_FIELDS: list[str] = [
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
    "binary_sha256",
    "config_hash",
    "session",
    "sequence",
    "status",
]

SURVIVAL_FIELDS: list[str] = [
    "system",
    "arm",
    "calls",
    "n_at_risk",
    "n_events",
    "n_censored",
    "survival",
    "greenwood_se",
]

SUMMARY_FIELDS: list[str] = [
    "system",
    "budget",
    "reference",
    "arm",
    "n",
    "hits",
    "errors",
    "timeouts",
    "fisher_p",
    "logrank_chi2",
    "logrank_p",
    "mw_u",
    "mw_p",
    "median_calls",
    "iqr_calls",
    "median_calls_hits_only",
    "km_median_calls",
    "median_wall_s",
    "median_outside_s",
    "median_outside_per_charged_s",
    "best_min",
    "config_hash",
    "n_tests",
]

README_TABLE_MARKS = (
    "<!-- g1-knockout table: start -->",
    "<!-- g1-knockout table: end -->",
)


def panel_specs() -> dict[str, dict[str, Any]]:
    """Return the nine G1 systems with budgets and refs from the panel harness."""
    by_name = {row[0]: row for row in PANEL_SYSTEMS}
    specs: dict[str, dict[str, Any]] = {}
    for name in SYSTEM_ORDER:
        row = by_name[name]
        specs[name] = {
            "system": name,
            "kind": row[1],
            "n": int(row[2]),
            "reference": float(row[3]),
            "rho": float(row[4]),
            "budget": int(PAPER_BUDGETS[name]),
        }
    return specs


def expected_diff_fields(n_atoms: int, arm: str) -> frozenset[str]:
    """Return the exact config keys expected to differ from ``recommended``."""
    if arm == "e_no_surfaces" and n_atoms >= 98:
        return frozenset()
    return ARM_DECLARED_DIFFS[arm]


def arm_order_for(sys_idx: int, seed_idx: int, arms: tuple[str, ...] = ARMS) -> tuple[str, ...]:
    """Balanced ABBA rotation of the arms for one ``(system, seed)`` block.

    Consecutive seed pairs ``(2k, 2k+1)`` run a cyclic rotation forward on
    ``2k`` and reversed on ``2k+1``, so every arm has exact mean slot
    ``(len(arms) - 1) / 2`` within each system and uniform ``(arm, slot)``
    counts across the 9-system panel.
    """
    m = len(arms)
    shift = (sys_idx + (seed_idx // 2)) % m
    rotated = arms[shift:] + arms[:shift]
    if seed_idx % 2 == 1:
        return tuple(reversed(rotated))
    return rotated


def build_schedule(
    systems: tuple[str, ...] = SYSTEM_ORDER,
    seeds: tuple[int, ...] = SEEDS,
    arms: tuple[str, ...] = ARMS,
) -> list[dict[str, Any]]:
    """Build the complete serial execution schedule."""
    schedule: list[dict[str, Any]] = []
    seq = 0
    for sys_idx, system in enumerate(systems):
        for seed_idx, seed in enumerate(seeds):
            order = arm_order_for(sys_idx, seed_idx, arms)
            for slot, arm in enumerate(order):
                seq += 1
                schedule.append(
                    {
                        "sequence": seq,
                        "system": system,
                        "seed": seed,
                        "slot": slot,
                        "arm": arm,
                    }
                )
    return schedule


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            h.update(chunk)
    return h.hexdigest()


def hashed_machine_id() -> str:
    """Return a deterministic SHA-256 machine identifier without raw hostnames."""
    parts: list[str] = []
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                parts.append(text)
                break
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                parts.append(line.split(":", 1)[1].strip())
                break
    uname = os.uname()
    parts.extend([uname.sysname, uname.machine, uname.nodename])
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def child_env() -> dict[str, str]:
    """Single-threaded environment with all ``H2H_*`` variables removed."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("H2H_")}
    env.update(THREAD_ENV_VARS)
    return env


def git_rev(ref: str, cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def fetch_config_json(binary: Path, system: str, arm: str) -> dict[str, Any]:
    proc = subprocess.run(
        [str(binary), "config", system, arm],
        env=child_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def fetch_config_hash(binary: Path, system: str, arm: str) -> str:
    proc = subprocess.run(
        [str(binary), "config-hash", system, arm],
        env=child_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def type7(values: list[float], p: float) -> float:
    """Hyndman-Fan type 7 quantile."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a non-empty sample")
    if count == 1:
        return ordered[0]
    pos = 1.0 + (count - 1) * p
    lower = int(pos)
    frac = pos - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher_exact_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Exact two-sided Fisher test summing tables no more likely than observed."""
    total_hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, total_hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, total_hits - k) / ways

    observed = prob(hits_a)
    lo = max(0, total_hits - n_b)
    hi = min(n_a, total_hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1.0 + 1e-12)))


def mann_whitney_two_sided(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U with tie and continuity corrections.

    Handles ``+inf`` values cleanly (misses ranked above every finite hit and
    tied with one another).
    """
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
    return u, min(1.0, math.erfc(dev / math.sqrt(2.0 * var)))


def kaplan_meier(times: list[int], events: list[int]) -> list[dict[str, Any]]:
    """Kaplan-Meier survival curve for calls-to-hit with misses right-censored."""
    if len(times) != len(events) or not times:
        raise ValueError("kaplan_meier needs non-empty equal-length times and events")
    n_total = len(times)
    by_time: dict[int, list[int]] = {}
    for t, e in zip(times, events):
        by_time.setdefault(int(t), []).append(int(e))

    curve: list[dict[str, Any]] = [
        {
            "calls": 0,
            "n_at_risk": n_total,
            "n_events": 0,
            "n_censored": 0,
            "survival": 1.0,
            "greenwood_se": 0.0,
        }
    ]
    at_risk = n_total
    surv = 1.0
    gw_sum = 0.0
    for t in sorted(by_time):
        evs = by_time[t]
        d = sum(1 for e in evs if e == 1)
        c = len(evs) - d
        if d > 0:
            surv *= 1.0 - d / at_risk
            if at_risk > d:
                gw_sum += d / (at_risk * (at_risk - d))
            else:
                gw_sum = 0.0
        se = surv * math.sqrt(gw_sum) if surv > 0.0 and gw_sum > 0.0 else 0.0
        curve.append(
            {
                "calls": t,
                "n_at_risk": at_risk,
                "n_events": d,
                "n_censored": c,
                "survival": surv,
                "greenwood_se": se,
            }
        )
        at_risk -= len(evs)
    return curve


def km_median(curve: list[dict[str, Any]]) -> float | None:
    """Smallest event time where Kaplan-Meier survival reaches <= 0.5."""
    for step in curve:
        if step["calls"] > 0 and step["n_events"] > 0 and step["survival"] <= 0.5:
            return float(step["calls"])
    return None


def logrank_two_sided(
    times_a: list[int],
    events_a: list[int],
    times_b: list[int],
    events_b: list[int],
) -> tuple[float, float]:
    """Two-sided Mantel-Haenszel log-rank test (hypergeometric variance)."""
    event_times = sorted(
        {
            *(t for t, e in zip(times_a, events_a) if e == 1),
            *(t for t, e in zip(times_b, events_b) if e == 1),
        }
    )
    if not event_times:
        return 0.0, 1.0
    obs_minus_exp = 0.0
    var_sum = 0.0
    for t in event_times:
        n1 = sum(1 for x in times_a if x >= t)
        n0 = sum(1 for x in times_b if x >= t)
        n = n1 + n0
        if n <= 1:
            continue
        d1 = sum(1 for x, e in zip(times_a, events_a) if x == t and e == 1)
        d0 = sum(1 for x, e in zip(times_b, events_b) if x == t and e == 1)
        d = d1 + d0
        if d == 0:
            continue
        e1 = d * (n1 / n)
        v1 = (n1 * n0 * d * (n - d)) / (n * n * (n - 1))
        obs_minus_exp += d1 - e1
        var_sum += v1
    if var_sum <= 0.0:
        return 0.0, 1.0
    chi2 = (obs_minus_exp * obs_minus_exp) / var_sum
    p_val = min(1.0, math.erfc(math.sqrt(chi2 / 2.0)))
    return chi2, p_val


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
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


def run_one(
    *,
    binary: Path,
    system: str,
    arm: str,
    seed: int,
    slot: int,
    sequence: int,
    budget: int,
    reference: float,
    core: int,
    machine_id: str,
    binary_sha256: str,
    config_hash: str,
    session: str,
    log_dir: Path,
    timeout_s: float,
) -> dict[str, str]:
    """Execute one serial pinned run; errors and timeouts are recorded as misses."""
    cmd = ["taskset", "-c", str(core), str(binary), "run", system, arm, str(seed), str(budget)]
    env = child_env()
    started = time.monotonic()
    status = "ok"
    returncode: int | None = None
    stdout_text = ""
    stderr_text = ""
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.monotonic() - started
        returncode = proc.returncode
        stdout_text = proc.stdout
        stderr_text = proc.stderr
        if proc.returncode != 0:
            status = "error"
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        status = "timeout"
        stdout_text = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr_text = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{sequence:04d}_{system}_{arm}_s{seed}.log"
    safe_cmd = f"taskset -c {core} {binary.name} run {system} {arm} {seed} {budget}"
    log_file.write_text(
        f"cmd={safe_cmd}\n"
        f"status={status}\n"
        f"returncode={returncode}\n"
        f"elapsed_s={elapsed:.6f}\n"
        f"--- stdout ---\n{stdout_text}\n"
        f"--- stderr ---\n{stderr_text}\n",
        encoding="utf-8",
    )

    if status == "ok":
        try:
            lines = [line for line in stdout_text.splitlines() if line.strip()]
            payload = json.loads(lines[-1])
            best_val = float(payload["best"])
            hit_flag = energy_hits(best_val, reference, DEFAULT_HIT_TOL)
            charged = int(payload["charged"])
            calls = int(payload["calls"]) if hit_flag else budget
            if hit_flag and calls > budget:
                calls = budget
            wall_s = float(payload["wall_s"])
            pot_s = float(payload["pot_s"])
            outside_s = max(float(payload["outside_s"]), 0.0)
            return {
                "system": system,
                "arm": arm,
                "seed": str(seed),
                "slot": str(slot),
                "core": str(payload.get("core", core)),
                "hit": "1" if hit_flag else "0",
                "calls": str(calls),
                "charged": str(charged),
                "wall_s": f"{wall_s:.6f}",
                "pot_s": f"{pot_s:.6f}",
                "outside_s": f"{outside_s:.6f}",
                "best": str(payload["best"]),
                "best_bits": str(payload["best_bits"]),
                "budget": str(budget),
                "machine_id": machine_id,
                "binary_sha256": binary_sha256,
                "config_hash": str(payload.get("config_hash", config_hash)),
                "session": session,
                "sequence": str(sequence),
                "status": "ok",
            }
        except Exception as exc:  # noqa: BLE001
            status = "error"
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"\n--- parse_error ---\n{exc!r}\n")

    return {
        "system": system,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "core": str(core),
        "hit": "0",
        "calls": str(budget),
        "charged": "0",
        "wall_s": f"{elapsed:.6f}",
        "pot_s": "0.000000",
        "outside_s": f"{elapsed:.6f}",
        "best": "inf",
        "best_bits": "0x7ff0000000000000",
        "budget": str(budget),
        "machine_id": machine_id,
        "binary_sha256": binary_sha256,
        "config_hash": config_hash,
        "session": session,
        "sequence": str(sequence),
        "status": status,
    }


def summarize_rows(
    rows: list[dict[str, str]],
    *,
    specs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Compute survival curves and per-(system, arm) summary from raw rows only."""
    if specs is None:
        specs = panel_specs()

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["arm"]), []).append(row)

    present_systems = [s for s in SYSTEM_ORDER if any((s, a) in grouped for a in ARMS)]
    n_tests = sum(
        1
        for s in present_systems
        for a in ARMS
        if a != "recommended" and (s, a) in grouped and (s, "recommended") in grouped
    )

    survival_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for system in present_systems:
        spec = specs[system]
        budget = int(spec["budget"])
        reference = float(spec["reference"])
        rec_rows = sorted(grouped.get((system, "recommended"), []), key=lambda r: int(r["seed"]))

        def extract_group(group: list[dict[str, str]]) -> dict[str, Any]:
            hits_flags: list[int] = []
            calls_list: list[int] = []
            ranked_calls: list[float] = []
            hits_only_calls: list[float] = []
            wall_list: list[float] = []
            outside_list: list[float] = []
            outside_per_charged: list[float] = []
            finite_bests: list[float] = []
            n_err = 0
            n_to = 0
            for r in group:
                status = r.get("status", "ok")
                if status == "error":
                    n_err += 1
                elif status == "timeout":
                    n_to += 1
                best_v = float(r["best"])
                if math.isfinite(best_v):
                    finite_bests.append(best_v)
                is_hit = status == "ok" and energy_hits(best_v, reference, DEFAULT_HIT_TOL)
                c = int(r["calls"]) if is_hit else budget
                if is_hit and c > budget:
                    c = budget
                hits_flags.append(1 if is_hit else 0)
                calls_list.append(c)
                ranked_calls.append(float(c) if is_hit else math.inf)
                if is_hit:
                    hits_only_calls.append(float(c))
                wall_v = float(r["wall_s"])
                out_v = float(r["outside_s"])
                chg = int(r["charged"])
                wall_list.append(wall_v)
                outside_list.append(out_v)
                if chg > 0:
                    outside_per_charged.append(out_v / chg)
            return {
                "n": len(group),
                "hits": sum(hits_flags),
                "hits_flags": hits_flags,
                "calls": calls_list,
                "ranked_calls": ranked_calls,
                "hits_only_calls": hits_only_calls,
                "wall": wall_list,
                "outside": outside_list,
                "outside_per_charged": outside_per_charged,
                "best_min": min(finite_bests) if finite_bests else math.inf,
                "errors": n_err,
                "timeouts": n_to,
                "config_hashes": sorted({r["config_hash"] for r in group}),
            }

        rec_data = extract_group(rec_rows) if rec_rows else None

        for arm in ARMS:
            group = sorted(grouped.get((system, arm), []), key=lambda r: int(r["seed"]))
            if not group:
                continue
            data = extract_group(group)
            curve = kaplan_meier(data["calls"], data["hits_flags"])
            for step in curve:
                survival_rows.append(
                    {
                        "system": system,
                        "arm": arm,
                        "calls": str(step["calls"]),
                        "n_at_risk": str(step["n_at_risk"]),
                        "n_events": str(step["n_events"]),
                        "n_censored": str(step["n_censored"]),
                        "survival": f"{step['survival']:.6f}",
                        "greenwood_se": f"{step['greenwood_se']:.6f}",
                    }
                )

            kmed = km_median(curve)
            med_calls = type7([float(c) for c in data["calls"]], 0.5)
            iqr_calls = type7([float(c) for c in data["calls"]], 0.75) - type7(
                [float(c) for c in data["calls"]], 0.25
            )
            med_hits_only = (
                f"{type7(data['hits_only_calls'], 0.5):.2f}" if data["hits_only_calls"] else ""
            )
            med_wall = type7(data["wall"], 0.5)
            med_out = type7(data["outside"], 0.5)
            med_out_per_chg = (
                f"{type7(data['outside_per_charged'], 0.5):.9e}"
                if data["outside_per_charged"]
                else ""
            )

            if arm == "recommended" or rec_data is None:
                fisher_p_s = ""
                lr_chi2_s = ""
                lr_p_s = ""
                mw_u_s = ""
                mw_p_s = ""
            else:
                fp = fisher_exact_two_sided(
                    data["hits"], data["n"], rec_data["hits"], rec_data["n"]
                )
                lr_chi2, lr_p = logrank_two_sided(
                    data["calls"],
                    data["hits_flags"],
                    rec_data["calls"],
                    rec_data["hits_flags"],
                )
                mw_u, mw_p = mann_whitney_two_sided(
                    data["ranked_calls"], rec_data["ranked_calls"]
                )
                fisher_p_s = f"{fp:.6g}"
                lr_chi2_s = f"{lr_chi2:.6g}"
                lr_p_s = f"{lr_p:.6g}"
                mw_u_s = f"{mw_u:.1f}"
                mw_p_s = f"{mw_p:.6g}"

            summary_rows.append(
                {
                    "system": system,
                    "budget": str(budget),
                    "reference": f"{reference:.8f}",
                    "arm": arm,
                    "n": str(data["n"]),
                    "hits": str(data["hits"]),
                    "errors": str(data["errors"]),
                    "timeouts": str(data["timeouts"]),
                    "fisher_p": fisher_p_s,
                    "logrank_chi2": lr_chi2_s,
                    "logrank_p": lr_p_s,
                    "mw_u": mw_u_s,
                    "mw_p": mw_p_s,
                    "median_calls": f"{med_calls:.2f}",
                    "iqr_calls": f"{iqr_calls:.2f}",
                    "median_calls_hits_only": med_hits_only,
                    "km_median_calls": f"{kmed:.2f}" if kmed is not None else "",
                    "median_wall_s": f"{med_wall:.6f}",
                    "median_outside_s": f"{med_out:.6f}",
                    "median_outside_per_charged_s": med_out_per_chg,
                    "best_min": (
                        f"{data['best_min']:.8f}" if math.isfinite(data["best_min"]) else "inf"
                    ),
                    "config_hash": " ".join(data["config_hashes"]),
                    "n_tests": str(n_tests),
                }
            )

    return survival_rows, summary_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def format_p(text: str) -> str:
    if not text:
        return "-"
    p = float(text)
    if p >= 1.0:
        return "1"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.01:
        return f"{p:.3f}"
    return f"{p:.2e}"


def format_calls(text: str) -> str:
    val = float(text)
    return f"{val:.2f}".rstrip("0").rstrip(".")


def format_rate(text: str) -> str:
    if not text:
        return "-"
    us = float(text) * 1.0e6
    return f"{us:.3f}"


def render_markdown_table(summary_rows: list[dict[str, str]]) -> str:
    headers = [
        "System",
        "Budget",
        "Arm",
        "Hits",
        "Fisher p",
        "Log-rank p",
        "MW p",
        "Median calls",
        "Median outside/charged (us)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    last_sys = None
    for row in summary_rows:
        sys_key = row["system"]
        sys_label = SYSTEM_LABELS.get(sys_key, sys_key) if sys_key != last_sys else ""
        budget_label = row["budget"] if sys_key != last_sys else ""
        last_sys = sys_key
        cells = [
            sys_label,
            budget_label,
            f"`{row['arm']}`",
            f"{row['hits']}/{row['n']}",
            format_p(row["fisher_p"]),
            format_p(row["logrank_p"]),
            format_p(row["mw_p"]),
            format_calls(row["median_calls"]),
            format_rate(row["median_outside_per_charged_s"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_readme(readme_path: Path, summary_rows: list[dict[str, str]]) -> None:
    text = readme_path.read_text(encoding="utf-8")
    table = render_markdown_table(summary_rows)
    start_mark, end_mark = README_TABLE_MARKS
    block = f"{start_mark}\n{table}\n{end_mark}"
    if start_mark in text and end_mark in text:
        before, rest = text.split(start_mark, 1)
        _, after = rest.split(end_mark, 1)
        updated = before + block + after
    else:
        updated = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    readme_path.write_text(updated, encoding="utf-8")


def default_release_binary() -> Path:
    return (REPO_ROOT / ".." / "tmp" / "anneal-target" / "release" / "examples" / "g1_knockout").resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G1 single-knockout ablation runner and analyzer.")
    parser.add_argument("--binary", type=Path, default=default_release_binary())
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "bench" / "g1-knockout")
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/g1-knockout-logs"))
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--session", type=str, default="")
    args = parser.parse_args(argv)

    rows_path = args.out_dir / "rows.csv"
    survival_path = args.out_dir / "survival.csv"
    summary_path = args.out_dir / "summary.csv"
    sessions_path = args.out_dir / "sessions.jsonl"

    if args.summarize_only:
        rows = read_rows(rows_path)
        survival_rows, summary_rows = summarize_rows(rows)
        write_csv(survival_path, SURVIVAL_FIELDS, survival_rows)
        write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
        if args.update_readme:
            update_readme(args.out_dir / "README.md", summary_rows)
        return 0

    binary: Path = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"release binary not found: {binary}")
    if "release" not in binary.parts:
        raise SystemExit(f"refusing non-release binary: {binary}")

    specs = panel_specs()
    schedule = build_schedule()
    machine_id = hashed_machine_id()
    binary_sha = sha256_file(binary)
    session = args.session or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    config_hashes: dict[tuple[str, str], str] = {}
    for system in SYSTEM_ORDER:
        for arm in ARMS:
            config_hashes[(system, arm)] = fetch_config_hash(binary, system, arm)

    session_log_dir = args.log_dir / session
    session_log_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = read_rows(rows_path)
    done_keys = {
        (r["system"], r["arm"], int(r["seed"]))
        for r in existing_rows
        if r.get("session") == session
    }

    n_completed = len(done_keys)
    n_errors = sum(1 for r in existing_rows if r.get("session") == session and r.get("status") == "error")
    n_timeouts = sum(1 for r in existing_rows if r.get("session") == session and r.get("status") == "timeout")

    for item in schedule:
        key = (item["system"], item["arm"], item["seed"])
        if key in done_keys:
            continue
        spec = specs[item["system"]]
        row = run_one(
            binary=binary,
            system=item["system"],
            arm=item["arm"],
            seed=item["seed"],
            slot=item["slot"],
            sequence=item["sequence"],
            budget=spec["budget"],
            reference=spec["reference"],
            core=args.core,
            machine_id=machine_id,
            binary_sha256=binary_sha,
            config_hash=config_hashes[(item["system"], item["arm"])],
            session=session,
            log_dir=session_log_dir,
            timeout_s=args.timeout_s,
        )
        append_row(rows_path, row)
        n_completed += 1
        if row["status"] == "error":
            n_errors += 1
        elif row["status"] == "timeout":
            n_timeouts += 1
        print(
            f"[{item['sequence']:03d}/{len(schedule)}] {row['system']} {row['arm']} "
            f"seed={row['seed']} slot={row['slot']} hit={row['hit']} calls={row['calls']} "
            f"wall={row['wall_s']}s status={row['status']}",
            flush=True,
        )

    finished_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    session_record = {
        "session": session,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "head_commit": git_rev("HEAD"),
        "head_tree": git_rev("HEAD^{tree}"),
        "binary_sha256": binary_sha,
        "machine_id": machine_id,
        "core": args.core,
        "thread_env": THREAD_ENV_VARS,
        "systems": list(SYSTEM_ORDER),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "n_planned": len(schedule),
        "n_completed": n_completed,
        "n_errors": n_errors,
        "n_timeouts": n_timeouts,
    }
    sessions_path.parent.mkdir(parents=True, exist_ok=True)
    with sessions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(session_record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    all_rows = read_rows(rows_path)
    survival_rows, summary_rows = summarize_rows(all_rows, specs=specs)
    write_csv(survival_path, SURVIVAL_FIELDS, survival_rows)
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    if args.update_readme:
        update_readme(args.out_dir / "README.md", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
