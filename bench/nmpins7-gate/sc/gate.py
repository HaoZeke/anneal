#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Three-arm same-session gate for shard sc.

Cells are sc126-55, sc96-38, and BLJ7. The first two are the Sutton-Chen
panel names in ``scripts/metal_h2h/run_sc.py``. No panel definition names
BLJ7. The binary Lennard-Jones panel cell measured under that label is
``blj105n38`` (38 atoms, sigma_BB 1.05, budget 400000, Cambridge header
-177.2090846748), the first cell of ``examples/h2h_blj.rs``.

Arms A, B and C share seeds 2000-2011 and the geometry ``coords`` writes
for that seed. A is the develop binary. B and C are one kit binary. C sets
``general_symmetry=1``. B leaves the flag at its default. A second pair of
A runs on BLJ7, ordered ABBA, is the outside-time noise band.

Budgets and the hit window are the panel values. A seed that is still
running at 7200 s is killed and stored as a miss; its calls, best energy
and wall are kept when the process printed them. Medians and IQRs are
Hyndman-Fan type 7 from ``scripts/gmin_panel/run_panel.py``. Hit counts
use a one-sided Fisher exact test. Call counts use that module's
Mann-Whitney statistic, read one-sided for the higher-call direction.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SEEDS = tuple(range(2000, 2012))
HIT_TOL = 1.0e-3
WALL_CAP_S = 7200
BONFERRONI_M = 30
ALPHA = 0.05
CORE = "0"
FLAG = "general_symmetry"

PERMS = (
    ("A", "B", "C"),
    ("A", "C", "B"),
    ("B", "A", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("C", "B", "A"),
)

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "KMP_NUM_THREADS",
)

CELLS = (
    {
        "system": "sc126-55",
        "panel_system": "sc126-55",
        "driver": "h2h_sc",
        "budget": 1_000_000,
        "reference": -52884.6806,
    },
    {
        "system": "sc96-38",
        "panel_system": "sc96-38",
        "driver": "h2h_sc",
        "budget": 400_000,
        "reference": -8917.7056,
    },
    {
        "system": "BLJ7",
        "panel_system": "blj105n38",
        "driver": "h2h_blj",
        "budget": 400_000,
        "reference": -177.2090846748,
    },
)

FIELDS = (
    "system",
    "panel_system",
    "series",
    "arm",
    "slot",
    "seed",
    "budget",
    "hit",
    "calls",
    "calls_to_hit",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "outside_per_call",
    "capped",
    "exit_code",
    "core",
    "affinity",
    "machine_id",
    "coords_sha256",
    "binary_sha256",
    "source_sha",
    "flag",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel_stats():
    return load_module("gmin_panel_run", ROOT / "scripts" / "gmin_panel" / "run_panel.py")


def energy_hits(energy: float, target: float) -> bool:
    module = load_module("target_tol", ROOT / "scripts" / "target_tol.py")
    return bool(module.energy_hits(energy, target, HIT_TOL))


def abc_order(seed: int) -> tuple[str, str, str]:
    return PERMS[(seed - 2000) % 6]


def aa_order(seed: int) -> tuple[str, str]:
    if (seed - 2000) % 4 in (0, 3):
        return ("A", "A2")
    return ("A2", "A")


def fisher_fewer(hits_focus: int, n_focus: int, hits_other: int, n_other: int) -> float:
    """One-sided exact P(X <= hits_focus) under a shared hit rate.

    X is the focus arm's hit count in the hypergeometric table conditioned
    on the observed total.
    """
    total_n = n_focus + n_other
    total_hits = hits_focus + hits_other
    ways = math.comb(total_n, n_focus)
    lo = max(0, n_focus - (total_n - total_hits))
    hi = min(n_focus, total_hits)
    if hits_focus < lo or hits_focus > hi:
        raise ValueError("hit count is outside the hypergeometric support")
    mass = Fraction(0)
    for k in range(lo, hits_focus + 1):
        mass += Fraction(
            math.comb(total_hits, k) * math.comb(total_n - total_hits, n_focus - k),
            ways,
        )
    return float(mass)


def mann_whitney_higher(focus: list[float], other: list[float], stats) -> tuple[float, float, float]:
    """U of ``focus``, the repo's two-sided p, and the one-sided p that focus is higher."""
    u, two = stats.mann_whitney(focus, other)
    expected = len(focus) * len(other) / 2.0
    if u > expected:
        one = two / 2.0
    elif u < expected:
        one = 1.0 - two / 2.0
    else:
        one = 1.0
    return u, two, min(1.0, max(0.0, one))


def bonferroni(p: float) -> float:
    return min(1.0, BONFERRONI_M * p)


def type7(values: list[float], p: float, stats) -> float:
    return float(stats.type7(values, p))


def iqr(values: list[float], stats) -> float:
    return type7(values, 0.75, stats) - type7(values, 0.25, stats)


def machine_id_hash() -> str:
    raw = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit("machine id is empty")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def thread_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")) or key == FLAG:
            del env[key]
    for key in THREAD_VARS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["OMP_NESTED"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["KMP_DYNAMIC"] = "FALSE"
    return env


def require_object(sha: str) -> None:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        blob = subprocess.run(
            ["git", "cat-file", "-e", sha],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if blob.returncode != 0:
            raise SystemExit(f"git cat-file -e {sha} failed")


def packages_of(lock_text: str) -> dict[str, dict[str, str]]:
    wanted = {"anneal-core", "readcon-core", "rgmin", "rgpot-core", "rgpot-pair"}
    found: dict[str, dict[str, str]] = {}
    for block in lock_text.split("[[package]]")[1:]:
        name = version = source = ""
        for line in block.splitlines():
            if line.startswith("name = "):
                name = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("version = "):
                version = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("source = "):
                source = line.split("=", 1)[1].strip().strip('"')
        if name in wanted:
            found[name] = {"version": version, "source": source}
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"lock is missing {sorted(missing)}")
    return found


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({key: row[key] for key in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def child_affinity(pid: int, expected: str) -> str:
    status = Path(f"/proc/{pid}/status")
    comm = Path(f"/proc/{pid}/comm")
    last = ""
    for _ in range(4000):
        if not comm.exists():
            return last
        try:
            name = comm.read_text(encoding="utf-8").strip()
            text = status.read_text(encoding="utf-8")
        except OSError:
            return last
        for line in text.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                last = line.split(":", 1)[1].strip()
        if name == expected and last:
            return last
        time.sleep(0.0005)
    return last


def parse_report(stdout: str) -> dict[str, str] | None:
    lines = [line.strip() for line in stdout.splitlines() if "," in line]
    for index, line in enumerate(lines):
        if line.startswith("system,code,seed,") and index + 1 < len(lines):
            header = line.split(",")
            values = lines[index + 1].split(",")
            if len(values) < len(header):
                return None
            return dict(zip(header, values))
    return None


def run_command(command: list[str], env: dict[str, str], log: Path, timeout: float | None) -> tuple[int, str, str, str, bool, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    expected = Path(command[-4]).name if command[0] == "taskset" else Path(command[0]).name
    # command is taskset -c 0 binary ...
    expected = Path(command[3]).name
    affinity = child_affinity(proc.pid, expected)
    capped = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        capped = True
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - started
    log.write_text(
        " ".join(command) + f"\nexit={proc.returncode} capped={int(capped)} affinity={affinity}\n"
        + stdout
        + "\n"
        + stderr,
        encoding="utf-8",
    )
    return proc.returncode if proc.returncode is not None else -9, stdout, stderr, affinity, capped, elapsed


def budget_of(binary: Path, panel_system: str, env: dict[str, str]) -> tuple[int, float]:
    proc = subprocess.run(
        ["taskset", "-c", CORE, str(binary), "budget", panel_system],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"budget {panel_system} failed\n{proc.stderr}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"budget {panel_system} returned {proc.stdout!r}")
    return int(lines[0]), float(lines[1])


def write_coords(binary: Path, panel_system: str, seed: int, dest: Path, env: dict[str, str]) -> bytes:
    proc = subprocess.run(
        ["taskset", "-c", CORE, str(binary), "coords", panel_system, str(seed), str(dest)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not dest.is_file():
        raise SystemExit(f"coords {panel_system} {seed} failed\n{proc.stderr}")
    return dest.read_bytes()


def cell_by_name(name: str) -> dict:
    for cell in CELLS:
        if cell["system"] == name:
            return cell
    raise KeyError(name)


def fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}g}"


def pfmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if value >= 1.0:
        return "1"
    return f"{value:.6g}"


def group_rows(rows: list[dict[str, str]], system: str, series: str, arm: str) -> list[dict[str, str]]:
    chosen = [
        row
        for row in rows
        if row["system"] == system and row["series"] == series and row["arm"] == arm
    ]
    return sorted(chosen, key=lambda row: int(row["seed"]))


def numeric(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def ranked(rows: list[dict[str, str]], budget: int, stats) -> list[float] | None:
    adapted = []
    for row in rows:
        if row["capped"] == "1" and (row["calls"] == "" or int(row["calls"]) < budget):
            return None
        adapted.append(
            {
                "hit": row["hit"],
                "charged": row["calls"],
                "evals_to_hit": row["calls_to_hit"] if row["calls_to_hit"] else row["budget"],
            }
        )
    return stats.ranked_calls(adapted, budget)


def summarise_pair(label: str, left_name: str, right_name: str, left: list[dict[str, str]], right: list[dict[str, str]], budget: int, stats) -> dict:
    hits_l = sum(row["hit"] == "1" for row in left)
    hits_r = sum(row["hit"] == "1" for row in right)
    fisher_one = fisher_fewer(hits_l, len(left), hits_r, len(right))
    fisher_two = float(stats.fisher(hits_l, len(left), hits_r, len(right)))
    calls_l = ranked(left, budget, stats)
    calls_r = ranked(right, budget, stats)
    if calls_l is None or calls_r is None:
        call_u = call_two = call_one = None
        call_med_l = call_med_r = call_iqr_l = call_iqr_r = None
        calls_censored = True
    else:
        calls_censored = False
        call_u, call_two, call_one = mann_whitney_higher(calls_l, calls_r, stats)
        finite_l = [value for value in calls_l if math.isfinite(value)]
        finite_r = [value for value in calls_r if math.isfinite(value)]
        # Medians include the +inf rank of a full-budget miss by using the
        # calls-to-hit value the driver stored (the budget on a miss).
        stored_l = [float(row["calls_to_hit"]) for row in left]
        stored_r = [float(row["calls_to_hit"]) for row in right]
        call_med_l = type7(stored_l, 0.5, stats)
        call_med_r = type7(stored_r, 0.5, stats)
        call_iqr_l = iqr(stored_l, stats)
        call_iqr_r = iqr(stored_r, stats)
        del finite_l, finite_r
    out_l = numeric(left, "outside_per_call")
    out_r = numeric(right, "outside_per_call")
    out_med_l = type7(out_l, 0.5, stats)
    out_med_r = type7(out_r, 0.5, stats)
    out_iqr_l = iqr(out_l, stats)
    out_iqr_r = iqr(out_r, stats)
    _u, out_two, out_one = mann_whitney_higher(out_l, out_r, stats)
    if out_med_r == 0.0:
        ratio = 1.0 if out_med_l == 0.0 else math.inf
    else:
        ratio = out_med_l / out_med_r
    return {
        "label": label,
        "left": left_name,
        "right": right_name,
        "n_left": len(left),
        "n_right": len(right),
        "hits_left": hits_l,
        "hits_right": hits_r,
        "fisher_one": fisher_one,
        "fisher_two": fisher_two,
        "fisher_one_bon": bonferroni(fisher_one),
        "calls_censored": calls_censored,
        "call_u": call_u,
        "call_two": call_two,
        "call_one": call_one,
        "call_one_bon": None if call_one is None else bonferroni(call_one),
        "call_med_left": call_med_l,
        "call_med_right": call_med_r,
        "call_iqr_left": call_iqr_l,
        "call_iqr_right": call_iqr_r,
        "out_med_left": out_med_l,
        "out_med_right": out_med_r,
        "out_iqr_left": out_iqr_l,
        "out_iqr_right": out_iqr_r,
        "out_ratio": ratio,
        "out_two": out_two,
        "out_one": out_one,
        "out_two_bon": bonferroni(out_two),
        "out_one_bon": bonferroni(out_one),
    }


def loses(summary: dict) -> tuple[bool, list[str]]:
    reasons = []
    if summary["fisher_one_bon"] < ALPHA:
        reasons.append("hits")
    if summary["calls_censored"]:
        reasons.append("calls-censored")
    elif summary["call_one_bon"] < ALPHA:
        reasons.append("calls")
    if summary["out_ratio"] > 1.1:
        reasons.append("outside")
    return bool(reasons), reasons


def slot_counts(rows: list[dict[str, str]], system: str, series: str) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if row["system"] == system and row["series"] == series:
            key = (row["slot"], row["arm"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def assert_balance(rows: list[dict[str, str]]) -> None:
    for cell in CELLS:
        counts = slot_counts(rows, cell["system"], "abc")
        for slot in ("0", "1", "2"):
            for arm in ("A", "B", "C"):
                if counts.get((slot, arm), 0) != 4:
                    raise SystemExit(f"{cell['system']} abc balance {counts}")
    aa = slot_counts(rows, "BLJ7", "aa")
    for slot in ("0", "1"):
        for arm in ("A", "A2"):
            if aa.get((slot, arm), 0) != 6:
                raise SystemExit(f"BLJ7 aa balance {aa}")


def aa_band(rows: list[dict[str, str]], stats) -> dict[str, float]:
    by_seed: dict[str, dict[str, float]] = {}
    for row in group_rows(rows, "BLJ7", "aa", "A") + group_rows(rows, "BLJ7", "aa", "A2"):
        by_seed.setdefault(row["seed"], {})[row["arm"]] = float(row["outside_per_call"])
    ratios = []
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        base = pair["A"]
        ratios.append(pair["A2"] / base if base else math.inf)
    return {
        "n": float(len(ratios)),
        "median": type7(ratios, 0.5, stats),
        "q1": type7(ratios, 0.25, stats),
        "q3": type7(ratios, 0.75, stats),
        "min": min(ratios),
        "max": max(ratios),
    }


def inside_band(ratio: float, band: dict[str, float]) -> str:
    if band["q1"] <= ratio <= band["q3"]:
        return "inside the A/A interquartile range"
    if band["min"] <= ratio <= band["max"]:
        return "outside the A/A interquartile range and inside the A/A min-max range"
    return "outside the A/A min-max range"


def render_report(rows: list[dict[str, str]], session: dict, stats) -> str:
    assert_balance(rows)
    band = aa_band(rows, stats)
    lines = [
        "# nmpins7 gate, shard sc",
        "",
        "Same session, one core, one thread. Seeds 2000-2011. Hit window +0.001.",
        "Wall cap 7200 s. A capped seed is a miss. Medians and IQRs are Hyndman-Fan type 7.",
        "B versus A loses on hits when the one-sided Fisher probability that B has fewer hits,",
        f"multiplied by {BONFERRONI_M}, is below {ALPHA}; on calls when the one-sided Mann-Whitney",
        f"probability that B's calls-to-hit are higher, multiplied by {BONFERRONI_M}, is below {ALPHA};",
        "and on outside time when the median outside-potential time per call of B exceeds 1.1 times A.",
        "C versus A is the same arithmetic with no pass or fail.",
        "",
        "BLJ7 is not a name in the panel tables. The cell measured here is blj105n38,",
        "budget 400000, reference -177.2090846748. sc126-55 and sc96-38 use the budgets",
        "and references in `scripts/metal_h2h/run_sc.py`.",
        "",
        "## Provenance",
        "",
        f"Base `{session['base']}`. Arm A source `{session['arm_a_source']}`.",
        f"Kit source `{session['kit_source']}`, tree `{session['kit_tree']}`.",
        f"Kit build `{session['kit_build']}` reads `{FLAG}` and is the binary for B and C.",
        f"Binary Lennard-Jones driver commit `{session['blj_driver_commit']}`, blob `{session['blj_driver_blob']}`.",
        "On this commit `BinaryLennardJones::doye_meyer` is `cfg(test)`, so the built driver calls `BinaryLennardJones::new` with that constructor's arguments.",
        f"rgpot `{session['rgpot']}`, librgpot.so.3.7.0 sha256 `{session['rgpot_sha256']}`.",
        f"h2h_sc A sha256 `{session['binary_sha']['h2h_sc_a']}`.",
        f"h2h_sc kit sha256 `{session['binary_sha']['h2h_sc_kit']}`.",
        f"h2h_blj A sha256 `{session['binary_sha']['h2h_blj_a']}`.",
        f"h2h_blj kit sha256 `{session['binary_sha']['h2h_blj_kit']}`.",
        f"rustc `{session['rustc']}`.",
        "Packages: "
        + ", ".join(
            f"{name} {meta['version']}" for name, meta in sorted(session["packages"].items())
        )
        + ".",
        f"Machine id sha256 `{session['machine_id']}`. Core {session['core']}.",
        "Thread variables set to 1: " + ", ".join(session["thread_vars"]) + ".",
        f"Session `{session['session']}`.",
        "",
        "## Results",
        "",
        "| cell | arm | hits | median calls (IQR) | median outside/call (IQR) |",
        "| --- | --- | --- | --- | --- |",
    ]
    summaries = []
    for cell in CELLS:
        arms = {}
        for arm in ("A", "B", "C"):
            grouped = group_rows(rows, cell["system"], "abc", arm)
            if len(grouped) != 12:
                raise SystemExit(f"{cell['system']} {arm} has {len(grouped)} rows")
            arms[arm] = grouped
            calls = [float(row["calls_to_hit"]) for row in grouped]
            outside = numeric(grouped, "outside_per_call")
            hits = sum(row["hit"] == "1" for row in grouped)
            lines.append(
                f"| {cell['system']} | {arm} | {hits}/12 | {fmt(type7(calls, 0.5, stats))} ({fmt(iqr(calls, stats))}) | {fmt(type7(outside, 0.5, stats))} ({fmt(iqr(outside, stats))}) |"
            )
        b_vs_a = summarise_pair("B-A", "B", "A", arms["B"], arms["A"], cell["budget"], stats)
        c_vs_a = summarise_pair("C-A", "C", "A", arms["C"], arms["A"], cell["budget"], stats)
        failed, reasons = loses(b_vs_a)
        where = inside_band(b_vs_a["out_ratio"], band)
        b_vs_a["verdict"] = "loses on " + ", ".join(reasons) if failed else "passes"
        b_vs_a["band"] = where
        b_vs_a["system"] = cell["system"]
        c_vs_a["system"] = cell["system"]
        summaries.append((b_vs_a, c_vs_a))
    lines.extend(["", "## B versus A", ""])
    for b_vs_a, _c in summaries:
        call_p = "censored" if b_vs_a["calls_censored"] else (
            f"one-sided {pfmt(b_vs_a['call_one'])}, Bonferroni {pfmt(b_vs_a['call_one_bon'])}, two-sided {pfmt(b_vs_a['call_two'])}"
        )
        lines.append(
            f"{b_vs_a['system']}: hits {b_vs_a['hits_left']}/12 against {b_vs_a['hits_right']}/12, "
            f"Fisher one-sided {pfmt(b_vs_a['fisher_one'])}, Bonferroni {pfmt(b_vs_a['fisher_one_bon'])}, "
            f"two-sided {pfmt(b_vs_a['fisher_two'])}. "
            f"Calls {call_p}. "
            f"Outside median ratio {fmt(b_vs_a['out_ratio'])}, "
            f"Mann-Whitney two-sided {pfmt(b_vs_a['out_two'])}, Bonferroni {pfmt(b_vs_a['out_two_bon'])}, "
            f"one-sided higher {pfmt(b_vs_a['out_one'])}, Bonferroni {pfmt(b_vs_a['out_one_bon'])}. "
            f"The ratio is {b_vs_a['band']}. Verdict: {b_vs_a['verdict']}."
        )
    lines.extend(["", "## C versus A", ""])
    for _b, c_vs_a in summaries:
        call_p = "censored" if c_vs_a["calls_censored"] else (
            f"one-sided {pfmt(c_vs_a['call_one'])}, Bonferroni {pfmt(c_vs_a['call_one_bon'])}, two-sided {pfmt(c_vs_a['call_two'])}"
        )
        lines.append(
            f"{c_vs_a['system']}: hits {c_vs_a['hits_left']}/12 against {c_vs_a['hits_right']}/12, "
            f"Fisher one-sided {pfmt(c_vs_a['fisher_one'])}, Bonferroni {pfmt(c_vs_a['fisher_one_bon'])}, "
            f"two-sided {pfmt(c_vs_a['fisher_two'])}. "
            f"Calls {call_p}. "
            f"Outside median ratio {fmt(c_vs_a['out_ratio'])}, "
            f"Mann-Whitney two-sided {pfmt(c_vs_a['out_two'])}, Bonferroni {pfmt(c_vs_a['out_two_bon'])}, "
            f"one-sided higher {pfmt(c_vs_a['out_one'])}, Bonferroni {pfmt(c_vs_a['out_one_bon'])}."
        )
    lines.extend(
        [
            "",
            "## A/A noise band on BLJ7",
            "",
            f"Twelve pairs of the arm A binary, order ABBA. Per-seed ratio of outside time per call, label A2 over label A: "
            f"median {fmt(band['median'])}, IQR {fmt(band['q1'])} to {fmt(band['q3'])}, "
            f"range {fmt(band['min'])} to {fmt(band['max'])}. The ratio is outside time per call of label A2 divided by label A.",
            "",
            "## Slot counts",
            "",
            "Each of A, B and C occupies each of the three slots four times on every cell.",
            "On the BLJ7 A/A series each of A and A2 occupies each of the two slots six times.",
            "",
        ]
    )
    return "\n".join(lines)


def selftest() -> None:
    stats = panel_stats()
    if type7([1, 2, 3, 4], 0.5, stats) != 2.5:
        raise SystemExit("type 7 median")
    if abs(type7([1, 2, 3, 4], 0.25, stats) - 1.75) > 1e-12:
        raise SystemExit("type 7 quartile")
    if fisher_fewer(12, 12, 12, 12) != 1.0:
        raise SystemExit("fisher tie")
    if not fisher_fewer(0, 12, 12, 12) < 1e-6:
        raise SystemExit("fisher separation")
    _u, two, one = mann_whitney_higher([1, 2, 3], [1, 2, 3], stats)
    if two != 1.0 or one != 1.0:
        raise SystemExit("mann whitney tie")
    _u, two, one = mann_whitney_higher([10, 11, 12, 13], [0, 1, 2, 3], stats)
    if not (one < two and one < 0.05):
        raise SystemExit(f"mann whitney direction {one} {two}")
    counts: dict[tuple[str, str], int] = {}
    for seed in SEEDS:
        for slot, arm in enumerate(abc_order(seed)):
            counts[(str(slot), arm)] = counts.get((str(slot), arm), 0) + 1
    if any(counts[(str(slot), arm)] != 4 for slot in range(3) for arm in ("A", "B", "C")):
        raise SystemExit(f"perm balance {counts}")
    aa: dict[tuple[str, str], int] = {}
    for seed in SEEDS:
        for slot, arm in enumerate(aa_order(seed)):
            aa[(str(slot), arm)] = aa.get((str(slot), arm), 0) + 1
    if any(aa[(str(slot), arm)] != 6 for slot in range(2) for arm in ("A", "A2")):
        raise SystemExit(f"aa balance {aa}")
    # The recorded Sutton-Chen budgets are the module the panel already ships.
    run_sc = load_module("run_sc", ROOT / "scripts" / "metal_h2h" / "run_sc.py")
    for cell in CELLS[:2]:
        if run_sc.PAPER_BUDGETS[cell["system"]] != cell["budget"]:
            raise SystemExit(f"budget drift {cell['system']}")
        if run_sc.REFERENCES[cell["system"]] != cell["reference"]:
            raise SystemExit(f"reference drift {cell['system']}")
    print("selftest ok", flush=True)


def source_for(arm: str, session: dict) -> str:
    if arm in ("A", "A2"):
        return session["arm_a_source"]
    return session["kit_build"]


def binary_for(arm: str, driver: str, bins: dict[str, Path]) -> Path:
    if driver == "h2h_sc":
        return bins["h2h_sc_a"] if arm in ("A", "A2") else bins["h2h_sc_kit"]
    return bins["h2h_blj_a"] if arm in ("A", "A2") else bins["h2h_blj_kit"]


def execute(args: list[str]) -> int:
    if args and args[0] == "selftest":
        selftest()
        return 0
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "report", "selftest"))
    parser.add_argument("--a-sc", type=Path)
    parser.add_argument("--kit-sc", type=Path)
    parser.add_argument("--a-blj", type=Path)
    parser.add_argument("--kit-blj", type=Path)
    parser.add_argument("--rgpot", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--rows", type=Path, default=Path(__file__).resolve().parent / "rows.csv")
    parser.add_argument("--sessions", type=Path, default=Path(__file__).resolve().parent / "sessions.jsonl")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent / "report.md")
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--arm-a-source", default="5eb79cdcb0a986984dc7b487e5a3cc93224246f9")
    parser.add_argument("--kit-source", default="4cd811004c67f26b0e216664cb51352f04654c09")
    parser.add_argument("--kit-tree", default="51c1a7e0180585d078c3457b5550b2963d1ab752")
    parser.add_argument("--kit-build")
    parser.add_argument("--blj-driver-commit", default="8de2a95c3ff3eb2b173b88b4a8cb48c144cbfe19")
    parser.add_argument("--rgpot-sha", default="8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e")
    parser.add_argument("--base", default="5eb79cdcb0a986984dc7b487e5a3cc93224246f9")
    parsed = parser.parse_args(args)
    if parsed.command == "selftest":
        selftest()
        return 0
    if parsed.command == "report":
        rows = read_rows(parsed.rows)
        header = json.loads(parsed.sessions.read_text(encoding="utf-8").splitlines()[0])
        for sha in (
            header["base"],
            header["arm_a_source"],
            header["kit_source"],
            header["kit_tree"],
            header["kit_build"],
            header["blj_driver_commit"],
            header["blj_driver_blob"],
            header["rgpot"],
        ):
            require_object(sha)
        text = render_report(rows, header, panel_stats())
        parsed.report.write_text(text, encoding="utf-8")
        print(parsed.report, flush=True)
        return 0
    if os.sched_getaffinity(0) != {0}:
        raise SystemExit(f"affinity is {sorted(os.sched_getaffinity(0))}, expected core 0")
    if parsed.logs is None or parsed.a_sc is None or parsed.kit_sc is None or parsed.a_blj is None or parsed.kit_blj is None or parsed.rgpot is None or parsed.lock is None or parsed.kit_build is None:
        raise SystemExit("run needs binaries, rgpot, lock, kit build and a log directory")
    try:
        parsed.logs.resolve().relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("log directory is inside the repository")
    selftest()
    for sha in (parsed.base, parsed.arm_a_source, parsed.kit_source, parsed.kit_build, parsed.blj_driver_commit, parsed.rgpot_sha):
        require_object(sha)
    blob = subprocess.run(
        ["git", "rev-parse", f"{parsed.blj_driver_commit}:examples/h2h_blj.rs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    require_object(blob)
    kit_tree = subprocess.run(
        ["git", "rev-parse", f"{parsed.kit_source}^{{tree}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if kit_tree != parsed.kit_tree:
        raise SystemExit(f"kit tree {kit_tree} is not {parsed.kit_tree}")
    bins = {
        "h2h_sc_a": parsed.a_sc,
        "h2h_sc_kit": parsed.kit_sc,
        "h2h_blj_a": parsed.a_blj,
        "h2h_blj_kit": parsed.kit_blj,
    }
    binary_sha = {name: sha256_file(path) for name, path in bins.items()}
    if binary_sha["h2h_sc_a"] == binary_sha["h2h_sc_kit"] or binary_sha["h2h_blj_a"] == binary_sha["h2h_blj_kit"]:
        raise SystemExit("arm A and the kit produced the same binary")
    env = thread_env()
    env["RGPOT_SC_LIB"] = str(parsed.rgpot)
    for cell in CELLS:
        binary = bins["h2h_sc_a"] if cell["driver"] == "h2h_sc" else bins["h2h_blj_a"]
        kit_binary = bins["h2h_sc_kit"] if cell["driver"] == "h2h_sc" else bins["h2h_blj_kit"]
        for candidate in (binary, kit_binary):
            budget, reference = budget_of(candidate, cell["panel_system"], env)
            if budget != cell["budget"] or abs(reference - cell["reference"]) > 1e-8:
                raise SystemExit(f"{candidate} {cell['panel_system']} budget {budget} reference {reference}")
    parsed.logs.mkdir(parents=True, exist_ok=True)
    probe_a = parsed.logs / "probe-a.xyz"
    probe_k = parsed.logs / "probe-k.xyz"
    write_coords(bins["h2h_sc_a"], "sc96-38", 2000, probe_a, env)
    write_coords(bins["h2h_sc_kit"], "sc96-38", 2000, probe_k, env)
    if probe_a.read_bytes() != probe_k.read_bytes():
        raise SystemExit("sc96-38 seed 2000 starts differ")
    energies = []
    for binary in (bins["h2h_sc_a"], bins["h2h_sc_kit"]):
        proc = subprocess.run(
            ["taskset", "-c", CORE, str(binary), "energy", "sc96-38", str(probe_a)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(f"probe energy failed\n{proc.stderr}")
        energies.append(float(proc.stdout.strip().split()[-1]))
    if abs(energies[0] - energies[1]) > 1e-8:
        raise SystemExit(f"Sutton-Chen energies differ {energies}")
    rustc = subprocess.run(["rustc", "+1.88.0", "--version"], capture_output=True, text=True, check=True).stdout.strip()
    session_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    machine = machine_id_hash()
    session = {
        "kind": "session",
        "session": session_id,
        "base": parsed.base,
        "arm_a_source": parsed.arm_a_source,
        "kit_source": parsed.kit_source,
        "kit_tree": kit_tree,
        "kit_build": parsed.kit_build,
        "blj_driver_commit": parsed.blj_driver_commit,
        "blj_driver_blob": blob,
        "rgpot": parsed.rgpot_sha,
        "rgpot_sha256": sha256_file(parsed.rgpot),
        "binary_sha": binary_sha,
        "rustc": rustc,
        "packages": packages_of(parsed.lock.read_text(encoding="utf-8")),
        "machine_id": machine,
        "core": CORE,
        "thread_vars": list(THREAD_VARS),
        "hit_tol": HIT_TOL,
        "wall_cap_s": WALL_CAP_S,
        "bonferroni_m": BONFERRONI_M,
        "flag": FLAG,
        "cells": [
            {"system": cell["system"], "panel_system": cell["panel_system"], "budget": cell["budget"], "reference": cell["reference"]}
            for cell in CELLS
        ],
    }
    if parsed.sessions.is_file() and parsed.sessions.stat().st_size:
        header = json.loads(parsed.sessions.read_text(encoding="utf-8").splitlines()[0])
        if header.get("kind") != "session":
            raise SystemExit("sessions file does not start with a session record")
        if header.get("binary_sha") != binary_sha or header.get("rgpot_sha256") != sha256_file(parsed.rgpot):
            raise SystemExit("binaries changed since the session header")
        session = header
        session_id = header["session"]
        machine = header["machine_id"]
    else:
        append_jsonl(parsed.sessions, session)
    done = {(row["system"], row["series"], row["arm"], row["seed"]) for row in read_rows(parsed.rows)}
    work = parsed.logs / "coords"
    work.mkdir(parents=True, exist_ok=True)

    def one(cell: dict, series: str, slot: int, arm: str, seed: int, coords_sha: str) -> None:
        key = (cell["system"], series, arm, str(seed))
        if key in done:
            print(f"skip {key}", flush=True)
            return
        binary = binary_for(arm, cell["driver"], bins)
        run_env = thread_env(env)
        run_env["RGPOT_SC_LIB"] = str(parsed.rgpot)
        flag = "0"
        if arm == "C":
            run_env[FLAG] = "1"
            flag = "1"
        log = parsed.logs / cell["system"] / f"{series}-{arm}-{seed}.log"
        command = ["taskset", "-c", CORE, str(binary), "run", cell["panel_system"], str(seed)]
        print(f"start {cell['system']} {series} {arm} slot {slot} seed {seed}", flush=True)
        code, stdout, _stderr, affinity, capped, elapsed = run_command(command, run_env, log, WALL_CAP_S)
        parsed_row = parse_report(stdout)
        if parsed_row is None and not capped:
            raise SystemExit(f"{key} exited {code} without a row")
        if parsed_row is not None:
            best = float(parsed_row["best"])
            hit = energy_hits(best, cell["reference"])
            calls = parsed_row["charged"]
            calls_to_hit = parsed_row["evals_to_hit"]
            wall = parsed_row["wall_s"]
            pot = parsed_row["pot_s"]
            if "outside_s" in parsed_row:
                outside = parsed_row["outside_s"]
            else:
                outside = f"{max(float(wall) - float(pot), 0.0):.6f}"
            per = float(outside) / max(int(calls), 1)
            if int(parsed_row["budget"]) != cell["budget"]:
                raise SystemExit(f"{key} budget {parsed_row['budget']}")
            if parsed_row["system"] != cell["panel_system"]:
                raise SystemExit(f"{key} system {parsed_row['system']}")
            if capped:
                hit = False
        else:
            best = ""
            hit = False
            calls = ""
            calls_to_hit = ""
            wall = f"{elapsed:.6f}"
            pot = ""
            outside = ""
            per = ""
        if affinity != CORE:
            raise SystemExit(f"{key} affinity {affinity!r}")
        row = {
            "system": cell["system"],
            "panel_system": cell["panel_system"],
            "series": series,
            "arm": arm,
            "slot": str(slot),
            "seed": str(seed),
            "budget": str(cell["budget"]),
            "hit": "1" if hit else "0",
            "calls": calls,
            "calls_to_hit": calls_to_hit,
            "best": f"{best:.8f}" if best != "" else "",
            "wall_s": wall,
            "pot_s": pot,
            "outside_s": outside,
            "outside_per_call": f"{per:.9f}" if per != "" else "",
            "capped": "1" if capped else "0",
            "exit_code": str(code),
            "core": CORE,
            "affinity": affinity,
            "machine_id": machine,
            "coords_sha256": coords_sha,
            "binary_sha256": sha256_file(binary),
            "source_sha": source_for(arm, session),
            "flag": flag,
        }
        append_row(parsed.rows, row)
        payload = {"kind": "run", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "session": session_id}
        payload.update(row)
        append_jsonl(parsed.sessions, payload)
        done.add(key)
        print(
            f"done {cell['system']} {series} {arm} {seed} hit={row['hit']} calls={row['calls']} wall_s={row['wall_s']} outside_per_call={row['outside_per_call']}",
            flush=True,
        )

    for seed in SEEDS:
        for cell in CELLS:
            a_bin = bins["h2h_sc_a"] if cell["driver"] == "h2h_sc" else bins["h2h_blj_a"]
            k_bin = bins["h2h_sc_kit"] if cell["driver"] == "h2h_sc" else bins["h2h_blj_kit"]
            a_path = work / f"{cell['system']}-{seed}-a.xyz"
            k_path = work / f"{cell['system']}-{seed}-k.xyz"
            a_bytes = write_coords(a_bin, cell["panel_system"], seed, a_path, env)
            k_bytes = write_coords(k_bin, cell["panel_system"], seed, k_path, env)
            flagged = thread_env(env)
            flagged["RGPOT_SC_LIB"] = env["RGPOT_SC_LIB"]
            flagged[FLAG] = "1"
            c_path = work / f"{cell['system']}-{seed}-c.xyz"
            c_bytes = write_coords(k_bin, cell["panel_system"], seed, c_path, flagged)
            if a_bytes != k_bytes or a_bytes != c_bytes:
                raise SystemExit(f"start geometry differs for {cell['system']} seed {seed}")
            digest = sha256_bytes(a_bytes)
            for slot, arm in enumerate(abc_order(seed)):
                one(cell, "abc", slot, arm, seed, digest)
        blj = cell_by_name("BLJ7")
        a_bytes = write_coords(bins["h2h_blj_a"], blj["panel_system"], seed, work / f"BLJ7-{seed}-aa.xyz", env)
        digest = sha256_bytes(a_bytes)
        for slot, arm in enumerate(aa_order(seed)):
            one(blj, "aa", slot, arm, seed, digest)
    rows = read_rows(parsed.rows)
    text = render_report(rows, session, panel_stats())
    parsed.report.write_text(text, encoding="utf-8")
    print(f"report {parsed.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(execute(sys.argv[1:]))
