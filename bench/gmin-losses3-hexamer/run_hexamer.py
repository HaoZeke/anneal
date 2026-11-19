#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Panel 1 water-hexamer guard for the gmin-losses3 gate.

A is develop 5eb79cdc. B is that tree plus the pair-scale and plateau-heat
kit. AP is a second release build of A. The search is the default
``gfn2_water_h2h`` path: no ``H2H_*`` variable is set. One process runs
at a time on core 0. A hit is a finite engine energy at or below the
reference plus 0.001. A miss is charged the budget, which ranks it worst
on the call axis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

BASE = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
BASE_TREE = "7a2ba49b37241a8fe42b2c45c226374de6d92274"
B_TREE = "70871f7307271806926b2aec9870e838dffbe924"
B_MBOX_SHA256 = "14d0cf394fa5146bd1f769f5e8ab5a26a9afc7d8fe7dbba3e9cd118698a7288d"
HEXAMER_REF = -829.8522275088
HIT_TOL = 0.001
HEXAMER_BUDGET = 4000
SEEDS = list(range(2000, 2012))
N_SEEDS = 12
CORE = "0"
ALPHA = 0.05
PANEL = "1"
SYSTEM = "water_hexamer"
BAND = "ab"

# Twelve seeds. Each (slot, arm) appears four times. Dropping AP leaves
# AB, BA, BA, AB, repeated three times.
ORDER = [
    ("A", "AP", "B"),
    ("AP", "B", "A"),
    ("AP", "B", "A"),
    ("A", "AP", "B"),
    ("A", "B", "AP"),
    ("B", "A", "AP"),
    ("B", "A", "AP"),
    ("A", "B", "AP"),
    ("AP", "A", "B"),
    ("B", "AP", "A"),
    ("B", "AP", "A"),
    ("AP", "A", "B"),
]

ROW_FIELDS = [
    "panel",
    "system",
    "arm",
    "seed",
    "slot",
    "core",
    "hit",
    "calls",
    "wall",
    "best",
    "best_bits",
    "budget",
    "machine_id_sha256",
    "base_commit",
    "base_tree",
    "harness_blob",
    "best_text",
    "charged",
    "band",
]

THREAD_ONES = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
)


def energy_hits(energy: float, target: float, tol: float = HIT_TOL) -> bool:
    limit = target + tol
    return math.isfinite(energy) and math.isfinite(limit) and energy <= limit


def fisher_twosided(a_hit: int, a_n: int, b_hit: int, b_n: int) -> float:
    """Two-sided Fisher exact p, summed over tables at least as unlikely."""
    lo = max(0, a_hit + b_hit - b_n)
    hi = min(a_hit + b_hit, a_n)
    point = {}
    for k in range(lo, hi + 1):
        point[k] = math.comb(a_n, k) * math.comb(b_n, a_hit + b_hit - k)
    total = sum(point.values())
    threshold = point[a_hit]
    return sum(v for v in point.values() if v <= threshold) / total


def exact_mann_whitney(left: list[float], right: list[float]) -> tuple[float, bool]:
    """Exact two-sided Mann-Whitney p, and whether right ranks higher.

    Ranks use the mid-rank for ties. The p-value is the share of subsets
    whose rank sum is at least as far from the mean as the left sample.
    A larger value ranks higher. Callers pass misses as the worst values.
    """
    n1 = len(left)
    n2 = len(right)
    n = n1 + n2
    paired = [(float(v), 0) for v in left] + [(float(v), 1) for v in right]
    order = sorted(range(n), key=lambda i: (paired[i][0], i))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and paired[order[j + 1]][0] == paired[order[i]][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    doubled = [int(round(r * 2)) for r in ranks]
    if sum(doubled) != n * (n + 1):
        raise RuntimeError("rank total drifted")
    max_s = sum(doubled)
    dp = [[0] * (max_s + 1) for _ in range(n1 + 1)]
    dp[0][0] = 1
    for value in doubled:
        for k in range(n1, 0, -1):
            prev = dp[k - 1]
            row = dp[k]
            for s in range(max_s - value, -1, -1):
                count = prev[s]
                if count:
                    row[s + value] += count
    total = math.comb(n, n1)
    if sum(dp[n1]) != total:
        raise RuntimeError("subset count drifted")
    observed = int(round(sum(ranks[:n1]) * 2))
    mean = n1 * (n + 1)
    extreme = abs(observed - mean)
    tail = sum(c for s, c in enumerate(dp[n1]) if abs(s - mean) >= extreme)
    right_sum = sum(ranks[n1:])
    right_higher = right_sum > n2 * (n + 1) / 2
    return tail / total, right_higher


def type7(values: list[float], p: float) -> float:
    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return xs[0]
    h = 1 + (n - 1) * p
    left = int(math.floor(h))
    right = int(math.ceil(h))
    return xs[left - 1] + (h - left) * (xs[right - 1] - xs[left - 1])


def median(values: list[float]) -> float:
    return type7(values, 0.5)


def bits(value: float) -> str:
    return format(struct.unpack(">Q", struct.pack(">d", value))[0], "016x")


def balanced(order: list[tuple[str, ...]], expected: int) -> bool:
    counts: dict[tuple[int, str], int] = {}
    width = len(order[0])
    for arms in order:
        for slot, arm in enumerate(arms):
            counts[(slot, arm)] = counts.get((slot, arm), 0) + 1
    names = set(order[0])
    return all(counts.get((slot, arm)) == expected for slot in range(width) for arm in names)


def slot_counts(order: list[tuple[str, ...]]) -> dict[str, int]:
    counts: dict[tuple[int, str], int] = {}
    for arms in order:
        for slot, arm in enumerate(arms):
            counts[(slot, arm)] = counts.get((slot, arm), 0) + 1
    return {f"{slot},{arm}": counts[(slot, arm)] for slot, arm in sorted(counts)}


def machine_hash() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_tree(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()


def thread_env(extra: dict[str, str]) -> dict[str, str]:
    keep = {}
    for key in ("PATH", "HOME", "LANG", "LC_ALL", "USER", "LOGNAME", "TMPDIR"):
        if key in os.environ and not key.startswith("H2H_"):
            keep[key] = os.environ[key]
    for key in THREAD_ONES:
        keep[key] = "1"
    keep["OMP_DYNAMIC"] = "FALSE"
    keep["MKL_DYNAMIC"] = "FALSE"
    keep["LJOS_SEAT"] = "benchbot"
    for key, value in extra.items():
        if key.startswith("H2H_") or key.startswith("H2O_"):
            raise RuntimeError(f"refusing {key}")
        keep[key] = value
    leaked = [key for key in keep if key.startswith("H2H_") or key.startswith("H2O_")]
    if leaked:
        raise RuntimeError(f"search env carries {leaked}")
    return keep


def affinity_ok() -> bool:
    text = Path("/proc/self/status").read_text()
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip() == CORE
    return False


def reexec_on_core() -> None:
    if os.environ.get("HEXAMER_ON_CORE") == "1" and affinity_ok():
        return
    os.environ["HEXAMER_ON_CORE"] = "1"
    os.execvp("taskset", ["taskset", "-c", CORE, sys.executable, *sys.argv])


def parse_seed_text(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key and " " not in key:
            out[key] = value.strip()
    return out


def water_result(outdir: Path) -> dict[str, object]:
    text = (outdir / "run.txt").read_text()
    row = parse_seed_text(text)
    if "search_error" in row:
        raise RuntimeError(row["search_error"])
    token = row["engine_best_eV"]
    best = float(token)
    hit = energy_hits(best, HEXAMER_REF)
    charged = int(row["energy_calls"])
    first = row.get("first_hit_call", "none")
    if hit and first not in ("", "none"):
        calls = int(first)
    elif hit:
        calls = charged
    else:
        calls = HEXAMER_BUDGET
    wall_token = row["wall_s"]
    return {
        "hit": int(hit),
        "calls": calls,
        "wall": wall_token,
        "best": token,
        "best_text": token,
        "best_bits": bits(best),
        "charged": charged,
    }


def call_rank(row: dict) -> float:
    """Misses sit strictly above every budget call, so they rank worst."""
    if int(row["hit"]) == 1:
        return float(row["calls"])
    return float(HEXAMER_BUDGET) + 1.0


def self_test() -> None:
    if not balanced(ORDER, 4):
        raise SystemExit("three-arm order is not slot-balanced")
    if len(ORDER) != N_SEEDS or len(SEEDS) != N_SEEDS:
        raise SystemExit("seed count")
    ab = [tuple(arm for arm in arms if arm != "AP") for arms in ORDER]
    expect = [("A", "B"), ("B", "A"), ("B", "A"), ("A", "B")] * 3
    if ab != expect:
        raise SystemExit(f"A/B projection is {ab}")
    p = fisher_twosided(0, 12, 12, 12)
    if abs(p - 2 / math.comb(24, 12)) > 1e-15:
        raise SystemExit(f"fisher zero-versus-all is {p}")
    if abs(fisher_twosided(6, 12, 6, 12) - 1.0) > 1e-12:
        raise SystemExit("fisher tie is not 1")
    p, higher = exact_mann_whitney([1, 2, 3], [4, 5, 6])
    if abs(p - 0.1) > 1e-12 or not higher:
        raise SystemExit(f"mann-whitney 1..3 vs 4..6 is {p} higher={higher}")
    p, higher = exact_mann_whitney([1, 1, 1], [1, 1, 1])
    if abs(p - 1.0) > 1e-12 or higher:
        raise SystemExit(f"mann-whitney ties are {p}")
    if energy_hits(HEXAMER_REF, HEXAMER_REF) is not True:
        raise SystemExit("hit on the reference")
    boundary = HEXAMER_REF + HIT_TOL
    if energy_hits(boundary, HEXAMER_REF) is not True:
        raise SystemExit("hit on the tolerance boundary")
    if energy_hits(math.nextafter(boundary, math.inf), HEXAMER_REF) is not False:
        raise SystemExit("value above the tolerance counted as a hit")
    if energy_hits(math.nextafter(boundary, -math.inf), HEXAMER_REF) is not True:
        raise SystemExit("value below the tolerance was a miss")
    if energy_hits(float("nan"), HEXAMER_REF) is not False:
        raise SystemExit("nan counted as a hit")
    if bits(0.0) != "0000000000000000":
        raise SystemExit("zero bits")
    miss = {"hit": 0, "calls": HEXAMER_BUDGET}
    late = {"hit": 1, "calls": HEXAMER_BUDGET}
    if not call_rank(miss) > call_rank(late):
        raise SystemExit("a miss does not rank worse than a last-call hit")
    env = thread_env({"RGPOT_XTB_ENGINE": "engine", "LD_LIBRARY_PATH": "libs"})
    if any(key.startswith("H2H_") or key.startswith("H2O_") for key in env):
        raise SystemExit("thread env kept a search override")
    for key in THREAD_ONES:
        if env.get(key) != "1":
            raise SystemExit(f"{key} is not 1")
    if env.get("LJOS_SEAT") != "benchbot":
        raise SystemExit("seat")
    try:
        thread_env({"H2H_PAIR_SCALE": "1"})
    except RuntimeError:
        pass
    else:
        raise SystemExit("an H2H variable was accepted")
    print("self-test ok")


def load_done(path: Path) -> dict[tuple, dict]:
    done = {}
    if not path.is_file():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done[(row["panel"], row["system"], row["arm"], int(row["seed"]))] = row
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_cmd(cmd: list[str], env: dict[str, str], log: Path) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        handle.write("cmd " + " ".join(cmd) + "\n")
        proc = subprocess.run(
            ["taskset", "-c", CORE, *cmd],
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc


def rows_of(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def arm_rows(rows: list[dict], arm: str) -> list[dict]:
    found = [row for row in rows if row["arm"] == arm and row["system"] == SYSTEM and row["panel"] == PANEL]
    found.sort(key=lambda row: int(row["seed"]))
    return found


def paired(rows: list[dict], left: str, right: str) -> tuple[list[dict], list[dict]]:
    a = {int(row["seed"]): row for row in arm_rows(rows, left)}
    b = {int(row["seed"]): row for row in arm_rows(rows, right)}
    if set(a) != set(SEEDS) or set(b) != set(SEEDS):
        return [], []
    return [a[seed] for seed in SEEDS], [b[seed] for seed in SEEDS]


def identical_seeds(a: list[dict], b: list[dict]) -> list[bool]:
    flags = []
    for left, right in zip(a, b, strict=True):
        flags.append(left["best_bits"] == right["best_bits"] and int(left["calls"]) == int(right["calls"]))
    return flags


def aa_band(a: list[dict], ap: list[dict]) -> tuple[float, float, list[float]] | None:
    ratios = []
    for left, right in zip(a, ap, strict=True):
        den = float(left["wall"])
        if den <= 0.0:
            return None
        ratios.append(float(right["wall"]) / den)
    return min(ratios), max(ratios), ratios


def wall_ratio(num: list[dict], den: list[dict]) -> float | None:
    base = median([float(row["wall"]) for row in den])
    if base <= 0.0:
        return None
    return median([float(row["wall"]) for row in num]) / base


def judge(a: list[dict], b: list[dict], band: tuple[float, float] | None) -> dict:
    """PASS if every seed matches on bits and calls.

    Otherwise PASS only when B is not worse on hits at the exact two-sided
    Fisher test, not worse on calls at the exact two-sided Mann-Whitney
    test with misses ranked worst, and the median wall ratio sits inside
    the A/A band. Worse means the point estimate goes the wrong way and
    the exact p-value is below 0.05.
    """
    out = {
        "status": "INCOMPLETE",
        "identical": False,
        "hit_p": None,
        "call_p": None,
        "calls_higher": None,
        "wall_ratio": None,
        "reasons": [],
    }
    if len(a) != N_SEEDS or len(b) != N_SEEDS:
        out["reasons"].append("incomplete")
        return out
    flags = identical_seeds(a, b)
    out["per_seed_identical"] = flags
    out["identical"] = all(flags)
    hits_a = sum(int(row["hit"]) for row in a)
    hits_b = sum(int(row["hit"]) for row in b)
    hit_p = fisher_twosided(hits_a, N_SEEDS, hits_b, N_SEEDS)
    call_p, calls_higher = exact_mann_whitney([call_rank(row) for row in a], [call_rank(row) for row in b])
    ratio = wall_ratio(b, a)
    out.update(
        {
            "hits_a": hits_a,
            "hits_b": hits_b,
            "hit_p": hit_p,
            "call_p": call_p,
            "calls_higher": calls_higher,
            "wall_ratio": ratio,
        }
    )
    if out["identical"]:
        out["status"] = "PASS"
        return out
    worse_hits = hits_b < hits_a and hit_p < ALPHA
    worse_calls = calls_higher and call_p < ALPHA
    inside = False
    if band is not None and ratio is not None:
        lo, hi = band
        inside = lo <= ratio <= hi
        out["band_lo"] = lo
        out["band_hi"] = hi
    if worse_hits:
        out["reasons"].append("fewer hits")
    if worse_calls:
        out["reasons"].append("more calls")
    if not inside:
        out["reasons"].append("wall ratio outside the A/A band")
    out["status"] = "FAIL" if (worse_hits or worse_calls or not inside) else "PASS"
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (int(row["seed"]), int(row["slot"]), row["arm"]))
    with path.open("w", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: row[key] for key in ROW_FIELDS})


def measured_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[tuple[int, str], int] = {}
    for row in rows:
        if row["panel"] == PANEL and row["system"] == SYSTEM:
            key = (int(row["slot"]), row["arm"])
            counts[key] = counts.get(key, 0) + 1
    return {f"{slot},{arm}": counts[(slot, arm)] for slot, arm in sorted(counts)}


def write_session(path: Path, rows: list[dict], meta: dict) -> None:
    session = {
        "base_commit": BASE,
        "base_tree": BASE_TREE,
        "b_tree": B_TREE,
        "b_kit_sha256": B_MBOX_SHA256,
        "seeds": SEEDS,
        "core": int(CORE),
        "budget": HEXAMER_BUDGET,
        "reference": f"{HEXAMER_REF:.10f}",
        "hit_tol": HIT_TOL,
        "alpha": ALPHA,
        "machine_id_sha256": meta["machine_id_sha256"],
        "harness_blob": meta["harness_blob"],
        "order": ORDER,
        "order_slot_arm": slot_counts(ORDER),
        "measured_slot_arm": measured_counts(rows),
        "engine_version": meta["engine_version"],
        "engine_revision": meta["engine_revision"],
        "engine_rgpot": meta["engine_rgpot"],
        "engine_sha256": meta["engine_sha256"],
        "bin_sha256": meta["bin_sha256"],
        "n_rows": len(rows),
        "panel": PANEL,
        "system": SYSTEM,
    }
    path.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n")


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{float(value):.6g}"


def report_text(rows: list[dict], session: dict) -> str:
    a, b = paired(rows, "A", "B")
    _a2, ap = paired(rows, "A", "AP")
    band = aa_band(a, ap) if a and ap else None
    judged = judge(a, b, None if band is None else (band[0], band[1]))
    lines = [
        "panel 1 water-hexamer guard",
        f"base {BASE} tree {BASE_TREE}",
        f"B tree {B_TREE}",
        f"kit sha256 {B_MBOX_SHA256}",
        f"engine {session.get('engine_version', '-')} {session.get('engine_revision', '-')}",
        f"engine rgpot {session.get('engine_rgpot', '-')}",
        f"engine sha256 {session.get('engine_sha256', '-')}",
        f"seeds {SEEDS[0]}..{SEEDS[-1]} n={N_SEEDS} budget {HEXAMER_BUDGET} core {CORE}",
        f"reference {HEXAMER_REF:.10f} tol {HIT_TOL}",
        f"identical {fmt(judged['identical'])}",
        f"hits A {judged.get('hits_a', 0)}/{len(a)} B {judged.get('hits_b', 0)}/{len(b)} fisher {fmt(judged['hit_p'])}",
        f"calls mannwhitney {fmt(judged['call_p'])} B_higher {fmt(judged['calls_higher'])} misses ranked at budget+1",
    ]
    if band is None:
        lines.append("A/A band unavailable")
    else:
        lines.append(f"A/A band {band[0]:.6f} .. {band[1]:.6f}")
        lines.append("A/A per-seed ratios " + " ".join(f"{value:.6f}" for value in band[2]))
    lines.append(f"B/A wall ratio {fmt(judged['wall_ratio'])}")
    lines.append(f"verdict {judged['status']}")
    if judged["reasons"]:
        lines.append("reasons " + "; ".join(judged["reasons"]))
    lines.append("slot,arm " + " ".join(f"{key}={value}" for key, value in measured_counts(rows).items()))
    lines.append("seed identical hit_A hit_AP hit_B calls_A calls_AP calls_B wall_A wall_AP wall_B best_A best_B")
    by_seed: dict[int, dict[str, dict]] = {}
    for row in rows:
        if row["panel"] != PANEL or row["system"] != SYSTEM:
            continue
        by_seed.setdefault(int(row["seed"]), {})[row["arm"]] = row
    flags = judged.get("per_seed_identical") or []
    for index, seed in enumerate(SEEDS):
        arms = by_seed.get(seed, {})
        if not {"A", "AP", "B"} <= set(arms):
            lines.append(f"{seed} missing")
            continue
        same = "yes" if index < len(flags) and flags[index] else "no"
        lines.append(
            " ".join(
                [
                    str(seed),
                    same,
                    str(arms["A"]["hit"]),
                    str(arms["AP"]["hit"]),
                    str(arms["B"]["hit"]),
                    str(arms["A"]["calls"]),
                    str(arms["AP"]["calls"]),
                    str(arms["B"]["calls"]),
                    str(arms["A"]["wall"]),
                    str(arms["AP"]["wall"]),
                    str(arms["B"]["wall"]),
                    str(arms["A"]["best"]),
                    str(arms["B"]["best"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def measure(args: argparse.Namespace) -> None:
    reexec_on_core()
    if not affinity_ok() or os.sched_getaffinity(0) != {0}:
        raise SystemExit(f"affinity is {sorted(os.sched_getaffinity(0))}, expected core 0")
    src_a = Path(args.src_a)
    src_b = Path(args.src_b)
    if git_tree(src_a) != BASE_TREE:
        raise SystemExit("A source tree is not the base tree")
    if git_tree(src_b) != B_TREE:
        raise SystemExit("B source tree is not the kit tree")
    commit_a = subprocess.check_output(["git", "-C", str(src_a), "rev-parse", "HEAD"], text=True).strip()
    commit_b = subprocess.check_output(["git", "-C", str(src_b), "rev-parse", "HEAD"], text=True).strip()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "raw.jsonl"
    done = load_done(raw)
    harness = Path(__file__).resolve()
    blob = blob_id(harness)
    machine = machine_hash()
    xtb = Path(args.xtb)
    if not xtb.is_file():
        raise SystemExit("engine library is missing")
    bins = {"A": Path(args.bin_a), "AP": Path(args.bin_ap), "B": Path(args.bin_b)}
    for arm, path in bins.items():
        if not path.is_file():
            raise SystemExit(f"{arm} binary is missing")
    engine_sha = file_sha256(xtb)
    bin_sha = {arm: file_sha256(path) for arm, path in bins.items()}
    lib = str(Path(args.lib_dir))
    stop = f"{HEXAMER_REF + HIT_TOL:.10f}"
    for seed, slots in zip(SEEDS, ORDER, strict=True):
        for slot, arm in enumerate(slots):
            key = (PANEL, SYSTEM, arm, seed)
            if key in done:
                continue
            out = work / "scratch" / f"{PANEL}-{SYSTEM}-{arm}-{seed}"
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            cmd = [str(bins[arm]), "6", str(HEXAMER_BUDGET), str(seed), str(out), stop]
            env = thread_env(
                {
                    "RGPOT_XTB_ENGINE": str(xtb),
                    "LD_LIBRARY_PATH": lib,
                    "ANNEAL_SOURCE_COMMIT": commit_a if arm != "B" else commit_b,
                    "ANNEAL_SOURCE_TREE": BASE_TREE if arm != "B" else B_TREE,
                }
            )
            log = work / "logs" / f"{PANEL}-{BAND}-{SYSTEM}-{arm}-{seed}.log"
            proc = run_cmd(cmd, env, log)
            if proc.returncode != 0:
                raise SystemExit(f"{arm} seed {seed} exited {proc.returncode}")
            scored = water_result(out)
            row = {
                "panel": PANEL,
                "system": SYSTEM,
                "arm": arm,
                "seed": seed,
                "slot": slot,
                "core": int(CORE),
                "hit": scored["hit"],
                "calls": scored["calls"],
                "wall": scored["wall"],
                "best": scored["best"],
                "best_text": scored["best_text"],
                "best_bits": scored["best_bits"],
                "budget": HEXAMER_BUDGET,
                "machine_id_sha256": machine,
                "base_commit": BASE,
                "base_tree": B_TREE if arm == "B" else BASE_TREE,
                "harness_blob": blob,
                "charged": scored["charged"],
                "band": BAND,
            }
            append_row(raw, row)
            done[key] = row
            print(
                f"{PANEL} {SYSTEM} {arm} {seed} hit={row['hit']} calls={row['calls']} wall={row['wall']}",
                flush=True,
            )
    rows = list(done.values())
    if len(rows) != N_SEEDS * 3:
        raise SystemExit(f"expected {N_SEEDS * 3} rows, have {len(rows)}")
    meta = {
        "machine_id_sha256": machine,
        "harness_blob": blob,
        "engine_version": args.engine_version,
        "engine_revision": args.engine_revision,
        "engine_rgpot": args.engine_rgpot,
        "engine_sha256": engine_sha,
        "bin_sha256": bin_sha,
    }
    write_rows(Path(args.rows), rows)
    write_session(Path(args.sessions), rows, meta)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("run")
    run.add_argument("--work", required=True)
    run.add_argument("--src-a", required=True)
    run.add_argument("--src-b", required=True)
    run.add_argument("--bin-a", required=True)
    run.add_argument("--bin-ap", required=True)
    run.add_argument("--bin-b", required=True)
    run.add_argument("--xtb", required=True)
    run.add_argument("--lib-dir", required=True)
    run.add_argument("--engine-version", required=True)
    run.add_argument("--engine-revision", required=True)
    run.add_argument("--engine-rgpot", required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--sessions", required=True)
    report = sub.add_parser("report")
    report.add_argument("--rows", required=True)
    report.add_argument("--sessions", required=True)
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv[1:])
    if args.cmd == "self-test":
        self_test()
        return 0
    if args.cmd == "run":
        measure(args)
        return 0
    rows = rows_of(Path(args.rows))
    session = json.loads(Path(args.sessions).read_text())
    text = report_text(rows, session)
    Path(args.out).write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
