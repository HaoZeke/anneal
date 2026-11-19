#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session A/B gate for the gmin-losses3 default and pair-scaled arms.

A is develop 5eb79cdc. B is that tree plus the three-patch pair-scale and
plateau-heat kit (tree 70871f73). Panel 1 runs the default path with no
H2H_* variable set. Panel 2 runs the 13-system GMIN-loss panel with
H2H_PAIR_SCALE=1 on B only. Every measured process is one thread on core 0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
BASE_TREE = "7a2ba49b37241a8fe42b2c45c226374de6d92274"
B_TREE = "70871f7307271806926b2aec9870e838dffbe924"
B_MBOX_SHA256 = "14d0cf394fa5146bd1f769f5e8ab5a26a9afc7d8fe7dbba3e9cd118698a7288d"
RECORD_REF = "origin/record/gmin-losses"
HIT_TOL = 0.001
SEEDS = list(range(2000, 2012))
N_SEEDS = 12
CORE = "0"
ALPHA = 0.05
HEXAMER_REF = -829.8522275088
HEXAMER_BUDGET = 4000

# Exact copy of the 13-system panel on origin/record/gmin-losses.
# assert_record_panel checks the branch text before any measurement.
PANEL2 = [
    ("lj38", "lj", 400_000, -173.928427),
    ("lj55", "lj", 1_000_000, -279.248470),
    ("lj75", "lj", 4_000_000, -397.492331),
    ("lj98", "lj", 4_000_000, -543.665361),
    ("lj102", "lj", 4_000_000, -569.363652),
    ("lj103", "lj", 4_000_000, -575.766131),
    ("lj104", "lj", 4_000_000, -582.086642),
    ("lj150", "lj", 4_000_000, -893.310258),
    ("au55", "au", 1_000_000, -191.03085280),
    ("m55r6", "morse", 1_000_000, -250.286609),
    ("m55r14", "morse", 1_000_000, -220.646208),
    ("m80r6", "morse", 4_000_000, -378.333471),
    ("m80r14", "morse", 4_000_000, -340.811371),
]

PANEL1 = [
    ("lj38", "lj", 400_000, -173.928427),
    ("lj75", "lj", 4_000_000, -397.492331),
    ("au55", "au", 1_000_000, -191.03085280),
    ("m38r10", "morse-seed", 400_000, -145.849817),
    ("m55r6", "morse", 1_000_000, -250.286609),
    ("m80r6", "morse", 4_000_000, -378.333471),
    ("water_hexamer", "water", HEXAMER_BUDGET, HEXAMER_REF),
    ("tip4p8", "tip4p", 2_000_000, -305.5183221676),
    ("rq3", "conf", 28457, None),
    ("xld", "conf", 22494, None),
]

# Placement is measured on the fast cells. Slow cells use the pooled band.
PLACE_SYSTEMS = {"lj38", "m38r10", "m55r6", "tip4p8", "rq3", "xld", "water_hexamer"}

# Three-arm order. Each (slot, arm) appears four times. The A/B projection
# is AB, BA, BA, AB, repeated.
ORDER3 = [
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
ORDER2 = [("A", "B"), ("B", "A"), ("B", "A"), ("A", "B")] * 3
PLACE_ARMS = ("PA", "L1", "L2", "L3", "L4", "L5")
ORDER6 = []
for turn in range(N_SEEDS):
    arms = list(PLACE_ARMS)
    rot = turn % len(PLACE_ARMS)
    step = arms[rot:] + arms[:rot]
    if turn >= len(PLACE_ARMS):
        step = list(reversed(step))
    ORDER6.append(tuple(step))

LAYOUT_FLAGS = {
    "L1": "-C link-arg=-Wl,-z,max-page-size=65536",
    "L2": "-C link-arg=-Wl,-z,max-page-size=16384",
    "L3": "-C link-arg=-Wl,--hash-style=sysv",
    "L4": "-C link-arg=-Wl,-z,norelro",
    "L5": "-C link-arg=-Wl,-z,max-page-size=2097152",
}

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
    "machine_id_sha256",
    "base_commit",
    "base_tree",
    "harness_blob",
    "budget",
    "best_bits",
    "best_text",
    "charged",
    "band",
]


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
    return format(struct_pack_double(value), "016x")


def struct_pack_double(value: float) -> int:
    import struct

    return struct.unpack(">Q", struct.pack(">d", value))[0]


def balanced(order: list[tuple[str, ...]], expected: int) -> bool:
    counts: dict[tuple[int, str], int] = {}
    width = len(order[0])
    for arms in order:
        for slot, arm in enumerate(arms):
            counts[(slot, arm)] = counts.get((slot, arm), 0) + 1
    names = set(order[0])
    return all(counts.get((slot, arm)) == expected for slot in range(width) for arm in names)


def assert_record_panel(repo: Path) -> None:
    text = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{RECORD_REF}:scripts/gmin_panel/run_panel.py"],
        text=True,
    )
    collapsed = " ".join(text.split())
    if "A hit is a quenched energy at or below the reference plus 0.001" not in collapsed:
        raise SystemExit("record hit criterion text drifted")
    if "SEEDS = tuple(range(2000, 2012))" not in text and "SEEDS = list(range(2000, 2012))" not in text:
        raise SystemExit("record seed list drifted")
    start = text.index("SYSTEMS = ")
    end = text.index("LABELS", start)
    body = text[start:end]
    kind_on_branch = {"lj": "lj", "au": "gupta", "morse": "morse"}
    for name, kind, budget, reference in PANEL2:
        branch_kind = kind_on_branch[kind]
        needle = f'("{name}", "{branch_kind}"'
        if needle not in body:
            raise SystemExit(f"record panel is missing {name}")
        if f"{reference:.8f}".rstrip("0").rstrip(".") not in body and f"{reference}" not in body:
            raise SystemExit(f"record reference for {name} drifted")
        token = f'"{name}": {budget}'
        token_us = f'"{name}": {budget:_}'
        if token not in text and token_us not in text:
            raise SystemExit(f"record PAPER_BUDGETS missing {name} {budget}")


def self_test() -> None:
    if not balanced(ORDER3, 4):
        raise SystemExit("three-arm order is not slot-balanced")
    if not balanced(ORDER2, 6):
        raise SystemExit("two-arm order is not slot-balanced")
    if not balanced(ORDER6, 2):
        raise SystemExit("placement order is not slot-balanced")
    ab = []
    for arms in ORDER3:
        ab.append(tuple(arm for arm in arms if arm != "AP"))
    expect = [("A", "B"), ("B", "A"), ("B", "A"), ("A", "B")] * 3
    if ab != expect:
        raise SystemExit(f"A/B projection is {ab}")
    if len(SEEDS) != N_SEEDS:
        raise SystemExit("seed count")
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
    if energy_hits(-173.928427, -173.928427) is not True:
        raise SystemExit("hit on the reference")
    if energy_hits(-173.927, -173.928427) is not False:
        raise SystemExit("miss above tolerance")
    if bits(0.0) != "0000000000000000":
        raise SystemExit("zero bits")
    print("self-test ok")


def prepare_rdkit_counter() -> None:
    """Point the call counter at the wheel's hashed ForceField library.

    The counter looks up ``libRDKitForceField.so.1``. The wheel ships that
    library under a hashed soname, so the unhashed name has to resolve to it.
    """
    import rdkit

    root = Path(rdkit.__file__).resolve().parent
    libs = root.parent / "rdkit.libs"
    hashed = sorted(libs.glob("libRDKitForceField-*.so.1"))
    if not hashed:
        return
    alias = libs / "libRDKitForceField.so.1"
    if not alias.exists():
        alias.symlink_to(hashed[0].name)
    old = hashed[0].name
    for module in root.rglob("*.so"):
        needed = subprocess.check_output(["objdump", "-p", str(module)], text=True, errors="replace")
        if old in needed:
            subprocess.check_call(["patchelf", "--replace-needed", old, alias.name, str(module)])


def complete_python_package(repo: Path, bindir: Path) -> None:
    py = bindir / "py" / "anneal"
    if not (py / "_core.so").is_file():
        return
    py.mkdir(parents=True, exist_ok=True)
    for source in (repo / "python/anneal").glob("*.py"):
        dest = py / source.name
        if not dest.is_file() or dest.read_bytes() != source.read_bytes():
            shutil.copy2(source, dest)


def machine_hash() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def thread_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    keep = {}
    for key in ("PATH", "HOME", "LANG", "LC_ALL", "USER", "LOGNAME", "TMPDIR", "PYTHONPATH", "LD_LIBRARY_PATH"):
        if key in os.environ:
            keep[key] = os.environ[key]
    keep.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "GOTO_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TBB_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "LJOS_SEAT": "benchbot",
        }
    )
    if extra:
        keep.update(extra)
    return keep


def affinity_ok() -> bool:
    text = Path("/proc/self/status").read_text()
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip() == CORE
    return False


def reexec_on_core() -> None:
    if os.environ.get("GATE_ON_CORE") == "1" and affinity_ok():
        return
    os.environ["GATE_ON_CORE"] = "1"
    os.execvp("taskset", ["taskset", "-c", CORE, sys.executable, *sys.argv])


def run_cmd(cmd: list[str], env: dict[str, str], log: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        handle.write("cmd " + " ".join(cmd) + "\n")
        proc = subprocess.run(
            ["taskset", "-c", CORE, *cmd],
            cwd=cwd,
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc


def parse_driver_csv(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = None
    data = None
    for line in lines:
        if line.startswith("system,code,seed,"):
            header = line.split(",")
        elif header and not line.startswith("arm ") and "," in line:
            data = line.split(",")
    if not header or not data:
        raise RuntimeError("driver csv missing")
    if len(data) != len(header):
        raise RuntimeError(f"driver csv width {len(data)} != {len(header)}")
    return dict(zip(header, data, strict=True))


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


def cluster_result(text: str, reference: float, budget: int) -> dict[str, object]:
    row = parse_driver_csv(text)
    best = float(row["best"])
    hit = energy_hits(best, reference)
    calls = int(row["evals_to_hit"]) if hit else budget
    return {
        "hit": int(hit),
        "calls": calls,
        "wall": float(row["wall_s"]),
        "best": best,
        "best_text": row["best"],
        "charged": int(row["charged"]),
    }


def seed_result(text: str, reference: float, budget: int) -> dict[str, object]:
    row = parse_seed_text(text)
    best = float(row["best"])
    hit = energy_hits(best, reference)
    charged = int(row["charged"])
    calls = int(row["calls"]) if hit else budget
    return {
        "hit": int(hit),
        "calls": calls,
        "wall": float(row["wall_seconds"]),
        "best": best,
        "best_text": row["best"],
        "charged": charged,
    }


def water_result(outdir: Path, budget: int) -> dict[str, object]:
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
        calls = budget
    return {
        "hit": int(hit),
        "calls": calls,
        "wall": float(row["wall_s"]),
        "best": best,
        "best_text": token,
        "charged": charged,
    }


def load_done(path: Path) -> set[tuple]:
    done = set()
    if not path.is_file():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["panel"], row["system"], row["band"], row["arm"], row["seed"]))
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def binaries(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.work)
    return {
        "A": root / "bin" / "A",
        "AP": root / "bin" / "AP",
        "B": root / "bin" / "B",
        "PA": root / "bin" / "A",
        "L1": root / "bin" / "L1",
        "L2": root / "bin" / "L2",
        "L3": root / "bin" / "L3",
        "L4": root / "bin" / "L4",
        "L5": root / "bin" / "L5",
    }


def command_for(cell: tuple, arm: str, seed: int, bins: dict[str, Path], work: Path, panel: str) -> tuple[list[str], dict[str, str]]:
    name, kind, budget, reference = cell
    bindir = bins[arm]
    extra: dict[str, str] = {}
    if panel == "opt" and arm == "B":
        extra["H2H_PAIR_SCALE"] = "1"
    if kind == "lj":
        cmd = [str(bindir / "h2h_gmin_proof"), "run", name, str(seed)]
    elif kind == "au":
        cmd = [str(bindir / "h2h_metal"), "run", name, str(seed)]
    elif kind == "morse":
        cmd = [str(bindir / "h2h_morse"), "run", name, str(seed)]
        if panel == "opt":
            shared = work / "starts" / f"{name}-{seed}.xyz"
            cmd.append(str(shared))
    elif kind == "morse-seed":
        stop = reference + HIT_TOL
        cmd = [
            str(bindir / "h2h_seed"),
            "morse",
            "38",
            "10",
            str(budget),
            str(seed),
            "recommended",
            f"{stop:.6f}",
            str(work / "scratch" / f"{panel}-{name}-{arm}-{seed}"),
        ]
    elif kind == "tip4p":
        stop = reference + HIT_TOL
        cmd = [
            str(bindir / "h2h_seed"),
            "tip4p",
            "8",
            "0",
            str(budget),
            str(seed),
            "recommended",
            f"{stop:.10f}",
            str(work / "scratch" / f"{panel}-{name}-{arm}-{seed}"),
        ]
    elif kind == "water":
        out = work / "scratch" / f"{panel}-{name}-{arm}-{seed}"
        cmd = [str(bindir / "gfn2_water_h2h"), "6", str(budget), str(seed), str(out), f"{HEXAMER_REF + HIT_TOL:.10f}"]
    elif kind == "conf":
        out = work / "scratch" / f"{panel}-{name}-{arm}-{seed}"
        cmd = [
            sys.executable,
            str(Path(os.environ["GATE_REPO"]) / "scripts/openconf_h2h/h2h_job.py"),
            "anneal",
            name,
            str(seed),
            "--budget",
            str(budget),
            "--out",
            str(out),
            "--anneal-path",
            str(bindir / "py"),
        ]
    else:
        raise RuntimeError(kind)
    return cmd, extra


def ensure_shared_start(cell: tuple, seed: int, bins: dict[str, Path], work: Path, env: dict[str, str]) -> None:
    name = cell[0]
    dest = work / "starts" / f"{name}-{seed}.xyz"
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(bins["A"] / "h2h_morse"), "coords", name, str(seed), str(dest)]
    log = work / "logs" / f"start-{name}-{seed}.log"
    proc = run_cmd(cmd, env, log, work)
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(f"shared start failed for {name} {seed}")


def score_conf(job_dir: Path) -> dict[str, object]:
    info = json.loads((job_dir / "job.json").read_text())
    if info.get("error"):
        raise RuntimeError(str(info["error"]))
    calls = int(info["energy_calls"]) + int(info["grad_calls"])
    best = float(info["best_kcal"])
    return {
        "hit": 0,
        "calls": calls,
        "wall": float(info["wall_s"]),
        "best": best,
        "best_text": f"{best:.10f}",
        "charged": calls,
        "pending": True,
    }


def measure(args: argparse.Namespace) -> None:
    reexec_on_core()
    repo = Path(args.repo)
    os.environ["GATE_REPO"] = str(repo)
    assert_record_panel(repo)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "scratch").mkdir(parents=True, exist_ok=True)
    (work / "starts").mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    raw = work / "raw.jsonl"
    done = load_done(raw)
    harness = repo / "bench/gmin-losses3-gate/run_gate.py"
    blob = blob_id(harness)
    for obj in (blob, BASE, BASE_TREE, B_TREE):
        probe = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", obj], check=False)
        if probe.returncode != 0:
            raise SystemExit(f"git object {obj} is not in the repository")
    machine = machine_hash()
    bins = binaries(args)
    for bindir in bins.values():
        if bindir.is_dir():
            complete_python_package(repo, bindir)
    prepare_rdkit_counter()
    base_env = thread_env()
    preload = args.preload
    if not preload:
        preload = str(work / "librdff_count.so")
        if not Path(preload).is_file():
            source = repo / "scripts/openconf_h2h/rdff_count.c"
            subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", preload, str(source), "-ldl"])
    plans = []
    for cell in PANEL1:
        plans.append(("default", "ab", cell, ORDER3, ("A", "AP", "B")))
        if cell[0] in PLACE_SYSTEMS:
            plans.append(("default", "place", cell, ORDER6, PLACE_ARMS))
    for cell in PANEL2:
        plans.append(("opt", "opt", cell, ORDER2, ("A", "B")))
    if args.only:
        wanted = set(args.only.split(","))
        plans = [item for item in plans if item[2][0] in wanted]
    for panel, band, cell, order, arms in plans:
        name, kind, budget, reference = cell
        if kind == "water" and (not args.xtb or not (bins["A"] / "gfn2_water_h2h").is_file() or (work / "skip-water").is_file()):
            (work / "skip-water").write_text("engine library absent\n")
            continue
        abort_cell = False
        for seed, slots in zip(SEEDS, order, strict=True):
            if abort_cell:
                break
            if kind == "morse" and panel == "opt":
                ensure_shared_start(cell, seed, bins, work, base_env)
            for slot, arm in enumerate(slots):
                key = (panel, name, band, arm, seed)
                if key in done:
                    continue
                if arm not in arms:
                    continue
                cmd, extra = command_for(cell, arm, seed, bins, work, panel)
                env = thread_env(extra)
                if kind == "conf":
                    env["LD_PRELOAD"] = preload
                if kind == "water" and args.xtb:
                    env["RGPOT_XTB_ENGINE"] = args.xtb
                log = work / "logs" / f"{panel}-{band}-{name}-{arm}-{seed}.log"
                started = time.time()
                proc = run_cmd(cmd, env, log, work)
                text = log.read_text(errors="replace")
                if proc.returncode != 0:
                    if kind == "water":
                        (work / "skip-water").write_text("engine library absent\n")
                        abort_cell = True
                        break
                    raise RuntimeError(f"{panel} {name} {arm} seed {seed} exited {proc.returncode}")
                job = work / "scratch" / f"{panel}-{name}-{arm}-{seed}"
                if kind in ("lj", "au", "morse"):
                    scored = cluster_result(text, reference, budget)
                elif kind in ("morse-seed", "tip4p"):
                    scored = seed_result(text, reference, budget)
                elif kind == "water":
                    scored = water_result(job, budget)
                else:
                    scored = score_conf(job)
                row = {
                    "panel": panel,
                    "system": name,
                    "arm": arm,
                    "seed": seed,
                    "slot": slot,
                    "core": int(CORE),
                    "hit": scored["hit"],
                    "calls": scored["calls"],
                    "wall": scored["wall"],
                    "best": scored["best"],
                    "best_text": scored["best_text"],
                    "machine_id_sha256": machine,
                    "base_commit": BASE,
                    "base_tree": BASE_TREE if arm != "B" else B_TREE,
                    "harness_blob": blob,
                    "budget": budget,
                    "best_bits": bits(float(scored["best"])),
                    "band": band,
                    "kind": kind,
                    "charged": scored["charged"],
                    "elapsed": time.time() - started,
                }
                if kind == "conf":
                    row["job"] = str(job)
                append_row(raw, row)
                done.add(key)
                print(f"{panel} {band} {name} {arm} {seed} hit={row['hit']} calls={row['calls']} wall={row['wall']:.3f}", flush=True)
    finalize_conformers(raw)
    write_outputs(work, repo)


def finalize_conformers(raw: Path) -> None:
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    pending = [row for row in rows if row.get("kind") == "conf"]
    if not pending:
        return
    script_dir = str(Path(os.environ["GATE_REPO"]) / "scripts/openconf_h2h")
    sys.path.insert(0, script_dir)
    import h2h_common as hc
    from h2h_score import Molecule

    def frames_of(job: Path, mol: Molecule) -> list[tuple]:
        path = job / "candidates.xyz"
        found = hc.read_xyz(path) if path.is_file() else []
        mine = []
        for _, coords, _ in found:
            point, energy, valid = mol.polish(coords)
            if valid:
                mine.append((point, float(energy)))
        return mol.dedupe(mine)

    groups: dict[tuple, list] = {}
    for row in pending:
        groups.setdefault((row["panel"], row["system"]), []).append(row)
    updated = []
    for (_panel, name), items in groups.items():
        mol = Molecule(hc.molecules()[name])
        pool = []
        polished = {}
        for row in items:
            if row["arm"] not in ("A", "B"):
                continue
            mine = frames_of(Path(row["job"]), mol)
            polished[row["job"]] = mine
            pool.extend(mine)
        if not pool:
            continue
        lowest = min(pool, key=lambda item: item[1])
        for row in items:
            job = Path(row["job"])
            mine = polished.get(row["job"])
            if mine is None:
                mine = frames_of(job, mol)
            hit = any(mol.same(lowest, item) for item in mine)
            calls = int(row["charged"])
            if hit:
                trace = hc.read_xyz(job / "trace.xyz")
                candidates = hc.read_xyz(job / "candidates.xyz")
                for (_, _, comment), (_, coords, _) in zip(trace, candidates, strict=False):
                    point, energy, valid = mol.polish(coords)
                    if valid and mol.same(lowest, (point, float(energy))):
                        calls = min(int(comment.split()[0]), int(row["charged"]))
                        break
            row["hit"] = int(hit)
            row["calls"] = calls if hit else int(row["charged"])
            updated.append(row)
    by_key = {(r["panel"], r["system"], r["band"], r["arm"], r["seed"]): r for r in rows}
    for row in updated:
        by_key[(row["panel"], row["system"], row["band"], row["arm"], row["seed"])] = row
    with raw.open("w") as handle:
        for row in by_key.values():
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def balance_faults(rows: list[dict]) -> list[str]:
    specs = [
        ("default", "ab", ("A", "AP", "B"), 3, 4),
        ("default", "place", PLACE_ARMS, 6, 2),
        ("opt", "opt", ("A", "B"), 2, 6),
    ]
    seen = {(r["panel"], r["band"], r["system"]) for r in rows}
    faults = []
    for panel, band, arms, nslots, expect in specs:
        systems = {system for p, b, system in seen if p == panel and b == band}
        for system in sorted(systems):
            counts: dict[tuple[int, str], int] = {}
            for row in rows:
                if row["panel"] == panel and row["band"] == band and row["system"] == system:
                    key = (int(row["slot"]), row["arm"])
                    counts[key] = counts.get(key, 0) + 1
            if sum(counts.values()) != N_SEEDS * len(arms):
                continue
            for slot in range(nslots):
                for arm in arms:
                    got = counts.get((slot, arm), 0)
                    if got != expect:
                        faults.append(f"{panel} {band} {system} slot {slot} {arm} {got}")
    return faults


def rows_of(work: Path) -> list[dict]:
    raw = work / "raw.jsonl"
    return [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]


def cell_table(rows: list[dict], panel: str, band: str, system: str, left: str, right: str) -> dict:
    a = [r for r in rows if r["panel"] == panel and r["band"] == band and r["system"] == system and r["arm"] == left]
    b = [r for r in rows if r["panel"] == panel and r["band"] == band and r["system"] == system and r["arm"] == right]
    a.sort(key=lambda r: r["seed"])
    b.sort(key=lambda r: r["seed"])
    by_b = {r["seed"]: r for r in b}
    identical = True
    for row in a:
        other = by_b.get(row["seed"])
        if (
            other is None
            or other.get("best_text") != row.get("best_text")
            or other["calls"] != row["calls"]
            or other["hit"] != row["hit"]
            or other.get("charged") != row.get("charged")
        ):
            identical = False
            break
    return {
        "n_a": len(a),
        "n_b": len(b),
        "hits_a": sum(int(r["hit"]) for r in a),
        "hits_b": sum(int(r["hit"]) for r in b),
        "calls_a": [float(r["calls"]) for r in a],
        "calls_b": [float(r["calls"]) for r in b],
        "wall_a": [float(r["wall"]) for r in a],
        "wall_b": [float(r["wall"]) for r in b],
        "identical": identical and len(a) == N_SEEDS and len(b) == N_SEEDS,
        "complete": len(a) == N_SEEDS and len(b) == N_SEEDS,
    }


def ratio(num: list[float], den: list[float]) -> float | None:
    d = median(den)
    if d <= 0:
        return None
    return median(num) / d


def placement_band(rows: list[dict]) -> tuple[float | None, float | None, list[str]]:
    ratios = []
    notes = []
    for system in sorted(PLACE_SYSTEMS):
        pair = cell_table(rows, "default", "ab", system, "A", "AP")
        if pair["complete"] and pair["identical"]:
            value = ratio(pair["wall_b"], pair["wall_a"])
            if value is not None:
                ratios.append(value)
                notes.append(f"AP/A {system} {value:.6f}")
        elif pair["complete"]:
            notes.append(f"AP/A {system} excluded (not bit-identical)")
        place_a = cell_table(rows, "default", "place", system, "PA", "PA")
        if not place_a["complete"]:
            continue
        for layout in ("L1", "L2", "L3", "L4", "L5"):
            pair = cell_table(rows, "default", "place", system, "PA", layout)
            if pair["complete"] and pair["identical"]:
                value = ratio(pair["wall_b"], pair["wall_a"])
                if value is not None:
                    ratios.append(value)
                    notes.append(f"{layout}/PA {system} {value:.6f}")
            elif pair["complete"]:
                notes.append(f"{layout}/PA {system} excluded (not bit-identical)")
    if not ratios:
        return None, None, notes
    return min(ratios), max(ratios), notes


def judge(table: dict, band_hi: float | None, m: int) -> dict:
    out = {"status": "INCOMPLETE", "hit_p": None, "call_p": None, "wall_p": None, "reasons": []}
    if not table["complete"]:
        out["reasons"].append("incomplete")
        return out
    hit_p = fisher_twosided(table["hits_a"], table["n_a"], table["hits_b"], table["n_b"])
    call_p, calls_higher = exact_mann_whitney(table["calls_a"], table["calls_b"])
    wall_p, wall_higher = exact_mann_whitney(table["wall_a"], table["wall_b"])
    out.update({"hit_p": hit_p, "call_p": call_p, "wall_p": wall_p})
    worse_hits = table["hits_b"] < table["hits_a"] and hit_p * m < ALPHA
    worse_calls = calls_higher and call_p * m < ALPHA
    wall_ratio = ratio(table["wall_b"], table["wall_a"])
    worse_wall = False
    if wall_higher and wall_p * m < ALPHA:
        if not table["identical"]:
            worse_wall = True
            out["reasons"].append("wall worse and trajectories differ")
        elif band_hi is None:
            out["status"] = "INCOMPLETE"
            out["reasons"].append("wall slower and placement band is empty")
            return out
        elif wall_ratio is not None and wall_ratio > band_hi:
            worse_wall = True
            out["reasons"].append(f"wall ratio {wall_ratio:.4f} above band {band_hi:.4f}")
    out["wall_ratio"] = wall_ratio
    if worse_hits:
        out["reasons"].append("fewer hits")
    if worse_calls:
        out["reasons"].append("more calls")
    out["status"] = "FAIL" if (worse_hits or worse_calls or worse_wall) else "PASS"
    return out


def write_outputs(work: Path, repo: Path) -> None:
    rows = rows_of(work)
    lo, hi, notes = placement_band(rows)
    lines = []
    lines.append("gmin-losses3 same-session gate")
    lines.append(f"base {BASE} tree {BASE_TREE}")
    lines.append(f"B tree {B_TREE}")
    lines.append(f"B kit sha256 {B_MBOX_SHA256}")
    lines.append(f"seeds {SEEDS[0]}..{SEEDS[-1]} n={N_SEEDS} core {CORE}")
    lines.append("placement cells " + " ".join(sorted(PLACE_SYSTEMS)))
    if (work / "skip-water").is_file():
        lines.append("water_hexamer not run: engine library absent")
    if lo is None:
        lines.append("placement band empty")
    else:
        lines.append(f"placement band median wall ratio {lo:.6f} .. {hi:.6f}")
    lines.extend(notes)
    verdicts = []
    for panel, band, cells in (("default", "ab", PANEL1), ("opt", "opt", PANEL2)):
        present = []
        for cell in cells:
            table = cell_table(rows, panel, band, cell[0], "A", "B")
            if table["complete"]:
                present.append(cell[0])
        m = max(1, len(present))
        lines.append("")
        lines.append(f"panel {panel}")
        lines.append("system hits_A hits_B hit_p call_p wall_p wall_ratio identical status")
        for cell in cells:
            table = cell_table(rows, panel, band, cell[0], "A", "B")
            judged = judge(table, hi, m)
            verdicts.append((panel, cell[0], judged["status"]))
            lines.append(
                " ".join(
                    [
                        cell[0],
                        f"{table['hits_a']}/{table['n_a']}",
                        f"{table['hits_b']}/{table['n_b']}",
                        fmt(judged["hit_p"]),
                        fmt(judged["call_p"]),
                        fmt(judged["wall_p"]),
                        fmt(judged.get("wall_ratio")),
                        "yes" if table["identical"] else "no",
                        judged["status"],
                        ";".join(judged["reasons"]),
                    ]
                )
            )
            if cell[0] == "m80r6":
                lines.append(
                    f"m80r6 detail {panel}: A {table['hits_a']}/12 B {table['hits_b']}/12 identical={table['identical']}"
                )
    faults = balance_faults(rows)
    if faults:
        lines.append("order faults")
        lines.extend(faults)
    overall = "PASS"
    if faults or any(status == "FAIL" for _, _, status in verdicts):
        overall = "FAIL"
    elif any(status != "PASS" for _, _, status in verdicts) or len(verdicts) < len(PANEL1) + len(PANEL2):
        overall = "INCOMPLETE"
    lines.append("")
    lines.append(f"overall {overall}")
    text = "\n".join(lines) + "\n"
    (work / "REPORT.txt").write_text(text)
    fieldnames = ROW_FIELDS
    with (work / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["panel"], r["system"], r["band"], r["seed"], r["slot"], r["arm"])):
            writer.writerow(row)
    session = {
        "base_commit": BASE,
        "base_tree": BASE_TREE,
        "b_tree": B_TREE,
        "b_kit_sha256": B_MBOX_SHA256,
        "record_ref": RECORD_REF,
        "seeds": SEEDS,
        "core": int(CORE),
        "machine_id_sha256": machine_hash(),
        "harness_blob": blob_id(repo / "bench/gmin-losses3-gate/run_gate.py"),
        "order3": ORDER3,
        "order2": ORDER2,
        "order6": ORDER6,
        "placement_band": [lo, hi],
        "layouts": LAYOUT_FLAGS,
        "bin_sha256": {},
        "n_rows": len(rows),
    }
    if (work / "builds.json").is_file():
        session["builds"] = json.loads((work / "builds.json").read_text())
    bin_root = work / "bin"
    if bin_root.is_dir():
        for path in sorted(bin_root.rglob("*")):
            if path.is_file() and path.suffix != ".py":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                session["bin_sha256"][str(path.relative_to(bin_root))] = digest
    (work / "sessions.json").write_text(json.dumps(session, indent=2, sort_keys=True) + "\n")
    print(text)


def fmt(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6g}"


def build_one(src: Path, target: Path, flags: str, log: Path) -> None:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["RUSTUP_TOOLCHAIN"] = "1.88.0"
    if flags:
        env["RUSTFLAGS"] = flags
    env["LJOS_SEAT"] = "benchbot"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cargo",
        "+1.88.0",
        "build",
        "--release",
        "--locked",
        "--example",
        "h2h_gmin_proof",
        "--example",
        "h2h_morse",
        "--example",
        "h2h_metal",
        "--example",
        "h2h_seed",
        "-j",
        "4",
    ]
    with log.open("w") as handle:
        proc = subprocess.run(cmd, cwd=src, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"build failed for {src}; see {log}")
    cmd_py = ["cargo", "+1.88.0", "build", "--release", "--locked", "-p", "anneal-python", "--features", "python", "-j", "4"]
    with log.open("a") as handle:
        proc = subprocess.run(cmd_py, cwd=src, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"python build failed for {src}; see {log}")


def stage_bin(src: Path, target: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    release = target / "release"
    for name in ("h2h_gmin_proof", "h2h_morse", "h2h_metal", "h2h_seed", "gfn2_water_h2h"):
        source = release / "examples" / name
        if source.is_file():
            shutil.copy2(source, dest / name)
    py = dest / "py" / "anneal"
    py.mkdir(parents=True)
    so = release / "libanneal_python.so"
    if not so.is_file():
        matches = list(release.glob("libanneal*.so"))
        if not matches:
            raise SystemExit(f"python extension missing under {release}")
        so = matches[0]
    for source in (src / "python/anneal").glob("*.py"):
        shutil.copy2(source, py / source.name)
    shutil.copy2(so, py / "_core.so")


def cmd_build(args: argparse.Namespace) -> None:
    work = Path(args.work)
    jobs = [
        ("A", Path(args.wt_a), work / "target-a", ""),
        ("AP", Path(args.wt_ap), work / "target-ap", ""),
        ("B", Path(args.wt_b), work / "target-b", ""),
    ]
    for name, flags in LAYOUT_FLAGS.items():
        jobs.append((name, Path(args.wt_a), work / f"target-{name.lower()}", flags))
    if args.only:
        wanted = set(args.only.split(","))
        jobs = [job for job in jobs if job[0] in wanted]
    for name, src, target, flags in jobs:
        marker = work / "bin" / name / "h2h_gmin_proof"
        if marker.is_file() and not args.force:
            print(f"skip {name}")
            continue
        print(f"build {name}", flush=True)
        build_one(src, target, flags, work / "logs" / f"build-{name}.log")
        stage_bin(src, target, work / "bin" / name)
        commit = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD^{tree}"], text=True).strip()
        meta_path = work / "builds.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        meta[name] = {"commit": commit, "tree": tree}
        if name == "A":
            meta["PA"] = {"commit": commit, "tree": tree}
            link = work / "bin" / "PA"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to("A")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("run")
    run.add_argument("--work", required=True)
    run.add_argument("--repo", required=True)
    run.add_argument("--only", default="")
    run.add_argument("--preload", default="")
    run.add_argument("--xtb", default="")
    build = sub.add_parser("build")
    build.add_argument("--work", required=True)
    build.add_argument("--wt-a", required=True)
    build.add_argument("--wt-ap", required=True)
    build.add_argument("--wt-b", required=True)
    build.add_argument("--only", default="")
    build.add_argument("--force", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--work", required=True)
    report.add_argument("--repo", required=True)
    args = parser.parse_args()
    if args.cmd == "self-test":
        self_test()
    elif args.cmd == "run":
        measure(args)
    elif args.cmd == "build":
        cmd_build(args)
    else:
        write_outputs(Path(args.work), Path(args.repo))


if __name__ == "__main__":
    main()
