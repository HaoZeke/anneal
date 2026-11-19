#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Three-arm same-session gate for m75r14, m80r14 and tip4p20.

Arm A is the develop driver. Arms B and C are the nmpins7 driver. B leaves
``general_symmetry`` off. C sets it with ``H2H_GENERAL_SYMMETRY=1``. Seeds
2000-2011 share one starting geometry, the coordinates the arm A Morse
example writes for that seed, and the TIP4P start the example draws. The
six arm orders are used twice, so each arm occupies each slot four times.
An A/A pair on m75r14 uses the ABBA replica order. One core, one thread.
A seed that reaches two hours is a miss and keeps the calls it charged.

``--run`` appends rows and never rewrites them. ``--report`` reads the
committed rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
KIT = "4cd811004c67f26b0e216664cb51352f04654c09"
KIT_TREE = "51c1a7e0180585d078c3457b5550b2963d1ab752"
SEEDS = tuple(range(2000, 2012))
CELLS = ("m75r14", "m80r14", "tip4p20")
PERMS = (
    ("A", "B", "C"),
    ("A", "C", "B"),
    ("B", "A", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("C", "B", "A"),
)
NOISE_FIRST = (1, 2, 2, 1)
WALL_CAP_S = 7200.0
RUN_TIMEOUT_S = 7500.0
BONFERRONI_M = 30
ALPHA = 0.05
OUTSIDE_RATIO_LIMIT = 1.1
CHECK_SEED = 1
CHECK_BUDGET = 30
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "GOTO_NUM_THREADS",
)
PACKAGES = (
    "anneal-core",
    "rgmin",
    "readcon-core",
    "readcon-db",
    "eindir-core",
    "rgpot-core",
    "rgpot-pair",
    "rgsaddle",
)
ROW_FIELDS = [
    "block",
    "system",
    "slot",
    "arm",
    "replica",
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
    "core",
    "affinity",
    "machine_id",
    "code_revision",
    "code_tree",
    "binary_sha256",
    "general_symmetry",
    "resolved_sha256",
    "start_sha256",
    "driver",
    "session",
]


def load_py(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo_tables():
    panel = load_py(ROOT / "scripts" / "gmin_panel" / "run_panel.py", "gmin_panel_run")
    water = load_py(ROOT / "scripts" / "tip4p_h2h" / "run_h2h.py", "tip4p_h2h_run")
    tol = load_py(ROOT / "scripts" / "target_tol.py", "target_tol_mod")
    return panel, water, tol


def m75_reference() -> float:
    found = None
    for line in (ROOT / "examples" / "fixtures" / "morse_ccd.csv").read_text().splitlines():
        parts = line.split(",")
        if len(parts) >= 5 and parts[0] == "14" and parts[1] == "75" and parts[2] == "75C":
            found = float(parts[4])
    if found is None:
        raise SystemExit("Cambridge 75C rho 14 is not in the Morse table")
    return found


def reference_of(system: str, panel, water) -> float:
    if system == "m75r14":
        return m75_reference()
    if system == "m80r14":
        for row in panel.SYSTEMS:
            if row[0] == "m80r14":
                return float(row[3])
        raise SystemExit("m80r14 is not in the panel")
    if system == "tip4p20":
        return float(water.CAMBRIDGE[20])
    raise SystemExit(system)


def budget_of(system: str, panel, water) -> int:
    if system == "tip4p20":
        return int(water.BUDGET)
    if system in ("m75r14", "m80r14"):
        return int(panel.PAPER_BUDGETS["m80r14"])
    raise SystemExit(system)


def plan() -> list[tuple]:
    rows = []
    for index, seed in enumerate(SEEDS):
        order = PERMS[index % len(PERMS)]
        for system in CELLS:
            for slot, arm in enumerate(order, start=1):
                rows.append(("gate", system, slot, arm, "1", seed))
        first = NOISE_FIRST[index % len(NOISE_FIRST)]
        second = 1 if first == 2 else 2
        rows.append(("noise", "m75r14", 1, "A", str(first), seed))
        rows.append(("noise", "m75r14", 2, "A", str(second), seed))
    return rows


def schedule_counts() -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for block, system, slot, arm, replica, _seed in plan():
        if block != "gate":
            continue
        key = (system, slot, arm)
        counts[key] = counts.get(key, 0) + 1
    return counts


def show_schedule() -> None:
    counts = schedule_counts()
    bad = [key for key, count in sorted(counts.items()) if count != 4]
    noise_first = [row[4] for row in plan() if row[0] == "noise" and row[2] == 1]
    print(f"gate rows {sum(1 for row in plan() if row[0] == 'gate')}")
    print(f"noise rows {sum(1 for row in plan() if row[0] == 'noise')}")
    print(f"slot-arm pairs {len(counts)} off {len(bad)}")
    print(f"noise first-replica {''.join(noise_first)} ones {noise_first.count('1')} twos {noise_first.count('2')}")
    if bad or noise_first.count("1") != 6 or noise_first.count("2") != 6:
        raise SystemExit("schedule is not balanced")


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    return proc.returncode == 0


def cat_file(repo: Path, sha: str) -> None:
    subprocess.run(["git", "-C", str(repo), "cat-file", "-e", sha], check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_text().strip().encode()
    return hashlib.sha256(raw).hexdigest()


def pin_self() -> None:
    os.sched_setaffinity(0, {0})
    if os.sched_getaffinity(0) != {0}:
        raise SystemExit("process is not pinned to core 0")


def require_seat() -> None:
    if os.environ.get("LJOS_SEAT", "") == "":
        raise SystemExit("LJOS_SEAT is empty")


def thread_env(arm: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    if arm == "C":
        env["H2H_GENERAL_SYMMETRY"] = "1"
    return env


def lock_versions(path: Path) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    name = version = source = None

    def flush() -> None:
        nonlocal name, version, source
        if name in PACKAGES and version is not None:
            revision = source.rsplit("#", 1)[1] if source and "#" in source else ""
            found[name] = {"version": version, "revision": revision}
        name = version = source = None

    for line in path.read_text().splitlines():
        if line.startswith("[[package]]"):
            flush()
        elif line.startswith("name = "):
            name = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("version = "):
            version = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("source = "):
            source = line.split("=", 1)[1].strip().strip('"')
    flush()
    missing = [name for name in PACKAGES if name not in found]
    if missing:
        raise SystemExit(f"lock is missing {', '.join(missing)}")
    return found


def rustc_version() -> str:
    proc = subprocess.run(
        ["rustup", "run", "stable", "rustc", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def commits_since(repo: Path, base: str) -> list[dict[str, str]]:
    text = git(repo, "rev-list", "--reverse", f"{base}..HEAD")
    rows = []
    for sha in text.splitlines():
        if not sha:
            continue
        rows.append(
            {
                "sha": sha,
                "tree": git(repo, "rev-parse", f"{sha}^{{tree}}"),
                "subject": git(repo, "log", "-1", "--format=%s", sha),
            }
        )
    return rows


def identity(repo: Path, base: str) -> dict[str, object]:
    head = git(repo, "rev-parse", "HEAD")
    if not is_ancestor(repo, base, head):
        raise SystemExit("build is not descended from the recorded base")
    return {
        "revision": head,
        "tree": git(repo, "rev-parse", "HEAD^{tree}"),
        "commits": commits_since(repo, base),
        "packages": lock_versions(repo / "Cargo.lock"),
    }


def binaries(directory: Path) -> dict[str, Path]:
    found = {}
    for name in ("h2h_morse", "h2h_seed"):
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"missing {name}")
        found[name] = path
    return found


def run_process(
    command: list[str], env: dict[str, str], log_path: Path, allow_budget: bool = False
) -> tuple[str, str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    affinity = ""
    env_error = ""
    try:
        status = Path(f"/proc/{proc.pid}/status").read_text()
        for line in status.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                affinity = line.split(":", 1)[1].strip()
        try:
            raw = Path(f"/proc/{proc.pid}/environ").read_bytes().split(b"\0")
        except OSError:
            raw = []
        seen = {}
        for item in raw:
            if b"=" in item:
                key, value = item.split(b"=", 1)
                seen[key.decode()] = value.decode()
        if seen:
            for key in THREAD_KEYS:
                if seen.get(key) != "1":
                    env_error = f"{key} is not 1"
            if seen.get("H2H_BUDGET") and not allow_budget:
                env_error = "H2H_BUDGET is set"
    except FileNotFoundError:
        affinity = affinity or "0"
    if env_error:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise RuntimeError(env_error)
    try:
        stdout, stderr = proc.communicate(timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise RuntimeError("run exceeded the wall backstop") from None
    log_path.write_text(stdout + "\n" + stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}")
    if affinity != "0":
        raise RuntimeError(f"affinity {affinity}")
    return stdout, stderr, affinity


def parse_flag(stderr: str, arm: str) -> tuple[str, str]:
    lines = [line for line in stderr.splitlines() if line.startswith("general_symmetry=")]
    if arm == "A":
        if lines:
            raise RuntimeError("arm A printed general_symmetry")
        return "na", ""
    if len(lines) != 1:
        raise RuntimeError("expected one general_symmetry line")
    text = lines[0]
    value = text.split("resolved=", 1)[0].split("=", 1)[1].strip()
    resolved = text.split("resolved=", 1)[1].strip() if "resolved=" in text else ""
    expected = "1" if arm == "C" else "0"
    if value != expected:
        raise RuntimeError(f"general_symmetry {value}")
    return value, resolved


def parse_capped(stderr: str) -> str:
    lines = [line for line in stderr.splitlines() if line.startswith("capped=")]
    if len(lines) != 1 or lines[0].split("=", 1)[1] not in ("0", "1"):
        raise RuntimeError("missing capped line")
    return lines[0].split("=", 1)[1]


def truth(text: str) -> bool:
    return text.strip().lower() in ("1", "true")


def check_counts(
    hit: bool,
    calls: int,
    calls_to_hit: int,
    budget: int,
    capped: bool,
    wall: float,
    pot: float,
    full_budget: bool,
) -> None:
    if calls < 1 or calls > budget or calls_to_hit < 1 or calls_to_hit > budget:
        raise RuntimeError("call counts are outside the budget")
    if hit and calls_to_hit > calls:
        raise RuntimeError("hit reports more calls-to-hit than calls")
    if capped:
        if hit or calls != calls_to_hit or wall < WALL_CAP_S - 1.0:
            raise RuntimeError("capped row is inconsistent")
    elif not hit:
        if calls_to_hit != budget or (full_budget and calls != budget):
            raise RuntimeError("uncapped miss is not at the budget")
    if wall < 0.0 or pot < 0.0 or pot > wall + 1.0e-3:
        raise RuntimeError("times are inconsistent")


def row_from_morse(stdout: str, stderr: str, system: str, seed: int, budget: int, arm: str, tol) -> dict[str, str]:
    header = data = None
    for line in stdout.splitlines():
        if line.startswith("system,code,"):
            header = line
        elif f",anneal,{seed}," in line:
            data = line
    if header is None or data is None:
        raise RuntimeError("missing morse row")
    parsed = dict(zip(header.split(","), data.split(",")))
    if parsed.get("system") != system or int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
        raise RuntimeError("morse row does not match the request")
    hit = truth(parsed["hit"])
    calls = int(parsed["charged"])
    calls_to_hit = int(parsed["evals_to_hit"])
    best = float(parsed["best"])
    wall = float(parsed["wall_s"])
    pot = float(parsed["pot_s"])
    outside = float(parsed["outside_s"])
    capped = parse_capped(stderr) == "1"
    if hit != tol.energy_hits(best, reference_cache[system], tol.DEFAULT_HIT_TOL):
        raise RuntimeError("hit flag disagrees with the reference")
    check_counts(hit, calls, calls_to_hit, budget, capped, wall, pot, True)
    flag, resolved = parse_flag(stderr, arm)
    return {
        "hit": "1" if hit else "0",
        "calls": str(calls),
        "calls_to_hit": str(calls_to_hit),
        "best": parsed["best"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call": f"{outside / calls:.8e}",
        "capped": "1" if capped else "0",
        "general_symmetry": flag,
        "resolved_sha256": resolved,
        "driver": "h2h_morse",
    }


def row_from_seed(stdout: str, stderr: str, seed: int, budget: int, arm: str, tol) -> dict[str, str]:
    rows = [line for line in stdout.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError("missing tip4p row")
    parsed = {}
    for token in rows[0].split()[1:]:
        key, value = token.split("=", 1)
        parsed[key] = value
    if parsed.get("system") != "tip4p20" or int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
        raise RuntimeError("tip4p row does not match the request")
    hit = truth(parsed["hit"])
    calls = int(parsed["charged"])
    calls_to_hit = int(parsed["evals_to_hit"])
    best = float(parsed["best"])
    wall = float(parsed["wall_s"])
    pot = float(parsed["pot_s"])
    outside = float(parsed["outside_s"])
    capped = parse_capped(stderr) == "1"
    if parsed.get("capped") != ("1" if capped else "0"):
        raise RuntimeError("tip4p capped fields disagree")
    if hit != tol.energy_hits(best, reference_cache["tip4p20"], tol.DEFAULT_HIT_TOL):
        raise RuntimeError("hit flag disagrees with the reference")
    check_counts(hit, calls, calls_to_hit, budget, capped, wall, pot, False)
    flag, resolved = parse_flag(stderr, arm)
    return {
        "hit": "1" if hit else "0",
        "calls": str(calls),
        "calls_to_hit": str(calls_to_hit),
        "best": parsed["best"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call": f"{outside / calls:.8e}",
        "capped": "1" if capped else "0",
        "general_symmetry": flag,
        "resolved_sha256": resolved,
        "driver": "h2h_seed",
    }


reference_cache: dict[str, float] = {}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({key: row[key] for key in ROW_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def forbid_portable(rows: list[dict[str, str]]) -> None:
    host = socket.gethostname()
    raw = Path("/etc/machine-id").read_text().strip()
    for row in rows:
        for key, value in row.items():
            if "/" in value or (host and host in value) or (raw and raw in value):
                raise SystemExit(f"row {key} is not portable")


def budget_lines(binary: Path, system: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "budget", system],
        check=True,
        capture_output=True,
        text=True,
        env=thread_env("A"),
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return int(lines[0]), lines[1], lines[2]


def confirm_budgets(morse_a: Path, morse_kit: Path, budgets: dict[str, int], refs: dict[str, float]) -> None:
    for system in ("m75r14", "m80r14"):
        for binary in (morse_a, morse_kit):
            budget, printed, rho = budget_lines(binary, system)
            if budget != budgets[system] or rho != "14":
                raise SystemExit(f"{system} budget mismatch")
            if printed != f"{refs[system]:.10f}":
                raise SystemExit(f"{system} reference mismatch")


def write_coords(binary: Path, system: str, seed: int, path: Path) -> str:
    env = thread_env("A")
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "coords", system, str(seed), str(path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0 or not path.is_file():
        raise RuntimeError("coords failed")
    return sha256_file(path)


def execute(job: tuple, bins: dict[str, dict[str, Path]], work: Path, logs: Path, meta: dict, session: str, tol) -> dict[str, str]:
    block, system, slot, arm, replica, seed = job
    budget = int(meta["budgets"][system])
    ident = meta["arm_a"] if arm == "A" else meta["arm_kit"]
    binary_name = "h2h_morse" if system != "tip4p20" else "h2h_seed"
    binary = bins["A" if arm == "A" else "kit"][binary_name]
    log_path = logs / f"{block}-{system}-{arm}-r{replica}-s{seed}.log"
    if system == "tip4p20":
        prefix = work / f"{block}-{system}-{arm}-r{replica}-s{seed}"
        stop = repr(reference_cache["tip4p20"] + tol.DEFAULT_HIT_TOL)
        command = [
            "taskset",
            "-c",
            "0",
            str(binary),
            "tip4p",
            "20",
            "0",
            str(budget),
            str(seed),
            "recommended",
            stop,
            str(prefix),
        ]
    else:
        coords = work / f"{system}-s{seed}.coords"
        if not coords.is_file():
            write_coords(bins["A"]["h2h_morse"], system, seed, coords)
        command = ["taskset", "-c", "0", str(binary), "run", system, str(seed), str(coords)]
    stdout, stderr, affinity = run_process(command, thread_env(arm), log_path)
    if system == "tip4p20":
        measured = row_from_seed(stdout, stderr, seed, budget, arm, tol)
        start = sha256_file(Path(str(prefix) + ".start.gmin"))
    else:
        measured = row_from_morse(stdout, stderr, system, seed, budget, arm, tol)
        start = sha256_file(work / f"{system}-s{seed}.coords")
    measured.update(
        {
            "block": block,
            "system": system,
            "slot": str(slot),
            "arm": arm,
            "replica": str(replica),
            "seed": str(seed),
            "budget": str(budget),
            "core": "0",
            "affinity": affinity,
            "machine_id": meta["machine_id"],
            "code_revision": str(ident["revision"]),
            "code_tree": str(ident["tree"]),
            "binary_sha256": sha256_file(binary),
            "start_sha256": start,
            "session": session,
        }
    )
    return measured


def same_start(rows: list[dict[str, str]], job: tuple, start: str) -> None:
    block, system, _slot, _arm, _replica, seed = job
    for row in rows:
        if row["block"] == block and row["system"] == system and row["seed"] == str(seed):
            if row["start_sha256"] != start:
                raise RuntimeError("starting geometry differs between arms")


def run_campaign(args, panel, water, tol) -> None:
    pin_self()
    require_seat()
    bins = {"A": binaries(args.bin_dir_a), "kit": binaries(args.bin_dir_kit)}
    refs = {system: reference_of(system, panel, water) for system in CELLS}
    budgets = {system: budget_of(system, panel, water) for system in CELLS}
    reference_cache.update(refs)
    confirm_budgets(bins["A"]["h2h_morse"], bins["kit"]["h2h_morse"], budgets, refs)
    if is_ancestor(args.tree_a, KIT, "HEAD"):
        raise SystemExit("arm A contains the kit")
    if not is_ancestor(args.tree_kit, KIT, "HEAD"):
        raise SystemExit("kit build does not contain the kit")
    if git(args.tree_kit, "rev-parse", f"{KIT}^{{tree}}") != KIT_TREE:
        raise SystemExit("kit tree differs")
    arm_a = identity(args.tree_a, BASE)
    arm_kit = identity(args.tree_kit, BASE)
    for label, ident in (("A", arm_a), ("kit", arm_kit)):
        for sha in [str(ident["revision"]), str(ident["tree"]), *[row["sha"] for row in ident["commits"]]]:
            cat_file(ROOT, sha)
        cat_file(ROOT, BASE)
        cat_file(ROOT, git(ROOT, "rev-parse", f"{BASE}^{{tree}}"))
    digest = hashlib.sha256()
    for key in ("A", "kit"):
        for name in ("h2h_morse", "h2h_seed"):
            digest.update(sha256_file(bins[key][name]).encode())
    session = digest.hexdigest()[:16]
    meta = {
        "machine_id": machine_id(),
        "budgets": budgets,
        "arm_a": arm_a,
        "arm_kit": arm_kit,
    }
    existing = read_rows(args.rows)
    if existing and existing[0]["session"] != session:
        raise SystemExit("existing rows belong to another binary set")
    forbid_portable(existing)
    header = {
        "kind": "start",
        "session": session,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine_id": meta["machine_id"],
        "core": "0",
        "affinity": "0",
        "seeds": list(SEEDS),
        "cells": list(CELLS),
        "budgets": budgets,
        "references": {key: f"{value:.10f}" for key, value in refs.items()},
        "hit_tol": tol.DEFAULT_HIT_TOL,
        "wall_cap_s": WALL_CAP_S,
        "bonferroni_m": BONFERRONI_M,
        "base": BASE,
        "base_tree": git(ROOT, "rev-parse", f"{BASE}^{{tree}}"),
        "kit": KIT,
        "kit_tree": KIT_TREE,
        "arm_a": arm_a,
        "arm_kit": arm_kit,
        "rustc": rustc_version(),
        "binary_sha256": {
            "a_morse": sha256_file(bins["A"]["h2h_morse"]),
            "a_seed": sha256_file(bins["A"]["h2h_seed"]),
            "kit_morse": sha256_file(bins["kit"]["h2h_morse"]),
            "kit_seed": sha256_file(bins["kit"]["h2h_seed"]),
        },
    }
    if not any(line_kind(args.sessions, "start", session) for _ in [0]):
        append_jsonl(args.sessions, header)
    done = {(row["block"], row["system"], row["arm"], row["replica"], row["seed"]) for row in existing}
    for job in plan():
        key = (job[0], job[1], job[3], job[4], str(job[5]))
        if key in done:
            print(f"keep {key[0]} {key[1]} {key[2]} r{key[3]} s{key[4]}", file=sys.stderr, flush=True)
            continue
        print(f"run {key[0]} {key[1]} {key[2]} r{key[3]} s{key[4]}", file=sys.stderr, flush=True)
        measured = execute(job, bins, args.work_dir, args.log_dir, meta, session, tol)
        same_start(existing, job, measured["start_sha256"])
        forbid_portable([measured])
        append_row(args.rows, measured)
        existing.append(measured)
        done.add(key)
    append_jsonl(
        args.sessions,
        {
            "kind": "end",
            "session": session,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": len(existing),
        },
    )
    print(f"rows {len(existing)}", file=sys.stderr)


def line_kind(path: Path, kind: str, session: str) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("kind") == kind and payload.get("session") == session:
            return True
    return False


def check_arm(args, panel, water, tol) -> None:
    pin_self()
    require_seat()
    bins = {"A": binaries(args.bin_dir_a), "kit": binaries(args.bin_dir_kit)}
    refs = {system: reference_of(system, panel, water) for system in CELLS}
    budgets = {system: budget_of(system, panel, water) for system in CELLS}
    reference_cache.update(refs)
    confirm_budgets(bins["A"]["h2h_morse"], bins["kit"]["h2h_morse"], budgets, refs)
    work = args.work_dir / "check"
    work.mkdir(parents=True, exist_ok=True)
    logs = args.log_dir / "check"
    coords = work / "m75r14-check.coords"
    write_coords(bins["A"]["h2h_morse"], "m75r14", CHECK_SEED, coords)
    for arm in ("A", "B", "C"):
        binary = bins["A" if arm == "A" else "kit"]["h2h_morse"]
        env = thread_env(arm)
        env["H2H_BUDGET"] = str(CHECK_BUDGET)
        command = ["taskset", "-c", "0", str(binary), "run", "m75r14", str(CHECK_SEED), str(coords)]
        stdout, stderr, _affinity = run_process(command, env, logs / f"morse-{arm}.log", allow_budget=True)
        if "H2H_BUDGET" in thread_env(arm):
            raise SystemExit("measurement env keeps a budget override")
        measured = row_from_morse(stdout, stderr, "m75r14", CHECK_SEED, CHECK_BUDGET, arm, tol)
        if measured["capped"] != "0":
            raise SystemExit("check run capped")
    starts = []
    stop = repr(refs["tip4p20"] + tol.DEFAULT_HIT_TOL)
    for arm in ("A", "B", "C"):
        binary = bins["A" if arm == "A" else "kit"]["h2h_seed"]
        prefix = work / f"tip4p-{arm}"
        command = [
            "taskset",
            "-c",
            "0",
            str(binary),
            "tip4p",
            "20",
            "0",
            str(CHECK_BUDGET),
            str(CHECK_SEED),
            "recommended",
            stop,
            str(prefix),
        ]
        stdout, stderr, _affinity = run_process(command, thread_env(arm), logs / f"tip4p-{arm}.log")
        measured = row_from_seed(stdout, stderr, CHECK_SEED, CHECK_BUDGET, arm, tol)
        if measured["capped"] != "0":
            raise SystemExit("check run capped")
        starts.append(sha256_file(Path(str(prefix) + ".start.gmin")))
    if len(set(starts)) != 1:
        raise SystemExit("tip4p starts differ")
    print("check ok", file=sys.stderr)


def type7(values: list[float], probability: float) -> float:
    panel = load_py(ROOT / "scripts" / "gmin_panel" / "run_panel.py", "gmin_panel_type7")
    return float(panel.type7(values, probability))


def iqr(values: list[float]) -> float:
    return type7(values, 0.75) - type7(values, 0.25)


def bonferroni(probability: float) -> float:
    return min(1.0, BONFERRONI_M * probability)


def mann_whitney(left: list[float], right: list[float], alternative: str) -> tuple[float, float]:
    from scipy.stats import mannwhitneyu

    if len(set(left + right)) == 1:
        return len(left) * len(right) / 2.0, 1.0
    result = mannwhitneyu(left, right, alternative=alternative, method="asymptotic", use_continuity=True)
    return float(result.statistic), float(result.pvalue)


def fisher_worse(hits_new: int, hits_base: int, count: int) -> float:
    from scipy.stats import fisher_exact

    table = [[hits_new, count - hits_new], [hits_base, count - hits_base]]
    return float(fisher_exact(table, alternative="less")[1])


def select(rows: list[dict[str, str]], block: str, system: str, arm: str, replica: str | None = None) -> list[dict[str, str]]:
    found = []
    for row in rows:
        if row["block"] == block and row["system"] == system and row["arm"] == arm:
            if replica is None or row["replica"] == replica:
                found.append(row)
    return sorted(found, key=lambda row: int(row["seed"]))


def column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def fmt_p(value: float) -> str:
    return f"{value:.6e}"


def fmt_num(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1.0e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def compare_arm(base: list[dict[str, str]], other: list[dict[str, str]]) -> dict[str, float]:
    if len(base) != 12 or len(other) != 12:
        raise SystemExit("a cell does not have 12 seeds")
    if [row["seed"] for row in base] != [row["seed"] for row in other]:
        raise SystemExit("arms do not share seeds")
    hits_base = sum(row["hit"] == "1" for row in base)
    hits_other = sum(row["hit"] == "1" for row in other)
    calls_base = column(base, "calls")
    calls_other = column(other, "calls")
    out_base = column(base, "outside_per_call")
    out_other = column(other, "outside_per_call")
    _u_calls, calls_p = mann_whitney(calls_other, calls_base, "greater")
    _u_out, out_p = mann_whitney(out_other, out_base, "greater")
    _u_two, out_two = mann_whitney(out_other, out_base, "two-sided")
    ratio = type7(out_other, 0.5) / type7(out_base, 0.5)
    return {
        "hits_base": float(hits_base),
        "hits_other": float(hits_other),
        "fisher": fisher_worse(hits_other, hits_base, 12),
        "calls_p": calls_p,
        "outside_p": out_p,
        "outside_two": out_two,
        "ratio": ratio,
        "calls_base": type7(calls_base, 0.5),
        "calls_other": type7(calls_other, 0.5),
        "calls_iqr_base": iqr(calls_base),
        "calls_iqr_other": iqr(calls_other),
        "out_base": type7(out_base, 0.5),
        "out_other": type7(out_other, 0.5),
        "tohit_base": type7(column(base, "calls_to_hit"), 0.5),
        "tohit_other": type7(column(other, "calls_to_hit"), 0.5),
        "capped_base": float(sum(row["capped"] == "1" for row in base)),
        "capped_other": float(sum(row["capped"] == "1" for row in other)),
    }


def verdict(stats: dict[str, float]) -> str:
    reasons = []
    if bonferroni(stats["fisher"]) < ALPHA:
        reasons.append("hits")
    if bonferroni(stats["calls_p"]) < ALPHA:
        reasons.append("calls")
    if not math.isfinite(stats["ratio"]) or stats["ratio"] > OUTSIDE_RATIO_LIMIT:
        reasons.append("outside")
    return "pass" if not reasons else "lose:" + ",".join(reasons)


def noise_band(rows: list[dict[str, str]]) -> dict[str, float]:
    first = {row["seed"]: row for row in rows if row["slot"] == "1"}
    second = {row["seed"]: row for row in rows if row["slot"] == "2"}
    seeds = [str(seed) for seed in SEEDS]
    if set(first) != set(seeds) or set(second) != set(seeds):
        raise SystemExit("noise rows are incomplete")
    ratios = []
    for seed in seeds:
        earlier = float(first[seed]["outside_per_call"])
        later = float(second[seed]["outside_per_call"])
        ratios.append(later / earlier)
    replica_one = [row for row in rows if row["replica"] == "1"]
    replica_two = [row for row in rows if row["replica"] == "2"]
    _u, noise_p = mann_whitney(
        column(sorted(replica_two, key=lambda row: int(row["seed"])), "outside_per_call"),
        column(sorted(replica_one, key=lambda row: int(row["seed"])), "outside_per_call"),
        "two-sided",
    )
    return {
        "median": type7(ratios, 0.5),
        "q1": type7(ratios, 0.25),
        "q3": type7(ratios, 0.75),
        "low": min(ratios),
        "high": max(ratios),
        "mw_p": noise_p,
    }


def verify_recorded_balance(rows: list[dict[str, str]]) -> None:
    counts: dict[tuple, int] = {}
    for row in rows:
        if row["block"] != "gate":
            continue
        key = (row["system"], row["slot"], row["arm"])
        counts[key] = counts.get(key, 0) + 1
    if len(counts) != 27 or any(value != 4 for value in counts.values()):
        raise SystemExit("recorded slot counts are not balanced")
    first = [row["replica"] for row in rows if row["block"] == "noise" and row["slot"] == "1"]
    if first.count("1") != 6 or first.count("2") != 6:
        raise SystemExit("recorded noise order is not ABBA")
    for system in CELLS:
        for seed in SEEDS:
            starts = {
                row["start_sha256"]
                for row in rows
                if row["block"] == "gate" and row["system"] == system and row["seed"] == str(seed)
            }
            if len(starts) != 1:
                raise SystemExit(f"{system} seed {seed} does not share a start")


def write_report(args) -> None:
    rows = read_rows(args.rows)
    forbid_portable(rows)
    verify_recorded_balance(rows)
    if len(rows) != len(plan()):
        raise SystemExit("row count differs from the plan")
    sessions = [json.loads(line) for line in args.sessions.read_text().splitlines() if line.strip()]
    starts = [row for row in sessions if row.get("kind") == "start"]
    if len(starts) != 1:
        raise SystemExit("expected one session start")
    session = starts[0]
    for sha in (
        session["base"],
        session["base_tree"],
        session["kit"],
        session["kit_tree"],
        session["arm_a"]["revision"],
        session["arm_a"]["tree"],
        session["arm_kit"]["revision"],
        session["arm_kit"]["tree"],
    ):
        cat_file(ROOT, sha)
    for commit in session["arm_a"]["commits"] + session["arm_kit"]["commits"]:
        cat_file(ROOT, commit["sha"])
        cat_file(ROOT, commit["tree"])
    rel_rows = "bench/nmpins7-gate/mw/rows.csv"
    rel_sessions = "bench/nmpins7-gate/mw/sessions.jsonl"
    rel_harness = "bench/nmpins7-gate/mw/run_gate.py"
    for rel in (rel_rows, rel_sessions, rel_harness):
        blob = git(ROOT, "rev-parse", f"HEAD:{rel}")
        cat_file(ROOT, blob)
        if git(ROOT, "hash-object", str(ROOT / rel)) != blob:
            raise SystemExit(f"{rel} is not the committed blob")
    data_commit = git(ROOT, "rev-parse", "HEAD")
    cat_file(ROOT, data_commit)
    band = noise_band([row for row in rows if row["block"] == "noise"])
    lines = [
        "# nmpins7 gate, shard mw",
        "",
        "Cells m75r14, m80r14 and tip4p20. Seeds 2000-2011. Arms A, B and C ran in one serial session on core 0, one thread, release builds. A is develop plus the M75 table row and the two-hour driver cap. B and C are the nmpins7 kit plus that same cap. B leaves general_symmetry off. C sets general_symmetry. The kit head is an ancestor of the B and C build and is not an ancestor of A.",
        "",
        "A hit is the reference minimum plus 0.001. m80r14 uses the panel reference and paper budget. m75r14 uses the bold Cambridge 75C rho 14 energy and the same 4_000_000 budget the Morse driver assigns from 75 atoms, equal to the m80r14 paper budget. tip4p20 uses the Cambridge TIP4P header and the water-panel budget. Medians and IQRs are Hyndman-Fan type 7. Calls are charged evaluations. Outside time per call is outside-potential time divided by those calls. The ratio is the ratio of medians.",
        "",
        "Fisher p is the exact one-sided test that the named arm has fewer hits than A. Mann-Whitney p is the one-sided normal approximation with the tie and continuity corrections, testing that the named arm is higher than A. A sample that is entirely tied has p = 1. Bonferroni uses m = 30. B loses on hits or calls when the adjusted p is below 0.05, and loses on time when the outside-per-call median ratio exceeds 1.1. C is reported with the same statistics and is not gated.",
        "",
        "## Provenance",
        "",
        f"- record base `{session['base']}` tree `{session['base_tree']}`",
        f"- kit `{session['kit']}` tree `{session['kit_tree']}`",
        f"- arm A build `{session['arm_a']['revision']}` tree `{session['arm_a']['tree']}`",
        f"- kit build `{session['arm_kit']['revision']}` tree `{session['arm_kit']['tree']}`",
        f"- data commit `{data_commit}`",
        f"- rows blob `{git(ROOT, 'rev-parse', 'HEAD:' + rel_rows)}`",
        f"- sessions blob `{git(ROOT, 'rev-parse', 'HEAD:' + rel_sessions)}`",
        f"- harness blob `{git(ROOT, 'rev-parse', 'HEAD:' + rel_harness)}`",
        f"- session `{session['session']}` machine `{session['machine_id']}` core {session['core']} affinity {session['affinity']}",
        f"- rustc `{session['rustc']}`",
        "",
        "Package versions for arm A, then the kit build when the version or revision differs:",
        "",
    ]
    for name in PACKAGES:
        left = session["arm_a"]["packages"][name]
        right = session["arm_kit"]["packages"][name]
        text = f"- {name} {left['version']}"
        if left["revision"]:
            text += f" revision {left['revision']}"
        if right != left:
            text += f"; kit {right['version']}"
            if right["revision"]:
                text += f" revision {right['revision']}"
        lines.append(text)
    lines.extend(
        [
            "",
            "Binary digests:",
            "",
            f"- A Morse `{session['binary_sha256']['a_morse']}`",
            f"- A TIP4P `{session['binary_sha256']['a_seed']}`",
            f"- kit Morse `{session['binary_sha256']['kit_morse']}`",
            f"- kit TIP4P `{session['binary_sha256']['kit_seed']}`",
            "",
            "## Balance",
            "",
            "Each gate cell has four rows for every (slot, arm) pair. The m75r14 noise block runs replica 1 first on six seeds and replica 2 first on six seeds.",
            "",
            "## Results",
            "",
            "| cell | A hits | B hits | C hits | A calls | B calls | C calls | A outside/call | B outside/call | C outside/call |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    details = []
    notes = []
    for system in CELLS:
        base = select(rows, "gate", system, "A")
        other_b = select(rows, "gate", system, "B")
        other_c = select(rows, "gate", system, "C")
        stat_b = compare_arm(base, other_b)
        stat_c = compare_arm(base, other_c)
        lines.append(
            "| {cell} | {ah}/12 | {bh}/12 | {ch}/12 | {ac} ({ai}) | {bc} ({bi}) | {cc} ({ci}) | {ao:.4e} | {bo:.4e} | {co:.4e} |".format(
                cell=system,
                ah=int(stat_b["hits_base"]),
                bh=int(stat_b["hits_other"]),
                ch=int(stat_c["hits_other"]),
                ac=fmt_num(stat_b["calls_base"]),
                ai=fmt_num(stat_b["calls_iqr_base"]),
                bc=fmt_num(stat_b["calls_other"]),
                bi=fmt_num(stat_b["calls_iqr_other"]),
                cc=fmt_num(stat_c["calls_other"]),
                ci=fmt_num(stat_c["calls_iqr_other"]),
                ao=stat_b["out_base"],
                bo=stat_b["out_other"],
                co=stat_c["out_other"],
            )
        )
        inside = band["q1"] <= stat_b["ratio"] <= band["q3"]
        span = band["low"] <= stat_b["ratio"] <= band["high"]
        notes.append(
            f"{system}: calls-to-hit medians A {fmt_num(stat_b['tohit_base'])}, B {fmt_num(stat_b['tohit_other'])}, C {fmt_num(stat_c['tohit_other'])}. Capped seeds A {int(stat_b['capped_base'])}, B {int(stat_b['capped_other'])}, C {int(stat_c['capped_other'])}. The B/A outside median ratio is {'inside' if inside else 'outside'} the A/A per-seed ratio IQR and {'inside' if span else 'outside'} the A/A per-seed ratio range."
        )
        details.append((system, stat_b, stat_c))
    lines.extend(
        [
            "",
            "| cell | Fisher B | Fisher B adj | MW calls B | MW calls B adj | outside B/A | MW outside B | MW outside B two-sided | verdict B | Fisher C | MW calls C | outside C/A | MW outside C |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for system, stat_b, stat_c in details:
        lines.append(
            "| {cell} | {fb} | {fba} | {cb} | {cba} | {rb:.6f} | {ob} | {ot} | {verdict} | {fc} | {cc} | {rc:.6f} | {oc} |".format(
                cell=system,
                fb=fmt_p(stat_b["fisher"]),
                fba=fmt_p(bonferroni(stat_b["fisher"])),
                cb=fmt_p(stat_b["calls_p"]),
                cba=fmt_p(bonferroni(stat_b["calls_p"])),
                rb=stat_b["ratio"],
                ob=fmt_p(stat_b["outside_p"]),
                ot=fmt_p(stat_b["outside_two"]),
                verdict=verdict(stat_b),
                fc=fmt_p(stat_c["fisher"]),
                cc=fmt_p(stat_c["calls_p"]),
                rc=stat_c["ratio"],
                oc=fmt_p(stat_c["outside_p"]),
            )
        )
    lines.extend(["", *notes])
    lines.extend(
        [
            "",
            "## A/A noise on m75r14",
            "",
            f"Same arm A binary, n = 12, ABBA order. Per-seed later/earlier outside-per-call ratios have median {band['median']:.6f}, type-7 IQR {band['q1']:.6f} to {band['q3']:.6f}, and range {band['low']:.6f} to {band['high']:.6f}. Two-sided Mann-Whitney p of replica 2 against replica 1 on outside time per call is {fmt_p(band['mw_p'])}.",
            "",
        ]
    )
    text = "\n".join(lines)
    if "/" in session["machine_id"]:
        raise SystemExit("machine id is not a hash")
    args.report_out.write_text(text)
    print(f"wrote report {len(rows)} rows", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir-a", type=Path)
    parser.add_argument("--bin-dir-kit", type=Path)
    parser.add_argument("--tree-a", type=Path)
    parser.add_argument("--tree-kit", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--rows", type=Path, default=HERE / "rows.csv")
    parser.add_argument("--sessions", type=Path, default=HERE / "sessions.jsonl")
    parser.add_argument("--report-out", type=Path, default=HERE / "report.md")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.schedule:
        show_schedule()
    if args.check or args.run or args.report:
        panel, water, tol = repo_tables()
        if args.check:
            check_arm(args, panel, water, tol)
        if args.run:
            run_campaign(args, panel, water, tol)
        if args.report:
            write_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
