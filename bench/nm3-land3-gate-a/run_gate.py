#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial no-loss gate for the default search path.

Arm A is the pinned base. Arm B is that base plus the four-commit kit.
Both arms use the paper budgets compiled into the drivers. No budget
override is set. Seeds 2000-2011 run in ABBA order on one core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import struct
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

BASE = "61fa67cf380e46876b569da45e7cc091647a775b"
A_TREE = "cff796c507a8fa1d0bfb28888c3570b61988055e"
B_TREE = "fb238a7ed83cda76d4a188b093c0001543b05254"
KIT_SHA256 = "bccb8c5bcd57571ac46333b3f98bbf8e9a07d634f0d3a341becc2f637a41077c"
KIT_COMMITS = 4
ALPHA = 0.05
OUTSIDE_RATIO_LIMIT = 1.1
HIT_TOL = 1.0e-3
SEEDS = tuple(range(2000, 2012))
JOB_TIMEOUT_S = 21600
CORE = "0"

# name, driver, source, budget, reference. Budgets are the paper map:
# 4e5 at 38 and below, 1e6 below 75, 4e6 from 75. References are the
# Cambridge values compiled into the drivers.
CELLS = (
    ("lj38", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 400_000, -173.928427),
    ("lj55", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 1_000_000, -279.248470),
    ("lj75", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -397.492331),
    ("lj98", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -543.665361),
    ("lj102", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -569.363652),
    ("lj103", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -575.766131),
    ("lj104", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -582.086642),
    ("lj150", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -893.310258),
    ("m30r6", "h2h_morse", "examples/h2h_morse.rs", 400_000, -118.432844),
    ("m30r14", "h2h_morse", "examples/h2h_morse.rs", 400_000, -106.835790),
    ("m38r6", "h2h_morse", "examples/h2h_morse.rs", 400_000, -157.477108),
    ("m38r14", "h2h_morse", "examples/h2h_morse.rs", 400_000, -144.321054),
    ("m55r6", "h2h_morse", "examples/h2h_morse.rs", 1_000_000, -250.286609),
    ("m55r14", "h2h_morse", "examples/h2h_morse.rs", 1_000_000, -220.646208),
    ("m70r3", "h2h_morse", "examples/h2h_morse.rs", 1_000_000, -577.739782),
    ("m70r6", "h2h_morse", "examples/h2h_morse.rs", 1_000_000, -325.887749),
    ("m70r10", "h2h_morse", "examples/h2h_morse.rs", 1_000_000, -296.412149),
    ("m70r14", "h2h_morse", "examples/h2h_morse.rs", 1_000_000, -292.462856),
    ("m80r3", "h2h_morse", "examples/h2h_morse.rs", 4_000_000, -690.577890),
    ("m80r6", "h2h_morse", "examples/h2h_morse.rs", 4_000_000, -378.333471),
    ("m80r10", "h2h_morse", "examples/h2h_morse.rs", 4_000_000, -344.911381),
    ("m80r14", "h2h_morse", "examples/h2h_morse.rs", 4_000_000, -340.811371),
)

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
    "TBB_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
)

ROW_FIELDS = (
    "system",
    "arm",
    "seed",
    "slot",
    "core",
    "hit",
    "calls",
    "charged",
    "wall_s",
    "potential_s",
    "outside_s",
    "outside_per_call_s",
    "best",
    "best_bits",
    "budget",
    "machine_id",
    "base_commit",
    "base_tree",
    "library_commit",
    "library_tree",
    "harness_blob",
    "driver_blob",
    "binary_sha256",
    "session",
    "sequence",
    "reference",
    "arm_order",
)

SUMMARY_FIELDS = (
    "system",
    "n_a",
    "n_b",
    "hits_a",
    "hits_b",
    "fisher_p",
    "median_calls_a",
    "median_calls_b",
    "mannwhitney_u_b",
    "mannwhitney_p_calls",
    "median_wall_s_a",
    "median_wall_s_b",
    "median_outside_per_call_s_a",
    "median_outside_per_call_s_b",
    "outside_per_call_ratio_b_over_a",
    "loss_hits",
    "loss_calls",
    "loss_outside",
    "verdict",
)

SCHEDULE_FIELDS = ("system", "slot", "arm", "count", "expected", "balanced")
SEED_FIELDS = ("system", "seed", "hit_a", "hit_b", "calls_a", "calls_b")
PROVENANCE_FIELDS = (
    "arm",
    "library_commit",
    "library_tree",
    "binary",
    "binary_sha256",
    "driver_blob",
    "harness_blob",
    "base_commit",
    "base_tree",
    "kit_sha256",
)


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        detail = (result.stdout + "\n" + result.stderr)[-4000:]
        raise RuntimeError(f"command exited {result.returncode}\n{detail}")
    return result


def git(tree: Path, *args: str) -> str:
    return run(["git", "-C", str(tree), *args], cwd=tree).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def machine_id() -> str:
    value = Path("/etc/machine-id").read_bytes().strip()
    if not value:
        raise RuntimeError("machine id is empty")
    return hashlib.sha256(value).hexdigest()


def child_env() -> dict[str, str]:
    blocked = ("H2H_", "PRIOR_", "GIT_")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(blocked)
    }
    for key in THREAD_KEYS:
        env[key] = "1"
    env["CARGO_BUILD_JOBS"] = "1"
    env["LJOS_SEAT"] = "benchbot"
    return env


def append_csv(path: Path, fields: tuple[str, ...], row: dict[str, str]) -> None:
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(command, cwd=cwd, env=env, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        log.write_text(f"exit=timeout\nstdout:\n{stdout}\nstderr:\n{stderr}")
        raise RuntimeError(f"command exceeded {timeout}s: {command[0]}") from exc
    log.write_text(
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def assert_object(root: Path, object_id: str, kind: str) -> None:
    found = git(root, "cat-file", "-t", object_id)
    if found != kind:
        raise RuntimeError(f"{object_id} is {found}, expected {kind}")


def verify_arms(a_tree: Path, b_tree: Path, kit: Path) -> dict[str, dict[str, str]]:
    if sha256_file(kit) != KIT_SHA256:
        raise RuntimeError("kit checksum mismatch")
    a_head = git(a_tree, "rev-parse", "HEAD")
    a_root = git(a_tree, "rev-parse", "HEAD^{tree}")
    b_head = git(b_tree, "rev-parse", "HEAD")
    b_root = git(b_tree, "rev-parse", "HEAD^{tree}")
    if a_head != BASE or a_root != A_TREE:
        raise RuntimeError(f"arm A mismatch: commit={a_head} tree={a_root}")
    if b_root != B_TREE:
        raise RuntimeError(f"arm B tree mismatch: commit={b_head} tree={b_root}")
    if git(b_tree, "merge-base", BASE, b_head) != BASE:
        raise RuntimeError("arm B is not based on arm A")
    commits = git(b_tree, "rev-list", "--count", f"{BASE}..{b_head}")
    if int(commits) != KIT_COMMITS:
        raise RuntimeError(f"arm B has {commits} commits, expected {KIT_COMMITS}")
    if git(a_tree, "status", "--porcelain") or git(b_tree, "status", "--porcelain"):
        raise RuntimeError("an arm worktree is not clean")
    return {
        "A": {"commit": a_head, "tree": a_root},
        "B": {"commit": b_head, "tree": b_root},
    }


def build_arm(name: str, tree: Path, work: Path, logs: Path, env: dict[str, str]) -> dict[str, Path]:
    target = work / f"target-{name}"
    temporary = work / f"tmp-{name}"
    target.mkdir(parents=True)
    temporary.mkdir(parents=True)
    build_env = env.copy()
    build_env["CARGO_TARGET_DIR"] = str(target)
    build_env["TMPDIR"] = str(temporary)
    cargo_home = build_env.get("CARGO_HOME", str(Path.home() / ".cargo"))
    build_env["RUSTFLAGS"] = " ".join(
        (
            f"--remap-path-prefix={tree}=/src",
            f"--remap-path-prefix={Path.home()}=/user",
            f"--remap-path-prefix={cargo_home}=/cargo",
        )
    )
    cargo = "cargo"
    command = [
        cargo,
        "+1.88.0",
        "build",
        "--release",
        "--locked",
        "-j",
        "1",
        "--example",
        "h2h_gmin_proof",
        "--example",
        "h2h_morse",
    ]
    result = logged(command, cwd=tree, env=build_env, log=logs / "build" / f"{name}.log")
    if result.returncode != 0:
        raise RuntimeError(f"arm {name} release build exited {result.returncode}")
    binaries = {
        binary: target / "release" / "examples" / binary
        for binary in ("h2h_gmin_proof", "h2h_morse")
    }
    for binary, path in binaries.items():
        if not path.is_file():
            raise RuntimeError(f"arm {name} did not build {binary}")
    return binaries


def probe_budget(
    arm: str,
    tree: Path,
    binary: Path,
    cell: str,
    budget: int,
    reference: float,
    env: dict[str, str],
    logs: Path,
) -> None:
    result = logged(
        ["taskset", "-c", CORE, str(binary), "budget", cell],
        cwd=tree,
        env=env,
        log=logs / "budget" / f"{cell}-{arm}.log",
    )
    if result.returncode != 0:
        raise RuntimeError(f"{cell} arm {arm} budget query exited {result.returncode}")
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) < 2 or int(values[0]) != budget:
        raise RuntimeError(f"{cell} arm {arm} budget mismatch: {values}")
    if abs(float(values[1]) - reference) > 1.0e-9:
        raise RuntimeError(f"{cell} arm {arm} reference mismatch: {values[1]}")


def energy_hits(energy: float, reference: float) -> bool:
    return (
        math.isfinite(energy)
        and math.isfinite(reference)
        and energy <= reference + HIT_TOL
    )


def best_bits(token: str) -> str:
    value = float(token)
    if not math.isfinite(value):
        raise RuntimeError(f"best is not finite: {token}")
    packed = struct.pack(">d", value)
    return f"{struct.unpack('>Q', packed)[0]:016x}"


def parse_result(text: str, cell: str, seed: int, budget: int, reference: float) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    headers = [line for line in lines if line.startswith("system,")]
    records = [line for line in lines if line.startswith(f"{cell},anneal,{seed},")]
    if len(headers) != 1 or len(records) != 1:
        raise RuntimeError(
            f"{cell} seed {seed}: expected one result, got headers={len(headers)} rows={len(records)}"
        )
    row = next(csv.DictReader([headers[0], records[0]]))
    if int(row["budget"]) != budget:
        raise RuntimeError(f"{cell} seed {seed}: budget {row['budget']} != {budget}")
    hit = "1" if row["hit"].strip().lower() in ("1", "true") else "0"
    calls = int(row["evals_to_hit"])
    charged = int(row["charged"])
    if calls < 0 or calls > budget or charged < 1 or charged > budget:
        raise RuntimeError(f"{cell} seed {seed}: calls={calls} charged={charged}")
    if hit == "0" and calls != budget:
        raise RuntimeError(f"{cell} seed {seed}: a miss is not at the budget")
    best = row["best"].strip()
    parsed_best = float(best)
    recomputed = energy_hits(parsed_best, reference)
    boundary = reference + HIT_TOL
    if (hit == "1") != recomputed and abs(parsed_best - boundary) > 5.0e-9:
        raise RuntimeError(f"{cell} seed {seed}: hit flag disagrees with the printed best")
    wall = float(row["wall_s"])
    potential = float(row["pot_s"])
    if wall < 0.0 or potential < 0.0:
        raise RuntimeError(f"{cell} seed {seed}: negative time")
    if "outside_s" in row and row["outside_s"] != "":
        outside = float(row["outside_s"])
    else:
        outside = max(wall - potential, 0.0)
    if outside < 0.0:
        raise RuntimeError(f"{cell} seed {seed}: negative outside time")
    return {
        "hit": hit,
        "calls": str(calls),
        "charged": str(charged),
        "best": best,
        "best_bits": best_bits(best),
        "wall_s": f"{wall:.6f}",
        "potential_s": f"{potential:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call_s": format(outside / charged, ".16g"),
        "reported_commit": row.get("head", ""),
        "reported_tree": row.get("tree", ""),
        "reported_pins": row.get("pins", ""),
    }


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total = hits_a + hits_b
    low = max(0, total - n_b)
    high = min(n_a, total)

    def weight(k: int) -> int:
        return math.comb(n_a, k) * math.comb(n_b, total - k)

    observed = weight(hits_a)
    extreme = sum(weight(k) for k in range(low, high + 1) if weight(k) <= observed)
    ways = math.comb(n_a + n_b, total)
    return float(Fraction(extreme, ways))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney, tie correction and continuity correction.

    `left` is arm B. A larger U means B ranks higher (more calls).
    """
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    count = n1 + n2
    ranks: dict[float, float] = {}
    ties = 0.0
    index = 0
    while index < count:
        end = index
        while end + 1 < count and pooled[end + 1] == pooled[index]:
            end += 1
        ranks[pooled[index]] = (index + end + 2) / 2.0
        size = end - index + 1
        ties += size**3 - size
        index = end + 1
    u_value = sum(ranks[value] for value in left) - n1 * (n1 + 1) / 2.0
    variance = n1 * n2 / 12.0 * ((count + 1) - ties / (count * (count - 1)))
    if variance <= 0.0:
        return u_value, 1.0
    deviation = max(abs(u_value - n1 * n2 / 2.0) - 0.5, 0.0)
    p_value = math.erfc(deviation / math.sqrt(2.0 * variance))
    return u_value, min(1.0, p_value)


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fmt(value: float) -> str:
    return format(value, ".12g")


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for cell, _binary, _source, _budget, _reference in CELLS:
        arms = {
            arm: [row for row in rows if row["system"] == cell and row["arm"] == arm]
            for arm in ("A", "B")
        }
        if any(len(group) != len(SEEDS) for group in arms.values()):
            raise RuntimeError(f"{cell}: incomplete rows")
        hits = {arm: sum(row["hit"] == "1" for row in group) for arm, group in arms.items()}
        calls = {arm: [float(row["calls"]) for row in group] for arm, group in arms.items()}
        fisher_p = fisher(hits["A"], len(SEEDS), hits["B"], len(SEEDS))
        u_calls, p_calls = mann_whitney(calls["B"], calls["A"])
        med_calls = {arm: type7(values) for arm, values in calls.items()}
        outside = {
            arm: [float(row["outside_per_call_s"]) for row in group]
            for arm, group in arms.items()
        }
        med_out = {arm: type7(values) for arm, values in outside.items()}
        if med_out["A"] == 0.0:
            ratio = 1.0 if med_out["B"] == 0.0 else math.inf
        else:
            ratio = med_out["B"] / med_out["A"]
        loss_hits = hits["B"] < hits["A"]
        expected_u = len(SEEDS) * len(SEEDS) / 2.0
        loss_calls = p_calls < ALPHA and u_calls > expected_u
        loss_outside = ratio > OUTSIDE_RATIO_LIMIT
        walls = {
            arm: type7([float(row["wall_s"]) for row in group])
            for arm, group in arms.items()
        }
        summaries.append(
            {
                "system": cell,
                "n_a": str(len(arms["A"])),
                "n_b": str(len(arms["B"])),
                "hits_a": str(hits["A"]),
                "hits_b": str(hits["B"]),
                "fisher_p": fmt(fisher_p),
                "median_calls_a": fmt(med_calls["A"]),
                "median_calls_b": fmt(med_calls["B"]),
                "mannwhitney_u_b": fmt(u_calls),
                "mannwhitney_p_calls": fmt(p_calls),
                "median_wall_s_a": fmt(walls["A"]),
                "median_wall_s_b": fmt(walls["B"]),
                "median_outside_per_call_s_a": fmt(med_out["A"]),
                "median_outside_per_call_s_b": fmt(med_out["B"]),
                "outside_per_call_ratio_b_over_a": fmt(ratio),
                "loss_hits": str(int(loss_hits)),
                "loss_calls": str(int(loss_calls)),
                "loss_outside": str(int(loss_outside)),
                "verdict": "FIX" if loss_hits or loss_calls or loss_outside else "PASS",
            }
        )
    return summaries


def schedule_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    expected = len(SEEDS) // 2
    output: list[dict[str, str]] = []
    for cell, *_rest in CELLS:
        selected = [row for row in rows if row["system"] == cell]
        for slot in ("0", "1"):
            for arm in ("A", "B"):
                count = sum(row["slot"] == slot and row["arm"] == arm for row in selected)
                output.append(
                    {
                        "system": cell,
                        "slot": slot,
                        "arm": arm,
                        "count": str(count),
                        "expected": str(expected),
                        "balanced": str(int(count == expected)),
                    }
                )
    if any(row["balanced"] != "1" for row in output):
        raise RuntimeError("ABBA schedule is not balanced")
    return output


def seed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for cell, *_rest in CELLS:
        for seed in SEEDS:
            pair = {
                row["arm"]: row
                for row in rows
                if row["system"] == cell and int(row["seed"]) == seed
            }
            if set(pair) != {"A", "B"}:
                raise RuntimeError(f"{cell} seed {seed}: missing an arm")
            output.append(
                {
                    "system": cell,
                    "seed": str(seed),
                    "hit_a": pair["A"]["hit"],
                    "hit_b": pair["B"]["hit"],
                    "calls_a": pair["A"]["calls"],
                    "calls_b": pair["B"]["calls"],
                }
            )
    return output


def failing_text(summaries: list[dict[str, str]]) -> list[str]:
    named: list[str] = []
    for row in summaries:
        reasons = []
        if row["loss_hits"] == "1":
            reasons.append(f"hits {row['hits_a']}/{row['hits_b']}")
        if row["loss_calls"] == "1":
            reasons.append(f"calls p={row['mannwhitney_p_calls']}")
        if row["loss_outside"] == "1":
            reasons.append(f"outside/call ratio {row['outside_per_call_ratio_b_over_a']}")
        if reasons:
            named.append(f"{row['system']} ({', '.join(reasons)})")
    return named


def render_readme(
    *,
    session: str,
    machine: str,
    arms: dict[str, dict[str, str]],
    harness_blob: str,
    driver_blobs: dict[str, dict[str, str]],
    summaries: list[dict[str, str]],
    seeds: list[dict[str, str]],
) -> str:
    losses = failing_text(summaries)
    verdict = "FIX" if losses else "PASS"
    if losses:
        headline = f"verdict: FIX — {'; '.join(losses)}"
    else:
        headline = (
            "verdict: PASS — no hit loss, no calls loss at p < 0.05, "
            "no outside-time-per-call median ratio above 1.1"
        )
    lines = [
        "# nm3-land3 no-loss gate A",
        "",
        headline,
        "",
        "## Provenance",
        "",
        f"- session: `{session}`",
        f"- machine id: `sha256:{machine}`",
        f"- base: commit `{BASE}`, tree `{A_TREE}`",
        f"- arm A: commit `{arms['A']['commit']}`, tree `{arms['A']['tree']}`",
        f"- arm B: commit `{arms['B']['commit']}`, tree `{arms['B']['tree']}`",
        f"- kit: `sha256:{KIT_SHA256}`",
        f"- harness blob: `{harness_blob}`",
        f"- arm A driver blobs: h2h_gmin_proof `{driver_blobs['A']['h2h_gmin_proof']}`, "
        f"h2h_morse `{driver_blobs['A']['h2h_morse']}`",
        f"- arm B driver blobs: h2h_gmin_proof `{driver_blobs['B']['h2h_gmin_proof']}`, "
        f"h2h_morse `{driver_blobs['B']['h2h_morse']}`",
        "",
        "Every recorded commit, tree, and blob resolves with `git cat-file -e`.",
        "Arm B is four commits above the base and has the required tree.",
        "The drivers run with no H2H setting. A hit is a finite best at or below",
        "the driver reference plus 1e-3. A miss is ranked at the budget.",
        "best is the driver's printed value. best_bits is the IEEE-754 encoding",
        "of that printed value.",
        "",
        "## Protocol",
        "",
        "One session, serial, affinity fixed to core 0, each listed thread pool",
        "fixed to one thread. Separate release builds. Seeds 2000-2011 alternate",
        "AB and BA, which is ABBA across each adjacent pair. Each (slot, arm)",
        "count is 6. Both arms draw the start for that seed; the draws were",
        "compared and matched before the scored runs. No budget override,",
        "pre-warm, smoke, rescue, or retry.",
        "",
        "Lennard-Jones systems are the paper-panel sizes, including LJ38, LJ55,",
        "and LJ75. Morse systems are every cluster compiled into the Morse",
        "driver, including m38r6. Sutton-Chen, binary Lennard-Jones, and TIP4P",
        "are outside this gate. The paper panel has no further in-tree molecular",
        "system on this default path.",
        "",
        "## Systems",
        "",
        "| system | hits A/B | Fisher p | median calls A/B | MW p | wall s A/B | outside/call s A/B | B/A | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['system']} | {row['hits_a']}/{row['hits_b']} | {row['fisher_p']} | "
            f"{row['median_calls_a']}/{row['median_calls_b']} | {row['mannwhitney_p_calls']} | "
            f"{row['median_wall_s_a']}/{row['median_wall_s_b']} | "
            f"{row['median_outside_per_call_s_a']}/{row['median_outside_per_call_s_b']} | "
            f"{row['outside_per_call_ratio_b_over_a']} | {row['verdict']} |"
        )
    lines.extend(["", "## Per-seed hits", ""])
    for index, row in enumerate(seeds):
        if index == 0 or seeds[index - 1]["system"] != row["system"]:
            lines.extend(
                [
                    f"### {row['system']}",
                    "",
                    "| seed | A hit | B hit | A calls | B calls |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
        lines.append(
            f"| {row['seed']} | {row['hit_a']} | {row['hit_b']} | {row['calls_a']} | {row['calls_b']} |"
        )
        if index + 1 == len(seeds) or seeds[index + 1]["system"] != row["system"]:
            lines.append("")
    return "\n".join(lines)


def self_check() -> None:
    lj = parse_result(
        "\n".join(
            (
                "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s",
                "lj38,anneal,2000,400000,true,10,12,0.100000,-173.92842700,abc,def,model,0,0.040000",
            )
        ),
        "lj38",
        2000,
        400_000,
        -173.928427,
    )
    if lj["hit"] != "1" or lj["reported_pins"] != "0":
        raise RuntimeError("parser self-check failed on a hit row")
    morse = parse_result(
        "\n".join(
            (
                "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s",
                "m38r6,anneal,2000,400000,false,400000,400000,1.000000,-150.00000000,0.500000,0.500000",
                "arm surface draws 1 accepts 0 best -150.000000",
            )
        ),
        "m38r6",
        2000,
        400_000,
        -157.477108,
    )
    if morse["hit"] != "0" or morse["calls"] != "400000":
        raise RuntimeError("parser self-check failed on a miss row")
    if abs(fisher(12, 12, 12, 12) - 1.0) > 1.0e-15:
        raise RuntimeError("fisher self-check failed")
    if not energy_hits(-173.927427, -173.928427):
        raise RuntimeError("hit boundary self-check failed")
    if energy_hits(-173.927426, -173.928427):
        raise RuntimeError("hit boundary self-check accepted a miss")


def compare_starts(
    binaries: dict[str, dict[str, Path]],
    a_tree: Path,
    b_tree: Path,
    env: dict[str, str],
    logs: Path,
) -> None:
    for cell, binary, _source, _budget, _reference in CELLS:
        for seed in SEEDS:
            paths = {}
            for arm, tree in (("A", a_tree), ("B", b_tree)):
                path = logs / "starts" / cell / f"{seed}-{arm}.xyz"
                path.parent.mkdir(parents=True, exist_ok=True)
                result = logged(
                    [
                        "taskset",
                        "-c",
                        CORE,
                        str(binaries[arm][binary]),
                        "coords",
                        cell,
                        str(seed),
                        str(path),
                    ],
                    cwd=tree,
                    env=env,
                    log=logs / "starts" / cell / f"{seed}-{arm}.log",
                )
                if result.returncode != 0 or not path.is_file():
                    raise RuntimeError(f"{cell} seed {seed} arm {arm}: start was not written")
                paths[arm] = path.read_bytes()
            if paths["A"] != paths["B"] or not paths["A"]:
                raise RuntimeError(f"{cell} seed {seed}: arm starts differ")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    self_check()
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=True)
    output_names = (
        "rows.csv",
        "summary.csv",
        "schedule.csv",
        "seeds.csv",
        "provenance.csv",
        "README.md",
    )
    if any((args.out / name).exists() for name in output_names):
        raise RuntimeError("refusing to resume or overwrite an existing gate")
    if args.work.exists() or args.logs.exists():
        raise RuntimeError("work and log directories must not already exist")
    args.work.mkdir(parents=True)
    args.logs.mkdir(parents=True)

    if hasattr(os, "sched_getaffinity") and os.sched_getaffinity(0) != {0}:
        raise RuntimeError(f"process affinity is {sorted(os.sched_getaffinity(0))}, expected [0]")
    if os.environ.get("LJOS_SEAT") != "benchbot":
        raise RuntimeError("LJOS_SEAT is not benchbot")
    if any(key.startswith("H2H_") or key.startswith("PRIOR_") for key in os.environ):
        raise RuntimeError("an override is set in the environment")
    env = child_env()
    if any(key.startswith("H2H_") or key.startswith("PRIOR_") for key in env):
        raise RuntimeError("an override survived environment sanitization")
    if any(env.get(key) != "1" for key in THREAD_KEYS):
        raise RuntimeError("a thread pool is not fixed to one thread")

    a_tree = args.a_tree.resolve()
    b_tree = args.b_tree.resolve()
    arms = verify_arms(a_tree, b_tree, args.kit.resolve())
    harness_blob = git(root, "rev-parse", "HEAD:bench/nm3-land3-gate-a/run_gate.py")
    assert_object(root, harness_blob, "blob")

    binaries = {
        "A": build_arm("A", a_tree, args.work, args.logs, env),
        "B": build_arm("B", b_tree, args.work, args.logs, env),
    }
    binary_hashes = {
        arm: {name: sha256_file(path) for name, path in built.items()}
        for arm, built in binaries.items()
    }
    driver_blobs: dict[str, dict[str, str]] = {"A": {}, "B": {}}
    for arm, tree in (("A", a_tree), ("B", b_tree)):
        for binary, source in (
            ("h2h_gmin_proof", "examples/h2h_gmin_proof.rs"),
            ("h2h_morse", "examples/h2h_morse.rs"),
        ):
            driver_blobs[arm][binary] = git(tree, "rev-parse", f"HEAD:{source}")
            assert_object(root, driver_blobs[arm][binary], "blob")

    for cell, binary, _source, budget, reference in CELLS:
        for arm, tree in (("A", a_tree), ("B", b_tree)):
            probe_budget(arm, tree, binaries[arm][binary], cell, budget, reference, env, args.logs)

    print(f"comparing starts for {len(CELLS)} systems", flush=True)
    compare_starts(binaries, a_tree, b_tree, env, args.logs)

    started = time.time_ns()
    machine = machine_id()
    session = hashlib.sha256(
        f"{BASE}:{arms['B']['tree']}:{machine}:{started}".encode()
    ).hexdigest()[:20]
    rows_path = args.out / "rows.csv"
    sequence = 0
    total = len(CELLS) * len(SEEDS) * 2
    print(f"session {session}: running {len(CELLS)} systems, {total} jobs", flush=True)

    for cell, binary, _source, budget, reference in CELLS:
        for offset, seed in enumerate(SEEDS):
            order = ("A", "B") if offset % 2 == 0 else ("B", "A")
            for slot, arm in enumerate(order):
                tree = a_tree if arm == "A" else b_tree
                (args.logs / "progress.txt").write_text(
                    f"{sequence + 1}/{total} {cell} {seed} {arm}\n"
                )
                result = logged(
                    [
                        "timeout",
                        "-k",
                        "30",
                        str(JOB_TIMEOUT_S),
                        "taskset",
                        "-c",
                        CORE,
                        str(binaries[arm][binary]),
                        "run",
                        cell,
                        str(seed),
                    ],
                    cwd=tree,
                    env=env,
                    log=args.logs / "runs" / cell / f"{seed}-{arm}.log",
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{cell} seed {seed} arm {arm} exited {result.returncode}"
                    )
                parsed = parse_result(result.stdout, cell, seed, budget, reference)
                if parsed["reported_commit"] and parsed["reported_commit"] != arms[arm]["commit"]:
                    raise RuntimeError(f"{cell} seed {seed} arm {arm}: reported commit mismatch")
                if parsed["reported_tree"] and parsed["reported_tree"] != arms[arm]["tree"]:
                    raise RuntimeError(f"{cell} seed {seed} arm {arm}: reported tree mismatch")
                if parsed["reported_pins"] and parsed["reported_pins"] != CORE:
                    raise RuntimeError(f"{cell} seed {seed} arm {arm}: affinity mismatch")
                row = {
                    "system": cell,
                    "arm": arm,
                    "seed": str(seed),
                    "slot": str(slot),
                    "core": CORE,
                    "hit": parsed["hit"],
                    "calls": parsed["calls"],
                    "charged": parsed["charged"],
                    "wall_s": parsed["wall_s"],
                    "potential_s": parsed["potential_s"],
                    "outside_s": parsed["outside_s"],
                    "outside_per_call_s": parsed["outside_per_call_s"],
                    "best": parsed["best"],
                    "best_bits": parsed["best_bits"],
                    "budget": str(budget),
                    "machine_id": machine,
                    "base_commit": BASE,
                    "base_tree": A_TREE,
                    "library_commit": arms[arm]["commit"],
                    "library_tree": arms[arm]["tree"],
                    "harness_blob": harness_blob,
                    "driver_blob": driver_blobs[arm][binary],
                    "binary_sha256": binary_hashes[arm][binary],
                    "session": session,
                    "sequence": str(sequence),
                    "reference": f"{reference:.10f}",
                    "arm_order": "".join(order),
                }
                append_csv(rows_path, ROW_FIELDS, row)
                sequence += 1
                print(
                    f"{sequence}/{total} {cell} seed={seed} arm={arm} hit={row['hit']} "
                    f"calls={row['calls']} wall_s={row['wall_s']}",
                    flush=True,
                )

    rows = read_csv(rows_path)
    if len(rows) != total:
        raise RuntimeError("row count mismatch")
    for row in rows:
        for field, kind in (
            ("library_commit", "commit"),
            ("library_tree", "tree"),
            ("base_commit", "commit"),
            ("base_tree", "tree"),
            ("harness_blob", "blob"),
            ("driver_blob", "blob"),
        ):
            assert_object(root, row[field], kind)
    summaries = summarize(rows)
    schedules = schedule_rows(rows)
    seeds = seed_rows(rows)
    write_csv(args.out / "summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(args.out / "schedule.csv", SCHEDULE_FIELDS, schedules)
    write_csv(args.out / "seeds.csv", SEED_FIELDS, seeds)
    provenance: list[dict[str, str]] = []
    for arm in ("A", "B"):
        for binary in ("h2h_gmin_proof", "h2h_morse"):
            provenance.append(
                {
                    "arm": arm,
                    "library_commit": arms[arm]["commit"],
                    "library_tree": arms[arm]["tree"],
                    "binary": binary,
                    "binary_sha256": binary_hashes[arm][binary],
                    "driver_blob": driver_blobs[arm][binary],
                    "harness_blob": harness_blob,
                    "base_commit": BASE,
                    "base_tree": A_TREE,
                    "kit_sha256": KIT_SHA256,
                }
            )
    write_csv(args.out / "provenance.csv", PROVENANCE_FIELDS, provenance)
    text = render_readme(
        session=session,
        machine=machine,
        arms=arms,
        harness_blob=harness_blob,
        driver_blobs=driver_blobs,
        summaries=summaries,
        seeds=seeds,
    )
    readme = args.out / "README.md"
    readme.write_text(text)
    if "\r" in text:
        raise RuntimeError("verdict text contains CR")
    verdict = "FIX" if any(row["verdict"] == "FIX" for row in summaries) else "PASS"
    print(f"completed verdict={verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
