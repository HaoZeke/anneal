#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the fixed-arm nm3-land no-loss gate without retries or resume."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
A_TREE = "7a2ba49b37241a8fe42b2c45c226374de6d92274"
B_TREE = "2ba43832b540c6e9005cfb6be31243b36ab086a7"
KIT_SHA256 = "cddaf32978ceeaeedb73d7ebff0f00d5006689c19d110ce8df69fb2881204134"
ALPHA = 0.05
SEEDS = tuple(range(2000, 2012))
MASK_SEEDS = tuple(range(16))

CELLS = (
    ("lj38", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 400_000, -173.928427),
    ("lj75", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -397.492331),
    ("lj98", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -543.665361),
    ("lj103", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -575.766131),
    ("lj104", "h2h_gmin_proof", "examples/h2h_gmin_proof.rs", 4_000_000, -582.086642),
    ("m80r6", "h2h_morse", "examples/h2h_morse.rs", 4_000_000, -378.333471),
    ("m80r14", "h2h_morse", "examples/h2h_morse.rs", 4_000_000, -340.811371),
    ("au38", "h2h_metal", "examples/h2h_metal.rs", 400_000, -130.54001708),
)

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
)

ROW_FIELDS = (
    "session",
    "machine_id",
    "sequence",
    "cell",
    "seed",
    "arm",
    "slot",
    "arm_order",
    "budget",
    "reference",
    "hit",
    "calls_to_hit",
    "charged",
    "best",
    "wall_s",
    "potential_s",
    "outside_s",
    "outside_per_call_s",
    "library_commit",
    "library_tree",
    "runner_blob",
    "driver_blob",
    "binary_sha256",
)

MASK_FIELDS = (
    "session",
    "machine_id",
    "sequence",
    "seed",
    "arm",
    "slot",
    "arm_order",
    "budget",
    "hit",
    "calls_to_hit",
    "charged",
    "best",
    "library_commit",
    "library_tree",
    "runner_blob",
    "driver_blob",
    "binary_sha256",
)

SUMMARY_FIELDS = (
    "cell",
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
    "median_potential_s_a",
    "median_potential_s_b",
    "median_outside_s_a",
    "median_outside_s_b",
    "median_outside_per_call_s_a",
    "median_outside_per_call_s_b",
    "loss_hits",
    "loss_calls",
    "verdict",
)

SCHEDULE_FIELDS = ("panel", "cell", "slot", "arm", "count", "expected", "balanced")
MASK_SUMMARY_FIELDS = ("arm", "n", "hits", "mask", "median_calls_to_hit")
PROVENANCE_FIELDS = (
    "arm",
    "library_commit",
    "library_tree",
    "binary",
    "binary_sha256",
    "driver_blob",
    "runner_blob",
    "probe_blob",
    "kit_sha256",
)


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stdout + "\n" + result.stderr)[-4000:]
        raise RuntimeError(f"command exited {result.returncode}\n{detail}")
    return result


def git(tree: Path, *args: str) -> str:
    return run(["git", *args], cwd=tree).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def machine_id() -> str:
    value = Path("/etc/machine-id").read_bytes().strip()
    if not value:
        raise RuntimeError("machine id is empty")
    return hashlib.sha256(value).hexdigest()


def child_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("H2H_")}
    for key in THREAD_KEYS:
        env[key] = "1"
    env["CARGO_BUILD_JOBS"] = "1"
    env["LJOS_SEAT"] = "benchbot"
    return env


def append_csv(path: Path, fields: tuple[str, ...], row: dict[str, str]) -> None:
    fresh = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
) -> subprocess.CompletedProcess[str]:
    result = run(command, cwd=cwd, env=env, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        f"exit={result.returncode}\n"
        + "stdout:\n"
        + result.stdout
        + "\nstderr:\n"
        + result.stderr
    )
    return result


def assert_blob(root: Path, blob: str) -> None:
    run(["git", "cat-file", "-e", f"{blob}^{{blob}}"], cwd=root)


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
    commits = git(b_tree, "rev-list", "--reverse", f"{BASE}..{b_head}").splitlines()
    if len(commits) != 2:
        raise RuntimeError(f"arm B has {len(commits)} commits, expected 2")
    if git(a_tree, "status", "--porcelain") or git(b_tree, "status", "--porcelain"):
        raise RuntimeError("an arm worktree is not clean")
    return {
        "A": {"commit": a_head, "tree": a_root},
        "B": {"commit": b_head, "tree": b_root},
    }


def build_arm(
    name: str,
    tree: Path,
    work: Path,
    logs: Path,
    probe_source: Path,
    env: dict[str, str],
) -> dict[str, Path]:
    copied_probe = tree / "examples" / "nm3_land_mask.rs"
    shutil.copyfile(probe_source, copied_probe)
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
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("build executable not found")
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
        "--example",
        "h2h_metal",
        "--example",
        "nm3_land_mask",
    ]
    result = logged(command, cwd=tree, env=build_env, log=logs / "build" / f"{name}.log")
    if result.returncode != 0:
        raise RuntimeError(f"arm {name} release build exited {result.returncode}")
    binaries = {
        binary: target / "release" / "examples" / binary
        for binary in ("h2h_gmin_proof", "h2h_morse", "h2h_metal", "nm3_land_mask")
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
        [str(binary), "budget", cell],
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


def parse_result(text: str, cell: str, seed: int, budget: int) -> dict[str, str]:
    headers = [line for line in text.splitlines() if line.startswith("system,")]
    records = [line for line in text.splitlines() if line.startswith(f"{cell},anneal,{seed},")]
    if len(headers) != 1 or len(records) != 1:
        raise RuntimeError(
            f"{cell} seed {seed}: expected one result, got headers={len(headers)} rows={len(records)}"
        )
    row = next(csv.DictReader([headers[0], records[0]]))
    if int(row["budget"]) != budget:
        raise RuntimeError(f"{cell} seed {seed}: budget {row['budget']} != {budget}")
    wall = float(row["wall_s"])
    potential = float(row["pot_s"])
    outside = float(row.get("outside_s", str(max(wall - potential, 0.0))))
    charged = int(row["charged"])
    return {
        "hit": "1" if row["hit"].lower() in ("1", "true") else "0",
        "calls_to_hit": str(int(row["evals_to_hit"])),
        "charged": str(charged),
        "best": row["best"],
        "wall_s": f"{wall:.9f}",
        "potential_s": f"{potential:.9f}",
        "outside_s": f"{outside:.9f}",
        "outside_per_call_s": f"{outside / max(charged, 1):.12g}",
        "reported_commit": row.get("head", ""),
        "reported_tree": row.get("tree", ""),
        "reported_pins": row.get("pins", ""),
    }


def parse_mask(text: str, seed: int) -> dict[str, str]:
    headers = [line for line in text.splitlines() if line.startswith("system,")]
    records = [line for line in text.splitlines() if line.startswith(f"lj38-plain-mask,{seed},")]
    if len(headers) != 1 or len(records) != 1:
        raise RuntimeError(f"mask seed {seed}: malformed output")
    row = next(csv.DictReader([headers[0], records[0]]))
    return {
        "budget": row["budget"],
        "hit": "1" if row["hit"].lower() in ("1", "true") else "0",
        "calls_to_hit": row["calls_to_hit"],
        "charged": row["charged"],
        "best": row["best"],
    }


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total_hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, total_hits)

    def probability(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, total_hits - k) / ways

    observed = probability(hits_a)
    low = max(0, total_hits - n_b)
    high = min(n_a, total_hits)
    return min(
        1.0,
        sum(
            probability(k)
            for k in range(low, high + 1)
            if probability(k) <= observed * (1.0 + 1.0e-7)
        ),
    )


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
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
    p_value = math.erfc(deviation / math.sqrt(variance) / math.sqrt(2.0))
    return u_value, min(1.0, p_value)


def fmt(value: float) -> str:
    return f"{value:.9g}"


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for cell, _binary, _source, _budget, _reference in CELLS:
        arms = {
            arm: [row for row in rows if row["cell"] == cell and row["arm"] == arm]
            for arm in ("A", "B")
        }
        if any(len(group) != len(SEEDS) for group in arms.values()):
            raise RuntimeError(f"{cell}: incomplete rows")
        hits = {arm: sum(row["hit"] == "1" for row in group) for arm, group in arms.items()}
        calls = {
            arm: [float(row["calls_to_hit"]) for row in group] for arm, group in arms.items()
        }
        fisher_p = fisher(hits["A"], len(SEEDS), hits["B"], len(SEEDS))
        u_calls, p_calls = mann_whitney(calls["B"], calls["A"])
        med_calls = {arm: statistics.median(values) for arm, values in calls.items()}
        loss_hits = hits["B"] < hits["A"] and fisher_p < ALPHA
        loss_calls = med_calls["B"] > med_calls["A"] and p_calls < ALPHA
        summary: dict[str, str] = {
            "cell": cell,
            "n_a": str(len(arms["A"])),
            "n_b": str(len(arms["B"])),
            "hits_a": str(hits["A"]),
            "hits_b": str(hits["B"]),
            "fisher_p": fmt(fisher_p),
            "median_calls_a": fmt(med_calls["A"]),
            "median_calls_b": fmt(med_calls["B"]),
            "mannwhitney_u_b": fmt(u_calls),
            "mannwhitney_p_calls": fmt(p_calls),
            "loss_hits": str(int(loss_hits)),
            "loss_calls": str(int(loss_calls)),
            "verdict": "FIX" if loss_hits or loss_calls else "PASS",
        }
        for field, column in (
            ("wall_s", "wall_s"),
            ("potential_s", "potential_s"),
            ("outside_s", "outside_s"),
            ("outside_per_call_s", "outside_per_call_s"),
        ):
            for arm, suffix in (("A", "a"), ("B", "b")):
                values = [float(row[column]) for row in arms[arm]]
                summary[f"median_{field}_{suffix}"] = fmt(statistics.median(values))
        summaries.append(summary)
    return summaries


def schedule_rows(
    rows: list[dict[str, str]],
    mask_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    panels = [("paper", cell, rows, 6) for cell, *_ in CELLS]
    panels.append(("mask", "lj38-plain-mask", mask_rows, 8))
    for panel, cell, source, expected in panels:
        selected = source if panel == "mask" else [row for row in source if row["cell"] == cell]
        for slot in ("0", "1"):
            for arm in ("A", "B"):
                count = sum(row["slot"] == slot and row["arm"] == arm for row in selected)
                output.append(
                    {
                        "panel": panel,
                        "cell": cell,
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


def mask_summaries(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for arm in ("A", "B"):
        group = sorted(
            (row for row in rows if row["arm"] == arm),
            key=lambda row: int(row["seed"]),
        )
        if len(group) != len(MASK_SEEDS):
            raise RuntimeError(f"mask arm {arm}: incomplete rows")
        mask = sum((row["hit"] == "1") << int(row["seed"]) for row in group)
        output.append(
            {
                "arm": arm,
                "n": str(len(group)),
                "hits": str(sum(row["hit"] == "1" for row in group)),
                "mask": f"0x{mask:04x}",
                "median_calls_to_hit": fmt(
                    statistics.median(float(row["calls_to_hit"]) for row in group)
                ),
            }
        )
    return output


def render_readme(
    *,
    session: str,
    machine: str,
    arms: dict[str, dict[str, str]],
    runner_blob: str,
    probe_blob: str,
    summaries: list[dict[str, str]],
    mask_rows: list[dict[str, str]],
    masks: list[dict[str, str]],
) -> str:
    losses = [row["cell"] for row in summaries if row["verdict"] == "FIX"]
    verdict = "FIX" if losses else "PASS"
    lines = [
        "# nm3-land no-loss gate A",
        "",
        f"verdict: {verdict}"
        + (f" — losses in {', '.join(losses)}" if losses else " — no hit or calls-to-hit loss at p < 0.05"),
        "",
        "## Provenance",
        "",
        f"- session: `{session}`",
        f"- machine id: `sha256:{machine}`",
        f"- arm A: commit `{arms['A']['commit']}`, tree `{arms['A']['tree']}`",
        f"- arm B: commit `{arms['B']['commit']}`, tree `{arms['B']['tree']}`",
        f"- kit: `sha256:{KIT_SHA256}`",
        f"- runner blob: `{runner_blob}`",
        f"- plain-mask probe blob: `{probe_blob}`",
        "",
        "Every recorded object id resolves with `git cat-file -e`. Arm B is exactly two commits",
        "above arm A and has the required tree. The change is exercised with its default settings.",
        "",
        "## Protocol",
        "",
        "All jobs ran in one session, serially, with process affinity fixed to CPU 0 and every",
        "declared thread pool fixed to one thread. Both arms used separate locked release builds.",
        "No budget override, pre-warm, smoke, rescue, retry, or retuning run was used. Seeds",
        "2000–2011 used alternating AB/BA pairs, producing ABBA over each adjacent seed pair.",
        "For every paper cell, each `(slot, arm)` count is 6; for the mask probe each count is 8.",
        "",
        "## Paper-budget cells",
        "",
        "| cell | hits A/B | Fisher p | median calls A/B | MW p | wall s A/B | potential s A/B | outside s A/B | outside/call s A/B | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['cell']} | {row['hits_a']}/{row['hits_b']} | {row['fisher_p']} | "
            f"{row['median_calls_a']}/{row['median_calls_b']} | {row['mannwhitney_p_calls']} | "
            f"{row['median_wall_s_a']}/{row['median_wall_s_b']} | "
            f"{row['median_potential_s_a']}/{row['median_potential_s_b']} | "
            f"{row['median_outside_s_a']}/{row['median_outside_s_b']} | "
            f"{row['median_outside_per_call_s_a']}/{row['median_outside_per_call_s_b']} | "
            f"{row['verdict']} |"
        )
    by_arm_seed = {
        (row["arm"], int(row["seed"])): row
        for row in mask_rows
    }
    mask_by_arm = {row["arm"]: row for row in masks}
    changed_hits = [
        seed
        for seed in MASK_SEEDS
        if by_arm_seed[("A", seed)]["hit"] != by_arm_seed[("B", seed)]["hit"]
    ]
    changed_calls = [
        seed
        for seed in MASK_SEEDS
        if by_arm_seed[("A", seed)]["calls_to_hit"]
        != by_arm_seed[("B", seed)]["calls_to_hit"]
    ]
    lines.extend(
        [
            "",
            "## Plain LJ38 two-phase mask",
            "",
            f"Arm A is `{mask_by_arm['A']['mask']}` ({mask_by_arm['A']['hits']}/16); "
            f"arm B is `{mask_by_arm['B']['mask']}` ({mask_by_arm['B']['hits']}/16). "
            f"Hit outcomes changed on seeds {', '.join(map(str, changed_hits)) or 'none'}; "
            f"calls-to-hit changed on seeds {', '.join(map(str, changed_calls)) or 'none'}.",
            "",
            "| seed | A hit | A calls | B hit | B calls |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in MASK_SEEDS:
        a_row = by_arm_seed[("A", seed)]
        b_row = by_arm_seed[("B", seed)]
        lines.append(
            f"| {seed} | {a_row['hit']} | {a_row['calls_to_hit']} | "
            f"{b_row['hit']} | {b_row['calls_to_hit']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=True)
    output_names = (
        "rows.csv",
        "mask.csv",
        "summary.csv",
        "mask-summary.csv",
        "schedule.csv",
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
    env = child_env()
    if any(key.startswith("H2H_") for key in env):
        raise RuntimeError("an H2H override survived environment sanitization")
    if any(env.get(key) != "1" for key in THREAD_KEYS):
        raise RuntimeError("a thread pool is not fixed to one thread")

    a_tree = args.a_tree.resolve()
    b_tree = args.b_tree.resolve()
    arms = verify_arms(a_tree, b_tree, args.kit.resolve())
    probe_source = args.out / "mask_probe.rs"
    runner_blob = git(root, "rev-parse", "HEAD:bench/nm3-land-gate-a/run_gate.py")
    probe_blob = git(root, "rev-parse", "HEAD:bench/nm3-land-gate-a/mask_probe.rs")
    assert_blob(root, runner_blob)
    assert_blob(root, probe_blob)

    binaries = {
        "A": build_arm("A", a_tree, args.work, args.logs, probe_source, env),
        "B": build_arm("B", b_tree, args.work, args.logs, probe_source, env),
    }
    binary_hashes = {
        arm: {name: sha256_file(path) for name, path in built.items()}
        for arm, built in binaries.items()
    }
    driver_blobs: dict[str, dict[str, str]] = {"A": {}, "B": {}}
    for arm, tree in (("A", a_tree), ("B", b_tree)):
        for _cell, binary, source, _budget, _reference in CELLS:
            driver_blobs[arm][binary] = git(tree, "rev-parse", f"HEAD:{source}")
            assert_blob(root, driver_blobs[arm][binary])
        driver_blobs[arm]["nm3_land_mask"] = probe_blob

    for cell, binary, _source, budget, reference in CELLS:
        for arm, tree in (("A", a_tree), ("B", b_tree)):
            probe_budget(
                arm,
                tree,
                binaries[arm][binary],
                cell,
                budget,
                reference,
                env,
                args.logs,
            )

    started = time.time_ns()
    machine = machine_id()
    session = hashlib.sha256(
        f"{BASE}:{arms['B']['tree']}:{machine}:{started}".encode()
    ).hexdigest()[:20]
    rows_path = args.out / "rows.csv"
    mask_path = args.out / "mask.csv"
    sequence = 0

    print(f"session {session}: running {len(CELLS)} cells", flush=True)
    for cell, binary, _source, budget, reference in CELLS:
        for offset, seed in enumerate(SEEDS):
            order = ("A", "B") if offset % 2 == 0 else ("B", "A")
            for slot, arm in enumerate(order):
                tree = a_tree if arm == "A" else b_tree
                result = logged(
                    [str(binaries[arm][binary]), "run", cell, str(seed)],
                    cwd=tree,
                    env=env,
                    log=args.logs / "runs" / cell / f"{seed}-{arm}.log",
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{cell} seed {seed} arm {arm} exited {result.returncode}"
                    )
                parsed = parse_result(result.stdout, cell, seed, budget)
                if parsed["reported_commit"] and parsed["reported_commit"] != arms[arm]["commit"]:
                    raise RuntimeError(f"{cell} seed {seed} arm {arm}: reported commit mismatch")
                if parsed["reported_tree"] and parsed["reported_tree"] != arms[arm]["tree"]:
                    raise RuntimeError(f"{cell} seed {seed} arm {arm}: reported tree mismatch")
                if parsed["reported_pins"] and parsed["reported_pins"] != "0":
                    raise RuntimeError(f"{cell} seed {seed} arm {arm}: affinity mismatch")
                row = {
                    "session": session,
                    "machine_id": machine,
                    "sequence": str(sequence),
                    "cell": cell,
                    "seed": str(seed),
                    "arm": arm,
                    "slot": str(slot),
                    "arm_order": "".join(order),
                    "budget": str(budget),
                    "reference": f"{reference:.10f}",
                    **{
                        key: parsed[key]
                        for key in (
                            "hit",
                            "calls_to_hit",
                            "charged",
                            "best",
                            "wall_s",
                            "potential_s",
                            "outside_s",
                            "outside_per_call_s",
                        )
                    },
                    "library_commit": arms[arm]["commit"],
                    "library_tree": arms[arm]["tree"],
                    "runner_blob": runner_blob,
                    "driver_blob": driver_blobs[arm][binary],
                    "binary_sha256": binary_hashes[arm][binary],
                }
                append_csv(rows_path, ROW_FIELDS, row)
                sequence += 1
                print(
                    f"{cell} seed={seed} arm={arm} hit={row['hit']} "
                    f"calls={row['calls_to_hit']} wall_s={row['wall_s']}",
                    flush=True,
                )

    print("running plain LJ38 mask", flush=True)
    for offset, seed in enumerate(MASK_SEEDS):
        order = ("A", "B") if offset % 2 == 0 else ("B", "A")
        for slot, arm in enumerate(order):
            tree = a_tree if arm == "A" else b_tree
            result = logged(
                [str(binaries[arm]["nm3_land_mask"]), str(seed)],
                cwd=tree,
                env=env,
                log=args.logs / "mask" / f"{seed}-{arm}.log",
            )
            if result.returncode != 0:
                raise RuntimeError(f"mask seed {seed} arm {arm} exited {result.returncode}")
            parsed = parse_mask(result.stdout, seed)
            row = {
                "session": session,
                "machine_id": machine,
                "sequence": str(sequence),
                "seed": str(seed),
                "arm": arm,
                "slot": str(slot),
                "arm_order": "".join(order),
                **parsed,
                "library_commit": arms[arm]["commit"],
                "library_tree": arms[arm]["tree"],
                "runner_blob": runner_blob,
                "driver_blob": probe_blob,
                "binary_sha256": binary_hashes[arm]["nm3_land_mask"],
            }
            append_csv(mask_path, MASK_FIELDS, row)
            sequence += 1
            print(
                f"mask seed={seed} arm={arm} hit={row['hit']} calls={row['calls_to_hit']}",
                flush=True,
            )

    rows = read_csv(rows_path)
    mask_rows = read_csv(mask_path)
    if len(rows) != len(CELLS) * len(SEEDS) * 2:
        raise RuntimeError("paper row count mismatch")
    if len(mask_rows) != len(MASK_SEEDS) * 2:
        raise RuntimeError("mask row count mismatch")
    summaries = summarize(rows)
    schedules = schedule_rows(rows, mask_rows)
    masks = mask_summaries(mask_rows)
    if {row["arm"]: row["mask"] for row in masks} != {"A": "0x240d", "B": "0x0014"}:
        raise RuntimeError(f"plain mask mismatch: {masks}")

    write_csv(args.out / "summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(args.out / "schedule.csv", SCHEDULE_FIELDS, schedules)
    write_csv(args.out / "mask-summary.csv", MASK_SUMMARY_FIELDS, masks)
    provenance: list[dict[str, str]] = []
    for arm in ("A", "B"):
        for binary in ("h2h_gmin_proof", "h2h_morse", "h2h_metal", "nm3_land_mask"):
            provenance.append(
                {
                    "arm": arm,
                    "library_commit": arms[arm]["commit"],
                    "library_tree": arms[arm]["tree"],
                    "binary": binary,
                    "binary_sha256": binary_hashes[arm][binary],
                    "driver_blob": driver_blobs[arm][binary],
                    "runner_blob": runner_blob,
                    "probe_blob": probe_blob,
                    "kit_sha256": KIT_SHA256,
                }
            )
    write_csv(args.out / "provenance.csv", PROVENANCE_FIELDS, provenance)

    for row in rows + mask_rows:
        for field in ("library_commit", "library_tree", "runner_blob", "driver_blob"):
            object_id = row[field]
            kind = git(root, "cat-file", "-t", object_id)
            expected = "commit" if field == "library_commit" else "tree" if field == "library_tree" else "blob"
            if kind != expected:
                raise RuntimeError(f"{field} {object_id} is {kind}, expected {expected}")
    (args.out / "README.md").write_text(
        render_readme(
            session=session,
            machine=machine,
            arms=arms,
            runner_blob=runner_blob,
            probe_blob=probe_blob,
            summaries=summaries,
            mask_rows=mask_rows,
            masks=masks,
        )
    )
    for tree in (a_tree, b_tree):
        copied_probe = tree / "examples" / "nm3_land_mask.rs"
        copied_probe.unlink()
    print(
        "completed verdict="
        + ("FIX" if any(row["verdict"] == "FIX" for row in summaries) else "PASS"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
