#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial plain-LJ38 probe at the mask budget and the LJ38 paper budget."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE = "10e8c5a04cefee885d3ea093c3768ae1fd4c66d2"
ARM_A = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
ARM_B = "8f1e0b5c5078062d5f241371554555421526452c"
TREE_A = "7a2ba49b37241a8fe42b2c45c226374de6d92274"
TREE_B = "2ba43832b540c6e9005cfb6be31243b36ab086a7"
KIT_SHA256 = "cddaf32978ceeaeedb73d7ebff0f00d5006689c19d110ce8df69fb2881204134"
MASK_BINARY_SHA256 = {
    "A": "a6f7e64c76deeb6411bfde1d3cc2f15513d71b0c4d8adcdb3a54c8a55baba7d3",
    "B": "b0f8152cdbaee2b50695e04b08baba3031471cea45f21c5efc52e2dfbcaa1601",
}
MASK_BUDGET = 30_000
SEEDS = tuple(range(2012, 2060))
ALPHA = 0.05
FAMILY = 4
HIT_TOL = 1.0e-4
REFERENCE = -173.928427

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_MAX_THREADS",
    "GOTO_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
)

ROW_FIELDS = (
    "session",
    "machine_id",
    "sequence",
    "budget",
    "seed",
    "arm",
    "slot",
    "arm_order",
    "hit",
    "calls_to_hit",
    "charged",
    "wall_s",
    "potential_s",
    "outside_s",
    "outside_per_call_s",
    "best",
    "kernel_calls",
    "library_commit",
    "library_tree",
    "runner_blob",
    "probe_blob",
    "binary_sha256",
)

SUMMARY_FIELDS = (
    "budget",
    "n_a",
    "n_b",
    "hits_a",
    "hits_b",
    "fisher_extreme",
    "fisher_total",
    "fisher_p",
    "fisher_p_bonferroni",
    "loss_hits",
    "median_calls_a",
    "median_calls_b",
    "mannwhitney_u_b",
    "mannwhitney_calls_numerator",
    "mannwhitney_calls_denominator",
    "mannwhitney_p_calls",
    "mannwhitney_p_calls_bonferroni",
    "loss_calls",
    "n_hit_a",
    "n_hit_b",
    "median_calls_hit_a",
    "median_calls_hit_b",
    "mannwhitney_u_hit_b",
    "mannwhitney_hit_numerator",
    "mannwhitney_hit_denominator",
    "mannwhitney_p_hit_calls",
    "median_wall_s_a",
    "median_wall_s_b",
    "median_potential_s_a",
    "median_potential_s_b",
    "median_outside_per_call_s_a",
    "median_outside_per_call_s_b",
    "seeds_hit_changed",
    "seeds_calls_changed",
    "verdict",
)

SEED_FIELDS = (
    "budget",
    "seed",
    "hit_a",
    "calls_a",
    "hit_b",
    "calls_b",
    "hit_changed",
    "calls_changed",
)

SCHEDULE_FIELDS = ("budget", "slot", "arm", "count", "expected", "balanced")
PROVENANCE_FIELDS = (
    "arm",
    "library_commit",
    "library_tree",
    "binary_sha256",
    "mask_binary_sha256",
    "probe_blob",
    "runner_blob",
    "kit_sha256",
)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def git(tree: Path, *args: str) -> str:
    result = run_command(["git", *args], cwd=tree)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout.strip()


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


def paper_budget(root: Path) -> int:
    path = root / "scripts/paper_0120/run_panel.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "PAPER_BUDGETS" for target in node.targets):
            continue
        budgets = ast.literal_eval(node.value)
        budget = int(budgets["lj38"])
        if budget != 400_000:
            raise RuntimeError(f"LJ38 paper budget is {budget}")
        return budget
    raise RuntimeError("PAPER_BUDGETS is missing")


def mask_budget(root: Path) -> int:
    text = (root / "bench/nm3-land-gate-a/mask_probe.rs").read_text()
    match = re.search(r"const BUDGET: usize = ([0-9_]+);", text)
    if match is None:
        raise RuntimeError("mask budget constant is missing")
    budget = int(match.group(1).replace("_", ""))
    if budget != MASK_BUDGET:
        raise RuntimeError(f"mask budget is {budget}")
    return budget


def child_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("H2H_")}
    for key in THREAD_KEYS:
        env[key] = "1"
    env["CARGO_BUILD_JOBS"] = "1"
    env["LJOS_SEAT"] = "benchbot"
    return env


def fisher_counts(hits_a: int, n_a: int, hits_b: int, n_b: int) -> tuple[int, int]:
    hits = hits_a + hits_b
    low = max(0, hits - n_b)
    high = min(n_a, hits)
    weights = [
        math.comb(n_a, k) * math.comb(n_b, hits - k) for k in range(low, high + 1)
    ]
    observed = math.comb(n_a, hits_a) * math.comb(n_b, hits_b)
    return sum(weight for weight in weights if weight <= observed), sum(weights)


def mann_whitney(left: list[int], right: list[int]) -> tuple[float, int, int]:
    """Return U for `left` and the exact two-sided probability as a ratio."""
    n1 = len(left)
    pooled = list(left) + list(right)
    order = sorted(range(len(pooled)), key=pooled.__getitem__)
    ranks = [0.0] * len(pooled)
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[order[end + 1]] == pooled[order[index]]:
            end += 1
        average = (index + end + 2) / 2.0
        for cursor in range(index, end + 1):
            ranks[order[cursor]] = average
        index = end + 1
    scaled = [int(round(rank * 2.0)) for rank in ranks]
    if any(abs(rank * 2.0 - value) > 1.0e-6 for rank, value in zip(ranks, scaled)):
        raise RuntimeError("rank scaling is not integral")
    observed = sum(scaled[:n1])
    counts: list[dict[int, int]] = [{} for _ in range(n1 + 1)]
    counts[0][0] = 1
    for rank in scaled:
        for chosen in range(n1 - 1, -1, -1):
            source = counts[chosen]
            if not source:
                continue
            destination = counts[chosen + 1]
            for total, ways in source.items():
                key = total + rank
                destination[key] = destination.get(key, 0) + ways
    distribution = counts[n1]
    total_ways = sum(distribution.values())
    if total_ways != math.comb(len(pooled), n1):
        raise RuntimeError("Mann-Whitney null distribution does not sum to the binomial count")
    low = sum(ways for total, ways in distribution.items() if total <= observed)
    high = sum(ways for total, ways in distribution.items() if total >= observed)
    numerator = min(total_ways, 2 * min(low, high))
    return (observed - n1 * (n1 + 1)) / 2.0, numerator, total_ways


def fraction_less(numerator: int, denominator: int, tests: int = FAMILY) -> bool:
    return numerator * tests * 20 < denominator


def ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0 or numerator < 0:
        raise RuntimeError("probability is not a ratio")
    divisor = math.gcd(numerator, denominator)
    return f"{numerator // divisor}/{denominator // divisor}"


def adjusted_ratio(numerator: int, denominator: int) -> str:
    scaled = FAMILY * numerator
    if scaled >= denominator:
        return "1/1"
    return ratio(scaled, denominator)


def number_text(value: float) -> str:
    return f"{value:.12g}"


def half_text(value: float) -> str:
    twice = round(value * 2.0)
    if abs(value * 2.0 - twice) > 1.0e-6:
        return f"{value:.12g}"
    if twice % 2 == 0:
        return str(twice // 2)
    return f"{twice // 2}.5"


def self_test() -> None:
    extreme, total = fisher_counts(2, 2, 0, 2)
    if extreme != 2 or total != 6:
        raise RuntimeError(f"Fisher self-test returned {extreme}/{total}")
    extreme, total = fisher_counts(4, 4, 0, 4)
    if extreme != 2 or total != 70:
        raise RuntimeError(f"tea-tasting Fisher returned {extreme}/{total}")
    extreme, total = fisher_counts(10, 10, 0, 10)
    if extreme != 2 or total != math.comb(20, 10):
        raise RuntimeError(f"Fisher tail returned {extreme}/{total}")
    extreme, total = fisher_counts(6, 12, 6, 12)
    if extreme != total:
        raise RuntimeError("balanced Fisher table is not probability 1")
    u_value, numerator, denominator = mann_whitney([4, 5, 6], [1, 2, 3])
    if (u_value, numerator, denominator) != (9.0, 2, 20):
        raise RuntimeError(f"untied Mann-Whitney returned {u_value} {numerator}/{denominator}")
    u_value, numerator, denominator = mann_whitney([2, 3], [1, 2])
    if (u_value, numerator, denominator) != (3.5, 4, 6):
        raise RuntimeError(f"tied Mann-Whitney returned {u_value} {numerator}/{denominator}")
    u_value, numerator, denominator = mann_whitney([5, 5], [5, 5])
    if (u_value, numerator, denominator) != (2.0, 6, 6):
        raise RuntimeError(f"equal Mann-Whitney returned {u_value} {numerator}/{denominator}")
    if fraction_less(1, 80) or not fraction_less(1, 81):
        raise RuntimeError("Bonferroni threshold is not strict")
    if ratio(2, 70) != "1/35" or adjusted_ratio(2, 20) != "2/5":
        raise RuntimeError("ratio formatting changed")
    if len(SEEDS) != 48 or SEEDS[0] != 2012 or SEEDS[-1] != 2059:
        raise RuntimeError("seed range is not 2012-2059")
    if ALPHA != 0.05 or FAMILY != 4:
        raise RuntimeError("Bonferroni family changed")


def append_csv(path: Path, fields: tuple[str, ...], row: dict[str, str]) -> None:
    fresh = not path.exists()
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
    result = run_command(command, cwd=cwd, env=env)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "exit="
        + str(result.returncode)
        + "\nstdout:\n"
        + result.stdout
        + "\nstderr:\n"
        + result.stderr
    )
    return result


def verify_arms(a_tree: Path, b_tree: Path, kit: Path) -> None:
    if sha256_file(kit) != KIT_SHA256:
        raise RuntimeError("kit checksum mismatch")
    if git(a_tree, "rev-parse", "HEAD") != ARM_A or git(a_tree, "rev-parse", "HEAD^{tree}") != TREE_A:
        raise RuntimeError("arm A identity mismatch")
    if git(b_tree, "rev-parse", "HEAD") != ARM_B or git(b_tree, "rev-parse", "HEAD^{tree}") != TREE_B:
        raise RuntimeError("arm B identity mismatch")
    if git(a_tree, "status", "--porcelain") or git(b_tree, "status", "--porcelain"):
        raise RuntimeError("an arm worktree is dirty")


def library_hashes(target: Path) -> dict[str, str]:
    return {
        name: sha256_file(target / "release" / "deps" / name)
        for name in ("libanneal_core.rlib", "libanneal_core.a")
    }


def build_probe(
    arm: str,
    tree: Path,
    target: Path,
    probe_source: Path,
    logs: Path,
    env: dict[str, str],
) -> Path:
    destination = tree / "examples" / "lj38_plain_probe.rs"
    if destination.exists():
        raise RuntimeError(f"arm {arm} already has a probe example")
    mask_binary = target / "release" / "examples" / "nm3_land_mask"
    if sha256_file(mask_binary) != MASK_BINARY_SHA256[arm]:
        raise RuntimeError(f"arm {arm} mask binary hash mismatch")
    before = library_hashes(target)
    shutil.copyfile(probe_source, destination)
    try:
        build_env = env.copy()
        build_env["CARGO_TARGET_DIR"] = str(target)
        build_env["TMPDIR"] = str(target.parent / f"tmp-{arm}")
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
        result = logged(
            [
                cargo,
                "+1.88.0",
                "build",
                "--release",
                "--locked",
                "--offline",
                "-j",
                "1",
                "--example",
                "lj38_plain_probe",
            ],
            cwd=tree,
            env=build_env,
            log=logs / "build" / f"{arm}.log",
        )
        if result.returncode != 0:
            raise RuntimeError(f"arm {arm} probe build exited {result.returncode}")
    finally:
        destination.unlink(missing_ok=True)
    if git(tree, "status", "--porcelain"):
        raise RuntimeError(f"arm {arm} worktree changed during the build")
    expected = TREE_A if arm == "A" else TREE_B
    if git(tree, "rev-parse", "HEAD^{tree}") != expected:
        raise RuntimeError(f"arm {arm} tree changed during the build")
    if library_hashes(target) != before:
        raise RuntimeError(f"arm {arm} library artifact changed")
    if sha256_file(mask_binary) != MASK_BINARY_SHA256[arm]:
        raise RuntimeError(f"arm {arm} verified binary changed")
    binary = target / "release" / "examples" / "lj38_plain_probe"
    if not binary.is_file():
        raise RuntimeError(f"arm {arm} probe binary is missing")
    return binary


def parse_result(text: str, seed: int, budget: int) -> dict[str, str]:
    headers = [line for line in text.splitlines() if line.startswith("system,")]
    records = [line for line in text.splitlines() if line.startswith(f"lj38-plain,{seed},{budget},")]
    if len(headers) != 1 or len(records) != 1:
        raise RuntimeError(f"seed {seed} budget {budget}: malformed probe output")
    row = next(csv.DictReader([headers[0], records[0]]))
    charged = int(row["charged"])
    if charged < 1:
        raise RuntimeError(f"seed {seed} budget {budget}: no charged calls")
    kernel_calls = int(row["kernel_calls"])
    if kernel_calls < 1:
        raise RuntimeError(f"seed {seed} budget {budget}: potential timer was not called")
    if row["affinity"] != "0":
        raise RuntimeError(f"seed {seed} budget {budget}: affinity is {row['affinity']}")
    outside = float(row["outside_s"])
    return {
        "hit": "1" if row["hit"] == "1" else "0",
        "calls_to_hit": str(int(row["calls_to_hit"])),
        "charged": str(charged),
        "wall_s": row["wall_s"],
        "potential_s": row["potential_s"],
        "outside_s": f"{outside:.9f}",
        "outside_per_call_s": f"{outside / charged:.12g}",
        "best": row["best"],
        "kernel_calls": str(kernel_calls),
    }


def grouped(rows: list[dict[str, str]], budget: int) -> dict[str, list[dict[str, str]]]:
    selected = [row for row in rows if int(row["budget"]) == budget]
    return {
        arm: sorted(
            (row for row in selected if row["arm"] == arm),
            key=lambda row: int(row["seed"]),
        )
        for arm in ("A", "B")
    }


def summarize_budget(rows: list[dict[str, str]], budget: int) -> tuple[dict[str, str], list[dict[str, str]]]:
    arms = grouped(rows, budget)
    if any(len(group) != len(SEEDS) for group in arms.values()):
        raise RuntimeError(f"budget {budget} is incomplete")
    if [int(row["seed"]) for row in arms["A"]] != list(SEEDS):
        raise RuntimeError(f"budget {budget} arm A seeds mismatch")
    if [int(row["seed"]) for row in arms["B"]] != list(SEEDS):
        raise RuntimeError(f"budget {budget} arm B seeds mismatch")
    hits = {arm: sum(row["hit"] == "1" for row in group) for arm, group in arms.items()}
    calls = {arm: [int(row["calls_to_hit"]) for row in group] for arm, group in arms.items()}
    fisher_num, fisher_den = fisher_counts(hits["A"], len(SEEDS), hits["B"], len(SEEDS))
    u_calls, call_num, call_den = mann_whitney(calls["B"], calls["A"])
    medians = {arm: statistics.median(values) for arm, values in calls.items()}
    loss_hits = hits["B"] < hits["A"] and fraction_less(fisher_num, fisher_den)
    loss_calls = medians["B"] > medians["A"] and fraction_less(call_num, call_den)
    hit_calls = {
        arm: [int(row["calls_to_hit"]) for row in group if row["hit"] == "1"]
        for arm, group in arms.items()
    }
    if hit_calls["A"] and hit_calls["B"]:
        u_hit, hit_num, hit_den = mann_whitney(hit_calls["B"], hit_calls["A"])
        hit_fields = {
            "median_calls_hit_a": half_text(statistics.median(hit_calls["A"])),
            "median_calls_hit_b": half_text(statistics.median(hit_calls["B"])),
            "mannwhitney_u_hit_b": half_text(u_hit),
            "mannwhitney_hit_numerator": str(hit_num),
            "mannwhitney_hit_denominator": str(hit_den),
            "mannwhitney_p_hit_calls": ratio(hit_num, hit_den),
        }
    else:
        hit_fields = {
            "median_calls_hit_a": "",
            "median_calls_hit_b": "",
            "mannwhitney_u_hit_b": "",
            "mannwhitney_hit_numerator": "",
            "mannwhitney_hit_denominator": "",
            "mannwhitney_p_hit_calls": "",
        }
    summary = {
        "budget": str(budget),
        "n_a": str(len(arms["A"])),
        "n_b": str(len(arms["B"])),
        "hits_a": str(hits["A"]),
        "hits_b": str(hits["B"]),
        "fisher_extreme": str(fisher_num),
        "fisher_total": str(fisher_den),
        "fisher_p": ratio(fisher_num, fisher_den),
        "fisher_p_bonferroni": adjusted_ratio(fisher_num, fisher_den),
        "loss_hits": str(int(loss_hits)),
        "median_calls_a": half_text(medians["A"]),
        "median_calls_b": half_text(medians["B"]),
        "mannwhitney_u_b": half_text(u_calls),
        "mannwhitney_calls_numerator": str(call_num),
        "mannwhitney_calls_denominator": str(call_den),
        "mannwhitney_p_calls": ratio(call_num, call_den),
        "mannwhitney_p_calls_bonferroni": adjusted_ratio(call_num, call_den),
        "loss_calls": str(int(loss_calls)),
        "n_hit_a": str(len(hit_calls["A"])),
        "n_hit_b": str(len(hit_calls["B"])),
        **hit_fields,
        "verdict": "FAIL" if loss_hits or loss_calls else "PASS",
    }
    for column, suffix in (
        ("wall_s", "wall_s"),
        ("potential_s", "potential_s"),
        ("outside_per_call_s", "outside_per_call_s"),
    ):
        for arm, label in (("A", "a"), ("B", "b")):
            values = [float(row[column]) for row in arms[arm]]
            summary[f"median_{suffix}_{label}"] = number_text(statistics.median(values))
    seed_rows = []
    hit_changed = 0
    calls_changed = 0
    by_seed = {
        arm: {int(row["seed"]): row for row in group} for arm, group in arms.items()
    }
    for seed in SEEDS:
        a_row = by_seed["A"][seed]
        b_row = by_seed["B"][seed]
        hit_diff = a_row["hit"] != b_row["hit"]
        call_diff = a_row["calls_to_hit"] != b_row["calls_to_hit"]
        hit_changed += int(hit_diff)
        calls_changed += int(call_diff)
        seed_rows.append(
            {
                "budget": str(budget),
                "seed": str(seed),
                "hit_a": a_row["hit"],
                "calls_a": a_row["calls_to_hit"],
                "hit_b": b_row["hit"],
                "calls_b": b_row["calls_to_hit"],
                "hit_changed": str(int(hit_diff)),
                "calls_changed": str(int(call_diff)),
            }
        )
    summary["seeds_hit_changed"] = str(hit_changed)
    summary["seeds_calls_changed"] = str(calls_changed)
    return summary, seed_rows


def schedule_rows(rows: list[dict[str, str]], budgets: tuple[int, int]) -> list[dict[str, str]]:
    output = []
    for budget in budgets:
        selected = [row for row in rows if int(row["budget"]) == budget]
        for slot in ("0", "1"):
            for arm in ("A", "B"):
                count = sum(row["slot"] == slot and row["arm"] == arm for row in selected)
                output.append(
                    {
                        "budget": str(budget),
                        "slot": slot,
                        "arm": arm,
                        "count": str(count),
                        "expected": "24",
                        "balanced": str(int(count == 24)),
                    }
                )
    if any(row["balanced"] != "1" for row in output):
        raise RuntimeError("ABBA counts are not balanced")
    return output


def render_readme(
    *,
    session: str,
    machine: str,
    budgets: tuple[int, int],
    summaries: list[dict[str, str]],
    seed_rows: list[dict[str, str]],
    probe_blob: str,
    runner_blob: str,
) -> str:
    verdict = "FAIL" if any(row["verdict"] == "FAIL" for row in summaries) else "PASS"
    losses = [
        f"budget {row['budget']} "
        + ", ".join(
            name
            for name, flag in (("hits", row["loss_hits"]), ("calls", row["loss_calls"]))
            if flag == "1"
        )
        for row in summaries
        if row["verdict"] == "FAIL"
    ]
    lines = [
        "# Plain LJ38 probe",
        "",
        "verdict: " + verdict + (f" - {'; '.join(losses)}" if losses else ""),
        "",
        "## Budgets",
        "",
        f"- mask-probe budget: `{budgets[0]}`",
        f"- `PAPER_BUDGETS` LJ38 budget: `{budgets[1]}`",
        f"- reference: `{REFERENCE}`",
        f"- hit tolerance: `{HIT_TOL}`",
        "",
        "## Definition",
        "",
        "Both budgets use the plain LJ38 walk: `Config::recommended(38)`, 200 relax steps,",
        "`MoveLibrary::LeanBurst`, stall trail, escape, and restart off, Bayes screening off,",
        "and no surface portfolio. A run spends its full budget. A hit is a recorded best at",
        "or below the reference plus `1e-4`. Calls-to-hit are the first recorded improvement",
        "inside that window, or the budget when no such improvement is present. Potential time",
        "is time inside the Lennard-Jones kernel. Outside time per call divides the remaining",
        "wall time by the charged call count. Probabilities are exact reduced ratios.",
        "",
        "## Provenance",
        "",
        f"- session: `{session}`",
        f"- machine id: `sha256:{machine}`",
        f"- arm A: commit `{ARM_A}`, tree `{TREE_A}`",
        f"- arm B: commit `{ARM_B}`, tree `{TREE_B}`",
        f"- kit: `sha256:{KIT_SHA256}`",
        f"- probe blob: `{probe_blob}`",
        f"- runner blob: `{runner_blob}`",
        "",
        "## Protocol",
        "",
        "The probe executable for each arm was linked to that arm's existing release library.",
        "Library artifact hashes were unchanged by the link. One session ran every job serially",
        "on CPU 0, with every listed thread-pool variable set to 1. Seeds 2012-2059 are common",
        "to both arms. Even offsets use AB and odd offsets use BA. Each `(budget, slot, arm)`",
        "count is 24. The verdict family is exact two-sided Fisher hits and exact two-sided",
        "Mann-Whitney calls with non-hits censored at the budget, at both budgets. Fisher sums",
        "every table whose probability is no larger than the observed table. Mann-Whitney uses",
        "the exact midrank distribution and doubles the smaller tail, capped at 1. Bonferroni",
        f"uses family size {FAMILY}. A hits loss requires fewer B hits. A calls loss requires a",
        "larger B median. Either loss also requires its exact p to be below",
        f"{ALPHA / FAMILY}. The hit-only Mann-Whitney is reported separately and is outside",
        "that family. U counts B-versus-A pairs, with ties scored one half.",
        "",
        "## Results",
        "",
        "| budget | hits A/B | Fisher p | adjusted p | calls A/B | U | MW p | adjusted p | verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['budget']} | {row['hits_a']}/{row['hits_b']} | {row['fisher_p']} | "
            f"{row['fisher_p_bonferroni']} | {row['median_calls_a']}/{row['median_calls_b']} | "
            f"{row['mannwhitney_u_b']} | {row['mannwhitney_p_calls']} | "
            f"{row['mannwhitney_p_calls_bonferroni']} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "| budget | hit-only n A/B | hit-only calls A/B | hit-only U | hit-only p | wall s A/B | potential s A/B | outside/call s A/B |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['budget']} | {row['n_hit_a']}/{row['n_hit_b']} | "
            f"{row['median_calls_hit_a']}/{row['median_calls_hit_b']} | "
            f"{row['mannwhitney_u_hit_b']} | {row['mannwhitney_p_hit_calls']} | "
            f"{row['median_wall_s_a']}/{row['median_wall_s_b']} | "
            f"{row['median_potential_s_a']}/{row['median_potential_s_b']} | "
            f"{row['median_outside_per_call_s_a']}/{row['median_outside_per_call_s_b']} |"
        )
    for budget in budgets:
        selected = [row for row in seed_rows if int(row["budget"]) == budget]
        changed = [row["seed"] for row in selected if row["hit_changed"] == "1"]
        call_changed = sum(row["calls_changed"] == "1" for row in selected)
        lines.extend(
            [
                "",
                f"## Seeds at budget {budget}",
                "",
                f"Hit outcomes differ on {len(changed)} seeds"
                + (f": {', '.join(changed)}" if changed else "")
                + f". Calls differ on {call_changed} seeds.",
                "",
                "| seed | A hit | A calls | B hit | B calls |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            lines.append(
                f"| {row['seed']} | {row['hit_a']} | {row['calls_a']} | "
                f"{row['hit_b']} | {row['calls_b']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    self_test()
    if argv == ["--self-test"]:
        print("ok")
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--a-target", type=Path, required=True)
    parser.add_argument("--b-target", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    budgets = (mask_budget(root), paper_budget(root))
    outputs = (
        "rows.csv",
        "summary.csv",
        "seeds.csv",
        "schedule.csv",
        "provenance.csv",
        "README.md",
    )
    if any((args.out / name).exists() for name in outputs) or args.logs.exists():
        raise RuntimeError("refusing to resume or overwrite an existing probe")
    if hasattr(os, "sched_getaffinity") and os.sched_getaffinity(0) != {0}:
        raise RuntimeError(f"affinity is {sorted(os.sched_getaffinity(0))}")
    env = child_env()
    if os.environ.get("LJOS_SEAT") != "benchbot" or any(env.get(key) != "1" for key in THREAD_KEYS):
        raise RuntimeError("thread environment is incomplete")
    if any(key.startswith("H2H_") for key in env):
        raise RuntimeError("an H2H override is present")
    a_tree = args.a_tree.resolve()
    b_tree = args.b_tree.resolve()
    verify_arms(a_tree, b_tree, args.kit.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True)
    probe_source = Path(__file__).resolve().parent / "probe.rs"
    probe_blob = git(root, "rev-parse", "HEAD:bench/nm3-land-gate-a/lj38-probe/probe.rs")
    runner_blob = git(root, "rev-parse", "HEAD:bench/nm3-land-gate-a/lj38-probe/run_probe.py")
    for blob in (probe_blob, runner_blob):
        git(root, "cat-file", "-e", f"{blob}^{{blob}}")
    binaries = {
        "A": build_probe("A", a_tree, args.a_target.resolve(), probe_source, args.logs, env),
        "B": build_probe("B", b_tree, args.b_target.resolve(), probe_source, args.logs, env),
    }
    binary_hashes = {arm: sha256_file(path) for arm, path in binaries.items()}
    arms = {
        "A": {"commit": ARM_A, "tree": TREE_A, "path": a_tree},
        "B": {"commit": ARM_B, "tree": TREE_B, "path": b_tree},
    }
    started = time.time_ns()
    machine = machine_id()
    session = hashlib.sha256(f"{TREE_B}:{machine}:{started}".encode()).hexdigest()[:20]
    rows_path = args.out / "rows.csv"
    sequence = 0
    print(f"session {session}: budgets {budgets[0]} {budgets[1]}", flush=True)
    for budget in budgets:
        for offset, seed in enumerate(SEEDS):
            order = ("A", "B") if offset % 2 == 0 else ("B", "A")
            for slot, arm in enumerate(order):
                result = logged(
                    [str(binaries[arm]), str(seed), str(budget)],
                    cwd=arms[arm]["path"],
                    env=env,
                    log=args.logs / str(budget) / f"{seed}-{arm}.log",
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"budget {budget} seed {seed} arm {arm} exited {result.returncode}"
                    )
                parsed = parse_result(result.stdout, seed, budget)
                row = {
                    "session": session,
                    "machine_id": machine,
                    "sequence": str(sequence),
                    "budget": str(budget),
                    "seed": str(seed),
                    "arm": arm,
                    "slot": str(slot),
                    "arm_order": "".join(order),
                    "library_commit": arms[arm]["commit"],
                    "library_tree": arms[arm]["tree"],
                    "runner_blob": runner_blob,
                    "probe_blob": probe_blob,
                    "binary_sha256": binary_hashes[arm],
                    **parsed,
                }
                append_csv(rows_path, ROW_FIELDS, row)
                sequence += 1
                print(
                    f"budget {budget} seed {seed} arm {arm} hit={row['hit']} "
                    f"calls={row['calls_to_hit']} wall_s={row['wall_s']}",
                    flush=True,
                )
    rows = read_csv(rows_path)
    if len(rows) != len(budgets) * len(SEEDS) * 2:
        raise RuntimeError("row count mismatch")
    summaries = []
    seed_rows = []
    for budget in budgets:
        summary, seeds = summarize_budget(rows, budget)
        summaries.append(summary)
        seed_rows.extend(seeds)
    schedules = schedule_rows(rows, budgets)
    write_csv(args.out / "summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(args.out / "seeds.csv", SEED_FIELDS, seed_rows)
    write_csv(args.out / "schedule.csv", SCHEDULE_FIELDS, schedules)
    write_csv(
        args.out / "provenance.csv",
        PROVENANCE_FIELDS,
        [
            {
                "arm": arm,
                "library_commit": arms[arm]["commit"],
                "library_tree": arms[arm]["tree"],
                "binary_sha256": binary_hashes[arm],
                "mask_binary_sha256": MASK_BINARY_SHA256[arm],
                "probe_blob": probe_blob,
                "runner_blob": runner_blob,
                "kit_sha256": KIT_SHA256,
            }
            for arm in ("A", "B")
        ],
    )
    for row in rows:
        for field, kind in (
            ("library_commit", "commit"),
            ("library_tree", "tree"),
            ("runner_blob", "blob"),
            ("probe_blob", "blob"),
        ):
            if git(root, "cat-file", "-t", row[field]) != kind:
                raise RuntimeError(f"{field} has the wrong object type")
    verify_arms(a_tree, b_tree, args.kit.resolve())
    (args.out / "README.md").write_text(
        render_readme(
            session=session,
            machine=machine,
            budgets=budgets,
            summaries=summaries,
            seed_rows=seed_rows,
            probe_blob=probe_blob,
            runner_blob=runner_blob,
        )
    )
    verdict = "FAIL" if any(row["verdict"] == "FAIL" for row in summaries) else "PASS"
    print(f"completed verdict={verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
