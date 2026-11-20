#!/usr/bin/env python3
"""Build committed OpenMP arms, measure them, and render the v3 report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

THREADS = (1, 2, 4, 8)
STAGES = ("build", "force")
P_FLOOR_N12 = 1.0 / math.comb(24, 12)
CROSSOVER_MIN_SPEEDUP = {2: 1.2, 8: 2.0}
CROSSOVER_P_LIMIT = 0.01
DEFAULT_BOOTSTRAP_SEED = 173
SAMPLES_NAME = "rgpot_omp_crossover_v3_samples.json.gz"


@dataclass(frozen=True)
class Arm:
    name: str
    sha: str
    threshold: int
    worktree: Path
    build: Path
    worker: Path
    library: Path
    library_sha256: str
    identity: dict[str, int]


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def output(command: list[str], *, cwd: Path | None = None) -> str:
    return run(command, cwd=cwd, capture=True).stdout.strip()


def commit_provenance(repo: Path, sha: str) -> dict[str, Any]:
    tree = output(["git", "rev-parse", sha + "^{tree}"], cwd=repo)
    subject = output(["git", "show", "-s", "--format=%s", sha], cwd=repo)
    if output(["git", "cat-file", "-t", tree], cwd=repo) != "tree":
        raise RuntimeError(f"arm tree does not resolve: {tree}")
    if not subject:
        raise RuntimeError(f"arm commit has no subject: {sha}")
    threshold = tree_threshold_from_git(repo, tree)
    return {
        "commit_sha": sha,
        "tree": tree,
        "commit_tree": tree,
        "tree_threshold": threshold,
        "subject": subject,
    }


def tree_file_from_git(repo: Path, tree: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{tree}:{path}"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else ""


def tree_threshold_from_git(repo: Path, tree: str) -> int:
    options = tree_file_from_git(repo, tree, "meson_options.txt")
    match = re.search(
        r"option\(\s*'team_min_atoms'.*?value:\s*([0-9]+)",
        options,
        flags=re.DOTALL,
    )
    if match is None:
        slices = tree_file_from_git(repo, tree, "CppCore/rgpot/Slices.hpp")
        match = re.search(r"kTeamMinAtoms\s*=\s*([0-9]+)", slices)
    if match is None:
        raise RuntimeError(f"tree {tree} has no kTeamMinAtoms value")
    return int(match.group(1))


def add_arm_provenance(payload: dict[str, Any], repo: Path) -> None:
    for arm in payload["metadata"]["arms"]:
        arm.update(commit_provenance(repo, arm["sha"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arm(text: str) -> tuple[str, str, int]:
    fields = text.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("arm must be NAME:SHA:THRESHOLD")
    name, sha, raw_threshold = fields
    threshold = int(raw_threshold)
    if not name or not re.fullmatch(r"[0-9a-fA-F]{7,40}", sha) or threshold < 1:
        raise argparse.ArgumentTypeError("invalid arm")
    return name, sha, threshold


def require_clean_worktree(path: Path, expected_sha: str) -> tuple[str, str]:
    actual = output(["git", "rev-parse", "HEAD"], cwd=path)
    status = output(["git", "status", "--porcelain"], cwd=path)
    if actual != expected_sha:
        raise RuntimeError(f"worktree HEAD mismatch: expected {expected_sha}, got {actual}")
    if status:
        raise RuntimeError(f"refusing dirty worktree for {actual}: {status}")
    return actual, status


def meson_prefix() -> list[str]:
    executable = shutil.which("meson")
    if executable:
        return [executable]
    return [shutil.which("python3") or "python3", "-m", "mesonbuild.mesonmain"]


def configure_arm(worktree: Path, build: Path) -> None:
    command = meson_prefix() + [
        "setup",
        str(build),
        str(worktree),
        "-Dbuildtype=release",
        "-Dwith_tests=false",
        "-Dwith_examples=false",
        "-Dwith_rpc=false",
        "-Dwith_cache=false",
        "-Dwith_xtensor=false",
        "-Dwith_eigen=false",
        "-Dwith_python=false",
        "-Dwith_fortran_pots=disabled",
        "-Dwith_water=false",
        "-Dwith_ipi=disabled",
        "-Dwith_mpi=disabled",
        "-Dwith_dftd3=false",
        "-Dwith_dftd4=false",
        "-Dwith_expr=false",
        "-Dwith_tblite=false",
        "-Dwith_metatomic=false",
        "-Dwith_xckernel=false",
        "-Dwith_skala=false",
        "-Dwith_long_range=false",
        "-Dwith_highway=disabled",
        "-Dwith_fast_nlist=disabled",
    ]
    run(command)
    run(meson_prefix() + ["compile", "-C", str(build)])


def find_library(build: Path) -> Path:
    candidates = sorted(build.rglob("librgpot.so"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one librgpot.so under {build}, got {len(candidates)}")
    return candidates[0]


def build_worker(
    source: Path,
    worktree: Path,
    build: Path,
    library: Path,
    cxx: str,
    threshold: int,
) -> Path:
    worker = build / "rgpot_omp_worker"
    run(
        [
            cxx,
            "-O3",
            "-DNDEBUG",
            f"-DRGPOT_TEAM_MIN_ATOMS={threshold}",
            "-std=c++20",
            "-fopenmp",
            str(source),
            "-I" + str(worktree / "CppCore"),
            "-I" + str(worktree / "include"),
            "-L" + str(library.parent),
            "-Wl,-rpath," + str(library.parent),
            "-lrgpot",
            "-o",
            str(worker),
        ]
    )
    return worker


def prepare_arm(
    repo: Path,
    worktree_root: Path,
    build_root: Path,
    worker_source: Path,
    cxx: str,
    spec: tuple[str, str, int],
) -> Arm:
    name, sha, threshold = spec
    worktree = worktree_root / name
    build = build_root / name
    if worktree.exists():
        run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo)
    if build.exists():
        shutil.rmtree(build)
    run(["git", "worktree", "add", "--detach", str(worktree), sha], cwd=repo)
    actual_sha, _ = require_clean_worktree(worktree, sha)
    configure_arm(worktree, build)
    require_clean_worktree(worktree, sha)
    library = find_library(build)
    worker = build_worker(worker_source, worktree, build, library, cxx, threshold)
    identity = json.loads(output([str(worker), "--identity"]))
    if identity["configured_team_min_atoms"] != threshold:
        raise RuntimeError(f"{name} worker reports the wrong threshold")
    expected_activation = max(2, threshold)
    if identity["observed_team_activation_atoms"] != expected_activation:
        raise RuntimeError(f"{name} library activates at the wrong atom count")
    return Arm(
        name,
        actual_sha,
        threshold,
        worktree,
        build,
        worker,
        library,
        sha256_file(library),
        identity,
    )


def exclusive_cpus() -> list[int]:
    chosen: list[int] = []
    siblings: set[str] = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        key = topology.read_text(encoding="ascii").strip()
        if key not in siblings:
            siblings.add(key)
            chosen.append(cpu)
    if len(chosen) < max(THREADS):
        raise RuntimeError("eight physical CPUs are required")
    return chosen


def worker_env(threads: int, cpus: list[int]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "OMP_DYNAMIC": "false",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": ",".join(f"{{{cpu}}}" for cpu in cpus),
            "GOMP_CPU_AFFINITY": ",".join(str(cpu) for cpu in cpus),
        }
    )
    return env


def measure(
    arm: Arm,
    potential: str,
    atoms: int,
    threads: int,
    seconds: float,
    cpus: list[int],
) -> dict[str, Any]:
    command = [
        "taskset",
        "-c",
        ",".join(str(cpu) for cpu in cpus),
        str(arm.worker),
        "--potential",
        potential,
        "--atoms",
        str(atoms),
        "--threads",
        str(threads),
        "--samples",
        "1",
        "--sample-seconds",
        f"{seconds:.12g}",
    ]
    return json.loads(
        run(command, cwd=arm.worktree, env=worker_env(threads, cpus), capture=True).stdout
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def speedup_ci(stock: list[float], team: list[float], seed: int) -> tuple[float, float, float]:
    estimate = statistics.median(stock) / statistics.median(team)
    rng = random.Random(seed)
    if len(stock) == 24 and len(team) == 24:
        getrandbits = rng.getrandbits
        samples = []
        for _ in range(2000):
            left = []
            for _ in range(24):
                while (index := getrandbits(5)) >= 24:
                    pass
                left.append(stock[index])
            right = []
            for _ in range(24):
                while (index := getrandbits(5)) >= 24:
                    pass
                right.append(team[index])
            left.sort()
            right.sort()
            samples.append((left[11] + left[12]) / (right[11] + right[12]))
        return estimate, percentile(samples, 0.025), percentile(samples, 0.975)
    samples = []
    for _ in range(2000):
        left = [stock[rng.randrange(len(stock))] for _ in stock]
        right = [team[rng.randrange(len(team))] for _ in team]
        samples.append(statistics.median(left) / statistics.median(right))
    return estimate, percentile(samples, 0.025), percentile(samples, 0.975)


def mann_whitney_u(x: list[float], y: list[float]) -> int:
    merged = sorted([(value, 0) for value in x] + [(value, 1) for value in y])
    ranks = [0.0] * len(merged)
    rank_sum = 0.0
    index = 0
    while index < len(merged):
        end = index + 1
        while end < len(merged) and merged[end][0] == merged[index][0]:
            end += 1
        average = 0.5 * (index + 1 + end)
        for offset in range(index, end):
            ranks[offset] = average
        index = end
    for rank, (_, group) in zip(ranks, merged):
        if group == 0:
            rank_sum += rank
    return math.floor(rank_sum - len(x) * (len(x) + 1) / 2 + 1.0e-12)


@lru_cache(maxsize=None)
def u_counts(nx: int, ny: int) -> list[int]:
    maximum = nx * ny
    table = [[0] * (maximum + 1) for _ in range(nx + 1)]
    table[0][0] = 1
    for _ in range(ny):
        next_table = [[0] * (maximum + 1) for _ in range(nx + 1)]
        for selected in range(nx + 1):
            for u, count in enumerate(table[selected]):
                if count:
                    next_table[selected][u] += count
                    if selected < nx:
                        next_table[selected + 1][u + ny] += count
        table = next_table
    counts = table[nx]
    if sum(counts) != math.comb(nx + ny, nx):
        # Rank-subset recurrence is used when the compact recurrence degenerates.
        total_ranks = nx + ny
        max_sum = total_ranks * (total_ranks + 1) // 2
        dp = [[0] * (max_sum + 1) for _ in range(nx + 1)]
        dp[0][0] = 1
        for rank in range(1, total_ranks + 1):
            for selected in range(min(nx, rank), 0, -1):
                for value in range(max_sum, rank - 1, -1):
                    dp[selected][value] += dp[selected - 1][value - rank]
        offset = nx * (nx + 1) // 2
        counts = [0] * (maximum + 1)
        for value, count in enumerate(dp[nx]):
            if count and 0 <= value - offset <= maximum:
                counts[value - offset] += count
    return counts


def mw_less(x: list[float], y: list[float]) -> float:
    counts = u_counts(len(x), len(y))
    return sum(counts[: mann_whitney_u(x, y) + 1]) / math.comb(len(x) + len(y), len(x))


def machine_metadata(cxx: str, worker: Path, affinity: dict[str, Any]) -> dict[str, Any]:
    cpu_model = "unknown"
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    lscpu = output(["lscpu", "-J"])
    runtime = output(["ldd", str(worker)])
    runtime_line = next((line.strip() for line in runtime.splitlines() if "libgomp" in line), "unknown")
    runtime_path = runtime_line.split("=>", 1)[1].split("(", 1)[0].strip() if "=>" in runtime_line else ""
    runtime_version = output(["strings", runtime_path]) if runtime_path else ""
    versions = sorted(set(re.findall(r"GOMP_[0-9.]+", runtime_version)))
    omp_environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("OMP_") or key.startswith("GOMP_")
    }
    return {
        "cpu_model": cpu_model,
        "lscpu": json.loads(lscpu),
        "compiler": output([cxx, "--version"]).splitlines()[0],
        "libgomp": {
            "library": Path(runtime_path).name if runtime_path else "unknown",
            "symbol_versions": versions,
        },
        "inherited_openmp_environment": omp_environment,
        "benchmark_affinity": affinity,
    }


def summarize_cell(
    potential: str,
    atoms: int,
    threads: int,
    stock: Arm,
    team: Arm,
    rounds: int,
    seconds: float,
    cpus: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    samples: dict[str, list[dict[str, Any]]] = {"stock_a": [], "stock_b": [], "team_a": [], "team_b": []}
    for round_index in range(rounds):
        order = ("stock_a", "team_a", "team_b", "stock_b")
        if round_index % 2:
            order = tuple(reversed(order))
        for label in order:
            arm = stock if label.startswith("stock") else team
            samples[label].append(measure(arm, potential, atoms, threads, seconds, cpus))
    rows = []
    for stage in STAGES:
        key = f"{stage}_ns_per_eval"
        stock_values = [float(item[key][0]) for side in ("stock_a", "stock_b") for item in samples[side]]
        team_values = [float(item[key][0]) for side in ("team_a", "team_b") for item in samples[side]]
        aa_left = [float(item[key][0]) for item in samples["stock_a"]]
        aa_right = [float(item[key][0]) for item in samples["stock_b"]]
        aa_ratios = [left / right for left, right in zip(aa_left, aa_right)]
        speedup, ci_low, ci_high = speedup_ci(stock_values, team_values, seed)
        rows.append(
            {
                "arm": team.name,
                "potential": potential,
                "stage": stage,
                "atoms": atoms,
                "threads": threads,
                "n_per_binary": len(stock_values),
                "order": "ABBA/BAAB",
                "stock_ns": stock_values,
                "team_ns": team_values,
                "speedup": speedup,
                "speedup_ci95": [ci_low, ci_high],
                "mw_p_team_faster": mw_less(team_values, stock_values),
                "aa_band": [min(aa_ratios), max(aa_ratios)],
            }
        )
    return rows


def add_crossovers(payload: dict[str, Any]) -> None:
    rows = payload["measurements"]
    stock_threshold = next(
        arm["threshold"] for arm in payload["metadata"]["arms"] if arm["name"] == "stock"
    )
    crossovers = []
    keys = sorted(
        {
            (row["arm"], row["potential"], row["stage"], row["threads"])
            for row in rows
            if row["threads"] in CROSSOVER_MIN_SPEEDUP
        }
    )
    for arm, potential, stage, threads in keys:
        candidates = sorted(
            (
                row
                for row in rows
                if (row["arm"], row["potential"], row["stage"], row["threads"])
                == (arm, potential, stage, threads)
                and row["atoms"] < stock_threshold
            ),
            key=lambda row: row["atoms"],
        )
        minimum_speedup = CROSSOVER_MIN_SPEEDUP[threads]
        qualifies = [
            row["speedup"] >= minimum_speedup
            and row["speedup"] > row["aa_band"][1]
            and row["mw_p_team_faster"] < CROSSOVER_P_LIMIT
            for row in candidates
        ]
        crossover = None
        for index, row in enumerate(candidates):
            if all(qualifies[index:]):
                crossover = row["atoms"]
                break
        crossovers.append(
            {
                "arm": arm,
                "potential": potential,
                "stage": stage,
                "threads": threads,
                "atoms": crossover,
                "minimum_speedup": minimum_speedup,
            }
        )
    payload["crossovers"] = crossovers


def one_thread_no_regression(
    payload: dict[str, Any],
    arm: str,
    potential: str,
    stage: str,
    stock_threshold: int,
) -> bool:
    rows = [
        row
        for row in payload["measurements"]
        if (row["arm"], row["potential"], row["stage"], row["threads"])
        == (arm, potential, stage, 1)
        and row["atoms"] < stock_threshold
    ]
    return bool(rows) and all(row["speedup"] >= row["aa_band"][0] for row in rows)


def add_recommendation(payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    crossovers = payload["crossovers"]
    stock = next(arm for arm in metadata["arms"] if arm["name"] == "stock")
    teams = [arm for arm in metadata["arms"] if arm["name"] != "stock"]
    phases = sorted(
        {(row["potential"], row["stage"]) for row in payload["measurements"]}
    )
    recommendations = []
    for potential, stage in phases:
        eligible = []
        for arm in teams:
            rows = [
                row
                for row in crossovers
                if (row["arm"], row["potential"], row["stage"])
                == (arm["name"], potential, stage)
            ]
            by_thread = {row["threads"]: row for row in rows}
            if set(by_thread) != set(CROSSOVER_MIN_SPEEDUP):
                continue
            if any(row["atoms"] is None for row in by_thread.values()):
                continue
            if not one_thread_no_regression(
                payload,
                arm["name"],
                potential,
                stage,
                stock["threshold"],
            ):
                continue
            threshold = max(row["atoms"] for row in by_thread.values())
            eligible.append((threshold, arm["threshold"], arm["name"]))
        if eligible:
            threshold, _, supporting_arm = min(eligible)
        else:
            threshold, supporting_arm = stock["threshold"], "stock"
        recommendations.append(
            {
                "potential": potential,
                "stage": stage,
                "team_min_atoms": threshold,
                "supporting_arm": supporting_arm,
            }
        )
    payload["recommendations"] = recommendations
    metadata.pop("recommended_team_min_atoms", None)
    metadata["recommendation"] = (
        "The corrected analysis reports kTeamMinAtoms per potential and phase; "
        "the shipped default is unchanged."
    )


def corrected_analysis(payload: dict[str, Any]) -> None:
    payload["metadata"]["analysis_revision"] = "persistent-below-stock-threshold"
    payload["metadata"]["crossover_rule"] = (
        "The crossover is the smallest measured N below the stock threshold where the team arm "
        "beats stock outside the same-binary A/A band with exact one-sided Mann–Whitney p<0.01, "
        "reaches 1.2x at 2 threads or 2x at 8 threads, and keeps doing so at every larger "
        "measured N below the stock threshold. One-thread measurements must show no slowdown "
        "outside the A/A band."
    )
    add_crossovers(payload)
    add_recommendation(payload)


def format_p(value: float, n: int) -> str:
    rendered = f"{value:.3g}"
    if n == 12 and math.isclose(value, P_FLOOR_N12):
        return f"{rendered} (n=12 exact floor)"
    return rendered


def render_report(
    payload: dict[str, Any], *, explain_team_one_activation: bool = False
) -> str:
    metadata = payload["metadata"]
    lines = [
        "# OpenMP crossover benchmark v3",
        "",
        "## Declared rule",
        "",
        metadata["crossover_rule"],
        "",
        f"For an exact one-sided Mann–Whitney comparison at n=12 per binary, the minimum p-value is {P_FLOOR_N12:.8g} (about 3.7e-7).",
        "",
        "## Provenance and machine",
        "",
        f"- Base and stock: `{metadata['base_sha']}`",
        f"- CPU: {metadata['machine']['cpu_model']}",
        f"- Compiler: {metadata['machine']['compiler']}",
        f"- libgomp: `{metadata['machine']['libgomp']['library']}` "
        f"(`{','.join(metadata['machine']['libgomp']['symbol_versions'])}`)",
        f"- Physical cores used: {len(metadata['affinity']['physical_cpus'])}; logical CPUs reported: {metadata['logical_cpu_count']}",
        f"- Pinning: `{json.dumps(metadata['affinity'], sort_keys=True)}`",
        f"- Inherited OMP/GOMP variables: `{json.dumps(metadata['machine']['inherited_openmp_environment'], sort_keys=True)}`",
        "",
        "Commit SHAs are pre-landing identifiers. Tree SHAs are landing-invariant and identify the applied arm contents.",
        "",
        "|Arm|Pre-landing commit|Landing-invariant tree|Commit subject|Parameter|Observed activation|Library SHA-256|Clean status|",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for arm in metadata["arms"]:
        lines.append(
            f"|{arm['name']}|`{arm['sha']}`|`{arm['tree']}`|{arm['subject']}|{arm['threshold']}|"
            f"{arm['identity']['observed_team_activation_atoms']}|`{arm['library_sha256']}`|"
            f"`{arm['git_status_porcelain']}`|"
        )
    if explain_team_one_activation:
        lines += [
            "",
            "The team-1 activation value comes from the worker --identity probe. "
            "The slice count is capped at the atom count, so one atom remains one "
            "slice and two atoms are the first multi-slice activation.",
        ]
    lines += [
        "",
        "## Crossovers",
        "",
        "|Arm|Potential|Stage|Threads|Smallest persistent N|",
        "|---|---|---|---:|---:|",
    ]
    for row in payload["crossovers"]:
        value = row["atoms"] if row["atoms"] is not None else "none"
        lines.append(f"|{row['arm']}|{row['potential']}|{row['stage']}|{row['threads']}|{value}|")
    lines += [
        "",
        "## Every measured size",
        "",
        "|Arm|Potential|Stage|Threads|N|Speedup|95% CI|A/A band|MW p|n per binary|",
        "|---|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in payload["measurements"]:
        lines.append(
            f"|{row['arm']}|{row['potential']}|{row['stage']}|{row['threads']}|{row['atoms']}|"
            f"{row['speedup']:.4f}|[{row['speedup_ci95'][0]:.4f}, {row['speedup_ci95'][1]:.4f}]|"
            f"[{row['aa_band'][0]:.4f}, {row['aa_band'][1]:.4f}]|"
            f"{format_p(row['mw_p_team_faster'], row['n_per_binary'])}|{row['n_per_binary']}|"
        )
    recommendation = metadata["recommendation"]
    lines += ["", "## Recommendation", "", recommendation, ""]
    if "recommendations" in payload:
        lines += [
            "|Potential|Phase|Recommended kTeamMinAtoms|Supporting arm|",
            "|---|---|---:|---|",
        ]
        for row in payload["recommendations"]:
            lines.append(
                f"|{row['potential']}|{row['stage']}|{row['team_min_atoms']}|"
                f"{row['supporting_arm']}|"
            )
        lines.append("")
    return "\n".join(lines)


def default_samples_path(output_json: Path) -> Path:
    if output_json.name.endswith("_results.json"):
        stem = output_json.name[: -len("_results.json")]
        return output_json.with_name(f"{stem}_samples.json.gz")
    return output_json.with_name(SAMPLES_NAME)


def sample_measurement_key(row: dict[str, Any]) -> tuple[str, str, str, int, int, str]:
    return (
        str(row["arm"]),
        str(row["potential"]),
        str(row["stage"]),
        int(row["atoms"]),
        int(row["threads"]),
        str(row["order"]),
    )


def split_samples(
    payload: dict[str, Any], *, seed: int = DEFAULT_BOOTSTRAP_SEED
) -> dict[str, Any]:
    sample_rows = []
    for index, row in enumerate(payload["measurements"]):
        stock_ns = [float(value) for value in row.pop("stock_ns")]
        team_ns = [float(value) for value in row.pop("team_ns")]
        sample_rows.append(
            {
                "arm": row["arm"],
                "potential": row["potential"],
                "stage": row["stage"],
                "atoms": row["atoms"],
                "threads": row["threads"],
                "order": row["order"],
                "seed": int(row.get("seed", seed + (index // len(STAGES)) * len(STAGES))),
                "stock_ns": stock_ns,
                "team_ns": team_ns,
            }
        )
    return {
        "seed": seed,
        "measurements": sample_rows,
    }


def write_samples_gzip(path: Path, samples_payload: dict[str, Any]) -> None:
    encoded = (json.dumps(samples_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))


def read_samples_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def benchmark(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    if output(["git", "rev-parse", args.base_sha + "^{commit}"], cwd=repo) != args.base_sha:
        raise RuntimeError("base SHA is not the requested full commit")
    specs = [("stock", args.base_sha, 8192)] + args.arm
    args.worktree_root.mkdir(parents=True, exist_ok=True)
    args.build_root.mkdir(parents=True, exist_ok=True)
    arms = [
        prepare_arm(repo, args.worktree_root, args.build_root, args.worker_source.resolve(), args.cxx, spec)
        for spec in specs
    ]
    if len({arm.sha for arm in arms}) != len(arms):
        raise RuntimeError("arm commits are not distinct")
    if len({arm.library_sha256 for arm in arms}) != len(arms):
        raise RuntimeError("arm libraries are not distinct")
    if len({arm.identity["configured_team_min_atoms"] for arm in arms}) != len(arms):
        raise RuntimeError("arm identities are not distinct")
    stock, teams = arms[0], arms[1:]
    cpus = exclusive_cpus()
    sizes = [int(value) for value in args.sizes.split(",")]
    potentials = args.potentials.split(",")
    if sizes[0] > 64 or sizes[-1] < 16384:
        raise RuntimeError("sizes must span 64 through at least 16384")
    measurements = []
    skipped_cells = []
    for team in teams:
        for potential in potentials:
            for atoms in sizes:
                if potential == "emt" and atoms < 256:
                    skipped_cells.append(
                        {
                            "arm": team.name,
                            "potential": potential,
                            "atoms": atoms,
                            "reason": "EMT rejects this periodic cell size",
                        }
                    )
                    continue
                for threads in THREADS:
                    measurements.extend(
                        summarize_cell(
                            potential,
                            atoms,
                            threads,
                            stock,
                            team,
                            args.rounds,
                            args.sample_seconds,
                            cpus[:threads],
                            args.seed + len(measurements),
                        )
                    )
    affinity = {
        "physical_cpus": cpus[: max(THREADS)],
        "policy": "taskset plus OMP_PLACES and GOMP_CPU_AFFINITY; close binding",
        "environment_by_threads": {
            str(threads): {
                key: worker_env(threads, cpus[:threads])[key]
                for key in (
                    "OMP_NUM_THREADS",
                    "OMP_DYNAMIC",
                    "OMP_PROC_BIND",
                    "OMP_PLACES",
                    "GOMP_CPU_AFFINITY",
                )
            }
            for threads in THREADS
        },
    }
    arm_metadata = []
    for arm in arms:
        actual, status = require_clean_worktree(arm.worktree, arm.sha)
        arm_metadata.append(
            {
                "name": arm.name,
                "sha": actual,
                "threshold": arm.threshold,
                "git_status_porcelain": status,
                "library_sha256": arm.library_sha256,
                "identity": arm.identity,
            }
        )
    payload = {
        "metadata": {
            "base_sha": args.base_sha,
            "rounds": args.rounds,
            "n_per_binary": args.rounds * 2,
            "sample_seconds": args.sample_seconds,
            "sizes": sizes,
            "potentials": potentials,
            "threads": list(THREADS),
            "order": "ABBA on even rounds, BAAB on odd rounds",
            "prewarm": False,
            "rescue_runs": False,
            "crossover_rule": (
                "Before measurement: the crossover is the smallest measured N where the team arm "
                "beats stock above the same-binary A/A band with exact one-sided Mann–Whitney p<0.05, "
                "and does so at every larger measured N."
            ),
            "mw_n12_floor": P_FLOOR_N12,
            "logical_cpu_count": os.cpu_count(),
            "affinity": affinity,
            "machine": machine_metadata(args.cxx, stock.worker, affinity),
            "arms": arm_metadata,
        },
        "measurements": measurements,
        "skipped_cells": skipped_cells,
    }
    add_arm_provenance(payload, repo)
    corrected_analysis(payload)
    samples_payload = split_samples(payload, seed=args.seed)
    samples_path = args.output_samples or default_samples_path(args.output_json)
    write_samples_gzip(samples_path, samples_payload)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_report.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--repo", type=Path, required=True)
    measure_parser.add_argument("--worktree-root", type=Path, required=True)
    measure_parser.add_argument("--build-root", type=Path, required=True)
    measure_parser.add_argument("--worker-source", type=Path, required=True)
    measure_parser.add_argument("--output-json", type=Path, required=True)
    measure_parser.add_argument("--output-samples", type=Path)
    measure_parser.add_argument("--output-report", type=Path, required=True)
    measure_parser.add_argument("--base-sha", required=True)
    measure_parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    measure_parser.add_argument("--sizes", required=True)
    measure_parser.add_argument("--potentials", required=True)
    measure_parser.add_argument("--rounds", type=int, required=True)
    measure_parser.add_argument("--sample-seconds", type=float, required=True)
    measure_parser.add_argument("--seed", type=int, required=True)
    measure_parser.add_argument("--cxx", required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--input-json", type=Path, required=True)
    render_parser.add_argument("--output-report", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input-json", type=Path, required=True)
    analyze_parser.add_argument("--output-report", type=Path, required=True)
    clarified_parser = subparsers.add_parser("analyze-v3e")
    clarified_parser.add_argument("--input-json", type=Path, required=True)
    clarified_parser.add_argument("--output-report", type=Path, required=True)
    provenance_parser = subparsers.add_parser("provenance")
    provenance_parser.add_argument("--repo", type=Path, required=True)
    provenance_parser.add_argument("--input-json", type=Path, required=True)
    provenance_parser.add_argument("--output-json", type=Path, required=True)
    split_parser = subparsers.add_parser("split-samples")
    split_parser.add_argument("--input-json", type=Path, required=True)
    split_parser.add_argument("--output-json", type=Path, required=True)
    split_parser.add_argument("--output-samples", type=Path)
    split_parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.command == "split-samples":
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        samples_payload = split_samples(payload, seed=args.seed)
        samples_path = args.output_samples or default_samples_path(args.output_json)
        write_samples_gzip(samples_path, samples_payload)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if args.command == "render":
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        args.output_report.write_text(render_report(payload), encoding="utf-8")
        return
    if args.command == "analyze":
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        corrected_analysis(payload)
        args.output_report.write_text(render_report(payload), encoding="utf-8")
        return
    if args.command == "analyze-v3e":
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        corrected_analysis(payload)
        args.output_report.write_text(
            render_report(payload, explain_team_one_activation=True),
            encoding="utf-8",
        )
        return
    if args.command == "provenance":
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        add_arm_provenance(payload, args.repo.resolve())
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if args.rounds < 12:
        raise RuntimeError("at least 12 ABBA rounds are required")
    if args.sample_seconds <= 0:
        raise RuntimeError("sample seconds must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
