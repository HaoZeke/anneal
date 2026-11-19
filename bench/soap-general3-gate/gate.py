#!/usr/bin/env python3
"""Run and report the default-on SOAP cloud-gradient gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

A_COMMIT = "dd054749d4703923f4cf4c34f50589c574ee1276"
A_TREE = "f1705d6f02f7f257f50e831c16057740283efea5"
B_TREE = "e497d9777a725b0d9d732c6dadcf1f0e20aaa68f"
HIT_TOL = 1.0e-3
ALPHA = 0.05
SEEDS = tuple(range(2000, 2012))


@dataclass(frozen=True)
class System:
    name: str
    kind: str
    n: int
    rho: int
    budget: int
    reference: float | None
    reference_source: str


SYSTEMS = (
    System("lj38", "lj", 38, 0, 400_000, -173.928427, "scripts/gmin_panel/run_panel.py"),
    System("lj75", "lj", 75, 0, 4_000_000, -397.492331, "scripts/gmin_panel/run_panel.py"),
    System("lj98", "lj", 98, 0, 4_000_000, -543.665361, "scripts/gmin_panel/run_panel.py"),
    System("lj150", "lj", 150, 0, 4_000_000, -893.310258, "scripts/gmin_panel/run_panel.py"),
    System("m55r14", "morse", 55, 14, 1_000_000, -220.646208, "scripts/gmin_panel/run_panel.py"),
    System("m80r6", "morse", 80, 6, 4_000_000, -378.333471, "scripts/gmin_panel/run_panel.py"),
    System(
        "m38r10",
        "morse",
        38,
        10,
        400_000,
        -145.849817,
        "scripts/ase_nonlj/run_one.py@cf6833e1316cf8c7a1bb7879a477ef2b32e9d8f5",
    ),
    System("tip4p8", "tip4p", 8, 0, 2_000_000, -305.5183221676, "scripts/tip4p_h2h/run_h2h.py"),
    System("tip4p15", "tip4p", 15, 0, 2_000_000, -628.3722525077, "scripts/tip4p_h2h/run_h2h.py"),
    System("tip4p20", "tip4p", 20, 0, 2_000_000, -872.9887544952, "scripts/tip4p_h2h/run_h2h.py"),
    System("tip4p21", "tip4p", 21, 0, 2_000_000, -916.7074302940, "scripts/tip4p_h2h/run_h2h.py"),
    System("tip4p25", "tip4p", 25, 0, 2_000_000, None, "scripts/tip4p_h2h/run_h2h.py"),
)
AA_SYSTEMS = frozenset(("lj75", "m80r6", "tip4p15"))

ROW_FIELDS = (
    "session",
    "panel",
    "system",
    "seed",
    "arm",
    "arm_order",
    "commit",
    "tree",
    "budget",
    "reference",
    "reference_source",
    "calls",
    "charged",
    "best",
    "hit",
    "wall_s",
    "pot_s",
    "outside_s",
    "machine_id_hash",
    "cpu_model",
    "core",
    "affinity",
)

SUMMARY_FIELDS = (
    "system",
    "n_a",
    "n_b",
    "hits_a",
    "hits_b",
    "fisher_p_exact",
    "median_calls_a",
    "median_calls_b",
    "mann_whitney_u_calls_b",
    "mann_whitney_p_calls_exact",
    "median_wall_s_a",
    "median_wall_s_b",
    "median_pot_s_a",
    "median_pot_s_b",
    "median_outside_s_a",
    "median_outside_s_b",
    "mann_whitney_u_outside_b",
    "mann_whitney_p_outside_exact",
    "outside_delta_rel",
    "aa_noise_band_rel",
    "verdict",
    "reason",
)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", newline="\n") as handle:
        handle.write("command=" + " ".join(command) + "\n")
        handle.write(f"exit={process.returncode}\n")
        handle.write(process.stdout)
        handle.write(process.stderr)
    if process.returncode != 0:
        raise RuntimeError(f"command failed with exit {process.returncode}; see {log}")
    return process


def git_value(tree: Path, expression: str) -> str:
    return subprocess.run(
        ["git", "-C", str(tree), "rev-parse", expression],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()


def verify_source(tree: Path, expected_commit: str | None, expected_tree: str) -> tuple[str, str]:
    commit = git_value(tree, "HEAD")
    tree_sha = git_value(tree, "HEAD^{tree}")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(f"source commit is {commit}, expected {expected_commit}")
    if tree_sha != expected_tree:
        raise RuntimeError(f"source tree is {tree_sha}, expected {expected_tree}")
    status = subprocess.run(
        ["git", "-C", str(tree), "status", "--porcelain"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    if status:
        raise RuntimeError(f"source worktree is not clean: {tree}")
    return commit, tree_sha


def build_arm(
    label: str,
    source: Path,
    cargo: Path,
    work: Path,
    logs: Path,
) -> Path:
    target = work / f"target-{label}"
    target.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["CARGO_INCREMENTAL"] = "0"
    env["TMPDIR"] = str(work / f"tmp-{label}")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=False)
    run_command(
        [
            str(cargo),
            "+1.88.0",
            "build",
            "--release",
            "--locked",
            "-j",
            "1",
            "--example",
            "h2h_seed",
        ],
        cwd=source,
        env=env,
        log=logs / f"build-{label}.log",
    )
    binary = work / "bin" / f"h2h_seed-{label}"
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target / "release" / "examples" / "h2h_seed", binary)
    binary.chmod(0o755)
    return binary


def machine_id_hash() -> str:
    machine_id = Path("/etc/machine-id").read_bytes().strip()
    if not machine_id:
        raise RuntimeError("machine id is empty")
    return hashlib.sha256(machine_id).hexdigest()


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip().replace(",", " ")
    raise RuntimeError("CPU model is unavailable")


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[key] = "1"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    return env


def parse_result(stdout: str, system: System) -> dict[str, str]:
    lines = [line for line in stdout.splitlines() if line.startswith("h2h_row ")]
    if len(lines) != 1:
        raise RuntimeError(f"{system.name}: expected one h2h_row, found {len(lines)}")
    fields = dict(token.split("=", 1) for token in lines[0].split()[1:] if "=" in token)
    if fields.get("system") != f"{system.kind}{system.n}":
        raise RuntimeError(f"{system.name}: malformed system field {fields.get('system')}")
    if int(fields["budget"]) != system.budget:
        raise RuntimeError(f"{system.name}: malformed budget {fields['budget']}")
    if int(fields["charged"]) != system.budget:
        raise RuntimeError(f"{system.name}: run stopped at {fields['charged']} charged calls")
    wall = float(fields["wall_s"])
    pot = float(fields["pot_s"])
    outside = float(fields["outside_s"])
    if min(wall, pot, outside) < 0.0 or abs(max(wall - pot, 0.0) - outside) > 3.0e-6:
        raise RuntimeError(f"{system.name}: inconsistent timing fields")
    return {
        "charged": fields["charged"],
        "best": f"{float(fields['best']):.10f}",
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
    }


def calls_to_reference(trace: Path, reference: float, budget: int) -> tuple[int, int]:
    with trace.open(newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            if float(row["energy"]) <= reference + HIT_TOL:
                return int(row["calls"]), 1
    return budget, 0


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def execute_run(args: argparse.Namespace) -> None:
    if args.rows.exists():
        raise RuntimeError(f"rows output already exists: {args.rows}")
    if args.work.exists():
        raise RuntimeError(f"work directory already exists: {args.work}")
    core = args.core
    affinity = os.sched_getaffinity(0)
    if core not in affinity:
        raise RuntimeError(f"core {core} is not in the process affinity")

    source_a = args.a_source.resolve()
    source_b = args.b_source.resolve()
    source_aa = args.aa_source.resolve()
    a_commit, a_tree = verify_source(source_a, A_COMMIT, A_TREE)
    b_commit, b_tree = verify_source(source_b, None, B_TREE)
    if git_value(source_b, "HEAD^") != A_COMMIT:
        raise RuntimeError("B is not the single child of A")
    aa_commit, aa_tree = verify_source(source_aa, A_COMMIT, A_TREE)

    args.work.mkdir(parents=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    machine_hash = machine_id_hash()
    cpu = cpu_model()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session_record: dict[str, object] = {
        "session": session,
        "started_utc": started,
        "status": "started",
        "machine_id_hash": machine_hash,
        "cpu_model": cpu,
        "core": core,
        "affinity": str(core),
        "seeds": list(SEEDS),
        "arms": {
            "A": {"commit": a_commit, "tree": a_tree},
            "B": {"commit": b_commit, "tree": b_tree},
            "A1": {"commit": a_commit, "tree": a_tree},
            "A2": {"commit": aa_commit, "tree": aa_tree},
        },
    }
    append_jsonl(args.sessions, session_record)

    binaries = {
        "A": build_arm("A", source_a, args.cargo, args.work, args.logs),
        "B": build_arm("B", source_b, args.cargo, args.work, args.logs),
        "A1": None,
        "A2": build_arm("A2", source_aa, args.cargo, args.work, args.logs),
    }
    binaries["A1"] = binaries["A"]
    identities = {
        "A": (a_commit, a_tree),
        "B": (b_commit, b_tree),
        "A1": (a_commit, a_tree),
        "A2": (aa_commit, aa_tree),
    }
    env = single_thread_env()
    pending: list[dict[str, object]] = []

    for seed in SEEDS:
        ab_order = ("A", "B") if seed % 2 == 0 else ("B", "A")
        aa_order = ("A1", "A2") if seed % 2 == 0 else ("A2", "A1")
        for system in SYSTEMS:
            panels = (("AB", ab_order),)
            if system.name in AA_SYSTEMS:
                panels += (("AA", aa_order),)
            for panel, order in panels:
                for arm in order:
                    prefix = args.logs / "rows" / panel / system.name / f"{seed}-{arm}"
                    prefix.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        "taskset",
                        "-c",
                        str(core),
                        str(binaries[arm]),
                        system.kind,
                        str(system.n),
                        str(system.rho),
                        str(system.budget),
                        str(seed),
                        "recommended",
                        "-",
                        str(prefix),
                    ]
                    process = run_command(
                        command,
                        cwd=args.work,
                        env=env,
                        log=prefix.with_suffix(".log"),
                    )
                    parsed = parse_result(process.stdout, system)
                    commit, tree_sha = identities[arm]
                    pending.append(
                        {
                            "session": session,
                            "panel": panel,
                            "system": system,
                            "seed": seed,
                            "arm": arm,
                            "arm_order": "".join(order),
                            "commit": commit,
                            "tree": tree_sha,
                            "trace": prefix.with_suffix(".trace.csv"),
                            **parsed,
                        }
                    )
                    print(
                        f"{len(pending)}/360 {panel} {system.name} seed={seed} arm={arm} "
                        f"best={parsed['best']} wall_s={parsed['wall_s']}",
                        flush=True,
                    )

    tip25_primary = [
        float(row["best"])
        for row in pending
        if row["panel"] == "AB" and row["system"].name == "tip4p25"
    ]
    if len(tip25_primary) != 24:
        raise RuntimeError("TIP4P25 primary pool is incomplete")
    pooled_tip25 = min(tip25_primary)

    rows: list[dict[str, str]] = []
    for pending_row in pending:
        system = pending_row["system"]
        reference = pooled_tip25 if system.reference is None else system.reference
        calls, hit = calls_to_reference(pending_row["trace"], reference, system.budget)
        if (float(pending_row["best"]) <= reference + HIT_TOL) != bool(hit):
            raise RuntimeError(f"{system.name}: trace and final best disagree on hit")
        rows.append(
            {
                "session": str(pending_row["session"]),
                "panel": str(pending_row["panel"]),
                "system": system.name,
                "seed": str(pending_row["seed"]),
                "arm": str(pending_row["arm"]),
                "arm_order": str(pending_row["arm_order"]),
                "commit": str(pending_row["commit"]),
                "tree": str(pending_row["tree"]),
                "budget": str(system.budget),
                "reference": f"{reference:.10f}",
                "reference_source": system.reference_source,
                "calls": str(calls),
                "charged": str(pending_row["charged"]),
                "best": str(pending_row["best"]),
                "hit": str(hit),
                "wall_s": str(pending_row["wall_s"]),
                "pot_s": str(pending_row["pot_s"]),
                "outside_s": str(pending_row["outside_s"]),
                "machine_id_hash": machine_hash,
                "cpu_model": cpu,
                "core": str(core),
                "affinity": str(core),
            }
        )
    write_csv(args.rows, ROW_FIELDS, rows)
    append_jsonl(
        args.sessions,
        {
            **session_record,
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "completed",
            "rows": len(rows),
            "tip4p25_pooled_reference": f"{pooled_tip25:.10f}",
        },
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(rows[0]) != ROW_FIELDS:
        raise RuntimeError("rows file has an unexpected schema")
    return rows


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total_hits = hits_a + hits_b
    denominator = math.comb(n_a + n_b, total_hits)

    def probability(left_hits: int) -> float:
        right_hits = total_hits - left_hits
        if not (0 <= left_hits <= n_a and 0 <= right_hits <= n_b):
            return 0.0
        return math.comb(n_a, left_hits) * math.comb(n_b, right_hits) / denominator

    observed = probability(hits_a)
    low = max(0, total_hits - n_b)
    high = min(n_a, total_hits)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(low, high + 1)
            if probability(value) <= observed * (1.0 + 1.0e-12)
        ),
    )


def exact_mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    if not left or not right:
        raise ValueError("both samples are required")
    n_left = len(left)
    pooled = sorted(left + right)
    doubled_rank: dict[float, int] = {}
    index = 0
    while index < len(pooled):
        end = index + 1
        while end < len(pooled) and pooled[end] == pooled[index]:
            end += 1
        doubled_rank[pooled[index]] = (index + 1) + end
        index = end
    ranks = [doubled_rank[value] for value in left + right]
    observed_sum = sum(doubled_rank[value] for value in left)
    distributions: list[defaultdict[int, int]] = [defaultdict(int) for _ in range(n_left + 1)]
    distributions[0][0] = 1
    used = 0
    for rank in ranks:
        used += 1
        for count in range(min(used, n_left), 0, -1):
            for rank_sum, ways in tuple(distributions[count - 1].items()):
                distributions[count][rank_sum + rank] += ways
    distribution = distributions[n_left]
    total = math.comb(len(ranks), n_left)
    lower = sum(ways for rank_sum, ways in distribution.items() if rank_sum <= observed_sum)
    upper = sum(ways for rank_sum, ways in distribution.items() if rank_sum >= observed_sum)
    p_value = min(1.0, 2.0 * min(lower, upper) / total)
    rank_sum = observed_sum / 2.0
    u_value = rank_sum - n_left * (n_left + 1) / 2.0
    return u_value, p_value


def median(rows: list[dict[str, str]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def format_number(value: float) -> str:
    return f"{value:.8g}"


def execute_report(args: argparse.Namespace) -> None:
    rows = read_rows(args.rows)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["system"])].append(row)

    aa_lines: list[dict[str, float | str]] = []
    for system in sorted(AA_SYSTEMS):
        sample = grouped[("AA", system)]
        left = [row for row in sample if row["arm"] == "A1"]
        right = [row for row in sample if row["arm"] == "A2"]
        if len(left) != 12 or len(right) != 12:
            raise RuntimeError(f"{system}: incomplete A/A band")
        med_left = median(left, "outside_s")
        med_right = median(right, "outside_s")
        delta = (med_right - med_left) / med_left
        _u, p_value = exact_mann_whitney(
            [float(row["outside_s"]) for row in right],
            [float(row["outside_s"]) for row in left],
        )
        aa_lines.append(
            {
                "system": system,
                "median_a1": med_left,
                "median_a2": med_right,
                "delta": delta,
                "p": p_value,
            }
        )
    noise_band = max(abs(float(line["delta"])) for line in aa_lines)

    summary: list[dict[str, str]] = []
    for system in SYSTEMS:
        sample = grouped[("AB", system.name)]
        a_rows = sorted((row for row in sample if row["arm"] == "A"), key=lambda row: int(row["seed"]))
        b_rows = sorted((row for row in sample if row["arm"] == "B"), key=lambda row: int(row["seed"]))
        if len(a_rows) != 12 or len(b_rows) != 12:
            raise RuntimeError(f"{system.name}: incomplete A/B panel")
        if [row["seed"] for row in a_rows] != [row["seed"] for row in b_rows]:
            raise RuntimeError(f"{system.name}: A/B seeds differ")

        hits_a = sum(int(row["hit"]) for row in a_rows)
        hits_b = sum(int(row["hit"]) for row in b_rows)
        calls_a = [float(row["calls"]) for row in a_rows]
        calls_b = [float(row["calls"]) for row in b_rows]
        outside_a = [float(row["outside_s"]) for row in a_rows]
        outside_b = [float(row["outside_s"]) for row in b_rows]
        calls_u, calls_p = exact_mann_whitney(calls_b, calls_a)
        outside_u, outside_p = exact_mann_whitney(outside_b, outside_a)
        median_calls_a = statistics.median(calls_a)
        median_calls_b = statistics.median(calls_b)
        median_outside_a = statistics.median(outside_a)
        median_outside_b = statistics.median(outside_b)
        outside_delta = (median_outside_b - median_outside_a) / median_outside_a

        losses: list[str] = []
        wins: list[str] = []
        if hits_b < hits_a:
            losses.append("fewer hits")
        elif hits_b > hits_a:
            wins.append("more hits")
        if median_calls_b > median_calls_a and calls_p < ALPHA:
            losses.append("calls worse at p<0.05")
        elif median_calls_b < median_calls_a and calls_p < ALPHA:
            wins.append("calls better at p<0.05")
        if outside_delta > noise_band:
            losses.append("outside time exceeds A/A band")
        elif outside_delta < -noise_band:
            wins.append("outside time improves beyond A/A band")
        if losses:
            verdict = "LOSES"
            reason = "; ".join(losses)
        elif wins:
            verdict = "WINS"
            reason = "; ".join(wins)
        else:
            verdict = "TIE"
            reason = "no change beyond a gate threshold"

        summary.append(
            {
                "system": system.name,
                "n_a": str(len(a_rows)),
                "n_b": str(len(b_rows)),
                "hits_a": str(hits_a),
                "hits_b": str(hits_b),
                "fisher_p_exact": format_number(fisher_two_sided(hits_a, 12, hits_b, 12)),
                "median_calls_a": format_number(median_calls_a),
                "median_calls_b": format_number(median_calls_b),
                "mann_whitney_u_calls_b": format_number(calls_u),
                "mann_whitney_p_calls_exact": format_number(calls_p),
                "median_wall_s_a": format_number(median(a_rows, "wall_s")),
                "median_wall_s_b": format_number(median(b_rows, "wall_s")),
                "median_pot_s_a": format_number(median(a_rows, "pot_s")),
                "median_pot_s_b": format_number(median(b_rows, "pot_s")),
                "median_outside_s_a": format_number(median_outside_a),
                "median_outside_s_b": format_number(median_outside_b),
                "mann_whitney_u_outside_b": format_number(outside_u),
                "mann_whitney_p_outside_exact": format_number(outside_p),
                "outside_delta_rel": format_number(outside_delta),
                "aa_noise_band_rel": format_number(noise_band),
                "verdict": verdict,
                "reason": reason,
            }
        )
    write_csv(args.summary, SUMMARY_FIELDS, summary)

    losing = [row["system"] for row in summary if row["verdict"] == "LOSES"]
    overall = "DEFAULT-ON REJECTED" if losing else "DEFAULT-ON OK"
    lines = [
        "# SOAP general3 default-on gate",
        "",
        f"Overall verdict: **{overall}**.",
        "",
        "B is acceptable as the default only when no system loses. A system loses if B has fewer hits, "
        "if its median budget-censored calls are higher with exact two-sided Mann–Whitney p < 0.05, "
        "or if its median outside-potential time is higher than A by more than the A/A relative noise band.",
        "",
        "Calls and outside-potential p-values use the exact permutation distribution of the pooled "
        "midranks, including ties. Fisher p-values are exact and two-sided. Wall, potential, and "
        "outside-potential entries are medians in seconds.",
        "",
        "## A/B results",
        "",
        "| System | Hits A/B | Fisher p | Calls A/B | Calls p | Wall A/B | Pot A/B | Outside A/B | Outside p | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['system']} | {row['hits_a']}/12 / {row['hits_b']}/12 | "
            f"{row['fisher_p_exact']} | {row['median_calls_a']} / {row['median_calls_b']} | "
            f"{row['mann_whitney_p_calls_exact']} | {row['median_wall_s_a']} / {row['median_wall_s_b']} | "
            f"{row['median_pot_s_a']} / {row['median_pot_s_b']} | "
            f"{row['median_outside_s_a']} / {row['median_outside_s_b']} | "
            f"{row['mann_whitney_p_outside_exact']} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## A/A outside-potential band",
            "",
            "| System | Median A1 | Median A2 | Relative delta | Exact p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for line in aa_lines:
        lines.append(
            f"| {line['system']} | {format_number(float(line['median_a1']))} | "
            f"{format_number(float(line['median_a2']))} | {format_number(float(line['delta']))} | "
            f"{format_number(float(line['p']))} |"
        )
    lines.extend(
        [
            "",
            f"The A/A band is the largest absolute relative median delta: {format_number(noise_band)}.",
            "",
            "## Provenance",
            "",
            f"- A: commit `{A_COMMIT}`, tree `{A_TREE}`.",
            f"- B: commit `{next(row['commit'] for row in rows if row['arm'] == 'B')}`, tree `{B_TREE}`.",
            "- Seeds 2000–2011; full paper budgets; serial fresh processes pinned to core 0.",
            "- References: `scripts/gmin_panel/run_panel.py` for LJ and M55/M80; "
            "`scripts/ase_nonlj/run_one.py` at commit "
            "`cf6833e1316cf8c7a1bb7879a477ef2b32e9d8f5` for M38 ρ=10; "
            "`scripts/tip4p_h2h/run_h2h.py` for TIP4P, including its pooled-session rule for TIP4P25.",
        ]
    )
    if losing:
        lines.append(f"- Losing systems: {', '.join(losing)}.")
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.gate.write_text("\n".join(lines) + "\n", newline="\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--a-source", type=Path, required=True)
    run_parser.add_argument("--b-source", type=Path, required=True)
    run_parser.add_argument("--aa-source", type=Path, required=True)
    run_parser.add_argument("--cargo", type=Path, required=True)
    run_parser.add_argument("--core", type=int, required=True)
    run_parser.add_argument("--work", type=Path, required=True)
    run_parser.add_argument("--rows", type=Path, required=True)
    run_parser.add_argument("--sessions", type=Path, required=True)
    run_parser.add_argument("--logs", type=Path, required=True)
    run_parser.set_defaults(function=execute_run)

    report_parser = commands.add_parser("report")
    report_parser.add_argument("--rows", type=Path, required=True)
    report_parser.add_argument("--summary", type=Path, required=True)
    report_parser.add_argument("--gate", type=Path, required=True)
    report_parser.set_defaults(function=execute_report)
    return root


def main(argv: list[str]) -> int:
    args = parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
