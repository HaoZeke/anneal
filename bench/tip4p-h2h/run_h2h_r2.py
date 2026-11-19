#!/usr/bin/env python3
# Copyright (c) anneal developers
"""TIP4P vs GMIN rerun with per-run provenance and ABBA arm order."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


TIP4P_REFERENCES_KJ_MOL = {
    8: -305.5183221676,
    15: -628.3722525077,
    20: -872.9887544952,
    21: -916.7074302940,
}


def parse_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lo = int(h)
    frac = h - lo
    i = lo - 1
    if frac == 0.0:
        return ordered[i]
    return ordered[i] * (1.0 - frac) + ordered[i + 1] * frac


def read_cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def read_mem_total_kb() -> str:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return line.split()[1]
    return "0"


def read_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def machine_id() -> str:
    payload = f"{read_cpu_model()}|{read_mem_total_kb()}|{read_boot_id()}".encode()
    return hashlib.sha256(payload).hexdigest()


def parse_kv(line: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in line.split()[1:])


def parse_anneal_row(log_text: str) -> dict[str, str]:
    for line in log_text.splitlines():
        if line.startswith("h2h_row "):
            return parse_kv(line)
    raise RuntimeError("missing h2h_row in anneal log")


def parse_affinity(log_text: str) -> str:
    for line in log_text.splitlines():
        if line.startswith("affinity_cores="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("missing affinity_cores in log")


def env_for_budget(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMBA_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    return env


def run_with_affinity(
    core: int,
    command: list[str],
    env: dict[str, str],
    cwd: Path | None,
    log_path: Path,
) -> tuple[int, str]:
    wrapper = [
        "python3",
        "-c",
        (
            "import os,sys; "
            "print('affinity_cores=' + ','.join(map(str, sorted(os.sched_getaffinity(0))))); "
            "sys.stdout.flush(); "
            "os.execv(sys.argv[1], sys.argv[1:])"
        ),
        *command,
    ]
    with log_path.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", str(core), *wrapper],
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=False,
        )
    return proc.returncode, log_path.read_text()


@dataclass
class ArmRun:
    molecules: int
    seed: int
    arm: str
    start_id: str
    hit: int
    calls: int
    wall_s: float
    best_energy: float
    core: int
    affinity_source: str
    affinity_seen: str
    run_order_index: int
    arm_order: str
    start_ts: str
    end_ts: str


def mann_whitney_u_two_sided_exact(x: list[float], y: list[float]) -> tuple[float, str]:
    """Exact two-sided Mann-Whitney p via rank-sum enumeration."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), "exact-rank-enumeration"
    pooled = x + y
    order = sorted(enumerate(pooled), key=lambda item: item[1])
    ranks = [0.0] * len(pooled)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k][0]] = rank
        i = j + 1
    ranks2 = [int(round(rank * 2.0)) for rank in ranks]
    obs_sum2 = sum(ranks2[:n1])
    obs_u2 = obs_sum2 - n1 * (n1 + 1)
    total_n = n1 + n2

    dp: list[dict[int, int]] = [{0: 1}] + [{} for _ in range(n1)]
    seen = 0
    for rank2 in ranks2:
        seen += 1
        top = min(seen, n1)
        for picked in range(top, 0, -1):
            current = dp[picked]
            previous = dp[picked - 1]
            for score, ways in previous.items():
                key = score + rank2
                current[key] = current.get(key, 0) + ways
    distribution = dp[n1]
    total = math.comb(total_n, n1)

    low = sum(
        ways for rank_sum2, ways in distribution.items() if rank_sum2 - n1 * (n1 + 1) <= obs_u2
    ) / total
    high = sum(
        ways for rank_sum2, ways in distribution.items() if rank_sum2 - n1 * (n1 + 1) >= obs_u2
    ) / total
    p_two = min(1.0, 2.0 * min(low, high))
    return p_two, "exact-rank-enumeration"


def compare_calls_hits(
    observed: list[tuple[int, int, str, int, int]], reference_rows: Path
) -> list[str]:
    """Return one line per (molecules, seed, arm, hit, calls) that departs from the reference."""
    baseline: dict[tuple[int, int, str], tuple[int, int]] = {}
    with reference_rows.open() as handle:
        for row in csv.DictReader(handle):
            baseline[(int(row["molecules"]), int(row["seed"]), row["arm"])] = (
                int(row["hit"]),
                int(row["calls"]),
            )
    mismatches: list[str] = []
    for molecules, seed, arm, hit, calls in observed:
        key = (molecules, seed, arm)
        if key not in baseline:
            mismatches.append(f"{molecules},{seed},{arm},missing-baseline")
            continue
        hit_ref, calls_ref = baseline[key]
        if hit_ref != hit or calls_ref != calls:
            mismatches.append(
                f"{molecules},{seed},{arm},hit:{hit}!={hit_ref},calls:{calls}!={calls_ref}"
            )
    return mismatches


def refuse_existing(*paths: Path) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise SystemExit("refusing to overwrite existing records: " + ", ".join(present))


def write_once(path: Path, text: str) -> None:
    """Write a record that is never rewritten: an existing file must already hold ``text``."""
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"refusing to rewrite {path}: its contents differ")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        handle.write(text)


def csv_text(fieldnames: list[str], rows: list[dict[str, object]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def report_command(args: argparse.Namespace) -> int:
    with args.rows.open() as handle:
        rows_data = list(csv.DictReader(handle))
    if not rows_data:
        raise RuntimeError("rows file is empty")
    sizes = sorted({int(row["molecules"]) for row in rows_data})

    machine_ids = sorted({row["machine_id"] for row in rows_data})
    cores = sorted({row["core"] for row in rows_data})
    rows_total = len(rows_data)

    mismatches = compare_calls_hits(
        [
            (int(row["molecules"]), int(row["seed"]), row["arm"], int(row["hit"]), int(row["calls"]))
            for row in rows_data
        ],
        args.reference_rows,
    )

    summary_rows: list[dict[str, str]] = []
    for n in sizes:
        a = [row for row in rows_data if int(row["molecules"]) == n and row["arm"] == "anneal"]
        g = [row for row in rows_data if int(row["molecules"]) == n and row["arm"] == "gmin"]
        aw = [float(row["wall_s"]) for row in a]
        gw = [float(row["wall_s"]) for row in g]
        ac = [int(row["calls"]) for row in a]
        gc = [int(row["calls"]) for row in g]
        ae = [float(row["best_energy"]) for row in a]
        ge = [float(row["best_energy"]) for row in g]
        scored = n in TIP4P_REFERENCES_KJ_MOL
        ah = f"{sum(int(row['hit']) for row in a)}/{len(a)}" if scored else "unscored"
        gh = f"{sum(int(row['hit']) for row in g)}/{len(g)}" if scored else "unscored"
        p, method = mann_whitney_u_two_sided_exact(aw, gw)
        summary_rows.append(
            {
                "molecules": str(n),
                "anneal_n": str(len(a)),
                "gmin_n": str(len(g)),
                "mwu_method": method,
                "anneal_hits_n": ah,
                "gmin_hits_n": gh,
                "anneal_calls_median": f"{type7([float(x) for x in ac]):.1f}",
                "gmin_calls_median": f"{type7([float(x) for x in gc]):.1f}",
                "anneal_energy_best_kj_mol": f"{min(ae):.10f}",
                "anneal_energy_median_kj_mol": f"{type7(ae):.10f}",
                "gmin_energy_best_kj_mol": f"{min(ge):.10f}",
                "gmin_energy_median_kj_mol": f"{type7(ge):.10f}",
                "anneal_wall_median_s": f"{type7(aw):.6f}",
                "anneal_wall_iqr_s": f"{type7(aw, 0.75) - type7(aw, 0.25):.6f}",
                "anneal_wall_max_s": f"{max(aw):.6f}",
                "gmin_wall_median_s": f"{type7(gw):.6f}",
                "gmin_wall_iqr_s": f"{type7(gw, 0.75) - type7(gw, 0.25):.6f}",
                "gmin_wall_max_s": f"{max(gw):.6f}",
                "wall_mwu_p": f"{p:.6g}",
            }
        )

    wall_text = csv_text(list(summary_rows[0].keys()), summary_rows)

    check_lines = [
        "# TIP4P ABBA wall-time check",
        "",
        f"rows compared: {rows_total}",
        f"mismatches: {len(mismatches)}",
        "",
        "Wall times are within-run only and cannot be compared outside this run.",
        "",
    ]

    if len(machine_ids) == 1 and cores == ["0"]:
        check_lines.append("All runs used one machine id and core 0.")
    else:
        check_lines.append(
            f"Runs used machine ids: {', '.join(machine_ids)}; cores: {', '.join(cores)}."
        )
    check_lines.append("")
    check_lines.append("## Mismatches")
    check_lines.append("")
    if mismatches:
        for line in mismatches:
            check_lines.append(f"- {line}")
    else:
        check_lines.append("- none")
    check_lines.append("")
    check_lines.append("## Wall statistics")
    check_lines.append("")
    check_lines.append(
        "| molecules | n anneal | n gmin | MW method | hits/n anneal | hits/n gmin | calls median anneal | calls median gmin | energy best anneal | energy median anneal | energy best gmin | energy median gmin | wall median anneal | wall iqr anneal | wall max anneal | wall median gmin | wall iqr gmin | wall max gmin | MW p |"
    )
    check_lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for row in summary_rows:
        check_lines.append(
            f"| {row['molecules']} | {row['anneal_n']} | {row['gmin_n']} | {row['mwu_method']} | "
            f"{row['anneal_hits_n']} | {row['gmin_hits_n']} | {row['anneal_calls_median']} | {row['gmin_calls_median']} | "
            f"{row['anneal_energy_best_kj_mol']} | {row['anneal_energy_median_kj_mol']} | "
            f"{row['gmin_energy_best_kj_mol']} | {row['gmin_energy_median_kj_mol']} | "
            f"{row['anneal_wall_median_s']} | {row['anneal_wall_iqr_s']} | {row['anneal_wall_max_s']} | "
            f"{row['gmin_wall_median_s']} | {row['gmin_wall_iqr_s']} | {row['gmin_wall_max_s']} | {row['wall_mwu_p']} |"
        )
    check_lines.append("")

    n8 = next((row for row in summary_rows if row["molecules"] == "8"), None)
    if n8 is not None:
        gmin_iqr = float(n8["gmin_wall_iqr_s"])
        gmin_median = float(n8["gmin_wall_median_s"])
        if gmin_iqr > gmin_median:
            check_lines.append(
                "At n=8, gmin wall iqr is larger than its median; this spread is high."
            )
            check_lines.append(
                f"At n=8, wall max is anneal {n8['anneal_wall_max_s']} and gmin {n8['gmin_wall_max_s']}."
            )
            check_lines.append("")

    write_once(args.wall_out, wall_text)
    write_once(args.check_out, "\n".join(check_lines).rstrip("\n") + "\n")

    print(f"mismatch_count={len(mismatches)}")
    return 1 if mismatches else 0


def run_command(args: argparse.Namespace) -> int:
    sizes = parse_list(args.sizes)
    seeds = parse_list(args.seeds)
    affinity_source = f"taskset -c {args.core}"
    machine = machine_id()
    hostname = socket.gethostname()
    cpu_model = read_cpu_model()
    arch = platform.machine()
    logical_cpus = os.cpu_count() or 0
    references = TIP4P_REFERENCES_KJ_MOL

    refuse_existing(args.rows_out, args.sessions_out, args.hosts_out)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.work_dir / "runs-r2"
    run_dir.mkdir(parents=True, exist_ok=True)

    expected_start_ids: dict[tuple[int, int], str] = {}
    for row in csv.DictReader(args.reference_rows.open()):
        key = (int(row["molecules"]), int(row["seed"]))
        expected_start_ids[key] = row["start_id"]

    rows: list[ArmRun] = []
    order = 0

    def start_hash_from(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    for n in sizes:
        for seed in seeds:
            arm_order = "anneal,gmin" if seed % 2 == 0 else "gmin,anneal"
            ref_start = args.start_dir / f"tip4p{n}-s{seed}.start.gmin"
            if not ref_start.is_file():
                raise RuntimeError(f"missing reference start for n={n} seed={seed}")
            start_id = start_hash_from(ref_start)
            want = expected_start_ids.get((n, seed))
            if want is None:
                raise RuntimeError(f"missing reference row for n={n} seed={seed}")
            if start_id != want:
                raise RuntimeError(
                    f"start_id mismatch for n={n} seed={seed}: {start_id} != {want}"
                )

            arms = ["anneal", "gmin"] if seed % 2 == 0 else ["gmin", "anneal"]
            for arm in arms:
                order += 1
                started = datetime.now().astimezone().isoformat()
                if arm == "anneal":
                    prefix = run_dir / f"tip4p{n}-s{seed}-r2"
                    log = run_dir / f"tip4p{n}-s{seed}-anneal-r2.log"
                    stop = references.get(n)
                    stop_text = "-" if stop is None else repr(stop + args.hit_tol)
                    rc, text = run_with_affinity(
                        args.core,
                        [
                            str(args.binary),
                            "tip4p",
                            str(n),
                            "0",
                            str(args.budget),
                            str(seed),
                            "recommended",
                            stop_text,
                            str(prefix),
                        ],
                        env_for_budget(args.budget),
                        None,
                        log,
                    )
                    if rc != 0:
                        raise RuntimeError(f"anneal failed n={n} seed={seed}")
                    row = parse_anneal_row(text)
                    seen = parse_affinity(text)
                    start_file = prefix.with_suffix(".start.gmin")
                    if not start_file.is_file():
                        raise RuntimeError(f"anneal start file missing for n={n} seed={seed}")
                    got_start = start_hash_from(start_file)
                    if got_start != start_id:
                        raise RuntimeError(
                            f"anneal start_id mismatch for n={n} seed={seed}: {got_start} != {start_id}"
                        )
                    ended = datetime.now().astimezone().isoformat()
                    rows.append(
                        ArmRun(
                            molecules=n,
                            seed=seed,
                            arm="anneal",
                            start_id=start_id,
                            hit=int(row["hit"]),
                            calls=int(row["evals_to_hit"]),
                            wall_s=float(row["wall_s"]),
                            best_energy=float(row["best"]),
                            core=args.core,
                            affinity_source=affinity_source,
                            affinity_seen=seen,
                            run_order_index=order,
                            arm_order=arm_order,
                            start_ts=started,
                            end_ts=ended,
                        )
                    )
                else:
                    gmin_run = run_dir / f"gmin-tip4p{n}-s{seed}-r2"
                    gmin_run.mkdir(parents=True, exist_ok=True)
                    coords = gmin_run / "coords"
                    coords.write_bytes(ref_start.read_bytes())
                    data = gmin_run / "data"
                    data.write_text(args.gmin_data.read_text() + f"RANSEED {seed}\n")
                    row_txt = gmin_run / "row.txt"
                    log = run_dir / f"tip4p{n}-s{seed}-gmin-r2.log"
                    env = env_for_budget(args.budget)
                    env.update(
                        {
                            "H2H_SYSTEM": f"tip4p{n}",
                            "H2H_SEED": str(seed),
                            "H2H_ROW": str(row_txt),
                        }
                    )
                    if n in references:
                        env["H2H_TARGET"] = f"{references[n]:.10f}"
                        env["H2H_TOL"] = f"{args.hit_tol}"
                    rc, text = run_with_affinity(
                        args.core,
                        [str(args.gmin)],
                        env,
                        gmin_run,
                        log,
                    )
                    if rc != 0 or not row_txt.is_file():
                        raise RuntimeError(f"gmin failed n={n} seed={seed}")
                    seen = parse_affinity(text)
                    row = parse_kv(row_txt.read_text().strip())
                    ended = datetime.now().astimezone().isoformat()
                    rows.append(
                        ArmRun(
                            molecules=n,
                            seed=seed,
                            arm="gmin",
                            start_id=start_id,
                            hit=int(row["hit"]),
                            calls=int(row["evals_to_hit"]),
                            wall_s=float(row["wall_s"]),
                            best_energy=float(row["best"]),
                            core=args.core,
                            affinity_source=affinity_source,
                            affinity_seen=seen,
                            run_order_index=order,
                            arm_order=arm_order,
                            start_ts=started,
                            end_ts=ended,
                        )
                    )

    rows.sort(key=lambda r: (r.molecules, r.seed, 0 if r.arm == "anneal" else 1))

    row_fields = [
        "system",
        "molecules",
        "arm",
        "seed",
        "start_id",
        "best_energy",
        "hit",
        "calls",
        "wall_s",
        "anneal_sha",
        "gmin_revision",
        "run_order_index",
        "arm_order",
        "machine_id",
        "core",
        "affinity_source",
        "affinity_seen",
        "start_ts",
        "end_ts",
        "cpu_model",
        "arch",
        "logical_cpus",
        "hostname",
    ]
    rows_text = csv_text(
        row_fields,
        [
            {
                "system": "tip4p",
                "molecules": r.molecules,
                "arm": r.arm,
                "seed": r.seed,
                "start_id": r.start_id,
                "best_energy": f"{r.best_energy:.10f}",
                "hit": r.hit,
                "calls": r.calls,
                "wall_s": f"{r.wall_s:.6f}",
                "anneal_sha": args.anneal_sha,
                "gmin_revision": args.gmin_revision,
                "run_order_index": r.run_order_index,
                "arm_order": r.arm_order,
                "machine_id": machine,
                "core": r.core,
                "affinity_source": r.affinity_source,
                "affinity_seen": r.affinity_seen,
                "start_ts": r.start_ts,
                "end_ts": r.end_ts,
                "cpu_model": cpu_model,
                "arch": arch,
                "logical_cpus": logical_cpus,
                "hostname": hostname,
            }
            for r in rows
        ],
    )

    session_lines = []
    for n in sizes:
        for seed in seeds:
            pair = [r for r in rows if r.molecules == n and r.seed == seed]
            pair.sort(key=lambda r: 0 if r.arm == "anneal" else 1)
            payload = {
                "system": "tip4p",
                "molecules": n,
                "seed": seed,
                "start_id": pair[0].start_id,
                "arm_order": pair[0].arm_order.split(","),
                "machine_id": machine,
                "hostname": hostname,
                "cpu_model": cpu_model,
                "arch": arch,
                "logical_cpus": logical_cpus,
                "affinity_source": affinity_source,
                "hit_tol": args.hit_tol,
                "hit_tol_unit": "kJ/mol",
                "arms": [
                    {
                        "arm": r.arm,
                        "hit": r.hit,
                        "calls": r.calls,
                        "wall_s": r.wall_s,
                        "best_energy": r.best_energy,
                        "core": r.core,
                        "affinity_seen": r.affinity_seen,
                        "run_order_index": r.run_order_index,
                        "start_ts": r.start_ts,
                        "end_ts": r.end_ts,
                    }
                    for r in pair
                ],
                "anneal_sha": args.anneal_sha,
                "gmin_revision": args.gmin_revision,
            }
            session_lines.append(json.dumps(payload, separators=(",", ":")) + "\n")

    hosts_text = csv_text(
        ["molecules", "seed", "arm", "machine_id", "hostname", "core", "affinity_source"],
        [
            {
                "molecules": r.molecules,
                "seed": r.seed,
                "arm": r.arm,
                "machine_id": machine,
                "hostname": hostname,
                "core": r.core,
                "affinity_source": r.affinity_source,
            }
            for r in rows
        ],
    )

    write_once(args.rows_out, rows_text)
    write_once(args.sessions_out, "".join(session_lines))
    write_once(args.hosts_out, hosts_text)

    mismatches = compare_calls_hits(
        [(r.molecules, r.seed, r.arm, r.hit, r.calls) for r in rows], args.reference_rows
    )
    print(f"mismatch_count={len(mismatches)}")
    print(f"rows_compared={len(rows)}")
    return 1 if mismatches else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--gmin", type=Path, required=True)
    run.add_argument("--gmin-data", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--start-dir", type=Path, required=True)
    run.add_argument("--reference-rows", type=Path, required=True)
    run.add_argument("--rows-out", type=Path, required=True)
    run.add_argument("--sessions-out", type=Path, required=True)
    run.add_argument("--hosts-out", type=Path, required=True)
    run.add_argument("--anneal-sha", required=True)
    run.add_argument("--gmin-revision", required=True)
    run.add_argument("--sizes", required=True)
    run.add_argument("--seeds", required=True)
    run.add_argument("--budget", type=int, required=True)
    run.add_argument("--core", type=int, required=True)
    run.add_argument("--hit-tol", type=float, required=True)
    run.set_defaults(handler=run_command)

    report = sub.add_parser("report")
    report.add_argument("--rows", type=Path, required=True)
    report.add_argument("--reference-rows", type=Path, required=True)
    report.add_argument("--check-out", type=Path, required=True)
    report.add_argument("--wall-out", type=Path, required=True)
    report.set_defaults(handler=report_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
