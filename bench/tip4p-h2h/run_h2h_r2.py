#!/usr/bin/env python3
# Copyright (c) anneal developers
"""TIP4P vs GMIN rerun with per-run provenance and ABBA arm order."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
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


def mann_whitney_u_two_sided(x: list[float], y: list[float]) -> float:
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = sorted(x + y)
    ranks: dict[float, float] = {}
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        rank = (i + 1 + j + 1) / 2.0
        ranks[pooled[i]] = rank
        i = j + 1
    r1 = sum(ranks[v] for v in x)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie = 0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        t = j - i + 1
        tie += t**3 - t
        i = j + 1
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if sigma2 <= 0.0:
        return 1.0
    z = (u - mu) / math.sqrt(sigma2)
    return max(0.0, min(1.0, 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))))


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--gmin-data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--start-dir", type=Path, required=True)
    parser.add_argument("--reference-rows", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--sessions-out", type=Path, required=True)
    parser.add_argument("--hosts-out", type=Path, required=True)
    parser.add_argument("--check-out", type=Path, required=True)
    parser.add_argument("--verify-out", type=Path, required=True)
    parser.add_argument("--anneal-sha", required=True)
    parser.add_argument("--gmin-revision", required=True)
    parser.add_argument("--sizes", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--core", type=int, required=True)
    parser.add_argument("--hit-tol", type=float, required=True)
    args = parser.parse_args()

    sizes = parse_list(args.sizes)
    seeds = parse_list(args.seeds)
    affinity_source = f"taskset -c {args.core}"
    machine = machine_id()
    cpu_model = read_cpu_model()
    arch = platform.machine()
    logical_cpus = os.cpu_count() or 0
    references = TIP4P_REFERENCES_KJ_MOL

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

    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
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
                }
            )

    with args.sessions_out.open("w") as handle:
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
                    "cpu_model": cpu_model,
                    "arch": arch,
                    "logical_cpus": logical_cpus,
                    "affinity_source": affinity_source,
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
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    with args.hosts_out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["molecules", "seed", "arm", "machine_id", "core", "affinity_source"],
            lineterminator="\n",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "molecules": r.molecules,
                    "seed": r.seed,
                    "arm": r.arm,
                    "machine_id": machine,
                    "core": r.core,
                    "affinity_source": affinity_source,
                }
            )

    baseline: dict[tuple[int, int, str], tuple[int, int]] = {}
    for row in csv.DictReader(args.reference_rows.open()):
        baseline[(int(row["molecules"]), int(row["seed"]), row["arm"])] = (
            int(row["hit"]),
            int(row["calls"]),
        )
    mismatches: list[str] = []
    for r in rows:
        key = (r.molecules, r.seed, r.arm)
        if key not in baseline:
            mismatches.append(f"{r.molecules},{r.seed},{r.arm},missing-baseline")
            continue
        bhit, bcalls = baseline[key]
        if bhit != r.hit or bcalls != r.calls:
            mismatches.append(
                f"{r.molecules},{r.seed},{r.arm},hit:{r.hit}!= {bhit},calls:{r.calls}!= {bcalls}"
            )

    verify_rows = []
    for n in sizes:
        a = [r for r in rows if r.molecules == n and r.arm == "anneal"]
        g = [r for r in rows if r.molecules == n and r.arm == "gmin"]
        aw = [r.wall_s for r in a]
        gw = [r.wall_s for r in g]
        verify_rows.append(
            {
                "molecules": n,
                "anneal_wall_median_s": f"{type7(aw):.6f}",
                "anneal_wall_iqr_s": f"{type7(aw, 0.75) - type7(aw, 0.25):.6f}",
                "gmin_wall_median_s": f"{type7(gw):.6f}",
                "gmin_wall_iqr_s": f"{type7(gw, 0.75) - type7(gw, 0.25):.6f}",
                "wall_mwu_p": f"{mann_whitney_u_two_sided(aw, gw):.6g}",
            }
        )
    with args.verify_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verify_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(verify_rows)

    check_lines = [
        "# R2 check",
        "",
        f"rows compared: {len(rows)}",
        f"mismatches: {len(mismatches)}",
        "",
    ]
    if mismatches:
        check_lines.append("## Mismatches")
        check_lines.append("")
        for line in mismatches:
            check_lines.append(f"- {line}")
        check_lines.append("")
    else:
        check_lines.append("## Mismatches")
        check_lines.append("")
        check_lines.append("- none")
        check_lines.append("")
    check_lines.append("## R2 wall stats")
    check_lines.append("")
    check_lines.append("| molecules | anneal median | anneal iqr | gmin median | gmin iqr | MW p |")
    check_lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in verify_rows:
        check_lines.append(
            f"| {row['molecules']} | {row['anneal_wall_median_s']} | {row['anneal_wall_iqr_s']} | "
            f"{row['gmin_wall_median_s']} | {row['gmin_wall_iqr_s']} | {row['wall_mwu_p']} |"
        )
    args.check_out.write_text("\n".join(check_lines) + "\n")

    if mismatches:
        print(f"mismatch_count={len(mismatches)}")
    else:
        print("mismatch_count=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
