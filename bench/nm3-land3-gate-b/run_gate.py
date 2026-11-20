#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial two-arm no-loss gate for nm3-land3, part B.

Sutton-Chen sc126-55 (seeds 2000-2023) and sc96-38 (seeds 2000-2011),
the seven Cambridge binary Lennard-Jones clusters (seeds 2000-2011),
one labelled blj105n38 seed 2101 reproduction, and rigid TIP4P (H2O)n
for every n from 8 to 21 that has a row in data/tip4p_ccd.csv (seeds
2000-2011). Paper budgets, the default search, no H2H_* variables.

One process tree on core 0, one thread in every pool, arms in ABBA
order, release builds, one session. Rows are appended and never rewritten.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path

GATE_SEEDS = tuple(range(2000, 2012))
SC126_SEEDS = tuple(range(2000, 2024))
REPRO_SEED = 2101
JOB_TIMEOUT_S = 43200.0
BASE_COMMIT = "61fa67cf380e46876b569da45e7cc091647a775b"
BASE_TREE = "cff796c507a8fa1d0bfb28888c3570b61988055e"
TIP4P_MIN_N = 8
TIP4P_MAX_N = 21

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
)

BLJ_SYSTEMS = (
    "blj105n38",
    "blj115n38",
    "blj130n38",
    "blj105n55",
    "blj115n55",
    "blj130n55",
    "blj130n98",
)

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "slot",
    "core",
    "cohort",
    "hit",
    "calls",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "outside_per_call_s",
    "best",
    "best_bits",
    "budget",
    "reference",
    "machine_id",
    "base_commit",
    "base_tree",
    "library",
    "library_tree",
    "harness_blobs",
    "driver",
    "status",
    "start_sha256",
    "trace_sha256",
    "xyz_sha256",
    "gmin_sha256",
    "session_id",
    "utc",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return sha256_bytes(path.read_bytes())


def float_bits(value: float) -> str:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"{raw:016x}"


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                return sha256_bytes(raw.encode("utf-8"))
    raise RuntimeError("machine id is not readable")


def git_blob(repo: Path, relpath: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "hash-object", str(repo / relpath)],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def arm_blob(tree: Path, relpath: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", f"HEAD:{relpath}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def read_tip4p_references(repo: Path) -> dict[int, float]:
    found: dict[int, float] = {}
    with (repo / "data" / "tip4p_ccd.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            size = int(row["n"])
            if TIP4P_MIN_N <= size <= TIP4P_MAX_N:
                found[size] = float(row["energy"])
    return found


def tip4p_budget_and_tol(repo: Path) -> tuple[int, float]:
    scripts = repo / "scripts"
    tip4p = scripts / "tip4p_h2h"
    for entry in (str(scripts), str(tip4p)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import run_h2h  # noqa: PLC0415
    from target_tol import DEFAULT_HIT_TOL  # noqa: PLC0415

    return int(run_h2h.BUDGET), float(DEFAULT_HIT_TOL)


def sc_budgets(repo: Path) -> dict[str, int]:
    scripts = repo / "scripts"
    metal = scripts / "metal_h2h"
    for entry in (str(scripts), str(metal)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import run_sc  # noqa: PLC0415

    return {
        "sc126-55": int(run_sc.PAPER_BUDGETS["sc126-55"]),
        "sc96-38": int(run_sc.PAPER_BUDGETS["sc96-38"]),
    }


def verify_affinity(cpu: int) -> None:
    aff = sorted(os.sched_getaffinity(0))
    if aff != [cpu]:
        raise SystemExit(f"affinity is {aff}, expected [{cpu}]")


def child_env(work: Path, rgpot: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(work / "home"),
        "TMPDIR": str(work / "tmp"),
        "LANG": "C",
        "LC_ALL": "C",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "PYTHONHASHSEED": "0",
        "LJOS_SEAT": os.environ.get("LJOS_SEAT", "benchbot"),
    }
    for key in THREAD_ENV:
        env[key] = "1"
    if rgpot:
        lib = Path(rgpot).resolve()
        env["RGPOT_SC_LIB"] = str(lib)
        env["LD_LIBRARY_PATH"] = str(lib.parent)
    forbidden = [key for key in env if key.startswith(("H2H_", "ANNEAL_"))]
    if forbidden:
        raise SystemExit(f"refusing environment {forbidden}")
    return env


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        if header != fields:
            raise SystemExit(f"{path.name} header does not match the runner")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
        handle.flush()
        os.fsync(handle.fileno())


def query_budget(binary: Path, system: str, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        [str(binary), "budget", system],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return int(lines[0]), lines[1]


def parse_result(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    if "h2h_row " in text:
        for line in text.splitlines():
            if line.startswith("h2h_row "):
                parsed = dict(part.split("=", 1) for part in line.split()[1:])
                if parsed.get("system") != system:
                    raise ValueError(f"unexpected h2h_row system {parsed.get('system')}")
                if int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
                    raise ValueError(f"h2h_row does not match {system} {seed} {budget}")
                hit = parsed["hit"]
                parsed["hit"] = "1" if hit in {"1", "true", "True"} else "0"
                parsed["calls"] = str(int(float(parsed["evals_to_hit"])))
                if "best_bits" not in parsed:
                    parsed["best_bits"] = float_bits(float(parsed["best"]))
                return parsed
        raise ValueError("missing h2h_row")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("system,code,seed,"):
            continue
        header = line.split(",")
        for candidate in lines[index + 1 :]:
            values = candidate.split(",")
            if len(values) != len(header) or values[0] != system or values[2] != str(seed):
                continue
            row = dict(zip(header, values, strict=True))
            if int(float(row["budget"])) != budget:
                raise ValueError(f"budget {row['budget']} != {budget}")
            row["hit"] = "1" if row["hit"].strip().lower() in {"1", "true"} else "0"
            row["calls"] = str(int(float(row["evals_to_hit"])))
            if "best_bits" not in row:
                row["best_bits"] = float_bits(float(row["best"]))
            return row
    raise ValueError("missing result row")


def run_job(argv: list[str], cpu: str, env: dict[str, str], cwd: Path, log: Path, timeout_s: float) -> tuple[str, str]:
    pinned = ["taskset", "-c", cpu, *argv]
    proc = subprocess.Popen(
        pinned,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        out, _ = proc.communicate()
        status = "timeout"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"# status={status}\n" + (out or ""), encoding="utf-8")
    return status, out or ""


def arm_order(seed: int) -> tuple[str, str]:
    if (seed - 2000) % 2 == 0:
        return ("A", "B")
    return ("B", "A")


def ccd_xyz(repo: Path, system: str, dest: Path) -> None:
    text = (repo / "tests" / "fixtures" / "blj-ccd" / system).read_text(encoding="utf-8")
    rows: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in {"LA", "LB", "A", "B"}:
            rows.append(" ".join(parts[1:]))
    dest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def jobs() -> list[tuple[str, str, tuple[int, ...], str]]:
    """(system, driver, seeds, cohort) in run order."""
    planned: list[tuple[str, str, tuple[int, ...], str]] = [
        ("sc126-55", "h2h_sc", SC126_SEEDS, "gate"),
        ("sc96-38", "h2h_sc", GATE_SEEDS, "gate"),
    ]
    for system in BLJ_SYSTEMS:
        planned.append((system, "nm3_gate_driver", GATE_SEEDS, "gate"))
        if system == "blj105n38":
            planned.append((system, "nm3_gate_driver", (REPRO_SEED,), "reproduction"))
    for size in range(TIP4P_MIN_N, TIP4P_MAX_N + 1):
        planned.append((f"tip4p{size}", "h2h_seed", GATE_SEEDS, "gate"))
    return planned


def binary_of(arm_bin: Path, driver: str) -> Path:
    return arm_bin / driver


def command_for(
    driver: str,
    binary: Path,
    system: str,
    seed: int,
    budget: int,
    reference: float,
    tol: float,
    prefix: Path,
    start: Path | None,
) -> list[str]:
    if driver == "h2h_sc":
        return [str(binary), "run", system, str(seed)]
    if driver == "nm3_gate_driver":
        argv = [str(binary), "run", system, str(seed)]
        if start is not None:
            argv.append(str(start))
        return argv
    stop = repr(reference + tol)
    size = system.removeprefix("tip4p")
    return [
        str(binary),
        "tip4p",
        size,
        "0",
        str(budget),
        str(seed),
        "recommended",
        stop,
        str(prefix),
    ]


def preflight(repo: Path, arms: dict[str, Path], env: dict[str, str], scratch: Path) -> None:
    for system in ("sc126-55", "sc96-38"):
        hashes = []
        for name, bindir in arms.items():
            path = scratch / f"pre-{system}-{name}.xyz"
            subprocess.run(
                [str(bindir / "h2h_sc"), "coords", system, "2000", str(path)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            hashes.append(sha256_file(path))
            energy = subprocess.run(
                [str(bindir / "h2h_sc"), "energy", system, str(path)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.split()
            if not energy:
                raise SystemExit(f"{system} arm {name} returned no energy")
            value = float(energy[0])
            if value != value:
                raise SystemExit(f"{system} arm {name} energy is not finite")
        if hashes[0] != hashes[1]:
            raise SystemExit(f"{system} seed 2000 starts differ")
    for system in BLJ_SYSTEMS:
        xyz = scratch / f"ccd-{system}.xyz"
        ccd_xyz(repo, system, xyz)
        seen = []
        for _name, bindir in arms.items():
            proc = subprocess.run(
                [str(bindir / "nm3_gate_driver"), "energy", system, str(xyz)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            seen.append(float(proc.stdout.splitlines()[0]))
        _budget, reference = query_budget(arms["A"] / "nm3_gate_driver", system, env)
        for energy in seen:
            if abs(energy - float(reference)) > 1e-8:
                raise SystemExit(f"{system} single point {energy} is far from {reference}")


def main(argv: list[str]) -> int:
    if "--quick" in argv:
        raise SystemExit("quick runs are not part of this gate")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arm-a-bin", type=Path, required=True)
    parser.add_argument("--arm-b-bin", type=Path, required=True)
    parser.add_argument("--arm-a-tree", type=Path, required=True)
    parser.add_argument("--arm-b-tree", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--rgpot-lib", required=True)
    parser.add_argument("--cpu", default="0")
    parser.add_argument("--job-timeout-s", type=float, default=JOB_TIMEOUT_S)
    args = parser.parse_args(argv)

    verify_affinity(int(args.cpu))
    repo = args.repo.resolve()
    work = args.work.resolve()
    logs = args.logs.resolve()
    for sub in ("home", "tmp", "scratch", "starts", "tip4p"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = child_env(work, args.rgpot_lib)
    arms = {"A": args.arm_a_bin.resolve(), "B": args.arm_b_bin.resolve()}
    trees = {"A": args.arm_a_tree.resolve(), "B": args.arm_b_tree.resolve()}

    tip4p_refs = read_tip4p_references(repo)
    if tuple(sorted(tip4p_refs)) != tuple(range(TIP4P_MIN_N, TIP4P_MAX_N + 1)):
        raise SystemExit(f"TIP4P references are {sorted(tip4p_refs)}")
    tip4p_budget, tip4p_tol = tip4p_budget_and_tol(repo)
    sc_expected = sc_budgets(repo)

    libraries = {}
    library_trees = {}
    for name, tree in trees.items():
        libraries[name] = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        library_trees[name] = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD^{tree}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    if libraries["A"] != BASE_COMMIT or library_trees["A"] != BASE_TREE:
        raise SystemExit("arm A is not the pinned base")
    if library_trees["B"] != "fb238a7ed83cda76d4a188b093c0001543b05254":
        raise SystemExit(f"arm B tree is {library_trees['B']}")

    harness = {
        "run": git_blob(repo, "bench/nm3-land3-gate-b/run_gate.py"),
        "sum": git_blob(repo, "bench/nm3-land3-gate-b/summarize.py"),
        "toml": git_blob(repo, "bench/nm3-land3-gate-b/driver/Cargo.toml"),
        "rs": git_blob(repo, "bench/nm3-land3-gate-b/driver/src/main.rs"),
    }
    example_blobs = {
        name: {
            "h2h_sc": arm_blob(tree, "examples/h2h_sc.rs"),
            "h2h_seed": arm_blob(tree, "examples/h2h_seed.rs"),
        }
        for name, tree in trees.items()
    }

    def blobs_for(arm: str, driver: str) -> str:
        parts = [
            f"bench/nm3-land3-gate-b/run_gate.py:{harness['run']}",
            f"bench/nm3-land3-gate-b/summarize.py:{harness['sum']}",
        ]
        if driver == "h2h_sc":
            parts.append(f"examples/h2h_sc.rs:{example_blobs[arm]['h2h_sc']}")
        elif driver == "h2h_seed":
            parts.append(f"examples/h2h_seed.rs:{example_blobs[arm]['h2h_seed']}")
        else:
            parts.append(f"bench/nm3-land3-gate-b/driver/Cargo.toml:{harness['toml']}")
            parts.append(f"bench/nm3-land3-gate-b/driver/src/main.rs:{harness['rs']}")
        return ";".join(parts)

    preflight(repo, arms, env, work / "scratch")

    rows_path = args.rows
    existing = read_csv(rows_path)
    done = {
        (row["cohort"], row["system"], row["arm"], int(row["seed"]))
        for row in existing
        if row.get("status") == "ok"
    }
    session_id = existing[0]["session_id"] if existing else dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    machine = hashed_machine_id()
    planned = jobs()
    total = sum(len(seeds) * 2 for _s, _d, seeds, _c in planned)
    print(f"session={session_id} jobs={total} done={len(done)}", flush=True)

    for system, driver, seeds, cohort in planned:
        if driver == "h2h_seed":
            size = int(system.removeprefix("tip4p"))
            budget = tip4p_budget
            reference = tip4p_refs[size]
            reference_text = f"{reference:.16g}"
        else:
            seen = {
                query_budget(binary_of(bindir, driver), system, env) for bindir in arms.values()
            }
            if len(seen) != 1:
                raise SystemExit(f"{system} arms disagree on budget {seen}")
            budget, reference_text = seen.pop()
            reference = float(reference_text)
            if driver == "h2h_sc" and budget != sc_expected[system]:
                raise SystemExit(f"{system} budget {budget} is not the paper budget")
        for seed in seeds:
            order = arm_order(seed)
            start_path = None
            start_hash = ""
            if driver == "nm3_gate_driver":
                start_path = work / "starts" / f"{system}-{seed}.xyz"
                if not start_path.is_file():
                    subprocess.run(
                        [
                            str(binary_of(arms["A"], driver)),
                            "coords",
                            system,
                            str(seed),
                            str(start_path),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                start_hash = sha256_file(start_path)
            elif driver == "h2h_sc":
                hashes = []
                for name in ("A", "B"):
                    path = work / "starts" / f"{system}-{name}-{seed}.xyz"
                    subprocess.run(
                        [str(binary_of(arms[name], driver)), "coords", system, str(seed), str(path)],
                        check=True,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    hashes.append(sha256_file(path))
                if hashes[0] != hashes[1]:
                    raise SystemExit(f"{system} seed {seed} starts differ")
                start_hash = hashes[0]
            for slot, arm in enumerate(order, start=1):
                key = (cohort, system, arm, seed)
                if key in done:
                    continue
                prefix = work / "tip4p" / f"{cohort}-{system}-{arm}-s{seed}"
                argv = command_for(
                    driver,
                    binary_of(arms[arm], driver),
                    system,
                    seed,
                    budget,
                    reference,
                    tip4p_tol,
                    prefix,
                    start_path,
                )
                log = logs / f"{cohort}-{system}-{arm}-s{seed}.log"
                status, out = run_job(argv, args.cpu, env, work / "scratch", log, args.job_timeout_s)
                if status != "ok":
                    print(f"FAIL {system} {cohort} seed {seed} arm {arm} {status}", flush=True)
                    return 1
                parsed = parse_result(out, system, seed, budget)
                wall = float(parsed["wall_s"])
                pot = float(parsed["pot_s"])
                outside = float(parsed["outside_s"]) if "outside_s" in parsed else max(wall - pot, 0.0)
                outside = max(outside, 0.0)
                charged = int(float(parsed["charged"]))
                per_call = outside / charged if charged else float("nan")
                row = {
                    "system": system,
                    "arm": arm,
                    "seed": str(seed),
                    "slot": str(slot),
                    "core": args.cpu,
                    "cohort": cohort,
                    "hit": parsed["hit"],
                    "calls": parsed["calls"],
                    "charged": str(charged),
                    "wall_s": f"{wall:.6f}",
                    "pot_s": f"{pot:.6f}",
                    "outside_s": f"{outside:.6f}",
                    "outside_per_call_s": f"{per_call:.12g}",
                    "best": parsed["best"],
                    "best_bits": parsed["best_bits"],
                    "budget": str(budget),
                    "reference": reference_text,
                    "machine_id": machine,
                    "base_commit": BASE_COMMIT,
                    "base_tree": BASE_TREE,
                    "library": libraries[arm],
                    "library_tree": library_trees[arm],
                    "harness_blobs": blobs_for(arm, driver),
                    "driver": driver,
                    "status": "ok",
                    "start_sha256": start_hash,
                    "trace_sha256": sha256_file(Path(f"{prefix}.trace.csv")) if driver == "h2h_seed" else "",
                    "xyz_sha256": sha256_file(Path(f"{prefix}.xyz")) if driver == "h2h_seed" else "",
                    "gmin_sha256": sha256_file(Path(f"{prefix}.gmin")) if driver == "h2h_seed" else "",
                    "session_id": session_id,
                    "utc": utc_now(),
                }
                if driver == "h2h_seed":
                    start_file = Path(f"{prefix}.start.gmin")
                    if start_file.is_file():
                        row["start_sha256"] = sha256_file(start_file)
                append_csv(rows_path, ROW_FIELDS, row)
                done.add(key)
                print(
                    f"[{len(done)}/{total}] {cohort} {system} seed={seed} slot={slot} arm={arm} "
                    f"hit={row['hit']} calls={row['calls']} wall={row['wall_s']} "
                    f"out_per_call={row['outside_per_call_s']}",
                    flush=True,
                )
            if driver == "h2h_seed":
                starts = [
                    sha256_file(work / "tip4p" / f"{cohort}-{system}-{arm}-s{seed}.start.gmin")
                    for arm in ("A", "B")
                ]
                if not starts[0] or not starts[1] or starts[0] != starts[1]:
                    raise SystemExit(f"{system} seed {seed} tip4p starts differ")
    print(f"complete session={session_id} rows={len(done)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
