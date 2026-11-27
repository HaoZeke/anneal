#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Full-budget (H2O)8 polish-on-improve rows for two shards.

``calls_to_first_hit`` is ``first_hit_call`` from ``account.patch`` applied to
``examples/gfn2_mol_h2h.rs`` at ``b35601b7``. That is the first finite
objective evaluation at or below the harness threshold. ``H2O_TARGET_EV`` is
left unset, so ``Ledger::stop_below`` is not installed. The crossing index
does not depend on that stop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE_EH = -40.681336013150
TOL_EH = 1.0e-6
HARTREE_EV = 27.211386245988
REFERENCE_EV = REFERENCE_EH * HARTREE_EV
TOL_EV = TOL_EH * HARTREE_EV
THRESHOLD_EV = REFERENCE_EV + TOL_EV
BUDGET = 151738
SEEDS_A = (2000, 2001, 2002, 2003, 2004, 2005)
ENGINE_COMMIT = "b35601b7b7b2325c28f20ac604d267965cd64648"
ENGINE_TREE = "9ac668b61723fed09525bf8bc5d555d3afceeaaf"
EXPECTED_CONFIG = "985afef7a27e65816786baf78db2a2eda7c2363a1ed929467ca04697f4119f15"
RGPOT_COMMIT = "8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e"

ROW_FIELDS = [
    "seed",
    "budget",
    "calls_charged",
    "wall_s",
    "cpu_s",
    "best_Eh",
    "hit",
    "calls_to_first_hit",
    "revision",
    "host_id_hash",
    "core",
    "cpu_model",
]


def energy_hits(energy: float, target: float, tol: float) -> bool:
    if not (math.isfinite(energy) and math.isfinite(target) and math.isfinite(tol)):
        return False
    return tol >= 0.0 and energy <= target + tol


def recorded_hit(best_eV: float) -> bool:
    """Final energy at or below reference plus tolerance, in eV.

    Same test as ``bench/octamer-matched/h2o8/run.py``.
    """
    return energy_hits(best_eV, REFERENCE_EV, TOL_EV)


def scan_calls(lines, threshold: float) -> tuple[int | None, int]:
    hit = None
    count = 0
    for line in lines:
        text = line.strip()
        if not text:
            continue
        index_s, energy_s = text.split()
        count += 1
        energy = float(energy_s)
        if hit is None and energy_hits(energy, threshold, 0.0):
            hit = int(index_s)
    return hit, count


def self_test_a() -> None:
    threshold = (REFERENCE_EH + TOL_EH) * HARTREE_EV
    if threshold != THRESHOLD_EV:
        raise SystemExit("threshold")
    if not energy_hits(THRESHOLD_EV, THRESHOLD_EV, 0.0):
        raise SystemExit("boundary")
    if energy_hits(math.nextafter(THRESHOLD_EV, math.inf), THRESHOLD_EV, 0.0):
        raise SystemExit("boundary above")
    hit, count = scan_calls(["1 -1.0\n", "2 -1106.9955200453\n"], THRESHOLD_EV)
    if hit != 2 or count != 2:
        raise SystemExit(f"scan {hit} {count}")
    miss, count = scan_calls(["1 -1.0\n"], THRESHOLD_EV)
    if miss is not None or count != 1:
        raise SystemExit("scan miss")
    best = -1106.9955472624
    if f"{best / HARTREE_EV:.12f}" != "-40.681336013361":
        raise SystemExit("hartree")


def read_fields_a(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def host_id_hash() -> str:
    return "vm-A"


def cpu_model_a() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "model name":
            return value.strip()
    raise SystemExit("cpu model missing")


def need(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"missing {name}")
    return found


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_env_a(libdir: Path, revision: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    env["H2H_POLISH_ON_IMPROVE"] = "1"
    env["H2O_HIT_EV"] = f"{THRESHOLD_EV:.17g}"
    env["ANNEAL_SOURCE_COMMIT"] = revision
    env["ANNEAL_SOURCE_TREE"] = ENGINE_TREE
    env.pop("H2O_TARGET_EV", None)
    env.pop("H2H_START_ONLY", None)
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(libdir) if not previous else str(libdir) + os.pathsep + previous
    return env


def run_checked(
    binary: Path,
    args: list[str],
    out: Path,
    env: dict[str, str],
    core: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    command = [need("taskset"), "-c", core, str(binary), *args]
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        proc = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    if proc.returncode != 0:
        raise SystemExit(f"{args} exited {proc.returncode}")


def preflight(binary: Path, libdir: Path, runs: Path, core: str) -> None:
    out = runs / "preflight"
    env = base_env_a(libdir, ENGINE_COMMIT)
    env["H2H_START_ONLY"] = "1"
    run_checked(binary, ["h2o-8", "1", "2000", str(out)], out, env, core)
    fields = read_fields_a(out / "run.txt")
    if fields.get("polish_on_improve") != "true":
        raise SystemExit(f"polish_on_improve {fields.get('polish_on_improve')}")
    if fields.get("config_sha256") != EXPECTED_CONFIG:
        raise SystemExit(f"config_sha256 {fields.get('config_sha256')}")
    if fields.get("source_commit") != ENGINE_COMMIT:
        raise SystemExit(f"source_commit {fields.get('source_commit')}")


def parse_time(path: Path) -> tuple[float, float]:
    fields = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        fields[key] = float(value)
    return fields["elapsed"], fields["user"] + fields["sys"]


def finished_run(out: Path, budget: int) -> bool:
    run_txt = out / "run.txt"
    timing = out / "time.txt"
    if not (run_txt.is_file() and timing.is_file()):
        return False
    fields = read_fields_a(run_txt)
    return (
        fields.get("polish_on_improve") == "true"
        and fields.get("objective_best_eV") is not None
        and fields.get("first_hit_call") is not None
        and fields.get("energy_calls") == str(budget)
        and fields.get("config_sha256") == EXPECTED_CONFIG
    )


def run_seed(
    binary: Path,
    engine: Path,
    libdir: Path,
    seed: int,
    budget: int,
    out: Path,
    core: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    env = base_env_a(libdir, ENGINE_COMMIT)
    env["RGPOT_XTB_ENGINE"] = str(engine)
    command = [
        need("taskset"),
        "-c",
        core,
        str(binary),
        "h2o-8",
        str(budget),
        str(seed),
        str(out),
    ]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        proc = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    wall_s = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    user_s = after.ru_utime - before.ru_utime
    sys_s = after.ru_stime - before.ru_stime
    (out / "time.txt").write_text(f"elapsed {wall_s:.6f}\nuser {user_s:.6f}\nsys {sys_s:.6f}\n")
    if proc.returncode != 0 or not finished_run(out, budget):
        raise SystemExit(f"seed {seed} exited {proc.returncode}")


def row_from(
    out: Path,
    seed: int,
    budget: int,
    core: str,
    host: str,
    model: str,
    engine: Path,
) -> dict[str, str]:
    fields = read_fields_a(out / "run.txt")
    best_eV = float(fields["objective_best_eV"])
    best_Eh = best_eV / HARTREE_EV
    charged = int(fields["energy_calls"])
    first = fields.get("first_hit_call")
    if first is None:
        raise SystemExit(
            f"seed {seed} run.txt has no first_hit_call; account.patch applies to "
            "examples/gfn2_mol_h2h.rs at b35601b7"
        )
    hit_at = None if first == "none" else int(first)
    best_eV_hit = recorded_hit(best_eV)
    if best_eV_hit and hit_at is None:
        raise SystemExit(f"seed {seed} final energy hits and first_hit_call is none")
    wall_s, cpu_s = parse_time(out / "time.txt")
    session = {
        "seed": seed,
        "budget": budget,
        "calls_charged": charged,
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "search_wall_s": float(fields["wall_s"]),
        "best_Eh": best_Eh,
        "best_eV": best_eV,
        "hit": int(recorded_hit(best_eV)),
        "calls_to_first_hit": hit_at if hit_at is not None else budget + 1,
        "revision": fields["source_commit"],
        "host_id_hash": host,
        "core": core,
        "cpu_model": model,
        "engine_calls": int(fields["objective_calls"]) if fields.get("objective_calls") else charged,
        "intact": fields.get("best_intact"),
        "config_sha256": fields["config_sha256"],
        "polish_on_improve": fields["polish_on_improve"],
        "rgpot": RGPOT_COMMIT,
        "engine_sha256": file_sha256(engine),
    }
    (out / "session.json").write_text(json.dumps(session, sort_keys=True) + "\n")
    return {
        "seed": str(seed),
        "budget": str(budget),
        "calls_charged": str(charged),
        "wall_s": f"{wall_s:.6f}",
        "cpu_s": f"{cpu_s:.6f}",
        "best_Eh": f"{best_Eh:.12f}",
        "hit": str(session["hit"]),
        "calls_to_first_hit": str(session["calls_to_first_hit"]),
        "revision": session["revision"],
        "host_id_hash": host,
        "core": core,
        "cpu_model": model,
    }


def write_tables(rows: list[dict[str, str]], sessions: list[dict], rows_path: Path, sessions_path: Path) -> None:
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with sessions_path.open("w") as handle:
        for session in sessions:
            handle.write(json.dumps(session, sort_keys=True) + "\n")


def campaign_a(args: argparse.Namespace) -> None:
    self_test_a()
    binary = args.binary.resolve()
    engine = args.engine.resolve()
    libdir = args.libdir.resolve()
    runs = args.runs.resolve()
    runs.mkdir(parents=True, exist_ok=True)
    preflight(binary, libdir, runs, args.core)
    host = host_id_hash()
    model = cpu_model_a()
    rows = []
    sessions = []
    for seed in args.seeds:
        out = runs / str(seed)
        if not finished_run(out, args.budget):
            print(f"SEED {seed} start budget {args.budget}", flush=True)
            run_seed(binary, engine, libdir, seed, args.budget, out, args.core)
        row = row_from(out, seed, args.budget, args.core, host, model, engine)
        rows.append(row)
        sessions.append(json.loads((out / "session.json").read_text()))
        write_tables(rows, sessions, args.rows, args.sessions)
        print(
            f"SEED {seed} calls {row['calls_charged']} wall {row['wall_s']} "
            f"best {row['best_Eh']} hit {row['hit']} ctoh {row['calls_to_first_hit']}",
            flush=True,
        )


def main_a() -> None:
    if "--self-test" in sys.argv:
        self_test_a()
        print("self-test ok")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--libdir", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=HERE / "rows-A.csv")
    parser.add_argument("--sessions", type=Path, default=HERE / "sessions-A.jsonl")
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_A))
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--core", default="0")
    args = parser.parse_args()
    campaign_a(args)


MISS = BUDGET + 1
SEEDS = (2006, 2007, 2008, 2009, 2010, 2011)
ENGINE = "b35601b7b7b2325c28f20ac604d267965cd64648"
CONFIG_SHA = "985afef7a27e65816786baf78db2a2eda7c2363a1ed929467ca04697f4119f15"

COLUMNS = [
    "seed",
    "budget",
    "calls_charged",
    "wall_s",
    "cpu_s",
    "best_Eh",
    "best_eV",
    "hit",
    "calls_to_first_hit",
    "revision",
    "host_id",
    "core",
    "cpu_model",
]

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)



def read_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key] = value.strip()
    return fields


def host_id() -> str:
    return "vm0"


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def calls_to_first_hit(first: str) -> int:
    if first == "none":
        return MISS
    return int(first)


def measure_row(
    seed: int,
    fields: dict[str, str],
    wall_s: float,
    cpu_s: float,
    host: str,
    core: str,
    model: str,
) -> dict[str, str]:
    best_eV = float(fields["objective_best_eV"])
    hit = recorded_hit(best_eV)
    return {
        "seed": str(seed),
        "budget": str(BUDGET),
        "calls_charged": str(int(fields["energy_calls"])),
        "wall_s": f"{wall_s:.6f}",
        "cpu_s": f"{cpu_s:.6f}",
        "best_Eh": f"{best_eV / HARTREE_EV:.12f}",
        "best_eV": f"{best_eV:.10f}",
        "hit": "1" if hit else "0",
        "calls_to_first_hit": str(calls_to_first_hit(fields["first_hit_call"])),
        "revision": ENGINE,
        "host_id": host,
        "core": core,
        "cpu_model": model,
    }


def finished(run_txt: Path) -> bool:
    if not run_txt.is_file():
        return False
    fields = read_fields(run_txt)
    needed = ("wall_s", "objective_best_eV", "energy_calls", "first_hit_call", "config_sha256")
    return all(name in fields for name in needed)


def base_env(xtb_engine: Path, xtb_lib: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in THREAD_VARS:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["H2H_POLISH_ON_IMPROVE"] = "1"
    env["H2O_HIT_EV"] = f"{THRESHOLD_EV:.17g}"
    env["ANNEAL_SOURCE_COMMIT"] = ENGINE
    env["RGPOT_XTB_ENGINE"] = str(xtb_engine)
    env["LD_LIBRARY_PATH"] = str(xtb_lib)
    env.pop("H2H_START_ONLY", None)
    env.pop("H2O_TARGET_EV", None)
    return env


def run_process(binary: Path, seed: int, out: Path, core: str, env: dict[str, str]) -> tuple[int, float, float]:
    out.mkdir(parents=True, exist_ok=True)
    command = ["taskset", "-c", core, str(binary), "h2o-8", str(BUDGET), str(seed), str(out)]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        proc = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    wall_s = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_s = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    (out / "time.txt").write_text(f"wall_s: {wall_s:.6f}\ncpu_s: {cpu_s:.6f}\n")
    return proc.returncode, wall_s, cpu_s


def check_config(binary: Path, core: str, env: dict[str, str]) -> None:
    probe = HERE / "runs" / "probe"
    if probe.exists():
        for child in probe.iterdir():
            child.unlink()
    probe_env = dict(env)
    probe_env["H2H_START_ONLY"] = "1"
    code, _, _ = run_process(binary, SEEDS[0], probe, core, probe_env)
    fields = read_fields(probe / "run.txt") if (probe / "run.txt").is_file() else {}
    if code != 0 or fields.get("config_sha256") != CONFIG_SHA or fields.get("polish_on_improve") != "true":
        raise SystemExit(
            f"config probe failed code={code} sha={fields.get('config_sha256', 'missing')} "
            f"polish={fields.get('polish_on_improve', 'missing')}"
        )


def load_time(out: Path) -> tuple[float, float]:
    fields = read_fields(out / "time.txt")
    return float(fields["wall_s"]), float(fields["cpu_s"])


def session_record(row: dict[str, str], fields: dict[str, str]) -> dict[str, object]:
    record: dict[str, object] = dict(row)
    record["intact"] = fields.get("best_intact", "")
    record["config_sha256"] = fields.get("config_sha256", "")
    record["polish_on_improve"] = fields.get("polish_on_improve", "")
    record["objective_calls"] = fields.get("objective_calls", "")
    record["search_wall_s"] = fields.get("wall_s", "")
    record["search_calls"] = fields.get("search_calls", "")
    record["polish_calls"] = fields.get("polish_calls", "")
    return record


def write_outputs(rows: list[dict[str, str]], sessions: list[dict[str, object]], rows_path: Path, sessions_path: Path) -> None:
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with sessions_path.open("w") as handle:
        for record in sessions:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def campaign(binary: Path, xtb_engine: Path, xtb_lib: Path, core: str, rows_path: Path, sessions_path: Path) -> None:
    env = base_env(xtb_engine, xtb_lib)
    check_config(binary, core, env)
    host = host_id()
    model = cpu_model()
    rows: list[dict[str, str]] = []
    sessions: list[dict[str, object]] = []
    for seed in SEEDS:
        out = HERE / "runs" / str(seed)
        run_txt = out / "run.txt"
        if not (finished(run_txt) and (out / "time.txt").is_file()):
            print(f"seed {seed} start budget {BUDGET}", flush=True)
            code, wall_s, cpu_s = run_process(binary, seed, out, core, env)
            if code != 0 or not finished(run_txt):
                raise SystemExit(f"seed {seed} exited {code}")
        else:
            wall_s, cpu_s = load_time(out)
        fields = read_fields(run_txt)
        if fields.get("config_sha256") != CONFIG_SHA or fields.get("polish_on_improve") != "true":
            raise SystemExit(f"seed {seed} config {fields.get('config_sha256')} polish {fields.get('polish_on_improve')}")
        row = measure_row(seed, fields, wall_s, cpu_s, host, core, model)
        best_eV = float(fields["objective_best_eV"])
        crossed = recorded_hit(best_eV)
        if crossed and fields["first_hit_call"] == "none":
            raise SystemExit(f"seed {seed} final energy hits and first_hit_call is none")
        rows.append(row)
        sessions.append(session_record(row, fields))
        write_outputs(rows, sessions, rows_path, sessions_path)
        print(
            f"seed {seed} calls {row['calls_charged']} wall_s {row['wall_s']} "
            f"best_Eh {row['best_Eh']} hit {row['hit']} ctoh {row['calls_to_first_hit']}",
            flush=True,
        )
        if int(row["calls_charged"]) != BUDGET:
            raise SystemExit(f"seed {seed} charged {row['calls_charged']} budget {BUDGET}")


def self_test() -> None:
    if MISS != 151739 or BUDGET != 151738:
        raise SystemExit("budget")
    if abs((REFERENCE_EH + TOL_EH) * HARTREE_EV - THRESHOLD_EV) > 0.0:
        raise SystemExit("threshold")
    if not energy_hits(THRESHOLD_EV, THRESHOLD_EV, 0.0):
        raise SystemExit("boundary")
    if energy_hits(math.nextafter(THRESHOLD_EV, 0.0), THRESHOLD_EV, 0.0):
        raise SystemExit("above")
    if host_id() != "vm0":
        raise SystemExit("host")
    fields = {
        "objective_best_eV": f"{THRESHOLD_EV:.10f}",
        "energy_calls": str(BUDGET),
        "first_hit_call": "6254",
    }
    row = measure_row(2006, fields, 1.5, 1.25, "abc", "0", "cpu")
    if row["hit"] != "1" or row["calls_to_first_hit"] != "6254" or row["calls_charged"] != str(BUDGET):
        raise SystemExit(f"row {row}")
    miss = dict(fields)
    miss["objective_best_eV"] = "0"
    miss["first_hit_call"] = "none"
    missed = measure_row(2006, miss, 1.0, 1.0, "abc", "0", "cpu")
    if missed["hit"] != "0" or missed["calls_to_first_hit"] != str(MISS):
        raise SystemExit(f"miss {missed}")
    print("self-test ok")


def main_b() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--xtb-engine", type=Path)
    parser.add_argument("--xtb-lib", type=Path)
    parser.add_argument("--core", default=os.environ.get("H2O8_CORE", "0"))
    parser.add_argument("--rows", type=Path, default=HERE / "rows-B.csv")
    parser.add_argument("--sessions", type=Path, default=HERE / "sessions-B.jsonl")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.binary is None or args.xtb_engine is None or args.xtb_lib is None:
        raise SystemExit("binary, xtb-engine, and xtb-lib are required")
    campaign(args.binary, args.xtb_engine, args.xtb_lib, args.core, args.rows, args.sessions)


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--shard" or sys.argv[2] not in ("A", "B"):
        raise SystemExit("pass --shard A or --shard B before that shard's arguments")
    shard = sys.argv[2]
    sys.argv = [sys.argv[0], *sys.argv[3:]]
    if shard == "A":
        main_a()
    else:
        main_b()


if __name__ == "__main__":
    main()
