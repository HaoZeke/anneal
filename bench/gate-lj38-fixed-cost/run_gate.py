#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session default-on gate for the lj38 fixed-cost change.

One serial session on core 0. Arms A, B and A2 (a second release build
of A) run on every cell in palindromic blocks. Cells use the default
cluster presets. Each row records outside time, calls and in-process
peak RssAnon. The statistics are
in analyze.py and are fixed by PROTOCOL.md before any data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BASE_REF = "9b65fda3972fd8c014bce886a3ea5c05956d64af"
B_COMMIT = "f9bbb5e86a67b7bbfaa629ac7378306f1f22cd1d"
B_TREE = "510c087e8c6e4c9aeedae202204a7919cee95a60"
FIRST_SEED = 2000
N_SEEDS = {
    "lj38": 24,
    "m38r14d": 12,
    "lj75": 12,
    "lj98": 12,
    "m80r6": 12,
    "au55": 12,
    "hexamer": 12,
}
MOLECULES: list[str] = []
GUARDS = list(N_SEEDS)
CELL_ORDER = MOLECULES + GUARDS
GUARD_BUDGET = {
    "lj38": 400_000,
    "m38r14d": 400_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "m80r6": 4_000_000,
    "au55": 1_000_000,
    "hexamer": 500_000,
}
# Separate from the 288-job panel. Strict identity only, sealed on its own.
IDENTITY_N = {
    "tip4p6": 12,
    "sc126-38": 12,
}
IDENTITY_ORDER = list(IDENTITY_N)
IDENTITY_BUDGET = {
    "tip4p6": 500_000,
    "sc126-38": 400_000,
}
HARNESS_FILES = (
    "bench/gate-lj38-fixed-cost/run_gate.py",
    "bench/gate-lj38-fixed-cost/guard_trace.rs",
    "bench/gate-lj38-fixed-cost/analyze.py",
)
ROW_FIELDS = [
    "cell", "kind", "seed", "arm", "slot", "core", "hit", "budget", "calls", "charged",
    "wall_s", "inside_s", "outside_per_call_s", "rss_anon_kb", "rss_end_kb", "energy", "energy_bits", "trajectory",
    "machine_id", "base_ref", "base_tree", "harness_blobs", "error",
]


def kind_of(cell: str) -> str:
    return "guard"


def block(cell: str, pair: int) -> list[tuple[str, int, int]]:
    first = FIRST_SEED + 2 * pair
    second = first + 1
    return [
        ("A", first, 0),
        ("B", first, 1),
        ("Ap", first, 2),
        ("Ap", second, 3),
        ("B", second, 4),
        ("A", second, 5),
    ]


def jobs_for(order: list[str], counts: dict[str, int], phase: str) -> list[dict]:
    jobs = []
    index = 0
    for pair in range(max(counts.values()) // 2):
        for cell in order:
            if pair >= counts[cell] // 2:
                continue
            for arm, seed, slot in block(cell, pair):
                jobs.append(
                    {
                        "cell": cell,
                        "kind": kind_of(cell),
                        "phase": phase,
                        "seed": seed,
                        "arm": arm,
                        "slot": slot,
                        "index": index,
                        "block": pair,
                    }
                )
                index += 1
    return jobs


def schedule() -> list[dict]:
    return jobs_for(CELL_ORDER, N_SEEDS, "session")


def identity_schedule() -> list[dict]:
    return jobs_for(IDENTITY_ORDER, IDENTITY_N, "identity")


def schedule_id(jobs: list[dict]) -> str:
    blob = json.dumps(
        [(j["cell"], j["seed"], j["arm"], j["slot"], j["index"]) for j in jobs],
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def expected_balance(cell: str, n_seeds: int) -> Counter:
    pairs = [(0, "A"), (5, "A"), (1, "B"), (4, "B"), (2, "Ap"), (3, "Ap")]
    return Counter({pair: n_seeds // 2 for pair in pairs})


def check_balance_for(jobs: list[dict], order: list[str], counts: dict[str, int]) -> dict[str, Counter]:
    found = {}
    for cell in order:
        mine = [j for j in jobs if j["cell"] == cell]
        got = Counter((j["slot"], j["arm"]) for j in mine)
        want = expected_balance(cell, counts[cell])
        if got != want:
            raise SystemExit(f"unbalanced {cell}: {got} != {want}")
        arms = sorted({j["arm"] for j in mine})
        means = []
        for arm in arms:
            idx = [j["index"] for j in mine if j["arm"] == arm]
            means.append(sum(idx) / len(idx))
            if len(idx) != counts[cell]:
                raise SystemExit(f"{cell} {arm} has {len(idx)} runs")
        if max(means) - min(means) > 1e-9:
            raise SystemExit(f"unbalanced position {cell}: {means}")
        found[cell] = got
    return found


def check_balance(jobs: list[dict]) -> dict[str, Counter]:
    return check_balance_for(jobs, CELL_ORDER, N_SEEDS)


def budget_of(cell: str) -> int:
    if cell in GUARD_BUDGET:
        return GUARD_BUDGET[cell]
    return IDENTITY_BUDGET[cell]


def outside_per_call(wall: float, inside: float, calls: int) -> float:
    if calls <= 0:
        return float("nan")
    return (wall - inside) / calls


def thread_env(extra: dict[str, str]) -> dict[str, str]:
    kept = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "RUSTUP_HOME", "CARGO_HOME")
    env = {key: os.environ[key] for key in kept if key in os.environ}
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "GOTO_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "NUMBA_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
            "JULIA_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    env.update(extra)
    return env


def affinity() -> str:
    text = Path("/proc/self/status").read_text()
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def require_core() -> None:
    if affinity() != "0":
        raise SystemExit("this process is not pinned to core 0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def require_object(repo: Path, obj: str) -> None:
    proc = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", obj], capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"git object missing: {obj}")


def provenance(repo: Path) -> dict[str, str]:
    require_object(repo, BASE_REF)
    base_tree = git_text(repo, "rev-parse", f"{BASE_REF}^{{tree}}")
    require_object(repo, base_tree)
    blobs = []
    for rel in HARNESS_FILES:
        digest = git_text(repo, "hash-object", rel)
        require_object(repo, digest)
        blobs.append(f"{Path(rel).name}:{digest}")
    return {"base_ref": BASE_REF, "base_tree": base_tree, "harness_blobs": " ".join(blobs)}


def cpu_info() -> tuple[str, str]:
    model, flags = "", ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name") and not model:
            model = line.split(":", 1)[1].strip()
        if line.startswith("flags") and not flags:
            flags = line.split(":", 1)[1].strip()
    return model, flags


def machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/sys/class/dmi/id/product_uuid")):
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes().strip()).hexdigest()
    raise SystemExit("machine id is unavailable")


def job_dir(run_dir: Path, job: dict) -> Path:
    return run_dir / "jobs" / job["phase"] / job["cell"] / str(job["seed"]) / job["arm"]


def result_ok(path: Path, ident: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return data.get("complete") is True and data.get("schedule") == ident


def classify_error(text: str) -> str:
    if "no finite start" in text or "finite start" in text:
        return "no finite start"
    if text.strip():
        return "run failed"
    return ""


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def log_line(journal: Path, text: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{stamp} {text}\n"
    with journal.open("a") as handle:
        handle.write(line)
    print(text, flush=True)


def journal_raw(journal: Path, text: str) -> None:
    with journal.open("a") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def run_guard(args, job: dict, binary: Path, ident: str, journal: Path) -> None:
    out = job_dir(args.run_dir, job)
    dest = out / "result.json"
    if result_ok(dest, ident):
        return
    out.mkdir(parents=True, exist_ok=True)
    env = thread_env({})
    cmd = ["taskset", "-c", "0", str(binary), job["cell"], str(job["seed"])]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.stderr:
        journal_raw(journal, proc.stderr)
    gate = parse_gate(proc.stdout)
    if gate is None or gate.get("core") != "0" or int(gate.get("budget", "0")) != budget_of(job["cell"]):
        write_result(
            dest,
            {
                "complete": True,
                "schedule": ident,
                "cell": job["cell"],
                "kind": "guard",
                "phase": job["phase"],
                "seed": job["seed"],
                "arm": job["arm"],
                "slot": job["slot"],
                "index": job["index"],
                "core": gate.get("core") if gate else "",
                "budget": budget_of(job["cell"]),
                "hit": 0,
                "calls": 0,
                "charged": 0,
                "wall_s": 0.0,
                "inside_s": 0.0,
                "energy_bits": "",
                "trajectory": "",
                "error": classify_error(proc.stderr) or "run failed",
                "returncode": proc.returncode,
            },
        )
    else:
        write_result(
            dest,
            {
                "complete": True,
                "schedule": ident,
                "cell": job["cell"],
                "kind": "guard",
                "phase": job["phase"],
                "seed": job["seed"],
                "arm": job["arm"],
                "slot": job["slot"],
                "index": job["index"],
                "core": gate["core"],
                "budget": int(gate["budget"]),
                "hit": int(gate["hit"]),
                "calls": int(gate["calls"]),
                "charged": int(gate["charged"]),
                "wall_s": float(gate["wall_s"]),
                "inside_s": float(gate["inside_s"]),
                "rss_anon_kb": int(gate["rss_anon_kb"]),
                "rss_end_kb": int(gate["rss_end_kb"]),
                "energy_bits": gate["best_bits"],
                "trajectory": gate["trajectory"],
                "error": "",
            },
        )
    log_line(
        journal,
        f"guard {job['cell']} seed {job['seed']} {job['arm']} slot {job['slot']} exit {proc.returncode}",
    )


def parse_gate(text: str) -> dict[str, str] | None:
    for line in reversed(text.splitlines()):
        if line.startswith("GATE "):
            fields = {}
            for part in line.split()[1:]:
                key, value = part.split("=", 1)
                fields[key] = value
            return fields
    return None


def bits_to_float(text: str) -> float:
    return struct.unpack("<d", int(text, 16).to_bytes(8, "little"))[0]


def assemble_records(run_dir: Path, jobs: list[dict], prov: dict, machine: str) -> list[dict]:
    records = []
    for job in jobs:
        info = json.loads((job_dir(run_dir, job) / "result.json").read_text())
        if True:
            hit = int(info.get("hit") or 0)
            calls = int(info.get("calls") or 0)
            charged = int(info.get("charged") or 0)
            wall = float(info.get("wall_s") or 0.0)
            inside = float(info.get("inside_s") or 0.0)
            bits = info.get("energy_bits") or ""
            energy = f"{bits_to_float(bits):.16e}" if bits else ""
            trajectory = info.get("trajectory") or ""
            error = info.get("error") or ""
            core = info.get("core") or ""
            budget = int(info.get("budget") or 0)
            rss = int(info.get("rss_anon_kb") or 0)
            rss_end = int(info.get("rss_end_kb") or 0)
        outside = outside_per_call(wall, inside, calls)
        records.append(
            {
                "cell": job["cell"],
                "kind": job["kind"],
                "phase": job["phase"],
                "seed": job["seed"],
                "arm": job["arm"],
                "slot": job["slot"],
                "core": core,
                "hit": hit,
                "budget": budget,
                "calls": calls,
                "charged": charged,
                "wall_s": f"{wall:.9f}",
                "inside_s": f"{inside:.9f}",
                "outside_per_call_s": f"{outside:.9e}",
                "rss_anon_kb": rss,
                "rss_end_kb": rss_end,
                "energy": energy,
                "energy_bits": bits,
                "trajectory": trajectory,
                "machine_id": machine,
                "base_ref": prov["base_ref"],
                "base_tree": prov["base_tree"],
                "harness_blobs": prov["harness_blobs"],
                "error": error,
            }
        )
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def self_check() -> None:
    jobs = schedule()
    check_balance(jobs)
    want = sum(N_SEEDS.values()) * 3
    if len(jobs) != want:
        raise SystemExit(f"schedule length {len(jobs)} != {want}")


def write_sessions(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def run(args) -> int:
    require_core()
    if "avx2" not in cpu_info()[1].split():
        raise SystemExit("host has no avx2; the changed code needs an avx2 host")
    self_check()
    identity = args.cmd == "identity"
    counts_map = IDENTITY_N if identity else N_SEEDS
    order = IDENTITY_ORDER if identity else CELL_ORDER
    budgets = IDENTITY_BUDGET if identity else GUARD_BUDGET
    b_tree = git_text(args.src_b, "rev-parse", "HEAD^{tree}")
    if b_tree != B_TREE:
        raise SystemExit("B tree does not match")
    a_tree = git_text(args.src_a, "rev-parse", "HEAD^{tree}")
    ap_tree = git_text(args.src_ap, "rev-parse", "HEAD^{tree}")
    if a_tree != ap_tree or a_tree != git_text(args.repo, "rev-parse", f"{BASE_REF}^{{tree}}"):
        raise SystemExit("A and A2 are not the develop tree")
    prov = provenance(args.repo)
    machine = machine_id()
    jobs = identity_schedule() if identity else schedule()
    balances = check_balance_for(jobs, order, counts_map)
    if len(jobs) != sum(counts_map.values()) * 3:
        raise SystemExit(f"schedule length {len(jobs)}")
    ident = schedule_id(jobs)
    journal = args.run_dir / "journal.log"
    journal.parent.mkdir(parents=True, exist_ok=True)
    log_line(journal, f"first seed {FIRST_SEED} n {dict(counts_map)} cells {len(order)}")
    log_line(journal, f"session {ident[:12]} jobs {len(jobs)}")
    for position, job in enumerate(jobs, 1):
        run_guard(args, job, args.bin[job["arm"]], ident, journal)
        if position % 20 == 0:
            log_line(journal, f"progress {position}/{len(jobs)}")
    records = assemble_records(args.run_dir, jobs, prov, machine)
    builds = {arm: sha256_file(args.bin[arm]) for arm in ("A", "B", "Ap")}
    sessions = [
        ("base_ref", prov["base_ref"]),
        ("base_tree", prov["base_tree"]),
        ("b_commit", B_COMMIT),
        ("b_tree", B_TREE),
        ("harness_blobs", prov["harness_blobs"]),
        ("machine_id", machine),
        ("seeds_lj38", f"{FIRST_SEED}-{FIRST_SEED + N_SEEDS['lj38'] - 1}"),
        ("seeds_others", f"{FIRST_SEED}-{FIRST_SEED + 11}"),
        ("n", json.dumps(counts_map, sort_keys=True)),
        ("panel", "identity" if identity else "main"),
        ("cpu_model", cpu_info()[0]),
        ("cpu_flags", cpu_info()[1]),
        ("avx2", str("avx2" in cpu_info()[1].split()).lower()),
        ("core", "0"),
        ("threads", "1"),
        ("schedule", ident),
        ("build_a", builds["A"]),
        ("build_b", builds["B"]),
        ("build_a2", builds["Ap"]),
    ]
    for cell, counts in balances.items():
        text = " ".join(f"{slot}:{arm}={count}" for (slot, arm), count in sorted(counts.items()))
        sessions.append((f"balance_{cell}", text))
    for cell, budget in budgets.items():
        sessions.append((f"budget_{cell}", str(budget)))
    write_csv(args.run_dir / "rows.csv", records)
    write_csv(args.rows, records)
    write_sessions(args.sessions, sessions)
    log_line(journal, "rows written; run analyze.py on rows.csv")
    return 0


COUNTER_NAMES = [
    "solve_shortest_paths_loop_avx2",
    "kuhn_munkres_loop_avx2",
    "augment_tight_avx2",
    "csm_recall_op",
    "candidate_from_basis",
]
COUNTER_RUNS = [("lj38", 2000), ("lj38", 2001), ("m38r14d", 2000)]


def counters(args) -> int:
    """Runs the counter build of B on a few seeds and checks the paths fire.

    The counter build is not a timed arm. Its trajectory, energy bits,
    calls and charged calls must equal the timed B row of the same seed.
    """
    require_core()
    with args.rows.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    total = [0] * len(COUNTER_NAMES)
    lines = []
    ok = True
    for cell, seed in COUNTER_RUNS:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(args.bin_bc), cell, str(seed)],
            env=thread_env({}),
            capture_output=True,
            text=True,
        )
        gate = parse_gate(proc.stdout)
        counts = None
        for line in proc.stdout.splitlines():
            if line.startswith("COUNTERS "):
                counts = [int(x) for x in line.split()[1:]]
        if gate is None or counts is None or len(counts) != len(COUNTER_NAMES):
            raise SystemExit(f"counter run failed for {cell} {seed}")
        match = [r for r in rows if r["cell"] == cell and r["arm"] == "B" and int(r["seed"]) == seed]
        same = len(match) == 1 and all(
            str(match[0][k]) == str(v)
            for k, v in (
                ("trajectory", gate["trajectory"]),
                ("energy_bits", gate["best_bits"]),
                ("calls", gate["calls"]),
                ("charged", gate["charged"]),
            )
        )
        ok = ok and same
        total = [a + b for a, b in zip(total, counts)]
        lines.append([cell, seed, *counts, "identical_to_B" if same else "DIFFERS_FROM_B"])
    lines.append(["total", "", *total, ""])
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cell", "seed", *COUNTER_NAMES, "check"])
        writer.writerows(lines)
    zero = [n for n, c in zip(COUNTER_NAMES, total) if c == 0]
    print("zero counters:", zero or "none")
    print("counter build identical to timed B:", ok)
    return 0 if ok and not zero else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("schedule")

    run_p = sub.add_parser("run")
    run_p.add_argument("--repo", type=Path, required=True)
    run_p.add_argument("--run-dir", type=Path, required=True)
    run_p.add_argument("--src-a", type=Path, required=True)
    run_p.add_argument("--src-b", type=Path, required=True)
    run_p.add_argument("--src-ap", type=Path, required=True)
    run_p.add_argument("--bin-a", type=Path, required=True)
    run_p.add_argument("--bin-b", type=Path, required=True)
    run_p.add_argument("--bin-ap", type=Path, required=True)
    run_p.add_argument("--rows", type=Path, required=True)
    run_p.add_argument("--sessions", type=Path, required=True)

    ident_p = sub.add_parser("identity")
    ident_p.add_argument("--repo", type=Path, required=True)
    ident_p.add_argument("--run-dir", type=Path, required=True)
    ident_p.add_argument("--src-a", type=Path, required=True)
    ident_p.add_argument("--src-b", type=Path, required=True)
    ident_p.add_argument("--src-ap", type=Path, required=True)
    ident_p.add_argument("--bin-a", type=Path, required=True)
    ident_p.add_argument("--bin-b", type=Path, required=True)
    ident_p.add_argument("--bin-ap", type=Path, required=True)
    ident_p.add_argument("--rows", type=Path, required=True)
    ident_p.add_argument("--sessions", type=Path, required=True)

    count_p = sub.add_parser("counters")
    count_p.add_argument("--bin-bc", type=Path, required=True)
    count_p.add_argument("--rows", type=Path, required=True)
    count_p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "schedule":
        self_check()
        jobs = schedule()
        counts = check_balance(jobs)
        for cell, counter in counts.items():
            text = " ".join(f"{slot}:{arm}={n}" for (slot, arm), n in sorted(counter.items()))
            print(f"{cell} {text}")
        print(f"jobs {len(jobs)} id {schedule_id(jobs)[:12]}")
        return 0
    if args.cmd == "counters":
        return counters(args)
    if args.cmd == "identity":
        jobs = identity_schedule()
        check_balance_for(jobs, IDENTITY_ORDER, IDENTITY_N)
        print(f"identity jobs {len(jobs)} id {schedule_id(jobs)[:12]}")
    args.bin = {"A": args.bin_a, "B": args.bin_b, "Ap": args.bin_ap}
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
