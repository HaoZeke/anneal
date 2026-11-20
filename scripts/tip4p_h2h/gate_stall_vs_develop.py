#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial gate for fix/tip4p-generic-stall and PR #160 vs develop.

Rules and bars are in gates/anneal-tip4p-stall-vs-develop-gate.md.
``build`` compiles one release arm in its own target directory.
``run --campaign rigid`` runs TIP4P9..21 (seeds 2000..2023) on arms A,
B, C and A2 in a 4x4 Williams order rotated by seed; ``run --campaign
nonrigid`` runs the thirteen-system paper panel on A, B and C in a 3-arm
Williams order rotated by seed. ``score`` reads the journals and writes
rows.csv, summary.json and session.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_h2h  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

COMMITS = {
    "A": "61fa67cf380e46876b569da45e7cc091647a775b",
    "A2": "61fa67cf380e46876b569da45e7cc091647a775b",
    "B": "87280ef9745f8367d9857d757e21af35410d7b25",
    "C": "ea777eafb9b80d8b93d94e68f499d66740e8a05b",
}
TREES = {
    "A": "cff796c507a8fa1d0bfb28888c3570b61988055e",
    "A2": "cff796c507a8fa1d0bfb28888c3570b61988055e",
    "B": "b778132b34901c95570b129acb44ab6bb0e06402",
    "C": "da15bc0a999531019e6e53fcc6b6b104afc47187",
}
LABELS = {"A": "develop", "A2": "develop-2", "B": "stall", "C": "pr160"}
SIZES = tuple(range(9, 22))
PRIMARY_SEEDS = tuple(range(2000, 2012))
SECOND_SEEDS = tuple(range(2012, 2024))
SEEDS = PRIMARY_SEEDS + SECOND_SEEDS
NONRIGID_SEEDS = (2000, 2001, 2002)
GATED_WINDOW = tuple(range(15, 22))
BUDGET = 2_000_000
CORE = "0"
COST_LIMIT = 1.10
PAUSE_GAP_S = 5.0
BINARIES = ("h2h_seed", "h2h_gmin_proof", "h2h_morse", "h2h_metal")
WILLIAMS_4 = (
    ("A", "B", "C", "A2"),
    ("B", "A2", "A", "C"),
    ("A2", "C", "B", "A"),
    ("C", "A", "A2", "B"),
)
WILLIAMS_3 = (
    ("A", "B", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("C", "B", "A"),
    ("A", "C", "B"),
    ("B", "A", "C"),
)
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
    "TBB_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "TOKIO_WORKER_THREADS",
)
FIELDS = [
    "campaign",
    "seed_set",
    "driver",
    "cell",
    "seed",
    "arm",
    "label",
    "round",
    "slot",
    "commit",
    "tree",
    "machine_id",
    "core",
    "budget",
    "reference",
    "hit",
    "evals_to_hit",
    "calls",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "best",
    "search_best",
    "lowest",
    "geometry_energy",
    "hops",
    "basins",
    "start_sha",
    "trace_sha",
    "geometry_sha",
    "rss_anon_kb",
    "ru_maxrss_kb",
]
PUBLIC_FIELDS = FIELDS + ["pause_gap_s"]
IDENTITY = ("hit", "calls", "charged", "best", "search_best", "basins", "trace_sha", "geometry_sha")
NONRIGID_IDENTITY = (
    "hit",
    "evals_to_hit",
    "calls",
    "charged",
    "best",
    "search_best",
    "lowest",
    "geometry_energy",
    "hops",
    "basins",
    "start_sha",
    "trace_sha",
    "geometry_sha",
)


def ccd_references() -> dict[int, float]:
    out = {}
    with (ROOT / "data" / "tip4p_ccd.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            out[int(row["n"])] = float(row["energy"])
    for n in (8, 15, 20, 21):
        if out[n] != run_h2h.CAMBRIDGE[n]:
            raise SystemExit(f"Cambridge table disagrees with run_h2h at {n}")
    return out


def load_panel():
    import importlib.util

    path = ROOT / "scripts" / "gmin_panel" / "run_panel.py"
    spec = importlib.util.spec_from_file_location("gmin_panel_run", path)
    if spec is None or spec.loader is None:
        raise SystemExit("panel module is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    if not raw:
        raise SystemExit("machine id is empty")
    return hashlib.sha256(raw).hexdigest()


def thread_env_dict() -> dict[str, str]:
    out = {key: "1" for key in THREAD_KEYS}
    out["OMP_DYNAMIC"] = "FALSE"
    out["MKL_DYNAMIC"] = "FALSE"
    return out


def run_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_", "PRIOR_")):
            del env[key]
        elif key.endswith(("_NUM_THREADS", "_MAX_THREADS")):
            env[key] = "1"
    env.update(thread_env_dict())
    if extra:
        env.update(extra)
    return env


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def type7(values: list[float]) -> float:
    return run_h2h.type7(values)


def rigid_order(seed: int) -> tuple[str, ...]:
    return WILLIAMS_4[(seed - SEEDS[0]) % len(WILLIAMS_4)]


def nonrigid_order(seed: int) -> tuple[str, ...]:
    return WILLIAMS_3[(seed - NONRIGID_SEEDS[0]) % len(WILLIAMS_3)]


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class Arm:
    def __init__(self, name: str, spec: str):
        bin_dir, _, worktree = spec.partition(":")
        self.name = name
        self.bin = Path(bin_dir) / "bin"
        self.worktree = Path(worktree)
        self.commit = (Path(bin_dir) / "commit").read_text().strip()
        self.tree = (Path(bin_dir) / "tree").read_text().strip()
        if self.commit != COMMITS[name]:
            raise SystemExit(f"STOP: arm {name} commit {self.commit} is not {COMMITS[name]}")
        if self.tree != TREES[name]:
            raise SystemExit(f"STOP: arm {name} tree {self.tree} is not {TREES[name]}")
        live_commit = git(self.worktree, "rev-parse", "HEAD")
        live_tree = git(self.worktree, "rev-parse", "HEAD^{tree}")
        if live_commit != self.commit or live_tree != self.tree or git(self.worktree, "status", "--porcelain"):
            raise SystemExit(f"STOP: arm {name} worktree is not clean at {self.commit} / {self.tree}")
        for binary in BINARIES:
            if not (self.bin / binary).is_file():
                raise SystemExit(f"arm {name} lacks {binary}")

    def metadata(self) -> dict[str, object]:
        return {
            "label": LABELS[self.name],
            "commit": self.commit,
            "tree": self.tree,
            "worktree_clean": True,
            "binaries_sha256": {b: sha256_file(self.bin / b) for b in BINARIES},
        }


def read_proc_status(pid: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return out
    for line in text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def execute(
    argv: list[str], log: Path, cwd: Path, extra: dict[str, str] | None = None, sample_taskset: bool = False
) -> tuple[str, int, int, dict[str, str]]:
    log.parent.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {int(CORE)})
    command = ["taskset", "-c", CORE, *argv]
    peak_anon = [0]
    pin_info: dict[str, str] = {}
    stop_flag = [False]

    with log.open("w") as handle:
        proc = subprocess.Popen(
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, env=run_env(extra)
        )

        def sampler() -> None:
            try:
                if os.cpu_count() and os.cpu_count() > 1:
                    os.sched_setaffinity(threading.get_native_id(), {1})
            except OSError:
                pass
            first = True
            while not stop_flag[0]:
                st = read_proc_status(proc.pid)
                if not st:
                    break
                if "RssAnon" in st:
                    parts = st["RssAnon"].split()
                    if parts:
                        val = int(parts[0])
                        if val > peak_anon[0]:
                            peak_anon[0] = val
                if first and "Cpus_allowed_list" in st:
                    pin_info["cpus_allowed"] = st.get("Cpus_allowed", "")
                    pin_info["cpus_allowed_list"] = st["Cpus_allowed_list"]
                    if sample_taskset:
                        tp = subprocess.run(
                            ["taskset", "-p", str(proc.pid)],
                            capture_output=True,
                            text=True,
                            check=False,
                        ).stdout.strip()
                        pin_info["taskset_p"] = tp
                    first = False
                time.sleep(0.004)

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        _, wait_status, rusage = os.wait4(proc.pid, 0)
        stop_flag[0] = True
        thread.join()

    rc = os.waitstatus_to_exitcode(wait_status)
    text = log.read_text()
    if rc != 0:
        raise SystemExit(f"driver exited {rc}; see {log.name}")
    if pin_info.get("cpus_allowed_list", CORE) != CORE:
        raise SystemExit(f"STOP: child Cpus_allowed_list {pin_info.get('cpus_allowed_list')} != {CORE}")
    return text, peak_anon[0], int(rusage.ru_maxrss), pin_info


def capture_pinning_evidence() -> dict[str, str]:
    os.sched_setaffinity(0, {int(CORE)})
    parent_tp = subprocess.run(
        ["taskset", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    ).stdout.strip()
    parent_st = read_proc_status(os.getpid())
    proc = subprocess.Popen(
        ["taskset", "-c", CORE, "sh", "-c", "sleep 0.05"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=run_env(),
    )
    time.sleep(0.01)
    child_tp = subprocess.run(
        ["taskset", "-p", str(proc.pid)], capture_output=True, text=True, check=True
    ).stdout.strip()
    child_st = read_proc_status(proc.pid)
    os.wait4(proc.pid, 0)
    if parent_st.get("Cpus_allowed_list") != CORE or child_st.get("Cpus_allowed_list") != CORE:
        raise SystemExit("STOP: taskset pinning check failed")
    return {
        "core": CORE,
        "parent_taskset_p": parent_tp,
        "parent_cpus_allowed": parent_st.get("Cpus_allowed", ""),
        "parent_cpus_allowed_list": parent_st.get("Cpus_allowed_list", ""),
        "child_taskset_p": child_tp,
        "child_cpus_allowed": child_st.get("Cpus_allowed", ""),
        "child_cpus_allowed_list": child_st.get("Cpus_allowed_list", ""),
    }


def field(text: str, key: str) -> str:
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"driver printed no {key}")


def h2h_row(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            return dict(part.split("=", 1) for part in line.split()[1:])
    raise SystemExit("driver printed no h2h_row")


def seed_run(
    arm: Arm, kind: str, n: int, rho: float, budget: int, seed: int, stop: str, cell: str, work: Path
) -> dict[str, str]:
    prefix = work / "runs" / f"{cell}-s{seed}-{arm.name}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log = work / "logs" / f"seed-{cell}-s{seed}-{arm.name}.log"
    argv = [str(arm.bin / "h2h_seed"), kind, str(n), repr(rho) if kind == "morse" else "0", str(budget)]
    argv += [str(seed), "recommended", stop, str(prefix)]
    text, rss_anon_kb, ru_maxrss_kb, _pin = execute(argv, log, arm.worktree)
    row = h2h_row(text)
    wall = float(field(text, "wall_seconds"))
    pot = float(field(text, "pot_s"))
    start = Path(str(prefix) + ".start.gmin")
    out = {
        "driver": "h2h_seed",
        "hit": row["hit"],
        "evals_to_hit": row["evals_to_hit"],
        "calls": field(text, "calls"),
        "charged": field(text, "charged"),
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{wall - pot:.6f}",
        "best": row["best"],
        "search_best": field(text, "best"),
        "lowest": field(text, "lowest"),
        "geometry_energy": field(text, "geometry_energy"),
        "hops": field(text, "hops"),
        "basins": field(text, "basins"),
        "start_sha": sha256_file(start) if start.is_file() else "",
        "trace_sha": sha256_file(Path(str(prefix) + ".trace.csv")),
        "geometry_sha": sha256_file(Path(str(prefix) + ".xyz")),
        "rss_anon_kb": str(rss_anon_kb),
        "ru_maxrss_kb": str(ru_maxrss_kb),
    }
    for suffix in (".trace.csv", ".xyz", ".start.gmin"):
        Path(str(prefix) + suffix).unlink(missing_ok=True)
    return out


def panel_run(arm: Arm, panel, system: str, seed: int, work: Path) -> dict[str, str]:
    _name, _kind, _n, _reference, _rho, _driver, binary = panel.spec(system)
    if system == "lj103":
        binary = panel.resolve_driver(arm.bin, arm.worktree, system)[1]
    coords = work / "runs" / f"{system}-s{seed}-{arm.name}.start"
    best = work / "runs" / f"{system}-s{seed}-{arm.name}.best"
    coords.parent.mkdir(parents=True, exist_ok=True)
    best.unlink(missing_ok=True)
    log = work / "logs" / f"panel-{system}-s{seed}-{arm.name}.log"
    extra = {"H2H_BEST": str(best)} if binary == "h2h_metal" else None
    argv = [str(arm.bin / binary), "run", system, str(seed)]
    if binary == "h2h_morse":
        coords_log = work / "logs" / f"coords-{system}-s{seed}-{arm.name}.log"
        execute([str(arm.bin / binary), "coords", system, str(seed), str(coords)], coords_log, arm.worktree)
    else:
        argv.append(str(coords))
    text, rss_anon_kb, ru_maxrss_kb, _pin = execute(argv, log, arm.worktree, extra)
    budget = panel.PAPER_BUDGETS[system]
    row = panel.anneal_result(text, system, seed, budget)
    if row["head"] and row["head"] != arm.commit:
        raise SystemExit(f"{system} {seed} {arm.name}: driver head {row['head']} is not {arm.commit}")
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    out = {
        "driver": binary,
        "hit": row["hit"],
        "evals_to_hit": row["evals_to_hit"],
        "calls": row["charged"],
        "charged": row["charged"],
        "wall_s": row["wall_s"],
        "pot_s": row["pot_s"],
        "outside_s": f"{wall - pot:.6f}",
        "best": row["best"],
        "start_sha": sha256_file(coords),
        "geometry_sha": sha256_file(best) if best.is_file() else "",
        "rss_anon_kb": str(rss_anon_kb),
        "ru_maxrss_kb": str(ru_maxrss_kb),
    }
    coords.unlink(missing_ok=True)
    best.unlink(missing_ok=True)
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def rigid_jobs() -> list[tuple[int, int, int, int, str]]:
    jobs = []
    for n in SIZES:
        for block, seed in enumerate(SEEDS):
            for slot, arm in enumerate(rigid_order(seed), start=1):
                jobs.append((n, seed, block, slot, arm))
    return jobs


def nonrigid_jobs(panel) -> list[tuple[str, int, int, int, str]]:
    jobs = []
    for system in [row[0] for row in panel.SYSTEMS]:
        for block, seed in enumerate(NONRIGID_SEEDS):
            for slot, arm in enumerate(nonrigid_order(seed), start=1):
                jobs.append((system, seed, block, slot, arm))
    return jobs


def base_row(campaign: str, cell: str, seed: int, block: int, slot: int, arm: Arm, budget: int, ref: str) -> dict:
    seed_set = "primary" if seed in PRIMARY_SEEDS else "second" if seed in SECOND_SEEDS else ""
    return {
        "campaign": campaign,
        "seed_set": "" if campaign == "nonrigid" else seed_set,
        "label": LABELS[arm.name],
        "cell": cell,
        "seed": str(seed),
        "arm": arm.name,
        "round": str(block),
        "slot": str(slot),
        "commit": arm.commit,
        "tree": arm.tree,
        "machine_id": machine_id(),
        "core": CORE,
        "budget": str(budget),
        "reference": ref,
    }


def check_starts(rows: list[dict[str, str]]) -> None:
    seen: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (row["driver"], row["cell"], row["seed"])
        if key in seen and seen[key] != row["start_sha"]:
            raise SystemExit(f"STOP: starts differ for {key}")
        seen.setdefault(key, row["start_sha"])


def done_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["driver"], row["cell"], row["seed"], row["arm"])


def update_session_file(session_path: Path | None, arms: dict[str, Arm], campaign: str) -> None:
    if session_path is None:
        return
    session_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {}
    if session_path.is_file():
        data = json.loads(session_path.read_text())
    data["machine_id"] = machine_id()
    data["core"] = CORE
    data["pinning"] = capture_pinning_evidence()
    data["thread_env"] = thread_env_dict()
    arm_meta = data.get("arms", {})
    if not isinstance(arm_meta, dict):
        arm_meta = {}
    for name, arm in sorted(arms.items()):
        arm_meta[name] = arm.metadata()
    data["arms"] = arm_meta
    campaigns = data.get("campaigns_run", [])
    if not isinstance(campaigns, list):
        campaigns = []
    if campaign not in campaigns:
        campaigns.append(campaign)
    data["campaigns_run"] = campaigns
    session_path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def cmd_run(args: argparse.Namespace) -> int:
    arms = {}
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        arms[name] = Arm(name, rest)
    update_session_file(args.session, arms, args.campaign)
    journal = args.journal
    rows = read_rows(journal)
    done = {done_key(row) for row in rows}
    refs = ccd_references()
    if args.campaign == "rigid":
        if set(arms) != {"A", "A2", "B", "C"}:
            raise SystemExit("rigid campaign needs arms A, A2, B and C")
        for n, seed, block, slot, name in rigid_jobs():
            cell = f"tip4p{n}"
            if ("h2h_seed", cell, str(seed), name) in done:
                continue
            ref = refs[n]
            stop = repr(ref + DEFAULT_HIT_TOL)
            row = base_row("rigid", cell, seed, block, slot, arms[name], BUDGET, f"{ref:.10f}")
            row.update(seed_run(arms[name], "tip4p", n, 0.0, BUDGET, seed, stop, cell, args.work))
            if energy_hits(float(row["best"]), ref, DEFAULT_HIT_TOL) != (row["hit"] == "1"):
                raise SystemExit(f"hit flag disagrees for {cell} {seed} {name}")
            append_row(journal, row)
            rows.append(row)
            check_starts(rows)
            print(
                f"done rigid {cell} s{seed} {name} hit {row['hit']} calls {row['calls']} "
                f"wall {row['wall_s']} rss_anon {row['rss_anon_kb']} maxrss {row['ru_maxrss_kb']}",
                flush=True,
            )
    elif args.campaign == "nonrigid":
        if set(arms) != {"A", "B", "C"}:
            raise SystemExit("nonrigid campaign needs arms A, B and C")
        panel = load_panel()
        for system, seed, block, slot, name in nonrigid_jobs(panel):
            _s, kind, n, reference, rho, _driver, _binary = panel.spec(system)
            budget = panel.PAPER_BUDGETS[system]
            arm = arms[name]
            driver = panel.resolve_driver(arm.bin, arm.worktree, system)[1] if system == "lj103" else _binary
            if (driver, system, str(seed), name) not in done:
                row = base_row("nonrigid", system, seed, block, slot, arm, budget, f"{reference:.10f}")
                row.update(panel_run(arm, panel, system, seed, args.work))
                append_row(journal, row)
                rows.append(row)
                check_starts(rows)
                print(
                    f"done panel {system} s{seed} {name} best {row['best']} charged {row['charged']} "
                    f"rss_anon {row['rss_anon_kb']} maxrss {row['ru_maxrss_kb']}",
                    flush=True,
                )
            if kind in ("lj", "morse") and ("h2h_seed", system, str(seed), name) not in done:
                stop = repr(reference + DEFAULT_HIT_TOL)
                row = base_row("nonrigid", system, seed, block, slot, arm, budget, f"{reference:.10f}")
                row.update(seed_run(arm, kind, n, rho, budget, seed, stop, system, args.work))
                append_row(journal, row)
                rows.append(row)
                check_starts(rows)
                print(
                    f"done seed {system} s{seed} {name} best {row['best']} calls {row['calls']} "
                    f"rss_anon {row['rss_anon_kb']} maxrss {row['ru_maxrss_kb']}",
                    flush=True,
                )
    else:
        raise SystemExit(f"unknown campaign {args.campaign}")
    print("CAMPAIGN_OK", flush=True)
    return 0


def opc(row: dict[str, str]) -> float:
    return (float(row["wall_s"]) - float(row["pot_s"])) / float(row["charged"])


def pooled_opc(rows: list[dict[str, str]]) -> float:
    return sum(float(r["wall_s"]) - float(r["pot_s"]) for r in rows) / sum(float(r["charged"]) for r in rows)


def pick(rows: list[dict[str, str]], cell: str, arm: str, driver: str = "h2h_seed") -> list[dict[str, str]]:
    found = [r for r in rows if r["cell"] == cell and r["arm"] == arm and r["driver"] == driver]
    return sorted(found, key=lambda r: int(r["seed"]))


def log_name(row: dict[str, str]) -> str:
    if row["driver"] == "h2h_seed":
        return f"seed-{row['cell']}-s{row['seed']}-{row['arm']}.log"
    return f"panel-{row['cell']}-s{row['seed']}-{row['arm']}.log"


def mark_pauses(rows: list[dict[str, str]], work: Path) -> None:
    previous = None
    for row in rows:
        end = (work / "logs" / log_name(row)).stat().st_mtime
        row["pause_gap_s"] = "" if previous is None else f"{end - previous - float(row['wall_s']):.3f}"
        previous = end


def paused(row: dict[str, str]) -> bool:
    return row.get("pause_gap_s", "") != "" and float(row["pause_gap_s"]) > PAUSE_GAP_S


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def hit_string(rows: list[dict[str, str]]) -> str:
    return "".join("1" if r["hit"] == "1" else "0" for r in rows)


def pair_stats(ref_rows: list[dict[str, str]], cand_rows: list[dict[str, str]]) -> dict[str, object]:
    b_seeds = [r["seed"] for r, o in zip(ref_rows, cand_rows) if r["hit"] == "1" and o["hit"] == "0"]
    c_seeds = [r["seed"] for r, o in zip(ref_rows, cand_rows) if r["hit"] == "0" and o["hit"] == "1"]
    b = len(b_seeds)
    c = len(c_seeds)
    return {
        "b": b,
        "c": c,
        "p": mcnemar(b, c),
        "ref_hit_cand_miss_seeds": " ".join(b_seeds),
        "ref_miss_cand_hit_seeds": " ".join(c_seeds),
    }


def cell_entry(rows: list[dict[str, str]], cell: str, arms: tuple[str, ...], seeds: tuple[int, ...]) -> dict:
    arm_rows = {arm: pick(rows, cell, arm) for arm in arms}
    for arm, found in arm_rows.items():
        if [int(r["seed"]) for r in found] != list(seeds):
            raise SystemExit(f"{cell} arm {arm} lacks rows")
    med = {arm: type7([opc(r) for r in v]) for arm, v in arm_rows.items()}
    pooled = {arm: pooled_opc(v) for arm, v in arm_rows.items()}
    hits = {arm: sum(r["hit"] == "1" for r in v) for arm, v in arm_rows.items()}
    hits_primary = {
        arm: sum(r["hit"] == "1" for r in v if int(r["seed"]) in PRIMARY_SEEDS) for arm, v in arm_rows.items()
    }
    hits_second = {
        arm: sum(r["hit"] == "1" for r in v if int(r["seed"]) in SECOND_SEEDS) for arm, v in arm_rows.items()
    }
    ratio_aa = med["A2"] / med["A"]
    cost_cap = max(COST_LIMIT, ratio_aa)
    ba_pair = pair_stats(arm_rows["A"], arm_rows["B"])
    ca_pair = pair_stats(arm_rows["A"], arm_rows["C"])
    bc_pair = pair_stats(arm_rows["B"], arm_rows["C"])
    return {
        "cell": cell,
        "n": len(seeds),
        "hits": hits,
        "hits_primary_2000_2011": hits_primary,
        "hits_second_2012_2023": hits_second,
        "hit_table": {arm: hit_string(v) for arm, v in arm_rows.items()},
        "opc_us": {arm: med[arm] * 1e6 for arm in arms},
        "pooled_opc_us": {arm: pooled[arm] * 1e6 for arm in arms},
        "ratio_to_a": {arm: med[arm] / med["A"] for arm in arms if arm != "A"},
        "pooled_ratio_to_a": {arm: pooled[arm] / pooled["A"] for arm in arms if arm != "A"},
        "ratio_c_to_b": med["C"] / med["B"],
        "ratio_b_to_c": med["B"] / med["C"],
        "pooled_ratio_c_to_b": pooled["C"] / pooled["B"],
        "pooled_ratio_b_to_c": pooled["B"] / pooled["C"],
        "cost_cap": cost_cap,
        "aa_noisy": int(not (1.0 / COST_LIMIT <= ratio_aa <= COST_LIMIT)),
        "median_charged": {arm: type7([float(r["charged"]) for r in v]) for arm, v in arm_rows.items()},
        "peak_rss_anon_kb": {arm: max(int(r["rss_anon_kb"]) for r in v) for arm, v in arm_rows.items()},
        "median_rss_anon_kb": {arm: type7([float(r["rss_anon_kb"]) for r in v]) for arm, v in arm_rows.items()},
        "peak_ru_maxrss_kb": {arm: max(int(r["ru_maxrss_kb"]) for r in v) for arm, v in arm_rows.items()},
        "median_ru_maxrss_kb": {arm: type7([float(r["ru_maxrss_kb"]) for r in v]) for arm, v in arm_rows.items()},
        "paused_rows": {arm: sum(paused(r) for r in v) for arm, v in arm_rows.items()},
        "b_vs_a": {
            **ba_pair,
            "flag_fewer_hits": int(hits["B"] < hits["A"]),
            "fail_cost": int(med["B"] / med["A"] > cost_cap),
        },
        "c_vs_a": {
            **ca_pair,
            "flag_fewer_hits": int(hits["C"] < hits["A"]),
            "fail_cost": int(med["C"] / med["A"] > cost_cap),
        },
        "c_vs_b": bc_pair,
        "rows_a_eq_a2": sum(
            all(a[k] == a2[k] for k in IDENTITY) for a, a2 in zip(arm_rows["A"], arm_rows["A2"])
        ),
        "rows_b_eq_c": sum(
            all(b[k] == c[k] for k in IDENTITY) for b, c in zip(arm_rows["B"], arm_rows["C"])
        ),
        "rows_a_eq_b": sum(
            all(a[k] == b[k] for k in IDENTITY) for a, b in zip(arm_rows["A"], arm_rows["B"])
        ),
        "rows_a_eq_c": sum(
            all(a[k] == c[k] for k in IDENTITY) for a, c in zip(arm_rows["A"], arm_rows["C"])
        ),
    }


def size_of(cell: str) -> int:
    return int(cell[len("tip4p") :])


def pooled_mcnemar_window(
    rigid: list[dict[str, str]], sizes: tuple[int, ...], seeds: tuple[int, ...], ref_arm: str, cand_arm: str
) -> dict[str, object]:
    seed_set = set(seeds)
    pairs = []
    for n in sizes:
        cell = f"tip4p{n}"
        ref_rows = [r for r in pick(rigid, cell, ref_arm) if int(r["seed"]) in seed_set]
        cand_rows = [r for r in pick(rigid, cell, cand_arm) if int(r["seed"]) in seed_set]
        if len(ref_rows) != len(seeds) or len(cand_rows) != len(seeds):
            raise SystemExit(f"{cell} pooled pairs missing for {ref_arm} vs {cand_arm}")
        pairs.extend(zip(ref_rows, cand_rows))
    b = sum(a["hit"] == "1" and o["hit"] == "0" for a, o in pairs)
    c = sum(a["hit"] == "0" and o["hit"] == "1" for a, o in pairs)
    p = mcnemar(b, c)
    return {
        "n_pairs": len(pairs),
        "hits_ref": sum(a["hit"] == "1" for a, _ in pairs),
        "hits_cand": sum(o["hit"] == "1" for _, o in pairs),
        "b": b,
        "c": c,
        "p": p,
        "fail_mcnemar_005": bool(b > c and p < 0.05),
    }


def rss_arm_summary(rigid: list[dict[str, str]], nonrigid: list[dict[str, str]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for arm in ("A", "A2", "B", "C"):
        r_rows = [r for r in rigid if r["arm"] == arm]
        nr_rows = [r for r in nonrigid if r["arm"] == arm]
        all_rows = r_rows + nr_rows
        if not all_rows:
            continue
        out[arm] = {
            "rigid_peak_rss_anon_kb": max(int(r["rss_anon_kb"]) for r in r_rows) if r_rows else 0,
            "rigid_median_rss_anon_kb": type7([float(r["rss_anon_kb"]) for r in r_rows]) if r_rows else 0.0,
            "rigid_peak_ru_maxrss_kb": max(int(r["ru_maxrss_kb"]) for r in r_rows) if r_rows else 0,
            "rigid_median_ru_maxrss_kb": type7([float(r["ru_maxrss_kb"]) for r in r_rows]) if r_rows else 0.0,
            "nonrigid_peak_rss_anon_kb": max(int(r["rss_anon_kb"]) for r in nr_rows) if nr_rows else 0,
            "nonrigid_peak_ru_maxrss_kb": max(int(r["ru_maxrss_kb"]) for r in nr_rows) if nr_rows else 0,
            "overall_peak_rss_anon_kb": max(int(r["rss_anon_kb"]) for r in all_rows),
            "overall_peak_ru_maxrss_kb": max(int(r["ru_maxrss_kb"]) for r in all_rows),
        }
    return out


def cmd_score(args: argparse.Namespace) -> int:
    rigid = read_rows(args.rigid)
    nonrigid = read_rows(args.nonrigid)
    for rows in (rigid, nonrigid):
        mark_pauses(rows, args.work)
        check_starts(rows)
    bad = [r for r in rigid if r["calls"] != r["charged"]]
    if bad:
        raise SystemExit(f"{len(bad)} TIP4P rows have calls != charged")

    cells = [cell_entry(rigid, f"tip4p{n}", ("A", "A2", "B", "C"), SEEDS) for n in SIZES]
    nonrigid_report = []
    for system, driver in sorted({(r["cell"], r["driver"]) for r in nonrigid}):
        a_rows = pick(nonrigid, system, "A", driver)
        b_rows = pick(nonrigid, system, "B", driver)
        c_rows = pick(nonrigid, system, "C", driver)
        if any(len(v) != len(NONRIGID_SEEDS) for v in (a_rows, b_rows, c_rows)):
            raise SystemExit(f"{system} {driver} lacks rows")
        diffs_ab = []
        diffs_ac = []
        diffs_bc = []
        for a, b, c in zip(a_rows, b_rows, c_rows):
            diffs_ab += [f"{a['seed']}:{k}" for k in NONRIGID_IDENTITY if a.get(k, "") != b.get(k, "")]
            diffs_ac += [f"{a['seed']}:{k}" for k in NONRIGID_IDENTITY if a.get(k, "") != c.get(k, "")]
            diffs_bc += [f"{b['seed']}:{k}" for k in NONRIGID_IDENTITY if b.get(k, "") != c.get(k, "")]
        nonrigid_report.append(
            {
                "cell": system,
                "driver": driver,
                "seeds": len(a_rows),
                "best_a": " ".join(r["best"] for r in a_rows),
                "charged_a": " ".join(r["charged"] for r in a_rows),
                "identical_ab": int(not diffs_ab),
                "identical_ac": int(not diffs_ac),
                "identical_bc": int(not diffs_bc),
                "diffs_ab": " ".join(diffs_ab),
                "diffs_ac": " ".join(diffs_ac),
                "diffs_bc": " ".join(diffs_bc),
            }
        )

    pooled = {
        "tip4p15_21_all_24": {
            "b_vs_a": pooled_mcnemar_window(rigid, GATED_WINDOW, SEEDS, "A", "B"),
            "c_vs_a": pooled_mcnemar_window(rigid, GATED_WINDOW, SEEDS, "A", "C"),
            "c_vs_b": pooled_mcnemar_window(rigid, GATED_WINDOW, SEEDS, "B", "C"),
            "a2_vs_a": pooled_mcnemar_window(rigid, GATED_WINDOW, SEEDS, "A", "A2"),
        },
        "tip4p15_21_primary_2000_2011": {
            "b_vs_a": pooled_mcnemar_window(rigid, GATED_WINDOW, PRIMARY_SEEDS, "A", "B"),
            "c_vs_a": pooled_mcnemar_window(rigid, GATED_WINDOW, PRIMARY_SEEDS, "A", "C"),
            "c_vs_b": pooled_mcnemar_window(rigid, GATED_WINDOW, PRIMARY_SEEDS, "B", "C"),
        },
        "tip4p15_21_second_2012_2023": {
            "b_vs_a": pooled_mcnemar_window(rigid, GATED_WINDOW, SECOND_SEEDS, "A", "B"),
            "c_vs_a": pooled_mcnemar_window(rigid, GATED_WINDOW, SECOND_SEEDS, "A", "C"),
            "c_vs_b": pooled_mcnemar_window(rigid, GATED_WINDOW, SECOND_SEEDS, "B", "C"),
        },
        "tip4p9_21_all_24": {
            "b_vs_a": pooled_mcnemar_window(rigid, SIZES, SEEDS, "A", "B"),
            "c_vs_a": pooled_mcnemar_window(rigid, SIZES, SEEDS, "A", "C"),
            "c_vs_b": pooled_mcnemar_window(rigid, SIZES, SEEDS, "B", "C"),
            "a2_vs_a": pooled_mcnemar_window(rigid, SIZES, SEEDS, "A", "A2"),
        },
    }

    def arm_verdict(cand: str, key: str, nr_key: str) -> dict[str, object]:
        flagged_cells = [c["cell"] for c in cells if c[key]["flag_fewer_hits"]]
        cost_failed_cells = [c["cell"] for c in cells if c[key]["fail_cost"]]
        nr_failed = [r["cell"] + ":" + r["driver"] for r in nonrigid_report if not r[nr_key]]
        mcnemar_15_21 = pooled["tip4p15_21_all_24"][key]
        rule1_mcnemar_pass = not mcnemar_15_21["fail_mcnemar_005"]
        rule1_no_cell_flags = len(flagged_cells) == 0
        rule1_pass = rule1_mcnemar_pass and rule1_no_cell_flags
        rule2_pass = len(cost_failed_cells) == 0
        rule3_pass = bool(nonrigid_report) and len(nr_failed) == 0
        return {
            "candidate": cand,
            "flagged_hit_loss_cells": flagged_cells,
            "cost_failed_cells": cost_failed_cells,
            "nonrigid_failed_cells": nr_failed,
            "rule1_pooled_mcnemar_pass": rule1_mcnemar_pass,
            "rule1_no_flagged_cells": rule1_no_cell_flags,
            "rule1_pass": rule1_pass,
            "rule2_pass": rule2_pass,
            "rule3_pass": rule3_pass,
            "pass_pooled_rule1": rule1_mcnemar_pass and rule2_pass and rule3_pass,
            "pass_strict": rule1_pass and rule2_pass and rule3_pass,
        }

    a_rows = [r for r in rigid if r["arm"] == "A"]
    a2_rows = [r for r in rigid if r["arm"] == "A2"]
    b_rows = [r for r in rigid if r["arm"] == "B"]
    c_rows = [r for r in rigid if r["arm"] == "C"]
    a_15_21 = [r for r in a_rows if size_of(r["cell"]) in GATED_WINDOW]
    a2_15_21 = [r for r in a2_rows if size_of(r["cell"]) in GATED_WINDOW]
    b_15_21 = [r for r in b_rows if size_of(r["cell"]) in GATED_WINDOW]
    c_15_21 = [r for r in c_rows if size_of(r["cell"]) in GATED_WINDOW]

    rss_summary = rss_arm_summary(rigid, nonrigid)
    report = {
        "cells": cells,
        "pooled_mcnemar": pooled,
        "nonrigid": nonrigid_report,
        "verdicts": {
            "b_vs_a": arm_verdict("B", "b_vs_a", "identical_ab"),
            "c_vs_a": arm_verdict("C", "c_vs_a", "identical_ac"),
        },
        "comparisons": {
            "a2_to_a_ratio_range_9_21": [
                min(c["ratio_to_a"]["A2"] for c in cells),
                max(c["ratio_to_a"]["A2"] for c in cells),
            ],
            "b_to_a_ratio_range_9_21": [
                min(c["ratio_to_a"]["B"] for c in cells),
                max(c["ratio_to_a"]["B"] for c in cells),
            ],
            "c_to_a_ratio_range_9_21": [
                min(c["ratio_to_a"]["C"] for c in cells),
                max(c["ratio_to_a"]["C"] for c in cells),
            ],
            "b_to_a_ratio_range_15_21": [
                min(c["ratio_to_a"]["B"] for c in cells if size_of(c["cell"]) in GATED_WINDOW),
                max(c["ratio_to_a"]["B"] for c in cells if size_of(c["cell"]) in GATED_WINDOW),
            ],
            "c_to_a_ratio_range_15_21": [
                min(c["ratio_to_a"]["C"] for c in cells if size_of(c["cell"]) in GATED_WINDOW),
                max(c["ratio_to_a"]["C"] for c in cells if size_of(c["cell"]) in GATED_WINDOW),
            ],
            "c_to_b_ratio_range_9_21": [
                min(c["ratio_c_to_b"] for c in cells),
                max(c["ratio_c_to_b"] for c in cells),
            ],
            "a2_to_a_pooled_9_21": pooled_opc(a2_rows) / pooled_opc(a_rows),
            "b_to_a_pooled_9_21": pooled_opc(b_rows) / pooled_opc(a_rows),
            "c_to_a_pooled_9_21": pooled_opc(c_rows) / pooled_opc(a_rows),
            "c_to_b_pooled_9_21": pooled_opc(c_rows) / pooled_opc(b_rows),
            "a2_to_a_pooled_15_21": pooled_opc(a2_15_21) / pooled_opc(a_15_21),
            "b_to_a_pooled_15_21": pooled_opc(b_15_21) / pooled_opc(a_15_21),
            "c_to_a_pooled_15_21": pooled_opc(c_15_21) / pooled_opc(a_15_21),
            "c_to_b_pooled_15_21": pooled_opc(c_15_21) / pooled_opc(b_15_21),
            "rows_a_eq_a2_9_21": [sum(c["rows_a_eq_a2"] for c in cells), len(cells) * len(SEEDS)],
            "rows_b_eq_c_9_21": [sum(c["rows_b_eq_c"] for c in cells), len(cells) * len(SEEDS)],
            "rows_a_eq_b_9_21": [sum(c["rows_a_eq_b"] for c in cells), len(cells) * len(SEEDS)],
            "rows_a_eq_c_9_21": [sum(c["rows_a_eq_c"] for c in cells), len(cells) * len(SEEDS)],
            "paused_rows_rigid": [
                f"{r['cell']}:s{r['seed']}:{r['arm']}:{r['pause_gap_s']}s" for r in rigid if paused(r)
            ],
            "paused_rows_nonrigid": [
                f"{r['cell']}:{r['driver']}:s{r['seed']}:{r['arm']}:{r['pause_gap_s']}s"
                for r in nonrigid
                if paused(r)
            ],
            "total_rows": len(rigid) + len(nonrigid),
        },
        "rss": rss_summary,
    }

    public = [{k: r.get(k, "") for k in PUBLIC_FIELDS} for r in rigid + nonrigid]
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(public)
    args.summary.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.session and args.session.is_file():
        sess = json.loads(args.session.read_text())
        sess["rss"] = rss_summary
        sess["scored_rows"] = {
            "rigid": len(rigid),
            "nonrigid": len(nonrigid),
            "total": len(rigid) + len(nonrigid),
        }
        args.session.write_text(json.dumps(sess, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report["verdicts"]), flush=True)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    src = args.src.resolve()
    out = args.out.resolve()
    for sub in ("target", "tmp", "bin"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cargo_home = os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(out / "target")
    env["TMPDIR"] = str(out / "tmp")
    env["RUSTFLAGS"] = (
        f"--remap-path-prefix={src}=/src --remap-path-prefix={Path.home()}=/user "
        f"--remap-path-prefix={cargo_home}=/cargo"
    )
    command = ["cargo", "+1.88.0", "build", "--release", "--locked"]
    for binary in BINARIES:
        command += ["--example", binary]
    subprocess.run(command + ["-j", "2"], cwd=src, env=env, check=True)
    for binary in BINARIES:
        target = out / "bin" / binary
        target.write_bytes((out / "target" / "release" / "examples" / binary).read_bytes())
        target.chmod(0o755)
    (out / "commit").write_text(git(src, "rev-parse", "HEAD") + "\n")
    (out / "tree").write_text(git(src, "rev-parse", "HEAD^{tree}") + "\n")
    return 0


def self_test() -> None:
    jobs = rigid_jobs()
    if len(jobs) != len(SIZES) * len(SEEDS) * 4:
        raise SystemExit("rigid job count")
    first_four = [
        [arm for n, seed, *_rest, arm in jobs if n == 9 and seed == s]
        for s in (2000, 2001, 2002, 2003)
    ]
    if [tuple(x) for x in first_four] != list(WILLIAMS_4):
        raise SystemExit(f"rigid Williams order {first_four}")
    slot_counts = {arm: [0, 0, 0, 0] for arm in ("A", "A2", "B", "C")}
    for s in SEEDS:
        for slot_idx, arm in enumerate(rigid_order(s)):
            slot_counts[arm][slot_idx] += 1
    if any(counts != [6, 6, 6, 6] for counts in slot_counts.values()):
        raise SystemExit(f"unbalanced Williams slots {slot_counts}")
    refs = ccd_references()
    if sorted(n for n in refs if 9 <= n <= 21) != list(SIZES):
        raise SystemExit("Cambridge references")
    panel = load_panel()
    if len(panel.SYSTEMS) != 13 or len(nonrigid_jobs(panel)) != 13 * 3 * 3:
        raise SystemExit("panel size")
    pin = capture_pinning_evidence()
    if pin["child_cpus_allowed_list"] != CORE:
        raise SystemExit("pinning self-test failed")
    print("ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "build", "run", "score"))
    parser.add_argument("--campaign", choices=("rigid", "nonrigid"))
    parser.add_argument("--arm", action="append", default=[], help="NAME=BUILD_DIR:WORKTREE")
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--src", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rigid", type=Path)
    parser.add_argument("--nonrigid", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv[1:])
    if args.command == "self-test":
        self_test()
        return 0
    if args.command == "build":
        return cmd_build(args)
    if args.command == "run":
        return cmd_run(args)
    return cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
