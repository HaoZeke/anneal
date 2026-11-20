#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial gate for the TIP4P b11 series, rules in gates/anneal-tip4p-b11-gate.md.

``build`` makes one release arm. ``run --campaign rigid`` runs the TIP4P
cells on arms A, B, C and A2; ``run --campaign nonrigid`` runs the paper
panel on A and B. ``score`` reads the journals and writes the public rows
and the per-cell summary. Every run is one process on CPU 0 with every
thread pool at one thread. A journal is appended and synced after each
run, and a rerun skips the rows it already holds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_h2h  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

TREES = {
    "A": "cff796c507a8fa1d0bfb28888c3570b61988055e",
    "A2": "cff796c507a8fa1d0bfb28888c3570b61988055e",
    "B": "75f19ed42a5f3f7311cf670c6b9af15b8eeaa68f",
    "C": "14b537fd9b73b8fe21e01721954b41a278c5d223",
}
SIZES = tuple(range(8, 22)) + (25,)
SEEDS = tuple(range(2000, 2012))
SECOND_SEEDS = tuple(range(2012, 2024))
SECOND_SIZES = tuple(range(15, 22))
PROBE_SIZES = (9, 15, 17)
LABELS = {"A": "develop", "A2": "develop-2", "B": "b11", "C": "b6", "P": "probe"}
NONRIGID_SEEDS = (2000, 2001, 2002)
BUDGET = 2_000_000
CORE = "0"
COST_LIMIT = 1.10
BINARIES = ("h2h_seed", "h2h_gmin_proof", "h2h_morse", "h2h_metal")
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
    "probe",
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
]


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


def run_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_", "PRIOR_")):
            del env[key]
        elif key.endswith(("_NUM_THREADS", "_MAX_THREADS")):
            env[key] = "1"
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    if extra:
        env.update(extra)
    return env


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def type7(values: list[float]) -> float:
    return run_h2h.type7(values)


def arm_order(seed: int, arms: tuple[str, ...]) -> tuple[str, ...]:
    return arms if (seed - SEEDS[0]) % 2 == 0 else tuple(reversed(arms))


class Arm:
    def __init__(self, name: str, spec: str):
        bin_dir, _, worktree = spec.partition(":")
        self.name = name
        self.bin = Path(bin_dir) / "bin"
        self.worktree = Path(worktree)
        self.commit = (Path(bin_dir) / "commit").read_text().strip()
        self.tree = (Path(bin_dir) / "tree").read_text().strip()
        if name in TREES and self.tree != TREES[name]:
            raise SystemExit(f"STOP: arm {name} tree {self.tree} is not {TREES[name]}")
        live = git(self.worktree, "rev-parse", "HEAD^{tree}")
        if live != self.tree or git(self.worktree, "status", "--porcelain"):
            raise SystemExit(f"STOP: arm {name} worktree is not clean at {self.tree}")
        for binary in BINARIES:
            if not (self.bin / binary).is_file():
                raise SystemExit(f"arm {name} lacks {binary}")


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def execute(argv: list[str], log: Path, cwd: Path, extra: dict[str, str] | None = None) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    command = ["taskset", "-c", CORE, *argv]
    with log.open("w") as handle:
        proc = subprocess.run(
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, env=run_env(extra), check=False
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise SystemExit(f"driver exited {proc.returncode}; see {log.name}")
    return text


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
    text = execute(argv, log, arm.worktree)
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
        # Its run reads a start from this argument instead of writing one.
        execute([str(arm.bin / binary), "coords", system, str(seed), str(coords)], log, arm.worktree)
    else:
        argv.append(str(coords))
    text = execute(argv, log, arm.worktree, extra)
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
            for slot, arm in enumerate(arm_order(seed, ("A", "B", "C", "A2")), start=1):
                jobs.append((n, seed, block, slot, arm))
    return jobs


def nonrigid_jobs(panel) -> list[tuple[str, int, int, int, str]]:
    jobs = []
    for system in [row[0] for row in panel.SYSTEMS]:
        for block, seed in enumerate(NONRIGID_SEEDS):
            for slot, arm in enumerate(arm_order(seed, ("A", "B")), start=1):
                jobs.append((system, seed, block, slot, arm))
    return jobs


def base_row(campaign: str, cell: str, seed: int, block: int, slot: int, arm: Arm, budget: int, ref: str) -> dict:
    seed_set = "primary" if seed in SEEDS else "second" if seed in SECOND_SEEDS else ""
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


def cmd_run(args: argparse.Namespace) -> int:
    arms = {}
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        arms[name] = Arm(name, rest)
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
            ref = refs.get(n) if n <= 21 else None
            stop = "-" if ref is None else repr(ref + DEFAULT_HIT_TOL)
            row = base_row("rigid", cell, seed, block, slot, arms[name], BUDGET, "" if ref is None else f"{ref:.10f}")
            row.update(seed_run(arms[name], "tip4p", n, 0.0, BUDGET, seed, stop, cell, args.work))
            if ref is not None and energy_hits(float(row["best"]), ref, DEFAULT_HIT_TOL) != (row["hit"] == "1"):
                raise SystemExit(f"hit flag disagrees for {cell} {seed} {name}")
            append_row(journal, row)
            rows.append(row)
            check_starts(rows)
            print(f"done rigid {cell} s{seed} {name} hit {row['hit']} calls {row['calls']} wall {row['wall_s']}", flush=True)
    elif args.campaign in ("second", "probe"):
        if args.campaign == "second":
            names, sizes, seeds, stop_on = ("A", "B"), SECOND_SIZES, SECOND_SEEDS, False
        else:
            names, sizes, seeds, stop_on = ("A", "P"), PROBE_SIZES, SEEDS, True
        if set(arms) != set(names):
            raise SystemExit(f"{args.campaign} campaign needs arms {names}")
        probe = arms["P"].commit if "P" in arms else ""
        for n in sizes:
            ref = refs[n]
            cell = f"tip4p{n}"
            for block, seed in enumerate(seeds):
                for slot, name in enumerate(arm_order(seed, names), start=1):
                    if ("h2h_seed", cell, str(seed), name) in done:
                        continue
                    stop = repr(ref + DEFAULT_HIT_TOL) if stop_on else "-"
                    row = base_row(args.campaign, cell, seed, block, slot, arms[name], BUDGET, f"{ref:.10f}")
                    row["probe"] = probe
                    row.update(seed_run(arms[name], "tip4p", n, 0.0, BUDGET, seed, stop, cell, args.work))
                    hit = energy_hits(float(row["best"]), ref, DEFAULT_HIT_TOL)
                    if stop_on and hit != (row["hit"] == "1"):
                        raise SystemExit(f"hit flag disagrees for {cell} {seed} {name}")
                    if not stop_on:
                        row["hit"] = "1" if hit else "0"
                        row["evals_to_hit"] = ""
                    append_row(journal, row)
                    rows.append(row)
                    check_starts(rows)
                    print(f"done {args.campaign} {cell} s{seed} {name} hit {row['hit']} calls {row['calls']} wall {row['wall_s']}", flush=True)
    elif args.campaign == "nonrigid":
        if set(arms) != {"A", "B"}:
            raise SystemExit("nonrigid campaign needs arms A and B")
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
                print(f"done panel {system} s{seed} {name} best {row['best']} charged {row['charged']}", flush=True)
            if kind in ("lj", "morse") and ("h2h_seed", system, str(seed), name) not in done:
                stop = repr(reference + DEFAULT_HIT_TOL)
                row = base_row("nonrigid", system, seed, block, slot, arm, budget, f"{reference:.10f}")
                row.update(seed_run(arm, kind, n, rho, budget, seed, stop, system, args.work))
                append_row(journal, row)
                rows.append(row)
                print(f"done seed {system} s{seed} {name} best {row['best']} calls {row['calls']}", flush=True)
    else:
        raise SystemExit(f"unknown campaign {args.campaign}")
    print("CAMPAIGN_OK", flush=True)
    return 0


def opc(row: dict[str, str]) -> float:
    return (float(row["wall_s"]) - float(row["pot_s"])) / float(row["charged"])


def pooled_opc(rows: list[dict[str, str]]) -> float:
    return sum(float(r["wall_s"]) - float(r["pot_s"]) for r in rows) / sum(float(r["charged"]) for r in rows)


def score_size25(rows: list[dict[str, str]]) -> float:
    group = [row for row in rows if row["cell"] == "tip4p25"]
    reference = min(float(row["best"]) for row in group)
    for row in group:
        row["reference"] = f"{reference:.10f}"
        row["hit"] = "1" if energy_hits(float(row["best"]), reference, DEFAULT_HIT_TOL) else "0"
    return reference


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
PUBLIC_FIELDS = FIELDS + ["pause_gap_s"]
PAUSE_GAP_S = 5.0


def pick(rows: list[dict[str, str]], cell: str, arm: str, driver: str = "h2h_seed") -> list[dict[str, str]]:
    found = [r for r in rows if r["cell"] == cell and r["arm"] == arm and r["driver"] == driver]
    return sorted(found, key=lambda r: int(r["seed"]))


def log_name(row: dict[str, str]) -> str:
    if row["driver"] == "h2h_seed":
        return f"seed-{row['cell']}-s{row['seed']}-{row['arm']}.log"
    return f"panel-{row['cell']}-s{row['seed']}-{row['arm']}.log"


def mark_pauses(rows: list[dict[str, str]], work: Path) -> None:
    """Time between consecutive log ends not covered by the run's own wall.

    Runs are serial in journal order, so a gap of seconds means the host
    stopped the machine between the end of one run and the end of the
    next. The driver's monotonic clock does not count such a stop.
    """
    previous = None
    for row in rows:
        end = (work / "logs" / log_name(row)).stat().st_mtime
        row["pause_gap_s"] = "" if previous is None else f"{end - previous - float(row['wall_s']):.3f}"
        previous = end


def paused(row: dict[str, str]) -> bool:
    return row.get("pause_gap_s", "") != "" and float(row["pause_gap_s"]) > PAUSE_GAP_S


def mcnemar(b: int, c: int) -> float:
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def hit_string(rows: list[dict[str, str]]) -> str:
    return "".join("1" if r["hit"] == "1" else "0" for r in rows)


def cell_entry(rows: list[dict[str, str]], cell: str, arms: tuple[str, ...], seeds: tuple[int, ...]) -> dict:
    arm_rows = {arm: pick(rows, cell, arm) for arm in arms}
    for arm, found in arm_rows.items():
        if [int(r["seed"]) for r in found] != list(seeds):
            raise SystemExit(f"{cell} arm {arm} lacks rows")
    med = {arm: type7([opc(r) for r in v]) for arm, v in arm_rows.items()}
    pooled = {arm: pooled_opc(v) for arm, v in arm_rows.items()}
    hits = {arm: sum(r["hit"] == "1" for r in v) for arm, v in arm_rows.items()}
    a_rows, b_rows = arm_rows["A"], arm_rows[arms[1] if "B" not in arms else "B"]
    entry = {
        "cell": cell,
        "hits": hits,
        "hit_table": {arm: hit_string(v) for arm, v in arm_rows.items()},
        "opc_us": {arm: med[arm] * 1e6 for arm in arms},
        "ratio_to_a": {arm: med[arm] / med["A"] for arm in arms if arm != "A"},
        "pooled_ratio_to_a": {arm: pooled[arm] / pooled["A"] for arm in arms if arm != "A"},
        "median_charged": {arm: type7([float(r["charged"]) for r in v]) for arm, v in arm_rows.items()},
        "paused_rows": {arm: sum(paused(r) for r in v) for arm, v in arm_rows.items()},
        "seeds_a_hit_other_miss": " ".join(
            r["seed"] for r, o in zip(a_rows, b_rows) if r["hit"] == "1" and o["hit"] == "0"
        ),
        "seeds_a_miss_other_hit": " ".join(
            r["seed"] for r, o in zip(a_rows, b_rows) if r["hit"] == "0" and o["hit"] == "1"
        ),
    }
    if "A2" in arms:
        ratio_aa = entry["ratio_to_a"]["A2"]
        entry["aa_noisy"] = int(not (1.0 / COST_LIMIT <= ratio_aa <= COST_LIMIT))
    if "B" in arms:
        entry["fail_hits"] = int(hits["B"] < hits["A"])
        entry["fail_cost"] = int(entry["ratio_to_a"]["B"] > COST_LIMIT)
    if "C" in arms:
        entry["ratio_b_to_c"] = med["B"] / med["C"]
        entry["pooled_ratio_b_to_c"] = pooled["B"] / pooled["C"]
        entry["rows_b_eq_c"] = sum(
            all(b[k] == c[k] for k in IDENTITY) for b, c in zip(arm_rows["B"], arm_rows["C"])
        )
        entry["rows_a_eq_c"] = sum(
            all(a[k] == c[k] for k in IDENTITY) for a, c in zip(arm_rows["A"], arm_rows["C"])
        )
    return entry


def size_of(cell: str) -> int:
    return int(cell[len("tip4p") :])


def cmd_score(args: argparse.Namespace) -> int:
    rigid = read_rows(args.rigid)
    second = read_rows(args.second)
    nonrigid = read_rows(args.nonrigid)
    for rows in (rigid, second, nonrigid):
        mark_pauses(rows, args.work)
        check_starts(rows)
    probes = []
    for spec in args.probe:
        journal, _, work = spec.partition(":")
        rows = read_rows(Path(journal))
        mark_pauses(rows, Path(work))
        check_starts(rows)
        probes.append(rows)
    tip4p = rigid + second + [r for rows in probes for r in rows]
    bad = [r for r in tip4p if r["calls"] != r["charged"]]
    if bad:
        raise SystemExit(f"{len(bad)} TIP4P rows have calls != charged")
    report: dict = {"reference_tip4p25": score_size25(rigid)}
    report["primary"] = [cell_entry(rigid, f"tip4p{n}", ("A", "A2", "B", "C"), SEEDS) for n in SIZES]
    report["second"] = [cell_entry(second, f"tip4p{n}", ("A", "B"), SECOND_SEEDS) for n in SECOND_SIZES]
    pooled_pairs = []
    mc = []
    for n in SECOND_SIZES:
        cell = f"tip4p{n}"
        pairs = list(zip(pick(rigid, cell, "A") + pick(second, cell, "A"), pick(rigid, cell, "B") + pick(second, cell, "B")))
        if len(pairs) != 24 or any(a["seed"] != b["seed"] for a, b in pairs):
            raise SystemExit(f"{cell} pooled pairs")
        pooled_pairs += pairs
        b = sum(a["hit"] == "1" and o["hit"] == "0" for a, o in pairs)
        c = sum(a["hit"] == "0" and o["hit"] == "1" for a, o in pairs)
        mc.append({"cell": cell, "n": 24, "hits_a": sum(a["hit"] == "1" for a, _ in pairs),
                   "hits_b": sum(o["hit"] == "1" for _, o in pairs), "b": b, "c": c, "p": mcnemar(b, c)})
    b = sum(a["hit"] == "1" and o["hit"] == "0" for a, o in pooled_pairs)
    c = sum(a["hit"] == "0" and o["hit"] == "1" for a, o in pooled_pairs)
    mc.append({"cell": "tip4p15-21", "n": len(pooled_pairs), "hits_a": sum(a["hit"] == "1" for a, _ in pooled_pairs),
               "hits_b": sum(o["hit"] == "1" for _, o in pooled_pairs), "b": b, "c": c, "p": mcnemar(b, c)})
    report["mcnemar"] = mc
    nonrigid_report = []
    for system, driver in sorted({(r["cell"], r["driver"]) for r in nonrigid}):
        a_rows = pick(nonrigid, system, "A", driver)
        b_rows = pick(nonrigid, system, "B", driver)
        if len(a_rows) != len(NONRIGID_SEEDS) or len(b_rows) != len(NONRIGID_SEEDS):
            raise SystemExit(f"{system} {driver} lacks rows")
        diffs = []
        for a, o in zip(a_rows, b_rows):
            diffs += [f"{a['seed']}:{k}" for k in NONRIGID_IDENTITY if a.get(k, "") != o.get(k, "")]
        nonrigid_report.append({"cell": system, "driver": driver, "seeds": len(a_rows),
                                "best_a": " ".join(r["best"] for r in a_rows),
                                "charged_a": " ".join(r["charged"] for r in a_rows),
                                "identical": int(not diffs), "diffs": " ".join(diffs)})
    report["nonrigid"] = nonrigid_report
    cells = report["primary"] + report["second"]
    rule1 = not any(c["fail_hits"] for c in cells)
    rule2 = not any(c["fail_cost"] for c in cells)
    rule3 = bool(nonrigid_report) and all(r["identical"] for r in nonrigid_report)
    report["rules"] = {"hits": rule1, "outside": rule2, "nonrigid": rule3, "pass": rule1 and rule2 and rule3}
    by = {c["cell"]: c for c in report["primary"]}
    window = [by[f"tip4p{n}"] for n in SECOND_SIZES]
    report["b6"] = {
        "ratio_tip4p9": by["tip4p9"]["ratio_to_a"]["C"],
        "ratio_tip4p15": by["tip4p15"]["ratio_to_a"]["C"],
        "hits_tip4p17": [by["tip4p17"]["hits"]["A"], by["tip4p17"]["hits"]["C"]],
        "hits_15_21": {arm: sum(c["hits"][arm] for c in window) for arm in ("A", "A2", "B", "C")},
    }
    report["b6"]["bad"] = bool(
        report["b6"]["ratio_tip4p9"] > COST_LIMIT
        or report["b6"]["ratio_tip4p15"] > COST_LIMIT
        or by["tip4p17"]["hits"]["C"] < by["tip4p17"]["hits"]["A"]
        or report["b6"]["hits_15_21"]["C"] < report["b6"]["hits_15_21"]["A"]
    )
    probe_report = []
    for rows in probes:
        entries = {n: cell_entry(rows, f"tip4p{n}", ("A", "P"), SEEDS) for n in PROBE_SIZES}
        commit = rows[0]["probe"]
        verdict = (
            entries[9]["ratio_to_a"]["P"] > COST_LIMIT
            or entries[15]["ratio_to_a"]["P"] > COST_LIMIT
            or entries[17]["hits"]["P"] < entries[17]["hits"]["A"]
        )
        probe_report.append({"commit": commit, "tree": [r for r in rows if r["arm"] == "P"][0]["tree"],
                             "ratio_tip4p9": entries[9]["ratio_to_a"]["P"],
                             "ratio_tip4p15": entries[15]["ratio_to_a"]["P"],
                             "hits_tip4p17": [entries[17]["hits"]["A"], entries[17]["hits"]["P"]],
                             "hits_tip4p9": [entries[9]["hits"]["A"], entries[9]["hits"]["P"]],
                             "hits_tip4p15": [entries[15]["hits"]["A"], entries[15]["hits"]["P"]],
                             "paused_rows": sum(paused(r) for r in rows),
                             "bad": bool(verdict)})
    report["probes"] = probe_report
    sub = [c for c in report["primary"] if 9 <= size_of(c["cell"]) <= 21]
    b_rows = [r for r in rigid if r["arm"] == "B" and size_of(r["cell"]) <= 21]
    c_rows = [r for r in rigid if r["arm"] == "C" and size_of(r["cell"]) <= 21]
    a_rows = [r for r in rigid if r["arm"] == "A"]
    report["claims"] = {
        "b_to_c_ratio_range_8_21": [
            min(c["ratio_b_to_c"] for c in report["primary"] if size_of(c["cell"]) <= 21),
            max(c["ratio_b_to_c"] for c in report["primary"] if size_of(c["cell"]) <= 21),
        ],
        "b_to_c_pooled_8_21": pooled_opc(b_rows) / pooled_opc(c_rows),
        "b_to_a_pooled_all": pooled_opc([r for r in rigid if r["arm"] == "B"]) / pooled_opc(a_rows),
        "a2_to_a_pooled_all": pooled_opc([r for r in rigid if r["arm"] == "A2"]) / pooled_opc(a_rows),
        "c_to_a_pooled_all": pooled_opc([r for r in rigid if r["arm"] == "C"]) / pooled_opc(a_rows),
        "hits_equal_b_c_all_cells": all(c["hits"]["B"] == c["hits"]["C"] for c in report["primary"]),
        "rows_b_eq_c_9_21": [sum(c["rows_b_eq_c"] for c in sub), len(sub) * len(SEEDS)],
        "rows_a_eq_c_9_21": [sum(c["rows_a_eq_c"] for c in sub), len(sub) * len(SEEDS)],
        "paused_rows": sum(paused(r) for r in rigid + second + nonrigid),
        "rows": len(rigid) + len(second) + len(nonrigid) + sum(len(p) for p in probes),
    }
    public = [{k: r.get(k, "") for k in PUBLIC_FIELDS} for r in rigid + second + nonrigid + [r for p in probes for r in p]]
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(public)
    args.summary.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report["rules"]), flush=True)
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
    first = [arm for n, seed, *_rest, arm in jobs if n == 8 and seed in (2000, 2001)]
    if first != ["A", "B", "C", "A2", "A2", "C", "B", "A"]:
        raise SystemExit(f"rigid order {first}")
    refs = ccd_references()
    if sorted(n for n in refs if 8 <= n <= 21) != list(range(8, 22)):
        raise SystemExit("Cambridge references")
    panel = load_panel()
    if len(panel.SYSTEMS) != 13 or len(nonrigid_jobs(panel)) != 13 * 3 * 2:
        raise SystemExit("panel size")
    print("ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "build", "run", "score"))
    parser.add_argument("--campaign", choices=("rigid", "second", "probe", "nonrigid"))
    parser.add_argument("--arm", action="append", default=[], help="NAME=BUILD_DIR:WORKTREE")
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--src", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rigid", type=Path)
    parser.add_argument("--nonrigid", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--probe", action="append", default=[], help="JOURNAL:WORK")
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
