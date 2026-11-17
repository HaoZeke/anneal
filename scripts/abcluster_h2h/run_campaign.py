#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Anneal against ABCluster 3.4 at the paper call budgets.

Systems are the Cambridge Lennard-Jones minima LJ38, LJ55, LJ75, LJ98 and
LJ104, and Gupta Au55 with the Cleri-Rosato gold parameters. ABCluster runs
its own ``atom`` program with its built-in ``LennardJones`` and ``Gupta``
force fields, unchanged, under ``abc_shim.c``, which counts every call of
the force-field kernel, times it, seeds the program from the run seed and
stops the run at the first quench at or below the Cambridge energy plus
0.001 or at the budget. Anneal runs ``h2h_abcluster`` with
``Config::recommended`` and with ``Config::for_cluster``. The budget is
counted in energy-and-gradient calls, the ``PAPER_BUDGETS`` of the Lennard-Jones
paper map: 400000 at 38 atoms, 1000000 at 55, 4000000 at 75, 98 and 104.

A hit is a quenched energy within 0.001 of the Cambridge minimum. A miss is
stored at the full budget for the calls to first hit. Twelve seeds, 2000 to
2011, the same list on every arm, one process pinned to core 0. Systems run in the
listed order; inside a system the (seed, arm) pairs are shuffled. ABCluster's
population size is chosen per system on the pilot seeds 1000 to 1002, which
the campaign never uses.

Rows are append-only. A (system, arm, seed) already in ``seeds.csv`` is
skipped, so an interrupted run resumes. Before any timed run the ABCluster
kernel is checked against anneal on the same geometries and the shim's call
count against a debugger's breakpoint count. This process refuses to run
unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import math
import os
import queue
import random
import re
import subprocess
import sys
import time
from pathlib import Path

PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj104": 4_000_000,
    "au55": 1_000_000,
}
REFERENCES = {
    "lj38": -173.928427,
    "lj55": -279.248470,
    "lj75": -397.492331,
    "lj98": -543.665361,
    "lj104": -582.086642,
    "au55": -191.03085280,
}
SIZES = {"lj38": 38, "lj55": 55, "lj75": 75, "lj98": 98, "lj104": 104, "au55": 55}
KERNELS = {"lj38": "lj", "lj55": "lj", "lj75": "lj", "lj98": "lj", "lj104": "lj", "au55": "gupta"}
ELEMENTS = {"lj38": "C", "lj55": "C", "lj75": "C", "lj98": "C", "lj104": "C", "au55": "Au"}
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011)
PILOT_SEEDS = (1000, 1001, 1002)
PARITY_SEEDS = (2000, 2001, 2002)
ARMS = ("abcluster", "anneal-recommended", "anneal-cluster")
PRESETS = {"anneal-recommended": "recommended", "anneal-cluster": "cluster"}
SCHEDULE_SEED = 20261117
HIT_TOL = 0.001
PARITY_TOL = 1e-8

# ABCluster inputs. The Lennard-Jones length is the 1.4 of the manual's own
# LJ38 example and the energy unit is 1, so energies are reduced units.
# The amplitude is the manual's 5.0 for 38 atoms, grown with N^(1/3); the
# gold amplitude keeps the same size in units of the nearest-neighbour
# distance (2.884 against 1.4 * 2^(1/6)). The scout limit is the manual's 5.
# The generation limit is never reached; the budget stops the run.
LJ_SIGMA = 1.4
GOLD_PARAMETERS = (0.2061, 1.790, 2.884, 10.229, 4.036)
ABC_AMPLITUDE = {
    "lj38": 5.00,
    "lj55": 5.66,
    "lj75": 6.27,
    "lj98": 6.86,
    "lj104": 6.99,
    "au55": 10.40,
}
ABC_GLIMIT = 5
ABC_GMAX = 100_000_000
ABC_SAVES = 30
# Population size, chosen on the pilot seeds by ``choose_population``.
ABC_SN = {
    "lj38": 30,
    "lj55": 30,
    "lj75": 30,
    "lj98": 30,
    "lj104": 30,
    "au55": 30,
}
PILOT_SN = (30, 100)
PILOT_CORES = (1, 2, 3, 4, 5, 6)

# Offsets of the hooked functions in the ABCluster 3.4 Linux ``atom``,
# the same constants as abc_shim.c, for the debugger cross-check.
KERNEL_OFFSET = {"lj": 0x2CCC6, "gupta": 0x38CB4}
QUENCH_OFFSET = 0x2D676
PIE_BASE = 0x555555554000

SEED_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "calls_to_hit",
    "time_to_hit_s",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "best",
    "stop",
    "process_s",
    "setting",
    "code_revision",
    "anneal_head",
    "host",
    "cpu",
]
PILOT_FIELDS = [
    "system",
    "sn",
    "seed",
    "budget",
    "hit",
    "calls_to_hit",
    "charged",
    "wall_s",
    "best",
    "stop",
    "code_revision",
    "host",
]
PARITY_FIELDS = [
    "system",
    "seed",
    "atoms",
    "abcluster",
    "anneal",
    "abs_diff",
    "rel_diff",
    "tolerance",
    "parameters",
    "code_revision",
    "host",
]
COUNTER_FIELDS = [
    "system",
    "seed",
    "generations",
    "shim_calls",
    "gdb_calls",
    "shim_quenches",
    "gdb_quenches",
    "match",
    "code_revision",
    "host",
]


def type7(sorted_values: list[float], p: float) -> float:
    """Hyndman-Fan type 7. ``sorted_values`` is ascending."""
    n = len(sorted_values)
    if n < 1:
        raise ValueError("type 7 needs a sample")
    if n == 1:
        return sorted_values[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def schedule(systems, seeds, arms, rng_seed: int) -> list[tuple[str, int, str]]:
    """Systems in the given order. Pairs inside a system are shuffled."""
    rng = random.Random(rng_seed)
    order: list[tuple[str, int, str]] = []
    for system in systems:
        jobs = [(system, seed, arm) for seed in seeds for arm in arms]
        rng.shuffle(jobs)
        order.extend(jobs)
    return order


def choose_population(rows: list[dict[str, str]], system: str) -> int:
    """The pilot population with the most hits, then the fewest calls, then the smallest."""
    best = None
    for sn in PILOT_SN:
        group = [r for r in rows if r["system"] == system and int(r["sn"]) == sn]
        if not group:
            continue
        hits = sum(int(r["hit"]) for r in group)
        calls = median([float(r["calls_to_hit"]) for r in group])
        key = (-hits, calls, sn)
        if best is None or key < best[0]:
            best = (key, sn)
    if best is None:
        raise SystemExit(f"no pilot rows for {system}")
    return best[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Append under the existing header. Never rewrites a stored value."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle))
        if header != fields:
            raise RuntimeError(f"{path} header {header} != {fields}")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def load_pins(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    for key in ("version", "url", "tarball_sha256", "atom_sha256", "commit"):
        if not fields.get(key):
            raise SystemExit(f"{path} has no {key}")
    return fields


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def abc_revision(pins: dict[str, str]) -> str:
    return f"abcluster-{pins['version']}@{pins['commit'][:12]}"


def child_env(budget: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if budget is not None:
        env["H2H_BUDGET"] = str(budget)
    for name in (
        "OMP_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[name] = "1"
    return env


def abc_input(system: str, sn: int, initial: str | None, sigma: float = LJ_SIGMA) -> tuple[str, str]:
    """The ``run.inp`` and ``run.par`` text for one system."""
    atoms = SIZES[system]
    if KERNELS[system] == "lj":
        field, line = "LennardJones", f"{sigma:.8f} 1.0\n"
    else:
        field, line = "Gupta", " ".join(f"{v:g}" for v in GOLD_PARAMETERS) + "\n"
    inp = (
        f"{atoms:<30}# number of atoms\n"
        f"{initial or '*':<30}# initial guess file;  * - random guess\n"
        f"{field:<30}# force field types\n"
        f"{'run.par':<30}# force field parameters\n"
        f"{sn:<30}# population size\n"
        f"{ABC_GMAX:<30}# maximal generations\n"
        f"{ABC_GLIMIT:<30}# scout limit\n"
        f"{ABC_AMPLITUDE[system]:<30.8f}# amplitude\n"
        f"{'run':<30}# save optimized configuration to .xyz and .gjf\n"
        f"{ABC_SAVES:<30}# number of LMs to be saved\n"
    )
    return inp, f"{atoms}\n" + line * atoms


def parse_shim(text: str) -> dict[str, str]:
    fields = dict(re.findall(r"(\w+)=(\S+)", text))
    for key in ("hit", "calls", "calls_at_hit", "wall_ns", "inside_ns", "quenches", "stop"):
        if key not in fields:
            raise ValueError(f"shim row has no {key}: {text!r}")
    return fields


def run_abc(
    atom: Path,
    shim: Path,
    system: str,
    seed: int,
    budget: int,
    sn: int,
    work: Path,
    core: int = 0,
    initial: str | None = None,
    trace: int = 0,
    sigma: float = LJ_SIGMA,
) -> tuple[dict[str, str], float]:
    """One ABCluster run in a fresh directory. Returns the shim row and the process seconds."""
    work.mkdir(parents=True, exist_ok=False)
    inp, par = abc_input(system, sn, initial, sigma)
    (work / "run.inp").write_text(inp)
    (work / "run.par").write_text(par)
    env = child_env()
    env.update(
        ABC_KERNEL=KERNELS[system],
        ABC_OUT=str(work / "shim.txt"),
        ABC_SEED=str(seed),
        ABC_BUDGET=str(budget),
        ABC_TARGET=repr(REFERENCES[system] + HIT_TOL),
        ABC_TRACE=str(trace),
    )
    cmd = ["taskset", "-c", str(core), "env", f"LD_PRELOAD={shim}", str(atom), "run.inp"]
    with (work / "run.out").open("w") as handle:
        mark = time.perf_counter()
        proc = subprocess.run(
            cmd, cwd=work, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
        elapsed = time.perf_counter() - mark
    if proc.returncode != 0 or not (work / "shim.txt").is_file():
        tail = (work / "run.out").read_text()[-600:]
        raise RuntimeError(f"abcluster {system} seed {seed} exited {proc.returncode}\n{tail}")
    return parse_shim((work / "shim.txt").read_text()), elapsed


def best_of(fields: dict[str, str]) -> str:
    value = float(fields["best_quench"])
    return f"{value:.8f}" if value < 1e299 else ""


def abc_row(
    system: str,
    seed: int,
    budget: int,
    sn: int,
    shim_row: dict[str, str],
    process_s: float,
    revision: str,
    head: str,
    cpu: str,
) -> dict[str, str]:
    wall = int(shim_row["wall_ns"]) / 1e9
    pot = int(shim_row["inside_ns"]) / 1e9
    hit = shim_row["hit"] == "1"
    calls = int(shim_row["calls"])
    best = best_of(shim_row)
    if hit != (best != "" and float(best) <= REFERENCES[system] + HIT_TOL):
        raise RuntimeError(f"abcluster {system} {seed}: hit {hit} against best {best}")
    return {
        "system": system,
        "arm": "abcluster",
        "seed": str(seed),
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "calls_to_hit": shim_row["calls_at_hit"] if hit else str(budget),
        "time_to_hit_s": f"{wall:.6f}" if hit else "",
        "charged": str(calls),
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{max(wall - pot, 0.0):.6f}",
        "best": best,
        "stop": shim_row["stop"],
        "process_s": f"{process_s:.6f}",
        "setting": (
            f"SN={sn} L={ABC_AMPLITUDE[system]} glimit={ABC_GLIMIT} "
            f"sigma={LJ_SIGMA if KERNELS[system] == 'lj' else 'gupta'}"
        ),
        "code_revision": revision,
        "anneal_head": head,
        "host": os.uname().nodename,
        "cpu": cpu,
    }


def parse_anneal(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    header = next((line for line in lines if line.startswith("system,code,")), None)
    data = [line for line in lines if line.startswith(tuple(f"{s}," for s in PAPER_BUDGETS))]
    if header is None or len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}")
    return next(csv.DictReader([header, data[0]]))


def run_anneal(
    binary: Path, system: str, seed: int, preset: str, budget: int, log: Path, core: int = 0
) -> tuple[dict[str, str], float]:
    cmd = ["taskset", "-c", str(core), str(binary), "run", system, str(seed), preset]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        mark = time.perf_counter()
        proc = subprocess.run(
            cmd, env=child_env(budget), stdout=handle, stderr=subprocess.STDOUT, check=False
        )
        elapsed = time.perf_counter() - mark
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal {system} seed {seed} exited {proc.returncode}\n{text[-500:]}")
    return parse_anneal(text), elapsed


def anneal_row(
    arm: str, row: dict[str, str], process_s: float, budget: int, head: str, cpu: str
) -> dict[str, str]:
    if row["code"] != arm or int(row["budget"]) != budget or int(row["charged"]) > budget:
        raise RuntimeError(f"anneal row {row}")
    hit = row["hit"] == "1"
    wall, pot = float(row["wall_s"]), float(row["pot_s"])
    return {
        "system": row["system"],
        "arm": arm,
        "seed": row["seed"],
        "budget": row["budget"],
        "hit": "1" if hit else "0",
        "calls_to_hit": row["evals_to_hit"],
        "time_to_hit_s": f"{wall:.6f}" if hit else "",
        "charged": row["charged"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{max(wall - pot, 0.0):.6f}",
        "best": row["best"],
        "stop": "hit" if hit else "budget",
        "process_s": f"{process_s:.6f}",
        "setting": f"Config::{PRESETS[arm]}",
        "code_revision": head,
        "anneal_head": head,
        "host": os.uname().nodename,
        "cpu": cpu,
    }


def note(progress: Path, line: str) -> None:
    print(line, flush=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a") as handle:
        handle.write(line + "\n")


def parity(args, pins: dict[str, str], head: str) -> None:
    """Both codes' energy of one geometry, before any timed run."""
    path = args.out / "parity.csv"
    done = {(r["system"], r["seed"]) for r in read_rows(path)}
    for system in PAPER_BUDGETS:
        for seed in PARITY_SEEDS:
            if (system, str(seed)) in done:
                continue
            work = args.runs / "parity" / f"{system}-{seed}"
            work.mkdir(parents=True, exist_ok=True)
            raw = work / "start.xyz"
            subprocess.run(
                [str(args.anneal), "coords", system, str(seed), str(raw)], check=True
            )
            rows = [line.split() for line in raw.read_text().splitlines() if line.strip()]
            rounded = [[f"{float(v):.8f}" for v in row] for row in rows]
            plain = work / "rounded.xyz"
            plain.write_text("".join(" ".join(row) + "\n" for row in rounded))
            element = ELEMENTS[system]
            (work / "init.xyz").write_text(
                f"{len(rounded)}\nanneal start\n"
                + "".join(f"{element:<2} " + " ".join(f"{v:>15}" for v in row) + "\n" for row in rounded)
            )
            theirs = subprocess.run(
                [str(args.anneal), "energy", system, str(plain)],
                check=True, capture_output=True, text=True,
            )
            anneal = float(theirs.stdout.strip().splitlines()[-1])
            shim_row, _ = run_abc(
                args.atom, args.shim, system, 1, 1, ABC_SN[system], work / "abc",
                initial=str(work / "init.xyz"), trace=1, sigma=1.0,
            )
            abc = float(shim_row["e0"])
            diff = abs(abc - anneal)
            rel = diff / max(1.0, abs(anneal))
            if rel > PARITY_TOL:
                raise SystemExit(f"{system} {seed}: ABCluster {abc!r} != anneal {anneal!r}")
            parameters = (
                "sigma=1 epsilon=1"
                if KERNELS[system] == "lj"
                else "A=%g xi=%g d=%g p=%g q=%g" % GOLD_PARAMETERS
            )
            append_rows(path, PARITY_FIELDS, [{
                "system": system,
                "seed": str(seed),
                "atoms": str(SIZES[system]),
                "abcluster": f"{abc:.12f}",
                "anneal": f"{anneal:.12f}",
                "abs_diff": f"{diff:.3e}",
                "rel_diff": f"{rel:.3e}",
                "tolerance": f"{PARITY_TOL:.0e}",
                "parameters": parameters,
                "code_revision": f"{abc_revision(pins)} + {head}",
                "host": os.uname().nodename,
            }])
            note(args.runs / "progress.txt", f"parity {system} {seed} abs {diff:.3e} rel {rel:.3e}")


def gdb_counts(args, system: str, seed: int, work: Path) -> tuple[int, int] | None:
    from shutil import which

    if which("gdb") is None:
        return None
    inp, par = abc_input(system, ABC_SN[system], None)
    inp = inp.replace(f"{ABC_GMAX:<30}", f"{COUNTER_GENERATIONS:<30}")
    work.mkdir(parents=True)
    (work / "run.inp").write_text(inp)
    (work / "run.par").write_text(par)
    script = work / "count.gdb"
    kernel = KERNEL_OFFSET[KERNELS[system]]
    script.write_text(
        "set pagination off\n"
        "set disable-randomization on\n"
        f"set environment LD_PRELOAD={args.shim}\n"
        f"set environment ABC_SEED={seed}\n"
        f"break *({PIE_BASE:#x}+{kernel:#x})\n"
        "ignore 1 1000000000\n"
        f"break *({PIE_BASE:#x}+{QUENCH_OFFSET:#x})\n"
        "ignore 2 1000000000\n"
        "run run.inp > gdb.out 2>&1\n"
        "info breakpoints\n"
    )
    proc = subprocess.run(
        ["taskset", "-c", "0", "gdb", "-batch", "-x", str(script), str(args.atom)],
        cwd=work, env=child_env(), capture_output=True, text=True, check=False,
    )
    hits = [int(v) for v in re.findall(r"already hit (\d+) time", proc.stdout)]
    if len(hits) != 2:
        raise RuntimeError(f"gdb counted {proc.stdout[-400:]}")
    return hits[0], hits[1]


COUNTER_GENERATIONS = 40


def counter_check(args, pins: dict[str, str], head: str) -> None:
    """The shim's call and quench counts against a debugger's breakpoint counts."""
    path = args.out / "counter.csv"
    if read_rows(path):
        return
    for system, seed in (("lj38", 2000), ("au55", 2000)):
        work = args.runs / "counter" / f"{system}-{seed}"
        shim_dir = work / "shim"
        shim_dir.parent.mkdir(parents=True, exist_ok=True)
        inp, par = abc_input(system, ABC_SN[system], None)
        inp = inp.replace(f"{ABC_GMAX:<30}", f"{COUNTER_GENERATIONS:<30}")
        shim_dir.mkdir()
        (shim_dir / "run.inp").write_text(inp)
        (shim_dir / "run.par").write_text(par)
        env = child_env()
        env.update(
            ABC_KERNEL=KERNELS[system], ABC_OUT=str(shim_dir / "shim.txt"), ABC_SEED=str(seed),
            ABC_BUDGET=str(10**12), ABC_TARGET="-1e300",
        )
        subprocess.run(
            ["taskset", "-c", "0", "env", f"LD_PRELOAD={args.shim}", str(args.atom), "run.inp"],
            cwd=shim_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True,
        )
        shim_row = parse_shim((shim_dir / "shim.txt").read_text())
        counted = gdb_counts(args, system, seed, work / "gdb")
        gdb_calls, gdb_quenches = counted if counted else ("", "")
        match = counted is not None and counted == (
            int(shim_row["calls"]), int(shim_row["quenches"])
        )
        if counted is not None and not match:
            raise SystemExit(f"{system}: shim {shim_row['calls']}/{shim_row['quenches']} != gdb {counted}")
        append_rows(path, COUNTER_FIELDS, [{
            "system": system,
            "seed": str(seed),
            "generations": str(COUNTER_GENERATIONS),
            "shim_calls": shim_row["calls"],
            "gdb_calls": str(gdb_calls),
            "shim_quenches": shim_row["quenches"],
            "gdb_quenches": str(gdb_quenches),
            "match": "1" if match else "",
            "code_revision": f"{abc_revision(pins)} + {head}",
            "host": os.uname().nodename,
        }])
        note(args.runs / "progress.txt", f"counter {system} {seed} shim {shim_row['calls']} gdb {gdb_calls}")


def pilot(args, pins: dict[str, str], head: str) -> None:
    """ABCluster population sizes on seeds the campaign never uses, on spare cores."""
    path = args.out / "pilot.csv"
    done = {(r["system"], r["sn"], r["seed"]) for r in read_rows(path)}
    systems = tuple(args.system) if args.system else tuple(PAPER_BUDGETS)
    jobs = [
        (system, sn, seed)
        for system in systems
        for sn in PILOT_SN
        for seed in PILOT_SEEDS
        if (system, str(sn), str(seed)) not in done
    ]
    jobs.sort(key=lambda job: -PAPER_BUDGETS[job[0]])
    cores: queue.Queue[int] = queue.Queue()
    for core in PILOT_CORES:
        cores.put(core)
    revision = abc_revision(pins)
    progress = args.runs / "progress.txt"

    def one(job):
        system, sn, seed = job
        core = cores.get()
        try:
            budget = PAPER_BUDGETS[system]
            shim_row, _ = run_abc(
                args.atom, args.shim, system, seed, budget, sn,
                args.runs / "pilot" / system / f"sn{sn}-{seed}", core=core,
            )
            hit = shim_row["hit"] == "1"
            return {
                "system": system,
                "sn": str(sn),
                "seed": str(seed),
                "budget": str(budget),
                "hit": "1" if hit else "0",
                "calls_to_hit": shim_row["calls_at_hit"] if hit else str(budget),
                "charged": shim_row["calls"],
                "wall_s": f"{int(shim_row['wall_ns']) / 1e9:.6f}",
                "best": best_of(shim_row),
                "stop": shim_row["stop"],
                "code_revision": revision,
                "host": os.uname().nodename,
            }
        finally:
            cores.put(core)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PILOT_CORES)) as pool:
        for row in pool.map(one, jobs):
            append_rows(path, PILOT_FIELDS, [row])
            note(progress, f"pilot {row['system']} sn={row['sn']} {row['seed']} hit={row['hit']} calls={row['charged']} best={row['best']}")


def campaign(args, pins: dict[str, str], head: str) -> int:
    cpu = cpu_model()
    systems = tuple(args.system) if args.system else tuple(PAPER_BUDGETS)
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = tuple(args.arm) if args.arm else ARMS
    for system in systems:
        if system not in PAPER_BUDGETS:
            raise SystemExit(f"no paper budget for {system}")
        if args.budget is not None and args.budget != PAPER_BUDGETS[system]:
            raise SystemExit(f"{system} budget {args.budget} != {PAPER_BUDGETS[system]}")
        proc = subprocess.run(
            [str(args.anneal), "budget", system], check=True, capture_output=True, text=True
        )
        lines = proc.stdout.split()
        if int(lines[0]) != PAPER_BUDGETS[system] or abs(float(lines[1]) - REFERENCES[system]) > 1e-6:
            raise SystemExit(f"{system} anneal binary has {lines}")
    revision = abc_revision(pins)
    progress = args.runs / "progress.txt"
    parity(args, pins, head)
    counter_check(args, pins, head)
    order = schedule(systems, seeds, arms, SCHEDULE_SEED)
    note(progress, f"head {head} abcluster {revision} cpu {cpu}")
    note(progress, "schedule " + " ".join(f"{s}:{a}:{d}" for s, d, a in order))
    seeds_csv = args.out / "seeds.csv"
    logged = {(r["system"], r["arm"], r["seed"]) for r in read_rows(seeds_csv)}
    for system, seed, arm in order:
        if (system, arm, str(seed)) in logged:
            note(progress, f"skip {system} {arm} {seed}")
            continue
        budget = PAPER_BUDGETS[system]
        note(progress, f"start {system} {arm} {seed} budget={budget}")
        if arm == "abcluster":
            shim_row, process_s = run_abc(
                args.atom, args.shim, system, seed, budget, ABC_SN[system],
                args.runs / system / f"{arm}-{seed}",
            )
            row = abc_row(system, seed, budget, ABC_SN[system], shim_row, process_s, revision, head, cpu)
        else:
            parsed, process_s = run_anneal(
                args.anneal, system, seed, PRESETS[arm], budget,
                args.runs / system / f"{arm}-{seed}.log",
            )
            row = anneal_row(arm, parsed, process_s, budget, head, cpu)
        append_rows(seeds_csv, SEED_FIELDS, [row])
        logged.add((system, arm, str(seed)))
        note(
            progress,
            f"done {system} {arm} {seed} hit={row['hit']} calls={row['calls_to_hit']} "
            f"charged={row['charged']} wall_s={row['wall_s']} pot_s={row['pot_s']} "
            f"best={row['best']} stop={row['stop']}",
        )
    return 0


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    if PAPER_BUDGETS != {
        "lj38": 400_000,
        "lj55": 1_000_000,
        "lj75": 4_000_000,
        "lj98": 4_000_000,
        "lj104": 4_000_000,
        "au55": 1_000_000,
    }:
        raise SystemExit("paper budgets")
    if len(SEEDS) != 12 or SEEDS != tuple(range(2000, 2012)):
        raise SystemExit("seeds")
    if set(PILOT_SEEDS) & set(SEEDS):
        raise SystemExit("pilot seeds overlap the campaign")
    names = list(PAPER_BUDGETS)
    for table in (REFERENCES, SIZES, KERNELS, ELEMENTS, ABC_AMPLITUDE, ABC_SN):
        if list(table) != names:
            raise SystemExit("system tables disagree")
    for system in names:
        if KERNELS[system] == "lj":
            expected = 5.0 * (SIZES[system] / 38.0) ** (1.0 / 3.0)
            if abs(ABC_AMPLITUDE[system] - expected) > 0.005:
                raise SystemExit(f"{system} amplitude {ABC_AMPLITUDE[system]} != {expected:.3f}")
    gold = 5.0 * (55 / 38.0) ** (1.0 / 3.0) * GOLD_PARAMETERS[2] / (LJ_SIGMA * 2.0 ** (1.0 / 6.0))
    if abs(ABC_AMPLITUDE["au55"] - gold) > 0.05:
        raise SystemExit(f"gold amplitude {ABC_AMPLITUDE['au55']} != {gold:.2f}")
    order = schedule(names, SEEDS, ARMS, SCHEDULE_SEED)
    seen: list[str] = []
    for system, _seed, _arm in order:
        if not seen or seen[-1] != system:
            seen.append(system)
    if seen != names:
        raise SystemExit(f"system order {seen}")
    for system in names:
        jobs = sorted((seed, arm) for name, seed, arm in order if name == system)
        if jobs != sorted((seed, arm) for seed in SEEDS for arm in ARMS):
            raise SystemExit(f"{system} is not the shared seed list")
    inp, par = abc_input("lj38", 30, None)
    if inp.splitlines()[2].split("#")[0].strip() != "LennardJones" or par.count("\n") != 39:
        raise SystemExit("lennard-jones input")
    inp, par = abc_input("au55", 30, "start.xyz")
    if "Gupta" not in inp or "start.xyz" not in inp or par.splitlines()[1] != "0.2061 1.79 2.884 10.229 4.036":
        raise SystemExit("gupta input")
    row = parse_shim(
        "hit=1 calls=77 calls_at_hit=70 wall_ns=5000 inside_ns=2000 quenches=3 "
        "best_quench=-173.9284265903 best_kernel=-173.9284265903 stop=hit seed=2000 e0=1.5"
    )
    out = abc_row("lj38", 2000, 400_000, 30, row, 0.01, "abcluster-3.4@x", "head", "cpu")
    if out["calls_to_hit"] != "70" or out["outside_s"] != "0.000003" or out["time_to_hit_s"] != "0.000005":
        raise SystemExit(f"abcluster row {out}")
    miss = parse_shim(
        "hit=0 calls=400000 calls_at_hit=400000 wall_ns=9000000000 inside_ns=4000000000 "
        "quenches=1500 best_quench=-173.1343170088 best_kernel=-173.1 stop=budget seed=2001"
    )
    out = abc_row("lj38", 2001, 400_000, 30, miss, 9.1, "abcluster-3.4@x", "head", "cpu")
    if out["calls_to_hit"] != "400000" or out["time_to_hit_s"] != "":
        raise SystemExit(f"abcluster miss row {out}")
    parsed = parse_anneal(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s\n"
        "au55,anneal-cluster,2000,1000000,0,1000000,1000000,3.5,-190.7,2.0,1.5\n"
    )
    out = anneal_row("anneal-cluster", parsed, 3.6, 1_000_000, "head", "cpu")
    if out["hit"] != "0" or out["stop"] != "budget" or out["calls_to_hit"] != "1000000":
        raise SystemExit(f"anneal row {out}")
    pilot_rows = [
        {"system": "lj38", "sn": "30", "seed": "1000", "hit": "1", "calls_to_hit": "100"},
        {"system": "lj38", "sn": "100", "seed": "1000", "hit": "1", "calls_to_hit": "50"},
        {"system": "lj38", "sn": "30", "seed": "1001", "hit": "0", "calls_to_hit": "400000"},
        {"system": "lj38", "sn": "100", "seed": "1001", "hit": "1", "calls_to_hit": "60"},
    ]
    if choose_population(pilot_rows, "lj38") != 100:
        raise SystemExit("population choice")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--anneal", type=Path, required=True, help="the h2h_abcluster binary")
    parser.add_argument("--atom", type=Path, required=True, help="ABCluster 3.4 atom")
    parser.add_argument("--shim", type=Path, required=True, help="abc_shim.so")
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv[1:])
    pins = load_pins(args.pins)
    if sha256_of(args.atom) != pins["atom_sha256"]:
        raise SystemExit(f"{args.atom} is not the pinned ABCluster {pins['version']} atom")
    head = args.head_file.read_text().strip()
    if args.pilot:
        pilot(args, pins, head)
        return 0
    return campaign(args, pins, head)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
