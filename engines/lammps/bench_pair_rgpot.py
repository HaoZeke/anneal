#!/usr/bin/env python3
"""Wall time of pair_style rgpot (gather and domain) against native styles.

    bench_pair_rgpot.py --lmp LMP --plugin libpair_rgpot.so \
        --eam CuNi.eam.alloy [--ranks 1 2 4 8] [--repeats 3] [--out bench.json]

Each case is one lmp process under mpirun -np N: fcc Lennard-Jones
(lj/cut, shifted, 8.5 A) or CuNi EAM (eam/alloy), 300 K NVE, a warm-up
run, then a timed run. The time is the "Loop time" LAMMPS prints for the
timed run, so start-up and setup are left out. Strong scaling keeps the
box; weak scaling grows the box along x with the ranks. The script
prints a Markdown table of the medians and writes every repeat to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

LOOP = re.compile(r"Loop time of ([0-9.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


def pair_lines(system: str, mode: str, plugin: str, eam: str) -> list[str]:
    if system == "lj":
        if mode == "native":
            return ["pair_style lj/cut 8.5", "pair_modify shift yes", "pair_coeff * * 0.0104 3.40"]
        return [f"plugin load {plugin}", f"pair_style rgpot 8.5 mode {mode}",
                "pair_coeff * * LJ u0 0.0104 psi 3.40 cutoff 8.5 Ar"]
    if mode == "native":
        return ["pair_style eam/alloy", f"pair_coeff * * {eam} Cu Ni"]
    return [f"plugin load {plugin}", f"pair_style rgpot 6.5 mode {mode}",
            f"pair_coeff * * EAMTable file {eam} Cu Ni"]


def deck(system: str, cells: tuple[int, int, int], mode: str, args) -> str:
    a, mass, ntypes = (5.26, 39.948, 1) if system == "lj" else (3.6, 60.0, 2)
    lines = [
        "units metal",
        "atom_style atomic",
        "atom_modify map array",
        "boundary p p p",
        f"lattice fcc {a}",
        f"region box block 0 {cells[0]} 0 {cells[1]} 0 {cells[2]}",
        f"create_box {ntypes} box",
        "create_atoms 1 box",
        f"mass * {mass}",
    ]
    if system == "eam":
        lines.append("set type 1 type/fraction 2 0.5 9137")
    lines += pair_lines(system, mode, args.plugin, args.eam)
    lines += [
        "velocity all create 300 4928 dist gaussian",
        "fix 1 all nve",
        "thermo 50",
        f"run {args.warmup}",
        f"run {args.steps[system]}",
    ]
    return "\n".join(lines) + "\n"


def run_one(system, cells, mode, nprocs, args, work: Path) -> dict:
    inp = work / f"in.{system}.{mode}.{nprocs}"
    inp.write_text(deck(system, cells, mode, args))
    cmd = [args.mpirun, "-np", str(nprocs), "--bind-to", "core", args.lmp,
           "-in", str(inp), "-log", "none", "-nocite"]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=work, check=False)
    loops = LOOP.findall(proc.stdout)
    if proc.returncode != 0 or len(loops) < 2:
        raise RuntimeError(f"{' '.join(cmd)} failed\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    seconds, procs, steps, atoms = loops[-1]
    pe = re.findall(r"^\s+\d+\s+[0-9.eE+-]+\s+([-0-9.eE+]+)", proc.stdout, re.M)
    return {"seconds": float(seconds), "procs": int(procs), "steps": int(steps),
            "atoms": int(atoms), "last_pe": float(pe[-1]) if pe else None}


def hardware() -> dict:
    info = {"machine": platform.machine(), "cpus": os.cpu_count()}
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("Model name", "Hypervisor vendor", "L3 cache"):
                info[key.strip()] = value.strip()
    except OSError:
        pass
    return info


def versions(args) -> dict:
    lmp = subprocess.run([args.lmp, "-h"], capture_output=True, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="1")).stdout
    mpi = subprocess.run([args.mpirun, "--version"], capture_output=True, text=True).stdout
    head = [line for line in lmp.splitlines() if line.startswith("Large-scale")]
    return {"lammps": head[0] if head else lmp.splitlines()[0:3],
            "lammps_tag": next((line.strip() for line in lmp.splitlines()
                                if "Git info" in line), ""),
            "mpi": mpi.splitlines()[0] if mpi else ""}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lmp", required=True)
    p.add_argument("--plugin", required=True)
    p.add_argument("--eam", required=True)
    p.add_argument("--mpirun", default="mpirun")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--systems", nargs="+", default=["lj", "eam"])
    p.add_argument("--modes", nargs="+", default=["native", "gather", "domain"])
    p.add_argument("--scaling", nargs="+", default=["strong", "weak"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lj-steps", type=int, default=100)
    p.add_argument("--eam-steps", type=int, default=100)
    p.add_argument("--out", default="bench_pair_rgpot.json")
    args = p.parse_args()
    args.steps = {"lj": args.lj_steps, "eam": args.eam_steps}
    # Strong: one box for every rank count. Weak: 8 x 16 x 16 cells per
    # rank along x (4 atoms per cell).
    strong = {"lj": (16, 16, 16), "eam": (20, 20, 20)}
    results = {"hardware": hardware(), "versions": versions(args), "repeats": args.repeats,
               "steps": args.steps, "warmup": args.warmup, "runs": []}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for scaling in args.scaling:
            for system in args.systems:
                for n in args.ranks:
                    cells = strong[system] if scaling == "strong" else (8 * n, 16, 16)
                    for mode in args.modes:
                        times = []
                        for _ in range(args.repeats):
                            r = run_one(system, cells, mode, n, args, work)
                            times.append(r["seconds"])
                        row = {"scaling": scaling, "system": system, "ranks": n, "mode": mode,
                               "atoms": r["atoms"], "steps": r["steps"], "seconds": times,
                               "last_pe": r["last_pe"]}
                        results["runs"].append(row)
                        print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print()
    print("| scaling | system | atoms | ranks | " + " | ".join(args.modes) + " |")
    print("|---|---|---|---|" + "---|" * len(args.modes))
    keyed = {(r["scaling"], r["system"], r["ranks"], r["mode"]): r for r in results["runs"]}
    for scaling in args.scaling:
        for system in args.systems:
            for n in args.ranks:
                cells = [keyed[(scaling, system, n, m)] for m in args.modes]
                ms = [f"{1e3 * statistics.median(c['seconds']) / c['steps']:.2f}" for c in cells]
                print(f"| {scaling} | {system} | {cells[0]['atoms']} | {n} | " + " | ".join(ms)
                      + " |")
    print("\nms per step, median of", args.repeats, "repeats")


if __name__ == "__main__":
    main()
