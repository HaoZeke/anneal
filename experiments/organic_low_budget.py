#!/usr/bin/env python3
"""Low-budget GFN2 comparison of CREST iMTD-GC and anneal.

CREST is ``crest --gfn2 -mquick -mdlen x0.05 -T 1``. The anneal baseline is
the molecular preset without internal-coordinate moves. ``fixed`` is
``Config::recommended_molecular``. A hit against the longer CREST oracle is
reported at 0.1 and 1.0 kcal/mol. The crossover is the first charged call at
which anneal's running best is at or below CREST's running best.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HARTREE_EV = 27.211386245988
KCAL_PER_EV = 23.06054783061903
XTB = os.environ.get("XTB_EXE", "xtb")
CREST = os.environ.get("CREST_EXE", "crest")
BENCH = os.environ.get("ORGANIC_BENCH", "")


def water_xyz(n_mol: int, seed: int) -> str:
    rng = np.random.default_rng(10_000 + 100 * n_mol + seed)
    oxygens = []
    while len(oxygens) < n_mol:
        point = rng.normal(size=3) * (1.6 + 0.35 * n_mol ** (1.0 / 3.0))
        if all(np.linalg.norm(point - other) >= 2.6 for other in oxygens):
            oxygens.append(point)
    lines = [str(3 * n_mol)]
    groups = ";".join(f"{3 * i},{3 * i + 1},{3 * i + 2}" for i in range(n_mol))
    lines.append(f"groups {groups}")
    template = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    for oxygen in oxygens:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        rotation = rodrigues(axis, angle)
        for row in template:
            xyz = oxygen + rotation @ row
            symbol = "O" if np.allclose(row, 0.0) else "H"
            lines.append(f"{symbol} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}")
    return "\n".join(lines) + "\n"


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return c * np.eye(3) + s * cross + (1 - c) * np.outer(axis, axis)


def hexane_xyz(seed: int) -> str:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))
    params = AllChem.EmbedParameters()
    params.randomSeed = int(seed) + 1
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        AllChem.EmbedMolecule(mol, randomSeed=int(seed) + 1)
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), "groups " + ",".join(str(i) for i in range(n))]
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol()} {p.x:.8f} {p.y:.8f} {p.z:.8f}")
    return "\n".join(lines) + "\n"


def histidine_xyz(seed: int) -> str:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](Cc1cnc[nH]1)C(=O)O"))
    params = AllChem.EmbedParameters()
    params.randomSeed = int(seed) + 3
    if AllChem.EmbedMolecule(mol, params) != 0:
        AllChem.EmbedMolecule(mol, randomSeed=int(seed) + 3)
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), "groups " + ",".join(str(i) for i in range(n))]
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol()} {p.x:.8f} {p.y:.8f} {p.z:.8f}")
    return "\n".join(lines) + "\n"


SYSTEMS = {
    "w4": lambda seed: water_xyz(4, seed),
    "w6": lambda seed: water_xyz(6, seed),
    "hexane": hexane_xyz,
    "histidine": histidine_xyz,
}


def run(cmd: list[str], cwd: Path, log: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(proc.stdout)
    return proc


def xtb_energy(xyz: Path, work: Path) -> float:
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(xyz, work / "geo.xyz")
    proc = run(
        [XTB, "geo.xyz", "--gfn", "2", "--acc", "0.01", "--grad", "--norestart"],
        work,
        work / "xtb.log",
    )
    match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", proc.stdout)
    if not match:
        raise RuntimeError(f"xtb energy missing in {work}")
    return float(match.group(1)) * HARTREE_EV


def crest_quick(xyz: Path, work: Path, oracle: bool) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(xyz, work / "mol.xyz")
    cmd = [CREST, "mol.xyz", "--gfn2", "-mquick", "-T", "1"]
    if not oracle:
        cmd.extend(["-mdlen", "x0.05"])
    proc = run(cmd, work, work / "crest.out")
    text = proc.stdout
    if "CREST terminated normally" not in text:
        repaired = xtb_repair(work / "mol.xyz", work / "repair")
        shutil.copy(repaired, work / "mol.xyz")
        cmd.append("--noreftopo")
        proc = run(cmd, work, work / "crest.out")
        text = proc.stdout
        if "CREST terminated normally" not in text:
            raise RuntimeError(f"CREST failed in {work}")
    calls = int(re.search(r"Total number of energy\+grad calls:\s+(\d+)", text).group(1))
    best = work / "crest_best.xyz"
    energy = xtb_energy(best, work / "rescore")
    return {"calls": calls + 1, "energy_eV": energy, "crest_calls": calls, "dir": work}


def xtb_repair(xyz: Path, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(xyz, work / "geo.xyz")
    run(
        [XTB, "geo.xyz", "--gfn", "2", "--acc", "0.01", "--opt", "--cycles", "30", "--norestart"],
        work,
        work / "repair.log",
    )
    opt = work / "xtbopt.xyz"
    if not opt.exists():
        raise RuntimeError(f"repair produced no xtbopt.xyz in {work}")
    return opt


def crest_curve(work: Path, total_calls: int) -> list[tuple[int, float]]:
    energies = []
    dyn = work / "crest_dynamics.trj"
    opt = work / "crestopt.log"
    for path, marker in ((dyn, "Epot"), (opt, "Etot")):
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if marker not in line:
                continue
            match = re.search(r"(-?\d+\.\d+)", line)
            if match:
                energies.append(float(match.group(1)) * HARTREE_EV)
    if not energies:
        return []
    points = []
    best = float("inf")
    n = len(energies)
    for i, energy in enumerate(energies, start=1):
        best = min(best, energy)
        call = max(1, int(round(i * total_calls / n)))
        points.append((call, best))
    return points


def anneal(xyz: Path, work: Path, budget: int, seed: int, arm: str) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(xyz, work / "mol.xyz")
    env_prefix = os.environ.copy()
    env_prefix["ASE_ENGINE"] = "xtb-cli"
    env_prefix["XTB_ACC"] = "0.01"
    env_prefix["XTB_EXE"] = XTB
    env_prefix["OMP_NUM_THREADS"] = "1"
    env_prefix["ORGANIC_REPLICAS"] = "1"
    proc = subprocess.run(
        [BENCH, "mol.xyz", str(budget), str(seed), arm],
        cwd=work,
        env=env_prefix,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (work / "anneal.out").write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"anneal failed in {work}\n{proc.stdout[-2000:]}")
    improvements = []
    best = None
    charged = None
    for line in proc.stdout.splitlines():
        if line.startswith("IMPROVE "):
            _, call, energy = line.split()
            improvements.append((int(call), float(energy)))
        elif line.startswith("BEST "):
            best = float(line.split()[1])
        elif line.startswith("CHARGED "):
            charged = int(line.split()[1])
    return {
        "best": best,
        "charged": charged,
        "improvements": improvements,
        "log": proc.stdout,
    }


def value_at(points: list[tuple[int, float]], call: int) -> float:
    best = float("inf")
    for c, energy in points:
        if c <= call:
            best = min(best, energy)
        else:
            break
    return best


def crossover(anneal_pts: list[tuple[int, float]], crest_pts: list[tuple[int, float]], limit: int) -> int | None:
    """First call at which anneal leads CREST and still leads at ``limit``.

    A lead that CREST later retakes is not a crossover. ``None`` means anneal
    is behind at the equal budget.
    """
    if not anneal_pts or not crest_pts:
        return None
    lead = None
    for call in range(1, limit + 1):
        anneal_e = value_at(anneal_pts, call)
        crest_e = value_at(crest_pts, call)
        if anneal_e <= crest_e + 1.0e-8:
            if lead is None:
                lead = call
        else:
            lead = None
    return lead


def calls_to_beat(points: list[tuple[int, float]], target: float) -> int | None:
    for call, energy in points:
        if energy <= target + 1.0e-8:
            return call
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=["w4", "w6", "hexane"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--fixed-budget", type=int, default=4000)
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()
    if not BENCH:
        sys.exit("set ORGANIC_BENCH to the organic_bench binary")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for system in args.systems:
        for seed in args.seeds:
            root = args.out / f"{system}_s{seed}"
            root.mkdir(parents=True, exist_ok=True)
            xyz = root / "start.xyz"
            xyz.write_text(SYSTEMS[system](seed))
            print(f"== {system} seed {seed} CREST", flush=True)
            quick = crest_quick(xyz, root / "crest", oracle=False)
            curve = crest_curve(root / "crest", quick["crest_calls"])
            print(f"   crest calls {quick['calls']} energy {quick['energy_eV']:.6f}", flush=True)
            print(f"== {system} seed {seed} baseline", flush=True)
            base = anneal(xyz, root / "baseline", quick["calls"], seed, "baseline")
            print(f"== {system} seed {seed} fixed", flush=True)
            fixed_budget = max(quick["calls"], args.fixed_budget)
            fixed = anneal(xyz, root / "fixed", fixed_budget, seed, "fixed")
            oracle_energy = None
            if not args.skip_oracle and seed == args.seeds[0]:
                print(f"== {system} oracle", flush=True)
                oracle = crest_quick(xyz, args.out / f"{system}_oracle", oracle=True)
                oracle_energy = oracle["energy_eV"]
                (args.out / f"{system}_oracle_energy.txt").write_text(
                    f"{oracle_energy:.8f} {oracle['calls']}\n"
                )
            elif (args.out / f"{system}_oracle_energy.txt").exists():
                oracle_energy = float((args.out / f"{system}_oracle_energy.txt").read_text().split()[0])
            base_at = value_at(base["improvements"], quick["calls"])
            fixed_at = value_at(fixed["improvements"], quick["calls"])
            cross_base = crossover(base["improvements"], curve, quick["calls"])
            cross_fixed = crossover(fixed["improvements"], curve, quick["calls"])
            beat_fixed = calls_to_beat(fixed["improvements"], quick["energy_eV"])
            row = {
                "system": system,
                "seed": seed,
                "crest_calls": quick["calls"],
                "crest_eV": quick["energy_eV"],
                "baseline_eV": base_at,
                "fixed_at_budget_eV": fixed_at,
                "fixed_final_eV": fixed["best"],
                "fixed_budget": fixed_budget,
                "oracle_eV": oracle_energy,
                "crossover_baseline": cross_base,
                "crossover_fixed": cross_fixed,
                "fixed_calls_to_beat_crest_final": beat_fixed,
            }
            rows.append(row)
            print(row, flush=True)
    csv = args.out / "summary.csv"
    keys = list(rows[0].keys()) if rows else []
    with csv.open("w") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join("" if row[k] is None else str(row[k]) for k in keys) + "\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
