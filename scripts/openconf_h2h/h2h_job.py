#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One seeded conformer search: anneal, openconf or CREST on one molecule.

Each job writes ``job.json`` (counts, clocks, settings) and
``candidates.xyz`` (every geometry the arm reports, unpolished) into its
directory. Scoring is a separate step, ``h2h_score.py``, and is the same
for every arm.

The RDKit arms need ``librdff_count.so`` preloaded. ``grad_calls`` and
``energy_calls`` are outermost RDKit ForceField calls, ``pot_s`` the time
inside them. ETKDG distance-geometry calls are reported apart, as
``embed_*``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2h_common as hc  # noqa: E402

# A recorded minimum of a molecular search is at 5e-2 eV/Å.
RECORD_GRADIENT = 5.0e-2


class Harvest:
    """Keeps the lowest sub-gate point of each run of sub-gate evaluations.

    A quench ends where its gradient first stays under the record gate; the
    next evaluation above the gate is a new proposal or a new quench.
    """

    def __init__(self, gate: float) -> None:
        self.gate = gate
        self.minima: list[tuple[np.ndarray, float]] = []
        self._run: tuple[np.ndarray, float] | None = None
        self._last: tuple[bytes, float] | None = None

    def energy(self, x: np.ndarray, e: float) -> None:
        self._last = (x.tobytes(), e)

    def gradient(self, x: np.ndarray, g: np.ndarray) -> None:
        if self._last is None or self._last[0] != x.tobytes():
            return
        e = self._last[1]
        if np.max(np.abs(g)) < self.gate:
            if self._run is None or e < self._run[1]:
                self._run = (x.copy(), e)
        else:
            self.close()

    def close(self) -> None:
        if self._run is not None:
            self.minima.append(self._run)
            self._run = None


def run_anneal(row: dict, seed: int, budget: int, out: Path, local_step: str | None = None) -> dict:
    import anneal

    mol = hc.build_mol(row["smiles"])
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    groups = hc.rigid_groups(mol)
    count = hc.counter()
    count.reset()
    t0 = time.perf_counter()
    x0 = hc.embed(mol, seed)
    ff = hc.mmff94(mol)
    harvest = Harvest(RECORD_GRADIENT)

    def energy(x):
        e = ff.CalcEnergy(x.tolist()) / hc.KCAL_PER_EV
        harvest.energy(x, e)
        return e

    def gradient(x):
        g = np.asarray(ff.CalcGrad(x.tolist())) / hc.KCAL_PER_EV
        harvest.gradient(x, g)
        return g

    extra = {} if local_step is None else {"local_step": local_step}
    result = anneal.cluster_search(
        energy,
        gradient,
        len(z),
        budget,
        seed=seed,
        recommended=True,
        start=x0.reshape(-1),
        species=z,
        groups=groups,
        **extra,
    )
    harvest.close()
    wall = time.perf_counter() - t0
    counts = count.read()
    frames = [(x, e * hc.KCAL_PER_EV) for x, e in harvest.minima]
    best = np.asarray(result["best"])
    frames.append((best, float(result["best_energy"]) * hc.KCAL_PER_EV))
    hc.write_xyz(out / "candidates.xyz", hc.symbols(mol), frames)
    return {
        "wall_s": wall,
        **counts,
        "budget": budget,
        "hops": int(result["hops"]),
        "accepted": result.get("accepted"),
        "best_kcal": float(result["best_energy"]) * hc.KCAL_PER_EV,
        "candidates": len(frames),
        "groups": len(groups),
        "local_step": result.get("local_step"),
        "quench_seconds": result.get("quench_seconds"),
        "settings": "cluster_search recommended_molecular, energy scale 1 eV, local step "
        + (local_step or "preset"),
    }


def _pin_mmff94():
    """Make every openconf MMFF call MMFF94; openconf names MMFF94s in places."""
    from rdkit.Chem import AllChem

    props = AllChem.MMFFGetMoleculeProperties
    optimize = AllChem.MMFFOptimizeMoleculeConfs

    def pinned_props(mol, mmffVariant="MMFF94", **kwargs):
        return props(mol, mmffVariant="MMFF94", **kwargs)

    def pinned_optimize(mol, **kwargs):
        kwargs["mmffVariant"] = "MMFF94"
        return optimize(mol, **kwargs)

    AllChem.MMFFGetMoleculeProperties = pinned_props
    AllChem.MMFFOptimizeMoleculeConfs = pinned_optimize


def _phase_embedding():
    from rdkit.Chem import AllChem

    for name in ("EmbedMultipleConfs", "EmbedMolecule"):
        inner = getattr(AllChem, name)

        def wrapped(*args, _inner=inner, **kwargs):
            with hc.phase(1):
                return _inner(*args, **kwargs)

        setattr(AllChem, name, wrapped)


def run_openconf(row: dict, seed: int, out: Path) -> dict:
    _pin_mmff94()
    _phase_embedding()
    import openconf
    from openconf import generate_conformers
    from openconf.config import preset_config

    preset = "macrocycle" if row["macrocycle"] == "yes" else "spectroscopic"
    config = dataclasses.replace(
        preset_config(preset),
        num_threads=1,
        random_seed=int(seed),
        fast_dielectric=1.0,
        final_dielectric=1.0,
    )
    mol = hc.build_mol(row["smiles"])
    count = hc.counter()
    count.reset()
    t0 = time.perf_counter()
    ensemble = generate_conformers(mol, config=config)
    wall = time.perf_counter() - t0
    counts = count.read()
    out_mol = ensemble.mol
    if [a.GetAtomicNum() for a in out_mol.GetAtoms()] != [a.GetAtomicNum() for a in mol.GetAtoms()]:
        raise RuntimeError("openconf reordered atoms")
    frames = [
        (out_mol.GetConformer(cid).GetPositions(), float(e))
        for cid, e in zip(ensemble.conf_ids, ensemble.energies, strict=True)
    ]
    hc.write_xyz(out / "candidates.xyz", hc.symbols(mol), frames)
    return {
        "wall_s": wall,
        **counts,
        "candidates": len(frames),
        "best_kcal": min((e for _, e in frames), default=float("nan")),
        "settings": f"openconf {openconf.__version__ if hasattr(openconf, '__version__') else ''} "
        f"preset {preset}, num_threads 1, dielectric 1, MMFF94",
    }


_CALLS = re.compile(r"Total number of energy\+grad calls:\s+(\d+)")
_INSIDE = re.compile(r"Wall time inside energy\+grad calls:\s+([0-9.]+)")
_RUNTIME = re.compile(r"CREST runtime \(total\)\s+(\d+) d,\s+(\d+) h,\s+(\d+) min,\s+([0-9.]+) sec")


def run_crest(row: dict, seed: int, out: Path, crest: str) -> dict:
    mol = hc.build_mol(row["smiles"])
    x0 = hc.embed(mol, seed)
    hc.write_xyz(out / "start.xyz", hc.symbols(mol), [(x0, 0.0)])
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_STACKSIZE="1G")
    env.pop("LD_PRELOAD", None)
    t0 = time.perf_counter()
    with (out / "crest.out").open("w") as stdout:
        proc = subprocess.run(
            [crest, "start.xyz", "--gfnff", "-T", "1"],
            cwd=out,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.perf_counter() - t0
    text = (out / "crest.out").read_text(errors="replace")
    calls = _CALLS.search(text)
    inside = _INSIDE.search(text)
    runtime = _RUNTIME.search(text)
    ensemble = out / "crest_conformers.xyz"
    frames = []
    if ensemble.is_file():
        want = hc.symbols(mol)
        for sym, x, comment in hc.read_xyz(ensemble):
            if sym != want:
                raise RuntimeError("CREST reordered atoms")
            frames.append((x, float(comment.split()[0]) * 627.509474))
        hc.write_xyz(out / "candidates.xyz", want, frames)
    for name in ("MDFILES", "crest_dynamics.trj", "crest_rotamers_0.xyz", "crest_rotamers_1.xyz"):
        path = out / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    return {
        "wall_s": wall,
        "crest_runtime_s": (
            int(runtime[1]) * 86400 + int(runtime[2]) * 3600 + int(runtime[3]) * 60 + float(runtime[4])
            if runtime
            else None
        ),
        "grad_calls": int(calls[1]) if calls else None,
        "energy_calls": int(calls[1]) if calls else None,
        "pot_s": float(inside[1]) if inside else None,
        "exit": proc.returncode,
        "candidates": len(frames),
        "settings": "crest --gfnff -T 1 (iMTD-GC, GFN-FF)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["anneal", "openconf", "crest"])
    parser.add_argument("molecule")
    parser.add_argument("seed", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--crest", default=os.environ.get("CREST_BIN", "crest"))
    parser.add_argument("--local-step", choices=["lbfgs", "tric", "auto"], help="anneal quench; default the preset's")
    args = parser.parse_args()
    row = hc.molecules()[args.molecule]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.arm == "anneal":
        if args.budget < 1:
            raise SystemExit("anneal needs --budget")
        info = run_anneal(row, args.seed, args.budget, args.out, args.local_step)
    elif args.arm == "openconf":
        info = run_openconf(row, args.seed, args.out)
    else:
        info = run_crest(row, args.seed, args.out, args.crest)
    info.update(
        arm=args.arm,
        molecule=args.molecule,
        seed=args.seed,
        hostname=os.uname().nodename,
    )
    if info.get("pot_s") is not None:
        info["outside_s"] = max(info["wall_s"] - info["pot_s"], 0.0)
    (args.out / "job.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: info[k] for k in ("arm", "molecule", "seed", "wall_s", "grad_calls", "candidates")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
