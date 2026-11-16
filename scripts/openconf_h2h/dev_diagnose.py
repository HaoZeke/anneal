# Copyright (c) anneal developers
"""Why anneal runs miss a molecule's lowest conformer, on development seeds.

``run`` searches each (molecule, seed) as ``h2h_job.py`` does, with the
recommended preset and the paired budget, and keeps every minimum the
harvest records together with every evaluated geometry within 4 kcal/mol
of the reference. ``analyse`` polishes each run's minima within 3 kcal/mol
of the reference as ``h2h_score.py`` does and writes one row per run:

- ``hit``: a valid polished minimum within 0.1 kcal/mol of the reference
- ``gap_kcal``: lowest valid polished energy minus the reference, empty
  when no minimum lies within 3 kcal/mol
- ``invalid_low``: polished minima at the reference energy whose
  stereocentres or double bonds differ from the SMILES
- ``near_minimum``, ``near_evaluation``: smallest heavy-atom RMSD to the
  reference, over automorphisms, of a recorded minimum within 8 kcal/mol
  and of a low evaluation (one in every len/400)
- for a miss, its lowest valid conformer against the reference, atoms
  mapped by the best automorphism: rotatable torsions more than 45 degrees
  apart (``rotors``, amide-type bonds apart as ``amides``), non-aromatic
  rings with an endocyclic torsion more than 30 degrees apart (``rings``),
  and intramolecular hydrogen bonds (H to N or O under 2.5 Å, D-H...A over
  110 degrees, ends at least three bonds apart) of the reference it lacks
  (``hbonds_lost``) and that it has beyond them (``hbonds_new``)

The reference of each molecule is ``reference/<molecule>.xyz`` beside the
rows: the lowest valid polished conformer of every session on disk.

    dev_diagnose.py run --molecules z34 --seeds 3000-3059 --out DIR [--kwargs JSON]
    dev_diagnose.py analyse --runs DIR --molecules z34 --rows rows.csv --arm develop
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2h_common as hc
from h2h_job import RECORD_GRADIENT, Harvest
from h2h_score import Molecule

REFERENCE = Path(__file__).resolve().parents[2] / "bench" / "openconf-h2h" / "conf-fix6" / "reference"
BUDGET = {"915": 24205, "fluvastatin": 39708, "amprenavir": 37880, "lorlatinib": 41771,
          "xld": 42076, "rq3": 49297, "z34": 43421, "lopinavir": 45759}
LOW = 4.0
HIT = 0.1
POLISH_WINDOW = 3.0
NEAR_WINDOW = 8.0
ROTATABLE = Chem.MolFromSmarts("[!D1&!$(*#*)]-&!@[!D1&!$(*#*)]")
AMIDE = Chem.MolFromSmarts("[CX3,SX4](=O)-&!@[NX3]")
FIELDS = ["molecule", "arm", "seed", "hit", "gap_kcal", "invalid_low", "near_minimum", "near_evaluation",
          "rotors", "amides", "rings", "hbonds_reference", "hbonds_lost", "hbonds_new", "minima",
          "low_evaluations", "calls", "wall_s", "pot_s"]


def reference(name: str) -> tuple[np.ndarray, float]:
    _, x, comment = hc.read_xyz(REFERENCE / f"{name}.xyz")[0]
    return x, float(comment)


def run_one(args) -> None:
    import anneal

    name, seed, out, kwargs = args
    mol = hc.build_mol(hc.molecules()[name]["smiles"])
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    x0 = hc.embed(mol, seed)
    ff = hc.mmff94(mol)
    e_ref = reference(name)[1]
    harvest = Harvest(RECORD_GRADIENT)
    low_x, low_e = [], []
    pot = [0.0]

    def energy(x):
        t = time.perf_counter()
        e = ff.CalcEnergy(x.tolist())
        pot[0] += time.perf_counter() - t
        harvest.energy(x, e / hc.KCAL_PER_EV)
        if e < e_ref + LOW:
            low_x.append(np.array(x, dtype=np.float32))
            low_e.append(e)
        return e / hc.KCAL_PER_EV

    def gradient(x):
        t = time.perf_counter()
        g = np.asarray(ff.CalcGrad(x.tolist())) / hc.KCAL_PER_EV
        pot[0] += time.perf_counter() - t
        harvest.gradient(x, g)
        return g

    t0 = time.perf_counter()
    result = anneal.cluster_search(energy, gradient, len(z), BUDGET[name], seed=seed, recommended=True,
                                   start=x0.reshape(-1), species=z, groups=hc.rigid_groups(mol), **kwargs)
    wall = time.perf_counter() - t0
    harvest.close()
    mx = [x for x, _ in harvest.minima] + [np.asarray(result["best"])]
    me = [e * hc.KCAL_PER_EV for _, e in harvest.minima] + [result["best_energy"] * hc.KCAL_PER_EV]
    np.savez_compressed(Path(out) / f"{name}_{seed}.npz", mx=np.asarray(mx), me=np.asarray(me),
                        lx=np.asarray(low_x).reshape(-1, 3 * len(z)), le=np.asarray(low_e),
                        calls=harvest.calls, wall=wall, pot=pot[0])


class Compare:
    def __init__(self, name: str) -> None:
        self.m = Molecule(hc.molecules()[name])
        mol = self.m.mol
        self.rx, self.e_ref = reference(name)
        heavy = set(self.m.heavy)
        amide = {frozenset(b) for b in mol.GetSubstructMatches(AMIDE)}
        self.torsions = []
        for b, c in mol.GetSubstructMatches(ROTATABLE):
            nb = [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != c and n.GetIdx() in heavy]
            nc = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != b and n.GetIdx() in heavy]
            if nb and nc:
                kind = "amides" if frozenset((b, c)) in amide else "rotors"
                self.torsions.append((kind, (max(nb), b, c, max(nc))))
        self.rings = []
        for ring in mol.GetRingInfo().AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                continue
            n = len(ring)
            self.rings.append([tuple(ring[(k + j) % n] for j in range(4)) for k in range(n)])
        self.donors = [(h.GetIdx(), h.GetNeighbors()[0].GetIdx()) for h in mol.GetAtoms()
                       if h.GetAtomicNum() == 1 and h.GetNeighbors()[0].GetAtomicNum() in (7, 8)]
        self.acceptors = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8)]
        self.bonds_apart = Chem.GetDistanceMatrix(mol)

    def align(self, x: np.ndarray) -> tuple[float, dict[int, int]]:
        heavy = np.asarray(self.m.heavy)
        ha = self.rx[heavy] - self.rx[heavy].mean(0)
        hb_all = x[heavy] - x[heavy].mean(0)
        best, best_perm = np.inf, None
        for perm in self.m.perms:
            hb = hb_all[perm]
            u, _, vt = np.linalg.svd(hb.T @ ha)
            if np.linalg.det(u @ vt) < 0:
                u[:, -1] *= -1
            d = hb @ (u @ vt) - ha
            value = float(np.sqrt((d * d).sum() / len(ha)))
            if value < best:
                best, best_perm = value, perm
        return best, {int(heavy[i]): int(heavy[j]) for i, j in enumerate(best_perm)}

    @staticmethod
    def dihedral(x, a, b, c, d) -> float:
        b0, b1, b2 = x[a] - x[b], x[c] - x[b], x[d] - x[c]
        b1 = b1 / np.linalg.norm(b1)
        v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
        return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))

    def hbonds(self, x: np.ndarray) -> set[tuple[int, int]]:
        out = set()
        for h, donor in self.donors:
            for a in self.acceptors:
                if a == donor or self.bonds_apart[donor, a] < 3 or np.linalg.norm(x[h] - x[a]) >= 2.5:
                    continue
                u, v = x[donor] - x[h], x[a] - x[h]
                if np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))) > 110:
                    out.add((donor, a))
        return out

    def differ(self, x: np.ndarray) -> dict:
        _, amap = self.align(x)
        to = lambda i: amap.get(i, i)
        apart = lambda a, b: abs((a - b + 180) % 360 - 180)
        out = {"rotors": 0, "amides": 0}
        for kind, (a, b, c, d) in self.torsions:
            if apart(self.dihedral(self.rx, a, b, c, d), self.dihedral(x, to(a), to(b), to(c), to(d))) > 45:
                out[kind] += 1
        out["rings"] = sum(any(apart(self.dihedral(self.rx, *q), self.dihedral(x, *map(to, q))) > 30 for q in ring)
                           for ring in self.rings)
        ref = self.hbonds(self.rx)
        back = {v: k for k, v in amap.items()}
        mine = {(back.get(d, d), back.get(a, a)) for d, a in self.hbonds(x)}
        out.update(hbonds_reference=len(ref), hbonds_lost=len(ref - mine), hbonds_new=len(mine - ref))
        return out


_COMPARE: dict[str, Compare] = {}


def analyse_one(args) -> dict:
    path, arm = args
    name, seed = path.stem.rsplit("_", 1)
    c = _COMPARE.setdefault(name, Compare(name))
    d = np.load(path)
    n = c.m.mol.GetNumAtoms()
    mx, me = d["mx"].reshape(-1, n, 3), d["me"]
    polished = [c.m.polish(mx[i]) for i in np.where(me < c.e_ref + POLISH_WINDOW)[0]]
    valid = [(e, x) for x, e, ok in polished if ok]
    best = min(valid, key=lambda t: t[0]) if valid else None
    lx = d["lx"].reshape(-1, n, 3)
    step = max(1, len(lx) // 400)
    row = {
        "molecule": name, "arm": arm, "seed": int(seed),
        "hit": int(best is not None and best[0] - c.e_ref <= HIT),
        "gap_kcal": None if best is None else best[0] - c.e_ref,
        "invalid_low": sum(1 for _, e, ok in polished if not ok and e - c.e_ref <= HIT),
        "near_minimum": min((c.align(mx[i])[0] for i in np.where(me < c.e_ref + NEAR_WINDOW)[0]), default=None),
        "near_evaluation": min((c.align(lx[i].astype(float))[0] for i in range(0, len(lx), step)), default=None),
        "minima": len(me), "low_evaluations": len(lx), "calls": int(d["calls"]),
        "wall_s": float(d["wall"]), "pot_s": float(d["pot"]),
    }
    if best is not None and not row["hit"]:
        row.update(c.differ(best[1]))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--molecules", required=True)
    run.add_argument("--seeds", required=True, help="first-last, inclusive")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--kwargs", default="{}", help="extra cluster_search keywords, JSON")
    run.add_argument("--workers", type=int, default=7)
    ana = sub.add_parser("analyse")
    ana.add_argument("--runs", type=Path, required=True)
    ana.add_argument("--molecules", required=True)
    ana.add_argument("--rows", type=Path, required=True, help="appended to")
    ana.add_argument("--arm", required=True)
    ana.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    names = args.molecules.split(",")
    if args.cmd == "run":
        lo, hi = map(int, args.seeds.split("-"))
        os.makedirs(args.out, exist_ok=True)
        jobs = [(n, s, args.out, json.loads(args.kwargs)) for n in names for s in range(lo, hi + 1)]
        with ProcessPoolExecutor(args.workers) as ex:
            list(ex.map(run_one, jobs))
        return 0
    files = sorted(p for n in names for p in args.runs.glob(f"{n}_*.npz"))
    new = not args.rows.exists()
    with ProcessPoolExecutor(args.workers) as ex, args.rows.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        for row in ex.map(analyse_one, [(p, args.arm) for p in files]):
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else ("" if v is None else v))
                             for k, v in ((k, row.get(k)) for k in FIELDS)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
