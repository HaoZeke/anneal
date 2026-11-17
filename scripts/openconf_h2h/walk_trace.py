# Copyright (c) anneal developers
"""Why a molecular chain's walk misses the lowest conformer.

``run`` repeats a development-seed search of ``recommended=True`` with the hop
trace on and keeps, for every hop, what the walk did: the energies before, after
the partial relaxation and after the quench, whether the partial relaxation fell
back into the walker's own basin, whether the screen refused it, the largest
gradient component of the quench, the acceptance and the quenched geometry. It
keeps every harvested minimum as well. Extra ``cluster_search`` keywords come as
JSON. ``--budgets`` takes each pair's budget from a paired run's ``seeds.csv``.

``diagnose`` reads those runs against reference minima, one xyz frame per
molecule with ``name energy_kcal`` on the comment line, and writes one CSV row
per run:

* where the hops go: the share that fell back into the walker's basin before the
  quench, the share the screen refused, the share quenched in full, and of the
  quenched hops the share accepted, in each third of the run;
* how far a quenched proposal moved the walker, as the number of rotatable
  torsions that changed by more than 45 degrees and of non-aromatic rings with a
  dihedral that changed by more than 30, and the acceptance at each count;
* whether the quench converged: the share of full quenches whose largest
  gradient component exceeds ``GRADIENT_TOL``, and how far polishing lowers the
  best harvested minimum;
* the energy gap of the lowest valid polished minimum to the reference, which
  torsions and rings it has wrong, and how many of the wrong torsion pairs sit
  within two bonds of each other, against the share of all pairs that do;
* how many torsions the walker has wrong at a quarter, a half and the end of
  the run, against the best minimum found.

    walk_trace.py run --molecules lopinavir,lorlatinib --seeds 3000 3120 --out DIR [--kw JSON]
    walk_trace.py diagnose DIR --references refs.xyz --out rows.csv --arm NAME
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2h_common as hc
from h2h_job import RECORD_GRADIENT, Harvest
from h2h_score import Molecule

BUDGET = {
    "915": 24205,
    "fluvastatin": 39708,
    "amprenavir": 37880,
    "lorlatinib": 41771,
    "xld": 42076,
    "rq3": 49297,
    "z34": 43421,
    "lopinavir": 45759,
}
HIT = 0.1
POLISH_WINDOW = 3.0
TORSION_OFF = 45.0
RING_OFF = 30.0
GRADIENT_TOL = 1.0e-3
SAME_BASIN = 0.02
OPTIONAL = ("temperature", "delta")
TRACE = ("spent", "e", "e_screen", "e_new", "best", "drift", "max_gradient", "temperature", "delta")
FLAGS = ("returning", "screened", "unquenched", "accept")


def run_one(job: tuple[str, int, int, str, dict]) -> str:
    import anneal

    name, seed, budget, out, kw = job
    mol = hc.build_mol(hc.molecules()[name]["smiles"])
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    start = hc.embed(mol, seed).reshape(-1)
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

    os.environ["ANNEAL_HOP_TRACE"] = "1"
    saved = os.dup(2)
    with tempfile.TemporaryFile("w+") as trace:
        os.dup2(trace.fileno(), 2)
        try:
            result = anneal.cluster_search(
                energy,
                gradient,
                len(z),
                budget,
                seed=seed,
                recommended=True,
                start=start,
                species=z,
                groups=hc.rigid_groups(mol),
                **kw,
            )
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        trace.seek(0)
        lines = trace.read().splitlines()
    harvest.close()
    cols = {k: [] for k in (*TRACE, *FLAGS)}
    xs, turned, changed = [], [], []
    for line in lines:
        if line.startswith("HOPTRACE "):
            d = json.loads(line[9:])
            for key, column in cols.items():
                column.append(d.get(key, float("nan")) if key in OPTIONAL else d[key])
            xs.append(np.asarray(d["x"], dtype=np.float32))
            turned.append(json.dumps(d.get("turned", [])))
            changed.append(json.dumps(d.get("changed", [])))
    k = hc.KCAL_PER_EV
    minima_x = [x for x, _ in harvest.minima] + [np.asarray(result["best"])]
    minima_e = [e * k for _, e in harvest.minima] + [result["best_energy"] * k]
    energies = ("e", "e_screen", "e_new", "best", "temperature", "delta")
    np.savez_compressed(
        Path(out) / f"{name}_{seed}.npz",
        minima_x=np.asarray(minima_x),
        minima_e=np.asarray(minima_e),
        x=np.asarray(xs),
        turned=np.asarray(turned),
        changed=np.asarray(changed),
        calls=harvest.calls,
        hops=result["hops"],
        host=socket.gethostname(),
        **{
            key: np.asarray(column) * (k if key in energies else 1)
            for key, column in cols.items()
        },
    )
    return f"{name} {seed}"


def circular(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def dihedrals(x: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Dihedrals in degrees of ``x`` (frames, atoms, 3) over rows of atom quadruples."""
    p = x[:, quads]
    b0 = p[:, :, 0] - p[:, :, 1]
    b1 = p[:, :, 2] - p[:, :, 1]
    b2 = p[:, :, 3] - p[:, :, 2]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    y = (np.cross(b1, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, (v * w).sum(-1)))


class Reference:
    """One molecule's reference minimum, rotatable torsions and flexible rings."""

    def __init__(self, name: str, x: np.ndarray, energy: float) -> None:
        from rdkit import Chem

        self.m = Molecule(hc.molecules()[name])
        mol = self.m.mol
        self.x, self.energy = x, energy
        heavy = set(self.m.heavy)
        pattern = Chem.MolFromSmarts("[!D1&!$(*#*)]-&!@[!D1&!$(*#*)]")
        quads = []
        for b, c in mol.GetSubstructMatches(pattern):
            nb = [
                n.GetIdx()
                for n in mol.GetAtomWithIdx(b).GetNeighbors()
                if n.GetIdx() != c and n.GetIdx() in heavy
            ]
            nc = [
                n.GetIdx()
                for n in mol.GetAtomWithIdx(c).GetNeighbors()
                if n.GetIdx() != b and n.GetIdx() in heavy
            ]
            if nb and nc:
                quads.append((max(nb), b, c, max(nc)))
        self.torsions = np.asarray(quads, dtype=np.intp).reshape(-1, 4)
        self.rings = []
        for ring in mol.GetRingInfo().AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                continue
            size = len(ring)
            self.rings.append(
                np.asarray(
                    [
                        (ring[i], ring[(i + 1) % size], ring[(i + 2) % size], ring[(i + 3) % size])
                        for i in range(size)
                    ],
                    dtype=np.intp,
                )
            )
        dist = Chem.GetDistanceMatrix(mol)
        mid = [(q[1], q[2]) for q in quads]
        self.near = np.zeros((len(mid), len(mid)), dtype=bool)
        for i, j in combinations(range(len(mid)), 2):
            gap = min(dist[a][b] for a in mid[i] for b in mid[j])
            self.near[i, j] = self.near[j, i] = gap <= 2
        self.base = dihedrals(x[None], self.torsions)[0]
        self.ring_base = [dihedrals(x[None], r)[0] for r in self.rings]

    def atom_map(self, x: np.ndarray) -> np.ndarray:
        """Atom of ``x`` that sits on each atom of the reference, over graph automorphisms."""
        heavy = np.asarray(self.m.heavy)
        ha = self.x[heavy] - self.x[heavy].mean(0)
        hb_all = x[heavy] - x[heavy].mean(0)
        best, chosen = np.inf, None
        for perm in self.m.perms:
            u, _, vt = np.linalg.svd(hb_all[perm].T @ ha)
            if np.linalg.det(u @ vt) < 0:
                u[:, -1] *= -1
            d = hb_all[perm] @ (u @ vt) - ha
            value = float((d * d).sum())
            if value < best:
                best, chosen = value, perm
        out = np.arange(self.x.shape[0])
        out[heavy] = heavy[np.asarray(chosen)]
        return out

    def wrong(self, x: np.ndarray) -> tuple[np.ndarray, list[int]]:
        amap = self.atom_map(x)
        t = circular(self.base, dihedrals(x[None], amap[self.torsions])[0]) > TORSION_OFF
        rings = [
            k
            for k, r in enumerate(self.rings)
            if (circular(self.ring_base[k], dihedrals(x[None], amap[r])[0]) > RING_OFF).any()
        ]
        return t, rings


def references(path: Path) -> dict[str, Reference]:
    out = {}
    for _, x, comment in hc.read_xyz(path):
        name, energy = comment.split()
        out[name] = Reference(name, x, float(energy))
    return out


REFS: dict[str, Reference] = {}


def share(mask: np.ndarray) -> float:
    return round(float(mask.mean()), 4) if mask.size else float("nan")


def diagnose_one(path: str) -> dict:
    d = np.load(path)
    name, seed = Path(path).stem.rsplit("_", 1)
    ref = REFS[name]
    n = ref.x.shape[0]
    xs, es = d["minima_x"].reshape(-1, n, 3), d["minima_e"]
    polished = [(i, *ref.m.polish(xs[i])) for i in np.where(es < ref.energy + POLISH_WINDOW)[0]]
    valid = [p for p in polished if p[3]]
    if valid:
        idx, bx, be, ok = min(valid, key=lambda p: p[2])
    else:
        idx = len(xs) - 1
        bx, be, ok = ref.m.polish(xs[idx])
    gap = be - ref.energy
    wrong, rings = ref.wrong(bx)
    ids = np.where(wrong)[0]
    pairs = [(i, j) for i, j in combinations(ids, 2)]
    near_wrong = float(np.mean([ref.near[i, j] for i, j in pairs])) if pairs else float("nan")
    all_pairs = ref.near[np.triu_indices(len(ref.torsions), 1)]

    hops = len(d["e"])
    ret, scr, unq = (d[k].astype(bool) for k in ("returning", "screened", "unquenched"))
    acc, e, e_new = d["accept"].astype(bool), d["e"], d["e_new"]
    quenched = ~unq
    third = np.minimum((np.arange(hops) * 3) // max(hops, 1), 2)
    row = {
        "molecule": name,
        "seed": int(seed),
        "energy_hit": int(bool(valid) and gap <= HIT),
        "gap_kcal": round(gap, 4),
        "hops": int(d["hops"]),
        "returning_share": share(ret),
        "screened_share": share(scr & ~ret),
        "quenched_share": share(quenched),
        "quench_same_basin": share(np.abs(e_new[quenched] - e[quenched]) < SAME_BASIN),
        "accept_quenched": share(acc[quenched]),
    }
    for t in range(3):
        sel = quenched & (third == t)
        row[f"accept_third{t + 1}"] = share(acc[sel])
        row[f"returning_third{t + 1}"] = share(ret[third == t])
    grad = d["max_gradient"][quenched]
    row["quench_unconverged"] = share(grad > GRADIENT_TOL)
    row["polish_drop_kcal"] = round(float(es[idx] - be), 4)
    row["valid"] = int(ok)

    walker = np.full(hops, -1)
    last = -1
    for h in range(hops):
        walker[h] = last
        if acc[h]:
            last = h
    use = np.where(quenched & (walker >= 0))[0]
    moved = np.zeros(len(use), dtype=int)
    ringed = np.zeros(len(use), dtype=int)
    if len(use):
        cur = d["x"][walker[use]].reshape(-1, n, 3).astype(np.float64)
        new = d["x"][use].reshape(-1, n, 3).astype(np.float64)
        if len(ref.torsions):
            moved = (circular(dihedrals(cur, ref.torsions), dihedrals(new, ref.torsions)) > TORSION_OFF).sum(1)
        for r in ref.rings:
            ringed += (circular(dihedrals(cur, r), dihedrals(new, r)) > RING_OFF).any(1)
    changed = moved + ringed
    row["moved_mean"] = round(float(changed.mean()), 3) if len(use) else float("nan")
    for k in range(4):
        sel = changed == k if k < 3 else changed >= 3
        row[f"moved{k}_share"] = share(sel)
        row[f"moved{k}_accept"] = share(acc[use][sel])
    row["ring_moved_share"] = share(ringed > 0)
    row["ring_moved_accept"] = share(acc[use][ringed > 0])

    for tag, frac in (("q1", 0.25), ("q2", 0.5), ("q4", 1.0)):
        h = int(frac * hops) - 1
        w = walker[h] if h < hops and walker[h] >= 0 else -1
        if w >= 0:
            wx = d["x"][w].reshape(n, 3).astype(np.float64)
            row[f"walker_wrong_{tag}"] = int(ref.wrong(wx)[0].sum())
        else:
            row[f"walker_wrong_{tag}"] = -1
    row["wrong_torsions"] = len(ids)
    row["wrong_rings"] = len(rings)
    row["wrong_pairs_near"] = round(near_wrong, 4) if pairs else ""
    row["all_pairs_near"] = round(float(all_pairs.mean()), 4)
    above = e - np.minimum.accumulate(e)
    row["near_own_best"] = share(above < 1.0)
    row["above_own_best_kcal"] = round(float(np.median(above)), 4)
    row["host"] = str(d["host"])
    return row


def init_diagnose(path: str) -> None:
    REFS.update(references(Path(path)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--molecules", required=True)
    run.add_argument("--seeds", nargs=2, type=int, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--kw", default="{}")
    run.add_argument("--budgets", type=Path, help="seeds.csv of a paired run")
    run.add_argument("--jobs", type=int, default=1)
    diag = sub.add_parser("diagnose")
    diag.add_argument("runs", type=Path)
    diag.add_argument("--references", type=Path, required=True)
    diag.add_argument("--out", type=Path, required=True)
    diag.add_argument("--arm", required=True)
    diag.add_argument("--revision", required=True)
    diag.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.cmd == "run":
        args.out.mkdir(parents=True, exist_ok=True)
        kw = json.loads(args.kw)
        budgets = {}
        if args.budgets:
            with args.budgets.open() as fh:
                budgets = {
                    (r["molecule"], int(r["seed"])): int(r["budget"])
                    for r in csv.DictReader(fh)
                    if r["budget"]
                }
        jobs = [
            (m, s, budgets[(m, s)] if args.budgets else BUDGET[m], str(args.out), kw)
            for m in args.molecules.split(",")
            for s in range(*args.seeds)
        ]
        with ProcessPoolExecutor(args.jobs) as ex:
            for done in ex.map(run_one, jobs):
                print(done, flush=True)
        return
    files = sorted(str(p) for p in args.runs.glob("*.npz"))
    with ProcessPoolExecutor(
        args.jobs, initializer=init_diagnose, initargs=(str(args.references),)
    ) as ex:
        rows = list(ex.map(diagnose_one, files))
    rows.sort(key=lambda r: (r["molecule"], r["seed"]))
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["arm", *rows[0], "code_revision"], lineterminator="\n"
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({"arm": args.arm, **r, "code_revision": args.revision})


if __name__ == "__main__":
    main()
