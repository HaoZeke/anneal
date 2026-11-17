# Copyright (c) anneal developers
"""Where a molecular chain misses the lowest conformer, and what a stall return changes.

``run`` repeats a development-seed search of ``recommended=True`` at the
median budgets of ``bench/openconf-h2h/conf-fix4`` with the hop trace on,
keeping every harvested minimum and every hop: energies, acceptance and the
quenched geometry. Extra
``cluster_search`` keywords come as JSON, so ``'{"elite_return": false}'`` is
the develop search of a tree where the return is on. ``--budgets`` takes each
pair's budget from a paired run's ``seeds.csv`` instead.

``score`` reads those runs against reference minima, one xyz frame per
molecule with ``name energy_kcal`` on the comment line, and writes one CSV row
per run. ``energy_hit`` is set when a valid polished minimum among its
harvested minima within 3 kcal/mol of the reference lies within 0.1 kcal/mol
of it; a paired run's ``hit`` matches structures among the conformers the run
reports, so the two can differ on the same trajectory. The row
then describes the lowest valid polished minimum: which rotatable torsions
stand more than 45 degrees from the reference and which non-aromatic rings
have a dihedral more than 30 degrees off, after mapping atoms over graph
automorphisms; whether setting one of those torsions, or all of them, to the
reference and polishing reaches the reference; and how far greedy repair
gets, taking each time the single torsion whose setting lowers the polished
energy most. From the trace it reads how often the walker stands within
1 kcal/mol of its own running best, the median height above it, the share of
calls spent when the last new best fell by more than 0.05 kcal/mol, the share
spent when the first stall return moved the walker, and how many of the
torsions are wrong in every one of the eight lowest distinct accepted minima. ``host`` is where the run ran.

    elite_trace.py run --molecules 915,amprenavir --seeds 3000 3120 --out DIR [--kw JSON]
    elite_trace.py run --molecules 915 --seeds 2000 2024 --budgets seeds.csv --out DIR
    elite_trace.py score DIR --references refs.xyz --out rows.csv --arm NAME
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2h_common as hc
from h2h_job import RECORD_GRADIENT, Harvest
from h2h_score import Molecule

# Median search calls of the gate pairs, halved as in ``run_bench.paired``.
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
ROTOR = Chem.MolFromSmarts("[!D1&!$(*#*)]-&!@[!D1&!$(*#*)]")


def circular(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def dihedral(x: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    b0, b1, b2 = x[a] - x[b], x[c] - x[b], x[d] - x[c]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


class Reference:
    """One molecule's reference minimum, its rotatable torsions and its flexible rings."""

    def __init__(self, name: str, x: np.ndarray, energy: float) -> None:
        self.m = Molecule(hc.molecules()[name])
        mol = self.m.mol
        self.x, self.energy = x, energy
        heavy = set(self.m.heavy)
        self.torsions = []
        for b, c in mol.GetSubstructMatches(ROTOR):
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
                self.torsions.append((max(nb), b, c, max(nc)))
        self.rings = []
        for ring in mol.GetRingInfo().AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                continue
            k = len(ring)
            self.rings.append(
                [
                    (ring[i], ring[(i + 1) % k], ring[(i + 2) % k], ring[(i + 3) % k])
                    for i in range(k)
                ]
            )

    def atom_map(self, x: np.ndarray) -> dict[int, int]:
        """Heavy atom of the reference to the atom of ``x`` it sits on, over graph automorphisms."""
        heavy = np.asarray(self.m.heavy)
        ha = self.x[heavy] - self.x[heavy].mean(0)
        hb_all = x[heavy] - x[heavy].mean(0)
        best, chosen = np.inf, None
        for perm in self.m.perms:
            hb = hb_all[perm]
            u, _, vt = np.linalg.svd(hb.T @ ha)
            if np.linalg.det(u @ vt) < 0:
                u[:, -1] *= -1
            d = hb @ (u @ vt) - ha
            value = float((d * d).sum())
            if value < best:
                best, chosen = value, perm
        return {int(heavy[i]): int(heavy[j]) for i, j in enumerate(chosen)}

    def wrong(self, x: np.ndarray) -> tuple[list[int], list[int], dict[int, int]]:
        amap = self.atom_map(x)
        mp = lambda i: amap.get(i, i)
        torsions = [
            k
            for k, q in enumerate(self.torsions)
            if circular(dihedral(self.x, *q), dihedral(x, *map(mp, q))) > TORSION_OFF
        ]
        rings = [
            k
            for k, ring in enumerate(self.rings)
            if any(
                circular(dihedral(self.x, *q), dihedral(x, *map(mp, q))) > RING_OFF
                for q in ring
            )
        ]
        return torsions, rings, amap

    def set_torsions(
        self, x: np.ndarray, which: list[int], amap: dict[int, int]
    ) -> np.ndarray:
        mol = Chem.Mol(self.m.mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, p in enumerate(x):
            conf.SetAtomPosition(i, tuple(float(v) for v in p))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        c = mol.GetConformer()
        mp = lambda i: amap.get(i, i)
        for k in which:
            q = self.torsions[k]
            rdMolTransforms.SetDihedralDeg(c, *map(mp, q), dihedral(self.x, *q))
        return np.asarray(c.GetPositions())


def references(path: Path) -> dict[str, Reference]:
    out = {}
    for _, x, comment in hc.read_xyz(path):
        name, energy = comment.split()
        out[name] = Reference(name, x, float(energy))
    return out


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
    cols = {k: [] for k in ("spent", "e", "e_new", "best", "accept", "unquenched")}
    xs, returns, first = [], 0, -1
    for line in lines:
        if line.startswith("ELITETRACE "):
            first = len(xs) if returns == 0 else first
            returns += 1
        elif line.startswith("HOPTRACE "):
            d = json.loads(line[9:])
            for key, column in cols.items():
                column.append(d[key])
            xs.append(np.asarray(d["x"], dtype=np.float32))
    k = hc.KCAL_PER_EV
    minima_x = [x for x, _ in harvest.minima] + [np.asarray(result["best"])]
    minima_e = [e * k for _, e in harvest.minima] + [result["best_energy"] * k]
    np.savez_compressed(
        Path(out) / f"{name}_{seed}.npz",
        minima_x=np.asarray(minima_x),
        minima_e=np.asarray(minima_e),
        spent=np.asarray(cols["spent"]),
        e=np.asarray(cols["e"]) * k,
        e_new=np.asarray(cols["e_new"]) * k,
        best=np.asarray(cols["best"]) * k,
        accept=np.asarray(cols["accept"]),
        unquenched=np.asarray(cols["unquenched"]),
        x=np.asarray(xs),
        returns=returns,
        first_return=first,
        calls=harvest.calls,
        hops=result["hops"],
        host=socket.gethostname(),
    )
    return f"{name} {seed}"


def greedy(
    ref: Reference, x: np.ndarray, e: float, wrong: list[int]
) -> tuple[int, float]:
    steps = 0
    while wrong:
        trials = []
        for k in wrong:
            y = ref.set_torsions(x, [k], ref.atom_map(x))
            trials.append(ref.m.polish(y))
        px, pe, ok = min(trials, key=lambda t: t[1])
        if pe >= e - 1e-3 or not ok:
            break
        x, e, steps = px, pe, steps + 1
        wrong = ref.wrong(x)[0]
    return steps, e


def low_cover(ref: Reference, d) -> int:
    """Torsions wrong in every one of the eight lowest distinct accepted minima."""
    n = ref.x.shape[0]
    accepted = np.where(d["accept"] & ~d["unquenched"])[0]
    seen, common = set(), None
    for i in accepted[np.argsort(d["e_new"][accepted])]:
        t, r, _ = ref.wrong(d["x"][i].reshape(n, 3).astype(np.float64))
        key = (tuple(t), tuple(r))
        if key in seen:
            continue
        seen.add(key)
        common = set(t) if common is None else common & set(t)
        if len(seen) == 8:
            break
    return len(common or ())


REFS: dict[str, Reference] = {}


def score_one(path: str) -> dict:
    d = np.load(path)
    name, seed = Path(path).stem.rsplit("_", 1)
    ref = REFS[name]
    n = ref.x.shape[0]
    xs, es = d["minima_x"].reshape(-1, n, 3), d["minima_e"]
    polished = [
        ref.m.polish(xs[i]) for i in np.where(es < ref.energy + POLISH_WINDOW)[0]
    ]
    valid = [p for p in polished if p[2]]
    if valid:
        bx, be, ok = min(valid, key=lambda p: p[1])
    else:
        bx, be, ok = ref.m.polish(xs[-1])
    gap = be - ref.energy
    hit = bool(valid) and gap <= HIT
    torsions, rings, amap = ref.wrong(bx)
    single = any(
        p[2] and p[1] - ref.energy <= HIT
        for p in (ref.m.polish(ref.set_torsions(bx, [k], amap)) for k in torsions)
    )
    if torsions:
        _, ae, aok = ref.m.polish(ref.set_torsions(bx, torsions, amap))
        every = bool(aok and ae - ref.energy <= HIT)
    else:
        every = False
    steps, ge = greedy(ref, bx, be, torsions) if not hit else (0, be)
    e, spent, best = d["e"], d["spent"], d["best"]
    above = e - np.minimum.accumulate(e)
    drops = np.where(np.diff(best) < -0.05)[0]
    last = float(spent[drops[-1] + 1] / spent[-1]) if len(drops) else 0.0
    first = int(d["first_return"]) if "first_return" in d else -1
    first_share = round(float(spent[first - 1] / spent[-1]), 4) if first > 0 else ""
    return {
        "molecule": name,
        "seed": int(seed),
        "energy_hit": int(hit),
        "gap_kcal": round(gap, 4),
        "valid": int(ok),
        "hops": int(d["hops"]),
        "returns": int(d["returns"]),
        "first_return_share": first_share,
        "accepted_share": round(float(np.mean(d["accept"])), 4),
        "wrong_torsions": len(torsions),
        "wrong_torsion_ids": " ".join(map(str, torsions)),
        "wrong_rings": len(rings),
        "one_torsion_fix": int(single),
        "all_torsion_fix": int(every),
        "greedy_steps": steps,
        "greedy_gap_kcal": round(ge - ref.energy, 4),
        "near_own_best": round(float(np.mean(above < 1.0)), 4),
        "above_own_best_kcal": round(float(np.median(above)), 4),
        "current_above_reference_kcal": round(float(np.median(e - ref.energy)), 4),
        "last_new_best": round(last, 4),
        "low8_wrong_torsions": low_cover(ref, d),
        "host": str(d["host"]),
    }


def init_score(path: str) -> None:
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
    score = sub.add_parser("score")
    score.add_argument("runs", type=Path)
    score.add_argument("--references", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--arm", required=True)
    score.add_argument(
        "--revision", default=None, help="code revision of the runs; default HEAD"
    )
    score.add_argument("--jobs", type=int, default=1)
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
    revision = (
        args.revision
        or subprocess.run(
            ["git", "-C", str(hc.ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    files = sorted(str(p) for p in args.runs.glob("*.npz"))
    with ProcessPoolExecutor(
        args.jobs, initializer=init_score, initargs=(str(args.references),)
    ) as ex:
        rows = list(ex.map(score_one, files))
    rows.sort(key=lambda r: (r["molecule"], r["seed"]))
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["arm", *rows[0], "code_revision"], lineterminator="\n"
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({"arm": args.arm, **r, "code_revision": revision})


if __name__ == "__main__":
    main()
