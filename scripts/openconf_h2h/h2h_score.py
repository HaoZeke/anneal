#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Score every arm's candidates by one rule.

Each candidate geometry is polished on MMFF94 (RDKit, dielectric one) until
RDKit's minimiser reports convergence. A polished geometry whose
stereocentres or double bonds differ from the input SMILES is invalid and
dropped. Two valid geometries are the same conformer when their heavy-atom
RMSD, minimised over graph automorphisms after a Kabsch fit, is below
0.25 Å and their energies differ by less than 0.5 kcal/mol; the lower one
is kept.

Per molecule, E_ref is the lowest valid polished energy over every arm and
seed. The reference ensemble is the deduplicated union of every run's
conformers within 3 kcal/mol of E_ref. Per run:

- ``delta_min_kcal``: lowest polished energy minus E_ref
- ``n_3kcal``: distinct conformers within 3 kcal/mol of E_ref
- ``coverage``: share of the reference ensemble that the run found
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2h_common as hc  # noqa: E402

RMSD = 0.25
ENERGY_TIE = 0.5
WINDOW = 3.0


def _stereo_key(mol: Chem.Mol) -> str:
    heavy = Chem.RemoveHs(mol)
    for atom in heavy.GetAtoms():
        if atom.GetAtomicNum() == 7:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.MolToSmiles(heavy)


class Molecule:
    def __init__(self, row: dict) -> None:
        self.row = row
        self.mol = hc.build_mol(row["smiles"])
        self.reference = _stereo_key(Chem.MolFromSmiles(row["smiles"]))
        self.heavy = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() > 1]
        # RemoveHs keeps stereo-defining hydrogens; the permutations must index self.heavy.
        heavy_mol = Chem.RemoveAllHs(self.mol)
        perms = heavy_mol.GetSubstructMatches(
            heavy_mol, uniquify=False, useChirality=False, maxMatches=100000
        )
        self.perms = np.asarray(perms, dtype=np.intp)

    def polish(self, x: np.ndarray) -> tuple[np.ndarray, float, bool]:
        mol = Chem.Mol(self.mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (a, b, c) in enumerate(np.asarray(x).reshape(-1, 3)):
            conf.SetAtomPosition(i, (float(a), float(b), float(c)))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        ff = hc.mmff94(mol)
        status = 1
        for _ in range(20):
            status = ff.Minimize(maxIts=2000)
            if status == 0:
                break
        energy = float(ff.CalcEnergy())
        pos = np.asarray(ff.Positions()).reshape(-1, 3)
        for i, (a, b, c) in enumerate(pos):
            mol.GetConformer().SetAtomPosition(i, (float(a), float(b), float(c)))
        Chem.AssignStereochemistryFrom3D(mol)
        valid = _stereo_key(mol) == self.reference and np.isfinite(energy)
        return pos, energy, bool(valid and status == 0)

    def rmsd(self, a: np.ndarray, b: np.ndarray) -> float:
        """Heavy-atom RMSD of b onto a, minimised over automorphisms."""
        ha = a[self.heavy]
        ha = ha - ha.mean(axis=0)
        hb_all = b[self.heavy]
        hb_all = hb_all - hb_all.mean(axis=0)
        best = np.inf
        for perm in self.perms:
            hb = hb_all[perm]
            h = hb.T @ ha
            u, _, vt = np.linalg.svd(h)
            if np.linalg.det(u @ vt) < 0:
                u[:, -1] *= -1
            diff = hb @ (u @ vt) - ha
            value = float(np.sqrt((diff * diff).sum() / len(ha)))
            if value < best:
                best = value
        return best

    def same(self, a, b) -> bool:
        return abs(a[1] - b[1]) < ENERGY_TIE and self.rmsd(a[0], b[0]) < RMSD

    def dedupe(self, items: list[tuple[np.ndarray, float]]) -> list[tuple[np.ndarray, float]]:
        kept: list[tuple[np.ndarray, float]] = []
        for item in sorted(items, key=lambda t: t[1]):
            if not any(self.same(item, other) for other in kept if item[1] - other[1] < ENERGY_TIE):
                kept.append(item)
        return kept


def _jobs(runs: Path):
    for path in sorted(runs.glob("*/*/*/job.json")):
        yield path.parent, json.loads(path.read_text())


def score(runs: Path, seeds_out: Path, summary_out: Path, polish_cache: Path) -> None:
    mols = {name: Molecule(row) for name, row in hc.molecules().items()}
    per_run: list[dict] = []
    polished: dict[tuple[str, str, int], list[tuple[np.ndarray, float]]] = {}
    for job_dir, info in _jobs(runs):
        name, arm, seed = info["molecule"], info["arm"], int(info["seed"])
        m = mols[name]
        cache = job_dir / "polished.npz"
        if cache.is_file() and polish_cache.exists():
            data = np.load(cache)
            xs, es, ok = data["x"], data["e"], data["ok"]
        else:
            xs, es, ok = [], [], []
            path = job_dir / "candidates.xyz"
            frames = hc.read_xyz(path) if path.is_file() else []
            for _, x, _ in frames:
                px, pe, pv = m.polish(x)
                xs.append(px)
                es.append(pe)
                ok.append(pv)
            xs = np.asarray(xs).reshape(-1, m.mol.GetNumAtoms(), 3)
            es, ok = np.asarray(es), np.asarray(ok, dtype=bool)
            np.savez(cache, x=xs, e=es, ok=ok)
        valid = [(x, float(e)) for x, e, v in zip(xs, es, ok, strict=True) if v]
        polished[(name, arm, seed)] = m.dedupe(valid)
        per_run.append(
            {
                "molecule": name,
                "arm": arm,
                "seed": seed,
                "candidates": int(len(es)),
                "invalid": int((~ok).sum()) if len(es) else 0,
                **{k: info.get(k) for k in ("wall_s", "pot_s", "outside_s", "grad_calls", "energy_calls", "embed_s", "budget", "hostname")},
            }
        )
    references = {}
    for name, m in mols.items():
        pool = [item for (n, _, _), items in polished.items() if n == name for item in items]
        if not pool:
            continue
        e_ref = min(e for _, e in pool)
        ensemble = m.dedupe([item for item in pool if item[1] - e_ref <= WINDOW])
        references[name] = (e_ref, ensemble)
    for row in per_run:
        name = row["molecule"]
        if name not in references:
            continue
        e_ref, ensemble = references[name]
        mine = polished[(name, row["arm"], row["seed"])]
        m = mols[name]
        row["e_ref_kcal"] = e_ref
        row["delta_min_kcal"] = (min(e for _, e in mine) - e_ref) if mine else None
        window = [item for item in mine if item[1] - e_ref <= WINDOW]
        row["n_3kcal"] = len(window)
        found = sum(1 for ref in ensemble if any(m.same(ref, item) for item in window))
        row["reference_3kcal"] = len(ensemble)
        row["coverage"] = found / len(ensemble) if ensemble else None
    fields = [
        "molecule", "arm", "seed", "delta_min_kcal", "n_3kcal", "reference_3kcal", "coverage",
        "candidates", "invalid", "grad_calls", "energy_calls", "budget", "wall_s", "pot_s",
        "outside_s", "embed_s", "e_ref_kcal", "hostname",
    ]
    with seeds_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in sorted(per_run, key=lambda r: (r["molecule"], r["arm"], r["seed"])):
            writer.writerow({k: _fmt(row.get(k)) for k in fields})
    summary_fields = [
        "molecule", "arm", "n", "e_ref_kcal", "delta_min_median_kcal", "delta_min_best_kcal",
        "hits_0p1", "n_3kcal_median", "reference_3kcal", "coverage_median", "grad_calls_median",
        "wall_median_s", "pot_median_s", "outside_median_s", "pot_per_grad_s", "outside_per_grad_s",
        "invalid_total",
    ]
    order = list(mols)
    extra = sorted({r["arm"] for r in per_run} - {"anneal", "openconf", "crest"})
    arms = ("anneal", *extra, "openconf", "crest")
    with summary_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        for name in order:
            for arm in arms:
                rows = [r for r in per_run if r["molecule"] == name and r["arm"] == arm and r.get("delta_min_kcal") is not None]
                if not rows:
                    continue

                def med(key):
                    vals = [r[key] for r in rows if r.get(key) is not None]
                    return statistics.median(vals) if vals else None

                grad_total = sum(r["grad_calls"] or 0 for r in rows)
                pot_total = sum(r["pot_s"] or 0 for r in rows)
                out_total = sum(r["outside_s"] or 0 for r in rows)
                writer.writerow(
                    {
                        "molecule": name,
                        "arm": arm,
                        "n": len(rows),
                        "e_ref_kcal": _fmt(rows[0]["e_ref_kcal"]),
                        "delta_min_median_kcal": _fmt(med("delta_min_kcal")),
                        "delta_min_best_kcal": _fmt(min(r["delta_min_kcal"] for r in rows)),
                        "hits_0p1": f"{sum(r['delta_min_kcal'] <= 0.1 for r in rows)}/{len(rows)}",
                        "n_3kcal_median": _fmt(med("n_3kcal")),
                        "reference_3kcal": rows[0]["reference_3kcal"],
                        "coverage_median": _fmt(med("coverage")),
                        "grad_calls_median": _fmt(med("grad_calls")),
                        "wall_median_s": _fmt(med("wall_s")),
                        "pot_median_s": _fmt(med("pot_s")),
                        "outside_median_s": _fmt(med("outside_s")),
                        "pot_per_grad_s": _fmt(pot_total / grad_total if grad_total and pot_total else None, 9),
                        "outside_per_grad_s": _fmt(out_total / grad_total if grad_total and out_total else None, 9),
                        "invalid_total": sum(r["invalid"] for r in rows),
                    }
                )


def _fmt(value, digits: int = 6):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seeds-out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--repolish", action="store_true", help="ignore polished.npz caches")
    args = parser.parse_args()
    cache_flag = Path("/nonexistent") if args.repolish else args.runs
    score(args.runs, args.seeds_out, args.summary, cache_flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
