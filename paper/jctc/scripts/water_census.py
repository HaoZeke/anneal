#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Basin census of a water hexamer search against the CREST minimum.

Usage:
  water_census.py hops <crest_best.xyz> <hoptrace.log> [<label>]
  water_census.py relaxed <crest_best.xyz> <hoptrace.log> <label> <workdir>
  water_census.py compare <reference.xyz> <trial.xyz> [...]

The hop mode reads the HOPTRACE lines that ANNEAL_HOP_TRACE=1 writes,
one per hop with the quenched geometry, and says for every hop how far
the geometry is from the CREST minimum. The distance is the all-atom
RMSD minimised over rotations, reflections, the 720 relabellings of
the six molecules and the two labellings of each pair of hydrogens.
A mirror image is the same minimum on a non-chiral potential.

Two structures are one minimum when they pass the census radius the
water arm already uses, 0.1 angstrom on the sorted oxygen-oxygen
distances (OXYGEN_DISTANCE_CENSUS_RADIUS_A in src/catalog/molecular.rs),
and the permutation-invariant RMSD reported beside it says the same.

Hydrogen bonds use the Luzar and Chandler geometric definition,
Nature 379, 55 (1996): oxygen-oxygen distance below 3.5 angstrom and
the angle between the donor O-H bond and the O-O axis below 30 degrees.
"""

import itertools
import json
import math
import re
import sys
from collections import Counter

import numpy as np

CENSUS_RADIUS_A = 0.1
HBOND_OO_A = 3.5
HBOND_ANGLE_DEG = 30.0
EV_PER_HARTREE = 27.211386245988
PERMUTATIONS = np.array(list(itertools.permutations(range(6))))


def read_xyz(path):
    lines = open(path).read().splitlines()
    count = int(lines[0].split()[0])
    symbols = []
    rows = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        symbols.append(parts[0])
        rows.append([float(value) for value in parts[1:4]])
    return symbols, np.array(rows)


def molecules(symbols, coordinates):
    """(m, 3, 3) array of [O, H, H] with each hydrogen on its nearest oxygen."""
    oxygens = [i for i, s in enumerate(symbols) if s.upper().startswith("O")]
    hydrogens = [i for i, s in enumerate(symbols) if s.upper().startswith("H")]
    owned = {o: [] for o in oxygens}
    for h in hydrogens:
        nearest = min(oxygens, key=lambda o: np.linalg.norm(coordinates[h] - coordinates[o]))
        owned[nearest].append(h)
    out = []
    for o in oxygens:
        if len(owned[o]) != 2:
            return None
        out.append([coordinates[o], coordinates[owned[o][0]], coordinates[owned[o][1]]])
    return np.array(out)


def from_trace(x):
    return molecules(["O", "H", "H"] * (len(x) // 9), np.array(x).reshape(-1, 3))


def sorted_oo(mol):
    oxygens = mol[:, 0, :]
    return np.sort(
        [np.linalg.norm(oxygens[i] - oxygens[j]) for i in range(len(oxygens)) for j in range(i + 1, len(oxygens))]
    )


def hbonds(mol):
    count = 0
    cos_limit = math.cos(math.radians(HBOND_ANGLE_DEG))
    for d in range(len(mol)):
        for a in range(len(mol)):
            if a == d:
                continue
            axis = mol[a, 0] - mol[d, 0]
            distance = np.linalg.norm(axis)
            if distance >= HBOND_OO_A:
                continue
            for h in (1, 2):
                bond = mol[d, h] - mol[d, 0]
                if np.dot(bond, axis) / (np.linalg.norm(bond) * distance) > cos_limit:
                    count += 1
    return count


def _kabsch_batch(trial, reference, proper):
    covariance = np.einsum("pni,nj->pij", trial, reference)
    u, _, vt = np.linalg.svd(covariance)
    if proper:
        sign = np.sign(np.linalg.det(u @ vt))
        u[:, :, 2] *= sign[:, None]
    rotation = u @ vt
    aligned = np.einsum("pni,pij->pnj", trial, rotation)
    return rotation, np.sqrt(((aligned - reference) ** 2).sum(axis=(1, 2)) / reference.shape[0])


def perm_rmsd(reference, trial):
    """All-atom RMSD minimised over molecule order, hydrogen order, rotation and reflection."""
    ref = reference.reshape(-1, 3) - reference.reshape(-1, 3).mean(axis=0)
    ref = ref.reshape(reference.shape)
    best = math.inf
    for mirror in (False, True):
        tri = trial.copy()
        if mirror:
            tri[:, :, 0] *= -1.0
        tri = tri - tri.reshape(-1, 3).mean(axis=0)
        ordered = tri[PERMUTATIONS]
        _, oxygen_rmsd = _kabsch_batch(ordered[:, :, 0, :], ref[:, 0, :], True)
        for p in np.argsort(oxygen_rmsd)[:24]:
            candidate = ordered[p].copy()
            for _ in range(3):
                flat = candidate.reshape(-1, 3)
                rotation, _ = _kabsch_batch(flat[None], ref.reshape(-1, 3), True)
                rotated = (flat @ rotation[0]).reshape(candidate.shape)
                for k in range(len(rotated)):
                    keep = np.sum((rotated[k, 1:] - ref[k, 1:]) ** 2)
                    swap = np.sum((rotated[k, [2, 1]] - ref[k, 1:]) ** 2)
                    if swap < keep:
                        candidate[k, [1, 2]] = candidate[k, [2, 1]]
            flat = candidate.reshape(-1, 3)
            _, value = _kabsch_batch(flat[None], ref.reshape(-1, 3), True)
            best = min(best, float(value[0]))
    return best


def label_free_rmsd(reference_path, trial_path):
    """perm_rmsd between two xyz files, or None when either is not intact water."""
    reference = molecules(*read_xyz(reference_path))
    trial = molecules(*read_xyz(trial_path))
    if reference is None or trial is None or reference.shape != trial.shape:
        return None
    return perm_rmsd(reference, trial)


def same_minimum(reference, trial):
    return float(np.max(np.abs(sorted_oo(reference) - sorted_oo(trial)))) <= CENSUS_RADIUS_A


def load_trace(path):
    hops = []
    for line in open(path, errors="ignore"):
        if not line.startswith("HOPTRACE "):
            continue
        text = line[len("HOPTRACE ") :]
        text = re.sub(r"(?<![\w.])-inf\b", "-Infinity", text)
        text = re.sub(r"(?<![\w.])inf\b", "Infinity", text)
        hops.append(json.loads(text))
    return hops


def hop_census(reference_path, trace_path, label):
    symbols, coordinates = read_xyz(reference_path)
    reference = molecules(symbols, coordinates)
    hops = load_trace(trace_path)
    reference_oo = sorted_oo(reference)
    rows = []
    for hop in hops:
        mol = from_trace(hop["x"])
        if mol is None:
            rows.append((hop, math.inf, math.inf, -1))
            continue
        oo_gap = float(np.max(np.abs(sorted_oo(mol) - reference_oo)))
        rows.append((hop, oo_gap, None, hbonds(mol)))
    hits = [r for r in rows if r[1] <= CENSUS_RADIUS_A]
    full = [r for r in rows if not r[0]["unquenched"]]
    recordable = [r for r in rows if r[0]["recordable"]]
    print(f"census {label}")
    print(f"reference {reference_path} hbonds {hbonds(reference)}")
    print(f"hops {len(rows)} full_quench {len(full)} recordable {len(recordable)} spent {hops[-1]['spent'] if hops else 0}")
    print("arms " + " ".join(f"{a}={n}" for a, n in sorted(Counter(r[0]['arm'] for r in rows).items())))
    print("hbond_hist_recordable " + " ".join(f"{k}:{v}" for k, v in sorted(Counter(r[3] for r in recordable).items())))
    print(
        "reference_hits total {} recordable {} recordable_accepted {} recordable_rejected {} unquenched {}".format(
            len(hits),
            sum(1 for r in hits if r[0]["recordable"]),
            sum(1 for r in hits if r[0]["recordable"] and r[0]["accept"]),
            sum(1 for r in hits if r[0]["recordable"] and not r[0]["accept"]),
            sum(1 for r in hits if r[0]["unquenched"]),
        )
    )
    nearest = sorted(rows, key=lambda r: r[1])[:5]
    for hop, gap, _, nhb in nearest:
        mol = from_trace(hop["x"])
        rmsd = perm_rmsd(reference, mol) if mol is not None else math.inf
        print(
            "nearest hop {} arm {} oo_gap {:.4f} perm_rmsd {:.4f} hbonds {} e_new {:.6f} unquenched {} recordable {} accept {} max_gradient {}".format(
                hop["hop"], hop["arm"], gap, rmsd, nhb, hop["e_new"], hop["unquenched"], hop["recordable"], hop["accept"], hop["max_gradient"]
            )
        )
    lowest = sorted(recordable, key=lambda r: r[0]["e_new"])
    distinct = []
    for hop, gap, _, nhb in lowest:
        mol = from_trace(hop["x"])
        if any(same_minimum(mol, kept) for kept, *_ in distinct):
            continue
        distinct.append((mol, hop, gap, nhb))
        if len(distinct) == 8:
            break
    for mol, hop, gap, nhb in distinct:
        print(
            "low_basin e_new {:.6f} hbonds {} oo_gap {:.4f} perm_rmsd {:.4f} arm {} hop {} max_gradient {}".format(
                hop["e_new"], nhb, gap, perm_rmsd(reference, mol), hop["arm"], hop["hop"], hop["max_gradient"]
            )
        )
    accepted = sum(1 for r in rows if r[0]["accept"])
    print(f"accepted {accepted} of {len(rows)}")
    return rows


def write_xyz(path, mol, comment=""):
    with open(path, "w") as handle:
        handle.write(f"{3 * len(mol)}\n{comment}\n")
        for molecule in mol:
            for symbol, row in zip(("O", "H", "H"), molecule):
                handle.write(f"{symbol} {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")


def xtb_relax(mol, work):
    """Relax with the scoring engine to its vtight level; (energy_Eh, molecules)."""
    import os
    import subprocess

    os.makedirs(work, exist_ok=True)
    write_xyz(os.path.join(work, "in.xyz"), mol)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run(
        [os.environ["XTB_BIN"], "in.xyz", "--gfn", "2", "--opt", "vtight", "--acc", "0.01", "--iterations", "500"],
        cwd=work,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out = os.path.join(work, "xtbopt.xyz")
    if not os.path.exists(out):
        return None, None
    symbols, coordinates = read_xyz(out)
    energy = float(open(out).read().splitlines()[1].split()[1])
    return energy, molecules(symbols, coordinates)


def relaxed_census(reference_path, trace_path, label, work):
    """Every recordable quench relaxed tightly, then classified against the reference."""
    symbols, coordinates = read_xyz(reference_path)
    reference = molecules(symbols, coordinates)
    hops = [h for h in load_trace(trace_path) if h["recordable"]]
    best_loose = min((h["e_new"] for h in hops), default=math.nan)
    rows = []
    for hop in hops:
        mol = from_trace(hop["x"])
        energy, relaxed = xtb_relax(mol, f"{work}/hop{hop['hop']}")
        if relaxed is None:
            continue
        rows.append((hop, energy, same_minimum(reference, relaxed), perm_rmsd(reference, relaxed)))
    hits = [r for r in rows if r[2]]
    print(f"relaxed_census {label} recordable {len(rows)} reference_basin_hits {len(hits)} best_loose_eV {best_loose:.6f}")
    for hop, energy, _, rmsd in hits:
        print(
            "hit hop {} spent {} arm {} loose_eV {:.6f} loose_above_best_meV {:.2f} relaxed_Eh {:.9f} perm_rmsd {:.4f} accept {}".format(
                hop["hop"], hop["spent"], hop["arm"], hop["e_new"], 1000 * (hop["e_new"] - best_loose), energy, rmsd, hop["accept"]
            )
        )
    winner = min(rows, key=lambda r: r[0]["e_new"]) if rows else None
    if winner:
        hop, energy, same, rmsd = winner
        print(f"ledger_best hop {hop['hop']} loose_eV {hop['e_new']:.6f} relaxed_Eh {energy:.9f} reference_basin {same} perm_rmsd {rmsd:.4f}")
    relaxed_best = min(rows, key=lambda r: r[1]) if rows else None
    if relaxed_best:
        hop, energy, same, rmsd = relaxed_best
        print(f"relaxed_best hop {hop['hop']} loose_eV {hop['e_new']:.6f} relaxed_Eh {energy:.9f} reference_basin {same}")
    first = min((r[0]["spent"] for r in hits), default=None)
    print(f"first_reference_hit_spent {first}")


def compare(reference_path, trial_paths):
    symbols, coordinates = read_xyz(reference_path)
    reference = molecules(symbols, coordinates)
    print(f"reference {reference_path} hbonds {hbonds(reference)}")
    for path in trial_paths:
        s, c = read_xyz(path)
        mol = molecules(s, c)
        gap = float(np.max(np.abs(sorted_oo(mol) - sorted_oo(reference))))
        print(f"trial {path} hbonds {hbonds(mol)} oo_gap {gap:.4f} perm_rmsd {perm_rmsd(reference, mol):.4f} same_minimum {gap <= CENSUS_RADIUS_A}")


def main():
    mode = sys.argv[1]
    if mode == "hops":
        hop_census(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else sys.argv[3])
    elif mode == "relaxed":
        relaxed_census(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "compare":
        compare(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
