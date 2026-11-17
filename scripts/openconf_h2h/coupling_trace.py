#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Which hops coupled torsion draws change, seed by seed, at the gate budgets.

Each (molecule, seed) of a paired ``seeds.csv`` runs twice at the budget the
gate gave it: the plain search (``torsion_coupling=False``) and coupled draws
at ``--patience``, both under ``recommended=True`` with ``ANNEAL_HOP_TRACE``
on. One row per pair: where the two chains first part, the first hop with a
coupled change on file and the first coupled draw, how often coupled and
plain draws are accepted, and the median energy the chain stands at above
the reference, next to the hits of the gate's develop and branch arms
(``seeds.csv``'s ``develop`` and ``anneal`` rows, whatever patience the
branch ran at). Energies are the unpolished quench energies of the trace,
in kcal/mol.

    coupling_trace.py --gate bench/openconf-h2h/conf-fix8/seeds.csv \\
        --patience 0 --jobs 6 --out bench/openconf-h2h/conf-fix8/trace-patience0.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h2h_common as hc

COLUMNS = [
    "molecule",
    "seed",
    "budget",
    "patience",
    "hit_develop_gate",
    "hit_branch_gate",
    "diverge_hop",
    "first_set_hop",
    "first_coupled_hop",
    "hops_plain",
    "hops_coupled",
    "coupled_draws",
    "coupled_accepted",
    "plain_draws_coupled_arm",
    "plain_accepted_coupled_arm",
    "plain_draws",
    "plain_accepted",
    "current_median_plain",
    "current_median_coupled",
    "best_plain",
    "best_coupled",
]


def one(molecule: str, seed: int, budget: int, coupling: bool, patience: int) -> None:
    import anneal
    import numpy as np

    row = hc.molecules()[molecule]
    mol = hc.build_mol(row["smiles"])
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    start = hc.embed(mol, seed)
    ff = hc.mmff94(mol)
    anneal.cluster_search(
        lambda x: ff.CalcEnergy(x.tolist()) / hc.KCAL_PER_EV,
        lambda x: np.asarray(ff.CalcGrad(x.tolist())) / hc.KCAL_PER_EV,
        len(z),
        budget,
        seed=seed,
        recommended=True,
        start=start.reshape(-1),
        species=z,
        groups=hc.rigid_groups(mol),
        torsion_coupling=coupling,
        torsion_patience=patience,
    )


def trace(
    molecule: str, seed: int, budget: int, coupling: bool, patience: int
) -> list[dict]:
    """Hops of one run, each torsion hop with the draw that made it."""
    env = dict(
        os.environ, ANNEAL_HOP_TRACE="1", OMP_NUM_THREADS="1", PYTHONHASHSEED="0"
    )
    cmd = [
        sys.executable,
        __file__,
        "one",
        molecule,
        str(seed),
        str(budget),
        str(int(coupling)),
        str(patience),
    ]
    err = subprocess.run(
        cmd, env=env, capture_output=True, text=True, check=True
    ).stderr
    hops, draw = [], None
    for line in err.splitlines():
        if line.startswith("DRAWTRACE "):
            draw = json.loads(line.split(" ", 1)[1])
        elif line.startswith("HOPTRACE "):
            hop = json.loads(line.split(" ", 1)[1])
            hop.pop("x", None)
            hop["draw"] = draw if hop["arm"] == "torsion" else None
            draw = None
            hops.append(hop)
    return hops


def pair(job: tuple[str, int, int, int, str, str, float]) -> dict:
    molecule, seed, budget, patience, hit_develop, hit_branch, e_ref = job
    plain = trace(molecule, seed, budget, False, patience)
    coupled = trace(molecule, seed, budget, True, patience)
    k = hc.KCAL_PER_EV
    diverge = next(
        (
            a["hop"]
            for a, b in zip(plain, coupled)
            if (a["e_new"], a["arm"]) != (b["e_new"], b["arm"])
        ),
        "",
    )
    torsion = [h for h in coupled if h["arm"] == "torsion" and h["draw"]]
    drawn = [h for h in torsion if h["draw"]["coupled"]]
    rest = [h for h in torsion if not h["draw"]["coupled"]]
    base = [h for h in plain if h["arm"] == "torsion"]
    return {
        "molecule": molecule,
        "seed": seed,
        "budget": budget,
        "patience": patience,
        "hit_develop_gate": hit_develop,
        "hit_branch_gate": hit_branch,
        "diverge_hop": diverge,
        "first_set_hop": next((h["hop"] for h in torsion if h["draw"]["sets"]), ""),
        "first_coupled_hop": next((h["hop"] for h in drawn), ""),
        "hops_plain": len(plain),
        "hops_coupled": len(coupled),
        "coupled_draws": len(drawn),
        "coupled_accepted": sum(h["accept"] for h in drawn),
        "plain_draws_coupled_arm": len(rest),
        "plain_accepted_coupled_arm": sum(h["accept"] for h in rest),
        "plain_draws": len(base),
        "plain_accepted": sum(h["accept"] for h in base),
        "current_median_plain": f"{statistics.median(h['e'] * k - e_ref for h in plain):.3f}",
        "current_median_coupled": f"{statistics.median(h['e'] * k - e_ref for h in coupled):.3f}",
        "best_plain": f"{min(h['e_new'] for h in plain) * k - e_ref:.3f}",
        "best_coupled": f"{min(h['e_new'] for h in coupled) * k - e_ref:.3f}",
    }


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        name, seed, budget, coupling, patience = sys.argv[2:7]
        one(name, int(seed), int(budget), coupling == "1", int(patience))
        return 0
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gate",
        type=Path,
        required=True,
        help="paired seeds.csv with budgets and hits",
    )
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--molecules", nargs="*")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.gate.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    hits = {(r["molecule"], int(r["seed"]), r["arm"]): r["hit"] for r in rows}
    jobs = [
        (
            r["molecule"],
            int(r["seed"]),
            int(r["budget"]),
            args.patience,
            hits[(r["molecule"], int(r["seed"]), "develop")],
            r["hit"],
            float(r["e_ref_kcal"]),
        )
        for r in rows
        if r["arm"] == "anneal"
        and (not args.molecules or r["molecule"] in args.molecules)
    ]
    with (
        ProcessPoolExecutor(args.jobs) as pool,
        args.out.open("w", newline="") as handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for out in pool.map(pair, jobs):
            writer.writerow(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
