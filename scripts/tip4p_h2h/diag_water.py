#!/usr/bin/env python3
# Copyright (c) anneal developers
"""(H2O)n miss diagnosis against the Cambridge minimum, any n.

``structures`` reads the lowest geometry each seed wrote
(``<runs>/tip4p<n>-s<seed>.xyz``, the ``water_gmin`` layout) and, with the
Cambridge minimum as the first row, prints one row per seed:

* the oxygen framework, the O-O contact graph at ``OO_CONTACT``, and whether
  it is isomorphic to the minimum's;
* the directed hydrogen-bond graph and whether it is isomorphic to the
  minimum's; with the framework equal and the digraph not, the difference is
  the proton arrangement alone;
* hydrogen-bond and dangling-OH counts and the donor/acceptor census
  (``DA``, ``DDA``, ``DAA``, ``DDAA``, ...);
* chordless rings of the contact graph, lengths 3 to 8;
* the cage shape: radius of gyration and asphericity of the oxygens;
* the oxygen RMSD to the minimum over permutation, rotation and reflection.

Distinct frameworks are numbered in order of appearance, ``F0`` being the
minimum's. Needs numpy, scipy and networkx.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag21 import hbond_digraph, oxygen_rmsd, parse_xyz  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(2000, 2012))
CCD = REPO / "paper" / "jctc" / "data" / "system_visuals" / "ccd" / "TIP4P" / "xyz"
MINIMA = {
    8: REPO / "bench" / "tip4p-h2h" / "radius" / "TIP4P-8.xyz",
    12: CCD / "TIP4P-12.xyz",
    16: CCD / "TIP4P-16.xyz",
    20: REPO / "tests" / "fixtures" / "tip4p-20.xyz",
    21: CCD / "TIP4P-21.xyz",
}
# In the five minima and every n = 16 and 21 seed geometry the first-shell
# O-O pairs end by 2.85 A and the next begin at 3.22 A.
OO_CONTACT = 3.1


def contact_graph(o: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(o)))
    for i in range(len(o)):
        for j in range(i + 1, len(o)):
            if np.linalg.norm(o[i] - o[j]) < OO_CONTACT:
                g.add_edge(i, j)
    return g


def rings(g: nx.Graph, longest: int = 8) -> Counter:
    counts: Counter = Counter()
    for cycle in nx.chordless_cycles(g, length_bound=longest):
        counts[len(cycle)] += 1
    return counts


def shape(o: np.ndarray) -> tuple[float, float]:
    c = o - o.mean(0)
    w = np.sort(np.linalg.eigvalsh(c.T @ c / len(o)))
    rg = float(np.sqrt(w.sum()))
    asph = float(((w[2] - w[1]) ** 2 + (w[2] - w[0]) ** 2 + (w[1] - w[0]) ** 2)
                 / (2.0 * w.sum() ** 2))
    return rg, asph


def census(bonds: nx.DiGraph) -> str:
    kinds = Counter("D" * bonds.out_degree(i) + "A" * bonds.in_degree(i) for i in bonds)
    return " ".join(f"{k}:{v}" for k, v in sorted(kinds.items()))


def energy_of(path: Path) -> float:
    line = path.read_text().splitlines()[1]
    match = re.search(r"-?\d+\.\d+", line.split("=", 1)[-1])
    return float(match.group(0)) if match else float("nan")


def cmd_structures(args: argparse.Namespace) -> None:
    og, hg = parse_xyz(MINIMA[args.n])
    gm_contacts, gm_bonds = contact_graph(og), hbond_digraph(og, hg)
    frameworks = [gm_contacts]
    fields = ["seed", "energy_kj", "framework", "framework_iso_gm", "hbond_iso_gm",
              "o_contacts", "hbonds", "dangling_oh", "census", "rings", "rg_a",
              "asphericity", "o_rmsd_gm_a"]
    writer = csv.DictWriter(args.output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    entries = [("gm", MINIMA[args.n], og, hg)]
    for seed in SEEDS:
        path = args.runs / f"tip4p{args.n}-s{seed}.xyz"
        o, h = parse_xyz(path)
        entries.append((str(seed), path, o, h))
    for label, path, o, h in entries:
        contacts, bonds = contact_graph(o), hbond_digraph(o, h)
        cls = next((i for i, c in enumerate(frameworks) if nx.is_isomorphic(contacts, c)), None)
        if cls is None:
            frameworks.append(contacts)
            cls = len(frameworks) - 1
        rg, asph = shape(o)
        ring = rings(contacts)
        writer.writerow(dict(
            seed=label,
            energy_kj=f"{energy_of(path):.4f}",
            framework=f"F{cls}",
            framework_iso_gm=int(cls == 0),
            hbond_iso_gm=int(nx.is_isomorphic(bonds, gm_bonds)),
            o_contacts=contacts.number_of_edges(),
            hbonds=bonds.number_of_edges(),
            dangling_oh=2 * len(o) - bonds.number_of_edges(),
            census=census(bonds),
            rings=" ".join(f"{k}:{v}" for k, v in sorted(ring.items())),
            rg_a=f"{rg:.3f}",
            asphericity=f"{asph:.4f}",
            o_rmsd_gm_a=f"{oxygen_rmsd(o, og):.3f}",
        ))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    st = sub.add_parser("structures")
    st.add_argument("--n", type=int, choices=sorted(MINIMA), required=True)
    st.add_argument("--runs", type=Path, required=True)
    st.add_argument("--output", type=argparse.FileType("w"), default=sys.stdout)
    args = parser.parse_args(argv[1:])
    {"structures": cmd_structures}[args.command](args)


if __name__ == "__main__":
    main(sys.argv)
