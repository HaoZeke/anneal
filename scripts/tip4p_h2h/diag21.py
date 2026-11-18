#!/usr/bin/env python3
# Copyright (c) anneal developers
"""(H2O)21 miss diagnosis: the rows behind ``diag-h2o21.csv`` and
``summary-h2o21-mechanisms.csv`` in ``bench/tip4p-h2h``.

Three subcommands, all serial:

``run``
    Twelve ``h2h_seed tip4p <n>`` seeds at 2M charged calls with the panel's
    stop target (the Cambridge header plus ``HIT_TOL``), written as
    ``<runs>/n<n>-s<seed>.{out,xyz,trace.csv}``.

``diag``
    One row per seed from a ``run`` directory at n = 21. The oxygen framework
    is the O-O contact graph at ``OO_CONTACT``; the proton topology is the
    directed hydrogen-bond graph. Both are tested for isomorphism with the
    Cambridge minimum. ``o_rmsd_gm_a`` is the oxygen RMSD after the optimal
    permutation, rotation and reflection. Rings are chordless cycles of the
    contact graph. The budget columns are the search's own counters from the
    ``.out``: arm accepts, hops, and the minima-hopping visit classification.

``mechanisms``
    Hits, best and median per ``variant:n:directory`` triple, in the order
    given. A directory holds ``n<n>-s<seed>.out`` or ``.log`` files carrying
    the ``h2h_row`` line. The variant builds were scratch builds; only their
    logs are read here.

Needs numpy, scipy and networkx.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import statistics
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from target_tol import DEFAULT_HIT_TOL  # noqa: E402

SEEDS = tuple(range(2000, 2012))
BUDGET = 2_000_000
HIT_TOL = DEFAULT_HIT_TOL
CAMBRIDGE = {
    8: -305.5183221676,
    15: -628.3722525077,
    20: -872.9887544952,
    21: -916.7074302940,
}
PLATEAU = -915.7735
GM_XYZ = Path(__file__).resolve().parents[2] / (
    "paper/jctc/data/system_visuals/ccd/TIP4P/xyz/TIP4P-21.xyz"
)
# The minimum's O-O distances leave an empty band at [2.9, 3.1) A between the
# first-shell pairs and the rest, as do those of every seed here.
OO_CONTACT = 3.0
HB_HO_MAX = 2.5
HB_ANGLE_MIN = 130.0
ALIGN_STARTS = 400


def parse_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    nat = int(lines[0].split()[0])
    oxygens, hydrogens = [], []
    for line in lines[2 : 2 + nat]:
        parts = line.split()
        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        (oxygens if parts[0] == "O" else hydrogens).append(xyz)
    return np.array(oxygens), np.array(hydrogens)


def hbonds(o: np.ndarray, h: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for i in range(len(o)):
        for hk in (2 * i, 2 * i + 1):
            best, best_d = None, HB_HO_MAX
            for j in range(len(o)):
                if j == i:
                    continue
                d = np.linalg.norm(h[hk] - o[j])
                if d < best_d:
                    v1, v2 = o[i] - h[hk], o[j] - h[hk]
                    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                    if np.degrees(np.arccos(np.clip(c, -1, 1))) >= HB_ANGLE_MIN:
                        best, best_d = j, d
            if best is not None:
                out.append((i, best))
    return out


def kabsch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    v, _s, wt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(v @ wt))
    return v @ np.diag([1, 1, d]) @ wt


def refine(os_c: np.ndarray, og_c: np.ndarray, r: np.ndarray, iters: int = 80):
    x = os_c @ r
    perm = None
    for _ in range(iters):
        d = ((x[:, None, :] - og_c[None, :, :]) ** 2).sum(-1)
        _rows, perm = linear_sum_assignment(d)
        xn = os_c @ kabsch(os_c, og_c[perm])
        if np.allclose(xn, x, atol=1e-8):
            x = xn
            break
        x = xn
    return np.sqrt(((x - og_c[perm]) ** 2).sum(-1).mean()), perm


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ]
    )


def oxygen_rmsd(os_: np.ndarray, og: np.ndarray) -> float:
    """Lowest RMSD over permutation, rotation and reflection.

    Principal-axis starts and random rotations, each refined by alternating
    assignment and Kabsch, so a local optimum of the joint problem is not
    taken for the answer.
    """
    rng = np.random.default_rng(0)
    og_c = og - og.mean(0)
    best = 1e9
    for mirror in (False, True):
        os_c = os_ - os_.mean(0)
        if mirror:
            os_c = os_c * np.array([-1, 1, 1])
        _w, axes_s = np.linalg.eigh(os_c.T @ os_c)
        _w, axes_g = np.linalg.eigh(og_c.T @ og_c)
        starts = []
        for signs in itertools.product((1, -1), repeat=3):
            r = axes_s @ np.diag(signs) @ axes_g.T
            if np.linalg.det(r) > 0:
                starts.append(r)
        starts += [random_rotation(rng) for _ in range(ALIGN_STARTS)]
        for r in starts:
            rmsd, _perm = refine(os_c, og_c, r)
            best = min(best, rmsd)
    return best


def contact_graph(o: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(o)))
    for i in range(len(o)):
        for j in range(i + 1, len(o)):
            if np.linalg.norm(o[i] - o[j]) < OO_CONTACT:
                g.add_edge(i, j)
    return g


def hbond_digraph(o: np.ndarray, h: np.ndarray) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(len(o)))
    g.add_edges_from(hbonds(o, h))
    return g


def ring_counts(g: nx.Graph, longest: int = 6) -> dict[int, int]:
    counts = {k: 0 for k in range(3, longest + 1)}
    for cycle in nx.chordless_cycles(g, length_bound=longest):
        if 3 <= len(cycle) <= longest:
            counts[len(cycle)] += 1
    return counts


def last_improvement(path: Path) -> int:
    best, last = float("inf"), 0
    with path.open() as handle:
        next(handle)
        for line in handle:
            calls, energy = line.split(",")
            if float(energy) < best - 1e-6:
                best, last = float(energy), int(calls)
    return last


def stop_target(n: int) -> str:
    return repr(CAMBRIDGE[n] + HIT_TOL)


def cmd_run(args: argparse.Namespace) -> None:
    args.runs.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        prefix = args.runs / f"n{args.n}-s{seed}"
        with Path(f"{prefix}.out").open("w") as handle:
            subprocess.run(
                ["taskset", "-c", "0", str(args.binary), "tip4p", str(args.n), "0",
                 str(BUDGET), str(seed), "recommended", stop_target(args.n), str(prefix)],
                check=True, stdout=handle, stderr=subprocess.STDOUT,
            )


def cmd_diag(args: argparse.Namespace) -> None:
    og, hg = parse_xyz(GM_XYZ)
    gm_contacts, gm_hbonds = contact_graph(og), hbond_digraph(og, hg)
    classes = [gm_contacts]
    fields = ["seed", "lowest_kj", "gap_plateau_kj", "gap_gm_kj", "framework_class",
              "framework_iso_gm", "hbond_iso_gm", "o_rmsd_gm_a", "o_contacts", "hbonds",
              "rings4", "rings5", "rings6", "charged", "hops", "accepted", "rejected",
              "quench_return", "quench_known", "quench_new", "calls_per_hop",
              "last_improvement_call", "verdict"]
    writer = csv.DictWriter(args.output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for seed in SEEDS:
        stem = args.runs / f"n21-s{seed}"
        text = Path(f"{stem}.out").read_text()

        def field(key: str) -> str:
            return re.search(rf"^{key}: (\S+)", text, re.M).group(1)

        lowest = float(field("lowest"))
        hops = int(field("hops"))
        ret, known, new = map(int, re.search(
            r"^visits: return (\d+) known (\d+) new (\d+)", text, re.M).groups())
        accepted = sum(int(a) for a in re.findall(
            r"^arm \S+ draws \d+ accepts (\d+)", text, re.M))
        charged = int(re.search(r"charged=(\d+)", text).group(1))
        o, h = parse_xyz(Path(f"{stem}.xyz"))
        contacts, bonds = contact_graph(o), hbond_digraph(o, h)
        framework_iso = nx.is_isomorphic(contacts, gm_contacts)
        hbond_iso = nx.is_isomorphic(bonds, gm_hbonds)
        cls = next((i for i, c in enumerate(classes) if nx.is_isomorphic(contacts, c)), None)
        if cls is None:
            classes.append(contacts)
            cls = len(classes) - 1
        rings = ring_counts(contacts)
        if not framework_iso:
            verdict = "oxygen framework"
        elif not hbond_iso:
            verdict = "proton topology (GM oxygen cage)"
        else:
            verdict = "GM"
        writer.writerow(dict(
            seed=seed,
            lowest_kj=f"{lowest:.10f}",
            gap_plateau_kj=f"{lowest - PLATEAU:.4f}".replace("-0.0000", "0.0000"),
            gap_gm_kj=f"{lowest - CAMBRIDGE[21]:.4f}",
            framework_class=f"F{cls}",
            framework_iso_gm=int(framework_iso),
            hbond_iso_gm=int(hbond_iso),
            o_rmsd_gm_a=f"{oxygen_rmsd(o, og):.3f}",
            o_contacts=contacts.number_of_edges(),
            hbonds=bonds.number_of_edges(),
            rings4=rings[4], rings5=rings[5], rings6=rings[6],
            charged=charged, hops=hops, accepted=accepted, rejected=hops - accepted,
            quench_return=ret, quench_known=known, quench_new=new,
            calls_per_hop=round(charged / hops),
            last_improvement_call=last_improvement(Path(f"{stem}.trace.csv")),
            verdict=verdict,
        ))


def h2h_row(directory: Path, n: int, seed: int) -> dict[str, str]:
    for suffix in ("out", "log"):
        path = directory / f"n{n}-s{seed}.{suffix}"
        if path.is_file():
            line = next(l for l in path.read_text().splitlines() if l.startswith("h2h_row "))
            return dict(token.split("=", 1) for token in line.split()[1:])
    raise SystemExit(f"no run for n={n} seed={seed} in {directory}")


def cmd_mechanisms(args: argparse.Namespace) -> None:
    writer = csv.writer(args.output, lineterminator="\n")
    writer.writerow(["variant", "n", "seeds", "budget", "hits", "best_kj", "median_kj"])
    for spec in args.rows:
        variant, n, directory = spec.split(":", 2)
        rows = [h2h_row(Path(directory), int(n), seed) for seed in SEEDS]
        best = [float(r["best"]) for r in rows]
        hits = sum(int(r["hit"]) for r in rows)
        budgets = {r["budget"] for r in rows}
        if len(budgets) != 1:
            raise SystemExit(f"{spec}: mixed budgets {budgets}")
        writer.writerow([variant, n, len(rows), budgets.pop(), f"{hits}/{len(rows)}",
                         f"{min(best):.10f}", f"{statistics.median(best):.6f}"])


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--runs", type=Path, required=True)
    run.add_argument("--n", type=int, choices=sorted(CAMBRIDGE), default=21)
    diag = sub.add_parser("diag")
    diag.add_argument("--runs", type=Path, required=True)
    diag.add_argument("--output", type=argparse.FileType("w"), default=sys.stdout)
    mech = sub.add_parser("mechanisms")
    mech.add_argument("rows", nargs="+", metavar="VARIANT:N:DIR")
    mech.add_argument("--output", type=argparse.FileType("w"), default=sys.stdout)
    args = parser.parse_args(argv[1:])
    {"run": cmd_run, "diag": cmd_diag, "mechanisms": cmd_mechanisms}[args.command](args)


if __name__ == "__main__":
    main(sys.argv)
