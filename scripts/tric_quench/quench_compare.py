#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Gradients per quench: anneal's Cartesian L-BFGS against TRIC.

Every arm calls the same in-process GFN2-xTB (xtb 6.7.1) and stops at the
same test, the largest Cartesian gradient component below the gate in
eV/Å, at a point no higher than any it has accepted. geomeTRIC's own
criteria are set out of reach so the gate is the only exit. The
Cartesian geomeTRIC arm separates the coordinate system from the
optimizer. ``anneal_tric`` is anneal's own internal-coordinate quench
(``anneal.quench(method="tric")``).

The gates are the molecular preset's record gradient, 0.05 eV/Å, and the
GFN2 water campaign's, 2e-4 Hartree/Bohr.

Starts:
  water   (H2O)20, the quench starts of one anneal molecular search
  nh3     (NH3)4, the same
  drug    50 raw ETKDG conformers of the openconf head-to-head molecules

The two clusters carry the GFN2 water campaign's X-H wall (zero inside
1.15 Å, 200 eV/Å² outside) in the search and in every replay, so a
flexible move cannot dissociate a monomer.

``starts`` writes ``starts.npz``; ``run`` replays them and writes one row
per (start, gate, arm); ``summary`` reduces the rows. ``search`` runs the
cluster searches themselves with each anneal quench and writes one row per
(system, seed, quench).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import logging
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "openconf_h2h"))

BOHR = 0.529177210903
HARTREE = 27.211386245988
EV_PER_HB = HARTREE / BOHR
SEARCH_GATE = 5e-2
WATER_GATE = 2e-4 * EV_PER_HB
GATES = (SEARCH_GATE, WATER_GATE)
WALL_ONSET = 1.15
WALL_EV_PER_A2 = 200.0
SCC_FAILURE_EV = 1.0e6
ARMS = ("anneal", "tric", "cart", "anneal_tric")
SAME_EV = 1e-3
SAME_RMSD = 0.1

WATER = (np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]), [8, 1, 1])
_A = np.radians(67.9)
NH3 = (
    np.array(
        [[0.0, 0.0, 0.0]]
        + [[1.012 * np.sin(_A) * np.cos(p), 1.012 * np.sin(_A) * np.sin(p), -1.012 * np.cos(_A)]
           for p in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
    ),
    [7, 1, 1, 1],
)
CLUSTERS = {"water": (WATER, 20, 2.75, 1500), "nh3": (NH3, 4, 3.2, 1000)}


class Gfn2:
    """GFN2-xTB through the xtb C API, in eV and Å, with a call count and time inside.

    One results object lives as long as the engine, so every call restarts
    from the previous wavefunction; ``reset`` starts a quench cold.
    """

    def __init__(self, numbers, wall=()):
        self.wall = list(wall)
        lib = ctypes.CDLL(os.environ.get("XTB_LIBRARY", os.path.join(sys.base_prefix, "lib", "libxtb.so.6")))
        p = ctypes.c_void_p
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int)
        for name, args, res in (
            ("xtb_newEnvironment", [], p),
            ("xtb_newCalculator", [], p),
            ("xtb_newResults", [], p),
            ("xtb_delResults", [ctypes.POINTER(p)], None),
            ("xtb_setVerbosity", [p, ctypes.c_int], None),
            ("xtb_checkEnvironment", [p], ctypes.c_int),
            ("xtb_getError", [p, ctypes.c_char_p, ip], None),
            ("xtb_newMolecule", [p, ip, ip, dp, dp, ip, dp, ctypes.POINTER(ctypes.c_bool)], p),
            ("xtb_updateMolecule", [p, p, dp, dp], None),
            ("xtb_loadGFN2xTB", [p, p, p, ctypes.c_char_p], None),
            ("xtb_singlepoint", [p, p, p, p], None),
            ("xtb_getEnergy", [p, p, dp], None),
            ("xtb_getGradient", [p, p, dp], None),
        ):
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = res
        self._lib = lib
        self.numbers = np.ascontiguousarray(numbers, dtype=np.int32)
        self.n = len(self.numbers)
        self._env = lib.xtb_newEnvironment()
        lib.xtb_setVerbosity(self._env, 0)
        self._calc = lib.xtb_newCalculator()
        self._mol = None
        self._res = None
        self.calls = 0
        self.pot_s = 0.0
        self.cold_retries = 0
        self.scc_failures = 0

    def _check(self):
        if self._lib.xtb_checkEnvironment(self._env):
            buf = ctypes.create_string_buffer(2048)
            size = ctypes.c_int(2048)
            self._lib.xtb_getError(self._env, buf, ctypes.byref(size))
            raise RuntimeError(buf.value.decode())

    def reset(self):
        self.calls = 0
        self.pot_s = 0.0
        self.cold_retries = 0
        self.scc_failures = 0
        if self._res is not None:
            self._lib.xtb_delResults(ctypes.byref(ctypes.c_void_p(self._res)))
        self._res = self._lib.xtb_newResults()

    @staticmethod
    def _ptr(a, kind=ctypes.c_double):
        return a.ctypes.data_as(ctypes.POINTER(kind))

    def __call__(self, x_ang):
        t0 = time.perf_counter()
        pos = np.ascontiguousarray(np.asarray(x_ang, dtype=float).reshape(-1) / BOHR)
        lib = self._lib
        if self._mol is None:
            n = ctypes.c_int(self.n)
            charge = ctypes.c_double(0.0)
            uhf = ctypes.c_int(0)
            self._mol = lib.xtb_newMolecule(
                self._env, ctypes.byref(n), self._ptr(self.numbers, ctypes.c_int), self._ptr(pos),
                ctypes.byref(charge), ctypes.byref(uhf), None, None,
            )
            self._check()
            lib.xtb_loadGFN2xTB(self._env, self._mol, self._calc, None)
            self._check()
        else:
            lib.xtb_updateMolecule(self._env, self._mol, self._ptr(pos), None)
            self._check()
        if self._res is None:
            self._res = lib.xtb_newResults()
        lib.xtb_singlepoint(self._env, self._mol, self._calc, self._res)
        try:
            self._check()
        except RuntimeError:
            # A guess carried from a distant geometry can stall the SCC.
            self.cold_retries += 1
            lib.xtb_delResults(ctypes.byref(ctypes.c_void_p(self._res)))
            self._res = lib.xtb_newResults()
            lib.xtb_singlepoint(self._env, self._mol, self._calc, self._res)
            try:
                self._check()
            except RuntimeError:
                # The search scores a failed call as its worst value with a
                # zero gradient; every replay arm sees the same.
                self.scc_failures += 1
                self.pot_s += time.perf_counter() - t0
                self.calls += 1
                return SCC_FAILURE_EV, np.zeros(3 * self.n)
        e = ctypes.c_double()
        g = np.empty(3 * self.n)
        lib.xtb_getEnergy(self._env, self._res, ctypes.byref(e))
        lib.xtb_getGradient(self._env, self._res, self._ptr(g))
        self._check()
        energy = e.value * HARTREE
        g *= EV_PER_HB
        x = pos.reshape(-1, 3) * BOHR
        for heavy, hydrogen in self.wall:
            delta = x[hydrogen] - x[heavy]
            distance = float(np.linalg.norm(delta))
            if distance > WALL_ONSET:
                excess = distance - WALL_ONSET
                energy += WALL_EV_PER_A2 * excess * excess
                slope = 2.0 * WALL_EV_PER_A2 * excess / distance * delta
                g[3 * heavy:3 * heavy + 3] -= slope
                g[3 * hydrogen:3 * hydrogen + 3] += slope
        self.pot_s += time.perf_counter() - t0
        self.calls += 1
        return energy, g


def pack(template, count, spacing, seed):
    """Molecules at random orientations in a sphere, centres ``spacing`` apart."""
    coords, numbers = template
    rng = np.random.default_rng(seed)
    radius = spacing * (count / 0.6) ** (1 / 3) * 0.5
    centres = []
    misses = 0
    while len(centres) < count:
        c = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(c) <= radius and all(np.linalg.norm(c - o) >= spacing for o in centres):
            centres.append(c)
            misses = 0
        else:
            misses += 1
            if misses == 10_000:
                # The first centres can leave no room for the rest.
                centres, misses = [], 0
    out = []
    for c in centres:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        out.append(coords @ rot.T + c)
    return np.vstack(out), numbers * count


def wall_pairs(numbers, size):
    """Each monomer's heavy atom, first in its block, with its hydrogens."""
    return [(b, b + k) for b in range(0, len(numbers), size) for k in range(1, size)]


def cluster_search_run(name, seed, local_step):
    """One anneal molecular search on GFN2 with the given quench."""
    import anneal

    template, count, spacing, budget = CLUSTERS[name]
    x0, numbers = pack(template, count, spacing, seed)
    size = len(template[1])
    groups = [list(range(i * size, (i + 1) * size)) for i in range(count)]
    eng = Gfn2(numbers, wall_pairs(numbers, size))
    eng.reset()
    cache = {}

    def evaluate(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = eng(x)
        return cache[key]

    t0 = time.perf_counter()
    out = anneal.cluster_search(
        lambda x: evaluate(x)[0],
        lambda x: evaluate(x)[1],
        len(numbers),
        budget,
        seed=seed,
        recommended=True,
        start=x0.reshape(-1),
        species=numbers,
        groups=groups,
        local_step=local_step,
    )
    wall = time.perf_counter() - t0
    call_s, step_s = out.get("quench_seconds") or (float("nan"), float("nan"))
    return {
        "system": name, "seed": seed, "local_step": local_step, "budget": budget,
        "calls": eng.calls, "hops": out["hops"], "best_eV": f"{out['best_energy']:.6f}",
        "wall_s": f"{wall:.3f}", "pot_s": f"{eng.pot_s:.3f}", "outside_s": f"{wall - eng.pot_s:.3f}",
        "scc_failures": eng.scc_failures, "quench": out.get("local_step", local_step),
        "call_ms": f"{1e3 * call_s:.3f}", "tric_step_ms": f"{1e3 * step_s:.3f}",
    }


def cmd_search(args):
    fields = ["system", "seed", "local_step", "budget", "calls", "hops", "best_eV", "wall_s", "pot_s",
              "outside_s", "scc_failures", "quench", "call_ms", "tric_step_ms"]
    done = set()
    if args.searches.is_file():
        with args.searches.open(newline="") as handle:
            done = {(r["system"], r["seed"], r["local_step"]) for r in csv.DictReader(handle)}
    new = not args.searches.is_file()
    with args.searches.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        for name in ("water", "nh3"):
            for seed in args.search_seeds:
                for step in ("lbfgs", "tric", "auto"):
                    if (name, str(seed), step) in done:
                        continue
                    row = cluster_search_run(name, seed, step)
                    writer.writerow(row)
                    handle.flush()
                    print(",".join(str(row[f]) for f in fields), flush=True)


def search_starts(name, seed, take):
    """Quench starts and in-search calls of one anneal molecular search."""
    import anneal

    template, count, spacing, budget = CLUSTERS[name]
    x0, numbers = pack(template, count, spacing, seed)
    size = len(template[1])
    groups = [list(range(i * size, (i + 1) * size)) for i in range(count)]
    eng = Gfn2(numbers, wall_pairs(numbers, size))
    stream = []
    cache = {}

    def evaluate(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            e, g = eng(x)
            cache[key] = (e, g)
            gmax = np.inf if e >= SCC_FAILURE_EV else float(np.max(np.abs(g)))
            stream.append((x.copy(), e, gmax))
            if len(stream) % 100 == 0:
                print(f"  {name}: {len(stream)} calls, {eng.pot_s:.1f} s inside, "
                      f"{time.perf_counter() - t0:.1f} s wall", flush=True)
        return cache[key]

    t0 = time.perf_counter()
    anneal.cluster_search(
        lambda x: evaluate(x)[0],
        lambda x: evaluate(x)[1],
        len(numbers),
        budget,
        seed=seed,
        recommended=True,
        start=x0.reshape(-1),
        species=numbers,
        groups=groups,
    )
    wall = time.perf_counter() - t0
    quenches = []
    begin = 0
    for i, (_, _, gmax) in enumerate(stream):
        if gmax < SEARCH_GATE:
            if i > begin:
                quenches.append((stream[begin][0], i - begin + 1))
            begin = i + 1
    picks = np.linspace(0, len(quenches) - 1, min(take, len(quenches))).round().astype(int)
    print(
        f"{name}: {len(stream)} calls, {len(quenches)} completed quenches, "
        f"median {statistics.median(q[1] for q in quenches)} calls, wall {wall:.1f} s",
        flush=True,
    )
    return (
        [quenches[i][0] for i in picks],
        [quenches[i][1] for i in picks],
        np.array(numbers),
        [q[1] for q in quenches],
    )


def drug_starts(total):
    import h2h_common as hc
    from rdkit.Chem import AllChem

    rows = list(hc.molecules().items())
    per = [total // len(rows) + (i < total % len(rows)) for i in range(len(rows))]
    out = []
    for (name, row), k in zip(rows, per):
        mol = hc.build_mol(row["smiles"])
        numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        for seed in range(1, k + 1):
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            conf = AllChem.EmbedMolecule(mol, params)
            if conf < 0:
                raise RuntimeError(f"ETKDG failed on {name} seed {seed}")
            x = mol.GetConformer(conf).GetPositions()
            out.append((name, seed, x - x.mean(axis=0), numbers))
    return out


def cmd_starts(args):
    payload = {}
    for name in ("water", "nh3"):
        xs, calls, numbers, every = search_starts(name, args.search_seed, args.take)
        payload[f"{name}_x"] = np.array([x.reshape(-1) for x in xs])
        payload[f"{name}_search_calls"] = np.array(calls)
        payload[f"{name}_all_search_calls"] = np.array(every)
        payload[f"{name}_numbers"] = numbers
    drugs = drug_starts(args.drugs)
    for i, (name, seed, x, numbers) in enumerate(drugs):
        payload[f"drug{i:02d}_x"] = x.reshape(1, -1)
        payload[f"drug{i:02d}_numbers"] = numbers
        payload[f"drug{i:02d}_label"] = np.array(f"{name}:{seed}")
    np.savez_compressed(args.starts, **payload)


class GateReached(Exception):
    pass


def geometric_quench(eng, numbers, x0, gate, coordsys, max_iter):
    from geometric.engine import Engine
    from geometric.internal import CartesianCoordinates, DelocalizedInternalCoordinates
    from geometric.molecule import Molecule
    from geometric.nifty import ang2bohr
    from geometric.optimize import Optimize
    from geometric.params import OptParams

    symbols = [ELEMENTS[z] for z in numbers]
    best = {"e": np.inf, "x": x0.copy()}

    class Gated(Engine):
        def calc_new(self, coords, dirname):
            x = coords * BOHR
            e, g = eng(x)
            if e <= best["e"]:
                best.update(e=e, x=x.copy())
                if np.max(np.abs(g)) < gate:
                    raise GateReached
            return {"energy": e / HARTREE, "gradient": g / EV_PER_HB}

    mol = Molecule()
    mol.elem = symbols
    mol.xyzs = [x0.reshape(-1, 3)]
    mol.comms = [""]
    mol.build_topology()
    engine = Gated(mol)
    if coordsys == "tric":
        ic = DelocalizedInternalCoordinates(mol, build=True, connect=False, addcart=False)
    else:
        ic = CartesianCoordinates(mol)
    unreachable = 1e-30
    params = OptParams(
        maxiter=max_iter,
        convergence_energy=unreachable,
        convergence_grms=unreachable,
        convergence_gmax=unreachable,
        convergence_drms=unreachable,
        convergence_dmax=unreachable,
    )
    scratch = ROOT / "build" / "tric_quench"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        try:
            Optimize(x0.reshape(-1) * ang2bohr, mol, ic, engine, tmp, params, print_info=False)
            reached = False
        except GateReached:
            reached = True
        except Exception as err:  # geomeTRIC gives up on a degenerate step
            print(f"  geomeTRIC {coordsys} stopped: {type(err).__name__}: {err}", flush=True)
            reached = False
    return best["x"], best["e"], reached


ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br"}


def anneal_quench(eng, x0, gate, max_iter, numbers=None):
    import anneal

    seen = {}

    def fg(x):
        e, g = eng(x)
        seen[x.tobytes()] = float(np.max(np.abs(g)))
        return e, g

    if numbers is None:
        out = anneal.quench(fg, x0.reshape(-1), gate, max_iter=max_iter)
    else:
        out = anneal.quench(fg, x0.reshape(-1), gate, max_iter=max_iter, method="tric",
                            species=[int(z) for z in numbers])
    x = np.asarray(out["x"], dtype=float)
    return x, float(out["energy"]), seen.get(x.tobytes(), np.inf) < gate


def heavy_rmsd(a, b, numbers):
    mask = np.asarray(numbers) > 1
    p = a.reshape(-1, 3)[mask]
    q = b.reshape(-1, 3)[mask]
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    u, s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    return float(np.sqrt(max(0.0, (np.sum(p * p) + np.sum(q * q) - 2 * s.sum()) / len(p))))


def load_starts(path):
    data = np.load(path)
    starts = []
    for name in ("water", "nh3"):
        for i, (x, calls) in enumerate(zip(data[f"{name}_x"], data[f"{name}_search_calls"])):
            starts.append((name, f"{name}{i:02d}", x, data[f"{name}_numbers"], int(calls)))
    i = 0
    while f"drug{i:02d}_x" in data:
        starts.append(("drug", str(data[f"drug{i:02d}_label"]), data[f"drug{i:02d}_x"][0],
                       data[f"drug{i:02d}_numbers"], None))
        i += 1
    return starts


def cmd_run(args):
    for name in ("geometric", "geometric.nifty", "geometric.optimize"):
        logging.getLogger(name).setLevel(logging.ERROR)
    starts = load_starts(args.starts)
    if args.only:
        starts = [s for s in starts if s[0] in args.only]
    done = set()
    if args.out.is_file():
        with args.out.open(newline="") as handle:
            done = {(r["start"], r["gate"], r["arm"]) for r in csv.DictReader(handle)}
    fields = ["system", "start", "atoms", "gate", "arm", "calls", "wall_s", "pot_s", "outside_s",
              "energy_eV", "reached", "scc_failures", "search_calls", "same_as_anneal", "de_vs_anneal_eV",
              "rmsd_vs_anneal"]
    new = not args.out.is_file()
    with args.out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        for system, label, x0, numbers, search_calls in starts:
            if all((label, f"{g:g}", a) in done for g in GATES for a in args.arms):
                continue
            size = {"water": 3, "nh3": 4}.get(system)
            eng = Gfn2(numbers, wall_pairs(numbers, size) if size else ())
            for gate in GATES:
                reference = None
                for arm in ARMS:
                    key = (label, f"{gate:g}", arm)
                    if arm != "anneal" and (key in done or arm not in args.arms):
                        continue
                    eng.reset()
                    t0 = time.perf_counter()
                    if arm == "anneal":
                        x, e, reached = anneal_quench(eng, x0, gate, args.max_iter)
                    elif arm == "anneal_tric":
                        x, e, reached = anneal_quench(eng, x0, gate, args.max_iter, numbers)
                    else:
                        x, e, reached = geometric_quench(eng, numbers, x0, gate, arm, args.max_iter)
                    wall = time.perf_counter() - t0
                    if arm == "anneal":
                        reference = (x, e)
                    de = e - reference[1]
                    rmsd = heavy_rmsd(x, reference[0], numbers)
                    if key in done:
                        continue
                    writer.writerow(
                        {
                            "system": system, "start": label, "atoms": len(numbers),
                            "gate": f"{gate:g}", "arm": arm, "calls": eng.calls,
                            "wall_s": f"{wall:.6f}", "pot_s": f"{eng.pot_s:.6f}",
                            "outside_s": f"{wall - eng.pot_s:.6f}", "energy_eV": f"{e:.6f}",
                            "reached": int(reached), "scc_failures": eng.scc_failures,
                            "search_calls": "" if search_calls is None else search_calls,
                            "same_as_anneal": int(abs(de) <= SAME_EV and rmsd <= SAME_RMSD),
                            "de_vs_anneal_eV": f"{de:.6f}", "rmsd_vs_anneal": f"{rmsd:.4f}",
                        }
                    )
                    handle.flush()
                    print(f"{label} gate {gate:g} {arm}: {eng.calls} calls {wall:.2f} s "
                          f"dE {de:+.4f} rmsd {rmsd:.3f} reached {reached}", flush=True)


def cmd_summary(args):
    with args.out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for r in rows:
        groups.setdefault((r["system"], r["gate"]), []).append(r)
    fields = ["system", "gate", "arm", "starts", "reached", "calls_total", "calls_median",
              "saving_total_vs_anneal", "saving_median_ratio_vs_anneal", "wall_total_s",
              "pot_total_s", "outside_total_s", "same_as_anneal", "lower_than_anneal",
              "search_calls_median"]
    out = []
    for (system, gate), rs in sorted(groups.items()):
        arms = [arm for arm in ARMS if any(r["arm"] == arm for r in rs)]
        by = {arm: {r["start"]: r for r in rs if r["arm"] == arm} for arm in arms}
        common = sorted(set.intersection(*(set(v) for v in by.values())))
        base = sum(int(by["anneal"][s]["calls"]) for s in common)
        for arm in arms:
            sel = [by[arm][s] for s in common]
            calls = [int(r["calls"]) for r in sel]
            ratios = [int(by[arm][s]["calls"]) / int(by["anneal"][s]["calls"]) for s in common]
            search = [int(r["search_calls"]) for r in sel if r["search_calls"]]
            out.append(
                {
                    "system": system, "gate": gate, "arm": arm, "starts": len(sel),
                    "reached": sum(int(r["reached"]) for r in sel),
                    "calls_total": sum(calls), "calls_median": statistics.median(calls),
                    "saving_total_vs_anneal": f"{1 - sum(calls) / base:.3f}",
                    "saving_median_ratio_vs_anneal": f"{1 - statistics.median(ratios):.3f}",
                    "wall_total_s": f"{sum(float(r['wall_s']) for r in sel):.2f}",
                    "pot_total_s": f"{sum(float(r['pot_s']) for r in sel):.2f}",
                    "outside_total_s": f"{sum(float(r['outside_s']) for r in sel):.2f}",
                    "same_as_anneal": sum(int(r["same_as_anneal"]) for r in sel),
                    "lower_than_anneal": sum(float(r["de_vs_anneal_eV"]) < -SAME_EV for r in sel),
                    "search_calls_median": statistics.median(search) if search else "",
                }
            )
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    for row in out:
        print(",".join(str(row[f]) for f in fields))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["starts", "run", "summary", "search"])
    parser.add_argument("--starts", type=Path, default=Path("/work/tric-quench/starts.npz"))
    parser.add_argument("--out", type=Path, default=Path("/work/tric-quench/quenches.csv"))
    parser.add_argument("--summary", type=Path, default=ROOT / "bench" / "tric-quench" / "results.csv")
    parser.add_argument("--search-seed", type=int, default=7001)
    parser.add_argument("--search-seeds", type=int, nargs="*", default=[7101, 7102, 7103])
    parser.add_argument("--searches", type=Path, default=Path("/work/tric-quench/searches.csv"))
    parser.add_argument("--take", type=int, default=10)
    parser.add_argument("--drugs", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    args = parser.parse_args()
    args.starts.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    {"starts": cmd_starts, "run": cmd_run, "summary": cmd_summary, "search": cmd_search}[args.command](args)


if __name__ == "__main__":
    main()
