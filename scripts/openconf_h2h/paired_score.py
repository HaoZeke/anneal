# Copyright (c) anneal developers
"""Score the paired anneal and openconf runs and append one row per run.

A third arm, ``develop`` (anneal at the base revision, same budget), is
scored with them when its runs are present. Candidates are polished, checked
and deduplicated by the rules of ``h2h_score.py``. Per molecule the
reference is the union of every arm over every seed: E_ref is its lowest
polished energy and the lowest reference conformer is that geometry. Per
run:

- ``hit``: one of the run's reported conformers is the lowest reference
  conformer (heavy-atom RMSD under 0.25 Å, energy within 0.5 kcal/mol)
- ``calls_to_hit``: energy plus gradient calls spent when the run's trace
  first held a minimum that polishes to that conformer; the run's total
  when only its final output does
- ``n_3kcal``: distinct conformers within 3 kcal/mol of E_ref
- ``error``: the exception a run raised; it is scored on what it harvested

Rows already in ``--rows`` (same molecule, arm and seed) are left as they
are. The summary and the tests are rebuilt from every row in the file:
Fisher's exact test on hits and the Mann-Whitney U test on calls to hit,
both two-sided, anneal against openconf in ``--stats`` and anneal against
develop in ``--versus-develop``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2h_common as hc  # noqa: E402
from h2h_score import WINDOW, Molecule, _fmt  # noqa: E402

ARMS = ("anneal", "openconf", "develop")
ROW_FIELDS = [
    "molecule", "arm", "seed", "budget", "calls", "energy_calls", "grad_calls", "hit", "calls_to_hit",
    "delta_min_kcal", "n_3kcal", "reference_3kcal", "candidates", "invalid", "wall_s", "pot_s",
    "outside_s", "embed_s", "e_ref_kcal", "error", "openconf_revision", "code_revision", "host",
]


_MOLS: dict[str, Molecule] = {}


def _molecule(name: str) -> Molecule:
    if name not in _MOLS:
        _MOLS[name] = Molecule(hc.molecules()[name])
    return _MOLS[name]


def polished_frames(name: str, job_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = job_dir / "polished.npz"
    if cache.is_file():
        data = np.load(cache)
        return data["x"], data["e"], data["ok"]
    m = _molecule(name)
    path = job_dir / "candidates.xyz"
    frames = hc.read_xyz(path) if path.is_file() else []
    xs, es, ok = [], [], []
    for _, x, _ in frames:
        px, pe, pv = m.polish(x)
        xs.append(px)
        es.append(pe)
        ok.append(pv)
    xs = np.asarray(xs).reshape(-1, m.mol.GetNumAtoms(), 3)
    es, ok = np.asarray(es), np.asarray(ok, dtype=bool)
    np.savez(cache, x=xs, e=es, ok=ok)
    return xs, es, ok


def first_hit(name: str, arm: str, job_dir: Path, lowest: tuple[np.ndarray, float]) -> int | None:
    """Calls spent at the first trace minimum that polishes to ``lowest``.

    An anneal arm's trace holds the same frames as its candidates, so their
    polished geometries are reused.
    """
    m = _molecule(name)
    path = job_dir / "trace.xyz"
    if not path.is_file():
        return None
    trace = hc.read_xyz(path)
    if arm in ("anneal", "develop"):
        xs, es, ok = polished_frames(name, job_dir)
        polished = zip(xs, es, ok, strict=True)
    else:
        polished = (m.polish(x) for _, x, _ in trace)
    for (_, _, comment), (px, pe, pv) in zip(trace, polished):
        if pv and m.same(lowest, (px, float(pe))):
            return int(comment.split()[0])
    return None


def score_runs(runs: Path, rows_path: Path, workers: int) -> None:
    done = set()
    if rows_path.is_file():
        with rows_path.open(newline="") as handle:
            done = {(r["molecule"], r["arm"], int(r["seed"])) for r in csv.DictReader(handle)}
    mols = {name: _molecule(name) for name in hc.molecules()}
    infos = [(path.parent, json.loads(path.read_text())) for path in sorted(runs.glob("*/*/*/job.json"))]
    infos = [(job_dir, info) for job_dir, info in infos if info["arm"] in ARMS]
    with ProcessPoolExecutor(workers) as pool:
        frames = list(pool.map(polished_frames, [i["molecule"] for _, i in infos], [d for d, _ in infos]))
    jobs = []
    for (job_dir, info), (xs, es, ok) in zip(infos, frames, strict=True):
        m = mols[info["molecule"]]
        mine = m.dedupe([(x, float(e)) for x, e, v in zip(xs, es, ok, strict=True) if v])
        jobs.append((job_dir, info, mine, len(es), int((~ok).sum()) if len(es) else 0))
    lowest_of = {}
    for name in mols:
        pool_items = [item for _, info, mine, _, _ in jobs if info["molecule"] == name for item in mine]
        if pool_items:
            lowest_of[name] = min(pool_items, key=lambda item: item[1])
    hitting = [
        (job_dir, info)
        for job_dir, info, mine, _, _ in jobs
        if (info["molecule"], info["arm"], int(info["seed"])) not in done
        and any(mols[info["molecule"]].same(lowest_of[info["molecule"]], item) for item in mine)
    ]
    with ProcessPoolExecutor(workers) as pool:
        hits = pool.map(
            first_hit,
            [i["molecule"] for _, i in hitting],
            [i["arm"] for _, i in hitting],
            [d for d, _ in hitting],
            [lowest_of[i["molecule"]] for _, i in hitting],
        )
        trace_hit = {d: h for (d, _), h in zip(hitting, hits, strict=True)}
    new = []
    for name, m in mols.items():
        if name not in lowest_of:
            continue
        pool = [item for _, info, mine, _, _ in jobs if info["molecule"] == name for item in mine]
        lowest = lowest_of[name]
        e_ref = lowest[1]
        ensemble = m.dedupe([item for item in pool if item[1] - e_ref <= WINDOW])
        for job_dir, info, mine, candidates, invalid in jobs:
            key = (name, info["arm"], int(info["seed"]))
            if info["molecule"] != name or key in done:
                continue
            calls = int(info["energy_calls"]) + int(info["grad_calls"])
            hit = any(m.same(lowest, item) for item in mine)
            to_hit = None
            if hit:
                to_hit = trace_hit[job_dir]
                to_hit = calls if to_hit is None else min(to_hit, calls)
            window = [item for item in mine if item[1] - e_ref <= WINDOW]
            new.append(
                {
                    "molecule": name,
                    "arm": info["arm"],
                    "seed": int(info["seed"]),
                    "budget": info.get("budget"),
                    "calls": calls,
                    "energy_calls": info["energy_calls"],
                    "grad_calls": info["grad_calls"],
                    "hit": int(hit),
                    "calls_to_hit": to_hit,
                    "delta_min_kcal": (min(e for _, e in mine) - e_ref) if mine else None,
                    "n_3kcal": len(window),
                    "reference_3kcal": len(ensemble),
                    "candidates": candidates,
                    "invalid": invalid,
                    "wall_s": info["wall_s"],
                    "pot_s": info["pot_s"],
                    "outside_s": info["outside_s"],
                    "embed_s": info.get("embed_s"),
                    "e_ref_kcal": e_ref,
                    "error": info.get("error"),
                    "openconf_revision": info.get("openconf_revision"),
                    "code_revision": info.get("code_revision"),
                    "host": info["hostname"],
                }
            )
    fresh = not rows_path.is_file()
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        for row in sorted(new, key=lambda r: (r["molecule"], r["arm"], r["seed"])):
            writer.writerow({k: _fmt(row.get(k)) for k in ROW_FIELDS})


def _num(value: str) -> float | None:
    return float(value) if value != "" else None


def summarise(rows_path: Path, summary_path: Path, stats_path: Path, versus_path: Path | None = None) -> None:
    with rows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = [name for name in hc.molecules() if any(r["molecule"] == name for r in rows)]
    summary_fields = [
        "molecule", "arm", "n", "hits", "n_3kcal_median", "reference_3kcal", "calls_median",
        "calls_to_hit_median", "delta_min_median_kcal", "wall_median_s", "pot_median_s",
        "outside_median_s", "pot_share", "invalid_total", "errors",
    ]
    by = {(name, arm): [r for r in rows if r["molecule"] == name and r["arm"] == arm] for name in order for arm in ARMS}

    def med(group, key):
        vals = [_num(r[key]) for r in group if r[key] != ""]
        return statistics.median(vals) if vals else None

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        for name in order:
            for arm in ARMS:
                group = by[(name, arm)]
                if not group:
                    continue
                pot = sum(float(r["pot_s"]) for r in group)
                wall = sum(float(r["wall_s"]) for r in group)
                writer.writerow(
                    {
                        "molecule": name,
                        "arm": arm,
                        "n": len(group),
                        "hits": f"{sum(int(r['hit']) for r in group)}/{len(group)}",
                        "n_3kcal_median": _fmt(med(group, "n_3kcal")),
                        "reference_3kcal": group[-1]["reference_3kcal"],
                        "calls_median": _fmt(med(group, "calls")),
                        "calls_to_hit_median": _fmt(med(group, "calls_to_hit")),
                        "delta_min_median_kcal": _fmt(med(group, "delta_min_kcal")),
                        "wall_median_s": _fmt(med(group, "wall_s")),
                        "pot_median_s": _fmt(med(group, "pot_s")),
                        "outside_median_s": _fmt(med(group, "outside_s")),
                        "pot_share": _fmt(pot / wall if wall else None, 4),
                        "invalid_total": sum(int(r["invalid"]) for r in group),
                        "errors": sum(r["error"] != "" for r in group),
                    }
                )
    write_stats(stats_path, order, by, "openconf")
    if versus_path is not None:
        write_stats(versus_path, order, by, "develop")


def write_stats(path: Path, order: list[str], by: dict, other: str) -> None:
    fields = [
        "molecule", "hits_anneal", f"hits_{other}", "fisher_odds_ratio", "fisher_p",
        "hit_runs_anneal", f"hit_runs_{other}", "mannwhitney_u", "mannwhitney_p",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in order:
            a, o = by[(name, "anneal")], by[(name, other)]
            if not a or not o:
                continue
            ha, ho = sum(int(r["hit"]) for r in a), sum(int(r["hit"]) for r in o)
            odds, p = fisher_exact([[ha, len(a) - ha], [ho, len(o) - ho]], alternative="two-sided")
            ca = [float(r["calls_to_hit"]) for r in a if r["calls_to_hit"] != ""]
            co = [float(r["calls_to_hit"]) for r in o if r["calls_to_hit"] != ""]
            u = mw_p = None
            if ca and co:
                test = mannwhitneyu(ca, co, alternative="two-sided")
                u, mw_p = float(test.statistic), float(test.pvalue)
            writer.writerow(
                {
                    "molecule": name,
                    "hits_anneal": f"{ha}/{len(a)}",
                    f"hits_{other}": f"{ho}/{len(o)}",
                    "fisher_odds_ratio": _fmt(float(odds), 4),
                    "fisher_p": _fmt(float(p), 4),
                    "hit_runs_anneal": len(ca),
                    f"hit_runs_{other}": len(co),
                    "mannwhitney_u": _fmt(u, 1),
                    "mannwhitney_p": _fmt(mw_p, 4),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--versus-develop", type=Path, help="anneal against the develop arm")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    score_runs(args.runs, args.rows, args.workers)
    summarise(args.rows, args.summary, args.stats, args.versus_develop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
