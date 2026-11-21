#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rescore the stored conf-miss4 conformers under two rules.

R-A is the develop comparison: heavy-atom RMSD after a proper Kabsch
fit, minimised over automorphisms. R-K is the kit comparison: R-A, or
the same test after reflecting the candidate through the xy plane.
Polish, the stereo filter, the energy window and the reference are
shared. A miss enters the rank tests as positive infinity. The test
family is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "openconf_h2h"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h2h_common as hc  # noqa: E402
from h2h_score import ENERGY_TIE, RMSD, Molecule  # noqa: E402
from run_gate import (  # noqa: E402
    AA_CELLS,
    ALPHA,
    CELL_ORDER,
    FAMILY,
    GUARDS,
    MOLECULES,
    OUTSIDE_CAP,
    SEEDS,
    adjust,
    call_bands,
    endpoint_loss,
    fisher_worse,
    guard_identity,
    hit_bands,
    kind_of,
    mcnemar_worse,
    median,
    numeric,
    outside_bands,
    ratio,
    rows_for,
    schedule,
    wall_bands,
)

RULES = ("R-A", "R-K")
MIRROR = np.array([1.0, 1.0, -1.0])
ROW_FIELDS = [
    "cell",
    "arm",
    "seed",
    "rule",
    "hit",
    "calls",
    "wall_s",
    "outside_per_call_s",
    "energy",
    "ref_energy",
    "energy_gap",
    "machine_id",
]
FLIP_FIELDS = [
    "cell",
    "arm",
    "seed",
    "hit_ra",
    "hit_rk",
    "class",
    "rmsd",
    "rmsd_inverted",
    "energy_gap",
]


def same_or_mirror(mol: Molecule, a, b) -> bool:
    """Kit rule. ``b`` matches ``a``, or its mirror image does.

    A polished geometry has already passed the stereo check, so a
    mirror image that passes it too is the same species. The Kabsch
    fit keeps proper rotations, so the mirror is invisible to ``same``.
    """
    if mol.same(a, b):
        return True
    flipped = np.asarray(b[0]).reshape(-1, 3) * MIRROR
    return abs(a[1] - b[1]) < ENERGY_TIE and mol.rmsd(a[0], flipped) < RMSD


def load_valid(path: Path) -> list[tuple[np.ndarray, float]]:
    if not path.is_file():
        raise SystemExit(f"missing polished conformers for {path.parent.name}")
    data = np.load(path)
    xs, es, ok = data["x"], data["e"], data["ok"]
    return [(x, float(e)) for x, e, flag in zip(xs, es, ok, strict=True) if flag]


def dedupe_near(mol: Molecule, items: list[tuple[np.ndarray, float]], emin: float):
    """Dedupe the slice that can match the lowest energy.

    A conformer more than the energy window above the minimum cannot
    be the reference and cannot pass ``same`` against it. Inside the
    window every comparison is against another conformer that is also
    inside it, so the kept set matches a full dedupe.
    """
    window = [item for item in items if item[1] < emin + ENERGY_TIE]
    return mol.dedupe(window)


def geometry_id(item) -> str:
    blob = np.ascontiguousarray(item[0], dtype=np.float64).tobytes()
    return hashlib.sha256(blob).hexdigest()


def load_gate_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["phase"] = "session"
    return rows


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
    return {(row["cell"], row["arm"], int(row["seed"])): row for row in rows}


def score_cell(mol: Molecule, run_dir: Path, cell: str) -> dict:
    jobs = [job for job in schedule() if job["cell"] == cell]
    valid = {}
    emin = float("inf")
    for job in jobs:
        directory = run_dir / "jobs" / job["phase"] / cell / str(job["seed"]) / job["arm"]
        items = load_valid(directory / "polished.npz")
        valid[(job["arm"], job["seed"])] = items
        if items:
            emin = min(emin, min(e for _, e in items))
    if emin == float("inf"):
        raise SystemExit(f"no valid conformers for {cell}")
    kept = {key: dedupe_near(mol, items, emin) for key, items in valid.items()}
    pool = [item for job in jobs for item in kept[(job["arm"], job["seed"])]]
    lowest = min(pool, key=lambda item: item[1])
    return {"emin": float(lowest[1]), "ref": lowest, "ref_id": geometry_id(lowest), "kept": kept}


def match_detail(mol: Molecule, ref, items: list) -> dict | None:
    """Closest mirror explanation, if a kept conformer matches only inverted."""
    best = None
    for item in items:
        gap = float(item[1] - ref[1])
        if abs(gap) >= ENERGY_TIE:
            continue
        before = mol.rmsd(ref[0], item[0])
        after = mol.rmsd(ref[0], np.asarray(item[0]).reshape(-1, 3) * MIRROR)
        if after >= RMSD or before < RMSD:
            continue
        if best is None or after < best["rmsd_inverted"]:
            best = {"rmsd": before, "rmsd_inverted": after, "energy_gap": gap}
    return best


def closest_detail(mol: Molecule, ref, items: list) -> dict:
    """Nearest kept conformer, used when a flip is not a mirror image."""
    if not items:
        return {"rmsd": float("inf"), "rmsd_inverted": float("inf"), "energy_gap": float("inf")}
    item = min(items, key=lambda it: mol.rmsd(ref[0], it[0]))
    return {
        "rmsd": mol.rmsd(ref[0], item[0]),
        "rmsd_inverted": mol.rmsd(ref[0], np.asarray(item[0]).reshape(-1, 3) * MIRROR),
        "energy_gap": float(item[1] - ref[1]),
    }


def hits_of(mol: Molecule, ref, items: list, rule: str) -> int:
    if rule == "R-A":
        return int(any(mol.same(ref, item) for item in items))
    return int(any(same_or_mirror(mol, ref, item) for item in items))


def mw_exact(b_values: list[float], a_values: list[float]) -> float:
    """One-sided exact Mann-Whitney, larger B. Ties keep their exact p."""
    if not a_values or not b_values:
        return 1.0
    return float(mannwhitneyu(b_values, a_values, alternative="greater", method="exact").pvalue)


def censored(rows: list[dict], key: str) -> list[float]:
    """Recorded values, with a miss replaced by positive infinity."""
    out = []
    for row in rows:
        if int(row["hit"]) == 0:
            out.append(float("inf"))
        else:
            out.append(float(row[key]))
    return out


def fmt_p(value: float) -> str:
    return f"{value:.4e}"


def fmt_num(value: float) -> str:
    if value != value or value == float("inf"):
        return ""
    if abs(value - round(value)) < 1e-9 and abs(value) > 1:
        return str(int(round(value)))
    return f"{value:.6g}"


def fmt_fixed(value: float) -> str:
    if value != value or abs(value) == float("inf"):
        return ""
    return f"{value:.6f}"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()


def judge(records: list[dict], limits: dict) -> list[dict]:
    verdicts = []
    for cell in CELL_ORDER:
        a = rows_for(records, cell, "A")
        b = rows_for(records, cell, "B")
        n = len(SEEDS)
        hits_a = sum(int(row["hit"]) for row in a)
        hits_b = sum(int(row["hit"]) for row in b)
        calls_a = numeric(a, "calls")
        calls_b = numeric(b, "calls")
        wall_a = numeric(a, "wall_s")
        wall_b = numeric(b, "wall_s")
        out_a = numeric(a, "outside_per_call_s")
        out_b = numeric(b, "outside_per_call_s")
        p_hits = fisher_worse(hits_b, hits_a, n) if kind_of(cell) == "molecule" else 1.0
        p_mcnemar = mcnemar_worse(records, cell)
        p_calls = mw_exact(censored(b, "calls"), censored(a, "calls"))
        p_wall = mw_exact(censored(b, "wall_s"), censored(a, "wall_s"))
        p_out = mw_exact(censored(b, "outside_per_call_s"), censored(a, "outside_per_call_s"))
        med_calls_a, med_calls_b = median(calls_a), median(calls_b)
        med_wall_a, med_wall_b = median(wall_a), median(wall_b)
        med_out_a, med_out_b = median(out_a), median(out_b)
        wall_ratio = ratio(med_wall_b, med_wall_a)
        call_ratio = ratio(med_calls_b, med_calls_a)
        out_ratio = ratio(med_out_b, med_out_a)
        outside_wall = wall_ratio > limits[cell]
        outside_calls = call_ratio > limits.get(f"calls:{cell}", 1.0)
        outside_out = out_ratio > limits.get(f"outside:{cell}", 1.0)
        outside_hits = (hits_a - hits_b) > limits.get(f"hits:{cell}", 0.0)
        identical, bad = (True, [])
        if kind_of(cell) == "guard":
            identical, bad = guard_identity(records, cell)
        losses = []
        if kind_of(cell) == "molecule":
            if endpoint_loss(p_hits, hits_b < hits_a, outside_hits):
                losses.append("hits")
            if endpoint_loss(p_calls, med_calls_b > med_calls_a, outside_calls):
                losses.append("calls")
            if endpoint_loss(p_wall, med_wall_b > med_wall_a, outside_wall):
                losses.append("wall")
            if endpoint_loss(p_out, med_out_b > med_out_a, outside_out):
                losses.append("outside")
            if out_ratio > OUTSIDE_CAP:
                losses.append("outside_cap")
        else:
            if not identical:
                losses.append("identity")
            if endpoint_loss(p_wall, med_wall_b > med_wall_a, outside_wall):
                losses.append("wall")
        ap = rows_for(records, cell, "Ap")
        verdicts.append(
            {
                "cell": cell,
                "kind": kind_of(cell),
                "hits_a": hits_a,
                "hits_b": hits_b,
                "hits_ap": sum(int(row["hit"]) for row in ap) if ap else "",
                "p_hits": p_hits,
                "p_mcnemar": p_mcnemar,
                "p_calls": p_calls,
                "p_wall": p_wall,
                "p_out": p_out,
                "calls_a": med_calls_a,
                "calls_b": med_calls_b,
                "wall_a": med_wall_a,
                "wall_b": med_wall_b,
                "out_a": med_out_a,
                "out_b": med_out_b,
                "out_ratio": out_ratio,
                "wall_limit": limits[cell],
                "identical": identical,
                "mismatch": bad,
                "losses": losses,
                "verdict": "FAIL" if losses else "PASS",
                "n_a": len(a),
                "n_b": len(b),
            }
        )
    return verdicts


def table(verdicts: list[dict]) -> list[str]:
    lines = [
        "| cell | hits A | hits B | hits A' | p hits | p mcnemar | calls A | calls B | p calls | wall A | wall B | p wall | outside A | outside B | ratio | p outside | wall limit | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in verdicts:
        lines.append(
            "| {cell} | {ha} | {hb} | {hp} | {ph} | {pm} | {ca} | {cb} | {pc} | {wa} | {wb} | {pw} | {oa} | {ob} | {ratio} | {po} | {lim} | {verdict} |".format(
                cell=row["cell"],
                ha=f"{row['hits_a']}/24",
                hb=f"{row['hits_b']}/24",
                hp="" if row["hits_ap"] == "" else f"{row['hits_ap']}/24",
                ph=fmt_p(row["p_hits"]),
                pm=fmt_p(row["p_mcnemar"]),
                ca=fmt_num(row["calls_a"]),
                cb=fmt_num(row["calls_b"]),
                pc=fmt_p(row["p_calls"]),
                wa=fmt_num(row["wall_a"]),
                wb=fmt_num(row["wall_b"]),
                pw=fmt_p(row["p_wall"]),
                oa=fmt_num(row["out_a"]),
                ob=fmt_num(row["out_b"]),
                ratio=fmt_num(row["out_ratio"]),
                po=fmt_p(row["p_out"]),
                lim=fmt_num(row["wall_limit"]),
                verdict=row["verdict"],
            )
        )
    return lines


def write_report(path: Path, refs: dict, flips: list[dict], verdicts: dict[str, list[dict]], machine: str) -> None:
    lines = [
        "Copyright (c) anneal developers",
        "",
        "Rescore of the stored conf-miss4 conformers. No search was repeated.",
        "Both arms keep the geometries recorded in the gate session.",
        "",
        "Polish is MMFF94 at dielectric one until the minimiser reports",
        "convergence, the same polished cache the gate scored. A conformer",
        "whose stereocentres or double bonds differ from the SMILES is",
        "invalid and is dropped. Two valid geometries are the same conformer",
        "when their heavy-atom RMSD, minimised over graph automorphisms",
        "after a Kabsch fit, is under 0.25 angstrom and their energies",
        "differ by less than 0.5 kcal/mol.",
        "",
        "R-A applies that comparison. R-K applies it and also accepts a",
        "candidate whose mirror image meets it. The mirror is a reflection",
        "through the xy plane. The Kabsch fit still keeps proper rotations.",
        "",
        "The reference is the lowest valid conformer of the union of that",
        "cell's arms and seeds, including the second develop arm where the",
        "session recorded one. R-K uses the same reference. It does not",
        "replace the reference with a mirror image, and the reference",
        "energy and geometry do not change under R-K.",
        "",
        "## Hits",
        "",
        "| molecule | A R-A | B R-A | A R-K | B R-K |",
        "|---|---|---|---|---|",
    ]
    by_rule = {}
    for rule, rows in verdicts.items():
        by_rule[rule] = {row["cell"]: row for row in rows}
    for cell in MOLECULES:
        ra = by_rule["R-A"][cell]
        rk = by_rule["R-K"][cell]
        lines.append(
            f"| {cell} | {ra['hits_a']}/24 | {ra['hits_b']}/24 | {rk['hits_a']}/24 | {rk['hits_b']}/24 |"
        )
    lines += [
        "",
        "## Reference",
        "",
        "Changed is no on every molecule: one geometry, one energy, both rules.",
        "",
        "| molecule | ref kcal | geometry | changed |",
        "|---|---|---|---|",
    ]
    for cell in MOLECULES:
        ref = refs[cell]
        lines.append(f"| {cell} | {ref['emin']:.6f} | {ref['ref_id']} | no |")
    lines += [
        "",
        "## Flips",
        "",
        "A flip is a seed whose hit status differs between R-A and R-K.",
        "Mirror means the inverted heavy-atom RMSD is under 0.25 angstrom,",
        "the direct RMSD is not, and the energy gap to the reference is",
        "under 0.5 kcal/mol. Anything else is a distinct minimum.",
        "",
    ]
    if not flips:
        lines.append("No seed changes hit status.")
    else:
        lines.append("| molecule | arm | seed | R-A | R-K | class | rmsd | rmsd inverted | energy gap |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in flips:
            lines.append(
                "| {cell} | {arm} | {seed} | {hit_ra} | {hit_rk} | {klass} | {rmsd} | {inv} | {gap} |".format(
                    cell=row["cell"],
                    arm=row["arm"],
                    seed=row["seed"],
                    hit_ra=row["hit_ra"],
                    hit_rk=row["hit_rk"],
                    klass=row["class"],
                    rmsd=row["rmsd"],
                    inv=row["rmsd_inverted"],
                    gap=row["energy_gap"],
                )
            )
        n_mirror = sum(1 for row in flips if row["class"] == "mirror")
        n_distinct = sum(1 for row in flips if row["class"] == "distinct")
        lines += [
            "",
            f"Flips: {len(flips)}. Mirror images: {n_mirror}. Distinct minima: {n_distinct}.",
        ]
    lines += [
        "",
        "## Tests",
        "",
        "The family is unchanged. It is the eight molecules' four endpoints",
        f"(hits, calls, wall, outside time) and the five guard wall endpoints, {FAMILY} tests.",
        "Bonferroni multiplies a family probability by 37 and caps the product at 1.",
        f"A loss needs that adjusted probability below {ALPHA}, a worse hit count",
        "or median, and a median ratio outside the develop band. McNemar is",
        "paired over seeds, is reported beside Fisher, and is not in the family.",
        "",
        "Calls, wall and outside time enter the Mann-Whitney test with a miss",
        "replaced by positive infinity. The medians, the outside-time ratio and",
        f"the {OUTSIDE_CAP:.2f} cap use the recorded finite times. On every molecule",
        "the median ratio of outside time per call must be at most that cap.",
        "",
        "Guard call and outside probabilities below are the same one-sided",
        "exact Mann-Whitney test, including the tied case. They are reported",
        "and they are not members of the 37-test family, so they do not move",
        "a guard verdict. The guard wall endpoint remains the family member.",
        "",
    ]
    for rule in RULES:
        rows = verdicts[rule]
        overall = "PASS" if all(row["verdict"] == "PASS" and row["n_a"] == 24 and row["n_b"] == 24 for row in rows) else "FAIL"
        lines += [f"### {rule}", ""]
        lines.extend(table(rows))
        lines += ["", f"Overall: {overall}", ""]
    lines += [
        "## Session",
        "",
        "- base_ref: 61fa67cf380e46876b569da45e7cc091647a775b",
        "- base_tree: cff796c507a8fa1d0bfb28888c3570b61988055e",
        "- b_tree: 91af99fceaf0591edddb071988c149cc711eaf37",
        "- kit_sha256: 5ca02856b67ecd4a0e938d50d91f73536660d6536cd739befdbcea53d05406f1",
        f"- machine_id: {machine}",
        "- seeds: 2000-2023",
        "- n: 24",
        "- family: 37",
        "- family_unchanged: yes",
        "- guard_calls_outside_in_family: no",
        "- miss_rank: inf",
        "- reference_changes_under_R-K: no",
        "",
    ]
    path.write_text("\n".join(lines))


def apply_hits(records: list[dict], hits: dict[tuple, int]) -> list[dict]:
    out = []
    for row in records:
        copied = dict(row)
        key = (row["cell"], row["arm"], int(row["seed"]))
        if key in hits:
            copied["hit"] = hits[key]
        out.append(copied)
    return out


def limits_for(records: list[dict]) -> dict:
    limits = dict(wall_bands(records, set()))
    for cell, value in call_bands(records).items():
        limits[f"calls:{cell}"] = value
    for cell, value in hit_bands(records).items():
        limits[f"hits:{cell}"] = value
    for cell, value in outside_bands(records).items():
        limits[f"outside:{cell}"] = value
    return limits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    gate = load_gate_rows(args.rows)
    by_key = index_rows(gate)
    machine = gate[0]["machine_id"]
    catalog = hc.molecules()
    run_dir = args.run_dir
    refs = {}
    per_rule: dict[str, dict[tuple, int]] = {rule: {} for rule in RULES}
    energies: dict[tuple, float] = {}
    flips = []
    for cell in MOLECULES:
        print(f"score {cell}", flush=True)
        mol = Molecule(catalog[cell])
        scored = score_cell(mol, run_dir, cell)
        refs[cell] = scored
        jobs = [job for job in schedule() if job["cell"] == cell]
        for job in jobs:
            key = (cell, job["arm"], job["seed"])
            items = scored["kept"][(job["arm"], job["seed"])]
            if items:
                energies[key] = min(e for _, e in items)
            for rule in RULES:
                per_rule[rule][key] = hits_of(mol, scored["ref"], items, rule)
            if job["arm"] not in ("A", "B"):
                continue
            hit_a = per_rule["R-A"][key]
            hit_k = per_rule["R-K"][key]
            recorded = int(by_key[key]["hit"])
            if hit_a != recorded:
                raise SystemExit(f"R-A disagrees with the gate at {cell} {job['arm']} {job['seed']}")
            if hit_a == hit_k:
                continue
            detail = match_detail(mol, scored["ref"], items)
            klass = "mirror"
            if detail is None:
                detail = closest_detail(mol, scored["ref"], items)
                klass = "distinct"
            flips.append(
                {
                    "cell": cell,
                    "arm": job["arm"],
                    "seed": job["seed"],
                    "hit_ra": hit_a,
                    "hit_rk": hit_k,
                    "class": klass,
                    "rmsd": fmt_fixed(detail["rmsd"]),
                    "rmsd_inverted": fmt_fixed(detail["rmsd_inverted"]),
                    "energy_gap": fmt_fixed(detail["energy_gap"]),
                }
            )
    out_rows = []
    for cell in MOLECULES:
        for arm in ("A", "B"):
            for seed in SEEDS:
                key = (cell, arm, seed)
                gate_row = by_key[key]
                energy = energies.get(key)
                ref_e = refs[cell]["emin"]
                gap = "" if energy is None else f"{energy - ref_e:.6f}"
                for rule in RULES:
                    out_rows.append(
                        {
                            "cell": cell,
                            "arm": arm,
                            "seed": seed,
                            "rule": rule,
                            "hit": per_rule[rule][key],
                            "calls": gate_row["calls"],
                            "wall_s": gate_row["wall_s"],
                            "outside_per_call_s": gate_row["outside_per_call_s"],
                            "energy": "" if energy is None else f"{energy:.6f}",
                            "ref_energy": f"{ref_e:.6f}",
                            "energy_gap": gap,
                            "machine_id": machine,
                        }
                    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "rescore-rows.csv", ROW_FIELDS, out_rows)
    write_csv(out_dir / "flips.csv", FLIP_FIELDS, flips)
    verdicts = {}
    for rule in RULES:
        rescored = apply_hits(gate, per_rule[rule])
        verdicts[rule] = judge(rescored, limits_for(rescored))
    write_report(out_dir / "results-rescore.md", refs, flips, verdicts, machine)
    print(f"rows {len(out_rows)} flips {len(flips)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
