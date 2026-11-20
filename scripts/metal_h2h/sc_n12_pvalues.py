# Copyright (c) anneal developers
"""Anneal against GMIN p-values of the twelve-seed Sutton-Chen panel.

The p-values on the anneal rows of results.csv test the anneal seeds against
the GMIN rows of the anneal harness, e3ed7dad, whose data file used the
Lennard-Jones container, temperature and step. This script tests the same
anneal seeds against the GMIN rows of the corrected harness only, with the
Fisher and Mann-Whitney tests of run_sc_n12.py, and appends one row per
system to pvalues.csv. Each row names, in supersedes, the results.csv row
whose p-values it replaces. Existing rows are left as they are.

    python3 scripts/metal_h2h/sc_n12_pvalues.py            # append missing rows
    python3 scripts/metal_h2h/sc_n12_pvalues.py --check    # exit 1 if a row is missing or differs
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_sc_n12 import EIGHTY, SMALLER, fisher, mann_whitney  # noqa: E402

PANEL = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12"
SEEDS = PANEL / "seeds.csv"
OUT = PANEL / "pvalues.csv"
FIELDS = (
    "system",
    "n_anneal",
    "n_gmin",
    "hits_anneal",
    "hits_gmin",
    "fisher_p",
    "mannwhitney_p_evals",
    "mannwhitney_p_wall",
    "code_revision",
    "gmin_revision",
    "harness_revision",
    "host",
    "supersedes",
    "script",
)
SCRIPT = "scripts/metal_h2h/sc_n12_pvalues.py"
SUPERSEDED = ("fisher_p_vs_gmin", "mannwhitney_p_evals", "mannwhitney_p_wall")


def one(rows: list[dict[str, str]], column: str) -> str:
    values = {r[column] for r in rows}
    if len(values) != 1:
        raise SystemExit(f"expected one {column}, found {sorted(values)}")
    return values.pop()


def compute(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for system in list(EIGHTY) + list(SMALLER):
        rows = [r for r in seed_rows if r["system"] == system]
        anneal = [r for r in rows if r["arm"] == "anneal"]
        own = one(anneal, "harness_revision")
        gmin = [r for r in rows if r["arm"] == "gmin" and r["harness_revision"] != own]
        harness = one(gmin, "harness_revision")
        hits_a = sum(r["hit"] == "1" for r in anneal)
        hits_g = sum(r["hit"] == "1" for r in gmin)
        out.append({
            "system": system,
            "n_anneal": str(len(anneal)),
            "n_gmin": str(len(gmin)),
            "hits_anneal": str(hits_a),
            "hits_gmin": str(hits_g),
            "fisher_p": f"{fisher(hits_a, len(anneal), hits_g, len(gmin)):.6g}",
            "mannwhitney_p_evals": f"{mann_whitney([float(r['evals_to_hit']) for r in anneal], [float(r['evals_to_hit']) for r in gmin]):.6g}",
            "mannwhitney_p_wall": f"{mann_whitney([float(r['wall_s']) for r in anneal], [float(r['wall_s']) for r in gmin]):.6g}",
            "code_revision": one(anneal, "code_revision"),
            "gmin_revision": one(gmin, "code_revision"),
            "harness_revision": harness,
            "host": one(anneal + gmin, "host"),
            "supersedes": f"results.csv system={system} arm=anneal harness_revision={own[:8]}: " + " ".join(SUPERSEDED),
            "script": SCRIPT,
        })
    return out


def load(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str]) -> int:
    computed = compute(load(SEEDS))
    stored = load(OUT)
    key = ("system", "harness_revision")
    have = {tuple(r[k] for k in key): r for r in stored}
    missing = [r for r in computed if tuple(r[k] for k in key) not in have]
    differs = [r for r in computed if tuple(r[k] for k in key) in have and have[tuple(r[k] for k in key)] != r]
    if "--check" in argv:
        for r in missing:
            print(f"{r['system']} {r['harness_revision'][:8]}: no row in {OUT.name}", file=sys.stderr)
        for r in differs:
            print(f"{r['system']} {r['harness_revision'][:8]}: stored row differs from the seeds", file=sys.stderr)
        return 1 if missing or differs else 0
    if differs:
        raise SystemExit(f"{OUT.name} has rows that differ from the seeds; it is append-only, so fix by hand")
    if missing:
        new = not OUT.is_file()
        with OUT.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerows(missing)
    print(f"{len(missing)} rows appended to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
