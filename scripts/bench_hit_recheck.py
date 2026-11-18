#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Recompute bench CSV/TSV hit columns with energy_hits and report disagreements."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SYSTEM_REFERENCES: dict[str, float] = {
    "lj38": -173.928427,
    "lj55": -279.248470,
    "lj75": -397.492331,
    "lj98": -543.665361,
    "lj102": -569.363652,
    "lj103": -575.766131,
    "lj104": -582.086642,
    "lj150": -893.310258,
    "lj31": -133.58642192,
    "au55": -191.03085280,
    "m55r6": -250.286609,
    "m55r14": -220.646208,
    "m80r6": -378.333471,
    "m80r14": -340.811371,
}


def parse_bool(val: str) -> int | None:
    text = val.strip().lower()
    if text in {"1", "true", "yes", "hit"}:
        return 1
    if text in {"0", "false", "no", ""}:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return None


def tracked(paths: list[str]) -> list[Path]:
    out = subprocess.check_output(["git", "ls-files", *paths], cwd=ROOT, text=True)
    return [ROOT / line for line in out.splitlines() if line]


def energy_col(fieldnames: list[str]) -> str | None:
    for name in ("best", "energy", "best_eV", "best_ev", "e", "delta_min_kcal"):
        if name in fieldnames:
            return name
    return None


def hit_col(fieldnames: list[str]) -> str | None:
    """Per-run binary hit flags only (not aggregate seed-count columns named hits)."""
    for name in ("hit", "anneal_hit", "gmin_hit"):
        if name in fieldnames:
            return name
    return None


def reference_for(row: dict[str, str], fieldnames: list[str]) -> float | None:
    for key in ("reference", "ref", "target", "target_energy", "TARGET"):
        if key in fieldnames and row.get(key, "").strip():
            try:
                return float(row[key])
            except ValueError:
                continue
    system = row.get("system", "").strip().lower()
    if system in SYSTEM_REFERENCES:
        return SYSTEM_REFERENCES[system]
    return None


def audit_file(path: Path) -> tuple[int, int] | None:
    with path.open(newline="") as handle:
        dialect = csv.excel_tab if path.suffix == ".tsv" else csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            return None
        fields = list(reader.fieldnames)
        ecol = energy_col(fields)
        hcol = hit_col(fields)
        if not ecol or not hcol:
            return None
        total = 0
        disagree = 0
        for row in reader:
            energy_raw = row.get(ecol, "").strip()
            hit_raw = row.get(hcol, "").strip()
            if not energy_raw or hit_raw == "":
                continue
            try:
                energy = float(energy_raw)
            except ValueError:
                continue
            committed = parse_bool(hit_raw)
            if committed is None:
                continue
            ref = reference_for(row, fields)
            if ref is None:
                continue
            tol = DEFAULT_HIT_TOL
            if "tolerance" in fields and row.get("tolerance", "").strip():
                try:
                    tol = float(row["tolerance"])
                except ValueError:
                    pass
            expected = 1 if energy_hits(energy, ref, tol) else 0
            total += 1
            if expected != committed:
                disagree += 1
        return (total, disagree)


def main() -> int:
    paths = tracked(["bench/**/*.csv", "bench/**/*.tsv"])
    grand_total = 0
    grand_disagree = 0
    any_disagree = False
    for path in sorted(paths):
        stats = audit_file(path)
        if stats is None:
            continue
        total, disagree = stats
        if total == 0:
            continue
        rel = path.relative_to(ROOT).as_posix()
        print(f"{rel}: {disagree}/{total} disagree")
        grand_total += total
        grand_disagree += disagree
        any_disagree |= disagree > 0
    print(f"TOTAL: {grand_disagree}/{grand_total} rows disagree")
    return 1 if any_disagree else 0


if __name__ == "__main__":
    raise SystemExit(main())