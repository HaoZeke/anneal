# Copyright (c) anneal developers
"""gmin_hits counts only code=gmin rows, re-derived from best energy."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SUMMARY = ROOT / "bench" / "g1-develop-stock" / "summary.csv"
PANEL = ROOT / "bench" / "gmin-panel" / "rows.csv"
MANIFEST = ROOT / "bench" / "g0-lj-stock" / "manifest.csv"


def _summarize():
    path = ROOT / "bench" / "g1-develop-stock" / "summarize.py"
    spec = importlib.util.spec_from_file_location("g1_develop_stock_summarize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _references() -> dict[str, float]:
    refs: dict[str, float] = {}
    for row in _rows(MANIFEST):
        if row["cell"].startswith("gmin-"):
            refs[row["cell"][5:]] = float(row["reference"])
    return refs


def _panel_row(code: str, seed: int, best: str, revision: str, hit: str) -> dict[str, str]:
    return {
        "system": "lj38",
        "code": code,
        "seed": str(seed),
        "budget": "400000",
        "hit": hit,
        "evals_to_hit": "10" if hit == "1" else "400000",
        "best": best,
        "code_revision": revision,
    }


def test_gmin_hits_ignore_rows_that_are_not_gmin() -> None:
    """An anneal row that hits must not add to gmin_hits."""
    module = _summarize()
    reference = -173.928427
    hitting = f"{reference:.6f}"
    missing = "-100.0"
    rows: list[dict[str, str]] = []
    for seed in range(2000, 2012):
        rows.append(_panel_row("anneal", seed, hitting, "c2de652dca0bc75e2b0cb4afffac91fd656404e4", "1"))
        rows.append(_panel_row("gmin", seed, hitting, "not-a-gmin-revision", "1"))
        rows.append(_panel_row("gmin", seed, missing, module.GMIN_REVISION, "1"))
    # An earlier GMIN row that hits is replaced by the miss above.
    rows.insert(0, _panel_row("gmin", 2000, hitting, module.GMIN_REVISION, "1"))
    arm = module.gmin_arm(rows, {"lj38": reference})
    assert [row["hit"] for row in arm["lj38"]] == ["0"] * 12
    assert sum(row["hit"] == "1" for row in arm["lj38"]) == 0


def test_a_gmin_row_on_the_anneal_boundary_counts() -> None:
    module = _summarize()
    reference = -173.928427
    boundary = f"{reference + DEFAULT_HIT_TOL:.12f}"
    above = f"{reference + DEFAULT_HIT_TOL + 0.01:.12f}"
    rows = []
    for seed in range(2000, 2012):
        best = boundary if seed == 2000 else above
        rows.append(_panel_row("anneal", seed, above, "c2de652dca0bc75e2b0cb4afffac91fd656404e4", "0"))
        rows.append(_panel_row("gmin", seed, best, module.GMIN_REVISION, "0"))
    arm = module.gmin_arm(rows, {"lj38": reference})
    hits = [row["hit"] for row in arm["lj38"]]
    assert hits[0] == "1"
    assert hits[1:] == ["0"] * 11
    assert energy_hits(float(boundary), reference, DEFAULT_HIT_TOL)
    assert not energy_hits(float(above), reference, DEFAULT_HIT_TOL)


def test_summary_gmin_hits_match_gmin_rows_only() -> None:
    module = _summarize()
    references = _references()
    panel = _rows(PANEL)
    latest: dict[tuple[str, str], dict[str, str]] = {}
    seen: set[tuple[str, str]] = set()
    anneal_hits: dict[str, int] = {}
    for row in panel:
        key = (row["system"], row["seed"])
        if row["code"] == "anneal" and row["code_revision"].startswith("c2de652d") and key not in seen:
            seen.add(key)
            anneal_hits[row["system"]] = anneal_hits.get(row["system"], 0) + int(
                energy_hits(float(row["best"]), references[row["system"]], DEFAULT_HIT_TOL)
            )
        if row["code"] != "gmin" or row["code_revision"] != module.GMIN_REVISION:
            continue
        latest[key] = row
    expected: dict[str, int] = {}
    for (system, _seed), row in latest.items():
        expected[system] = expected.get(system, 0) + int(
            energy_hits(float(row["best"]), references[system], DEFAULT_HIT_TOL)
        )
    summary = {row["system"]: row for row in _rows(SUMMARY)}
    assert summary["lj38"]["gmin_hits"] != str(anneal_hits["lj38"])
    for system, row in summary.items():
        if system in expected:
            assert row["gmin_hits"] == str(expected[system])
            assert row["gmin_revision"] == module.GMIN_REVISION
        else:
            assert row["gmin_hits"] == ""
            assert row["gmin_revision"] == ""
            assert row["fisher_p"] == ""
    # The published summary agrees with the selector, not with a non-gmin source.
    selected = module.gmin_arm(_rows(PANEL), references)
    for system, arm in selected.items():
        assert summary[system]["gmin_hits"] == str(sum(row["hit"] == "1" for row in arm))
