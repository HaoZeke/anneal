# Copyright (c) anneal developers
"""The panel view pools the systems of a panel as documented."""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "no_motif_pins_gate" / "panel_summary.py"
SPEC = importlib.util.spec_from_file_location("no_motif_pins_panels", SCRIPT)
PANELS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PANELS
SPEC.loader.exec_module(PANELS)
GATE = PANELS.run_gate


def test_one_stratum_is_the_fisher_exact_test():
    assert PANELS.stratified_fisher([(9, 3, 2, 10)]) == pytest.approx(
        GATE.fisher_two_sided(9, 3, 2, 10), rel=1e-12
    )
    assert PANELS.stratified_fisher([(12, 0, 0, 12)]) == pytest.approx(7.39602e-07, rel=1e-5)


def test_strata_are_permuted_within_themselves():
    # Two strata of three runs per arm. Every way to hand three of the six
    # outcomes of a stratum to arm A is equally likely, strata are independent.
    strata = [(3, 0, 0, 3), (2, 1, 1, 2)]
    outcomes = [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]
    mass: dict[int, float] = {}
    shares = [list(itertools.combinations(range(6), 3)) for _ in strata]
    count = len(shares[0]) * len(shares[1])
    for first in shares[0]:
        for second in shares[1]:
            total = sum(outcomes[0][i] for i in first) + sum(outcomes[1][i] for i in second)
            mass[total] = mass.get(total, 0.0) + 1.0 / count
    observed = 3 + 2
    expected = sum(m for m in mass.values() if m <= mass[observed] * (1.0 + 1e-9))
    assert PANELS.stratified_fisher(strata) == pytest.approx(expected, rel=1e-12)


def test_strata_without_a_split_add_nothing():
    base = PANELS.stratified_fisher([(9, 3, 2, 10)])
    padded = PANELS.stratified_fisher([(9, 3, 2, 10), (12, 0, 12, 0), (0, 12, 0, 12)])
    assert padded == pytest.approx(base, rel=1e-12)


def test_one_stratum_is_the_mann_whitney_test_without_continuity():
    x = [1, 2, 2, 3, 5, 5, 5, 8]
    y = [2, 3, 3, 5, 9, 9, 10, 10, 10]
    p, z = PANELS.van_elteren([(x, y)])
    assert p == pytest.approx(0.07095043811925086, rel=1e-9)
    assert z < 0.0
    swapped, z_swapped = PANELS.van_elteren([(y, x)])
    assert swapped == pytest.approx(p)
    assert z_swapped == pytest.approx(-z)


def test_van_elteren_weights_each_stratum_by_its_size_plus_one():
    strata = [([1, 2], [3, 4]), ([1, 3], [2, 4])]
    p, z = PANELS.van_elteren(strata)
    assert z == pytest.approx(-0.6 / math.sqrt(2.0 / 15.0), rel=1e-12)
    assert p == pytest.approx(0.10034824646229074, rel=1e-9)


def test_all_tied_strata_give_p_one():
    assert PANELS.van_elteren([([4e6] * 12, [4e6] * 12)]) == (1.0, 0.0)


SYSTEMS = PANELS.run_gate.BLJ_SYSTEMS


def row(system, arm, seed, hit, calls, wall=2.0, inside=1.5, status="ok"):
    return {
        "system": system,
        "arm": arm,
        "seed": str(seed),
        "position": "1",
        "budget": "1000000",
        "hit": str(hit),
        "evals_to_hit": str(calls),
        "charged": str(calls),
        "best": "-100.0",
        "wall_s": f"{wall}",
        "pot_s": f"{inside}",
        "outside_s": f"{wall - inside}",
        "status": status,
        "library": "x",
        "harness": "y",
    }


def table(systems):
    return {s: {"system": s, "budget": "1000000", "reference": "-100.0"} for s in systems}


def build(systems, shift):
    rows = []
    for index, system in enumerate(systems):
        for seed in PANELS.run_gate.SEEDS:
            base = int((index + 1) * 1000 * ((seed - 2000) + 1))
            rows.append(row(system, "A", seed, 1, base))
            rows.append(row(system, "B", seed, 1, base + int(shift * (index + 1) * 1000)))
    return rows


def blj(rows):
    return PANELS.panel_summary("BLJ", SYSTEMS, rows, table(SYSTEMS))


def test_equal_arms_pass_and_return_the_same_results():
    summary = blj(build(SYSTEMS, 0))
    assert summary["verdict"] == "PASS"
    assert summary["identical_runs"] == str(len(SYSTEMS) * len(PANELS.run_gate.SEEDS))
    assert summary["hits_a"] == summary["hits_b"] == str(len(SYSTEMS) * 12)
    assert summary["p_calls"] == "1"
    assert summary["wall_ratio"] == "1.0000"
    assert summary["inside_ratio"] == "1.0000"


def test_a_small_loss_on_every_system_is_a_panel_loss():
    rows = []
    for index, system in enumerate(SYSTEMS):
        scale = (index + 1) * 1000
        for seed in PANELS.run_gate.SEEDS:
            i = seed - 2000 + 1
            rows.append(row(system, "A", seed, 1, int(scale * i)))
            rows.append(row(system, "B", seed, 1, int(scale * (i + 2.5))))
    summary = blj(rows)
    systems = [GATE.system_summary(s, rows, table(SYSTEMS)[s]) for s in SYSTEMS]
    assert all(float(r["mw_p_calls"]) > 0.05 for r in systems)
    assert summary["lost_systems"] == ""
    assert summary["loss_calls"] == "1"
    assert float(summary["p_calls"]) < 0.05
    assert summary["calls_direction"] == "B more calls"
    assert summary["verdict"] == "FIX"
    assert PANELS.overall([summary]) == "FIX: BLJ"


def test_the_same_shift_in_favour_of_b_is_not_a_loss():
    rows = []
    for index, system in enumerate(SYSTEMS):
        scale = (index + 1) * 1000
        for seed in PANELS.run_gate.SEEDS:
            i = seed - 2000 + 1
            rows.append(row(system, "A", seed, 1, int(scale * (i + 2.5))))
            rows.append(row(system, "B", seed, 1, int(scale * i)))
    summary = blj(rows)
    assert float(summary["p_calls"]) < 0.05
    assert summary["calls_direction"] == "B fewer calls"
    assert summary["loss_calls"] == "0"
    assert summary["verdict"] == "PASS"


def test_a_lost_system_names_itself_and_fixes_the_panel():
    rows = build(SYSTEMS, 0)
    for r in rows:
        if r["system"] == "blj130n98" and r["arm"] == "B":
            r.update(hit="0", evals_to_hit="1000000")
    summary = blj(rows)
    assert "blj130n98:hits+calls" in summary["lost_systems"]
    assert summary["loss_hits"] == "1"
    assert summary["verdict"] == "FIX"


def test_a_failed_run_of_b_is_a_loss_and_not_a_timing():
    rows = build(SYSTEMS, 0)
    for r in rows:
        if r["system"] == "blj105n38" and r["arm"] == "B" and r["seed"] == "2003":
            r.update(hit="0", evals_to_hit="1000000", status="exit1", wall_s="0.5")
    summary = blj(rows)
    assert summary["failed_b"] == "1"
    assert summary["failed_a"] == "0"
    assert "blj105n38:failure" in summary["lost_systems"]
    assert summary["verdict"] == "FIX"
    assert float(summary["wall_b"]) == pytest.approx(float(summary["wall_a"]) - 2.0)


def test_a_panel_of_one_system_is_that_system():
    rows = build(("m80r6",), 0)
    summary = PANELS.panel_summary("Morse M80 rho6", ("m80r6",), rows, table(("m80r6",)))
    assert summary["verdict"] == "PASS"
    assert summary["systems"] == "1"
    assert summary["p_hits"] == "1"


def test_missing_runs_leave_the_panel_incomplete():
    rows = build(SYSTEMS[:3], 0)
    assert blj(rows)["verdict"] == "INCOMPLETE"
