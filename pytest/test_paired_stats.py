# Copyright (c) anneal developers
"""The seed-paired comparison counts discordant seeds and reports per-call time."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "openconf_h2h"))

import paired_stats as ps  # noqa: E402


def row(arm, seed, hit, calls=1000, outside=0.5, invalid=0, molecule="m"):
    return {
        "molecule": molecule, "arm": arm, "seed": str(seed), "hit": str(hit), "calls": str(calls),
        "outside_s": str(outside), "invalid": str(invalid),
    }


def test_exact_mcnemar_is_a_binomial_test_of_the_discordant_split():
    assert ps.mcnemar_exact(0, 0) == 1.0
    assert ps.mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert ps.mcnemar_exact(0, 6) == pytest.approx(2 * 0.5**6)
    assert ps.mcnemar_exact(6, 0) == pytest.approx(ps.mcnemar_exact(0, 6))


def test_identical_arms_are_one_everywhere():
    rows = [row(a, s, s % 2) for a in ("anneal", "develop") for s in range(8)]
    out = ps.compare(rows, "anneal", "develop")[0]
    assert (out["only_arm"], out["only_base"], out["mcnemar_p"], out["wilcoxon_p"]) == (0, 0, "1.0000", "1.0000")
    assert out["outside_ratio_median"] == "1.0000"


def test_discordant_seeds_and_per_call_ratio():
    rows = []
    for s in range(6):
        rows.append(row("anneal", s, 1, outside=0.6, invalid=1))
        rows.append(row("develop", s, 1 if s < 2 else 0, outside=0.5, invalid=2))
    out = ps.compare(rows, "anneal", "develop")[0]
    assert (out["hits_arm"], out["hits_base"], out["only_arm"], out["only_base"]) == (6, 2, 4, 0)
    assert out["mcnemar_p"] == "0.1250"
    assert (out["invalid_arm"], out["invalid_base"]) == (6, 12)
    assert out["outside_us_per_call_arm"] == "600.00" and out["outside_us_per_call_base"] == "500.00"
    assert out["outside_ratio_median"] == "1.2000"
    assert float(out["wilcoxon_p"]) < 0.05


def test_only_seeds_of_both_arms_pair_up_and_local_steps_are_counted():
    rows = [row("anneal", 0, 1), row("anneal", 1, 0), row("develop", 1, 1)]
    steps = {("m", "anneal", "1"): "tric", ("m", "develop", "1"): "lbfgs"}
    out = ps.compare(rows, "anneal", "develop", steps)[0]
    assert out["n"] == 1 and out["only_base"] == 1
    assert out["local_step_arm"] == "tric:1" and out["local_step_base"] == "lbfgs:1"
