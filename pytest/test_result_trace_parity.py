# Copyright (c) anneal developers
"""Parity tests for cluster_search, Ledger, polish, box, and portfolio results."""

from __future__ import annotations

import math

import numpy as np
import pytest

import anneal
from anneal.ase import Anneal


def _sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.dot(x - 0.25, x - 0.25))


def _sphere_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * (x - 0.25)


def test_cluster_search_exposes_outcome_relax_trace_and_phase_fields():
    pot = anneal.PairPotential.lennard_jones(5)
    cfg = anneal.Config.recommended(5)
    cfg.trace_quenched = True
    cfg.energy_trace = True

    out = anneal.cluster_search(pot, None, 5, 600, seed=1, config=cfg)
    # Archive-only keys remain exclusive to ras=True.
    for key in ("charged", "floors", "returned", "events"):
        assert key not in out

    assert isinstance(out["best"], np.ndarray)
    assert out["best"].shape == (15,)
    assert math.isfinite(out["best_energy"])
    assert isinstance(out["final"], np.ndarray)
    assert out["final"].shape == (15,)
    assert math.isfinite(out["final_energy"])

    assert 0 < out["spent"] <= 600
    assert out["hops"] >= 1
    assert out["basins"] >= 1
    assert isinstance(out["screened_out"], int)
    assert isinstance(out["restarts"], int)
    assert len(out["visit_counts"]) == 3
    assert math.isfinite(out["escape_scale"])
    assert math.isfinite(out["escape_threshold"])
    assert out["merge_radius"] > 0.0
    assert math.isfinite(out["mean_step"])
    assert isinstance(out["arms"], list) and len(out["arms"]) >= 1
    assert isinstance(out["improvements"], list) and len(out["improvements"]) >= 1
    assert out["stop"] == "budget"

    # Traces enabled on Config must populate in the returned dict.
    assert isinstance(out["quenched"], list) and len(out["quenched"]) >= 1
    first_q = out["quenched"][0]
    assert len(first_q) == 4
    assert isinstance(first_q[3], bool)
    assert isinstance(out["energy_trace"], list) and len(out["energy_trace"]) >= 1

    # RelaxStats and SearchPhase fields.
    for key in (
        "converged",
        "capped",
        "screens",
        "screen_charged",
        "full_charged",
        "check_charged",
        "overlap_rejected",
        "refused",
        "screen_steps_taken",
        "polish_calls",
    ):
        assert isinstance(out[key], int)
        assert out[key] >= 0
    assert out["search_seconds"] >= 0.0
    assert out["polish_seconds"] >= 0.0

    # Target stop reason when target is met.
    out_target = anneal.cluster_search(pot, None, 5, 600, seed=1, target=0.0)
    assert out_target["stop"] == "target"
    assert anneal.energy_hits(out_target["best_energy"], 0.0)


def test_cluster_search_ras_exposes_full_archive_outcome():
    pot = anneal.PairPotential.lennard_jones(4)
    try:
        out = anneal.cluster_search(pot, None, 4, 500, seed=0, ras=True)
    except ValueError as exc:
        assert "graphkey" in str(exc)
        return
    for key in (
        "best",
        "best_energy",
        "hops",
        "charged",
        "screens",
        "same_floor",
        "artn",
        "best_at",
        "floors",
        "returned",
        "events",
        "nve_unavailable",
    ):
        assert key in out
    assert 0 < out["best_at"] <= out["charged"]


def test_ledger_methods_and_properties():
    led = anneal.Ledger(10)
    assert led.budget == 10
    assert led.spent == 0
    assert led.remaining == 10
    assert led.best == float("inf")
    assert led.best_state is None
    assert led.target is None
    assert led.target_reached is False
    assert led.stopped is False
    assert led.answer_polish_calls == 0
    assert led.start_redraws == 0
    assert led.encounters == []

    led.stop_below(-1.0)
    assert led.target == -1.0
    assert led.below_target(-1.5) is True
    assert led.below_target(-0.5) is False

    assert led.charge() is True
    assert led.spent == 1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    led.record(-0.5, x)
    assert led.best == -0.5
    assert led.target_reached is False
    np.testing.assert_allclose(led.best_state, x)
    assert led.encounters == [(1, -0.5)]

    assert led.charge_frac(0.6) is True
    assert led.spent == 1
    assert led.charge_frac(0.6) is True
    assert led.spent == 2

    led.record(-1.2, x * 2.0)
    assert led.best == -1.2
    assert led.target_reached is True
    assert len(led.encounters) == 2

    assert led.charge_many(8) is True
    assert led.remaining == 0
    assert led.charge() is False


def test_polish_box_portfolio_and_ase_charged_fields():
    low = np.full(3, -2.0, dtype=np.float64)
    high = np.full(3, 2.0, dtype=np.float64)
    x0 = np.array([1.2, -0.8, 0.5], dtype=np.float64)

    pol = anneal.polish(_sphere, _sphere_grad, low, high, x0, max_fevals=80)
    assert "best_grad" in pol
    assert pol["best_grad"] is not None
    assert isinstance(pol["best_grad"], np.ndarray)
    assert pol["best_grad"].shape == (3,)
    assert np.linalg.norm(pol["best_grad"]) < 1e-5

    box_out = anneal.box_ensemble_optimize(
        _sphere, low, high, budget=120, seed=2, grad_fn=_sphere_grad, replicas=2
    )
    assert box_out["charged"] == box_out["n_evals"] + box_out["n_grads"]
    assert isinstance(box_out["history_attempts"], int)
    assert box_out["history_attempts"] >= box_out["history_observations"]

    ens_out = anneal.ensemble_optimize(
        _sphere, low, high, budget=120, seed=2, grad_fn=_sphere_grad, replicas=2
    )
    assert isinstance(ens_out["history_attempts"], int)
    assert ens_out["history_attempts"] >= ens_out["history_observations"]

    port_grad = anneal.global_optimize(
        _sphere, low, high, 200, seed=3, grad_fn=_sphere_grad
    )
    assert "hop_state" in port_grad
    if port_grad["hop_state"] is not None:
        pos, val = port_grad["hop_state"]
        assert isinstance(pos, np.ndarray) and pos.shape == (3,)
        assert math.isfinite(val)

    port_val = anneal.global_optimize(_sphere, low, high, 120, seed=3)
    assert port_val["hop_state"] is None

    # Anneal._hop accumulates cluster_search spent charges on 4+ atoms.
    ase = pytest.importorskip("ase")
    atoms = ase.Atoms(
        "Ar4",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.6, 1.1, 0.0], [0.6, 0.4, 1.0]],
    )
    atoms.calc = anneal.ase.PotentialCalculator(anneal.PairPotential.lennard_jones(4))
    opt = Anneal(atoms, logfile=None, mode="search", budget=120, seed=0)
    opt.step()
    assert opt.charged > 0
