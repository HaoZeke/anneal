# Copyright (c) anneal developers
"""Parity tests for cluster_search, Ledger, polish, box, and portfolio results."""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

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
        assert str(exc) == "ras=True requires building with the graphkey feature"
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


def test_core_docstrings_match_live_result_fields():
    # Every public callable and class on anneal._core must carry a docstring.
    for name in sorted(dir(anneal._core)):
        if name.startswith("_"):
            continue
        obj = getattr(anneal._core, name)
        if callable(obj):
            assert getattr(obj, "__doc__", None), f"anneal._core.{name} has no __doc__"

    # 1. anneal._core.cluster_search (standard + rigid + ras=True) vs live result dicts.
    pot = anneal.PairPotential.lennard_jones(4)
    cs_out = anneal._core.cluster_search(pot, None, 4, 120, seed=0)
    rigid_out = anneal._core.cluster_search(
        anneal.GminTip4p(2), None, 2, 40, seed=0, rigid=True
    )
    assert "rigid_path" in rigid_out
    rigid_path_keys = set(rigid_out["rigid_path"].keys())
    cs_core_doc = anneal._core.cluster_search.__doc__ or ""
    cs_wrap_doc = anneal.cluster_search.__doc__ or ""
    for key in sorted(set(cs_out.keys()) | set(rigid_out.keys()) | rigid_path_keys):
        assert f"`{key}`" in cs_core_doc, (
            f"anneal._core.cluster_search.__doc__ missing result key {key!r}"
        )
        assert f"``{key}``" in cs_wrap_doc, (
            f"anneal.cluster_search.__doc__ missing result key {key!r}"
        )
    assert "Returns `{best, best_energy, hops, nve_unavailable}`" not in cs_core_doc
    assert "Ignored once" not in cs_core_doc
    assert "at energy scale one" not in cs_core_doc

    repo_root = Path(__file__).resolve().parents[1]
    py_rs = (repo_root / "src" / "python.rs").read_text(encoding="utf-8")
    ras_fn_match = re.search(
        r"#\[cfg\(feature = \"graphkey\"\)\]\s*fn cluster_archive_search.*?\n\}",
        py_rs,
        re.S,
    )
    assert ras_fn_match is not None
    expected_ras_keys = set(
        re.findall(r'dict\.set_item\(\s*"([^"]+)"', ras_fn_match.group(0))
    )
    assert len(expected_ras_keys) == 14

    try:
        ras_out = anneal._core.cluster_search(pot, None, 4, 200, seed=0, ras=True)
    except ValueError as exc:
        assert str(exc) == "ras=True requires building with the graphkey feature"
    else:
        assert set(ras_out.keys()) == expected_ras_keys

    for key in sorted(expected_ras_keys):
        assert f"`{key}`" in cs_core_doc, (
            f"anneal._core.cluster_search.__doc__ missing ras=True key {key!r}"
        )
        assert f"``{key}``" in cs_wrap_doc, (
            f"anneal.cluster_search.__doc__ missing ras=True key {key!r}"
        )

    # 2. anneal._core.Ledger docstring must describe Ledger (no orphaned Config text).
    ledger_doc = anneal._core.Ledger.__doc__ or ""
    assert "Config.recommended" not in ledger_doc
    assert "Driver settings" not in ledger_doc
    for attr in (
        "budget",
        "spent",
        "remaining",
        "best",
        "best_state",
        "target",
        "target_reached",
        "stopped",
        "answer_polish_calls",
        "start_redraws",
        "encounters",
    ):
        assert f"`{attr}`" in ledger_doc, f"Ledger.__doc__ missing {attr!r}"

    # 3. Other touched _core optimizer/quench docstrings vs live result dicts.
    low = np.full(3, -2.0, dtype=np.float64)
    high = np.full(3, 2.0, dtype=np.float64)
    x0 = np.array([1.2, -0.8, 0.5], dtype=np.float64)

    q_x0 = np.array(
        [0.0, 0.0, 0.0, 1.12, 0.0, 0.0, 0.56, 0.97, 0.0, 0.56, 0.32, 0.91],
        dtype=np.float64,
    )
    q_out = anneal._core.quench(pot, q_x0, 1e-4)
    q_doc = anneal._core.quench.__doc__ or ""
    for key in sorted(q_out.keys()):
        assert f"`{key}`" in q_doc, f"quench.__doc__ missing {key!r}"
    assert "PairPotential" in q_doc

    pol_out = anneal._core.polish(_sphere, _sphere_grad, low, high, x0, max_fevals=60)
    for key in sorted(pol_out.keys()):
        assert f"`{key}`" in (anneal._core.polish.__doc__ or ""), (
            f"_core.polish.__doc__ missing {key!r}"
        )
        assert f"``{key}``" in (anneal.polish.__doc__ or ""), (
            f"polish.__doc__ missing {key!r}"
        )

    gle_out = anneal._core.gle_langevin(_sphere, _sphere_grad, low, high, 60, seed=1)
    for fn in (
        anneal._core.gle_langevin,
        anneal._core.gle_langevin_preconditioned,
        anneal._core.gle_langevin_objective,
        anneal._core.gle_langevin_preconditioned_objective,
    ):
        doc = fn.__doc__ or ""
        for key in sorted(gle_out.keys()):
            assert f"`{key}`" in doc, f"{fn.__name__}.__doc__ missing {key!r}"
    for fn in (
        anneal.gle_langevin,
        anneal.gle_langevin_preconditioned,
        anneal.gle_langevin_objective,
        anneal.gle_langevin_preconditioned_objective,
    ):
        doc = fn.__doc__ or ""
        for key in sorted(gle_out.keys()):
            assert re.search(rf"\b{re.escape(key)}\b", doc), (
                f"anneal.{fn.__name__}.__doc__ missing {key!r}"
            )

    dmc_out = anneal._core.dmc_population_optimize(_sphere, low, high, 80, seed=1)
    for key in sorted(dmc_out.keys()):
        assert f"`{key}`" in (anneal._core.dmc_population_optimize.__doc__ or ""), (
            f"_core.dmc_population_optimize.__doc__ missing {key!r}"
        )
        assert re.search(
            rf"\b{re.escape(key)}\b",
            anneal.dmc_population_optimize.__doc__ or "",
        ), f"dmc_population_optimize.__doc__ missing {key!r}"

    box_out = anneal._core.box_ensemble_optimize(
        _sphere, low, high, 100, seed=1, grad_fn=_sphere_grad, replicas=2
    )
    box_doc = anneal._core.box_ensemble_optimize.__doc__ or ""
    assert "langevin" in box_doc
    for key in sorted(box_out.keys()):
        assert f"`{key}`" in box_doc, (
            f"_core.box_ensemble_optimize.__doc__ missing {key!r}"
        )

    ens_out = anneal._core.ensemble_optimize(
        _sphere, low, high, 100, seed=1, grad_fn=_sphere_grad, replicas=2
    )
    ens_doc = anneal._core.ensemble_optimize.__doc__ or ""
    assert "langevin" in ens_doc
    for key in sorted(ens_out.keys()):
        assert f"`{key}`" in ens_doc, (
            f"_core.ensemble_optimize.__doc__ missing {key!r}"
        )

    go_single = anneal._core.global_optimize(
        _sphere, low, high, 120, seed=1, grad_fn=_sphere_grad
    )
    go_multi = anneal._core.global_optimize(
        _sphere, low, high, 120, seed=1, grad_fn=_sphere_grad, replicas=2
    )
    go_doc = anneal._core.global_optimize.__doc__ or ""
    go_wrap_doc = anneal.global_optimize.__doc__ or ""
    for key in sorted(set(go_single.keys()) | set(go_multi.keys())):
        assert f"`{key}`" in go_doc, (
            f"_core.global_optimize.__doc__ missing {key!r}"
        )
        assert f"``{key}``" in go_wrap_doc, (
            f"global_optimize.__doc__ missing {key!r}"
        )
    for param in inspect.signature(anneal._core.global_optimize).parameters:
        assert re.search(rf"\b{re.escape(param)}\b", go_doc), (
            f"_core.global_optimize.__doc__ missing parameter {param!r}"
        )
    for key in sorted(go_single.keys()):
        assert f"`{key}`" in (anneal._core.global_optimize_objective.__doc__ or ""), (
            f"_core.global_optimize_objective.__doc__ missing {key!r}"
        )
        assert f"``{key}``" in (anneal.global_optimize_objective.__doc__ or ""), (
            f"global_optimize_objective.__doc__ missing {key!r}"
        )
