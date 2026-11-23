# Copyright (c) anneal developers
"""Box ensembles take every setting of their Rust configuration."""

import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

import anneal
from anneal import _core

ROOT = Path(__file__).resolve().parent.parent
BOX_RS = ROOT / "src" / "methods" / "box_hopping.rs"
COVERAGE_RS = ROOT / "src" / "methods" / "box_hopping" / "coverage.rs"

LOW = np.full(2, -5.0)
HIGH = np.full(2, 5.0)


def wells(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.cos(3.0 * x)) + 0.05 * np.sum(x * x))


def wells_grad(x):
    x = np.asarray(x, dtype=float)
    return -3.0 * np.sin(3.0 * x) + 0.1 * x


def hop(**kwargs):
    return anneal.box_ensemble_optimize(
        wells, LOW, HIGH, budget=3000, seed=4, grad_fn=wells_grad, replicas=4, **kwargs
    )


@pytest.fixture(scope="module")
def default():
    return hop()


def struct_fields(path, name):
    text = path.read_text()
    start = text.index(f"pub struct {name} {{")
    body = text[start : text.index("\n}\n", start)]
    return re.findall(r"^    pub (\w+): ", body, re.MULTILINE)


# Each field of the three Rust configurations and the keyword that sets it.
KEYWORD = {
    ("BoxEnsembleConfig", "replicas"): "replicas",
    ("BoxEnsembleConfig", "budget"): "budget",
    ("BoxEnsembleConfig", "history"): "history",
    ("BoxEnsembleConfig", "membership"): "membership",
    ("BoxEnsembleConfig", "identity_tol"): "identity_tol",
    ("BoxEnsembleConfig", "shared_deposits"): "shared_deposits",
    ("BoxEnsembleConfig", "shared_visit_policy"): "shared_visit_policy",
    ("BoxEnsembleConfig", "escape"): "escape",
    ("GleEscapeConfig", "steps"): "langevin_steps",
    ("GleEscapeConfig", "omega0"): "langevin_omega0",
    ("GleEscapeConfig", "dt"): "langevin_dt",
    ("GleEscapeConfig", "noise"): "langevin_friction",
    ("BoxCoverageConfig", "radius"): "coverage_radius",
    ("BoxCoverageConfig", "height"): "coverage_height",
    ("BoxCoverageConfig", "well_tempering"): "coverage_well_tempering",
    ("BoxCoverageConfig", "peer_weight"): "coverage_peer_weight",
    ("BoxCoverageConfig", "shared"): "coverage_shared",
    ("BoxCoverageConfig", "neighbors"): "coverage_neighbors",
}
STRUCTS = {
    "BoxEnsembleConfig": BOX_RS,
    "GleEscapeConfig": BOX_RS,
    "BoxCoverageConfig": COVERAGE_RS,
}
POSITIONAL = {"replicas", "budget", "history", "membership"}


@pytest.mark.parametrize("struct", sorted(STRUCTS))
def test_every_field_of_the_rust_configuration_has_a_keyword(struct):
    fields = struct_fields(STRUCTS[struct], struct)
    assert fields
    unmapped = [name for name in fields if (struct, name) not in KEYWORD]
    assert unmapped == [], f"{struct} gained {unmapped}; give each a keyword"
    for name in fields:
        keyword = KEYWORD[(struct, name)]
        for driver in (_core.box_ensemble_optimize, _core.ensemble_optimize):
            parameters = inspect.signature(driver).parameters
            assert keyword in parameters, (driver.__name__, keyword)
            if keyword not in POSITIONAL:
                assert parameters[keyword].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_public_wrappers_take_the_same_keywords():
    for driver in (anneal.box_ensemble_optimize, anneal.ensemble_optimize):
        parameters = inspect.signature(driver).parameters
        for keyword in KEYWORD.values():
            assert keyword in parameters, (driver.__name__, keyword)


def test_the_result_reports_the_charge(default):
    assert default["charged"] == default["n_evals"] + default["n_grads"]
    assert default["charged"] <= 3000


def test_the_defaults_are_the_documented_values(default):
    explicit = hop(
        identity_tol=1e-3,
        shared_deposits=8,
        shared_visit_policy="tabu",
        escape="gaussian",
        coverage_shared=True,
        coverage_radius=0.05,
        coverage_height=0.1,
        coverage_well_tempering=5.0,
        coverage_peer_weight=1.0,
        coverage_neighbors=0,
    )
    for key in ("best_val", "n_evals", "n_grads", "hops", "history_minima", "shared_deposits"):
        assert explicit[key] == default[key], key


def test_identity_tol_decides_which_minima_are_one(default):
    merged = hop(identity_tol=1e6)
    split = hop(identity_tol=1e-12)
    assert merged["history_minima"] == 1
    assert split["history_minima"] > default["history_minima"] > 1


def test_shared_deposits_zero_turns_incoming_deposits_off(default):
    assert default["shared_deposits"] > 0
    off = hop(shared_deposits=0)
    assert off["shared_deposits"] == 0
    assert off["coverage_applied_foreign"] == 0


def test_the_visit_policy_changes_how_peers_steer_the_chains(default):
    recognition = hop(shared_visit_policy="recognition")
    key = ("n_evals", "hops", "history_minima", "coverage_repelled_proposals")
    assert [recognition[k] for k in key] != [default[k] for k in key]


def test_langevin_escape_integrates_forces(default):
    langevin = hop(escape="langevin")
    assert langevin["charged"] <= 3000
    assert math.isfinite(langevin["best_val"])
    assert langevin["hops"] != default["hops"]
    assert langevin["n_grads"] > default["n_grads"]


def test_langevin_settings_each_change_the_run():
    key = ("n_evals", "hops", "history_minima", "coverage_repelled_proposals")
    edits = (
        {},
        {"langevin_steps": 4},
        {"langevin_friction": 2.0},
        {"langevin_dt": 0.001},
        {"langevin_omega0": 1.0},
    )
    seen = set()
    for edit in edits:
        run = hop(escape="langevin", **edit)
        seen.add(tuple(run[k] for k in key))
    assert len(seen) == len(edits)


def test_coverage_settings_reach_the_coverage_ledger(default):
    assert default["coverage_applied_foreign"] > 0
    assert default["coverage_repelled_proposals"] > 0
    assert hop(coverage_shared=False)["coverage_applied_foreign"] == 0
    assert hop(coverage_height=0.0)["coverage_repelled_proposals"] == 0
    assert hop(coverage_peer_weight=0.0)["coverage_applied_foreign"] == 0
    assert hop(coverage_neighbors=1)["shared_deposits"] < default["shared_deposits"]
    assert (
        hop(coverage_radius=0.5)["coverage_repelled_proposals"]
        != default["coverage_repelled_proposals"]
    )
    assert hop(coverage_well_tempering=50.0)["hops"] != default["hops"]


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("identity_tol", 0.0),
        ("identity_tol", -1.0),
        ("identity_tol", math.nan),
        ("identity_tol", math.inf),
        ("shared_visit_policy", "sometimes"),
        ("escape", "metropolis"),
        ("coverage_radius", 0.0),
        ("coverage_radius", math.nan),
        ("coverage_height", -0.1),
        ("coverage_well_tempering", 1.0),
        ("coverage_well_tempering", math.nan),
        ("coverage_peer_weight", -1.0),
    ],
)
@pytest.mark.parametrize("driver", ["box_ensemble_optimize", "ensemble_optimize"])
def test_a_value_rust_would_assert_on_is_refused_with_a_typed_error(driver, keyword, value):
    name = "coverage radius" if keyword == "coverage_radius" else keyword
    with pytest.raises(anneal.InvalidArgumentError, match=name):
        getattr(anneal, driver)(
            wells, LOW, HIGH, budget=200, seed=1, grad_fn=wells_grad, **{keyword: value}
        )


@pytest.mark.parametrize(
    "edit, text",
    [
        ({"escape": "langevin", "langevin_steps": 0}, "langevin_steps"),
        ({"escape": "langevin", "langevin_omega0": 0.0}, "langevin_omega0"),
        ({"escape": "langevin", "langevin_dt": -1.0}, "langevin_dt"),
        ({"escape": "langevin", "langevin_friction": 0.0}, "langevin_friction"),
        ({"langevin_steps": 4}, "escape"),
        ({"langevin_friction": 1.0}, "escape"),
    ],
)
def test_langevin_settings_are_checked_before_any_callback(edit, text):
    calls = []

    def counting(x):
        calls.append(1)
        return wells(x)

    with pytest.raises(anneal.InvalidArgumentError, match=text):
        anneal.box_ensemble_optimize(
            counting, LOW, HIGH, budget=200, seed=1, grad_fn=wells_grad, **edit
        )
    assert calls == []


def test_langevin_without_a_gradient_is_refused_not_a_panic():
    with pytest.raises(anneal.InvalidArgumentError, match="grad_fn"):
        anneal.ensemble_optimize(
            wells, LOW, HIGH, budget=200, seed=1, replicas=2, escape="langevin"
        )


def test_ensemble_optimize_takes_the_same_settings_with_a_gradient():
    out = anneal.ensemble_optimize(
        wells,
        LOW,
        HIGH,
        budget=3000,
        seed=4,
        grad_fn=wells_grad,
        replicas=4,
        escape="langevin",
        identity_tol=1e6,
    )
    assert out["history_minima"] == 1
    assert out["charged"] <= 3000


def test_a_setting_selects_hop_chains_for_one_values_only_replica():
    portfolio = anneal.ensemble_optimize(wells, LOW, HIGH, budget=300, seed=2, replicas=1)
    assert portfolio["hops"] == 0
    chains = anneal.ensemble_optimize(
        wells, LOW, HIGH, budget=300, seed=2, replicas=1, identity_tol=1e-3
    )
    assert chains["hops"] > 0


def test_a_grid_of_values_never_reaches_a_rust_panic():
    grid = {
        "identity_tol": [1e-12, 1.0, 1e6],
        "shared_deposits": [0, 1, 10**6],
        "langevin_steps": [1, 10**4],
        "langevin_omega0": [1e-9, 1e6],
        "langevin_dt": [1e-9, 1e6],
        "langevin_friction": [1e-9, 1e6],
        "coverage_radius": [1e-9, 1e6],
        "coverage_height": [0.0, 1e6],
        "coverage_well_tempering": [1.0 + 1e-9, 1e6],
        "coverage_peer_weight": [0.0, 1e6],
        "coverage_neighbors": [0, 10**6],
    }
    runs = 0
    for keyword, values in grid.items():
        for value in values:
            extra = {"escape": "langevin"} if keyword.startswith("langevin") else {}
            for replicas in (1, 3):
                out = anneal.box_ensemble_optimize(
                    wells,
                    LOW,
                    HIGH,
                    budget=120,
                    seed=1,
                    grad_fn=wells_grad,
                    replicas=replicas,
                    **{keyword: value},
                    **extra,
                )
                assert out["charged"] <= 120
                runs += 1
    assert runs == 2 * sum(len(values) for values in grid.values())
