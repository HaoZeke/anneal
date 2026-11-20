# Copyright (c) anneal developers
"""``Ledger`` exposes the accounting of the Rust ``Ledger``."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

import anneal
from anneal import Ledger

ROOT = Path(__file__).resolve().parent.parent
HOPPING_RS = ROOT / "src" / "methods" / "cluster_hopping.rs"

# Each public method of the Rust ledger and the Python name that reaches it.
# The rest belong to the search drivers, which build their own ledger and
# are not callable from Python with one.
REACHED = {
    "new": "Ledger",
    "stop_below": "stop_below",
    "target": "target",
    "below_target": "below_target",
    "target_reached": "target_reached",
    "slice": "slice",
    "charge": "charge",
    "charge_frac": "charge_frac",
    "charge_many": "charge_many",
    "record": "record",
    "encounters": "encounters",
    "replace_best": "replace_best",
    "grant": "grant",
    "budget": "budget",
    "remaining": "remaining",
    "spent": "spent",
    "best": "best",
    "best_state": "best_state",
    "record_quench_boundary": "record_quench_boundary",
    "quench_boundaries": "quench_boundaries",
}
DRIVER_ONLY = {
    "keep_answer_seats": "polish seats of a search that runs a final polish",
    "answer_seats": "polish seats of a search that runs a final polish",
    "hold_tail": "tail reserve of a search driver",
    "release_tail": "tail reserve of a search driver",
    "answer_polish_calls": "count of the polish a search driver ran",
    "set_start_attempts": "retries of a search's first quench",
    "start_redraws": "geometries a search drew after a failed first quench",
    "with_stop": "the stop flag of a callback harness, which cluster_search owns",
    "bind_stop": "the stop flag of a callback harness, which cluster_search owns",
    "stopped": "the stop flag of a callback harness, which cluster_search owns",
    "is_diagnostic_quench": "marks a relaxation inside a search driver",
}


def rust_methods():
    text = HOPPING_RS.read_text()
    start = text.index("impl Ledger {")
    body = text[start : text.index("\n}\n", start)]
    return re.findall(r"^    pub fn (\w+)", body, re.MULTILINE)


def test_the_impl_was_read():
    methods = rust_methods()
    assert "charge" in methods
    assert len(methods) > 20


def test_every_public_method_is_reached_or_named_as_driver_only():
    missing = [
        name
        for name in rust_methods()
        if name not in REACHED and name not in DRIVER_ONLY
    ]
    assert missing == [], f"Ledger gained {missing}; expose each or say why not"
    stale = (set(REACHED) | set(DRIVER_ONLY)) - set(rust_methods()) - {"spent", "best", "best_state"}
    assert stale == set()


def test_every_reached_method_exists_in_python():
    for rust, python in REACHED.items():
        if python == "Ledger":
            continue
        assert hasattr(Ledger, python), (rust, python)


def test_charging_counts_until_the_budget_is_gone():
    led = Ledger(3)
    assert [led.charge() for _ in range(5)] == [True, True, True, False, False]
    assert (led.spent, led.remaining, led.budget) == (3, 0, 3)


def test_charge_many_settles_partway_and_reports_it():
    led = Ledger(10)
    assert led.charge_many(4) is True
    assert led.spent == 4
    assert led.charge_many(20) is False
    assert (led.spent, led.remaining) == (10, 0)


def test_charge_frac_pays_a_whole_unit_each_time_the_sum_crosses_one():
    led = Ledger(10)
    for _ in range(3):
        assert led.charge_frac(0.4) is True
    assert led.spent == 1
    assert led.charge_frac(0.0) is True
    assert led.charge_frac(math.nan) is True
    assert led.spent == 1
    assert Ledger(0).charge_frac(2.0) is False


def test_record_keeps_the_lowest_finite_value_and_its_state():
    led = Ledger(10)
    assert led.best == math.inf and led.best_state is None
    led.record(5.0, np.array([1.0, 2.0]))
    led.record(7.0, np.array([9.0, 9.0]))
    led.record(math.nan, np.array([8.0, 8.0]))
    led.record(-math.inf, np.array([8.0, 8.0]))
    assert led.best == 5.0
    assert np.array_equal(led.best_state, [1.0, 2.0])
    led.record(3.0, np.array([0.0, 1.0]))
    assert led.best == 3.0
    assert np.array_equal(led.best_state, [0.0, 1.0])


def test_best_state_is_a_copy():
    led = Ledger(10)
    state = np.array([1.0, 2.0])
    led.record(1.0, state)
    state[0] = 99.0
    got = led.best_state
    got[1] = -5.0
    assert np.array_equal(led.best_state, [1.0, 2.0])


def test_encounters_name_the_charge_at_each_new_incumbent():
    led = Ledger(100)
    led.charge_many(3)
    led.record(4.0, np.zeros(2))
    led.charge_many(5)
    led.record(6.0, np.zeros(2))
    led.record(2.0, np.zeros(2))
    assert led.encounters == [(3, 4.0), (8, 2.0)]


def test_encounters_are_capped_like_the_outcome_improvements():
    text = HOPPING_RS.read_text()
    cap = int(re.search(r"const IMPROVEMENT_CAP: usize = (\d+);", text).group(1))
    led = Ledger(10**6)
    for i in range(cap + 10):
        led.record(-float(i), np.zeros(1))
    assert len(led.encounters) == cap
    assert led.best == -float(cap + 9)


def test_replace_best_moves_value_and_state_together():
    led = Ledger(10)
    led.record(1.0, np.array([1.0]))
    led.replace_best(1.5, np.array([2.0]))
    assert led.best == 1.5
    assert np.array_equal(led.best_state, [2.0])


@pytest.mark.parametrize("value, state", [(math.nan, [1.0]), (math.inf, [1.0]), (1.0, [math.nan])])
def test_replace_best_refuses_what_rust_would_ignore(value, state):
    led = Ledger(10)
    led.record(1.0, np.array([1.0]))
    with pytest.raises(anneal.InvalidArgumentError, match="finite"):
        led.replace_best(value, np.array(state))
    assert led.best == 1.0


def test_stop_below_sets_a_target_the_incumbent_can_reach():
    led = Ledger(10)
    assert led.target is None and led.target_reached is False
    led.stop_below(2.0)
    assert led.target == 2.0
    assert led.below_target(1.0) and led.below_target(2.0) and not led.below_target(2.5)
    led.record(3.0, np.zeros(1))
    assert led.target_reached is False
    led.record(1.5, np.zeros(1))
    assert led.target_reached is True


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_stop_below_refuses_a_threshold_rust_would_ignore(value):
    led = Ledger(10)
    with pytest.raises(anneal.InvalidArgumentError, match="finite"):
        led.stop_below(value)
    assert led.target is None


def test_grant_adds_charged_evaluations_after_the_ceiling():
    led = Ledger(2)
    led.charge_many(2)
    assert led.charge() is False
    led.grant(3)
    assert led.budget == 5 and led.remaining == 3
    assert led.charge() is True
    led.grant(2**64 - 1)
    assert led.remaining > 0


def test_a_slice_has_its_own_spend():
    parent = Ledger(10)
    parent.charge_many(4)
    child = parent.slice(3)
    assert (child.budget, child.spent) == (3, 0)
    child.charge_many(3)
    assert parent.spent == 4
    parent.charge_many(child.spent)
    assert parent.spent == 7


def test_a_boundary_with_a_fresh_gradient_is_validated():
    led = Ledger(100)
    led.charge_many(7)
    assert led.record_quench_boundary(2, -1.5, np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    (boundary,) = led.quench_boundaries
    assert boundary["status"] == "validated"
    assert boundary["charged_calls"] == 5
    assert boundary["energy"] == -1.5
    assert np.array_equal(boundary["state"], [1.0, 2.0])
    assert np.array_equal(boundary["gradient"], [0.0, 0.0])


@pytest.mark.parametrize(
    "energy, state, gradient",
    [
        (-1.0, [1.0, 2.0], None),
        (-1.0, [1.0, 2.0], [0.0]),
        (-1.0, [1.0, 2.0], [0.0, math.nan]),
        (math.nan, [1.0, 2.0], [0.0, 0.0]),
        (-1.0, [math.inf, 2.0], [0.0, 0.0]),
    ],
)
def test_a_boundary_without_good_evidence_is_rejected_and_keeps_its_work(energy, state, gradient):
    led = Ledger(100)
    led.charge_many(4)
    grad = None if gradient is None else np.array(gradient)
    assert led.record_quench_boundary(0, energy, np.array(state), grad)
    (boundary,) = led.quench_boundaries
    assert boundary["status"] == "rejected"
    assert boundary["charged_calls"] == 4
    assert boundary["gradient"] is None


def test_a_relaxation_that_spent_nothing_is_not_a_boundary():
    led = Ledger(100)
    led.charge_many(3)
    assert led.record_quench_boundary(3, 0.0, np.zeros(2), np.zeros(2)) is False
    assert led.record_quench_boundary(9, 0.0, np.zeros(2), np.zeros(2)) is False
    assert led.quench_boundaries == []


def test_repr_names_the_state():
    led = Ledger(5)
    led.charge()
    assert repr(led) == "Ledger(budget=5, spent=1, best=inf)"
