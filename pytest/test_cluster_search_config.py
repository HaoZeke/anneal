# Copyright (c) anneal developers
"""``cluster_search(config=)`` and the full ``Outcome`` it returns."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

import anneal
from anneal import Config

ROOT = Path(__file__).resolve().parent.parent
HOPPING_RS = ROOT / "src" / "methods" / "cluster_hopping.rs"


def energy(x):
    return float(np.sum(x**2))


def gradient(x):
    return 2.0 * x


def search(n=4, budget=400, seed=3, full=True, **kwargs):
    return anneal.cluster_search(
        energy, gradient, n=n, budget=budget, seed=seed, full=full, **kwargs
    )


def outcome_fields():
    text = HOPPING_RS.read_text()
    start = text.index("pub struct Outcome {")
    body = text[start : text.index("\n}\n", start)]
    return re.findall(r"^    pub (\w+): ", body, re.MULTILINE)


RENAMED = {"best": "best_energy", "best_state": "best"}


def test_the_struct_was_read():
    fields = outcome_fields()
    assert "accepted_transitions" in fields
    assert len(fields) > 50


def test_every_outcome_field_is_a_key_of_the_result():
    result = search(budget=300)
    missing = [
        RENAMED.get(name, name)
        for name in outcome_fields()
        if RENAMED.get(name, name) not in result
    ]
    assert missing == []
    assert "stop_reason" in result


def test_the_result_carries_nothing_the_struct_lacks():
    result = search(budget=300)
    known = {RENAMED.get(name, name) for name in outcome_fields()}
    known |= {"stop_reason", "local_step", "quench_seconds"}
    extra = sorted(set(result) - known)
    assert extra == [], extra


def test_the_default_result_keeps_its_short_shape():
    result = search(budget=200, full=False)
    assert set(result) == {
        "best",
        "best_energy",
        "hops",
        "accepted",
        "nve_unavailable",
        "local_step",
        "quench_seconds",
    }
    assert set(result) < set(search(budget=200))


def test_full_adds_keys_and_changes_no_value_of_the_short_result():
    short = search(budget=300, full=False)
    full = search(budget=300)
    for key in ("best_energy", "hops", "accepted", "nve_unavailable", "local_step"):
        assert full[key] == short[key], key
    assert np.array_equal(full["best"], short["best"])


@pytest.mark.parametrize(
    "preset, flags",
    [
        (lambda n: Config.recommended(n), {}),
        (lambda n: Config.for_cluster(n), {"recommended": False}),
        (lambda n: Config.derived(n), {"derived": True}),
        (lambda n: Config.communicating(n), {"communicating": True}),
    ],
    ids=["recommended", "for_cluster", "derived", "communicating"],
)
def test_a_preset_config_reproduces_the_flag_run(preset, flags):
    by_flags = search(n=5, budget=500, **flags)
    by_config = search(n=5, budget=500, config=preset(5))
    for key in ("best_energy", "hops", "charged", "basins", "visit_counts", "screened_out"):
        assert by_config[key] == by_flags[key], key
    assert np.array_equal(by_config["final_state"], by_flags["final_state"])


def test_a_length_scale_flag_equals_the_config_it_builds():
    by_flags = search(n=5, budget=500, length_scale=1.5)
    cfg = Config.with_scales(5, 1.5, 1.0).replace(
        allocate_moves=True, return_screen=True, orbit_complete_on_new=True
    )
    by_config = search(n=5, budget=500, config=cfg)
    assert by_config["charged"] == by_flags["charged"]
    assert by_config["best_energy"] == by_flags["best_energy"]


def test_an_edited_config_changes_the_run():
    base = Config.recommended(5)
    capped = search(n=5, budget=2000, config=base.replace(max_hops=3))
    assert capped["hops"] <= 3
    assert capped["stop"] == "max_hops"
    free = search(n=5, budget=2000, config=base)
    assert free["hops"] > capped["hops"]


def test_stop_codes_name_the_reason_the_run_ended():
    assert search(budget=150)["stop"] == "budget"
    assert search(n=4, budget=5000, target=1e9)["stop"] == "target"
    assert search(budget=5000, config=Config.recommended(4).replace(max_hops=1))["stop"] == "max_hops"


def test_a_config_for_the_wrong_cluster_size_is_a_typed_error():
    with pytest.raises(anneal.InvalidArgumentError, match="config has 5 points"):
        search(n=4, config=Config.recommended(5))


@pytest.mark.parametrize(
    "clash",
    [
        {"recommended": False},
        {"derived": True},
        {"communicating": True},
        {"length_scale": 1.5},
        {"rigid": True},
    ],
)
def test_flags_that_a_config_already_decides_are_refused(clash):
    with pytest.raises(anneal.InvalidArgumentError, match="Config"):
        search(config=Config.recommended(4), **clash)


def test_a_molecular_config_needs_a_start():
    water = ([8, 1, 1, 8, 1, 1], [[0, 1, 2], [3, 4, 5]])
    with pytest.raises(anneal.InvalidArgumentError, match="start"):
        search(n=6, config=Config.for_molecular(*water))


def test_a_rigid_body_config_runs():
    result = search(n=3, budget=400, config=Config.communicating_tip4p(3))
    assert result["best"].shape == (18,)


def test_the_result_arrays_and_records_have_the_documented_shape():
    result = search(n=4, budget=600, config=Config.recommended(4).replace(max_hops=20))
    assert result["final_state"].shape == (12,)
    assert math.isfinite(result["final_energy"])
    assert len(result["visit_counts"]) == 3
    assert result["accepted_transitions"]
    record = result["accepted_transitions"][0]
    assert set(record) == {
        "hop",
        "action",
        "from_energy",
        "to_energy",
        "from_state",
        "to_state",
        "from_gradient",
        "to_gradient",
        "validated",
        "adopted",
    }
    assert record["from_state"].shape == (12,)
    counts = result["proposal_counts"]
    assert counts
    assert set(counts[0]) == {"move_type", "proposed", "accepted", "improved"}


@pytest.mark.parametrize(
    "edit, key, kind",
    [
        ({"energy_bias": True}, "energy_bias", dict),
        ({"superbasin_features": True, "superbasin_report": True, "superbasin_period": 20}, "superbasin", dict),
        ({"delayed_acceptance": True}, "delayed", tuple),
        ({"energy_trace": True}, "energy_trace", list),
    ],
    ids=["energy_bias", "superbasin", "delayed", "energy_trace"],
)
def test_optional_reports_appear_exactly_when_the_config_asks_for_them(edit, key, kind):
    base = Config.recommended(6)
    off = search(n=6, budget=3000, config=base)
    on = search(n=6, budget=3000, config=base.replace(**edit))
    assert not off[key]
    assert isinstance(on[key], kind)


def test_the_superbasin_report_names_its_counters():
    cfg = Config.recommended(6).replace(
        superbasin_quotient=True, superbasin_report=True, superbasin_period=20
    )
    result = search(n=6, budget=3000, config=cfg)
    report = result["superbasin"]
    for key in ("nodes", "edges", "distortion", "jumps", "refusals", "archived", "quotient"):
        assert key in report, key
    assert isinstance(result["superbasin_archive"], list)
    assert "observations" in result["superbasin_counts"]


CONSTRAINED = {
    "theta": [0.0, 2.0, -1.0, 3.0],
    "calibrate_quantile": [1.0, -0.5, 2.0],
    "bayes_threshold": [1.0, -0.5],
    "bayes_exploration": [1.5, -0.5],
    "contextual_floor": [1.5, -0.5],
    "diversity_floor": [0.0, 1.5],
    "bias_gamma": [1.0, 0.5],
    "bias_height": [-1.0],
    "height_revisits": [0.0, -1.0],
    "length_scale": [0.0, -1.0],
    "merge_radius": [0.0, -1.0],
    "call_floor": [0.0, -1.0],
}


@pytest.mark.parametrize(
    "name, value",
    [(name, value) for name, values in CONSTRAINED.items() for value in values],
)
def test_a_setting_the_search_asserts_on_is_refused_where_it_is_set(name, value):
    with pytest.raises(anneal.InvalidArgumentError, match=name):
        Config.recommended(4).replace(**{name: value})


def test_shape_keyed_is_refused_cleanly_without_the_ira_feature():
    try:
        cfg = Config.recommended(4).replace(shape_keyed=True)
    except anneal.FeatureUnavailableError as err:
        assert "ira" in str(err)
    else:
        assert cfg.get("shape_keyed") is True


def grid(value):
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [0, 1, 2, 10**6]
    if isinstance(value, float):
        return [0.0, 1e-12, 1.0, 1e6, -1.0]
    if value is None:
        return [0, 1, 0.5, -1.0]
    return []


def test_no_scalar_setting_reaches_a_rust_panic_through_the_python_api():
    base = Config.recommended(4)
    defaults = base.to_dict()
    survived = 0
    for name in Config.field_names():
        if name == "n_points":
            continue
        for value in grid(defaults[name]):
            try:
                cfg = base.replace(**{name: value})
            except (anneal.AnnealError, TypeError, ValueError):
                continue
            try:
                search(n=4, budget=60, seed=1, config=cfg)
            except Exception as err:
                assert isinstance(err, (anneal.AnnealError, ValueError)), (name, value, err)
            survived += 1
    assert survived > 200
