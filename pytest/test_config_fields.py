# Copyright (c) anneal developers
"""``Config`` exposes every field of the Rust struct and every preset."""

import math
import re
from pathlib import Path

import pytest

import anneal
from anneal import Config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_RS = ROOT / "src" / "methods" / "cluster_hopping" / "config.rs"
STRUCT_FIELD = re.compile(r"^    pub (\w+): (.+),$", re.MULTILINE)


def rust_fields():
    text = CONFIG_RS.read_text()
    start = text.index("pub struct Config {")
    body = text[start : text.index("\n}\n", start)]
    return STRUCT_FIELD.findall(body)


FIELDS = rust_fields()
KIND = {
    "bool": bool,
    "f64": float,
    "usize": int,
    "u64": int,
}
OPAQUE = {"reoccupy", "hmc"}


def scalar(rust_type):
    return rust_type in KIND or rust_type in ("Option<f64>", "Option<usize>")


def test_the_struct_was_read():
    assert len(FIELDS) > 150
    assert ("n_points", "usize") in FIELDS


def test_field_names_are_the_struct_fields_sorted():
    assert Config.field_names() == sorted(name for name, _ in FIELDS)
    assert len(set(Config.field_names())) == len(FIELDS)


def test_to_dict_has_every_field_with_the_type_the_struct_declares():
    values = Config.recommended(13).to_dict()
    assert list(values) == sorted(name for name, _ in FIELDS)
    for name, rust_type in FIELDS:
        value = values[name]
        if name in OPAQUE:
            assert isinstance(value, bool)
        elif rust_type in KIND:
            assert type(value) is KIND[rust_type], name
        elif rust_type == "Option<f64>":
            assert value is None or type(value) is float, name
        elif rust_type == "Option<usize>":
            assert value is None or type(value) is int, name
        else:
            assert value is None or isinstance(value, (str, int, float, bool, list, dict)), name


@pytest.mark.parametrize(
    "preset",
    [
        Config.recommended(13),
        Config.derived(13),
        Config.for_cluster(13),
        Config.communicating(13),
        Config.recommended_tip4p(4),
        Config.packing_superbasin(13),
    ],
    ids=repr,
)
def test_get_agrees_with_the_named_properties(preset):
    for name in (
        "n_points",
        "screen_margin",
        "calibrated_screen",
        "finish_improving",
        "allocate_moves",
        "depth_reward",
        "orbit_complete_on_new",
        "tabu_on_stall",
        "bayes_threshold",
        "budget_window",
    ):
        assert preset.get(name) == getattr(preset, name), name
    assert preset.get("shared_visit_policy") == preset.shared_visit_policy
    assert (preset.get("move_library") == "lean_burst") == preset.burst_moves
    assert (preset.get("rigid_body_geometry") is not None) == preset.rigid_body


def test_every_default_scalar_is_finite_so_replace_can_demand_it():
    for name, rust_type in FIELDS:
        if rust_type in ("f64", "Option<f64>"):
            for preset in (Config.recommended(13), Config.for_cluster(13), Config.derived(13)):
                value = preset.get(name)
                assert value is None or math.isfinite(value), name


CONSTRAINED = {
    "theta": 0.75,
    "calibrate_quantile": 0.25,
    "bayes_threshold": 0.25,
    "bayes_exploration": 0.25,
    "contextual_floor": 0.25,
    "diversity_floor": 0.5,
    "bias_gamma": 3.0,
}


def changed(name, value):
    if name in CONSTRAINED:
        return CONSTRAINED[name]
    if isinstance(value, bool):
        return not value
    if isinstance(value, float):
        return value + 1.0
    return value + 1


@pytest.mark.parametrize("name, rust_type", [field for field in FIELDS if scalar(field[1]) and field[0] != "n_points"])
def test_replace_sets_one_scalar_and_leaves_the_original(name, rust_type):
    base = Config.recommended(13)
    before = base.resolved_sha256()
    old = base.get(name)
    new = changed(name, 0 if old is None else old)
    try:
        edited = base.replace(**{name: new})
    except anneal.FeatureUnavailableError:
        assert name == "shape_keyed" and new is True
        return
    assert edited.get(name) == new
    assert base.get(name) == old
    assert base.resolved_sha256() == before
    assert edited.resolved_sha256() != before
    others = {key: value for key, value in edited.to_dict().items() if key != name}
    assert others == {key: value for key, value in base.to_dict().items() if key != name}


@pytest.mark.parametrize("name, value", [("target_energy", -3.5), ("call_floor", 3.5)])
def test_an_optional_float_takes_none_and_a_number(name, value):
    cfg = Config.recommended(13).replace(**{name: value})
    assert cfg.get(name) == value
    assert cfg.replace(**{name: None}).get(name) is None


def test_an_optional_count_takes_none_and_a_number():
    cfg = Config.recommended(13).replace(max_hops=7)
    assert cfg.get("max_hops") == 7
    assert cfg.replace(max_hops=None).get("max_hops") is None


def test_replace_with_nothing_is_a_copy():
    cfg = Config.recommended(13)
    assert cfg.replace().resolved_sha256() == cfg.resolved_sha256()
    assert repr(cfg.replace()) == repr(cfg)


def test_replace_refuses_what_it_cannot_set_with_a_typed_error():
    cfg = Config.recommended(13)
    with pytest.raises(anneal.InvalidArgumentError, match="n_points"):
        cfg.replace(n_points=5)
    with pytest.raises(anneal.InvalidArgumentError, match="structure"):
        cfg.replace(move_library="atomic")
    with pytest.raises(anneal.InvalidArgumentError, match="structure"):
        cfg.replace(hmc=True)
    with pytest.raises(anneal.InvalidArgumentError, match="no field"):
        cfg.replace(not_a_field=1)
    with pytest.raises(anneal.InvalidArgumentError, match="finite"):
        cfg.replace(temperature=float("nan"))
    with pytest.raises(anneal.InvalidArgumentError, match="finite"):
        cfg.replace(temperature=math.inf)
    with pytest.raises(anneal.InvalidArgumentError, match="whole"):
        cfg.replace(relax_steps=-1)
    with pytest.raises(TypeError):
        cfg.replace(relax_steps="many")
    with pytest.raises(TypeError):
        cfg.replace(budget_window=1)


def test_get_refuses_an_unknown_name():
    with pytest.raises(anneal.InvalidArgumentError, match="no field"):
        Config.recommended(13).get("not_a_field")


def test_resolved_json_names_its_schema_and_hashes_stably():
    cfg = Config.recommended(13)
    assert cfg.resolved_json().startswith('{"schema":"anneal-cluster-config-v1"')
    assert re.fullmatch(r"[0-9a-f]{64}", cfg.resolved_sha256())
    assert cfg.resolved_sha256() == Config.recommended(13).resolved_sha256()
    assert cfg.resolved_sha256() != Config.recommended(14).resolved_sha256()


@pytest.mark.parametrize(
    "build, text",
    [
        (lambda: Config.recommended(5), "Config.recommended(5)"),
        (lambda: Config.derived(5), "Config.derived(5)"),
        (lambda: Config.for_cluster(5), "Config.for_cluster(5)"),
        (lambda: Config.communicating(5), "Config.communicating(5)"),
        (lambda: Config.communicating_tip4p(5), "Config.communicating_tip4p(5)"),
        (lambda: Config.recommended_tip4p(5), "Config.recommended_tip4p(5)"),
        (lambda: Config.for_tip4p(5), "Config.for_tip4p(5)"),
        (lambda: Config.packing_superbasin(5), "Config.packing_superbasin(5)"),
        (lambda: Config.for_cluster_dual_core(5), "Config.for_cluster_dual_core(5)"),
        (lambda: Config.with_scales(5, 1.5, 2.0), "Config.with_scales(5, 1.5, 2.0)"),
        (
            lambda: Config.recommended(5).replace(relax_steps=50),
            "Config.recommended(5).replace(relax_steps=50)",
        ),
        (
            lambda: Config.for_cluster(5).with_dual_orbit_core(),
            "Config.for_cluster(5).with_dual_orbit_core()",
        ),
    ],
)
def test_repr_names_how_the_config_was_built(build, text):
    assert repr(build()) == text


def test_the_class_docstring_describes_the_current_surface():
    doc = Config.__doc__
    assert "field_names" in doc
    assert "replace" in doc
    assert "cluster_search(config=" in doc


def test_scaled_presets_take_declared_units():
    cfg = Config.recommended_with_scales(13, 1.0, 2.0)
    assert (cfg.get("length_scale"), cfg.get("energy_scale")) == (1.0, 2.0)
    assert cfg.get("relax_steps") == 1000
    plain = Config.for_cluster_with_scales(13, 1.5, 3.0)
    assert (plain.get("length_scale"), plain.get("energy_scale")) == (1.5, 3.0)
    assert plain.get("move_library") == "atomic"
    bare = Config.with_scales(13, 1.5, 3.0)
    assert (bare.get("length_scale"), bare.get("energy_scale")) == (1.5, 3.0)


def test_the_pair_preset_matches_the_rust_rule():
    lj = Config.recommended_for_pair(13, "lennard_jones")
    assert lj.resolved_sha256() == Config.recommended(13).resolved_sha256()
    morse = Config.recommended_for_pair(13, "morse", rho=6.0)
    assert morse.get("length_scale") == pytest.approx(2.0 ** (-1.0 / 6.0))
    assert morse.get("length_scale") < 1.0


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: Config.recommended_for_pair(13, "morse"), "rho"),
        (lambda: Config.recommended_for_pair(13, "morse", rho=-1.0), "rho"),
        (lambda: Config.recommended_for_pair(13, "lennard_jones", rho=6.0), "no rho"),
        (lambda: Config.recommended_for_pair(13, "coulomb"), "pair must be"),
        (lambda: Config.with_scales(13, 0.0, 1.0), "length_scale"),
        (lambda: Config.with_scales(13, 1.0, math.nan), "energy_scale"),
        (lambda: Config.recommended_with_scales(13, -1.0, 1.0), "length_scale"),
        (lambda: Config.for_cluster_with_scales(13, 1.0, 0.0), "energy_scale"),
        (lambda: Config.for_tip4p(1), "at least 2"),
        (lambda: Config.packing_superbasin(1), "at least 2"),
        (lambda: Config.for_cluster_dual_core(0), "at least 2"),
    ],
)
def test_presets_reject_what_would_panic_in_rust(call, message):
    with pytest.raises(anneal.InvalidArgumentError, match=message):
        call()


def test_tip4p_and_packing_presets_match_their_rust_doc_examples():
    assert Config.for_tip4p(6).get("temperature") == 2.0
    assert Config.for_tip4p(6).get("n_points") == 6
    packing = Config.packing_superbasin(38)
    assert packing.get("adaptive_height") is True
    assert packing.get("height_revisits") == 20.0


def test_the_dual_core_is_one_step_from_the_plain_protocol():
    direct = Config.for_cluster_dual_core(13)
    built = Config.for_cluster(13).with_dual_orbit_core()
    assert direct.get("restart_patience") == 4000
    assert direct.get("orbit_core") == built.get("orbit_core")
    assert direct.to_dict() == built.to_dict()
    assert Config.for_cluster(13).get("restart_patience") != 4000


WATER = ([8, 1, 1, 8, 1, 1], [[0, 1, 2], [3, 4, 5]])


def test_the_molecular_presets_differ_by_the_measured_controls():
    plain = Config.for_molecular(*WATER)
    measured = Config.recommended_molecular(*WATER, energy_scale=1.0)
    assert plain.get("n_points") == measured.get("n_points") == 6
    assert plain.get("ring_moves") is False
    assert measured.get("ring_moves") is True
    assert measured.get("local_step") == "auto"
    assert plain.get("species") == [8, 1, 1, 8, 1, 1]
    assert plain.get("move_library")["molecular"]["groups"] == [[0, 1, 2], [3, 4, 5]]


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: Config.for_molecular([], []), "empty"),
        (lambda: Config.for_molecular([8, 1, 1], [[0, 1, 2]], energy_scale=0.0), "energy_scale"),
        (lambda: Config.for_molecular([0, 0], [[0, 1]]), "covalent"),
        (lambda: Config.for_molecular([8, 1, 1], [[0, 1, 3]]), "past"),
        (lambda: Config.for_molecular([8, 1, 1], [[]]), "at least one atom"),
        (lambda: Config.recommended_molecular([], []), "empty"),
    ],
)
def test_molecular_presets_reject_what_would_panic_in_rust(call, message):
    with pytest.raises(anneal.InvalidArgumentError, match=message):
        call()


def test_with_species_labels_a_preset_and_checks_the_lengths():
    cfg = Config.recommended(4).with_species([1, 2, 1, 2], sigma=[1.0, 1.2, 1.0, 1.2])
    assert cfg.get("species") == [1, 2, 1, 2]
    assert cfg.get("species_sigma") == [1.0, 1.2, 1.0, 1.2]
    assert Config.recommended(4).get("species") is None
    with pytest.raises(anneal.InvalidArgumentError, match="labels"):
        Config.recommended(4).with_species([1, 2])
    with pytest.raises(anneal.InvalidArgumentError, match="sigma"):
        Config.recommended(4).with_species([1, 2, 1, 2], sigma=[1.0])
    with pytest.raises(anneal.InvalidArgumentError, match="positive"):
        Config.recommended(4).with_species([1, 2, 1, 2], sigma=[1.0, -1.0, 1.0, 1.0])


def test_with_flexible_molecules_keeps_a_config_it_does_not_change():
    import numpy as np

    cfg = Config.for_molecular(*WATER)
    start = np.array(
        [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 5.0, 0.0, 0.0, 5.96, 0.0, 0.0, 4.76, 0.93, 0.0]
    )
    assert cfg.with_flexible_molecules(start) is None
    assert Config.recommended(4).with_flexible_molecules(np.zeros(12)) is None
