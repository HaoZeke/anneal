# Copyright (c) anneal developers
"""Every public method of the Rust ``Config`` is reached from Python."""

import re
from pathlib import Path

import numpy as np
import pytest

import anneal
from anneal import Config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_RS = ROOT / "src" / "methods" / "cluster_hopping" / "config.rs"

# Each public method of the Rust ``Config`` and the Python name that reaches it.
REACHED = {
    "resolved_json": "resolved_json",
    "resolved_sha256": "resolved_sha256",
    "validate_json_keys": "validate_json",
    "validate_toml_keys": "validate_toml",
    "recommended": "recommended",
    "recommended_for_pair": "recommended_for_pair",
    "recommended_with_scales": "recommended_with_scales",
    "communicating": "communicating",
    "packing_surface": "packing_surface",
    "packing_superbasin": "packing_superbasin",
    "monoatomic": "monoatomic",
    "packing_cna_applies": "packing_cna_applies",
    "derived": "derived",
    "for_cluster": "for_cluster",
    "for_cluster_with_scales": "for_cluster_with_scales",
    "for_cluster_dual_core": "for_cluster_dual_core",
    "with_dual_orbit_core": "with_dual_orbit_core",
    "with_scales": "with_scales",
    "for_tip4p": "for_tip4p",
    "with_species": "with_species",
    "with_rigid_bodies": "with_rigid_bodies",
    "communicating_tip4p": "communicating_tip4p",
    "recommended_tip4p": "recommended_tip4p",
    "for_molecular": "for_molecular",
    "recommended_molecular": "recommended_molecular",
    "with_flexible_molecules": "with_flexible_molecules",
    "start_radius": "start_radius",
    "proposal_kernel": "proposal_arms",
}
PLUMBING = {
    "parse_prior_switches": "reads the H2H_PRIORS text of the Rust binaries",
    "check_removed_env_vars": "screens the environment variables of the Rust binaries",
    "check_removed_env": "screens the environment variables of the Rust binaries",
}

SITES = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
LABELS = [8, 1, 1]


def rust_methods():
    text = CONFIG_RS.read_text()
    start = text.index("\nimpl Config {")
    body = text[start : text.index("\n}\n", start)]
    return re.findall(r"^    pub fn (\w+)", body, re.MULTILINE)


def test_the_impl_was_read():
    methods = rust_methods()
    assert "recommended" in methods
    assert len(methods) > 25


def test_every_public_method_is_reached_or_named_as_plumbing():
    missing = [
        name
        for name in rust_methods()
        if name not in REACHED and name not in PLUMBING
    ]
    assert not missing, f"public Config methods with no Python name: {missing}"


def test_the_lists_name_only_methods_that_exist():
    methods = set(rust_methods())
    stale = [name for name in (*REACHED, *PLUMBING) if name not in methods]
    assert not stale, f"named here but not in impl Config: {stale}"
    assert not set(REACHED) & set(PLUMBING)


@pytest.mark.parametrize("rust,python", sorted(REACHED.items()))
def test_each_reached_method_exists_on_config(rust, python):
    assert hasattr(Config, python), f"{rust} has no Config.{python}"


@pytest.mark.parametrize("name", sorted(PLUMBING))
def test_plumbing_is_not_pretended_to_be_exposed(name):
    assert not hasattr(Config, name)


def test_monoatomic_follows_the_species_labels():
    cfg = Config.recommended(4)
    assert cfg.monoatomic
    assert cfg.with_species([1, 1, 1, 1]).monoatomic
    assert not cfg.with_species([1, 2, 1, 2]).monoatomic


def test_the_packing_diagnostics_apply_to_a_monoatomic_cluster_only():
    assert Config.recommended(13).packing_cna_applies
    assert not Config.recommended(13).with_species([1, 2] * 6 + [1]).packing_cna_applies
    assert not Config.for_tip4p(4).packing_cna_applies


def test_start_radius_is_the_container_the_preset_chose():
    small = Config.recommended(13)
    assert small.start_radius == small.get("container")
    assert Config.recommended(38).start_radius > small.start_radius


def test_the_proposal_has_arms_and_they_follow_the_move_library():
    assert Config.recommended(13).proposal_arms >= 1
    assert Config.for_tip4p(4).proposal_arms != Config.recommended(13).proposal_arms


def test_packing_surface_reports_the_four_flags():
    surface = Config.recommended(13).packing_surface()
    assert set(surface) == {"twin_as_move", "orbit_on_new", "angular", "psym_on_new"}
    assert all(type(v) is bool for v in surface.values())
    assert surface["orbit_on_new"] == Config.recommended(13).get("orbit_complete_on_new")
    assert Config.communicating(13).packing_surface()["orbit_on_new"]


def test_with_rigid_bodies_installs_the_rigid_chart():
    plain = Config.recommended(4)
    rigid = plain.with_rigid_bodies(SITES, LABELS)
    assert not plain.rigid_body
    assert rigid.rigid_body
    assert rigid.n_points == 4
    assert rigid.start_radius >= plain.start_radius
    assert rigid.proposal_arms != plain.proposal_arms
    assert repr(rigid).endswith(".with_rigid_bodies(sites, species)")


def test_a_rigid_config_runs_a_search_on_six_numbers_per_body():
    rigid = Config.recommended(4).with_rigid_bodies(SITES, LABELS)
    start = np.random.default_rng(0).normal(size=24)
    out = anneal.cluster_search(
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        lambda x: 2.0 * np.asarray(x),
        4,
        1500,
        seed=1,
        config=rigid,
        start=start,
    )
    assert np.asarray(out["best"]).shape == (24,)


@pytest.mark.parametrize(
    "sites,species,message",
    [
        (SITES, [8, 1], "species"),
        (SITES, [0, 1, 1], "species"),
        (np.zeros((3, 2)), [1, 1, 1], "shape"),
        (np.array([[np.nan, 0.0, 0.0]]), [1], "finite"),
        (np.zeros((0, 3)), [], "geometry"),
    ],
)
def test_with_rigid_bodies_refuses_what_rust_refuses(sites, species, message):
    with pytest.raises(anneal.InvalidArgumentError, match=message):
        Config.recommended(4).with_rigid_bodies(sites, species)
