# Copyright (c) anneal developers
"""Parity tests for anneal.Config and cluster_search scaled/custom configs."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import anneal


def test_with_scales_defaults_match_rust_config_exactly():
    cfg = anneal.Config.with_scales(13, 1.0, 1.0)
    assert cfg.n_points == 13
    assert cfg.rigid_body is False
    assert cfg.move_library == "atomic"
    assert cfg.burst_moves is False
    assert cfg.length_scale == 1.0
    assert cfg.energy_scale == 1.0
    assert cfg.temperature == 0.8
    assert cfg.bias_height == 0.25
    assert cfg.bias_gamma == 5.0
    assert cfg.basin_stride == 1
    assert cfg.merge_radius == 0.7
    assert cfg.keying == "distances"
    assert cfg.keying_sigma == 2.5
    assert cfg.morphology_scale == 1.0
    assert cfg.shape_keyed is False
    assert cfg.theta == 0.5
    assert cfg.budget_window is False
    assert cfg.allocate_moves is False
    assert cfg.adaptive_height is False
    assert cfg.max_hops is None
    assert cfg.replicas == 1
    assert cfg.swap_period == 50
    assert cfg.bias_by_rung is False
    assert cfg.contextual_moves is False
    assert cfg.contextual_floor == 0.1
    assert cfg.bayes_screen is False
    assert cfg.bayes_exploration == 0.1
    assert cfg.bayes_threshold == 0.05
    assert cfg.bayes_warmup == 300
    assert cfg.flat_histogram is False
    assert cfg.flat_sweep == 400
    assert cfg.flat_quantile == 0.5
    assert cfg.statistical_temperature is False
    assert cfg.energy_bias is False
    assert cfg.depth_reward is False
    assert cfg.arm_selection == "posterior"
    assert cfg.novel_reward is False
    assert cfg.soft_perturb is False
    assert cfg.soft_modes == 6
    assert cfg.soft_steps == 30
    assert cfg.cov_perturb is False
    assert cfg.staged_quench is False
    assert cfg.settle_iters == 20
    assert cfg.species is None
    assert cfg.species_sigma is None
    assert cfg.bond_tolerance == 1.25
    assert cfg.frozen is None
    assert cfg.active_region is None
    assert cfg.track_funnels is False
    assert cfg.funnel_period == 20_000
    assert cfg.symmetrise_on_stall is False
    assert cfg.point_symmetrise is False
    assert cfg.point_symmetrise_on_new is False
    assert cfg.orbit_complete_on_new is False
    assert cfg.orbit_arm is False
    assert cfg.orbit_core == "fraction"
    assert cfg.surface_hop_on_best is False
    assert cfg.soap_repel is False
    assert cfg.exchange_metropolis is False
    assert cfg.md_escape is False
    assert cfg.md_escape_dt == 0.005
    assert cfg.md_escape_kinetic == 1.0
    assert cfg.md_escape_minima == 2
    assert cfg.md_escape_max_steps == 2_000
    assert cfg.md_escape_soften == 0
    assert cfg.shared_deposits == 8
    assert cfg.shared_visit_policy == "tabu"
    assert cfg.jump_on_stall is False
    assert cfg.jump_patience == 5_000
    assert cfg.jump_steps == 10
    assert cfg.symmetrise_core_fraction == 0.6
    assert cfg.point_symmetrise_every_accept is False
    assert cfg.continuous_symmetry == {"group": "off"}
    assert cfg.symmetrise_patience == 2_000
    assert cfg.tabu_on_stall is False
    assert cfg.tabu_capacity == 8
    assert cfg.angular_moves is False
    assert cfg.angular_target == 0.5
    assert cfg.restart_on_stall is False
    assert cfg.restart_patience == 5_000
    assert cfg.calibrate_radius is False
    assert cfg.calibrate_quantile == 0.9
    assert cfg.calibrate_warmup == 200
    assert cfg.minima_hopping is False
    assert cfg.escape_lanczos_steps == 16
    assert cfg.escape_overshoot == 1.5
    assert cfg.escape_max_climb == 24
    assert cfg.escape_on_stall is False
    assert cfg.retry_noop_stall_leave is False
    assert cfg.align_directed_stall_source is False
    assert cfg.trail_on_stall is False
    assert cfg.escape_stall_patience == 5_000
    assert cfg.escape_stall_factor == 2.0
    assert cfg.ladder_top == 4.0
    assert cfg.ladder_mode == "shipped"
    assert cfg.ladder_window == 10
    assert cfg.ladder_pilot == 200
    assert cfg.return_screen is False
    assert cfg.soap_class_residual is False
    assert cfg.soap_mode == "flexible"
    assert cfg.return_polish == 0
    assert cfg.return_polish_after == 0
    assert cfg.path_on_stall is False
    assert cfg.stall_patience == 60
    assert cfg.path_images == 9
    assert cfg.anneal_diversity is False
    assert cfg.prior_soap is True
    assert cfg.general_collective is False
    assert cfg.homotop_descent is True
    assert cfg.homotop_ranked is True
    assert cfg.homotop_window is True
    assert cfg.homotop_reorder is False
    assert cfg.diversity_floor == 0.75
    assert cfg.height_revisits == 4.0
    assert cfg.calibrated_screen is True
    assert cfg.finish_improving is True
    assert cfg.screen_steps == 25
    assert cfg.adaptive_screen is False
    assert cfg.precondition_bonds is False
    assert cfg.capped_lindh_quench is False
    assert cfg.local_step == "lbfgs"
    assert cfg.answer_gradient == 0.0
    assert cfg.trace_quenched is False
    assert cfg.long_ranged is False
    assert cfg.plateau_heat is False
    assert cfg.polish_records == 0
    assert cfg.surrogate_tolerance == 0.5
    assert cfg.delayed_acceptance is False
    assert cfg.construct_width == 4
    assert cfg.probe_screen is False
    assert cfg.quench_warmup == 4
    assert cfg.quench_confidence == 2.0
    assert cfg.quench_once is False
    assert cfg.target_energy is None
    assert cfg.improvement_rewards is False
    assert cfg.incumbent_gap == 0.0
    assert cfg.relax_steps == 200
    assert cfg.reoccupy_interval == 5_000
    assert cfg.energy_lattice is False
    assert cfg.pair_range is None
    assert cfg.call_floor is None
    assert cfg.call_floor_elements is None
    assert cfg.displacement_only is False
    assert cfg.superbasin_report is False
    assert cfg.superbasin_escape is False
    assert cfg.superbasin_period == 2_000
    assert cfg.energy_trace is False
    assert cfg.superbasin_quotient is False
    assert cfg.superbasin_features is False
    assert cfg.witness_tripleton_ceiling is False
    assert cfg.ring_moves is False
    assert cfg.hold_stereocentres is False
    assert cfg.measured_energy_scale is False
    assert cfg.branch_on_stall is False


def test_config_constructors_and_digests_match_rust_presets():
    rec = anneal.Config.recommended(13)
    rec_scaled = anneal.Config.recommended_with_scales(13, 1.0, 1.0)
    rec_lj = anneal.Config.recommended_for_pair(13, "lj")
    assert rec.resolved_sha256() == rec_scaled.resolved_sha256()
    assert rec.resolved_sha256() == rec_lj.resolved_sha256()
    assert rec.relax_steps == 1000
    assert rec.energy_lattice is True
    assert rec.continuous_symmetry == {"group": "measure", "interval": 10}

    morse_cfg = anneal.Config.recommended_for_pair(13, "morse", rho=6.0)
    morse_pot = anneal.PairPotential.morse(13, 6.0)
    assert math.isclose(morse_cfg.length_scale, morse_pot.length_scale, rel_tol=1e-12)

    fc = anneal.Config.for_cluster(13)
    fc_scaled = anneal.Config.for_cluster_with_scales(13, 1.0, 1.0)
    assert fc.resolved_sha256() == fc_scaled.resolved_sha256()
    assert fc.orbit_core == "both"

    dual = anneal.Config.for_cluster_dual_core(13)
    dual_method = fc.with_dual_orbit_core()
    assert dual.resolved_sha256() == dual_method.resolved_sha256()
    assert dual.orbit_complete_on_new is True
    assert dual.restart_on_stall is True
    assert dual.restart_patience == 4_000

    pack = anneal.Config.packing_superbasin(38)
    assert pack.adaptive_height is True
    assert pack.height_revisits == 20.0

    tip = anneal.Config.for_tip4p(6)
    assert tip.rigid_body is True
    assert tip.move_library == "rigid_body"
    assert tip.temperature == 2.0
    assert tip.relax_steps == 80
    assert tip.screen_steps == 15

    mol = anneal.Config.for_molecular([8, 1, 1], [[0, 1, 2]], 1.0)
    rmol = anneal.Config.recommended_molecular([8, 1, 1], [[0, 1, 2]], 1.0)
    assert mol.move_library == "molecular"
    assert mol.local_step == "lbfgs"
    assert rmol.local_step == "auto"
    assert rmol.ring_moves is True
    assert rmol.hold_stereocentres is True
    assert rmol.measured_energy_scale is True

    parsed = json.loads(rec.resolved_json())
    assert parsed["schema"] == "anneal-cluster-config-v1"
    assert parsed["n_points"] == 13
    assert len(rec.resolved_sha256()) == 64


def test_config_builders_mutations_and_prior_validators():
    cfg = anneal.Config.recommended(6)
    assert cfg.monoatomic() is True
    assert cfg.packing_cna_applies() is True
    assert cfg.start_radius() == cfg.container

    bimet = cfg.with_species([29, 29, 29, 79, 79, 79], [1.0, 1.0, 1.0, 1.2, 1.2, 1.2])
    assert bimet.monoatomic() is False
    assert bimet.packing_cna_applies() is False
    assert bimet.species == [29, 29, 29, 79, 79, 79]

    mol_split = anneal.Config.recommended_molecular([6, 6, 6], [[0], [1, 2]], 1.0)
    coords = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 2.2, 1.2, 0.0], dtype=np.float64)
    merged = mol_split.with_flexible_molecules(coords)
    assert merged is not None
    assert merged.with_flexible_molecules(coords) is None

    cloned = cfg.copy()
    cloned.temperature = 1.25
    cloned.relax_steps = 400
    cloned.keying = "q4q6"
    cloned.ladder_mode = "reversible"
    cloned.soap_mode = "off"
    cloned.arm_selection = "uniform"
    cloned.shared_visit_policy = "recognition"
    cloned.orbit_core = "shells"
    cloned.continuous_symmetry = ("inversion", 5)
    cloned.pair_range = (3.0, 1.0)
    cloned.max_hops = 25
    cloned.target_energy = -12.5
    assert cfg.temperature == 0.8
    assert cloned.temperature == 1.25
    assert cloned.relax_steps == 400
    assert cloned.keying == "q4q6"
    assert cloned.ladder_mode == "reversible"
    assert cloned.soap_mode == "off"
    assert cloned.arm_selection == "uniform"
    assert cloned.shared_visit_policy == "recognition"
    assert cloned.orbit_core == "shells"
    assert cloned.continuous_symmetry == {"group": "inversion", "interval": 5}
    assert cloned.pair_range == (3.0, 1.0)
    assert cloned.max_hops == 25
    assert cloned.target_energy == -12.5

    with pytest.raises(ValueError, match="was removed"):
        cloned.continuous_symmetry = {
            "group": "measure",
            "interval": 10,
            "reseed_hops": 4,
        }

    assert anneal.Config.parse_prior_switches("") == {"soap": True, "all_off": False}
    assert anneal.Config.parse_prior_switches("0") == {"soap": False, "all_off": True}
    assert anneal.Config.parse_prior_switches("soap") == {"soap": False, "all_off": False}
    with pytest.raises(ValueError, match="was removed"):
        anneal.Config.parse_prior_switches("reseed")
    anneal.Config.check_removed_env()


def test_cluster_search_scaled_presets_and_custom_config_match_rust():
    lj4 = anneal.PairPotential.lennard_jones(4)

    rec_plain = anneal.cluster_search(lj4, None, 4, 300, seed=9, recommended=True)
    rec_unit_scale = anneal.cluster_search(
        lj4,
        None,
        4,
        300,
        seed=9,
        recommended=True,
        length_scale=1.0,
        energy_scale=1.0,
    )
    rec_cfg = anneal.cluster_search(
        lj4,
        None,
        4,
        300,
        seed=9,
        config=anneal.Config.recommended(4),
    )
    assert math.isclose(rec_plain["best_energy"], rec_unit_scale["best_energy"], rel_tol=1e-12)
    assert rec_plain["hops"] == rec_unit_scale["hops"]
    assert math.isclose(rec_plain["best_energy"], rec_cfg["best_energy"], rel_tol=1e-12)
    assert rec_plain["hops"] == rec_cfg["hops"]

    fc_plain = anneal.cluster_search(lj4, None, 4, 300, seed=9, recommended=False)
    fc_unit_scale = anneal.cluster_search(
        lj4,
        None,
        4,
        300,
        seed=9,
        recommended=False,
        length_scale=1.0,
        energy_scale=1.0,
    )
    fc_cfg = anneal.cluster_search(
        lj4,
        None,
        4,
        300,
        seed=9,
        config=anneal.Config.for_cluster(4),
    )
    assert math.isclose(fc_plain["best_energy"], fc_unit_scale["best_energy"], rel_tol=1e-12)
    assert fc_plain["hops"] == fc_unit_scale["hops"]
    assert math.isclose(fc_plain["best_energy"], fc_cfg["best_energy"], rel_tol=1e-12)
    assert fc_plain["hops"] == fc_cfg["hops"]

    capped_cfg = anneal.Config.recommended(4)
    capped_cfg.max_hops = 2
    capped_out = anneal.cluster_search(lj4, None, 4, 2000, seed=9, config=capped_cfg)
    assert capped_out["hops"] <= 2

    with pytest.raises(ValueError, match="must match n"):
        anneal.cluster_search(lj4, None, 5, 200, config=capped_cfg)
    with pytest.raises(ValueError, match="excludes"):
        anneal.cluster_search(lj4, None, 4, 200, config=capped_cfg, length_scale=1.2)
