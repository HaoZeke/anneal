"""TDD assertions for occupancy cooperative-search identities.

Each test names one algebraic identity or inequality. The SymPy module
must return a vanishing residual, a stated limit, or a tautology.
No numeric sampling of the identities.
"""

import sympy as sp

from proofs.occupancy_cooperative_search import (
    MIXED_RHAT,
    REDUCTION_FACTOR,
    ASKMC_ALPHA,
    CERTIFY_MIN_SAMPLES,
    PRODUCTION_MAX_UNSEEN_MASS,
    PRODUCTION_MINIMUM_VISITS,
    all_identities,
    identity_rhat_squared,
    identity_w_zero_b_positive_limit,
    identity_two_chain_between,
    identity_constant_traces_b_vanishes_iff_equal,
    identity_mcmc_skip_misclassifies_unmixed,
    identity_mixed_threshold_bw,
    identity_lone_floor_is_sampled_mode_certificate,
    identity_certificate_four_conjuncts,
    identity_equal_occupancy_not_stronger,
    identity_strict_occupancy_is_stronger,
    identity_good_turing_estimator,
    identity_revisit_singleton_drops_mass,
    identity_revisit_nonsingleton_drops_mass,
    identity_fixed_family_mass_vanishes,
    identity_one_family_saturation_is_not_stop,
    identity_gt_stop_needs_two_families,
    identity_default_floor_is_packing_gt_alone,
    identity_new_type_raises_p0,
    identity_hatch_stable_is_next_below_ceiling,
    identity_visits_floor_three_fifths,
    identity_certify_min_is_split_rhat_length,
    identity_fes_factor_scale_and_tie,
    identity_cheeger_code_cut,
    identity_path_rg2,
    identity_forest_edge_bound,
    identity_mixing_does_not_retire_without_packing,
    identity_keep_count_partitions,
    identity_keep_independent_of_resource,
    identity_champion_survives_where_halving_zeros,
    identity_occ_minus_sh_residue,
    identity_keep_extras_floor,
    identity_askmc_refuse_probability,
    identity_askmc_rate_scale,
    identity_untempered_height_is_visit_count,
    identity_tempered_height_implies_visits,
    identity_exploit_is_same_packing_lower,
    identity_different_packing_never_exploit,
    identity_copy_collapse_is_mixed,
    identity_copy_collapse_is_not_certificate,
)


def test_source_constants_are_exact_rationals():
    assert MIXED_RHAT == sp.Rational(6, 5)
    assert REDUCTION_FACTOR == 3
    assert ASKMC_ALPHA == 2
    assert CERTIFY_MIN_SAMPLES == 16
    assert PRODUCTION_MAX_UNSEEN_MASS == sp.Rational(1, 5)
    assert PRODUCTION_MINIMUM_VISITS == 20


def test_rhat_squared_splits_into_within_and_between():
    ok, residual = identity_rhat_squared()
    assert ok
    assert residual == 0


def test_w_zero_b_positive_rhat_is_infinity():
    ok, limit = identity_w_zero_b_positive_limit()
    assert ok
    assert limit is sp.oo


def test_two_chain_between_is_half_n_gap_squared():
    ok, residual = identity_two_chain_between()
    assert ok
    assert residual == 0


def test_constant_traces_b_vanishes_iff_means_equal():
    ok, residual = identity_constant_traces_b_vanishes_iff_equal()
    assert ok
    assert residual == 0


def test_mcmc_w_zero_skip_would_pass_mixed_threshold():
    ok, skip_mixed, inverted_mixed = identity_mcmc_skip_misclassifies_unmixed()
    assert ok
    assert skip_mixed is True
    assert inverted_mixed is False


def test_mixed_rhat_is_bw_inequality():
    ok, residual = identity_mixed_threshold_bw()
    assert ok
    assert residual == 0


def test_lone_mixed_floor_is_the_sampled_mode_certificate():
    ok, tautology = identity_lone_floor_is_sampled_mode_certificate()
    assert ok
    assert tautology is True


def test_certificate_is_the_four_conjuncts():
    ok, tautology = identity_certificate_four_conjuncts()
    assert ok
    assert tautology is True


def test_equal_occupancy_is_not_strictly_stronger():
    ok, tautology = identity_equal_occupancy_not_stronger()
    assert ok
    assert tautology is True


def test_strict_occupancy_is_stronger():
    ok, tautology = identity_strict_occupancy_is_stronger()
    assert ok
    assert tautology is True


def test_good_turing_unseen_mass_is_n1_over_n():
    ok, residual = identity_good_turing_estimator()
    assert ok
    assert residual == 0


def test_revisit_of_a_singleton_strictly_drops_mass():
    ok, residual, sign = identity_revisit_singleton_drops_mass()
    assert ok
    assert residual == 0
    assert sign is True


def test_revisit_of_a_nonsingleton_does_not_raise_mass():
    ok, residual, sign = identity_revisit_nonsingleton_drops_mass()
    assert ok
    assert residual == 0
    assert sign is True


def test_fixed_family_count_drives_unseen_mass_to_zero():
    ok, limit = identity_fixed_family_mass_vanishes()
    assert ok
    assert limit == 0


def test_one_family_saturation_is_not_a_gt_stop():
    ok, tautology = identity_one_family_saturation_is_not_stop()
    assert ok
    assert tautology is True


def test_gt_stop_requires_two_occupied_families():
    ok, tautology = identity_gt_stop_needs_two_families()
    assert ok
    assert tautology is True


def test_default_family_floor_is_packing_gt_alone():
    ok, tautology = identity_default_floor_is_packing_gt_alone()
    assert ok
    assert tautology is True


def test_new_type_raises_unseen_mass():
    ok, residual = identity_new_type_raises_p0()
    assert ok
    assert residual == 0


def test_hatch_stable_is_the_next_estimator():
    ok, residual = identity_hatch_stable_is_next_below_ceiling()
    assert ok
    assert residual == 0


def test_visit_floor_is_twenty_for_three_singletons():
    ok, _payload = identity_visits_floor_three_fifths()
    assert ok


def test_certify_length_is_split_rhat_product():
    ok, residual = identity_certify_min_is_split_rhat_length()
    assert ok
    assert residual == 0


def test_fes_boltzmann_factor_identities():
    ok, residual = identity_fes_factor_scale_and_tie()
    assert ok
    assert residual == 0


def test_fiedler_cut_implies_cheeger_upper():
    ok, residual = identity_cheeger_code_cut()
    assert ok
    assert residual == 0


def test_path_rg2_is_the_uniform_second_moment():
    ok, residual = identity_path_rg2()
    assert ok
    assert residual == 0


def test_forest_has_at_most_n_minus_c_edges():
    ok, tautology = identity_forest_edge_bound()
    assert ok
    assert tautology is True


def test_mixing_does_not_retire_without_packing_saturation():
    ok, tautology = identity_mixing_does_not_retire_without_packing()
    assert ok
    assert tautology is True


def test_keep_plus_leave_is_the_cohort():
    ok, residual = identity_keep_count_partitions()
    assert ok
    assert residual == 0


def test_keep_count_does_not_depend_on_resource():
    ok, derivative = identity_keep_independent_of_resource()
    assert ok
    assert derivative == 0


def test_champion_survives_the_halving_zero_set():
    ok, tautology = identity_champion_survives_where_halving_zeros()
    assert ok
    assert tautology is True


def test_occupancy_and_halving_differ_off_eta_multiples():
    ok, residual = identity_occ_minus_sh_residue()
    assert ok
    assert residual == 0


def test_kept_extras_are_the_floor_fraction():
    ok, residual = identity_keep_extras_floor()
    assert ok
    assert residual == 0


def test_askmc_refuse_probability_is_one_half():
    ok, residual = identity_askmc_refuse_probability()
    assert ok
    assert residual == 0


def test_askmc_rate_is_inverse_alpha():
    ok, residual = identity_askmc_rate_scale()
    assert ok
    assert residual == 0


def test_untempered_bias_height_tracks_visit_count():
    ok, residual = identity_untempered_height_is_visit_count()
    assert ok
    assert residual == 0


def test_tempered_frequent_height_implies_nf_visits():
    ok, tautology = identity_tempered_height_implies_visits()
    assert ok
    assert tautology is True


def test_exploit_is_a_deeper_isomer_of_the_same_packing():
    ok, tautology = identity_exploit_is_same_packing_lower()
    assert ok
    assert tautology is True


def test_a_different_packing_is_explore_never_exploit():
    ok, tautology = identity_different_packing_never_exploit()
    assert ok
    assert tautology is True


def test_copying_one_floor_mixes_explore_chains():
    ok, residual, mixed = identity_copy_collapse_is_mixed()
    assert ok
    assert residual == 0
    assert mixed is True


def test_explore_collapse_is_leave_not_a_retire_conjunct():
    ok, tautology = identity_copy_collapse_is_not_certificate()
    assert ok
    assert tautology is True


def test_every_identity_holds():
    assert all_identities() is True
