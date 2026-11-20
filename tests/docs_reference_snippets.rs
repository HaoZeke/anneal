// Copyright (c) anneal developers
//! The Rust blocks of `docs/orgmode/reference/presets-and-advanced.org`
//! that declare search options, run as written.

#[test]
fn for_cluster_with_scales_matches_for_cluster_at_unit_scales() {
    use anneal_core::methods::cluster_hopping::Config;
    let plain = Config::for_cluster(13);
    let same = Config::for_cluster_with_scales(13, 1.0, 1.0);
    assert_eq!(
        plain.resolved_sha256().unwrap(),
        same.resolved_sha256().unwrap()
    );
    let morse = Config::for_cluster_with_scales(43, 0.891, 1.0);
    assert_eq!((morse.length_scale, morse.energy_scale), (0.891, 1.0));
}

#[test]
fn call_floor_options_are_unset_and_per_element() {
    use anneal_core::methods::cluster_hopping::{Config, element_call_floor};
    let mut cfg = Config::recommended(13);
    assert!(cfg.call_floor.is_none() && cfg.call_floor_elements.is_none());
    cfg.call_floor = Some(0.90);
    cfg.call_floor_elements = Some(vec![14; 13]);
    assert!((element_call_floor(14, 14, 0.90) - 1.55).abs() < 0.01);
    assert!((element_call_floor(6, 6, 0.90) - 1.06).abs() < 0.01);
    assert_eq!(element_call_floor(1, 1, 0.90), 0.90);
    let json = cfg.resolved_json().unwrap();
    assert!(json.contains("call_floor"));
}

#[test]
fn stall_options_are_off_and_leave_the_digest_alone() {
    use anneal_core::methods::cluster_hopping::Config;
    let stock = Config::recommended(38);
    assert!(!stock.branch_on_stall && !stock.retry_noop_stall_leave);
    assert!(!stock.align_directed_stall_source && !stock.capped_lindh_quench);
    let mut cfg = stock.clone();
    cfg.branch_on_stall = true;
    cfg.retry_noop_stall_leave = true;
    cfg.align_directed_stall_source = true;
    assert_ne!(
        cfg.resolved_sha256().unwrap(),
        stock.resolved_sha256().unwrap()
    );
    assert!(!stock.resolved_json().unwrap().contains("branch_on_stall"));
    assert!(cfg.resolved_json().unwrap().contains("branch_on_stall"));
}

#[test]
fn energy_hits_is_one_sided_and_inclusive() {
    use anneal_core::{DEFAULT_HIT_TOL, SEARCH_HIT_TOL, energy_hits};
    let target = -173.928427;
    assert!(energy_hits(
        target + DEFAULT_HIT_TOL,
        target,
        DEFAULT_HIT_TOL
    ));
    assert!(energy_hits(target - 5.0, target, SEARCH_HIT_TOL));
    assert!(!energy_hits(
        target + 2.0 * DEFAULT_HIT_TOL,
        target,
        DEFAULT_HIT_TOL
    ));
    assert!(!energy_hits(f64::NAN, target, DEFAULT_HIT_TOL));
    assert!(!energy_hits(target, target, -1.0));
}

#[test]
fn removed_prior_keys_are_refused() {
    use anneal_core::methods::cluster_hopping::Config;
    let err = Config::validate_json_keys(r#"{"prior_reseed": true}"#).unwrap_err();
    assert_eq!(err.key, "prior_reseed");
    let err = Config::validate_toml_keys("[search]\nprior_core = 1\n").unwrap_err();
    assert_eq!(err.key, "search.prior_core");
    assert!(Config::validate_json_keys(r#"{"prior_soap": true}"#).is_ok());
    let switches = Config::parse_prior_switches("soap").unwrap();
    let _ = switches;
    assert!(Config::parse_prior_switches("reseed").is_err());
}

#[test]
fn binary_lennard_jones_evaluates() {
    use anneal_core::potentials::BinaryLennardJones;
    use eindir_core::gradient::DifferentiableObjective;
    use ndarray::Array1;
    let pot = BinaryLennardJones::new(3, 2, 1.0, 1.0, 1.0, 1.0, 1.15, 1.3);
    assert_eq!(pot.n_points(), 5);
    let x = Array1::from_iter((0..15).map(|i| 0.9 * (i as f64).sin() * 1.5));
    let (e, g) = pot.value_and_gradient(x.view());
    assert!(e.is_finite() && g.len() == 15);
    let _ = <BinaryLennardJones as DifferentiableObjective<f64>>::value_and_gradient;
}

#[test]
fn recommended_for_pair_scales_morse_and_keeps_lennard_jones() {
    use anneal_core::methods::cluster_hopping::Config;
    use anneal_core::potentials::PairKind;
    let lj = Config::recommended_for_pair(13, PairKind::LennardJones);
    assert_eq!(
        lj.resolved_json().unwrap(),
        Config::recommended(13).resolved_json().unwrap()
    );
    let morse = Config::recommended_for_pair(13, PairKind::Morse { rho: 6.0 });
    assert_eq!(morse.energy_scale, 1.0);
    assert!((morse.length_scale - 2.0_f64.powf(-1.0 / 6.0)).abs() < 1e-15);
}

#[test]
fn plateau_heat_is_off_and_leaves_the_digest_alone() {
    use anneal_core::methods::cluster_hopping::Config;
    let stock = Config::recommended(38);
    assert!(!stock.plateau_heat);
    assert!(!stock.resolved_json().unwrap().contains("plateau_heat"));
    let mut cfg = stock.clone();
    cfg.plateau_heat = true;
    assert!(cfg.resolved_json().unwrap().contains("plateau_heat"));
}

#[test]
fn rust_api_crate_root_helpers_match_docs() {
    use anneal_core::{
        ANNEAL_VERSION, DEFAULT_HIT_TOL, Ladder, SwapScheme, TARGET_SWAP_ACCEPT,
        beta_step_for_acceptance, energy_hits, qmc_skip_from_seed,
    };
    assert!(!ANNEAL_VERSION.is_empty());
    let ladder = Ladder::from_temperatures(&[1.0, 2.0, 4.0], SwapScheme::DeterministicEvenOdd);
    assert_eq!(ladder.len(), 3);
    assert!(energy_hits(-10.0 + DEFAULT_HIT_TOL, -10.0, DEFAULT_HIT_TOL));
    assert!(qmc_skip_from_seed(7) > 0);
    assert!(beta_step_for_acceptance(2.0, TARGET_SWAP_ACCEPT) > 0.0);
}
