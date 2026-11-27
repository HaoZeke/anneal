// Copyright (c) anneal developers
//! The crate-root sample in `docs/orgmode/reference/rust-api.org`.

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
