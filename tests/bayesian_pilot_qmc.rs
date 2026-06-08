use anneal_core::{PilotPrior, Q_V_MAX, Q_V_MIN, pilot_draws_qmc};

#[test]
fn qmc_pilot_draws_are_seeded_deterministic_and_bounded() {
    let prior = PilotPrior::default();

    let first = pilot_draws_qmc(&prior, 16, 7);
    let second = pilot_draws_qmc(&prior, 16, 7);
    let third = pilot_draws_qmc(&prior, 16, 8);

    assert_eq!(first, second);
    assert_ne!(first, third);
    assert_eq!(first.len(), 16);

    for (t_init, sigma, q_v) in first {
        assert!(t_init.is_finite() && t_init > 0.0);
        assert!(sigma.is_finite() && sigma > 0.0);
        assert!(
            q_v > Q_V_MIN && q_v < Q_V_MAX,
            "q_v {q_v} outside the pilot support"
        );
    }
}
