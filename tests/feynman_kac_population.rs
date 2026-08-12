use anneal_core::methods::feynman_kac::{
    BasinEvidence, SelectionCoefficients, ascending_fractional_ranks, reconfiguration_plan,
};

fn coefficients() -> SelectionCoefficients {
    SelectionCoefficients {
        energy: 1.0,
        novelty: 0.8,
        scarcity: 0.6,
        log_weight_clip: 4.0,
    }
}

#[test]
fn energy_ranks_are_invariant_to_positive_affine_units() {
    let energies = [-543.6, -541.2, -539.0, -530.5];
    let converted = energies.map(|energy| 7.5 * energy + 19.0);

    assert_eq!(
        ascending_fractional_ranks(&energies).unwrap(),
        ascending_fractional_ranks(&converted).unwrap()
    );
}

#[test]
fn novelty_and_scarcity_protect_a_productive_second_funnel() {
    let evidence = [
        BasinEvidence::new(0.0, 0.0, 0.0).unwrap(),
        BasinEvidence::new(0.25, 1.0, 1.0).unwrap(),
        BasinEvidence::new(0.75, 0.4, 0.5).unwrap(),
        BasinEvidence::new(1.0, 0.2, 0.1).unwrap(),
    ];
    let plan = reconfiguration_plan(&evidence, coefficients(), 0.125, 2).unwrap();

    assert!(plan.weights()[1] > plan.weights()[0]);
    assert_eq!(plan.parents().len(), evidence.len());
    assert!(plan.diagnostics().unique_parents >= 2);
    assert!(plan.diagnostics().max_family_size <= 2);
}

#[test]
fn systematic_reconfiguration_is_fixed_size_capped_and_replayable() {
    let evidence = [
        BasinEvidence::new(0.0, 0.0, 0.0).unwrap(),
        BasinEvidence::new(0.4, 0.2, 0.2).unwrap(),
        BasinEvidence::new(0.7, 0.8, 0.7).unwrap(),
        BasinEvidence::new(1.0, 1.0, 1.0).unwrap(),
    ];

    let first = reconfiguration_plan(&evidence, coefficients(), 0.375, 2).unwrap();
    let replay = reconfiguration_plan(&evidence, coefficients(), 0.375, 2).unwrap();

    assert_eq!(first.parents(), replay.parents());
    assert_eq!(first.parents().len(), 4);
    assert!(first.parents().iter().all(|parent| *parent < 4));
    assert!(first.diagnostics().max_family_size <= 2);
    assert!(first.diagnostics().effective_sample_size >= 1.0);
    assert!(first.diagnostics().effective_sample_size <= 4.0);
}

#[test]
fn equal_evidence_preserves_one_parent_per_chain() {
    let evidence = [BasinEvidence::new(0.5, 0.5, 0.5).unwrap(); 4];
    let plan = reconfiguration_plan(&evidence, coefficients(), 0.91, 2).unwrap();

    assert_eq!(plan.parents(), &[0, 1, 2, 3]);
    assert_eq!(plan.diagnostics().unique_parents, 4);
    assert_eq!(plan.diagnostics().max_family_size, 1);
    assert!((plan.diagnostics().effective_sample_size - 4.0).abs() < 1e-12);
    assert!(plan.diagnostics().offspring_variance.abs() < 1e-12);
}

#[test]
fn invalid_ranks_and_offsets_are_rejected() {
    assert!(BasinEvidence::new(-0.1, 0.5, 0.5).is_err());
    assert!(BasinEvidence::new(0.5, f64::NAN, 0.5).is_err());

    let evidence = [BasinEvidence::new(0.5, 0.5, 0.5).unwrap(); 4];
    assert!(reconfiguration_plan(&evidence, coefficients(), 1.0, 2).is_err());
    assert!(reconfiguration_plan(&evidence, coefficients(), 0.5, 0).is_err());
}
