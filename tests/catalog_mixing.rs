use anneal_core::catalog::{
    AttractorStrength, certified_global_minimum, explore_collapsed, invert_mixing, mixed,
    rhat_series, stronger, MIXED_RHAT,
};

fn constant(value: f64, len: usize) -> Vec<f64> {
    vec![value; len]
}

fn ico_floor() -> AttractorStrength {
    AttractorStrength {
        energy: -173.252378,
        occupancy: 4,
        occupant_rhat: 0.0,
    }
}

fn oh_minority() -> AttractorStrength {
    AttractorStrength {
        energy: -173.928427,
        occupancy: 1,
        occupant_rhat: f64::INFINITY,
    }
}

fn oh_majority() -> AttractorStrength {
    AttractorStrength {
        energy: -173.928427,
        occupancy: 3,
        occupant_rhat: 0.0,
    }
}

#[test]
fn identical_constant_traces_are_mixed() {
    let rhat = rhat_series(&[constant(-173.252378, 16), constant(-173.252378, 16)]);
    assert_eq!(rhat, 0.0);
    assert!(mixed(rhat));
    assert!(rhat < MIXED_RHAT);
}

#[test]
fn constant_traces_at_different_floors_are_not_mixed() {
    let rhat = rhat_series(&[constant(-173.928427, 16), constant(-173.252378, 16)]);
    assert!(rhat.is_infinite());
    assert!(!mixed(rhat));
}

#[test]
fn a_lone_mixed_floor_is_not_a_certified_global_minimum() {
    let ico = ico_floor();
    assert!(ico.mixed());
    assert!(!certified_global_minimum(&ico, &[], true));
}

#[test]
fn a_deeper_minority_does_not_certify_against_a_more_occupied_floor() {
    let oh = oh_minority();
    let ico = ico_floor();
    assert!(stronger(&ico, &oh));
    assert!(!stronger(&oh, &ico));
    assert!(!certified_global_minimum(&oh, &[ico], true));
}

#[test]
fn a_deeper_majority_that_has_mixed_is_certified() {
    let oh = oh_majority();
    let ico = AttractorStrength {
        energy: -173.252378,
        occupancy: 1,
        occupant_rhat: f64::INFINITY,
    };
    assert!(stronger(&oh, &ico));
    assert!(certified_global_minimum(&oh, &[ico], true));
}

#[test]
fn equal_occupancy_is_not_a_stronger_attractor() {
    let oh = AttractorStrength {
        energy: -173.928427,
        occupancy: 2,
        occupant_rhat: 0.0,
    };
    let ico = AttractorStrength {
        energy: -173.252378,
        occupancy: 2,
        occupant_rhat: 0.0,
    };
    assert!(!stronger(&oh, &ico));
    assert!(!certified_global_minimum(&oh, &[ico], true));
}

#[test]
fn energy_tie_is_not_uniquely_deepest() {
    let left = AttractorStrength {
        energy: -173.25,
        occupancy: 3,
        occupant_rhat: 0.0,
    };
    let right = AttractorStrength {
        energy: -173.25,
        occupancy: 1,
        occupant_rhat: f64::INFINITY,
    };
    let verdict = invert_mixing(&[left, right], &[constant(-173.25, 8), constant(-170.0, 8)]);
    assert!(!verdict.certified_attractor);
    assert!(!verdict.explore_collapsed);
}

#[test]
fn inverted_mixing_certifies_only_a_winning_deeper_attractor() {
    let series = [
        constant(-173.928427, 8),
        constant(-173.928427, 8),
        constant(-173.928427, 8),
        constant(-173.252378, 8),
    ];
    let verdict = invert_mixing(&[oh_majority(), ico_floor()], &series[3..]);
    assert!(verdict.certified_attractor);
    assert!(!verdict.explore_collapsed);
}

#[test]
fn a_lone_ico_collapse_is_explore_failure_not_a_certificate() {
    let series = [
        constant(-173.252378, 8),
        constant(-173.252378, 8),
        constant(-173.252378, 8),
        constant(-173.252378, 8),
    ];
    let verdict = invert_mixing(&[ico_floor()], &series);
    assert!(!verdict.certified_attractor);
    assert!(verdict.explore_collapsed);
    assert!(explore_collapsed(&series));
}
