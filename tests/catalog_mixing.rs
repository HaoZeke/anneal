use anneal_core::catalog::{
    explore_collapsed, explore_must_leave, invert_mixing, mixed, occupancy_complete_at,
    occupant_rhat, rhat_series, sampled_minimum_is_dominant, stronger, AttractorStrength,
    OccupancyCertificate, MIXED_RHAT,
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
fn a_lone_mixed_floor_is_dominant_in_the_sample() {
    let ico = ico_floor();
    assert!(ico.mixed());
    assert!(sampled_minimum_is_dominant(&ico, &[], true));
}

#[test]
fn an_unmixed_lone_floor_is_not_dominant_in_the_sample() {
    let ico = AttractorStrength {
        energy: -173.252378,
        occupancy: 4,
        occupant_rhat: f64::INFINITY,
    };
    assert!(!sampled_minimum_is_dominant(&ico, &[], true));
}

#[test]
fn a_deeper_minority_is_not_dominant_against_a_more_occupied_floor() {
    let oh = oh_minority();
    let ico = ico_floor();
    assert!(stronger(&ico, &oh));
    assert!(!stronger(&oh, &ico));
    assert!(!sampled_minimum_is_dominant(&oh, &[ico], true));
}

#[test]
fn a_deeper_majority_that_has_mixed_is_dominant_in_the_sample() {
    let oh = oh_majority();
    let ico = AttractorStrength {
        energy: -173.252378,
        occupancy: 2,
        occupant_rhat: 0.0,
    };
    assert!(stronger(&oh, &ico));
    assert!(sampled_minimum_is_dominant(&oh, &[ico], true));
}

#[test]
fn a_mixed_deepest_well_is_not_dominant_against_a_flyby() {
    let deep = AttractorStrength {
        energy: -170.294257,
        occupancy: 4,
        occupant_rhat: 0.0,
    };
    let flyby = AttractorStrength {
        energy: -168.1,
        occupancy: 1,
        occupant_rhat: f64::INFINITY,
    };
    assert!(stronger(&deep, &flyby));
    assert!(!sampled_minimum_is_dominant(&deep, &[flyby], true));
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
    assert!(!sampled_minimum_is_dominant(&oh, &[ico], true));
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
fn occupant_rhat_cannot_drop_an_off_family_prefix() {
    let mut prefix_then_sibling = constant(-173.252378, 16);
    prefix_then_sibling.push(-170.0);
    assert!(
        mixed(occupant_rhat(&[
            prefix_then_sibling.clone(),
            prefix_then_sibling
        ])),
        "16 ico samples plus one sibling quench look mixed; the prefix must be dropped before occupant_rhat"
    );
}

#[test]
fn two_point_traces_are_not_a_mixing_certificate() {
    let short = [constant(-170.0, 2), constant(-170.0, 2)];
    assert_eq!(rhat_series(&short), 0.0);
    assert!(occupant_rhat(&short).is_infinite());
    let left = AttractorStrength {
        energy: -170.0,
        occupancy: 12,
        occupant_rhat: occupant_rhat(&short),
    };
    let right = AttractorStrength {
        energy: -168.0,
        occupancy: 12,
        occupant_rhat: occupant_rhat(&[constant(-168.0, 2), constant(-168.0, 2)]),
    };
    let verdict = invert_mixing(&[left, right], &[]);
    assert!(!verdict.certified_attractor);
}

#[test]
fn leftover_unsaturated_does_not_retire() {
    let ico = ico_floor();
    let shallower = AttractorStrength {
        energy: -170.0,
        occupancy: 2,
        occupant_rhat: 0.0,
    };
    let verdict = invert_mixing(&[ico, shallower], &[]);
    assert!(verdict.certified_attractor);
    assert_ne!(
        occupancy_complete_at(verdict.certified_attractor, true, false, 2, 2),
        Some(OccupancyCertificate::MixingCertified)
    );
    assert_eq!(
        occupancy_complete_at(verdict.certified_attractor, true, true, 2, 2),
        Some(OccupancyCertificate::MixingCertified)
    );
    let ico_minority = AttractorStrength {
        energy: -173.252378,
        occupancy: 2,
        occupant_rhat: 0.0,
    };
    let deep = invert_mixing(&[oh_majority(), ico_minority], &[]);
    assert!(deep.certified_attractor);
    assert_eq!(
        occupancy_complete_at(deep.certified_attractor, true, false, 1, 1),
        Some(OccupancyCertificate::MixingCertified)
    );
}

#[test]
fn inverted_mixing_certifies_only_a_winning_deeper_attractor() {
    let ico = AttractorStrength {
        energy: -173.252378,
        occupancy: 2,
        occupant_rhat: 0.0,
    };
    let series = [constant(-173.252378, 8), constant(-170.0, 8)];
    let verdict = invert_mixing(&[oh_majority(), ico], &series);
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
    assert!(verdict.certified_attractor);
    assert!(verdict.explore_collapsed);
    assert!(explore_collapsed(&series));
}

#[test]
fn mixed_decaf_labels_on_one_occupied_packing_force_leave() {
    // SOAP/DECAF already say these four walks sit on one packing.
    // Unmixed ico isomer energies are not a new family. Inverted GR
    // must leave that well; waiting for a second packing never opens it.
    let energies = [
        constant(-173.252378, 8),
        constant(-171.38, 8),
        constant(-170.45, 8),
        constant(-166.64, 8),
    ];
    let families = [
        constant(0.0, 8),
        constant(0.0, 8),
        constant(0.0, 8),
        constant(0.0, 8),
    ];
    assert!(!explore_collapsed(&energies));
    assert!(mixed(rhat_series(&families)));
    assert!(explore_must_leave(&energies, 4, 4));
}

#[test]
fn mixed_labels_on_one_of_two_known_packings_force_leave() {
    let energies = [
        constant(-171.38, 8),
        constant(-170.45, 8),
        constant(-166.64, 8),
    ];
    let families = [constant(0.0, 8), constant(0.0, 8), constant(0.0, 8)];
    assert!(mixed(rhat_series(&families)));
    assert!(!explore_collapsed(&energies));
    assert!(explore_must_leave(&energies, 3, 3));
}

#[test]
fn distinct_packing_labels_are_not_a_mixed_explore_set() {
    let families = [constant(0.0, 8), constant(1.0, 8)];
    assert!(!mixed(rhat_series(&families)));
    assert!(!explore_must_leave(&families, 1, 2));
}

#[test]
fn a_lone_walk_on_the_incumbent_packing_is_not_collapse() {
    let energies = [constant(-171.38, 8)];
    assert!(!explore_must_leave(&energies, 1, 1));
}
