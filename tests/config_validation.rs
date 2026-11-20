// Copyright (c) anneal developers
//! `Config::validate` and the `try_*` molecular constructors refuse what the
//! search would assert on, and the search returns the same refusal as an
//! error before it evaluates anything.

use anneal_core::methods::cluster_hopping::{
    Config, InvalidConfig, Ledger, MoveLibrary, SearchError, check_molecular_groups,
    random_cluster_in_radius, run_with_gradient,
};
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;

const ATOMS: usize = 7;

fn run(cfg: &Config) -> (Result<(), SearchError>, usize) {
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut rng = StdRng::seed_from_u64(3);
    let start = random_cluster_in_radius(ATOMS, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(200);
    let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, _steps: usize| {
        led.charge();
        (potential.value_and_gradient(x).0, x.to_owned())
    };
    let mut grad = |led: &mut Ledger, x: ndarray::ArrayView1<f64>| {
        led.charge().then(|| potential.value_and_gradient(x).1)
    };
    let out = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    );
    (out.map(|_| ()), ledger.spent())
}

#[test]
fn groups_must_be_ordered_and_in_range() {
    assert!(check_molecular_groups(&[vec![0, 1, 2], vec![3, 4, 5]], 6).is_ok());
    assert!(check_molecular_groups(&[vec![0]], 1).is_ok());
    for (groups, n, words) in [
        (vec![vec![2, 1, 0]], 3, "increasing"),
        (vec![vec![0, 2, 1]], 3, "increasing"),
        (vec![vec![0, 0, 1]], 3, "repeats"),
        (vec![vec![0, 1, 3]], 3, "past the last"),
        (vec![vec![0, 1, usize::MAX]], 3, "past the last"),
        (vec![vec![]], 3, "at least one atom"),
    ] {
        let err = check_molecular_groups(&groups, n).unwrap_err();
        assert!(err.0.contains(words), "{groups:?}: {err}");
    }
}

#[test]
fn a_bad_molecular_preset_is_an_error_not_a_panic() {
    let water = |groups: Vec<Vec<usize>>| Config::try_for_molecular(vec![8, 1, 1], groups, 1.0);
    assert!(water(vec![vec![0, 1, 2]]).is_ok());
    assert!(water(vec![vec![2, 1, 0]]).is_err());
    assert!(water(vec![vec![1, 0, 2]]).is_err());
    assert!(Config::try_recommended_molecular(vec![8, 1, 1], vec![vec![2, 1, 0]], 1.0).is_err());
    assert!(Config::try_for_molecular(vec![], vec![], 1.0).is_err());
    assert!(Config::try_for_molecular(vec![0, 0], vec![vec![0, 1]], 1.0).is_err());
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(Config::try_for_molecular(vec![8, 1, 1], vec![vec![0, 1, 2]], scale).is_err());
    }
}

#[test]
fn a_water_group_after_the_first_is_still_recognised() {
    let cfg = Config::try_for_molecular(
        vec![8, 1, 1, 8, 1, 1],
        vec![vec![0, 1, 2], vec![3, 4, 5]],
        1.0,
    )
    .unwrap();
    assert!(matches!(
        cfg.move_library,
        MoveLibrary::Molecular {
            reverse_cycles: true,
            ..
        }
    ));
}

#[test]
#[should_panic(expected = "increasing order")]
fn the_panicking_preset_names_the_reason() {
    let _ = Config::for_molecular(vec![8, 1, 1], vec![vec![2, 1, 0]], 1.0);
}

#[test]
fn delayed_acceptance_needs_one_chain() {
    let mut cfg = Config::recommended(ATOMS);
    cfg.delayed_acceptance = true;
    cfg.replicas = 1;
    assert_eq!(cfg.validate(), Ok(()));
    cfg.replicas = 2;
    let err = cfg.validate().unwrap_err();
    assert!(
        err.0.starts_with("delayed acceptance needs a single chain"),
        "{err}"
    );
    let (out, spent) = run(&cfg);
    assert_eq!(out, Err(SearchError::InvalidConfig(err)));
    assert_eq!(spent, 0);
}

#[test]
fn every_refusal_reaches_the_search_as_an_error() {
    let mut groups = Config::recommended(ATOMS);
    groups.move_library = MoveLibrary::Molecular {
        groups: vec![vec![2, 1, 0]],
        reactive: false,
        reverse_cycles: false,
    };
    let mut floor = Config::recommended(ATOMS);
    floor.call_floor = Some(f64::NAN);
    let mut elements = Config::recommended(ATOMS);
    elements.call_floor_elements = Some(vec![1; ATOMS - 1]);
    for cfg in [groups, floor, elements] {
        let refused = cfg.validate().unwrap_err();
        let (out, spent) = run(&cfg);
        assert_eq!(out, Err(SearchError::InvalidConfig(refused.clone())));
        assert_eq!(spent, 0, "{refused}");
    }
    assert!(Config::recommended(ATOMS).validate().is_ok());
    let _: InvalidConfig = InvalidConfig("display".into());
}
