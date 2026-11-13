use super::*;
use crate::methods::cluster_hopping::min_pair_distance;
use crate::potentials::Gupta;
use ndarray::Array1;
use proptest::prelude::*;

fn pair(distance: f64) -> Array1<f64> {
    Array1::from(vec![0.0, 0.0, 0.0, distance, 0.0, 0.0])
}

fn molecular(species: Vec<u32>, groups: Vec<Vec<usize>>, reactive: bool) -> Config {
    let mut cfg = Config::for_molecular(species, groups, 1.0);
    if let MoveLibrary::Molecular { reactive: flag, .. } = &mut cfg.move_library {
        *flag = reactive;
    }
    cfg
}

fn two_waters(reactive: bool) -> Config {
    molecular(
        vec![8, 1, 1, 8, 1, 1],
        vec![vec![0, 1, 2], vec![3, 4, 5]],
        reactive,
    )
}

/// Water A's second hydrogen sits `reach` from water B's oxygen.
fn transferred(reach: f64) -> Array1<f64> {
    let oxygen_b = 2.0 + reach;
    Array1::from(vec![
        0.0,
        0.0,
        0.0, // O, water A
        -0.24,
        0.93,
        0.0, // H, water A
        2.0,
        0.0,
        0.0, // H, water A, on its way to B
        oxygen_b,
        0.0,
        0.0, // O, water B
        oxygen_b + 0.24,
        0.93,
        0.0, // H, water B
        oxygen_b + 0.24,
        -0.93,
        0.0, // H, water B
    ])
}

#[test]
fn a_species_free_search_is_never_walled() {
    let cfg = Config::for_cluster(2);
    let wall = CallWall::of(&cfg);
    assert!(matches!(wall, CallWall::Open));
    assert!(!wall.refuses(pair(0.0).view()));
    assert!(!wall.refuses(pair(f64::NAN).view()));
}

#[test]
fn a_labelled_metal_keeps_the_absolute_floor() {
    for species in [vec![29, 29], vec![29, 79], vec![79, 79], vec![78, 29]] {
        let cfg = Config::for_cluster(2).with_species(species.clone(), None);
        let wall = CallWall::of(&cfg);
        assert!(matches!(wall, CallWall::Absolute), "{species:?}");
        assert!(wall.refuses(pair(OVERLAP_CALL_LIMIT).view()));
        assert!(!wall.refuses(pair(OVERLAP_CALL_LIMIT + 1e-9).view()));
        // Bulk copper's nearest neighbour, inside the Cu-Cu covalent sum.
        assert!(!wall.refuses(pair(2.556).view()), "{species:?}");
    }
}

/// A labelled metal dimer is not a molecule. Its Gupta minimum is inside the
/// summed covalent radii, and a search that refuses that contact never
/// reaches it.
#[test]
fn labelled_metal_dimers_reach_their_gupta_minimum() {
    for species in [vec![29, 29], vec![29, 79], vec![79, 79]] {
        let gupta = Gupta::elements(&species);
        let r0 = Gupta::homonuclear_r0(species[0]).expect("tabulated");
        let mut polish = Ledger::new(10_000);
        let (reference, minimum) =
            polish_answer(&gupta, &mut polish, pair(r0).view(), 1e-9, 2_000).expect("a minimum");
        let bond = min_pair_distance(minimum.view());
        let covalent = crate::methods::cluster_hopping::covalent_radius(species[0])
            + crate::methods::cluster_hopping::covalent_radius(species[1]);
        if species == [29, 29] {
            assert!(bond < covalent, "premise: Cu2 at {bond} inside {covalent}");
        }
        let cfg = Config::for_cluster(2).with_species(species.clone(), None);
        let mut ledger = Ledger::new(4_000);
        let (out, _) = search(&gupta, &cfg, &mut ledger, 5).expect("finite start");
        assert!(
            out.best <= reference + 1e-6,
            "{species:?}: best {} against the dimer minimum {reference} at {bond}",
            out.best
        );
        let reached = min_pair_distance(out.best_state.expect("a state").view());
        assert!(
            (reached - bond).abs() < 1e-3,
            "{species:?}: {reached} against {bond}"
        );
    }
}

#[test]
fn a_molecular_library_reads_the_covalent_sums() {
    let cfg = two_waters(false);
    let wall = CallWall::of(&cfg);
    assert!(matches!(
        wall,
        CallWall::Covalent {
            reactive: false,
            ..
        }
    ));
    // An O-H contact between rigid waters inside the 0.97 Å sum.
    assert!(wall.refuses(transferred(0.95).view()));
    assert!(!wall.refuses(transferred(1.30).view()));
}

#[test]
fn a_reactive_library_may_form_a_bond_between_groups() {
    let cfg = two_waters(true);
    let wall = CallWall::of(&cfg);
    assert!(matches!(wall, CallWall::Covalent { reactive: true, .. }));
    assert!(!wall.refuses(transferred(0.95).view()));
    assert!(!wall.refuses(transferred(1.30).view()));
    // A compressed bond is still refused.
    assert!(wall.refuses(transferred(0.85).view()));
}

#[test]
fn a_non_finite_geometry_is_refused_when_walled() {
    let labelled = Config::for_cluster(2).with_species(vec![29, 79], None);
    let water = two_waters(false);
    let reactive = two_waters(true);
    for cfg in [&labelled, &water, &reactive] {
        let wall = CallWall::of(cfg);
        let mut point = if cfg.n_points == 2 {
            pair(2.5)
        } else {
            transferred(1.3)
        };
        assert!(!wall.refuses(point.view()));
        for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            point[1] = poison;
            assert!(wall.refuses(point.view()), "{wall:?} {poison}");
        }
    }
}

#[test]
fn a_single_atom_is_never_refused() {
    let cfg = molecular(vec![8], vec![vec![0]], false);
    let point = Array1::from(vec![0.0, 0.0, 0.0]);
    assert!(!CallWall::of(&cfg).refuses(point.view()));
}

#[test]
fn the_call_floor_is_symmetric_and_never_below_the_absolute_one() {
    for za in 0..120 {
        for zb in 0..120 {
            for bonded in [false, true] {
                let floor = molecular_call_floor(za, zb, bonded);
                assert_eq!(floor, molecular_call_floor(zb, za, bonded));
                assert!(floor >= OVERLAP_CALL_LIMIT);
            }
            assert!(molecular_call_floor(za, zb, true) <= molecular_call_floor(za, zb, false));
        }
    }
}

const ELEMENTS: [u32; 6] = [1, 6, 7, 8, 29, 79];

fn system() -> impl Strategy<Value = (Vec<f64>, Vec<u32>, Vec<usize>)> {
    (2usize..8).prop_flat_map(|n| {
        (
            proptest::collection::vec(-2.5f64..2.5, 3 * n),
            proptest::collection::vec(proptest::sample::select(&ELEMENTS[..]), n),
            proptest::collection::vec(0usize..3, n),
        )
    })
}

fn grouped(labels: &[usize]) -> Vec<Vec<usize>> {
    (0..3)
        .map(|group| {
            labels
                .iter()
                .enumerate()
                .filter(|(_, label)| **label == group)
                .map(|(atom, _)| atom)
                .collect()
        })
        .filter(|atoms: &Vec<usize>| !atoms.is_empty())
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn the_walls_nest_and_follow_the_geometry(
        (coordinates, species, labels) in system(),
        stretch in 1.01f64..3.0,
    ) {
        let n = species.len();
        let point = Array1::from(coordinates);
        let groups = grouped(&labels);
        let labelled = Config::for_cluster(n).with_species(species.clone(), None);
        let rigid = molecular(species.clone(), groups.clone(), false);
        let reactive = molecular(species.clone(), groups.clone(), true);
        let absolute = CallWall::of(&labelled).refuses(point.view());
        let covalent = CallWall::of(&rigid).refuses(point.view());
        let bonding = CallWall::of(&reactive).refuses(point.view());

        prop_assert_eq!(absolute, min_pair_distance(point.view()) <= OVERLAP_CALL_LIMIT);
        prop_assert!(!absolute || bonding, "the bond floor is looser than the absolute one");
        prop_assert!(!bonding || covalent, "a reactive wall refuses more than a rigid one");

        let far = &point * stretch;
        for cfg in [&labelled, &rigid, &reactive] {
            let wall = CallWall::of(cfg);
            if !wall.refuses(point.view()) {
                prop_assert!(!wall.refuses(far.view()), "{:?} refused a stretched cluster", wall);
            }
        }

        // Reversing the atoms, with their species and groups, is the same cluster.
        let order: Vec<usize> = (0..n).rev().collect();
        let mut reversed = Array1::zeros(3 * n);
        for (to, &from) in order.iter().enumerate() {
            for axis in 0..3 {
                reversed[3 * to + axis] = point[3 * from + axis];
            }
        }
        let species_r: Vec<u32> = order.iter().map(|&i| species[i]).collect();
        let labels_r: Vec<usize> = order.iter().map(|&i| labels[i]).collect();
        let groups_r = grouped(&labels_r);
        let rigid_r = molecular(species_r.clone(), groups_r.clone(), false);
        let reactive_r = molecular(species_r, groups_r, true);
        prop_assert_eq!(CallWall::of(&rigid_r).refuses(reversed.view()), covalent);
        prop_assert_eq!(CallWall::of(&reactive_r).refuses(reversed.view()), bonding);
    }
}
