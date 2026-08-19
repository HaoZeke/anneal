use anneal_core::catalog::{
    different_decaf_family, occupancy_retire_at, packing_distance, packing_fingerprint,
    same_packing, OccupancyCertificate, PackingBook, PACKING_MERGE, PACKING_MOVE_EPS,
};
use ndarray::Array1;

fn load_xyz(text: &str) -> Array1<f64> {
    let coordinates = text
        .lines()
        .skip(2)
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .take(3)
                .map(str::parse::<f64>)
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    Array1::from_vec(coordinates)
}

#[test]
fn decaf_histogram_separates_lj38_oh_from_the_icosahedral_competitor() {
    let ico = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let mut book = PackingBook::default();
    let ico_family = book
        .observe(ico.as_slice().unwrap())
        .expect("LJ38 ico has a class histogram");
    let ico_hist = book.histogram(ico.as_slice().unwrap()).unwrap();
    let oh_hist = book.histogram(oh.as_slice().unwrap()).unwrap();
    let gap = packing_distance(&ico_hist, &oh_hist);
    eprintln!("LJ38 ico–Oh DECAF L1 {gap}");
    assert!(
        gap > PACKING_MERGE,
        "LJ38 ico–Oh DECAF L1 {gap} must sit outside the packing well {PACKING_MERGE}"
    );
    let oh_family = book
        .observe(oh.as_slice().unwrap())
        .expect("Oh opens a second family");
    assert_ne!(oh_family, ico_family);
    assert_eq!(book.occupied_family_count(), 2);
}

#[test]
fn dimer_coordinates_have_no_packing_fingerprint() {
    assert!(packing_fingerprint(&[0.0, 0.0, 0.0, 1.2, 0.0, 0.0]).is_none());
}

#[test]
fn different_decaf_family_is_ico_versus_marks_not_an_ico_isomer() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let ico = ico.as_slice().unwrap();
    let marks = marks.as_slice().unwrap();
    assert!(different_decaf_family(ico, marks));
    assert!(!different_decaf_family(ico, ico));
}

#[test]
fn decaf_histogram_separates_lj75_marks_from_the_icosahedral_floor() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let mut book = PackingBook::default();
    let ico_family = book
        .observe(ico.as_slice().unwrap())
        .expect("LJ75 ico has a class histogram");
    let ico_hist = book
        .histogram(ico.as_slice().unwrap())
        .expect("ico histogram stays readable");
    let marks_hist = book
        .histogram(marks.as_slice().unwrap())
        .expect("Marks histogram stays readable");

    assert!(same_packing(&ico_hist, &ico_hist));
    let unseen_gap = packing_distance(&ico_hist, &marks_hist);
    assert!(
        unseen_gap > PACKING_MERGE,
        "ico–Marks DECAF L1 {unseen_gap} must sit outside the packing well {PACKING_MERGE}"
    );
    let marks_family = book
        .observe(marks.as_slice().unwrap())
        .expect("Marks opens a second family");
    assert_ne!(marks_family, ico_family);
    let ico_after = book.histogram(ico.as_slice().unwrap()).unwrap();
    let marks_after = book.histogram(marks.as_slice().unwrap()).unwrap();
    let pooled_gap = packing_distance(&ico_after, &marks_after);
    assert!(
        pooled_gap > PACKING_MERGE,
        "pooled ico–Marks L1 {pooled_gap} must stay a different family"
    );
    assert_eq!(book.visits(ico_family), 1);
    assert_eq!(book.visits(marks_family), 1);
    assert_eq!(book.occupied_family_count(), 2);
    assert!(book.novelty(&ico_after) > PACKING_MERGE);
}

#[test]
fn leftover_first_wave_arrivals_are_not_saturated() {
    use anneal_core::catalog::{leftover_arrivals_saturated, GoodTuringSample};
    assert!(!leftover_arrivals_saturated(std::iter::repeat_n(1u64, 48)));
    assert!(leftover_arrivals_saturated(std::iter::repeat_n(2u64, 20)));
    let first = GoodTuringSample::from_counts(std::iter::repeat_n(1u64, 48));
    assert_eq!(first.n, 48);
    assert_eq!(first.n1, 48);
    assert!((first.unseen().unwrap() - 1.0).abs() < 1e-12);
    assert!(!first.saturated());
}

#[test]
fn leftover_soap_gt_plus_two_singleton_packings_is_not_a_certificate() {
    let ico = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let mut book = PackingBook::default();
    book.observe(ico.as_slice().unwrap()).unwrap();
    book.observe(oh.as_slice().unwrap()).unwrap();
    let live = [ico.as_slice().unwrap(), oh.as_slice().unwrap()];
    assert_eq!(book.occupied_family_count(), 2);
    assert!(!book.families_saturated());
    assert_eq!(book.certificate_family_count(live), 1);
}

#[test]
fn packing_good_turing_uses_the_census_production_floor() {
    use anneal_core::catalog::PRODUCTION_MINIMUM_VISITS;
    let ico = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let mut book = PackingBook::default();
    let ico_family = book.observe(ico.as_slice().unwrap()).unwrap();
    let oh_family = book.observe(oh.as_slice().unwrap()).unwrap();
    assert!(!book.families_saturated());
    for _ in 0..PRODUCTION_MINIMUM_VISITS {
        book.observe(ico.as_slice().unwrap()).unwrap();
        book.observe(oh.as_slice().unwrap()).unwrap();
    }
    assert!(
        !book.families_saturated(),
        "hop re-observes of the same two structures are not leftover-well arrivals"
    );
    for _ in 0..PRODUCTION_MINIMUM_VISITS {
        book.credit_well(ico_family);
        book.credit_well(oh_family);
    }
    let live = [ico.as_slice().unwrap(), oh.as_slice().unwrap()];
    assert!(book.families_saturated());
    assert_eq!(book.certificate_family_count(live), 2);
    assert_eq!(
        book.occupied_among([ico.as_slice().unwrap(), ico.as_slice().unwrap()]),
        1
    );
}

#[test]
fn leftover_waiver_uses_book_families_not_live_rematch() {
    let ico = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let mut book = PackingBook::default();
    book.observe(ico.as_slice().unwrap()).unwrap();
    book.observe(oh.as_slice().unwrap()).unwrap();
    assert_eq!(book.occupied_family_count(), 2);
    assert_eq!(book.occupied_among([ico.as_slice().unwrap()]), 1);
    assert!(
        !occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            false,
            true,
            book.occupied_family_count(),
            1
        ),
        "Oh on the book is a second Boender cell; extras Leaving it must not waive leftover SOAP"
    );
}

#[test]
fn throwaway_fingerprint_is_not_the_shared_book() {
    let ico = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let ico = ico.as_slice().unwrap();
    let oh = oh.as_slice().unwrap();
    let fp_oh = packing_fingerprint(oh).expect("Oh has a private codebook histogram");
    let mut book = PackingBook::default();
    book.observe(ico).unwrap();
    book.observe(oh).unwrap();
    let shared_oh = book.histogram(oh).unwrap();
    assert_ne!(
        fp_oh, shared_oh,
        "FunnelModel EI must observe the coordinator book, not a per-structure codebook"
    );
}

#[test]
fn a_second_look_at_the_same_ico_does_not_open_a_family() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let mut book = PackingBook::default();
    let first = book.observe(ico.as_slice().unwrap()).unwrap();
    let second = book.observe(ico.as_slice().unwrap()).unwrap();
    assert_eq!(first, second);
    assert_eq!(book.visits(first), 2);
    assert_eq!(book.occupied_family_count(), 1);
    assert_eq!(
        book.novelty(&book.histogram(ico.as_slice().unwrap()).unwrap()),
        0.0
    );
}

#[test]
fn a_sub_threshold_displacement_reuses_the_decaf_histogram() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let mut book = PackingBook::default();
    book.observe(ico.as_slice().unwrap()).unwrap();
    let first = book.histogram(ico.as_slice().unwrap()).unwrap();
    let mut nudged = ico.clone();
    nudged[0] += 0.1 * PACKING_MOVE_EPS;
    let reused = book.histogram(nudged.as_slice().unwrap()).unwrap();
    assert_eq!(first, reused);
}

#[test]
fn a_query_histogram_does_not_change_what_the_book_learns() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let ico = ico.as_slice().unwrap();
    let marks = marks.as_slice().unwrap();

    // The coordinator queries every live structure on each policy
    // request, and a query folds environments the codebook has not seen
    // into one shared bin. Crediting a visit from that histogram counts
    // the structure against a family it was never assigned to and leaves
    // its environments out of the codebook for good.
    let mut queried = PackingBook::default();
    queried
        .observe(ico)
        .expect("LJ75 ico has a class histogram");
    let _ = queried.histogram(marks).expect("a query answers");
    let queried_family = queried.observe(marks).expect("Marks opens a family");

    let mut direct = PackingBook::default();
    direct.observe(ico).expect("LJ75 ico has a class histogram");
    let direct_family = direct.observe(marks).expect("Marks opens a family");

    assert_eq!(queried_family, direct_family);
    assert_eq!(
        queried.histogram(marks).unwrap(),
        direct.histogram(marks).unwrap()
    );
    assert_eq!(queried.visits(direct_family), direct.visits(direct_family));
    assert_eq!(queried.occupied_family_count(), 2);
}
