use anneal_core::catalog::{
    PACKING_MERGE, PACKING_MOVE_EPS, PackingBook, different_decaf_family, packing_distance,
    packing_fingerprint, same_packing,
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
    assert!(book.novelty(&ico_after) > PACKING_MERGE);
}

#[test]
fn a_second_look_at_the_same_ico_does_not_open_a_family() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let mut book = PackingBook::default();
    let first = book.observe(ico.as_slice().unwrap()).unwrap();
    let second = book.observe(ico.as_slice().unwrap()).unwrap();
    assert_eq!(first, second);
    assert_eq!(book.visits(first), 2);
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
