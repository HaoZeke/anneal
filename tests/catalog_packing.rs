use anneal_core::catalog::{
    packing_distance, packing_fingerprint, same_packing, PackingBook, PACKING_MERGE,
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
fn dimer_coordinates_have_no_packing_fingerprint() {
    assert!(packing_fingerprint(&[0.0, 0.0, 0.0, 1.2, 0.0, 0.0]).is_none());
}

#[test]
fn lj75_marks_is_a_different_packing_from_the_icosahedral_competitor() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let ico_fp =
        packing_fingerprint(ico.as_slice().unwrap()).expect("LJ75 ico has a leftover mean");
    let marks_fp =
        packing_fingerprint(marks.as_slice().unwrap()).expect("LJ75 Marks has a leftover mean");

    assert!(same_packing(&ico_fp, &ico_fp));
    let gap = packing_distance(&ico_fp, &marks_fp);
    assert!(
        gap > PACKING_MERGE,
        "Mackay–Marks leftover {gap} must sit outside the packing well {PACKING_MERGE}"
    );
}

#[test]
fn packing_book_counts_one_family_and_reports_the_other_as_novel() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let ico_fp = packing_fingerprint(ico.as_slice().unwrap()).unwrap();
    let marks_fp = packing_fingerprint(marks.as_slice().unwrap()).unwrap();

    let mut book = PackingBook::default();
    let ico_family = book.observe(&ico_fp).unwrap();
    assert_eq!(book.observe(&ico_fp), Some(ico_family));
    assert_eq!(book.visits(ico_family), 2);
    assert!(book.novelty(&ico_fp) == 0.0);

    let marks_family = book.observe(&marks_fp).unwrap();
    assert_ne!(marks_family, ico_family);
    assert!(book.novelty(&ico_fp) > PACKING_MERGE);
    assert_eq!(book.visits(ico_family), 2);
    assert_eq!(book.visits(marks_family), 1);
}
