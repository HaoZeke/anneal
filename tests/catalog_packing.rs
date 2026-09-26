use anneal_core::catalog::{
    OccupancyCertificate, PACKING_MERGE, PACKING_MOVE_EPS, PackingBook, different_decaf_family,
    lens_ring_displacement, occupancy_landfold_floor, occupancy_retire_at, occupancy_ring_census,
    occupancy_ring_floor, occupancy_ring_profile, packing_distance, packing_fingerprint,
    ring_leave_weight, same_packing,
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
    use anneal_core::catalog::{GoodTuringSample, leftover_arrivals_saturated};
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
fn landfold_floor_separates_lj75_marks_from_ico() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let mut book = PackingBook::default();
    book.observe(ico.as_slice().unwrap()).unwrap();
    book.observe(marks.as_slice().unwrap()).unwrap();
    let occupied = book.occupied_histograms();
    assert_eq!(occupied.len(), 2);
    let hists: Vec<Vec<f64>> = occupied.iter().map(|(_, h)| h.clone()).collect();
    let fams: Vec<usize> = occupied.iter().map(|(i, _)| *i).collect();
    assert_eq!(
        occupancy_landfold_floor(&hists, &fams),
        2,
        "book ico and Marks must be two landfold communities"
    );
}

#[test]
fn landfold_floor_on_the_book_sees_oh_after_live_ico_only() {
    let ico = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let mut book = PackingBook::default();
    book.observe(ico.as_slice().unwrap()).unwrap();
    book.observe(oh.as_slice().unwrap()).unwrap();
    let occupied = book.occupied_histograms();
    let hists: Vec<Vec<f64>> = occupied.iter().map(|(_, h)| h.clone()).collect();
    let fams: Vec<usize> = occupied.iter().map(|(i, _)| *i).collect();
    assert_eq!(book.occupied_among([ico.as_slice().unwrap()]), 1);
    assert_eq!(
        occupancy_landfold_floor(&hists, &fams),
        2,
        "Oh on the book is a second map community after extras Leave it"
    );
}

#[test]
fn landfold_two_means_bipartitions_a_leftover_decaf_chain() {
    let mut hists = Vec::new();
    let mut fams = Vec::new();
    for i in 0..8 {
        let t = 0.01 * i as f64;
        hists.push(vec![1.0 - t, t, 0.0]);
        fams.push(i);
    }
    assert_eq!(
        occupancy_landfold_floor(&hists, &fams),
        2,
        "2-means splits a leftover DECAF chain; FES maxima are the landfold figure, not this floor"
    );
}

#[test]
fn landfold_floor_separates_lj38_oh_from_ico() {
    let ico = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let mut book = PackingBook::default();
    let ico_h = {
        book.observe(ico.as_slice().unwrap()).unwrap();
        book.histogram(ico.as_slice().unwrap()).unwrap()
    };
    let oh_h = {
        book.observe(oh.as_slice().unwrap()).unwrap();
        book.histogram(oh.as_slice().unwrap()).unwrap()
    };
    let ico_f = book.family_of(&ico_h).unwrap();
    let oh_f = book.family_of(&oh_h).unwrap();
    assert_ne!(ico_f, oh_f);
    assert_eq!(
        occupancy_landfold_floor(&[ico_h.clone(), ico_h, oh_h], &[ico_f, ico_f, oh_f]),
        2,
        "Torgerson of DECAF L1 must split Oh from leftover ico"
    );
}

#[test]
fn landfold_floor_does_not_split_leftover_ico() {
    let ico = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let mut book = PackingBook::default();
    let h = {
        book.observe(ico.as_slice().unwrap()).unwrap();
        book.histogram(ico.as_slice().unwrap()).unwrap()
    };
    let family = book.family_of(&h).unwrap();
    let mut nudged = ico.clone();
    nudged[0] += 0.2;
    let h2 = book.histogram(nudged.as_slice().unwrap()).unwrap();
    assert_eq!(book.family_of(&h2), Some(family));
    assert_eq!(
        occupancy_landfold_floor(&[h, h2], &[family, family]),
        1,
        "leftover ico wells of one packing are one landfold community"
    );
}

fn rings(x: &Array1<f64>) -> (usize, usize, usize) {
    occupancy_ring_profile(x.as_slice().unwrap()).expect("cluster has three atoms")
}

#[test]
fn primitive_rings_separate_packings_and_keep_leftover_ico() {
    let ico38 = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh38 = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let ico75 = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks75 = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let r_ico38 = rings(&ico38);
    let r_oh38 = rings(&oh38);
    let r_ico75 = rings(&ico75);
    let r_marks = rings(&marks75);
    eprintln!("LJ38 ico rings {r_ico38:?} Oh {r_oh38:?}");
    eprintln!("LJ75 ico rings {r_ico75:?} Marks {r_marks:?}");
    assert_ne!(
        r_ico38, r_oh38,
        "LJ38 ico and Oh must differ in primitive 3/4/5 rings"
    );
    assert_ne!(
        r_ico75, r_marks,
        "LJ75 ico and Marks must differ in primitive 3/4/5 rings"
    );
    let mut nudged = ico38.clone();
    nudged[0] += 0.2;
    let r_left = rings(&nudged);
    eprintln!("LJ38 leftover-ico rings {r_left:?}");
    assert_eq!(
        r_left, r_ico38,
        "a leftover ico well must keep the ico ring profile"
    );
    assert_eq!(
        occupancy_ring_floor(&[r_ico38, r_left]),
        1,
        "leftover ico wells of one packing are one ring community"
    );
    assert_eq!(
        occupancy_ring_floor(&[r_ico38, r_oh38]),
        2,
        "LJ38 ico and Oh are two ring communities"
    );
    assert_eq!(
        occupancy_ring_floor(&[r_ico75, r_marks]),
        2,
        "LJ75 ico and Marks are two ring communities"
    );
}

#[test]
fn ring_leave_lens_sees_pentagon_atoms_on_ico_not_oh() {
    let ico38 = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh38 = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let ico = occupancy_ring_census(ico38.as_slice().unwrap()).unwrap();
    let oh = occupancy_ring_census(oh38.as_slice().unwrap()).unwrap();
    let ico_pent: usize = ico.atom.iter().filter(|w| w[2] > 0).count();
    let oh_pent: usize = oh.atom.iter().filter(|w| w[2] > 0).count();
    eprintln!(
        "LJ38 ico pentagon atoms {ico_pent}/{} Oh {oh_pent}/{}",
        ico.atom.len(),
        oh.atom.len()
    );
    assert!(
        ico_pent > 0 && ico_pent < ico.atom.len(),
        "ico pentagon incidence must be a proper subset so the lens can steer"
    );
    assert_eq!(oh_pent, 0, "Oh has no 5-rings to concentrate on");
    let ico_w: Vec<f64> = ico
        .atom
        .iter()
        .map(|&w| ring_leave_weight(ico.profile, w))
        .collect();
    assert!(
        ico_w.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            > ico_w.iter().copied().fold(f64::INFINITY, f64::min),
        "ico leave weights must not be uniform"
    );
    let mut dr: Vec<f64> = (0..ico38.len())
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    lens_ring_displacement(ico38.as_slice().unwrap(), &mut dr);
    let pent_step: f64 = ico
        .atom
        .iter()
        .enumerate()
        .filter(|(_, w)| w[2] > 0)
        .map(|(i, _)| dr[3 * i].abs())
        .sum();
    let other_step: f64 = ico
        .atom
        .iter()
        .enumerate()
        .filter(|(_, w)| w[2] == 0)
        .map(|(i, _)| dr[3 * i].abs())
        .sum();
    assert!(
        pent_step > other_step,
        "lensed ico step must put more length on pentagon atoms ({pent_step} vs {other_step})"
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
