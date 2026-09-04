use anneal_core::catalog::{
    OccupancyCertificate, OccupancyFold, PACKING_LINK, PACKING_MERGE, PACKING_MOVE_EPS,
    PackingBook, different_decaf_family, include_packing_reference, leaves_packing,
    nearby_packing,
    lens_ring_displacement, occupancy_fes_delta, occupancy_fes_from_histograms,
    occupancy_landfold_floor, occupancy_leave_new_class, occupancy_leave_new_packing,
    occupancy_map_fold, occupancy_retire_at, occupancy_ring_census, occupancy_ring_floor,
    occupancy_ring_profile, occupancy_sparsify_packing, packing_community_count, packing_distance,
    packing_fingerprint, packing_link_labels, packing_reference_book, remember_packing_reference,
    ring_leave_weight, same_packing, set_packing_references,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::{Array1, ArrayView1};

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
fn invert_neighbours_are_nearby_in_the_packing_map() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let ico = ico.as_slice().unwrap();
    let marks = marks.as_slice().unwrap();
    assert!(nearby_packing(ico, ico));
    assert!(
        !nearby_packing(ico, marks),
        "a far packing pair is a hear, not an invert neighbour"
    );
}

#[test]
fn catalog_reference_refresh_does_not_count_a_well_arrival() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let ico = ico.as_slice().unwrap();
    set_packing_references(Vec::new());

    include_packing_reference(ico);
    include_packing_reference(ico);
    let refreshed = packing_reference_book();
    assert_eq!(refreshed.len(), 1);
    assert_eq!(refreshed[0].visits, 1);

    remember_packing_reference(ico);
    let arrived = packing_reference_book();
    assert_eq!(arrived.len(), 1);
    assert_eq!(arrived[0].visits, 2);
    set_packing_references(Vec::new());
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
    assert!(!first.chao1_complete());
}

#[test]
fn packing_chao1_needs_no_singletons() {
    use anneal_core::catalog::GoodTuringSample;
    // n=20, n1=3, n2=0: leftover p0=0.15 is under the ceiling,
    // Chao1 is unbounded, packing is not complete.
    let leftover_ok = GoodTuringSample {
        n: 20,
        n1: 3,
        n2: 0,
    };
    assert!(leftover_ok.saturated());
    assert!(leftover_ok.chao1_unseen().is_none());
    assert!(!leftover_ok.chao1_complete());
    let complete = GoodTuringSample {
        n: 20,
        n1: 0,
        n2: 4,
    };
    assert!(complete.chao1_complete());
    assert_eq!(complete.chao1_unseen(), Some(0.0));
    let bounded = GoodTuringSample {
        n: 20,
        n1: 4,
        n2: 2,
    };
    assert!((bounded.chao1_unseen().unwrap() - 4.0).abs() < 1e-12);
    assert!(!bounded.chao1_complete());
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
    assert_eq!(book.well_sample().n1, 0);
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
    let fes = occupancy_fes_from_histograms(&hists).unwrap();
    assert_eq!(fes.minima, 1, "two equal-weight packings have the same F");
    assert!(fes.delta.is_none());
    let ico_f = fams[0];
    let marks_f = fams[1];
    for _ in 0..20 {
        book.credit_well(ico_f);
    }
    for _ in 0..5 {
        book.credit_well(marks_f);
    }
    let delta = occupancy_fes_delta(&book.occupied_well_counts()).unwrap();
    assert!(
        (delta - 4.0_f64.ln()).abs() < 1e-12,
        "packing F/kT is ln(n_max/n_2), got {delta}"
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
    let fes = occupancy_fes_from_histograms(&hists).unwrap();
    assert_eq!(
        fes.minima, 1,
        "the connected interpolant chain has one landfold KDE mode"
    );
    assert!(fes.delta.is_none());
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

#[test]
fn landfold_sparsify_collapses_leftover_ico_and_keeps_oh_as_a_hole() {
    use anneal_core::catalog::PRODUCTION_MINIMUM_VISITS;
    let ico = load_xyz(include_str!("fixtures/lj38_ico.xyz"));
    let oh = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let mut leftover = PackingBook::default();
    leftover.observe(ico.as_slice().unwrap()).unwrap();
    let ico_family = leftover
        .family_of(&leftover.histogram(ico.as_slice().unwrap()).unwrap())
        .unwrap();
    for _ in 0..PRODUCTION_MINIMUM_VISITS {
        leftover.credit_well(ico_family);
    }
    let leftover_map = occupancy_sparsify_packing(&leftover);
    assert_eq!(leftover_map.communities, 1);
    assert!(
        !leftover_map.holes,
        "a one-packing book with Chao1 well credits has no holes"
    );

    let mut book = PackingBook::default();
    let ico_family = book.observe(ico.as_slice().unwrap()).unwrap();
    book.observe(oh.as_slice().unwrap()).unwrap();
    for _ in 0..PRODUCTION_MINIMUM_VISITS {
        book.credit_well(ico_family);
    }
    let open = occupancy_sparsify_packing(&book);
    assert_eq!(open.floor, 2);
    assert_eq!(open.communities, 2);
    assert!(
        open.holes,
        "Oh on the book with no well arrivals is a hole extras Leave into"
    );

    let oh_family = book
        .family_of(&book.histogram(oh.as_slice().unwrap()).unwrap())
        .unwrap();
    for _ in 0..PRODUCTION_MINIMUM_VISITS {
        book.credit_well(oh_family);
    }
    let closed = occupancy_sparsify_packing(&book);
    assert!(!closed.holes, "both book packings credited closes holes");
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
    assert!(
        occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            false,
            true,
            book.occupied_among([ico.as_slice().unwrap()]),
            1
        ),
        "live rematch of last extras is not the book count; passing it would waive leftover"
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
    let version = book.version();
    let second = book.observe(ico.as_slice().unwrap()).unwrap();
    assert_eq!(first, second);
    assert_eq!(book.visits(first), 2);
    assert_eq!(book.version(), version);
    assert_eq!(book.occupied_family_count(), 1);
    assert_eq!(book.occupied_packing_count(), 1);
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
    assert_eq!(queried.occupied_packing_count(), 2);
}

#[test]
fn a_query_inherits_the_cached_packing_community() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let ico = ico.as_slice().unwrap();
    let marks = marks.as_slice().unwrap();
    let mut book = PackingBook::default();
    let family = book.observe(ico).expect("LJ75 ico has a class histogram");
    let histogram = book.histogram(ico).expect("ico histogram");
    assert_eq!(book.families_sharing_community(&histogram), vec![family]);
    assert_eq!(
        book.occupied_packing_count(),
        book.occupied_packing_count(),
        "the community fold is stable for an unchanged book"
    );
    let marks_family = book.observe(marks).expect("Marks opens a family");
    assert_ne!(family, marks_family);
    assert_eq!(book.occupied_packing_count(), 2);
    let marks_histogram = book.histogram(marks).expect("Marks histogram");
    assert_eq!(
        book.families_sharing_community(&marks_histogram),
        vec![marks_family],
        "Marks stays in its own packing community"
    );
}

#[test]
fn switch_saturates_far_l1_asinh_does_not() {
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    let c = vec![0.0, 0.0, 0.0, 1.0];
    let d_ab = packing_distance(&a, &b);
    let d_ac = packing_distance(&a, &c);
    assert!(d_ab > 0.0 && (d_ab - d_ac).abs() < 1e-12);
    let far = vec![0.0, 0.0, 1.0, 0.0];
    let d_near = packing_distance(&a, &b);
    let d_far = packing_distance(&a, &far) + packing_distance(&b, &c);
    assert!(d_far > d_near);
    let sigma = d_near;
    let sw_near = 1.0 - 1.0 / (1.0 + (d_near / sigma).powi(2));
    let sw_far = 1.0 - 1.0 / (1.0 + ((10.0 * d_near) / sigma).powi(2));
    let sw_farer = 1.0 - 1.0 / (1.0 + ((20.0 * d_near) / sigma).powi(2));
    assert!(
        (sw_farer - sw_far).abs() < 0.05,
        "switch must flatten the far tail ({sw_far} vs {sw_farer})"
    );
    let as_far = (10.0_f64).asinh() / (2.0 * 1.0_f64.asinh());
    let as_farer = (20.0_f64).asinh() / (2.0 * 1.0_f64.asinh());
    assert!(
        as_farer - as_far > 0.05,
        "asinh must keep 10σ and 20σ ordered ({as_far} vs {as_farer})"
    );
}

/// A quenched icosahedral isomer: a distinct minimum on the ico shelf.
///
/// One surface atom is moved and the structure relaxed, which is the move
/// that isomerises a shell without unbuilding it. The search over sites and
/// amplitudes is deterministic, and the result is checked to be a different
/// minimum from the sealed floor and still on the shelf, so a test using this
/// is never quietly comparing the floor with itself.
fn lj75_ico_isomer() -> Array1<f64> {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let potential = PairPotential::new(75, PairKind::LennardJones, 40.0);
    let floor = potential.value_and_gradient(ico.view()).0;
    let atoms = ico.len() / 3;
    let mut centre = [0.0; 3];
    for atom in 0..atoms {
        for axis in 0..3 {
            centre[axis] += ico[3 * atom + axis] / atoms as f64;
        }
    }
    let mut surface: Vec<(usize, f64)> = (0..atoms)
        .map(|atom| {
            let radius = (0..3)
                .map(|axis| (ico[3 * atom + axis] - centre[axis]).powi(2))
                .sum::<f64>();
            (atom, radius)
        })
        .collect();
    surface.sort_by(|left, right| right.1.total_cmp(&left.1));
    for &(atom, _) in surface.iter().take(8) {
        for axis in 0..3 {
            for step in 1..=6 {
                let shift = 0.4 * f64::from(step);
                for sign in [1.0, -1.0] {
                    let mut start = ico.clone();
                    start[3 * atom + axis] += sign * shift;
                    let mut opt = WarmLbfgs::default();
                    let (energy, relaxed, _) =
                        opt.minimize(start.view(), 2000, |v: ArrayView1<f64>| {
                            Some(potential.value_and_gradient(v))
                        });
                    if energy > floor + 1e-6 && energy < floor + 8.0 {
                        return relaxed;
                    }
                }
            }
        }
    }
    panic!("no icosahedral isomer on the shelf under the surface-atom search");
}

#[test]
fn a_packing_is_what_its_cells_chain_into_not_a_radius() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let isomer = lj75_ico_isomer();
    let mut book = PackingBook::default();
    for state in [&ico, &marks, &isomer] {
        book.observe(state.as_slice().unwrap());
    }
    let histograms: Vec<Vec<f64>> = [&ico, &marks, &isomer]
        .into_iter()
        .map(|state| book.histogram(state.as_slice().unwrap()).unwrap())
        .collect();
    let to_marks = packing_distance(&histograms[0], &histograms[1]);
    let to_isomer = packing_distance(&histograms[0], &histograms[2]);
    let funnel = anneal_core::funnel_bo::FunnelModel::new(0.15, 20.0, 1e-2);
    let k_iso = funnel.similarity(
        ndarray::Array1::from(histograms[0].clone()).view(),
        ndarray::Array1::from(histograms[2].clone()).view(),
    );
    let k_marks = funnel.similarity(
        ndarray::Array1::from(histograms[0].clone()).view(),
        ndarray::Array1::from(histograms[1].clone()).view(),
    );
    assert!(
        k_iso > k_marks,
        "Hellinger kernel must rank ico-isomer {k_iso} closer than ico-Marks {k_marks}"
    );
    assert!(
        to_marks > PACKING_MERGE,
        "ico-Marks {to_marks} must clear the cell grain"
    );
    let labels = packing_link_labels(&histograms, PACKING_LINK);
    assert_eq!(
        labels[0], labels[2],
        "a quenched ico isomer chains to ico: cell L1 {to_isomer}, ico-Marks {to_marks}"
    );
    assert_ne!(labels[0], labels[1], "Marks chains to neither");
    assert_eq!(packing_community_count(&histograms), 2);
}

#[test]
fn leave_accepts_marks_and_refuses_an_ico_isomer() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("fixtures/lj75_marks.xyz"));
    let isomer = lj75_ico_isomer();
    let ico = ico.as_slice().unwrap();
    let references = vec![ico.to_vec()];
    assert!(
        leaves_packing(ico, marks.as_slice().unwrap(), &references),
        "Marks is a packing the book does not hold"
    );
    assert!(
        !leaves_packing(ico, isomer.as_slice().unwrap(), &references),
        "an isomer of the occupied packing is not a Leave"
    );
    assert!(
        occupancy_leave_new_packing(ico, marks.as_slice().unwrap()),
        "the Leave accept fires on Marks"
    );
    assert!(
        !occupancy_leave_new_packing(ico, isomer.as_slice().unwrap()),
        "the Leave accept does not fire on an ico isomer"
    );
}

#[test]
fn the_cell_grain_adopts_the_isomer_the_packing_grain_refuses() {
    let ico = load_xyz(include_str!("fixtures/lj75_ico.xyz"));
    let isomer = lj75_ico_isomer();
    let ico = ico.as_slice().unwrap();
    let isomer = isomer.as_slice().unwrap();
    assert!(
        occupancy_leave_new_class(ico, isomer),
        "the cell grain calls an ico isomer a new class, which is the leak"
    );
    assert!(
        !occupancy_leave_new_packing(ico, isomer),
        "the packing grain does not"
    );
}
