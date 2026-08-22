//! Which observable separates the LJ75 Marks packing from the ico shelf.
//!
//!     decaf_packing_separator [ISOMERS]
//!
//! A scalar L1 radius on DECAF histograms does not: quenched icosahedral
//! isomers reach further from the ico reference than Marks does. This tool
//! measures the three candidate separators on one sample of shelf isomers
//! plus both sealed fixtures:
//!
//! * nearest-reference assignment in DECAF L1 on a shared codebook,
//! * single-linkage components of the shared book at a radius ladder,
//! * the Franzblau ring profile, raw counts and normalized shares.

use anneal_core::catalog::{PackingBook, occupancy_ring_profile, packing_distance};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn load_xyz(text: &str) -> Array1<f64> {
    let coordinates = text
        .lines()
        .skip(2)
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .take(3)
                .map(|value| value.parse::<f64>().expect("fixture coordinate parses"))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<f64>>();
    Array1::from(coordinates)
}

fn quench(potential: &PairPotential, x: ArrayView1<f64>, steps: usize) -> (f64, Array1<f64>) {
    let mut opt = WarmLbfgs::default();
    let (energy, relaxed, _) = opt.minimize(x, steps, |v| Some(potential.value_and_gradient(v)));
    (energy, relaxed)
}

fn perturbed(x: ArrayView1<f64>, sigma: f64, rng: &mut StdRng) -> Array1<f64> {
    let normal = Normal::new(0.0, sigma).expect("sigma is positive");
    let mut y = x.to_owned();
    for value in y.iter_mut() {
        *value += normal.sample(rng);
    }
    y
}

/// Connected components of a single-linkage graph at `radius`.
fn components(distance: &[Vec<f64>], radius: f64) -> Vec<usize> {
    let n = distance.len();
    let mut label: Vec<usize> = (0..n).collect();
    let find = |label: &mut Vec<usize>, mut i: usize| {
        while label[i] != i {
            label[i] = label[label[i]];
            i = label[i];
        }
        i
    };
    for i in 0..n {
        for j in (i + 1)..n {
            if distance[i][j] <= radius {
                let a = find(&mut label, i);
                let b = find(&mut label, j);
                if a != b {
                    label[a] = b;
                }
            }
        }
    }
    (0..n).map(|i| find(&mut label, i)).collect()
}

fn ring_share(profile: (usize, usize, usize)) -> [f64; 3] {
    let total = (profile.0 + profile.1 + profile.2).max(1) as f64;
    [
        profile.0 as f64 / total,
        profile.1 as f64 / total,
        profile.2 as f64 / total,
    ]
}

fn share_l1(left: [f64; 3], right: [f64; 3]) -> f64 {
    (0..3).map(|i| (left[i] - right[i]).abs()).sum()
}

fn main() {
    let isomers: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);
    let lj75_ico = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
    let lj75_marks = load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz"));
    let potential = PairPotential::new(75, PairKind::LennardJones, 40.0);
    let (ico_energy, ico_state) = quench(&potential, lj75_ico.view(), 600);
    let (marks_energy, marks_state) = quench(&potential, lj75_marks.view(), 600);

    // Index 0 is ico, index 1 is Marks, then the shelf sample.
    let mut states: Vec<Array1<f64>> = vec![ico_state.clone(), marks_state.clone()];
    let mut energies: Vec<f64> = vec![ico_energy, marks_energy];
    let mut rng = StdRng::seed_from_u64(20260821);
    for index in 0..isomers {
        let sigma = 0.10 + 0.04 * ((index % 14) as f64);
        let start = perturbed(ico_state.view(), sigma, &mut rng);
        let (energy, relaxed) = quench(&potential, start.view(), 1200);
        // The shelf the live book holds: icosahedral minima, not the
        // amorphous floor a large perturbation reaches.
        if !energy.is_finite() || energy > ico_energy + 8.0 {
            continue;
        }
        states.push(relaxed);
        energies.push(energy);
    }
    let n = states.len();
    println!("{{\"kind\":\"separator_sample\",\"n\":{n},\"ico\":{ico_energy:.6},\"marks\":{marks_energy:.6}}}");

    // One shared codebook over every structure, the live PackingBook form.
    let mut book = PackingBook::default();
    let mut histograms: Vec<Vec<f64>> = Vec::with_capacity(n);
    for state in &states {
        book.observe(state.as_slice().expect("state is contiguous"));
    }
    for state in &states {
        histograms.push(
            book.histogram(state.as_slice().expect("state is contiguous"))
                .expect("histogram exists"),
        );
    }
    let mut distance = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            distance[i][j] = packing_distance(&histograms[i], &histograms[j]);
        }
    }

    // Nearest-reference assignment: every shelf isomer should sit nearer ico.
    let mut misassigned = 0usize;
    for i in 2..n {
        if distance[i][1] < distance[i][0] {
            misassigned += 1;
        }
    }
    println!(
        "{{\"kind\":\"separator_nearest_reference\",\"shelf\":{},\"nearer_marks\":{misassigned},\"ico_marks\":{:.4}}}",
        n - 2,
        distance[0][1]
    );

    for radius in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45] {
        let label = components(&distance, radius);
        let mut distinct: Vec<usize> = label.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let marks_alone = label.iter().filter(|&&c| c == label[1]).count() == 1;
        let ico_with_marks = label[0] == label[1];
        println!(
            "{{\"kind\":\"separator_linkage\",\"radius\":{radius:.2},\"components\":{},\"marks_alone\":{marks_alone},\"ico_with_marks\":{ico_with_marks},\"ico_component\":{}}}",
            distinct.len(),
            label.iter().filter(|&&c| c == label[0]).count()
        );
    }

    // Franzblau rings. Raw counts, then the normalized share the packing
    // signature actually lives in.
    let profiles: Vec<Option<(usize, usize, usize)>> = states
        .iter()
        .map(|state| occupancy_ring_profile(state.as_slice().expect("state is contiguous")))
        .collect();
    let (Some(ico_profile), Some(marks_profile)) = (profiles[0], profiles[1]) else {
        println!("{{\"kind\":\"separator_rings\",\"available\":false}}");
        return;
    };
    let ico_share = ring_share(ico_profile);
    let marks_share = ring_share(marks_profile);
    println!(
        "{{\"kind\":\"separator_rings_reference\",\"ico\":[{},{},{}],\"marks\":[{},{},{}],\"ico_share\":[{:.4},{:.4},{:.4}],\"marks_share\":[{:.4},{:.4},{:.4}],\"share_l1\":{:.4}}}",
        ico_profile.0, ico_profile.1, ico_profile.2,
        marks_profile.0, marks_profile.1, marks_profile.2,
        ico_share[0], ico_share[1], ico_share[2],
        marks_share[0], marks_share[1], marks_share[2],
        share_l1(ico_share, marks_share)
    );
    let mut to_ico: Vec<f64> = Vec::new();
    let mut nearer_marks_rings = 0usize;
    let mut novelty_nonzero = 0usize;
    for profile in profiles.iter().skip(2).flatten() {
        let share = ring_share(*profile);
        let d_ico = share_l1(share, ico_share);
        let d_marks = share_l1(share, marks_share);
        to_ico.push(d_ico);
        if d_marks < d_ico {
            nearer_marks_rings += 1;
        }
        if anneal_core::catalog::ring_novelty(ico_profile, *profile) > 0 {
            novelty_nonzero += 1;
        }
    }
    to_ico.sort_by(|a, b| a.partial_cmp(b).expect("share L1 is finite"));
    println!(
        "{{\"kind\":\"separator_rings_shelf\",\"n\":{},\"share_l1_median\":{:.4},\"share_l1_max\":{:.4},\"nearer_marks\":{nearer_marks_rings},\"ring_novelty_nonzero\":{novelty_nonzero}}}",
        to_ico.len(),
        to_ico.get(to_ico.len() / 2).copied().unwrap_or(f64::NAN),
        to_ico.last().copied().unwrap_or(f64::NAN)
    );
}
