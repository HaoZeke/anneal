//! How often an independent start quenches into each LJ75 funnel.
//!
//!     catchment_rate [STARTS] [SEED]
//!
//! Wales and Doye report the decahedral global minimum in four of 100 basin
//! hopping runs and state that a successful run requires an initial geometry
//! inside the decahedral catchment, the others producing the lowest
//! icosahedral minimum after which the decahedron is not found
//! (*J. Phys. Chem. A* **1997**, *101*, 5111,
//! <https://doi.org/10.1021/jp970984n>). That makes the catchment rate of the
//! start distribution, not the escape move, the quantity that decides whether
//! an ensemble sees Marks.
//!
//! This measures it directly: uniform starts in a sphere, one quench each,
//! classified against the two sealed fixtures by the same single linkage the
//! Leave accept uses. No hopping, no bias, no bank.

use anneal_core::catalog::{PACKING_LINK, PackingBook, packing_link_labels};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

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
    let (energy, relaxed, _) = opt.minimize(x, steps, |v: ArrayView1<f64>| {
        Some(potential.value_and_gradient(v))
    });
    (energy, relaxed)
}

/// 0 ico, 1 Marks, 2 neither, by single linkage on a shared codebook.
fn community_of(ico: &[f64], marks: &[f64], trial: &[f64]) -> usize {
    let mut book = PackingBook::default();
    for state in [ico, marks, trial] {
        book.observe(state);
    }
    let histograms: Vec<Vec<f64>> = [ico, marks, trial]
        .into_iter()
        .filter_map(|state| book.histogram(state))
        .collect();
    if histograms.len() != 3 {
        return 2;
    }
    let labels = packing_link_labels(&histograms, PACKING_LINK);
    if labels[2] == labels[0] {
        0
    } else if labels[2] == labels[1] {
        1
    } else {
        2
    }
}

fn main() {
    let starts: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);
    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(20260821);
    let ico = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
    let marks = load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz"));
    let potential = PairPotential::new(75, PairKind::LennardJones, 40.0);
    let (ico_energy, ico_min) = quench(&potential, ico.view(), 2000);
    let (marks_energy, marks_min) = quench(&potential, marks.view(), 2000);
    let ico_slice = ico_min.as_slice().expect("contiguous").to_vec();
    let marks_slice = marks_min.as_slice().expect("contiguous").to_vec();
    println!(
        "{{\"kind\":\"catchment_floors\",\"ico\":{ico_energy:.6},\"marks\":{marks_energy:.6},\"starts\":{starts}}}"
    );

    // Wales and Doye confine the random start to a sphere of radius 5.5 for
    // LJ75; the container is theirs, the classification is ours.
    let radius = 5.5;
    let mut rng = StdRng::seed_from_u64(seed);
    let uniform = Uniform::new_inclusive(-radius, radius).expect("valid range");
    let mut counts = [0usize; 3];
    let mut best = f64::INFINITY;
    let mut below_ico = 0usize;
    for index in 0..starts {
        let mut start = Array1::zeros(ico.len());
        for atom in 0..ico.len() / 3 {
            loop {
                let x = uniform.sample(&mut rng);
                let y = uniform.sample(&mut rng);
                let z = uniform.sample(&mut rng);
                if x * x + y * y + z * z <= radius * radius {
                    start[3 * atom] = x;
                    start[3 * atom + 1] = y;
                    start[3 * atom + 2] = z;
                    break;
                }
            }
        }
        let (energy, relaxed) = quench(&potential, start.view(), 4000);
        if !energy.is_finite() {
            continue;
        }
        let community = relaxed
            .as_slice()
            .map_or(2, |slice| community_of(&ico_slice, &marks_slice, slice));
        counts[community] += 1;
        if energy < best {
            best = energy;
        }
        if energy < ico_energy - 1e-6 {
            below_ico += 1;
        }
        if community == 1 || energy < ico_energy - 1e-6 {
            println!(
                "{{\"kind\":\"catchment_hit\",\"index\":{index},\"energy\":{energy:.6},\"community\":{community}}}"
            );
        }
        if index % 25 == 24 {
            println!(
                "{{\"kind\":\"catchment_progress\",\"seen\":{},\"ico\":{},\"marks\":{},\"other\":{},\"best\":{best:.6}}}",
                index + 1,
                counts[0],
                counts[1],
                counts[2]
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
    println!(
        "{{\"kind\":\"catchment_rate\",\"starts\":{starts},\"ico\":{},\"marks\":{},\"other\":{},\"below_ico\":{below_ico},\"best\":{best:.6}}}",
        counts[0], counts[1], counts[2]
    );
}
