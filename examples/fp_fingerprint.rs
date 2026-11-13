// Copyright (c) anneal developers
//! Quench-to-quench fingerprint distances.
//!
//! ```text
//! fp_fingerprint <system> <seeds>
//! ```
//!
//! `system` is `au55`, `lj38`, `lj75` or `tip4p15`. Each seed is a random
//! start, quenched, then shaken and quenched again. The second quench of
//! the same start is one minimum seen twice. Its fingerprint distance is
//! the width of that minimum. A pair of starts whose energies differ by
//! more than that width are distinct minima. The mean SOAP spectrum is
//! the one the taboo compares at distance 0.02. The other three spectra
//! are candidates, and each is called a match inside its own re-quench
//! width.

use anneal_core::methods::cluster_hopping::fingerprint::{
    l2, mean_soap, pair_histogram, points_of, radial_spectrum, requench_threshold, sorted_soap,
};
use anneal_core::methods::cluster_hopping::random_cluster;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{GOLD_Z, Gupta, PairPotential, Tip4pCluster, random_tip4p_cluster};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

struct Minimum {
    energy: f64,
    soap: Vec<f64>,
    radial: Vec<f64>,
    hist: Vec<f64>,
    sorted: Vec<f64>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let system = args.next().unwrap_or_else(|| "au55".to_string());
    let seeds: usize = args.next().and_then(|text| text.parse().ok()).unwrap_or(16);
    let rows = match system.as_str() {
        "au55" => survey_atomic(&system, 55, seeds, Kind::Gold),
        "lj38" => survey_atomic(&system, 38, seeds, Kind::LennardJones),
        "lj75" => survey_atomic(&system, 75, seeds, Kind::LennardJones),
        "tip4p15" => survey_tip4p(seeds),
        other => {
            eprintln!("unknown system {other}");
            std::process::exit(2);
        }
    };
    println!("{rows}");
}

enum Kind {
    LennardJones,
    Gold,
}

fn survey_atomic(system: &str, n: usize, seeds: usize, kind: Kind) -> String {
    let mut minima = Vec::new();
    let mut same_energy = Vec::new();
    let mut same_soap = Vec::new();
    let mut same_radial = Vec::new();
    let mut same_hist = Vec::new();
    let mut same_sorted = Vec::new();
    for seed in 0..seeds {
        let mut rng = StdRng::seed_from_u64(2_000 + seed as u64);
        let start = random_cluster(n, 0.5, 0.8, &mut rng);
        let (energy, quenched) = quench_atomic(start.view(), &kind);
        if !energy.is_finite() {
            continue;
        }
        let species = species_of(n, &kind);
        let Some(first) = fingerprints(quenched.view(), Some(&species), None) else {
            continue;
        };
        let shaken = jitter(&quenched, 0.02, &mut rng);
        let (again, requenched) = quench_atomic(shaken.view(), &kind);
        if again.is_finite() && same_geometry(quenched.view(), requenched.view()) {
            if let Some(second) = fingerprints(requenched.view(), Some(&species), None) {
                same_energy.push((again - energy).abs());
                same_soap.push(l2(&first.soap, &second.soap));
                same_radial.push(l2(&first.radial, &second.radial));
                same_hist.push(l2(&first.hist, &second.hist));
                same_sorted.push(l2(&first.sorted, &second.sorted));
            }
        }
        minima.push(Minimum { energy, ..first });
    }
    summarise(
        system,
        &minima,
        &same_energy,
        &same_soap,
        &same_radial,
        &same_hist,
        &same_sorted,
    )
}

fn survey_tip4p(seeds: usize) -> String {
    let mut minima = Vec::new();
    let mut same_energy = Vec::new();
    let mut same_soap = Vec::new();
    let mut same_radial = Vec::new();
    let mut same_hist = Vec::new();
    let mut same_sorted = Vec::new();
    for seed in 0..seeds {
        let mut rng = StdRng::seed_from_u64(2_000 + seed as u64);
        let start = random_tip4p_cluster(15, &mut rng);
        let (energy, quenched) = quench_tip4p(start.view());
        if !energy.is_finite() {
            continue;
        }
        let Some(first) = fingerprints(quenched.view(), None, Some(15)) else {
            continue;
        };
        let shaken = jitter_rigid(&quenched, 15, 0.05, &mut rng);
        let (again, requenched) = quench_tip4p(shaken.view());
        let centres = points_of(quenched.view(), Some(15));
        let centres_again = points_of(requenched.view(), Some(15));
        if again.is_finite() && same_geometry(centres.view(), centres_again.view()) {
            if let Some(second) = fingerprints(requenched.view(), None, Some(15)) {
                same_energy.push((again - energy).abs());
                same_soap.push(l2(&first.soap, &second.soap));
                same_radial.push(l2(&first.radial, &second.radial));
                same_hist.push(l2(&first.hist, &second.hist));
                same_sorted.push(l2(&first.sorted, &second.sorted));
            }
        }
        minima.push(Minimum { energy, ..first });
    }
    summarise(
        "tip4p15",
        &minima,
        &same_energy,
        &same_soap,
        &same_radial,
        &same_hist,
        &same_sorted,
    )
}

fn fingerprints(
    coords: ArrayView1<f64>,
    species: Option<&[u32]>,
    rigid: Option<usize>,
) -> Option<Minimum> {
    let points = points_of(coords, rigid);
    Some(Minimum {
        energy: 0.0,
        soap: mean_soap(points.view(), species)?,
        radial: radial_spectrum(points.view())?,
        hist: pair_histogram(points.view(), 32)?,
        sorted: sorted_soap(points.view(), species)?,
    })
}

fn summarise(
    system: &str,
    minima: &[Minimum],
    same_energy: &[f64],
    same_soap: &[f64],
    same_radial: &[f64],
    same_hist: &[f64],
    same_sorted: &[f64],
) -> String {
    let energy_width = requench_threshold(same_energy);
    let distinct = |gap: f64| gap > energy_width && gap > 100.0 * energy_width.max(0.0);
    let mut pairs_01 = 0usize;
    let mut pairs_05 = 0usize;
    let mut soap_002_01 = 0usize;
    let mut soap_002_05 = 0usize;
    let mut soap_cal_01 = 0usize;
    let mut radial_cal_01 = 0usize;
    let mut hist_cal_01 = 0usize;
    let mut sorted_cal_01 = 0usize;
    let mut min_soap = f64::MAX;
    let mut min_radial = f64::MAX;
    let mut min_hist = f64::MAX;
    let mut min_sorted = f64::MAX;
    let soap_width = requench_threshold(same_soap);
    let radial_width = requench_threshold(same_radial);
    let hist_width = requench_threshold(same_hist);
    let sorted_width = requench_threshold(same_sorted);
    for i in 0..minima.len() {
        for j in (i + 1)..minima.len() {
            let gap = (minima[i].energy - minima[j].energy).abs();
            if !distinct(gap) {
                continue;
            }
            let soap = l2(&minima[i].soap, &minima[j].soap);
            let radial = l2(&minima[i].radial, &minima[j].radial);
            let hist = l2(&minima[i].hist, &minima[j].hist);
            let sorted = l2(&minima[i].sorted, &minima[j].sorted);
            if gap > 0.1 {
                pairs_01 += 1;
                soap_002_01 += usize::from(soap < 0.02);
                soap_cal_01 += usize::from(soap <= soap_width);
                radial_cal_01 += usize::from(radial <= radial_width);
                hist_cal_01 += usize::from(hist <= hist_width);
                sorted_cal_01 += usize::from(sorted <= sorted_width);
                min_soap = min_soap.min(soap);
                min_radial = min_radial.min(radial);
                min_hist = min_hist.min(hist);
                min_sorted = min_sorted.min(sorted);
            }
            if gap > 0.5 {
                pairs_05 += 1;
                soap_002_05 += usize::from(soap < 0.02);
            }
        }
    }
    let finite = |value: f64| {
        if pairs_01 == 0 {
            "nan".to_string()
        } else {
            format!("{value:.6e}")
        }
    };
    format!(
        "{system},{n},{requench},{energy_width:.6e},{soap_width:.6e},{radial_width:.6e},{hist_width:.6e},{sorted_width:.6e},{pairs_01},{soap_002_01},{soap_cal_01},{radial_cal_01},{hist_cal_01},{sorted_cal_01},{pairs_05},{soap_002_05},{},{},{},{}",
        finite(min_soap),
        finite(min_radial),
        finite(min_hist),
        finite(min_sorted),
        n = minima.len(),
        requench = same_energy.len(),
    )
}

fn species_of(n: usize, kind: &Kind) -> Vec<u32> {
    match kind {
        Kind::LennardJones => vec![0; n],
        Kind::Gold => vec![GOLD_Z; n],
    }
}

fn quench_atomic(start: ArrayView1<f64>, kind: &Kind) -> (f64, Array1<f64>) {
    match kind {
        Kind::LennardJones => {
            let potential = PairPotential::lennard_jones(start.len() / 3);
            relax(start, |point| potential.value_and_gradient(point))
        }
        Kind::Gold => {
            let potential = Gupta::copper_gold(0, start.len() / 3);
            relax(start, |point| potential.value_and_gradient(point))
        }
    }
}

fn quench_tip4p(start: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let potential = Tip4pCluster::new(15);
    relax(start, |point| potential.value_and_gradient(point))
}

fn relax<F>(start: ArrayView1<f64>, mut energy: F) -> (f64, Array1<f64>)
where
    F: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let mut optimiser = WarmLbfgs::default();
    optimiser.gtol = 1e-8;
    let (value, point, _) = optimiser.minimize(start, 250, |point| {
        let (value, gradient) = energy(point);
        value.is_finite().then_some((value, gradient))
    });
    (value, point)
}

fn same_geometry(left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
    let n = left.len() / 3;
    if n < 2 || left.len() != right.len() {
        return false;
    }
    let mut shortest = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = 0.0;
            for axis in 0..3 {
                let delta = left[3 * i + axis] - left[3 * j + axis];
                d += delta * delta;
            }
            shortest = shortest.min(d.sqrt());
        }
    }
    if shortest <= 1e-8 {
        return false;
    }
    let centre = |coords: ArrayView1<f64>| {
        let mut c = [0.0; 3];
        for i in 0..n {
            for axis in 0..3 {
                c[axis] += coords[3 * i + axis];
            }
        }
        for value in &mut c {
            *value /= n as f64;
        }
        c
    };
    let left_c = centre(left);
    let right_c = centre(right);
    let mut residual = 0.0;
    for i in 0..n {
        for axis in 0..3 {
            let delta = (left[3 * i + axis] - left_c[axis]) - (right[3 * i + axis] - right_c[axis]);
            residual += delta * delta;
        }
    }
    (residual / n as f64).sqrt() < 0.05 * shortest
}

fn jitter(coords: &Array1<f64>, amplitude: f64, rng: &mut StdRng) -> Array1<f64> {
    coords.mapv(|value| value + (rng.random::<f64>() - 0.5) * 2.0 * amplitude)
}

fn jitter_rigid(
    coords: &Array1<f64>,
    bodies: usize,
    amplitude: f64,
    rng: &mut StdRng,
) -> Array1<f64> {
    let mut shaken = coords.clone();
    for value in shaken.iter_mut().take(3 * bodies) {
        *value += (rng.random::<f64>() - 0.5) * 2.0 * amplitude;
    }
    shaken
}
