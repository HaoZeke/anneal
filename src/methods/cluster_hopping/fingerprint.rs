// Copyright (c) anneal developers
//! Structural fingerprints of a quenched cluster.
//!
//! The mean of the per-atom SOAP vectors is the spectrum the taboo
//! compares at a fixed distance of 0.02. A mean washes out which atom
//! carries an environment, so two minima can sit inside that distance
//! while their energies differ. The alternatives here keep the atom
//! resolution or the pair distances, and a match is decided by the
//! distances of re-quenches of one minimum rather than by a fixed cutoff.

use ndarray::{Array1, ArrayView1};

use crate::soap::{SoapSpec, local_spectra_z};

/// SOAP settings of the mean spectrum the taboo stores.
pub fn taboo_soap() -> SoapSpec {
    SoapSpec {
        n_max: 3,
        l_max: 3,
        rcut_nn: 6.0,
    }
}

/// Mean per-atom SOAP, the spectrum [`super::general`] stores.
pub fn mean_soap(coords: ArrayView1<f64>, species: Option<&[u32]>) -> Option<Vec<f64>> {
    let local = local_rows(coords, species)?;
    let n = local.len();
    let dim = local[0].len();
    let mut mean = vec![0.0; dim];
    for row in &local {
        for (bin, value) in mean.iter_mut().zip(row) {
            *bin += *value;
        }
    }
    let scale = n as f64;
    for value in &mut mean {
        *value /= scale;
    }
    Some(mean)
}

/// Per-atom SOAP rows sorted lexicographically, then concatenated.
///
/// Sorting removes the atom order. Two minima whose environments differ
/// stay apart even when the average of those environments does not.
pub fn sorted_soap(coords: ArrayView1<f64>, species: Option<&[u32]>) -> Option<Vec<f64>> {
    let mut local = local_rows(coords, species)?;
    local.sort_by(|left, right| {
        left.iter()
            .zip(right)
            .map(|(a, b)| a.total_cmp(b))
            .find(|order| !order.is_eq())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(local.into_iter().flatten().collect())
}

/// Sorted distances from the centroid, in units of their mean.
pub fn radial_spectrum(coords: ArrayView1<f64>) -> Option<Vec<f64>> {
    let n = atom_count(coords)?;
    let mut centre = [0.0; 3];
    for i in 0..n {
        for axis in 0..3 {
            centre[axis] += coords[3 * i + axis];
        }
    }
    let scale = n as f64;
    for value in &mut centre {
        *value /= scale;
    }
    let mut radii = Vec::with_capacity(n);
    for i in 0..n {
        radii.push(distance_to(coords, i, centre));
    }
    radii.sort_by(|left, right| left.total_cmp(right));
    let mean = radii.iter().copied().sum::<f64>() / scale;
    if !(mean > 1e-12) {
        return None;
    }
    for value in &mut radii {
        *value /= mean;
    }
    Some(radii)
}

/// Histogram of pair distances in units of the shortest pair.
///
/// `bins` covers zero to eight shortest-pair lengths. The counts are
/// normalised, so the size of the cluster does not set the scale.
pub fn pair_histogram(coords: ArrayView1<f64>, bins: usize) -> Option<Vec<f64>> {
    let n = atom_count(coords)?;
    if bins < 2 {
        return None;
    }
    let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
    let mut shortest = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            let distance = pair_distance(coords, i, j);
            if distance < shortest {
                shortest = distance;
            }
            pairs.push(distance);
        }
    }
    if !(shortest > 1e-12) {
        return None;
    }
    let mut hist = vec![0.0; bins];
    let width = 8.0 / bins as f64;
    for distance in pairs {
        let bin = ((distance / shortest) / width) as usize;
        hist[bin.min(bins - 1)] += 1.0;
    }
    let total = hist.iter().copied().sum::<f64>();
    if !(total > 0.0) {
        return None;
    }
    for value in &mut hist {
        *value /= total;
    }
    Some(hist)
}

/// Euclidean distance. Unequal lengths compare the common prefix.
pub fn l2(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

/// Largest distance in a sample of re-quenches of one minimum.
///
/// An empty sample has no measured width, so nothing is called a match.
pub fn requench_threshold(distances: &[f64]) -> f64 {
    distances.iter().copied().fold(0.0_f64, f64::max)
}

/// Centres of a rigid state, or the whole state when it is already
/// one point per atom.
pub fn points_of(coords: ArrayView1<f64>, rigid_bodies: Option<usize>) -> Array1<f64> {
    if let Some(bodies) = rigid_bodies
        && coords.len() >= 6 * bodies
    {
        return coords.slice(ndarray::s![..3 * bodies]).to_owned();
    }
    coords.to_owned()
}

fn local_rows(coords: ArrayView1<f64>, species: Option<&[u32]>) -> Option<Vec<Vec<f64>>> {
    let n = atom_count(coords)?;
    if let Some(labels) = species
        && labels.len() != n
    {
        return None;
    }
    let spec = taboo_soap();
    let local = local_spectra_z(coords, spec, species);
    if local.ncols() == 0 {
        return None;
    }
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(local.row(i).to_vec());
    }
    Some(rows)
}

fn atom_count(coords: ArrayView1<f64>) -> Option<usize> {
    let n = coords.len() / 3;
    (n >= 2 && coords.len() == 3 * n).then_some(n)
}

fn pair_distance(coords: ArrayView1<f64>, i: usize, j: usize) -> f64 {
    let mut d = 0.0;
    for axis in 0..3 {
        let delta = coords[3 * i + axis] - coords[3 * j + axis];
        d += delta * delta;
    }
    d.sqrt()
}

fn distance_to(coords: ArrayView1<f64>, i: usize, centre: [f64; 3]) -> f64 {
    let mut d = 0.0;
    for axis in 0..3 {
        let delta = coords[3 * i + axis] - centre[axis];
        d += delta * delta;
    }
    d.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_requench_threshold_is_the_widest_same_minimum_distance() {
        assert_eq!(requench_threshold(&[]), 0.0);
        assert!((requench_threshold(&[0.01, 0.4, 0.02]) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn mean_soap_calls_distinct_gold_minima_the_same_at_the_taboo_cutoff() {
        use crate::methods::warm_lbfgs::WarmLbfgs;
        use crate::potentials::{GOLD_Z, Gupta};
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let potential = Gupta::copper_gold(0, 55);
        let quench = |seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            let start = super::super::random_cluster(55, 0.5, 0.8, &mut rng);
            let mut optimiser = WarmLbfgs::default();
            optimiser.gtol = 1e-8;
            let (energy, point, _) = optimiser.minimize(start.view(), 250, |point| {
                let (energy, gradient) = potential.value_and_gradient(point);
                energy.is_finite().then_some((energy, gradient))
            });
            (energy, point)
        };
        let species = vec![GOLD_Z; 55];
        let mut found = false;
        let mut best_gap = 0.0;
        let mut best_soap = f64::MAX;
        for left in 0..6 {
            let (energy_left, point_left) = quench(2_000 + left);
            let soap_left = mean_soap(point_left.view(), Some(&species)).expect("soap");
            let hist_left = pair_histogram(point_left.view(), 32).expect("histogram");
            for right in (left + 1)..6 {
                let (energy_right, point_right) = quench(2_000 + right);
                let gap = (energy_left - energy_right).abs();
                let soap_right = mean_soap(point_right.view(), Some(&species)).expect("soap");
                let distance = l2(&soap_left, &soap_right);
                if gap > 0.5 && distance < 0.02 {
                    let hist_right = pair_histogram(point_right.view(), 32).expect("histogram");
                    let hist = l2(&hist_left, &hist_right);
                    assert!(
                        hist > 1.0e-2,
                        "pair histogram also collided, gap {gap} soap {distance} hist {hist}"
                    );
                    found = true;
                    best_gap = gap;
                    best_soap = distance;
                    break;
                }
            }
            if found {
                break;
            }
        }
        assert!(
            found,
            "no Au55 pair among seeds 2000-2005 was 0.5 eV apart inside mean-SOAP distance 0.02"
        );
        assert!(best_gap > 0.5);
        assert!(best_soap < 0.02);
    }

    #[test]
    fn a_translated_copy_matches_its_radial_spectrum_and_its_histogram() {
        let coords = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.9, 0.1]);
        let shifted = Array1::from_vec(vec![4.0, -2.0, 7.0, 5.0, -2.0, 7.0, 4.2, -1.1, 7.1]);
        let radial = radial_spectrum(coords.view()).unwrap();
        let radial_shifted = radial_spectrum(shifted.view()).unwrap();
        assert!(l2(&radial, &radial_shifted) < 1e-9);
        let hist = pair_histogram(coords.view(), 16).unwrap();
        let hist_shifted = pair_histogram(shifted.view(), 16).unwrap();
        assert!(l2(&hist, &hist_shifted) < 1e-9);
    }
}
