//! Fingerprint taboo and the energy spread of the structures already kept.
//!
//! A proposal whose SOAP spectrum matches one already stored is refused
//! unless it is lower than that structure. The spread of the stored
//! energies is the temperature of a later hop. No packing is named.

use ndarray::ArrayView1;

use crate::soap::{SoapSpec, local_spectra_z};

struct Seat {
    energy: f64,
    soap: Vec<f64>,
}

/// Structures already accepted, keyed by their SOAP spectrum.
#[derive(Default)]
pub struct FingerprintTaboo {
    seats: Vec<Seat>,
}

impl FingerprintTaboo {
    /// Standard deviation of the stored energies, once three exist.
    pub fn spread(&self) -> Option<f64> {
        if self.seats.len() < 3 {
            return None;
        }
        let mean = self.seats.iter().map(|seat| seat.energy).sum::<f64>() / self.seats.len() as f64;
        let var = self
            .seats
            .iter()
            .map(|seat| {
                let delta = seat.energy - mean;
                delta * delta
            })
            .sum::<f64>()
            / self.seats.len() as f64;
        Some(var.sqrt())
    }

    /// True when `coords` matches a stored spectrum and is not lower.
    pub fn forbids(&self, energy: f64, coords: ArrayView1<f64>, species: &[u32]) -> bool {
        let Some(soap) = spectrum(coords, species) else {
            return false;
        };
        self.seats
            .iter()
            .any(|seat| soap_distance(&seat.soap, &soap) < 0.02 && energy >= seat.energy - 1e-6)
    }

    /// Records a structure that was accepted.
    pub fn remember(&mut self, energy: f64, coords: ArrayView1<f64>, species: &[u32]) {
        let Some(soap) = spectrum(coords, species) else {
            return;
        };
        if let Some(seat) = self
            .seats
            .iter_mut()
            .find(|seat| soap_distance(&seat.soap, &soap) < 0.02)
        {
            if energy < seat.energy {
                seat.energy = energy;
                seat.soap = soap;
            }
            return;
        }
        if self.seats.len() >= 24 {
            if let Some(worst) = self
                .seats
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.energy.total_cmp(&right.energy))
                .map(|(index, _)| index)
            {
                self.seats.remove(worst);
            }
        }
        self.seats.push(Seat { energy, soap });
    }
}

fn spectrum(coords: ArrayView1<f64>, species: &[u32]) -> Option<Vec<f64>> {
    let n = coords.len() / 3;
    if n < 2 || species.len() != n {
        return None;
    }
    let spec = SoapSpec {
        n_max: 3,
        l_max: 3,
        rcut_nn: 6.0,
    };
    let local = local_spectra_z(coords, spec, Some(species));
    if local.nrows() == 0 || local.ncols() == 0 {
        return None;
    }
    let mut mean = vec![0.0; local.ncols()];
    for row in 0..local.nrows() {
        for col in 0..local.ncols() {
            mean[col] += local[[row, col]];
        }
    }
    let scale = local.nrows() as f64;
    for value in &mut mean {
        *value /= scale;
    }
    Some(mean)
}

fn soap_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn a_repeated_fingerprint_is_refused_unless_it_is_lower() {
        let species = [29u32, 29, 79, 79];
        let coords = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 2.6, 0.0, 0.0, 1.3, 2.4, 0.1, 1.2, 0.4, 2.7,
        ]);
        let mut taboo = FingerprintTaboo::default();
        taboo.remember(-1.0, coords.view(), &species);
        assert!(taboo.forbids(-0.5, coords.view(), &species));
        assert!(!taboo.forbids(-1.5, coords.view(), &species));
    }

    #[test]
    fn a_plane_move_keeps_distances_inside_the_moved_set() {
        let x = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.2, 1.3, 0.1, 2.4, 0.4, 0.2, 1.1, 2.2, 0.3, 0.4, 0.5,
            1.6,
        ]);
        let n = 6;
        let mut moved = false;
        for seed in 0..16 {
            let mut rng = StdRng::seed_from_u64(seed);
            let y = super::super::moves::plane_rearrangement(x.view(), n, &mut rng);
            assert_eq!(y.len(), x.len());
            if x.iter().zip(y.iter()).any(|(a, b)| (a - b).abs() > 1e-6) {
                moved = true;
                break;
            }
        }
        assert!(moved, "the plane move left every atom where it was");
    }
}
