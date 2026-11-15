// Copyright (c) anneal developers
//! The one-sided covalent O-H wall shared by the (H2O)m GFN2 drivers.
//!
//! Atoms are ordered O, H, H per molecule. Each covalent O-H distance
//! beyond the onset pays `stiffness * (r - onset)^2` in eV; inside the
//! onset the wall is zero.

pub(crate) const OH_WALL_ONSET_A: f64 = 1.15;
pub(crate) const OH_WALL_EV_PER_A2: f64 = 200.0;

pub(crate) fn oh_wall(coordinates: &[f64], n_molecules: usize) -> (f64, Vec<f64>) {
    let mut energy = 0.0;
    let mut gradient = vec![0.0; coordinates.len()];
    if coordinates.len() != 9 * n_molecules {
        return (0.0, gradient);
    }
    for molecule in 0..n_molecules {
        let oxygen = 9 * molecule;
        for hydrogen in [oxygen + 3, oxygen + 6] {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = coordinates[hydrogen + axis] - coordinates[oxygen + axis];
                square += delta * delta;
            }
            let distance = square.sqrt();
            if !distance.is_finite() || distance <= OH_WALL_ONSET_A || distance < 1e-12 {
                continue;
            }
            let excess = distance - OH_WALL_ONSET_A;
            energy += OH_WALL_EV_PER_A2 * excess * excess;
            let slope = 2.0 * OH_WALL_EV_PER_A2 * excess / distance;
            for axis in 0..3 {
                let delta = coordinates[hydrogen + axis] - coordinates[oxygen + axis];
                gradient[oxygen + axis] -= slope * delta;
                gradient[hydrogen + axis] += slope * delta;
            }
        }
    }
    (energy, gradient)
}
