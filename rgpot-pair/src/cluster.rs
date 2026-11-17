//! All-pairs cluster kernels.
//!
//! These are the Cambridge sums: every pair, no cutoff. The walk and the
//! rounding live in the C sources next to this crate. Callers pass one
//! contiguous coordinate buffer.

unsafe extern "C" {
    fn rgpot_cambridge_morse(
        x: *const f64,
        n: usize,
        rho: f64,
        energy: *mut f64,
        gradient: *mut f64,
    );
    fn rgpot_cluster_lj(x: *const f64, n: usize, energy: *mut f64, gradient: *mut f64);
    fn rgpot_gupta(
        x: *const f64,
        n: usize,
        slot: *const u8,
        n_species: usize,
        a: *const f64,
        xi: *const f64,
        p: *const f64,
        q: *const f64,
        r0: *const f64,
        energy: *mut f64,
        gradient: *mut f64,
    );
}

/// Cambridge Morse energy and gradient. `positions` is `3n` coordinates.
pub fn cambridge_morse(positions: &[f64], rho: f64, energy: &mut f64, gradient: &mut [f64]) {
    let n = positions.len() / 3;
    debug_assert_eq!(positions.len(), 3 * n);
    debug_assert_eq!(gradient.len(), 3 * n);
    unsafe {
        rgpot_cambridge_morse(positions.as_ptr(), n, rho, energy, gradient.as_mut_ptr());
    }
}

/// Reduced Lennard-Jones energy and gradient. `positions` is `3n` coordinates.
pub fn cluster_lj(positions: &[f64], energy: &mut f64, gradient: &mut [f64]) {
    let n = positions.len() / 3;
    debug_assert_eq!(positions.len(), 3 * n);
    debug_assert_eq!(gradient.len(), 3 * n);
    unsafe {
        rgpot_cluster_lj(positions.as_ptr(), n, energy, gradient.as_mut_ptr());
    }
}

/// Cleri–Rosato Gupta energy and gradient.
///
/// `slot` is one species index per atom. The parameter tables are
/// row-major `n_species * n_species`, indexed as `slot[i] * n_species + slot[j]`.
pub fn gupta(
    positions: &[f64],
    slot: &[u8],
    n_species: usize,
    a: &[f64],
    xi: &[f64],
    p: &[f64],
    q: &[f64],
    r0: &[f64],
    energy: &mut f64,
    gradient: &mut [f64],
) {
    let n = slot.len();
    debug_assert_eq!(positions.len(), 3 * n);
    debug_assert_eq!(gradient.len(), 3 * n);
    unsafe {
        rgpot_gupta(
            positions.as_ptr(),
            n,
            slot.as_ptr(),
            n_species,
            a.as_ptr(),
            xi.as_ptr(),
            p.as_ptr(),
            q.as_ptr(),
            r0.as_ptr(),
            energy,
            gradient.as_mut_ptr(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morse_dimer_at_equilibrium_is_minus_one() {
        let x = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut energy = 0.0;
        let mut gradient = [0.0; 6];
        cambridge_morse(&x, 6.0, &mut energy, &mut gradient);
        assert_eq!(energy, -1.0);
        assert_eq!(gradient, [0.0; 6]);
    }

    #[test]
    fn a_nan_coordinate_returns_nan_energy_and_forces() {
        let x = [f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 1.1, 0.0];
        let mut energy = 0.0;
        let mut gradient = [0.0; 9];
        cambridge_morse(&x, 6.0, &mut energy, &mut gradient);
        assert!(energy.is_nan());
        assert!(gradient.iter().all(|component| component.is_nan()));

        energy = 0.0;
        gradient = [0.0; 9];
        cluster_lj(&x, &mut energy, &mut gradient);
        assert!(energy.is_nan());
        assert!(gradient.iter().all(|component| component.is_nan()));

        energy = 0.0;
        gradient = [0.0; 9];
        gupta(
            &x,
            &[0, 0, 0],
            1,
            &[0.2061],
            &[1.790],
            &[10.229],
            &[4.036],
            &[2.884],
            &mut energy,
            &mut gradient,
        );
        assert!(energy.is_nan());
        assert!(gradient.iter().all(|component| component.is_nan()));
    }

    #[test]
    fn lj_dimer_at_one_has_force_24() {
        let x = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut energy = 0.0;
        let mut gradient = [0.0; 6];
        cluster_lj(&x, &mut energy, &mut gradient);
        assert_eq!(energy, 0.0);
        assert_eq!(gradient[0], 24.0);
        assert_eq!(gradient[3], -24.0);
    }
}
