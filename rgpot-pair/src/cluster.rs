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
    #[cfg(test)]
    fn rgpot_pair_test_have_avx2_fma() -> i32;
    #[cfg(test)]
    fn rgpot_pair_test_morse_scalar(
        x: *const f64,
        n: usize,
        rho: f64,
        energy: *mut f64,
        gradient: *mut f64,
    );
    #[cfg(test)]
    fn rgpot_pair_test_morse_avx2(
        x: *const f64,
        n: usize,
        rho: f64,
        energy: *mut f64,
        gradient: *mut f64,
    );
    #[cfg(test)]
    fn rgpot_pair_test_lj_scalar(x: *const f64, n: usize, energy: *mut f64, gradient: *mut f64);
    #[cfg(test)]
    fn rgpot_pair_test_lj_avx2(x: *const f64, n: usize, energy: *mut f64, gradient: *mut f64);
    #[cfg(test)]
    fn rgpot_pair_test_gupta_scalar(
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
    #[cfg(test)]
    fn rgpot_pair_test_gupta_avx2(
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
#[allow(
    clippy::too_many_arguments,
    reason = "public rgpot-pair API; its parameters mirror the rgpot_gupta C tables"
)]
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

    fn assert_close(left: f64, right: f64) {
        let tolerance = 8.0 * f64::EPSILON * (1.0 + left.abs().max(right.abs()));
        assert!(
            (left - right).abs() <= tolerance,
            "{left:.17e} differs from {right:.17e} by more than {tolerance:.3e}"
        );
    }

    fn compare_outputs(
        scalar_energy: f64,
        scalar_gradient: &[f64],
        avx2_energy: f64,
        avx2_gradient: &[f64],
    ) {
        assert_close(scalar_energy, avx2_energy);
        assert_eq!(scalar_gradient.len(), avx2_gradient.len());
        for (&scalar, &avx2) in scalar_gradient.iter().zip(avx2_gradient) {
            assert_close(scalar, avx2);
        }
    }

    #[test]
    fn scalar_and_avx2_cluster_paths_agree() {
        if unsafe { rgpot_pair_test_have_avx2_fma() } == 0 {
            eprintln!("AVX2/FMA unavailable; path comparison skipped");
            return;
        }

        for n in [2, 9, 31] {
            let positions: Vec<f64> = (0..n)
                .flat_map(|i| {
                    let i = i as f64;
                    [1.17 * i, 0.31 * i * i + 0.2, (i * 0.73).sin()]
                })
                .collect();
            let mut scalar_gradient = vec![0.0; 3 * n];
            let mut avx2_gradient = vec![0.0; 3 * n];
            let mut scalar_energy = 0.0;
            let mut avx2_energy = 0.0;

            unsafe {
                rgpot_pair_test_morse_scalar(
                    positions.as_ptr(),
                    n,
                    6.0,
                    &mut scalar_energy,
                    scalar_gradient.as_mut_ptr(),
                );
                rgpot_pair_test_morse_avx2(
                    positions.as_ptr(),
                    n,
                    6.0,
                    &mut avx2_energy,
                    avx2_gradient.as_mut_ptr(),
                );
            }
            compare_outputs(scalar_energy, &scalar_gradient, avx2_energy, &avx2_gradient);

            scalar_gradient.fill(0.0);
            avx2_gradient.fill(0.0);
            unsafe {
                rgpot_pair_test_lj_scalar(
                    positions.as_ptr(),
                    n,
                    &mut scalar_energy,
                    scalar_gradient.as_mut_ptr(),
                );
                rgpot_pair_test_lj_avx2(
                    positions.as_ptr(),
                    n,
                    &mut avx2_energy,
                    avx2_gradient.as_mut_ptr(),
                );
            }
            compare_outputs(scalar_energy, &scalar_gradient, avx2_energy, &avx2_gradient);

            let slot = vec![0; n];
            let a = [0.2061];
            let xi = [1.790];
            let p = [10.229];
            let q = [4.036];
            let r0 = [2.884];
            scalar_gradient.fill(0.0);
            avx2_gradient.fill(0.0);
            unsafe {
                rgpot_pair_test_gupta_scalar(
                    positions.as_ptr(),
                    n,
                    slot.as_ptr(),
                    1,
                    a.as_ptr(),
                    xi.as_ptr(),
                    p.as_ptr(),
                    q.as_ptr(),
                    r0.as_ptr(),
                    &mut scalar_energy,
                    scalar_gradient.as_mut_ptr(),
                );
                rgpot_pair_test_gupta_avx2(
                    positions.as_ptr(),
                    n,
                    slot.as_ptr(),
                    1,
                    a.as_ptr(),
                    xi.as_ptr(),
                    p.as_ptr(),
                    q.as_ptr(),
                    r0.as_ptr(),
                    &mut avx2_energy,
                    avx2_gradient.as_mut_ptr(),
                );
            }
            compare_outputs(scalar_energy, &scalar_gradient, avx2_energy, &avx2_gradient);
        }
    }

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
