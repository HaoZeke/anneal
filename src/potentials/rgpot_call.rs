//! Cluster pair calls from the `rgpot-pair` crate.
//!
//! The hot Lennard-Jones and Morse loops inline `lj_unit` and
//! `morse_unit`. These wrappers cover the whole-configuration, per-atom,
//! partial, arithmetic, binary, and TIP4P entry points. Gradients are
//! derivatives of the energy.

use ndarray::{Array1, ArrayView1};

fn with_slice<R>(x: ArrayView1<f64>, body: impl FnOnce(&[f64]) -> R) -> R {
    if let Some(slice) = x.as_slice() {
        body(slice)
    } else {
        let owned = x.to_vec();
        body(&owned)
    }
}

/// Reduced Lennard-Jones, `epsilon = 1`, `sigma = 1`, no cutoff.
#[inline]
pub fn reduced_lennard_jones(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    lennard_jones(1.0, 1.0, 0.0, x)
}

/// Unshifted Lennard-Jones. `cutoff <= 0` keeps every pair.
#[inline]
pub fn lennard_jones(
    epsilon: f64,
    sigma: f64,
    cutoff: f64,
    x: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    with_slice(x, |pos| {
        let mut gradient = vec![0.0; pos.len()];
        let energy = rgpot_pair::lj_gradient(epsilon, sigma, cutoff, pos, &mut gradient);
        (energy, Array1::from_vec(gradient))
    })
}

/// Energy, `(dV/dr) / r`, and `d2V/dr2` of one Lennard-Jones pair.
#[inline]
pub fn lj_pair_terms(epsilon: f64, sigma: f64, r2: f64) -> (f64, f64, f64) {
    rgpot_pair::lj_pair(epsilon, sigma, r2)
}

/// Pair energy of each atom. The total energy is half the sum.
pub fn lennard_jones_atom_energies(epsilon: f64, sigma: f64, x: ArrayView1<f64>) -> Array1<f64> {
    with_slice(x, |pos| {
        let mut atoms = vec![0.0; pos.len() / 3];
        rgpot_pair::lj_atom_energies(epsilon, sigma, pos, &mut atoms);
        Array1::from_vec(atoms)
    })
}

/// Gradient rows of `moved` only.
pub fn lennard_jones_partial(x: ArrayView1<f64>, moved: &[usize]) -> Array1<f64> {
    with_slice(x, |pos| {
        let mut gradient = vec![0.0; pos.len()];
        rgpot_pair::lj_partial(1.0, 1.0, pos, moved, &mut gradient);
        Array1::from_vec(gradient)
    })
}

/// Arithmetic-mean diameters, one `sigma` per atom.
pub fn lennard_jones_arithmetic(
    sigma: &[f64],
    cutoff: f64,
    x: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    with_slice(x, |pos| {
        let mut gradient = vec![0.0; pos.len()];
        let energy = rgpot_pair::lj_arithmetic(sigma, cutoff, pos, &mut gradient);
        (energy, Array1::from_vec(gradient))
    })
}

/// Doye and Wales Morse at `epsilon = 1`, `re = 1`, no cutoff.
#[inline]
pub fn morse(rho: f64, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    morse_cluster(1.0, rho, 1.0, 0.0, x)
}

/// Unshifted Morse. `cutoff <= 0` keeps every pair.
#[inline]
pub fn morse_cluster(
    epsilon: f64,
    rho: f64,
    re: f64,
    cutoff: f64,
    x: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    with_slice(x, |pos| {
        let mut gradient = vec![0.0; pos.len()];
        let energy = rgpot_pair::morse_gradient(epsilon, rho, re, cutoff, pos, &mut gradient);
        (energy, Array1::from_vec(gradient))
    })
}

/// Energy, `(dV/dr) / r`, and `d2V/dr2` of one Morse pair.
#[inline]
pub fn morse_pair_terms(epsilon: f64, rho: f64, re: f64, r2: f64) -> (f64, f64, f64) {
    rgpot_pair::morse_pair(epsilon, rho, re, r2)
}

/// Binary Lennard-Jones. Species A is the first `n_a` atoms.
pub fn binary_lennard_jones(
    n_a: usize,
    epsilon: [f64; 3],
    sigma: [f64; 3],
    cutoff: f64,
    x: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    with_slice(x, |pos| {
        let mut gradient = vec![0.0; pos.len()];
        let energy = rgpot_pair::binary_lj(n_a, epsilon, sigma, cutoff, pos, &mut gradient);
        (energy, Array1::from_vec(gradient))
    })
}

/// Parameters of one TIP4P site evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Tip4pSites {
    /// Attractive coefficient. Used when `use_epsilon` is false.
    pub c6: f64,
    /// Repulsive coefficient. Used when `use_epsilon` is false.
    pub c12: f64,
    /// Well depth. Used when `use_epsilon` is true.
    pub epsilon: f64,
    /// Diameter. Used when `use_epsilon` is true.
    pub sigma: f64,
    /// Select the epsilon and sigma oxygen term.
    pub use_epsilon: bool,
    /// Coulomb prefactor.
    pub coulomb: f64,
    /// Hydrogen charge.
    pub q_h: f64,
    /// M-site charge.
    pub q_m: f64,
}

/// TIP4P site energy and gradient. `sites` is molecule-major O, H, H, M.
pub fn tip4p_site_gradient(cfg: Tip4pSites, sites: &[f64]) -> (f64, Vec<f64>) {
    assert!(
        sites.len().is_multiple_of(12) && !sites.is_empty(),
        "TIP4P sites are 12 coordinates per molecule"
    );
    let mut gradient = vec![0.0; sites.len()];
    let energy = rgpot_pair::tip4p_sites(
        cfg.use_epsilon,
        cfg.c6,
        cfg.c12,
        cfg.epsilon,
        cfg.sigma,
        cfg.coulomb,
        cfg.q_h,
        cfg.q_m,
        sites,
        &mut gradient,
    );
    (energy, gradient)
}
