//! Binary Lennard-Jones cluster, with no cutoff.
//!
//! Pair energy is `4 ε [(σ/r)^12 - (σ/r)^6]`. Atom indices `0..n_a` are
//! species A and the rest are species B. The arithmetic mean of `σ_AA` and
//! `σ_BB` is the usual unlike diameter; the caller passes that value in.

use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};

/// Two-species Lennard-Jones cluster.
#[derive(Debug, Clone)]
pub struct BinaryLennardJones {
    n_a: usize,
    n_b: usize,
    epsilon_aa: f64,
    epsilon_ab: f64,
    epsilon_bb: f64,
    sigma_aa: f64,
    sigma_ab: f64,
    sigma_bb: f64,
    bounds: Bounds<f64>,
}

impl BinaryLennardJones {
    /// A cluster with `n_a` atoms of species A followed by `n_b` of species B.
    pub fn new(
        n_a: usize,
        n_b: usize,
        epsilon_aa: f64,
        epsilon_ab: f64,
        epsilon_bb: f64,
        sigma_aa: f64,
        sigma_ab: f64,
        sigma_bb: f64,
    ) -> Self {
        let n = n_a + n_b;
        assert!(n >= 2, "a binary cluster needs at least two atoms");
        assert!(
            [
                epsilon_aa, epsilon_ab, epsilon_bb, sigma_aa, sigma_ab, sigma_bb
            ]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
            "Lennard-Jones parameters must be finite and positive"
        );
        let extent = 4.0
            * sigma_aa.max(sigma_bb).max(sigma_ab)
            * 2.0_f64.powf(1.0 / 6.0)
            * (n as f64).cbrt();
        Self {
            n_a,
            n_b,
            epsilon_aa,
            epsilon_ab,
            epsilon_bb,
            sigma_aa,
            sigma_ab,
            sigma_bb,
            bounds: Bounds::new(
                Array1::from_elem(3 * n, -extent),
                Array1::from_elem(3 * n, extent),
                0.0,
            ),
        }
    }

    /// Atoms in the cluster.
    pub fn n_points(&self) -> usize {
        self.n_a + self.n_b
    }

    /// Energy and Cartesian gradient in one pair pass.
    pub fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = self.n_points();
        assert_eq!(
            x.len(),
            3 * n,
            "binary Lennard-Jones state is 3 coordinates per atom"
        );
        super::rgpot_call::binary_lennard_jones(
            self.n_a,
            [self.epsilon_aa, self.epsilon_ab, self.epsilon_bb],
            [self.sigma_aa, self.sigma_ab, self.sigma_bb],
            0.0,
            x,
        )
    }
}

impl Objective<f64> for BinaryLennardJones {
    fn dim(&self) -> usize {
        3 * self.n_points()
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for BinaryLennardJones {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }

    fn dim(&self) -> usize {
        3 * self.n_points()
    }
}

impl DifferentiableObjective<f64> for BinaryLennardJones {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        BinaryLennardJones::value_and_gradient(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potentials::PairPotential;

    #[test]
    fn equal_diameters_match_the_monoatomic_potential() {
        let binary = BinaryLennardJones::new(2, 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let mono = PairPotential::lennard_jones(4);
        let x = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.2, 0.1, -0.2, 0.4, 1.3, 0.2, -0.3, 0.5, 1.1,
        ]);
        let (energy, gradient) = binary.value_and_gradient(x.view());
        let (mono_energy, mono_gradient) = mono.value_and_gradient(x.view());
        assert!((energy - mono_energy).abs() < 1e-12);
        for (left, right) in gradient.iter().zip(mono_gradient.iter()) {
            assert!((left - right).abs() < 1e-12);
        }
    }

    #[test]
    fn the_gradient_matches_a_finite_difference() {
        let binary = BinaryLennardJones::new(2, 2, 1.0, 1.0, 1.0, 1.0, 1.15, 1.3);
        let x = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.1, 0.05, 0.0, 2.4, 0.2, 0.1, 1.3, 1.6, -0.2,
        ]);
        let (_, gradient) = binary.value_and_gradient(x.view());
        let step = 1e-6;
        for k in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[k] += step;
            minus[k] -= step;
            let numeric = (binary.eval(plus.view()) - binary.eval(minus.view())) / (2.0 * step);
            let scale = 1.0 + numeric.abs();
            assert!(
                (gradient[k] - numeric).abs() / scale < 1e-6,
                "coordinate {k}: analytic {} against numeric {numeric}",
                gradient[k]
            );
        }
    }

    #[test]
    fn the_pair_minimum_sits_at_the_unlike_diameter() {
        let binary = BinaryLennardJones::new(1, 1, 1.0, 1.0, 1.0, 1.0, 1.15, 1.3);
        let r_min = 1.15 * 2.0_f64.powf(1.0 / 6.0);
        let x = Array1::from_vec(vec![0.0, 0.0, 0.0, r_min, 0.0, 0.0]);
        let energy = binary.eval(x.view());
        assert!(
            (energy + 1.0).abs() < 1e-12,
            "unlike well depth is {energy}"
        );
    }
}
