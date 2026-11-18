//! Binary Lennard-Jones cluster, with no cutoff.
//!
//! Pair energy is `4 ε [(σ/r)^12 - (σ/r)^6]`. Atom indices `0..n_a` are
//! species A and the rest are species B. The arithmetic mean of `σ_AA` and
//! `σ_BB` is the usual unlike diameter; the caller passes that value in.
//!
//! When the process has installed the rgpot kernels, the evaluation is
//! `rgpot_pair::binary_lj` with a non-positive cutoff, the same dispatch
//! `PairPotential` uses for the one-component clusters. The pair loop below
//! is the fallback for a process that has not installed a kernel.

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

    /// Doye-Meyer parameters: every `ε` is 1, `σ_AA` is 1, and
    /// `σ_AB = (σ_AA + σ_BB) / 2`.
    #[cfg(test)]
    pub fn doye_meyer(n_a: usize, n_b: usize, sigma_bb: f64) -> Self {
        Self::new(
            n_a,
            n_b,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5 * (1.0 + sigma_bb),
            sigma_bb,
        )
    }

    /// Atoms in the cluster.
    pub fn n_points(&self) -> usize {
        self.n_a + self.n_b
    }

    fn pair_params(&self, i: usize, j: usize) -> (f64, f64) {
        let a_i = i < self.n_a;
        let a_j = j < self.n_a;
        match (a_i, a_j) {
            (true, true) => (self.epsilon_aa, self.sigma_aa),
            (false, false) => (self.epsilon_bb, self.sigma_bb),
            _ => (self.epsilon_ab, self.sigma_ab),
        }
    }

    /// Energy and Cartesian gradient in one pair pass.
    pub fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = self.n_points();
        assert_eq!(
            x.len(),
            3 * n,
            "binary Lennard-Jones state is 3 coordinates per atom"
        );
        if let (Some(kernels), Some(positions)) = (super::cluster_kernels(), x.as_slice()) {
            let mut gradient = vec![0.0; positions.len()];
            let energy = (kernels.binary_lj)(
                self.n_a,
                [self.epsilon_aa, self.epsilon_ab, self.epsilon_bb],
                [self.sigma_aa, self.sigma_ab, self.sigma_bb],
                0.0,
                positions,
                &mut gradient,
            );
            return (energy, Array1::from(gradient));
        }
        self.local_value_and_gradient(x)
    }

    /// The pair sum used when no kernel is installed.
    fn local_value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = self.n_points();
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 <= 0.0 {
                    continue;
                }
                let (eps, sigma) = self.pair_params(i, j);
                let a = (sigma * sigma) / r2;
                let a3 = a * a * a;
                let a6 = a3 * a3;
                energy += 4.0 * eps * (a6 - a3);
                let coef = 4.0 * eps * (-12.0 * a6 + 6.0 * a3) / r2;
                for k in 0..3 {
                    gradient[3 * i + k] += coef * d[k];
                    gradient[3 * j + k] -= coef * d[k];
                }
            }
        }
        (energy, gradient)
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

    /// Header fields of a Cambridge Cluster Database points file.
    fn ccd_header(line: &str) -> (f64, usize, f64) {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let mut energy = None;
        let mut n_a = None;
        let mut sigma = None;
        for (i, token) in tokens.iter().enumerate() {
            if *token == "1=" {
                energy = tokens[i + 1].parse().ok();
            }
            if *token == "(NA,SIG_BB)" {
                n_a = tokens[i + 1].parse().ok();
                sigma = tokens[i + 2].parse().ok();
            }
        }
        (
            energy.expect("header energy"),
            n_a.expect("header NA"),
            sigma.expect("header sigma"),
        )
    }

    fn load_ccd(name: &str) -> (f64, usize, f64, Array1<f64>) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/blj-ccd")
            .join(name);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        let mut lines = text.lines();
        let n: usize = lines
            .next()
            .and_then(|line| line.trim().parse().ok())
            .unwrap_or_else(|| panic!("{name}: atom count"));
        let (energy, n_a, sigma) = ccd_header(lines.next().unwrap_or(""));
        let mut a = Vec::new();
        let mut b = Vec::new();
        for line in lines {
            let mut parts = line.split_whitespace();
            let Some(label) = parts.next() else {
                continue;
            };
            let coords: Vec<f64> = parts
                .map(|word| word.parse().expect("coordinate"))
                .collect();
            assert_eq!(coords.len(), 3, "{name}: {label}");
            match label {
                "LA" | "A" => a.extend(coords),
                "LB" | "B" => b.extend(coords),
                other => panic!("{name}: unknown label {other}"),
            }
        }
        assert_eq!(a.len() / 3, n_a, "{name}: species A");
        assert_eq!(a.len() / 3 + b.len() / 3, n, "{name}: atom count");
        a.extend(b);
        (energy, n_a, sigma, Array1::from_vec(a))
    }

    #[test]
    fn ccd_structures_match_their_headers_and_the_local_loop() {
        for name in [
            "blj105n38",
            "blj105n55",
            "blj115n38",
            "blj115n55",
            "blj130n38",
            "blj130n55",
            "blj130n98",
            "blj130n98-original",
        ] {
            let (header, n_a, sigma, coords) = load_ccd(name);
            let n = coords.len() / 3;
            let potential = BinaryLennardJones::doye_meyer(n_a, n - n_a, sigma);
            let (energy, gradient) = potential.value_and_gradient(coords.view());
            let (local, local_gradient) = potential.local_value_and_gradient(coords.view());
            assert!(
                (energy - header).abs() < 1e-9,
                "{name}: kernel {energy} against header {header}"
            );
            assert_eq!(
                energy.to_bits(),
                local.to_bits(),
                "{name}: kernel against the pair loop"
            );
            assert_eq!(gradient.len(), local_gradient.len());
            for (left, right) in gradient.iter().zip(local_gradient.iter()) {
                assert_eq!(left.to_bits(), right.to_bits(), "{name}");
            }
        }
    }
}
