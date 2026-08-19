//! `BarEstimator`: Bennett's acceptance-ratio estimator for the
//! free-energy difference `Delta F = F_B - F_A` between two ensembles.
//! The inputs use reduced energy differences, with inverse-temperature
//! factors supplied for the two ensembles.
//!
//! Pure observable: takes the two energy series and returns
//! `(Delta F, sigma^2)`. No kernel perturbation.
//! The estimator is exposed as an observable so kernels can compare
//! ensembles without changing the proposal or acceptance machinery.

/// Fermi function `f(x) = 1 / (1 + exp(x))`, numerically stable.
fn fermi(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + x.exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Bennett 1976 self-consistent free-energy estimator.
///
/// Given samples from ensembles `A` and `B`, `du_a` is `U_B - U_A` at points
/// sampled from A, while `du_b` is `U_A - U_B` at points sampled from B.
/// The implementation multiplies these by `beta_a` and `beta_b`, respectively.
/// This raw-difference representation is exact for a shared energy scale; for
/// arbitrary reduced potentials, callers must construct the reduced
/// differences themselves because this API cannot encode both cross energies.
/// The BAR estimate solves the equality between the two Fermi sums, where
/// `C = Delta F + log(N_A / N_B)`.
#[derive(Clone, Debug)]
pub struct BarEstimator {
    /// Samples of `U_B - U_A` evaluated at points drawn from A.
    pub du_a: Vec<f64>,
    /// Samples of `U_A - U_B` evaluated at points drawn from B.
    pub du_b: Vec<f64>,
    /// Inverse temperature multiplying the A-side difference.
    pub beta_a: f64,
    /// Inverse temperature multiplying the B-side difference.
    pub beta_b: f64,
}

impl BarEstimator {
    /// Constructs from cross-energy lists and inverse temperatures.
    pub fn new(du_a: Vec<f64>, du_b: Vec<f64>, beta_a: f64, beta_b: f64) -> Self {
        Self {
            du_a,
            du_b,
            beta_a,
            beta_b,
        }
    }

    /// Returns `g(C) = sum_A f(beta_a * du_a - C) - sum_B f(beta_b * du_b + C)`.
    /// The BAR root is at `g(C) = 0`.
    fn root_residual(&self, c: f64) -> f64 {
        let lhs: f64 = self.du_a.iter().map(|du| fermi(self.beta_a * du - c)).sum();
        let rhs: f64 = self.du_b.iter().map(|du| fermi(self.beta_b * du + c)).sum();
        lhs - rhs
    }

    /// Solves the BAR self-consistency by bisection. Returns the root
    /// `C` and the corresponding `Delta F = C - log(N_A / N_B)`.
    pub fn solve(&self) -> (f64, f64) {
        let n_a = self.du_a.len();
        let n_b = self.du_b.len();
        if n_a == 0
            || n_b == 0
            || !(self.beta_a.is_finite() && self.beta_a > 0.0)
            || !(self.beta_b.is_finite() && self.beta_b > 0.0)
            || self.du_a.iter().any(|x| !x.is_finite())
            || self.du_b.iter().any(|x| !x.is_finite())
        {
            return (f64::NAN, f64::NAN);
        }

        // The residual is monotone increasing in C. Expand from a finite
        // central bracket so saturated Fermi tails cannot choose a false
        // sign, then bisect only after both endpoint signs are established.
        let mut lo = -1.0;
        let mut hi = 1.0;
        for _ in 0..100 {
            if self.root_residual(lo) <= 0.0 {
                break;
            }
            lo *= 2.0;
        }
        for _ in 0..100 {
            if self.root_residual(hi) >= 0.0 {
                break;
            }
            hi *= 2.0;
        }
        if self.root_residual(lo) > 0.0 || self.root_residual(hi) < 0.0 {
            return (f64::NAN, f64::NAN);
        }
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if self.root_residual(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let c = 0.5 * (lo + hi);
        let delta_f = c - (n_a as f64 / n_b as f64).ln();
        (c, delta_f)
    }

    /// Asymptotic variance of the BAR estimator (Shirts/Chodera 2008).
    /// Returns the sample variance of `Delta F`.
    pub fn variance(&self) -> f64 {
        let (c, _) = self.solve();
        let n_a = self.du_a.len() as f64;
        let n_b = self.du_b.len() as f64;
        let f_a: Vec<f64> = self
            .du_a
            .iter()
            .map(|du| fermi(self.beta_a * du - c))
            .collect();
        let f_b: Vec<f64> = self
            .du_b
            .iter()
            .map(|du| fermi(self.beta_b * du + c))
            .collect();
        let mean_a: f64 = f_a.iter().sum::<f64>() / n_a;
        let mean_b: f64 = f_b.iter().sum::<f64>() / n_b;
        let var_a: f64 = f_a.iter().map(|fi| (fi - mean_a).powi(2)).sum::<f64>() / n_a;
        let var_b: f64 = f_b.iter().map(|fi| (fi - mean_b).powi(2)).sum::<f64>() / n_b;
        var_a / (n_a * mean_a.powi(2).max(1e-18)) + var_b / (n_b * mean_b.powi(2).max(1e-18))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    /// Toy: two zero-mean Gaussians with different variances.
    /// p_A(x) = N(0, sigma_A^2); F_A = ln(sigma_A) + const.
    /// p_B(x) = N(0, sigma_B^2); F_B = ln(sigma_B) + const.
    /// Delta F = -ln(sigma_B / sigma_A), independent of the constant.
    #[test]
    fn bar_recovers_gaussian_log_ratio_free_energy() {
        let sigma_a = 1.0_f64;
        let sigma_b = 2.0_f64;
        let beta = 1.0_f64;
        let true_df = -(sigma_b / sigma_a).ln();

        let mut rng = StdRng::seed_from_u64(42);
        let normal_a = Normal::new(0.0_f64, sigma_a).unwrap();
        let normal_b = Normal::new(0.0_f64, sigma_b).unwrap();
        let n = 2000;
        let xs_a: Vec<f64> = (0..n).map(|_| normal_a.sample(&mut rng)).collect();
        let xs_b: Vec<f64> = (0..n).map(|_| normal_b.sample(&mut rng)).collect();

        // U_A(x) = 0.5 (x / sigma_A)^2; U_B(x) = 0.5 (x / sigma_B)^2.
        let du_a: Vec<f64> = xs_a
            .iter()
            .map(|&x| 0.5 * (x / sigma_b).powi(2) - 0.5 * (x / sigma_a).powi(2))
            .collect();
        let du_b: Vec<f64> = xs_b
            .iter()
            .map(|&x| 0.5 * (x / sigma_a).powi(2) - 0.5 * (x / sigma_b).powi(2))
            .collect();

        let bar = BarEstimator::new(du_a, du_b, beta, beta);
        let (_c, delta_f) = bar.solve();
        assert!(
            delta_f.is_finite(),
            "BAR estimate {} should be finite; truth is {}",
            delta_f,
            true_df
        );
        assert!(
            (delta_f - true_df).abs() < 0.12,
            "BAR estimate {} differs from truth {}",
            delta_f,
            true_df,
        );
    }
}
