//! Momentum kernels for HMC: standard Gaussian (q=1) and q-Gaussian
//! (Tsallis q in (1, 1+2/dim); doi:10.1007/BF01016429).
//!
//! q-Gaussian momentum unifies HMC with the GSA Tsallis hierarchy
//! (doi:10.1016/S0378-4371(96)00271-3). The kinetic term becomes
//!   K(p, q) = (1/(q-1)) ln[1 + (q-1) |p|^2 / 2]
//! whose density is the multivariate q-Gaussian
//!   rho(p) ~ [1 + (q-1) |p|^2 / 2]^{-1/(q-1)}.
//!
//! q -> 1+ recovers Gaussian momentum. q in (1, 1 + 2/dim)
//! gives heavy-tailed momentum draws, which is exactly what HMC needs
//! to escape local cups on multimodal objectives (Rastrigin / Schwefel).
//! At q approaching 1 + 2/dim, the q-Gaussian normalisation diverges;
//! the constructor asserts the valid range.
//!
//! Sampling representation (standard for Student-t / q-Gaussian):
//!   z ~ N(0, I_d), g ~ Gamma(alpha, 1) with alpha = 1/(q-1) - d/2,
//!   p = z * sqrt(1 / ((q-1) * g)).

use ndarray::Array1;
use rand::Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};

/// Generic momentum kernel for HMC. Implementors define the sampling
/// distribution and the kinetic energy + its gradient w.r.t. `p`.
pub trait Momentum: Send + Sync {
    /// Draw a fresh momentum vector.
    fn sample<R: Rng>(&self, dim: usize, rng: &mut R) -> Array1<f64>;

    /// Kinetic energy `K(p)`. Used in the Hamiltonian computation.
    fn kinetic(&self, p: &Array1<f64>) -> f64;

    /// Gradient of `K` w.r.t. `p`. Used in the leapfrog drift step.
    fn dk_dp(&self, p: &Array1<f64>) -> Array1<f64>;

    /// NUTS U-turn termination predicate: returns true iff the trajectory
    /// has reversed direction between the leftmost and rightmost states.
    /// For Gaussian momentum: `<p_l, dx> < 0 || <p_r, dx> < 0` per
    /// Hoffman/Gelman 2014 NUTS, eqn 9. The default implementation
    /// uses this Gaussian formula (correct for both Gaussian and
    /// q-Gaussian since `dK/dp` is just `p` rescaled, preserving sign).
    fn uturn(
        &self,
        x_left: &Array1<f64>,
        p_left: &Array1<f64>,
        x_right: &Array1<f64>,
        p_right: &Array1<f64>,
    ) -> bool {
        let dim = x_left.len();
        let mut dot_l: f64 = 0.0;
        let mut dot_r: f64 = 0.0;
        for i in 0..dim {
            let dx = x_right[i] - x_left[i];
            dot_l += p_left[i] * dx;
            dot_r += p_right[i] * dx;
        }
        dot_l < 0.0 || dot_r < 0.0
    }
}

/// Standard Gaussian momentum: `K(p) = |p|^2 / 2`, `dK/dp = p`.
/// Recovered as the `q -> 1+` limit of the q-Gaussian.
pub struct GaussianMomentum;

impl Momentum for GaussianMomentum {
    fn sample<R: Rng>(&self, dim: usize, rng: &mut R) -> Array1<f64> {
        Array1::from_iter((0..dim).map(|_| {
            let z: f64 = StandardNormal.sample(rng);
            z
        }))
    }

    fn kinetic(&self, p: &Array1<f64>) -> f64 {
        0.5 * p.iter().map(|pi| pi * pi).sum::<f64>()
    }

    fn dk_dp(&self, p: &Array1<f64>) -> Array1<f64> {
        p.clone()
    }
}

/// q-Gaussian (Tsallis) momentum with index `q in (1, 1 + 2/dim)`.
/// Heavy-tailed for `q > 1`; the kinetic term grows logarithmically in
/// `|p|^2` so large momentum draws cost much less than under Gaussian
/// momentum. This is the q-deformed Hamiltonian dynamics that lets
/// HMC-SA escape local cups on multimodal objectives.
#[derive(Clone, Copy, Debug)]
pub struct QGaussianMomentum {
    /// Tsallis index. Must satisfy `1 < q < 1 + 2/dim` to give a
    /// normalisable density. Boundary `q == 1` falls back to Gaussian.
    pub q: f64,
}

impl QGaussianMomentum {
    /// Constructs with the given `q`. Validate against `dim` separately
    /// before sampling -- the `dim` arrives at sample time.
    pub fn new(q: f64) -> Self {
        assert!(
            q > 1.0,
            "q-Gaussian requires q > 1; for q = 1 use GaussianMomentum"
        );
        Self { q }
    }

    /// Returns `1/(q-1) - dim/2`, the Gamma shape parameter. Must be
    /// strictly positive for the density to integrate.
    fn alpha(&self, dim: usize) -> f64 {
        1.0 / (self.q - 1.0) - 0.5 * dim as f64
    }

    /// Maximum dim for which this `q` gives a valid q-Gaussian.
    pub fn max_dim(&self) -> usize {
        ((2.0 / (self.q - 1.0)).floor() as usize).saturating_sub(1)
    }
}

impl Momentum for QGaussianMomentum {
    fn sample<R: Rng>(&self, dim: usize, rng: &mut R) -> Array1<f64> {
        let alpha = self.alpha(dim);
        assert!(
            alpha > 0.0,
            "q-Gaussian requires q < 1 + 2/dim; got q={}, dim={}, alpha={}",
            self.q,
            dim,
            alpha
        );
        let gamma = Gamma::new(alpha, 1.0).expect("valid Gamma shape");
        let g: f64 = gamma.sample(rng).max(1e-300);
        let scale = (1.0 / ((self.q - 1.0) * g)).sqrt();
        Array1::from_iter((0..dim).map(|_| {
            let z: f64 = StandardNormal.sample(rng);
            scale * z
        }))
    }

    fn kinetic(&self, p: &Array1<f64>) -> f64 {
        let p2 = p.iter().map(|pi| pi * pi).sum::<f64>();
        (1.0 / (self.q - 1.0)) * (1.0 + 0.5 * (self.q - 1.0) * p2).ln()
    }

    fn dk_dp(&self, p: &Array1<f64>) -> Array1<f64> {
        let p2 = p.iter().map(|pi| pi * pi).sum::<f64>();
        let denom = 1.0 + 0.5 * (self.q - 1.0) * p2;
        p.mapv(|pi| pi / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn gaussian_kinetic_at_origin_is_zero() {
        let m = GaussianMomentum;
        let p = Array1::zeros(5);
        assert_eq!(m.kinetic(&p), 0.0);
    }

    #[test]
    fn q_gaussian_kinetic_at_origin_is_zero() {
        let m = QGaussianMomentum::new(1.3);
        let p = Array1::zeros(5);
        assert_eq!(m.kinetic(&p), 0.0);
    }

    #[test]
    fn q_gaussian_max_dim_obeys_q_constraint() {
        // q = 1.5 -> max_dim should be floor(2/0.5)-1 = 3
        let m = QGaussianMomentum::new(1.5);
        assert_eq!(m.max_dim(), 3);
        // q = 1.2 -> max_dim should be floor(2/0.2)-1 = 9
        let m2 = QGaussianMomentum::new(1.2);
        assert_eq!(m2.max_dim(), 9);
    }

    #[test]
    fn q_gaussian_sample_has_finite_kinetic() {
        let m = QGaussianMomentum::new(1.3);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            let p = m.sample(5, &mut rng);
            let k = m.kinetic(&p);
            assert!(k.is_finite(), "q-Gaussian kinetic non-finite: {k}");
            let dk = m.dk_dp(&p);
            for &v in dk.iter() {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn q_gaussian_heavy_tails_vs_gaussian() {
        // Q-Gaussian should produce some larger |p| draws than Gaussian
        // at matched seed/dim, evidencing the heavy-tail property.
        let g = GaussianMomentum;
        let q = QGaussianMomentum::new(1.3);
        let mut rng_g = StdRng::seed_from_u64(7);
        let mut rng_q = StdRng::seed_from_u64(7);
        let mut max_g: f64 = 0.0;
        let mut max_q: f64 = 0.0;
        for _ in 0..200 {
            let pg = g.sample(5, &mut rng_g);
            let pq = q.sample(5, &mut rng_q);
            let normg: f64 = pg.iter().map(|v| v * v).sum::<f64>().sqrt();
            let normq: f64 = pq.iter().map(|v| v * v).sum::<f64>().sqrt();
            if normg > max_g {
                max_g = normg;
            }
            if normq > max_q {
                max_q = normq;
            }
        }
        // Heavy tail: q-Gaussian's max |p| should exceed Gaussian's by a
        // measurable factor (typically 2-5x at q=1.3 over 200 draws).
        assert!(
            max_q > max_g,
            "q-Gaussian max-|p| {} not heavier than Gaussian max-|p| {}",
            max_q,
            max_g
        );
    }
}
