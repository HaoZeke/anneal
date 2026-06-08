//! `trait Bias<S>`: cost-augmenting operator for enhanced-sampling
//! methods inside SA. The standard well-tempered metadynamics
//! (Barducci/Bussi/Parrinello 2008) is the canonical impl; the trait
//! is open so SGOOP-style or VES bias kernels can implement the same surface.
//!
//! Composition with the existing typed algebra:
//!   - `Bias` augments the cost: `F_eff(x) = F(x) + V(s(x))`.
//!   - `Sampler` sees `F_eff` via a wrapped objective.
//!   - The bias is a stateful component: `deposit` mutates internal
//!     state, `potential` reads it.
//! This keeps biasing in the objective transform instead of folding it into
//! the sampler implementation.

use ndarray::{Array1, Array2, ArrayView1};

/// Cost-augmenting bias on a low-dimensional collective variable `s = phi(x)`.
/// Implementors maintain internal state that is updated by `deposit`
/// and read by `potential`.
pub trait Bias: Send + Sync {
    /// Map a position to its CV value.
    fn cv(&self, x: ArrayView1<f64>) -> Array1<f64>;

    /// Bias potential at the CV value `s`.
    fn potential(&self, s: ArrayView1<f64>) -> f64;

    /// Deposit a Gaussian at CV value `s` at temperature `temp`.
    /// Mutating; the well-tempered weight depends on the current
    /// `potential(s)` and `temp`.
    fn deposit(&mut self, s: ArrayView1<f64>, temp: f64);

    /// Reweighting factor `exp(+V(s) / T)` for post-hoc unbiasing of
    /// observables computed under the biased ensemble.
    fn reweight(&self, s: ArrayView1<f64>, temp: f64) -> f64 {
        (self.potential(s) / temp).exp()
    }
}

/// Well-tempered metadynamics (Barducci, Bussi, Parrinello 2008).
///
/// State: a 2D grid of bin values storing the deposited bias.
/// Deposit rule:
///   `w_k = w_0 * exp(-V_{k-1}(s_k) / ((gamma - 1) * T))`
/// so the asymptotic bias is `V_inf(s) = -((gamma - 1)/gamma) * F(s)`.
///
/// Asymmetric box bounds are supported on each CV axis. The CV map
/// is a fixed linear projection from `R^dim` to `R^2` configured at
/// construction time -- typical use: pass the top-2 TICA components.
pub struct WellTemperedBias {
    /// Linear projection `phi: R^dim -> R^2`, shape (dim, 2).
    pub projector: Array2<f64>,
    /// Mean offset in `R^dim` subtracted before projection.
    pub mu: Array1<f64>,
    /// CV-space lower corner.
    pub low: [f64; 2],
    /// CV-space upper corner.
    pub high: [f64; 2],
    /// Gaussian width in CV space.
    pub sigma: f64,
    /// Initial deposition height.
    pub w0: f64,
    /// Well-tempered bias factor; `gamma > 1`.
    pub gamma: f64,
    /// Number of bins per CV axis.
    pub grid_n: usize,
    /// Bias values on the `(grid_n, grid_n)` grid.
    pub v: Array2<f64>,
}

impl WellTemperedBias {
    /// Constructs with the given linear projector + bias parameters.
    /// Asserts `gamma > 1`, `sigma > 0`, `grid_n >= 2`.
    pub fn new(
        projector: Array2<f64>,
        mu: Array1<f64>,
        low: [f64; 2],
        high: [f64; 2],
        sigma: f64,
        w0: f64,
        gamma: f64,
        grid_n: usize,
    ) -> Self {
        assert!(gamma > 1.0, "gamma must be > 1");
        assert!(sigma > 0.0, "sigma must be > 0");
        assert!(grid_n >= 2, "grid_n must be >= 2");
        assert!(low[0] < high[0] && low[1] < high[1], "low < high required");
        assert_eq!(projector.shape()[1], 2, "projector must be (dim, 2)");
        assert_eq!(
            projector.shape()[0],
            mu.len(),
            "mu / projector dim mismatch"
        );
        Self {
            projector,
            mu,
            low,
            high,
            sigma,
            w0,
            gamma,
            grid_n,
            v: Array2::zeros((grid_n, grid_n)),
        }
    }

    /// Returns the (i_f, j_f) fractional grid indices for the CV value `s`.
    fn frac_indices(&self, s: ArrayView1<f64>) -> (f64, f64) {
        let sx = s[0].clamp(self.low[0], self.high[0]);
        let sy = s[1].clamp(self.low[1], self.high[1]);
        let i_f = (sx - self.low[0]) / (self.high[0] - self.low[0]) * (self.grid_n as f64 - 1.0);
        let j_f = (sy - self.low[1]) / (self.high[1] - self.low[1]) * (self.grid_n as f64 - 1.0);
        (i_f, j_f)
    }
}

impl Bias for WellTemperedBias {
    fn cv(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let n = x.len();
        let mut centred = Array1::zeros(n);
        for i in 0..n {
            centred[i] = x[i] - self.mu[i];
        }
        // s = projector^T * centred, shape (2,).
        let mut s = Array1::zeros(2);
        for d in 0..2 {
            let mut acc = 0.0;
            for i in 0..n {
                acc += self.projector[[i, d]] * centred[i];
            }
            s[d] = acc;
        }
        s
    }

    fn potential(&self, s: ArrayView1<f64>) -> f64 {
        let (i_f, j_f) = self.frac_indices(s);
        let i0 = (i_f.floor() as usize).min(self.grid_n - 2);
        let j0 = (j_f.floor() as usize).min(self.grid_n - 2);
        let di = i_f - i0 as f64;
        let dj = j_f - j0 as f64;
        let v00 = self.v[[i0, j0]];
        let v10 = self.v[[i0 + 1, j0]];
        let v01 = self.v[[i0, j0 + 1]];
        let v11 = self.v[[i0 + 1, j0 + 1]];
        v00 * (1.0 - di) * (1.0 - dj)
            + v10 * di * (1.0 - dj)
            + v01 * (1.0 - di) * dj
            + v11 * di * dj
    }

    fn deposit(&mut self, s: ArrayView1<f64>, temp: f64) {
        let v_at_s = self.potential(s);
        let w = self.w0 * (-v_at_s / ((self.gamma - 1.0) * temp)).exp();
        let dx = (self.high[0] - self.low[0]) / (self.grid_n as f64 - 1.0);
        let dy = (self.high[1] - self.low[1]) / (self.grid_n as f64 - 1.0);
        let two_sigma2 = 2.0 * self.sigma * self.sigma;
        for i in 0..self.grid_n {
            let gx = self.low[0] + i as f64 * dx;
            for j in 0..self.grid_n {
                let gy = self.low[1] + j as f64 * dy;
                let dx2 = (gx - s[0]).powi(2) + (gy - s[1]).powi(2);
                self.v[[i, j]] += w * (-dx2 / two_sigma2).exp();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn identity_projector_2d() -> WellTemperedBias {
        // 2D identity projector: CV is x itself.
        let projector = array![[1.0, 0.0], [0.0, 1.0]];
        let mu = array![0.0, 0.0];
        WellTemperedBias::new(projector, mu, [-5.0, -5.0], [5.0, 5.0], 0.5, 1.0, 5.0, 32)
    }

    #[test]
    fn potential_is_zero_before_deposit() {
        let b = identity_projector_2d();
        let s = array![0.0, 0.0];
        assert_eq!(b.potential(s.view()), 0.0);
    }

    #[test]
    fn deposit_increases_potential_at_centre() {
        let mut b = identity_projector_2d();
        let s = array![1.0, 1.0];
        let v_before = b.potential(s.view());
        b.deposit(s.view(), 1.0);
        let v_after = b.potential(s.view());
        assert!(
            v_after > v_before,
            "deposit should increase V at the deposit point: {} -> {}",
            v_before,
            v_after,
        );
    }

    #[test]
    fn well_tempered_height_decays_with_existing_bias() {
        let mut b = identity_projector_2d();
        let s = array![0.0, 0.0];
        // Deposit 10x at the same place; the height should monotonically shrink.
        let mut last_increment = f64::INFINITY;
        for _ in 0..10 {
            let v_before = b.potential(s.view());
            b.deposit(s.view(), 1.0);
            let v_after = b.potential(s.view());
            let delta = v_after - v_before;
            assert!(delta > 0.0);
            assert!(
                delta <= last_increment + 1e-12,
                "well-tempered height failed to decay"
            );
            last_increment = delta;
        }
    }
}
