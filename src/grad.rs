//! `trait Gradient<T>`: typed first-derivative interface for the HMC
//! sampler family (Method B). Mirrors Stan's separation of the
//! log-density and its gradient into separate compute paths.
//!
//! See `~/Git/Gitlab/obsidian-notes/Software/anneal/design_pass_09_method_b_hmc.org`.

use eindir_core::Objective;
use ndarray::{Array1, ArrayView1};
use num_traits::Float;

/// First-derivative interface. Implementors that own an analytic
/// gradient implement this directly; black-box objectives use
/// `FiniteDiffGradient`.
pub trait Gradient<T: Float>: Send + Sync {
    /// Gradient of the underlying objective at `x`.
    fn grad(&self, x: ArrayView1<T>) -> Array1<T>;

    /// Number of input dimensions; matches the underlying `Objective::dim`.
    fn dim(&self) -> usize;
}

/// Central-difference finite-difference gradient adapter for any
/// `Objective<f64>`. Costs `2 * dim` objective evaluations per
/// `grad` call. Use for black-box objectives; replace with an
/// analytic impl when one is available.
pub struct FiniteDiffGradient<O: Objective<f64>> {
    /// The wrapped objective.
    pub obj: O,
    /// Step size for the central difference. Default `1e-5`.
    pub h: f64,
}

impl<O: Objective<f64>> FiniteDiffGradient<O> {
    /// Constructs with the default step `h = 1e-5`.
    pub fn new(obj: O) -> Self {
        Self { obj, h: 1e-5 }
    }

    /// Constructs with a user-specified step `h`.
    pub fn with_step(obj: O, h: f64) -> Self {
        assert!(h > 0.0, "h must be positive");
        Self { obj, h }
    }
}

impl<O: Objective<f64> + Send + Sync> Gradient<f64> for FiniteDiffGradient<O> {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let n = x.len();
        let mut g = Array1::zeros(n);
        let mut xp = x.to_owned();
        let mut xm = x.to_owned();
        for i in 0..n {
            xp[i] = x[i] + self.h;
            xm[i] = x[i] - self.h;
            g[i] = (self.obj.eval(xp.view()) - self.obj.eval(xm.view())) / (2.0 * self.h);
            xp[i] = x[i];
            xm[i] = x[i];
        }
        g
    }

    fn dim(&self) -> usize {
        self.obj.dim()
    }
}
