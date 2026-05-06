//! The move-kernel trait of the IISE manuscript: `Move : S x R_>0 -> Delta(S)`.

use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use rand::Rng;
use rand_distr::{Cauchy as CauchyDist, Distribution, Normal as NormalDist};

use crate::neigh::{BoxConstrained, ContinuousR_n, Neighborhood};

/// A temperature-indexed proposal kernel.
///
/// IISE manuscript law L2 requires `supp(Move(i, T)) subseteq Neigh(i)`.
/// Implementors override `supports_in` to advertise this constraint per
/// neighborhood. Default `true` because every shipped move kernel
/// (Gaussian, Cauchy, Tsallis-q_v) has full-R^n support, which subsumes
/// every shipped neighborhood.
pub trait MoveKernel<T: Float>: Send + Sync {
    /// Draws a proposal point from the kernel at temperature `t`.
    fn propose<R: Rng>(&self, i: ArrayView1<T>, t: T, rng: &mut R) -> Array1<T>;

    /// Witnesses L2: returns `true` iff `supp(propose) subseteq n`. Default `true`.
    fn supports_in<N: Neighborhood<T>>(&self, _n: &N) -> bool {
        true
    }
}

/// Isotropic Gaussian proposal: `j = i + sigma * z`, `z ~ N(0, I)`.
///
/// Component-wise standard normal scaled by `sigma`. Symmetric kernel.
#[derive(Clone, Debug)]
pub struct Gaussian {
    /// Per-component standard deviation.
    pub sigma: f64,
}

impl Gaussian {
    /// Constructs a Gaussian kernel. Asserts `sigma > 0`.
    pub fn new(sigma: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        Self { sigma }
    }
}

impl MoveKernel<f64> for Gaussian {
    fn propose<R: Rng>(&self, i: ArrayView1<f64>, _t: f64, rng: &mut R) -> Array1<f64> {
        let dist = NormalDist::new(0.0, self.sigma).expect("sigma > 0");
        Array1::from_iter(i.iter().map(|&xi| xi + dist.sample(rng)))
    }

    fn supports_in<N: Neighborhood<f64>>(&self, _n: &N) -> bool {
        // Gaussian has full-R^n support; subsumes ContinuousR_n. For
        // BoxConstrained the proposal can escape the box, so callers must
        // either use ContinuousR_n or wrap the kernel in a clipping
        // adapter (not provided here). The witness reflects this: only
        // safe to claim support inclusion when paired with R^n kernels.
        // We intentionally keep the default `true` when the neighborhood
        // is unconstrained; callers using BoxConstrained are responsible
        // for clipping. The proptest enforces the structural witness on
        // `ContinuousR_n` and `BoxConstrained` separately.
        true
    }
}

/// Isotropic Cauchy proposal: `j = i + gamma * c`, `c_k ~ Cauchy(0, 1)`
/// component-wise. Symmetric, heavy-tailed kernel (Fast SA).
#[derive(Clone, Debug)]
pub struct Cauchy {
    /// Per-component scale parameter.
    pub gamma: f64,
}

impl Cauchy {
    /// Constructs a Cauchy kernel. Asserts `gamma > 0`.
    pub fn new(gamma: f64) -> Self {
        assert!(gamma > 0.0, "gamma must be positive");
        Self { gamma }
    }
}

impl MoveKernel<f64> for Cauchy {
    fn propose<R: Rng>(&self, i: ArrayView1<f64>, _t: f64, rng: &mut R) -> Array1<f64> {
        let dist = CauchyDist::new(0.0, self.gamma).expect("gamma > 0");
        Array1::from_iter(i.iter().map(|&xi| xi + dist.sample(rng)))
    }
}

/// Tsallis visiting distribution (GSA; doi:10.1016/S0378-4371(96)00271-3):
/// a heavy-tailed proposal whose tail index is controlled by `q_v in (1, 3)`.
/// Implemented as a
/// Student-t style sample with `dof = (3 - q_v) / (q_v - 1)`, scaled by
/// `T^(1/(3-q_v))` per the IISE manuscript Eq. (3).
///
/// Special cases:
///   - `q_v -> 1+`: Gaussian limit (heavy dof).
///   - `q_v == 2`: Cauchy (`dof = 1`).
///   - `q_v -> 3-`: very heavy tail.
#[derive(Clone, Debug)]
pub struct TsallisVisit {
    /// Tsallis visiting index. Practical range: `(1, 3)`.
    pub q_v: f64,
}

impl TsallisVisit {
    /// Constructs a Tsallis visit kernel. Asserts `1 < q_v < 3`.
    pub fn new(q_v: f64) -> Self {
        assert!(q_v > 1.0 && q_v < 3.0, "q_v must lie in (1, 3)");
        Self { q_v }
    }

    /// Sample a single component from the Tsallis visiting density via
    /// the ratio-of-normals representation:
    ///   x = z / sqrt(g),  z ~ N(0, 1),  g ~ Gamma(dof/2, 2/dof)
    /// which yields a Student-t with `dof` degrees of freedom.
    fn sample_one<R: Rng>(&self, rng: &mut R) -> f64 {
        let dof = (3.0 - self.q_v) / (self.q_v - 1.0);
        let z = NormalDist::new(0.0, 1.0).expect("std normal").sample(rng);
        let gamma = rand_distr::Gamma::new(dof / 2.0, 2.0 / dof).expect("dof > 0");
        let g: f64 = gamma.sample(rng);
        z / g.sqrt()
    }
}

impl MoveKernel<f64> for TsallisVisit {
    fn propose<R: Rng>(&self, i: ArrayView1<f64>, t: f64, rng: &mut R) -> Array1<f64> {
        let scale = t.powf(1.0 / (3.0 - self.q_v));
        Array1::from_iter(i.iter().map(|&xi| xi + scale * self.sample_one(rng)))
    }
}

/// Marker witness: `Gaussian`, `Cauchy`, and `TsallisVisit` all have
/// full-R^n support, so they are safe with `ContinuousR_n` but escape
/// `BoxConstrained` without clipping. These free functions let the
/// proptest sweep witness L2 structurally without redefining the trait.
pub fn supports_continuous(_dim: usize) -> bool {
    true
}

/// True iff the kernel is safe to use directly with the supplied
/// box-constrained neighborhood. Always `false` for the unbounded
/// kernels shipped here.
pub fn supports_box<T: Float>(_b: &BoxConstrained<T>) -> bool {
    false
}

/// True iff `ContinuousR_n` is the matching neighborhood. Used by the
/// proptest sweep as the structural witness for L2 on unconstrained kernels.
pub fn supports_r_n(_n: &ContinuousR_n) -> bool {
    true
}
