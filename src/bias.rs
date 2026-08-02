//! `trait Bias<S>`: cost-augmenting operator for enhanced-sampling
//! methods inside SA. The standard well-tempered metadynamics
//! (Barducci/Bussi/Parrinello 2008) is the canonical impl; the trait
//! is open so SGOOP-style or VES bias kernels can implement the same surface.
//!
//! In the typed algebra, `Bias` augments the cost as
//! `F_eff(x) = F(x) + V(s(x))`. `Sampler` sees `F_eff` via a wrapped
//! objective, while the bias remains stateful: `deposit` mutates internal
//! state and `potential` reads it.
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
/// The deposit rule is `w_k = w_0 * exp(-V_{k-1}(s_k) / ((gamma - 1) * T))`,
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
    /// Requires gamma > 1, sigma > 0, and grid_n >= 2.
    #[allow(clippy::too_many_arguments)]
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

/// Well-tempered bias keyed on discrete basin identity rather than on a
/// collective variable.
///
/// A grid bias has to be told which projection to watch. That works when the
/// competing structures separate along the chosen axis and fails silently when
/// they do not: on the 38-atom Lennard-Jones cluster the close-packed and
/// icosahedral funnels differ by 0.19 in the fourth Steinhardt parameter, while
/// at 75 atoms the decahedral and icosahedral minima differ by 0.023, which is
/// narrower than a sensible deposition width. The bias then fills a region that
/// contains both competitors and the search never leaves.
///
/// Keying on identity removes the choice. Two states are the same basin when
/// their fingerprints lie within `merge_radius`, so there is no axis to be
/// blind along. Revisiting a basin raises its bias, which is the superbasin
/// escape acceleration of Chatterjee and Voter (J Chem Phys 132, 194101, 2010)
/// with the Barducci well-tempered weight on top.
///
/// The fingerprint must be invariant to whatever the objective is invariant
/// under, or the same physical state registers as many basins. [`SortedPairs`]
/// is the default for point sets: invariant to permutation, translation and
/// rotation, and free of external dependencies.
pub struct BasinBias<F: Fingerprint> {
    fingerprint: F,
    merge_radius: f64,
    w0: f64,
    gamma: f64,
    centres: Vec<Array1<f64>>,
    v: Vec<f64>,
    visits: Vec<u64>,
}

/// Maps a state to a vector that compares equal for states in the same basin.
pub trait Fingerprint: Send + Sync {
    /// Descriptor of `x`. Two states in the same basin must map to vectors
    /// within the bias's merge radius, and states in different basins must not.
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64>;
}

/// Sorted pairwise distances of a flattened `(n, 3)` point set.
///
/// Invariant to permutation of the points and to rigid motions, which are the
/// symmetries of a cluster energy. Sorting is what supplies permutation
/// invariance and is also why the descriptor is cheap.
pub struct SortedPairs {
    /// Points per state; the state length must be `3 * n_points`.
    pub n_points: usize,
}

impl Fingerprint for SortedPairs {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let n = self.n_points;
        let mut d = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = x[3 * i] - x[3 * j];
                let dy = x[3 * i + 1] - x[3 * j + 1];
                let dz = x[3 * i + 2] - x[3 * j + 2];
                d.push((dx * dx + dy * dy + dz * dz).sqrt());
            }
        }
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Array1::from(d)
    }
}

impl<F: Fingerprint> BasinBias<F> {
    /// Requires `gamma > 1`, `w0 > 0` and `merge_radius > 0`.
    pub fn new(fingerprint: F, merge_radius: f64, w0: f64, gamma: f64) -> Self {
        assert!(gamma > 1.0, "gamma must be > 1");
        assert!(w0 > 0.0, "w0 must be > 0");
        assert!(merge_radius > 0.0, "merge_radius must be > 0");
        Self {
            fingerprint,
            merge_radius,
            w0,
            gamma,
            centres: Vec::new(),
            v: Vec::new(),
            visits: Vec::new(),
        }
    }

    /// Index of the nearest registered basin within `merge_radius`.
    fn lookup(&self, d: ArrayView1<f64>) -> Option<usize> {
        let mut best = None;
        let mut best_dist = f64::INFINITY;
        for (i, c) in self.centres.iter().enumerate() {
            if c.len() != d.len() {
                continue;
            }
            let mut acc = 0.0;
            for k in 0..d.len() {
                let diff = c[k] - d[k];
                acc += diff * diff;
            }
            let dist = acc.sqrt();
            if dist < best_dist {
                best_dist = dist;
                best = Some(i);
            }
        }
        best.filter(|_| best_dist <= self.merge_radius)
    }

    /// Number of distinct basins registered so far.
    pub fn n_basins(&self) -> usize {
        self.centres.len()
    }

    /// Good-Turing missing mass: the share of basins seen exactly once, which
    /// estimates the probability that the next visit opens a new one.
    pub fn missing_mass(&self) -> f64 {
        let total: u64 = self.visits.iter().sum();
        if total == 0 {
            return 1.0;
        }
        let singletons = self.visits.iter().filter(|&&c| c == 1).count() as f64;
        singletons / total as f64
    }

    /// Deepest accumulated bias over all basins.
    pub fn deepest(&self) -> f64 {
        self.v.iter().copied().fold(0.0, f64::max)
    }
}

impl<F: Fingerprint> Bias for BasinBias<F> {
    /// The fingerprint itself is the collective variable: identity, not a
    /// projection onto a chosen axis.
    fn cv(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.fingerprint.describe(x)
    }

    fn potential(&self, s: ArrayView1<f64>) -> f64 {
        self.lookup(s).map(|i| self.v[i]).unwrap_or(0.0)
    }

    fn deposit(&mut self, s: ArrayView1<f64>, temp: f64) {
        let denom = (self.gamma - 1.0) * temp;
        match self.lookup(s) {
            Some(i) => {
                // Barducci well-tempered weight: deposition slows where the
                // bias is already deep, so a basin fills to a finite depth.
                let w = self.w0 * (-self.v[i] / denom).exp();
                self.v[i] += w;
                self.visits[i] += 1;
            }
            None => {
                self.centres.push(s.to_owned());
                self.v.push(self.w0);
                self.visits.push(1);
            }
        }
    }
}

#[cfg(test)]
mod basin_bias_tests {
    use super::*;
    use ndarray::array;

    fn tetra(scale: f64) -> Array1<f64> {
        array![
            0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale
        ]
    }

    #[test]
    fn sorted_pairs_is_permutation_invariant() {
        let fp = SortedPairs { n_points: 4 };
        let a = tetra(1.0);
        // Same point set, atoms 0 and 2 exchanged.
        let b = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let da = fp.describe(a.view());
        let db = fp.describe(b.view());
        for k in 0..da.len() {
            assert!((da[k] - db[k]).abs() < 1e-12, "component {k} differs");
        }
    }

    #[test]
    fn sorted_pairs_is_translation_invariant() {
        let fp = SortedPairs { n_points: 4 };
        let a = tetra(1.0);
        let mut b = a.clone();
        for i in 0..4 {
            b[3 * i] += 7.5;
            b[3 * i + 1] -= 2.25;
        }
        let da = fp.describe(a.view());
        let db = fp.describe(b.view());
        for k in 0..da.len() {
            assert!((da[k] - db[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn revisiting_a_basin_raises_only_that_basin() {
        let mut bias = BasinBias::new(SortedPairs { n_points: 4 }, 1e-6, 0.5, 5.0);
        let a = tetra(1.0);
        let b = tetra(2.0);
        let sa = bias.cv(a.view());
        let sb = bias.cv(b.view());

        bias.deposit(sa.view(), 1.0);
        assert_eq!(bias.n_basins(), 1);
        assert!(bias.potential(sa.view()) > 0.0);
        assert_eq!(bias.potential(sb.view()), 0.0, "unrelated basin untouched");

        bias.deposit(sb.view(), 1.0);
        assert_eq!(bias.n_basins(), 2);
    }

    #[test]
    fn well_tempered_deposits_shrink() {
        let mut bias = BasinBias::new(SortedPairs { n_points: 4 }, 1e-6, 1.0, 2.0);
        let a = tetra(1.0);
        let s = bias.cv(a.view());
        let mut last = 0.0;
        let mut increments = Vec::new();
        for _ in 0..6 {
            bias.deposit(s.view(), 1.0);
            let now = bias.potential(s.view());
            increments.push(now - last);
            last = now;
        }
        for w in increments.windows(2) {
            assert!(w[1] < w[0], "well-tempered weights must decrease: {increments:?}");
        }
    }

    /// The reason this type exists. A grid bias keyed on one projection cannot
    /// distinguish two structures whose separation along that axis is below the
    /// deposition width, so biasing one also biases the other. Keyed on
    /// identity they stay separate however close the projection puts them.
    #[test]
    fn separates_basins_a_narrow_projection_would_merge() {
        let fp = SortedPairs { n_points: 4 };
        // Two genuinely different point sets whose mean pair distance, a
        // one-dimensional projection, is nearly identical.
        let a = tetra(1.0);
        let b = array![
            0.0, 0.0, 0.0, 1.02, 0.0, 0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 1.0
        ];
        let da = fp.describe(a.view());
        let db = fp.describe(b.view());
        let mean = |v: &Array1<f64>| v.iter().sum::<f64>() / v.len() as f64;
        assert!(
            (mean(&da) - mean(&db)).abs() < 1e-2,
            "the scalar projection should barely separate these"
        );

        let mut bias = BasinBias::new(SortedPairs { n_points: 4 }, 1e-3, 0.5, 5.0);
        bias.deposit(da.view(), 1.0);
        assert_eq!(bias.n_basins(), 1);
        assert_eq!(
            bias.potential(db.view()),
            0.0,
            "identity keying must leave the neighbouring basin unbiased"
        );
        bias.deposit(db.view(), 1.0);
        assert_eq!(bias.n_basins(), 2, "these are two basins, not one");
    }

    #[test]
    fn missing_mass_tracks_singletons() {
        let mut bias = BasinBias::new(SortedPairs { n_points: 4 }, 1e-6, 0.5, 5.0);
        assert_eq!(bias.missing_mass(), 1.0, "no visits yet");
        let a = bias.cv(tetra(1.0).view());
        let b = bias.cv(tetra(2.0).view());
        bias.deposit(a.view(), 1.0);
        bias.deposit(b.view(), 1.0);
        assert!((bias.missing_mass() - 1.0).abs() < 1e-12, "both seen once");
        bias.deposit(a.view(), 1.0);
        assert!(bias.missing_mass() < 1.0, "a repeat lowers the missing mass");
    }
}
