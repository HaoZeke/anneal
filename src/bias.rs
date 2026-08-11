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
    /// The prune is exact, so it has to agree with an exhaustive scan on every
    /// query. A metric that reports no key is the exhaustive scan, which makes
    /// this a comparison of the two paths through the same code on the same
    /// data.
    #[test]
    fn the_triangle_prune_agrees_with_an_exhaustive_scan() {
        /// The descriptor is the state, so the test exercises the index and
        /// not a fingerprint.
        struct RawFingerprint;
        impl Fingerprint for RawFingerprint {
            fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
                x.to_owned()
            }
        }

        /// Euclidean distance with the key suppressed.
        struct NoKey;
        impl BasinMetric for NoKey {
            fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
                EuclideanMetric.distance(a, b)
            }
        }

        let mut rng = 88172645463325252u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        let dim = 24;
        let points: Vec<Array1<f64>> = (0..300)
            .map(|_| Array1::from((0..dim).map(|_| next() * 3.0).collect::<Vec<_>>()))
            .collect();

        for radius in [0.05, 0.4, 1.2, 3.0] {
            let mut fast = BasinIndex::new(RawFingerprint, radius);
            let mut slow = BasinIndex::new(RawFingerprint, radius).with_metric(Box::new(NoKey));
            for p in &points {
                let a = fast.basin_of(p.view());
                let b = slow.basin_of(p.view());
                assert_eq!(
                    a, b,
                    "radius {radius}: pruned scan gave {a}, exhaustive {b}"
                );
            }
            assert_eq!(fast.n_basins(), slow.n_basins(), "radius {radius}");
        }
    }

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
    index: BasinIndex<F>,
    w0: f64,
    gamma: f64,
    v: Vec<f64>,
}

/// Which basin a state is in, with no potential attached.
///
/// The identity half of [`BasinBias`], split out because more than one
/// mechanism needs to ask "have I been here before" and only one of them
/// answers by depositing. History-conditioned escape uses the same rule to
/// decide how hard to push next, and reading that off a bias would tie the two
/// together: under replica exchange each rung owns its own bias, so the basin
/// numbering of one rung means nothing in another, while a controller
/// following one chain needs a numbering that outlives the swap.
pub struct BasinIndex<F: Fingerprint> {
    fingerprint: F,
    metric: Box<dyn BasinMetric>,
    merge_radius: f64,
    centres: Vec<Array1<f64>>,
    /// Triangle-inequality keys, parallel to `centres`.
    keys: Vec<Option<f64>>,
    visits: Vec<u64>,
}

impl<F: Fingerprint> BasinIndex<F> {
    /// Index over `fingerprint`, calling two states the same within
    /// `merge_radius`.
    pub fn new(fingerprint: F, merge_radius: f64) -> Self {
        assert!(merge_radius > 0.0, "merge_radius must be > 0");
        Self {
            fingerprint,
            metric: Box::new(EuclideanMetric),
            merge_radius,
            centres: Vec::new(),
            keys: Vec::new(),
            visits: Vec::new(),
        }
    }

    /// Replaces the distance used to decide sameness.
    pub fn with_metric(mut self, metric: Box<dyn BasinMetric>) -> Self {
        self.metric = metric;
        // Keys belong to the metric that produced them, so any already held
        // are recomputed rather than carried into a different distance.
        self.keys = self
            .centres
            .iter()
            .map(|c| self.metric.key(c.view()))
            .collect();
        self
    }

    /// Descriptor of a state.
    pub fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.fingerprint.describe(x)
    }

    /// Index of the nearest registered basin within `merge_radius`.
    ///
    /// Most recently added first, returning the first centre inside the radius
    /// rather than scanning for the nearest.
    ///
    /// Both parts matter once the metric costs anything. A chain revisits the
    /// basin it is already in far more often than any other, measured at
    /// roughly nineteen proposals in twenty near a deep minimum, so the recent
    /// end is where the answer almost always is. And a merge radius asks
    /// whether any centre is within it, not which is closest, so the scan can
    /// stop at the first hit.
    ///
    /// With Euclidean distance the exhaustive scan was free and the
    /// distinction did not show. With a shape distance at milliseconds per
    /// comparison it is the difference between one call and one per basin, and
    /// the exhaustive version did not finish an LJ38 run.
    pub fn lookup(&self, d: ArrayView1<f64>) -> Option<usize> {
        let dk = self.metric.key(d);
        for (i, c) in self.centres.iter().enumerate().rev() {
            if c.len() != d.len() {
                continue;
            }
            // Exact, not a heuristic: the reverse triangle inequality says a
            // centre whose key differs by more than the radius is further away
            // than the radius, so skipping it cannot change the answer.
            if let (Some(k), Some(ck)) = (dk, self.keys[i]) {
                if (k - ck).abs() > self.merge_radius {
                    continue;
                }
            }
            if self.metric.distance_bounded(c.view(), d, self.merge_radius) <= self.merge_radius {
                return Some(i);
            }
        }
        None
    }

    /// Registers a descriptor as a new basin and returns its index.
    pub fn push(&mut self, d: Array1<f64>) -> usize {
        self.keys.push(self.metric.key(d.view()));
        self.centres.push(d);
        self.visits.push(0);
        self.centres.len() - 1
    }

    /// Records another visit to `i`.
    pub fn bump(&mut self, i: usize) {
        self.visits[i] += 1;
    }

    /// Index of the basin holding `x`, opening one if it is new, and counting
    /// the visit.
    pub fn basin_of(&mut self, x: ArrayView1<f64>) -> usize {
        let d = self.describe(x);
        let i = match self.lookup(d.view()) {
            Some(i) => i,
            None => self.push(d),
        };
        self.bump(i);
        i
    }

    /// Number of distinct basins registered so far.
    pub fn n_basins(&self) -> usize {
        self.centres.len()
    }

    /// Times basin `i` has been recorded.
    pub fn visits(&self, i: usize) -> u64 {
        self.visits.get(i).copied().unwrap_or(0)
    }

    /// Sets the distance below which two descriptors are one basin.
    pub fn set_merge_radius(&mut self, radius: f64) {
        assert!(radius > 0.0, "merge_radius must be > 0");
        self.merge_radius = radius;
    }

    /// Current merge radius.
    pub fn merge_radius(&self) -> f64 {
        self.merge_radius
    }
}

/// How far apart two descriptors are, for deciding whether they are one basin.
///
/// Separate from [`Fingerprint`] because the descriptor and the notion of
/// sameness are separate choices, and getting the second wrong is what a
/// merge radius in descriptor space does. A sorted distance spectrum compared
/// by Euclidean distance needs a threshold with no physical meaning, found
/// empirically and not transferable between system sizes; the same descriptor
/// compared by an optimal-permutation shape distance needs a length, which
/// transfers.
pub trait BasinMetric: Send + Sync {
    /// Distance between two descriptors. Must be symmetric and vanish on
    /// identical inputs.
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64;

    /// A scalar obeying `|key(a) - key(b)| <= distance(a, b)`, when the metric
    /// has one.
    ///
    /// The reverse triangle inequality turns a scan into a subtraction for
    /// every centre that cannot possibly be within the radius, which on a
    /// 98-point cluster is nearly all of them: the descriptor there has 4753
    /// entries, so one Euclidean comparison costs about as much as a
    /// Lennard-Jones gradient, and a chain that has opened twenty thousand
    /// basins pays that per centre on every miss.
    ///
    /// `None` for metrics with no such quantity, notably shape distances that
    /// minimise over permutations and rotations, where the scan stays exact
    /// and exhaustive.
    fn key(&self, _v: ArrayView1<f64>) -> Option<f64> {
        None
    }

    /// Distance, allowed to stop once it is certain to exceed `bound`.
    ///
    /// A caller asking whether a centre is within the merge radius does not
    /// need the distance to a centre that is not.
    fn distance_bounded(&self, a: ArrayView1<f64>, b: ArrayView1<f64>, _bound: f64) -> f64 {
        self.distance(a, b)
    }
}

/// Ordinary Euclidean distance between descriptors.
#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanMetric;

impl BasinMetric for EuclideanMetric {
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// The Euclidean norm, which obeys the reverse triangle inequality.
    fn key(&self, v: ArrayView1<f64>) -> Option<f64> {
        Some(v.iter().map(|x| x * x).sum::<f64>().sqrt())
    }

    fn distance_bounded(&self, a: ArrayView1<f64>, b: ArrayView1<f64>, bound: f64) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let limit = bound * bound;
        let mut acc = 0.0;
        // Checked in blocks rather than per element: the branch costs more
        // than the arithmetic on a descriptor whose leading entries are nearly
        // equal for every pair of structures, which is what a sorted distance
        // spectrum is.
        for chunk in 0..a.len().div_ceil(32) {
            let lo = chunk * 32;
            let hi = (lo + 32).min(a.len());
            for k in lo..hi {
                let d = a[k] - b[k];
                acc += d * d;
            }
            if acc > limit {
                return f64::INFINITY;
            }
        }
        acc.sqrt()
    }
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

/// Sorted per-point pair energies of a flattened `(n, 3)` point set.
///
/// `E(i) = sum_{j != i} 4 [ (1/r_ij)^12 - (1/r_ij)^6 ]`, sorted. Invariant to
/// permutation by the sort, and to rigid motions because it is built from
/// distances, so it has the same symmetries as [`SortedPairs`] and the same
/// cost, one pass over the pairs.
///
/// What it adds is that it is energetic. A sorted distance spectrum says how
/// far apart the points are; this says how well each one is bound, so two
/// structures with similar distance statistics and different coordination
/// separate. That matters because basin identity is what the merge radius is
/// measuring, and the radius is sharply sensitive: at 75 points, 0.7 in the
/// distance spectrum gives 13 seeds in 24 while 0.95 gives 0 in 8. A
/// descriptor whose distances between genuinely different structures are
/// better separated is what would widen that band.
///
/// A richer alternative was measured and rejected on cost. The eigenvalue
/// spectrum of the pairwise distance matrix is a stronger invariant, and it
/// needs an order `n^3` decomposition per hop against the order `n^2` of the
/// thirty energy evaluations a hop already spends, so it would cut the hop
/// count roughly threefold. Hops are the scarce resource in this search.
pub struct SiteEnergies {
    /// Points per state; the state length must be `3 * n_points`.
    pub n_points: usize,
}

impl Fingerprint for SiteEnergies {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let n = self.n_points;
        let mut e = vec![0.0_f64; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = x[3 * i] - x[3 * j];
                let dy = x[3 * i + 1] - x[3 * j + 1];
                let dz = x[3 * i + 2] - x[3 * j + 2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 <= 0.0 {
                    continue;
                }
                let s6 = 1.0 / (r2 * r2 * r2);
                let v = 4.0 * (s6 * s6 - s6);
                e[i] += v;
                e[j] += v;
            }
        }
        e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Array1::from(e)
    }
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
        Self {
            index: BasinIndex::new(fingerprint, merge_radius),
            w0,
            gamma,
            v: Vec::new(),
        }
    }

    /// Sets the distance below which two descriptors are one basin.
    ///
    /// Exposed because this is a schedule rather than a setting. Lee, Lee and
    /// Scheraga show the threshold plays the role of a temperature and is
    /// annealed from wide to narrow, and their method solves the cluster sizes
    /// a fixed threshold does not. See [`crate::diversity`].
    pub fn set_merge_radius(&mut self, radius: f64) {
        self.index.set_merge_radius(radius);
    }

    /// Current merge radius.
    pub fn merge_radius(&self) -> f64 {
        self.index.merge_radius()
    }

    /// The identity half, for a mechanism that keys on basins without
    /// depositing.
    pub fn index(&self) -> &BasinIndex<F> {
        &self.index
    }

    /// Sets the height deposited per revisit.
    ///
    /// Exposed because the right value is a property of the landscape rather
    /// than of this type: it has to be commensurate with the energy cost of
    /// leaving a basin, and a height above that cost empties a basin on a
    /// single revisit instead of filling it.
    pub fn set_height(&mut self, w0: f64) {
        self.w0 = w0;
    }

    /// Current deposit height.
    pub fn height(&self) -> f64 {
        self.w0
    }

    /// Replaces how sameness is measured.
    ///
    /// The descriptor and the notion of sameness are separate choices. Keying
    /// on a sorted distance spectrum compared by Euclidean distance needs a
    /// threshold with no physical meaning: measured on this crate's cluster
    /// driver, LJ38 solves 1 seed in 8 that way. Comparing the same structures
    /// by optimal-permutation shape distance makes the threshold a length.
    pub fn with_metric(mut self, metric: Box<dyn BasinMetric>) -> Self {
        self.index = self.index.with_metric(metric);
        self
    }

    /// Chatterjee–Voter AS-KMC: an intra-well hop after the well has
    /// been seen N_f times is a frequent superbasin process. Its rate
    /// is what gets lowered. `nf_height` is N_f times the deposit
    /// increment (the same number as recommended `height_revisits * w0`).
    pub fn frequent_superbasin(
        &self,
        s_old: ArrayView1<f64>,
        s_new: ArrayView1<f64>,
        nf_height: f64,
    ) -> bool {
        match (self.lookup(s_old), self.lookup(s_new)) {
            (Some(i), Some(j)) if i == j => self.v.get(i).copied().unwrap_or(0.0) >= nf_height,
            _ => false,
        }
    }

    /// Index of the nearest registered basin within `merge_radius`.
    fn lookup(&self, d: ArrayView1<f64>) -> Option<usize> {
        self.index.lookup(d)
    }

    /// Number of distinct basins registered so far.
    pub fn n_basins(&self) -> usize {
        self.index.n_basins()
    }

    /// Good-Turing missing mass: the share of basins seen exactly once, which
    /// estimates the probability that the next visit opens a new one.
    pub fn missing_mass(&self) -> f64 {
        let total: u64 = self.index.visits.iter().sum();
        if total == 0 {
            return 1.0;
        }
        let singletons = self.index.visits.iter().filter(|&&c| c == 1).count() as f64;
        singletons / total as f64
    }

    /// Deepest accumulated bias over all basins.
    pub fn deepest(&self) -> f64 {
        self.v.iter().copied().fold(0.0, f64::max)
    }

    /// Merge a packing well found by another chain.
    ///
    /// The descriptor is already a fingerprint (unit mean SOAP), not raw
    /// coordinates. Height is kept if this well is new, or raised to the
    /// remote value if the same packing is already local.
    pub fn import_well(&mut self, descriptor: Array1<f64>, height: f64) {
        if !height.is_finite() || height < 0.0 || descriptor.is_empty() {
            return;
        }
        match self.lookup(descriptor.view()) {
            Some(i) => {
                if height > self.v[i] {
                    self.v[i] = height;
                }
            }
            None => {
                self.index.push(descriptor);
                self.v.push(height);
            }
        }
    }
}

impl<F: Fingerprint> Bias for BasinBias<F> {
    /// The fingerprint itself is the collective variable: identity, not a
    /// projection onto a chosen axis.
    fn cv(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.index.describe(x)
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
                self.index.bump(i);
            }
            None => {
                self.index.push(s.to_owned());
                self.v.push(self.w0);
                let i = self.index.n_basins() - 1;
                self.index.bump(i);
            }
        }
    }
}

/// Deposit height estimated from the escape gaps a chain actually sees.
///
/// A per-basin bias only moves a chain once the bias in the occupied basin
/// exceeds the energy cost of leaving it, so the deposit height and that cost
/// have to be commensurate. Setting the height by hand does not survive contact
/// with a real landscape: on a 75-point Lennard-Jones cluster the cheapest
/// escape from the structure basin hopping plateaus at costs 0.0906 and the
/// tenth percentile costs 0.1831, against a hand-set default of 0.25, so one
/// deposit clears both and the chain abandons a basin after a single revisit
/// rather than filling what it is sitting in. The same default at a different
/// size, or on a different potential, is wrong in whichever direction that
/// landscape happens to run.
///
/// The gaps are observable: every rejected uphill proposal is a sample from the
/// escape distribution. This tracks a low quantile of those samples and sets
/// the height so that a chosen number of revisits clears it, which makes the
/// bias self-scaling in the units of the landscape rather than of the author.
///
/// The estimate is a P-square style quantile tracker rather than a stored
/// history, so the cost does not grow with the run.
pub struct AdaptiveHeight {
    /// Quantile of the escape gap the bias aims to clear, in [0, 1).
    pub quantile: f64,
    /// Revisits a basin should take before that gap is cleared.
    pub revisits: f64,
    /// Current quantile estimate.
    estimate: f64,
    /// Samples seen, which sets the step size and the warm-up.
    count: u64,
    /// Height used until enough samples have arrived.
    prior: f64,
    /// Samples required before the estimate replaces the prior.
    warmup: u64,
    /// Running mean gap, used only to scale the update step.
    scale: f64,
}

impl AdaptiveHeight {
    /// Tracker aiming at `quantile`, clearing it in `revisits` deposits.
    pub fn new(quantile: f64, revisits: f64, prior: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&quantile),
            "quantile must lie in [0, 1), got {quantile}"
        );
        assert!(revisits > 0.0, "revisits must be positive, got {revisits}");
        Self {
            quantile,
            revisits,
            estimate: prior * revisits,
            count: 0,
            prior,
            warmup: 32,
            scale: prior * revisits,
        }
    }

    /// Records one observed escape gap, which must be positive.
    ///
    /// A non-positive gap is a downhill move and says nothing about the cost of
    /// leaving, so it is ignored rather than dragging the estimate to zero.
    pub fn observe(&mut self, gap: f64) {
        if !(gap > 0.0) || !gap.is_finite() {
            return;
        }
        self.count += 1;
        let n = self.count as f64;
        self.scale += (gap - self.scale) / n;
        // Robbins-Monro stochastic approximation of the quantile. The step is
        // up by `q` when the sample lies above the estimate and down by
        // `1 - q` when it lies below, which is the pairing that puts the fixed
        // point at the quantile: there a fraction `1 - q` of samples lies above
        // and `q` below, so the expected drift is `(1 - q) q - q (1 - q) = 0`.
        // Weighting the other way round leaves a drift of `0.81` against `0.01`
        // at the tenth percentile and the estimate climbs away from it.
        let step = self.scale.max(1e-12) / n.sqrt();
        if gap > self.estimate {
            self.estimate += step * self.quantile;
        } else {
            self.estimate -= step * (1.0 - self.quantile);
        }
        self.estimate = self.estimate.max(0.0);
    }

    /// Height to deposit per revisit.
    pub fn height(&self) -> f64 {
        if self.count < self.warmup {
            return self.prior;
        }
        (self.estimate / self.revisits).max(1e-12)
    }

    /// Current estimate of the targeted escape-gap quantile.
    pub fn gap_estimate(&self) -> f64 {
        self.estimate
    }

    /// Samples recorded.
    pub fn samples(&self) -> u64 {
        self.count
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
    fn site_energies_are_permutation_invariant() {
        let f = SiteEnergies { n_points: 4 };
        let a = tetra(1.1);
        let mut b = Array1::zeros(12);
        // Relabelled: 0 -> 2, 1 -> 0, 2 -> 3, 3 -> 1.
        let perm = [2usize, 0, 3, 1];
        for (i, p) in perm.iter().enumerate() {
            for k in 0..3 {
                b[3 * p + k] = a[3 * i + k];
            }
        }
        let da = f.describe(a.view());
        let db = f.describe(b.view());
        for (p, q) in da.iter().zip(db.iter()) {
            assert!((p - q).abs() < 1e-12, "{p} against {q}");
        }
    }

    #[test]
    fn site_energies_are_invariant_to_rigid_motions() {
        let f = SiteEnergies { n_points: 4 };
        let a = tetra(1.1);
        let da = f.describe(a.view());
        // Translate, then rotate a quarter turn about z.
        let mut b = a.clone();
        for i in 0..4 {
            b[3 * i] += 3.0;
            b[3 * i + 1] -= 1.5;
            b[3 * i + 2] += 0.25;
        }
        let mut c = Array1::zeros(12);
        for i in 0..4 {
            c[3 * i] = -b[3 * i + 1];
            c[3 * i + 1] = b[3 * i];
            c[3 * i + 2] = b[3 * i + 2];
        }
        let dc = f.describe(c.view());
        for (p, q) in da.iter().zip(dc.iter()) {
            assert!((p - q).abs() < 1e-10, "{p} against {q}");
        }
    }

    /// The reason to have it: it reports how well each point is bound, so
    /// structures that differ in coordination separate even when their distance
    /// statistics are close.
    #[test]
    fn site_energies_separate_structures_by_coordination() {
        let n = 5;
        let pairs = SortedPairs { n_points: n };
        let sites = SiteEnergies { n_points: n };
        // A square with a point at its centre, against the same square with the
        // fifth point outside. The pair distances overlap heavily; the
        // coordination does not.
        let mut a = Array1::<f64>::zeros(3 * n);
        let mut b = Array1::<f64>::zeros(3 * n);
        for (i, (px, py)) in [(0.0, 0.0), (1.1, 0.0), (1.1, 1.1), (0.0, 1.1)]
            .iter()
            .enumerate()
        {
            for (t, v) in [(&mut a, 0), (&mut b, 0)] {
                let _ = v;
                t[3 * i] = *px;
                t[3 * i + 1] = *py;
            }
        }
        a[3 * 4] = 0.55;
        a[3 * 4 + 1] = 0.55;
        b[3 * 4] = 2.2;
        b[3 * 4 + 1] = 0.55;

        let dp: f64 = pairs
            .describe(a.view())
            .iter()
            .zip(pairs.describe(b.view()).iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        let ds: f64 = sites
            .describe(a.view())
            .iter()
            .zip(sites.describe(b.view()).iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        assert!(
            ds > dp,
            "site energies separated the two by {ds} where distances gave {dp}"
        );
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
            assert!(
                w[1] < w[0],
                "well-tempered weights must decrease: {increments:?}"
            );
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
        let b = array![0.0, 0.0, 0.0, 1.02, 0.0, 0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 1.0];
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
        assert!(
            bias.missing_mass() < 1.0,
            "a repeat lowers the missing mass"
        );
    }
}

#[cfg(test)]
mod adaptive_tests {
    use super::AdaptiveHeight;

    /// Deterministic exponential-ish stream, avoiding a dependency on an rng.
    fn gaps(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let u = ((i * 7919 + 13) % 1000) as f64 / 1000.0;
                -(1.0 - u).ln() * 2.5
            })
            .collect()
    }

    #[test]
    fn tracks_a_low_quantile_of_the_observed_gaps() {
        let mut h = AdaptiveHeight::new(0.1, 4.0, 0.25);
        let sample = gaps(4000);
        for g in &sample {
            h.observe(*g);
        }
        let mut sorted = sample.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let truth = sorted[(0.1 * sorted.len() as f64) as usize];
        let est = h.gap_estimate();
        assert!(
            (est - truth).abs() < 0.5 * truth.max(0.2),
            "estimate {est} far from the tenth percentile {truth}"
        );
    }

    #[test]
    fn holds_the_prior_until_warmed_up() {
        let mut h = AdaptiveHeight::new(0.1, 4.0, 0.25);
        h.observe(3.0);
        assert_eq!(h.height(), 0.25, "an estimate from one sample is noise");
    }

    #[test]
    fn clears_the_estimated_gap_in_the_requested_revisits() {
        let mut h = AdaptiveHeight::new(0.1, 5.0, 0.25);
        for g in gaps(2000) {
            h.observe(g);
        }
        let total = h.height() * 5.0;
        assert!(
            (total - h.gap_estimate()).abs() < 1e-9,
            "five deposits of {} should reach {}",
            h.height(),
            h.gap_estimate()
        );
    }

    #[test]
    fn ignores_downhill_and_non_finite_samples() {
        let mut h = AdaptiveHeight::new(0.1, 4.0, 0.25);
        for _ in 0..100 {
            h.observe(-1.0);
            h.observe(0.0);
            h.observe(f64::NAN);
        }
        assert_eq!(h.samples(), 0);
        assert_eq!(h.height(), 0.25);
    }
}
