//! Landfold: a sketch-map embedding of a minima book.
//!
//! A minima book is a set of quenched structures. [`book_descriptors`]
//! reads each one as a DECAF class histogram on a shared codebook, and two
//! minima are compared by the L1 distance of their histograms. [`landfold`]
//! maps any set of descriptor rows to the plane in four steps, the hybrid
//! `docs/landfold-survey.org` argues for:
//!
//! 1. Landmarks by farthest-point sampling (Gonzalez 1985), the first one
//!    drawn from the seed. Each landmark is weighted by the number of rows
//!    in its Voronoi cell raised to [`LandfoldConfig::weight_power`]. One is
//!    the reference `dimlandmark -w`. The default one half keeps sparse
//!    regions from being outvoted by the dense funnel bottoms.
//! 2. Classical MDS (Torgerson 1952) of the landmark distances. Block
//!    subspace iteration with Rayleigh-Ritz extraction keeps both axes of a
//!    repeated eigenvalue, which single-vector power iteration does not.
//! 3. The sketch-map stress of Ceriotti, Tribello and Parrinello (PNAS
//!    2011) on the landmarks. It is reached from the metric stress by the
//!    mixing homotopy of the reference `sketch-map.sh`, with a pointwise
//!    grid search between stages to leave local minima, and L-BFGS within
//!    each stage.
//! 4. Every other row is placed by minimising its own sketch-map stress
//!    against the fixed landmarks, as the reference `dimproj` does. The
//!    fitted [`LandfoldMap`] places rows that were never in the book.
//!
//! The switch on both sides is
//! \(s(r)=1-(1+(2^{a/b}-1)(r/\sigma)^a)^{-b/a}\), with the high-dimensional
//! pair \((A, B)\) and the low-dimensional pair \((a, b)\) set apart.
//!
//! Every random number comes from the seed, and every parallel loop writes
//! its own slot and is summed in index order, so a seed fixes the map bit
//! for bit at any thread count.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::methods::warm_lbfgs::WarmLbfgs;

/// Default landmark count.
pub const DEFAULT_LANDMARKS: usize = 400;
/// Default high-dimensional switch exponents \((A, B)\).
pub const DEFAULT_HIGH: [f64; 2] = [2.0, 4.0];
/// Default low-dimensional switch exponents \((a, b)\).
pub const DEFAULT_LOW: [f64; 2] = [2.0, 4.0];
/// Default fraction of the median landmark distance used as \(\sigma\).
pub const DEFAULT_SIGMA_SCALE: f64 = 0.5;
/// Default grid points per axis of the pointwise global search.
pub const DEFAULT_GRID: usize = 21;
/// Default L-BFGS iterations per optimisation stage.
pub const DEFAULT_ITERATIONS: usize = 200;
/// Default weights of the metric stress, one optimisation stage each.
pub const DEFAULT_MIX: [f64; 5] = [1.0, 0.5, 0.25, 0.1, 0.0];
/// Default exponent on the Voronoi population of a landmark.
pub const DEFAULT_WEIGHT_POWER: f64 = 0.5;
/// Landmarks nearest a projected row that seed its local search.
const PROJECTION_SEEDS: usize = 6;
/// Subspace width of the landmark eigensolve.
const MDS_BLOCK: usize = 8;

/// Why a book cannot be folded.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum LandfoldError {
    /// No rows to embed.
    #[error("landfold needs at least one row")]
    Empty,
    /// A descriptor or coordinate is NaN or infinite.
    #[error("row {index} is not finite")]
    NonFinite {
        /// Zero-based row index.
        index: usize,
    },
    /// A structure has no DECAF class histogram.
    #[error("structure {index} has no DECAF histogram (fewer than 13 atoms or not 3N values)")]
    NoDescriptor {
        /// Zero-based structure index.
        index: usize,
    },
    /// A configuration value is out of range.
    #[error("invalid landfold configuration: {0}")]
    Config(String),
    /// A minima book could not be read.
    #[error("{0}")]
    Read(String),
}

/// Distance between descriptor rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LandfoldMetric {
    /// Sum of absolute differences: the DECAF packing distance.
    #[default]
    L1,
    /// Euclidean distance.
    L2,
}

impl LandfoldMetric {
    /// Distance between two rows. A shorter row is read as zero-padded.
    pub fn distance(self, a: &[f64], b: &[f64]) -> f64 {
        let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        let tail = &long[short.len()..];
        match self {
            Self::L1 => {
                short
                    .iter()
                    .zip(long)
                    .map(|(x, y)| (x - y).abs())
                    .sum::<f64>()
                    + tail.iter().map(|x| x.abs()).sum::<f64>()
            }
            Self::L2 => (short
                .iter()
                .zip(long)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                + tail.iter().map(|x| x * x).sum::<f64>())
            .sqrt(),
        }
    }
}

/// Sketch-map switch \(s(r)=1-(1+(2^{a/b}-1)(r/\sigma)^a)^{-b/a}\).
///
/// \(s(\sigma)=1/2\) for every \((a, b)\). The exponent \(a\) sets how
/// fast the switch leaves zero, \(b\) how fast it saturates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Switch {
    sigma: f64,
    a: f64,
    b: f64,
    inv_sigma: f64,
    c: f64,
    a_square: bool,
    exponent: f64,
    exponent_int: Option<i32>,
}

impl Switch {
    /// Switch of scale `sigma` and exponents `a`, `b`.
    pub fn new(sigma: f64, a: f64, b: f64) -> Result<Self, LandfoldError> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(LandfoldError::Config(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        if !(a.is_finite() && a > 1.0 && b.is_finite() && b > 0.0) {
            return Err(LandfoldError::Config(format!(
                "switch needs a > 1 and b > 0, got a = {a}, b = {b}"
            )));
        }
        let exponent = -b / a;
        let exponent_int =
            (exponent.fract() == 0.0 && exponent.abs() <= 64.0).then_some(exponent as i32);
        Ok(Self {
            sigma,
            a,
            b,
            inv_sigma: 1.0 / sigma,
            c: 2.0_f64.powf(a / b) - 1.0,
            a_square: a == 2.0,
            exponent,
            exponent_int,
        })
    }

    /// Scale \(\sigma\).
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Exponents \((a, b)\).
    pub fn exponents(&self) -> [f64; 2] {
        [self.a, self.b]
    }

    /// \(s(r)\).
    #[inline]
    pub fn value(&self, r: f64) -> f64 {
        self.value_and_slope(r).0
    }

    /// \(s(r)\) and \(s'(r)\).
    #[inline]
    pub fn value_and_slope(&self, r: f64) -> (f64, f64) {
        if r <= 0.0 {
            return (0.0, 0.0);
        }
        let t = r * self.inv_sigma;
        let (u, u_over_r) = if self.a_square {
            let u = self.c * t * t;
            (u, self.c * t * self.inv_sigma)
        } else {
            let u = self.c * t.powf(self.a);
            (u, u / r)
        };
        let base = 1.0 + u;
        let p = match self.exponent_int {
            Some(k) => base.powi(k),
            None => base.powf(self.exponent),
        };
        (1.0 - p, self.b * u_over_r * p / base)
    }
}

/// Settings of [`landfold`].
#[derive(Clone, Debug, PartialEq)]
pub struct LandfoldConfig {
    /// Landmark count. A smaller book is used whole.
    pub landmarks: usize,
    /// Seed of the first landmark and the eigensolve start block.
    pub seed: u64,
    /// High-dimensional \(\sigma\). `None` takes [`Self::sigma_scale`]
    /// times the median landmark distance.
    pub sigma: Option<f64>,
    /// Fraction of the median landmark distance used when `sigma` is unset.
    pub sigma_scale: f64,
    /// Low-dimensional \(\sigma\). `None` uses the high-dimensional one.
    pub sigma_low: Option<f64>,
    /// High-dimensional switch exponents \((A, B)\).
    pub high: [f64; 2],
    /// Low-dimensional switch exponents \((a, b)\).
    pub low: [f64; 2],
    /// Grid points per axis of the pointwise global search. Zero skips it.
    pub grid: usize,
    /// L-BFGS iterations per optimisation stage.
    pub iterations: usize,
    /// Weight of the metric stress in each stage. The last stage should be
    /// zero so the map minimises the sketch-map stress alone.
    pub mix: Vec<f64>,
    /// Landmark weight is its Voronoi population to this power. One is the
    /// reference `dimlandmark -w`, zero weights every landmark equally.
    pub weight_power: f64,
    /// Row distance.
    pub metric: LandfoldMetric,
}

impl Default for LandfoldConfig {
    fn default() -> Self {
        Self {
            landmarks: DEFAULT_LANDMARKS,
            seed: 0,
            sigma: None,
            sigma_scale: DEFAULT_SIGMA_SCALE,
            sigma_low: None,
            high: DEFAULT_HIGH,
            low: DEFAULT_LOW,
            grid: DEFAULT_GRID,
            iterations: DEFAULT_ITERATIONS,
            mix: DEFAULT_MIX.to_vec(),
            weight_power: DEFAULT_WEIGHT_POWER,
            metric: LandfoldMetric::L1,
        }
    }
}

impl LandfoldConfig {
    fn validate(&self) -> Result<(), LandfoldError> {
        if self.landmarks == 0 {
            return Err(LandfoldError::Config("landmarks must be positive".into()));
        }
        if !(self.sigma_scale.is_finite() && self.sigma_scale > 0.0) {
            return Err(LandfoldError::Config("sigma_scale must be positive".into()));
        }
        if !(self.weight_power.is_finite() && self.weight_power >= 0.0) {
            return Err(LandfoldError::Config(
                "weight_power must be non-negative".into(),
            ));
        }
        if self.mix.is_empty() {
            return Err(LandfoldError::Config("mix needs at least one stage".into()));
        }
        if self
            .mix
            .iter()
            .any(|m| !(m.is_finite() && (0.0..=1.0).contains(m)))
        {
            return Err(LandfoldError::Config("mix weights lie in [0, 1]".into()));
        }
        if self.grid == 1 {
            return Err(LandfoldError::Config(
                "grid is zero or at least two points per axis".into(),
            ));
        }
        Ok(())
    }
}

/// Farthest-point landmarks and every row's distance to each of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Landmarks {
    /// Row index of each landmark, in selection order.
    pub indices: Vec<usize>,
    /// Row-major `rows × landmarks` distances.
    pub distances: Vec<f64>,
}

impl Landmarks {
    /// Landmark count.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// True when no landmark was chosen.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Distances from row `i` to every landmark.
    pub fn row(&self, i: usize) -> &[f64] {
        let m = self.indices.len();
        &self.distances[i * m..(i + 1) * m]
    }
}

/// Greedy farthest-point sampling (Gonzalez 1985) from row `first`.
///
/// Each step takes the row farthest from every landmark so far, the lowest
/// index on a tie. Selection stops early once every row coincides with a
/// landmark, so a book of duplicates gets one landmark per distinct row.
pub fn farthest_point_landmarks(
    rows: &[Vec<f64>],
    count: usize,
    first: usize,
    metric: LandfoldMetric,
) -> Landmarks {
    let n = rows.len();
    if n == 0 || count == 0 {
        return Landmarks {
            indices: Vec::new(),
            distances: Vec::new(),
        };
    }
    let first = first.min(n - 1);
    let mut indices = vec![first];
    let mut columns: Vec<Vec<f64>> = Vec::new();
    let column = |k: usize| -> Vec<f64> {
        rows.par_iter()
            .map(|row| metric.distance(row, &rows[k]))
            .collect()
    };
    columns.push(column(first));
    let mut nearest = columns[0].clone();
    while indices.len() < count.min(n) {
        let mut best = 0usize;
        let mut best_d = f64::NEG_INFINITY;
        for (i, &d) in nearest.iter().enumerate() {
            if d > best_d {
                best_d = d;
                best = i;
            }
        }
        if best_d <= 0.0 {
            break;
        }
        indices.push(best);
        let next = column(best);
        for (held, &d) in nearest.iter_mut().zip(&next) {
            if d < *held {
                *held = d;
            }
        }
        columns.push(next);
    }
    let m = indices.len();
    let mut distances = vec![0.0; n * m];
    for (k, col) in columns.iter().enumerate() {
        for (i, &d) in col.iter().enumerate() {
            distances[i * m + k] = d;
        }
    }
    Landmarks { indices, distances }
}

/// Top two eigenpairs of the Torgerson Gram matrix of a distance matrix.
///
/// `dist` is row-major `m × m`. Returns the coordinates
/// \(\sqrt{\lambda_k}\,u_k\) as rows and \([\lambda_1, \lambda_2]\), each
/// clamped at zero. Block subspace iteration starts from a Gaussian block
/// drawn from `seed`, shifted so negative eigenvalues of a non-Euclidean
/// metric do not dominate, and extracts Ritz pairs from the block, so a
/// repeated eigenvalue keeps both of its axes.
pub fn classical_mds(dist: &[f64], m: usize, seed: u64) -> (Vec<[f64; 2]>, [f64; 2]) {
    assert_eq!(dist.len(), m * m, "classical_mds needs an m × m matrix");
    if m == 0 {
        return (Vec::new(), [0.0, 0.0]);
    }
    if m == 1 {
        return (vec![[0.0, 0.0]], [0.0, 0.0]);
    }
    let gram = torgerson_gram(dist, m);
    let p = MDS_BLOCK.min(m);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x6c61_6e64_666f_6c64);
    let mut q = Array2::<f64>::zeros((m, p));
    for v in q.iter_mut() {
        *v = rng.sample(StandardNormal);
    }
    orthonormalize(&mut q);
    let mut values = [0.0; 2];
    let mut vectors = Array2::<f64>::zeros((m, 2));
    let scale = gram.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    for _ in 0..500 {
        let z = gram_times(&gram, &q);
        let t = q.t().dot(&z);
        let t = (&t + &t.t()) * 0.5;
        let (theta, w) = crate::spectral::symmetric_eigen(t.view(), 100);
        let ritz_q = q.dot(&w);
        let ritz_z = z.dot(&w);
        let mut residual = 0.0_f64;
        for (slot, k) in (0..2.min(p)).map(|s| (s, p - 1 - s)) {
            values[slot] = theta[k];
            let r = &ritz_z.column(k) - &(&ritz_q.column(k) * theta[k]);
            residual = residual.max(r.dot(&r).sqrt());
            vectors.column_mut(slot).assign(&ritz_q.column(k));
        }
        if p == m || residual <= 1e-10 * scale {
            break;
        }
        let shift = (-theta[0]).max(0.0);
        let mut next = &ritz_z + &(&ritz_q * shift);
        orthonormalize(&mut next);
        q = next;
    }
    let mut xy = vec![[0.0; 2]; m];
    let mut spectrum = [0.0; 2];
    for k in 0..2.min(p) {
        let lambda = values[k].max(0.0);
        spectrum[k] = lambda;
        let mut u = vectors.column(k).to_owned();
        let flip = u.iter().fold(
            0.0_f64,
            |held, &v| if v.abs() > held.abs() { v } else { held },
        );
        if flip < 0.0 {
            u.mapv_inplace(|v| -v);
        }
        let root = lambda.sqrt();
        for i in 0..m {
            xy[i][k] = root * u[i];
        }
    }
    (xy, spectrum)
}

fn torgerson_gram(dist: &[f64], m: usize) -> Array2<f64> {
    let mut d2 = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            let d = 0.5 * (dist[i * m + j] + dist[j * m + i]);
            d2[[i, j]] = d * d;
        }
    }
    let row_mean: Vec<f64> = (0..m).map(|i| d2.row(i).sum() / m as f64).collect();
    let total = row_mean.iter().sum::<f64>() / m as f64;
    let mut gram = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            gram[[i, j]] = -0.5 * (d2[[i, j]] - row_mean[i] - row_mean[j] + total);
        }
    }
    gram
}

fn gram_times(gram: &Array2<f64>, q: &Array2<f64>) -> Array2<f64> {
    let (m, p) = q.dim();
    let rows: Vec<Vec<f64>> = (0..m)
        .into_par_iter()
        .map(|i| {
            let g = gram.row(i);
            (0..p).map(|c| g.dot(&q.column(c))).collect()
        })
        .collect();
    let mut out = Array2::<f64>::zeros((m, p));
    for (i, row) in rows.into_iter().enumerate() {
        for (c, v) in row.into_iter().enumerate() {
            out[[i, c]] = v;
        }
    }
    out
}

/// Modified Gram-Schmidt, twice. A column that collapses is replaced by a
/// unit vector orthogonal to the others.
fn orthonormalize(q: &mut Array2<f64>) {
    let (m, p) = q.dim();
    for c in 0..p {
        for _ in 0..2 {
            for prev in 0..c {
                let dot = q.column(prev).dot(&q.column(c));
                let prev_col = q.column(prev).to_owned();
                let mut col = q.column_mut(c);
                col.scaled_add(-dot, &prev_col);
            }
        }
        let norm = q.column(c).dot(&q.column(c)).sqrt();
        if norm > 1e-300 {
            q.column_mut(c).mapv_inplace(|v| v / norm);
        } else {
            let mut col = q.column_mut(c);
            col.fill(0.0);
            col[c % m] = 1.0;
        }
    }
}

/// Fitted landmark map: what is needed to place a new row.
#[derive(Clone, Debug, PartialEq)]
pub struct LandfoldMap {
    landmark_rows: Vec<Vec<f64>>,
    landmark_xy: Vec<[f64; 2]>,
    weights: Vec<f64>,
    high: Switch,
    low: Switch,
    metric: LandfoldMetric,
}

impl LandfoldMap {
    /// Landmark coordinates on the plane.
    pub fn landmark_xy(&self) -> &[[f64; 2]] {
        &self.landmark_xy
    }

    /// Landmark weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// High-dimensional switch.
    pub fn high(&self) -> Switch {
        self.high
    }

    /// Low-dimensional switch.
    pub fn low(&self) -> Switch {
        self.low
    }

    /// Place one descriptor row on the plane.
    pub fn project(&self, row: &[f64]) -> [f64; 2] {
        let distances: Vec<f64> = self
            .landmark_rows
            .iter()
            .map(|landmark| self.metric.distance(row, landmark))
            .collect();
        self.project_distances(&distances)
    }

    /// Place a row given its distance to every landmark, in landmark order.
    ///
    /// Minimises \(\sum_k w_k\,(s_D(\delta_k) - s_d(|y - Y_k|))^2\) by
    /// L-BFGS from the best of the nearest landmarks and their weighted
    /// centroid, as `dimproj` minimises from its grid.
    pub fn project_distances(&self, distances: &[f64]) -> [f64; 2] {
        assert_eq!(
            distances.len(),
            self.landmark_xy.len(),
            "one distance per landmark"
        );
        let targets: Vec<f64> = distances.iter().map(|&d| self.high.value(d)).collect();
        let objective = PointStress {
            targets: &targets,
            raw: distances,
            others: &self.landmark_xy,
            weights: &self.weights,
            low: &self.low,
            mix: 0.0,
            skip: None,
        };
        let mut order: Vec<usize> = (0..distances.len()).collect();
        order.sort_by(|&a, &b| distances[a].total_cmp(&distances[b]).then(a.cmp(&b)));
        let seeds = &order[..PROJECTION_SEEDS.min(order.len())];
        let mut candidates: Vec<[f64; 2]> = seeds.iter().map(|&k| self.landmark_xy[k]).collect();
        let mut centroid = [0.0; 2];
        let mut total = 0.0;
        for &k in seeds {
            let w = 1.0 / (distances[k] + 1e-12);
            centroid[0] += w * self.landmark_xy[k][0];
            centroid[1] += w * self.landmark_xy[k][1];
            total += w;
        }
        if total > 0.0 {
            candidates.push([centroid[0] / total, centroid[1] / total]);
        }
        let mut start = candidates[0];
        let mut start_value = objective.value(start);
        for &c in &candidates[1..] {
            let v = objective.value(c);
            if v < start_value {
                start = c;
                start_value = v;
            }
        }
        objective.relax(start)
    }
}

/// Per-row stress against fixed partners.
struct PointStress<'a> {
    targets: &'a [f64],
    raw: &'a [f64],
    others: &'a [[f64; 2]],
    weights: &'a [f64],
    low: &'a Switch,
    mix: f64,
    skip: Option<usize>,
}

impl PointStress<'_> {
    fn value(&self, y: [f64; 2]) -> f64 {
        let mut total = 0.0;
        for k in 0..self.others.len() {
            if Some(k) == self.skip {
                continue;
            }
            let dx = y[0] - self.others[k][0];
            let dy = y[1] - self.others[k][1];
            let d = (dx * dx + dy * dy).sqrt();
            let g = self.low.value(d);
            let e = self.targets[k] - g;
            let mut term = (1.0 - self.mix) * e * e;
            if self.mix > 0.0 {
                let r = self.raw[k] - d;
                term += self.mix * r * r;
            }
            total += self.weights[k] * term;
        }
        total
    }

    fn value_and_gradient(&self, y: [f64; 2]) -> (f64, [f64; 2]) {
        let mut total = 0.0;
        let mut grad = [0.0; 2];
        for k in 0..self.others.len() {
            if Some(k) == self.skip {
                continue;
            }
            let dx = y[0] - self.others[k][0];
            let dy = y[1] - self.others[k][1];
            let d = (dx * dx + dy * dy).sqrt();
            let (g, slope) = self.low.value_and_slope(d);
            let e = self.targets[k] - g;
            let w = self.weights[k];
            let mut term = (1.0 - self.mix) * e * e;
            let mut radial = -2.0 * (1.0 - self.mix) * e * slope;
            if self.mix > 0.0 {
                let r = self.raw[k] - d;
                term += self.mix * r * r;
                radial -= 2.0 * self.mix * r;
            }
            total += w * term;
            if d > 1e-300 {
                let coef = w * radial / d;
                grad[0] += coef * dx;
                grad[1] += coef * dy;
            }
        }
        (total, grad)
    }

    fn relax(&self, start: [f64; 2]) -> [f64; 2] {
        let mut opt = WarmLbfgs::default();
        opt.gtol = 1e-12;
        let x0 = Array1::from_vec(start.to_vec());
        let norm = self.weights.iter().sum::<f64>().max(1e-300);
        let (_, x, _) = opt.minimize(x0.view(), 200, |v: ArrayView1<f64>| {
            let (f, g) = self.value_and_gradient([v[0], v[1]]);
            Some((f / norm, Array1::from_vec(vec![g[0] / norm, g[1] / norm])))
        });
        let end = [x[0], x[1]];
        if self.value(end) <= self.value(start) {
            end
        } else {
            start
        }
    }
}

/// A folded book.
#[derive(Clone, Debug, PartialEq)]
pub struct Landfold {
    /// One point per input row, in input order.
    pub xy: Vec<[f64; 2]>,
    /// Row index of each landmark.
    pub landmarks: Vec<usize>,
    /// The fitted map, for rows outside the book.
    pub map: LandfoldMap,
    /// Sketch-map stress of the landmarks, normalised by the total pair weight.
    pub stress: f64,
    /// The two leading Torgerson eigenvalues of the landmark distances.
    pub spectrum: [f64; 2],
}

/// Fold descriptor rows onto the plane. See the module documentation.
pub fn landfold(rows: &[Vec<f64>], config: &LandfoldConfig) -> Result<Landfold, LandfoldError> {
    config.validate()?;
    let n = rows.len();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if let Some(index) = rows.iter().position(|r| r.iter().any(|v| !v.is_finite())) {
        return Err(LandfoldError::NonFinite { index });
    }
    let mut rng = StdRng::seed_from_u64(config.seed);
    let first = rng.random_range(0..n);
    let marks = farthest_point_landmarks(rows, config.landmarks, first, config.metric);
    let m = marks.len();

    let mut weights = vec![0.0_f64; m];
    for i in 0..n {
        let row = marks.row(i);
        let mut best = 0;
        for k in 1..m {
            if row[k] < row[best] {
                best = k;
            }
        }
        weights[best] += 1.0;
    }
    for w in &mut weights {
        *w = w.powf(config.weight_power);
    }

    let mut dist_ll = vec![0.0; m * m];
    for (a, &ia) in marks.indices.iter().enumerate() {
        dist_ll[a * m..(a + 1) * m].copy_from_slice(marks.row(ia));
    }
    let sigma = match config.sigma {
        Some(s) => s,
        None => config.sigma_scale * median_offdiagonal(&dist_ll, m),
    };
    let sigma = if sigma.is_finite() && sigma > 0.0 {
        sigma
    } else {
        1.0
    };
    let high = Switch::new(sigma, config.high[0], config.high[1])?;
    let low = Switch::new(
        config.sigma_low.unwrap_or(sigma),
        config.low[0],
        config.low[1],
    )?;

    let (mut ys, spectrum) = classical_mds(&dist_ll, m, config.seed);
    if m == 2 {
        let d = dist_ll[1];
        ys = vec![[-0.5 * d, 0.0], [0.5 * d, 0.0]];
    }
    let targets: Vec<f64> = dist_ll.iter().map(|&d| high.value(d)).collect();
    let fit = LandmarkStress {
        m,
        raw: &dist_ll,
        targets: &targets,
        weights: &weights,
        low: &low,
    };
    if m >= 3 {
        for &mix in &config.mix {
            fit.relax(&mut ys, mix, config.iterations);
            if config.grid >= 2 && mix < 1.0 {
                fit.grid_sweep(&mut ys, mix, config.grid);
                fit.relax(&mut ys, mix, config.iterations);
            }
        }
    }
    center(&mut ys);
    let stress = fit.value(&ys, 0.0);

    let map = LandfoldMap {
        landmark_rows: marks.indices.iter().map(|&i| rows[i].clone()).collect(),
        landmark_xy: ys.clone(),
        weights,
        high,
        low,
        metric: config.metric,
    };
    let mut slot = vec![usize::MAX; n];
    for (k, &i) in marks.indices.iter().enumerate() {
        slot[i] = k;
    }
    let xy: Vec<[f64; 2]> = (0..n)
        .into_par_iter()
        .map(|i| {
            if slot[i] != usize::MAX {
                ys[slot[i]]
            } else {
                map.project_distances(marks.row(i))
            }
        })
        .collect();
    Ok(Landfold {
        xy,
        landmarks: marks.indices,
        map,
        stress,
        spectrum,
    })
}

fn median_offdiagonal(dist: &[f64], m: usize) -> f64 {
    let mut values: Vec<f64> = (0..m)
        .flat_map(|i| ((i + 1)..m).map(move |j| (i, j)))
        .map(|(i, j)| dist[i * m + j])
        .filter(|d| *d > 0.0)
        .collect();
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn center(ys: &mut [[f64; 2]]) {
    let n = ys.len().max(1) as f64;
    let mx = ys.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = ys.iter().map(|p| p[1]).sum::<f64>() / n;
    for p in ys.iter_mut() {
        p[0] -= mx;
        p[1] -= my;
    }
}

/// Mixed stress over landmark pairs.
///
/// \(\chi = W^{-1}\sum_{i<j} w_i w_j[(1-\mu)(s_D(D_{ij})-s_d(d_{ij}))^2 +
/// \mu (D_{ij}-d_{ij})^2]\) with \(W=\sum_{i<j} w_i w_j\), the stress of
/// the reference `NLDRITERChi` at mixing \(\mu\).
struct LandmarkStress<'a> {
    m: usize,
    raw: &'a [f64],
    targets: &'a [f64],
    weights: &'a [f64],
    low: &'a Switch,
}

impl LandmarkStress<'_> {
    fn point<'s>(&'s self, ys: &'s [[f64; 2]], i: usize, mix: f64) -> PointStress<'s> {
        let m = self.m;
        PointStress {
            targets: &self.targets[i * m..(i + 1) * m],
            raw: &self.raw[i * m..(i + 1) * m],
            others: ys,
            weights: self.weights,
            low: self.low,
            mix,
            skip: Some(i),
        }
    }

    fn pair_weight(&self) -> f64 {
        let total: f64 = self.weights.iter().sum();
        let squares: f64 = self.weights.iter().map(|w| w * w).sum();
        (0.5 * (total * total - squares)).max(1e-300)
    }

    fn value(&self, ys: &[[f64; 2]], mix: f64) -> f64 {
        let rows: Vec<f64> = (0..self.m)
            .into_par_iter()
            .map(|i| self.weights[i] * self.point(ys, i, mix).value(ys[i]))
            .collect();
        0.5 * rows.iter().sum::<f64>() / self.pair_weight()
    }

    fn value_and_gradient(&self, ys: &[[f64; 2]], mix: f64) -> (f64, Vec<f64>) {
        let rows: Vec<(f64, [f64; 2])> = (0..self.m)
            .into_par_iter()
            .map(|i| {
                let (v, g) = self.point(ys, i, mix).value_and_gradient(ys[i]);
                let w = self.weights[i];
                (w * v, [w * g[0], w * g[1]])
            })
            .collect();
        let norm = self.pair_weight();
        let mut grad = vec![0.0; 2 * self.m];
        let mut total = 0.0;
        for (i, (v, g)) in rows.into_iter().enumerate() {
            total += v;
            grad[2 * i] = g[0] / norm;
            grad[2 * i + 1] = g[1] / norm;
        }
        (0.5 * total / norm, grad)
    }

    fn relax(&self, ys: &mut [[f64; 2]], mix: f64, iterations: usize) {
        if iterations == 0 {
            return;
        }
        let mut opt = WarmLbfgs::default();
        opt.gtol = 1e-10;
        let x0 = Array1::from_iter(ys.iter().flat_map(|p| [p[0], p[1]]));
        let unpack = |v: ArrayView1<f64>| -> Vec<[f64; 2]> {
            (0..self.m).map(|i| [v[2 * i], v[2 * i + 1]]).collect()
        };
        let (_, x, _) = opt.minimize(x0.view(), iterations, |v: ArrayView1<f64>| {
            let (f, g) = self.value_and_gradient(&unpack(v), mix);
            Some((f, Array1::from_vec(g)))
        });
        let relaxed = unpack(x.view());
        if self.value(&relaxed, mix) <= self.value(ys, mix) {
            ys.copy_from_slice(&relaxed);
        }
    }

    /// One Gauss-Seidel sweep of the reference pointwise global search: each
    /// landmark in turn moves to the grid node of lowest stress against the
    /// others, when that node beats where it sits.
    fn grid_sweep(&self, ys: &mut [[f64; 2]], mix: f64, grid: usize) {
        let mut ys_vec = ys.to_vec();
        center(&mut ys_vec);
        let reach = 1.2
            * ys_vec
                .iter()
                .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
                .fold(0.0, f64::max)
                .max(1e-12);
        let nodes: Vec<[f64; 2]> = (0..grid * grid)
            .map(|g| {
                let step = 2.0 * reach / (grid - 1) as f64;
                [
                    -reach + (g / grid) as f64 * step,
                    -reach + (g % grid) as f64 * step,
                ]
            })
            .collect();
        for i in 0..self.m {
            let here = self.point(&ys_vec, i, mix).value(ys_vec[i]);
            let values: Vec<f64> = {
                let point = self.point(&ys_vec, i, mix);
                nodes.par_iter().map(|&y| point.value(y)).collect()
            };
            let mut best = 0;
            for g in 1..values.len() {
                if values[g] < values[best] {
                    best = g;
                }
            }
            if values[best] < here {
                ys_vec[i] = nodes[best];
            }
        }
        ys.copy_from_slice(&ys_vec);
    }
}

/// One minimum of a book.
#[derive(Clone, Debug, PartialEq)]
pub struct Minimum {
    /// Energy, or NaN when the book has none.
    pub energy: f64,
    /// Cartesian coordinates, `3N` values.
    pub coordinates: Vec<f64>,
}

/// Parse a minima book.
///
/// `.xyz` text is a sequence of frames: an atom count, a comment, and one
/// `symbol x y z` line per atom. The energy is the value after `e=` or
/// `energy=` in the comment, or else its first number. Any other text is a
/// `.min` dump: one minimum per line, energy first, then `3N` coordinates.
/// Blank lines and lines starting with `#` are skipped.
pub fn parse_book(text: &str, xyz: bool) -> Result<Vec<Minimum>, LandfoldError> {
    if xyz {
        return parse_xyz_frames(text);
    }
    let mut book = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values: Result<Vec<f64>, _> = line.split_whitespace().map(str::parse::<f64>).collect();
        let values =
            values.map_err(|err| LandfoldError::Read(format!("line {}: {err}", line_no + 1)))?;
        if values.len() < 4 || !(values.len() - 1).is_multiple_of(3) {
            return Err(LandfoldError::Read(format!(
                "line {}: expected an energy and 3N coordinates, got {} values",
                line_no + 1,
                values.len()
            )));
        }
        book.push(Minimum {
            energy: values[0],
            coordinates: values[1..].to_vec(),
        });
    }
    Ok(book)
}

fn parse_xyz_frames(text: &str) -> Result<Vec<Minimum>, LandfoldError> {
    let lines: Vec<&str> = text.lines().collect();
    let mut book = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        let head = lines[at].trim();
        if head.is_empty() {
            at += 1;
            continue;
        }
        let atoms: usize = head
            .parse()
            .map_err(|_| LandfoldError::Read(format!("line {}: expected an atom count", at + 1)))?;
        if at + 2 + atoms > lines.len() {
            return Err(LandfoldError::Read(format!(
                "line {}: frame of {atoms} atoms runs past the end",
                at + 1
            )));
        }
        let energy = comment_energy(lines[at + 1]);
        let mut coordinates = Vec::with_capacity(3 * atoms);
        for (offset, line) in lines[at + 2..at + 2 + atoms].iter().enumerate() {
            let xyz: Result<Vec<f64>, _> = line
                .split_whitespace()
                .skip(1)
                .take(3)
                .map(str::parse::<f64>)
                .collect();
            match xyz {
                Ok(v) if v.len() == 3 => coordinates.extend(v),
                _ => {
                    return Err(LandfoldError::Read(format!(
                        "line {}: expected `symbol x y z`",
                        at + 3 + offset
                    )));
                }
            }
        }
        book.push(Minimum {
            energy,
            coordinates,
        });
        at += 2 + atoms;
    }
    Ok(book)
}

fn comment_energy(comment: &str) -> f64 {
    for token in comment.split_whitespace() {
        let lower = token.to_ascii_lowercase();
        for key in ["e=", "energy="] {
            if let Some(value) = lower.strip_prefix(key)
                && let Ok(e) = value.parse::<f64>()
            {
                return e;
            }
        }
    }
    comment
        .split_whitespace()
        .find_map(|token| token.parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// Read a minima book from disk. See [`parse_book`].
pub fn read_book(path: &Path) -> Result<Vec<Minimum>, LandfoldError> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| LandfoldError::Read(format!("{}: {err}", path.display())))?;
    let xyz = path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("xyz"));
    parse_book(&text, xyz)
}

/// DECAF rows of a book on one shared codebook.
#[derive(Clone, Debug, PartialEq)]
pub struct BookDescriptors {
    /// One normalised class histogram per minimum, all of one length.
    pub rows: Vec<Vec<f64>>,
    /// Packing-book cell of each minimum.
    pub family: Vec<usize>,
    /// Packing community of each minimum, single linkage at
    /// [`crate::catalog::PACKING_LINK`] over the occupied cells.
    pub community: Vec<usize>,
}

/// DECAF class histograms of every minimum against the final codebook.
///
/// Every structure is observed into one [`crate::catalog::PackingBook`]
/// first, so the codebook contains every environment the book has, and each
/// row is then read against that codebook.
pub fn book_descriptors(coordinates: &[Vec<f64>]) -> Result<BookDescriptors, LandfoldError> {
    use crate::catalog::PackingBook;
    use crate::catalog::packing::prepare_rows;
    const CHUNK: usize = 128;
    if let Some(index) = coordinates
        .iter()
        .position(|c| c.iter().any(|v| !v.is_finite()))
    {
        return Err(LandfoldError::NonFinite { index });
    }
    let mut book = PackingBook::default();
    let mut family = Vec::with_capacity(coordinates.len());
    for (c, chunk) in coordinates.chunks(CHUNK).enumerate() {
        chunk.par_iter().for_each(|x| prepare_rows(x));
        for (offset, x) in chunk.iter().enumerate() {
            let index = c * CHUNK + offset;
            family.push(
                book.observe(x)
                    .ok_or(LandfoldError::NoDescriptor { index })?,
            );
        }
    }
    let books: Vec<PackingBook> = coordinates.chunks(CHUNK).map(|_| book.clone()).collect();
    let rows: Vec<Vec<f64>> = coordinates
        .par_chunks(CHUNK)
        .zip(books)
        .map(|(chunk, local)| {
            chunk
                .iter()
                .map(|x| local.codebook_histogram(x).unwrap_or_default())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();
    if let Some(index) = rows.iter().position(Vec::is_empty) {
        return Err(LandfoldError::NoDescriptor { index });
    }
    let (occupied, labels) = book.occupied_community_labels();
    let mut community_of = std::collections::BTreeMap::new();
    for ((cell, _), label) in occupied.iter().zip(&labels) {
        community_of.insert(*cell, *label);
    }
    let community = family
        .iter()
        .map(|f| community_of.get(f).copied().unwrap_or(usize::MAX))
        .collect();
    Ok(BookDescriptors {
        rows,
        family,
        community,
    })
}

/// Coordinates as CSV with a header.
///
/// Columns: `index,energy,x,y,landmark,family,community`. An absent
/// energy, family, or community is an empty field.
pub fn to_csv(
    fold: &Landfold,
    energy: Option<&[f64]>,
    family: Option<&[usize]>,
    community: Option<&[usize]>,
) -> String {
    let mut is_landmark = vec![false; fold.xy.len()];
    for &i in &fold.landmarks {
        is_landmark[i] = true;
    }
    let mut out = String::from("index,energy,x,y,landmark,family,community\n");
    for (i, p) in fold.xy.iter().enumerate() {
        let e = energy
            .and_then(|e| e.get(i))
            .filter(|e| e.is_finite())
            .map_or(String::new(), |e| format!("{e:.10}"));
        let f = family
            .and_then(|f| f.get(i))
            .map_or(String::new(), ToString::to_string);
        let c = community
            .and_then(|c| c.get(i))
            .filter(|c| **c != usize::MAX)
            .map_or(String::new(), ToString::to_string);
        let _ = writeln!(
            out,
            "{i},{e},{:.10},{:.10},{},{f},{c}",
            p[0],
            p[1],
            u8::from(is_landmark[i])
        );
    }
    out
}

/// Coordinates and fit metadata as one JSON object.
pub fn to_json(
    fold: &Landfold,
    energy: Option<&[f64]>,
    family: Option<&[usize]>,
    community: Option<&[usize]>,
) -> String {
    let mut is_landmark = vec![false; fold.xy.len()];
    for &i in &fold.landmarks {
        is_landmark[i] = true;
    }
    let points: Vec<serde_json::Value> = fold
        .xy
        .iter()
        .enumerate()
        .map(|(i, p)| {
            serde_json::json!({
                "index": i,
                "energy": energy.and_then(|e| e.get(i)).copied().filter(|e| e.is_finite()),
                "x": p[0],
                "y": p[1],
                "landmark": is_landmark[i],
                "family": family.and_then(|f| f.get(i)).copied(),
                "community": community.and_then(|c| c.get(i)).copied().filter(|c| *c != usize::MAX),
            })
        })
        .collect();
    let high = fold.map.high();
    let low = fold.map.low();
    serde_json::json!({
        "kind": "landfold",
        "n": fold.xy.len(),
        "landmarks": fold.landmarks,
        "sigma": high.sigma(),
        "sigma_low": low.sigma(),
        "high": high.exponents(),
        "low": low.exponents(),
        "stress": fold.stress,
        "spectrum": fold.spectrum,
        "points": points,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l1_square() -> Vec<f64> {
        vec![
            0.0, 1.0, 2.0, 1.0, //
            1.0, 0.0, 1.0, 2.0, //
            2.0, 1.0, 0.0, 1.0, //
            1.0, 2.0, 1.0, 0.0,
        ]
    }

    fn l1_matrix(points: &[[f64; 2]]) -> Vec<f64> {
        let m = points.len();
        let mut d = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                d[i * m + j] =
                    (points[i][0] - points[j][0]).abs() + (points[i][1] - points[j][1]).abs();
            }
        }
        d
    }

    fn spread(xy: &[[f64; 2]], axis: usize) -> f64 {
        let n = xy.len() as f64;
        let mean = xy.iter().map(|p| p[axis]).sum::<f64>() / n;
        (xy.iter().map(|p| (p[axis] - mean).powi(2)).sum::<f64>() / n).sqrt()
    }

    #[test]
    fn switch_is_half_at_sigma_and_slope_matches_finite_difference() {
        for (a, b) in [(2.0, 4.0), (8.0, 8.0), (2.0, 8.0), (3.0, 1.5), (4.0, 2.0)] {
            let s = Switch::new(1.7, a, b).unwrap();
            assert!((s.value(1.7) - 0.5).abs() < 1e-12, "s(sigma) for {a},{b}");
            assert_eq!(s.value(0.0), 0.0);
            for r in [0.1, 0.9, 1.7, 3.0, 9.0] {
                let h = 1e-6;
                let fd = (s.value(r + h) - s.value(r - h)) / (2.0 * h);
                let (_, slope) = s.value_and_slope(r);
                assert!(
                    (fd - slope).abs() < 1e-6 * (1.0 + slope.abs()),
                    "slope at {r} for {a},{b}: {slope} vs {fd}"
                );
            }
        }
        assert!(Switch::new(0.0, 2.0, 2.0).is_err());
        assert!(Switch::new(1.0, 1.0, 2.0).is_err());
    }

    #[test]
    fn classical_mds_square_has_lambda1_equal_lambda2_equal_two() {
        for seed in 0..8 {
            let (xy, spectrum) = classical_mds(&l1_square(), 4, seed);
            assert!(
                (spectrum[0] - 2.0).abs() < 1e-9 && (spectrum[1] - 2.0).abs() < 1e-9,
                "seed {seed}: square spectrum {spectrum:?}"
            );
            assert!(spectrum[1] > 0.0, "lambda2 > 0 on the square");
            let lambda = |axis: usize| xy.iter().map(|p| p[axis] * p[axis]).sum::<f64>();
            assert!((lambda(0) - 2.0).abs() < 1e-9 && (lambda(1) - 2.0).abs() < 1e-9);
            assert!(spread(&xy, 0) > 0.2 && spread(&xy, 1) > 0.2, "{xy:?}");
        }
    }

    #[test]
    fn classical_mds_rectangle_keeps_lambda2_positive() {
        let rect = [[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0]];
        let (xy, spectrum) = classical_mds(&l1_matrix(&rect), 4, 3);
        assert!(spectrum[0] > spectrum[1], "{spectrum:?}");
        assert!(
            spectrum[1] > 0.2,
            "lambda2 > 0 on the 3x1 rectangle: {spectrum:?}"
        );
        assert!(spread(&xy, 0) > 0.2 && spread(&xy, 1) > 0.2, "{xy:?}");
        let euclid = |p: &[[f64; 2]]| {
            let m = p.len();
            let mut d = vec![0.0; m * m];
            for i in 0..m {
                for j in 0..m {
                    d[i * m + j] =
                        ((p[i][0] - p[j][0]).powi(2) + (p[i][1] - p[j][1]).powi(2)).sqrt();
                }
            }
            d
        };
        let (_, exact) = classical_mds(&euclid(&rect), 4, 3);
        assert!(
            (exact[0] - 9.0).abs() < 1e-9 && (exact[1] - 1.0).abs() < 1e-9,
            "Euclidean 3x1 rectangle has lambda1 = 9, lambda2 = 1: {exact:?}"
        );
    }

    #[test]
    fn landfold_square_keeps_two_axes() {
        let rows = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ];
        let fold = landfold(&rows, &LandfoldConfig::default()).unwrap();
        assert!(
            (fold.spectrum[0] - 2.0).abs() < 1e-9 && (fold.spectrum[1] - 2.0).abs() < 1e-9,
            "{:?}",
            fold.spectrum
        );
        assert!(spread(&fold.xy, 0) > 0.2 && spread(&fold.xy, 1) > 0.2);
    }

    #[test]
    fn farthest_points_start_at_first_and_stop_at_distinct_rows() {
        let rows = vec![vec![0.0], vec![0.0], vec![1.0], vec![5.0], vec![5.0]];
        let marks = farthest_point_landmarks(&rows, 10, 2, LandfoldMetric::L1);
        assert_eq!(marks.indices, vec![2, 3, 0]);
        assert_eq!(marks.row(4), &[4.0, 0.0, 5.0]);
    }

    #[test]
    fn parse_book_reads_min_and_xyz() {
        let min = "# energy then coordinates\n-1.5 0 0 0 1 0 0\n\n-2.5 0 0 0 0 1 0\n";
        let book = parse_book(min, false).unwrap();
        assert_eq!(book.len(), 2);
        assert_eq!(book[1].energy, -2.5);
        assert_eq!(book[1].coordinates, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let xyz = "2\nLJ2 e=-1.0\nX 0 0 0\nX 1 0 0\n2\n-0.5\nX 0 0 0\nX 2 0 0\n";
        let book = parse_book(xyz, true).unwrap();
        assert_eq!(book.len(), 2);
        assert_eq!(book[0].energy, -1.0);
        assert_eq!(book[1].energy, -0.5);
        assert_eq!(book[1].coordinates[3], 2.0);
        assert!(parse_book("1.0 2.0\n", false).is_err());
    }
}
