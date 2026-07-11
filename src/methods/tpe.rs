//! TPE-style dual-density allocation (Bergstra et al. 2011).
//!
//! Observations `(config, score)` are split by a quantile `gamma` into a
//! "good" set (top scores) and a "bad" set. Independent Parzen / Dirichlet
//! densities `l(config)` and `g(config)` are fit to each set; selection
//! maximises the density ratio `l/g` (equivalent to expected improvement
//! under TPE's model). This module is pure: no I/O, no portfolio types.

use rand::Rng;

/// Default good-quantile used by Auto portfolio allocation.
pub const DEFAULT_GAMMA: f64 = 0.25;
/// Dirichlet / Laplace smoothing for categorical densities.
pub const DEFAULT_ALPHA: f64 = 1.0;

/// Categorical TPE over `n_categories` discrete choices (e.g. arm indices).
///
/// Score convention: **higher is better** (improvement magnitude, not cost).
#[derive(Clone, Debug)]
pub struct TpeCategorical {
    n_categories: usize,
    /// `(category, score)` history.
    history: Vec<(usize, f64)>,
    gamma: f64,
    alpha: f64,
}

impl TpeCategorical {
    /// Empty history over `n_categories` choices.
    pub fn new(n_categories: usize) -> Self {
        Self::with_params(n_categories, DEFAULT_GAMMA, DEFAULT_ALPHA)
    }

    /// Full constructor.
    pub fn with_params(n_categories: usize, gamma: f64, alpha: f64) -> Self {
        assert!(n_categories >= 1, "n_categories must be >= 1");
        assert!((0.0..1.0).contains(&gamma), "gamma must be in (0, 1)");
        assert!(alpha > 0.0 && alpha.is_finite(), "alpha must be positive");
        Self {
            n_categories,
            history: Vec::new(),
            gamma,
            alpha,
        }
    }

    /// Number of recorded observations.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns `true` when no observations have been recorded.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Record one (category, score) observation. Higher score = better.
    pub fn record(&mut self, category: usize, score: f64) {
        assert!(
            category < self.n_categories,
            "category {category} out of range {}",
            self.n_categories
        );
        if score.is_finite() {
            self.history.push((category, score));
        }
    }

    /// Split point: top `ceil(gamma * n)` scores (at least 1 when n > 0) are good.
    fn n_good(&self) -> usize {
        let n = self.history.len();
        if n == 0 {
            return 0;
        }
        ((self.gamma * n as f64).ceil() as usize).clamp(1, n)
    }

    /// Counts of good and bad observations per category.
    fn good_bad_counts(&self) -> (Vec<f64>, Vec<f64>) {
        let mut good = vec![0.0; self.n_categories];
        let mut bad = vec![0.0; self.n_categories];
        if self.history.is_empty() {
            return (good, bad);
        }
        let mut indexed: Vec<(usize, f64)> = self.history.clone();
        // Higher score first.
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let n_good = self.n_good();
        for (i, (cat, _)) in indexed.into_iter().enumerate() {
            if i < n_good {
                good[cat] += 1.0;
            } else {
                bad[cat] += 1.0;
            }
        }
        (good, bad)
    }

    /// Density ratio `l(cat) / g(cat)` with Dirichlet smoothing.
    ///
    /// `l` and `g` are smoothed multinomials over the good and bad sets.
    pub fn density_ratio(&self, category: usize) -> f64 {
        assert!(category < self.n_categories);
        let (good, bad) = self.good_bad_counts();
        let k = self.n_categories as f64;
        let sum_g: f64 = good.iter().sum();
        let sum_b: f64 = bad.iter().sum();
        // When history is empty, all ratios are 1.
        if sum_g + sum_b == 0.0 {
            return 1.0;
        }
        let l = (good[category] + self.alpha) / (sum_g + self.alpha * k);
        let g = (bad[category] + self.alpha) / (sum_b + self.alpha * k).max(self.alpha * k);
        (l / g).max(1e-300)
    }

    /// Density ratio for every category.
    pub fn density_ratios(&self) -> Vec<f64> {
        (0..self.n_categories)
            .map(|c| self.density_ratio(c))
            .collect()
    }

    /// Sample a category with probability proportional to `l/g`.
    pub fn pick<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let ratios = self.density_ratios();
        let total: f64 = ratios.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return rng.random_range(0..self.n_categories);
        }
        let mut u = rng.random::<f64>() * total;
        for (i, r) in ratios.iter().enumerate() {
            u -= r;
            if u <= 0.0 {
                return i;
            }
        }
        self.n_categories - 1
    }

    /// Argmax of density ratio (deterministic exploit).
    pub fn best(&self) -> usize {
        let ratios = self.density_ratios();
        ratios
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// One-dimensional continuous TPE with isotropic Gaussian kernels.
///
/// Used for continuous portfolio knobs (e.g. relative slice scale). Score is
/// higher-is-better. Proposal support is the observed range expanded by one
/// bandwidth on each side.
#[derive(Clone, Debug, Default)]
pub struct TpeContinuous1d {
    history: Vec<(f64, f64)>,
    gamma: f64,
}

impl TpeContinuous1d {
    /// Empty continuous TPE with default gamma.
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            gamma: DEFAULT_GAMMA,
        }
    }

    /// Record `(x, score)`.
    pub fn record(&mut self, x: f64, score: f64) {
        if x.is_finite() && score.is_finite() {
            self.history.push((x, score));
        }
    }

    /// Number of recorded `(x, score)` pairs.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns `true` when no observations have been recorded.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    fn split(&self) -> (Vec<f64>, Vec<f64>) {
        if self.history.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut indexed = self.history.clone();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let n_good = ((self.gamma * indexed.len() as f64).ceil() as usize).clamp(1, indexed.len());
        let good: Vec<f64> = indexed[..n_good].iter().map(|(x, _)| *x).collect();
        let bad: Vec<f64> = indexed[n_good..].iter().map(|(x, _)| *x).collect();
        (good, bad)
    }

    fn bandwidth(xs: &[f64]) -> f64 {
        if xs.len() < 2 {
            return 1.0;
        }
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        let std = var.sqrt().max(1e-6);
        // Scott's rule scaled for 1D.
        std * (xs.len() as f64).powf(-0.2)
    }

    fn kde(xs: &[f64], x: f64, bw: f64) -> f64 {
        if xs.is_empty() {
            return 1e-12;
        }
        let inv = 1.0 / (bw * (2.0 * std::f64::consts::PI).sqrt());
        let dens: f64 = xs
            .iter()
            .map(|&xi| {
                let z = (x - xi) / bw;
                inv * (-0.5 * z * z).exp()
            })
            .sum::<f64>()
            / xs.len() as f64;
        dens.max(1e-300)
    }

    /// Density ratio `l(x)/g(x)`.
    pub fn density_ratio(&self, x: f64) -> f64 {
        let (good, bad) = self.split();
        if good.is_empty() && bad.is_empty() {
            return 1.0;
        }
        let bw_g = Self::bandwidth(&good);
        let bw_b = Self::bandwidth(if bad.is_empty() { &good } else { &bad });
        let l = Self::kde(&good, x, bw_g);
        let g = if bad.is_empty() {
            1e-6
        } else {
            Self::kde(&bad, x, bw_b)
        };
        (l / g).max(1e-300)
    }

    /// Sample candidates uniformly in the expanded range and return the
    /// maximiser of `l/g`.
    pub fn pick_candidates<R: Rng + ?Sized>(&self, n_candidates: usize, rng: &mut R) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        let xs: Vec<f64> = self.history.iter().map(|(x, _)| *x).collect();
        let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bw = Self::bandwidth(&xs);
        let a = lo - bw;
        let b = hi + bw;
        let mut best_x = xs[0];
        let mut best_r = f64::NEG_INFINITY;
        for _ in 0..n_candidates.max(1) {
            let x = a + (b - a) * rng.random::<f64>();
            let r = self.density_ratio(x);
            if r > best_r {
                best_r = r;
                best_x = x;
            }
        }
        best_x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn density_ratio_ranks_good_category_higher() {
        // Category 0 always improves; category 1 never does.
        let mut tpe = TpeCategorical::new(2);
        for i in 0..40 {
            tpe.record(0, 1.0 + (i as f64) * 0.01);
            tpe.record(1, 0.0);
        }
        let r0 = tpe.density_ratio(0);
        let r1 = tpe.density_ratio(1);
        assert!(
            r0 > r1,
            "good category must have higher l/g: r0={r0} r1={r1}"
        );
        assert_eq!(tpe.best(), 0);
    }

    #[test]
    fn pick_samples_preferentially_from_good_category() {
        let mut tpe = TpeCategorical::new(3);
        for _ in 0..60 {
            tpe.record(2, 5.0);
            tpe.record(0, 0.1);
            tpe.record(1, 0.0);
        }
        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = [0usize; 3];
        for _ in 0..200 {
            counts[tpe.pick(&mut rng)] += 1;
        }
        assert!(
            counts[2] > counts[0] && counts[2] > counts[1],
            "TPE pick must prefer category 2: {counts:?}"
        );
    }

    #[test]
    fn continuous_tpe_prefers_high_score_region() {
        let mut tpe = TpeContinuous1d::new();
        // High scores near x=2, low scores near x=0.
        for i in 0..30 {
            tpe.record(2.0 + 0.01 * i as f64, 10.0);
            tpe.record(0.0 + 0.01 * i as f64, 0.1);
        }
        let r_good = tpe.density_ratio(2.0);
        let r_bad = tpe.density_ratio(0.0);
        assert!(
            r_good > r_bad,
            "continuous TPE must prefer high-score region: {r_good} vs {r_bad}"
        );
        let mut rng = StdRng::seed_from_u64(3);
        let x = tpe.pick_candidates(64, &mut rng);
        assert!(
            x > 1.0,
            "pick_candidates should land near the good region, got {x}"
        );
    }
}
