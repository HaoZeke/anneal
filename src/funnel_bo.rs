//! Choosing which morphology to search next, by expected improvement.
//!
//! The searches here fail by funnel, not by local optimization. At 75 points
//! every failing run reaches the icosahedral plateau and stays; the relaxation
//! is fine, the descent is fine, and the answer is in a region the chain never
//! visits. Eight mechanisms built to make the chain *leave* a funnel were
//! measured and none helped, because leaving is not the problem: the chain has
//! nowhere better to be told to go.
//!
//! That is a decision problem over a small space, and it is the shape Bayesian
//! optimization is for. Not over the coordinates, where three hundred
//! dimensions puts a Gaussian process out of reach, but over a *structural
//! descriptor*: the share of points in each local environment, five numbers
//! from [`crate::structure::ptm_fractions`]. A model of "how low does this
//! morphology go" over those five numbers is cheap to fit and is exactly the
//! surface the search is blind to.
//!
//! # What is modelled
//!
//! For each distinct morphology the search has visited, the lowest energy found
//! there. A Gaussian process over that gives a mean and a variance everywhere,
//! including at morphologies never visited, and expected improvement turns the
//! pair into a single number: how much better than the incumbent this
//! morphology is likely to be, integrated over the model's own uncertainty.
//!
//! The point is what expected improvement does with a region that has never
//! been sampled. Its mean reverts to the prior and its variance is large, so it
//! scores highly; a region sampled repeatedly and found mediocre scores low
//! however close it is to the incumbent. That is the opposite of what a bias
//! does, and it is the missing half: a bias says where not to go, this says
//! where to go instead.
//!
//! # Why a Gaussian process and not something larger
//!
//! The observation count is the number of distinct morphologies, which is
//! hundreds rather than millions, and the dimension is five. Exact inference is
//! a Cholesky factorisation of a few-hundred-square matrix, done once per
//! refit. Nothing here needs approximating.

use ndarray::{Array1, Array2, ArrayView1};

/// A Gaussian process over a structural descriptor.
///
/// Squared-exponential kernel with a single length scale, which is right when
/// the inputs are fractions on a common scale, and a noise term that also keeps
/// the factorisation well conditioned when two morphologies are nearly equal.
#[derive(Debug, Clone)]
pub struct FunnelModel {
    /// Kernel length scale, in units of the descriptor.
    pub length_scale: f64,
    /// Signal standard deviation.
    pub amplitude: f64,
    /// Observation noise standard deviation.
    pub noise: f64,
    /// Prior mean, used where nothing has been observed.
    ///
    /// Set to the mean of the observations on each refit rather than to zero.
    /// A zero prior on an energy scale of minus four hundred makes every
    /// unvisited region look catastrophic and expected improvement then never
    /// leaves the data.
    prior_mean: f64,
    xs: Vec<Array1<f64>>,
    ys: Vec<f64>,
    /// Cholesky factor of the kernel matrix plus noise, lower triangular.
    chol: Option<Array2<f64>>,
    /// `K^-1 (y - prior)`, precomputed for the mean.
    alpha: Option<Array1<f64>>,
}

impl FunnelModel {
    /// A model over descriptors, with a length scale in descriptor units.
    pub fn new(length_scale: f64, amplitude: f64, noise: f64) -> Self {
        assert!(length_scale > 0.0, "the length scale is a distance");
        assert!(amplitude > 0.0, "the amplitude is a standard deviation");
        assert!(noise > 0.0, "a positive noise keeps the factorisation stable");
        Self {
            length_scale,
            amplitude,
            noise,
            prior_mean: 0.0,
            xs: Vec::new(),
            ys: Vec::new(),
            chol: None,
            alpha: None,
        }
    }

    /// Morphologies observed.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Whether nothing has been observed.
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    /// The best value seen, which is what improvement is measured against.
    pub fn incumbent(&self) -> Option<f64> {
        self.ys.iter().copied().fold(None, |acc: Option<f64>, v| {
            Some(acc.map_or(v, |a| a.min(v)))
        })
    }

    fn kernel(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let d2: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum();
        self.amplitude * self.amplitude * (-0.5 * d2 / (self.length_scale * self.length_scale)).exp()
    }

    /// Records the lowest energy found at a morphology.
    ///
    /// A morphology already present is updated to the lower of the two rather
    /// than added again: the quantity modelled is how low a region goes, not
    /// how often it was sampled.
    pub fn observe(&mut self, x: ArrayView1<f64>, y: f64) {
        if !y.is_finite() || x.iter().any(|v| !v.is_finite()) {
            return;
        }
        for (i, existing) in self.xs.iter().enumerate() {
            if existing.len() == x.len()
                && existing
                    .iter()
                    .zip(x.iter())
                    .all(|(p, q)| (p - q).abs() < 1e-9)
            {
                if y < self.ys[i] {
                    self.ys[i] = y;
                    self.chol = None;
                }
                return;
            }
        }
        self.xs.push(x.to_owned());
        self.ys.push(y);
        self.chol = None;
    }

    /// Refits the factorisation. Called automatically when needed.
    fn fit(&mut self) {
        let n = self.xs.len();
        if n == 0 {
            return;
        }
        self.prior_mean = self.ys.iter().sum::<f64>() / n as f64;
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k[[i, j]] = self.kernel(self.xs[i].view(), self.xs[j].view());
            }
            k[[i, i]] += self.noise * self.noise;
        }
        // Cholesky, lower triangular.
        let mut l = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let mut s = k[[i, j]];
                for m in 0..j {
                    s -= l[[i, m]] * l[[j, m]];
                }
                if i == j {
                    if s <= 0.0 {
                        // Not positive definite, which happens only if the
                        // noise was set to zero; refuse rather than return a
                        // mean built from a broken factorisation.
                        self.chol = None;
                        self.alpha = None;
                        return;
                    }
                    l[[i, j]] = s.sqrt();
                } else {
                    l[[i, j]] = s / l[[j, j]];
                }
            }
        }
        // alpha = K^-1 (y - prior), by forward then back substitution.
        let mut v = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = self.ys[i] - self.prior_mean;
            for m in 0..i {
                s -= l[[i, m]] * v[m];
            }
            v[i] = s / l[[i, i]];
        }
        let mut a = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut s = v[i];
            for m in (i + 1)..n {
                s -= l[[m, i]] * a[m];
            }
            a[i] = s / l[[i, i]];
        }
        self.chol = Some(l);
        self.alpha = Some(a);
    }

    /// Posterior mean and standard deviation at a morphology.
    pub fn predict(&mut self, x: ArrayView1<f64>) -> (f64, f64) {
        if self.chol.is_none() {
            self.fit();
        }
        let n = self.xs.len();
        if n == 0 {
            return (self.prior_mean, self.amplitude);
        }
        let (l, a) = match (&self.chol, &self.alpha) {
            (Some(l), Some(a)) => (l, a),
            _ => return (self.prior_mean, self.amplitude),
        };
        let ks: Array1<f64> = (0..n)
            .map(|i| self.kernel(self.xs[i].view(), x))
            .collect();
        let mean = self.prior_mean + ks.iter().zip(a.iter()).map(|(p, q)| p * q).sum::<f64>();
        // v = L^-1 ks, and the variance is k(x,x) - v'v.
        let mut v = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = ks[i];
            for m in 0..i {
                s -= l[[i, m]] * v[m];
            }
            v[i] = s / l[[i, i]];
        }
        let var = (self.kernel(x, x) - v.iter().map(|z| z * z).sum::<f64>()).max(0.0);
        (mean, var.sqrt())
    }

    /// Expected improvement at a morphology, for a minimisation.
    ///
    /// Zero where the model is confident nothing better lives, and large where
    /// the mean is low *or* the uncertainty is high. The second half is what
    /// makes this different from following the model's mean: a morphology never
    /// visited scores on its variance alone, which is how a search reaches a
    /// funnel it has no evidence about.
    pub fn expected_improvement(&mut self, x: ArrayView1<f64>) -> f64 {
        let best = match self.incumbent() {
            Some(b) => b,
            None => return f64::INFINITY,
        };
        let (mean, sd) = self.predict(x);
        if sd < 1e-12 {
            return (best - mean).max(0.0);
        }
        let z = (best - mean) / sd;
        // EI = (best - mean) Phi(z) + sd phi(z).
        (best - mean) * normal_cdf(z) + sd * normal_pdf(z)
    }
}

fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (std::f64::consts::TAU).sqrt()
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz and Stegun 7.1.26.
fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn pt(v: &[f64]) -> Array1<f64> {
        Array1::from(v.to_vec())
    }

    /// The model has to reproduce what it was told, or nothing above it means
    /// anything.
    #[test]
    fn it_interpolates_its_observations() {
        let mut m = FunnelModel::new(0.3, 50.0, 1e-3);
        let data = [
            (pt(&[0.0, 0.0]), -100.0),
            (pt(&[1.0, 0.0]), -200.0),
            (pt(&[0.0, 1.0]), -150.0),
        ];
        for (x, y) in &data {
            m.observe(x.view(), *y);
        }
        for (x, y) in &data {
            let (mean, sd) = m.predict(x.view());
            assert!((mean - y).abs() < 1.0, "at {x:?} predicted {mean} for {y}");
            assert!(sd < 1.0, "standard deviation {sd} where data sits");
        }
    }

    /// And it has to be uncertain where it was told nothing. This is the
    /// property expected improvement runs on.
    #[test]
    fn it_is_uncertain_away_from_its_observations() {
        let mut m = FunnelModel::new(0.2, 50.0, 1e-3);
        m.observe(pt(&[0.0, 0.0]).view(), -100.0);
        m.observe(pt(&[0.1, 0.0]).view(), -110.0);
        let (_, near) = m.predict(pt(&[0.05, 0.0]).view());
        let (_, far) = m.predict(pt(&[0.9, 0.9]).view());
        assert!(
            far > near * 5.0,
            "uncertainty {far} far from data against {near} inside it"
        );
    }

    /// The whole point: an unvisited morphology outscores a well-sampled
    /// mediocre one, so the search is sent somewhere it has no evidence about
    /// rather than back where it has been.
    #[test]
    fn an_unexplored_morphology_outscores_a_sampled_mediocre_one() {
        let mut m = FunnelModel::new(0.25, 50.0, 1e-3);
        // A region sampled repeatedly and found mediocre.
        for k in 0..6 {
            let d = 0.02 * k as f64;
            m.observe(pt(&[0.1 + d, 0.1]).view(), -390.0 + d);
        }
        // One good point, so the incumbent is not the mediocre region.
        m.observe(pt(&[0.15, 0.12]).view(), -396.0);

        let sampled = m.expected_improvement(pt(&[0.12, 0.1]).view());
        let unexplored = m.expected_improvement(pt(&[0.8, 0.05]).view());
        assert!(
            unexplored > sampled,
            "unexplored scored {unexplored}, sampled mediocre scored {sampled}"
        );
    }

    /// Expected improvement is never negative, which is what makes it a
    /// quantity that can be compared across candidates.
    #[test]
    fn expected_improvement_is_non_negative() {
        let mut m = FunnelModel::new(0.3, 30.0, 1e-3);
        m.observe(pt(&[0.0, 0.0]).view(), -50.0);
        m.observe(pt(&[0.5, 0.5]).view(), -20.0);
        for k in 0..20 {
            let a = k as f64 / 20.0;
            let ei = m.expected_improvement(pt(&[a, 1.0 - a]).view());
            assert!(ei >= 0.0, "expected improvement {ei} at {a}");
            assert!(ei.is_finite(), "expected improvement not finite at {a}");
        }
    }

    /// Revisiting a morphology keeps the lower energy rather than adding a
    /// second observation, since what is modelled is how low a region goes.
    #[test]
    fn revisiting_a_morphology_keeps_the_better_value() {
        let mut m = FunnelModel::new(0.3, 30.0, 1e-3);
        m.observe(pt(&[0.2, 0.3]).view(), -100.0);
        m.observe(pt(&[0.2, 0.3]).view(), -150.0);
        m.observe(pt(&[0.2, 0.3]).view(), -120.0);
        assert_eq!(m.len(), 1);
        assert_eq!(m.incumbent(), Some(-150.0));
    }

    /// The prior mean tracks the data. A zero prior on an energy scale of
    /// hundreds makes every unvisited region look catastrophic, and expected
    /// improvement then never leaves the data at all.
    #[test]
    fn the_prior_follows_the_energy_scale() {
        let mut m = FunnelModel::new(0.2, 20.0, 1e-3);
        for k in 0..5 {
            m.observe(pt(&[0.1 * k as f64, 0.0]).view(), -400.0 - k as f64);
        }
        let (mean, _) = m.predict(pt(&[5.0, 5.0]).view());
        assert!(
            mean < -300.0,
            "far-field mean {mean} should revert to the data's scale, not zero"
        );
    }

    #[test]
    fn nothing_observed_gives_the_prior_and_infinite_improvement() {
        let mut m = FunnelModel::new(0.3, 10.0, 1e-3);
        let (mean, sd) = m.predict(pt(&[0.5, 0.5]).view());
        assert_eq!(mean, 0.0);
        assert_eq!(sd, 10.0);
        assert!(m.expected_improvement(pt(&[0.5, 0.5]).view()).is_infinite());
    }
}
