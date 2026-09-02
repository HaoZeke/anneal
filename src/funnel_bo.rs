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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DescriptorMetric {
    AutoSimplex,
    Euclidean,
}

/// Auditable covariance loss from a pivoted-Cholesky kernel compression.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FunnelCompression {
    /// Number of observations offered to this compression step.
    pub input_count: usize,
    /// Number of kernel pivots retained by the GP.
    pub retained_rank: usize,
    /// Remaining prior-covariance trace divided by the original trace.
    pub residual_fraction: f64,
    /// Whether the rank ceiling was reached above the numerical covariance floor.
    pub rank_limited: bool,
}

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
    metric: DescriptorMetric,
    /// Prior mean, used where nothing has been observed.
    ///
    /// Set to the mean of the observations on each refit rather than to zero.
    /// A zero prior on an energy scale of minus four hundred makes every
    /// unvisited region look catastrophic and expected improvement then never
    /// leaves the data.
    prior_mean: f64,
    fixed_prior_mean: Option<f64>,
    xs: Vec<Array1<f64>>,
    ys: Vec<f64>,
    /// Bumped whenever an observation changes the data.
    version: u64,
    /// Cholesky factor of the kernel matrix plus noise, lower triangular.
    chol: Option<Array2<f64>>,
    /// `K^-1 (y - prior)`, precomputed for the mean.
    alpha: Option<Array1<f64>>,
}

impl FunnelModel {
    /// A model over descriptors, with a length scale in descriptor units.
    pub fn new(length_scale: f64, amplitude: f64, noise: f64) -> Self {
        Self::with_metric(
            length_scale,
            amplitude,
            noise,
            DescriptorMetric::AutoSimplex,
        )
    }

    /// A model that always uses Euclidean distance between descriptor vectors.
    ///
    /// Universal multiblock descriptors are not probability simplices even
    /// when every coordinate happens to be nonnegative. This constructor keeps
    /// their block amplitudes and concatenated geometry intact.
    pub fn new_euclidean(length_scale: f64, amplitude: f64, noise: f64) -> Self {
        Self::with_metric(length_scale, amplitude, noise, DescriptorMetric::Euclidean)
    }

    fn with_metric(
        length_scale: f64,
        amplitude: f64,
        noise: f64,
        metric: DescriptorMetric,
    ) -> Self {
        assert!(length_scale > 0.0, "the length scale is a distance");
        assert!(amplitude > 0.0, "the amplitude is a standard deviation");
        assert!(
            noise > 0.0,
            "a positive noise keeps the factorisation stable"
        );
        Self {
            length_scale,
            amplitude,
            noise,
            metric,
            prior_mean: 0.0,
            fixed_prior_mean: None,
            xs: Vec::new(),
            ys: Vec::new(),
            version: 0,
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

    /// Fix the GP prior mean independently of the retained observation subset.
    pub fn set_prior_mean(&mut self, mean: f64) {
        assert!(mean.is_finite(), "the GP prior mean must be finite");
        self.prior_mean = mean;
        self.fixed_prior_mean = Some(mean);
        self.chol = None;
        self.alpha = None;
    }

    /// The best value seen, which is what improvement is measured against.
    pub fn incumbent(&self) -> Option<f64> {
        self.ys.iter().copied().fold(None, |acc: Option<f64>, v| {
            Some(acc.map_or(v, |a| a.min(v)))
        })
    }

    /// Hellinger (or Euclidean) kernel between two descriptors.
    pub fn similarity(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        self.kernel(a, b)
    }

    fn kernel(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let amp2 = self.amplitude * self.amplitude;
        let ell2 = self.length_scale * self.length_scale;
        match self.metric {
            DescriptorMetric::AutoSimplex => match (simplex_weights(a), simplex_weights(b)) {
                (Some(p), Some(q)) => amp2 * (-hellinger2(&p, &q) / ell2).exp(),
                _ => {
                    let d2: f64 = a.iter().zip(b.iter()).map(|(u, v)| (u - v) * (u - v)).sum();
                    amp2 * (-0.5 * d2 / ell2).exp()
                }
            },
            DescriptorMetric::Euclidean => {
                let d2: f64 = a.iter().zip(b.iter()).map(|(u, v)| (u - v) * (u - v)).sum();
                amp2 * (-0.5 * d2 / ell2).exp()
            }
        }
    }

    /// Changes to the observed data since this model was created.
    ///
    /// [`Self::max_expected_improvement_at_data`] predicts at every observed
    /// site and each prediction is quadratic in their number, so the sweep is
    /// cubic: at 497 observations that is of order 1e8 operations, and the
    /// occupancy coordinator asked for it once per policy request per
    /// replica per slice. Measured with perf on a live 48-replica run,
    /// FunnelModel::predict was 74 percent of the coordinator's cycles. The
    /// version lets a caller keep the verdict while the data has not moved.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Records the lowest energy found at a morphology, bumping
    /// [`Self::version`] when that changes the data a prediction is built
    /// from.
    ///
    /// A morphology already present is updated to the lower of the two rather
    /// than added again: the quantity modelled is how low a region goes, not
    /// how often it was sampled.
    ///
    /// Non-finite coordinates or scores are dropped rather than stored: a
    /// NaN in the design matrix poisons every later prediction, and the
    /// caller that produced it is an exhausted budget answering with an
    /// infinity, not a landing worth keeping.
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
                    if self.fixed_prior_mean.is_some() {
                        self.refresh_alpha();
                    } else {
                        self.chol = None;
                        self.alpha = None;
                    }
                    self.version = self.version.wrapping_add(1);
                }
                return;
            }
        }
        let extended_cholesky = self
            .fixed_prior_mean
            .and_then(|_| self.extended_cholesky(x));
        self.xs.push(x.to_owned());
        self.ys.push(y);
        if let Some(cholesky) = extended_cholesky {
            self.alpha = Some(alpha_from_cholesky(&cholesky, &self.ys, self.prior_mean));
            self.chol = Some(cholesky);
        } else {
            self.chol = None;
            self.alpha = None;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Bound the retained kernel rank by pivoted Cholesky.
    ///
    /// The lowest observed response is always a pivot. Remaining pivots greedily
    /// maximize conditional prior variance until either the covariance floor or
    /// `maximum_rank` is reached. Exact caller-owned observations can therefore
    /// remain outside this numerical surrogate.
    pub fn compress(&mut self, maximum_rank: usize) -> FunnelCompression {
        assert!(maximum_rank > 0, "kernel compression needs positive rank");
        let input_count = self.xs.len();
        if input_count <= maximum_rank {
            return FunnelCompression {
                input_count,
                retained_rank: input_count,
                residual_fraction: 0.0,
                rank_limited: false,
            };
        }

        let diagonal = self.amplitude * self.amplitude;
        let covariance_floor = (self.noise * self.noise)
            .max(diagonal * f64::EPSILON.sqrt())
            .min(diagonal);
        let target_trace = covariance_floor * input_count as f64;
        let mut residual = vec![diagonal; input_count];
        let mut columns = Vec::<Vec<f64>>::new();
        let mut selected = vec![false; input_count];
        let mut indices = Vec::with_capacity(maximum_rank.min(input_count));
        let incumbent = self
            .ys
            .iter()
            .enumerate()
            .min_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| index)
            .expect("a nonempty GP has an incumbent");
        self.append_kernel_pivot(incumbent, &mut residual, &mut columns, &mut selected);
        indices.push(incumbent);

        while indices.len() < maximum_rank.min(input_count) {
            let residual_trace = residual.iter().sum::<f64>();
            if residual_trace <= target_trace {
                break;
            }
            let pivot = (0..input_count)
                .filter(|index| !selected[*index])
                .max_by(|left, right| {
                    residual[*left]
                        .total_cmp(&residual[*right])
                        .then_with(|| right.cmp(left))
                });
            let Some(pivot) = pivot else {
                break;
            };
            if residual[pivot] <= covariance_floor {
                break;
            }
            self.append_kernel_pivot(pivot, &mut residual, &mut columns, &mut selected);
            indices.push(pivot);
        }

        let residual_trace = residual.iter().sum::<f64>();
        let residual_fraction = (residual_trace / (diagonal * input_count as f64)).clamp(0.0, 1.0);
        let retained_rank = indices.len();
        let rank_limited = retained_rank == maximum_rank.min(input_count)
            && retained_rank < input_count
            && residual_trace > target_trace;
        self.xs = indices
            .iter()
            .map(|index| self.xs[*index].clone())
            .collect();
        self.ys = indices.iter().map(|index| self.ys[*index]).collect();
        self.chol = None;
        self.alpha = None;
        FunnelCompression {
            input_count,
            retained_rank,
            residual_fraction,
            rank_limited,
        }
    }

    fn append_kernel_pivot(
        &self,
        pivot: usize,
        residual: &mut [f64],
        columns: &mut Vec<Vec<f64>>,
        selected: &mut [bool],
    ) {
        let diagonal = self.amplitude * self.amplitude;
        let denominator = residual[pivot].max(diagonal * f64::EPSILON).sqrt();
        let mut column = vec![0.0; self.xs.len()];
        for index in 0..self.xs.len() {
            let projection = columns
                .iter()
                .map(|held| held[index] * held[pivot])
                .sum::<f64>();
            let covariance = self.kernel(self.xs[index].view(), self.xs[pivot].view());
            column[index] = (covariance - projection) / denominator;
        }
        selected[pivot] = true;
        for index in 0..residual.len() {
            residual[index] = if selected[index] {
                0.0
            } else {
                (residual[index] - column[index] * column[index]).max(0.0)
            };
        }
        columns.push(column);
    }

    fn extended_cholesky(&self, x: ArrayView1<f64>) -> Option<Array2<f64>> {
        let held = self.chol.as_ref()?;
        let count = self.xs.len();
        if held.nrows() != count || held.ncols() != count {
            return None;
        }
        let covariance = self
            .xs
            .iter()
            .map(|site| self.kernel(site.view(), x))
            .collect::<Array1<_>>();
        let projection = forward_substitute(held, &covariance);
        let conditional = self.kernel(x, x) + self.noise * self.noise
            - projection.iter().map(|value| value * value).sum::<f64>();
        if !conditional.is_finite() || conditional <= 0.0 {
            return None;
        }
        let mut extended = Array2::<f64>::zeros((count + 1, count + 1));
        for row in 0..count {
            for column in 0..=row {
                extended[[row, column]] = held[[row, column]];
            }
        }
        for column in 0..count {
            extended[[count, column]] = projection[column];
        }
        extended[[count, count]] = conditional.sqrt();
        Some(extended)
    }

    fn refresh_alpha(&mut self) {
        self.alpha = self
            .chol
            .as_ref()
            .map(|cholesky| alpha_from_cholesky(cholesky, &self.ys, self.prior_mean));
    }

    /// Refits the factorisation. Called automatically when needed.
    fn fit(&mut self) {
        let n = self.xs.len();
        if n == 0 {
            return;
        }
        self.prior_mean = self
            .fixed_prior_mean
            .unwrap_or_else(|| self.ys.iter().sum::<f64>() / n as f64);
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
        let a = alpha_from_cholesky(&l, &self.ys, self.prior_mean);
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
        let ks: Array1<f64> = (0..n).map(|i| self.kernel(self.xs[i].view(), x)).collect();
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

    /// Latent joint posterior over an ordered set of prospective sites.
    pub(crate) fn posterior_joint(
        &mut self,
        sites: &[ArrayView1<'_, f64>],
    ) -> (Vec<f64>, Array2<f64>) {
        if self.chol.is_none() {
            self.fit();
        }
        let count = sites.len();
        let mut covariance = Array2::<f64>::zeros((count, count));
        let (Some(cholesky), Some(alpha)) = (&self.chol, &self.alpha) else {
            let means = vec![self.prior_mean; count];
            for row in 0..count {
                for column in 0..=row {
                    let value = self.kernel(sites[row], sites[column]);
                    covariance[[row, column]] = value;
                    covariance[[column, row]] = value;
                }
            }
            return (means, covariance);
        };
        let mut projections = Vec::with_capacity(count);
        let mut means = Vec::with_capacity(count);
        for site in sites {
            let kernel = self
                .xs
                .iter()
                .map(|observed| self.kernel(observed.view(), *site))
                .collect::<Array1<_>>();
            means.push(
                self.prior_mean
                    + kernel
                        .iter()
                        .zip(alpha.iter())
                        .map(|(left, right)| left * right)
                        .sum::<f64>(),
            );
            projections.push(forward_substitute(cholesky, &kernel));
        }
        for row in 0..count {
            for column in 0..=row {
                let value = self.kernel(sites[row], sites[column])
                    - projections[row]
                        .iter()
                        .zip(projections[column].iter())
                        .map(|(left, right)| left * right)
                        .sum::<f64>();
                let value = if row == column { value.max(0.0) } else { value };
                covariance[[row, column]] = value;
                covariance[[column, row]] = value;
            }
        }
        (means, covariance)
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

    /// Largest expected improvement at the morphologies already observed.
    ///
    /// Occupancy retire uses this as the remaining-improvement bound on
    /// the seen packing codebook. Unseen families stay leftover-dwell's
    /// job. Empty and unfitted models return infinity so they cannot
    /// look exhausted.
    pub fn max_expected_improvement_at_data(&mut self) -> f64 {
        if self.xs.is_empty() {
            return f64::INFINITY;
        }
        let sites: Vec<Array1<f64>> = self.xs.clone();
        sites.iter().fold(0.0_f64, |held, x| {
            held.max(self.expected_improvement(x.view()))
        })
    }

    /// Wang and Jegelka MES for a minimizer, given samples of the min value.
    ///
    /// \(I(E^\star; y) = H[y] - \mathbb{E}_{E^\star}[H[y \mid E^\star]]\).
    /// Jones EI is kept for retire ([`Self::max_expected_improvement_at_data`]).
    /// Occupancy Leave ranks holes with this, not with EI. Independent
    /// marginal draws, not the GIBBON determinant bound.
    pub fn max_value_entropy(&mut self, x: ArrayView1<f64>, minima: &[f64]) -> f64 {
        if minima.is_empty() {
            return 0.0;
        }
        let (mean, sd) = self.predict(x);
        if !mean.is_finite() || !sd.is_finite() || sd < 1e-12 {
            return 0.0;
        }
        let mut acc = 0.0;
        let mut n = 0usize;
        for &eta in minima {
            if !eta.is_finite() {
                continue;
            }
            acc += mes_given_min(mean, sd, eta);
            n += 1;
        }
        if n == 0 { 0.0 } else { acc / n as f64 }
    }

    /// GIBBON information about the minimum value from one noisy evaluation.
    ///
    /// This is the minimization form of the closed-form lower bound in Moss
    /// et al. The signal-to-observation variance ratio accounts for the model's
    /// observation noise; `minima` contains posterior samples of the minimum
    /// function value over the candidate set.
    pub fn gibbon_information(&mut self, x: ArrayView1<f64>, minima: &[f64]) -> f64 {
        self.gibbon_information_with_offset(x, 0.0, minima)
    }

    /// GIBBON information when the GP models a change added to a known offset.
    ///
    /// Search operators are modelled through their terminal energy change. A
    /// candidate starting at energy `offset` therefore has terminal prediction
    /// `offset + f(x)`, while its posterior variance is unchanged.
    pub(crate) fn gibbon_information_with_offset(
        &mut self,
        x: ArrayView1<f64>,
        offset: f64,
        minima: &[f64],
    ) -> f64 {
        if minima.is_empty() {
            return 0.0;
        }
        let (delta_mean, sd) = self.predict(x);
        let mean = offset + delta_mean;
        if !mean.is_finite() || !sd.is_finite() || sd < 1e-12 {
            return 0.0;
        }
        let latent_variance = sd * sd;
        let observation_variance = latent_variance + self.noise * self.noise;
        let rho2 = (latent_variance / observation_variance).clamp(0.0, 1.0);
        let mut information = 0.0;
        let mut samples = 0usize;
        for &minimum in minima {
            if !minimum.is_finite() {
                continue;
            }
            let gamma = (mean - minimum) / sd;
            let mills = inverse_mills_lower(gamma);
            let truncation = (mills * (gamma + mills)).clamp(0.0, 1.0);
            let retained = (1.0 - rho2 * truncation).max(f64::MIN_POSITIVE);
            information -= 0.5 * retained.ln();
            samples += 1;
        }
        if samples == 0 {
            0.0
        } else {
            information / samples as f64
        }
    }

    /// Posterior correlation between two prospective noisy observations.
    ///
    /// Independent observation noise contributes to each marginal variance,
    /// but not to the covariance between two prospective evaluations.
    pub fn predictive_observation_correlation(
        &mut self,
        left: ArrayView1<f64>,
        right: ArrayView1<f64>,
    ) -> f64 {
        let covariance = self.posterior_latent_covariance(left, right);
        let left_variance =
            self.posterior_latent_covariance(left, left).max(0.0) + self.noise * self.noise;
        let right_variance =
            self.posterior_latent_covariance(right, right).max(0.0) + self.noise * self.noise;
        let scale = (left_variance * right_variance).sqrt();
        if !covariance.is_finite() || !scale.is_finite() || scale <= 0.0 {
            0.0
        } else {
            (covariance / scale).clamp(-1.0, 1.0)
        }
    }

    /// Batch GIBBON acquisition for prospective descriptor-space evaluations.
    ///
    /// The singleton information terms target the unknown minimum value. The
    /// half log-determinant of the predictive correlation matrix discounts
    /// redundant launches without an independently tuned diversity weight.
    pub fn gibbon_batch(&mut self, batch: &[ArrayView1<f64>], minima: &[f64]) -> f64 {
        if batch.is_empty() {
            return 0.0;
        }
        let singleton = batch
            .iter()
            .map(|point| self.gibbon_information(*point, minima))
            .sum::<f64>();
        let mut correlation = Array2::<f64>::eye(batch.len());
        for row in 0..batch.len() {
            for column in 0..row {
                let value = self.predictive_observation_correlation(batch[row], batch[column]);
                correlation[[row, column]] = value;
                correlation[[column, row]] = value;
            }
        }
        let Some(log_determinant) = positive_definite_log_determinant(&correlation) else {
            return f64::NEG_INFINITY;
        };
        singleton + 0.5 * log_determinant
    }

    /// Greedily fill a batch by GIBBON under a hard per-source family cap.
    ///
    /// The candidate table is a finite search domain. Posterior minimum samples
    /// are joint Gaussian draws over that domain, so correlated descriptors are
    /// not treated as independent opportunities. Greedy filling is the standard
    /// submodular approximation to the corresponding determinantal MAP problem.
    pub fn assign_gibbon(
        &mut self,
        candidates: &[(usize, Vec<f64>)],
        q: usize,
        max_family_size: usize,
        minimum_samples: usize,
    ) -> Vec<usize> {
        if candidates.is_empty() || q == 0 || max_family_size == 0 {
            return Vec::new();
        }
        let candidates = candidates
            .iter()
            .filter(|(_, descriptor)| {
                !descriptor.is_empty() && descriptor.iter().all(|value| value.is_finite())
            })
            .map(|(source, descriptor)| (*source, Array1::from(descriptor.clone())))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Vec::new();
        }
        let candidate_views = candidates
            .iter()
            .map(|(_, descriptor)| descriptor.view())
            .collect::<Vec<_>>();
        let minima = self.sample_joint_minima(&candidate_views, minimum_samples);
        let mut selected_sources = Vec::with_capacity(q);
        let mut selected_descriptors = Vec::<Array1<f64>>::with_capacity(q);
        let mut family_sizes = std::collections::BTreeMap::<usize, usize>::new();
        while selected_sources.len() < q {
            let mut best: Option<(f64, usize)> = None;
            for (candidate_index, (source, descriptor)) in candidates.iter().enumerate() {
                if family_sizes.get(source).copied().unwrap_or(0) >= max_family_size {
                    continue;
                }
                let mut batch = selected_descriptors
                    .iter()
                    .map(|point| point.view())
                    .collect::<Vec<_>>();
                batch.push(descriptor.view());
                let score = self.gibbon_batch(&batch, &minima);
                let replace = best.as_ref().is_none_or(|(held, held_index)| {
                    score.total_cmp(held).is_gt()
                        || (score.total_cmp(held).is_eq()
                            && family_sizes.get(source).copied().unwrap_or(0)
                                < family_sizes
                                    .get(&candidates[*held_index].0)
                                    .copied()
                                    .unwrap_or(0))
                });
                if score.is_finite() && replace {
                    best = Some((score, candidate_index));
                }
            }
            let Some((_, candidate_index)) = best else {
                break;
            };
            let (source, descriptor) = &candidates[candidate_index];
            selected_sources.push(*source);
            selected_descriptors.push(descriptor.clone());
            *family_sizes.entry(*source).or_default() += 1;
        }
        selected_sources
    }

    /// Independent-marginal samples of the posterior minimum at the
    /// observed sites plus `extras`. Seeded from the book version so a
    /// ranking is reproducible without a caller rng.
    pub fn sample_minima(&mut self, extras: &[ArrayView1<f64>], n_samples: usize) -> Vec<f64> {
        let xs = self.xs.clone();
        let mut sites: Vec<(f64, f64)> = Vec::with_capacity(xs.len() + extras.len());
        for x in &xs {
            sites.push(self.predict(x.view()));
        }
        for extra in extras {
            sites.push(self.predict(*extra));
        }
        if sites.is_empty() || n_samples == 0 {
            return Vec::new();
        }
        let mut state = 0x9E37_79B9_7F4A_7C15u64
            ^ self.version.wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ (sites.len() as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
        if state == 0 {
            state = 1;
        }
        let incumbent = self.incumbent();
        let mut out = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut eta = f64::INFINITY;
            for &(mean, sd) in &sites {
                eta = eta.min(mean + sd.max(0.0) * unit_normal(&mut state));
            }
            if let Some(best) = incumbent {
                eta = eta.min(best);
            }
            out.push(eta);
        }
        out
    }

    fn sample_joint_minima(&mut self, extras: &[ArrayView1<f64>], n_samples: usize) -> Vec<f64> {
        let shifted = extras.iter().map(|point| (*point, 0.0)).collect::<Vec<_>>();
        self.sample_shifted_joint_minima(&shifted, n_samples, self.incumbent(), 0)
    }

    /// Correlated posterior samples of the smallest offset GP value.
    ///
    /// Each site represents `offset + f(x)`. `incumbent` is an exact terminal
    /// energy already observed across every operator in the enclosing search.
    pub(crate) fn sample_shifted_joint_minima(
        &mut self,
        extras: &[(ArrayView1<'_, f64>, f64)],
        n_samples: usize,
        incumbent: Option<f64>,
        draw_salt: u64,
    ) -> Vec<f64> {
        if extras.is_empty() || n_samples == 0 {
            return Vec::new();
        }
        let (means, covariance, draws) =
            self.sample_shifted_joint_values(extras, n_samples, draw_salt);
        debug_assert_eq!(means.len(), covariance.nrows());
        draws
            .into_iter()
            .map(|draw| {
                let mut minimum = draw
                    .into_iter()
                    .min_by(f64::total_cmp)
                    .unwrap_or(f64::INFINITY);
                if let Some(best) = incumbent {
                    minimum = minimum.min(best);
                }
                minimum
            })
            .collect()
    }

    /// Correlated terminal-energy draws over an ordered finite action set.
    ///
    /// Every returned mean includes its known source-energy offset. The latent
    /// covariance excludes prospective observation noise, which lets an
    /// acquisition condition one query on a sampled optimum value without
    /// treating numerical evaluation noise as a property of the PES.
    pub(crate) fn sample_shifted_joint_values(
        &mut self,
        extras: &[(ArrayView1<'_, f64>, f64)],
        n_samples: usize,
        draw_salt: u64,
    ) -> (Vec<f64>, Array2<f64>, Vec<Vec<f64>>) {
        if extras.is_empty() || n_samples == 0 {
            return (Vec::new(), Array2::zeros((0, 0)), Vec::new());
        }
        let views = extras.iter().map(|(site, _)| *site).collect::<Vec<_>>();
        let (mut means, covariance) = self.posterior_joint(&views);
        for (mean, (_, offset)) in means.iter_mut().zip(extras) {
            *mean += offset;
        }
        let mut stabilized_covariance = covariance.clone();
        let jitter = self.amplitude * self.amplitude * 1e-12;
        for index in 0..extras.len() {
            stabilized_covariance[[index, index]] += jitter;
        }
        let Some(cholesky) = positive_definite_cholesky(&stabilized_covariance) else {
            return (means, covariance, Vec::new());
        };
        let mut state = 0xD1B5_4A32_D192_ED03u64
            ^ self.version.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (extras.len() as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
            ^ draw_salt;
        if state == 0 {
            state = 1;
        }
        let mut out = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let normal = (0..extras.len())
                .map(|_| unit_normal(&mut state))
                .collect::<Vec<_>>();
            let mut values = Vec::with_capacity(extras.len());
            for row in 0..extras.len() {
                let value = means[row]
                    + (0..=row)
                        .map(|column| cholesky[[row, column]] * normal[column])
                        .sum::<f64>();
                values.push(value);
            }
            out.push(values);
        }
        (means, covariance, out)
    }

    fn posterior_latent_covariance(
        &mut self,
        left: ArrayView1<f64>,
        right: ArrayView1<f64>,
    ) -> f64 {
        if self.chol.is_none() {
            self.fit();
        }
        let Some(cholesky) = &self.chol else {
            return self.kernel(left, right);
        };
        let left_kernel = self
            .xs
            .iter()
            .map(|site| self.kernel(site.view(), left))
            .collect::<Array1<_>>();
        let right_kernel = self
            .xs
            .iter()
            .map(|site| self.kernel(site.view(), right))
            .collect::<Array1<_>>();
        let left_solved = forward_substitute(cholesky, &left_kernel);
        let right_solved = forward_substitute(cholesky, &right_kernel);
        self.kernel(left, right)
            - left_solved
                .iter()
                .zip(right_solved.iter())
                .map(|(left, right)| left * right)
                .sum::<f64>()
    }

    /// Rank-and-cycle q-EI on a discrete family table.
    ///
    /// Independent argmax EI repeats the same family `q` times. Rank the
    /// table by Jones EI and cycle without replacement so a WAVE spreads
    /// across families that still have remaining improvement.
    pub fn assign_q_ei(&mut self, candidates: &[(usize, Vec<f64>)], q: usize) -> Vec<usize> {
        if candidates.is_empty() || q == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(f64, usize)> = Vec::new();
        for (id, histogram) in candidates {
            if histogram.is_empty() {
                continue;
            }
            let ei = self.expected_improvement(Array1::from(histogram.clone()).view());
            if ei.is_finite() {
                scored.push((ei, *id));
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        if scored.is_empty() {
            return Vec::new();
        }
        (0..q).map(|i| scored[i % scored.len()].1).collect()
    }

    /// `q` copies of the lowest posterior mean. The greedy lie q-EI replaces.
    pub fn assign_by_mean(&mut self, candidates: &[(usize, Vec<f64>)], q: usize) -> Vec<usize> {
        if candidates.is_empty() || q == 0 {
            return Vec::new();
        }
        let mut best: Option<(f64, usize)> = None;
        for (id, histogram) in candidates {
            if histogram.is_empty() {
                continue;
            }
            let (mean, _) = self.predict(Array1::from(histogram.clone()).view());
            if mean.is_finite() && best.as_ref().is_none_or(|(held, _)| mean < *held) {
                best = Some((mean, *id));
            }
        }
        match best {
            Some((_, id)) => vec![id; q],
            None => Vec::new(),
        }
    }
}

fn simplex_weights(x: ArrayView1<f64>) -> Option<Vec<f64>> {
    let sum: f64 = x.iter().copied().sum();
    if !sum.is_finite() || sum <= 0.0 || x.iter().any(|v| !v.is_finite() || *v < -1e-15) {
        return None;
    }
    Some(x.iter().map(|v| v.max(0.0) / sum).collect())
}

fn hellinger2(p: &[f64], q: &[f64]) -> f64 {
    let n = p.len().max(q.len());
    let mut acc = 0.0;
    for i in 0..n {
        let u = p.get(i).copied().unwrap_or(0.0).sqrt();
        let v = q.get(i).copied().unwrap_or(0.0).sqrt();
        let d = u - v;
        acc += d * d;
    }
    0.5 * acc
}

/// Minimization MES term: \(\gamma\varphi(\gamma)/(2\Phi(\gamma)) - \log\Phi(\gamma)\),
/// \(\gamma = (\mu - \eta)/\sigma\).
fn mes_given_min(mean: f64, sd: f64, eta: f64) -> f64 {
    let gamma = (mean - eta) / sd;
    let cdf = normal_cdf(gamma).clamp(1e-15, 1.0 - 1e-15);
    (gamma * normal_pdf(gamma)) / (2.0 * cdf) - cdf.ln()
}

pub(crate) fn inverse_mills_lower(value: f64) -> f64 {
    if value < -5.0 {
        let magnitude = -value;
        let inverse = 1.0 / magnitude;
        magnitude + inverse - 2.0 * inverse.powi(3) + 10.0 * inverse.powi(5)
    } else {
        normal_pdf(value) / normal_cdf(value).max(f64::MIN_POSITIVE)
    }
}

fn forward_substitute(cholesky: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    let mut solved = Array1::<f64>::zeros(rhs.len());
    for row in 0..rhs.len() {
        let mut value = rhs[row];
        for column in 0..row {
            value -= cholesky[[row, column]] * solved[column];
        }
        solved[row] = value / cholesky[[row, row]];
    }
    solved
}

fn alpha_from_cholesky(cholesky: &Array2<f64>, ys: &[f64], prior_mean: f64) -> Array1<f64> {
    let count = ys.len();
    let residual = Array1::from_iter(ys.iter().map(|value| value - prior_mean));
    let forward = forward_substitute(cholesky, &residual);
    let mut alpha = Array1::<f64>::zeros(count);
    for row in (0..count).rev() {
        let mut value = forward[row];
        for column in (row + 1)..count {
            value -= cholesky[[column, row]] * alpha[column];
        }
        alpha[row] = value / cholesky[[row, row]];
    }
    alpha
}

fn positive_definite_cholesky(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    if matrix.nrows() != matrix.ncols() {
        return None;
    }
    let mut cholesky = Array2::<f64>::zeros(matrix.raw_dim());
    for row in 0..matrix.nrows() {
        for column in 0..=row {
            let mut value = matrix[[row, column]];
            for inner in 0..column {
                value -= cholesky[[row, inner]] * cholesky[[column, inner]];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return None;
                }
                cholesky[[row, column]] = value.sqrt();
            } else {
                cholesky[[row, column]] = value / cholesky[[column, column]];
            }
        }
    }
    Some(cholesky)
}

pub(crate) fn positive_definite_log_determinant(matrix: &Array2<f64>) -> Option<f64> {
    let cholesky = positive_definite_cholesky(matrix)?;
    Some(
        2.0 * (0..cholesky.nrows())
            .map(|i| cholesky[[i, i]].ln())
            .sum::<f64>(),
    )
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn unit_normal(state: &mut u64) -> f64 {
    let u1 = ((xorshift64(state) as f64) / (u64::MAX as f64)).clamp(1e-12, 1.0);
    let u2 = (xorshift64(state) as f64) / (u64::MAX as f64);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
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
    use ndarray::{Array1, ArrayView1};

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
        m.observe(pt(&[1.0, 0.0]).view(), -100.0);
        m.observe(pt(&[0.92, 0.08]).view(), -110.0);
        let (_, near) = m.predict(pt(&[0.96, 0.04]).view());
        let (_, far) = m.predict(pt(&[0.0, 1.0]).view());
        assert!(
            far > near * 2.0,
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
            m.observe(pt(&[0.85 + d, 0.15 - d]).view(), -390.0 + d);
        }
        // One good point, so the incumbent is not the mediocre region.
        m.observe(pt(&[0.88, 0.12]).view(), -396.0);

        let sampled = m.expected_improvement(pt(&[0.87, 0.13]).view());
        let unexplored = m.expected_improvement(pt(&[0.1, 0.9]).view());
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
    fn an_explicit_prior_mean_survives_a_sparse_observation_subset() {
        let mut m = FunnelModel::new_euclidean(0.2, 20.0, 1e-3);
        m.set_prior_mean(-75.0);
        m.observe(pt(&[0.0, 0.0]).view(), -120.0);

        let (mean, _) = m.predict(pt(&[100.0, 100.0]).view());

        assert!((mean + 75.0).abs() < 1e-10, "far-field prior mean {mean}");
    }

    #[test]
    fn max_ei_at_data_is_finite_after_three_observations() {
        let mut m = FunnelModel::new(0.15, 20.0, 1e-2);
        assert!(m.max_expected_improvement_at_data().is_infinite());
        m.observe(pt(&[0.0, 0.0]).view(), -44.0);
        m.observe(pt(&[0.2, 0.0]).view(), -40.0);
        m.observe(pt(&[0.0, 0.2]).view(), -42.0);
        let max_ei = m.max_expected_improvement_at_data();
        assert!(max_ei.is_finite(), "max EI at data {max_ei}");
        assert!(max_ei >= 0.0, "max EI at data {max_ei}");
    }

    #[test]
    fn nothing_observed_gives_the_prior_and_infinite_improvement() {
        let mut m = FunnelModel::new(0.3, 10.0, 1e-3);
        let (mean, sd) = m.predict(pt(&[0.5, 0.5]).view());
        assert_eq!(mean, 0.0);
        assert_eq!(sd, 10.0);
        assert!(m.expected_improvement(pt(&[0.5, 0.5]).view()).is_infinite());
    }

    #[test]
    fn hellinger_kernel_is_one_on_the_same_composition() {
        let m = FunnelModel::new(0.15, 20.0, 1e-2);
        let a = pt(&[2.0, 0.0, 0.0]);
        let b = pt(&[1.0, 0.0, 0.0]);
        let k = m.kernel(a.view(), b.view());
        assert!(
            (k - 400.0).abs() < 1e-9,
            "identical compositions after normalize: {k}"
        );
    }

    #[test]
    fn mes_is_non_negative_and_finite() {
        let mut m = FunnelModel::new(0.15, 20.0, 1e-2);
        m.observe(pt(&[1.0, 0.0, 0.0]).view(), -396.28);
        m.observe(pt(&[0.0, 1.0, 0.0]).view(), -380.0);
        let extras = [pt(&[0.0, 0.0, 1.0])];
        let views: Vec<ArrayView1<f64>> = extras.iter().map(|x| x.view()).collect();
        let minima = m.sample_minima(&views, 16);
        for hist in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let mes = m.max_value_entropy(pt(&hist).view(), &minima);
            assert!(mes.is_finite(), "MES {mes} at {hist:?}");
            assert!(mes >= 0.0, "MES {mes} at {hist:?}");
        }
    }

    #[test]
    fn mes_and_ei_disagree_on_an_ico_shelf() {
        let mut m = FunnelModel::new(0.15, 20.0, 1e-2);
        m.observe(pt(&[1.0, 0.0, 0.0]).view(), -396.28);
        m.observe(pt(&[0.98, 0.02, 0.0]).view(), -396.30);
        let nearby = pt(&[0.97, 0.03, 0.0]);
        let unseen = pt(&[0.0, 0.0, 1.0]);
        let extras = [nearby.clone(), unseen.clone()];
        let views: Vec<ArrayView1<f64>> = extras.iter().map(|x| x.view()).collect();
        let minima = m.sample_minima(&views, 16);
        let ei_near = m.expected_improvement(nearby.view());
        let ei_far = m.expected_improvement(unseen.view());
        let mes_near = m.max_value_entropy(nearby.view(), &minima);
        let mes_far = m.max_value_entropy(unseen.view(), &minima);
        assert!(ei_far > ei_near && mes_far > mes_near);
        let ei_ratio = ei_far / ei_near.max(1e-12);
        let mes_ratio = mes_far / mes_near.max(1e-12);
        assert!(
            ei_ratio > 2.0 * mes_ratio || mes_ratio > 2.0 * ei_ratio,
            "EI ratio {ei_ratio} MES ratio {mes_ratio} (near EI {ei_near} MES {mes_near}, far EI {ei_far} MES {mes_far})"
        );
    }

    #[test]
    fn q_ei_puts_a_slot_on_the_defect() {
        let mut m = FunnelModel::new(0.15, 20.0, 1e-2);
        let ico = pt(&[1.0, 0.0, 0.0]);
        let defect = pt(&[0.0, 1.0, 0.0]);
        for _ in 0..6 {
            m.observe(ico.view(), -396.28);
        }
        m.observe(defect.view(), -380.0);
        let cands = [(0usize, ico.to_vec()), (1usize, defect.to_vec())];
        let mean = m.assign_by_mean(&cands, 4);
        assert!(
            mean.iter().all(|&id| id == 0),
            "mean-greedy WAVE {mean:?} stays on the ico floor"
        );
        let qei = m.assign_q_ei(&cands, 4);
        assert!(
            qei.contains(&1),
            "q-EI WAVE {qei:?} must send a replica to the defect"
        );
    }

    #[test]
    fn hellinger_ranks_marks_farther_than_a_near_ico() {
        let m = FunnelModel::new(0.15, 20.0, 1e-2);
        let ico = pt(&[1.0, 0.0, 0.0]);
        let near = pt(&[0.97, 0.03, 0.0]);
        let marks = pt(&[0.0, 0.0, 1.0]);
        let k_near = m.similarity(ico.view(), near.view());
        let k_marks = m.similarity(ico.view(), marks.view());
        assert!(
            k_near > k_marks,
            "near-ico kernel {k_near} must beat ico-Marks {k_marks}"
        );
    }

    #[test]
    fn gibbon_batch_reproduces_the_two_point_log_determinant() {
        let mut model = FunnelModel::new_euclidean(0.8, 2.0, 0.1);
        model.observe(pt(&[-1.0, 0.0]).view(), -2.0);
        model.observe(pt(&[1.0, 0.0]).view(), -1.0);
        let left = pt(&[-0.4, 0.7]);
        let right = pt(&[0.6, 0.8]);
        let minimum_samples = [-2.2, -2.0, -1.9];

        let left_information = model.gibbon_information(left.view(), &minimum_samples);
        let right_information = model.gibbon_information(right.view(), &minimum_samples);
        let correlation = model.predictive_observation_correlation(left.view(), right.view());
        let expected =
            left_information + right_information + 0.5 * (1.0 - correlation * correlation).ln();
        let batch = model.gibbon_batch(&[left.view(), right.view()], &minimum_samples);

        assert!((batch - expected).abs() < 1e-10);
    }

    #[test]
    fn greedy_gibbon_batch_repels_a_redundant_launch() {
        let mut model = FunnelModel::new_euclidean(0.2, 1.0, 1e-3);
        let candidates = [
            (0, vec![0.0, 0.0]),
            (1, vec![0.01, 0.0]),
            (2, vec![2.0, 0.0]),
        ];

        let selected = model.assign_gibbon(&candidates, 2, 1, 64);

        assert_eq!(selected, vec![0, 2]);
    }

    #[test]
    fn joint_posterior_matches_scalar_predictions_and_covariances() {
        let mut model = FunnelModel::new_euclidean(0.7, 2.0, 0.05);
        model.set_prior_mean(0.0);
        for (x, y) in [
            (pt(&[-1.0, 0.0]), -0.5),
            (pt(&[0.0, 0.5]), 0.25),
            (pt(&[1.0, 0.0]), -0.2),
        ] {
            model.observe(x.view(), y);
        }
        let sites = [pt(&[-0.4, 0.2]), pt(&[0.3, 0.7]), pt(&[0.8, -0.1])];
        let views = sites.iter().map(|site| site.view()).collect::<Vec<_>>();

        let (means, covariance) = model.posterior_joint(&views);

        for (index, site) in sites.iter().enumerate() {
            let (mean, standard_deviation) = model.predict(site.view());
            assert!((means[index] - mean).abs() < 1e-12);
            assert!(
                (covariance[[index, index]] - standard_deviation * standard_deviation).abs()
                    < 1e-12
            );
            for (other_index, other) in sites.iter().enumerate() {
                let expected = model.posterior_latent_covariance(site.view(), other.view());
                assert!((covariance[[index, other_index]] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pivoted_kernel_compression_bounds_rank_and_preserves_the_incumbent() {
        let mut model = FunnelModel::new_euclidean(0.2, 1.0, 1e-3);
        model.set_prior_mean(0.0);
        for index in 0..40 {
            model.observe(pt(&[0.1 * index as f64, 0.0]).view(), -(index as f64));
        }
        let incumbent = model.incumbent();

        let compression = model.compress(8);

        assert_eq!(compression.input_count, 40);
        assert!(compression.retained_rank <= 8);
        assert!(compression.residual_fraction.is_finite());
        assert!((0.0..=1.0).contains(&compression.residual_fraction));
        assert_eq!(model.len(), compression.retained_rank);
        assert_eq!(model.incumbent(), incumbent);
    }
}
