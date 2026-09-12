//! First-passage model of a search chain, for choosing how to split a
//! budget over independent chains.
//!
//! One campaign of single chains records, per seed, the forces at which
//! the chain first reached the target, or that it had not by the budget's
//! end (a right-censored observation). Procacci (J. Chem. Phys. 142,
//! 154117, 2015) treats the work distribution of many non-communicating
//! nonequilibrium trajectories as a mixture whose components are the
//! metastable basins the starts fall into; the same picture fits a
//! chain's first-passage forces, whose components are the funnels a
//! random start lands in (a fast one that sits over the target and a slow
//! one that must cross a funnel boundary first). A mixture of
//! exponentials is fitted by expectation-maximisation with the censoring
//! handled exactly (the exponential is memoryless, so a censored seed's
//! expected first passage is the budget plus the component's mean). The
//! fitted distribution then gives the hit probability of `k` independent
//! chains sharing an aggregate budget, `1 - (1 - F(B/k))^k`, at any split
//! without another campaign; a mixture of exponentials has a
//! non-increasing hazard, so the bound of independent starts applies.
//!
//! A chain also has a warm-up: on LJ75 no orbit chain reaches Marks
//! before a few hundred thousand forces except the rare start that lands
//! beside it, and a memoryless mixture cannot say that one chain of 4e6
//! beats two of 2e6 (measured 31 against 20 of 48). Each component
//! therefore carries a shift below which it cannot fire; [`fit_shifted`]
//! profiles the slow component's shift over a grid and keeps the
//! likelihood's maximum.

/// One seed's outcome: the forces at first passage, or the budget it ran
/// out at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirstPassage {
    /// Reached the target after this many forces.
    Hit(f64),
    /// Had not reached it when the budget of this many forces ended.
    Censored(f64),
}

/// A mixture of exponential first-passage components.
#[derive(Debug, Clone, PartialEq)]
pub struct ExponentialMixture {
    /// Component weights, summing to one.
    pub weights: Vec<f64>,
    /// Component means beyond the shift, in forces.
    pub means: Vec<f64>,
    /// Forces below which each component cannot fire.
    pub shifts: Vec<f64>,
    /// Log-likelihood of the data the fit was made on.
    pub log_likelihood: f64,
    /// EM iterations taken.
    pub iterations: usize,
}

/// Why a fit could not be made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FirstPassageError {
    /// No observation.
    #[error("no observations")]
    Empty,
    /// Fewer components than one, or more than observations.
    #[error("invalid component count {0}")]
    Components(usize),
    /// A non-positive or non-finite value.
    #[error("observation {0} is not a positive finite force count")]
    Invalid(usize),
}

impl ExponentialMixture {
    /// Fit `components` exponential components to `observations` by EM
    /// from a spread of initial means. Deterministic: initial means are
    /// quantiles of the observed values, and the iteration stops when the
    /// log-likelihood moves by less than `1e-9` or after `max_iterations`.
    pub fn fit(
        observations: &[FirstPassage],
        components: usize,
        max_iterations: usize,
    ) -> Result<Self, FirstPassageError> {
        Self::fit_with_shifts(observations, &vec![0.0; components], max_iterations)
    }

    /// Two components, the fast one unshifted and the slow one shifted by
    /// the grid value that maximises the likelihood. `shift_grid` is in
    /// forces; an empty grid means no shift.
    pub fn fit_shifted(
        observations: &[FirstPassage],
        shift_grid: &[f64],
        max_iterations: usize,
    ) -> Result<Self, FirstPassageError> {
        let mut best: Option<Self> = None;
        let grid: Vec<f64> = if shift_grid.is_empty() {
            vec![0.0]
        } else {
            shift_grid.to_vec()
        };
        for &shift in &grid {
            let candidate = Self::fit_with_shifts(observations, &[0.0, shift], max_iterations)?;
            if best
                .as_ref()
                .is_none_or(|held| candidate.log_likelihood > held.log_likelihood)
            {
                best = Some(candidate);
            }
        }
        best.ok_or(FirstPassageError::Empty)
    }

    /// EM with every component's shift fixed.
    pub fn fit_with_shifts(
        observations: &[FirstPassage],
        shifts: &[f64],
        max_iterations: usize,
    ) -> Result<Self, FirstPassageError> {
        let components = shifts.len();
        if observations.is_empty() {
            return Err(FirstPassageError::Empty);
        }
        if components == 0 || components > observations.len() {
            return Err(FirstPassageError::Components(components));
        }
        for (index, observation) in observations.iter().enumerate() {
            let value = match observation {
                FirstPassage::Hit(v) | FirstPassage::Censored(v) => *v,
            };
            if !value.is_finite() || value <= 0.0 {
                return Err(FirstPassageError::Invalid(index));
            }
        }
        let mut values: Vec<f64> = observations
            .iter()
            .map(|o| match o {
                FirstPassage::Hit(v) | FirstPassage::Censored(v) => *v,
            })
            .collect();
        values.sort_by(f64::total_cmp);
        // Initial means spread over the observed range so the components
        // start apart: geometric between the smallest value and the largest.
        let low = values[0].max(1.0);
        let high = values[values.len() - 1].max(low * 2.0);
        let mut means: Vec<f64> = (0..components)
            .map(|i| {
                let f = (i as f64 + 0.5) / components as f64;
                (low.ln() + f * (high.ln() - low.ln())).exp()
            })
            .collect();
        let mut weights = vec![1.0 / components as f64; components];
        let mut previous = f64::NEG_INFINITY;
        let mut iterations = 0;
        let mut log_likelihood = previous;
        while iterations < max_iterations {
            iterations += 1;
            // E-step: responsibilities and the expected first passage of
            // censored seeds under each component.
            let mut sum_r = vec![0.0; components];
            let mut sum_rt = vec![0.0; components];
            log_likelihood = 0.0;
            for observation in observations {
                let (t, censored) = match observation {
                    FirstPassage::Hit(v) => (*v, false),
                    FirstPassage::Censored(v) => (*v, true),
                };
                let terms: Vec<f64> = (0..components)
                    .map(|i| {
                        let m = means[i];
                        let u = t - shifts[i];
                        if censored {
                            // Survival: one below the shift.
                            weights[i] * if u <= 0.0 { 1.0 } else { (-u / m).exp() }
                        } else if u <= 0.0 {
                            0.0
                        } else {
                            weights[i] * (-u / m).exp() / m
                        }
                    })
                    .collect();
                let total: f64 = terms.iter().sum();
                if total <= 0.0 || !total.is_finite() {
                    // A hit below every shift has zero density; it counts
                    // against the fit through the likelihood floor.
                    log_likelihood += f64::MIN_POSITIVE.ln();
                    continue;
                }
                log_likelihood += total.ln();
                for i in 0..components {
                    let r = terms[i] / total;
                    sum_r[i] += r;
                    // Expected first passage beyond the shift: a censored
                    // seed still below the shift waits the mean from there.
                    let beyond = if censored {
                        (t - shifts[i]).max(0.0) + means[i]
                    } else {
                        t - shifts[i]
                    };
                    sum_rt[i] += r * beyond;
                }
            }
            // M-step.
            let n = observations.len() as f64;
            for i in 0..components {
                if sum_r[i] > 0.0 {
                    weights[i] = sum_r[i] / n;
                    means[i] = (sum_rt[i] / sum_r[i]).max(1.0);
                }
            }
            if (log_likelihood - previous).abs() < 1e-9 {
                break;
            }
            previous = log_likelihood;
        }
        Ok(Self {
            weights,
            means,
            shifts: shifts.to_vec(),
            log_likelihood,
            iterations,
        })
    }

    /// Probability that one chain has reached the target within `budget`
    /// forces.
    pub fn hit_probability(&self, budget: f64) -> f64 {
        if budget <= 0.0 {
            return 0.0;
        }
        self.weights
            .iter()
            .zip(&self.means)
            .zip(&self.shifts)
            .map(|((w, m), s)| {
                let u = budget - s;
                if u <= 0.0 {
                    0.0
                } else {
                    w * (1.0 - (-u / m).exp())
                }
            })
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    /// Probability that at least one of `chains` independent chains sharing
    /// `aggregate` forces equally has reached the target.
    pub fn ensemble_hit_probability(&self, chains: usize, aggregate: f64) -> f64 {
        if chains == 0 {
            return 0.0;
        }
        let per_chain = self.hit_probability(aggregate / chains as f64);
        1.0 - (1.0 - per_chain).powi(i32::try_from(chains).unwrap_or(i32::MAX))
    }

    /// The chain count in `candidates` with the highest ensemble hit
    /// probability at `aggregate` forces, with that probability.
    pub fn best_split(&self, aggregate: f64, candidates: &[usize]) -> Option<(usize, f64)> {
        candidates
            .iter()
            .map(|&k| (k, self.ensemble_hit_probability(k, aggregate)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }
}
