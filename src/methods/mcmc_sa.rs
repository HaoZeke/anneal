//! MCMC-style simulated annealing with Gelman-Rubin convergence
//! termination at each epoch.
//!
//! Classical SA runs a fixed number of inner-loop steps `K` per
//! temperature epoch. MCMC tradition runs until a between-chain
//! convergence diagnostic crosses a threshold (typically `Rhat < 1.1`
//! across `M >= 2` chains). The two are mathematically identical at
//! fixed temperature -- both implement a Metropolis-Hastings kernel
//! with Boltzmann-Gibbs stationary distribution -- but differ in
//! loop control. This module ships the MCMC-style point of the
//! typed algebra.
//!
//! See the IISE manuscript Section 8.3 "Extension to distributed and
//! parallel-tempering settings" and design_pass_07 in obsidian-notes
//! for the full unification story.

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::history::{EpochLine, History, State};
use crate::sampler::Sampler;

/// Per-chain state for a multi-chain MCMC-SA run.
#[derive(Clone, Debug)]
pub struct MultiChainState {
    /// One per-chain `State`.
    pub chains: Vec<State>,
    /// Per-chain RNG state, advanced independently.
    pub seeds: Vec<u64>,
}

/// Gelman-Rubin convergence diagnostic on a multi-chain trace.
///
/// For `M` chains of length `N`, define
///   - chain mean `theta_m = mean over draws in chain m`
///   - chain variance `s2_m = sample variance within chain m`
///   - between-chain variance `B = N / (M - 1) * sum_m (theta_m - theta_bar)^2`
///   - within-chain variance `W = (1 / M) * sum_m s2_m`
///   - pooled estimate `Var = (N - 1)/N * W + B/N`
///   - Rhat = sqrt(Var / W)
///
/// `M >= 2` and `N >= 2` required; otherwise returns `f64::INFINITY`.
/// We track per-coordinate Rhat and return the maximum across coords;
/// SA practice converges on the worst-mixing coordinate.
pub struct GelmanRubin;

impl GelmanRubin {
    /// Computes max-per-coordinate Rhat from a `(n_chains, n_draws, dim)` view
    /// passed as a flat slice with row-major `(chain, draw, dim)` layout.
    pub fn compute(traces: &[Vec<Vec<f64>>]) -> f64 {
        let m = traces.len();
        if m < 2 {
            return f64::INFINITY;
        }
        let n = traces[0].len();
        if n < 2 {
            return f64::INFINITY;
        }
        let dim = traces[0][0].len();
        let mut max_rhat: f64 = 0.0;
        for d in 0..dim {
            let mut chain_means = Vec::with_capacity(m);
            let mut chain_vars = Vec::with_capacity(m);
            for chain in traces.iter().take(m) {
                let mean: f64 = chain.iter().map(|x| x[d]).sum::<f64>() / n as f64;
                chain_means.push(mean);
                let var: f64 = chain
                    .iter()
                    .map(|x| (x[d] - mean).powi(2))
                    .sum::<f64>()
                    / (n as f64 - 1.0);
                chain_vars.push(var);
            }
            let theta_bar: f64 = chain_means.iter().sum::<f64>() / m as f64;
            let b: f64 = (n as f64 / (m as f64 - 1.0))
                * chain_means.iter().map(|t| (t - theta_bar).powi(2)).sum::<f64>();
            let w: f64 = chain_vars.iter().sum::<f64>() / m as f64;
            if w <= 0.0 {
                continue;
            }
            let var_hat = ((n as f64 - 1.0) / n as f64) * w + b / n as f64;
            let rhat = (var_hat / w).sqrt();
            if rhat > max_rhat {
                max_rhat = rhat;
            }
        }
        max_rhat
    }
}

/// MCMC-style multi-chain SA driver.
///
/// At each epoch:
///   1. Run `k_min` steps per chain (minimum-history requirement for the
///      Gelman-Rubin computation, which needs `n >= 2` per chain to be
///      defined).
///   2. Compute Rhat across chains. If `Rhat < threshold`, advance to
///      the next epoch.
///   3. Otherwise run another `k_check` steps and recheck. Bail out at
///      `k_max` total steps for the epoch even if `Rhat >= threshold`
///      (avoids non-convergent infinite loops on stiff objectives).
pub struct MultiChainSampler<S: Sampler<f64>> {
    /// The per-chain sampler (shared across chains; each chain owns its
    /// `State` + RNG).
    pub sampler: S,
    /// Number of chains.
    pub n_chains: usize,
    /// Minimum inner-loop steps per epoch before checking Rhat.
    pub k_min: usize,
    /// Step batch size between Rhat checks after `k_min`.
    pub k_check: usize,
    /// Hard cap on inner-loop steps per epoch.
    pub k_max: usize,
    /// Convergence threshold (typical: 1.1).
    pub rhat_threshold: f64,
}

/// Aggregated multi-chain history: one `History` per chain plus the
/// per-epoch Rhat trace.
#[derive(Debug)]
pub struct MultiChainResult {
    /// One `History` per chain.
    pub chain_histories: Vec<History>,
    /// Final Rhat at each epoch.
    pub epoch_rhat: Vec<f64>,
    /// Inner-loop step count used at each epoch (varies; the MCMC payoff).
    pub epoch_steps: Vec<usize>,
}

impl<S: Sampler<f64>> MultiChainSampler<S> {
    /// Drives the multi-chain MCMC-SA loop. Each chain's RNG is seeded
    /// from `seed.wrapping_add(chain_idx as u64)` for reproducibility.
    pub fn run(
        &self,
        cooling: &dyn crate::cool::Cooling<f64>,
        n_epochs: usize,
        seed: u64,
    ) -> MultiChainResult {
        let mut rngs: Vec<StdRng> = (0..self.n_chains)
            .map(|c| StdRng::seed_from_u64(seed.wrapping_add(c as u64)))
            .collect();
        let mut states: Vec<State> = rngs
            .iter_mut()
            .map(|rng| self.sampler.initial_state(rng))
            .collect();
        let mut chain_histories: Vec<History> = states
            .iter()
            .map(|s| History::with_capacity(n_epochs, s.best.clone()))
            .collect();
        let mut epoch_rhat: Vec<f64> = Vec::with_capacity(n_epochs);
        let mut epoch_steps: Vec<usize> = Vec::with_capacity(n_epochs);

        for epoch in 0..n_epochs {
            let temp = cooling.temperature(epoch);
            let mut accepted_per_chain = vec![0usize; self.n_chains];
            let mut rejected_per_chain = vec![0usize; self.n_chains];
            let mut traces: Vec<Vec<Vec<f64>>> =
                (0..self.n_chains).map(|_| Vec::with_capacity(self.k_min)).collect();

            // Phase 1: minimum-K steps.
            for _ in 0..self.k_min {
                for c in 0..self.n_chains {
                    if self.sampler.step(&mut states[c], epoch, &mut rngs[c]) {
                        accepted_per_chain[c] += 1;
                    } else {
                        rejected_per_chain[c] += 1;
                    }
                    traces[c].push(states[c].cur.pos.to_vec());
                }
            }

            let mut total_steps = self.k_min;
            let mut current_rhat = GelmanRubin::compute(&traces);

            // Phase 2: keep stepping in batches of k_check until convergence
            // or k_max budget.
            while current_rhat > self.rhat_threshold && total_steps < self.k_max {
                for _ in 0..self.k_check {
                    for c in 0..self.n_chains {
                        if self.sampler.step(&mut states[c], epoch, &mut rngs[c]) {
                            accepted_per_chain[c] += 1;
                        } else {
                            rejected_per_chain[c] += 1;
                        }
                        traces[c].push(states[c].cur.pos.to_vec());
                    }
                }
                total_steps += self.k_check;
                current_rhat = GelmanRubin::compute(&traces);
                if total_steps >= self.k_max {
                    break;
                }
            }

            for c in 0..self.n_chains {
                chain_histories[c].epochs.push(EpochLine {
                    epoch,
                    temp,
                    accepted: accepted_per_chain[c],
                    rejected: rejected_per_chain[c],
                    best_val: states[c].best.val,
                });
            }
            epoch_rhat.push(current_rhat);
            epoch_steps.push(total_steps);
        }

        for (c, hist) in chain_histories.iter_mut().enumerate() {
            hist.best = states[c].best.clone();
        }

        MultiChainResult {
            chain_histories,
            epoch_rhat,
            epoch_steps,
        }
    }
}
