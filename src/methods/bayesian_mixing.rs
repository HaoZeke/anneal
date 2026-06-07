//! Automatic Bayesian chain-mixing driver.
//!
//! The public surface deliberately has one tuning knob: a proposal budget.
//! Chain count, incumbent protection, and proposal allocation are derived
//! online from a Beta-Bernoulli posterior over global-best improvements.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Beta, Distribution};

use crate::history::{EpochLine, History, State};
use crate::runner::qmc_skip_from_seed;
use crate::sampler::Sampler;

/// Result of an automatic Bayesian chain-mixing run.
#[derive(Clone, Debug)]
pub struct BayesianMixingResult {
    /// One compact history per chain.
    pub chain_histories: Vec<History>,
    /// Best-seen position across all chains.
    pub best_pos: Vec<f64>,
    /// Best-seen objective value across all chains.
    pub best_val: f64,
    /// Number of chains inferred from the budget and dimension.
    pub n_chains: usize,
    /// Proposal counts allocated to each chain.
    pub proposal_counts: Vec<usize>,
}

impl BayesianMixingResult {
    /// Total proposal calls made after chain initialisation.
    pub fn total_proposals(&self) -> usize {
        self.proposal_counts.iter().sum()
    }
}

/// Automatic Bayesian chain mixer around any inner sampler.
pub struct BayesianMixingSampler<S: Sampler<f64>> {
    /// Shared inner sampler.
    pub sampler: S,
    /// Total proposal budget after chain initialisation.
    pub max_proposals: usize,
}

impl<S: Sampler<f64>> BayesianMixingSampler<S> {
    /// Creates a Bayesian mixer with a single user-visible budget.
    pub fn new(sampler: S, max_proposals: usize) -> Self {
        assert!(max_proposals > 0, "max_proposals must be positive");
        Self {
            sampler,
            max_proposals,
        }
    }

    /// Runs the online allocation loop.
    pub fn run(&self, seed: u64) -> BayesianMixingResult {
        let qmc_bounds = self.sampler.qmc_bounds();
        let dim = qmc_bounds.map(|bounds| bounds.dims).unwrap_or_else(|| {
            let mut first_rng = StdRng::seed_from_u64(seed);
            self.sampler.initial_state(&mut first_rng).cur.pos.len()
        });
        let n_chains = auto_chain_count(dim, self.max_proposals);

        let mut rngs: Vec<StdRng> = Vec::with_capacity(n_chains);
        let mut states: Vec<State> = Vec::with_capacity(n_chains);
        if let Some(bounds) = qmc_bounds {
            let starts =
                eindir_core::low_discrepancy_points(bounds, n_chains, qmc_skip_from_seed(seed));
            for c in 0..n_chains {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(c as u64));
                let state = self
                    .sampler
                    .initial_state_from_position(starts.row(c).to_owned())
                    .unwrap_or_else(|| self.sampler.initial_state(&mut rng));
                states.push(state);
                rngs.push(rng);
            }
        } else {
            for c in 0..n_chains {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(c as u64));
                states.push(self.sampler.initial_state(&mut rng));
                rngs.push(rng);
            }
        }

        let mut best_idx = 0usize;
        for c in 1..n_chains {
            if states[c].best.val < states[best_idx].best.val {
                best_idx = c;
            }
        }
        let mut best_val = states[best_idx].best.val;
        let mut best_pos = states[best_idx].best.pos.to_vec();
        let mut incumbent_chain = 0usize;
        let mut controller_rng = StdRng::seed_from_u64(seed.wrapping_add(10_007));

        let mut improve_alpha = vec![1.0_f64; n_chains];
        let mut improve_beta = vec![1.0_f64; n_chains];
        improve_alpha[0] = 4.0;
        for beta in improve_beta.iter_mut().skip(1) {
            *beta = 4.0;
        }
        let mut proposal_counts = vec![0usize; n_chains];
        let mut accepted = vec![0usize; n_chains];
        let mut rejected = vec![0usize; n_chains];

        for proposal_idx in 0..self.max_proposals {
            let chain_idx = select_chain(
                incumbent_chain,
                &improve_alpha,
                &improve_beta,
                &mut controller_rng,
            );
            let was_accepted =
                self.sampler
                    .step(&mut states[chain_idx], proposal_idx, &mut rngs[chain_idx]);
            proposal_counts[chain_idx] += 1;
            if was_accepted {
                accepted[chain_idx] += 1;
            } else {
                rejected[chain_idx] += 1;
            }

            if states[chain_idx].best.val < best_val {
                best_val = states[chain_idx].best.val;
                best_pos = states[chain_idx].best.pos.to_vec();
                incumbent_chain = chain_idx;
                improve_alpha[chain_idx] += 1.0;
            } else {
                improve_beta[chain_idx] += 1.0;
            }
        }

        let chain_histories = states
            .iter()
            .enumerate()
            .map(|(c, state)| {
                let mut history = History::with_capacity(1, state.best.clone());
                history.epochs.push(EpochLine {
                    epoch: 0,
                    temp: 0.0,
                    accepted: accepted[c],
                    rejected: rejected[c],
                    best_val: state.best.val,
                });
                history
            })
            .collect();

        BayesianMixingResult {
            chain_histories,
            best_pos,
            best_val,
            n_chains,
            proposal_counts,
        }
    }
}

fn auto_chain_count(dim: usize, max_proposals: usize) -> usize {
    let budget_limited = (max_proposals / 64).clamp(2, 4);
    let dim_limited = ((dim.max(1) as f64).sqrt().ceil() as usize).clamp(2, 4);
    budget_limited.min(dim_limited).min(max_proposals).max(1)
}

fn select_chain(incumbent_chain: usize, alpha: &[f64], beta: &[f64], rng: &mut StdRng) -> usize {
    let mut best_idx = incumbent_chain;
    let mut best_score = beta_draw(alpha[incumbent_chain], beta[incumbent_chain], rng);
    for idx in 0..alpha.len() {
        let score = beta_draw(alpha[idx], beta[idx], rng);
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    let incumbent_score = beta_draw(alpha[incumbent_chain], beta[incumbent_chain], rng);
    if best_idx != incumbent_chain && best_score > incumbent_score + 0.05 {
        best_idx
    } else {
        incumbent_chain
    }
}

fn beta_draw(alpha: f64, beta: f64, rng: &mut StdRng) -> f64 {
    Beta::new(alpha, beta)
        .expect("positive beta posterior parameters")
        .sample(rng)
}
