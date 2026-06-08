//! Parallel-tempering driver. Wraps any `Sampler<f64>` and runs M of
//! them in parallel at a geometric temperature ladder, with adjacent-
//! pair Metropolis swaps every `swap_period` steps.
//!
//! Composition: PT is orthogonal to the inner Sampler. Whether the
//! inner kernel is random-walk Metropolis (`SaVariant`),
//! HMC (`HmcSaSampler`), or a custom sampler, PT lifts it to a
//! multi-temperature ensemble.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::cool::Cooling;
use crate::exchange::{Exchange, MetropolisExchange};
use crate::history::{EpochLine, History, State};
use crate::sampler::Sampler;

/// Per-temperature chain state for a PT run.
#[derive(Clone, Debug)]
pub struct PtChainState {
    /// Inner-sampler state.
    pub state: State,
    /// Temperature this chain operates at.
    pub temp: f64,
}

/// Result of a PT run: per-temperature chain history + global best.
#[derive(Clone, Debug)]
pub struct PtResult {
    /// One History per chain (M entries).
    pub chain_histories: Vec<History>,
    /// Best-seen position across all chains.
    pub best_pos: Vec<f64>,
    /// Best-seen objective value across all chains.
    pub best_val: f64,
    /// Number of swap attempts.
    pub swap_attempts: usize,
    /// Number of accepted swaps.
    pub swap_accepts: usize,
}

/// Geometric temperature ladder spanning `[t_cold, t_hot]` with
/// `n_chains` rungs. `T_0 = t_cold` (production temperature),
/// `T_{M-1} = t_hot` (broad-exploration temperature).
pub fn geometric_ladder(t_cold: f64, t_hot: f64, n_chains: usize) -> Vec<f64> {
    assert!(n_chains >= 2, "PT requires at least 2 chains");
    assert!(t_cold > 0.0 && t_hot > t_cold);
    (0..n_chains)
        .map(|i| {
            let r = i as f64 / (n_chains - 1) as f64;
            t_cold * (t_hot / t_cold).powf(r)
        })
        .collect()
}

/// PT driver. Constraint: the inner sampler's `Cooling` is bypassed --
/// each chain operates at a fixed temperature from the ladder, NOT at
/// the cooling schedule's epoch-indexed value. PT and cooling are
/// orthogonal in this driver.
pub struct ParallelTemperingSampler<S: Sampler<f64>, E: Exchange<f64> = MetropolisExchange> {
    /// Shared inner-sampler logic.
    pub sampler: S,
    /// The exchange operator. Defaults to `MetropolisExchange`.
    pub exchange: E,
    /// Temperature ladder (length = n_chains).
    pub temps: Vec<f64>,
    /// Inner-loop step count per epoch per chain.
    pub k_inner: usize,
    /// Swap attempt period: try a swap every `swap_period` inner-loop steps.
    pub swap_period: usize,
}

impl<S: Sampler<f64>> ParallelTemperingSampler<S, MetropolisExchange> {
    /// Constructs with default Metropolis exchange.
    pub fn new(sampler: S, temps: Vec<f64>, k_inner: usize, swap_period: usize) -> Self {
        Self {
            sampler,
            exchange: MetropolisExchange,
            temps,
            k_inner,
            swap_period,
        }
    }
}

impl<S: Sampler<f64>, E: Exchange<f64>> ParallelTemperingSampler<S, E> {
    /// Constructs with a user-chosen exchange operator.
    pub fn with_exchange(
        sampler: S,
        exchange: E,
        temps: Vec<f64>,
        k_inner: usize,
        swap_period: usize,
    ) -> Self {
        Self {
            sampler,
            exchange,
            temps,
            k_inner,
            swap_period,
        }
    }

    /// Runs the PT loop for `n_epochs` epochs.
    ///
    /// Each chain's RNG is seeded from `seed.wrapping_add(c as u64)`.
    /// The shared swap RNG is `seed.wrapping_add(M+1)`. Per epoch we
    /// run `k_inner` inner-loop steps per chain, attempting adjacent-
    /// pair swaps every `swap_period` steps. The Cool inside the inner
    /// sampler is ignored at the chain temperature; the swap rule
    /// targets the Boltzmann distribution at each chain's fixed temp.
    pub fn run<C>(&self, _cooling: &C, n_epochs: usize, seed: u64) -> PtResult
    where
        C: Cooling<f64>,
    {
        let m = self.temps.len();
        let mut rngs: Vec<StdRng> = (0..m)
            .map(|c| StdRng::seed_from_u64(seed.wrapping_add(c as u64)))
            .collect();
        let mut swap_rng = StdRng::seed_from_u64(seed.wrapping_add(m as u64 + 1));
        let mut states: Vec<State> = rngs
            .iter_mut()
            .map(|rng| self.sampler.initial_state(rng))
            .collect();
        let mut chain_histories: Vec<History> = states
            .iter()
            .map(|s| History::with_capacity(n_epochs, s.best.clone()))
            .collect();
        let mut best_val = f64::INFINITY;
        let mut best_pos = states[0].cur.pos.to_vec();
        let mut swap_attempts = 0;
        let mut swap_accepts = 0;

        for epoch in 0..n_epochs {
            let mut accepted = vec![0usize; m];
            let mut rejected = vec![0usize; m];
            for inner in 0..self.k_inner {
                for c in 0..m {
                    if self.sampler.step(&mut states[c], epoch, &mut rngs[c]) {
                        accepted[c] += 1;
                    } else {
                        rejected[c] += 1;
                    }
                    if states[c].cur.val < best_val {
                        best_val = states[c].cur.val;
                        best_pos = states[c].cur.pos.to_vec();
                    }
                }
                if (inner + 1) % self.swap_period == 0 {
                    let i = swap_rng.random_range(0..m - 1);
                    let alpha = self.exchange.swap_accept_prob(
                        states[i].cur.val,
                        self.temps[i],
                        states[i + 1].cur.val,
                        self.temps[i + 1],
                    );
                    swap_attempts += 1;
                    let u: f64 = swap_rng.random();
                    if u < alpha {
                        states.swap(i, i + 1);
                        swap_accepts += 1;
                    }
                }
            }
            for c in 0..m {
                chain_histories[c].epochs.push(EpochLine {
                    epoch,
                    temp: self.temps[c],
                    accepted: accepted[c],
                    rejected: rejected[c],
                    best_val: states[c].best.val,
                });
            }
        }

        PtResult {
            chain_histories,
            best_pos,
            best_val,
            swap_attempts,
            swap_accepts,
        }
    }
}
