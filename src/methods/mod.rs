//! Annealing-method extensions built on top of the trait Sampler<T>
//! algebra: multi-chain MCMC-style SA with Gelman-Rubin termination,
//! parallel tempering, Bayesian pilot tuning, local polish, and
//! surrogate-assisted moves.
//!
//! The unification rationale: SA's sub-chain at fixed T is exactly an
//! MCMC Metropolis-Hastings kernel with Boltzmann-Gibbs stationary
//! distribution. Classical SA fixes the inner-loop iteration count K;
//! the MCMC tradition runs until a convergence diagnostic crosses a
//! threshold. The typed component algebra (Cool / Neigh / Move /
//! Accept) is silent on this loop control, so both styles are
//! equally well-typed points. `mcmc_sa` ships the MCMC-style point.

pub mod additive_independence;
pub mod bayesian_mixing;
pub mod bayesian_pilot;
pub mod gle_langevin;
pub mod local_polish;
pub mod mcmc_sa;
pub mod parallel_tempering;

pub use additive_independence::{additive_independence_sa, AdditiveIndependenceResult};
pub use bayesian_mixing::{BayesianMixingResult, BayesianMixingSampler};
pub use bayesian_pilot::{
    fit_laplace, pilot_draws, pilot_draws_qmc, LaplacePosterior, PilotObservation, PilotPrior,
    Q_V_MAX, Q_V_MIN, TARGET_ACCEPT_RATE,
};
pub use gle_langevin::{
    estimate_gle_omega0, gle_langevin_adaptive_sa, gle_langevin_sa, GleLangevinResult,
};
pub use local_polish::{
    projected_gradient_polish, qmc_projected_gradient_polish,
    shifted_qmc_projected_gradient_polish, LocalPolishResult, QmcPolishResult,
};
pub use mcmc_sa::{GelmanRubin, MultiChainResult, MultiChainSampler, MultiChainState};
pub use parallel_tempering::{geometric_ladder, ParallelTemperingSampler, PtChainState, PtResult};
