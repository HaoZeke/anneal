//! Annealing-method extensions built on top of the trait Sampler<T>
//! algebra: multi-chain MCMC-style SA with Gelman-Rubin termination,
//! adiabatic schedule wrappers (future), surrogate-accelerated moves
//! (future).
//!
//! The unification rationale: SA's sub-chain at fixed T is exactly an
//! MCMC Metropolis-Hastings kernel with Boltzmann-Gibbs stationary
//! distribution. Classical SA fixes the inner-loop iteration count K;
//! the MCMC tradition runs until a convergence diagnostic crosses a
//! threshold. The typed component algebra (Cool / Neigh / Move /
//! Accept) is silent on this loop control, so both styles are
//! equally well-typed points. `mcmc_sa` ships the MCMC-style point.

pub mod bayesian_pilot;
pub mod mcmc_sa;
pub mod parallel_tempering;

pub use bayesian_pilot::{
    LaplacePosterior, PilotObservation, PilotPrior, TARGET_ACCEPT_RATE, fit_laplace, pilot_draws,
};
pub use mcmc_sa::{GelmanRubin, MultiChainResult, MultiChainSampler, MultiChainState};
pub use parallel_tempering::{ParallelTemperingSampler, PtChainState, PtResult, geometric_ladder};
