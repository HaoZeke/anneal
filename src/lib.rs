#![warn(missing_docs)]
//! anneal-core: simulated-annealing components on the eindir typed primitives.
//!
//! This crate ships the typed component algebra (Cool / Neigh / Move / Accept
//! traits, Boltzmann / Fast / Tsallis points, and the SaVariant tuple)
//! consumed by the Python `anneal` package via pyo3.

/// The acceptance-rule trait: `(delta_e, T) -> p`.
pub mod accept;
/// Stan-style windowed adaptation: `trait Adapter<T, S>`.
pub mod adapter;
/// Cost-augmenting bias operators (well-tempered metadynamics, etc.).
pub mod bias;
/// The cooling-schedule trait: `epoch -> temperature`.
pub mod cool;
/// Error variants returned by `anneal-core`.
pub mod error;
/// Parallel-tempering exchange operator for multi-temperature ensembles.
pub mod exchange;
/// Free-energy estimators (Bennett's BAR + descendants).
pub mod free_energy;
/// First-derivative interface for HMC-style samplers.
pub mod grad;
/// Per-epoch run history returned by `run_rs`.
pub mod history;
/// Hamiltonian Monte Carlo inside SA (Method B Phase 1).
pub mod hmc;
/// Law-witness helpers and the `LawViolation` diagnostic type.
pub mod laws;
/// Annealing-method extensions built on the typed algebra:
/// MCMC-style multi-chain SA with Gelman-Rubin termination, Bayesian
/// pilot adaptation, parallel tempering, etc.
pub mod methods;
/// The move-kernel trait: temperature-indexed proposal sampling.
pub mod movekernel;
/// The neighborhood trait: `state -> set-of-states`.
pub mod neigh;
/// The pure-Rust SA driver loop.
pub mod runner;
/// Stan-style single-step sampler trait: `trait Sampler<T>`.
pub mod sampler;
/// Shared algebra-facing type definitions.
pub mod types;
/// The fully-typed `SaVariant` tuple satisfying L1-L4.
pub mod variant;
/// Package version metadata.
pub mod version;

/// C ABI surface (cargo-c builds).
#[cfg(feature = "capi")]
pub mod ffi;
/// pyo3 module entry point exposed as `anneal._core`.
#[cfg(feature = "python")]
pub mod python;

pub use accept::AcceptRule;
pub use adapter::{Adapter, IdentityAdapter};
pub use bias::{Bias, WellTemperedBias};
pub use cool::Cooling;
pub use error::Error;
pub use exchange::{Exchange, MetropolisExchange, TsallisExchange};
pub use free_energy::BarEstimator;
pub use grad::{AnalyticGradient, DifferentiableObjective, FiniteDiffGradient, Gradient};
pub use history::{EpochLine, History, State};
pub use hmc::{
    HmcIntegrator, HmcSaSampler, LeapfrogIntegrator, LeapfrogResult, NutsSaSampler, NutsTransition,
    OMELYAN_LAMBDA, OmelyanIntegrator,
};
pub use laws::LawViolation;
pub use methods::{
    BayesianMixingResult, BayesianMixingSampler, GelmanRubin, GleLangevinResult, LaplacePosterior,
    LocalPolishResult, MultiChainResult, MultiChainSampler, MultiChainState,
    ParallelTemperingSampler, PilotObservation, PilotPrior, PtChainState, PtResult, Q_V_MAX,
    Q_V_MIN, QmcPolishResult, TARGET_ACCEPT_RATE, estimate_gle_omega0, fit_laplace,
    geometric_ladder, gle_langevin_adaptive_sa, gle_langevin_sa, pilot_draws, pilot_draws_qmc,
    projected_gradient_polish, qmc_projected_gradient_polish,
    shifted_qmc_projected_gradient_polish,
};
pub use movekernel::MoveKernel;
pub use neigh::Neighborhood;
pub use runner::{qmc_skip_from_seed, run_rs, run_rs_qmc_variant, run_rs_variant};
pub use sampler::Sampler;
pub use variant::SaVariant;
pub use version::ANNEAL_VERSION;
