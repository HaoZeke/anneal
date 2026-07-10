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
/// Noise-aware acceptance (Ball, Branke & Meisel 2018 sequential OSA rule).
pub mod noise_accept;
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
    OmelyanIntegrator, OMELYAN_LAMBDA,
};
pub use laws::LawViolation;
pub use methods::{
    arm_prior_boost, arm_slice_multiplier, check_accept_path, estimate_gle_omega0,
    estimate_gle_preconditioner, exact_accept_allowed, fit_laplace, geometric_ladder,
    gle_langevin_adaptive_sa, gle_langevin_preconditioned_sa, gle_langevin_sa, order_arms,
    preferred_arm_tail, pilot_draws, regime_exploit_prob, regime_exploit_width,
    pilot_draws_qmc, portfolio_optimize, portfolio_optimize_with_policy, projected_gradient_polish,
    qmc_best1bin_scout, qmc_gsa_global_search, qmc_projected_gradient_polish, qmc_trust_region_poll,
    require_accept_compatible, select_regime, shifted_qmc_projected_gradient_polish, ArmStat,
    BayesianMixingResult, BayesianMixingSampler, GelmanRubin, GleLangevinResult, GlePreconditioner,
    LaplacePosterior, LocalPolishResult, MultiChainResult, MultiChainSampler, MultiChainState,
    OptimizationRegime, ParallelTemperingSampler, PilotObservation, PilotPrior, PortfolioPolicy,
    PortfolioResult, ProblemFeatures, PtChainState, PtResult, QmcPolishResult, RegimeError,
    TpeCategorical, TpeContinuous1d, ShootDirection, Q_V_MAX, Q_V_MIN, TARGET_ACCEPT_RATE,
    DEFAULT_ALPHA, DEFAULT_GAMMA,
};
pub use movekernel::{MoveKernel, Reflected};
pub use neigh::Neighborhood;
pub use noise_accept::{OsaAccept, OsaResult};
pub use runner::{
    qmc_skip_from_seed, run_rs, run_rs_qmc_variant, run_rs_variant, run_rs_variant_resumed,
};
pub use sampler::Sampler;
pub use variant::SaVariant;
pub use version::ANNEAL_VERSION;
