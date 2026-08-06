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
/// Proposal allocation and the budget-window temperature law.
pub mod allocate;
pub mod bias;
/// The cooling-schedule trait: `epoch -> temperature`.
pub mod contextual;
pub mod cool;
/// Spectral statistics of the curvature, without forming a Hessian.
pub mod calibrate;
pub mod curvature;
/// Error variants returned by `anneal-core`.
/// Annealing the distance at which two solutions count as one.
pub mod diversity;
pub mod error;
/// Parallel-tempering exchange operator for multi-temperature ensembles.
pub mod exchange;
/// Free-energy estimators (Bennett's BAR + descendants).
pub mod free_energy;
/// First-derivative interface for HMC-style samplers.
pub mod funnel_bo;
pub mod funnel_spectral;
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
/// Multi-step paths between minima, where one hop cannot cross.
pub mod path;
pub mod potentials;
pub mod runner;
/// Collective variables from the spectrum of the visited-basin graph.
pub mod construct;
pub mod delayed;
pub mod lattice;
pub mod model_hessian;
pub mod quench;
pub mod screen;
pub mod twin;
pub mod spectral;
pub mod structure;
pub mod symmetrise;
/// Replica exchange: the bias-aware swap ratio, the non-reversible sweep, and
/// a ladder placed by the communication barrier the run measures.
pub mod tempering;
pub mod terminate;
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
#[cfg(feature = "ira")]
pub mod shape;

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
pub use tempering::{
    IndexProcess, Ladder, ReplicaMove, ReplicaTarget, SwapScheme, TARGET_SWAP_ACCEPT,
    beta_step_for_acceptance, biased_swap_log_ratio, swap_log_ratio, swap_probability,
};
pub use methods::{
    AmsaResult, ArmStat, BayesianMixingResult, BayesianMixingSampler, DEFAULT_ALPHA, DEFAULT_BETA0,
    DEFAULT_GAMMA, DEFAULT_STEPS_PER_CONTROL, DEFAULT_TARGET_WALKERS, DmcPopulationResult,
    GelmanRubin, GleLangevinResult, GlePreconditioner, LaplacePosterior, LocalPolishResult,
    MultiChainResult, MultiChainSampler, MultiChainState, OptimizationRegime,
    ParallelTemperingSampler, PilotObservation, PilotPrior, Population, PortfolioPolicy,
    PortfolioResult, ProblemFeatures, PtChainState, PtResult, Q_V_MAX, Q_V_MIN, QmcPolishResult,
    RegimeError, ShootDirection, TARGET_ACCEPT_RATE, TpeCategorical, TpeContinuous1d, Walker,
    amsa_optimize, arm_prior_boost, arm_slice_multiplier, check_accept_path, default_sigma,
    diffusion_displace, dmc_population_optimize, estimate_gle_omega0, estimate_gle_preconditioner,
    exact_accept_allowed, fit_laplace, geometric_ladder, gle_langevin_adaptive_sa,
    gle_langevin_preconditioned_sa, gle_langevin_sa, order_arms, pilot_draws, pilot_draws_qmc,
    population_control, portfolio_optimize, portfolio_optimize_with_policy, preferred_arm_tail,
    projected_gradient_polish, qmc_best1bin_scout, qmc_gsa_global_search,
    qmc_projected_gradient_polish, qmc_trust_region_poll, regime_exploit_prob,
    regime_exploit_width, require_accept_compatible, run_dmc_population, select_regime,
    shifted_qmc_projected_gradient_polish, walker_weight,
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
