#![warn(missing_docs)]
// Fixed-axis tensor contractions are written with index notation so each loop
// follows the corresponding equation and preserves its bounds visibly.
#![allow(clippy::needless_range_loop)]
// Numerical kernels keep physical inputs explicit; packing them into transport
// structs would obscure the equations without reducing the required state.
#![allow(clippy::too_many_arguments)]
// Callback and evidence tuples encode coupled numerical contracts at their use
// sites, where aliases would hide rather than clarify their components.
#![allow(clippy::type_complexity)]
// Protocol enums keep payloads inline to avoid allocation in the cooperative
// search path; variant size is part of that performance choice.
#![allow(clippy::large_enum_variant)]
// Negated floating-point bounds intentionally classify NaN as invalid. Rewriting
// them as ordered comparisons silently admits NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
//! anneal-core: simulated-annealing components on the eindir typed primitives.
//!
//! This crate ships the typed component algebra (Cool / Neigh / Move / Accept
//! traits, Boltzmann / Fast / Tsallis points, and the SaVariant tuple)
//! consumed by the Python `anneal` package via pyo3.

/// Shared CSA bank over Cap'n Proto. Optional; process-local bank stays default.
#[cfg(feature = "bank-rpc")]
pub mod bank_rpc;
/// Generated Cap'n Proto schema for [`bank_rpc`].
#[cfg(feature = "bank-rpc")]
#[allow(missing_docs, dead_code, clippy::all, non_camel_case_types)]
#[rustfmt::skip]
#[allow(non_snake_case)]
pub mod Bank_capnp {
    include!("bank_rpc/Bank_capnp.rs");
}
/// Generated Cap'n Proto schema for [`raft`] consensus traffic.
#[cfg(feature = "bank-rpc")]
#[allow(missing_docs, dead_code, clippy::all, non_camel_case_types)]
#[rustfmt::skip]
#[allow(non_snake_case)]
pub mod Raft_capnp {
    include!("Raft_capnp.rs");
}
/// Generated Cap'n Proto schema for [`catalog_rpc`].
#[cfg(feature = "bank-rpc")]
#[allow(missing_docs, dead_code, clippy::all, non_camel_case_types)]
#[rustfmt::skip]
#[allow(non_snake_case)]
pub mod Catalog_capnp {
    include!("catalog_rpc/Catalog_capnp.rs");
}

/// The acceptance-rule trait: `(delta_e, T) -> p`.
pub mod accept;
/// ACE ν=3 / λ-SOAP CG contraction of a spherical expansion.
pub mod ace;
/// Stan-style windowed adaptation: `trait Adapter<T, S>`.
pub mod adapter;
/// Cost-augmenting bias operators (well-tempered metadynamics, etc.).
/// Proposal allocation and the budget-window temperature law.
pub mod allocate;
pub mod bias;
/// Transport proposals from observed attraction-region crossings.
pub mod boundary_transport;
/// Spectral statistics of the curvature, without forming a Hessian.
pub mod calibrate;
/// Event catalogue keyed by local topology.
pub mod catalog;
/// Pure cooperative-catalog policy and target-free proposals.
pub mod catalog_policy;
/// Versioned cooperative catalog protocol.
#[cfg(feature = "bank-rpc")]
pub mod catalog_rpc;
/// Collective variables from the spectrum of the visited-basin graph.
pub mod construct;
/// The cooling-schedule trait: `epoch -> temperature`.
pub mod contextual;
pub mod cool;
pub mod cooperative_search;
pub mod curvature;
pub mod delayed;
/// Versioned multiscale invariant coordinates for basin catalogs.
pub mod descriptor_space;
/// Annealing the distance at which two solutions count as one.
pub mod diversity;
/// A posterior over the density of minima, and acceptance by entropy rather
/// than by energy.
pub mod dos;
/// Error variants returned by `anneal-core`.
pub mod error;
/// Parallel-tempering exchange operator for multi-temperature ensembles.
pub mod exchange;
/// featomic SOAP power-spectrum leftover hop.
#[cfg(feature = "featomic")]
pub mod featomic_hop;
/// Energy-floor flicker components and record EI.
pub mod floors;
/// Free-energy estimators (Bennett's BAR + descendants).
pub mod free_energy;
/// First-derivative interface for HMC-style samplers.
pub mod funnel_bo;
pub mod funnel_spectral;
pub mod grad;
/// Exact basin identity by canonical contact-graph labelling (nauty).
#[cfg(feature = "graphkey")]
pub mod graphkey;
/// Per-epoch run history returned by `run_rs`.
pub mod history;
/// Hamiltonian Monte Carlo inside SA (Method B Phase 1).
pub mod hmc;
pub mod lattice;
/// Law-witness helpers and the `LawViolation` diagnostic type.
pub mod laws;
/// Local two-shell topology keys (k-ART).
#[cfg(feature = "graphkey")]
pub mod localkey;
/// Short MD segments through external engines (LAMMPS, GROMACS).
pub mod md_engine;
/// Leader consensus over exploration decrees between server brains.
pub mod raft;
/// Annealing-method extensions built on the typed algebra:
/// MCMC-style multi-chain SA with Gelman-Rubin termination, Bayesian
/// pilot adaptation, parallel tempering, etc.
pub mod methods;
pub mod model_hessian;
/// nng transport for consensus traffic between server brains.
#[cfg(all(feature = "nng-transport", feature = "bank-rpc"))]
pub mod decree_bus;
/// The move-kernel trait: temperature-indexed proposal sampling.
pub mod movekernel;
/// The neighborhood trait: `state -> set-of-states`.
pub mod neigh;
/// Incremental neighbour table shared across the hop.
pub mod neighbors;
/// Noise-aware acceptance (Ball, Branke & Meisel 2018 sequential OSA rule).
pub mod noise_accept;
/// The pure-Rust SA driver loop.
/// Multi-step paths between minima, where one hop cannot cross.
pub mod path;
pub mod potentials;
pub mod quench;
/// GMRF residual intensity on the class graph.
pub mod residual_field;
/// Joint population assignment across fixed-probe attraction regions.
pub mod region_assignment;
pub mod runner;
/// Stan-style single-step sampler trait: `trait Sampler<T>`.
pub mod sampler;
pub mod screen;
/// SOAP power spectrum and Cartesian pullback through `∂p/∂R`.
pub mod soap;
pub mod spectral;
pub mod structure;
/// How a leftover chain uses the shared packing catalog.
pub mod swarm;
pub mod symmetrise;
pub mod terminate;
/// Action-conditioned structural transition evidence.
pub mod transition_graph;
pub mod twin;
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
