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

/// History-conditioned escape feedback, after Goedecker's minima hopping.
pub mod activation;
/// Residual archive search: local events, floors, cheap novelty loop.
#[cfg(feature = "graphkey")]
pub mod archive_search;
pub mod bank;
/// Basin hopping over quenched minima with a basin-keyed bias.
pub mod cluster_hopping;
pub mod cluster_search;
/// Population resampled by estimated probability of improvement.
pub mod committor_pop;
pub mod csa_cluster;
/// Archive-ratcheted exploration of the minima network.
#[cfg(feature = "graphkey")]
pub mod ffs;
/// Spectral referee over the explored landscape's transition graph.
pub mod landscape_graph;
pub mod minima_hopping;
/// Nested search: population under a descending energy ceiling.
pub mod nested;
/// Umbrella bridges between two catalog minima in descriptor space.
pub mod neus_bridge;
/// Cut-and-splice mixing of two quenched clusters.
pub mod splice;
#[cfg(feature = "graphkey")]
pub use archive_search::{Archive, ArchiveOutcome, archive_search};
pub mod additive_independence;
pub mod amsa;
pub mod bayesian_mixing;
pub mod bayesian_pilot;
pub mod bfwt;
pub mod dmc_population;
/// Target-free Feynman--Kac reconfiguration for cooperative search chains.
pub mod feynman_kac;
pub mod gle_langevin;
pub mod gpmd;
pub mod local_polish;
pub mod mcmc_sa;
pub mod parallel_tempering;
pub mod portfolio;
pub mod regime;
pub mod routing_probe;
pub mod sketchmap;
pub mod tpe;
pub mod tps_shoot;
/// Quasi-Newton relaxation whose curvature persists between calls.
pub mod warm_lbfgs;

pub use additive_independence::{AdditiveIndependenceResult, additive_independence_sa};
pub use amsa::{AmsaResult, amsa_optimize};
pub use bayesian_mixing::{BayesianMixingResult, BayesianMixingSampler};
pub use bayesian_pilot::{
    LaplacePosterior, PilotObservation, PilotPrior, Q_V_MAX, Q_V_MIN, TARGET_ACCEPT_RATE,
    fit_laplace, fit_laplace_skew_corrected, pilot_draws, pilot_draws_qmc,
};
pub use bfwt::{
    BfwtMode, BfwtResult, EULER_E, THETA_STAR as BFWT_THETA_STAR, bfwt_optimize,
    budget_feasible_temp, t_des, t_hi, t_lo, window_nonempty,
};
pub use dmc_population::{
    DEFAULT_BETA0, DEFAULT_STEPS_PER_CONTROL, DEFAULT_TARGET_WALKERS, DmcPopulationResult,
    Population, Walker, default_sigma, diffusion_displace, dmc_population_optimize,
    population_control, run_dmc_population, walker_weight,
};
pub use gle_langevin::{
    GleLangevinResult, GlePreconditioner, estimate_gle_omega0, estimate_gle_preconditioner,
    gle_langevin_adaptive_sa, gle_langevin_preconditioned_sa, gle_langevin_sa,
};
pub use gpmd::{
    ALPHA_TARGET, GpmdResult, POLISH_FRACTION, THETA_STAR, gap_proportional_temp, gpmd_optimize,
    run_gpmd,
};
pub use local_polish::{
    LocalPolishResult, QmcPolishResult, projected_gradient_polish, qmc_best1bin_scout,
    qmc_gsa_global_search, qmc_projected_gradient_polish, qmc_trust_region_poll,
    shifted_qmc_projected_gradient_polish,
};
pub use mcmc_sa::{GelmanRubin, MultiChainResult, MultiChainSampler, MultiChainState};
pub use parallel_tempering::{ParallelTemperingSampler, PtChainState, PtResult, geometric_ladder};
pub use portfolio::{
    ArmStat, PortfolioPolicy, PortfolioResult, portfolio_optimize, portfolio_optimize_with_policy,
};
pub use regime::{
    OptimizationRegime, ProblemFeatures, RegimeError, arm_prior_boost, arm_slice_multiplier,
    check_accept_path, exact_accept_allowed, order_arms, preferred_arm_tail, regime_exploit_prob,
    regime_exploit_width, require_accept_compatible, select_regime,
};
pub use sketchmap::{
    DEFAULT_A, DEFAULT_A_LOW, DEFAULT_B, DEFAULT_B_LOW, SketchMap2d, farthest_point_landmarks,
    pairwise_l2, row_l2, sigmoid_switch,
};
pub use tpe::{DEFAULT_ALPHA, DEFAULT_GAMMA, TpeCategorical, TpeContinuous1d};
pub use tps_shoot::{
    ShootDirection, accept_reactive_shoot, apply_shoot, best_frame_index, linear_path,
    path_is_reactive, path_reactive_geometric, path_reactive_objective, pick_shoot_direction,
    pick_shoot_index,
};
