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
pub mod portfolio;
pub mod regime;
pub mod sketchmap;
pub mod tpe;
pub mod tps_shoot;

pub use additive_independence::{additive_independence_sa, AdditiveIndependenceResult};
pub use bayesian_mixing::{BayesianMixingResult, BayesianMixingSampler};
pub use bayesian_pilot::{
    fit_laplace, pilot_draws, pilot_draws_qmc, LaplacePosterior, PilotObservation, PilotPrior,
    Q_V_MAX, Q_V_MIN, TARGET_ACCEPT_RATE,
};
pub use gle_langevin::{
    estimate_gle_omega0, estimate_gle_preconditioner, gle_langevin_adaptive_sa,
    gle_langevin_preconditioned_sa, gle_langevin_sa, GleLangevinResult, GlePreconditioner,
};
pub use local_polish::{
    projected_gradient_polish, qmc_best1bin_scout, qmc_gsa_global_search,
    qmc_projected_gradient_polish, qmc_trust_region_poll, shifted_qmc_projected_gradient_polish,
    LocalPolishResult, QmcPolishResult,
};
pub use mcmc_sa::{GelmanRubin, MultiChainResult, MultiChainSampler, MultiChainState};
pub use parallel_tempering::{geometric_ladder, ParallelTemperingSampler, PtChainState, PtResult};
pub use portfolio::{
    portfolio_optimize, portfolio_optimize_with_policy, ArmStat, PortfolioPolicy, PortfolioResult,
};
pub use regime::{
    arm_prior_boost, arm_slice_multiplier, check_accept_path, exact_accept_allowed, order_arms,
    preferred_arm_tail, regime_exploit_prob, regime_exploit_width, require_accept_compatible,
    select_regime, OptimizationRegime, ProblemFeatures, RegimeError,
};
pub use sketchmap::{
    farthest_point_landmarks, pairwise_l2, row_l2, sigmoid_switch, SketchMap2d, DEFAULT_A,
    DEFAULT_A_LOW, DEFAULT_B, DEFAULT_B_LOW,
};
pub use tpe::{TpeCategorical, TpeContinuous1d, DEFAULT_ALPHA, DEFAULT_GAMMA};
pub use tps_shoot::{
    accept_reactive_shoot, apply_shoot, best_frame_index, linear_path, path_is_reactive,
    path_reactive_geometric, path_reactive_objective, pick_shoot_direction, pick_shoot_index,
    ShootDirection,
};
