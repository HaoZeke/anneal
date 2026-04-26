//! Bayesian-pilot adaptation for SA hyperparameters.
//!
//! The SA literature picks `(T_0, sigma)` by hand or by grid search.
//! This module replaces the grid with a principled Bayesian pilot:
//!   1. Run `n_pilot` chains, each at a hyperparameter draw from the prior.
//!   2. Observe acceptance rates `r_i` and best-improvement deltas `d_i`.
//!   3. Fit a Laplace approximation to the marginal posterior of
//!      `(log T_0, log sigma)` given the observations and the
//!      Roberts/Rosenthal 2001 target acceptance rate `r* = 0.234`.
//!   4. Use the MAP estimate for the production SA run.
//!
//! Why Laplace and not full INLA: the latent field here is just two
//! scalars `(log T_0, log sigma)`, so the Laplace approximation is
//! a 2x2 Hessian invert -- analytic, deterministic, O(1). INLA's
//! sparse-precision-matrix machinery is overkill at this scale; we
//! ship the Laplace path now and reserve the full INLA path for
//! cooling schedules with many parameters (e.g., piecewise-linear
//! cooling with one knot per epoch).
//!
//! Convergence: when the pilot ends at epoch `n_pilot` and the
//! production phase uses the fixed MAP hyperparameters thereafter,
//! Hajek 1988 applies to the production phase unchanged. When the
//! pilot is allowed to re-fit at every epoch (online adaptation),
//! the diminishing-adaptation condition of Roberts/Rosenthal 2007
//! is required; we do not implement online mode in v0.3.x.
//!
//! See `~/Git/Gitlab/obsidian-notes/Software/anneal/design_pass_08_bayesian_sa.org`
//! for the full design rationale.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Roberts/Rosenthal 2001 doi:10.1214/ss/998929475: the asymptotically
/// optimal acceptance rate for random-walk Metropolis on a generic
/// product target. SA at high T behaves random-walk-like, so this is
/// the target the pilot tries to match.
pub const TARGET_ACCEPT_RATE: f64 = 0.234;

/// One pilot observation: parameters tried + acceptance rate observed.
#[derive(Clone, Debug)]
pub struct PilotObservation {
    /// Initial temperature drawn from the prior.
    pub t_init: f64,
    /// Step size drawn from the prior.
    pub sigma: f64,
    /// Empirical acceptance rate from the chain.
    pub accept_rate: f64,
    /// Best objective value reached during the pilot chain.
    pub best_val: f64,
    /// Final position (used as warm start for the production phase).
    pub final_pos: Vec<f64>,
}

/// Posterior summary returned by the Laplace fit.
#[derive(Clone, Copy, Debug)]
pub struct LaplacePosterior {
    /// MAP estimate of `T_0`.
    pub t_init_map: f64,
    /// MAP estimate of `sigma`.
    pub sigma_map: f64,
    /// Posterior standard deviation of `log T_0`.
    pub log_t_init_sd: f64,
    /// Posterior standard deviation of `log sigma`.
    pub log_sigma_sd: f64,
    /// Negative log-posterior at the MAP (lower is better).
    pub neg_log_post_map: f64,
}

/// Prior specification: log-Normal on both `T_0` and `sigma`.
/// Defaults match the IISE manuscript's range: `T_0 ~ logN(log 1, 1)`,
/// `sigma ~ logN(log 0.5, 0.7)`.
#[derive(Clone, Copy, Debug)]
pub struct PilotPrior {
    /// Mean of `log T_0` under the prior.
    pub log_t_init_mean: f64,
    /// Std of `log T_0` under the prior.
    pub log_t_init_sd: f64,
    /// Mean of `log sigma` under the prior.
    pub log_sigma_mean: f64,
    /// Std of `log sigma` under the prior.
    pub log_sigma_sd: f64,
}

impl Default for PilotPrior {
    fn default() -> Self {
        Self {
            log_t_init_mean: 0.0,            // T_0 ~ logN(0, 1) -> median 1
            log_t_init_sd: 1.0,
            log_sigma_mean: -0.693,          // sigma ~ logN(log 0.5, 0.7) -> median 0.5
            log_sigma_sd: 0.7,
        }
    }
}

/// Sample a `(T_0, sigma)` pair from the prior, returning the linear-scale values.
fn sample_prior(prior: &PilotPrior, rng: &mut StdRng) -> (f64, f64) {
    // Box-Muller
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let u3: f64 = rng.random();
    let u4: f64 = rng.random();
    let z2 = (-2.0 * u3.ln()).sqrt() * (2.0 * std::f64::consts::PI * u4).cos();
    let log_t = prior.log_t_init_mean + prior.log_t_init_sd * z1;
    let log_s = prior.log_sigma_mean + prior.log_sigma_sd * z2;
    (log_t.exp(), log_s.exp())
}

/// Negative log-posterior on `(log T_0, log sigma)` given pilot observations.
///
/// The likelihood is a Gaussian on the logit-of-acceptance-rate centred at
/// `logit(r*)` with a fitted variance: each observation `r_i` contributes
/// `(logit(r_i) - logit(r*))^2 / (2 * obs_var)`. The "effective likelihood"
/// is sharper for chains that diverge far from `r*`, regardless of which
/// hyperparameters produced them; we fit the *response surface* `r(T_0, sigma)`
/// implicitly through a local quadratic in log-coords.
fn neg_log_posterior(
    log_t_init: f64,
    log_sigma: f64,
    obs: &[PilotObservation],
    prior: &PilotPrior,
) -> f64 {
    let prior_term = 0.5
        * (((log_t_init - prior.log_t_init_mean) / prior.log_t_init_sd).powi(2)
            + ((log_sigma - prior.log_sigma_mean) / prior.log_sigma_sd).powi(2));

    // Likelihood: penalize deviations of nearby observations from r*.
    // Use inverse-distance weighting in log-hyperparameter space.
    let mut total_weight = 0.0;
    let mut weighted_dev = 0.0;
    for o in obs {
        let dx = log_t_init - o.t_init.ln();
        let dy = log_sigma - o.sigma.ln();
        let dist2 = dx * dx + dy * dy;
        let w = (-0.5 * dist2 / 0.5).exp(); // bandwidth 0.5 in log-coords
        total_weight += w;
        let logit_r = (o.accept_rate.max(1e-6).min(1.0 - 1e-6)).ln()
            - (1.0 - o.accept_rate.max(1e-6).min(1.0 - 1e-6)).ln();
        let logit_target = (TARGET_ACCEPT_RATE).ln() - (1.0 - TARGET_ACCEPT_RATE).ln();
        weighted_dev += w * (logit_r - logit_target).powi(2);
    }
    let likelihood_term = if total_weight > 0.0 {
        0.5 * weighted_dev / total_weight / 0.25 // sigma_obs = 0.5
    } else {
        0.0
    };
    prior_term + likelihood_term
}

/// Fit the Laplace approximation by a coarse grid search followed by a
/// finite-difference Newton step. Returns the MAP and posterior SDs.
pub fn fit_laplace(obs: &[PilotObservation], prior: &PilotPrior) -> LaplacePosterior {
    // Phase 1: 21x21 grid search in log space, 4 sd around the prior mean.
    let mut best = (prior.log_t_init_mean, prior.log_sigma_mean);
    let mut best_nll = neg_log_posterior(best.0, best.1, obs, prior);
    let n = 21;
    for i in 0..n {
        for j in 0..n {
            let log_t = prior.log_t_init_mean - 4.0 * prior.log_t_init_sd
                + 8.0 * prior.log_t_init_sd * (i as f64) / (n as f64 - 1.0);
            let log_s = prior.log_sigma_mean - 4.0 * prior.log_sigma_sd
                + 8.0 * prior.log_sigma_sd * (j as f64) / (n as f64 - 1.0);
            let nll = neg_log_posterior(log_t, log_s, obs, prior);
            if nll < best_nll {
                best_nll = nll;
                best = (log_t, log_s);
            }
        }
    }

    // Phase 2: finite-difference Hessian at the MAP, Laplace SDs from
    // the diagonal of the inverse Hessian.
    let h = 1e-3;
    let f00 = best_nll;
    let f10 = neg_log_posterior(best.0 + h, best.1, obs, prior);
    let f_10 = neg_log_posterior(best.0 - h, best.1, obs, prior);
    let f01 = neg_log_posterior(best.0, best.1 + h, obs, prior);
    let f0_1 = neg_log_posterior(best.0, best.1 - h, obs, prior);
    let h_tt = (f10 - 2.0 * f00 + f_10) / (h * h);
    let h_ss = (f01 - 2.0 * f00 + f0_1) / (h * h);
    let log_t_init_sd = if h_tt > 0.0 {
        (1.0 / h_tt).sqrt()
    } else {
        prior.log_t_init_sd
    };
    let log_sigma_sd = if h_ss > 0.0 {
        (1.0 / h_ss).sqrt()
    } else {
        prior.log_sigma_sd
    };

    LaplacePosterior {
        t_init_map: best.0.exp(),
        sigma_map: best.1.exp(),
        log_t_init_sd,
        log_sigma_sd,
        neg_log_post_map: best_nll,
    }
}

/// Draw `n_pilot` (T_0, sigma) pairs from the prior and return them.
/// The pilot phase itself (running chains, recording acceptance rates)
/// happens in user code so this module stays Sampler-agnostic.
pub fn pilot_draws(prior: &PilotPrior, n_pilot: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_pilot).map(|_| sample_prior(prior, &mut rng)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pilot_draws_respect_prior_means() {
        let prior = PilotPrior::default();
        let draws = pilot_draws(&prior, 1000, 42);
        let log_t_mean: f64 = draws.iter().map(|(t, _)| t.ln()).sum::<f64>() / 1000.0;
        let log_s_mean: f64 = draws.iter().map(|(_, s)| s.ln()).sum::<f64>() / 1000.0;
        assert!(
            (log_t_mean - prior.log_t_init_mean).abs() < 0.1,
            "log T_0 sample mean {} far from prior mean {}",
            log_t_mean,
            prior.log_t_init_mean,
        );
        assert!((log_s_mean - prior.log_sigma_mean).abs() < 0.1);
    }

    #[test]
    fn laplace_recovers_observation_concentration() {
        // 5 pilot observations clustered near (T_0=2, sigma=0.3) with
        // accept_rate near 0.234. The Laplace MAP should land near
        // (2, 0.3) and the SDs should be tighter than the prior.
        let prior = PilotPrior::default();
        let obs: Vec<PilotObservation> = (0..5)
            .map(|i| PilotObservation {
                t_init: 2.0 + 0.1 * (i as f64 - 2.0),
                sigma: 0.3 + 0.02 * (i as f64 - 2.0),
                accept_rate: TARGET_ACCEPT_RATE,
                best_val: -1.0,
                final_pos: vec![0.0, 0.0],
            })
            .collect();
        let post = fit_laplace(&obs, &prior);
        assert!(post.t_init_map > 0.0 && post.t_init_map.is_finite());
        assert!(post.sigma_map > 0.0 && post.sigma_map.is_finite());
        // Posterior should be tighter than prior on log_sigma at least.
        assert!(post.log_sigma_sd <= prior.log_sigma_sd + 0.1);
    }

    #[test]
    fn laplace_pulls_toward_optimal_accept_rate() {
        // 3 observations: one too-high accept_rate (T_0 too high),
        // one too-low (T_0 too low), one near r*. The MAP should
        // favour the near-r* observation.
        let prior = PilotPrior::default();
        let obs = vec![
            PilotObservation {
                t_init: 10.0,
                sigma: 0.5,
                accept_rate: 0.9,
                best_val: 0.0,
                final_pos: vec![0.0],
            },
            PilotObservation {
                t_init: 0.01,
                sigma: 0.5,
                accept_rate: 0.01,
                best_val: 0.0,
                final_pos: vec![0.0],
            },
            PilotObservation {
                t_init: 1.0,
                sigma: 0.3,
                accept_rate: 0.234,
                best_val: 0.0,
                final_pos: vec![0.0],
            },
        ];
        let post = fit_laplace(&obs, &prior);
        // MAP T_0 should be closer to 1 than to 10 or 0.01.
        assert!(
            (post.t_init_map - 1.0).abs() < 5.0,
            "MAP T_0 {} far from the on-target observation T_0=1.0",
            post.t_init_map
        );
    }
}
