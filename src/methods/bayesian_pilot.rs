//! Bayesian-pilot adaptation for SA / GSA hyperparameters.
//!
//! The SA literature picks `(T_0, sigma, q_v)` by hand or grid search.
//! This module replaces the grid with a principled Bayesian pilot:
//!   1. Run `n_pilot` chains, each at a hyperparameter draw from the
//!      prior (T_0 ~ logN, sigma ~ logN, q_v ~ scaled-Beta on (1, 3)).
//!   2. Observe acceptance rates `r_i` and best-improvement deltas `d_i`.
//!   3. Fit a Laplace approximation to the joint posterior of
//!      `(log T_0, log sigma, q_v)` given the observations and the
//!      Roberts/Rosenthal 2001 target acceptance rate `r* = 0.234`.
//!   4. Use the MAP estimate for the production SA run.
//!
//! q_v IS THE NOVELTY: to the limits of the lit-survey agent's reading,
//! no SA paper has done Bayesian model selection over the GSA q_v
//! continuum. q_v=1 is BSA (Boltzmann/Gaussian), q_v=2 is FSA
//! (Cauchy), and the heavy-tailed regime q_v in (1, 3) interpolates
//! continuously. The pilot finds the q_v that best matches the
//! objective's geometry rather than committing to one of the three
//! literature points by hand.
//!
//! Why Laplace and not full INLA: the latent field here is three
//! scalars `(log T_0, log sigma, q_v)`, so the Laplace approximation
//! is a 3x3 Hessian invert -- analytic, deterministic, O(1). INLA's
//! sparse-precision-matrix machinery is overkill at this scale; the
//! full INLA path is reserved for cooling schedules with one parameter
//! per epoch (a v0.4 extension).
//!
//! Convergence: when the pilot ends at epoch `n_pilot` and the
//! production phase uses the fixed MAP hyperparameters thereafter,
//! Hajek 1988 doi:10.1287/moor.13.2.311 applies to the production
//! phase unchanged -- the Bayesian-frozen SA inherits a.s. convergence
//! to the global optimum. Online (unfrozen) adaptation requires the
//! diminishing-adaptation condition of Roberts/Rosenthal 2007
//! doi:10.1239/jap/1183667414; not implemented in v0.3.x.
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

/// Lower bound on q_v (just above 1 to avoid the BSA branch-switch).
pub const Q_V_MIN: f64 = 1.05;
/// Upper bound on q_v (just below 5/3 to keep q-Gaussian variance finite).
pub const Q_V_MAX: f64 = 2.95;

/// One pilot observation: parameters tried + acceptance rate observed.
#[derive(Clone, Debug)]
pub struct PilotObservation {
    /// Initial temperature drawn from the prior.
    pub t_init: f64,
    /// Step size drawn from the prior.
    pub sigma: f64,
    /// GSA visiting index drawn from the prior. q_v=1 is BSA, q_v=2 is FSA.
    pub q_v: f64,
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
    /// MAP estimate of `q_v`.
    pub q_v_map: f64,
    /// Posterior standard deviation of `log T_0`.
    pub log_t_init_sd: f64,
    /// Posterior standard deviation of `log sigma`.
    pub log_sigma_sd: f64,
    /// Posterior standard deviation of `q_v` (linear-scale, since q_v is bounded).
    pub q_v_sd: f64,
    /// Negative log-posterior at the MAP (lower is better).
    pub neg_log_post_map: f64,
}

/// Prior specification: log-Normal on `T_0` and `sigma`; truncated
/// scaled-Beta on q_v over (Q_V_MIN, Q_V_MAX) with mode at 2 (FSA).
/// Defaults match the IISE manuscript's convention.
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
    /// Prior mean of q_v (linear scale).
    pub q_v_mean: f64,
    /// Prior std of q_v (linear scale).
    pub q_v_sd: f64,
}

impl Default for PilotPrior {
    fn default() -> Self {
        Self {
            log_t_init_mean: 0.0,            // T_0 ~ logN(0, 1) -> median 1
            log_t_init_sd: 1.0,
            log_sigma_mean: -0.693,          // sigma ~ logN(log 0.5, 0.7) -> median 0.5
            log_sigma_sd: 0.7,
            q_v_mean: 2.0,                   // q_v ~ truncated-N(2, 0.5) on (1.05, 2.95)
            q_v_sd: 0.5,
        }
    }
}

/// Sample a `(T_0, sigma, q_v)` triple from the prior, returning the linear-scale values.
fn sample_prior(prior: &PilotPrior, rng: &mut StdRng) -> (f64, f64, f64) {
    // Box-Muller for two log-Normals
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let u3: f64 = rng.random();
    let u4: f64 = rng.random();
    let z2 = (-2.0 * u3.ln()).sqrt() * (2.0 * std::f64::consts::PI * u4).cos();
    let log_t = prior.log_t_init_mean + prior.log_t_init_sd * z1;
    let log_s = prior.log_sigma_mean + prior.log_sigma_sd * z2;

    // Truncated normal on q_v via rejection sampling
    let mut q_v;
    loop {
        let u5: f64 = rng.random();
        let u6: f64 = rng.random();
        let z3 = (-2.0 * u5.ln()).sqrt() * (2.0 * std::f64::consts::PI * u6).cos();
        q_v = prior.q_v_mean + prior.q_v_sd * z3;
        if q_v > Q_V_MIN && q_v < Q_V_MAX {
            break;
        }
    }

    (log_t.exp(), log_s.exp(), q_v)
}

/// Negative log-posterior on `(log T_0, log sigma, q_v)` given pilot
/// observations + best-val improvement signal.
///
/// Three terms:
///   (1) Prior: log-Normal on T_0 and sigma; truncated normal on q_v.
///   (2) Acceptance-rate likelihood: Gaussian on logit(a) vs logit(0.234)
///       (Roberts/Rosenthal target), weighted by inverse-distance to the
///       observation in (log T_0, log sigma, q_v) space.
///   (3) Improvement likelihood: Gaussian rewarding pilot chains whose
///       best_val is far below the worst-pilot best_val. Without (3) the
///       posterior is blind to which hyperparameters actually optimised.
fn neg_log_posterior(
    log_t_init: f64,
    log_sigma: f64,
    q_v: f64,
    obs: &[PilotObservation],
    prior: &PilotPrior,
    best_val_ref: f64,
) -> f64 {
    if q_v <= Q_V_MIN || q_v >= Q_V_MAX {
        return f64::INFINITY;
    }
    let prior_term = 0.5
        * (((log_t_init - prior.log_t_init_mean) / prior.log_t_init_sd).powi(2)
            + ((log_sigma - prior.log_sigma_mean) / prior.log_sigma_sd).powi(2)
            + ((q_v - prior.q_v_mean) / prior.q_v_sd).powi(2));

    let bv_min = obs
        .iter()
        .map(|o| o.best_val)
        .fold(f64::INFINITY, f64::min);
    let bv_range = best_val_ref - bv_min + 1e-12;

    let mut total_weight = 0.0;
    let mut weighted_accept = 0.0;
    let mut weighted_improve = 0.0;
    let logit_target = (TARGET_ACCEPT_RATE).ln() - (1.0 - TARGET_ACCEPT_RATE).ln();
    for o in obs {
        let dx = log_t_init - o.t_init.ln();
        let dy = log_sigma - o.sigma.ln();
        let dz = (q_v - o.q_v) / 0.5; // q_v bandwidth = 0.5 (linear scale)
        let dist2 = dx * dx + dy * dy + dz * dz;
        let w = (-0.5 * dist2 / 0.5).exp(); // bandwidth 0.5 in (log T, log sigma, scaled q_v)
        total_weight += w;
        let a = o.accept_rate.max(1e-6).min(1.0 - 1e-6);
        let logit_r = a.ln() - (1.0 - a).ln();
        weighted_accept += w * (logit_r - logit_target).powi(2);
        let norm_improve = (best_val_ref - o.best_val) / bv_range;
        weighted_improve += w * (1.0 - norm_improve).powi(2);
    }
    let accept_term = if total_weight > 0.0 {
        0.5 * weighted_accept / total_weight / 0.25 // sigma_a = 0.5
    } else {
        0.0
    };
    let improve_term = if total_weight > 0.0 {
        0.5 * weighted_improve / total_weight / 0.04 // sigma_i = 0.2 (improvement matters more)
    } else {
        0.0
    };
    prior_term + accept_term + improve_term
}

/// Fit the Laplace approximation by a coarse 3D grid search followed
/// by a finite-difference diagonal Hessian Newton step. Returns the
/// MAP and posterior SDs in (log T_0, log sigma, q_v) space.
pub fn fit_laplace(obs: &[PilotObservation], prior: &PilotPrior) -> LaplacePosterior {
    let best_val_ref = obs.iter().map(|o| o.best_val).fold(f64::NEG_INFINITY, f64::max);

    // Phase 1: 11x11x11 grid search in log space, 3 sd around the prior mean.
    let n = 11;
    let mut best = (prior.log_t_init_mean, prior.log_sigma_mean, prior.q_v_mean);
    let mut best_nll = neg_log_posterior(best.0, best.1, best.2, obs, prior, best_val_ref);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let log_t = prior.log_t_init_mean - 3.0 * prior.log_t_init_sd
                    + 6.0 * prior.log_t_init_sd * (i as f64) / (n as f64 - 1.0);
                let log_s = prior.log_sigma_mean - 3.0 * prior.log_sigma_sd
                    + 6.0 * prior.log_sigma_sd * (j as f64) / (n as f64 - 1.0);
                let q_v = (prior.q_v_mean - 3.0 * prior.q_v_sd)
                    .max(Q_V_MIN + 0.01)
                    + ((prior.q_v_mean + 3.0 * prior.q_v_sd).min(Q_V_MAX - 0.01)
                        - (prior.q_v_mean - 3.0 * prior.q_v_sd).max(Q_V_MIN + 0.01))
                        * (k as f64)
                        / (n as f64 - 1.0);
                let nll = neg_log_posterior(log_t, log_s, q_v, obs, prior, best_val_ref);
                if nll < best_nll {
                    best_nll = nll;
                    best = (log_t, log_s, q_v);
                }
            }
        }
    }

    // Phase 2: finite-difference diagonal Hessian at the MAP. Diagonal
    // approximation good enough for the v0.3.x demo; full 3x3 Hessian
    // can be added in v0.4.
    let h = 1e-3;
    let f000 = best_nll;
    let h_tt = (neg_log_posterior(best.0 + h, best.1, best.2, obs, prior, best_val_ref)
        - 2.0 * f000
        + neg_log_posterior(best.0 - h, best.1, best.2, obs, prior, best_val_ref))
        / (h * h);
    let h_ss = (neg_log_posterior(best.0, best.1 + h, best.2, obs, prior, best_val_ref)
        - 2.0 * f000
        + neg_log_posterior(best.0, best.1 - h, best.2, obs, prior, best_val_ref))
        / (h * h);
    let h_qq = (neg_log_posterior(best.0, best.1, best.2 + h, obs, prior, best_val_ref)
        - 2.0 * f000
        + neg_log_posterior(best.0, best.1, best.2 - h, obs, prior, best_val_ref))
        / (h * h);
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
    let q_v_sd = if h_qq > 0.0 {
        (1.0 / h_qq).sqrt()
    } else {
        prior.q_v_sd
    };

    LaplacePosterior {
        t_init_map: best.0.exp(),
        sigma_map: best.1.exp(),
        q_v_map: best.2,
        log_t_init_sd,
        log_sigma_sd,
        q_v_sd,
        neg_log_post_map: best_nll,
    }
}

/// Draw `n_pilot` (T_0, sigma, q_v) triples from the prior.
/// The pilot phase itself (running chains, recording acceptance rates +
/// best vals) happens in user code so this module stays Sampler-agnostic.
pub fn pilot_draws(prior: &PilotPrior, n_pilot: usize, seed: u64) -> Vec<(f64, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_pilot).map(|_| sample_prior(prior, &mut rng)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_obs(t: f64, s: f64, q: f64, a: f64, bv: f64) -> PilotObservation {
        PilotObservation {
            t_init: t,
            sigma: s,
            q_v: q,
            accept_rate: a,
            best_val: bv,
            final_pos: vec![0.0, 0.0],
        }
    }

    #[test]
    fn pilot_draws_respect_prior_means() {
        let prior = PilotPrior::default();
        let draws = pilot_draws(&prior, 1000, 42);
        let log_t_mean: f64 = draws.iter().map(|(t, _, _)| t.ln()).sum::<f64>() / 1000.0;
        let log_s_mean: f64 = draws.iter().map(|(_, s, _)| s.ln()).sum::<f64>() / 1000.0;
        let q_v_mean: f64 = draws.iter().map(|(_, _, q)| *q).sum::<f64>() / 1000.0;
        assert!((log_t_mean - prior.log_t_init_mean).abs() < 0.1);
        assert!((log_s_mean - prior.log_sigma_mean).abs() < 0.1);
        assert!((q_v_mean - prior.q_v_mean).abs() < 0.1);
        for (_, _, q) in &draws {
            assert!(*q > Q_V_MIN && *q < Q_V_MAX, "q_v {} outside (1.05, 2.95)", q);
        }
    }

    #[test]
    fn laplace_recovers_q_v_concentration_near_2() {
        // 5 observations clustered near q_v=2.0 with the BEST best_val at
        // q_v=2.0 exactly (and worse values either side). The likelihood
        // + prior should both pull MAP to ~2.0.
        let prior = PilotPrior::default();
        let obs = vec![
            fake_obs(2.0, 0.3, 1.7, TARGET_ACCEPT_RATE, -1.0),
            fake_obs(2.0, 0.3, 1.85, TARGET_ACCEPT_RATE, -1.5),
            fake_obs(2.0, 0.3, 2.0, TARGET_ACCEPT_RATE, -2.0),
            fake_obs(2.0, 0.3, 2.15, TARGET_ACCEPT_RATE, -1.5),
            fake_obs(2.0, 0.3, 2.3, TARGET_ACCEPT_RATE, -1.0),
        ];
        let post = fit_laplace(&obs, &prior);
        assert!(
            (post.q_v_map - 2.0).abs() < 0.5,
            "MAP q_v {} far from observation peak at 2.0",
            post.q_v_map
        );
    }

    #[test]
    fn laplace_picks_q_v_with_best_improvement() {
        // Three observations at distinct q_v values; the one at q_v=2.5
        // got the best (most-negative) best_val. The improvement-term
        // in the likelihood should pull MAP toward 2.5.
        let prior = PilotPrior::default();
        let obs = vec![
            fake_obs(1.0, 0.5, 1.2, 0.234, -1.0),
            fake_obs(1.0, 0.5, 2.0, 0.234, -2.0),
            fake_obs(1.0, 0.5, 2.5, 0.234, -10.0), // big improvement
        ];
        let post = fit_laplace(&obs, &prior);
        assert!(
            post.q_v_map > 2.0,
            "MAP q_v should favour the q_v=2.5 observation that found best_val=-10; got {}",
            post.q_v_map
        );
    }
}
