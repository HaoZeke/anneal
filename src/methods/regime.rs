//! GJQ-style auto-selection of optimization regimes for the portfolio.
//!
//! Policy (documented, safe, not oracle-optimal). Maps measurable problem
//! features onto preferred portfolio arm order and Beta prior boosts. The
//! restart arm (`explore`) always stays first so the almost-sure convergence
//! floor remains valid. Unknown / incomplete features fall to `Default`.
//!
//! Pattern mirrors GaussJacobiQuad `select_method_auto`: several specialized
//! kernels/arms, feature-based routing, and explicit refusal where a path is
//! invalid (exact Metropolis under declared objective noise).

use eindir_core::{box_geometry, Bounds, BoxGeometry};

/// Measurable inputs for regime selection.
#[derive(Clone, Debug)]
pub struct ProblemFeatures {
    /// State dimension.
    pub dim: usize,
    /// Whether a native gradient is available.
    pub has_grad: bool,
    /// Known observation-noise scale on the objective, if any.
    pub noise_sigma: Option<f64>,
    /// Box aspect ratio (max/min side length).
    pub aspect_ratio: f64,
    /// Shared work-unit budget.
    pub budget: usize,
}

impl ProblemFeatures {
    /// Build features from box bounds and caller-supplied flags.
    pub fn from_bounds(
        bounds: &Bounds<f64>,
        has_grad: bool,
        noise_sigma: Option<f64>,
        budget: usize,
    ) -> Self {
        let geom = box_geometry(bounds);
        Self::from_geometry(&geom, has_grad, noise_sigma, budget)
    }

    /// Build from precomputed geometry.
    pub fn from_geometry(
        geom: &BoxGeometry,
        has_grad: bool,
        noise_sigma: Option<f64>,
        budget: usize,
    ) -> Self {
        Self {
            dim: geom.dim,
            has_grad,
            noise_sigma,
            aspect_ratio: geom.aspect_ratio,
            budget,
        }
    }
}

/// Named optimization regimes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OptimizationRegime {
    /// dim ≤ 5 with gradient: polish-heavy local refinement.
    LowDimSmooth,
    /// dim ≥ 20 with gradient: colored-noise / reduced-space first.
    HighDimIllConditioned,
    /// Caller declared objective noise: OSA accept required.
    StochasticNoise,
    /// No gradient; multimodal / elongated box: DE / GSA / surrogate.
    MultimodalNoGrad,
    /// Safe fallback when no specialized regime matches.
    Default,
}

/// Error when a requested numerical path is invalid for the regime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegimeError {
    /// Exact Metropolis acceptance under declared stochastic costs.
    ExactAcceptUnderNoise,
}

impl std::fmt::Display for RegimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegimeError::ExactAcceptUnderNoise => write!(
                f,
                "exact Metropolis accept is out-of-regime when noise_sigma is set; use OSA"
            ),
        }
    }
}

impl std::error::Error for RegimeError {}

/// Select regime from features. Safe / documented, not oracle-optimal.
pub fn select_regime(f: &ProblemFeatures) -> OptimizationRegime {
    if f.noise_sigma.is_some_and(|s| s.is_finite() && s > 0.0) {
        return OptimizationRegime::StochasticNoise;
    }
    if f.has_grad && f.dim > 0 && f.dim <= 5 {
        return OptimizationRegime::LowDimSmooth;
    }
    if f.has_grad && f.dim >= 20 {
        return OptimizationRegime::HighDimIllConditioned;
    }
    if !f.has_grad && (f.dim >= 5 || f.aspect_ratio >= 8.0) {
        return OptimizationRegime::MultimodalNoGrad;
    }
    OptimizationRegime::Default
}

/// Preferred arm name order after the mandatory restart arm `explore`.
///
/// Returned names match portfolio arm identifiers. Unknown names are ignored
/// by the portfolio when mapping to internal arm kinds.
pub fn preferred_arm_tail(regime: OptimizationRegime) -> &'static [&'static str] {
    match regime {
        OptimizationRegime::LowDimSmooth => &[
            "shift", "am_sa", "hop", "metad", "tps", "gle", "tr_poll", "surrogate", "de", "gsa",
            "variant", "pt", "hmc",
        ],
        OptimizationRegime::HighDimIllConditioned => &[
            "am_sa", "gle", "hop", "reduced", "shift", "metad", "tps", "hmc", "surrogate", "de",
            "gsa", "tr_poll", "variant", "pt",
        ],
        OptimizationRegime::StochasticNoise => &[
            "explore", // note: still de-duplicated when building full list
            "gsa", "de", "surrogate", "metad", "tps", "variant", "pt", "hop", "gle", "tr_poll",
            "hmc", "reduced",
        ],
        OptimizationRegime::MultimodalNoGrad => &[
            "de", "gsa", "am_sa", "variant", "metad", "tps", "surrogate", "tr_poll", "pt",
        ],
        OptimizationRegime::Default => &[
            "shift", "hop", "am_sa", "metad", "tps", "gle", "surrogate", "de", "tr_poll", "gsa",
            "variant", "pt", "hmc", "reduced",
        ],
    }
}

/// Beta prior `(alpha0, beta0)` boost for an arm under a regime.
///
/// Larger alpha relative to beta biases Thompson sampling toward the arm early
/// on; `(1,1)` is the uninformative default. Boosts are intentionally strong
/// so Auto and Legacy policies differ in finite-budget allocation (not just
/// list order).
pub fn arm_prior_boost(regime: OptimizationRegime, arm: &str) -> (f64, f64) {
    let preferred = preferred_arm_tail(regime);
    // Top-3 preferred get a moderate success bias (rank 0 strongest).
    if let Some(rank) = preferred.iter().position(|&a| a == arm) {
        if rank < 3 {
            // alpha: 5, 3.5, 2.5 for ranks 0..2
            let alpha = 5.0 - 1.25 * rank as f64;
            return (alpha.max(2.0), 1.0);
        }
    }
    if arm == "explore" {
        return (2.0, 1.0);
    }
    (1.0, 1.0)
}

/// Probability of picking a regime-preferred arm instead of pure Thompson
/// after warmup (Auto policy only). Safe default: not 1.0 so exploration remains.
pub fn regime_exploit_prob(regime: OptimizationRegime) -> f64 {
    match regime {
        OptimizationRegime::LowDimSmooth => 0.35,
        OptimizationRegime::HighDimIllConditioned => 0.40,
        OptimizationRegime::MultimodalNoGrad => 0.50,
        OptimizationRegime::StochasticNoise => 0.35,
        OptimizationRegime::Default => 0.20,
    }
}

/// How many leading preferred arms (after explore) participate in the
/// regime-exploit lottery.
pub fn regime_exploit_width(regime: OptimizationRegime) -> usize {
    match regime {
        OptimizationRegime::HighDimIllConditioned => 3,
        OptimizationRegime::LowDimSmooth => 3,
        _ => 2,
    }
}

/// Slice-size multiplier for preferred arms (1.0 = baseline).
pub fn arm_slice_multiplier(regime: OptimizationRegime, arm: &str) -> f64 {
    let preferred = preferred_arm_tail(regime);
    if let Some(rank) = preferred.iter().position(|&a| a == arm) {
        if rank < 3 {
            return 1.0 + 0.35 * (3 - rank) as f64; // 2.05, 1.70, 1.35
        }
    }
    1.0
}

/// Whether exact Metropolis (non-OSA) is allowed under this regime.
pub fn exact_accept_allowed(regime: OptimizationRegime) -> bool {
    !matches!(regime, OptimizationRegime::StochasticNoise)
}

/// Refuse exact Metropolis under noise (out-of-regime path).

/// Shipped gate for accept-path choice (GJQ-style regime refusal).
///
/// When `noise_sigma` is a positive finite scale, exact Metropolis is
/// *out of regime*: callers must use a noise-aware rule (OSA). This is the
/// public check custom drivers and the portfolio entry path share.
pub fn require_accept_compatible(
    noise_sigma: Option<f64>,
    use_noise_aware: bool,
) -> Result<(), RegimeError> {
    let noisy = noise_sigma.is_some_and(|s| s.is_finite() && s > 0.0);
    let regime = if noisy {
        OptimizationRegime::StochasticNoise
    } else {
        OptimizationRegime::Default
    };
    check_accept_path(regime, use_noise_aware || !noisy)
}

/// Check whether an accept path is legal under `regime`.
///
/// Exact Metropolis under [`OptimizationRegime::StochasticNoise`] is refused.
pub fn check_accept_path(
    regime: OptimizationRegime,
    noise_aware: bool,
) -> Result<(), RegimeError> {
    if matches!(regime, OptimizationRegime::StochasticNoise) && !noise_aware {
        return Err(RegimeError::ExactAcceptUnderNoise);
    }
    Ok(())
}

/// Build the ordered arm list: `explore` first, then regime preference,
/// then remaining library arms, truncated by horizon capacity.
pub fn order_arms(
    available: &[&str],
    regime: OptimizationRegime,
    k_active: usize,
) -> Vec<String> {
    let k_active = k_active.max(1);
    let mut out: Vec<String> = Vec::new();
    // Restart floor: explore first when available.
    if available.iter().any(|&a| a == "explore") {
        out.push("explore".into());
    }
    for &name in preferred_arm_tail(regime) {
        if name == "explore" {
            continue;
        }
        if available.iter().any(|&a| a == name) && !out.iter().any(|x| x == name) {
            out.push(name.into());
        }
    }
    for &name in available {
        if !out.iter().any(|x| x == name) {
            out.push(name.into());
        }
    }
    out.truncate(k_active.min(out.len()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feats(
        dim: usize,
        has_grad: bool,
        noise: Option<f64>,
        aspect: f64,
    ) -> ProblemFeatures {
        ProblemFeatures {
            dim,
            has_grad,
            noise_sigma: noise,
            aspect_ratio: aspect,
            budget: 8000,
        }
    }

    #[test]
    fn low_dim_smooth_with_grad() {
        assert_eq!(
            select_regime(&feats(3, true, None, 1.0)),
            OptimizationRegime::LowDimSmooth
        );
    }

    #[test]
    fn high_dim_grad() {
        assert_eq!(
            select_regime(&feats(50, true, None, 1.0)),
            OptimizationRegime::HighDimIllConditioned
        );
    }

    #[test]
    fn noise_dominates() {
        assert_eq!(
            select_regime(&feats(3, true, Some(0.1), 1.0)),
            OptimizationRegime::StochasticNoise
        );
    }

    #[test]
    fn multimodal_no_grad() {
        assert_eq!(
            select_regime(&feats(10, false, None, 1.0)),
            OptimizationRegime::MultimodalNoGrad
        );
    }

    #[test]
    fn explore_always_first() {
        let avail = ["explore", "gle", "de", "gsa", "surrogate"];
        let ordered = order_arms(&avail, OptimizationRegime::HighDimIllConditioned, 5);
        assert_eq!(ordered[0], "explore");
        // GLE should appear early under high-dim.
        assert!(ordered.iter().position(|a| a == "gle").unwrap() < 3);
    }

    #[test]
    fn out_of_regime_exact_accept_fails() {
        let r = OptimizationRegime::StochasticNoise;
        assert!(check_accept_path(r, false).is_err());
        assert!(check_accept_path(r, true).is_ok());
        assert!(check_accept_path(OptimizationRegime::Default, false).is_ok());
    }

    #[test]
    fn require_accept_refuses_exact_under_noise() {
        assert!(require_accept_compatible(Some(0.1), false).is_err());
        assert!(require_accept_compatible(Some(0.1), true).is_ok());
        assert!(require_accept_compatible(None, false).is_ok());
    }
}

