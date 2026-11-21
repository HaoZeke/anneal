// Copyright (c) anneal developers

//! The settings of a box ensemble that the Python drivers take beyond
//! `replicas`, `history` and `membership`.
//!
//! Each one is a field of [`BoxEnsembleConfig`], [`GleEscapeConfig`] or
//! [`BoxCoverageConfig`]. Rust asserts on a bad value, and an assert reaches
//! Python as `PanicException`, so [`BoxTuning::resolve`] checks each range
//! first and raises `InvalidArgumentError`.

use pyo3::prelude::*;

use super::errors::invalid_argument;
use crate::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEscape, GleEscapeConfig,
};
use crate::methods::ensemble::HistoryMode;
use crate::methods::gle_langevin::GleNoise;
use crate::methods::minima_hopping::SharedVisitPolicy;

#[derive(Default)]
pub(super) struct BoxTuning<'a> {
    pub identity_tol: Option<f64>,
    pub shared_deposits: Option<usize>,
    pub shared_visit_policy: Option<&'a str>,
    pub escape: Option<&'a str>,
    pub langevin_steps: Option<usize>,
    pub langevin_omega0: Option<f64>,
    pub langevin_dt: Option<f64>,
    pub langevin_friction: Option<f64>,
    pub coverage_shared: Option<bool>,
    pub coverage_radius: Option<f64>,
    pub coverage_neighbors: Option<usize>,
    pub coverage_height: Option<f64>,
    pub coverage_well_tempering: Option<f64>,
    pub coverage_peer_weight: Option<f64>,
}

/// A finite number in a range, or `InvalidArgumentError` naming it.
fn ranged(name: &str, value: Option<f64>, inside: fn(f64) -> bool, rule: &str) -> PyResult<()> {
    match value {
        Some(number) if !(number.is_finite() && inside(number)) => Err(invalid_argument(format!(
            "{name} must be finite and {rule}, got {number}"
        ))),
        _ => Ok(()),
    }
}

/// The ensemble and coverage configuration a call asked for.
pub(super) struct Resolved {
    pub config: BoxEnsembleConfig,
    /// Coverage settings, when any keyword changed the ensemble or its
    /// coverage. The convenience policies, such as the values-only portfolio
    /// for one replica, apply only when none did.
    pub coverage: Option<BoxCoverageConfig>,
}

impl BoxTuning<'_> {
    fn tunes_config(&self) -> bool {
        self.identity_tol.is_some()
            || self.shared_deposits.is_some()
            || self.shared_visit_policy.is_some()
            || self.escape.is_some()
            || self.langevin_steps.is_some()
            || self.langevin_omega0.is_some()
            || self.langevin_dt.is_some()
            || self.langevin_friction.is_some()
    }

    fn tunes_coverage(&self) -> bool {
        self.coverage_shared.is_some()
            || self.coverage_radius.is_some()
            || self.coverage_neighbors.is_some()
            || self.coverage_height.is_some()
            || self.coverage_well_tempering.is_some()
            || self.coverage_peer_weight.is_some()
    }

    pub(super) fn resolve(
        &self,
        base: BoxEnsembleConfig,
        has_gradient: bool,
    ) -> PyResult<Resolved> {
        ranged("identity_tol", self.identity_tol, |v| v > 0.0, "positive")?;
        ranged(
            "langevin_omega0",
            self.langevin_omega0,
            |v| v > 0.0,
            "positive",
        )?;
        ranged("langevin_dt", self.langevin_dt, |v| v > 0.0, "positive")?;
        ranged(
            "langevin_friction",
            self.langevin_friction,
            |v| v > 0.0,
            "positive",
        )?;
        ranged(
            "coverage radius",
            self.coverage_radius,
            |v| v > 0.0,
            "positive",
        )?;
        ranged(
            "coverage_height",
            self.coverage_height,
            |v| v >= 0.0,
            "nonnegative",
        )?;
        ranged(
            "coverage_well_tempering",
            self.coverage_well_tempering,
            |v| v > 1.0,
            "greater than one",
        )?;
        ranged(
            "coverage_peer_weight",
            self.coverage_peer_weight,
            |v| v >= 0.0,
            "nonnegative",
        )?;
        if self.langevin_steps == Some(0) {
            return Err(invalid_argument("langevin_steps must be positive"));
        }
        let policy = match self.shared_visit_policy {
            None | Some("tabu") => SharedVisitPolicy::Tabu,
            Some("recognition") => SharedVisitPolicy::Recognition,
            Some(other) => {
                return Err(invalid_argument(format!(
                    "shared_visit_policy must be tabu or recognition, got {other:?}"
                )));
            }
        };
        let langevin_given = self.langevin_steps.is_some()
            || self.langevin_omega0.is_some()
            || self.langevin_dt.is_some()
            || self.langevin_friction.is_some();
        let escape = match self.escape {
            None | Some("gaussian") => {
                if langevin_given {
                    return Err(invalid_argument(
                        "langevin_* settings need escape=\"langevin\"",
                    ));
                }
                BoxEscape::Gaussian
            }
            Some("langevin") => {
                if !has_gradient {
                    return Err(invalid_argument(
                        "escape=\"langevin\" needs grad_fn: the segments integrate forces",
                    ));
                }
                let defaults = GleEscapeConfig::default();
                BoxEscape::Langevin(GleEscapeConfig {
                    steps: self.langevin_steps.unwrap_or(defaults.steps),
                    omega0: self.langevin_omega0.unwrap_or(defaults.omega0),
                    dt: self.langevin_dt.unwrap_or(defaults.dt),
                    noise: self
                        .langevin_friction
                        .map_or(GleNoise::Colored, |friction| GleNoise::White { friction }),
                })
            }
            Some(other) => {
                return Err(invalid_argument(format!(
                    "escape must be gaussian or langevin, got {other:?}"
                )));
            }
        };
        let config = BoxEnsembleConfig {
            identity_tol: self.identity_tol.unwrap_or(base.identity_tol),
            shared_deposits: self.shared_deposits.unwrap_or(base.shared_deposits),
            shared_visit_policy: policy,
            escape,
            ..base
        };
        let coverage = (self.tunes_coverage() || self.tunes_config()).then(|| {
            let defaults = BoxCoverageConfig::default();
            BoxCoverageConfig {
                shared: self.coverage_shared.unwrap_or(
                    matches!(config.history, HistoryMode::Shared) && config.shared_deposits > 0,
                ),
                radius: self.coverage_radius.unwrap_or(defaults.radius),
                height: self.coverage_height.unwrap_or(defaults.height),
                well_tempering: self
                    .coverage_well_tempering
                    .unwrap_or(defaults.well_tempering),
                peer_weight: self.coverage_peer_weight.unwrap_or(defaults.peer_weight),
                neighbors: self.coverage_neighbors.unwrap_or(defaults.neighbors),
            }
        });
        Ok(Resolved { config, coverage })
    }
}
