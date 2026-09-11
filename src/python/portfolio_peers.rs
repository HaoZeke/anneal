//! Thread-detached access to the same native portfolio replica controller.

use std::collections::BTreeMap;

use eindir_core::{Gradient, Objective};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::methods::portfolio::{PortfolioEnsembleConfig, portfolio_ensemble_optimize};

pub(super) fn run<O, G>(
    py: Python<'_>,
    obj: &O,
    grad: Option<&G>,
    seed: u64,
    config: &PortfolioEnsembleConfig,
) -> PyResult<Py<PyDict>>
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let result = super::with_replica_threads(py, || {
        portfolio_ensemble_optimize(obj, grad, seed, None, config)
    });
    let mut totals = BTreeMap::new();
    for replica in &result.replicas {
        for arm in &replica.arm_stats {
            let total = totals.entry(arm.name).or_insert((0, 0));
            total.0 += arm.pulls;
            total.1 += arm.successes;
        }
    }
    let out = super::portfolio_result_to_dict(py, crate::PortfolioResult {
        best_pos: result.best_pos,
        best_val: result.best_val,
        n_evals: result.n_evals,
        n_grads: result.n_grads,
        arm_stats: totals.into_iter().map(|(name, (pulls, successes))| {
            crate::ArmStat { name, pulls, successes }
        }).collect(),
        hop_state: None,
    })?;
    let replicas = result.replicas.into_iter()
        .map(|replica| super::portfolio_result_to_dict(py, replica))
        .collect::<PyResult<Vec<_>>>()?;
    let dict = out.bind(py);
    dict.set_item("replicas", replicas)?;
    dict.set_item("charged", result.n_evals + result.n_grads)?;
    dict.set_item("coverage_published_samples", result.coverage.published_samples)?;
    dict.set_item("coverage_applied_foreign_samples", result.coverage.applied_foreign_samples)?;
    dict.set_item("coverage_sample_peer_checks", result.coverage.sample_peer_checks)?;
    dict.set_item("coverage_sample_anchor_overlaps", result.coverage.sample_anchor_overlaps)?;
    dict.set_item("coverage_sample_anchor_only_overlaps", result.coverage.sample_anchor_only_overlaps)?;
    dict.set_item("coverage_sample_overlaps", result.coverage.sample_overlaps)?;
    dict.set_item("coverage_repelled_proposals", result.coverage.repelled_proposals)?;
    dict.set_item("coverage_constrained_repulsions", result.coverage.constrained_repulsions)?;
    Ok(out)
}
