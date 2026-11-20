// Copyright (c) anneal developers

//! The result dict of `cluster_search`: every field of
//! [`crate::methods::cluster_hopping::Outcome`].
//!
//! `pytest/test_cluster_search_outcome.py` reads the struct and fails when a
//! field has no key here. Two names differ from the struct: `Outcome::best`
//! is the energy and keeps the key `best_energy`, and `Outcome::best_state`
//! is the geometry and keeps the key `best`.

use ndarray::Array1;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::methods::cluster_hopping::{AcceptedTransition, ArmCount, HopStop, Outcome};

fn array<'py>(
    py: Python<'py>,
    state: &Array1<f64>,
    layout: &dyn Fn(Vec<f64>) -> Vec<f64>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, layout(state.to_vec()))
}

fn optional_array<'py>(
    py: Python<'py>,
    state: Option<&Array1<f64>>,
    layout: &dyn Fn(Vec<f64>) -> Vec<f64>,
) -> Option<Bound<'py, PyArray1<f64>>> {
    state.map(|state| array(py, state, layout))
}

fn transition<'py>(
    py: Python<'py>,
    record: &AcceptedTransition,
    layout: &dyn Fn(Vec<f64>) -> Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("hop", record.hop)?;
    dict.set_item("action", &record.action)?;
    dict.set_item("from_energy", record.from_energy)?;
    dict.set_item("to_energy", record.to_energy)?;
    dict.set_item("from_state", array(py, &record.from_state, layout))?;
    dict.set_item(
        "from_gradient",
        optional_array(py, record.from_gradient.as_ref(), layout),
    )?;
    dict.set_item("to_state", array(py, &record.to_state, layout))?;
    dict.set_item(
        "to_gradient",
        optional_array(py, record.to_gradient.as_ref(), layout),
    )?;
    dict.set_item("validated", record.validated)?;
    dict.set_item("adopted", record.adopted)?;
    Ok(dict)
}

fn arm_count<'py>(py: Python<'py>, count: &ArmCount) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("move_type", &count.move_type)?;
    dict.set_item("proposed", count.proposed)?;
    dict.set_item("accepted", count.accepted)?;
    dict.set_item("improved", count.improved)?;
    Ok(dict)
}

fn superbasin<'py>(
    py: Python<'py>,
    report: &crate::superbasin::SuperbasinReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("nodes", report.nodes)?;
    dict.set_item("edges", report.edges)?;
    dict.set_item("distortion", report.distortion)?;
    dict.set_item("levels", report.levels.clone())?;
    dict.set_item("top", report.top.clone())?;
    dict.set_item("jumps", report.jumps)?;
    dict.set_item("refusals", report.refusals)?;
    dict.set_item("refusals_by_kind", report.refusals_by_kind.to_vec())?;
    dict.set_item("mixed_ratio_max", report.mixed_ratio_max)?;
    dict.set_item("condition_max", report.condition_max)?;
    dict.set_item("condition_mean", report.condition_mean)?;
    dict.set_item("condition_residual_max", report.condition_residual_max)?;
    dict.set_item("solve_residual_max", report.solve_residual_max)?;
    dict.set_item("exact_solves", report.exact_solves)?;
    dict.set_item("hops_saved", report.hops_saved)?;
    dict.set_item("improvements", report.improvements)?;
    dict.set_item("archived", report.archived)?;
    let separability = match &report.separability {
        Some(found) => {
            let item = PyDict::new(py);
            item.set_item("f", found.f)?;
            item.set_item("groups", found.groups)?;
            item.set_item("points", found.points)?;
            item.set_item("per_dimension", found.per_dimension.clone())?;
            Some(item)
        }
        None => None,
    };
    dict.set_item("separability", separability)?;
    let quotient = match &report.quotient {
        Some(found) => {
            let item = PyDict::new(py);
            item.set_item("basins_raw", found.basins_raw)?;
            item.set_item("basins_quotiented", found.basins_quotiented)?;
            item.set_item("orbits_nontrivial", found.orbits_nontrivial)?;
            item.set_item("largest_orbit", found.largest_orbit)?;
            item.set_item("archived", found.archived)?;
            item.set_item("energy_buckets", found.energy_buckets)?;
            item.set_item("comparisons", found.comparisons)?;
            item.set_item("matched_max", found.matched_max)?;
            item.set_item("rejected_min", found.rejected_min)?;
            item.set_item("revisits_raw", found.revisits_raw)?;
            item.set_item("revisits_quotiented", found.revisits_quotiented)?;
            item.set_item("sources", found.sources)?;
            item.set_item("depth", found.depth)?;
            item.set_item("lumped_fraction", found.lumped_fraction)?;
            Some(item)
        }
        None => None,
    };
    dict.set_item("quotient", quotient)?;
    Ok(dict)
}

fn hop_diagnostics<'py>(
    py: Python<'py>,
    found: &crate::hmc::hop::HopDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("proposals", found.proposals)?;
    dict.set_item("divergences", found.divergences)?;
    dict.set_item("depth_capped", found.depth_capped)?;
    dict.set_item("depth_sum", found.depth_sum)?;
    dict.set_item("depth_histogram", found.depth_histogram.clone())?;
    dict.set_item("accept_sum", found.accept_sum)?;
    dict.set_item("leaves", found.leaves)?;
    dict.set_item("divergent_separation_sum", found.divergent_separation_sum)?;
    dict.set_item("metric_drift_sum", found.metric_drift_sum)?;
    dict.set_item("metric_drift_count", found.metric_drift_count)?;
    dict.set_item("epsilon_final", found.epsilon_final)?;
    dict.set_item("trajectory_temperature", found.trajectory_temperature)?;
    dict.set_item("metric_condition", found.metric_condition)?;
    dict.set_item("metric_span_sum", found.metric_span_sum)?;
    dict.set_item("reach_sum", found.reach_sum)?;
    dict.set_item("reach_target", found.reach_target)?;
    Ok(dict)
}

/// Why the search ended, as the stable lower-case code and the reason a
/// checkpoint gave when it retired the search.
fn stop(stop: Option<&HopStop>) -> (Option<&'static str>, Option<String>) {
    match stop {
        None => (None, None),
        Some(HopStop::Retired { reason }) => (Some("retired"), Some(reason.clone())),
        Some(other) => (Some(other.code()), None),
    }
}

/// Put every field of `out` in `dict`. `layout` maps a state from the chart
/// the search ran in to the layout the caller reads.
pub(super) fn fill(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    out: &Outcome,
    layout: &dyn Fn(Vec<f64>) -> Vec<f64>,
) -> PyResult<()> {
    macro_rules! put {
        ($($field:ident),* $(,)?) => {
            $(dict.set_item(stringify!($field), out.$field.clone())?;)*
        };
    }
    dict.set_item("best_energy", out.best)?;
    dict.set_item(
        "final_state",
        optional_array(py, out.final_state.as_ref(), layout),
    )?;
    put!(
        final_energy,
        hops,
        screened_out,
        basins,
        charged,
        overlap_rejected,
        returned,
        escape_scale,
        escape_threshold,
        visit_counts,
        soft_perturbs,
        soft_subspaces,
        soft_escapes,
        soft_crossed,
        improvements,
        quenched,
        energy_trace,
        merge_radius,
        mean_step,
        angular,
        contextual,
        screen,
        tabu,
        funnel,
        symmetrised,
        orbits,
        orbit_duplicates,
        surface_hops,
        continuous_symmetry,
        restarts,
        exchanges_refused,
        jumps,
        history_visits,
        shared_deposits,
        gossip_rounds,
        md_escape,
        nve_unavailable,
        history_seconds,
        stall_escapes,
        trail_escapes,
        stall_escape_gain,
        soft_lambda,
        rungs,
        swaps_tried,
        transport,
        accepted,
        unconverged_records,
        arms,
        delayed,
        swaps_accepted,
        paths,
        path_escapes,
        path_improvements,
        path_gain,
    );
    let transitions = out
        .accepted_transitions
        .iter()
        .map(|record| transition(py, record, layout))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("accepted_transitions", transitions)?;
    let counts = out
        .proposal_counts
        .iter()
        .map(|count| arm_count(py, count))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("proposal_counts", counts)?;
    let report = match &out.superbasin {
        Some(report) => Some(superbasin(py, report)?),
        None => None,
    };
    dict.set_item("superbasin", report)?;
    let observed = match &out.superbasin_counts {
        Some(counts) => {
            let item = PyDict::new(py);
            item.set_item("observations", counts.observations)?;
            Some(item)
        }
        None => None,
    };
    dict.set_item("superbasin_counts", observed)?;
    let archive = out.superbasin_archive.as_ref().map(|rows| {
        rows.iter()
            .map(|(basin, energy, state)| (*basin, *energy, array(py, state, layout)))
            .collect::<Vec<_>>()
    });
    dict.set_item("superbasin_archive", archive)?;
    let diagnostics = out
        .hmc
        .iter()
        .map(|found| hop_diagnostics(py, found))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("hmc", diagnostics)?;
    let bias = match &out.energy_bias {
        Some(found) => {
            let item = PyDict::new(py);
            item.set_item("w0", found.w0)?;
            item.set_item("gamma", found.gamma)?;
            item.set_item("sigma_bins", found.sigma_bins)?;
            item.set_item("deposits", found.deposits)?;
            Some(item)
        }
        None => None,
    };
    dict.set_item("energy_bias", bias)?;
    let (code, reason) = stop(out.stop.as_ref());
    dict.set_item("stop", code)?;
    dict.set_item("stop_reason", reason)?;
    Ok(())
}
