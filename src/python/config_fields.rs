// Copyright (c) anneal developers

//! Read and write access to every field of
//! [`crate::methods::cluster_hopping::Config`].
//!
//! The field lists below are checked against the struct by
//! `pytest/test_config_fields.py`, which reads `config.rs`, so a field that
//! is added to the struct without an entry here fails that test.

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::errors::{feature_unavailable, invalid_argument};
use crate::methods::cluster_hopping::Config;

fn unknown_field(name: &str) -> PyErr {
    invalid_argument(format!(
        "Config has no field {name:?}; Config.field_names() lists them"
    ))
}

fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value;
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(flag) => flag.into_py_any(py),
        Value::Number(number) => {
            if let Some(whole) = number.as_i64() {
                whole.into_py_any(py)
            } else if let Some(whole) = number.as_u64() {
                whole.into_py_any(py)
            } else {
                number.as_f64().unwrap_or(f64::NAN).into_py_any(py)
            }
        }
        Value::String(text) => text.into_py_any(py),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_py_any(py)
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            dict.into_py_any(py)
        }
    }
}

fn json_field<T: serde::Serialize>(py: Python<'_>, name: &str, field: &T) -> PyResult<Py<PyAny>> {
    let value = serde_json::to_value(field)
        .map_err(|err| invalid_argument(format!("field {name:?} is not serialisable: {err}")))?;
    json_to_py(py, &value)
}

fn whole<'py, T: FromPyObject<'py>>(name: &str, value: &Bound<'py, PyAny>) -> PyResult<T> {
    value.extract::<T>().map_err(|err| {
        if err.is_instance_of::<pyo3::exceptions::PyOverflowError>(value.py()) {
            invalid_argument(format!("{name} must be a non-negative whole number"))
        } else {
            err
        }
    })
}

/// A finite number inside the range the search asserts on. Outside it the
/// Rust constructor that reads the setting panics, and a panic reaches
/// Python as `PanicException`.
fn finite(name: &str, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    let number: f64 = value.extract()?;
    if !number.is_finite() {
        return Err(invalid_argument(format!(
            "{name} must be finite, got {number}"
        )));
    }
    let (inside, rule) = match name {
        "theta" => (number > 0.0 && number < 2.0, "lie strictly between 0 and 2"),
        "calibrate_quantile" | "bayes_threshold" => ((0.0..1.0).contains(&number), "lie in [0, 1)"),
        "contextual_floor" | "bayes_exploration" => {
            ((0.0..=1.0).contains(&number), "lie in [0, 1]")
        }
        "diversity_floor" => (number > 0.0 && number <= 1.0, "lie in (0, 1]"),
        "height_revisits" | "length_scale" | "merge_radius" | "call_floor" => {
            (number > 0.0, "be positive")
        }
        "bias_gamma" => (number > 1.0, "exceed 1"),
        "bias_height" => (number >= 0.0, "be at least 0"),
        _ => (true, ""),
    };
    if inside {
        Ok(number)
    } else {
        Err(invalid_argument(format!(
            "{name} must {rule}, got {number}"
        )))
    }
}

fn flag(name: &str, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let on: bool = value.extract()?;
    if name == "shape_keyed" && on && !cfg!(feature = "ira") {
        return Err(feature_unavailable(
            "shape_keyed needs the ira feature, which this build lacks",
        ));
    }
    Ok(on)
}

macro_rules! config_fields {
    (
        bool: [$($b:ident),* $(,)?];
        f64: [$($f:ident),* $(,)?];
        usize: [$($u:ident),* $(,)?];
        u64: [$($w:ident),* $(,)?];
        opt_f64: [$($of:ident),* $(,)?];
        opt_usize: [$($ou:ident),* $(,)?];
        json: [$($j:ident),* $(,)?];
        opaque: [$($o:ident),* $(,)?];
    ) => {
        const FIELD_NAMES: &[&str] = &[
            $(stringify!($b),)* $(stringify!($f),)* $(stringify!($u),)* $(stringify!($w),)*
            $(stringify!($of),)* $(stringify!($ou),)* $(stringify!($j),)* $(stringify!($o),)*
        ];

        /// Every public field of `Config`, sorted by name.
        pub(super) fn field_names() -> Vec<&'static str> {
            let mut names = FIELD_NAMES.to_vec();
            names.sort_unstable();
            names
        }

        /// The value of one field as a Python object.
        pub(super) fn get(py: Python<'_>, cfg: &Config, name: &str) -> PyResult<Py<PyAny>> {
            match name {
                $(stringify!($b) => cfg.$b.into_py_any(py),)*
                $(stringify!($f) => cfg.$f.into_py_any(py),)*
                $(stringify!($u) => cfg.$u.into_py_any(py),)*
                $(stringify!($w) => cfg.$w.into_py_any(py),)*
                $(stringify!($of) => cfg.$of.into_py_any(py),)*
                $(stringify!($ou) => cfg.$ou.into_py_any(py),)*
                $(stringify!($j) => json_field(py, name, &cfg.$j),)*
                $(stringify!($o) => cfg.$o.is_some().into_py_any(py),)*
                _ => Err(unknown_field(name)),
            }
        }

        /// Set one scalar field. A field whose value is tied to the state
        /// length, or is a structure, is set by a constructor instead.
        pub(super) fn set(cfg: &mut Config, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
            match name {
                $(stringify!($b) => cfg.$b = flag(name, value)?,)*
                $(stringify!($f) => cfg.$f = finite(name, value)?,)*
                $(stringify!($u) => {
                    if name == "n_points" {
                        return Err(invalid_argument(
                            "n_points sets the state length; build the Config for the size you need",
                        ));
                    }
                    cfg.$u = whole(name, value)?;
                })*
                $(stringify!($w) => cfg.$w = whole(name, value)?,)*
                $(stringify!($of) => {
                    cfg.$of = if value.is_none() { None } else { Some(finite(name, value)?) };
                })*
                $(stringify!($ou) => {
                    cfg.$ou = if value.is_none() { None } else { Some(whole(name, value)?) };
                })*
                $(stringify!($j) => {
                    return Err(invalid_argument(format!(
                        "{name} is a structure; a preset or a with_* method sets it"
                    )));
                })*
                $(stringify!($o) => {
                    return Err(invalid_argument(format!(
                        "{name} is a structure; a preset or a with_* method sets it"
                    )));
                })*
                _ => return Err(unknown_field(name)),
            }
            Ok(())
        }
    };
}

config_fields! {
    bool: [
        budget_window, allocate_moves, adaptive_height, minima_hopping, md_escape,
        escape_on_stall, retry_noop_stall_leave, align_directed_stall_source,
        trail_on_stall, track_funnels, symmetrise_on_stall, point_symmetrise,
        point_symmetrise_on_new, orbit_complete_on_new, orbit_arm,
        surface_hop_on_best, soap_repel, exchange_metropolis, jump_on_stall,
        point_symmetrise_every_accept, angular_moves, contextual_moves,
        bayes_screen, flat_histogram, statistical_temperature, energy_bias,
        depth_reward, novel_reward, soft_perturb, cov_perturb, staged_quench,
        tabu_on_stall, restart_on_stall, calibrate_radius, bias_by_rung,
        return_screen, soap_class_residual, path_on_stall, anneal_diversity,
        prior_soap, general_collective, homotop_descent, homotop_ranked,
        homotop_window, homotop_reorder, shape_keyed, calibrated_screen,
        finish_improving, adaptive_screen, precondition_bonds, capped_lindh_quench,
        trace_quenched, long_ranged, plateau_heat, delayed_acceptance,
        probe_screen, quench_once, improvement_rewards, energy_lattice,
        displacement_only, superbasin_report, superbasin_escape, energy_trace,
        superbasin_quotient, superbasin_features, witness_tripleton_ceiling,
        ring_moves, hold_stereocentres, measured_energy_scale, branch_on_stall,
    ];
    f64: [
        length_scale, energy_scale, neighbour_cutoff, symmetrise_cutoff,
        temperature, bias_height, bias_gamma, merge_radius, keying_sigma,
        morphology_scale, theta, md_escape_dt, md_escape_kinetic, escape_epsilon,
        escape_amplitude, escape_overshoot, escape_stall_factor, jump_step,
        symmetrise_core_fraction, symmetry_tolerance, symmetry_merge_radius,
        angular_target, contextual_floor, flat_quantile, group_cutoff,
        covalent_cutoff, bond_tolerance, bayes_exploration, bayes_threshold,
        calibrate_quantile, ladder_top, ladder_target_accept, diversity_floor,
        height_revisits, screen_margin, record_gradient, answer_gradient,
        surrogate_tolerance, quench_confidence, incumbent_gap, container,
        min_separation,
    ];
    usize: [
        n_points, basin_stride, replicas, swap_period, md_escape_minima,
        md_escape_max_steps, md_escape_soften, escape_lanczos_steps,
        escape_max_climb, escape_stall_patience, funnel_period, shared_deposits,
        jump_patience, jump_steps, symmetrise_patience, flat_sweep, soft_modes,
        soft_steps, settle_iters, bayes_warmup, tabu_capacity, restart_patience,
        ladder_window, ladder_pilot, return_polish, return_polish_after,
        stall_patience, path_images, screen_steps, polish_records, construct_width,
        quench_warmup, relax_steps, reoccupy_interval, superbasin_period,
    ];
    u64: [
        calibrate_warmup,
    ];
    opt_f64: [
        target_energy, call_floor,
    ];
    opt_usize: [
        max_hops,
    ];
    json: [
        move_library, rigid_body_geometry, orbit_core, shared_visit_policy,
        continuous_symmetry, keying, arm_selection, species, species_sigma, frozen,
        active_region, ladder_mode, soap_mode, local_step, two_phase, surfaces,
        pair_range, call_floor_elements,
    ];
    opaque: [
        reoccupy, hmc,
    ];
}
