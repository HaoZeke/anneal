//! Objective gradient against a central difference of the same energy.
//!
//! The searches take `value_and_gradient` on the eindir handle built in
//! `examples/common/rgpot_eindir.rs`. That gradient is the slope of the
//! energy the same callback returns. Negating a published force a second
//! time leaves the energy alone and reverses the gradient.
//!
//! xtb 6.7.1 stops the SCC when the energy change drops below
//! `1.d-6 * acc` Hartree (`scfconv` in `src/scf_module.F90`). The callback
//! passes `GFN2_ACCURACY`, and rgpot publishes the energy in eV, so the
//! absolute precision `δ` is that threshold times `HARTREE_EV`. An
//! analytic backend returns an f64, and `δ` is one unit in the last place
//! of that energy.
//!
//! A central difference at step `h` has noise `δ / h`. The shortest pair
//! distance `L` is the length on which a pair slope changes by its own
//! size, so the model third derivative is `|g| / L²` and the truncation
//! is `|g| h² / (6 L²)`. The step minimises the sum of noise and
//! truncation. The compared value is the finer difference, at `h/2`.
//! When the error is quadratic in `h`, its truncation is one third of the
//! gap between the two differences, and its noise is `2δ / h`. A component
//! counts only when that bound is smaller than `|g|` and `h` stays inside
//! the pair, off the overlap wall. The largest component has to count, and
//! every counted component has to fall inside its bound.
//!
//! A missing engine library is named and skipped. A library that is
//! present has to resolve each distorted geometry. The callback list is
//! checked either way, so a new force publisher does not pass unseen.

#[allow(unused_imports)]
#[path = "../../examples/common/rgpot_eindir.rs"]
mod rgpot_eindir;

use anneal_core::catalog::molecular::{GFN2_ACCURACY, HARTREE_EV, VACUUM_BOX};
use eindir_core::Objective;
use eindir_core::gradient::DifferentiableObjective;
use ndarray::ArrayView1;
use rgpot_eindir::{OVERLAP_CALL_LIMIT_A, RgpotObjective};
use std::path::PathBuf;
use std::process::ExitCode;

/// Hartree per unit accuracy. xtb 6.7.1 sets `scfconv = 1.d-6 * acc`.
const XTB_SCF_ENERGY_CHANGE_PER_ACCURACY: f64 = 1.0e-6;

const WATER_SPECIES: [i32; 3] = [8, 1, 1];
const WATER: [&[f64]; 3] = [
    &[0.20, 0.15, 0.05, 1.12, 0.28, -0.10, -0.18, 1.02, 0.22],
    &[1.40, -0.30, 0.80, 1.55, 0.62, 0.55, 0.70, -0.55, 1.45],
    &[-1.10, 2.40, -0.50, -0.20, 2.15, -0.30, -1.55, 1.70, 0.10],
];

const CUH2_SPECIES: [i32; 3] = [29, 29, 1];
const CUH2: [&[f64]; 3] = [
    &[0.40, 0.20, 0.10, 2.85, 0.55, 0.30, 1.55, 1.85, 0.95],
    &[5.10, 4.20, 3.40, 7.70, 4.05, 3.90, 6.05, 5.85, 4.55],
    &[8.20, 1.10, 6.40, 10.15, 2.40, 6.10, 9.55, 1.35, 7.90],
];

fn main() -> ExitCode {
    let mut failures = Vec::new();
    if let Err(error) = force_callbacks_are_the_ones_checked() {
        failures.push(error);
    }
    if let Err(error) = geometries_are_scalene() {
        failures.push(error);
    }
    if let Err(error) = the_step_minimizes_the_model_bound() {
        failures.push(error);
    }

    let box_ = vacuum_box();
    failures.extend(check_xtb(&box_));
    failures.extend(check_cuh2(&box_));

    if failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        for failure in &failures {
            eprintln!("FAIL {failure}");
        }
        ExitCode::from(1)
    }
}

fn vacuum_box() -> [f64; 9] {
    [
        VACUUM_BOX, 0.0, 0.0, 0.0, VACUUM_BOX, 0.0, 0.0, 0.0, VACUUM_BOX,
    ]
}

fn force_callbacks_are_the_ones_checked() -> Result<(), String> {
    let source = include_str!("../../examples/common/rgpot_eindir.rs");
    let mut callbacks = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        if line.contains("rgpot_potential_new_eindir(") {
            let next = lines
                .get(index + 1)
                .map(|line| line.trim().trim_end_matches(','))
                .unwrap_or("");
            callbacks.push(next.to_string());
        }
    }
    let expected = ["xtb_callback", "cuh2_callback"];
    if callbacks != expected {
        return Err(format!(
            "force callbacks are {callbacks:?}; the finite-difference check covers {expected:?}"
        ));
    }
    Ok(())
}

fn geometries_are_scalene() -> Result<(), String> {
    for (label, geometry) in water_cases().into_iter().chain(cuh2_cases()) {
        scalene(label, geometry)?;
    }
    Ok(())
}

fn water_cases() -> Vec<(&'static str, &'static [f64])> {
    WATER
        .iter()
        .enumerate()
        .map(|(index, geometry)| (["water-0", "water-1", "water-2"][index], *geometry))
        .collect()
}

fn cuh2_cases() -> Vec<(&'static str, &'static [f64])> {
    CUH2.iter()
        .enumerate()
        .map(|(index, geometry)| (["cuh2-0", "cuh2-1", "cuh2-2"][index], *geometry))
        .collect()
}

fn scalene(label: &str, coords: &[f64]) -> Result<(), String> {
    if coords.len() != 9 {
        return Err(format!("{label} has {} coordinates", coords.len()));
    }
    let distances = pair_distances(coords);
    let mut ordered = distances;
    ordered.sort_by(|left, right| left.total_cmp(right));
    if ordered[0] <= OVERLAP_CALL_LIMIT_A {
        return Err(format!(
            "{label} shortest pair {:.6} is inside the overlap wall {OVERLAP_CALL_LIMIT_A}",
            ordered[0]
        ));
    }
    if !(ordered[1] > ordered[0] && ordered[2] > ordered[1]) {
        return Err(format!(
            "{label} pair distances {distances:?} are not distinct"
        ));
    }
    if !(triangle_area(coords) > 0.0) {
        return Err(format!("{label} is flat"));
    }
    Ok(())
}

fn pair_distances(coords: &[f64]) -> [f64; 3] {
    let mut distances = [0.0; 3];
    let mut slot = 0;
    for left in 0..3 {
        for right in (left + 1)..3 {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = coords[3 * left + axis] - coords[3 * right + axis];
                square += delta * delta;
            }
            distances[slot] = square.sqrt();
            slot += 1;
        }
    }
    distances
}

fn triangle_area(coords: &[f64]) -> f64 {
    let ax = coords[3] - coords[0];
    let ay = coords[4] - coords[1];
    let az = coords[5] - coords[2];
    let bx = coords[6] - coords[0];
    let by = coords[7] - coords[1];
    let bz = coords[8] - coords[2];
    let cx = ay * bz - az * by;
    let cy = az * bx - ax * bz;
    let cz = ax * by - ay * bx;
    0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
}

fn the_step_minimizes_the_model_bound() -> Result<(), String> {
    // A unit sample of the model, to check that the critical point is a minimum.
    let precision = scc_precision_ev();
    let length = 1.0;
    let slope = 1.0;
    let step = model_step(precision, length, slope);
    let at = model_bound(precision, length, slope, step);
    let shorter = model_bound(precision, length, slope, step * 0.5);
    let longer = model_bound(precision, length, slope, step * 2.0);
    if !(at < shorter && at < longer) {
        return Err(format!(
            "model step {step:.6e} bound {at:.6e} is not below {shorter:.6e} and {longer:.6e}"
        ));
    }
    Ok(())
}

fn scc_precision_ev() -> f64 {
    XTB_SCF_ENERGY_CHANGE_PER_ACCURACY * GFN2_ACCURACY * HARTREE_EV
}

fn unit_in_the_last_place(value: f64) -> f64 {
    let magnitude = value.abs();
    if !magnitude.is_finite() || magnitude == 0.0 {
        return f64::MIN_POSITIVE;
    }
    (magnitude.next_up() - magnitude).abs()
}

/// `h = (3 δ L² / |g|)^(1/3)`, the minimum of `δ/h + |g| h² / (6 L²)`.
fn model_step(precision: f64, length: f64, slope: f64) -> f64 {
    (3.0 * precision * length * length / slope).cbrt()
}

fn model_bound(precision: f64, length: f64, slope: f64, step: f64) -> f64 {
    precision / step + slope * step * step / (6.0 * length * length)
}

fn check_xtb(box_: &[f64; 9]) -> Vec<String> {
    let precision = scc_precision_ev();
    println!(
        "xtb energy precision {precision:.6e} eV from scfconv {XTB_SCF_ENERGY_CHANGE_PER_ACCURACY:.1e} * accuracy {GFN2_ACCURACY} * hartree {HARTREE_EV}"
    );
    match library_file("RGPOT_XTB_ENGINE", "libxtb_engine.so") {
        Ok(None) => {
            let reason =
                "xtb: skipped, libxtb_engine.so is absent (RGPOT_XTB_ENGINE or ./libxtb_engine.so)";
            println!("{reason}");
            Vec::new()
        }
        Ok(Some(path)) => {
            println!("xtb library {}", path.display());
            let mut pot = RgpotObjective::xtb(&WATER_SPECIES, *box_);
            check_pot(
                "xtb",
                &mut pot,
                &WATER,
                EnergyPrecision::Fixed(precision),
                Some(OVERLAP_CALL_LIMIT_A),
            )
        }
        Err(error) => vec![error],
    }
}

fn check_cuh2(box_: &[f64; 9]) -> Vec<String> {
    match library_file("RGPOT_CUH2_LIBRARY", "librgpot_cuh2.so") {
        Ok(None) => {
            let reason = "cuh2: skipped, librgpot_cuh2.so is absent (RGPOT_CUH2_LIBRARY or ./librgpot_cuh2.so)";
            println!("{reason}");
            Vec::new()
        }
        Ok(Some(path)) => {
            println!("cuh2 library {}", path.display());
            let mut pot = RgpotObjective::cuh2(&CUH2_SPECIES, *box_);
            check_pot(
                "cuh2",
                &mut pot,
                &CUH2,
                EnergyPrecision::UnitInTheLastPlace,
                None,
            )
        }
        Err(error) => vec![error],
    }
}

fn library_file(var: &str, file_name: &str) -> Result<Option<PathBuf>, String> {
    let mut missing_env = None;
    if let Some(raw) = std::env::var_os(var)
        && !raw.is_empty()
    {
        let path = PathBuf::from(&raw);
        if path.is_file() {
            return Ok(Some(path));
        }
        missing_env = Some(path);
    }
    let local = PathBuf::from(file_name);
    if local.is_file() {
        return Ok(Some(local));
    }
    if let Some(path) = missing_env {
        return Err(format!(
            "{var} names {} which is not a file, and {file_name} is absent",
            path.display()
        ));
    }
    Ok(None)
}

#[derive(Clone, Copy)]
enum EnergyPrecision {
    /// The same absolute precision at every geometry.
    Fixed(f64),
    /// One unit in the last place of the energy just returned.
    UnitInTheLastPlace,
}

fn check_pot(
    backend: &'static str,
    pot: &mut RgpotObjective,
    geometries: &[&[f64]; 3],
    precision: EnergyPrecision,
    overlap_limit: Option<f64>,
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut compared = 0usize;
    for (index, geometry) in geometries.iter().enumerate() {
        pot.restart();
        match compare_geometry(backend, pot, index, geometry, precision, overlap_limit) {
            Ok(row) => {
                compared += row.resolved;
                println!("{row}");
            }
            Err(error) => failures.push(error),
        }
    }
    if compared == 0 && failures.is_empty() {
        failures.push(format!(
            "{backend}: no component was resolved, so the comparison did not run"
        ));
    }
    failures
}

struct Row {
    backend: &'static str,
    index: usize,
    energy: f64,
    resolved: usize,
    dimension: usize,
    worst: f64,
    worst_bound: f64,
    worst_slope: f64,
    worst_component: usize,
}

impl std::fmt::Display for Row {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} geometry {} energy {:.12} resolved {}/{} worst {:.3e} bound {:.3e} slope {:.3e} component {}",
            self.backend,
            self.index,
            self.energy,
            self.resolved,
            self.dimension,
            self.worst,
            self.worst_bound,
            self.worst_slope,
            self.worst_component
        )
    }
}

fn compare_geometry(
    backend: &'static str,
    pot: &RgpotObjective,
    index: usize,
    geometry: &[f64],
    precision: EnergyPrecision,
    overlap_limit: Option<f64>,
) -> Result<Row, String> {
    let objective = pot.wrapper();
    let (energy, gradient) = objective.value_and_gradient(ArrayView1::from(geometry));
    if !energy.is_finite() {
        return Err(format!("{backend} geometry {index} energy is not finite"));
    }
    if gradient.len() != geometry.len() || gradient.iter().any(|component| !component.is_finite()) {
        return Err(format!(
            "{backend} geometry {index} gradient is not a finite slope"
        ));
    }
    let precision = match precision {
        EnergyPrecision::Fixed(precision) => precision,
        EnergyPrecision::UnitInTheLastPlace => {
            let precision = unit_in_the_last_place(energy);
            println!(
                "{backend} geometry {index} energy precision {precision:.6e} eV (one ulp of {energy:.12})"
            );
            precision
        }
    };
    if !(precision > 0.0) {
        return Err(format!(
            "{backend} geometry {index} energy precision {precision} is not positive"
        ));
    }

    let length = pair_distances(geometry)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let step_limit = match overlap_limit {
        Some(limit) => (length - limit).min(length),
        None => length,
    };
    if !(step_limit > 0.0) {
        return Err(format!(
            "{backend} geometry {index} has no room for a step inside the pair"
        ));
    }

    let slopes: Vec<f64> = gradient.iter().copied().collect();
    let largest = slopes
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .ok_or_else(|| format!("{backend} geometry {index} has an empty gradient"))?;

    let mut resolved = 0usize;
    let mut worst = 0.0;
    let mut worst_bound = 0.0;
    let mut worst_slope = 0.0;
    let mut worst_component = 0usize;
    let mut largest_resolved = false;
    for (component, slope) in slopes.iter().copied().enumerate() {
        let magnitude = slope.abs();
        let step = model_step(precision, length, magnitude);
        let coordinate_place = unit_in_the_last_place(geometry[component]);
        if !step.is_finite() || step <= coordinate_place || step >= step_limit || step >= length {
            if component == largest {
                return Err(format!(
                    "{backend} geometry {index} component {component} slope {slope:.6e} step {step:.6e} does not fit inside {step_limit:.6e}"
                ));
            }
            continue;
        }
        let coarse = central(&objective, geometry, component, step)
            .map_err(|error| format!("{backend} geometry {index} {error}"))?;
        let fine = central(&objective, geometry, component, step * 0.5)
            .map_err(|error| format!("{backend} geometry {index} {error}"))?;
        let gap = (coarse - fine).abs();
        // Finer denominator is `h`, so an energy error of `2δ` is `2δ/h`.
        let noise = 2.0 * precision / step;
        let bound = gap / 3.0 + noise;
        let miss = (slope - fine).abs();
        if !(bound < magnitude) {
            if component == largest {
                return Err(format!(
                    "{backend} geometry {index} largest component {component} slope {slope:.6e} is not resolved (fine {fine:.6e}, gap {gap:.6e}, bound {bound:.6e}, step {step:.6e})"
                ));
            }
            continue;
        }
        if miss > bound {
            return Err(format!(
                "{backend} geometry {index} component {component} slope {slope:.6e} fine {fine:.6e} coarse {coarse:.6e} miss {miss:.6e} bound {bound:.6e} step {step:.6e} precision {precision:.6e}"
            ));
        }
        if miss > worst {
            worst = miss;
            worst_bound = bound;
            worst_slope = slope;
            worst_component = component;
        }
        resolved += 1;
        if component == largest {
            largest_resolved = true;
        }
    }
    if !largest_resolved || resolved < 3 {
        return Err(format!(
            "{backend} geometry {index} resolved {resolved} components; the comparison would be vacuous"
        ));
    }
    Ok(Row {
        backend,
        index,
        energy,
        resolved,
        dimension: slopes.len(),
        worst,
        worst_bound,
        worst_slope,
        worst_component,
    })
}

fn central(
    objective: &impl Objective<f64>,
    geometry: &[f64],
    component: usize,
    step: f64,
) -> Result<f64, String> {
    let mut plus = geometry.to_vec();
    let mut minus = geometry.to_vec();
    plus[component] += step;
    minus[component] -= step;
    let high = objective.eval(ArrayView1::from(&plus));
    let low = objective.eval(ArrayView1::from(&minus));
    if !high.is_finite() || !low.is_finite() {
        return Err(format!(
            "component {component} step {step:.6e} energy is not finite (plus {high}, minus {low})"
        ));
    }
    let slope = (high - low) / (2.0 * step);
    if !slope.is_finite() {
        return Err(format!(
            "component {component} step {step:.6e} difference is not finite"
        ));
    }
    Ok(slope)
}
