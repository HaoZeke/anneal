// Central-difference check for a published slope.
//
// The step and the bound are the ones in `tests/rgpot_force_fd`. xtb
// 6.7.1 stops the SCC when the energy change drops below `1.d-6 * acc`
// Hartree. An analytic energy that is one rounded value uses one unit in
// the last place. A sum uses the forward error of that sum, which is
// coarser than one ulp of a cancelled result. The model
// third derivative on a length `L` is `|g| / L²`, so the step that
// minimises noise plus truncation is `(3 δ L² / |g|)^(1/3)`. The compared
// slope is the difference at `h/2`. Its truncation is one third of the gap
// between the two differences, and its noise is `2δ / h`.

use std::path::PathBuf;

/// Hartree per unit accuracy. xtb 6.7.1 sets `scfconv = 1.d-6 * acc`.
pub(crate) const XTB_SCF_ENERGY_CHANGE_PER_ACCURACY: f64 = 1.0e-6;

/// `--acc` default printed by xtb 6.7.1. The pipe helper does not pass `--acc`.
pub(crate) const XTB_CLI_ACCURACY: f64 = 1.0;

pub(crate) fn scc_precision(accuracy: f64, hartree_ev: f64) -> f64 {
    XTB_SCF_ENERGY_CHANGE_PER_ACCURACY * accuracy * hartree_ev
}

/// Forward error of summing `terms` addends whose absolute values total
/// `absolute_sum`.
///
/// Higham's first-order bound is `n ε / (1 - n ε) * Σ|a_i|`. One ulp of
/// the returned energy is the floor: the stored result cannot be finer
/// than that, and a cancelled sum cannot be trusted to it.
fn strictly_positive(value: f64) -> bool {
    matches!(value.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater))
}

pub(crate) fn summed_precision(terms: usize, absolute_sum: f64, returned: f64) -> f64 {
    let place = unit_in_the_last_place(returned);
    let n = terms.max(1) as f64;
    let machine = f64::EPSILON;
    if n * machine >= 1.0 || !absolute_sum.is_finite() {
        return place;
    }
    let forward = n * machine / (1.0 - n * machine) * absolute_sum.abs();
    if forward.is_finite() {
        forward.max(place)
    } else {
        place
    }
}

pub(crate) fn unit_in_the_last_place(value: f64) -> f64 {
    let magnitude = value.abs();
    if !magnitude.is_finite() || magnitude == 0.0 {
        return f64::MIN_POSITIVE;
    }
    (magnitude.next_up() - magnitude).abs()
}

/// `h = (3 δ L² / |g|)^(1/3)`, the minimum of `δ/h + |g| h² / (6 L²)`.
pub(crate) fn model_step(precision: f64, length: f64, slope: f64) -> f64 {
    (3.0 * precision * length * length / slope).cbrt()
}

pub(crate) fn model_bound(precision: f64, length: f64, slope: f64, step: f64) -> f64 {
    precision / step + slope * step * step / (6.0 * length * length)
}

pub(crate) fn the_step_minimizes_the_model_bound() -> Result<(), String> {
    let precision = 1.0e-6;
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

/// `Ok(None)` when neither the variable nor `./file_name` is a file.
/// A variable that names a missing path is an error once `./file_name` is
/// absent too.
pub(crate) fn library_file(var: &str, file_name: &str) -> Result<Option<PathBuf>, String> {
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

pub(crate) fn shortest_pair(coords: &[f64]) -> Result<f64, String> {
    let atoms = coords.len() / 3;
    if atoms < 2 || coords.len() != 3 * atoms {
        return Err(format!(
            "{} coordinates are not three per atom",
            coords.len()
        ));
    }
    let mut shortest = f64::INFINITY;
    for left in 0..atoms {
        for right in (left + 1)..atoms {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = coords[3 * left + axis] - coords[3 * right + axis];
                square += delta * delta;
            }
            shortest = shortest.min(square.sqrt());
        }
    }
    if !shortest.is_finite() || shortest <= 0.0 {
        return Err("the geometry has no positive pair distance".into());
    }
    Ok(shortest)
}

pub(crate) struct Row {
    pub backend: String,
    pub index: usize,
    pub energy: f64,
    pub resolved: usize,
    pub dimension: usize,
    pub worst: f64,
    pub worst_bound: f64,
    pub worst_slope: f64,
    pub worst_component: usize,
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

/// `lengths` is the model length of each coordinate. `step_limits` is the
/// largest step that stays inside that coordinate's pair.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_slope(
    backend: &str,
    index: usize,
    geometry: &[f64],
    slope: &[f64],
    energy: f64,
    precision: f64,
    lengths: &[f64],
    step_limits: &[f64],
    mut energy_at: impl FnMut(&[f64]) -> Result<f64, String>,
) -> Result<Row, String> {
    if slope.len() != geometry.len()
        || lengths.len() != geometry.len()
        || step_limits.len() != geometry.len()
    {
        return Err(format!(
            "{backend} geometry {index} slope, length, and step limit do not match the coordinates"
        ));
    }
    if !energy.is_finite() {
        return Err(format!("{backend} geometry {index} energy is not finite"));
    }
    if slope.iter().any(|component| !component.is_finite()) {
        return Err(format!("{backend} geometry {index} slope is not finite"));
    }
    if !strictly_positive(precision) {
        return Err(format!(
            "{backend} geometry {index} energy precision {precision} is not positive"
        ));
    }
    let largest = slope
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .ok_or_else(|| format!("{backend} geometry {index} has an empty slope"))?;
    let mut resolved = 0usize;
    let mut worst = 0.0;
    let mut worst_bound = 0.0;
    let mut worst_slope = 0.0;
    let mut worst_component = 0usize;
    let mut largest_resolved = false;
    for (component, component_slope) in slope.iter().copied().enumerate() {
        let magnitude = component_slope.abs();
        let length = lengths[component];
        let step_limit = step_limits[component];
        let step = model_step(precision, length, magnitude);
        let coordinate_place = unit_in_the_last_place(geometry[component]);
        if !length.is_finite()
            || !strictly_positive(length)
            || !step_limit.is_finite()
            || !strictly_positive(step_limit)
            || !step.is_finite()
            || step <= coordinate_place
            || step >= step_limit
            || step >= length
        {
            if component == largest {
                return Err(format!(
                    "{backend} geometry {index} component {component} slope {component_slope:.6e} step {step:.6e} does not fit inside {step_limit:.6e}"
                ));
            }
            continue;
        }
        let coarse = central(&mut energy_at, geometry, component, step)
            .map_err(|error| format!("{backend} geometry {index} {error}"))?;
        let fine = central(&mut energy_at, geometry, component, step * 0.5)
            .map_err(|error| format!("{backend} geometry {index} {error}"))?;
        let gap = (coarse - fine).abs();
        let noise = 2.0 * precision / step;
        let bound = gap / 3.0 + noise;
        let miss = (component_slope - fine).abs();
        if bound >= magnitude || !bound.is_finite() {
            if component == largest {
                return Err(format!(
                    "{backend} geometry {index} largest component {component} slope {component_slope:.6e} is not resolved (fine {fine:.6e}, gap {gap:.6e}, bound {bound:.6e}, step {step:.6e})"
                ));
            }
            continue;
        }
        if miss > bound {
            return Err(format!(
                "{backend} geometry {index} component {component} slope {component_slope:.6e} fine {fine:.6e} coarse {coarse:.6e} miss {miss:.6e} bound {bound:.6e} step {step:.6e} precision {precision:.6e}"
            ));
        }
        if miss > worst {
            worst = miss;
            worst_bound = bound;
            worst_slope = component_slope;
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
        backend: backend.to_string(),
        index,
        energy,
        resolved,
        dimension: slope.len(),
        worst,
        worst_bound,
        worst_slope,
        worst_component,
    })
}

fn central(
    energy_at: &mut impl FnMut(&[f64]) -> Result<f64, String>,
    geometry: &[f64],
    component: usize,
    step: f64,
) -> Result<f64, String> {
    let mut plus = geometry.to_vec();
    let mut minus = geometry.to_vec();
    plus[component] += step;
    minus[component] -= step;
    let high = energy_at(&plus)?;
    let low = energy_at(&minus)?;
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

pub(crate) fn uniform_lengths(
    coords: &[f64],
    overlap_limit: Option<f64>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let length = shortest_pair(coords)?;
    let step_limit = match overlap_limit {
        Some(limit) => length - limit,
        None => length,
    };
    if !strictly_positive(step_limit) {
        return Err(format!(
            "shortest pair {length:.6e} leaves no room inside the overlap limit"
        ));
    }
    Ok((vec![length; coords.len()], vec![step_limit; coords.len()]))
}
