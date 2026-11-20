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
// between the two differences. The gap is noisy by `3δ / h`, so that third
// is uncertain by `δ / h`, and the fine difference adds `2δ / h`.

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
        // `gap / 3` is the fine truncation. The measured gap is uncertain
        // by the coarse noise `δ/h` plus the fine noise `2δ/h`, and a
        // third of that uncertainty is `δ/h`.
        let gap_noise = precision / step;
        let fine_noise = 2.0 * precision / step;
        let bound = gap / 3.0 + gap_noise + fine_noise;
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

/// Absolute energy error of the eOn CuH2 potential.
///
/// The Cu embedding polynomial has degree 8 and the H polynomial degree 5
/// (`f_cu` and `f_h` in rgpot's `rgpot_cuh2`, the eOn `pot.par` set).
/// Horner's rule bounds a degree-n evaluation by `γ_{2n}` times the sum of
/// the absolute terms, so the factor here is `γ_16`. The sum also includes
/// the two exponentials in each pair and the cutoff shift. One ulp of
/// `energy` is the floor. The distances are the direct ones: the vacuum
/// cell is 60 Å and the cutoff is 6.1 Å, so a periodic image of these
/// clusters does not enter the sum.
pub(crate) fn cuh2_precision(species: &[i32], coordinates: &[f64], energy: f64) -> f64 {
    summed_precision(
        CUH2_HORNER_STAGES,
        cuh2_absolute_terms(species, coordinates),
        energy,
    )
}

const CU_Z: i32 = 29;
const H_Z: i32 = 1;
const CUH2_CUTOFF_A: f64 = 6.1;
const CUH2_HORNER_STAGES: usize = 16;
const CU_CU: [f64; 4] = [2862.3, 3.51236, -109.107, 1.75618];
const H_H: [f64; 4] = [79.5013, 2.47961, -107.555, 2.99918];
const H_CU: [f64; 4] = [86.1495, 4.21154, 15355.5, 6.07604];
const CU_EMBEDDING: [f64; 8] = [
    -112.945,
    8510.04,
    -261_734.0,
    4_780_090.0,
    -52_341_900.0,
    339_124_000.0,
    -1_201_500_000.0,
    1_796_190_000.0,
];
const H_EMBEDDING: [f64; 5] = [-81.7537, 838.668, -3952.35, 8767.87, -6599.37];

fn cuh2_absolute_terms(species: &[i32], coordinates: &[f64]) -> f64 {
    let atoms = species.len();
    if coordinates.len() != 3 * atoms {
        return f64::NAN;
    }
    let mut density = vec![0.0; atoms];
    let mut absolute = 0.0;
    for i in 0..atoms {
        for j in (i + 1)..atoms {
            let distance = direct_distance(coordinates, i, j);
            if !distance.is_finite() || distance >= CUH2_CUTOFF_A {
                continue;
            }
            let pair = match (species[i], species[j]) {
                (CU_Z, CU_Z) => CU_CU,
                (H_Z, H_Z) => H_H,
                (CU_Z, H_Z) | (H_Z, CU_Z) => H_CU,
                _ => continue,
            };
            absolute += pair_absolute(pair, distance);
            density[i] += deposited_density(species[j], distance);
            density[j] += deposited_density(species[i], distance);
        }
    }
    for (atom, z) in species.iter().copied().enumerate() {
        let coeffs: &[f64] = match z {
            CU_Z => &CU_EMBEDDING,
            H_Z => &H_EMBEDDING,
            _ => continue,
        };
        absolute += polynomial_absolute(coeffs, density[atom]);
    }
    absolute
}

fn pair_absolute(coeff: [f64; 4], distance: f64) -> f64 {
    let terms = |r: f64| {
        let first = coeff[0] * (-coeff[1] * r).exp();
        let second = coeff[2] * (-coeff[3] * r).exp();
        first.abs() + second.abs()
    };
    // The cutoff shift is the same pair evaluated at `rcut`.
    terms(distance) + terms(CUH2_CUTOFF_A)
}

fn deposited_density(neighbour: i32, distance: f64) -> f64 {
    let shifted = |scale: f64, power: i32, beta_a: f64, beta_b: f64, gamma: f64| {
        let at =
            |r: f64| scale * r.powi(power) * ((-beta_a * r).exp() + gamma * (-beta_b * r).exp());
        at(distance) - at(CUH2_CUTOFF_A)
    };
    match neighbour {
        CU_Z => shifted(0.273072, 6, 3.69051, 7.38101, 512.0),
        H_Z => shifted(2.14389, 0, 3.77701, 0.0, 0.0),
        _ => 0.0,
    }
}

fn polynomial_absolute(coeffs: &[f64], rho: f64) -> f64 {
    let mut power = 1.0;
    let mut absolute = 0.0;
    for coeff in coeffs {
        power *= rho;
        absolute += (coeff * power).abs();
    }
    absolute
}

fn direct_distance(coordinates: &[f64], left: usize, right: usize) -> f64 {
    let mut acc = 0.0;
    for axis in 0..3 {
        let delta = coordinates[3 * left + axis] - coordinates[3 * right + axis];
        acc += delta * delta;
    }
    acc.sqrt()
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
