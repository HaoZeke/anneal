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

/// Absolute energy error bound for the eOn CuH2 potential (`rgpot_cuh2`).
///
/// Mirrors the exact `f64` operation sequence in `rgpot`'s
/// `rgpot_cuh2.f90` and `NeighbourList.cc` (`rgpot_nlist_fortran_build`):
/// - direct pair displacement `vx, vy, vz`, squared distance `r2`, and
///   `sqrt(r2)` (direct distances: the vacuum cell is 60 Å and the
///   cutoff is 6.1 Å, so no periodic image enters),
/// - cutoff shifts `phicut_*` and `rhocut_*` evaluated at `rcut = 6.1`,
/// - pair exponentials `da * exp(-alpha_a * r) + db * exp(-alpha_b * r)
///   - phicut`, halved (`0.5 * phi`) and accumulated per atom,
/// - deposited densities `scale * r^neta * (exp(-beta_a * r) + gamma *
///   exp(-beta_b * r)) - rhocut` accumulated per atom (`rho_i`), with
///   `r**6` bounded by five sequential multiplications (upper-bounding
///   both sequential and exponentiation-by-squaring lowerings),
/// - explicit monomial powers `rho2..rho8` (Cu) and `rho2..rho5` (H),
///   term products `f(k) * rho^k`, and polynomial summation,
/// - final reduction `sum(e_pair) + sum(e_embed)`.
///
/// Each arithmetic step propagates input error and IEEE-754 rounding
/// error (`0.5 ulp` for `+`, `-`, `*`, `sqrt`; `1 ulp` for libm `exp`)
/// with directed upward rounding (`.next_up()` / `.next_down()`). Every
/// multi-term sum also applies Higham's order-independent `γ_{n-1}`
/// bound so compiler reassociation cannot violate the bound. One ulp of
/// `energy` is the floor.
pub(crate) fn cuh2_precision(species: &[i32], coordinates: &[f64], energy: f64) -> f64 {
    let Some((_, bound)) = cuh2_eval_with_bound(species, coordinates) else {
        return f64::NAN;
    };
    bound.max(unit_in_the_last_place(energy))
}

const CU_Z: i32 = 29;
const H_Z: i32 = 1;
const CUH2_CUTOFF_A: f64 = 6.1;
const CU_CU: [f64; 4] = [2862.3, 3.51236, -109.107, 1.75618];
const H_H: [f64; 4] = [79.5013, 2.47961, -107.555, 2.99918];
const H_CU: [f64; 4] = [86.1495, 4.21154, 15355.5, 6.07604];
const CU_DENSITY: (f64, usize, f64, f64, f64) = (0.273072, 6, 3.69051, 7.38101, 512.0);
const H_DENSITY: (f64, usize, f64, f64, f64) = (2.14389, 0, 3.77701, 0.0, 0.0);
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

fn half_ulp(value: f64) -> f64 {
    let magnitude = value.abs();
    if !magnitude.is_finite() {
        return f64::INFINITY;
    }
    if magnitude == 0.0 {
        return f64::from_bits(1);
    }
    let step = magnitude.next_up() - magnitude;
    let half = 0.5 * step;
    if half > 0.0 { half } else { step }
}

fn up_add(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        return b;
    }
    if b == 0.0 {
        return a;
    }
    (a + b).next_up()
}

fn up_mul(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        return 0.0;
    }
    (a * b).next_up()
}

fn gamma_up(steps: usize) -> f64 {
    if steps == 0 {
        return 0.0;
    }
    let unit_roundoff = 0.5 * f64::EPSILON;
    let nu = up_mul(steps as f64, unit_roundoff);
    if nu >= 1.0 {
        return f64::INFINITY;
    }
    let denom = (1.0 - nu).next_down();
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    (nu / denom).next_up()
}

/// One `f64` intermediate together with a directed-upward upper bound on
/// `|val - exact|`.
#[derive(Clone, Copy)]
struct ErrVal {
    val: f64,
    err: f64,
}

impl ErrVal {
    fn exact(val: f64) -> Self {
        Self { val, err: 0.0 }
    }

    fn neg(self) -> Self {
        Self {
            val: -self.val,
            err: self.err,
        }
    }

    fn add(self, rhs: Self) -> Self {
        let val = self.val + rhs.val;
        let err = up_add(up_add(self.err, rhs.err), half_ulp(val));
        Self { val, err }
    }

    fn sub(self, rhs: Self) -> Self {
        let val = self.val - rhs.val;
        let err = up_add(up_add(self.err, rhs.err), half_ulp(val));
        Self { val, err }
    }

    fn mul(self, rhs: Self) -> Self {
        let val = self.val * rhs.val;
        let t1 = up_mul(self.val.abs(), rhs.err);
        let t2 = up_mul(rhs.val.abs(), self.err);
        let t3 = up_mul(self.err, rhs.err);
        let err = up_add(up_add(up_add(t1, t2), t3), half_ulp(val));
        Self { val, err }
    }

    fn sqrt(self) -> Self {
        let val = self.val.sqrt();
        let lo_arg = (self.val - self.err).next_down().max(0.0);
        let lo = lo_arg.sqrt().next_down().max(0.0);
        let v_lo = val.next_down().max(0.0);
        let denom = (v_lo + lo).next_down();
        let propagated = if denom > 0.0 {
            (self.err / denom).next_up()
        } else {
            self.err.sqrt().next_up()
        };
        let err = up_add(propagated, half_ulp(val));
        Self { val, err }
    }

    fn exp(self) -> Self {
        let val = self.val.exp();
        // Standard libm `exp` guarantees at most 1 ulp.
        let exp_round = unit_in_the_last_place(val);
        // `exp_m1` can round down by up to 1 ulp; step upward twice.
        let expm1_hi = self.err.exp_m1().next_up().next_up();
        let propagated = up_mul(up_add(val, exp_round), expm1_hi);
        let err = up_add(propagated, exp_round);
        Self { val, err }
    }

    fn pow_sequential(self, exponent: usize) -> Self {
        if exponent == 0 {
            return Self::exact(1.0);
        }
        let mut acc = self;
        for _ in 1..exponent {
            acc = acc.mul(self);
        }
        acc
    }

    /// Sum `terms` in left-to-right order for `val`, while bounding the
    /// summation rounding error over any binary association order.
    fn sum_any_order(terms: &[Self]) -> Self {
        if terms.is_empty() {
            return Self::exact(0.0);
        }
        let mut val = terms[0].val;
        let mut input_err = terms[0].err;
        let mut seq_round = 0.0;
        let mut abs_sum = terms[0].val.abs();
        for term in &terms[1..] {
            val += term.val;
            input_err = up_add(input_err, term.err);
            seq_round = up_add(seq_round, half_ulp(val));
            abs_sum = up_add(abs_sum, term.val.abs());
        }
        let order_free_round = up_mul(gamma_up(terms.len() - 1), abs_sum);
        let err = up_add(input_err, seq_round.max(order_free_round));
        Self { val, err }
    }
}

fn cuh2_phi_raw_ev(coeff: [f64; 4], r: ErrVal) -> ErrVal {
    let t1 = ErrVal::exact(coeff[0]).mul(ErrVal::exact(coeff[1]).neg().mul(r).exp());
    let t2 = ErrVal::exact(coeff[2]).mul(ErrVal::exact(coeff[3]).neg().mul(r).exp());
    t1.add(t2)
}

fn cuh2_rho_raw_ev(params: (f64, usize, f64, f64, f64), r: ErrVal) -> ErrVal {
    let (scale, power, beta_a, beta_b, gamma) = params;
    let t1 = ErrVal::exact(beta_a).neg().mul(r).exp();
    let t2 = ErrVal::exact(gamma).mul(ErrVal::exact(beta_b).neg().mul(r).exp());
    let t3 = ErrVal::exact(scale).mul(r.pow_sequential(power));
    t3.mul(t1.add(t2))
}

fn cuh2_eval_with_bound(species: &[i32], coordinates: &[f64]) -> Option<(f64, f64)> {
    let atoms = species.len();
    if coordinates.len() != 3 * atoms {
        return None;
    }
    let rcut = ErrVal::exact(CUH2_CUTOFF_A);
    let phicut_cucu = cuh2_phi_raw_ev(CU_CU, rcut);
    let phicut_hh = cuh2_phi_raw_ev(H_H, rcut);
    let phicut_hcu = cuh2_phi_raw_ev(H_CU, rcut);
    let rhocut_cu = cuh2_rho_raw_ev(CU_DENSITY, rcut);
    let rhocut_h = cuh2_rho_raw_ev(H_DENSITY, rcut);

    let mut rho = Vec::with_capacity(atoms);
    let mut e_pair = Vec::with_capacity(atoms);
    for i in 0..atoms {
        let zi = species[i];
        if zi != CU_Z && zi != H_Z {
            return None;
        }
        let mut rho_terms = Vec::new();
        let mut pair_terms = Vec::new();
        for (j, &zj) in species.iter().enumerate() {
            if i == j {
                continue;
            }
            let vx = ErrVal::exact(coordinates[3 * j]).sub(ErrVal::exact(coordinates[3 * i]));
            let vy =
                ErrVal::exact(coordinates[3 * j + 1]).sub(ErrVal::exact(coordinates[3 * i + 1]));
            let vz =
                ErrVal::exact(coordinates[3 * j + 2]).sub(ErrVal::exact(coordinates[3 * i + 2]));
            let r2 = vx.mul(vx).add(vy.mul(vy)).add(vz.mul(vz));
            let r = r2.sqrt();
            if !r.val.is_finite() || r.val >= CUH2_CUTOFF_A {
                continue;
            }
            let deposited = match zj {
                CU_Z => cuh2_rho_raw_ev(CU_DENSITY, r).sub(rhocut_cu),
                H_Z => cuh2_rho_raw_ev(H_DENSITY, r).sub(rhocut_h),
                _ => return None,
            };
            rho_terms.push(deposited);
            let (pair_coeff, phicut) = match (zi, zj) {
                (CU_Z, CU_Z) => (CU_CU, phicut_cucu),
                (H_Z, H_Z) => (H_H, phicut_hh),
                (CU_Z, H_Z) | (H_Z, CU_Z) => (H_CU, phicut_hcu),
                _ => return None,
            };
            let phi = cuh2_phi_raw_ev(pair_coeff, r).sub(phicut);
            pair_terms.push(ErrVal::exact(0.5).mul(phi));
        }
        rho.push(ErrVal::sum_any_order(&rho_terms));
        e_pair.push(ErrVal::sum_any_order(&pair_terms));
    }

    let mut e_embed = Vec::with_capacity(atoms);
    for (atom, &z) in species.iter().enumerate() {
        let coeffs: &[f64] = match z {
            CU_Z => &CU_EMBEDDING,
            H_Z => &H_EMBEDDING,
            _ => return None,
        };
        let rho_i = rho[atom];
        let mut power = rho_i;
        let mut poly_terms = Vec::with_capacity(coeffs.len());
        poly_terms.push(ErrVal::exact(coeffs[0]).mul(power));
        for &coeff in &coeffs[1..] {
            power = power.mul(rho_i);
            poly_terms.push(ErrVal::exact(coeff).mul(power));
        }
        e_embed.push(ErrVal::sum_any_order(&poly_terms));
    }

    let total = ErrVal::sum_any_order(&e_pair).add(ErrVal::sum_any_order(&e_embed));
    if total.val.is_finite() && total.err.is_finite() {
        Some((total.val, total.err))
    } else {
        None
    }
}

/// Double-double (`~106`-bit) arithmetic used to certify `cuh2_precision`
/// and demonstrate that the former Horner `γ_16` bound was invalid.
#[derive(Clone, Copy)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    // Exact binary64 double-double split of ln(2) (error 5.7e-34).
    const LN2: Self = Self {
        hi: std::f64::consts::LN_2,
        lo: 2.319_046_813_846_299_6e-17,
    };

    fn from_f64(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    fn quick_two_sum(a: f64, b: f64) -> Self {
        let s = a + b;
        let e = b - (s - a);
        Self { hi: s, lo: e }
    }

    fn two_sum(a: f64, b: f64) -> Self {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        Self { hi: s, lo: e }
    }

    fn two_prod(a: f64, b: f64) -> Self {
        let p = a * b;
        let e = a.mul_add(b, -p);
        Self { hi: p, lo: e }
    }

    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    fn add(self, rhs: Self) -> Self {
        let s = Self::two_sum(self.hi, rhs.hi);
        let t = Self::two_sum(self.lo, rhs.lo);
        let c = Self::quick_two_sum(s.hi, s.lo + t.hi);
        Self::quick_two_sum(c.hi, c.lo + t.lo)
    }

    fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    fn mul(self, rhs: Self) -> Self {
        let p = Self::two_prod(self.hi, rhs.hi);
        let e = self.hi.mul_add(rhs.lo, self.lo * rhs.hi);
        Self::quick_two_sum(p.hi, p.lo + e)
    }

    fn div(self, rhs: Self) -> Self {
        let q1 = self.hi / rhs.hi;
        let r = self.sub(rhs.mul(Self::from_f64(q1)));
        let q2 = r.hi / rhs.hi;
        let r = r.sub(rhs.mul(Self::from_f64(q2)));
        let q3 = r.hi / rhs.hi;
        let q = Self::quick_two_sum(q1, q2);
        q.add(Self::from_f64(q3))
    }

    fn sqrt(self) -> Self {
        if self.hi <= 0.0 {
            return Self::ZERO;
        }
        let mut x = Self::from_f64(self.hi.sqrt());
        // Two Newton refinements reach full double-double precision.
        for _ in 0..2 {
            let residual = self.sub(x.mul(x));
            let denom = Self {
                hi: 2.0 * x.hi,
                lo: 2.0 * x.lo,
            };
            x = x.add(residual.div(denom));
        }
        x
    }

    fn exp(self) -> Self {
        let k = (self.hi / Self::LN2.hi).round() as i32;
        let reduced = self.sub(Self::LN2.mul(Self::from_f64(f64::from(k))));
        let mut sum = Self::ONE;
        let mut term = Self::ONE;
        for m in 1..=28 {
            term = term.mul(reduced).div(Self::from_f64(m as f64));
            sum = sum.add(term);
        }
        let scale = 2.0_f64.powi(k);
        Self {
            hi: sum.hi * scale,
            lo: sum.lo * scale,
        }
    }

    fn pow_usize(self, exponent: usize) -> Self {
        let mut acc = Self::ONE;
        for _ in 0..exponent {
            acc = acc.mul(self);
        }
        acc
    }

    fn abs_diff_f64(self, approx: f64) -> f64 {
        let diff = self.sub(Self::from_f64(approx));
        (diff.hi + diff.lo).abs()
    }
}

fn cuh2_phi_raw_dd(coeff: [f64; 4], r: Dd) -> Dd {
    let t1 = Dd::from_f64(coeff[0]).mul(Dd::from_f64(coeff[1]).neg().mul(r).exp());
    let t2 = Dd::from_f64(coeff[2]).mul(Dd::from_f64(coeff[3]).neg().mul(r).exp());
    t1.add(t2)
}

fn cuh2_rho_raw_dd(params: (f64, usize, f64, f64, f64), r: Dd) -> Dd {
    let (scale, power, beta_a, beta_b, gamma) = params;
    let t1 = Dd::from_f64(beta_a).neg().mul(r).exp();
    let t2 = Dd::from_f64(gamma).mul(Dd::from_f64(beta_b).neg().mul(r).exp());
    let t3 = Dd::from_f64(scale).mul(r.pow_usize(power));
    t3.mul(t1.add(t2))
}

fn cuh2_energy_dd(species: &[i32], coordinates: &[f64]) -> Dd {
    let atoms = species.len();
    let rcut = Dd::from_f64(CUH2_CUTOFF_A);
    let phicut_cucu = cuh2_phi_raw_dd(CU_CU, rcut);
    let phicut_hh = cuh2_phi_raw_dd(H_H, rcut);
    let phicut_hcu = cuh2_phi_raw_dd(H_CU, rcut);
    let rhocut_cu = cuh2_rho_raw_dd(CU_DENSITY, rcut);
    let rhocut_h = cuh2_rho_raw_dd(H_DENSITY, rcut);

    let mut rho = vec![Dd::ZERO; atoms];
    let mut e_pair = Dd::ZERO;
    for i in 0..atoms {
        for j in 0..atoms {
            if i == j {
                continue;
            }
            let vx = Dd::from_f64(coordinates[3 * j]).sub(Dd::from_f64(coordinates[3 * i]));
            let vy = Dd::from_f64(coordinates[3 * j + 1]).sub(Dd::from_f64(coordinates[3 * i + 1]));
            let vz = Dd::from_f64(coordinates[3 * j + 2]).sub(Dd::from_f64(coordinates[3 * i + 2]));
            let r = vx.mul(vx).add(vy.mul(vy)).add(vz.mul(vz)).sqrt();
            if r.hi >= CUH2_CUTOFF_A {
                continue;
            }
            let deposited = match species[j] {
                CU_Z => cuh2_rho_raw_dd(CU_DENSITY, r).sub(rhocut_cu),
                H_Z => cuh2_rho_raw_dd(H_DENSITY, r).sub(rhocut_h),
                _ => Dd::ZERO,
            };
            rho[i] = rho[i].add(deposited);
            let (pair_coeff, phicut) = match (species[i], species[j]) {
                (CU_Z, CU_Z) => (CU_CU, phicut_cucu),
                (H_Z, H_Z) => (H_H, phicut_hh),
                _ => (H_CU, phicut_hcu),
            };
            let phi = cuh2_phi_raw_dd(pair_coeff, r).sub(phicut);
            e_pair = e_pair.add(Dd::from_f64(0.5).mul(phi));
        }
    }

    let mut e_embed = Dd::ZERO;
    for (atom, &z) in species.iter().enumerate() {
        let coeffs: &[f64] = if z == CU_Z {
            &CU_EMBEDDING
        } else {
            &H_EMBEDDING
        };
        let mut power = Dd::ONE;
        for &coeff in coeffs {
            power = power.mul(rho[atom]);
            e_embed = e_embed.add(Dd::from_f64(coeff).mul(power));
        }
    }
    e_pair.add(e_embed)
}

/// Former Horner `γ_16` heuristic kept only to demonstrate its invalidity
/// in tests.
fn legacy_horner_cuh2_bound(species: &[i32], coordinates: &[f64], energy: f64) -> f64 {
    let atoms = species.len();
    let mut density = vec![0.0; atoms];
    let mut absolute = 0.0;
    for i in 0..atoms {
        for j in (i + 1)..atoms {
            let mut acc = 0.0;
            for axis in 0..3 {
                let delta = coordinates[3 * i + axis] - coordinates[3 * j + axis];
                acc += delta * delta;
            }
            let r = acc.sqrt();
            if !r.is_finite() || r >= CUH2_CUTOFF_A {
                continue;
            }
            let pair = match (species[i], species[j]) {
                (CU_Z, CU_Z) => CU_CU,
                (H_Z, H_Z) => H_H,
                (CU_Z, H_Z) | (H_Z, CU_Z) => H_CU,
                _ => continue,
            };
            let pair_abs = |dist: f64| {
                (pair[0] * (-pair[1] * dist).exp()).abs()
                    + (pair[2] * (-pair[3] * dist).exp()).abs()
            };
            absolute += pair_abs(r) + pair_abs(CUH2_CUTOFF_A);
            let deposit = |neighbour: i32, dist: f64| match neighbour {
                CU_Z => {
                    let at = |x: f64| {
                        0.273072 * x.powi(6) * ((-3.69051 * x).exp() + 512.0 * (-7.38101 * x).exp())
                    };
                    at(dist) - at(CUH2_CUTOFF_A)
                }
                H_Z => {
                    let at = |x: f64| 2.14389 * (-3.77701 * x).exp();
                    at(dist) - at(CUH2_CUTOFF_A)
                }
                _ => 0.0,
            };
            density[i] += deposit(species[j], r);
            density[j] += deposit(species[i], r);
        }
    }
    for (atom, &z) in species.iter().enumerate() {
        let coeffs: &[f64] = match z {
            CU_Z => &CU_EMBEDDING,
            H_Z => &H_EMBEDDING,
            _ => continue,
        };
        let mut power = 1.0;
        for &coeff in coeffs {
            power *= density[atom];
            absolute += (coeff * power).abs();
        }
    }
    summed_precision(16, absolute, energy)
}

/// Check that a live `rgpot_cuh2` energy agrees with the 106-bit
/// double-double reference within `bound`.
pub(crate) fn cuh2_verify_energy(
    species: &[i32],
    coordinates: &[f64],
    energy: f64,
    bound: f64,
) -> Result<(), String> {
    let reference = cuh2_energy_dd(species, coordinates);
    let actual_err = reference.abs_diff_f64(energy);
    if actual_err <= bound {
        Ok(())
    } else {
        Err(format!(
            "live rgpot_cuh2 energy error {actual_err:.3e} exceeded forward bound {bound:.3e}"
        ))
    }
}

/// Certify that the former Horner CuH2 bound is invalid and that the
/// operation-by-operation forward error bound encloses the actual `f64`
/// rounding error measured against a 106-bit double-double evaluation.
pub(crate) fn cuh2_bound_is_valid_and_old_horner_is_invalid() -> Result<(), String> {
    let sqrt2_err = Dd::from_f64(2.0)
        .sqrt()
        .mul(Dd::from_f64(2.0).sqrt())
        .abs_diff_f64(2.0);
    if sqrt2_err > 1.0e-30 {
        return Err(format!(
            "double-double sqrt self-check error {sqrt2_err:.3e}"
        ));
    }
    let exp_err = Dd::ONE.exp().mul(Dd::ONE.neg().exp()).abs_diff_f64(1.0);
    if exp_err > 1.0e-30 {
        return Err(format!("double-double exp self-check error {exp_err:.3e}"));
    }

    // Near the 6.1 Å cutoff, `rho = rho_cu(r) - rhocut_cu` cancels most
    // digits while `r` and `exp(-beta * r)` still inject rounding error
    // proportional to `rho_cu(r)`. The old Horner bound drops distance,
    // exp, and density-propagation error and under-reports the actual
    // f64 error by orders of magnitude.
    let step = 6.0999 / 3.0_f64.sqrt();
    let near_cutoff = [0.1, 0.2, 0.3, 0.1 + step, 0.2 + step, 0.3 + step];
    let near_species = [CU_Z, H_Z];
    let (near_f64, near_bound) = cuh2_eval_with_bound(&near_species, &near_cutoff)
        .ok_or_else(|| "cuh2_eval_with_bound failed on near-cutoff Cu-H".to_string())?;
    let near_dd = cuh2_energy_dd(&near_species, &near_cutoff);
    let near_actual_err = near_dd.abs_diff_f64(near_f64);
    let near_old_bound = legacy_horner_cuh2_bound(&near_species, &near_cutoff, near_f64);
    if !near_actual_err.is_finite()
        || !near_old_bound.is_finite()
        || near_actual_err <= near_old_bound
    {
        return Err(format!(
            "expected old Horner bound ({near_old_bound:.3e}) to fail against actual f64 error ({near_actual_err:.3e})"
        ));
    }
    let near_new_bound = cuh2_precision(&near_species, &near_cutoff, near_f64);
    if !near_new_bound.is_finite()
        || near_actual_err > near_new_bound
        || near_new_bound < near_bound
    {
        return Err(format!(
            "new CuH2 bound ({near_new_bound:.3e}) failed to enclose near-cutoff actual error ({near_actual_err:.3e})"
        ));
    }

    let checks: [(&str, &[i32], &[f64]); 5] = [
        (
            "cuh2-0",
            &[CU_Z, CU_Z, H_Z],
            &[0.40, 0.20, 0.10, 2.85, 0.55, 0.30, 1.55, 1.85, 0.95],
        ),
        (
            "cuh2-1",
            &[CU_Z, CU_Z, H_Z],
            &[5.10, 4.20, 3.40, 7.70, 4.05, 3.90, 6.05, 5.85, 4.55],
        ),
        (
            "cuh2-2",
            &[CU_Z, CU_Z, H_Z],
            &[8.20, 1.10, 6.40, 10.15, 2.40, 6.10, 9.55, 1.35, 7.90],
        ),
        (
            "cuh2-close",
            &[CU_Z, CU_Z, H_Z],
            &[0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.9, 1.0, 0.0],
        ),
        (
            "cu4h-fcc",
            &[CU_Z, CU_Z, CU_Z, CU_Z, H_Z],
            &[
                0.0, 0.0, 0.0, 1.8075, 1.8075, 0.0, 1.8075, 0.0, 1.8075, 0.0, 1.8075, 1.8075, 0.9,
                0.9, 2.7,
            ],
        ),
    ];
    for (label, sp, coords) in checks {
        let (f64_energy, _) = cuh2_eval_with_bound(sp, coords)
            .ok_or_else(|| format!("cuh2_eval_with_bound failed on {label}"))?;
        let dd_energy = cuh2_energy_dd(sp, coords);
        let actual_err = dd_energy.abs_diff_f64(f64_energy);
        let bound = cuh2_precision(sp, coords, f64_energy);
        if !actual_err.is_finite() || !bound.is_finite() || actual_err > bound {
            return Err(format!(
                "{label}: actual f64 error {actual_err:.3e} exceeded new CuH2 bound {bound:.3e}"
            ));
        }
    }
    Ok(())
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
