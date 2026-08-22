//! Invert basins already occupied so xtsci will not walk back into them.
//!
//! Henkelman and Jónsson, *J. Chem. Phys.* **1999**, *111*, 7010
//! <https://doi.org/10.1063/1.480097>: the dimer replaces \(F\) by
//! \(F-2(F\cdot P)P\) so a first-order stepper walks *up* the lowest
//! mode to a saddle. Occupancy Leave applies the same Householder in
//! the DECAF packing map, not in Cartesian radius. \(\mu\) is the
//! mean of per-center [`crate::soap::local_nu3_z`] (SOAP plus ACE
//! \(\nu=3\)) at [`crate::catalog::packing::PACKING_SPEC`]. For the
//! nearest archived packing \(\mu_k\),
//! \(\hat u_\varphi=(\mu-\mu_k)/\|\mu-\mu_k\|\) and
//! \(P=J_\mu^{\mathsf T}\hat u_\varphi\) is the Cartesian pullback
//! through the analytic stacked Jacobian: the same increment-on-every-center
//! lift as [`crate::soap::kick_packing_nu3`]. Then
//! \(g\leftarrow g-2(g\cdot\hat P)\hat P\) when \(g\cdot P>0\).
//! Leftover SOAP \(p_i-\mu\) collapses across the paper funnels; this
//! map does not.
//!
//! When the packing mean is unavailable (fewer than a closed-shell
//! neighbour cloud), the same Householder falls back to the COM-free
//! Cartesian radius of each well. That fallback is not a packing walk.
//!
//! The Householder on \(g\) is not conservative. The line search is
//! run on the matching PES \(E+V\), a Gaussian hill on each known
//! well in the same map. After a transformed quench that changes
//! DECAF family, a raw-\(E\) polish sits on a true minimum of that
//! well.

use std::cell::RefCell;

use ndarray::{Array1, Array2, ArrayView1};

use crate::catalog::packing::{MINIMUM_PACKING_ATOMS, PACKING_SPEC};
use crate::soap::{jacobian_nu3, local_nu3_z};

struct Well {
    coords: Array1<f64>,
    packing_mean: Option<Array1<f64>>,
}

struct Armed {
    wells: Vec<Well>,
    sigma_rmsd: f64,
    /// Raw \(E\) at the last point the transform was asked about, so the
    /// walk can see the ridge it is crossing rather than the hill it is on.
    raw_energy: Option<f64>,
    lift: Option<f64>,
    sigma_phi: Option<f64>,
    /// Geometry the cached packing modes were lifted at.
    mode_x: Option<Array1<f64>>,
    /// Cached \((\hat P, \|P\|)\) per well, in `wells` order.
    modes: Vec<(usize, Array1<f64>, f64)>,
}

thread_local! {
    static ARMED: RefCell<Option<Armed>> = const { RefCell::new(None) };
}

/// Arm the transformed quench for one occupancy Leave.
///
/// `origin` is the live well this extra is leaving, and `references` are the
/// packings already on file, as coordinates. Every one of them contributes a
/// mode the Householder inverts, so the walk is pushed away from all of them
/// and not only out of the well it starts in. The leftover-SOAP archive is a
/// list of cloud means, not structures, so it cannot supply these: a Leave
/// that reads it sees vectors of the wrong length and inverts nothing but its
/// own well.
pub fn arm_leave(origin: ArrayView1<f64>, sigma_rmsd: f64, references: &[Vec<f64>]) {
    let mut wells = vec![Well {
        packing_mean: packing_mean(origin),
        coords: origin.to_owned(),
    }];
    for reference in references {
        if reference.len() != origin.len() {
            continue;
        }
        let coords = Array1::from(reference.clone());
        if same_point(coords.view(), origin) {
            continue;
        }
        wells.push(Well {
            packing_mean: packing_mean(coords.view()),
            coords,
        });
    }
    ARMED.with(|slot| {
        *slot.borrow_mut() = Some(Armed {
            wells,
            raw_energy: None,
            sigma_rmsd: sigma_rmsd.max(1e-6),
            lift: None,
            sigma_phi: None,
            mode_x: None,
            modes: Vec::new(),
        });
    });
}

/// Drop the transform. Later xtsci calls see the raw PES.
pub fn disarm() {
    ARMED.with(|slot| *slot.borrow_mut() = None);
}

/// Whether a Leave quench is currently transformed.
pub fn is_armed() -> bool {
    ARMED.with(|slot| slot.borrow().is_some())
}

/// Run `body` on the raw surface, then restore the armed transform.
///
/// The polish that decides where a walk landed has to run on \(E\), not on
/// \(E+V\): a hill the Leave put there is not part of the landscape, and a
/// minimum of the sum is not a minimum of the potential.
pub fn with_disarmed<T>(body: impl FnOnce() -> T) -> T {
    let held = ARMED.with(|slot| slot.borrow_mut().take());
    let out = body();
    ARMED.with(|slot| *slot.borrow_mut() = held);
    out
}

/// Largest Cartesian RMSD one armed step may take.
///
/// The hill width is the size of the well being left; the walk across it is
/// not one step of that size. A trust radius equal to the width lets an
/// accepted uphill step carry atoms through each other, and the walk then
/// climbs for the whole iteration budget and reports a structure at
/// \(10^{11}\varepsilon\) that no quench recovers.
pub const LEAVE_WALK_STEP: f64 = 0.10;

/// Largest Cartesian RMSD one armed walk may cover in total.
///
/// A cooperative rearrangement moves atoms by about a nearest-neighbour
/// separation. Past that the walk is not crossing a ridge between packings,
/// it is pulling the cluster apart, and the ridge rule cannot end a climb
/// that never turns over.
pub const LEAVE_WALK_SPAN: f64 = 0.8;

/// Consecutive falls in raw \(E\) that name the far side of a ridge.
pub const LEAVE_WALK_DESCENTS: usize = 3;

/// Largest climb in raw \(E\), in units of the well depth per atom.
///
/// The span rule alone accepts a step that puts two atoms on top of each
/// other, because overlapping atoms move \(\mu\) further than any
/// rearrangement does. Wales and Doye put the LJ75 ico-Marks barriers at
/// 8.69 and 7.48 \(\varepsilon\) against a well depth of 5.28
/// \(\varepsilon\) per atom, so four times the depth per atom clears the
/// ridge the walk is meant to cross and refuses the ones it is not.
pub const LEAVE_WALK_CLIMB: f64 = 4.0;

/// xtsci step on the transformed surface: two-loop direction, accept a step
/// that increases the span from the known wells. Span is packing L2
/// \(\min_k\|\mu-\mu_k\|\) when the wells carry a \(\nu=3\) mean,
/// otherwise COM-free RMSD. Raw \(E\) may rise; that is the dimer walk away
/// from an occupied packing.
///
/// The walk ends at the ridge, the way an activation-relaxation walk does:
/// raw \(E\) climbs while the invert holds the descent component that
/// points back at the wells on file, and the first steps where it falls
/// again are the far side. Quenching from there is what decides which
/// packing the walk landed in.
///
/// Two stopping rules that do not work, both measured from the LJ75
/// icosahedral minimum. Crossing the DECAF grain stops the walk on a
/// distorted geometry that has not crossed anything: the quench that follows
/// returns to \(-396.282249\), the floor it started on, every time. Having
/// no rule at all lets the walk climb for the whole iteration budget and
/// report structures between \(10^4\) and \(10^{11}\varepsilon\), which
/// is atoms sitting on each other. [`LEAVE_WALK_SPAN`] bounds the second
/// case, since a raw energy that rises monotonically never turns over.
pub fn step_xtsci<F>(
    opt: &mut crate::methods::warm_lbfgs::WarmLbfgs,
    x0: ArrayView1<f64>,
    max_iter: usize,
    mut fg: F,
) -> (f64, Array1<f64>)
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    let sigma = ARMED.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|a| a.sigma_rmsd.min(LEAVE_WALK_STEP))
            .unwrap_or(LEAVE_WALK_STEP)
    });
    let mut x = x0.to_owned();
    let Some((mut energy, mut grad)) = fg(x.view()) else {
        return (f64::INFINITY, x);
    };
    let n_at = (x.len() / 3).max(1) as f64;
    let start_raw = raw_energy();
    let ceiling = start_raw.map(|raw| raw + LEAVE_WALK_CLIMB * raw.abs() / n_at);
    let mut peak = start_raw.unwrap_or(f64::NEG_INFINITY);
    let mut descents = 0usize;
    for _ in 0..max_iter {
        let mut direction = opt.two_loop(grad.view());
        if direction.dot(&grad) >= 0.0 {
            direction = grad.mapv(|v| -v);
        }
        let step_rmsd = (direction.iter().map(|v| v * v).sum::<f64>() / n_at).sqrt();
        if step_rmsd < 1e-15 {
            break;
        }
        let mut alpha = (sigma / step_rmsd).min(1.0);
        let start_span = span_from_wells(x.view());
        let mut accepted = false;
        for _ in 0..8 {
            let mut trial = x.clone();
            trial.scaled_add(alpha, &direction);
            let Some((te, tg)) = fg(trial.view()) else {
                return (energy, x);
            };
            let trial_span = span_from_wells(trial.view());
            let climbed = match (raw_energy(), ceiling) {
                (Some(raw), Some(ceiling)) => raw > ceiling,
                _ => false,
            };
            if !climbed && (trial_span > start_span || te < energy) {
                let s = &trial - &x;
                let y = &tg - &grad;
                opt.record_pair(s, y);
                x = trial;
                energy = te;
                grad = tg;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
        match raw_energy() {
            Some(raw) if raw > peak => {
                peak = raw;
                descents = 0;
            }
            Some(_) => descents += 1,
            None => {}
        }
        if descents >= LEAVE_WALK_DESCENTS {
            break;
        }
        let travelled = (x
            .iter()
            .zip(x0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / n_at)
            .sqrt();
        if travelled >= LEAVE_WALK_SPAN {
            break;
        }
    }
    (energy, x)
}

/// Distance from `x` to the nearest armed well.
///
/// Packing L2 \(\min_k\|\mu-\mu_k\|\) when the wells carry a
/// \(\nu=3\) mean, otherwise COM-free RMSD. Zero when unarmed.
/// Occupancy Leave keeps the invert walk when this rises.
pub fn span(x: ArrayView1<f64>) -> f64 {
    span_from_wells(x)
}

fn span_from_wells(x: ArrayView1<f64>) -> f64 {
    ARMED.with(|slot| {
        let held = slot.borrow();
        let Some(armed) = held.as_ref() else {
            return 0.0;
        };
        if armed.wells.iter().any(|well| well.packing_mean.is_some()) {
            let Some(mu) = packing_mean(x) else {
                return 0.0;
            };
            return armed
                .wells
                .iter()
                .filter_map(|well| well.packing_mean.as_ref())
                .map(|mu_k| packing_l2(mu.view(), mu_k.view()))
                .fold(f64::INFINITY, f64::min);
        }
        armed
            .wells
            .iter()
            .filter(|well| well.coords.len() == x.len())
            .map(|well| com_free_delta(x, well.coords.view()).1)
            .fold(f64::INFINITY, f64::min)
    })
}

/// \((E,g)\) seen by xtsci: identity when unarmed, \(E+V\) and
/// Householder-\(g+\nabla V\) when a Leave is in flight.
pub fn effective(x: ArrayView1<f64>, energy: f64, grad: Array1<f64>) -> (f64, Array1<f64>) {
    ARMED.with(|slot| {
        let mut held = slot.borrow_mut();
        let Some(armed) = held.as_mut() else {
            return (energy, grad);
        };
        armed.raw_energy = Some(energy);
        transform(armed, x, energy, grad)
    })
}

/// Raw \(E\) at the last point the armed transform saw.
fn raw_energy() -> Option<f64> {
    ARMED.with(|slot| slot.borrow().as_ref().and_then(|armed| armed.raw_energy))
}

fn transform(
    armed: &mut Armed,
    x: ArrayView1<f64>,
    energy: f64,
    grad: Array1<f64>,
) -> (f64, Array1<f64>) {
    if armed.wells.iter().any(|well| well.packing_mean.is_some()) {
        return transform_packing(armed, x, energy, grad);
    }
    transform_cartesian(armed, x, energy, grad)
}

fn transform_packing(
    armed: &mut Armed,
    x: ArrayView1<f64>,
    energy: f64,
    grad: Array1<f64>,
) -> (f64, Array1<f64>) {
    let Some(mu) = packing_mean(x) else {
        return (energy, grad);
    };
    let modes = packing_modes(armed, x, mu.view());
    if armed.lift.is_none() {
        let mut amplitude = 0.0;
        let mut sigma_phi = 0.0;
        for (p, r_phi, _) in &modes {
            if *r_phi < 1e-12 {
                continue;
            }
            let lambda = dot(&grad, p) / r_phi;
            if lambda > 0.0 {
                let trial = lambda * r_phi * r_phi;
                if trial > amplitude {
                    amplitude = trial;
                    sigma_phi = *r_phi;
                }
            }
        }
        armed.lift = Some(amplitude);
        armed.sigma_phi = Some(sigma_phi);
    }
    let amplitude = armed.lift.unwrap_or(0.0);
    let sigma_phi = armed.sigma_phi.unwrap_or(0.0);
    let mut grad = householder_packing(&modes, grad);
    if amplitude <= 0.0 || sigma_phi <= 1e-12 {
        return (energy, grad);
    }
    let mut potential = 0.0;
    for (p, r_phi, p_norm) in &modes {
        if *r_phi < 1e-12 {
            continue;
        }
        let gauss = (-0.5 * (r_phi / sigma_phi) * (r_phi / sigma_phi)).exp();
        potential += amplitude * gauss;
        // ∇_x r_φ = P_unnorm = ||P|| P̂. dV/dr_φ = −A gauss r_φ / σ².
        let scale = -amplitude * gauss * r_phi / (sigma_phi * sigma_phi) * p_norm;
        for (g, pk) in grad.iter_mut().zip(p.iter()) {
            *g += scale * *pk;
        }
    }
    (energy + potential, grad)
}

/// Packing modes at `x`, one per well that carries a \(\nu=3\) mean.
///
/// \(\hat P\) and \(\|P\|\) come from the stacked Jacobian, which is
/// \((N d)\times 3N\) and is the whole cost of a transformed step, so they
/// are lifted again only once the geometry has moved by
/// [`crate::catalog::PACKING_MOVE_EPS`] -- the same staleness the packing
/// book uses on its histograms, and below the grain either can resolve.
/// \(r_\varphi\) is the distance in the map and is read fresh from
/// \(\mu\) on every call, so the hill and its gradient still track the
/// walk.
fn packing_modes(
    armed: &mut Armed,
    x: ArrayView1<f64>,
    mu: ArrayView1<f64>,
) -> Vec<(Array1<f64>, f64, f64)> {
    let stale = armed
        .mode_x
        .as_ref()
        .is_none_or(|held| moved(held.view(), x, crate::catalog::PACKING_MOVE_EPS));
    if stale {
        let j = jacobian_nu3(x, PACKING_SPEC, None);
        armed.modes = armed
            .wells
            .iter()
            .enumerate()
            .filter_map(|(index, well)| {
                let mu_k = well.packing_mean.as_ref()?;
                let (p, _, p_norm) = lift_packing_mode(x, mu, mu_k.view(), &j)?;
                Some((index, p, p_norm))
            })
            .collect();
        armed.mode_x = Some(x.to_owned());
    }
    armed
        .modes
        .iter()
        .filter_map(|(index, p, p_norm)| {
            let mu_k = armed.wells.get(*index)?.packing_mean.as_ref()?;
            if mu_k.len() != mu.len() {
                return None;
            }
            let r_phi = mu
                .iter()
                .zip(mu_k.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            Some((p.clone(), r_phi, *p_norm))
        })
        .collect()
}

/// Whether any atom moved by more than `eps`.
fn moved(left: ArrayView1<f64>, right: ArrayView1<f64>, eps: f64) -> bool {
    if left.len() != right.len() {
        return true;
    }
    let limit = eps * eps;
    (0..left.len() / 3).any(|atom| {
        let d0 = left[3 * atom] - right[3 * atom];
        let d1 = left[3 * atom + 1] - right[3 * atom + 1];
        let d2 = left[3 * atom + 2] - right[3 * atom + 2];
        d0 * d0 + d1 * d1 + d2 * d2 > limit
    })
}

fn transform_cartesian(
    armed: &mut Armed,
    x: ArrayView1<f64>,
    energy: f64,
    grad: Array1<f64>,
) -> (f64, Array1<f64>) {
    let n_at = (x.len() / 3).max(1) as f64;
    let sigma = armed.sigma_rmsd;
    let mut potential = 0.0;
    if armed.lift.is_none() {
        // Amplitude that inverts the measured radial curvature of the
        // nearest known well: A = λ N σ² so H_V(well) = -λ I in the
        // RMSD metric. λ ≈ (g·û)/r at the Leave start.
        let mut amplitude: f64 = 0.0;
        for well in &armed.wells {
            if well.coords.len() != x.len() {
                continue;
            }
            let (delta, rmsd) = com_free_delta(x, well.coords.view());
            if rmsd < 1e-12 {
                continue;
            }
            let radius = n_at.sqrt() * rmsd;
            let uh_dot_g = dot(&grad, &delta) / radius;
            let lambda = uh_dot_g / radius;
            if lambda > 0.0 {
                amplitude = amplitude.max(lambda * n_at * sigma * sigma);
            }
        }
        armed.lift = Some(amplitude);
    }
    let amplitude = armed.lift.unwrap_or(0.0);
    let mut grad = householder_cartesian(&armed.wells, x, grad);
    if amplitude <= 0.0 {
        return (energy, grad);
    }
    for well in &armed.wells {
        if well.coords.len() != x.len() {
            continue;
        }
        let (delta, rmsd) = com_free_delta(x, well.coords.view());
        if rmsd < 1e-12 {
            continue;
        }
        let gauss = (-0.5 * (rmsd / sigma) * (rmsd / sigma)).exp();
        potential += amplitude * gauss;
        let scale = -amplitude * gauss / (n_at * sigma * sigma);
        for (g, d) in grad.iter_mut().zip(delta.iter()) {
            *g += scale * *d;
        }
    }
    (energy + potential, grad)
}

/// Invert the force component that would walk toward a known packing.
///
/// Henkelman–Jónsson on the pulled-back packing mode: \(g \leftarrow
/// g-2(g\cdot\hat P)\hat P\) when \(g\cdot\hat P>0\) (descent
/// \(-\nabla E\) points at \(\mu_k\)).
fn householder_packing(modes: &[(Array1<f64>, f64, f64)], mut grad: Array1<f64>) -> Array1<f64> {
    for (p, r_phi, _) in modes {
        if *r_phi < 1e-12 {
            continue;
        }
        let proj = dot(&grad, p);
        if proj <= 0.0 {
            continue;
        }
        let two = 2.0 * proj;
        for (g, pk) in grad.iter_mut().zip(p.iter()) {
            *g -= two * *pk;
        }
    }
    grad
}

/// Invert the force component that would walk toward a known well.
///
/// Cartesian fallback: \(g \leftarrow g-2(g\cdot\hat u)\hat u\) when
/// \(g\cdot\hat u>0\) (descent \(-\nabla E\) points at the well).
fn householder_cartesian(wells: &[Well], x: ArrayView1<f64>, mut grad: Array1<f64>) -> Array1<f64> {
    for well in wells {
        if well.coords.len() != x.len() {
            continue;
        }
        let (delta, rmsd) = com_free_delta(x, well.coords.view());
        if rmsd < 1e-12 {
            continue;
        }
        let n_at = (x.len() / 3).max(1) as f64;
        let u_norm = (n_at).sqrt() * rmsd;
        let proj = dot(&grad, &delta) / u_norm;
        if proj <= 0.0 {
            continue;
        }
        let two = 2.0 * proj / u_norm;
        for (g, d) in grad.iter_mut().zip(delta.iter()) {
            *g -= two * *d;
        }
    }
    grad
}

/// Mean of per-center SOAP+ACE \(\nu=3\) at [`PACKING_SPEC`].
fn packing_mean(x: ArrayView1<f64>) -> Option<Array1<f64>> {
    let n_at = x.len() / 3;
    if n_at < MINIMUM_PACKING_ATOMS {
        return None;
    }
    let loc = local_nu3_z(x, PACKING_SPEC, None);
    if loc.nrows() == 0 || loc.ncols() == 0 {
        return None;
    }
    let dim = loc.ncols();
    let mut mu = Array1::<f64>::zeros(dim);
    for i in 0..loc.nrows() {
        for t in 0..dim {
            mu[t] += loc[[i, t]];
        }
    }
    let n = loc.nrows() as f64;
    for item in mu.iter_mut() {
        *item /= n;
    }
    if mu.iter().copied().all(f64::is_finite) {
        Some(mu)
    } else {
        None
    }
}

/// Cartesian packing mode \(P=J_\mu^{\mathsf T}\hat u_\varphi\).
///
/// Returns \((\hat P, \|\mu-\mu_k\|, \|P\|)\) so the hill gradient
/// can reconstruct \(\nabla_x r_\varphi=\|P\|\hat P\).
fn packing_pullback(x: ArrayView1<f64>, mu_k: ArrayView1<f64>) -> Option<(Array1<f64>, f64, f64)> {
    let mu = packing_mean(x)?;
    let j = jacobian_nu3(x, PACKING_SPEC, None);
    lift_packing_mode(x, mu.view(), mu_k, &j)
}

fn lift_packing_mode(
    x: ArrayView1<f64>,
    mu: ArrayView1<f64>,
    mu_k: ArrayView1<f64>,
    j: &Array2<f64>,
) -> Option<(Array1<f64>, f64, f64)> {
    if mu.len() != mu_k.len() {
        return None;
    }
    let mut dmu = &mu - &mu_k;
    let r_phi = dmu.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r_phi < 1e-15 {
        return None;
    }
    for item in dmu.iter_mut() {
        *item /= r_phi;
    }
    let n_at = x.len() / 3;
    let dim = mu.len();
    if j.nrows() != n_at * dim || j.ncols() != x.len() {
        return None;
    }
    let mut p = Array1::<f64>::zeros(x.len());
    let inv_n = 1.0 / (n_at as f64);
    for k in 0..x.len() {
        let mut s = 0.0;
        for i in 0..n_at {
            for t in 0..dim {
                s += j[[i * dim + t, k]] * dmu[t];
            }
        }
        p[k] = s * inv_n;
    }
    strip_com(&mut p);
    let p_norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    if p_norm < 1e-15 {
        return None;
    }
    for item in p.iter_mut() {
        *item /= p_norm;
    }
    Some((p, r_phi, p_norm))
}

fn packing_l2(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

fn strip_com(delta: &mut Array1<f64>) {
    let n_at = delta.len() / 3;
    if n_at == 0 {
        return;
    }
    let mut com = [0.0; 3];
    for i in 0..n_at {
        for ax in 0..3 {
            com[ax] += delta[3 * i + ax];
        }
    }
    let n = n_at as f64;
    for item in &mut com {
        *item /= n;
    }
    for i in 0..n_at {
        for ax in 0..3 {
            delta[3 * i + ax] -= com[ax];
        }
    }
}

fn com_free_delta(x: ArrayView1<f64>, well: ArrayView1<f64>) -> (Array1<f64>, f64) {
    let n_at = x.len() / 3;
    let mut delta = Array1::zeros(x.len());
    let mut com = [0.0; 3];
    for i in 0..n_at {
        for ax in 0..3 {
            let d = x[3 * i + ax] - well[3 * i + ax];
            delta[3 * i + ax] = d;
            com[ax] += d;
        }
    }
    if n_at > 0 {
        for item in &mut com {
            *item /= n_at as f64;
        }
        for i in 0..n_at {
            for ax in 0..3 {
                delta[3 * i + ax] -= com[ax];
            }
        }
    }
    let n = n_at.max(1) as f64;
    let rmsd = (delta.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    (delta, rmsd)
}

fn same_point(left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|(a, b)| (a - b).abs() <= 1e-12)
}

fn dot(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

/// First rung of the Leave ladder, in Cartesian RMSD.
///
/// The old fixed Leave size. It is where the ladder starts, not where it
/// stops: a 0.35 cap holds \(\mu\) inside one packing, and Wales and Doye put
/// the LJ75 ico-Marks barriers at 8.69 and 7.48 \(\varepsilon\), so a quench
/// from 0.35 is a projector back onto the packing it left.
///
/// Wales, D. J.; Doye, J. P. K. *J. Phys. Chem. A* **1997**, *101*, 5111.
/// <https://doi.org/10.1021/jp970984n>
pub const LEAVE_RUNG_RMSD: f64 = 0.35;

/// Ratio between ladder rungs.
pub const LEAVE_RUNG_GROWTH: f64 = 1.5;

/// Rungs a single Leave walks before it reports a refusal.
pub const LEAVE_RUNGS: usize = 8;

/// Rungs walked past the first escape before the best of them is taken.
///
/// The first rung that leaves is not the best one to leave by. Measured from
/// the LJ75 icosahedral minimum, the first escape lands anywhere between 5
/// and 25 \(\varepsilon\) above the floor, and the spread across
/// neighbouring rungs of one direction is most of that range.
pub const LEAVE_RUNG_EXTRA: usize = 2;

/// One rung of the Leave ladder: a packing-map step of Cartesian size
/// `rmsd` along the covering direction `cover_index`, pointed away from the
/// packings on file.
///
/// The direction is a covering point of \(S^{d-1}\) in the DECAF feature,
/// not in Cartesian space: an even covering of the packing map is what a
/// Gaussian kick cannot give, and the pullback through \(J_\mu\) is what
/// makes the increment a packing move rather than a rattle. The component
/// along \(\mu\) is removed, since scaling the mean is a breath of the same
/// packing.
///
/// Plasencia Gutiérrez, M.; Argáez, C.; Jónsson, H. *J. Chem. Theory
/// Comput.* **2017**, *13* (1), 125-134.
/// <https://doi.org/10.1021/acs.jctc.5b01216>
pub fn leave_packing_rung(
    x: ArrayView1<f64>,
    cover_index: usize,
    rmsd: f64,
    references: &[Vec<f64>],
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> Array1<f64> {
    let Some(mu) = packing_mean(x) else {
        return x.to_owned();
    };
    let dim = mu.len();
    if dim == 0 {
        return x.to_owned();
    }
    let n_cover = crate::hypersphere::default_cover_size();
    let mut direction = crate::hypersphere::cover_direction(n_cover, dim, cover_index);
    if direction.len() != dim {
        return x.to_owned();
    }
    let mu_norm2 = mu.iter().map(|v| v * v).sum::<f64>();
    if mu_norm2 > 1e-15 {
        let projection = direction
            .iter()
            .zip(mu.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        for (value, mean) in direction.iter_mut().zip(mu.iter()) {
            *value -= projection * *mean / mu_norm2;
        }
    }
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return x.to_owned();
    }
    for value in &mut direction {
        *value /= norm;
    }
    // Away from the nearest packing on file: the sign of a covering point is
    // arbitrary, and half of them point back at a well the run already holds.
    let mut nearest: Option<(f64, Array1<f64>)> = None;
    for reference in references {
        if reference.len() != x.len() {
            continue;
        }
        let Some(mu_k) = packing_mean(ArrayView1::from(reference.as_slice())) else {
            continue;
        };
        if mu_k.len() != dim {
            continue;
        }
        let distance = mu
            .iter()
            .zip(mu_k.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        if nearest.as_ref().is_none_or(|(held, _)| distance < *held) {
            nearest = Some((distance, mu_k));
        }
    }
    if let Some((distance, mu_k)) = nearest
        && distance > 1e-12
    {
        let away = direction
            .iter()
            .zip(mu.iter().zip(mu_k.iter()))
            .map(|(d, (a, b))| d * (a - b))
            .sum::<f64>();
        if away < 0.0 {
            for value in &mut direction {
                *value = -*value;
            }
        }
    }
    crate::soap::packing_step_nu3(x, PACKING_SPEC, &direction, rmsd, species, mobile)
}

/// Walk the Leave ladder until the quench installs a packing.
///
/// Each rung takes a packing-map step of growing Cartesian size and quenches
/// it; `quench` is the caller's relaxation, and the caller arms
/// [`arm_leave`] around it so the walk is not pulled back into the wells on
/// file. The rung that lands outside every packing on file is the Leave.
/// `None` means the ladder is spent, which is a refusal to report, not a
/// reason to fall back on a leftover-SOAP hole: a hole of the occupied
/// packing quenches into the occupied packing.
pub fn leave_packing_ladder<Q>(
    x: ArrayView1<f64>,
    cover_index: usize,
    references: &[Vec<f64>],
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    rungs: usize,
    mut quench: Q,
) -> Option<(f64, Array1<f64>, usize)>
where
    Q: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let origin = x.as_slice()?;
    let mut rmsd = LEAVE_RUNG_RMSD;
    let mut best: Option<(f64, Array1<f64>, usize)> = None;
    let mut spare = LEAVE_RUNG_EXTRA;
    for rung in 0..rungs.max(1) {
        let start = leave_packing_rung(x, cover_index, rmsd, references, species, mobile);
        let (_, walked) = quench(start.view());
        // The walk ends on a ridge, so what names the packing is the raw
        // minimum below it, not the geometry the walk stopped at. Measured
        // from LJ75 ico: the walk crosses the grain on every rung and the
        // polish returns to the floor it started on.
        let (energy, quenched) = with_disarmed(|| quench(walked.view()));
        if energy.is_finite()
            && let Some(trial) = quenched.as_slice()
            && crate::catalog::leaves_packing(origin, trial, references)
        {
            if best.as_ref().is_none_or(|(held, _, _)| energy < *held) {
                best = Some((energy, quenched, rung));
            }
            if spare == 0 {
                break;
            }
            spare -= 1;
        }
        rmsd *= LEAVE_RUNG_GROWTH;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn unarmed_effective_is_the_raw_surface() {
        let x = Array1::from(vec![1.0, 0.0, 0.0]);
        let g = Array1::from(vec![2.0, 0.0, 0.0]);
        let (e, gt) = effective(x.view(), 3.0, g.clone());
        assert_eq!(e, 3.0);
        assert_eq!(gt, g);
    }

    #[test]
    fn householder_flips_a_gradient_that_points_at_the_well() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        arm_leave(origin.view(), 0.35, &[]);
        // x is to the +x of the well; g points +x, so descent (-g) walks
        // toward the well.
        let x = Array1::from(vec![0.3, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let g = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (_, gt) = effective(x.view(), 1.0, g);
        assert!(
            gt[0] < 0.0,
            "transformed gradient must point away from the well, g0={}",
            gt[0]
        );
        disarm();
    }

    #[test]
    fn hill_raises_energy_near_a_known_well() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        arm_leave(origin.view(), 0.35, &[]);
        let x = Array1::from(vec![0.3, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let g = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (e, _) = effective(x.view(), 0.0, g);
        assert!(e > 0.0, "Gaussian hill must raise E, got {e}");
        disarm();
    }

    #[test]
    fn xtsci_on_the_transformed_surface_walks_away_from_the_well() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        arm_leave(origin.view(), 0.35, &[]);
        let start = Array1::from(vec![0.3, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
        let (_, x) = step_xtsci(&mut opt, start.view(), 40, |v| {
            // Harmonic well at `origin`. Raw descent walks back.
            let mut g = Array1::zeros(v.len());
            let mut e = 0.0;
            for i in 0..v.len() {
                let d = v[i] - origin[i];
                e += 0.5 * d * d;
                g[i] = d;
            }
            Some(effective(v, e, g))
        });
        let (_, start_rmsd) = com_free_delta(start.view(), origin.view());
        let (_, end_rmsd) = com_free_delta(x.view(), origin.view());
        disarm();
        assert!(
            end_rmsd > start_rmsd,
            "xtsci must walk away from the known well, start={start_rmsd} end={end_rmsd}"
        );
    }

    #[test]
    fn disarm_restores_the_raw_surface() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0]);
        arm_leave(origin.view(), 0.35, &[]);
        disarm();
        let x = Array1::from(vec![0.2, 0.0, 0.0]);
        let g = Array1::from(vec![1.0, 0.0, 0.0]);
        let (e, gt) = effective(x.view(), 4.0, g.clone());
        assert_eq!(e, 4.0);
        assert_eq!(gt, g);
    }

    fn ico13() -> Array1<f64> {
        let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let verts: [[f64; 3]; 12] = [
            [0.0, 1.0, p],
            [0.0, 1.0, -p],
            [0.0, -1.0, p],
            [0.0, -1.0, -p],
            [1.0, p, 0.0],
            [1.0, -p, 0.0],
            [-1.0, p, 0.0],
            [-1.0, -p, 0.0],
            [p, 0.0, 1.0],
            [-p, 0.0, 1.0],
            [p, 0.0, -1.0],
            [-p, 0.0, -1.0],
        ];
        let s = 1.0 / (1.0 + p * p).sqrt();
        let mut x = Array1::<f64>::zeros(3 * 13);
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                x[3 * (i + 1) + k] = s * v[k];
            }
        }
        x
    }

    #[test]
    fn span_rises_when_the_packing_mean_leaves_mu_k() {
        let origin = ico13();
        arm_leave(origin.view(), 0.35, &[]);
        let at_well = span(origin.view());
        let mut away = origin.clone();
        away[3] += 0.25;
        let off = span(away.view());
        disarm();
        assert!(
            off > at_well,
            "span must rise away from mu_k, well={at_well} off={off}"
        );
    }

    #[test]
    fn packing_mode_is_nu3_mean_not_leftover_soap() {
        let origin = ico13();
        let mu = packing_mean(origin.view()).expect("ico13 packing mean");
        let want = PACKING_SPEC.feat_dim(None) + PACKING_SPEC.nu3_feat_dim(None);
        assert_eq!(
            mu.len(),
            want,
            "packing invert must live in stacked SOAP+ν=3, not leftover SOAP"
        );
    }

    #[test]
    fn packing_householder_flips_force_along_the_nu3_pullback() {
        let origin = ico13();
        arm_leave(origin.view(), 0.35, &[]);
        let mut x = origin.clone();
        x[3] += 0.25;
        let mu0 = packing_mean(origin.view()).expect("origin packing mean");
        let (p, r_phi, _) = packing_pullback(x.view(), mu0.view()).expect("packing mode");
        assert!(r_phi > 0.0, "displaced ico13 must leave the origin μ");
        let (_, gt) = effective(x.view(), 1.0, p.clone());
        let proj = gt.iter().zip(p.iter()).map(|(a, b)| a * b).sum::<f64>();
        disarm();
        assert!(proj < 0.0, "packing Householder must flip g·P, got {proj}");
    }

    #[test]
    fn xtsci_walks_off_the_known_packing() {
        let origin = ico13();
        arm_leave(origin.view(), 0.35, &[]);
        let mut start = origin.clone();
        start[3] += 0.20;
        let mu0 = packing_mean(origin.view()).expect("origin packing mean");
        let start_span = packing_mean(start.view())
            .map(|mu| packing_l2(mu.view(), mu0.view()))
            .expect("start packing mean");
        let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
        let (_, x) = step_xtsci(&mut opt, start.view(), 16, |v| {
            let mut g = Array1::zeros(v.len());
            let mut e = 0.0;
            for i in 0..v.len() {
                let d = v[i] - origin[i];
                e += 0.5 * d * d;
                g[i] = d;
            }
            Some(effective(v, e, g))
        });
        let end_span = packing_mean(x.view())
            .map(|mu| packing_l2(mu.view(), mu0.view()))
            .unwrap_or(0.0);
        disarm();
        assert!(
            end_span > start_span,
            "xtsci must walk off the known packing, start={start_span} end={end_span}"
        );
    }
}
