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
    lift: Option<f64>,
    sigma_phi: Option<f64>,
}

thread_local! {
    static ARMED: RefCell<Option<Armed>> = const { RefCell::new(None) };
}

/// Arm the transformed quench for one occupancy Leave.
///
/// `origin` is the live well this extra is leaving. Previous chains
/// enter through the packing archive when featomic is on.
pub fn arm_leave(origin: ArrayView1<f64>, sigma_rmsd: f64) {
    #[allow(unused_mut)]
    let mut wells = vec![Well {
        packing_mean: packing_mean(origin),
        coords: origin.to_owned(),
    }];
    #[cfg(feature = "featomic")]
    {
        for well in crate::featomic_hop::packing_archive() {
            if well.len() == origin.len() && !same_point(well.view(), origin) {
                wells.push(Well {
                    packing_mean: packing_mean(well.view()),
                    coords: well,
                });
            }
        }
    }
    ARMED.with(|slot| {
        *slot.borrow_mut() = Some(Armed {
            wells,
            sigma_rmsd: sigma_rmsd.max(1e-6),
            lift: None,
            sigma_phi: None,
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

/// xtsci step on the transformed surface: two-loop direction, accept a
/// step that increases the span from the known wells. Span is packing
/// L2 \(\min_k\|\mu-\mu_k\|\) when the wells carry a \(\nu=3\) mean,
/// otherwise COM-free RMSD. Raw \(E\) may rise; that is the dimer
/// walk away from an occupied packing.
pub fn step_xtsci<F>(
    opt: &mut crate::methods::warm_lbfgs::WarmLbfgs,
    x0: ArrayView1<f64>,
    max_iter: usize,
    mut fg: F,
) -> (f64, Array1<f64>)
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    let sigma = ARMED.with(|slot| slot.borrow().as_ref().map(|a| a.sigma_rmsd).unwrap_or(0.35));
    let mut x = x0.to_owned();
    let Some((mut energy, mut grad)) = fg(x.view()) else {
        return (f64::INFINITY, x);
    };
    for _ in 0..max_iter {
        let mut direction = opt.two_loop(grad.view());
        if direction.dot(&grad) >= 0.0 {
            direction = grad.mapv(|v| -v);
        }
        let n_at = (x.len() / 3).max(1) as f64;
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
            if trial_span > start_span || te < energy {
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
    }
    (energy, x)
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
        transform(armed, x, energy, grad)
    })
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
    let j = jacobian_nu3(x, PACKING_SPEC, None);
    let modes: Vec<(Array1<f64>, f64, f64)> = armed
        .wells
        .iter()
        .filter_map(|well| {
            let mu_k = well.packing_mean.as_ref()?;
            lift_packing_mode(x, mu.view(), mu_k.view(), &j)
        })
        .collect();
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

#[cfg(feature = "featomic")]
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
        arm_leave(origin.view(), 0.35);
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
        arm_leave(origin.view(), 0.35);
        let x = Array1::from(vec![0.3, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let g = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (e, _) = effective(x.view(), 0.0, g);
        assert!(e > 0.0, "Gaussian hill must raise E, got {e}");
        disarm();
    }

    #[test]
    fn xtsci_on_the_transformed_surface_walks_away_from_the_well() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        arm_leave(origin.view(), 0.35);
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
        arm_leave(origin.view(), 0.35);
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
        arm_leave(origin.view(), 0.35);
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
        arm_leave(origin.view(), 0.35);
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
