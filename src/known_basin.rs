//! Invert basins already occupied so xtsci will not walk back into them.
//!
//! Henkelman and Jónsson, *J. Chem. Phys.* **1999**, *111*, 7010
//! <https://doi.org/10.1063/1.480097>: the dimer replaces \(F\) by
//! \(F-2(F\cdot P)P\) so a first-order stepper walks *up* the lowest
//! mode to a saddle. Occupancy Leave applies the same Householder to
//! the radial directions of known wells: this replica's origin and the
//! shared archive of previous chains. Those wells are no longer
//! attractors of the xtsci L-BFGS step.
//!
//! The Householder on \(g\) is not conservative. The line search is
//! run on the matching PES \(E+V\), a Gaussian hill on each known
//! well. At a well the hill Hessian is \(-A/(N\sigma^2)I\). \(A\) is
//! the measured covering lift \((g\cdot\hat u)\,r\) at the Leave
//! start, \(\sigma\) is the covering RMSD cap. After a transformed
//! quench that changes DECAF family, a raw-\(E\) polish sits on a
//! true minimum of that well.

use std::cell::RefCell;

use ndarray::{Array1, ArrayView1};

struct Well {
    coords: Array1<f64>,
}

struct Armed {
    wells: Vec<Well>,
    sigma_rmsd: f64,
    lift: Option<f64>,
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
        coords: origin.to_owned(),
    }];
    #[cfg(feature = "featomic")]
    {
        for well in crate::featomic_hop::packing_archive() {
            if well.len() == origin.len() && !same_point(well.view(), origin) {
                wells.push(Well { coords: well });
            }
        }
    }
    ARMED.with(|slot| {
        *slot.borrow_mut() = Some(Armed {
            wells,
            sigma_rmsd: sigma_rmsd.max(1e-6),
            lift: None,
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
/// step that increases COM-free RMSD from the known wells. Raw \(E\)
/// may rise; that is the dimer walk away from an occupied min.
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
    let mut grad = householder(&armed.wells, x, grad);
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

/// Invert the force component that would walk toward a known well.
///
/// Henkelman–Jónsson: \(g \leftarrow g-2(g\cdot\hat u)\hat u\) when
/// \(g\cdot\hat u>0\) (descent \(-\nabla E\) points at the well).
fn householder(wells: &[Well], x: ArrayView1<f64>, mut grad: Array1<f64>) -> Array1<f64> {
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
}
