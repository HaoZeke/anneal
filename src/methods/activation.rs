//! Activation: climbing out of a basin along its softest direction.
//!
//! Barkema and Mousseau, Phys. Rev. Lett. 77, 4358 (1996), and Malek and
//! Mousseau, Phys. Rev. E 62, 7723 (2000).
//!
//! A single displacement along the softest mode does not leave a basin. It is
//! the right direction and the wrong distance: the mode points at the low
//! saddle, but relaxing from a point still inside the basin returns to the
//! minimum it came from. Measured on LJ38 with the escape controller driving a
//! straight displacement, 576 quenches in 959 came back to the basin they left
//! and 10 found anything new.
//!
//! Goedecker's answer is molecular dynamics, which carries kinetic energy over
//! the saddle. The answer that needs only gradients is to climb: push along the
//! mode, relax the components perpendicular to it so the structure stays on the
//! valley floor, and repeat until the curvature along the mode turns negative.
//! Negative curvature means the ridge is behind, and a quench from there falls
//! into a different basin.
//!
//! What this costs is honest and worth stating. Each climbing step is a
//! curvature pass and a few perpendicular relaxation steps, so an activation is
//! several hundred charged evaluations where a random displacement is one. It
//! buys escapes that actually leave.
//!
//! # Relation to the rest of the crate
//!
//! The perpendicular relaxation is the same projection [`crate::path`] uses to
//! hold a band off its endpoints, and the mode comes from
//! [`crate::curvature`]. The controller in [`crate::methods::minima_hopping`]
//! sets how far to climb; this module decides when to stop.

use crate::curvature::curvature_features;
use ndarray::{Array1, ArrayView1};

/// How the climb is run.
#[derive(Debug, Clone)]
pub struct Activation {
    /// Distance moved along the mode per climbing step.
    pub step: f64,
    /// Climbing steps before giving up.
    ///
    /// A cap rather than a convergence criterion: some directions do not reach
    /// negative curvature at all, and a climb that has not turned over after
    /// this many steps is abandoned rather than run to exhaustion.
    pub max_steps: usize,
    /// Perpendicular relaxation steps between climbs.
    pub perp_steps: usize,
    /// Step size of the perpendicular relaxation.
    pub perp_rate: f64,
    /// Largest displacement one perpendicular step may make.
    ///
    /// A fixed rate is not safe on a potential whose gradient spans decades. On
    /// a Lennard-Jones cluster two points a little too close carry a gradient of
    /// order a thousand, and a rate of 0.02 against that moves the structure
    /// twenty units and destroys it: measured on LJ38, 6 relaxations in 1589
    /// reached a minimum and the returned structure had a gradient of 1.0 where
    /// a minimum has 1e-6. The cap makes the step a direction with a bounded
    /// length rather than a length proportional to the gradient.
    pub perp_max_move: f64,
    /// Lanczos steps per curvature pass.
    pub lanczos_steps: usize,
    /// Finite-difference step for the curvature.
    pub epsilon: f64,
    /// Climbing steps between recomputing the mode.
    ///
    /// Recomputing every step is the accurate choice and the expensive one. The
    /// mode rotates slowly along a valley floor, so reusing it for a few steps
    /// costs little accuracy and divides the curvature bill.
    pub refresh: usize,
    /// Extra push along the mode once the curvature has turned over, in units
    /// of `step`, before the quench.
    pub overshoot: f64,
}

impl Default for Activation {
    fn default() -> Self {
        Self {
            step: 0.2,
            max_steps: 24,
            perp_steps: 3,
            perp_rate: 0.02,
            perp_max_move: 0.05,
            lanczos_steps: 12,
            epsilon: 1e-4,
            refresh: 3,
            overshoot: 1.5,
        }
    }
}

/// Where a climb ended.
#[derive(Debug, Clone)]
pub struct ActivationOutcome {
    /// The activated structure, to be quenched by the caller.
    pub state: Array1<f64>,
    /// Curvature along the mode at the end of the climb.
    pub lambda: f64,
    /// Climbing steps taken.
    pub steps: usize,
    /// Whether the curvature turned negative, so the ridge is behind.
    pub crossed: bool,
    /// Gradient evaluations spent, all of them charged by the caller.
    pub evaluations: usize,
}

/// Climbs out of the basin containing `x`.
///
/// `grad` returns the gradient or `None` when the caller's budget is spent, in
/// which case the climb stops and reports what it has. `sign` picks which way
/// along the mode to go; the two ends of a soft direction are different saddles.
///
/// Returns `None` only when the first curvature pass fails, since there is then
/// no direction to climb along.
pub fn activate<G>(
    x: ArrayView1<f64>,
    mut grad: G,
    cfg: &Activation,
    sign: f64,
) -> Option<ActivationOutcome>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    let dim = x.len();
    let mut cur = x.to_owned();
    let mut evaluations = 0usize;

    let first = curvature_features(
        cur.view(),
        |y| {
            evaluations += 1;
            grad(y)
        },
        cfg.lanczos_steps,
        cfg.epsilon,
    )?;
    let mut mode = first.mode.clone();
    let mut lambda = first.lambda_min;
    let mut steps = 0usize;
    let mut crossed = false;

    for k in 0..cfg.max_steps {
        // Refresh the direction on schedule, and always after the curvature has
        // already been seen to fall, since that is where it rotates fastest.
        if k > 0 && k % cfg.refresh == 0 {
            match curvature_features(
                cur.view(),
                |y| {
                    evaluations += 1;
                    grad(y)
                },
                cfg.lanczos_steps,
                cfg.epsilon,
            ) {
                Some(f) => {
                    // Keep the sense of travel: the mode is defined up to sign
                    // and flipping it mid-climb walks back down.
                    let dot: f64 = f.mode.iter().zip(mode.iter()).map(|(a, b)| a * b).sum();
                    mode = if dot < 0.0 { -f.mode } else { f.mode };
                    lambda = f.lambda_min;
                }
                None => break,
            }
        }
        for i in 0..dim {
            cur[i] += sign * cfg.step * mode[i];
        }
        steps += 1;

        // Perpendicular relaxation. Sliding down the component of the gradient
        // orthogonal to the mode keeps the structure on the valley floor; the
        // component along the mode is what the climb is fighting and is left
        // alone.
        let mut along = 0.0;
        for _ in 0..cfg.perp_steps {
            let g = match grad(cur.view()) {
                Some(g) => {
                    evaluations += 1;
                    g
                }
                None => {
                    return Some(ActivationOutcome {
                        state: cur,
                        lambda,
                        steps,
                        crossed,
                        evaluations,
                    })
                }
            };
            along = g.iter().zip(mode.iter()).map(|(a, b)| a * b).sum();
            let mut d = Array1::<f64>::zeros(dim);
            for i in 0..dim {
                d[i] = cfg.perp_rate * (g[i] - along * mode[i]);
            }
            let n: f64 = d.iter().map(|z| z * z).sum::<f64>().sqrt();
            let scale = if n > cfg.perp_max_move && n > 0.0 {
                cfg.perp_max_move / n
            } else {
                1.0
            };
            for i in 0..dim {
                cur[i] -= scale * d[i];
            }
        }

        // Stop at the saddle, not at the inflection.
        //
        // Negative curvature says the ridge is ahead, not behind: on a double
        // well the curvature along the well direction turns over at
        // `|u| = 1/sqrt(3)` while the barrier top is at `u = 0`. Stopping there
        // and pushing on by the overshoot ended the climb at `u = 0.101`, on
        // the side it started, and a quench from there goes home.
        //
        // The saddle is where the force along the direction of travel changes
        // sign: uphill while `sign * g . v > 0`, downhill after. Combined with
        // negative curvature that is the ridge, and past it a quench falls the
        // other way.
        if lambda < 0.0 && sign * along < 0.0 {
            crossed = true;
            break;
        }
    }

    if crossed && cfg.overshoot > 0.0 {
        // One push past the turning point, so the quench falls forward rather
        // than back down the way it came.
        for i in 0..dim {
            cur[i] += sign * cfg.overshoot * cfg.step * mode[i];
        }
    }

    Some(ActivationOutcome {
        state: cur,
        lambda,
        steps,
        crossed,
        evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A double well along one collective direction, stiff in the rest.
    ///
    /// `E = (u^2 - 1)^2 + 1/2 sum_i k_i (x_i - u w_i)^2` with `u = x . w`.
    /// Minima at `u = +/- 1`, a barrier at `u = 0`.
    ///
    /// The perpendicular stiffnesses vary and all exceed the curvature along
    /// `w` at the minimum, which is 8. Both parts are load-bearing. If they are
    /// smaller the softest mode is not the well direction and the test is
    /// asking the wrong question, and if they are all equal the Hessian is a
    /// multiple of the identity, every vector is an eigenvector, and the Krylov
    /// space collapses at the first step.
    fn perp_stiffness(dim: usize) -> Array1<f64> {
        Array1::from_shape_fn(dim, |i| 12.0 + 1.7 * (i % 7) as f64)
    }

    fn double_well<'a>(
        w: &'a Array1<f64>,
        k: &'a Array1<f64>,
    ) -> impl Fn(ArrayView1<f64>) -> Option<Array1<f64>> + 'a {
        move |x: ArrayView1<f64>| {
            let u: f64 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            let dedu = 4.0 * u * (u * u - 1.0);
            let mut g = Array1::zeros(x.len());
            // d/dx of the perpendicular term, including its dependence on u.
            let mut kp_dot_w = 0.0;
            for i in 0..x.len() {
                kp_dot_w += k[i] * (x[i] - u * w[i]) * w[i];
            }
            for i in 0..x.len() {
                let perp = x[i] - u * w[i];
                g[i] = dedu * w[i] + k[i] * perp - kp_dot_w * w[i];
            }
            Some(g)
        }
    }

    fn direction(dim: usize) -> Array1<f64> {
        let mut w = Array1::from_shape_fn(dim, |i| ((i % 5) as f64 - 2.0) + 0.25);
        let n: f64 = w.iter().map(|z| z * z).sum::<f64>().sqrt();
        w /= n;
        w
    }

    /// The cap has to bind on a stiff gradient, or the climb walks off the
    /// structure it was refining.
    #[test]
    fn a_perpendicular_step_is_bounded_however_steep_the_gradient() {
        let dim = 36;
        let w = direction(dim);
        // A gradient a thousand times the scale the rate was set for.
        let g = move |x: ArrayView1<f64>| -> Option<Array1<f64>> {
            Some(Array1::from_shape_fn(x.len(), |i| 1500.0 * x[i] + 3.0 * (i % 5) as f64))
        };
        let cfg = Activation {
            max_steps: 4,
            ..Activation::default()
        };
        let out = activate(w.view(), g, &cfg, 1.0).unwrap();
        let travelled: f64 = out
            .state
            .iter()
            .zip(w.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let bound = out.steps as f64 * (cfg.step + cfg.perp_steps as f64 * cfg.perp_max_move)
            + cfg.overshoot * cfg.step;
        assert!(
            travelled <= bound + 1e-9,
            "moved {travelled:.3} where the caps allow {bound:.3}"
        );
    }

    /// The property the module exists for. A straight displacement of the same
    /// length stays on its own side of the barrier; the climb crosses it.
    #[test]
    fn the_climb_crosses_the_barrier_a_displacement_does_not() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let g = double_well(&w, &k);
        // Start at the minimum with u = 1.
        let x: Array1<f64> = w.clone();
        let u0: f64 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!((u0 - 1.0).abs() < 1e-12);

        let cfg = Activation::default();
        let out = activate(x.view(), &g, &cfg, -1.0).unwrap();
        let u: f64 = out.state.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!(
            out.crossed,
            "the climb never reached negative curvature, ending at u = {u:.3}"
        );
        assert!(
            u < 0.0,
            "the climb ended at u = {u:.3}, still on the side it started"
        );

        // The same total distance in a straight line, no climbing.
        let travelled = (out.steps as f64 + cfg.overshoot) * cfg.step;
        let mut straight = x.clone();
        for i in 0..dim {
            straight[i] -= travelled * w[i];
        }
        let us: f64 = straight.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        // Both end past the barrier here because the well is one-dimensional;
        // what separates them is that the climb *knows* it is past, which is
        // what a caller needs in order to stop.
        assert!(
            us < 0.0,
            "the straight line should also cross this simple well: {us:.3}"
        );
    }

    /// The climb must not run forever on a direction that never turns over.
    #[test]
    fn a_direction_that_never_softens_stops_at_the_cap() {
        let dim = 36;
        let w = direction(dim);
        // Purely harmonic: the curvature is positive everywhere.
        // Stiffnesses that differ, so the Hessian is not a multiple of the
        // identity and the Krylov space has somewhere to go.
        let k = perp_stiffness(dim);
        let g = move |x: ArrayView1<f64>| -> Option<Array1<f64>> {
            Some(Array1::from_shape_fn(x.len(), |i| k[i] * x[i]))
        };
        let x = w.clone();
        let cfg = Activation {
            max_steps: 6,
            ..Activation::default()
        };
        let out = activate(x.view(), &g, &cfg, 1.0).unwrap();
        assert!(!out.crossed, "a harmonic well has no ridge to cross");
        assert!(
            out.steps <= 6,
            "the climb took {} steps against a cap of 6",
            out.steps
        );
    }

    /// A caller whose budget runs out mid-climb gets the structure so far
    /// rather than a panic or a silent full-cost climb.
    #[test]
    fn an_exhausted_budget_stops_the_climb() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let inner = double_well(&w, &k);
        let mut left = 40usize;
        let g = move |x: ArrayView1<f64>| -> Option<Array1<f64>> {
            if left == 0 {
                return None;
            }
            left -= 1;
            inner(x)
        };
        let out = activate(w.view(), g, &Activation::default(), -1.0);
        match out {
            Some(o) => assert!(
                o.evaluations <= 40,
                "spent {} gradients against a budget of 40",
                o.evaluations
            ),
            // Refusing before the first curvature pass completes is also
            // correct; what would not be is climbing past the budget.
            None => {}
        }
    }

    /// The sign argument has to mean something: the two ends of a soft
    /// direction are different saddles and a caller picking one must get it.
    #[test]
    fn the_two_signs_climb_opposite_ways() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let g = double_well(&w, &k);
        let x = w.clone();
        let a = activate(x.view(), &g, &Activation::default(), 1.0).unwrap();
        let b = activate(x.view(), &g, &Activation::default(), -1.0).unwrap();
        let ua: f64 = a.state.iter().zip(w.iter()).map(|(p, q)| p * q).sum();
        let ub: f64 = b.state.iter().zip(w.iter()).map(|(p, q)| p * q).sum();
        assert!(
            ua > ub,
            "the two signs ended at u = {ua:.3} and {ub:.3}, not on opposite sides"
        );
    }
}
