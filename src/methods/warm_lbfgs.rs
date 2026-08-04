//! Limited-memory quasi-Newton relaxation whose curvature survives between calls.
//!
//! A hopping chain relaxes thousands of times, each from a perturbation of an
//! already-relaxed structure, so the curvature at a new start resembles the
//! curvature at the old minimum and a minimiser that forgets between calls pays
//! to rediscover it. Since a work ledger charges evaluations rather than
//! seconds, cutting evaluations per relaxation multiplies the hops a budget
//! buys.
//!
//! # Measured status
//!
//! On a 75-point Lennard-Jones cluster, 400 relaxations from perturbed minima
//! with overlapping pairs repaired first:
//!
//! | arm  | evals per relaxation | worst final `|g|` |
//! |------|---------------------|-------------------|
//! | cold | 4610                | 7.6               |
//! | warm | 3389                | 9.7               |
//!
//! Retaining curvature costs 1.36 times fewer evaluations, but both arms stop
//! against the iteration cap with gradients around 8 rather than the 1e-6
//! tolerance, and a reference L-BFGS-B relaxes the same problem in about 270
//! evaluations. This implementation is therefore an order of magnitude off a
//! production minimiser, and the warm-to-cold ratio compares two arms that both
//! fail to converge. Treat it as unvalidated on real potentials: the quadratic
//! tests below are the only supported claim.
//!
//! Two facts from building it do transfer. Backtracking that enforces only
//! sufficient decrease produces curvature pairs describing curvature never
//! measured, and since every direction is built from the stored pairs, one bad
//! pair degrades the whole memory: with Armijo alone, retaining curvature was
//! 1.8 times worse than discarding it, and the strong Wolfe conditions reverse
//! the sign. And a relaxation benchmark has to check the final gradient, since
//! an unrepaired overlap returns a gradient near 1e13 that no line search
//! recovers from, making a cost comparison count failures instead of work.

use ndarray::{Array1, ArrayView1};

/// Stored curvature pair from one accepted step.
struct Pair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
}

/// L-BFGS with memory that persists across relaxations.
pub struct WarmLbfgs {
    memory: Vec<Pair>,
    /// Pairs retained; the usual choice is between five and ten.
    pub max_pairs: usize,
    /// Gradient infinity norm below which a relaxation is converged.
    pub gtol: f64,
    /// Armijo sufficient-decrease constant, `c1` in the Wolfe conditions.
    pub armijo: f64,
    /// Curvature constant, `c2`. The usual choice for quasi-Newton is 0.9.
    pub curvature: f64,
    /// Line-search evaluations attempted before the direction is abandoned.
    pub max_line_evals: usize,
}

impl Default for WarmLbfgs {
    fn default() -> Self {
        Self {
            memory: Vec::new(),
            max_pairs: 8,
            gtol: 1e-6,
            armijo: 1e-4,
            curvature: 0.9,
            max_line_evals: 20,
        }
    }
}

impl WarmLbfgs {
    /// Discards the stored curvature.
    ///
    /// Called when the chain moves somewhere structurally different, where the
    /// retained pairs describe a Hessian that no longer applies.
    pub fn forget(&mut self) {
        self.memory.clear();
    }

    /// Pairs currently held.
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// True when no curvature is stored.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Two-loop recursion: applies the inverse-Hessian approximation to `g`.
    fn direction(&self, g: ArrayView1<f64>) -> Array1<f64> {
        let mut q = g.to_owned();
        let m = self.memory.len();
        let mut alpha = vec![0.0; m];
        for (i, p) in self.memory.iter().enumerate().rev() {
            let a = p.rho * p.s.dot(&q);
            alpha[i] = a;
            q.scaled_add(-a, &p.y);
        }
        // Scale by the most recent pair's curvature, which is what makes the
        // first step of a warm start the right size rather than a guess.
        if let Some(p) = self.memory.last() {
            let yy = p.y.dot(&p.y);
            if yy > 0.0 {
                q *= p.s.dot(&p.y) / yy;
            }
        }
        for (i, p) in self.memory.iter().enumerate() {
            let b = p.rho * p.y.dot(&q);
            q.scaled_add(alpha[i] - b, &p.s);
        }
        q.mapv_inplace(|v| -v);
        q
    }

    fn push(&mut self, s: Array1<f64>, y: Array1<f64>) {
        let sy = s.dot(&y);
        // Curvature condition: a non-positive pair would make the
        // approximation indefinite and the direction an ascent one.
        if sy <= 1e-12 {
            return;
        }
        self.memory.push(Pair { s, y, rho: 1.0 / sy });
        if self.memory.len() > self.max_pairs {
            self.memory.remove(0);
        }
    }


    /// Strong Wolfe line search by bracketing then cubic-interpolated zoom.
    ///
    /// Nocedal and Wright algorithms 3.5 and 3.6. Returns whether a step was
    /// accepted and how many evaluations it cost. On acceptance the point,
    /// value and gradient are advanced and the curvature pair is stored.
    fn line_search<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        d: &Array1<f64>,
        slope: f64,
        fg: &mut F,
    ) -> (bool, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let f0 = *f;
        let mut evals = 0usize;
        let mut probe = |a: f64, fg: &mut F, evals: &mut usize| {
            let mut t = x.clone();
            t.scaled_add(a, d);
            let r = fg(t.view());
            if r.is_some() {
                *evals += 1;
            }
            r
        };

        let mut a_prev = 0.0;
        let mut f_prev = f0;
        // The first trial step is scaled by the direction's size. Starting at
        // one is right for a well-scaled problem and useless where the
        // gradient is enormous: a Lennard-Jones configuration with two points
        // nearly coincident carries a gradient near 1e13, twenty backtracks
        // from one reach only 1e-6, the search fails, and the relaxation
        // returns its starting point.
        //
        // Measured against a reference minimiser on identical inputs, that is
        // exactly what happened: on trials the reference relaxed to -146.7 and
        // -168.0, this returned 3.8e8 and -4.1, having never left the start.
        // It is the difference between a weak minimiser and one that does not
        // run.
        let dnorm = d.iter().fold(0.0_f64, |acc, v| acc + v * v).sqrt();
        let mut a = if dnorm > 1.0 { 1.0 / dnorm } else { 1.0 };
        let mut lo = 0.0;
        let mut f_lo = f0;
        let mut slope_lo = slope;
        let mut hi = f64::NAN;
        let mut bracketed = false;

        for i in 0..self.max_line_evals {
            let (fa, ga) = match probe(a, fg, &mut evals) {
                Some(v) => v,
                None => return (false, evals),
            };
            let slope_a = d.dot(&ga);
            if fa > f0 + self.armijo * a * slope || (i > 0 && fa >= f_prev) {
                lo = a_prev;
                f_lo = f_prev;
                hi = a;
                bracketed = true;
                break;
            }
            if slope_a.abs() <= -self.curvature * slope {
                // Both Wolfe conditions hold: accept without zooming.
                self.accept(x, f, g, d, a, fa, ga);
                return (true, evals);
            }
            if slope_a >= 0.0 {
                lo = a;
                f_lo = fa;
                slope_lo = slope_a;
                hi = a_prev;
                bracketed = true;
                break;
            }
            a_prev = a;
            f_prev = fa;
            a *= 2.0;
        }

        if !bracketed {
            return (false, evals);
        }

        while evals < self.max_line_evals {
            // Quadratic interpolant of the low end, clamped away from the
            // bracket edges so the interval keeps shrinking.
            let width = hi - lo;
            let mut trial = lo + 0.5 * width;
            let denom = 2.0 * (f_lo - f_prev + slope_lo * width);
            if denom.abs() > 1e-16 {
                let q = lo + slope_lo * width * width / denom;
                if (q - lo) / width > 0.1 && (q - lo) / width < 0.9 {
                    trial = q;
                }
            }
            let (ft, gt) = match probe(trial, fg, &mut evals) {
                Some(v) => v,
                None => return (false, evals),
            };
            let slope_t = d.dot(&gt);
            if ft > f0 + self.armijo * trial * slope || ft >= f_lo {
                hi = trial;
                f_prev = ft;
            } else {
                if slope_t.abs() <= -self.curvature * slope {
                    self.accept(x, f, g, d, trial, ft, gt);
                    return (true, evals);
                }
                if slope_t * (hi - lo) >= 0.0 {
                    hi = lo;
                }
                lo = trial;
                f_lo = ft;
                slope_lo = slope_t;
            }
            if (hi - lo).abs() < 1e-14 {
                break;
            }
        }
        (false, evals)
    }

    /// Advances the iterate and records the curvature pair.
    fn accept(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        d: &Array1<f64>,
        step: f64,
        f_new: f64,
        g_new: Array1<f64>,
    ) {
        let mut s = d.clone();
        s *= step;
        let mut y = g_new.clone();
        y -= &*g;
        self.push(s, y);
        x.scaled_add(step, d);
        *f = f_new;
        *g = g_new;
    }

    /// Relaxes `x0`, calling `fg` for value and gradient.
    ///
    /// `fg` returns `None` when the caller's budget is spent, which ends the
    /// relaxation where it stands. Returns the value, the point, and the number
    /// of evaluations used.
    pub fn minimize<F>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        fg: F,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        self.minimize_watched(x0, max_iter, fg, |_, _| true)
    }

    /// Relaxes `x0`, offering each accepted iterate to `watch`.
    ///
    /// `watch` receives the iteration index and the value at that iterate, and
    /// returning `false` ends the relaxation there. The point is a caller that
    /// stops on a decision rather than on an iteration count: a screening pass
    /// exists to answer one question, and the trajectory it produces says when
    /// the answer is settled.
    ///
    /// The hook sits at the top of the iteration, not inside the line search,
    /// so what it sees is always an accepted point with its value and gradient
    /// consistent. Stopping mid-search would return a trial step the optimizer
    /// had not adopted.
    pub fn minimize_watched<F, W>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        mut fg: F,
        mut watch: W,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        W: FnMut(usize, f64) -> bool,
    {
        let mut x = x0.to_owned();
        let mut evals = 0usize;
        let (mut f, mut g) = match fg(x.view()) {
            Some(v) => v,
            None => return (f64::INFINITY, x, evals),
        };
        evals += 1;

        for it in 0..max_iter {
            if !watch(it, f) {
                break;
            }
            if g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < self.gtol {
                break;
            }
            let d = self.direction(g.view());
            let slope = d.dot(&g);
            if slope >= 0.0 {
                // A stored pair has gone stale: the direction points uphill.
                self.forget();
                continue;
            }
            // Strong Wolfe rather than plain backtracking. Armijo alone accepts
            // a step that decreases the value without saying anything about the
            // gradient, and the pair such a step contributes describes curvature
            // that was never measured. Since every later direction is built from
            // the stored pairs, one bad pair degrades the whole memory, which is
            // why a chain carrying its curvature forward needs the curvature
            // condition and not only the decrease condition.
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, &d, slope, &mut fg);
            evals += evals_used;
            let moved = ok;
            if !moved {
                // A failed line search along a quasi-Newton direction means the
                // stored curvature no longer describes this region, not that the
                // point is converged. Dropping the memory falls back to steepest
                // descent, which always has a usable direction. Only give up
                // when that fails too, since then the failure is the geometry
                // and not the approximation.
                if self.memory.is_empty() {
                    break;
                }
                self.forget();
            }
        }
        (f, x, evals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// The watched form has to agree with the plain one when the hook never
    /// stops it, or every measurement taken through one does not describe the
    /// other.
    #[test]
    fn watching_without_stopping_changes_nothing() {
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut a = WarmLbfgs::default();
        let (fa, xa, ea) = a.minimize(x0.view(), 50, |v| Some(quad(v)));
        let mut b = WarmLbfgs::default();
        let (fb, xb, eb) = b.minimize_watched(x0.view(), 50, |v| Some(quad(v)), |_, _| true);
        assert_eq!(ea, eb);
        assert_eq!(fa, fb);
        assert_eq!(xa, xb);
    }

    /// And it has to actually stop, at an accepted point rather than a trial
    /// step: the value handed back must be the one the hook last saw.
    #[test]
    fn a_hook_that_refuses_stops_at_an_accepted_point() {
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut opt = WarmLbfgs::default();
        let mut seen = Vec::new();
        let (f, _, _) = opt.minimize_watched(
            x0.view(),
            50,
            |v| Some(quad(v)),
            |it, fv| {
                seen.push(fv);
                it < 3
            },
        );
        assert_eq!(seen.len(), 4, "hook saw {} iterates", seen.len());
        assert_eq!(f, *seen.last().unwrap());
        assert!(f > 1e-10, "stopped hook still ran to convergence");
    }

    /// Ill-conditioned quadratic: the case where curvature reuse should pay.
    fn quad(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let scales = [1.0, 10.0, 100.0, 1000.0];
        let mut f = 0.0;
        let mut g = Array1::zeros(x.len());
        for i in 0..x.len() {
            let c = scales[i % scales.len()];
            f += 0.5 * c * x[i] * x[i];
            g[i] = c * x[i];
        }
        (f, g)
    }

    #[test]
    fn converges_on_an_ill_conditioned_quadratic() {
        let mut opt = WarmLbfgs::default();
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let (f, x, evals) = opt.minimize(x0.view(), 200, |v| Some(quad(v)));
        assert!(f < 1e-10, "did not converge, f = {f}, evals = {evals}");
        assert!(x.iter().all(|v| v.abs() < 1e-4));
    }

    /// The property the type exists for: a second relaxation starting near the
    /// first one's answer costs fewer evaluations when curvature is retained.
    #[test]
    fn retained_curvature_costs_fewer_evaluations() {
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let mut warm = WarmLbfgs::default();
        let (_, xa, _) = warm.minimize(x0.view(), 200, |v| Some(quad(v)));
        let mut perturbed = xa.clone();
        for (i, v) in perturbed.iter_mut().enumerate() {
            *v += if i % 2 == 0 { 0.02 } else { -0.02 };
        }
        let (_, _, warm_evals) = warm.minimize(perturbed.view(), 200, |v| Some(quad(v)));

        let mut cold = WarmLbfgs::default();
        let (_, _, cold_evals) = cold.minimize(perturbed.view(), 200, |v| Some(quad(v)));

        assert!(
            warm_evals < cold_evals,
            "retained curvature should cost less: warm {warm_evals}, cold {cold_evals}"
        );
    }

    #[test]
    fn forget_clears_the_memory() {
        let mut opt = WarmLbfgs::default();
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
        opt.minimize(x0.view(), 50, |v| Some(quad(v)));
        assert!(!opt.is_empty(), "a relaxation should store curvature");
        opt.forget();
        assert!(opt.is_empty());
    }

    #[test]
    fn stops_when_the_budget_ends() {
        let mut opt = WarmLbfgs::default();
        let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
        let mut left = 3;
        let (_, _, evals) = opt.minimize(x0.view(), 200, |v| {
            if left == 0 {
                return None;
            }
            left -= 1;
            Some(quad(v))
        });
        assert!(evals <= 3, "spent {evals} with a budget of 3");
    }
}
