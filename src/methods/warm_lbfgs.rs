//! Limited-memory quasi-Newton relaxation whose curvature survives between calls.
//!
//! The solver is [`rgmin::Lbfgs`]: L-BFGS two-loop recursion
//! (Nocedal-Wright 7.4, scaling 7.20) with the strong Wolfe conditions
//! (algorithms 3.5 and 3.6). This type is the hopping-chain handle: it
//! keeps the pair history across relaxations, exposes `forget` when the
//! structure changes, and offers a watch hook for screening predictors.
//!
//! # Measured status
//!
//! On a 75-point Lennard-Jones cluster, 400 relaxations from perturbed minima
//! with overlapping pairs repaired first, against SciPy's L-BFGS-B on the
//! identical protocol:
//!
//! | arm            | evals per relaxation | worst final `|g|` |
//! |----------------|----------------------|-------------------|
//! | cold           | 386.1                | 1.45e-5           |
//! | warm           | 386.1                | 1.45e-5           |
//! | L-BFGS-B       | 273                  | 1.43e-5           |
//!
//! Both arms converge and cost 1.4 times a production minimiser. The deepest
//! structure found over the 400 was -388.26 here against -386.84 for the
//! reference, which is a property of which basins the perturbations landed in
//! rather than of the minimisers.
//!
//! # The retained curvature does not survive, and the name is now vestigial
//!
//! Warm and cold agree to the last digit because the memory is empty at the
//! end of every relaxation: 0 of 400 started with stored curvature. A
//! relaxation terminates by line-search failure at a gradient near 1e-5,
//! short of the 1e-6 tolerance, and that path discards the memory before
//! returning. So the premise this type was built on, that curvature at a new
//! start resembles curvature at the old minimum and is worth carrying, is not
//! something this implementation tests. It is a correct L-BFGS whose warm
//! start never engages.
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
use rgmin::{GradNorm, Lbfgs};

/// L-BFGS with memory that persists across relaxations.
///
/// Delegates the two-loop map and strong Wolfe search to
/// [`rgmin::Lbfgs`]. Public fields are copied onto the inner
/// solver at the start of each [`WarmLbfgs::minimize`] call.
pub struct WarmLbfgs {
    inner: Lbfgs,
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
            inner: Lbfgs::default(),
            max_pairs: 8,
            gtol: 1e-6,
            armijo: 1e-4,
            curvature: 0.9,
            max_line_evals: 20,
        }
    }
}

impl WarmLbfgs {
    fn sync(&mut self) {
        self.inner.max_pairs = self.max_pairs;
        self.inner.gtol = self.gtol;
        self.inner.armijo = self.armijo;
        self.inner.curvature = self.curvature;
        self.inner.max_line_evals = self.max_line_evals;
        self.inner.norm = GradNorm::Infinity;
        self.inner.trim();
    }

    /// Discards the stored curvature.
    ///
    /// Called when the chain moves somewhere structurally different, where the
    /// retained pairs describe a Hessian that no longer applies.
    pub fn forget(&mut self) {
        self.inner.forget();
    }

    /// Pairs currently held.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True when no curvature is stored.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Two-loop direction \(d=-Hg\) on the stored pairs.
    pub fn two_loop(&self, g: ArrayView1<f64>) -> Array1<f64> {
        self.inner.two_loop(g)
    }

    /// Store one accepted curvature pair.
    pub fn record_pair(&mut self, s: Array1<f64>, y: Array1<f64>) {
        self.inner.record(s, y);
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
    /// Relaxes `x0`, consulting `recognise` at each accepted iterate.
    ///
    /// The warm layer's face of [`rgmin::Lbfgs::minimize_recognized`]:
    /// a recogniser that certifies where this descent ends -- a minimum
    /// already on file whose catchment the iterate has entered -- ends the
    /// relaxation with the stand-in and refunds the rest of the descent.
    /// The flag separates refunded descents from completed ones, which is
    /// what an auditing caller needs to estimate its recogniser's error
    /// rate against the budget `Hop.refund_with_errors` prices.
    pub fn minimize_recognized<F, R>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        fg: F,
        recognise: R,
    ) -> (f64, Array1<f64>, usize, bool)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        R: FnMut(usize, f64, ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        self.sync();
        self.inner.minimize_recognized(x0, max_iter, fg, recognise)
    }

    pub fn minimize_watched<F, W>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        fg: F,
        watch: W,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        W: FnMut(usize, f64) -> bool,
    {
        self.sync();
        self.inner.minimize_watched(x0, max_iter, fg, watch)
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
