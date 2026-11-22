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
//!
//! # Refused evaluations
//!
//! An engine that refuses a geometry, such as an SCF that does not
//! converge, answers with a non-finite value or gradient. `rgmin` 0.5.0's
//! bracketing phase compares a NaN value false against both Wolfe tests and
//! doubles the step, up to `max_line_evals` times, and a NaN gradient at the
//! start of a relaxation becomes a NaN direction. On a tblite water board
//! the farthest atom of the requested geometry went from 5.5 to 1.7e6
//! Angstrom from the centroid within 20 requests, and then the coordinates
//! went NaN. Here a non-finite answer after the start reaches the solver as
//! `+inf` with a zero gradient, which fails the Armijo test, so the zoom
//! bisects back towards the accepted step. A refused start has no accepted
//! step to return to and ends that relaxation at `+inf`.

use ndarray::{Array1, ArrayView1};
use rgmin::{GradNorm, Lbfgs};

use crate::methods::cluster_hopping::covalent_radius;

/// Stands between the caller's objective and the solver.
struct Refusals<F> {
    fg: F,
    answered: usize,
    refused: usize,
    refused_start: bool,
}

impl<F> Refusals<F>
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    fn new(fg: F) -> Self {
        Self {
            fg,
            answered: 0,
            refused: 0,
            refused_start: false,
        }
    }

    fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        let (f, g) = (self.fg)(x)?;
        self.answered += 1;
        if f.is_finite() && g.iter().all(|v| v.is_finite()) {
            return Some((f, g));
        }
        self.refused += 1;
        if self.answered == 1 {
            self.refused_start = true;
            return None;
        }
        Some((f64::INFINITY, Array1::zeros(x.len())))
    }
}

/// L-BFGS with memory that persists across relaxations.
///
/// Delegates the two-loop map and strong Wolfe search to
/// [`rgmin::Lbfgs`]. Public fields are copied onto the inner
/// solver at the start of each [`WarmLbfgs::minimize`] call.
pub struct WarmLbfgs {
    inner: Lbfgs,
    /// Declared bonds the opening trial must not stretch past.
    ///
    /// Unset, the relaxation is [`rgmin::Lbfgs`] unchanged. Set, an empty
    /// memory still opens along the steepest direction, and the trial is
    /// the longest fraction of that step that leaves every declared bond
    /// inside its bond check.
    bonds: Option<BondLimit>,
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
    refused: usize,
}

impl Default for WarmLbfgs {
    fn default() -> Self {
        Self {
            inner: Lbfgs::default(),
            bonds: None,
            max_pairs: 8,
            gtol: 1e-6,
            armijo: 1e-4,
            curvature: 0.9,
            max_line_evals: 20,
            refused: 0,
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

    /// Keep each declared bond inside `tolerance` times its covalent radii.
    ///
    /// The tolerance and the radii are the caller's bond check. A step that
    /// already fits is not shortened. No bond information leaves the
    /// relaxation on the solver's own line search.
    pub(crate) fn keep_declared_bonds(
        &mut self,
        species: &[u32],
        groups: &[Vec<usize>],
        tolerance: f64,
    ) {
        self.bonds = Some(BondLimit {
            species: species.to_vec(),
            groups: groups.to_vec(),
            tolerance,
        });
    }

    /// Drop a bond limit installed by [`Self::keep_declared_bonds`].
    pub(crate) fn clear_bond_limit(&mut self) {
        self.bonds = None;
    }

    /// Evaluations answered with a non-finite value or gradient, over every
    /// relaxation this handle has run.
    pub fn refused(&self) -> usize {
        self.refused
    }

    /// Relaxes `x0`, calling `fg` for value and gradient.
    ///
    /// `fg` returns `None` when the caller's budget is spent, which ends the
    /// relaxation where it stands. A non-finite value or gradient is a
    /// refused evaluation: the line search backtracks from it, and a refused
    /// start returns `+inf` at `x0`. Returns the value, the point, and the
    /// number of evaluations used.
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
        let mut guard = Refusals::new(fg);
        let (f, x, evals, recognized) =
            self.inner
                .minimize_recognized(x0, max_iter, |v| guard.eval(v), recognise);
        self.refused += guard.refused;
        (f, x, evals + usize::from(guard.refused_start), recognized)
    }

    /// Relaxes `x0`, offering each accepted iterate to `watch`.
    ///
    /// `watch` receives the iteration index and the value at that iterate;
    /// returning `false` ends the relaxation at that accepted point. The hook
    /// does not interrupt a line search, so the returned coordinates and value
    /// remain consistent. `fg` returns `None` when its budget is exhausted.
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
        let mut guard = Refusals::new(fg);
        let bonds = self.bonds.clone();
        let (f, x, evals) = if let Some(bonds) = bonds.as_ref() {
            self.descend_within_bonds(x0, max_iter, &mut |v| guard.eval(v), watch, bonds)
        } else {
            self.inner
                .minimize_watched(x0, max_iter, |v| guard.eval(v), watch)
        };
        self.refused += guard.refused;
        (f, x, evals + usize::from(guard.refused_start))
    }

    /// The same loop as [`rgmin::Lbfgs::minimize_watched`], with the opening
    /// trial held inside the bond check.
    ///
    /// An empty memory takes `min(1, 1/||g||)` along `-g`. That length is
    /// one coordinate unit whenever the gradient is longer than one, which
    /// stretches a covalent bond past the check before any curvature exists
    /// to size the step. The trial here is that same direction, shortened
    /// to the largest factor that leaves every currently bonded pair at or
    /// inside the check. A direction that never reaches the check keeps the
    /// factor the solver would have used.
    fn descend_within_bonds<F, W>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        fg: &mut F,
        mut watch: W,
        bonds: &BondLimit,
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
            if gradient_norm(self.inner.norm, &g) < self.inner.gtol {
                break;
            }
            let d = self.inner.two_loop(g.view());
            let slope = d.dot(&g);
            if slope >= 0.0 {
                self.inner.forget();
                continue;
            }
            let a_max = max_bond_alpha(x.view(), d.view(), bonds);
            let (ok, used) = line_search_capped(
                &mut self.inner,
                &mut x,
                &mut f,
                &mut g,
                &d,
                slope,
                fg,
                a_max,
            );
            evals += used;
            if !ok {
                if self.inner.is_empty() {
                    break;
                }
                self.inner.forget();
            }
        }
        (f, x, evals)
    }
}

/// Species, declared groups, and the bond-check factor `tolerance`.
#[derive(Clone, Debug)]
struct BondLimit {
    species: Vec<u32>,
    groups: Vec<Vec<usize>>,
    tolerance: f64,
}

fn gradient_norm(norm: GradNorm, g: &Array1<f64>) -> f64 {
    match norm {
        GradNorm::Euclidean => {
            if g.iter().any(|v| !v.is_finite()) {
                return f64::INFINITY;
            }
            l2(g)
        }
        GradNorm::Infinity => {
            let mut peak = 0.0_f64;
            for v in g.iter() {
                if !v.is_finite() {
                    return f64::INFINITY;
                }
                peak = peak.max(v.abs());
            }
            peak
        }
    }
}

fn l2(g: &Array1<f64>) -> f64 {
    g.iter().fold(0.0_f64, |acc, v| acc + v * v).sqrt()
}

/// Largest `alpha` for which `x + alpha * direction` keeps every declared
/// bond inside the check. Infinite when the direction does not stretch one.
fn max_bond_alpha(x: ArrayView1<f64>, direction: ArrayView1<f64>, bonds: &BondLimit) -> f64 {
    let n = bonds.species.len();
    if !(bonds.tolerance > 0.0) || x.len() != 3 * n || direction.len() != x.len() {
        return f64::INFINITY;
    }
    let mut cap = f64::INFINITY;
    for atoms in &bonds.groups {
        if atoms.len() < 2 {
            continue;
        }
        for (left, &a) in atoms.iter().enumerate() {
            if a >= n {
                continue;
            }
            for &b in atoms.iter().skip(left + 1) {
                if b >= n || a == b {
                    continue;
                }
                let reach = bonds.tolerance
                    * (covalent_radius(bonds.species[a]) + covalent_radius(bonds.species[b]));
                if !(reach > 0.0) {
                    continue;
                }
                let mut separated = [0.0; 3];
                let mut relative = [0.0; 3];
                for axis in 0..3 {
                    separated[axis] = x[3 * b + axis] - x[3 * a + axis];
                    relative[axis] = direction[3 * b + axis] - direction[3 * a + axis];
                }
                let step = alpha_until_reach(separated, relative, reach);
                if step < cap {
                    cap = step;
                }
            }
        }
    }
    cap
}

/// Largest nonnegative scale at which `||v + s w||` is still at most `limit`.
///
/// Infinite when `s = 0` is already past the limit, or when the ray never
/// reaches it. The closed form is the positive root of that quadratic. A
/// root that rounds past the limit is walked back by bisection onto the
/// legal side of the same comparison.
fn alpha_until_reach(v: [f64; 3], w: [f64; 3], limit: f64) -> f64 {
    let distance = |scale: f64| -> f64 {
        let mut square = 0.0;
        for axis in 0..3 {
            let delta = v[axis] + scale * w[axis];
            square += delta * delta;
        }
        square.sqrt()
    };
    if distance(0.0) > limit {
        return f64::INFINITY;
    }
    let vv = v.iter().map(|c| c * c).sum::<f64>();
    let ww = w.iter().map(|c| c * c).sum::<f64>();
    let vw = v
        .iter()
        .zip(w)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let limit2 = limit * limit;
    let root = if ww == 0.0 {
        if vw <= 0.0 {
            f64::INFINITY
        } else {
            (limit2 - vv) / (2.0 * vw)
        }
    } else {
        let linear = 2.0 * vw;
        let constant = vv - limit2;
        let disc = linear * linear - 4.0 * ww * constant;
        if disc < 0.0 || constant > 0.0 {
            f64::INFINITY
        } else {
            (-linear + disc.sqrt()) / (2.0 * ww)
        }
    };
    if !root.is_finite() || root < 0.0 {
        return f64::INFINITY;
    }
    if distance(root) <= limit {
        return root;
    }
    let mut lo = 0.0;
    let mut hi = root;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if distance(mid) <= limit {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Strong Wolfe search copied from `rgmin` 0.5.0, with `a_max` as a ceiling.
///
/// The opening length matches that solver: one, or `1/||d||` when the
/// memory is empty and the direction is longer than one. The ceiling is
/// the bond check. A direction the check does not constrain has an
/// infinite ceiling, and the arithmetic below is then the original search.
fn line_search_capped<F>(
    solver: &mut Lbfgs,
    x: &mut Array1<f64>,
    f: &mut f64,
    g: &mut Array1<f64>,
    d: &Array1<f64>,
    slope: f64,
    fg: &mut F,
    a_max: f64,
) -> (bool, usize)
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    let f0 = *f;
    let mut evals = 0usize;
    let mut scratch = x.clone();
    let probe = |a: f64, fg: &mut F, evals: &mut usize, scratch: &mut Array1<f64>| {
        scratch.assign(x);
        scratch.scaled_add(a, d);
        let answered = fg(scratch.view());
        if answered.is_some() {
            *evals += 1;
        }
        answered
    };

    let mut a_prev = 0.0;
    let mut f_prev = f0;
    let mut slope_prev = slope;
    let mut a = if solver.is_empty() {
        let dnorm = d.iter().fold(0.0_f64, |acc, v| acc + v * v).sqrt();
        if dnorm > 1.0 { 1.0 / dnorm } else { 1.0 }
    } else {
        1.0
    };
    if a > a_max {
        a = a_max;
    }
    if !(a > 0.0) {
        return (false, evals);
    }
    let mut lo = 0.0;
    let mut f_lo = f0;
    let mut slope_lo = slope;
    let mut hi = f64::NAN;
    let mut f_hi = f64::NAN;
    let mut slope_hi = f64::NAN;
    let mut bracketed = false;

    let bracket_cap = solver.max_line_evals;
    let total_cap = 2 * solver.max_line_evals;

    for i in 0..bracket_cap {
        let (fa, ga) = match probe(a, fg, &mut evals, &mut scratch) {
            Some(v) => v,
            None => return (false, evals),
        };
        let slope_a = d.dot(&ga);
        if fa > f0 + solver.armijo * a * slope || (i > 0 && fa >= f_prev) {
            lo = a_prev;
            f_lo = f_prev;
            slope_lo = slope_prev;
            hi = a;
            f_hi = fa;
            slope_hi = slope_a;
            bracketed = true;
            break;
        }
        if slope_a.abs() <= -solver.curvature * slope {
            accept_step(solver, x, f, g, d, a, fa, ga);
            return (true, evals);
        }
        if slope_a >= 0.0 {
            lo = a;
            f_lo = fa;
            slope_lo = slope_a;
            hi = a_prev;
            f_hi = f_prev;
            slope_hi = slope_prev;
            bracketed = true;
            break;
        }
        a_prev = a;
        f_prev = fa;
        slope_prev = slope_a;
        let grown = a * 2.0;
        if grown <= a_max {
            a = grown;
        } else if a < a_max {
            a = a_max;
        } else {
            // The bond check is the longest step on this ray, and the
            // step still decreases the value. Taking it is the trial the
            // check allows; a longer one is not evaluated.
            accept_step(solver, x, f, g, d, a, fa, ga);
            return (true, evals);
        }
    }

    if !bracketed {
        return (false, evals);
    }

    while evals < total_cap {
        let width = hi - lo;
        let mut trial = lo + 0.5 * width;
        if slope_hi.is_finite() {
            let d1 = slope_lo + slope_hi - 3.0 * (f_lo - f_hi) / (lo - hi);
            let disc = d1 * d1 - slope_lo * slope_hi;
            if disc >= 0.0 && (lo - hi).abs() > 1e-16 {
                let d2 = (hi - lo).signum() * disc.sqrt();
                let denom = slope_hi - slope_lo + 2.0 * d2;
                if denom.abs() > 1e-16 {
                    let q = hi - (hi - lo) * (slope_hi + d2 - d1) / denom;
                    if (q - lo) / width > 0.1 && (q - lo) / width < 0.9 {
                        trial = q;
                    }
                }
            }
        } else {
            let denom = 2.0 * (f_hi - f_lo - slope_lo * width);
            if denom.abs() > 1e-16 {
                let q = lo - slope_lo * width * width / denom;
                if (q - lo) / width > 0.1 && (q - lo) / width < 0.9 {
                    trial = q;
                }
            }
        }
        let (ft, gt) = match probe(trial, fg, &mut evals, &mut scratch) {
            Some(v) => v,
            None => return (false, evals),
        };
        let slope_t = d.dot(&gt);
        if ft > f0 + solver.armijo * trial * slope || ft >= f_lo {
            hi = trial;
            f_hi = ft;
            slope_hi = slope_t;
        } else {
            if slope_t.abs() <= -solver.curvature * slope {
                accept_step(solver, x, f, g, d, trial, ft, gt);
                return (true, evals);
            }
            if slope_t * (hi - lo) >= 0.0 {
                hi = lo;
                f_hi = f_lo;
                slope_hi = slope_lo;
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

fn accept_step(
    solver: &mut Lbfgs,
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
    solver.record(s, y);
    x.scaled_add(step, d);
    *f = f_new;
    *g = g_new;
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

    /// A bowl centred at 3 that the engine answers only inside `|x_i| <= 2`,
    /// as an SCF that fails away from a bound geometry does.
    fn refused_outside(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        if x.iter().any(|v| v.abs() > 2.0) {
            return (f64::NAN, Array1::from_elem(x.len(), f64::NAN));
        }
        let f = x.iter().map(|v| (v - 3.0) * (v - 3.0)).sum();
        (f, x.mapv(|v| 2.0 * (v - 3.0)))
    }

    #[test]
    fn a_refusal_shrinks_the_step_instead_of_doubling_it() {
        let mut opt = WarmLbfgs::default();
        let x0 = Array1::zeros(6);
        let mut farthest = 0.0_f64;
        let mut non_finite = 0;
        let (f, x, evals) = opt.minimize(x0.view(), 200, |v| {
            if v.iter().any(|c| !c.is_finite()) {
                non_finite += 1;
            }
            farthest = v.iter().fold(farthest, |m, c| m.max(c.abs()));
            Some(refused_outside(v))
        });
        assert_eq!(non_finite, 0, "NaN coordinates reached the engine");
        assert!(farthest < 8.0, "a trial went {farthest} out");
        assert!(f.is_finite() && f < refused_outside(x0.view()).0, "f = {f}");
        assert!(x.iter().all(|c| c.abs() <= 2.0), "ended outside at {x}");
        assert!(opt.refused() > 0 && opt.refused() < evals);
    }

    #[test]
    fn a_refused_start_ends_only_that_relaxation() {
        let mut opt = WarmLbfgs::default();
        let refused = Array1::from_elem(4, 2.5);
        let mut calls = 0;
        let (f, x, evals) = opt.minimize(refused.view(), 200, |v| {
            calls += 1;
            Some(refused_outside(v))
        });
        assert_eq!((calls, evals, opt.refused()), (1, 1, 1));
        assert_eq!(f, f64::INFINITY);
        assert_eq!(x, refused);
        let (f, _, _) = opt.minimize(Array1::zeros(4).view(), 200, |v| Some(refused_outside(v)));
        assert!(f.is_finite() && f < 36.0, "the next relaxation gave {f}");
    }

    /// Harmonic O–H. `k` is the test spring, not a quench parameter.
    fn oh_spring(x: ArrayView1<f64>, equilibrium: f64, stiffness: f64) -> (f64, Array1<f64>) {
        let mut square = 0.0;
        for axis in 0..3 {
            let delta = x[3 + axis] - x[axis];
            square += delta * delta;
        }
        let distance = square.sqrt();
        let gap = distance - equilibrium;
        let energy = 0.5 * stiffness * gap * gap;
        let mut gradient = Array1::zeros(x.len());
        if distance > 0.0 {
            let scale = stiffness * gap / distance;
            for axis in 0..3 {
                let delta = x[3 + axis] - x[axis];
                gradient[axis] = -scale * delta;
                gradient[3 + axis] = scale * delta;
            }
        }
        (energy, gradient)
    }

    fn bond_length(x: ArrayView1<f64>) -> f64 {
        (0..3)
            .map(|axis| {
                let delta = x[3 + axis] - x[axis];
                delta * delta
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Develop's empty memory steps one coordinate unit along `-g`. On this
    /// bond that point fails the bond check. The opening trial must not.
    #[test]
    fn an_empty_memory_opening_stays_inside_the_bond_check() {
        use crate::methods::cluster_hopping::declared_group_lost_a_bond;

        let tolerance = 1.25;
        let species = [8u32, 1];
        let groups = vec![vec![0usize, 1]];
        let mut x0 = Array1::zeros(6);
        x0[3] = 0.90;
        let (_, gradient) = oh_spring(x0.view(), 0.96, 400.0);
        let mut direction = gradient;
        direction.mapv_inplace(|component| -component);
        let norm = direction.iter().fold(0.0_f64, |acc, v| acc + v * v).sqrt();
        assert!(norm > 1.0, "||g|| = {norm}, so the opening is not one unit");
        let mut uncapped = x0.clone();
        uncapped.scaled_add(1.0 / norm, &direction);
        let uncapped_length = uncapped
            .iter()
            .zip(x0.iter())
            .fold(0.0_f64, |acc, (left, right)| {
                let delta = left - right;
                acc + delta * delta
            })
            .sqrt();
        assert!(
            (uncapped_length - 1.0).abs() < 1e-12,
            "uncapped length {uncapped_length}"
        );
        assert!(
            declared_group_lost_a_bond(uncapped.view(), &species, Some(&groups), tolerance),
            "the one-unit step has to be the trial the check refuses"
        );

        let mut opt = WarmLbfgs::default();
        opt.keep_declared_bonds(&species, &groups, tolerance);
        let mut probes = Vec::new();
        let start = oh_spring(x0.view(), 0.96, 400.0).0;
        let (energy, done, _) = opt.minimize(x0.view(), 40, |point| {
            probes.push(point.to_owned());
            Some(oh_spring(point, 0.96, 400.0))
        });
        assert!(probes.len() >= 2, "the line search never left the start");
        assert!(
            !declared_group_lost_a_bond(probes[1].view(), &species, Some(&groups), tolerance),
            "first trial broke the check at length {}",
            bond_length(probes[1].view())
        );
        let taken = probes[1]
            .iter()
            .zip(x0.iter())
            .fold(0.0_f64, |acc, (left, right)| {
                let delta = left - right;
                acc + delta * delta
            })
            .sqrt();
        assert!(taken > 0.0 && taken < 1.0, "first trial length {taken}");
        assert!(energy.is_finite() && energy < start, "energy {energy}");
        assert!(
            !declared_group_lost_a_bond(done.view(), &species, Some(&groups), tolerance),
            "settled length {}",
            bond_length(done.view())
        );
        assert!(
            (bond_length(done.view()) - 0.96).abs() < 1e-4,
            "settled at {}",
            bond_length(done.view())
        );
    }

    /// A first step the check already allows is the solver's own step.
    #[test]
    fn a_step_inside_the_check_matches_the_uncapped_solver() {
        use crate::methods::cluster_hopping::declared_group_lost_a_bond;

        let tolerance = 1.25;
        let species = [8u32, 1];
        let groups = vec![vec![0usize, 1]];
        let mut x0 = Array1::zeros(6);
        x0[3] = 0.962;
        let (_, gradient) = oh_spring(x0.view(), 0.96, 50.0);
        let mut direction = gradient;
        direction.mapv_inplace(|component| -component);
        let norm = direction.iter().fold(0.0_f64, |acc, v| acc + v * v).sqrt();
        let alpha = if norm > 1.0 { 1.0 / norm } else { 1.0 };
        let mut opening = x0.clone();
        opening.scaled_add(alpha, &direction);
        assert!(
            !declared_group_lost_a_bond(opening.view(), &species, Some(&groups), tolerance),
            "this opening is the control, and it has to fit the check"
        );

        let mut plain = WarmLbfgs::default();
        let mut capped = WarmLbfgs::default();
        capped.keep_declared_bonds(&species, &groups, tolerance);
        let (plain_energy, plain_x, plain_evals) =
            plain.minimize(x0.view(), 40, |point| Some(oh_spring(point, 0.96, 50.0)));
        let (capped_energy, capped_x, capped_evals) =
            capped.minimize(x0.view(), 40, |point| Some(oh_spring(point, 0.96, 50.0)));
        assert_eq!(plain_evals, capped_evals);
        assert_eq!(plain_energy.to_bits(), capped_energy.to_bits());
        assert_eq!(plain_x, capped_x);
    }
}
