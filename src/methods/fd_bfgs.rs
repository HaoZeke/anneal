//! Resumable projected BFGS on finite-difference gradients.
//!
//! An ask/tell state machine for box-constrained descent when only objective
//! values are available. Every stencil point and every line-search trial is
//! one candidate handed to the caller, so the caller charges and records each
//! evaluation and may pause between any two of them.
//!
//! Gradients are forward differences with steps `sqrt(eps)` times the
//! coordinate's typical scale (its magnitude, floored at a tenth of the box
//! width). When a forward-difference line search fails, or an accepted step
//! buys less than the slow-iteration threshold, the engine switches to central
//! differences with steps `eps^(1/3)` times the same scale, the
//! Gill-Murray-Wright rule for the point where truncation error starts to
//! dominate. A stencil that would leave the box steps inward instead (one-sided
//! and second order in the central scheme), so no candidate lies outside it.
//!
//! That rule assumes the third derivative is of order `|f|` over the cube of
//! the scale. Across a valley much narrower than the scale it is not: the
//! stencil straddles the valley floor and returns a secant slope. Each
//! central stencil also yields the second difference `c_i` along its
//! coordinate, so the next central gradient at a coordinate uses the interval
//! `0.1 |g_i / c_i|`, a tenth of the distance to the minimum of the local
//! quadratic, where the relative truncation error of the slope is about
//! `h^2 / (3 L^2)`, 0.3%. The interval never exceeds the `eps^(1/3)` rule and
//! never drops below `eps^(2/3)` times the scale or `sqrt(eps |f| / |c_i|)`,
//! below which rounding moves the slope more than curvature does. A failed
//! central line search re-differences at the same point when an interval
//! has since moved by more than a factor of two, at most twice per iterate,
//! before it falls back to steepest descent.
//!
//! The step is projected BFGS. Coordinates held at a bound by the gradient are
//! fixed, the free block of the dense inverse-Hessian approximation scales the
//! free gradient, and a backtracking line search along the projected arc
//! accepts the Armijo condition, shrinking by safeguarded quadratic
//! interpolation. A first trial that is accepted while the decrease is still
//! nearly linear gets one longer trial at the minimiser of the quadratic
//! through the two values and the slope: the long steps a Wolfe search finds
//! by extrapolation, without paying for a gradient at every trial. Pairs that
//! violate the curvature condition are skipped.
//!
//! The initial inverse Hessian is the box metric `D = diag(w^2)`, the identity
//! in box-normalised coordinates, so a side that spans decades moves as far,
//! relative to its width, as a unit one. The first accepted pair scales it to
//! `(s.y / y.D.y) D`, and the steepest-descent fallback steps along `-D g`.

use eindir_core::Bounds;
use ndarray::{Array1, Array2, ArrayView1};

/// Armijo sufficient-decrease constant.
const ARMIJO: f64 = 1e-4;
/// Line-search trials before the step direction is declared useless.
const MAX_LINE_TRIALS: usize = 40;
/// An accepted first trial that realises more than this fraction of its
/// linear decrease is probably short (an exact quadratic model realises
/// half), so the minimiser of the quadratic through the start value, the
/// slope, and the trial value gets one extra evaluation.
const EXTRAPOLATE_RATIO: f64 = 0.6;
/// Longest extrapolated trial, as a multiple of the accepted step.
const MAX_EXTRAPOLATION: f64 = 16.0;
/// Until a curvature pair has scaled the inverse Hessian, each line search's
/// first trial moves the coordinate that moves most by this fraction of its
/// box width, whatever the gradient's size, so the descent does not depend on
/// the scale of the objective.
const FIRST_STEP_FRACTION: f64 = 0.05;
/// Central interval as a fraction of the distance `|g_i / c_i|` to the
/// minimum of the coordinate's local quadratic.
const CENTRAL_INTERVAL_FRACTION: f64 = 0.1;
/// Factor by which an adapted interval must differ from the one a gradient
/// used before a failed line search re-differences.
const INTERVAL_MOVE: f64 = 2.0;
/// Re-differenced gradients allowed at one iterate.
const MAX_REDIFFERENCES: usize = 2;

/// Termination controls for one descent.
#[derive(Clone, Copy, Debug)]
pub struct FdBfgsOptions {
    /// Relative decrease per iteration below which an iteration is slow.
    pub ftol: f64,
    /// Consecutive slow iterations that end the descent.
    pub patience: usize,
    /// Accepted-iteration cap; zero means unlimited.
    pub max_iter: usize,
    /// Whether a failed or slow forward-difference iteration switches to
    /// central differences instead of ending (or slowing) the descent.
    pub refine: bool,
    /// Floor on the value scale of the slow-iteration test, which compares
    /// the decrease with `ftol * max(|f_k|, |f_k+1|, value_floor)`.
    /// L-BFGS-B uses 1.
    pub value_floor: f64,
    /// Infinity norm of the projected gradient at or below which the descent
    /// ends; zero disables the test. L-BFGS-B's `pgtol`.
    pub gtol: f64,
}

impl Default for FdBfgsOptions {
    /// Descend as deep as finite differences allow.
    fn default() -> Self {
        Self {
            ftol: 1e-12,
            patience: 3,
            max_iter: 0,
            refine: true,
            value_floor: f64::MIN_POSITIVE,
            gtol: 0.0,
        }
    }
}

/// Finite-difference stencil in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FdScheme {
    /// One extra evaluation per coordinate.
    Forward,
    /// Two extra evaluations per coordinate.
    Central,
}

struct GradientWork {
    coord: usize,
    point: usize,
    offsets: Vec<f64>,
    weights: Vec<f64>,
    base_weight: f64,
    acc: f64,
    values: Vec<f64>,
    finite: bool,
    grad: Array1<f64>,
}

struct LineWork {
    direction: Array1<f64>,
    alpha: f64,
    trials: usize,
    slope: f64,
    steepest: bool,
    trial: Array1<f64>,
    fallback: Option<(Array1<f64>, f64)>,
}

enum Phase {
    Value,
    Gradient(GradientWork),
    Line(LineWork),
    Done,
}

/// One projected-BFGS descent driven by finite differences.
pub struct FdBfgs {
    n: usize,
    low: Array1<f64>,
    high: Array1<f64>,
    width: Array1<f64>,
    /// Box metric `diag(w^2)`, kept as its diagonal.
    metric: Array1<f64>,
    x: Array1<f64>,
    f: f64,
    grad: Array1<f64>,
    inv_hessian: Array2<f64>,
    scaled: bool,
    gamma: f64,
    scheme: FdScheme,
    phase: Phase,
    pair_from: Option<(Array1<f64>, Array1<f64>)>,
    iterations: usize,
    slow: usize,
    options: FdBfgsOptions,
    /// Curvature-adapted central interval per coordinate, once a central
    /// stencil has measured it.
    intervals: Vec<Option<f64>>,
    /// Interval the last central stencil at each coordinate used.
    used_intervals: Vec<f64>,
    /// Gradients re-differenced at the current iterate.
    redifferences: usize,
}

impl FdBfgs {
    /// Starts a descent at `x0` (clipped into the box). With `f0` the start
    /// value is taken as known; without it the first candidate is `x0`.
    pub fn new(
        x0: ArrayView1<f64>,
        f0: Option<f64>,
        bounds: &Bounds<f64>,
        options: FdBfgsOptions,
    ) -> Self {
        let n = bounds.dims;
        assert!(n > 0, "descent needs a positive dimension");
        assert_eq!(x0.len(), n, "start length must match the box");
        let width = &bounds.high - &bounds.low;
        let metric = width.mapv(|w| w * w);
        let mut engine = Self {
            n,
            low: bounds.low.clone(),
            high: bounds.high.clone(),
            width,
            x: bounds.clip(x0),
            f: f64::INFINITY,
            grad: Array1::zeros(n),
            inv_hessian: Array2::from_diag(&metric),
            metric,
            scaled: false,
            gamma: 1.0,
            scheme: FdScheme::Forward,
            phase: Phase::Value,
            pair_from: None,
            iterations: 0,
            slow: 0,
            options,
            intervals: vec![None; n],
            used_intervals: vec![0.0; n],
            redifferences: 0,
        };
        if let Some(f0) = f0 {
            engine.set_start_value(f0);
        }
        engine
    }

    /// Moves the descent to `x`, keeping the curvature model when
    /// `keep_curvature` is set. With `f` the value at `x` is taken as known;
    /// without it the next candidate is `x`.
    pub fn restart_at(&mut self, x: ArrayView1<f64>, f: Option<f64>, keep_curvature: bool) {
        self.x = Array1::from_iter(
            x.iter()
                .enumerate()
                .map(|(i, v)| v.clamp(self.low[i], self.high[i])),
        );
        self.pair_from = None;
        self.scheme = FdScheme::Forward;
        self.iterations = 0;
        self.slow = 0;
        self.intervals.fill(None);
        self.redifferences = 0;
        if !keep_curvature {
            self.inv_hessian = Array2::from_diag(&self.metric);
            self.scaled = false;
            self.gamma = 1.0;
        }
        match f {
            Some(f) => self.set_start_value(f),
            None => {
                self.f = f64::INFINITY;
                self.phase = Phase::Value;
            }
        }
    }

    /// True once the descent has converged or exhausted its options.
    pub fn is_done(&self) -> bool {
        matches!(self.phase, Phase::Done)
    }

    /// Current iterate, the best point the descent has accepted.
    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    /// Value at the current iterate.
    pub fn value(&self) -> f64 {
        self.f
    }

    /// Stencil currently in use.
    pub fn scheme(&self) -> FdScheme {
        self.scheme
    }

    /// Accepted iterations since the last (re)start.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Next candidate to evaluate; always inside the box. Returns the
    /// iterate once the descent is done.
    pub fn ask(&self) -> Array1<f64> {
        match &self.phase {
            Phase::Value | Phase::Done => self.x.clone(),
            Phase::Gradient(work) => {
                let mut point = self.x.clone();
                point[work.coord] += work.offsets[work.point];
                point
            }
            Phase::Line(work) => work.trial.clone(),
        }
    }

    /// Value of the candidate last returned by [`FdBfgs::ask`].
    pub fn tell(&mut self, value: f64) {
        match std::mem::replace(&mut self.phase, Phase::Done) {
            Phase::Value => self.set_start_value(value),
            Phase::Done => {}
            Phase::Gradient(work) => self.tell_gradient(work, value),
            Phase::Line(work) => self.tell_line(work, value),
        }
    }

    fn set_start_value(&mut self, value: f64) {
        self.f = value;
        self.phase = if value.is_finite() {
            Phase::Gradient(self.gradient_work())
        } else {
            Phase::Done
        };
    }

    fn typical(&self, i: usize) -> f64 {
        self.x[i].abs().max(0.1 * self.width[i])
    }

    fn stencil(&self, i: usize) -> (Vec<f64>, Vec<f64>, f64) {
        let xi = self.x[i];
        let room_up = self.high[i] - xi;
        let room_down = xi - self.low[i];
        let realized = |h: f64| (xi + h) - xi;
        let h = match (self.scheme, self.intervals[i]) {
            (FdScheme::Forward, _) => f64::EPSILON.sqrt() * self.typical(i),
            (FdScheme::Central, Some(adapted)) => adapted,
            (FdScheme::Central, None) => f64::EPSILON.cbrt() * self.typical(i),
        }
        .min(0.25 * self.width[i]);
        if self.scheme == FdScheme::Central && room_up >= h && room_down >= h {
            let hp = realized(h);
            let hm = realized(-h);
            let span = hp - hm;
            return (vec![hp, hm], vec![1.0 / span, -1.0 / span], 0.0);
        }
        if self.scheme == FdScheme::Central && room_up.max(room_down) >= 2.0 * h {
            let sign = if room_up >= 2.0 * h { 1.0 } else { -1.0 };
            let h1 = realized(sign * h);
            let h2 = realized(2.0 * sign * h);
            let w0 = -(h1 + h2) / (h1 * h2);
            let w1 = h2 / (h1 * (h2 - h1));
            let w2 = -h1 / (h2 * (h2 - h1));
            return (vec![h1, h2], vec![w1, w2], w0);
        }
        let sign = if room_up >= h || room_up >= room_down {
            1.0
        } else {
            -1.0
        };
        let step = realized(sign * h.min(room_up.max(room_down)));
        if step == 0.0 {
            return (Vec::new(), Vec::new(), 0.0);
        }
        (vec![step], vec![1.0 / step], -1.0 / step)
    }

    fn gradient_work(&self) -> GradientWork {
        let mut work = GradientWork {
            coord: 0,
            point: 0,
            offsets: Vec::new(),
            weights: Vec::new(),
            base_weight: 0.0,
            acc: 0.0,
            values: Vec::new(),
            finite: true,
            grad: Array1::zeros(self.n),
        };
        self.load_coord(&mut work);
        work
    }

    /// Loads the stencil of `work.coord`, skipping degenerate coordinates.
    fn load_coord(&self, work: &mut GradientWork) {
        while work.coord < self.n {
            let (offsets, weights, base) = self.stencil(work.coord);
            if !offsets.is_empty() {
                work.offsets = offsets;
                work.weights = weights;
                work.base_weight = base;
                work.point = 0;
                work.acc = 0.0;
                work.values.clear();
                work.finite = true;
                return;
            }
            work.grad[work.coord] = 0.0;
            work.coord += 1;
        }
    }

    fn tell_gradient(&mut self, mut work: GradientWork, value: f64) {
        if value.is_finite() {
            work.acc += work.weights[work.point] * value;
        } else {
            work.finite = false;
        }
        work.values.push(value);
        work.point += 1;
        if work.point < work.offsets.len() {
            self.phase = Phase::Gradient(work);
            return;
        }
        let component = work.base_weight * self.f + work.acc;
        work.grad[work.coord] = if work.finite && component.is_finite() {
            component
        } else {
            0.0
        };
        if self.scheme == FdScheme::Central && work.offsets.len() == 2 {
            self.adapt_interval(&work, component);
        }
        work.coord += 1;
        self.load_coord(&mut work);
        if work.coord < self.n {
            self.phase = Phase::Gradient(work);
        } else {
            self.finish_gradient(work.grad);
        }
    }

    /// Sets the next central interval at `work.coord` from the slope and the
    /// second difference of the quadratic through its stencil.
    fn adapt_interval(&mut self, work: &GradientWork, slope: f64) {
        let i = work.coord;
        let (t1, t2) = (work.offsets[0], work.offsets[1]);
        let (f1, f2) = (work.values[0], work.values[1]);
        self.used_intervals[i] = t1.abs().min(t2.abs());
        let curvature = 2.0 * (self.f / (t1 * t2) + f1 / (t1 * (t1 - t2)) + f2 / (t2 * (t2 - t1)));
        if !(work.finite && slope.is_finite() && curvature.is_finite() && curvature != 0.0) {
            return;
        }
        let typical = self.typical(i);
        let ceiling = f64::EPSILON.cbrt() * typical;
        let floor = (f64::EPSILON.powf(2.0 / 3.0) * typical)
            .max((f64::EPSILON * self.f.abs() / curvature.abs()).sqrt())
            .min(ceiling);
        let reach = slope.abs() / curvature.abs();
        self.intervals[i] = Some((CENTRAL_INTERVAL_FRACTION * reach).clamp(floor, ceiling));
    }

    /// Whether some adapted interval differs from the one the current
    /// gradient used by more than [`INTERVAL_MOVE`].
    fn intervals_moved(&self) -> bool {
        self.intervals
            .iter()
            .zip(&self.used_intervals)
            .any(|(adapted, &used)| {
                adapted.is_some_and(|h| {
                    used > 0.0 && (h * INTERVAL_MOVE < used || h > INTERVAL_MOVE * used)
                })
            })
    }

    fn finish_gradient(&mut self, grad: Array1<f64>) {
        self.grad = grad;
        if let Some((x_prev, g_prev)) = self.pair_from.take() {
            let s = &self.x - &x_prev;
            let y = &self.grad - &g_prev;
            self.bfgs_update(&s, &y);
        }
        if self.options.max_iter > 0 && self.iterations >= self.options.max_iter {
            self.phase = Phase::Done;
            return;
        }
        if self.options.gtol > 0.0 && self.projected_gradient_norm() <= self.options.gtol {
            self.phase = Phase::Done;
            return;
        }
        self.begin_line_search(false);
    }

    /// Infinity norm of the gradient projected onto the box, each component
    /// capped by the room to its bound in the descent direction (L-BFGS-B's
    /// `projgr`).
    fn projected_gradient_norm(&self) -> f64 {
        (0..self.n)
            .map(|i| {
                let g = self.grad[i];
                if g < 0.0 {
                    (self.x[i] - self.high[i]).max(g).abs()
                } else {
                    (self.x[i] - self.low[i]).min(g).abs()
                }
            })
            .fold(0.0_f64, f64::max)
    }

    fn bfgs_update(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        let sy = s.dot(y);
        let s_norm = s.dot(s).sqrt();
        let y_norm = y.dot(y).sqrt();
        if !(sy.is_finite() && sy > 8.0 * f64::EPSILON * s_norm * y_norm) {
            return;
        }
        let ydy: f64 = (0..self.n).map(|i| self.metric[i] * y[i] * y[i]).sum();
        self.gamma = sy / ydy;
        if !self.scaled {
            self.inv_hessian = Array2::from_diag(&(&self.metric * self.gamma));
            self.scaled = true;
        }
        let hy = self.inv_hessian.dot(y);
        let yhy = y.dot(&hy);
        let rho = 1.0 / sy;
        let ss = rho * rho * yhy + rho;
        for i in 0..self.n {
            for j in 0..self.n {
                self.inv_hessian[[i, j]] += ss * s[i] * s[j] - rho * (s[i] * hy[j] + hy[i] * s[j]);
            }
        }
    }

    fn free(&self, i: usize) -> bool {
        let g = self.grad[i];
        !((self.x[i] <= self.low[i] && g > 0.0) || (self.x[i] >= self.high[i] && g < 0.0))
    }

    fn begin_line_search(&mut self, steepest: bool) {
        let free: Vec<usize> = (0..self.n).filter(|&i| self.free(i)).collect();
        let mut direction = Array1::<f64>::zeros(self.n);
        if !steepest {
            for &i in &free {
                direction[i] = -free
                    .iter()
                    .map(|&j| self.inv_hessian[[i, j]] * self.grad[j])
                    .sum::<f64>();
            }
        }
        let mut slope = self.grad.dot(&direction);
        let mut steepest = steepest;
        if steepest || !(slope.is_finite() && slope < 0.0) {
            steepest = true;
            direction.fill(0.0);
            for &i in &free {
                direction[i] = -self.metric[i] * self.grad[i];
            }
            slope = self.grad.dot(&direction);
        }
        if !(slope.is_finite() && slope < 0.0) {
            self.phase = Phase::Done;
            return;
        }
        let alpha = if !self.scaled {
            let reach = (0..self.n)
                .map(|i| direction[i].abs() / self.width[i].max(f64::MIN_POSITIVE))
                .fold(0.0_f64, f64::max);
            // A finite step keeps the held coordinates' zero components at
            // zero when the reach is subnormal.
            (FIRST_STEP_FRACTION / reach).min(f64::MAX)
        } else if steepest {
            self.gamma
        } else {
            1.0
        };
        let trial = self.projected(&direction, alpha);
        self.phase = Phase::Line(LineWork {
            direction,
            alpha,
            trials: 0,
            slope,
            steepest,
            trial,
            fallback: None,
        });
    }

    fn projected(&self, direction: &Array1<f64>, alpha: f64) -> Array1<f64> {
        Array1::from_iter(
            (0..self.n)
                .map(|i| (self.x[i] + alpha * direction[i]).clamp(self.low[i], self.high[i])),
        )
    }

    fn tell_line(&mut self, mut work: LineWork, value: f64) {
        if let Some((point, best)) = work.fallback.take() {
            if value.is_finite() && value < best {
                self.accept(work.trial, value);
            } else {
                self.accept(point, best);
            }
            return;
        }
        let moved = &work.trial - &self.x;
        // The strict decrease matters once the predicted decrease rounds
        // away: the Armijo bound then equals f and would accept a null step.
        let bound = self.f + ARMIJO * self.grad.dot(&moved);
        if value.is_finite() && value < self.f && value <= bound {
            if work.trials == 0 {
                let predicted = -self.grad.dot(&moved);
                let ratio = (self.f - value) / predicted;
                if predicted > 0.0 && ratio > EXTRAPOLATE_RATIO {
                    let factor = if ratio >= 1.0 {
                        MAX_EXTRAPOLATION
                    } else {
                        (0.5 / (1.0 - ratio)).min(MAX_EXTRAPOLATION)
                    };
                    let next = work.alpha * factor;
                    let trial = self.projected(&work.direction, next);
                    if trial != work.trial {
                        work.fallback = Some((std::mem::replace(&mut work.trial, trial), value));
                        work.alpha = next;
                        self.phase = Phase::Line(work);
                        return;
                    }
                }
            }
            self.accept(work.trial, value);
            return;
        }
        work.trials += 1;
        let alpha = work.alpha;
        let next = if value.is_finite() {
            let denom = value - self.f - work.slope * alpha;
            if denom > 0.0 && denom.is_finite() {
                (-work.slope * alpha * alpha / (2.0 * denom)).clamp(0.1 * alpha, 0.5 * alpha)
            } else {
                0.5 * alpha
            }
        } else {
            0.1 * alpha
        };
        let trial = self.projected(&work.direction, next);
        if work.trials >= MAX_LINE_TRIALS || trial == self.x {
            self.line_search_failed(work.steepest);
            return;
        }
        work.alpha = next;
        work.trial = trial;
        self.phase = Phase::Line(work);
    }

    fn accept(&mut self, trial: Array1<f64>, value: f64) {
        self.redifferences = 0;
        let previous = self.f;
        self.pair_from = Some((std::mem::replace(&mut self.x, trial), self.grad.clone()));
        self.f = value;
        self.iterations += 1;
        let scale = previous
            .abs()
            .max(value.abs())
            .max(self.options.value_floor);
        if (previous - value) <= self.options.ftol * scale {
            if self.scheme == FdScheme::Forward && self.options.refine {
                // A forward-difference direction that only buys a sliver of
                // decrease is usually its truncation error talking.
                self.scheme = FdScheme::Central;
                self.pair_from = None;
                self.slow = 0;
                self.phase = Phase::Gradient(self.gradient_work());
                return;
            }
            self.slow += 1;
        } else {
            self.slow = 0;
        }
        self.phase = if self.slow >= self.options.patience.max(1) {
            Phase::Done
        } else {
            Phase::Gradient(self.gradient_work())
        };
    }

    fn line_search_failed(&mut self, steepest: bool) {
        if self.scheme == FdScheme::Central
            && self.redifferences < MAX_REDIFFERENCES
            && self.intervals_moved()
        {
            self.redifferences += 1;
            self.pair_from = None;
            self.phase = Phase::Gradient(self.gradient_work());
            return;
        }
        match self.scheme {
            FdScheme::Forward if !self.options.refine => self.phase = Phase::Done,
            FdScheme::Forward => {
                self.scheme = FdScheme::Central;
                self.pair_from = None;
                self.phase = Phase::Gradient(self.gradient_work());
            }
            FdScheme::Central if !steepest && self.scaled => self.begin_line_search(true),
            FdScheme::Central => self.phase = Phase::Done,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(low: f64, high: f64, n: usize) -> Bounds<f64> {
        Bounds::new(Array1::from_elem(n, low), Array1::from_elem(n, high), 0.0)
    }

    fn rosenbrock(x: ArrayView1<f64>) -> f64 {
        (0..x.len() - 1)
            .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
            .sum()
    }

    fn drive<F: Fn(ArrayView1<f64>) -> f64>(
        engine: &mut FdBfgs,
        f: F,
        budget: usize,
        bounds: &Bounds<f64>,
    ) -> usize {
        let mut used = 0;
        while used < budget && !engine.is_done() {
            let x = engine.ask();
            assert!(bounds.contains(x.view()), "candidate left the box: {x}");
            engine.tell(f(x.view()));
            used += 1;
        }
        used
    }

    #[test]
    fn reaches_rosenbrock_minimum_from_values_only() {
        let bounds = boxed(-2.0, 2.0, 6);
        let start = Array1::from_vec(vec![-1.2, 1.0, -0.5, 0.3, 1.5, -1.0]);
        let mut engine = FdBfgs::new(start.view(), None, &bounds, FdBfgsOptions::default());
        drive(&mut engine, rosenbrock, 4000, &bounds);
        assert!(engine.value() < 1e-10, "value {}", engine.value());
    }

    #[test]
    fn scaling_the_objective_by_a_power_of_two_leaves_the_descent_unchanged() {
        // A power of two rescales every value exactly. The tests compare
        // values with values, and the first trial is a fraction of the box
        // however small the scaled gradient is, so the same points are asked.
        let bounds = boxed(-2.0, 2.0, 4);
        let start = Array1::from_vec(vec![0.5, -1.0, 1.5, 0.0]);
        let trace = |scale: f64| {
            let mut engine = FdBfgs::new(start.view(), None, &bounds, FdBfgsOptions::default());
            let mut asked = Vec::new();
            while asked.len() < 600 && !engine.is_done() {
                let x = engine.ask();
                engine.tell(scale * rosenbrock(x.view()));
                asked.push(x);
            }
            (asked, engine.value() / scale)
        };
        let (plain, value) = trace(1.0);
        for scale in [2f64.powi(-30), 2f64.powi(30)] {
            let (scaled, scaled_value) = trace(scale);
            assert_eq!(scaled, plain, "scale {scale}");
            assert_eq!(scaled_value, value, "scale {scale}");
        }
    }

    #[test]
    fn stops_on_an_active_bound_with_stencils_inside() {
        // Minimum at (3, -3) outside [-1, 1]^2: the solution is the corner.
        let bounds = boxed(-1.0, 1.0, 2);
        let f = |x: ArrayView1<f64>| (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 3.0).powi(2);
        let mut engine = FdBfgs::new(
            Array1::zeros(2).view(),
            None,
            &bounds,
            FdBfgsOptions::default(),
        );
        let used = drive(&mut engine, f, 500, &bounds);
        assert!(engine.is_done(), "used {used}");
        assert!((engine.position()[0] - 1.0).abs() < 1e-12);
        assert!((engine.position()[1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn pauses_between_evaluations_are_invisible() {
        let bounds = boxed(-2.0, 2.0, 4);
        let start = Array1::from_vec(vec![0.5, -1.0, 1.5, 0.0]);
        let mut whole = FdBfgs::new(start.view(), None, &bounds, FdBfgsOptions::default());
        let mut parts = FdBfgs::new(start.view(), None, &bounds, FdBfgsOptions::default());
        let mut trace_whole = Vec::new();
        for _ in 0..300 {
            let x = whole.ask();
            trace_whole.push(x.clone());
            whole.tell(rosenbrock(x.view()));
        }
        let mut trace_parts = Vec::new();
        for chunk in [1usize, 5, 13, 81, 200] {
            for _ in 0..chunk {
                let x = parts.ask();
                trace_parts.push(x.clone());
                parts.tell(rosenbrock(x.view()));
            }
        }
        assert_eq!(trace_whole, trace_parts);
    }

    #[test]
    fn switches_to_central_differences_before_giving_up() {
        let bounds = boxed(-3.0, 3.0, 3);
        let f = |x: ArrayView1<f64>| x.iter().map(|v| (v - 0.7).powi(2)).sum::<f64>();
        let mut engine = FdBfgs::new(
            Array1::from_elem(3, 2.0).view(),
            None,
            &bounds,
            FdBfgsOptions {
                ftol: 0.0,
                ..FdBfgsOptions::default()
            },
        );
        drive(&mut engine, f, 2000, &bounds);
        assert_eq!(engine.scheme(), FdScheme::Central);
        assert!(engine.is_done());
        assert!(engine.value() < 1e-14, "value {}", engine.value());
    }

    /// Evaluations until the descent's value first drops below `target`.
    fn evaluations_to<F: Fn(ArrayView1<f64>) -> f64>(
        engine: &mut FdBfgs,
        f: F,
        target: f64,
        budget: usize,
        bounds: &Bounds<f64>,
    ) -> usize {
        let mut used = 0;
        while used < budget && !engine.is_done() && engine.value() >= target {
            let x = engine.ask();
            assert!(bounds.contains(x.view()), "candidate left the box: {x}");
            engine.tell(f(x.view()));
            used += 1;
        }
        used
    }

    #[test]
    fn mixed_widths_descend_like_unit_widths() {
        // The same Rosenbrock on [-2, 2]^6 and stretched to [-2w, 2w] per
        // coordinate: under the box metric the two descents agree up to
        // rounding, where an identity start crawls along the wide sides.
        // They are compared at a common depth; below it rounding decides
        // how long each tail runs.
        let widths = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0];
        let start = [-1.2, 1.0, -0.5, 0.3, 1.5, -1.0];
        let unit = boxed(-2.0, 2.0, 6);
        let mut plain = FdBfgs::new(
            Array1::from_vec(start.to_vec()).view(),
            None,
            &unit,
            FdBfgsOptions::default(),
        );
        let plain_used = evaluations_to(&mut plain, rosenbrock, 1e-10, 4000, &unit);
        let w = Array1::from_vec(widths.to_vec());
        let stretched = Bounds::new(&w * -2.0, &w * 2.0, 0.0);
        let scaled = |x: ArrayView1<f64>| rosenbrock((&x / &w).view());
        let mut wide = FdBfgs::new(
            (&Array1::from_vec(start.to_vec()) * &w).view(),
            None,
            &stretched,
            FdBfgsOptions::default(),
        );
        let wide_used = evaluations_to(&mut wide, scaled, 1e-10, 4000, &stretched);
        assert!(plain.value() < 1e-10, "unit widths {}", plain.value());
        assert!(wide.value() < 1e-10, "mixed widths {}", wide.value());
        assert!(
            wide_used <= plain_used + plain_used / 10,
            "mixed widths took {wide_used} evaluations against {plain_used}"
        );
    }

    #[test]
    fn unit_value_floor_and_gradient_tolerance_end_converged_descents() {
        // Rastrigin-6 from inside its global basin, whose minimum value is 0:
        // without the unit floor the slow test shrinks with |f| and only a
        // failed line search ends the descent.
        let bounds = boxed(-5.12, 5.12, 6);
        let rastrigin = |x: ArrayView1<f64>| {
            x.iter()
                .map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos() + 10.0)
                .sum::<f64>()
        };
        let start = Array1::from_vec(vec![0.08, -0.05, 0.03, -0.07, 0.02, 0.06]);
        let base = FdBfgsOptions {
            ftol: 1e7 * f64::EPSILON,
            patience: 1,
            max_iter: 1000,
            refine: false,
            ..FdBfgsOptions::default()
        };
        let mut raw = FdBfgs::new(start.view(), None, &bounds, base);
        let raw_used = drive(&mut raw, rastrigin, 2000, &bounds);
        let mut floored = FdBfgs::new(
            start.view(),
            None,
            &bounds,
            FdBfgsOptions {
                value_floor: 1.0,
                gtol: 1e-5,
                ..base
            },
        );
        let floored_used = drive(&mut floored, rastrigin, 2000, &bounds);
        assert!(raw.is_done() && floored.is_done());
        assert!(floored.value() < 1e-8, "value {}", floored.value());
        assert!(
            floored_used < raw_used,
            "floored {floored_used} evaluations against {raw_used}"
        );
    }

    #[test]
    fn gradient_tolerance_stops_at_a_stationary_start() {
        let bounds = boxed(-1.0, 1.0, 3);
        let f = |x: ArrayView1<f64>| x.iter().map(|v| v * v).sum::<f64>();
        let mut engine = FdBfgs::new(
            Array1::zeros(3).view(),
            Some(0.0),
            &bounds,
            FdBfgsOptions {
                gtol: 1e-5,
                ..FdBfgsOptions::default()
            },
        );
        let used = drive(&mut engine, f, 100, &bounds);
        assert!(engine.is_done());
        assert_eq!(used, 3, "one forward stencil, then stop");
    }

    #[test]
    fn non_finite_start_finishes_immediately() {
        let bounds = boxed(-1.0, 1.0, 2);
        let mut engine = FdBfgs::new(
            Array1::zeros(2).view(),
            Some(f64::NAN),
            &bounds,
            FdBfgsOptions::default(),
        );
        assert!(engine.is_done());
        engine.restart_at(Array1::from_vec(vec![0.5, 0.5]).view(), Some(1.0), true);
        assert!(!engine.is_done());
        engine.restart_at(Array1::from_vec(vec![0.25, 0.5]).view(), None, true);
        assert_eq!(engine.ask(), Array1::from_vec(vec![0.25, 0.5]));
    }

    /// log10 of the mean squared residual of a linear fit whose singular
    /// values span 3.5 decades, floored at 1e-16 like `binary_lj_fit`.
    fn stiff_fit(x: ArrayView1<f64>) -> f64 {
        let scales = [1e3, 3e1, 1.0, 0.3];
        let mut total = 0.0;
        for (k, scale) in scales.iter().enumerate() {
            let residual: f64 = (0..4)
                .map(|j| {
                    let weight = if j <= k { 1.0 } else { 0.5 };
                    weight * (x[j] - 0.4 - 0.1 * j as f64)
                })
                .sum();
            total += (scale * residual).powi(2);
        }
        (total / 4.0 + 1e-16).log10()
    }

    #[test]
    fn central_intervals_follow_a_narrow_valley() {
        // The eps^(1/3) interval straddles the valley floor once the fit is
        // near 1e-8 and the descent stalls there; intervals adapted to the
        // measured curvature keep the slopes honest to near the 1e-16 floor.
        let bounds = boxed(0.0, 1.0, 4);
        let mut engine = FdBfgs::new(
            Array1::from_elem(4, 0.9).view(),
            None,
            &bounds,
            FdBfgsOptions::default(),
        );
        drive(&mut engine, stiff_fit, 1500, &bounds);
        assert!(engine.value() < -14.0, "value {}", engine.value());
        let stiff = engine.intervals[0].expect("central stencils ran");
        let cube_root = f64::EPSILON.cbrt() * engine.typical(0);
        assert!(
            stiff < 0.5 * cube_root,
            "interval {stiff} against {cube_root}"
        );
    }

    #[test]
    fn adapted_intervals_stay_between_the_rounding_floor_and_the_cube_root_rule() {
        let bounds = boxed(-3.0, 3.0, 3);
        let f = |x: ArrayView1<f64>| x.iter().map(|v| (v - 0.7).powi(2)).sum::<f64>();
        let mut engine = FdBfgs::new(
            Array1::from_elem(3, 2.0).view(),
            None,
            &bounds,
            FdBfgsOptions {
                ftol: 0.0,
                ..FdBfgsOptions::default()
            },
        );
        drive(&mut engine, f, 2000, &bounds);
        assert_eq!(engine.scheme(), FdScheme::Central);
        for i in 0..3 {
            let h = engine.intervals[i].expect("central stencils ran");
            let typical = engine.typical(i);
            assert!(h <= f64::EPSILON.cbrt() * typical, "interval {h}");
            assert!(h >= f64::EPSILON.powf(2.0 / 3.0) * typical, "interval {h}");
        }
    }
}
