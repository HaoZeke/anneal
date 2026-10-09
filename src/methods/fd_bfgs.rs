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
//! The step is projected BFGS. Coordinates held at a bound by the gradient are
//! fixed, the free block of the dense inverse-Hessian approximation scales the
//! free gradient, and a backtracking line search along the projected arc
//! accepts the Armijo condition, shrinking by safeguarded quadratic
//! interpolation. A first trial that is accepted while the decrease is still
//! nearly linear gets one longer trial at the minimiser of the quadratic
//! through the two values and the slope: the long steps a Wolfe search finds
//! by extrapolation, without paying for a gradient at every trial. Pairs that
//! violate the curvature condition are skipped, and the first accepted pair
//! scales the initial matrix by `s.y / y.y`.

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
/// Largest first step, as a fraction of each coordinate's box width, before
/// any curvature pair has scaled the inverse Hessian.
const FIRST_STEP_FRACTION: f64 = 0.05;

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
}

impl Default for FdBfgsOptions {
    /// Descend as deep as finite differences allow.
    fn default() -> Self {
        Self {
            ftol: 1e-12,
            patience: 3,
            max_iter: 0,
            refine: true,
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
        let mut engine = Self {
            n,
            low: bounds.low.clone(),
            high: bounds.high.clone(),
            width,
            x: bounds.clip(x0),
            f: f64::INFINITY,
            grad: Array1::zeros(n),
            inv_hessian: Array2::eye(n),
            scaled: false,
            gamma: 1.0,
            scheme: FdScheme::Forward,
            phase: Phase::Value,
            pair_from: None,
            iterations: 0,
            slow: 0,
            options,
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
        if !keep_curvature {
            self.inv_hessian = Array2::eye(self.n);
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
        let relative = match self.scheme {
            FdScheme::Forward => f64::EPSILON.sqrt(),
            FdScheme::Central => f64::EPSILON.cbrt(),
        };
        let h = (relative * self.typical(i)).min(0.25 * self.width[i]);
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
        work.coord += 1;
        self.load_coord(&mut work);
        if work.coord < self.n {
            self.phase = Phase::Gradient(work);
        } else {
            self.finish_gradient(work.grad);
        }
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
        self.begin_line_search(false);
    }

    fn bfgs_update(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        let sy = s.dot(y);
        let s_norm = s.dot(s).sqrt();
        let y_norm = y.dot(y).sqrt();
        if !(sy.is_finite() && sy > 8.0 * f64::EPSILON * s_norm * y_norm) {
            return;
        }
        let yy = y.dot(y);
        self.gamma = sy / yy;
        if !self.scaled {
            self.inv_hessian = Array2::eye(self.n) * self.gamma;
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
                direction[i] = -self.grad[i];
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
            (FIRST_STEP_FRACTION / reach).min(1.0)
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
        let previous = self.f;
        self.pair_from = Some((std::mem::replace(&mut self.x, trial), self.grad.clone()));
        self.f = value;
        self.iterations += 1;
        let scale = previous.abs().max(value.abs()).max(f64::MIN_POSITIVE);
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
}
