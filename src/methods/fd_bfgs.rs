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
//! through the two values and the slope, at most sixteen times the accepted
//! step: the long steps a Wolfe search finds by extrapolation, without paying
//! for a gradient at every trial. A first trial that falls at least as far
//! as the slope predicts leaves the quadratic without a minimiser, and the
//! longest trial then goes at most halfway from the iterate to the nearest
//! bound along the direction. Far from its fit, the logarithm of a squared
//! error is concave along a line, and sixteen times the accepted step can
//! cross most of the box. Projected onto a face and accepted there, such a
//! trial would leave the coordinates it clamped held at their bounds while
//! descending would carry them out of the box, and the descent could settle
//! in a minimum of the face rather than of the fit. Where a face does hold
//! the minimum over the box, holding the trial short of it usually brings
//! the descent there later, so at small budgets it can end higher. A trial
//! at the minimiser of the quadratic is not held: on a quadratic objective
//! it is the minimum along the line. Pairs that violate the curvature
//! condition are skipped.
//!
//! The initial inverse Hessian is the box metric `D = diag(w^2)`, the identity
//! in box-normalised coordinates, so a side that spans decades moves as far,
//! relative to its width, as a unit one. The first accepted pair scales it to
//! `(s.y / y.D.y) D`, and the steepest-descent fallback steps along `-D g`.
//! A quasi-Newton direction along which the gradient does not descend takes
//! that fallback, and the approximation starts over from `(s.y / y.D.y) D`,
//! `s` and `y` the latest pair it took. The updates keep it positive
//! definite in exact arithmetic, but rounding can leave it indefinite once
//! the curvature has grown by many decades along the descent, and kept, an
//! indefinite approximation refuses direction after direction. On a
//! six-parameter binary Lennard-Jones fit to clusters whose atoms are at
//! least 0.75 apart, with the loss `log10(1e-16 + e)`, `e` the mean squared
//! energy error per atom, every direction refused on one seed came from an
//! approximation with negative diagonal entries, the first once `e` had
//! fallen to 6.9e-16, and kept, the descent took 263 steepest-descent
//! steps, each moving the parameters by 1.4e-11 to 4e-11, over its last
//! 3700 evaluations, from -15.74 to -15.84. Of 40 runs of the portfolio on
//! it at 5000 evaluations, 38 reach -15; with the approximation kept, 17 of
//! them ended short of -15.999, and started over, it brings all 38 there.
//!
//! With [`FdBfgsOptions::log_model`] the engine keeps a second model, of
//! `ln f`, from the same steps and the gradient `g / f`, while every value
//! it has started from or accepted is positive. Each curvature pair also
//! tells which reading is nearer quadratic along its step: the trapezoid
//! rule through the two end slopes is exact for a quadratic, and the pair
//! counts for `ln f` when the rule misses the change of `f` more than
//! [`LOG_MODEL_MARGIN`] times as far as the change of `ln f` (read back as
//! a value), and for `f` when it misses the change of `ln f` that much
//! further. Once the model of `f` has taken [`LOG_MODEL_PAIRS`] curvature
//! pairs per coordinate, a pair after which more of the pairs since the
//! models were last reset count for `ln f` than for `f` moves the line
//! searches onto `ln f` (its direction, Armijo test, extrapolation and
//! interpolation). From then until the models are next reset the count no
//! longer decides: each accepted step puts the next line search on the
//! model that predicted its value more closely, `ln f`'s on a tie, so a
//! step that the model of `ln f` predicts less closely moves the next line
//! search back onto `f`, and one that it predicts at least as closely moves
//! it onto `ln f` again. A loss that falls through decades along a curved
//! valley, as a squared error far from its fit does, is far from quadratic
//! in `f` and closer to it in `ln f`, while near a minimum whose value is
//! zero `ln f` is far from quadratic and the model of `f` can predict the
//! steps more closely. The test that moves the steps onto `ln f` reads
//! neither model's curvature: on an ill-conditioned quadratic, while the
//! model of `f` is still learning the curvature, the model of `ln f`
//! predicts many steps more closely without stepping better, yet the
//! trapezoid rule finds `f` quadratic. It counts every pair rather than
//! reading the latest one alone, so the early pairs of a squared error far
//! from its fit still count once the model of `f` has its pairs, and one
//! step along which `ln f` happens to be nearer quadratic does not move the
//! steps of a descent whose pairs found `f` so. Gradients, intervals and
//! the slow test read the values themselves, and the logarithm enters only
//! through ratios of values, so a power of two multiplying the objective
//! still leaves the descent unchanged; a constant added to it changes the
//! logarithm.

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
/// Fraction of the room from the iterate to the nearest bound along the
/// direction that an extrapolated trial may cover when the quadratic has no
/// minimiser. On a six-parameter Lennard-Jones fit with a logarithmic loss
/// and minima on its faces, 26 of 40 runs of the portfolio ended with a mean
/// squared error below `1e-9` in 1000 evaluations with half the room and 4
/// with no cap. Holding every extrapolated trial, 25, 22 and 27 did with a
/// quarter, two fifths and a half of the room, 19 with three fifths and 8
/// with three quarters; with half the room the portfolio then ended higher
/// on rotated ellipsoids of 30 and 50 coordinates in 1000 evaluations and of
/// 50 and 100 in 5000. Where a face holds the minimum over the box, half the
/// room usually brings the descent there later than no cap: on the base-10
/// logarithm of a quadratic form in ten coordinates centred outside the unit
/// cube, the portfolio ended higher in 200 evaluations with half the room
/// than with no cap from 77 of 80 starts, and level with it in 500.
const EXTRAPOLATION_ROOM: f64 = 0.5;
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
/// Factor by which the trapezoid rule along a curvature pair's step must
/// miss the change of one reading, the values or their logarithm, further
/// than the change of the other for the pair to count for the other in a
/// descent with [`FdBfgsOptions::log_model`]. Pairs about as near quadratic
/// in both, as for a loss already on a logarithmic scale, count for neither.
const LOG_MODEL_MARGIN: f64 = 4.0;
/// Curvature pairs per coordinate the model of the values takes before the
/// steps may move to the logarithm. Exact line searches would give a
/// quadratic's Hessian in one pair per coordinate; with these Armijo
/// searches the descent of a quadratic in 12 dimensions with condition
/// number `1e3` turned superlinear only after about two.
const LOG_MODEL_PAIRS: usize = 2;

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
    /// Decrease per iteration at or below which an iteration is slow
    /// whatever the value's scale; zero leaves only the relative test.
    pub fatol: f64,
    /// Whether the descent also models the logarithm of its values while
    /// they stay positive, and steps on that model once more of its steps
    /// have been markedly nearer quadratic in the logarithm than in the
    /// values than the other way round, and from then on after each step
    /// that model predicts at least as closely as the model of the values
    /// (see the module notes).
    pub log_model: bool,
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
            fatol: 0.0,
            log_model: false,
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
    /// Coordinates whose component is known.
    done: usize,
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
    /// Directional derivative of the reading the search steps on.
    slope: f64,
    steepest: bool,
    trial: Array1<f64>,
    fallback: Option<(Array1<f64>, f64)>,
    /// Whether the search reads the logarithm of the values.
    log: bool,
}

/// Quasi-Newton curvature model of one reading of the values.
struct Curvature {
    /// Inverse Hessian approximation: the box metric until the first pair
    /// scales it.
    inv_hessian: Array2<f64>,
    /// Hessian approximation, kept only to predict accepted steps: the
    /// inverse of `inv_hessian` from the first pair on.
    hessian: Option<Array2<f64>>,
    scaled: bool,
    gamma: f64,
    /// Pairs the model has taken since it was last reset; starting the
    /// approximation over keeps the count.
    pairs: usize,
}

impl Curvature {
    fn new(metric: &Array1<f64>) -> Self {
        Self {
            inv_hessian: Array2::from_diag(metric),
            hessian: None,
            scaled: false,
            gamma: 1.0,
            pairs: 0,
        }
    }

    /// BFGS update with the pair `(s, y)`, skipped when it violates the
    /// curvature condition. With `direct` the Hessian approximation follows.
    fn update(&mut self, metric: &Array1<f64>, s: &Array1<f64>, y: &Array1<f64>, direct: bool) {
        let n = s.len();
        let sy = s.dot(y);
        let s_norm = s.dot(s).sqrt();
        let y_norm = y.dot(y).sqrt();
        if !(sy.is_finite() && sy > 8.0 * f64::EPSILON * s_norm * y_norm) {
            return;
        }
        let ydy: f64 = (0..n).map(|i| metric[i] * y[i] * y[i]).sum();
        self.gamma = sy / ydy;
        self.pairs += 1;
        if !self.scaled {
            self.inv_hessian = Array2::from_diag(&(metric * self.gamma));
            self.scaled = true;
            if direct {
                self.hessian = Some(Array2::from_diag(&metric.mapv(|w| 1.0 / (w * self.gamma))));
            }
        }
        let hy = self.inv_hessian.dot(y);
        let yhy = y.dot(&hy);
        let rho = 1.0 / sy;
        let ss = rho * rho * yhy + rho;
        for i in 0..n {
            for j in 0..n {
                self.inv_hessian[[i, j]] += ss * s[i] * s[j] - rho * (s[i] * hy[j] + hy[i] * s[j]);
            }
        }
        if let Some(hessian) = self.hessian.as_mut() {
            let hs = hessian.dot(s);
            let shs = s.dot(&hs);
            if shs > 0.0 && shs.is_finite() {
                for i in 0..n {
                    for j in 0..n {
                        hessian[[i, j]] += rho * y[i] * y[j] - hs[i] * hs[j] / shs;
                    }
                }
            }
        }
    }

    /// Starts the approximation over from the box metric scaled by the
    /// latest pair, `(s.y / y.D.y) D`, keeping the count of pairs.
    fn start_over(&mut self, metric: &Array1<f64>) {
        self.inv_hessian = Array2::from_diag(&(metric * self.gamma));
        if self.hessian.is_some() {
            self.hessian = Some(Array2::from_diag(&metric.mapv(|w| 1.0 / (w * self.gamma))));
        }
    }

    /// Second-order term `s.B.s / 2` of the model's change along `s`.
    fn curvature_term(&self, s: &Array1<f64>) -> Option<f64> {
        self.hessian
            .as_ref()
            .map(|hessian| 0.5 * s.dot(&hessian.dot(s)))
    }
}

/// Logarithm of `value / base`, minus infinity when the ratio is not
/// positive.
fn log_ratio(value: f64, base: f64) -> f64 {
    let ratio = value / base;
    if ratio > 0.0 {
        ratio.ln()
    } else {
        f64::NEG_INFINITY
    }
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
    /// Model of the values.
    curvature: Curvature,
    /// Model of their logarithm, with [`FdBfgsOptions::log_model`].
    log_curvature: Curvature,
    /// Whether every value the descent has started from or accepted is
    /// positive, so the logarithm is defined along its path.
    log_valid: bool,
    /// Whether the line searches step on the model of the logarithm.
    log_active: bool,
    /// Whether the count has moved the line searches onto the logarithm
    /// since the models were last reset; from then on each accepted step
    /// chooses the model the next line search steps on.
    log_engaged: bool,
    /// Curvature pairs since the models were last reset along which the
    /// trapezoid rule missed the change of the values more than
    /// [`LOG_MODEL_MARGIN`] times as far as the change of their logarithm.
    log_nearer: usize,
    /// Pairs since then along which it missed the change of the logarithm
    /// more than [`LOG_MODEL_MARGIN`] times as far as that of the values.
    values_nearer: usize,
    scheme: FdScheme,
    phase: Phase,
    pair_from: Option<(Array1<f64>, Array1<f64>)>,
    /// Value at the point the pending curvature pair starts from.
    pair_value: f64,
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
    /// Coordinate each gradient differences first.
    first: usize,
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
            curvature: Curvature::new(&metric),
            log_curvature: Curvature::new(&metric),
            metric,
            log_valid: true,
            log_active: false,
            log_engaged: false,
            log_nearer: 0,
            values_nearer: 0,
            scheme: FdScheme::Forward,
            phase: Phase::Value,
            pair_from: None,
            pair_value: f64::NAN,
            iterations: 0,
            slow: 0,
            options,
            intervals: vec![None; n],
            used_intervals: vec![0.0; n],
            redifferences: 0,
            first: 0,
        };
        if let Some(f0) = f0 {
            engine.set_start_value(f0);
        }
        engine
    }

    /// Differences every gradient from coordinate `first` (modulo the
    /// dimension) on, cyclically, instead of from coordinate 0, and starts a
    /// gradient under way again in that order. The gradient is the same in
    /// any order, so descents that share their evaluations can each start
    /// at a different coordinate and ask for different stencil points.
    pub fn with_first_coordinate(mut self, first: usize) -> Self {
        self.first = first % self.n;
        if matches!(self.phase, Phase::Gradient(_)) {
            self.phase = self.gradient_phase();
        }
        self
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
            self.curvature = Curvature::new(&self.metric);
            self.log_curvature = Curvature::new(&self.metric);
            self.log_active = false;
            self.log_engaged = false;
            self.log_nearer = 0;
            self.values_nearer = 0;
        }
        match f {
            Some(f) => self.set_start_value(f),
            None => {
                self.f = f64::INFINITY;
                self.phase = Phase::Value;
            }
        }
    }

    /// True once the descent has converged or exhausted its options, or at
    /// its start when no side of the box is wide enough for a difference
    /// step to change its coordinate.
    pub fn is_done(&self) -> bool {
        matches!(self.phase, Phase::Done)
    }

    /// True when `bounds` leave some coordinate of `x` a side wide enough
    /// for a difference step to change it. When none has one, as when each
    /// side is at most a unit in the last place wide or so near zero that
    /// the step underflows, a descent from `x` asks for nothing but `x`.
    pub fn can_difference(x: ArrayView1<f64>, bounds: &Bounds<f64>) -> bool {
        let engine = Self::new(x, None, bounds, FdBfgsOptions::default());
        engine.gradient_work().done < engine.n
    }

    /// Sets [`FdBfgsOptions::fatol`] for the iterations that follow.
    pub fn set_fatol(&mut self, fatol: f64) {
        self.options.fatol = fatol;
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
        if !(value > 0.0) {
            self.log_valid = false;
            self.log_active = false;
        }
        self.f = value;
        self.phase = if value.is_finite() {
            self.gradient_phase()
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
        // A central step that rounds away would ask the iterate itself, or
        // one point twice. The one-sided rule then takes the side, failing
        // that the forward rule, which asks nothing when its step does too.
        if self.scheme == FdScheme::Central && room_up >= h && room_down >= h {
            let hp = realized(h);
            let hm = realized(-h);
            if hp != 0.0 && hm != 0.0 {
                let span = hp - hm;
                return (vec![hp, hm], vec![1.0 / span, -1.0 / span], 0.0);
            }
        }
        if self.scheme == FdScheme::Central && room_up.max(room_down) >= 2.0 * h {
            let sign = if room_up >= 2.0 * h { 1.0 } else { -1.0 };
            let h1 = realized(sign * h);
            let h2 = realized(2.0 * sign * h);
            if h1 != 0.0 && h2 != h1 {
                let w0 = -(h1 + h2) / (h1 * h2);
                let w1 = h2 / (h1 * (h2 - h1));
                let w2 = -h1 / (h2 * (h2 - h1));
                return (vec![h1, h2], vec![w1, w2], w0);
            }
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
            done: 0,
            coord: self.first,
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

    /// A gradient at the iterate, or the end of the descent when no
    /// coordinate admits a difference step, so that nothing is asked.
    fn gradient_phase(&self) -> Phase {
        let work = self.gradient_work();
        if work.done < self.n {
            Phase::Gradient(work)
        } else {
            Phase::Done
        }
    }

    /// Loads the stencil of the next coordinate in the gradient's order,
    /// skipping degenerate coordinates.
    fn load_coord(&self, work: &mut GradientWork) {
        while work.done < self.n {
            work.coord = (self.first + work.done) % self.n;
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
            work.done += 1;
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
        work.done += 1;
        self.load_coord(&mut work);
        if work.done < self.n {
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
            let direct = self.options.log_model;
            self.curvature.update(&self.metric, &s, &y, direct);
            if direct && self.log_valid {
                let y = &self.grad / self.f - &g_prev / self.pair_value;
                self.log_curvature.update(&self.metric, &s, &y, true);
                self.weigh_trapezoid(&s, &g_prev);
                if !self.log_engaged && self.curvature.pairs >= LOG_MODEL_PAIRS * self.n {
                    self.log_active = self.log_nearer > self.values_nearer;
                    self.log_engaged = self.log_active;
                }
            }
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

    /// Model the line searches step on.
    fn stepping(&self) -> &Curvature {
        if self.log_active {
            &self.log_curvature
        } else {
            &self.curvature
        }
    }

    fn free(&self, i: usize) -> bool {
        let g = self.grad[i];
        !((self.x[i] <= self.low[i] && g > 0.0) || (self.x[i] >= self.high[i] && g < 0.0))
    }

    fn begin_line_search(&mut self, steepest: bool) {
        let log = self.log_active;
        let model = self.stepping();
        let grad = if log {
            &self.grad / self.f
        } else {
            self.grad.clone()
        };
        let free: Vec<usize> = (0..self.n).filter(|&i| self.free(i)).collect();
        let mut direction = Array1::<f64>::zeros(self.n);
        if !steepest {
            for &i in &free {
                direction[i] = -free
                    .iter()
                    .map(|&j| model.inv_hessian[[i, j]] * grad[j])
                    .sum::<f64>();
            }
        }
        let mut slope = grad.dot(&direction);
        let descends = slope.is_finite() && slope < 0.0;
        let refused = !(steepest || descends);
        let mut steepest = steepest;
        if steepest || refused {
            steepest = true;
            direction.fill(0.0);
            for &i in &free {
                direction[i] = -self.metric[i] * grad[i];
            }
            slope = grad.dot(&direction);
        }
        if !(slope.is_finite() && slope < 0.0) {
            self.phase = Phase::Done;
            return;
        }
        let alpha = if !model.scaled {
            let reach = (0..self.n)
                .map(|i| direction[i].abs() / self.width[i].max(f64::MIN_POSITIVE))
                .fold(0.0_f64, f64::max);
            // A finite step keeps the held coordinates' zero components at
            // zero when the reach is subnormal.
            (FIRST_STEP_FRACTION / reach).min(f64::MAX)
        } else if steepest {
            model.gamma
        } else {
            1.0
        };
        if refused {
            let model = if log {
                &mut self.log_curvature
            } else {
                &mut self.curvature
            };
            model.start_over(&self.metric);
        }
        let trial = self.projected(&direction, alpha);
        self.phase = Phase::Line(LineWork {
            direction,
            alpha,
            trials: 0,
            slope,
            steepest,
            trial,
            fallback: None,
            log,
        });
    }

    fn projected(&self, direction: &Array1<f64>, alpha: f64) -> Array1<f64> {
        Array1::from_iter(
            (0..self.n)
                .map(|i| (self.x[i] + alpha * direction[i]).clamp(self.low[i], self.high[i])),
        )
    }

    /// Smallest step along `direction` from the iterate at which a coordinate
    /// reaches the bound its component heads for, lower or upper, over every
    /// coordinate the direction moves. A coordinate already on that bound
    /// makes it zero, as a free coordinate on its bound does when the
    /// quasi-Newton direction points out of the box there, and as the
    /// extrapolated step is never shorter than the first trial's, no longer
    /// trial is then asked. Measured instead to the first positive breakpoint
    /// of the projected arc, which runs on along the face with that
    /// coordinate clamped, the room would let the trial carry the descent
    /// along the face, where it could settle in a minimum of the face: on a
    /// six-parameter Lennard-Jones fit whose loss is the logarithm of a mean
    /// squared error, one of 40 runs of the portfolio then ended at -1.43 in
    /// 1000 evaluations, as with no cap, rather than at -4.82.
    fn room(&self, direction: &Array1<f64>) -> f64 {
        (0..self.n)
            .map(|i| {
                let d = direction[i];
                if d > 0.0 {
                    (self.high[i] - self.x[i]) / d
                } else if d < 0.0 {
                    (self.low[i] - self.x[i]) / d
                } else {
                    f64::INFINITY
                }
            })
            .fold(f64::INFINITY, f64::min)
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
        let linear = self.grad.dot(&moved);
        // The strict decrease matters once the predicted decrease rounds
        // away: the Armijo bound then equals f and would accept a null step.
        let sufficient = if work.log {
            log_ratio(value, self.f) <= ARMIJO * (linear / self.f)
        } else {
            value <= self.f + ARMIJO * linear
        };
        if value.is_finite() && value < self.f && sufficient {
            if work.trials == 0 {
                let (realised, predicted) = if work.log {
                    (-log_ratio(value, self.f), -(linear / self.f))
                } else {
                    (self.f - value, -linear)
                };
                let ratio = realised / predicted;
                if predicted > 0.0 && ratio > EXTRAPOLATE_RATIO {
                    let next = if ratio >= 1.0 {
                        let reach = EXTRAPOLATION_ROOM * self.room(&work.direction);
                        (work.alpha * MAX_EXTRAPOLATION).min(reach.max(work.alpha))
                    } else {
                        work.alpha * (0.5 / (1.0 - ratio)).min(MAX_EXTRAPOLATION)
                    };
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
            let change = if work.log {
                log_ratio(value, self.f)
            } else {
                value - self.f
            };
            let denom = change - work.slope * alpha;
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

    /// Counts the step `s` just taken for the logarithm when the trapezoid
    /// rule through its end slopes, exact for a quadratic, misses the change
    /// of the values more than [`LOG_MODEL_MARGIN`] times as far as the
    /// change of their logarithm, read back as a value, and for the values
    /// when it misses the logarithm's that much further. `g_prev` is the
    /// gradient at the step's start.
    fn weigh_trapezoid(&mut self, s: &Array1<f64>, g_prev: &Array1<f64>) {
        let (from, to) = (g_prev.dot(s), self.grad.dot(s));
        let miss = (self.f - self.pair_value - 0.5 * (from + to)).abs();
        let log_slope = 0.5 * (from / self.pair_value + to / self.f);
        let log_miss = (self.f - self.pair_value * log_slope.exp()).abs();
        if miss > LOG_MODEL_MARGIN * log_miss {
            self.log_nearer += 1;
        } else if log_miss > LOG_MODEL_MARGIN * miss {
            self.values_nearer += 1;
        }
    }

    /// Once the count has moved the steps onto the model of the logarithm,
    /// puts the next line search on that model when it predicts the
    /// accepted step's value at least as closely as the model of the values,
    /// and on the model of the values otherwise. A value at or below zero
    /// retires the model of the logarithm.
    fn choose_log_steps(&mut self, trial: &Array1<f64>, value: f64) {
        if !(value > 0.0) {
            self.log_valid = false;
            self.log_active = false;
        }
        if !(self.log_valid && self.log_engaged) {
            return;
        }
        let s = trial - &self.x;
        let (Some(term), Some(log_term)) = (
            self.curvature.curvature_term(&s),
            self.log_curvature.curvature_term(&s),
        ) else {
            return;
        };
        let linear = self.grad.dot(&s);
        let predicted = self.f + linear + term;
        let log_predicted = self.f * (linear / self.f + log_term).exp();
        self.log_active = (log_predicted - value).abs() <= (predicted - value).abs();
    }

    fn accept(&mut self, trial: Array1<f64>, value: f64) {
        if self.options.log_model {
            self.choose_log_steps(&trial, value);
        }
        self.redifferences = 0;
        let previous = self.f;
        self.pair_value = previous;
        self.pair_from = Some((std::mem::replace(&mut self.x, trial), self.grad.clone()));
        self.f = value;
        self.iterations += 1;
        let scale = previous
            .abs()
            .max(value.abs())
            .max(self.options.value_floor);
        if (previous - value) <= (self.options.ftol * scale).max(self.options.fatol) {
            if self.scheme == FdScheme::Forward && self.options.refine {
                // A forward-difference direction that only buys a sliver of
                // decrease is usually its truncation error talking.
                self.scheme = FdScheme::Central;
                self.pair_from = None;
                self.slow = 0;
                self.phase = self.gradient_phase();
                return;
            }
            self.slow += 1;
        } else {
            self.slow = 0;
        }
        self.phase = if self.slow >= self.options.patience.max(1) {
            Phase::Done
        } else {
            self.gradient_phase()
        };
    }

    fn line_search_failed(&mut self, steepest: bool) {
        if self.scheme == FdScheme::Central
            && self.redifferences < MAX_REDIFFERENCES
            && self.intervals_moved()
        {
            self.redifferences += 1;
            self.pair_from = None;
            self.phase = self.gradient_phase();
            return;
        }
        match self.scheme {
            FdScheme::Forward if !self.options.refine => self.phase = Phase::Done,
            FdScheme::Forward => {
                self.scheme = FdScheme::Central;
                self.pair_from = None;
                self.phase = self.gradient_phase();
            }
            FdScheme::Central if !steepest && self.stepping().scaled => {
                self.begin_line_search(true)
            }
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
        for log_model in [false, true] {
            let options = FdBfgsOptions {
                log_model,
                ..FdBfgsOptions::default()
            };
            let mut engine = FdBfgs::new(start.view(), None, &bounds, options);
            drive(&mut engine, rosenbrock, 4000, &bounds);
            assert!(
                engine.value() < 1e-10,
                "log model {log_model}: value {}",
                engine.value()
            );
        }
    }

    /// A bowl whose logarithm is quadratic: in four dimensions it falls
    /// through nine decades from the tests' start to its minimum of one at
    /// 0.3.
    fn exp_bowl(x: ArrayView1<f64>) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, v)| (1 + i) as f64 * (v - 0.3).powi(2))
            .sum::<f64>()
            .exp()
    }

    /// Evaluations, and whether a line search stepped on the logarithm,
    /// until the descent's value first drops below `target`.
    fn log_steps_to<F: Fn(ArrayView1<f64>) -> f64>(
        engine: &mut FdBfgs,
        f: F,
        target: f64,
        budget: usize,
    ) -> (usize, bool) {
        let (mut used, mut log) = (0, false);
        while used < budget && !engine.is_done() && engine.value() >= target {
            let x = engine.ask();
            engine.tell(f(x.view()));
            log |= engine.log_active;
            used += 1;
        }
        (used, log)
    }

    #[test]
    fn the_logarithm_takes_the_steps_where_it_is_nearer_quadratic() {
        let bounds = boxed(-2.0, 2.0, 4);
        let start = Array1::from_vec(vec![-1.5, 1.5, -1.5, 1.5]);
        let run = |log_model: bool| {
            let options = FdBfgsOptions {
                log_model,
                ..FdBfgsOptions::default()
            };
            let mut engine = FdBfgs::new(start.view(), None, &bounds, options);
            log_steps_to(&mut engine, exp_bowl, 1.0 + 1e-8, 4000)
        };
        let (plain, plain_log) = run(false);
        let (dual, dual_log) = run(true);
        assert!(!plain_log);
        assert!(dual_log, "the logarithm never took the steps");
        assert!(
            3 * dual < plain,
            "{dual} evaluations with the model, {plain} without"
        );
    }

    #[test]
    fn an_ill_conditioned_quadratic_keeps_the_steps_on_the_values() {
        // Its minimum value is zero, and its condition number of 1e6 keeps
        // BFGS learning the curvature for many steps.
        let n = 12;
        let bounds = boxed(-5.0, 5.0, n);
        let start = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { -4.0 } else { 4.5 }));
        let ellipsoid = |x: ArrayView1<f64>| {
            x.iter()
                .enumerate()
                .map(|(i, v)| 10f64.powf(6.0 * i as f64 / (n - 1) as f64) * (v - 0.3).powi(2))
                .sum::<f64>()
        };
        let trace = |log_model: bool| {
            let options = FdBfgsOptions {
                log_model,
                ..FdBfgsOptions::default()
            };
            let mut engine = FdBfgs::new(start.view(), None, &bounds, options);
            let (mut asked, mut log) = (Vec::new(), false);
            while asked.len() < 4000 && !engine.is_done() {
                let x = engine.ask();
                engine.tell(ellipsoid(x.view()));
                log |= engine.log_active;
                asked.push(x);
            }
            (asked, engine.value(), log)
        };
        let (plain, value, _) = trace(false);
        let (dual, dual_value, log) = trace(true);
        assert!(!log, "the logarithm took the steps");
        assert_eq!(dual, plain);
        assert_eq!(dual_value, value);
        assert!(value < 1e-10, "value {value}");
    }

    /// The pairs for the logarithm and for the values that one step from
    /// `from` to `to` along `f`, whose derivative is `g`, adds to the count.
    fn counts(f: fn(f64) -> f64, g: fn(f64) -> f64, from: f64, to: f64) -> (usize, usize) {
        let options = FdBfgsOptions {
            log_model: true,
            ..FdBfgsOptions::default()
        };
        let at = Array1::from_elem(1, to);
        let mut engine = FdBfgs::new(at.view(), Some(f(to)), &boxed(-5.0, 5.0, 1), options);
        engine.pair_value = f(from);
        engine.grad = Array1::from_elem(1, g(to));
        let step = Array1::from_elem(1, to - from);
        engine.weigh_trapezoid(&step, &Array1::from_elem(1, g(from)));
        (engine.log_nearer, engine.values_nearer)
    }

    #[test]
    fn each_pair_counts_for_the_reading_nearer_quadratic_along_it() {
        // The trapezoid rule is exact for x^2 and for the logarithm of
        // exp(x^2); along x^3 it misses both by about as much.
        assert_eq!(counts(|x| x * x, |x| 2.0 * x, 2.0, 1.0), (0, 1));
        let exp_square = |x: f64| (x * x).exp();
        assert_eq!(
            counts(exp_square, |x| 2.0 * x * (x * x).exp(), 1.0, 0.5),
            (1, 0)
        );
        assert_eq!(counts(|x| x.powi(3), |x| 3.0 * x * x, 2.0, 1.5), (0, 0));
    }

    #[test]
    fn a_pair_counts_for_one_reading_only_where_the_other_misses_four_times_as_far() {
        // From 2 to 1 along x^3 the trapezoid rule misses the change of the
        // values 3.2 times as far as the change of the logarithm, and 15
        // times along x^4; along the square root it misses the logarithm's
        // 2.2 times as far as the values', and 4.6 times along x^(3/2).
        // Only the factors over four count.
        assert_eq!(counts(|x| x.powi(3), |x| 3.0 * x * x, 2.0, 1.0), (0, 0));
        assert_eq!(counts(|x| x.powi(4), |x| 4.0 * x.powi(3), 2.0, 1.0), (1, 0));
        assert_eq!(counts(f64::sqrt, |x| 0.5 / x.sqrt(), 2.0, 1.0), (0, 0));
        assert_eq!(
            counts(|x| x * x.sqrt(), |x| 1.5 * x.sqrt(), 2.0, 1.0),
            (0, 1)
        );
    }

    #[test]
    fn a_pair_counts_only_where_one_reading_misses_over_four_times_as_far() {
        // Along each unit step below the values fall from one to a half and
        // the end slopes of the logarithm cancel, so its trapezoid rule
        // predicts no change and misses by a half. With the slope `g` at the
        // start, the rule on the values misses by `|1 + g / 2|` times that.
        // Missing four times as far, or a quarter as far, counts for neither
        // reading; a little further counts for the logarithm, and a little
        // less far for the values.
        let options = FdBfgsOptions {
            log_model: true,
            ..FdBfgsOptions::default()
        };
        let weigh = |slope: f64| {
            let at = Array1::from_elem(1, 1.0);
            let mut engine = FdBfgs::new(at.view(), Some(0.5), &boxed(-5.0, 5.0, 1), options);
            engine.pair_value = 1.0;
            engine.grad = Array1::from_elem(1, -0.5 * slope);
            let step = Array1::from_elem(1, 1.0);
            engine.weigh_trapezoid(&step, &Array1::from_elem(1, slope));
            (engine.log_nearer, engine.values_nearer)
        };
        assert_eq!(weigh(-10.0), (0, 0));
        assert_eq!(weigh(-10.0078125), (1, 0));
        assert_eq!(weigh(-2.5), (0, 0));
        assert_eq!(weigh(-2.4921875), (0, 1));
    }

    /// The point, value and slope before and after a step along the first
    /// coordinate of a descent in two, along which the trapezoid rule finds
    /// the logarithm nearer quadratic (`exp(x^2)`, `'l'`), the values (`x^2`,
    /// `'v'`) or neither (`x^3`, `'n'`), as in the test above.
    fn pair(kind: char) -> ([f64; 3], [f64; 3]) {
        match kind {
            'l' => {
                let (e, root) = (1f64.exp(), 0.25f64.exp());
                ([1.0, e, 2.0 * e], [0.5, root, root])
            }
            'v' => ([2.0, 4.0, 4.0], [1.0, 1.0, 2.0]),
            _ => ([2.0, 8.0, 12.0], [1.5, 3.375, 6.75]),
        }
    }

    fn two_coordinate_descent() -> FdBfgs {
        let options = FdBfgsOptions {
            log_model: true,
            ..FdBfgsOptions::default()
        };
        let start = Array1::zeros(2);
        FdBfgs::new(start.view(), Some(1.0), &boxed(-5.0, 5.0, 2), options)
    }

    /// Closes the curvature pair of each of `kinds` ([`pair`]) as the
    /// gradient after an accepted step does, and says after each whether
    /// the line search that follows steps on the logarithm.
    fn steps_after(engine: &mut FdBfgs, kinds: &str) -> Vec<bool> {
        let along = |a: f64| Array1::from_vec(vec![a, 0.0]);
        kinds
            .chars()
            .map(|kind| {
                let (from, to) = pair(kind);
                engine.pair_from = Some((along(from[0]), along(from[2])));
                engine.pair_value = from[1];
                engine.x = along(to[0]);
                engine.f = to[1];
                engine.finish_gradient(along(to[2]));
                match &engine.phase {
                    Phase::Line(work) => work.log,
                    _ => panic!("no line search followed the pair"),
                }
            })
            .collect()
    }

    #[test]
    fn the_steps_reach_the_logarithm_from_two_pairs_per_coordinate() {
        // In two coordinates the fourth pair is the first that may move the
        // steps; from one pair per coordinate the second would, and the
        // first with none.
        let mut engine = two_coordinate_descent();
        let log = steps_after(&mut engine, "llln");
        assert_eq!(log, [false, false, false, true]);
    }

    #[test]
    fn every_pair_since_the_models_were_reset_counts_not_the_latest_alone() {
        // Above, three pairs for the logarithm before two per coordinate
        // move the steps at a fourth for neither. Here two for the values
        // and one for neither leave the count with the values at a fourth
        // for the logarithm and level at a fifth, and a sixth tips it.
        let mut engine = two_coordinate_descent();
        let log = steps_after(&mut engine, "vvnlll");
        assert_eq!(log, [false, false, false, false, false, true]);
        // A reset clears the count: four pairs for neither then leave the
        // steps on the values, where the count kept would have moved them.
        let at = Array1::from_vec(vec![0.5, 0.0]);
        engine.restart_at(at.view(), Some(1.0), false);
        assert_eq!(steps_after(&mut engine, "nnnn"), [false; 4]);
    }

    /// What the model of the values, or with `log` the model of their
    /// logarithm, predicts for the value at `trial`.
    fn predicted(engine: &FdBfgs, trial: &Array1<f64>, log: bool) -> f64 {
        let s = trial - &engine.x;
        let linear = engine.grad.dot(&s);
        if log {
            let term = engine.log_curvature.curvature_term(&s).expect("a pair");
            engine.f * (linear / engine.f + term).exp()
        } else {
            engine.f + linear + engine.curvature.curvature_term(&s).expect("a pair")
        }
    }

    #[test]
    fn each_accepted_step_puts_the_next_line_search_on_the_nearer_model() {
        // Four pairs move the steps onto the logarithm. A step down the
        // first coordinate, accepted at the value the model of the values
        // predicts there, moves the next line search back onto the values,
        // and the steps stay there through five more pairs, after which
        // more pairs count for the values than for the logarithm. A step
        // accepted at the value the model of the logarithm predicts then
        // puts the next line search on the logarithm again.
        let mut engine = two_coordinate_descent();
        let log = steps_after(&mut engine, "llln");
        assert_eq!(log, [false, false, false, true]);
        let trial = Array1::from_vec(vec![1.0, 0.0]);
        let (values, logarithm) = (
            predicted(&engine, &trial, false),
            predicted(&engine, &trial, true),
        );
        assert!(0.0 < values && values < logarithm && logarithm < engine.f);
        engine.accept(trial, values);
        assert_eq!(steps_after(&mut engine, "vvvvv"), [false; 5]);
        assert!(engine.values_nearer > engine.log_nearer);
        let trial = Array1::from_vec(vec![0.75, 0.0]);
        let (values, logarithm) = (
            predicted(&engine, &trial, false),
            predicted(&engine, &trial, true),
        );
        assert!(0.0 < values && values < logarithm && logarithm < engine.f);
        engine.accept(trial, logarithm);
        assert_eq!(steps_after(&mut engine, "v"), [true]);
    }

    #[test]
    fn a_tie_between_the_models_puts_the_next_line_search_on_the_logarithm() {
        // A step of 1e-9 along the second coordinate, where the gradient
        // vanishes, is too short for either model's curvature to move its
        // prediction off the iterate's value, so both models miss the value
        // the step reaches by as much. Accepted while the steps are on the
        // values, it puts the next line search on the logarithm.
        let mut engine = two_coordinate_descent();
        assert_eq!(
            steps_after(&mut engine, "llln"),
            [false, false, false, true]
        );
        let trial = Array1::from_vec(vec![1.0, 0.0]);
        let values = predicted(&engine, &trial, false);
        engine.accept(trial, values);
        assert_eq!(steps_after(&mut engine, "v"), [false]);
        let trial = Array1::from_vec(vec![engine.x[0], 1e-9]);
        assert_eq!(predicted(&engine, &trial, false), engine.f);
        assert_eq!(predicted(&engine, &trial, true), engine.f);
        engine.accept(trial, engine.f * (1.0 - f64::EPSILON));
        assert_eq!(steps_after(&mut engine, "v"), [true]);
    }

    #[test]
    fn a_tie_accepted_on_the_logarithm_keeps_the_next_line_search_there() {
        // The same step, accepted while the steps are on the logarithm,
        // leaves the next line search there: a tie goes to the logarithm
        // whichever model the steps are on. The tie rests only on exp
        // returning exactly 1 for arguments below 2^-54.
        let mut engine = two_coordinate_descent();
        assert_eq!(
            steps_after(&mut engine, "llln"),
            [false, false, false, true]
        );
        let trial = Array1::from_vec(vec![engine.x[0], 1e-9]);
        assert_eq!(predicted(&engine, &trial, false), engine.f);
        assert_eq!(predicted(&engine, &trial, true), engine.f);
        engine.accept(trial, engine.f * (1.0 - f64::EPSILON));
        assert_eq!(steps_after(&mut engine, "v"), [true]);
    }

    #[test]
    fn the_count_hands_the_choice_to_the_steps_until_a_reset() {
        // Before the count moves the steps onto the logarithm, a step
        // accepted at the value the model of the logarithm predicts leaves
        // the next line search on the values. Once the count has moved them,
        // a reset hands the choice back to it, and four pairs move the steps
        // as they did from the start.
        let mut engine = two_coordinate_descent();
        assert_eq!(steps_after(&mut engine, "ll"), [false; 2]);
        let trial = Array1::from_vec(vec![0.25, 0.0]);
        let logarithm = predicted(&engine, &trial, true);
        assert!(0.0 < logarithm && logarithm != predicted(&engine, &trial, false));
        engine.accept(trial, logarithm);
        assert_eq!(steps_after(&mut engine, "ln"), [false, true]);
        let at = Array1::from_vec(vec![0.5, 0.0]);
        engine.restart_at(at.view(), Some(1.0), false);
        let log = steps_after(&mut engine, "llln");
        assert_eq!(log, [false, false, false, true]);
    }

    #[test]
    fn a_value_at_or_below_zero_retires_the_logarithm_model() {
        let bounds = boxed(-2.0, 2.0, 4);
        let start = Array1::from_vec(vec![-1.5, 1.5, -1.5, 1.5]);
        let options = FdBfgsOptions {
            log_model: true,
            ..FdBfgsOptions::default()
        };
        let mut engine = FdBfgs::new(start.view(), None, &bounds, options);
        let shifted = |x: ArrayView1<f64>| exp_bowl(x) - 2.0;
        let (_, log) = log_steps_to(&mut engine, shifted, 0.0, 4000);
        assert!(log, "the logarithm never took the steps");
        assert!(!engine.log_valid && !engine.log_active);
        drive(&mut engine, shifted, 4000, &bounds);
        assert!(!engine.log_active);
        assert!(engine.value() < -1.0 + 1e-12, "value {}", engine.value());
    }

    #[test]
    fn a_start_at_or_below_zero_retires_the_logarithm_model() {
        // The portfolio restarts a finished descent at a kicked point with
        // its curvature kept, and the value there starts the next descent.
        // After the count has moved the steps onto the logarithm, a start
        // at or below zero retires that model, and the line search that
        // follows the gradient there steps downhill on the values.
        for start in [0.0, -1.0] {
            let mut engine = two_coordinate_descent();
            assert_eq!(
                steps_after(&mut engine, "llln"),
                [false, false, false, true]
            );
            engine.restart_at(Array1::zeros(2).view(), None, true);
            let plane = |x: ArrayView1<f64>| start + x[0] + 2.0 * x[1];
            while matches!(engine.phase, Phase::Value | Phase::Gradient(_)) {
                let x = engine.ask();
                engine.tell(plane(x.view()));
            }
            assert!(!engine.log_valid && !engine.log_active, "start {start}");
            match &engine.phase {
                Phase::Line(work) => assert!(
                    !work.log && engine.grad.dot(&work.direction) < 0.0,
                    "start {start}"
                ),
                _ => panic!("start {start}: no line search followed the gradient"),
            }
        }
    }

    #[test]
    fn a_restart_at_a_positive_value_keeps_the_steps_on_the_logarithm() {
        // The same restart at a start whose value is positive keeps the
        // steps on the logarithm, where the count had moved them: a restart
        // that keeps the curvature keeps the model the steps are on.
        let mut engine = two_coordinate_descent();
        assert_eq!(
            steps_after(&mut engine, "llln"),
            [false, false, false, true]
        );
        engine.restart_at(Array1::zeros(2).view(), None, true);
        let plane = |x: ArrayView1<f64>| 1.0 + x[0] + 2.0 * x[1];
        while matches!(engine.phase, Phase::Value | Phase::Gradient(_)) {
            let x = engine.ask();
            engine.tell(plane(x.view()));
        }
        assert!(engine.log_valid && engine.log_active);
        match &engine.phase {
            Phase::Line(work) => assert!(work.log),
            _ => panic!("no line search followed the gradient"),
        }
    }

    #[test]
    fn a_retired_logarithm_model_takes_no_more_pairs() {
        // A value at or below zero retires the model of the logarithm for
        // the rest of the run: accepted after two pairs for the logarithm,
        // it leaves that model and the count where they were through the
        // pairs that follow. After a restart that keeps the curvature, as
        // the portfolio's kicks do, at a point whose value is positive, a
        // pair for the logarithm and one for neither leave the steps on the
        // values, where from the start the same four pairs moved them onto
        // the logarithm.
        for value in [0.0, -1.0] {
            let mut engine = two_coordinate_descent();
            assert_eq!(steps_after(&mut engine, "ll"), [false; 2]);
            engine.accept(Array1::from_vec(vec![0.25, 0.0]), value);
            assert!(!engine.log_valid, "value {value}");
            let at = Array1::from_vec(vec![0.5, 0.0]);
            engine.restart_at(at.view(), None, true);
            engine.tell(1.0);
            assert_eq!(steps_after(&mut engine, "ln"), [false; 2], "value {value}");
            assert_eq!(engine.log_curvature.pairs, 2, "value {value}");
            let count = (engine.log_nearer, engine.values_nearer);
            assert_eq!(count, (2, 0), "value {value}");
        }
    }

    #[test]
    fn the_logarithm_model_leaves_power_of_two_scaling_invisible() {
        // The logarithm enters only through ratios of values, which a power
        // of two leaves exact.
        let bounds = boxed(-2.0, 2.0, 4);
        let start = Array1::from_vec(vec![-1.5, 1.5, -1.5, 1.5]);
        let trace = |scale: f64| {
            let options = FdBfgsOptions {
                log_model: true,
                ..FdBfgsOptions::default()
            };
            let mut engine = FdBfgs::new(start.view(), None, &bounds, options);
            let (mut asked, mut log) = (Vec::new(), false);
            while asked.len() < 600 && !engine.is_done() {
                let x = engine.ask();
                engine.tell(scale * exp_bowl(x.view()));
                log |= engine.log_active;
                asked.push(x);
            }
            (asked, engine.value() / scale, log)
        };
        let (plain, value, log) = trace(1.0);
        assert!(log, "the logarithm never took the steps");
        for scale in [2f64.powi(-30), 2f64.powi(30)] {
            let (scaled, scaled_value, _) = trace(scale);
            assert_eq!(scaled, plain, "scale {scale}");
            assert_eq!(scaled_value, value, "scale {scale}");
        }
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
    fn extrapolations_without_a_minimiser_stop_halfway_to_the_nearest_bound() {
        // Falling ever faster towards the upper bound, each accepted first
        // trial realises more than its predicted decrease and asks for the
        // longest extrapolation: from 0.1 the first would reach 0.9.
        let bounds = boxed(0.0, 1.0, 1);
        let f = |x: ArrayView1<f64>| -x[0] * x[0];
        let mut engine = FdBfgs::new(
            Array1::from_elem(1, 0.1).view(),
            None,
            &bounds,
            FdBfgsOptions::default(),
        );
        let mut asked = Vec::new();
        while asked.len() < 500 && !engine.is_done() {
            let x = engine.ask();
            asked.push(x[0]);
            engine.tell(f(x.view()));
        }
        let first_long = asked.iter().copied().find(|&x| x > 0.2).unwrap();
        assert!((first_long - 0.55).abs() < 1e-12, "{asked:?}");
        assert!(engine.is_done(), "asked {}", asked.len());
        assert_eq!(engine.position()[0], 1.0);
    }

    #[test]
    fn extrapolations_to_a_minimiser_go_past_halfway_to_the_nearest_bound() {
        // Up to the error of the forward difference, the quadratic through
        // the values at 0.1 and 0.15 and the slope is the objective, so the
        // longer trial goes to its minimum at 0.7, past halfway to the bound.
        let bounds = boxed(0.0, 1.0, 1);
        let f = |x: ArrayView1<f64>| (x[0] - 0.7).powi(2);
        let mut engine = FdBfgs::new(
            Array1::from_elem(1, 0.1).view(),
            None,
            &bounds,
            FdBfgsOptions::default(),
        );
        let mut asked = Vec::new();
        while asked.len() < 500 && !engine.is_done() {
            let x = engine.ask();
            asked.push(x[0]);
            engine.tell(f(x.view()));
        }
        let first_long = asked.iter().copied().find(|&x| x > 0.2).unwrap();
        assert!((first_long - 0.7).abs() < 1e-5, "{asked:?}");
        assert!((engine.position()[0] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn extrapolations_stop_halfway_to_the_nearest_bound_when_it_is_a_lower_one() {
        // Falling ever faster towards the corner (0, 1), the first trial from
        // (0.7, 0.1) lowers the first coordinate by 0.05 and raises the second
        // by a third of that. Along the direction the first coordinate's lower
        // bound is nearer than the second's upper bound, so the longer trial
        // takes the first coordinate halfway to zero.
        let bounds = boxed(0.0, 1.0, 2);
        let f = |x: ArrayView1<f64>| -(x[0] - 1.0).powi(2) - x[1] * x[1];
        let mut engine = FdBfgs::new(
            Array1::from_vec(vec![0.7, 0.1]).view(),
            None,
            &bounds,
            FdBfgsOptions::default(),
        );
        let mut asked = Vec::new();
        while asked.len() < 500 && !engine.is_done() {
            let x = engine.ask();
            engine.tell(f(x.view()));
            asked.push(x);
        }
        let long = asked.iter().find(|x| x[0] < 0.6).unwrap();
        assert!((long[0] - 0.35).abs() < 1e-12, "{long}");
        assert!((long[1] - (0.1 + 0.35 / 3.0)).abs() < 1e-6, "{long}");
        assert!(engine.is_done(), "asked {}", asked.len());
        assert_eq!(engine.position(), Array1::from_vec(vec![0.0, 1.0]));
    }

    #[test]
    fn extrapolations_never_ask_for_less_than_the_first_trial() {
        // From 0.92 the first trial reaches 0.97 and realises more than its
        // predicted decrease, but halfway to the bound is 0.96, short of it:
        // the longer trial would be the first trial itself, so none is asked,
        // and no point below 0.97 is asked after it.
        let bounds = boxed(0.0, 1.0, 1);
        let f = |x: ArrayView1<f64>| -x[0] * x[0];
        let mut engine = FdBfgs::new(
            Array1::from_elem(1, 0.92).view(),
            None,
            &bounds,
            FdBfgsOptions::default(),
        );
        let mut asked = Vec::new();
        while asked.len() < 500 && !engine.is_done() {
            let x = engine.ask();
            asked.push(x[0]);
            engine.tell(f(x.view()));
        }
        let first = asked.iter().position(|&x| x > 0.95).unwrap();
        assert!((asked[first] - 0.97).abs() < 1e-12, "{asked:?}");
        assert!(
            asked[first..].iter().all(|&x| x >= asked[first]),
            "{asked:?}"
        );
        assert!(engine.is_done(), "asked {}", asked.len());
        assert_eq!(engine.position()[0], 1.0);
    }

    #[test]
    fn a_direction_out_of_the_box_on_a_free_coordinate_leaves_no_room() {
        // The first coordinate sits on its lower bound where the objective
        // falls inwards, so it is free, but the model's cross term turns the
        // direction out of the box there. The full step keeps that coordinate
        // on its bound and realises more than its predicted decrease; with no
        // room, no longer trial is asked, and the next point is a difference
        // step from the accepted trial.
        let bounds = boxed(0.0, 1.0, 2);
        let f = |x: ArrayView1<f64>| -0.1 * x[0] - (x[1] - 0.9).powi(2);
        let start = Array1::from_vec(vec![0.0, 0.5]);
        let mut engine = FdBfgs::new(
            start.view(),
            Some(f(start.view())),
            &bounds,
            FdBfgsOptions::default(),
        );
        engine.curvature.inv_hessian =
            Array2::from_shape_vec((2, 2), vec![0.05, 0.1, 0.1, 0.25]).unwrap();
        engine.curvature.scaled = true;
        engine.finish_gradient(Array1::from_vec(vec![-0.1, 0.8]));
        let first = engine.ask();
        assert_eq!(first[0], 0.0);
        assert!((first[1] - 0.31).abs() < 1e-12, "{first}");
        engine.tell(f(first.view()));
        assert_eq!(engine.position(), first.view());
        let next = engine.ask();
        let moved: Vec<f64> = (0..2).map(|i| (next[i] - first[i]).abs()).collect();
        assert_eq!(moved.iter().filter(|&&d| d > 0.0).count(), 1, "{next}");
        assert!(moved.iter().all(|&d| d < 1e-6), "{next}");
    }

    #[test]
    fn a_direction_that_does_not_descend_restarts_the_model() {
        // An indefinite approximation, which rounding leaves once the
        // curvature has grown by many decades along the descent, gives a
        // direction along which the gradient climbs. The line search steps
        // down the steepest-descent direction instead, and the model the
        // steps are on starts over from the box metric scaled by its latest
        // pair, inverse and direct approximation alike, while the other
        // model is kept.
        for kinds in ["vv", "llln"] {
            let mut engine = two_coordinate_descent();
            let log = *steps_after(&mut engine, kinds).last().expect("a pair");
            let other = if log {
                &engine.curvature
            } else {
                &engine.log_curvature
            };
            let kept = (other.inv_hessian.clone(), other.hessian.clone());
            let model = if log {
                &mut engine.log_curvature
            } else {
                &mut engine.curvature
            };
            let gamma = model.gamma;
            model.inv_hessian = -Array2::<f64>::eye(2);
            model.hessian = Some(-Array2::<f64>::eye(2));
            engine.begin_line_search(false);
            match &engine.phase {
                Phase::Line(work) => assert!(
                    work.log == log && work.steepest && engine.grad.dot(&work.direction) < 0.0,
                    "after {kinds}"
                ),
                _ => panic!("after {kinds}: no line search followed"),
            }
            let (model, other) = if log {
                (&engine.log_curvature, &engine.curvature)
            } else {
                (&engine.curvature, &engine.log_curvature)
            };
            let metric = &engine.metric;
            let inverse = Array2::from_diag(&(metric * gamma));
            assert_eq!(model.inv_hessian, inverse, "after {kinds}");
            let direct = Array2::from_diag(&metric.mapv(|w| 1.0 / (w * gamma)));
            assert_eq!(model.hessian, Some(direct), "after {kinds}");
            assert_eq!(
                (&other.inv_hessian, &other.hessian),
                (&kept.0, &kept.1),
                "after {kinds}"
            );
        }
    }

    #[test]
    fn a_start_over_keeps_the_count_of_pairs_and_the_scale() {
        // Three pairs for the logarithm leave the steps on the values, whose
        // model then refuses a direction and starts over. It keeps its count
        // of pairs, so a fourth pair, for neither reading, still completes
        // two per coordinate and moves the steps onto the logarithm, as it
        // does without the start-over; and it stays scaled, so that pair
        // updates the approximation the start-over left instead of scaling
        // it again by its own s.y / y.D.y.
        let mut engine = two_coordinate_descent();
        assert_eq!(steps_after(&mut engine, "lll"), [false; 3]);
        engine.curvature.inv_hessian = -Array2::<f64>::eye(2);
        engine.begin_line_search(false);
        match &engine.phase {
            Phase::Line(work) => assert!(!work.log && work.steepest),
            _ => panic!("no line search followed"),
        }
        let model = &engine.curvature;
        assert_eq!(
            model.inv_hessian,
            Array2::from_diag(&(&engine.metric * model.gamma))
        );
        assert_eq!((model.pairs, model.scaled), (3, true));
        assert_eq!(steps_after(&mut engine, "n"), [true]);
    }

    #[test]
    fn a_start_over_scales_the_box_metric_by_the_latest_pair() {
        // The model of the values takes a pair for the values and then one
        // for neither reading, whose scales s.y / y.D.y differ, while the
        // steps stay on the values. Started over, it holds the box metric
        // scaled by the second pair, inverse and direct approximation
        // alike, and the steepest-descent line search's first trial moves
        // by that scale too.
        let mut engine = two_coordinate_descent();
        assert_eq!(steps_after(&mut engine, "vn"), [false, false]);
        let scale = |kind| {
            let (from, to) = pair(kind);
            let (s, y) = (to[0] - from[0], to[2] - from[2]);
            s * y / (engine.metric[0] * y * y)
        };
        let gamma = scale('n');
        assert_ne!(gamma, scale('v'));
        engine.curvature.inv_hessian = -Array2::<f64>::eye(2);
        engine.curvature.hessian = Some(-Array2::<f64>::eye(2));
        engine.begin_line_search(false);
        match &engine.phase {
            Phase::Line(work) => assert!(!work.log && work.steepest && work.alpha == gamma),
            _ => panic!("no line search followed"),
        }
        let metric = &engine.metric;
        assert_eq!(
            engine.curvature.inv_hessian,
            Array2::from_diag(&(metric * gamma))
        );
        let direct = Array2::from_diag(&metric.mapv(|w| 1.0 / (w * gamma)));
        assert_eq!(engine.curvature.hessian, Some(direct));
    }

    #[test]
    fn a_steepest_descent_step_after_a_failed_line_search_keeps_the_models() {
        // A line search along a quasi-Newton direction that descends can
        // fail, and once the stencils are central and the gradient has been
        // differenced again as often as it may be, the descent searches down
        // the steepest-descent direction. Nothing there shows the
        // approximation indefinite, so both models keep their
        // approximations, counts of pairs and scales.
        let models = |engine: &FdBfgs| {
            [&engine.curvature, &engine.log_curvature].map(|model| {
                (
                    model.inv_hessian.clone(),
                    model.hessian.clone(),
                    model.pairs,
                    model.scaled,
                )
            })
        };
        for kinds in ["vv", "llln"] {
            let mut engine = two_coordinate_descent();
            let log = *steps_after(&mut engine, kinds).last().expect("a pair");
            let model = if log {
                &mut engine.log_curvature
            } else {
                &mut engine.curvature
            };
            model.inv_hessian = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 1.0]).unwrap();
            model.hessian =
                Some(Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 2.0]).unwrap());
            engine.scheme = FdScheme::Central;
            engine.redifferences = MAX_REDIFFERENCES;
            let kept = models(&engine);
            engine.line_search_failed(false);
            match &engine.phase {
                Phase::Line(work) => assert!(work.log == log && work.steepest, "after {kinds}"),
                _ => panic!("after {kinds}: no line search followed"),
            }
            assert_eq!(models(&engine), kept, "after {kinds}");
        }
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
    fn the_first_coordinate_reorders_the_stencils_but_not_the_descent() {
        let bounds = boxed(-2.0, 2.0, 5);
        let start = Array1::from_vec(vec![0.5, -1.0, 1.5, 0.0, 2.0]);
        let run = |first: usize| {
            let f0 = Some(rosenbrock(start.view()));
            let mut engine = FdBfgs::new(start.view(), f0, &bounds, FdBfgsOptions::default())
                .with_first_coordinate(first);
            let mut asked = Vec::new();
            while asked.len() < 2000 && !engine.is_done() {
                let x = engine.ask();
                engine.tell(rosenbrock(x.view()));
                asked.push(x);
            }
            (asked, engine.position().to_owned(), engine.value())
        };
        let sorted = |points: &[Array1<f64>]| {
            let mut keys: Vec<Vec<u64>> = points
                .iter()
                .map(|x| x.iter().map(|v| v.to_bits()).collect())
                .collect();
            keys.sort();
            keys
        };
        let (plain, x, f) = run(0);
        assert!(plain.len() > 200, "descent ended after {}", plain.len());
        for first in [1usize, 3, 7] {
            let (rotated, rotated_x, rotated_f) = run(first);
            let moved: Vec<usize> = (0..5).filter(|&i| rotated[0][i] != start[i]).collect();
            assert_eq!(moved, vec![first % 5], "first {first}");
            assert_ne!(rotated, plain, "first {first}");
            assert_eq!(sorted(&rotated), sorted(&plain), "first {first}");
            assert_eq!((rotated_x, rotated_f), (x.clone(), f), "first {first}");
        }
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
    fn absolute_slow_floor_ends_descents_sooner() {
        // Rosenbrock-4 converges to 0, where the relative test alone keeps
        // the descent going down to the last units in the last place.
        let bounds = boxed(-2.0, 2.0, 4);
        let rosenbrock = |x: ArrayView1<f64>| {
            (0..3)
                .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
                .sum::<f64>()
        };
        let start = Array1::from_vec(vec![-1.2, 1.0, -1.2, 1.0]);
        let mut plain = FdBfgs::new(start.view(), None, &bounds, FdBfgsOptions::default());
        let plain_used = drive(&mut plain, rosenbrock, 5000, &bounds);
        let mut floored = FdBfgs::new(
            start.view(),
            None,
            &bounds,
            FdBfgsOptions {
                fatol: 1e-6,
                ..FdBfgsOptions::default()
            },
        );
        let floored_used = drive(&mut floored, rosenbrock, 5000, &bounds);
        assert!(plain.is_done() && floored.is_done());
        assert!(floored.value() < 1e-3, "value {}", floored.value());
        assert!(
            floored_used < plain_used,
            "floored {floored_used} evaluations against {plain_used}"
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

    #[test]
    fn a_box_too_narrow_for_a_difference_step_ends_the_descent_at_its_start() {
        // A side one unit in the last place wide, or of zero width, has no
        // step that changes its coordinate. With no other side, the descent
        // has nothing to ask once it knows its start's value, whether told
        // or given, and from any coordinate first; with one side wide
        // enough it differences that one.
        let up = f64::from_bits(1.0f64.to_bits() + 1);
        let f = |x: ArrayView1<f64>| x.iter().map(|v| (v - 0.3).powi(2)).sum::<f64>();
        let narrow = [
            (vec![1.0], vec![up]),
            (vec![0.5], vec![0.5]),
            (vec![1.0, 1.0], vec![up, up]),
            (vec![0.5, 1.0], vec![0.5, up]),
        ];
        for (low, high) in narrow {
            let bounds = Bounds::new(Array1::from(low.clone()), Array1::from(high), 0.0);
            let start = Array1::from(low);
            let mut told = FdBfgs::new(start.view(), None, &bounds, FdBfgsOptions::default());
            assert_eq!(told.ask(), start);
            told.tell(f(start.view()));
            let mut given = FdBfgs::new(
                start.view(),
                Some(f(start.view())),
                &bounds,
                FdBfgsOptions::default(),
            )
            .with_first_coordinate(1);
            let mut moved = FdBfgs::new(start.view(), Some(1.0), &bounds, FdBfgsOptions::default());
            moved.restart_at(start.view(), Some(f(start.view())), false);
            for engine in [&mut told, &mut given, &mut moved] {
                assert!(engine.is_done(), "{start}");
                assert_eq!(
                    (engine.ask(), engine.value()),
                    (start.clone(), f(start.view()))
                );
                engine.tell(0.0);
                assert_eq!(
                    (engine.position(), engine.value()),
                    (start.view(), f(start.view()))
                );
            }
        }
        let bounds = Bounds::new(
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![up, 1.0]),
            0.0,
        );
        let start = Array1::from_vec(vec![1.0, 0.5]);
        let mut engine = FdBfgs::new(start.view(), None, &bounds, FdBfgsOptions::default());
        let used = drive(&mut engine, f, 200, &bounds);
        assert!(engine.is_done(), "used {used}");
        assert_eq!(engine.position()[0], 1.0);
        assert!((engine.position()[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn a_step_that_underflows_near_zero_admits_no_difference() {
        // The step scales with the larger of |x| and a tenth of the side, so
        // near zero it underflows: [0, 1e-316] admits none from either end
        // or its middle, and [0, 2e-316] none from 0 but one from 2e-316.
        let can = |x: f64, high: f64| {
            let bounds = Bounds::new(
                Array1::from_vec(vec![0.0]),
                Array1::from_vec(vec![high]),
                0.0,
            );
            FdBfgs::can_difference(Array1::from_vec(vec![x]).view(), &bounds)
        };
        for x in [0.0, 0.5e-316, 1e-316] {
            assert!(!can(x, 1e-316), "{x:e}");
        }
        assert!(!can(0.0, 2e-316));
        assert!(can(2e-316, 2e-316));
        assert!(can(0.0, 1e-300) && can(1e-300, 1e-300));
    }

    #[test]
    fn on_a_side_two_units_wide_half_a_unit_rounds_to_an_even_last_bit() {
        // There the step is half a unit in the last place, and adding it
        // rounds to whichever neighbour has an even last bit: back to either
        // end of [1, 1 + 2 ulp] but away from its middle, and away from
        // either end of [1 + ulp, 1 + 3 ulp]. A side one unit wide admits
        // no step anywhere.
        let ulps = |k: u64| f64::from_bits(1.0f64.to_bits() + k);
        let can = |x: u64, low: u64, high: u64| {
            let bounds = Bounds::new(
                Array1::from_vec(vec![ulps(low)]),
                Array1::from_vec(vec![ulps(high)]),
                0.0,
            );
            FdBfgs::can_difference(Array1::from_vec(vec![ulps(x)]).view(), &bounds)
        };
        assert!(!can(0, 0, 2) && !can(2, 0, 2) && can(1, 0, 2));
        assert!(can(1, 1, 3) && can(3, 1, 3) && !can(2, 1, 3));
        assert!(!can(0, 0, 1) && !can(1, 0, 1) && !can(1, 1, 2));
    }

    #[test]
    fn a_central_stencil_asks_nothing_on_a_side_too_narrow_for_its_step() {
        // On a side one unit in the last place wide, from either end, or of
        // zero width, every step of either scheme rounds away, so neither
        // stencil asks anything there and no stencil point is the iterate.
        let up = f64::from_bits(1.0f64.to_bits() + 1);
        let f = |x: ArrayView1<f64>| x.iter().map(|v| (v - 0.3).powi(2)).sum::<f64>();
        for (high, at) in [(up, 1.0), (up, up), (1.0, 1.0)] {
            let bounds = Bounds::new(
                Array1::from_vec(vec![-1.0, 1.0, -1.0]),
                Array1::from_vec(vec![1.0, high, 1.0]),
                0.0,
            );
            let start = Array1::from_vec(vec![0.4, at, -0.3]);
            let mut engine = FdBfgs::new(start.view(), None, &bounds, FdBfgsOptions::default());
            for (scheme, points) in [(FdScheme::Forward, 1), (FdScheme::Central, 2)] {
                engine.scheme = scheme;
                assert!(engine.stencil(1).0.is_empty(), "{scheme:?} at {at}");
                assert_eq!(engine.stencil(0).0.len(), points, "{scheme:?} at {at}");
            }
            engine.scheme = FdScheme::Forward;
            let mut central = 0;
            let mut used = 0;
            while used < 300 && !engine.is_done() {
                let x = engine.ask();
                if matches!(engine.phase, Phase::Gradient(_)) {
                    assert_ne!(x.view(), engine.position(), "at {at}");
                    central += usize::from(engine.scheme == FdScheme::Central);
                }
                engine.tell(f(x.view()));
                used += 1;
            }
            assert!(
                engine.is_done() && central > 0,
                "at {at}: {central} of {used}"
            );
            assert_eq!(engine.position()[1], at);
        }
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
