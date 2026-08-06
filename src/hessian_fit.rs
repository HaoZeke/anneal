//! Calibrating the model Hessian on the descents the run already performs.
//!
//! [`crate::model_hessian`] predicts the energy a relaxation recovers as
//! `1/2 g^T H^-1 g` from an operator whose three constants are covalent
//! chemistry's, borrowed because they were the numbers available. A
//! Lennard-Jones or Morse cluster is not that regime: the pair force constant
//! of a rare-gas well falls off far faster than the exponential a bond does,
//! and the bending stiffness the diagonal floor stands in for is a different
//! fraction of the stretching stiffness. Constants set for the wrong system
//! predict the wrong depth, and the depth is what a first-stage acceptance
//! decides on.
//!
//! # The data is already being generated
//!
//! Every quench the search performs is a steepest-descent path, an intrinsic
//! reaction coordinate without the saddle at the top, and it emits exactly the
//! triple a calibration needs: the structure it started from, the gradient
//! there, and the energy it actually recovered. Recording those costs nothing,
//! because the run paid for them to make its own decision. So the operator can
//! be fitted to the potential it is being asked about, using observations of
//! that potential, without ever charging the ledger for one.
//!
//! # What is fitted, and what is not
//!
//! Every term of the operator carries `k0` linearly, so
//! `H(k0, alpha, floor) = k0 H(1, alpha, floor)` and the depth goes as `1/k0`.
//! The scale is therefore not a shape at all: it multiplies every prediction by
//! the same number, which the surrogate regressing on the depth absorbs into
//! its own coefficient. The search runs at `k0 = 1` over the two constants that
//! change the *ordering* of predictions, `alpha` and `floor`, and recovers the
//! scale afterwards in closed form as the factor that lines the predictions up
//! with the observations. Two parameters, not three, and the third comes free.
//!
//! # The objective
//!
//! Depths span orders of magnitude across a run, so the residual is taken in
//! logarithms: `l_i = ln(predicted_i) - ln(observed_i)`, which treats a factor
//! of two too large and a factor of two too small as the same error, where a
//! plain relative residual does not. The scale sets the centre, `ln k0 =
//! median(l)`, and what is minimised is `median(|l_i - median(l)|)`.
//!
//! The median rather than the mean, because the harmonic form is exact only
//! near the minimum. A perturbation large enough to change basin recovers
//! anharmonic energy no quadratic form predicts, and on real Lennard-Jones
//! descents the upper quartile of relative error runs three to four times the
//! lower one at every perturbation size measured. A least-squares objective
//! spends both parameters chasing that tail; the median leaves it where it is
//! and fits the bulk, which is also the quantity reported.
//!
//! # Degrading safely
//!
//! Below `min_samples` observations the accumulator hands back
//! [`ModelParams::default`] unchanged, so a cold run is exactly the uncalibrated
//! behaviour and no caller needs a branch. A fit that does not beat the defaults
//! by [`ADOPT_MARGIN`] on the samples it was given is discarded rather than
//! adopted, which matters because the objective on real descents is nearly
//! flat and a search will always return something nominally better.

use std::collections::VecDeque;

use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;

use crate::model_hessian::{ModelParams, depth_with};

/// Observations before the accumulator will fit anything.
///
/// Two parameters need enough geometries to separate them, and a fit on a
/// handful of descents out of one basin sees one geometry. On descents
/// generated from a known `alpha = 3.0, floor = 0.01`, eight and twelve samples
/// are rejected because nothing in the box beats the defaults by
/// [`ADOPT_MARGIN`], and sixteen recover `alpha = 3.005, floor = 0.00999`.
/// Going on to ninety-six changes the third decimal place.
pub const MIN_SAMPLES: usize = 16;

/// Observations between refits.
///
/// A refit is a two-dimensional search over the whole buffer, so it is not free
/// in arithmetic even though it is free on the ledger: five grids of 64 shapes,
/// each solving every retained sample. On a full buffer that is 196 ms for a
/// 38-point cluster and 808 ms for a 75-point one. Refitting every sample would
/// multiply that by the interval, for constants that stop moving in the third
/// decimal place after twenty samples.
pub const REFIT_EVERY: usize = 16;

/// Observations retained.
///
/// The buffer is a ring, because a long run drifts through different regions of
/// the landscape and the constants that matter describe where the search is
/// now. Forty-eight rather than more: on synthetic descents the recovered
/// `alpha` is 3.005 at sixteen samples and 2.991 at every count from twenty to
/// ninety-six, so the fit stops learning long before the buffer stops growing,
/// while the cost is linear in it. A 38-point cluster refits in 196 ms on 48
/// samples and 408 ms on 96.
pub const CAPACITY: usize = 48;

/// Conjugate-gradient iterations inside a fit evaluation.
///
/// The truncation is part of the operator being fitted, not an approximation to
/// it: a shape fitted against one truncation and consumed at another has
/// absorbed the difference, and the difference is not small. Measured against
/// descents generated at forty iterations, a fit at twenty-four recovers the
/// forty-iteration answer in every digit; at sixteen it returns `alpha = 3.20`
/// where the matched fit returns `2.99`; at twelve, `3.29`; at eight, `2.76`
/// with a floor twice too stiff; at six, `1.35` with a floor eight times too
/// stiff and a scale seven times too soft.
///
/// Twenty-four is where the shape stops moving, and it doubles the refit
/// against twelve. A consumer that solves shallower than this should fit there
/// too, and pay less for it.
pub const FIT_ITERS: usize = 24;

/// One observed descent: where it started, the gradient there, what it recovered.
#[derive(Clone, Debug)]
pub struct Descent {
    /// Flattened coordinates of the structure the descent started from.
    pub x: Array1<f64>,
    /// Gradient at that structure.
    pub gradient: Array1<f64>,
    /// Energy the relaxation actually recovered, `E(x) - E(Q(x))`.
    pub observed: f64,
}

impl Descent {
    /// Number of points, from the flattened length.
    pub fn n(&self) -> usize {
        self.x.len() / 3
    }

    /// Whether this triple can carry information about the operator.
    ///
    /// A descent that recovered nothing, or one from a structure already at a
    /// minimum, constrains no ratio of predicted to observed depth.
    pub fn is_usable(&self) -> bool {
        self.n() >= 2
            && self.gradient.len() == self.x.len()
            && self.observed.is_finite()
            && self.observed > 0.0
            && self.x.iter().all(|v| v.is_finite())
            && self.gradient.iter().all(|v| v.is_finite())
            && self.gradient.iter().any(|v| v.abs() > 0.0)
    }
}

/// A fitted operator together with how well it fitted.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// The three constants, scale included.
    pub params: ModelParams,
    /// `median(|l_i - median(l)|)` on the sample, in log units.
    ///
    /// A spread of `s` means half the sample is predicted within a factor
    /// `exp(s)` of the scale-matched prediction.
    pub spread: f64,
}

/// Corner of the search box, in the same units as [`ModelParams`].
const ALPHA_LO: f64 = 0.2;
/// Upper end of the decay search.
///
/// Lennard-Jones needs the room. Scanning a sample of small perturbations of an
/// LJ38 minimum, the spread falls monotonically from 0.158 at `alpha = 0.2`
/// through 0.112 at 6.6 to a minimum of 0.068 at 17.9, then turns and reaches
/// 0.15 by 29.5 and 0.25 by 48.6. A ceiling of 12, which is generous for a
/// covalent bond, would have clipped the answer: the pair force constant of a
/// rare-gas well falls off far faster than a bond's, and a decay of 18 in units
/// of the spacing is what that looks like when it is written as an exponential.
const ALPHA_HI: f64 = 30.0;
/// Lower end of the floor search. Below this the operator is numerically
/// singular against the transverse motions its stretch terms ignore.
const FLOOR_LO: f64 = 1e-4;
/// Upper end of the floor search.
///
/// The ceiling is a regularisation, not a range. Past a floor of about three
/// the diagonal dominates the pair terms and the operator becomes
/// `|g|^2 / 2 floor k0`, the isotropic proxy the model Hessian exists to
/// replace, in which `alpha` no longer enters at all. Scanning a Lennard-Jones
/// sample out to a floor of 300 shows the objective flattening onto that limit
/// at a spread of 0.080 against the best 0.074 anywhere in the box: a search
/// allowed to go there would trade a real anisotropy for a plateau it could
/// wander along, and would report an unidentifiable `alpha` while doing it.
const FLOOR_HI: f64 = 3.0;
/// Relative worsening within which two shapes count as equally good.
///
/// The floor is unidentifiable over a wide range whenever the gradients in the
/// sample barely excite the transverse motions the floor exists to resist: on
/// small perturbations of an LJ38 minimum the spread is 0.1525 at every floor
/// from 1e-9 to 2e-3, six decades wide. Somewhere on that plateau has to be
/// picked, and the low end is the numerically singular one, so the search walks
/// the floor up while it costs no more than this.
///
/// A tenth of a per cent, because the plateau it exists to resolve is flat to
/// four decimal places across those six decades, while a floor that genuinely
/// carries information moves the spread by whole per cent.
const FLAT_TOL: f64 = 1e-3;
/// Fractional reduction in spread a fit must show before it is adopted.
///
/// The objective on real Lennard-Jones descents is close to flat: over the
/// whole box the spread runs between 0.074 and 0.089, so a search will always
/// come back with something nominally better than the defaults, and successive
/// refits on overlapping buffers settle on constants a factor of two apart
/// while predicting the same depths to within a per cent. Requiring a five per
/// cent reduction keeps the constants still when the data has nothing to say,
/// and still admits the eight per cent the Lennard-Jones fit does earn.
const ADOPT_MARGIN: f64 = 0.05;
/// Points per axis on each grid.
const GRID: usize = 8;
/// Grids laid down, each inside the last one's best cell.
///
/// Every level narrows the box to two cells about the incumbent, a factor
/// `2 / (GRID - 1)` in each logarithm, so five levels take the decay's five
/// log units of range down to one per cent. That resolution is far finer than
/// the constants are identified to, and the levels are cheap next to the
/// samples they solve.
const LEVELS: usize = 5;

/// Median of a slice, by sorting a copy. Non-finite entries are not expected.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len() / 2;
    if v.len() % 2 == 0 {
        0.5 * (v[m - 1] + v[m])
    } else {
        v[m]
    }
}

/// Log-ratio spread of one candidate shape, and the scale that centres it.
///
/// Returns `None` when a candidate predicts a non-positive or non-finite depth
/// anywhere on the sample, which means the solve found no usable curvature and
/// the candidate cannot be ranked against one that did.
pub fn spread_of(samples: &[Descent], alpha: f64, floor: f64, iters: usize) -> Option<Calibration> {
    if samples.is_empty() {
        return None;
    }
    let shape = ModelParams::unit_scale(alpha, floor);
    // Each sample is an independent conjugate-gradient solve on its own
    // structure, and the solves are what a refit spends its time on: 136
    // candidates times the buffer. Splitting over samples rather than over
    // candidates keeps the golden-section sweeps, which are sequential by
    // construction, as parallel as the grid.
    let logs: Vec<f64> = samples
        .par_iter()
        .map(|s| {
            let p = depth_with(s.x.view(), s.n(), s.gradient.view(), iters, shape);
            if p > 0.0 && p.is_finite() {
                Some(p.ln() - s.observed.ln())
            } else {
                None
            }
        })
        .collect::<Option<Vec<f64>>>()?;
    // The scale that puts half the sample above the observation and half below:
    // `depth ~ 1/k0`, so a prediction too large by `exp(mu)` is corrected by a
    // force-constant scale of `exp(mu)`.
    let mu = median(&logs);
    if !mu.is_finite() {
        return None;
    }
    let deviations: Vec<f64> = logs.iter().map(|l| (l - mu).abs()).collect();
    Some(Calibration {
        params: ModelParams {
            k0: mu.exp(),
            alpha,
            floor,
        },
        spread: median(&deviations),
    })
}

/// Fits `alpha` and `floor` to a sample of descents, recovering `k0` in closed form.
///
/// Nested grids in the logarithms of both constants, since the constants are
/// scales and a linear grid would spend most of its points on the stiff end.
/// Each level lays a `GRID x GRID` mesh over the box, then narrows the box to
/// the cells either side of the best point and repeats.
///
/// Nested grids rather than coordinate descent, because the valley is not axis
/// aligned. Alternating golden sections on `alpha` and `floor` were measured to
/// stall on synthetic descents generated from `alpha = 2.4, floor = 0.03`,
/// returning `alpha = 3.96, floor = 0.0083` at a spread of 0.028 where the
/// generating shape sits at 8e-16: a stiffer decay needs a softer floor to
/// predict the same depth, so every axial move away from the stall looks worse
/// while the diagonal move does not exist. A grid has no such blind direction.
///
/// Returns `None` when the sample is too small to constrain two parameters, or
/// when nothing in the box beats [`ModelParams::default`] by [`ADOPT_MARGIN`]
/// on the sample's own spread. The second condition is what bounds the harm: a
/// fit that cannot clearly improve on the constants it replaces is discarded
/// rather than adopted, and on a flat objective that is the common case.
pub fn calibrate(samples: &[Descent], iters: usize) -> Option<Calibration> {
    if samples.len() < MIN_SAMPLES {
        return None;
    }
    let objective = |alpha: f64, floor: f64| -> f64 {
        spread_of(samples, alpha, floor, iters).map_or(f64::INFINITY, |c| c.spread)
    };

    let (la, lb) = (ALPHA_LO.ln(), ALPHA_HI.ln());
    let (lf, lg) = (FLOOR_LO.ln(), FLOOR_HI.ln());
    let (mut a0, mut a1) = (la, lb);
    let (mut f0, mut f1) = (lf, lg);
    let mut best = (f64::INFINITY, ALPHA_LO, FLOOR_LO);

    for _ in 0..LEVELS {
        let step = (GRID - 1) as f64;
        let level: Vec<(f64, f64, f64)> = (0..GRID * GRID)
            .into_par_iter()
            .map(|k| {
                let alpha = (a0 + (a1 - a0) * (k / GRID) as f64 / step).exp();
                let floor = (f0 + (f1 - f0) * (k % GRID) as f64 / step).exp();
                (objective(alpha, floor), alpha, floor)
            })
            .collect();
        for cand in level {
            if cand.0 < best.0 {
                best = cand;
            }
        }
        if !best.0.is_finite() {
            return None;
        }
        // Two cells wide about the incumbent, clipped to the box the constants
        // are physical in, so a minimum sitting on a face stays reachable.
        let half_a = (a1 - a0) / step;
        let half_f = (f1 - f0) / step;
        a0 = (best.1.ln() - half_a).max(la);
        a1 = (best.1.ln() + half_a).min(lb);
        f0 = (best.2.ln() - half_f).max(lf);
        f1 = (best.2.ln() + half_f).min(lg);
    }

    // The floor is flat over decades when the sample's gradients do not excite
    // the transverse motions. Take the stiffest floor that costs nothing, since
    // the other end of that plateau is an operator one sample away from
    // singular, and since a plateau minimum is otherwise decided by rounding.
    let tolerated = best.0 * (1.0 + FLAT_TOL);
    loop {
        let trial = best.2 * 2.0;
        if trial > FLOOR_HI {
            break;
        }
        let v = objective(best.1, trial);
        if v > tolerated {
            break;
        }
        best.2 = trial;
        best.0 = best.0.min(v);
    }

    let fitted = spread_of(samples, best.1, best.2, iters)?;
    let baseline = ModelParams::default();
    let default_spread = spread_of(samples, baseline.alpha, baseline.floor, iters)
        .map_or(f64::INFINITY, |c| c.spread);
    if !(fitted.spread < (1.0 - ADOPT_MARGIN) * default_spread) {
        return None;
    }
    Some(fitted)
}

/// Accumulates observed descents and keeps a calibrated operator current.
///
/// Charges nothing: it consumes triples the run produced for its own reasons
/// and does arithmetic on them.
pub struct HessianFit {
    samples: VecDeque<Descent>,
    calibration: Option<Calibration>,
    since_refit: usize,
    seen: usize,
    /// Observations retained before the oldest is dropped.
    pub capacity: usize,
    /// Observations required before a fit is attempted at all.
    pub min_samples: usize,
    /// Observations between refits.
    pub refit_every: usize,
    /// Conjugate-gradient iterations inside a fit evaluation.
    pub fit_iters: usize,
}

impl Default for HessianFit {
    fn default() -> Self {
        Self::new()
    }
}

impl HessianFit {
    /// An empty accumulator, handing back the default constants until fitted.
    pub fn new() -> Self {
        Self {
            samples: VecDeque::new(),
            calibration: None,
            since_refit: 0,
            seen: 0,
            capacity: CAPACITY,
            min_samples: MIN_SAMPLES,
            refit_every: REFIT_EVERY,
            fit_iters: FIT_ITERS,
        }
    }

    /// Records one descent, refitting when the schedule says to.
    ///
    /// Returns whether this observation triggered a refit, which a caller can
    /// log without having to track the schedule itself. Unusable triples are
    /// dropped rather than fitted around: a descent that recovered nothing
    /// carries no ratio.
    pub fn observe(&mut self, x: ArrayView1<f64>, gradient: ArrayView1<f64>, observed: f64) -> bool {
        let sample = Descent {
            x: x.to_owned(),
            gradient: gradient.to_owned(),
            observed,
        };
        if !sample.is_usable() {
            return false;
        }
        self.samples.push_back(sample);
        while self.samples.len() > self.capacity {
            self.samples.pop_front();
        }
        self.seen += 1;
        self.since_refit += 1;
        // The first fit lands the moment there is enough to fit; after that the
        // schedule takes over.
        let due = self.samples.len() >= self.min_samples
            && (self.calibration.is_none() || self.since_refit >= self.refit_every);
        if due { self.refit() } else { false }
    }

    /// Refits immediately, whatever the schedule says.
    ///
    /// Returns whether a calibration was adopted. A rejected fit leaves the
    /// previous one, or the defaults, in place.
    pub fn refit(&mut self) -> bool {
        self.since_refit = 0;
        let samples: Vec<Descent> = self.samples.iter().cloned().collect();
        match calibrate(&samples, self.fit_iters) {
            Some(c) => {
                self.calibration = Some(c);
                true
            }
            None => false,
        }
    }

    /// The constants to use: fitted if there is a fit, the defaults otherwise.
    pub fn params(&self) -> ModelParams {
        self.calibration
            .map_or_else(ModelParams::default, |c| c.params)
    }

    /// The fit itself, when there is one.
    pub fn calibration(&self) -> Option<Calibration> {
        self.calibration
    }

    /// Whether the constants being handed out came from observations.
    pub fn is_calibrated(&self) -> bool {
        self.calibration.is_some()
    }

    /// Predicted depth at `x`, through whichever constants are current.
    pub fn predict(&self, x: ArrayView1<f64>, gradient: ArrayView1<f64>, iters: usize) -> f64 {
        depth_with(x, x.len() / 3, gradient, iters, self.params())
    }

    /// Observations currently retained.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether anything has been recorded.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Usable observations accepted over the whole run, retained or dropped.
    pub fn seen(&self) -> usize {
        self.seen
    }

    /// The retained observations, for a caller that wants to score a held-out set.
    pub fn samples(&self) -> &VecDeque<Descent> {
        &self.samples
    }
}

/// Median relative error of a parameter set against a set of descents.
///
/// `|predicted / observed - 1|`, which is the number a report quotes, as
/// opposed to the log-ratio spread the fit minimises. The two agree to first
/// order and diverge for gross errors, where the relative form saturates at one
/// for under-prediction and the log form does not.
pub fn median_relative_error(samples: &[Descent], params: ModelParams, iters: usize) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let errors: Vec<f64> = samples
        .iter()
        .map(|s| {
            let p = depth_with(s.x.view(), s.n(), s.gradient.view(), iters, params);
            (p / s.observed - 1.0).abs()
        })
        .collect();
    median(&errors)
}

/// The scale that centres `params` on `samples`, leaving the shape alone.
///
/// A comparison between two shapes is only fair once each has been given its
/// own best scale, because `k0` is a free multiplier that any consumer of the
/// depth absorbs. Scoring an uncalibrated operator at `k0 = 1` against a fitted
/// one measures the scale mismatch and calls it a shape improvement.
pub fn rescaled(samples: &[Descent], params: ModelParams, iters: usize) -> ModelParams {
    match spread_of(samples, params.alpha, params.floor, iters) {
        Some(c) => c.params,
        None => params,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_hessian::depth_with;

    /// Deterministic uniform stream, so a failure is reproducible.
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
        }
    }

    /// A jittered compact packing, so the sample spans geometries rather than
    /// repeating one: two constants cannot be separated from a single shape.
    fn structure(rng: &mut Rng, n: usize) -> Array1<f64> {
        let mut x = Array1::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = (i % 3) as f64 * 1.1 + rng.uniform(-0.15, 0.15);
            x[3 * i + 1] = ((i / 3) % 3) as f64 * 1.1 + rng.uniform(-0.15, 0.15);
            x[3 * i + 2] = (i / 9) as f64 * 1.1 + rng.uniform(-0.15, 0.15);
        }
        x
    }

    /// Descents whose observed depth is exactly what an operator with the given
    /// constants predicts, so a fit that fails here fails on the easiest data
    /// that exists.
    fn synthetic(seed: u64, count: usize, truth: ModelParams) -> Vec<Descent> {
        let mut rng = Rng(seed);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let n = 8 + (out.len() % 5);
            let x = structure(&mut rng, n);
            let g: Array1<f64> = Array1::from(
                (0..3 * n)
                    .map(|_| rng.uniform(-0.4, 0.4))
                    .collect::<Vec<_>>(),
            );
            let observed = depth_with(x.view(), n, g.view(), 40, truth);
            if observed > 0.0 && observed.is_finite() {
                out.push(Descent {
                    x,
                    gradient: g,
                    observed,
                });
            }
        }
        out
    }

    /// The fit has to recover constants it was generated from. Without this the
    /// search is only asserted to reduce a number, not to find the right one.
    #[test]
    fn it_recovers_the_constants_that_generated_the_data() {
        let truth = ModelParams {
            k0: 1.0,
            alpha: 3.5,
            floor: 0.008,
        };
        let samples = synthetic(0xA1CE, 48, truth);
        let fit = calibrate(&samples, 40).expect("a fit on exact data was rejected");
        assert!(
            (fit.params.alpha / truth.alpha - 1.0).abs() < 0.02,
            "recovered alpha {} against a true {}",
            fit.params.alpha,
            truth.alpha
        );
        assert!(
            (fit.params.floor / truth.floor).ln().abs() < 0.05,
            "recovered floor {} against a true {}, a factor of {:.3}",
            fit.params.floor,
            truth.floor,
            fit.params.floor / truth.floor
        );
        // Exact data has an exact answer, so the residual has to collapse to
        // the grid's own resolution rather than merely to something small.
        assert!(
            fit.spread < 1e-3,
            "the fit left a log spread of {} on data with no noise",
            fit.spread
        );
    }

    /// The scale is recovered as well as the shape, which is the claim that
    /// justifies searching two parameters instead of three.
    #[test]
    fn it_recovers_the_force_constant_scale_it_never_searched() {
        let truth = ModelParams {
            k0: 6.5,
            alpha: 2.4,
            floor: 0.03,
        };
        let samples = synthetic(0xBEEF, 40, truth);
        let fit = calibrate(&samples, 40).expect("a fit on exact data was rejected");
        assert!(
            (fit.params.k0 / truth.k0).ln().abs() < 0.01,
            "recovered a scale of {} against a true {}",
            fit.params.k0,
            truth.k0
        );
    }

    /// The fit has to help on descents it never saw, or it is memorising the
    /// sample rather than calibrating an operator.
    #[test]
    fn the_fit_beats_the_defaults_on_held_out_descents() {
        let truth = ModelParams {
            k0: 1.0,
            alpha: 4.2,
            floor: 0.004,
        };
        let train = synthetic(0x51DE, 40, truth);
        let test = synthetic(0x0FF0, 40, truth);
        let fit = calibrate(&train, 40).expect("a fit on exact data was rejected");

        // Both shapes get their own best scale on the training set, so what is
        // compared is the shape and not a free multiplier.
        let base = rescaled(&train, ModelParams::default(), 40);
        let before = median_relative_error(&test, base, 40);
        let after = median_relative_error(&test, fit.params, 40);
        assert!(
            after < 0.5 * before,
            "held-out median relative error went {before} -> {after}, not at least halved"
        );
    }

    /// Too few observations returns the defaults untouched, so a cold run
    /// behaves exactly as if the calibration were not there.
    #[test]
    fn too_few_samples_leaves_the_defaults_alone() {
        let truth = ModelParams {
            k0: 1.0,
            alpha: 5.0,
            floor: 0.002,
        };
        let samples = synthetic(0x1234, MIN_SAMPLES - 1, truth);
        assert!(
            calibrate(&samples, 40).is_none(),
            "a fit was returned from fewer than {MIN_SAMPLES} samples"
        );

        let mut acc = HessianFit::new();
        for s in &samples {
            acc.observe(s.x.view(), s.gradient.view(), s.observed);
        }
        assert!(!acc.is_calibrated(), "the accumulator claimed a calibration");
        assert_eq!(
            acc.params(),
            ModelParams::default(),
            "the accumulator moved the constants without enough data"
        );
    }

    /// And it starts using observations the moment there are enough, without a
    /// caller having to ask, then holds off until the schedule comes round.
    #[test]
    fn the_accumulator_fits_on_its_own_schedule() {
        let truth = ModelParams {
            k0: 1.0,
            alpha: 3.0,
            floor: 0.01,
        };
        let samples = synthetic(0x77AA, MIN_SAMPLES + REFIT_EVERY + 4, truth);
        let mut acc = HessianFit::new();
        acc.fit_iters = 40;
        let mut at: Vec<usize> = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            if acc.observe(s.x.view(), s.gradient.view(), s.observed) {
                at.push(i + 1);
            }
        }
        assert_eq!(
            at,
            vec![MIN_SAMPLES, MIN_SAMPLES + REFIT_EVERY],
            "the schedule fired at {at:?}"
        );
        assert!(acc.is_calibrated(), "enough data arrived and nothing fitted");
        assert!(
            (acc.params().alpha / truth.alpha - 1.0).abs() < 0.15,
            "the accumulator settled on alpha {} against a true {}",
            acc.params().alpha,
            truth.alpha
        );
    }

    /// A stiffer decay and a softer floor predict nearly the same depths, so an
    /// axial search has no move that improves on a point between them and stops
    /// there. This is the case that caught alternating golden sections, which
    /// returned `alpha = 3.96, floor = 0.0083` on data generated from `2.4` and
    /// `0.03`, and it is the reason the search lays down grids instead.
    #[test]
    fn the_search_crosses_the_valley_it_cannot_walk_along_an_axis() {
        let truth = ModelParams {
            k0: 6.5,
            alpha: 2.4,
            floor: 0.03,
        };
        let samples = synthetic(0xBEEF, 40, truth);
        let fit = calibrate(&samples, 40).expect("a fit on exact data was rejected");
        // The point coordinate descent stalled on, as a shape the search must
        // not settle for: it predicts these same depths a factor of 400 worse.
        let stall = spread_of(&samples, 3.96, 0.0083, 40)
            .expect("the stalled shape predicts no depth at all")
            .spread;
        assert!(
            fit.spread < 0.05 * stall,
            "the search left a spread of {} against the axial stall's {stall}",
            fit.spread
        );
    }

    /// Descents that recovered nothing carry no ratio and must not enter the
    /// buffer, or the objective takes the logarithm of zero.
    #[test]
    fn unrecoverable_descents_are_refused() {
        let mut acc = HessianFit::new();
        let x = structure(&mut Rng(1), 8);
        let g = Array1::from(vec![0.1; 24]);
        assert!(!acc.observe(x.view(), g.view(), 0.0), "a zero depth entered");
        assert!(
            !acc.observe(x.view(), g.view(), -1.0),
            "a negative depth entered"
        );
        assert!(
            !acc.observe(x.view(), g.view(), f64::NAN),
            "a non-finite depth entered"
        );
        let zero = Array1::zeros(24);
        assert!(
            !acc.observe(x.view(), zero.view(), 1.0),
            "a descent from a stationary point entered"
        );
        assert!(acc.is_empty(), "{} unusable samples were kept", acc.len());
    }

    /// The ring drops the oldest, so a long run is calibrated on where it is.
    #[test]
    fn the_buffer_keeps_the_most_recent_observations() {
        let truth = ModelParams::default();
        let samples = synthetic(0x9001, 40, truth);
        let mut acc = HessianFit::new();
        acc.capacity = 10;
        for s in &samples {
            acc.observe(s.x.view(), s.gradient.view(), s.observed);
        }
        assert_eq!(acc.len(), 10, "the ring held {} against a cap of 10", acc.len());
        assert_eq!(acc.seen(), 40, "the run counted {} descents", acc.seen());
        let last = acc.samples().back().expect("an empty ring after 40 pushes");
        assert_eq!(
            last.observed,
            samples[39].observed,
            "the newest observation is not the one at the back"
        );
    }

    /// A fit that cannot beat the defaults on its own sample is discarded, which
    /// is what bounds the damage a calibration on unrepresentative data can do.
    #[test]
    fn a_fit_that_cannot_improve_is_rejected() {
        // Data generated from the defaults themselves: no shape in the box does
        // better, so the search has nothing to adopt.
        let samples = synthetic(0x3333, 32, ModelParams::default());
        let fit = calibrate(&samples, 40);
        let spread = fit.map_or(0.0, |c| c.spread);
        assert!(
            spread < 1e-6,
            "a fit on data generated from the defaults claimed a spread of {spread}"
        );
    }
}
