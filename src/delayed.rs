//! Delayed-acceptance basin hopping: pay for the quench only when it can matter.
//!
//! Basin hopping walks the transformed surface `E~(x) = E(Q(x))`, where `Q` is a
//! local minimisation. Every proposal costs a quench, and measured on 38 points
//! that is 87 to 93 per cent of the whole charged budget: 25 gradient
//! evaluations per proposal against one for the raw energy. Two attempts to
//! shorten the quench failed and failed informatively. Extrapolating the
//! descent geometrically mispredicts the limit by 1e4 at the step where its
//! rule fires. Simply running fewer steps solves 0 of 4 at six steps and 1 of 4
//! at fifteen, against 4 of 4 at twenty-five. The chain needs a converged
//! energy because that is the surface it walks on.
//!
//! What it does *not* need is a converged energy for proposals it is going to
//! reject. Most are: the accept rate runs near one half, and the rejected half
//! pays the same 25 evaluations as the accepted half for an answer that is
//! discarded.
//!
//! # The scheme
//!
//! Let `E^` be a cheap surrogate for `E~` and let `q` be the proposal. One step
//! from `x`:
//!
//! 1. Draw `y ~ q(x, .)` and evaluate the surrogate. Accept the first stage
//!    with
//!    ```text
//!    a1 = min(1, [q(y,x) exp(-E^(y)/T)] / [q(x,y) exp(-E^(x)/T)])
//!    ```
//!    On rejection the chain stays at `x` and no quench is paid.
//! 2. Only if the first stage accepted, quench `y` and accept with
//!    ```text
//!    a2 = min(1, exp(-[(E~(y) - E~(x)) - (E^(y) - E^(x))] / T))
//!    ```
//!
//! # Why the surrogate cannot bias the answer
//!
//! The composite kernel is reversible with respect to `pi ∝ exp(-E~/T)` for any
//! `E^` whatever, which is the Christen and Fox delayed-acceptance argument.
//! Write the first stage as a proposal in its own right: it draws from `q` and
//! keeps with `a1`, so its effective proposal density from `x` to `y != x` is
//! `Q1(x,y) = q(x,y) a1(x,y)`. That density satisfies
//! `Q1(x,y) exp(-E^(x)/T) = Q1(y,x) exp(-E^(y)/T)` by construction, since `a1`
//! is a Metropolis-Hastings ratio for the surrogate target. The second stage is
//! then a Metropolis-Hastings step with proposal `Q1` against the true target,
//! whose ratio is
//!
//! ```text
//! [Q1(y,x) exp(-E~(y)/T)] / [Q1(x,y) exp(-E~(x)/T)]
//!   = exp(-[(E~(y) - E~(x)) - (E^(y) - E^(x))] / T)
//! ```
//!
//! because the `Q1` ratio contributes exactly the surrogate difference back.
//! So detailed balance holds for the pair, and a poor surrogate costs
//! acceptance rate rather than correctness: a surrogate that is constant
//! reduces the scheme to ordinary basin hopping, and a perfect one makes the
//! second stage accept with probability one.
//!
//! This is the property the crate's existing screen does not have. That screen
//! decides on a partial quench with a hand-set margin and lets the trial into
//! the chain on the partial energy, which is a different chain from the one it
//! reports.
//!
//! # What the surrogate is
//!
//! The run labels its own rows. Every quench it pays produces a pair of an
//! unrelaxed structure and the energy it relaxed to, and a 38-point run
//! produces about twelve thousand of them. The surrogate regresses the quenched
//! energy `E(Q(x))` on features of the unrelaxed structure, so the prediction
//! needs one energy evaluation and one gradient.
//!
//! Energy rather than depth, because the depth is the difference of a bounded
//! quantity and an unbounded one. Over one seed the quenched energy stays
//! inside `[-174, -150]` for 96.4 per cent of the rows while the raw energy
//! reaches 2.4e24, so the depth inherits the divergence, and a model asked for
//! the depth has to hit four significant digits for the sum to land inside the
//! temperature. The numbers this cost are below.
//!
//! # Measured, and not yet paying
//!
//! The scheme is exact and the saving is real; the surrogate is not good
//! enough for the saving to be worth what it costs. On 38 points at 2e5:
//!
//! | variant | stage-1 reject | stage-2 reject | hops | acceptance | accepted moves |
//! |---------|----------------|----------------|------|------------|----------------|
//! | screen (control) | -- | -- | 5561 | 0.560 | 3114 |
//! | delayed, no abstention | 0.485 | 0.571 | 10058 | 0.223 | 2243 |
//! | delayed, abstention at 0.5 T | 0.868 | 0.960 | 21138 | 0.014 | 296 |
//!
//! Nearly half of all proposals avoid a quench and the hop count nearly
//! doubles, which is the mechanism working. What it buys is spent again at the
//! second stage, where the surrogate's error rejects more moves than the extra
//! proposals supply, and abstention on the predictive spread makes it worse
//! rather than better: the posterior's own uncertainty is not calibrated
//! against its error, so raising the bar filtered out the cases it was right
//! about along with the ones it was wrong about.
//!
//! # What the second stage is actually asking for
//!
//! Adding the gradient, the displacement split of [`features_orthogonal`] and
//! the model-Hessian depth of [`features_with_depth`] costs no charged
//! evaluations and does not help. Over 8 seeds at 2e5 on 38 points, means:
//!
//! | design | stage-1 reject | stage-2 reject | hops | acceptance | accepted moves | solved |
//! |--------|----------------|----------------|------|------------|----------------|--------|
//! | screen (control) | -- | -- | 6059 | 0.567 | 3432 | 7/8 |
//! | gradient norm only | 0.503 | 0.755 | 9704 | 0.255 | 1974 | 4/7 |
//! | split and depth | 0.588 | 0.907 | 15128 | 0.172 | 1469 | 3/8 |
//!
//! The reason sits in the rows the run labels for itself. Dumping the
//! 4676 design rows of one seed: the quenched energy of a proposal lies in
//! `[-174, -150]` for 96.4 per cent of them, while the raw energy of the same
//! proposals has a median of 6.3e3 and a maximum of 2.4e24, because a random
//! displacement routinely overlaps a pair. The depth is therefore almost
//! exactly minus the raw energy, and a model that predicts the depth to a
//! quarter of a decade, the accuracy these features support, misses the
//! quenched energy by thousands. The second stage needs it to within the
//! temperature, 0.8.
//!
//! Fitting the quenched energy directly on a held-out half of those rows gives
//! a median absolute error of 2.78 with the depth column and 2.77 without it,
//! against 1.54 for predicting the training median and ignoring the structure
//! entirely. The features are worse than a constant, and the module's own
//! reversibility argument says what a constant surrogate does: it reduces the
//! scheme to ordinary basin hopping, which is the control.
//!
//! The unbounded columns also destroy the posterior they are fitted in.
//! Accumulated over one seed the precision matrix `V0^-1 + X'X` has entries
//! spanning `1e110`, and its numerical rank in double precision is 1 of 11:
//! every column except the largest is annihilated by the rounding error of that
//! column. Both designs above measure 1 of 11, so the comparison between them
//! is a comparison of which single column survived.
//!
//! # Bounding the design, and what it settles
//!
//! Both faults have a fix and the fixes work. [`squash`] maps every column that
//! a repulsive wall drives into a bounded one, and the prediction is the
//! quenched energy itself rather than a depth added back to an energy of median
//! 6.5e3. Over four seeds, `V0^-1 + X'X` on the trial rows of a run:
//!
//! | design | condition number | numerical rank | diagonal spread |
//! |--------|------------------|----------------|-----------------|
//! | linear columns | 1.6e71 | 2 of 11 | 1.9e70 |
//! | bounded | 9.4e6 | 11 of 11 | 1.3e3 |
//! | bounded and standardised | 6.9e5 | 11 of 11 | 1.0 |
//!
//! Held out on half the rows of each of four seeds, median absolute error in
//! the quenched energy, against the constant that predicts the training median
//! and looks at no structure at all:
//!
//! | predictor | temporal split | random split |
//! |-----------|----------------|--------------|
//! | depth added to the raw energy, linear columns | 6.9e5 | 4.9e5 |
//! | quenched energy, bounded and standardised | 1.85 | 1.34 |
//! | the training median | 1.96 | 1.32 |
//!
//! A factor of 3.7e5 of the error was parametrisation and conditioning, on both
//! splits, and it is gone. What remains is a dead heat with a constant: the bounded
//! model wins 1 of 4 seeds on the temporal split and 2 of 4 on the random one.
//! A surrogate that ties a constant is a constant for the purposes of this
//! scheme, and the reversibility argument above says what a constant does. It
//! reduces the composite kernel to ordinary basin hopping.
//!
//! The search says the same thing twice. Over 8 seeds at 2e5 on 38 points:
//!
//! | arm | stage-1 reject | stage-2 reject | hops | acceptance | accepted moves | solved |
//! |-----|----------------|----------------|------|------------|----------------|--------|
//! | screen (control) | -- | -- | 6059 | 0.567 | 3432 | 7/8 |
//! | delayed, linear columns, depth | 0.503 | 0.755 | 9704 | 0.255 | 1974 | 4/7 |
//! | delayed, bounded, abstaining at 0.5 T | -- | -- | 5527 | 0.568 | 3140 | 7/8 |
//! | delayed, bounded, never abstaining | 0.865 | 0.671 | 26599 | 0.046 | 1153 | 3/7 |
//!
//! At the tolerance the driver configures, the first stage runs zero times in
//! 5527 hops. That is the calibration working rather than failing: the
//! posterior's predictive spread never falls below `0.4`, because its own
//! held-out error is 1.3 to 1.9, so it declines to speak and the run is the
//! control minus the evaluations the abstentions paid for. Forced to speak it
//! rejects 86.5 per cent of proposals at the first stage and 67.1 per cent of
//! the survivors at the second, and acceptance falls to 0.046.
//!
//! So the ceiling is the feature set. A first stage needs the quenched energy
//! to within the temperature, `0.8`; coordination, contact, spacing, gradient,
//! displacement split and a model-Hessian depth give 1.3 to 1.9, which a
//! constant also gives. Nothing about the encoding is left to try: the design
//! has full rank, the target is the bounded one, and the error did not move.

use crate::model_hessian;
use crate::screen::Screen;
use ndarray::{Array1, ArrayView1};

/// Features the surrogate regresses on.
///
/// The last of them is the model-Hessian depth, which is an estimate of the
/// regressand itself rather than a correlate of it, so it enters the design as
/// its own column and the fit supplies the one coefficient that rescales it.
pub const FEATURES: usize = 11;

/// Conjugate-gradient iterations spent on the model-Hessian solve.
///
/// A solve on 38 points costs this many operator products at 703 pairs each,
/// against the roughly 25 charged gradient evaluations of the quench it is
/// deciding about, so the count is set by where the depth stops moving and not
/// by what it costs. A truncated solve underestimates the depth by a factor
/// that varies slowly with the geometry, and the surrogate regresses on the
/// number, so what a longer solve would buy is mostly a scale the fit already
/// supplies.
pub const DEPTH_ITERS: usize = 12;

/// Bounded image of a column that is not bounded on the sampled distribution.
///
/// `sign(v) ln(1 + |v|)`, which is odd, monotone, smooth at zero and agrees
/// with the identity to first order there, so a column that never leaves the
/// ordinary range is barely touched while one that runs away is compressed.
///
/// The columns that need it are the ones a repulsive wall drives. A single
/// Lennard-Jones pair at 0.4 of the mean spacing already contributes 6.1e4 and
/// the `r^-12` term diverges below that, so over one seed the raw energy column
/// reaches 2.4e24 and the `|g|^2` column 1.5e55. A Gram matrix holding both
/// those and an intercept spans 1e110 and has numerical rank 1 of 11 in double
/// precision: every other column is annihilated by the rounding error of the
/// largest. Under this map the same columns reach 56 and 127, and the rank is
/// the full 11.
///
/// Rank is what the map buys, not accuracy. The prediction it supports is in
/// the module header.
pub fn squash(v: f64) -> f64 {
    v.signum() * v.abs().ln_1p()
}

/// Cheap structural summary of an unrelaxed structure.
///
/// Coordination counts and the closest contact, which is what says whether a
/// structure has room to relax: a proposal with an overlapping pair has a large
/// depth ahead of it, and a proposal already near a packing has little.
pub fn features(x: ArrayView1<f64>, n: usize, raw: f64) -> Array1<f64> {
    features_with_gradient(x, n, raw, 0.0)
}

/// As [`features_with_gradient`], splitting the displacement from the
/// incumbent along and across the descent direction.
///
/// A single gradient norm collapses an anisotropic quantity to one number. The
/// depth a relaxation finds is `1/2 g^T H^-1 g`, so it is set by how the
/// displacement distributes over the curvature, not by its length. The split
/// recovers the part of that which costs nothing: the component of `d = y - x`
/// along the gradient is what simple descent undoes and predicts how much
/// energy comes back, while the orthogonal component is the part descent does
/// not remove and is what decides which basin the trial lands in.
///
/// This is the orthogonal-deviation reading used to visualise how far a band
/// wanders off a reaction path, applied to a quench rather than a path: the
/// descent direction plays the role of the path tangent.
pub fn features_orthogonal(
    x: ArrayView1<f64>,
    n: usize,
    raw: f64,
    gradient: ArrayView1<f64>,
    from: ArrayView1<f64>,
) -> Array1<f64> {
    let gnorm = gradient.iter().fold(0.0_f64, |a, q| a + q * q).sqrt();
    let mut base = features_with_gradient(x, n, raw, gnorm);
    // Displacement from the structure the trial was proposed from, split by
    // the descent direction at the trial.
    let (mut along, mut dsq) = (0.0_f64, 0.0_f64);
    if from.len() == x.len() && gnorm > 1e-12 {
        for i in 0..x.len() {
            let d = x[i] - from[i];
            along += d * (gradient[i] / gnorm);
            dsq += d * d;
        }
    }
    let perp = (dsq - along * along).max(0.0).sqrt();
    let total = dsq.sqrt();
    let mut out = Array1::zeros(FEATURES);
    for (i, v) in base.iter_mut().enumerate() {
        out[i] = *v;
    }
    out[7] = along.abs();
    out[8] = perp;
    // How much of the step descent cannot undo, which is scale free and is the
    // quantity that separates a step within a basin from one that leaves it.
    out[9] = if total > 1e-12 { perp / total } else { 0.0 };
    out
}

/// As [`features_orthogonal`], with the depth a model Hessian predicts.
///
/// The feature every other column only gestures at. The others describe the
/// structure and let the fit find the map from a description to a depth; this
/// one is `1/2 g^T H^-1 g` computed for the geometry in hand, which is the
/// leading term of the depth itself. See [`crate::model_hessian`] for why the
/// operator costs no charged evaluations.
///
/// One builder for both ends of a quench, because the surrogate is fitted on
/// unrelaxed structures and consulted about the relaxed one the chain stands
/// on. Passing a zero gradient gives the relaxed end its correct features: the
/// gradient columns vanish, the displacement split vanishes with them, and the
/// depth is zero, which is the definition of a structure with nothing left to
/// recover.
///
/// Measured on 38 points, the column carries no information the design does not
/// already hold: against the true depth of 4676 quenches its log-log
/// correlation is 0.9932 where the `|g|^2` column alone gives 0.9927, and a
/// power law fitted to it leaves a residual of 0.245 decades against 0.255. The
/// module header has what that costs the second stage.
pub fn features_with_depth(
    x: ArrayView1<f64>,
    n: usize,
    raw: f64,
    gradient: ArrayView1<f64>,
    from: ArrayView1<f64>,
) -> Array1<f64> {
    let mut out = features_orthogonal(x, n, raw, gradient, from);
    // Bounded, for the same reason the gradient columns are: the depth inherits
    // the divergence of the gradient it is a quadratic form in, and reaches
    // 9.0e53 over one seed.
    out[10] = squash(model_hessian::depth(x, n, gradient, DEPTH_ITERS));
    out
}

/// As [`features`], with the gradient norm at the unrelaxed point.
///
/// A free feature: the evaluation the first stage already pays for computes the
/// gradient and throws it away. In a locally quadratic basin the depth a
/// relaxation will find goes as `|g|^2 / 2 lambda`, so the squared norm is the
/// leading term of the quantity being predicted rather than a proxy for it, and
/// the linear term is carried alongside because the basin is only approximately
/// quadratic.
///
/// The energy and both gradient columns go in through [`squash`]. All three
/// diverge on a proposal that overlaps a pair, and a design carrying them
/// linearly has numerical rank 1 whatever else is in it.
pub fn features_with_gradient(
    x: ArrayView1<f64>,
    n: usize,
    raw: f64,
    gnorm: f64,
) -> Array1<f64> {
    if n < 2 {
        let mut v = Array1::zeros(FEATURES);
        v[0] = 1.0;
        v[1] = squash(raw);
        v[5] = squash(gnorm);
        v[6] = squash(gnorm * gnorm);
        return v;
    }
    let mut nearest = vec![f64::INFINITY; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = (0..3)
                .map(|k| {
                    let v = x[3 * i + k] - x[3 * j + k];
                    v * v
                })
                .sum();
            if d < nearest[i] {
                nearest[i] = d;
            }
        }
    }
    for v in nearest.iter_mut() {
        *v = v.sqrt();
    }
    let scale = nearest.iter().sum::<f64>() / n as f64;
    let closest = nearest.iter().cloned().fold(f64::INFINITY, f64::min);
    let cut = 1.2 * scale;
    let mut coord = 0usize;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = (0..3)
                .map(|k| {
                    let v = x[3 * i + k] - x[3 * j + k];
                    v * v
                })
                .sum::<f64>()
                .sqrt();
            if d < cut {
                coord += 1;
            }
        }
    }
    let mean_coord = coord as f64 / n as f64;
    // The raw energy is the strongest single predictor and costs the one
    // evaluation the stage is allowed. The rest describe how much room the
    // structure has to fall.
    let mut v = Array1::zeros(FEATURES);
    v[0] = 1.0;
    v[1] = squash(raw);
    v[2] = mean_coord;
    v[3] = closest / scale.max(1e-12);
    v[4] = scale;
    v[5] = squash(gnorm);
    v[6] = squash(gnorm * gnorm);
    v
}

/// Running mean and spread of each column, for standardising the design.
///
/// [`Screen`] holds a conjugate Normal-Inverse-Gamma posterior with a prior
/// precision of `1e-3 I`, which is one ridge for every column whatever its
/// units. Under such a prior a column measured in thousands and a column
/// measured in tenths are not shrunk by the same amount, and the ridge that
/// regularises one leaves the other unconstrained. Standardising is what makes
/// the single prior mean the same thing for all of them.
///
/// Kept beside the model rather than inside the feature builders, which are
/// pure functions of one structure and cannot hold the statistics of a stream.
///
/// The statistics drift as the chain moves, so a row folded in early was
/// standardised against different numbers from a row folded in late. Welford
/// over the whole stream makes that drift slow, order `1/n` after the warmup,
/// and the alternative, refitting the posterior whenever the statistics move,
/// costs the storage the conjugate form exists to avoid.
#[derive(Debug, Clone)]
struct Columns {
    n: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl Columns {
    fn new(d: usize) -> Self {
        Self {
            n: 0,
            mean: vec![0.0; d],
            m2: vec![0.0; d],
        }
    }

    /// Folds a row in, by Welford, which is the stable way to accumulate a
    /// variance in one pass.
    fn observe(&mut self, f: ArrayView1<f64>) {
        if f.len() != self.mean.len() {
            return;
        }
        self.n += 1;
        for j in 0..self.mean.len() {
            let d = f[j] - self.mean[j];
            self.mean[j] += d / self.n as f64;
            self.m2[j] += d * (f[j] - self.mean[j]);
        }
    }

    /// Standard deviation of a column, or one where there is not yet a spread
    /// to speak of.
    fn spread(&self, j: usize) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        let v = (self.m2[j] / (self.n - 1) as f64).sqrt();
        if v > 1e-12 { v } else { 1.0 }
    }

    /// The row as the model sees it. Column zero is the intercept and is left
    /// alone: it has no spread, and centring it would remove the only column
    /// that can carry the mean of the target.
    fn standardise(&self, f: ArrayView1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(f.len());
        out[0] = f[0];
        for j in 1..f.len() {
            out[j] = (f[j] - self.mean[j]) / self.spread(j);
        }
        out
    }
}

/// A learned stand-in for the quenched energy, with its own uncertainty.
#[derive(Debug)]
pub struct Surrogate {
    model: Screen,
    /// Per-column statistics the design is standardised against.
    columns: Columns,
    /// Quenches required before the first stage is allowed to reject.
    ///
    /// Enforced here rather than left to the model: `Screen::predict` answers
    /// from whatever posterior it holds, including a prior fitted to nothing,
    /// and the warmup it carries gates a different decision. Acting on a
    /// posterior fitted to a handful of quenches would reject good proposals
    /// for no reason. The scheme is exact either way, so waiting costs only
    /// savings not yet being made.
    pub warmup: usize,
    /// First-stage tests run.
    pub stage_one: usize,
    /// First-stage rejections, each of which saved a quench.
    pub stage_one_rejected: usize,
    /// First stages skipped because the posterior was too uncertain to speak.
    ///
    /// An abstention pays the quench the run would have paid anyway, so it is
    /// the cheap way of being unsure.
    pub abstained: usize,
    /// Second-stage tests run.
    pub stage_two: usize,
    /// Second-stage rejections, which are the surrogate's mistakes.
    pub stage_two_rejected: usize,
}

impl Default for Surrogate {
    fn default() -> Self {
        Self::new()
    }
}

impl Surrogate {
    /// A surrogate with no evidence, which accepts every first stage.
    pub fn new() -> Self {
        Self {
            // A long warmup: acting on a posterior fitted to a handful of
            // quenches would reject good proposals for no reason, and the
            // scheme is exact either way, so waiting costs only the savings it
            // has not started making.
            model: Screen::new(FEATURES, 64, 0.0, 0.5),
            columns: Columns::new(FEATURES),
            warmup: 64,
            abstained: 0,
            stage_one: 0,
            stage_one_rejected: 0,
            stage_two: 0,
            stage_two_rejected: 0,
        }
    }

    /// Quenches the surrogate has been trained on.
    pub fn seen(&self) -> usize {
        self.model.observations()
    }

    /// Predicted quenched energy of an unrelaxed structure, or `None` while the
    /// posterior has too little evidence to be worth consulting.
    ///
    /// `raw` is the structure's own energy, which the caller has already paid
    /// for and which enters the design through [`squash`].
    pub fn predict(&self, x: ArrayView1<f64>, n: usize, raw: f64) -> Option<f64> {
        self.predict_at(x, n, raw, f64::INFINITY)
    }

    /// As [`Surrogate::predict`], abstaining when the predictive spread exceeds
    /// `tolerance`.
    ///
    /// Abstention is what an uncertainty is for here. The two ways of being
    /// wrong do not cost the same: abstaining pays one quench, which is what
    /// the run would have paid anyway, while guessing wrongly passes a proposal
    /// the second stage then rejects, and that costs an accepted move. So the
    /// first stage should only speak where the posterior is sharp relative to
    /// the temperature that scales the acceptance ratio.
    ///
    /// Measured without it, the second stage rejected 57 per cent of what the
    /// first passed, and composite acceptance fell from 0.56 to 0.223: the
    /// surrogate was confidently wrong often enough to lose more moves than
    /// the extra proposals bought.
    pub fn predict_at(
        &self,
        x: ArrayView1<f64>,
        n: usize,
        raw: f64,
        tolerance: f64,
    ) -> Option<f64> {
        self.predict_full(x, n, raw, 0.0, tolerance)
    }

    /// As [`Surrogate::predict_at`], given the gradient norm at `x`.
    pub fn predict_full(
        &self,
        x: ArrayView1<f64>,
        n: usize,
        raw: f64,
        gnorm: f64,
        tolerance: f64,
    ) -> Option<f64> {
        self.predict_features(features_with_gradient(x, n, raw, gnorm).view(), tolerance)
    }

    /// Prediction from a feature vector the caller built, which is how the
    /// orthogonal split is supplied.
    ///
    /// The number returned is the quenched energy, not a correction to be added
    /// to anything. Regressing the difference `E(Q(y)) - E(y)` and adding `E(y)`
    /// back asks the model for a quantity of median 6.5e3 accurately enough to
    /// leave the sum inside the temperature, 0.8, which is four significant
    /// digits. The quenched energy itself lies in `[-174, -150]` for 96.4 per
    /// cent of the rows a 38-point run labels, so predicting it directly is
    /// asking for the number that is actually bounded.
    pub fn predict_features(&self, f: ArrayView1<f64>, tolerance: f64) -> Option<f64> {
        if self.model.observations() < self.warmup {
            return None;
        }
        let (quenched, sd) = self.model.predict(self.columns.standardise(f).view())?;
        if !sd.is_finite() || sd.sqrt() > tolerance {
            return None;
        }
        Some(quenched)
    }

    /// Records a quench from a feature vector the caller built.
    ///
    /// The row updates the column statistics before it is standardised, so a
    /// surrogate that has seen one row has a spread to divide by.
    pub fn observe_features(&mut self, f: ArrayView1<f64>, quenched: f64) {
        if !quenched.is_finite() || f.iter().any(|v| !v.is_finite()) {
            return;
        }
        self.columns.observe(f);
        let z = self.columns.standardise(f);
        self.model.observe(z.view(), quenched);
    }

    /// Records a quench: the structure before it and the energy after.
    pub fn observe(&mut self, x: ArrayView1<f64>, n: usize, raw: f64, quenched: f64) {
        self.observe_full(x, n, raw, 0.0, quenched)
    }

    /// As [`Surrogate::observe`], with the gradient norm the prediction used.
    ///
    /// The training features must match the prediction features or the model
    /// is fitted on one thing and consulted about another.
    pub fn observe_full(
        &mut self,
        x: ArrayView1<f64>,
        n: usize,
        raw: f64,
        gnorm: f64,
        quenched: f64,
    ) {
        let f = features_with_gradient(x, n, raw, gnorm);
        self.observe_features(f.view(), quenched);
    }

    /// First-stage acceptance probability for a symmetric proposal.
    ///
    /// Symmetric because every move in this crate's library is: a displacement,
    /// a relocation, a twin and a symmetrisation all propose `y` from `x` with
    /// the same density as `x` from `y`. Under asymmetry the ratio `q(y,x) /
    /// q(x,y)` multiplies in, and this returns the wrong number rather than a
    /// slightly wrong one, which is why it says so here.
    pub fn stage_one_probability(&self, surrogate_x: f64, surrogate_y: f64, t: f64) -> f64 {
        let d = surrogate_y - surrogate_x;
        if d <= 0.0 {
            1.0
        } else {
            (-d / t.max(1e-12)).exp()
        }
    }

    /// Second-stage acceptance probability, which corrects the first.
    ///
    /// The surrogate difference is subtracted back out, so what remains is the
    /// error the surrogate made on this pair. A surrogate that was right about
    /// the difference gives one.
    pub fn stage_two_probability(
        &self,
        surrogate_x: f64,
        surrogate_y: f64,
        true_x: f64,
        true_y: f64,
        t: f64,
    ) -> f64 {
        let d = (true_y - true_x) - (surrogate_y - surrogate_x);
        if d <= 0.0 {
            1.0
        } else {
            (-d / t.max(1e-12)).exp()
        }
    }

    /// Quenches avoided as a share of proposals, which is the whole point.
    pub fn saved_share(&self) -> f64 {
        if self.stage_one == 0 {
            return 0.0;
        }
        self.stage_one_rejected as f64 / self.stage_one as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The claim the whole scheme rests on, checked by simulation rather than
    /// asserted from the algebra: the composite kernel has the right
    /// stationary distribution whatever the surrogate says.
    ///
    /// A five-state chain with a deliberately bad surrogate, run long, and its
    /// empirical occupancy compared against `exp(-E/T)` normalised. If the
    /// derivation were wrong this would come out skewed toward wherever the
    /// surrogate is optimistic.
    fn stationary_under(surrogate: &[f64], truth: &[f64], t: f64, steps: usize) -> Vec<f64> {
        let n = truth.len();
        let mut rng = StdRng::seed_from_u64(20260806);
        let s = Surrogate::new();
        let mut at = 0usize;
        let mut count = vec![0usize; n];
        for _ in 0..steps {
            // Symmetric proposal: any other state, uniformly.
            let mut y = rng.random_range(0..n);
            while y == at {
                y = rng.random_range(0..n);
            }
            let a1 = s.stage_one_probability(surrogate[at], surrogate[y], t);
            if rng.random::<f64>() < a1 {
                let a2 = s.stage_two_probability(
                    surrogate[at],
                    surrogate[y],
                    truth[at],
                    truth[y],
                    t,
                );
                if rng.random::<f64>() < a2 {
                    at = y;
                }
            }
            count[at] += 1;
        }
        count.iter().map(|c| *c as f64 / steps as f64).collect()
    }

    fn boltzmann(e: &[f64], t: f64) -> Vec<f64> {
        let w: Vec<f64> = e.iter().map(|v| (-v / t).exp()).collect();
        let z: f64 = w.iter().sum();
        w.into_iter().map(|v| v / z).collect()
    }

    #[test]
    fn the_composite_kernel_targets_the_true_distribution() {
        let truth = [0.0, 0.5, 1.0, 1.5, 2.0];
        let t = 0.8;
        let want = boltzmann(&truth, t);
        // A surrogate that is wrong in both directions and by a lot.
        let bad = [2.0, -1.0, 3.0, 0.0, -2.0];
        let got = stationary_under(&bad, &truth, t, 400_000);
        for i in 0..truth.len() {
            assert!(
                (got[i] - want[i]).abs() < 0.02,
                "state {i}: occupancy {:.4} against target {:.4}; a wrong surrogate has biased the chain",
                got[i],
                want[i]
            );
        }
    }

    /// And with a perfect surrogate, so that a passing test above is not
    /// passing for want of the surrogate mattering at all.
    #[test]
    fn a_perfect_surrogate_also_targets_it() {
        let truth = [0.0, 0.5, 1.0, 1.5, 2.0];
        let t = 0.8;
        let want = boltzmann(&truth, t);
        let got = stationary_under(&truth, &truth, t, 400_000);
        for i in 0..truth.len() {
            assert!((got[i] - want[i]).abs() < 0.02, "state {i}");
        }
    }

    /// A perfect surrogate must make the second stage certain, which is what
    /// says the correction is a correction and not a second filter.
    #[test]
    fn a_perfect_surrogate_never_rejects_at_the_second_stage() {
        let s = Surrogate::new();
        for (a, b) in [(0.0, 1.0), (1.0, 0.0), (-3.0, 2.5)] {
            let p = s.stage_two_probability(a, b, a, b, 0.8);
            assert!((p - 1.0).abs() < 1e-12, "second stage rejected a perfect surrogate: {p}");
        }
    }

    /// A constant surrogate must reduce the scheme to ordinary basin hopping:
    /// the first stage always passes and the second carries the real ratio.
    #[test]
    fn a_constant_surrogate_reduces_to_plain_metropolis() {
        let s = Surrogate::new();
        let t = 0.8;
        assert_eq!(s.stage_one_probability(1.0, 1.0, t), 1.0);
        let p = s.stage_two_probability(1.0, 1.0, 0.0, 0.7, t);
        assert!((p - (-0.7f64 / t).exp()).abs() < 1e-12, "got {p}");
    }

    /// The surrogate has to learn a quenched energy it is shown, and learn it
    /// as a function of the structure rather than of the raw energy it is
    /// handed alongside, since a relaxation lands where the structure says.
    #[test]
    fn the_surrogate_learns_a_quenched_energy_it_is_shown() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut s = Surrogate::new();
        let n = 8;
        let truth = |f: &Array1<f64>| -5.0 - 2.0 * f[2];
        for _ in 0..300 {
            let mut x = Array1::zeros(3 * n);
            for v in x.iter_mut() {
                *v = rng.random_range(-2.0..2.0);
            }
            // The raw energy carries no information about where the structure
            // relaxes to, which is the case the design has to survive.
            let raw: f64 = rng.random_range(-10.0..10.0);
            let f = features(x.view(), n, raw);
            s.observe(x.view(), n, raw, truth(&f));
        }
        assert!(s.seen() >= 200, "only {} observations", s.seen());
        let mut x = Array1::zeros(3 * n);
        for v in x.iter_mut() {
            *v = rng.random_range(-2.0..2.0);
        }
        let want = truth(&features(x.view(), n, 0.0));
        let got = s.predict(x.view(), n, 0.0).expect("no prediction after 300 quenches");
        assert!(
            (got - want).abs() < 0.2,
            "predicted quenched energy {got} against the {want} the structure sets"
        );
    }

    /// Abstention has to actually abstain: a tolerance of zero means the
    /// first stage never speaks, whatever the posterior holds.
    #[test]
    fn a_zero_tolerance_abstains_always() {
        let mut rng = StdRng::seed_from_u64(17);
        let mut s = Surrogate::new();
        let n = 8;
        for _ in 0..300 {
            let mut x = Array1::zeros(3 * n);
            for v in x.iter_mut() {
                *v = rng.random_range(-2.0..2.0);
            }
            let raw: f64 = rng.random_range(-10.0..10.0);
            let f = features(x.view(), n, raw);
            s.observe(x.view(), n, raw, -5.0 - 2.0 * f[2]);
        }
        let mut x = Array1::zeros(3 * n);
        for v in x.iter_mut() {
            *v = rng.random_range(-2.0..2.0);
        }
        assert!(s.predict_at(x.view(), n, 0.0, 0.0).is_none());
        assert!(s.predict_at(x.view(), n, 0.0, f64::INFINITY).is_some());
    }

    /// Every column has to stay bounded over the range a Lennard-Jones
    /// proposal actually reaches, which is what keeps the Gram matrix in the
    /// range double precision can hold.
    #[test]
    fn the_design_stays_bounded_over_twenty_four_decades() {
        let n = 8;
        let mut x = Array1::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = (i % 2) as f64 * 1.1;
            x[3 * i + 1] = (i / 2) as f64 * 1.1;
        }
        let mut worst_squashed: f64 = 0.0;
        let mut worst_plain: f64 = 0.0;
        for p in 0..25 {
            let raw = 10f64.powi(p);
            let gnorm = 10f64.powi(p + 2);
            let f = features_with_gradient(x.view(), n, raw, gnorm);
            for v in f.iter() {
                assert!(v.is_finite(), "column ran to {v} at raw {raw:e}");
                worst_squashed = worst_squashed.max(v.abs());
            }
            worst_plain = worst_plain.max(gnorm * gnorm);
        }
        // A design carrying these columns linearly reaches 1e54, and its Gram
        // matrix the square of that, which is where the rank goes.
        assert!(
            worst_plain > 1e50,
            "the test did not reach the range that breaks the design: {worst_plain:e}"
        );
        assert!(
            worst_squashed < 200.0,
            "largest design entry {worst_squashed} over 24 decades of energy"
        );
    }

    /// And the spread of `X'X` has to follow, since that is the matrix the
    /// posterior is solved in.
    #[test]
    fn the_gram_diagonal_spread_stays_within_double_precision() {
        let n = 8;
        let mut x = Array1::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = (i % 2) as f64 * 1.1;
            x[3 * i + 1] = (i / 2) as f64 * 1.1;
            x[3 * i + 2] = (i % 3) as f64 * 0.9;
        }
        let mut diag = vec![0.0f64; FEATURES];
        let mut plain = vec![0.0f64; FEATURES];
        for p in 0..25 {
            let raw = 10f64.powi(p);
            let gnorm = 10f64.powi(p + 2);
            let f = features_with_gradient(x.view(), n, raw, gnorm);
            for j in 0..FEATURES {
                diag[j] += f[j] * f[j];
            }
            // The same design without the bounding map, for the columns the
            // map acts on.
            plain[1] += raw * raw;
            plain[6] += (gnorm * gnorm) * (gnorm * gnorm);
        }
        let live: Vec<f64> = diag.iter().cloned().filter(|v| *v > 0.0).collect();
        let hi = live.iter().cloned().fold(0.0f64, f64::max);
        let lo = live.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            hi / lo < 1e8,
            "bounded Gram diagonal spans {:e}, which a single ridge cannot regularise",
            hi / lo
        );
        assert!(
            plain[6] / plain[1] > 1e50,
            "the unbounded comparison did not reach the range that breaks it: {:e}",
            plain[6] / plain[1]
        );
    }

    /// The depth column has to vanish exactly at a relaxed structure, since
    /// that is the row the both-ends training supplies and the row the first
    /// stage reads back when it prices the incumbent.
    #[test]
    fn the_depth_column_vanishes_for_a_relaxed_structure() {
        let n = 8;
        let mut x = Array1::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = (i % 2) as f64 * 1.1;
            x[3 * i + 1] = (i / 2) as f64 * 1.1;
        }
        let settled = Array1::zeros(3 * n);
        let f = features_with_depth(x.view(), n, -4.0, settled.view(), x.view());
        assert_eq!(f.len(), FEATURES);
        assert_eq!(f[10], 0.0, "a relaxed structure was given a depth of {}", f[10]);
        // And the row must be the one `features` builds, or the model is
        // fitted on relaxed structures it can never be consulted about.
        let plain = features(x.view(), n, -4.0);
        for i in 0..FEATURES {
            assert_eq!(f[i], plain[i], "column {i} differs between the two builders");
        }
    }

    /// And it has to be positive and grow with the gradient at an unrelaxed
    /// one, or the column carries no information about how far a trial falls.
    /// The quadratic scaling lives under the bounding map, so the check undoes
    /// the map first: what the column holds is `ln(1 + 1/2 g^T H^-1 g)`.
    #[test]
    fn the_depth_column_grows_with_the_gradient() {
        let n = 10;
        let mut x = Array1::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = (i % 3) as f64 * 1.1;
            x[3 * i + 1] = ((i / 3) % 3) as f64 * 1.1;
            x[3 * i + 2] = (i / 9) as f64 * 1.1;
        }
        let mut g = Array1::zeros(3 * n);
        for i in 0..n {
            g[3 * i + 1] = 0.1 * ((i % 5) as f64 - 2.0);
        }
        let from = Array1::zeros(3 * n);
        let small = features_with_depth(x.view(), n, 0.0, g.view(), from.view())[10];
        let big = features_with_depth(x.view(), n, 0.0, (&g * 2.0).view(), from.view())[10];
        assert!(small > 0.0, "depth column {small} is not positive");
        assert!(big > small, "doubling the gradient did not raise the column");
        let (a, b) = (small.exp_m1(), big.exp_m1());
        assert!(
            (b / a - 4.0).abs() < 1e-6,
            "under the bounding map, doubling the gradient scaled the depth by {}, wanted the quadratic form's 4",
            b / a
        );
    }

    /// The surrogate has to recover a quenched energy set by the model-Hessian
    /// column, which is the claim the column is there to make: the fit supplies
    /// a scale, not a shape.
    #[test]
    fn the_surrogate_recovers_an_energy_set_by_the_model_hessian() {
        let mut rng = StdRng::seed_from_u64(918);
        let mut s = Surrogate::new();
        let n = 8;
        let mut sample = |rng: &mut StdRng| {
            let mut x = Array1::zeros(3 * n);
            for v in x.iter_mut() {
                *v = rng.random_range(-2.0..2.0);
            }
            let mut g = Array1::zeros(3 * n);
            for v in g.iter_mut() {
                *v = rng.random_range(-0.5..0.5);
            }
            (x, g)
        };
        let from = Array1::zeros(3 * n);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for _ in 0..400 {
            let (x, g) = sample(&mut rng);
            let raw: f64 = rng.random_range(-10.0..10.0);
            let f = features_with_depth(x.view(), n, raw, g.view(), from.view());
            // The truth the model is shown: a minimum near -170, deeper where
            // the model Hessian says more energy is available.
            let y = -170.0 - 0.75 * f[10];
            lo = lo.min(y);
            hi = hi.max(y);
            s.observe_features(f.view(), y);
        }
        let (x, g) = sample(&mut rng);
        let f = features_with_depth(x.view(), n, 0.0, g.view(), from.view());
        let want = -170.0 - 0.75 * f[10];
        let got = s
            .predict_features(f.view(), f64::INFINITY)
            .expect("no prediction after 400 quenches");
        // Against the spread of the quantity, not against its magnitude: the
        // intercept carries the -170 and says nothing about whether the column
        // was used.
        let spread = hi - lo;
        assert!(
            (got - want).abs() < 0.05 * spread,
            "predicted quenched energy {got} against {want}, an error of {} over a signal spread of {spread}",
            (got - want).abs()
        );
    }

    /// A cold surrogate must decline to predict rather than guess, so the
    /// scheme starts as plain basin hopping and becomes cheaper as it learns.
    #[test]
    fn a_cold_surrogate_declines_to_predict() {
        let s = Surrogate::new();
        let x = Array1::zeros(24);
        assert!(s.predict(x.view(), 8, -1.0).is_none());
    }
}
