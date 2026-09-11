//! Box-constrained search, split on whether a gradient exists.
//!
//! * [`box_ensemble_optimize`] requires a gradient: kick, quench, one
//!   [`HistoryHook`] per replica over one [`MinimumHistory`].
//! * [`box_values_ensemble_optimize`] is the same replica/history object
//!   without a user gradient: kick, pattern-search, finite-difference
//!   certificate. [`MinimumHistory`] only admits a point when that
//!   certificate is flat. One replica and no gradient still uses
//!   [`crate::methods::portfolio::portfolio_optimize`].
//!
//! A box is not a point set. Cluster
//! [`crate::methods::cluster_hopping::Config::recommended`] does not apply.
//! Never a third history table.

use std::sync::Mutex;

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::bias::{BasinBias, Bias, Fingerprint};
use crate::descriptor_space::DescriptorGeometry;
use crate::methods::ensemble::HistoryMode;
use crate::methods::gle_langevin::{GleNoise, LangevinStepper};
use crate::methods::local_polish::{projected_gradient, projected_gradient_polish};
use crate::methods::minima_hopping::{
    EscapeFeedback, HistoryHook, HistoryMembership, HistoryReport, MinimumHistory,
    SharedDesignHistory,
};
use crate::methods::portfolio::{PortfolioPolicy, portfolio_optimize_seeded};
use crate::movekernel::reflect_into_box;
use crate::pes_exploration::{ExactStructureWitness, StructureContext};

#[cfg(feature = "history-nng")]
use crate::history_nng::{HistoryNngClient, HistoryNngServer};

/// Relative max-norm that identifies two quenched points as one minimum.
pub const IDENTITY_TOL: f64 = 1e-3;
/// Starting Gaussian scale, matching the portfolio hop arm.
const STEP0: f64 = 0.25;
/// Floor on the per-hop quench so a hop is more than a single evaluation.
const MIN_QUENCH: usize = 8;

/// Gradient-driven escape segment within an existing box hop chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GleEscapeConfig {
    /// Maximum integration steps per escape, shortened to reserve quench work.
    pub steps: usize,
    /// Lower frequency of the colored-noise fit; also sets the common timestep cap.
    pub omega0: f64,
    /// Requested integration timestep, not physical elapsed time.
    pub dt: f64,
    /// Colored extended-state noise or a scalar-white control.
    pub noise: GleNoise,
}

impl Default for GleEscapeConfig {
    fn default() -> Self {
        Self {
            steps: 16,
            omega0: 0.2,
            dt: 0.01,
            noise: GleNoise::Colored,
        }
    }
}

impl GleEscapeConfig {
    fn validate(self) {
        assert!(self.steps > 0, "Langevin escape steps must be positive");
        assert!(
            self.omega0.is_finite() && self.omega0 > 0.0,
            "omega0 must be finite and positive"
        );
        assert!(
            self.dt.is_finite() && self.dt > 0.0,
            "dt must be finite and positive"
        );
        self.noise.validate();
    }
}

/// Proposal mechanism; all choices retain the same quench and history path.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BoxEscape {
    /// Reflected Gaussian kick, without force work in the proposal.
    #[default]
    Gaussian,
    /// Persistent Langevin noise with caller-owned positions and fresh launch forces.
    Langevin(GleEscapeConfig),
}

/// Configuration for one communicating-chain box ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxEnsembleConfig {
    /// Chains that divide the aggregate budget.
    pub replicas: usize,
    /// Combined objective and gradient work units.
    pub budget: usize,
    /// Whether replicas share one history, keep a private one, or none.
    pub history: HistoryMode,
    /// Which visits count as known under the hook.
    pub membership: HistoryMembership,
    /// Scaled max-norm that identifies two quenched points.
    pub identity_tol: f64,
    /// Foreign visits paid into this chain's bias, capped per look.
    pub shared_deposits: usize,
    /// Per-chain escape mechanism. Langevin segments require an analytic gradient.
    pub escape: BoxEscape,
}

impl Default for BoxEnsembleConfig {
    fn default() -> Self {
        Self {
            replicas: 4,
            budget: 8_000,
            history: HistoryMode::Shared,
            membership: HistoryMembership::Accepted,
            identity_tol: IDENTITY_TOL,
            shared_deposits: 8,
            escape: BoxEscape::Gaussian,
        }
    }
}

impl BoxEnsembleConfig {
    /// Per-replica budgets, remainder to the low indices.
    pub fn budgets(&self) -> Vec<usize> {
        let n = self.replicas.max(1);
        (0..n)
            .map(|replica| self.budget / n + usize::from(replica < self.budget % n))
            .collect()
    }
}

/// Outcome of one box ensemble.
#[derive(Clone, Debug)]
pub struct BoxEnsembleResult {
    /// Best finite feasible point evaluated by any replica, including probes.
    pub best_pos: Array1<f64>,
    /// Objective at [`BoxEnsembleResult::best_pos`].
    pub best_val: f64,
    /// Direct objective evaluations charged by the hop (polish included).
    pub n_evals: usize,
    /// Gradient evaluations charged by the hop (polish included).
    pub n_grads: usize,
    /// Accepted plus rejected hops across replicas.
    pub hops: usize,
    /// Certified observations reported to a history.
    pub history_observations: usize,
    /// Distinct exact identities in the shared history, or the largest private table.
    pub history_minima: usize,
    /// History attempts, refusals and summed operation seconds across replicas.
    ///
    /// Includes validation and waiting, not connection setup or PES calls.
    pub history_cost: (usize, usize, f64),
    /// Bias deposits made on behalf of other chains' visits.
    pub shared_deposits: usize,
}

/// Outcome of [`ensemble_hop_optimize`]: hop ledger plus history size.
#[derive(Clone, Debug)]
pub struct EnsembleHopResult {
    /// Best design-space point.
    pub best_pos: Array1<f64>,
    /// Objective at [`EnsembleHopResult::best_pos`].
    pub best_val: f64,
    /// Charged hop-ledger calls (evals plus grads).
    pub charged: usize,
    /// Distinct exact identities in the shared or largest private history.
    ///
    /// Zero on the one-replica values-only portfolio, which has no hop
    /// history. Two or more values-only replicas share this table when
    /// a finite-difference certificate is flat.
    pub history_minima: usize,
    /// History attempts, refusals and summed operation seconds across replicas.
    /// Zero when the selected search does not use history.
    pub history_cost: (usize, usize, f64),
}

/// Search on `obj`: hop and quench when `grad` is present.
///
/// Without a gradient, one replica is the values-only portfolio. Two or
/// more replicas are communicating values-only hop chains: they divide
/// the budget, kick, pattern-search, and share [`MinimumHistory`] when
/// a finite-difference certificate is flat. `replicas` is not ignored.
pub fn ensemble_hop_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
    budget: usize,
    replicas: usize,
    history: HistoryMode,
    membership: HistoryMembership,
) -> EnsembleHopResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let budget = budget.max(1);
    if let Some(grad) = grad {
        let config = BoxEnsembleConfig {
            replicas: replicas.max(1),
            budget,
            history,
            membership,
            ..BoxEnsembleConfig::default()
        };
        let out = box_ensemble_optimize(obj, grad, seed, x0, &config);
        return EnsembleHopResult::from(out);
    }
    if replicas <= 1 {
        let out =
            portfolio_optimize_seeded(obj, grad, budget, seed, None, PortfolioPolicy::Auto, x0);
        return EnsembleHopResult {
            best_pos: Array1::from(out.best_pos),
            best_val: out.best_val,
            charged: out.n_evals + out.n_grads,
            history_minima: 0,
            history_cost: (0, 0, 0.0),
        };
    }
    let config = BoxEnsembleConfig {
        replicas,
        budget,
        history,
        membership,
        ..BoxEnsembleConfig::default()
    };
    EnsembleHopResult::from(box_values_ensemble_optimize(obj, seed, x0, &config))
}

impl From<BoxEnsembleResult> for EnsembleHopResult {
    fn from(out: BoxEnsembleResult) -> Self {
        Self {
            best_pos: out.best_pos,
            best_val: out.best_val,
            charged: out.n_evals + out.n_grads,
            history_minima: out.history_minima,
            history_cost: out.history_cost,
        }
    }
}

/// Communicating values-only hop chains. Same history object as the
/// gradient ensemble. A point is admitted only when a one-sided
/// finite-difference certificate is flatter than `1e-3`.
pub fn box_values_ensemble_optimize<O>(
    obj: &O,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
    config: &BoxEnsembleConfig,
) -> BoxEnsembleResult
where
    O: Objective<f64>,
{
    assert!(
        matches!(config.escape, BoxEscape::Gaussian),
        "Langevin escape requires a gradient"
    );
    let observed = ObservedObjective {
        inner: obj,
        incumbent: Mutex::new(None),
    };
    let obj = &observed;
    let bounds = obj.bounds().clone();
    let dim = bounds.dims.max(1);
    let widths = &bounds.high - &bounds.low;
    let mean_width = widths.iter().copied().sum::<f64>() / dim as f64;
    let replica_count = config.replicas.max(1);
    let budgets = config.budgets();
    let gradient_tolerance = 1e-3;
    let nng_url = shared_nng_url(config);
    #[cfg(feature = "history-nng")]
    let _nng_server = bind_shared_nng(config, nng_url.as_deref());
    let histories: Vec<Mutex<MinimumHistory>> = if nng_url.is_some() {
        Vec::new()
    } else {
        match config.history {
            HistoryMode::None => Vec::new(),
            HistoryMode::Private => (0..replica_count)
                .map(|_| {
                    Mutex::new(
                        MinimumHistory::new(gradient_tolerance).expect("finite gradient tolerance"),
                    )
                })
                .collect(),
            HistoryMode::Shared => vec![Mutex::new(
                MinimumHistory::new(gradient_tolerance).expect("finite gradient tolerance"),
            )],
        }
    };
    let witness = WidthWitness {
        widths: widths.clone(),
        identity_tol: config.identity_tol,
    };
    let context = StructureContext::new(
        None,
        DescriptorGeometry::finite(mean_width.max(1e-6)).ok(),
        Some("design-box".into()),
    );
    let merge = (config.identity_tol * mean_width.max(1e-12)).max(1e-12);
    let mut biases: Vec<BasinBias<RawCoordinates>> = (0..replica_count)
        .map(|_| BasinBias::new(RawCoordinates, merge, 0.1, 5.0))
        .collect();
    let mut history_seen: Vec<std::collections::HashMap<usize, u64>> = (0..replica_count)
        .map(|_| std::collections::HashMap::new())
        .collect();
    let mut hooks: Vec<ReplicaHook<'_>> = (0..replica_count)
        .map(|index| {
            replica_hook(
                config,
                nng_url.as_deref(),
                &histories,
                &witness,
                &context,
                &widths,
                index,
            )
        })
        .collect();

    let mut replicas: Vec<Replica> = (0..replica_count)
        .map(|index| {
            let mut rng = StdRng::seed_from_u64(seed ^ (index as u64).wrapping_mul(0x9E37_79B9));
            let start = if index == 0 {
                if let Some(x0) = x0 {
                    bounds.clip(x0)
                } else {
                    bounds.clip(((&bounds.low + &bounds.high) * 0.5).view())
                }
            } else {
                let mut draw = Array1::zeros(dim);
                for j in 0..dim {
                    draw[j] = bounds.low[j] + widths[j] * rng.random::<f64>();
                }
                bounds.clip(draw.view())
            };
            Replica {
                trial: start.clone(),
                cv: start.clone(),
                rng,
                x: start,
                f: f64::INFINITY,
                work: 0,
                budget: budgets[index],
                hops: 0,
                here: None,
                feedback: EscapeFeedback::new(1.0, 0.1),
                generation: 0,
            }
        })
        .collect();

    let mut n_evals = 0usize;
    let n_grads = 0usize;
    let mut history_observations = 0usize;
    let mut shared_deposits = 0usize;

    for (index, replica) in replicas.iter_mut().enumerate() {
        if replica.budget == 0 {
            continue;
        }
        let start_depth = values_search_depth(dim, replica.budget);
        let polish = pattern_search_polish(obj, replica.x.clone(), start_depth.max(1));
        replica.work += polish.n_evals;
        n_evals += polish.n_evals;
        if polish.best_val.is_finite() {
            replica.x = polish.best_pos;
            replica.f = polish.best_val;
            replica.cv = replica.x.clone();
        }
        let before = replica.work;
        let certified =
            values_certificate(obj, replica.x.view(), &mut replica.work, replica.budget);
        n_evals += replica.work - before;
        if let Some(gradient) = certified {
            if let Some(report) = hooks[index].observe(replica.f, replica.x.view(), gradient.view())
            {
                history_observations += 1;
                hooks[index].mark_accepted(report.minimum);
                replica.feedback.register_initial(report.minimum);
                replica.here = Some(report.minimum);
            }
        }
    }

    loop {
        let mut progressed = false;
        for (index, replica) in replicas.iter_mut().enumerate() {
            let remaining = replica.budget.saturating_sub(replica.work);
            let depth = values_search_depth(dim, remaining);
            if remaining < 4 || depth == 0 {
                continue;
            }
            progressed = true;
            replica.generation += 1;
            replica.hops += 1;
            let escape = replica.feedback.escape();
            replica.trial.assign(&replica.x);
            for j in 0..dim {
                let noise: f64 = StandardNormal.sample(&mut replica.rng);
                replica.trial[j] += STEP0 * escape * widths[j] * noise;
            }
            replica.trial = reflect_into_box(replica.trial.view(), &bounds);
            let polish = pattern_search_polish(obj, replica.trial.clone(), depth);
            replica.work += polish.n_evals;
            n_evals += polish.n_evals;
            let mut report = None;
            if polish.best_val.is_finite() {
                let before = replica.work;
                let certified = values_certificate(
                    obj,
                    polish.best_pos.view(),
                    &mut replica.work,
                    replica.budget,
                );
                n_evals += replica.work - before;
                if let Some(gradient) = certified {
                    report = hooks[index].observe(
                        polish.best_val,
                        polish.best_pos.view(),
                        gradient.view(),
                    );
                }
            }
            let trial_x = polish.best_pos;
            let trial_f = polish.best_val;
            if let Some(report) = report {
                history_observations += 1;
                replica.feedback.observe_shared(
                    replica.here,
                    report.minimum,
                    report.is_new,
                    report.visits.max(1),
                );
                if config.shared_deposits > 0 {
                    let seen = history_seen[index].entry(report.minimum).or_insert(0);
                    let foreign = report
                        .visits
                        .saturating_sub(*seen)
                        .saturating_sub(1)
                        .min(config.shared_deposits as u64);
                    let cv = biases[index].cv(trial_x.view());
                    for _ in 0..foreign {
                        biases[index].deposit(cv.view(), temp_of(replica.generation, replica.f));
                        shared_deposits += 1;
                    }
                    *seen = report.visits;
                }
            }
            if !trial_f.is_finite() {
                continue;
            }
            let temp = temp_of(replica.generation, replica.f);
            let trial_cv = biases[index].cv(trial_x.view());
            let v_here = biases[index].potential(replica.cv.view());
            let v_trial = biases[index].potential(trial_cv.view());
            let delta = (trial_f + v_trial) - (replica.f + v_here);
            let accept =
                delta <= 0.0 || replica.rng.random::<f64>() < (-delta / temp.max(1e-300)).exp();
            replica.adopt_trial(accept, trial_x, trial_f, trial_cv, report);
            if accept {
                biases[index].deposit(replica.cv.view(), temp);
                if let Some(report) = report {
                    hooks[index].mark_accepted(report.minimum);
                    if report.visits == 0 {
                        history_seen[index].insert(report.minimum, 1);
                    }
                }
            }
        }
        if !progressed {
            break;
        }
    }

    let mut best_pos = replicas[0].x.clone();
    let mut best_val = replicas[0].f;
    let mut hops = 0usize;
    for replica in &replicas {
        hops += replica.hops;
        if replica.f.is_finite() && replica.f < best_val {
            best_val = replica.f;
            best_pos = replica.x.clone();
        }
    }
    if let Some((position, value)) = observed.incumbent.into_inner().expect("incumbent lock") {
        best_pos = position;
        best_val = value;
    }
    let history_minima = nng_minimum_count(hooks.first()).unwrap_or_else(|| {
        histories
            .iter()
            .map(|history| history.lock().map(|h| h.minimum_count()).unwrap_or(0))
            .max()
            .unwrap_or(0)
    });

    BoxEnsembleResult {
        best_pos,
        best_val,
        n_evals,
        n_grads,
        hops,
        history_observations,
        history_minima,
        history_cost: total_history_cost(&hooks),
        shared_deposits,
    }
}

struct ValuesPolish {
    best_pos: Array1<f64>,
    best_val: f64,
    n_evals: usize,
}

fn pattern_search_polish<O: Objective<f64>>(
    obj: &O,
    mut x: Array1<f64>,
    max_evals: usize,
) -> ValuesPolish {
    let bounds = obj.bounds();
    let mut f = obj.eval(x.view());
    let mut n = 1usize;
    if !f.is_finite() || max_evals <= 1 {
        return ValuesPolish {
            best_pos: x,
            best_val: f,
            n_evals: n,
        };
    }
    let widths = &bounds.high - &bounds.low;
    let mut step = 0.1;
    while n + x.len() < max_evals && step > 1e-8 {
        let mut improved = false;
        for i in 0..x.len() {
            for sgn in [-1.0, 1.0] {
                if n >= max_evals {
                    return ValuesPolish {
                        best_pos: x,
                        best_val: f,
                        n_evals: n,
                    };
                }
                let mut trial = x.clone();
                trial[i] += sgn * step * widths[i].max(1e-12);
                trial = bounds.clip(trial.view());
                let ft = obj.eval(trial.view());
                n += 1;
                if ft.is_finite() && ft < f {
                    x = trial;
                    f = ft;
                    improved = true;
                }
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    ValuesPolish {
        best_pos: x,
        best_val: f,
        n_evals: n,
    }
}

fn values_search_depth(dim: usize, remaining: usize) -> usize {
    let certificate = dim + 1;
    let room = remaining.saturating_sub(certificate + 2);
    if room < dim + 2 {
        return 0;
    }
    room.min((4 * dim + 16).max(MIN_QUENCH))
}

fn values_certificate<O: Objective<f64>>(
    obj: &O,
    x: ArrayView1<f64>,
    work: &mut usize,
    budget: usize,
) -> Option<Array1<f64>> {
    let dim = x.len();
    if *work + dim + 1 > budget {
        return None;
    }
    let f0 = obj.eval(x);
    *work += 1;
    if !f0.is_finite() {
        return None;
    }
    let bounds = obj.bounds();
    let mut grad = Array1::zeros(dim);
    for i in 0..dim {
        if bounds.low[i] == bounds.high[i] && x[i] == bounds.low[i] {
            continue;
        }
        if *work >= budget {
            return None;
        }
        // A certificate needs a distinct feasible point, including when an
        // absolute step rounds away at the coordinate's floating-point scale.
        let step = 1e-6_f64.max(f64::EPSILON * x[i].abs());
        let forward = (x[i] + step).min(bounds.high[i]);
        let backward = (x[i] - step).max(bounds.low[i]);
        let probe = if forward.is_finite() && forward > x[i] {
            forward
        } else if backward.is_finite() && backward < x[i] {
            backward
        } else {
            return None;
        };
        let mut bumped = x.to_owned();
        bumped[i] = probe;
        let fi = obj.eval(bumped.view());
        *work += 1;
        if !fi.is_finite() {
            return None;
        }
        grad[i] = (fi - f0) / (probe - x[i]);
        if !grad[i].is_finite() {
            return None;
        }
    }
    Some(projected_gradient(
        &x.to_owned(),
        &grad,
        &bounds.low,
        &bounds.high,
    ))
}

fn certificate_gradient<O: Objective<f64>, G: Gradient<f64>>(
    paid: Option<Array1<f64>>,
    pos: ArrayView1<f64>,
    obj: &O,
    grad: &G,
    work: &mut usize,
    budget: usize,
    n_grads: &mut usize,
) -> Option<Array1<f64>> {
    let g = if let Some(g) = paid.filter(|g| g.len() == pos.len()) {
        g
    } else {
        if *work >= budget {
            return None;
        }
        *work += 1;
        *n_grads += 1;
        grad.grad(pos)
    };
    if g.len() != pos.len() || g.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let bounds = obj.bounds();
    Some(projected_gradient(
        &pos.to_owned(),
        &g,
        &bounds.low,
        &bounds.high,
    ))
}

/// Scaled max-norm witness on a box. Not IRA; not SOAP.
struct WidthWitness {
    widths: Array1<f64>,
    identity_tol: f64,
}

impl ExactStructureWitness for WidthWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        if left.len() != right.len() || left.len() != self.widths.len() {
            return false;
        }
        left.iter()
            .zip(right.iter())
            .zip(self.widths.iter())
            .all(|((&a, &b), &w)| (a - b).abs() <= self.identity_tol * w.max(1e-12))
    }
}

/// Design coordinates as the bias fingerprint. A box is not a point set.
struct RawCoordinates;

impl Fingerprint for RawCoordinates {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

/// Run `replicas` box hops that divide `config.budget` and talk through
/// one [`HistoryHook`] per replica.
pub fn box_ensemble_optimize<O, G>(
    obj: &O,
    grad: &G,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
    config: &BoxEnsembleConfig,
) -> BoxEnsembleResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    if let BoxEscape::Langevin(escape) = config.escape {
        escape.validate();
    }
    let observed = ObservedObjective {
        inner: obj,
        incumbent: Mutex::new(None),
    };
    let obj = &observed;
    let bounds = obj.bounds().clone();
    let dim = bounds.dims.max(1);
    let widths = &bounds.high - &bounds.low;
    let mean_width = widths.iter().copied().sum::<f64>() / dim as f64;
    let replica_count = config.replicas.max(1);
    let budgets = config.budgets();
    let gradient_tolerance = 1e-3;
    let nng_url = shared_nng_url(config);
    #[cfg(feature = "history-nng")]
    let _nng_server = bind_shared_nng(config, nng_url.as_deref());
    // Mutex table or nng client, never both.
    let histories: Vec<Mutex<MinimumHistory>> = if nng_url.is_some() {
        Vec::new()
    } else {
        match config.history {
            HistoryMode::None => Vec::new(),
            HistoryMode::Private => (0..replica_count)
                .map(|_| {
                    Mutex::new(
                        MinimumHistory::new(gradient_tolerance).expect("finite gradient tolerance"),
                    )
                })
                .collect(),
            HistoryMode::Shared => vec![Mutex::new(
                MinimumHistory::new(gradient_tolerance).expect("finite gradient tolerance"),
            )],
        }
    };
    let witness = WidthWitness {
        widths: widths.clone(),
        identity_tol: config.identity_tol,
    };
    let context = StructureContext::new(
        None,
        DescriptorGeometry::finite(mean_width.max(1e-6)).ok(),
        Some("design-box".into()),
    );
    let merge = (config.identity_tol * mean_width.max(1e-12)).max(1e-12);
    let mut biases: Vec<BasinBias<RawCoordinates>> = (0..replica_count)
        .map(|_| BasinBias::new(RawCoordinates, merge, 0.1, 5.0))
        .collect();
    let mut history_seen: Vec<std::collections::HashMap<usize, u64>> = (0..replica_count)
        .map(|_| std::collections::HashMap::new())
        .collect();
    let mut hooks: Vec<ReplicaHook<'_>> = (0..replica_count)
        .map(|index| {
            replica_hook(
                config,
                nng_url.as_deref(),
                &histories,
                &witness,
                &context,
                &widths,
                index,
            )
        })
        .collect();

    let mut replicas: Vec<Replica> = (0..replica_count)
        .map(|index| {
            let mut rng = StdRng::seed_from_u64(seed ^ (index as u64).wrapping_mul(0x9E37_79B9));
            let start = if index == 0 {
                if let Some(x0) = x0 {
                    bounds.clip(x0)
                } else {
                    bounds.clip(((&bounds.low + &bounds.high) * 0.5).view())
                }
            } else {
                let mut draw = Array1::zeros(dim);
                for j in 0..dim {
                    draw[j] = bounds.low[j] + widths[j] * rng.random::<f64>();
                }
                bounds.clip(draw.view())
            };
            Replica {
                trial: start.clone(),
                cv: start.clone(),
                rng,
                x: start,
                f: f64::INFINITY,
                work: 0,
                budget: budgets[index],
                hops: 0,
                here: None,
                feedback: EscapeFeedback::new(1.0, 0.1),
                generation: 0,
            }
        })
        .collect();

    let mut n_evals = 0usize;
    let mut n_grads = 0usize;
    let mut history_observations = 0usize;
    let mut shared_deposits = 0usize;

    for (index, replica) in replicas.iter_mut().enumerate() {
        if replica.budget == 0 {
            continue;
        }
        let start_depth = quench_depth(dim, replica.budget);
        let mut start_grad = None;
        if start_depth > 0 {
            let quench =
                projected_gradient_polish(obj, grad, replica.x.clone(), start_depth, 1.0, 1e-8);
            replica.work += quench.n_evals + quench.n_grads;
            n_evals += quench.n_evals;
            n_grads += quench.n_grads;
            if quench.best_val.is_finite() {
                replica.x = quench.best_pos;
                replica.f = quench.best_val;
                replica.cv = replica.x.clone();
            }
            start_grad = quench.best_grad;
        } else {
            replica.f = obj.eval(replica.x.view());
            replica.work += 1;
            n_evals += 1;
        }
        if let Some(gradient) = certificate_gradient(
            start_grad,
            replica.x.view(),
            obj,
            grad,
            &mut replica.work,
            replica.budget,
            &mut n_grads,
        ) {
            if let Some(report) = hooks[index].observe(replica.f, replica.x.view(), gradient.view())
            {
                history_observations += 1;
                hooks[index].mark_accepted(report.minimum);
                replica.feedback.register_initial(report.minimum);
                replica.here = Some(report.minimum);
            }
        }
    }

    let mut noise_states: Vec<Option<LangevinStepper>> = (0..replica_count).map(|_| None).collect();
    loop {
        let mut progressed = false;
        for (index, replica) in replicas.iter_mut().enumerate() {
            let remaining = replica.budget.saturating_sub(replica.work);
            let mut depth = quench_depth(dim, remaining);
            if remaining < 4 || depth == 0 {
                continue;
            }
            let escape_steps = match config.escape {
                BoxEscape::Gaussian => 0,
                BoxEscape::Langevin(escape) => {
                    // Reserve four combined work units for a bounded quench,
                    // then one launch gradient and two callbacks per step.
                    let steps = escape.steps.min(remaining.saturating_sub(5) / 2);
                    if steps == 0 {
                        continue;
                    }
                    steps
                }
            };
            progressed = true;
            replica.generation += 1;
            replica.hops += 1;
            let escape = replica.feedback.escape();
            replica.trial.assign(&replica.x);
            match config.escape {
                BoxEscape::Gaussian => {
                    for j in 0..dim {
                        let noise: f64 = StandardNormal.sample(&mut replica.rng);
                        replica.trial[j] += STEP0 * escape * widths[j] * noise;
                    }
                    replica.trial = reflect_into_box(replica.trial.view(), &bounds);
                }
                BoxEscape::Langevin(settings) => {
                    let temperature = temp_of(replica.generation, replica.f) * escape * escape;
                    let stepper = noise_states[index].get_or_insert_with(|| {
                        LangevinStepper::new(
                            settings.noise,
                            settings.omega0,
                            settings.dt,
                            temperature,
                            Array1::ones(dim),
                            seed ^ (index as u64).wrapping_mul(0x9E37_79B9),
                        )
                    });
                    stepper.set_temperature(temperature);
                    // Quench and acceptance can change the anchor. A fresh raw
                    // force avoids confusing projected stationarity with dynamics.
                    let mut force = grad.grad(replica.trial.view());
                    replica.work += 1;
                    n_grads += 1;
                    for _ in 0..escape_steps {
                        stepper.step(obj, grad, &mut replica.trial, &mut force);
                        replica.work += 2;
                        n_evals += 1;
                        n_grads += 1;
                    }
                    depth = quench_depth(dim, replica.budget.saturating_sub(replica.work));
                }
            }
            let polish =
                projected_gradient_polish(obj, grad, replica.trial.clone(), depth, 1.0, 1e-8);
            let used_evals = polish.n_evals;
            let mut used_grads = polish.n_grads;
            replica.work += used_evals + used_grads;
            let mut report = None;
            if polish.best_val.is_finite() {
                if let Some(gradient) = certificate_gradient(
                    polish.best_grad,
                    polish.best_pos.view(),
                    obj,
                    grad,
                    &mut replica.work,
                    replica.budget,
                    &mut used_grads,
                ) {
                    report = hooks[index].observe(
                        polish.best_val,
                        polish.best_pos.view(),
                        gradient.view(),
                    );
                }
            }
            let trial_x = polish.best_pos;
            let trial_f = polish.best_val;
            n_evals += used_evals;
            n_grads += used_grads;
            if let Some(report) = report {
                history_observations += 1;
                replica.feedback.observe_shared(
                    replica.here,
                    report.minimum,
                    report.is_new,
                    report.visits.max(1),
                );
                if config.shared_deposits > 0 {
                    let seen = history_seen[index].entry(report.minimum).or_insert(0);
                    let foreign = report
                        .visits
                        .saturating_sub(*seen)
                        .saturating_sub(1)
                        .min(config.shared_deposits as u64);
                    let cv = biases[index].cv(trial_x.view());
                    for _ in 0..foreign {
                        biases[index].deposit(cv.view(), temp_of(replica.generation, replica.f));
                        shared_deposits += 1;
                    }
                    *seen = report.visits;
                }
            }
            if !trial_f.is_finite() {
                continue;
            }
            let temp = temp_of(replica.generation, replica.f);
            let trial_cv = biases[index].cv(trial_x.view());
            let v_here = biases[index].potential(replica.cv.view());
            let v_trial = biases[index].potential(trial_cv.view());
            let delta = (trial_f + v_trial) - (replica.f + v_here);
            let accept =
                delta <= 0.0 || replica.rng.random::<f64>() < (-delta / temp.max(1e-300)).exp();
            replica.adopt_trial(accept, trial_x, trial_f, trial_cv, report);
            if accept {
                biases[index].deposit(replica.cv.view(), temp);
                if let Some(report) = report {
                    hooks[index].mark_accepted(report.minimum);
                    if report.visits == 0 {
                        history_seen[index].insert(report.minimum, 1);
                    }
                }
            }
        }
        if !progressed {
            break;
        }
    }

    let mut best_pos = replicas[0].x.clone();
    let mut best_val = replicas[0].f;
    let mut hops = 0usize;
    for replica in &replicas {
        hops += replica.hops;
        if replica.f.is_finite() && replica.f < best_val {
            best_val = replica.f;
            best_pos = replica.x.clone();
        }
    }
    if let Some((position, value)) = observed.incumbent.into_inner().expect("incumbent lock") {
        best_pos = position;
        best_val = value;
    }
    let history_minima = nng_minimum_count(hooks.first()).unwrap_or_else(|| {
        histories
            .iter()
            .map(|history| history.lock().map(|h| h.minimum_count()).unwrap_or(0))
            .max()
            .unwrap_or(0)
    });

    BoxEnsembleResult {
        best_pos,
        best_val,
        n_evals,
        n_grads,
        hops,
        history_observations,
        history_minima,
        history_cost: total_history_cost(&hooks),
        shared_deposits,
    }
}

/// Passive best-point observation is independent of occupied-state acceptance,
/// history certification and work charging. Every callback reaches the wrapped
/// objective exactly once at its original coordinates.
struct ObservedObjective<'a, O> {
    inner: &'a O,
    incumbent: Mutex<Option<(Array1<f64>, f64)>>,
}

impl<O: Objective<f64>> Objective<f64> for ObservedObjective<'_, O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let value = self.inner.eval(x);
        let bounds = self.bounds();
        if value.is_finite()
            && x.len() == bounds.dims
            && x.iter().zip(bounds.low.iter().zip(bounds.high.iter())).all(
                |(coordinate, (low, high))| {
                    coordinate.is_finite() && coordinate >= low && coordinate <= high
                },
            )
        {
            let mut incumbent = self.incumbent.lock().expect("incumbent lock");
            if incumbent.as_ref().is_none_or(|(_, best)| value < *best) {
                *incumbent = Some((x.to_owned(), value));
            }
        }
        value
    }
}

struct Replica {
    rng: StdRng,
    x: Array1<f64>,
    trial: Array1<f64>,
    cv: Array1<f64>,
    f: f64,
    work: usize,
    budget: usize,
    hops: usize,
    here: Option<usize>,
    feedback: EscapeFeedback,
    generation: usize,
}

impl Replica {
    fn adopt_trial(
        &mut self,
        accepted: bool,
        position: Array1<f64>,
        energy: f64,
        descriptor: Array1<f64>,
        report: Option<HistoryReport>,
    ) {
        if !accepted {
            return;
        }
        self.x = position;
        self.f = energy;
        self.cv = descriptor;
        self.here = report.map(|report| report.minimum);
    }
}

fn temp_of(generation: usize, energy: f64) -> f64 {
    let scale = (1.0 + energy.abs()).max(1e-6);
    scale * 5.0 * std::f64::consts::LN_2 / (generation as f64 + 1.0).ln().max(1e-12)
}

fn quench_depth(dim: usize, remaining: usize) -> usize {
    // `projected_gradient_polish` charges one eval and about one grad per
    // outer step, then one trailing grad. Leave a unit for the history
    // certificate so the hop cannot spend past the replica budget.
    let fevals = remaining.saturating_sub(2) / 2;
    if fevals == 0 {
        return 0;
    }
    let target = (2 * dim + 8).max(MIN_QUENCH);
    target.min(fevals)
}

/// One hook value per replica. Mutex table or nng client, never both.
enum ReplicaHook<'a> {
    Off,
    Mutex(SharedDesignHistory<'a, WidthWitness>),
    #[cfg(feature = "history-nng")]
    Nng(HistoryNngClient),
}

impl HistoryHook for ReplicaHook<'_> {
    fn observe(
        &mut self,
        energy: f64,
        state: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Option<HistoryReport> {
        match self {
            Self::Off => None,
            Self::Mutex(hook) => hook.observe(energy, state, gradient),
            #[cfg(feature = "history-nng")]
            Self::Nng(hook) => hook.observe(energy, state, gradient),
        }
    }

    fn mark_accepted(&mut self, minimum: usize) {
        match self {
            Self::Off => {}
            Self::Mutex(hook) => hook.mark_accepted(minimum),
            #[cfg(feature = "history-nng")]
            Self::Nng(hook) => hook.mark_accepted(minimum),
        }
    }

    fn cost(&self) -> (usize, usize, f64) {
        match self {
            Self::Off => (0, 0, 0.0),
            Self::Mutex(hook) => hook.cost(),
            #[cfg(feature = "history-nng")]
            Self::Nng(hook) => hook.cost(),
        }
    }
}

fn total_history_cost(hooks: &[ReplicaHook<'_>]) -> (usize, usize, f64) {
    hooks.iter().fold((0, 0, 0.0), |total, hook| {
        let cost = hook.cost();
        (total.0 + cost.0, total.1 + cost.1, total.2 + cost.2)
    })
}

fn shared_nng_url(config: &BoxEnsembleConfig) -> Option<String> {
    if !matches!(config.history, HistoryMode::Shared) {
        return None;
    }
    let url = match std::env::var("HISTORY_NNG") {
        Ok(url) if !url.is_empty() => url,
        Ok(_) | Err(std::env::VarError::NotPresent) => return None,
        Err(error) => panic!("HISTORY_NNG configuration: {error}"),
    };
    #[cfg(feature = "history-nng")]
    {
        Some(url)
    }
    #[cfg(not(feature = "history-nng"))]
    panic!("HISTORY_NNG={url:?} requires the history-nng feature")
}

#[cfg(feature = "history-nng")]
fn bind_shared_nng(config: &BoxEnsembleConfig, url: Option<&str>) -> Option<HistoryNngServer> {
    let url = url?;
    std::env::var("HISTORY_NNG_SERVE")
        .is_ok_and(|value| value == "1")
        .then(|| {
            HistoryNngServer::bind(url, config.identity_tol, 1e-3)
                .unwrap_or_else(|error| panic!("HISTORY_NNG server setup: {error}"))
        })
}

fn replica_hook<'a>(
    config: &BoxEnsembleConfig,
    nng_url: Option<&str>,
    histories: &'a [Mutex<MinimumHistory>],
    witness: &'a WidthWitness,
    context: &StructureContext,
    widths: &Array1<f64>,
    replica: usize,
) -> ReplicaHook<'a> {
    if let Some(url) = nng_url {
        #[cfg(feature = "history-nng")]
        {
            return HistoryNngClient::dial(url, widths.clone(), config.membership)
                .map(ReplicaHook::Nng)
                .unwrap_or_else(|error| panic!("HISTORY_NNG client setup: {error}"));
        }
        #[cfg(not(feature = "history-nng"))]
        {
            let _ = (url, widths);
        }
    }
    let history = match config.history {
        HistoryMode::None => return ReplicaHook::Off,
        HistoryMode::Private => match histories.get(replica) {
            Some(history) => history,
            None => return ReplicaHook::Off,
        },
        HistoryMode::Shared => match histories.first() {
            Some(history) => history,
            None => return ReplicaHook::Off,
        },
    };
    ReplicaHook::Mutex(SharedDesignHistory::new(
        history,
        context.clone(),
        witness,
        config.membership,
    ))
}

fn nng_minimum_count(hook: Option<&ReplicaHook<'_>>) -> Option<usize> {
    #[cfg(feature = "history-nng")]
    if let Some(ReplicaHook::Nng(client)) = hook {
        return client.minimum_count();
    }
    let _ = hook;
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::Bounds;
    use ndarray::{Array1, ArrayView1, array};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn occupied_replica() -> Replica {
        let occupied = array![2.0, 2.0];
        let mut feedback = EscapeFeedback::new(1.0, 0.1);
        feedback.register_initial(7);
        Replica {
            rng: StdRng::seed_from_u64(19),
            x: occupied.clone(),
            trial: occupied.clone(),
            cv: occupied,
            f: -1.0,
            work: 13,
            budget: 100,
            hops: 2,
            here: Some(7),
            feedback,
            generation: 2,
        }
    }

    #[test]
    fn accepted_uncertified_trial_clears_the_occupied_history_identity() {
        let mut replica = occupied_replica();
        let destination = array![-2.0, -2.0];
        replica.adopt_trial(true, destination.clone(), -1.5, destination.clone(), None);

        assert_eq!(replica.x, destination);
        assert_eq!(replica.cv, destination);
        assert_eq!(replica.f, -1.5);
        assert_eq!(replica.here, None);
        assert_eq!((replica.work, replica.hops, replica.generation), (13, 2, 2));
    }

    #[test]
    fn return_after_uncertified_adoption_uses_population_known_feedback() {
        let mut replica = occupied_replica();
        replica.adopt_trial(true, array![-2.0, -2.0], -1.5, array![-2.0, -2.0], None);
        let before = replica.feedback.escape();
        let visit = replica.feedback.observe_shared(replica.here, 7, false, 8);

        assert_eq!(visit, crate::methods::minima_hopping::Visit::Known);
        let expected = 1.05 * (1.0 + 0.1 * 7.0_f64.ln());
        assert!((replica.feedback.escape() / before - expected).abs() < 1e-12);
    }

    #[test]
    fn accepted_certified_trial_keeps_its_own_history_identity() {
        let mut replica = occupied_replica();
        let destination = array![-2.0, -2.0];
        let report = HistoryReport {
            minimum: 9,
            is_new: true,
            visits: 0,
            observed_visits: 1,
            first_observation: true,
        };
        replica.adopt_trial(
            true,
            destination.clone(),
            -1.5,
            destination.clone(),
            Some(report),
        );

        assert_eq!(replica.x, destination);
        assert_eq!(replica.cv, destination);
        assert_eq!(replica.f, -1.5);
        assert_eq!(replica.here, Some(9));
        assert_eq!(replica.feedback.escape(), 1.0);
        assert_eq!((replica.work, replica.hops, replica.generation), (13, 2, 2));
    }

    #[test]
    fn rejected_trials_preserve_the_occupied_state_and_history_identity() {
        let report = HistoryReport {
            minimum: 9,
            is_new: true,
            visits: 0,
            observed_visits: 1,
            first_observation: true,
        };
        for certification in [None, Some(report)] {
            let mut replica = occupied_replica();
            replica.adopt_trial(
                false,
                array![-2.0, -2.0],
                -1.5,
                array![-2.0, -2.0],
                certification,
            );

            assert_eq!(replica.x, array![2.0, 2.0]);
            assert_eq!(replica.cv, array![2.0, 2.0]);
            assert_eq!(replica.f, -1.0);
            assert_eq!(replica.here, Some(7));
            assert_eq!(replica.feedback.escape(), 1.0);
            assert_eq!((replica.work, replica.hops, replica.generation), (13, 2, 2));
        }
    }

    struct BoundaryLinear {
        bounds: Bounds<f64>,
        slope: f64,
        evals: AtomicUsize,
        grads: AtomicUsize,
    }

    impl BoundaryLinear {
        fn new(slope: f64) -> Self {
            Self {
                bounds: Bounds::new(array![0.0], array![1.0], 1e-9),
                slope,
                evals: AtomicUsize::new(0),
                grads: AtomicUsize::new(0),
            }
        }
    }

    impl Objective<f64> for BoundaryLinear {
        fn dim(&self) -> usize {
            1
        }

        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.evals.fetch_add(1, Ordering::SeqCst);
            assert_eq!(x.len(), 1);
            assert!(self.bounds.contains(x));
            self.slope * x[0]
        }
    }

    impl Gradient<f64> for BoundaryLinear {
        fn dim(&self) -> usize {
            1
        }

        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.grads.fetch_add(1, Ordering::SeqCst);
            assert_eq!(x.len(), 1);
            assert!(self.bounds.contains(x));
            array![self.slope]
        }
    }

    #[test]
    fn values_certificate_distinguishes_an_upper_bound_minimum_from_inward_descent() {
        for (slope, expected_projected_gradient) in [(-1.0, 0.0), (1.0, 1.0)] {
            let objective = BoundaryLinear::new(slope);
            let position = array![1.0];
            let mut work = 0;
            let certificate = values_certificate(&objective, position.view(), &mut work, 2)
                .expect("two objective calls certify one box coordinate");

            assert_eq!(work, 2);
            assert_eq!(objective.evals.load(Ordering::SeqCst), work);
            assert_eq!(objective.grads.load(Ordering::SeqCst), 0);
            assert_eq!(certificate.len(), 1);
            assert!(
                (certificate[0] - expected_projected_gradient).abs() < 1e-8,
                "f(x)={slope}x at x=1 requires projected gradient \
                 {expected_projected_gradient}, received {}",
                certificate[0]
            );
        }
    }

    #[test]
    fn analytic_boundary_minimum_enters_history_when_projected_polish_is_stationary() {
        let position = array![1.0];
        let polish_objective = BoundaryLinear::new(-1.0);
        let polished = projected_gradient_polish(
            &polish_objective,
            &polish_objective,
            position.clone(),
            1,
            1.0,
            1e-8,
        );
        assert!(polished.projected_stationary);
        assert_eq!(polished.projected_grad_norm, 0.0);
        assert_eq!(polished.best_grad, Some(array![-1.0]));
        assert_eq!(polished.best_pos, position);
        assert_eq!(polished.best_val, -1.0);
        assert_eq!(polished.n_evals, 1);
        assert_eq!(polished.n_grads, 1);
        assert_eq!(polish_objective.evals.load(Ordering::SeqCst), 1);
        assert_eq!(polish_objective.grads.load(Ordering::SeqCst), 1);

        let objective = BoundaryLinear::new(-1.0);
        let config = BoxEnsembleConfig {
            replicas: 1,
            budget: 4,
            history: HistoryMode::Private,
            ..BoxEnsembleConfig::default()
        };
        let result =
            box_ensemble_optimize(&objective, &objective, 7, Some(position.view()), &config);
        assert_eq!(result.best_pos, position);
        assert_eq!(result.best_val, -1.0);
        assert_eq!(result.n_evals, 1);
        assert_eq!(result.n_grads, 1);
        assert_eq!(objective.evals.load(Ordering::SeqCst), result.n_evals);
        assert_eq!(objective.grads.load(Ordering::SeqCst), result.n_grads);
        assert_eq!(result.hops, 0);
        assert_eq!(
            result.history_observations, 1,
            "a box-KKT minimum must not be rejected for its outward raw gradient"
        );
        assert_eq!(result.history_minima, 1);
    }

    struct TwoWell {
        bounds: Bounds<f64>,
        evals: AtomicUsize,
        grads: AtomicUsize,
    }

    impl TwoWell {
        fn new() -> Self {
            Self {
                bounds: Bounds::new(array![-5.0, -5.0], array![5.0, 5.0], 1e-9),
                evals: AtomicUsize::new(0),
                grads: AtomicUsize::new(0),
            }
        }

        fn wells() -> (Array1<f64>, Array1<f64>) {
            (array![2.0, 2.0], array![-2.0, -2.0])
        }
    }

    impl Objective<f64> for TwoWell {
        fn dim(&self) -> usize {
            2
        }

        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.evals.fetch_add(1, Ordering::Relaxed);
            let (a, b) = Self::wells();
            let da = &x - &a;
            let db = &x - &b;
            let ea = da.dot(&da);
            let eb = db.dot(&db) - 0.5;
            ea.min(eb)
        }
    }

    impl Gradient<f64> for TwoWell {
        fn dim(&self) -> usize {
            2
        }

        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.grads.fetch_add(1, Ordering::Relaxed);
            let (a, b) = Self::wells();
            let da = &x - &a;
            let db = &x - &b;
            let ea = da.dot(&da);
            let eb = db.dot(&db) - 0.5;
            if ea < eb { 2.0 * da } else { 2.0 * db }
        }
    }

    fn design_hooks(
        _history: &Mutex<MinimumHistory>,
        widths: Array1<f64>,
    ) -> (WidthWitness, StructureContext) {
        let witness = WidthWitness {
            widths,
            identity_tol: IDENTITY_TOL,
        };
        let context = StructureContext::new(None, None, Some("design-box".into()));
        (witness, context)
    }

    #[test]
    fn two_hooks_over_one_minimum_history_see_each_other() {
        let history = Mutex::new(MinimumHistory::new(1e-3).unwrap());
        let (witness, context) = design_hooks(&history, array![10.0, 10.0]);
        let mut first = SharedDesignHistory::new(
            &history,
            context.clone(),
            &witness,
            HistoryMembership::Accepted,
        );
        let mut second =
            SharedDesignHistory::new(&history, context, &witness, HistoryMembership::Accepted);
        let zero = array![0.0, 0.0];
        let a = first
            .observe(-1.0, array![2.0, 2.0].view(), zero.view())
            .unwrap();
        first.mark_accepted(a.minimum);
        let b = second
            .observe(-0.5, array![2.001, 2.0].view(), zero.view())
            .unwrap();
        assert_eq!(a.minimum, b.minimum);
        assert!(!b.is_new);
        assert_eq!(b.observed_visits, 2);
    }

    #[test]
    fn private_histories_do_not_share_identities() {
        let left = Mutex::new(MinimumHistory::new(1e-3).unwrap());
        let right = Mutex::new(MinimumHistory::new(1e-3).unwrap());
        let (witness, context) = design_hooks(&left, array![10.0, 10.0]);
        let mut first = SharedDesignHistory::new(
            &left,
            context.clone(),
            &witness,
            HistoryMembership::Accepted,
        );
        let mut second =
            SharedDesignHistory::new(&right, context, &witness, HistoryMembership::Accepted);
        let zero = array![0.0, 0.0];
        first
            .observe(-1.0, array![2.0, 2.0].view(), zero.view())
            .unwrap();
        let b = second
            .observe(-1.0, array![2.0, 2.0].view(), zero.view())
            .unwrap();
        assert!(b.first_observation);
        assert_eq!(b.observed_visits, 1);
    }

    #[test]
    fn ensemble_respects_the_budget_and_the_box() {
        let obj = TwoWell::new();
        let config = BoxEnsembleConfig {
            replicas: 4,
            budget: 200,
            history: HistoryMode::Shared,
            membership: HistoryMembership::Accepted,
            identity_tol: IDENTITY_TOL,
            shared_deposits: 8,
        };
        let out = box_ensemble_optimize(&obj, &obj, 7, None, &config);
        assert!(out.n_evals + out.n_grads <= 200);
        assert!(obj.evals.load(Ordering::Relaxed) + obj.grads.load(Ordering::Relaxed) <= 200);
        assert!(out.best_val.is_finite());
        for (value, (&lo, &hi)) in out
            .best_pos
            .iter()
            .zip(obj.bounds().low.iter().zip(obj.bounds().high.iter()))
        {
            assert!(*value >= lo - 1e-8 && *value <= hi + 1e-8);
        }
        assert!(out.hops > 0);
        assert!(out.history_minima >= 1);
    }

    #[test]
    fn shared_history_records_both_wells() {
        let obj = TwoWell::new();
        let start = array![2.0, 2.0];
        let config = BoxEnsembleConfig {
            replicas: 4,
            budget: 400,
            history: HistoryMode::Shared,
            membership: HistoryMembership::Accepted,
            identity_tol: IDENTITY_TOL,
            shared_deposits: 8,
        };
        let out = box_ensemble_optimize(&obj, &obj, 11, Some(start.view()), &config);
        assert!(out.best_val.is_finite());
        assert!(
            out.history_minima >= 1,
            "shared history stayed empty: minima={}",
            out.history_minima
        );
        assert!(out.history_observations >= out.history_minima);
    }

    struct Sphere {
        bounds: Bounds<f64>,
    }

    impl Sphere {
        fn new() -> Self {
            Self {
                bounds: Bounds::new(array![-2.0, -2.0, -2.0], array![2.0, 2.0, 2.0], 1e-9),
            }
        }
    }

    impl Objective<f64> for Sphere {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.dot(&x)
        }
    }

    impl Gradient<f64> for Sphere {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            2.0 * &x
        }
    }

    #[test]
    fn production_ensemble_runs_on_a_three_coordinate_box() {
        let obj = Sphere::new();
        let start = array![1.0, 1.0, 1.0];
        let out = ensemble_hop_optimize(
            &obj,
            Some(&obj),
            3,
            Some(start.view()),
            400,
            2,
            HistoryMode::Shared,
            HistoryMembership::Accepted,
        );
        assert!(out.best_val.is_finite());
        assert!(out.best_val <= 3.0);
        assert!(out.charged > 0);
        assert!(out.best_pos.iter().all(|v| v.abs() <= 2.0 + 1e-8));
    }

    struct FiveSphere {
        bounds: Bounds<f64>,
    }

    impl FiveSphere {
        fn new() -> Self {
            Self {
                bounds: Bounds::new(Array1::from_elem(5, -2.0), Array1::from_elem(5, 2.0), 1e-9),
            }
        }
    }

    impl Objective<f64> for FiveSphere {
        fn dim(&self) -> usize {
            5
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.dot(&x)
        }
    }

    impl Gradient<f64> for FiveSphere {
        fn dim(&self) -> usize {
            5
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            2.0 * &x
        }
    }

    #[test]
    fn a_five_coordinate_box_stays_five_coordinates() {
        let obj = FiveSphere::new();
        let start = Array1::from_elem(5, 1.0);
        let out = ensemble_hop_optimize(
            &obj,
            Some(&obj),
            3,
            Some(start.view()),
            64,
            2,
            HistoryMode::Shared,
            HistoryMembership::Accepted,
        );
        assert_eq!(out.best_pos.len(), 5);
        assert!(out.best_pos.iter().all(|v| v.abs() <= 2.0 + 1e-8));
    }

    #[test]
    fn values_only_one_replica_uses_the_portfolio() {
        let obj = Sphere::new();
        let start = array![1.0, 1.0, 1.0];
        let out = ensemble_hop_optimize::<_, Sphere>(
            &obj,
            None,
            3,
            Some(start.view()),
            64,
            1,
            HistoryMode::Shared,
            HistoryMembership::Accepted,
        );
        assert!(out.best_val.is_finite());
        assert!(out.charged > 0);
        assert!(out.charged <= 64);
        assert_eq!(out.history_minima, 0);
        assert_eq!(out.best_pos.len(), 3);
    }

    #[test]
    fn values_only_replicas_share_a_history() {
        let obj = Sphere::new();
        let start = array![1.0, 1.0, 1.0];
        let out = ensemble_hop_optimize::<_, Sphere>(
            &obj,
            None,
            3,
            Some(start.view()),
            400,
            2,
            HistoryMode::Shared,
            HistoryMembership::Accepted,
        );
        assert!(out.best_val.is_finite());
        assert!(out.charged > 0);
        assert!(out.charged <= 400);
        assert_eq!(out.best_pos.len(), 3);
        assert!(
            out.history_minima >= 1,
            "values-only replicas left history empty: minima={} best={}",
            out.history_minima,
            out.best_val
        );
        assert!(
            out.best_val < 3.0,
            "values-only replicas did not descend: best={}",
            out.best_val
        );
    }

    struct CountingSphere {
        bounds: Bounds<f64>,
        calls: AtomicUsize,
    }

    impl CountingSphere {
        fn new() -> Self {
            Self {
                bounds: Bounds::new(Array1::from_elem(6, -2.0), Array1::from_elem(6, 2.0), 1e-9),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Objective<f64> for CountingSphere {
        fn dim(&self) -> usize {
            6
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.calls.fetch_add(1, Ordering::SeqCst);
            x.dot(&x)
        }
    }

    impl Gradient<f64> for CountingSphere {
        fn dim(&self) -> usize {
            6
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            2.0 * &x
        }
    }

    #[test]
    fn every_box_callback_charges_the_ledger() {
        let obj = CountingSphere::new();
        let start = Array1::from_elem(6, 1.0);
        let out = ensemble_hop_optimize(
            &obj,
            Some(&obj),
            3,
            Some(start.view()),
            32,
            2,
            HistoryMode::Shared,
            HistoryMembership::Accepted,
        );
        let n = obj.calls.load(Ordering::SeqCst);
        assert!(n > 0);
        assert!(out.charged > 0);
        assert!(out.charged <= 32);
        assert!(
            n <= out.charged,
            "uncharged callback: n={n} charged={}",
            out.charged
        );
    }
}
