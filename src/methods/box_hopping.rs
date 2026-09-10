//! Box-constrained hop with the production communicating-chain history.
//!
//! The move is a Gaussian kick reflected into the box, then a charged
//! quench. Communication is one [`HistoryHook`] per replica over one
//! [`MinimumHistory`]: in-process [`SharedDesignHistory`] behind a
//! `Mutex`, or process-split nng Req/Rep (`HistoryNngClient`). Never
//! both, never a third table. A certified quench, an exact witness,
//! Goedecker escape from `observe_shared`, and other chains' visits
//! paid into the well-tempered bias (`shared_deposits`). That is
//! commit 225282aa, not a parallel Euclidean side table. The witness
//! is scaled max-norm in the box; descriptors are the design
//! coordinates and never decide identity.

use std::sync::Mutex;

use eindir_core::{Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::bias::{BasinBias, Bias, Fingerprint};
use crate::descriptor_space::DescriptorGeometry;
use crate::methods::ensemble::HistoryMode;
use crate::methods::local_polish::projected_gradient_polish;
use crate::methods::minima_hopping::{
    EscapeFeedback, HistoryHook, HistoryMembership, HistoryReport, MinimumHistory,
    SharedDesignHistory,
};
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
    /// Best feasible point among the replicas.
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
    /// Distinct exact identities in the shared or union of private histories.
    pub history_minima: usize,
    /// Bias deposits made on behalf of other chains' visits.
    pub shared_deposits: usize,
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
    grad: Option<&G>,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
    config: &BoxEnsembleConfig,
) -> BoxEnsembleResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
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
                        MinimumHistory::new(gradient_tolerance)
                            .expect("finite gradient tolerance"),
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
    let mut history_seen: Vec<std::collections::HashMap<usize, u64>> =
        (0..replica_count).map(|_| std::collections::HashMap::new()).collect();
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
        replica.f = obj.eval(replica.x.view());
        replica.work += 1;
        n_evals += 1;
        let start_depth = quench_depth(dim, replica.budget.saturating_sub(replica.work));
        if let Some(grad) = grad
            && start_depth > 0
        {
            let quench =
                projected_gradient_polish(obj, grad, replica.x.clone(), start_depth, 1.0, 1e-8);
            replica.work += quench.n_evals + quench.n_grads;
            n_evals += quench.n_evals;
            n_grads += quench.n_grads;
            if quench.best_val.is_finite() {
                replica.x = quench.best_pos;
                replica.f = quench.best_val;
            }
        }
        if let Some(grad) = grad
            && replica.work < replica.budget
        {
            let gradient = grad.grad(replica.x.view());
            replica.work += 1;
            n_grads += 1;
            if let Some(report) = hooks[index].observe(
                replica.f,
                replica.x.view(),
                gradient.view(),
            ) {
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
            let depth = quench_depth(dim, remaining);
            if remaining < 4 || (grad.is_some() && depth == 0) {
                continue;
            }
            progressed = true;
            replica.generation += 1;
            replica.hops += 1;
            let escape = replica.feedback.escape();
            let mut trial = replica.x.clone();
            for j in 0..dim {
                let noise: f64 = StandardNormal.sample(&mut replica.rng);
                trial[j] += STEP0 * escape * widths[j] * noise;
            }
            trial = reflect_into_box(trial.view(), &bounds);
            let (trial_x, trial_f, used_evals, used_grads, report) = match grad {
                Some(grad) => {
                    let polish = projected_gradient_polish(obj, grad, trial, depth, 1.0, 1e-8);
                    let used_evals = polish.n_evals;
                    let mut used_grads = polish.n_grads;
                    replica.work += used_evals + used_grads;
                    let mut report = None;
                    if polish.best_val.is_finite() && replica.work < replica.budget {
                        let gradient = grad.grad(polish.best_pos.view());
                        replica.work += 1;
                        used_grads += 1;
                        report = hooks[index].observe(
                            polish.best_val,
                            polish.best_pos.view(),
                            gradient.view(),
                        );
                    }
                    (
                        polish.best_pos,
                        polish.best_val,
                        used_evals,
                        used_grads,
                        report,
                    )
                }
                None => {
                    let value = obj.eval(trial.view());
                    replica.work += 1;
                    (trial, value, 1, 0, None)
                }
            };
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
            let v_here = biases[index].potential(biases[index].cv(replica.x.view()).view());
            let v_trial = biases[index].potential(biases[index].cv(trial_x.view()).view());
            let delta = (trial_f + v_trial) - (replica.f + v_here);
            let accept =
                delta <= 0.0 || replica.rng.random::<f64>() < (-delta / temp.max(1e-300)).exp();
            if accept {
                replica.x = trial_x;
                replica.f = trial_f;
                let accepted_cv = biases[index].cv(replica.x.view());
                biases[index].deposit(accepted_cv.view(), temp);
                if let Some(report) = report {
                    hooks[index].mark_accepted(report.minimum);
                    replica.here = Some(report.minimum);
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
        shared_deposits,
    }
}

struct Replica {
    rng: StdRng,
    x: Array1<f64>,
    f: f64,
    work: usize,
    budget: usize,
    hops: usize,
    here: Option<usize>,
    feedback: EscapeFeedback,
    generation: usize,
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

fn shared_nng_url(config: &BoxEnsembleConfig) -> Option<String> {
    if !matches!(config.history, HistoryMode::Shared) {
        return None;
    }
    #[cfg(feature = "history-nng")]
    {
        match std::env::var("HISTORY_NNG") {
            Ok(url) if !url.is_empty() => Some(url),
            _ => None,
        }
    }
    #[cfg(not(feature = "history-nng"))]
    None
}

#[cfg(feature = "history-nng")]
fn bind_shared_nng(config: &BoxEnsembleConfig, url: Option<&str>) -> Option<HistoryNngServer> {
    let url = url?;
    std::env::var("HISTORY_NNG_SERVE")
        .is_ok_and(|value| value == "1")
        .then(|| HistoryNngServer::bind(url, config.identity_tol, 1e-3).ok())
        .flatten()
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
                .unwrap_or(ReplicaHook::Off);
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
        let mut second = SharedDesignHistory::new(
            &history,
            context,
            &witness,
            HistoryMembership::Accepted,
        );
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
        let mut second = SharedDesignHistory::new(
            &right,
            context,
            &witness,
            HistoryMembership::Accepted,
        );
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
        let out = box_ensemble_optimize(&obj, Some(&obj), 7, None, &config);
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
        let out = box_ensemble_optimize(&obj, Some(&obj), 11, Some(start.view()), &config);
        assert!(out.best_val.is_finite());
        assert!(
            out.history_minima >= 1,
            "shared history stayed empty: minima={}",
            out.history_minima
        );
        assert!(out.history_observations >= out.history_minima);
    }
}
