//! Box-constrained hop with a Euclidean minimum history.
//!
//! Cluster hopping is 3N Cartesian and the wrong move on a CUTEst box.
//! This loop is the algebraic analogue: a Gaussian kick reflected into the
//! box, a charged quench, and Goedecker escape feedback driven by a
//! [`HistoryHook`]. Identity is scaled max-norm in the box, not SOAP.
//! Four replicas that share one history are the communicating arm.

use std::sync::Mutex;
use std::time::Instant;

use eindir_core::{Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::methods::ensemble::HistoryMode;
use crate::methods::local_polish::projected_gradient_polish;
use crate::methods::minima_hopping::{
    EscapeFeedback, HistoryHook, HistoryMembership, HistoryReport,
};
use crate::movekernel::reflect_into_box;

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
}

impl Default for BoxEnsembleConfig {
    fn default() -> Self {
        Self {
            replicas: 4,
            budget: 8_000,
            history: HistoryMode::Shared,
            membership: HistoryMembership::Accepted,
            identity_tol: IDENTITY_TOL,
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
    /// Distinct Euclidean identities in the shared or union of private histories.
    pub history_minima: usize,
}

/// Euclidean minimum table used as a [`HistoryHook`].
#[derive(Debug)]
pub struct BoxMinimumHistory {
    minima: Vec<BoxMinimum>,
    widths: Array1<f64>,
    identity_tol: f64,
}

#[derive(Debug, Clone)]
struct BoxMinimum {
    state: Array1<f64>,
    observed: u64,
    accepted: u64,
}

impl BoxMinimumHistory {
    /// Empty history for a box of the given side lengths.
    pub fn new(widths: Array1<f64>, identity_tol: f64) -> Self {
        Self {
            minima: Vec::new(),
            widths,
            identity_tol,
        }
    }

    /// Number of distinct quenched identities.
    pub fn minimum_count(&self) -> usize {
        self.minima.len()
    }

    fn same(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        if left.len() != right.len() || left.len() != self.widths.len() {
            return false;
        }
        left.iter()
            .zip(right.iter())
            .zip(self.widths.iter())
            .all(|((&a, &b), &w)| (a - b).abs() <= self.identity_tol * w.max(1e-12))
    }

    fn observe(
        &mut self,
        state: ArrayView1<f64>,
        policy: HistoryMembership,
    ) -> Option<HistoryReport> {
        if state.len() != self.widths.len() || state.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let existing = self
            .minima
            .iter()
            .position(|minimum| self.same(minimum.state.view(), state));
        let (id, first_observation) = match existing {
            Some(id) => (id, false),
            None => {
                self.minima.push(BoxMinimum {
                    state: state.to_owned(),
                    observed: 0,
                    accepted: 0,
                });
                (self.minima.len() - 1, true)
            }
        };
        let visits = self.minima[id].observed.checked_add(1)?;
        self.minima[id].observed = visits;
        let accepted = self.minima[id].accepted;
        let (is_new, policy_visits) = crate::methods::minima_hopping::history_feedback_membership(
            policy,
            first_observation,
            visits,
            accepted,
        );
        Some(HistoryReport {
            minimum: id,
            is_new,
            visits: policy_visits,
            observed_visits: visits,
            first_observation,
        })
    }

    fn mark_accepted(&mut self, minimum: usize) {
        if let Some(entry) = self.minima.get_mut(minimum) {
            entry.accepted = entry.accepted.max(1);
        }
    }
}

/// [`HistoryHook`] over a [`BoxMinimumHistory`] behind a lock.
pub struct BoxHistoryHook<'a> {
    history: &'a Mutex<BoxMinimumHistory>,
    policy: HistoryMembership,
    observations: usize,
    refusals: usize,
    seconds: f64,
}

impl<'a> BoxHistoryHook<'a> {
    /// Hook over `history` reporting under `policy`.
    pub fn new(history: &'a Mutex<BoxMinimumHistory>, policy: HistoryMembership) -> Self {
        Self {
            history,
            policy,
            observations: 0,
            refusals: 0,
            seconds: 0.0,
        }
    }
}

impl HistoryHook for BoxHistoryHook<'_> {
    fn observe(
        &mut self,
        _energy: f64,
        state: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Option<HistoryReport> {
        let started = Instant::now();
        let report = if gradient.iter().any(|v| !v.is_finite()) {
            None
        } else {
            self.history
                .lock()
                .ok()
                .and_then(|mut history| history.observe(state, self.policy))
        };
        self.seconds += started.elapsed().as_secs_f64();
        if report.is_some() {
            self.observations += 1;
        } else {
            self.refusals += 1;
        }
        report
    }

    fn mark_accepted(&mut self, minimum: usize) {
        let started = Instant::now();
        if let Ok(mut history) = self.history.lock() {
            history.mark_accepted(minimum);
        }
        self.seconds += started.elapsed().as_secs_f64();
    }

    fn cost(&self) -> (usize, usize, f64) {
        (self.observations, self.refusals, self.seconds)
    }
}

/// Run `replicas` box hops that divide `config.budget` and talk only through history.
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
    let replica_count = config.replicas.max(1);
    let budgets = config.budgets();
    let histories: Vec<Mutex<BoxMinimumHistory>> = match config.history {
        HistoryMode::None => Vec::new(),
        HistoryMode::Private => (0..replica_count)
            .map(|_| Mutex::new(BoxMinimumHistory::new(widths.clone(), config.identity_tol)))
            .collect(),
        HistoryMode::Shared => vec![Mutex::new(BoxMinimumHistory::new(
            widths.clone(),
            config.identity_tol,
        ))],
    };

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
            if let Some(report) = observe_replica(
                config,
                &histories,
                index,
                replica.f,
                replica.x.view(),
                gradient.view(),
            ) {
                history_observations += 1;
                mark_replica(config, &histories, index, report.minimum);
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
                        report = observe_replica(
                            config,
                            &histories,
                            index,
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
            }
            if !trial_f.is_finite() {
                continue;
            }
            let scale = (1.0 + replica.f.abs()).max(1e-6);
            let temp = scale * 5.0 * std::f64::consts::LN_2
                / (replica.generation as f64 + 1.0).ln().max(1e-12);
            let delta = trial_f - replica.f;
            let accept =
                delta <= 0.0 || replica.rng.random::<f64>() < (-delta / temp.max(1e-300)).exp();
            if accept {
                replica.x = trial_x;
                replica.f = trial_f;
                if let Some(report) = report {
                    mark_replica(config, &histories, index, report.minimum);
                    replica.here = Some(report.minimum);
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
    let history_minima = histories
        .iter()
        .map(|history| history.lock().map(|h| h.minimum_count()).unwrap_or(0))
        .max()
        .unwrap_or(0);

    BoxEnsembleResult {
        best_pos,
        best_val,
        n_evals,
        n_grads,
        hops,
        history_observations,
        history_minima,
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

fn observe_replica(
    config: &BoxEnsembleConfig,
    histories: &[Mutex<BoxMinimumHistory>],
    replica: usize,
    energy: f64,
    state: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
) -> Option<HistoryReport> {
    let history = match config.history {
        HistoryMode::None => return None,
        HistoryMode::Private => histories.get(replica)?,
        HistoryMode::Shared => histories.first()?,
    };
    let mut hook = BoxHistoryHook::new(history, config.membership);
    hook.observe(energy, state, gradient)
}

fn mark_replica(
    config: &BoxEnsembleConfig,
    histories: &[Mutex<BoxMinimumHistory>],
    replica: usize,
    minimum: usize,
) {
    let history = match config.history {
        HistoryMode::None => return,
        HistoryMode::Private => match histories.get(replica) {
            Some(history) => history,
            None => return,
        },
        HistoryMode::Shared => match histories.first() {
            Some(history) => history,
            None => return,
        },
    };
    let mut hook = BoxHistoryHook::new(history, config.membership);
    hook.mark_accepted(minimum);
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

    #[test]
    fn two_hooks_over_one_box_history_see_each_other() {
        let widths = array![10.0, 10.0];
        let history = Mutex::new(BoxMinimumHistory::new(widths, IDENTITY_TOL));
        let mut first = BoxHistoryHook::new(&history, HistoryMembership::Accepted);
        let mut second = BoxHistoryHook::new(&history, HistoryMembership::Accepted);
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
        let widths = array![10.0, 10.0];
        let left = Mutex::new(BoxMinimumHistory::new(widths.clone(), IDENTITY_TOL));
        let right = Mutex::new(BoxMinimumHistory::new(widths, IDENTITY_TOL));
        let mut first = BoxHistoryHook::new(&left, HistoryMembership::Accepted);
        let mut second = BoxHistoryHook::new(&right, HistoryMembership::Accepted);
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
