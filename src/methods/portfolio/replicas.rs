//! Batch replicas of the values-only portfolio.
//!
//! [`portfolio_replicas`] runs several instances of the values-only loop
//! together, each on its own thread and its own share of the budget. A
//! replica's evaluations go to one coordinator, which waits until every
//! replica still running has asked for a point, evaluates those points as
//! one batch through [`Objective::eval_batch`], records them and answers.
//! Only the coordinator writes an archive, and only while every replica
//! waits for it, so between two batches each replica reads a fixed archive
//! and a run depends on its seed alone, not on how the threads interleave.
//!
//! With [`ReplicaArchive::Shared`] the replicas keep one archive and so one
//! incumbent: arms that start from the incumbent start from the best point
//! any replica has found, and the additive surrogate fits every replica's
//! evaluations. A point another replica has already evaluated is answered
//! from the archive, once per replica, with no call and no charge, and
//! replicas that ask for one point in the same batch share its row, paid
//! for by the one with the most of its share left. Replicas whose descents
//! meet thus pay for each point once between them, and none spends its
//! share on a common descent before the others.
//! With [`ReplicaArchive::Private`] each replica sees only its own
//! evaluations, as independent runs batched together would.
//! Either way a row is paid for as above, but a point any replica has
//! already evaluated is answered from that evaluation instead of a call,
//! so the objective is called once at a point.
//!
//! The finite-difference descent is deterministic, so replicas that descend
//! from one shared incumbent ask for the same points. Replica `r` of `R`
//! therefore differences each gradient from coordinate `r * n / R` on: the
//! gradient is the same, but at each step the replicas ask for different
//! stencil points, so a gradient in `n` dimensions takes about `n / R`
//! batches, and with a shared archive the budget pays for it once.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};

use super::{
    Archive, ArmStat, ArmStates, BudgetLedger, BudgetedObjective, OBJECTIVE_WORK_UNIT,
    PortfolioResult, run_values_only_portfolio,
};

/// Whether batch replicas keep one archive between them or one each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplicaArchive {
    /// One archive and one incumbent for every replica.
    #[default]
    Shared,
    /// An archive per replica, each holding the start and that replica's
    /// own evaluations.
    Private,
}

/// Result of [`portfolio_replicas`].
#[derive(Clone, Debug)]
pub struct ReplicaResult {
    /// The best point any replica evaluated, its value, the calls of the
    /// objective (the start's once) and the replicas' arm statistics
    /// summed arm by arm.
    pub portfolio: PortfolioResult,
    /// Calls to [`Objective::eval_batch`], the start's included.
    pub batches: usize,
}

/// Mixes the replica index into the run's seed; replica 0 keeps the seed.
const REPLICA_STREAM: u64 = 0xD6E8_FEB8_6659_FD93;

fn replica_seed(seed: u64, replica: usize) -> u64 {
    seed ^ (replica as u64).wrapping_mul(REPLICA_STREAM)
}

/// Evaluations of `budget` left to replica `replica` once the start is
/// paid: the rest split as evenly as it goes, the first replicas taking one
/// more each.
fn replica_share(budget: usize, replicas: usize, replica: usize) -> usize {
    let rest = budget - 1;
    rest / replicas + usize::from(replica < rest % replicas)
}

impl BudgetLedger {
    /// Gives back the unit the budgeted objective charged for an
    /// evaluation that the coordinator answered from the archive.
    fn refund_eval(&self) {
        self.used.fetch_sub(OBJECTIVE_WORK_UNIT, Ordering::Relaxed);
        self.n_evals.fetch_sub(1, Ordering::Relaxed);
    }
}

/// What a replica thread tells the coordinator.
enum Message {
    /// Replica `.0` asks for the value at `.1`.
    Ask(usize, Array1<f64>),
    /// A replica has stopped, normally or by panicking.
    Done,
}

/// The coordinator's answer to an [`Message::Ask`].
struct Answer {
    value: f64,
    /// Whether the asking replica pays for the value; otherwise it came
    /// from the archive or another replica in the same batch pays for it.
    paid: bool,
}

/// Tells the coordinator a replica has stopped, on every way out of its
/// thread, so it never waits for a replica that will not ask again.
struct DoneGuard {
    asks: mpsc::Sender<Message>,
}

impl Drop for DoneGuard {
    fn drop(&mut self) {
        let _ = self.asks.send(Message::Done);
    }
}

/// A replica's view of the objective: each evaluation is a request to the
/// coordinator, answered once the batch it joins has been evaluated.
struct ReplicaObjective<'a> {
    replica: usize,
    bounds: &'a Bounds<f64>,
    ledger: &'a BudgetLedger,
    asks: mpsc::Sender<Message>,
    answers: Mutex<mpsc::Receiver<Answer>>,
}

impl Objective<f64> for ReplicaObjective<'_> {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if self
            .asks
            .send(Message::Ask(self.replica, x.to_owned()))
            .is_err()
        {
            return f64::INFINITY;
        }
        let answer = self.answers.lock().expect("answer lock").recv();
        match answer {
            Ok(Answer { value, paid }) => {
                if !paid {
                    self.ledger.refund_eval();
                }
                value
            }
            Err(_) => f64::INFINITY,
        }
    }
}

/// The values-only loop's gradient type; it is never called.
struct NoGradient;

impl Gradient<f64> for NoGradient {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Array1::zeros(x.len())
    }

    fn dim(&self) -> usize {
        0
    }
}

/// Runs one replica's values-only loop on its share of the budget and
/// returns its arm statistics, the evaluations charged to it, and how many
/// of those were at a point with a non-finite coordinate, which asked the
/// coordinator nothing.
fn run_replica(
    replica: usize,
    bounds: &Bounds<f64>,
    archive: Arc<Archive>,
    share: usize,
    seed: u64,
    first: usize,
    asks: mpsc::Sender<Message>,
    answers: mpsc::Receiver<Answer>,
) -> (Vec<ArmStat>, usize, usize) {
    let _done = DoneGuard { asks: asks.clone() };
    // Each replica runs as a solo run of `share + 1` evaluations whose
    // first, the start, every replica paid for together.
    let ledger = BudgetLedger::with_archive(share + 1, bounds.dims, archive, false);
    ledger.try_charge(OBJECTIVE_WORK_UNIT);
    let objective = ReplicaObjective {
        replica,
        bounds,
        ledger: &ledger,
        asks,
        answers: Mutex::new(answers),
    };
    let budgeted = BudgetedObjective {
        inner: &objective,
        ledger: &ledger,
    };
    let mut states = ArmStates {
        qn_first: first,
        ..ArmStates::default()
    };
    let stats = run_values_only_portfolio::<_, NoGradient>(
        &budgeted,
        &ledger,
        &mut states,
        seed,
        share + 1,
    );
    (
        stats,
        ledger.n_evals.load(Ordering::Relaxed),
        ledger.n_unseen.load(Ordering::Relaxed),
    )
}

/// The points a shared archive has answered, by bit pattern, with the
/// replicas that have had each one.
struct Seen {
    value: f64,
    by: Vec<u64>,
}

impl Seen {
    fn new(value: f64, replicas: usize) -> Self {
        Self {
            value,
            by: vec![0; replicas.div_ceil(64)],
        }
    }

    fn has(&self, replica: usize) -> bool {
        self.by[replica / 64] & (1 << (replica % 64)) != 0
    }

    fn add(&mut self, replica: usize) {
        self.by[replica / 64] |= 1 << (replica % 64);
    }
}

fn point_key(x: ArrayView1<f64>) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Batches the replicas' requests until every replica has stopped and
/// returns the number of batches evaluated and of rows answered from
/// `called`, the value of every point evaluated so far by bit pattern.
#[allow(clippy::too_many_arguments)]
fn coordinate<O: Objective<f64>>(
    obj: &O,
    archives: &[Arc<Archive>],
    shared: bool,
    asks: &mpsc::Receiver<Message>,
    answers: &[mpsc::Sender<Answer>],
    shares: &[usize],
    seen: &mut HashMap<Vec<u64>, Seen>,
    called: &mut HashMap<Vec<u64>, f64>,
) -> (usize, usize) {
    let replicas = answers.len();
    let bounds = obj.bounds();
    let running = shares.iter().filter(|&&share| share > 0).count();
    // Evaluations each replica has left to pay for.
    let mut left = shares.to_vec();
    let mut pending: Vec<Option<Array1<f64>>> = vec![None; replicas];
    let mut awaiting = running;
    let mut batches = 0usize;
    let mut recalled = 0usize;
    loop {
        while awaiting > 0 {
            match asks.recv() {
                Ok(Message::Ask(replica, x)) => pending[replica] = Some(x),
                Ok(Message::Done) => {}
                Err(_) => break,
            }
            awaiting -= 1;
        }
        if pending.iter().all(Option::is_none) {
            return (batches, recalled);
        }
        // A shared archive answers what it holds, once to each replica;
        // those replicas ask again before the batch is evaluated.
        if shared {
            for (replica, slot) in pending.iter_mut().enumerate() {
                let Some(x) = slot.as_ref() else { continue };
                if let Some(entry) = seen.get_mut(&point_key(x.view()))
                    && !entry.has(replica)
                {
                    entry.add(replica);
                    let _ = answers[replica].send(Answer {
                        value: entry.value,
                        paid: false,
                    });
                    *slot = None;
                    awaiting += 1;
                }
            }
            if awaiting > 0 {
                continue;
            }
        }
        // One row per distinct point. With a shared archive, replicas that
        // ask for the same point share one row, and the one with the most
        // of its share left pays for it, so that a descent they follow
        // together does not spend one replica's share before the others'.
        let mut payers: Vec<usize> = Vec::new();
        let mut keys: Vec<Vec<u64>> = Vec::new();
        let mut row_of = vec![usize::MAX; replicas];
        for (replica, slot) in pending.iter().enumerate() {
            let Some(x) = slot.as_ref() else { continue };
            let key = point_key(x.view());
            let row = if shared {
                keys.iter().position(|k| *k == key)
            } else {
                None
            };
            row_of[replica] = match row {
                Some(row) => {
                    if left[replica] > left[payers[row]] {
                        payers[row] = replica;
                    }
                    row
                }
                None => {
                    payers.push(replica);
                    keys.push(key);
                    payers.len() - 1
                }
            };
        }
        // A row is paid for as it is, but a point evaluated before, or in
        // another row of this batch, is answered from that evaluation; only
        // the others go to the objective.
        let mut values = vec![f64::INFINITY; keys.len()];
        let mut fresh: Vec<usize> = Vec::new();
        let mut source = vec![None; keys.len()];
        for (row, key) in keys.iter().enumerate() {
            if let Some(&value) = called.get(key) {
                values[row] = value;
            } else if let Some(first) = fresh.iter().position(|&other| keys[other] == *key) {
                source[row] = Some(first);
            } else {
                source[row] = Some(fresh.len());
                fresh.push(row);
            }
        }
        recalled += keys.len() - fresh.len();
        if !fresh.is_empty() {
            let mut batch = Array2::<f64>::zeros((fresh.len(), bounds.dims));
            for (index, &row) in fresh.iter().enumerate() {
                batch
                    .row_mut(index)
                    .assign(pending[payers[row]].as_ref().expect("payer asked"));
            }
            let evaluated = obj.eval_batch(batch.view());
            batches += 1;
            for (row, index) in source.iter().enumerate() {
                if let Some(index) = *index {
                    values[row] = evaluated.get(index).copied().unwrap_or(f64::INFINITY);
                }
            }
            for &row in &fresh {
                called.insert(keys[row].clone(), values[row]);
            }
        }
        let value_at = |row: usize| values[row];
        for (row, &replica) in payers.iter().enumerate() {
            let archive = if shared {
                &archives[0]
            } else {
                &archives[replica]
            };
            let x = pending[replica].as_ref().expect("payer asked");
            archive.record(x.view(), value_at(row), bounds);
            left[replica] -= 1;
        }
        for (replica, slot) in pending.iter_mut().enumerate() {
            if slot.take().is_none() {
                continue;
            }
            let row = row_of[replica];
            if shared {
                seen.entry(keys[row].clone())
                    .or_insert_with(|| Seen::new(value_at(row), replicas))
                    .add(replica);
            }
            let _ = answers[replica].send(Answer {
                value: value_at(row),
                paid: payers[row] == replica,
            });
            awaiting += 1;
        }
    }
}

/// Runs `replicas` instances of the values-only portfolio from `x0` as one
/// batched search under one total `budget`.
///
/// `x0` is evaluated first, once, as a batch of one, and every replica
/// starts from it. The rest of the budget is split between the replicas as
/// evenly as it goes; replica `r` runs the values-only loop as a solo run
/// of its share plus the start would ([`super::portfolio_optimize_from`]
/// with `x0`), seeded with `seed` mixed with `r` and with its descent's
/// gradients differenced from coordinate `r * n / replicas` on (replica 0
/// keeps `seed` and coordinate 0), so one replica repeats that run exactly.
/// At each step every replica still running asks for one point and those
/// not evaluated before go to [`Objective::eval_batch`] together, the
/// others answered from their evaluations; `archive` decides
/// whether the replicas share what they find (see the module notes). Every
/// point evaluated lies in the objective's bounds, and at most `budget`
/// evaluations are made. The result holds the best point any replica
/// evaluated.
///
/// # Panics
///
/// If `budget` or `replicas` is zero, if the bounds are not finite with
/// ``low < high``, or if `x0` has the wrong length or lies outside them.
pub fn portfolio_replicas<O>(
    obj: &O,
    budget: usize,
    seed: u64,
    x0: ArrayView1<f64>,
    replicas: usize,
    archive: ReplicaArchive,
) -> ReplicaResult
where
    O: Objective<f64>,
{
    assert!(budget > 0, "budget must be positive");
    assert!(replicas > 0, "replicas must be positive");
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    assert!(dim > 0, "objective dimension must be positive");
    for i in 0..dim {
        assert!(
            bounds.low[i].is_finite() && bounds.high[i].is_finite(),
            "bounds must be finite at dimension {i}"
        );
        assert!(
            bounds.low[i] < bounds.high[i],
            "low[{i}] must be strictly less than high[{i}]"
        );
    }
    assert_eq!(x0.len(), dim, "x0 must have the objective's dimension");
    assert!(bounds.contains(x0), "x0 must lie inside the bounds");

    let shared = archive == ReplicaArchive::Shared;
    let archives: Vec<Arc<Archive>> = if shared {
        vec![Arc::new(Archive::new(budget))]
    } else {
        (0..replicas)
            .map(|r| Arc::new(Archive::new(replica_share(budget, replicas, r) + 1)))
            .collect()
    };
    let start = x0.to_owned();
    let start_value = obj.eval_batch(start.view().insert_axis(ndarray::Axis(0)))[0];
    for archive in &archives {
        archive.record(start.view(), start_value, &bounds);
    }
    let mut seen: HashMap<Vec<u64>, Seen> = HashMap::new();
    if shared {
        let mut everyone = Seen::new(start_value, replicas);
        (0..replicas).for_each(|r| everyone.add(r));
        seen.insert(point_key(start.view()), everyone);
    }
    let mut called = HashMap::from([(point_key(start.view()), start_value)]);

    let shares: Vec<usize> = (0..replicas)
        .map(|r| replica_share(budget, replicas, r))
        .collect();
    let running = shares.iter().filter(|&&share| share > 0).count();
    let (ask_tx, ask_rx) = mpsc::channel::<Message>();
    let mut answer_txs = Vec::with_capacity(replicas);
    let mut answer_rxs = Vec::with_capacity(replicas);
    for _ in 0..replicas {
        let (tx, rx) = mpsc::channel::<Answer>();
        answer_txs.push(tx);
        answer_rxs.push(Some(rx));
    }
    let bounds_ref = &bounds;
    let (outcomes, batches, recalled) = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(running);
        for (replica, &share) in shares.iter().enumerate() {
            if share == 0 {
                continue;
            }
            let archive = Arc::clone(if shared {
                &archives[0]
            } else {
                &archives[replica]
            });
            let asks = ask_tx.clone();
            let answers = answer_rxs[replica]
                .take()
                .expect("one receiver per replica");
            let seed = replica_seed(seed, replica);
            let first = replica * dim / replicas;
            let handle = std::thread::Builder::new()
                .name(format!("anneal-replica-{replica}"))
                .spawn_scoped(scope, move || {
                    run_replica(
                        replica, bounds_ref, archive, share, seed, first, asks, answers,
                    )
                })
                .expect("spawn a replica thread");
            handles.push(handle);
        }
        drop(ask_tx);
        let (batches, recalled) = coordinate(
            obj,
            &archives,
            shared,
            &ask_rx,
            &answer_txs,
            &shares,
            &mut seen,
            &mut called,
        );
        drop(answer_txs);
        let outcomes: Vec<(Vec<ArmStat>, usize, usize)> = handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(outcome) => outcome,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect();
        (outcomes, batches + 1, recalled)
    });

    let best = archives
        .iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| a.best_get().total_cmp(&b.best_get()).then(i.cmp(j)))
        .map(|(_, archive)| archive)
        .expect("at least one archive");
    let mut arm_stats: Vec<ArmStat> = Vec::new();
    let mut n_evals = 1usize;
    let mut unseen = 0usize;
    for (stats, charged, unseen_by_replica) in outcomes {
        n_evals += charged;
        unseen += unseen_by_replica;
        for stat in stats {
            match arm_stats.iter_mut().find(|s| s.name == stat.name) {
                Some(total) => {
                    total.pulls += stat.pulls;
                    total.successes += stat.successes;
                }
                None => arm_stats.push(stat),
            }
        }
    }
    ReplicaResult {
        portfolio: PortfolioResult {
            best_pos: bounds.clip(best.incumbent(&bounds).view()).to_vec(),
            best_val: best.best_get(),
            n_evals: n_evals - recalled - unseen,
            n_grads: 0,
            n_answered: recalled + unseen,
            arm_stats,
            hop_state: None,
        },
        batches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::portfolio::{PortfolioPolicy, portfolio_optimize_from};
    use std::sync::atomic::AtomicUsize;

    /// An objective that records every point it is asked for and the size
    /// of every batch, with an optional batch method of its own.
    struct Recorder<F: Fn(ArrayView1<f64>) -> f64 + Send + Sync> {
        f: F,
        bounds: Bounds<f64>,
        points: Mutex<Vec<Vec<f64>>>,
        batches: Mutex<Vec<usize>>,
        calls: AtomicUsize,
    }

    impl<F: Fn(ArrayView1<f64>) -> f64 + Send + Sync> Recorder<F> {
        fn new(f: F, low: f64, high: f64, dim: usize) -> Self {
            Self {
                f,
                bounds: Bounds::new(
                    Array1::from_elem(dim, low),
                    Array1::from_elem(dim, high),
                    0.0,
                ),
                points: Mutex::new(Vec::new()),
                batches: Mutex::new(Vec::new()),
                calls: AtomicUsize::new(0),
            }
        }

        fn points(&self) -> Vec<Vec<f64>> {
            self.points.lock().expect("points").clone()
        }
    }

    impl<F: Fn(ArrayView1<f64>) -> f64 + Send + Sync> Objective<f64> for Recorder<F> {
        fn dim(&self) -> usize {
            self.bounds.dims
        }

        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.points.lock().expect("points").push(x.to_vec());
            (self.f)(x)
        }

        fn eval_batch(&self, x: ndarray::ArrayView2<f64>) -> Array1<f64> {
            self.batches.lock().expect("batches").push(x.nrows());
            x.outer_iter().map(|row| self.eval(row)).collect()
        }
    }

    fn rosenbrock(x: ArrayView1<f64>) -> f64 {
        (0..x.len() - 1)
            .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
            .sum()
    }

    fn rastrigin(x: ArrayView1<f64>) -> f64 {
        x.iter()
            .map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos() + 10.0)
            .sum()
    }

    fn start(dim: usize, value: f64) -> Array1<f64> {
        Array1::from_iter((0..dim).map(|i| value * (1.0 - 0.1 * i as f64)))
    }

    #[test]
    fn one_replica_repeats_the_solo_run_from_x0() {
        for (budget, seed) in [(300usize, 0u64), (1_500, 7)] {
            let x0 = start(6, 1.6);
            let solo = Recorder::new(rosenbrock, -2.0, 2.0, 6);
            let expected = portfolio_optimize_from::<_, NoGradient>(
                &solo,
                None,
                budget,
                seed,
                None,
                PortfolioPolicy::Auto,
                Some(x0.view()),
            );
            let batched = Recorder::new(rosenbrock, -2.0, 2.0, 6);
            let result =
                portfolio_replicas(&batched, budget, seed, x0.view(), 1, ReplicaArchive::Shared);
            assert_eq!(solo.points(), batched.points());
            assert_eq!(result.portfolio.best_pos, expected.best_pos);
            assert_eq!(
                result.portfolio.best_val.to_bits(),
                expected.best_val.to_bits()
            );
            assert_eq!(result.portfolio.n_evals, expected.n_evals);
            let pulls = |stats: &[ArmStat]| {
                stats
                    .iter()
                    .map(|s| (s.name, s.pulls, s.successes))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                pulls(&result.portfolio.arm_stats),
                pulls(&expected.arm_stats)
            );
        }
    }

    #[test]
    fn replicas_evaluate_the_start_once_then_batches_of_their_count() {
        for archive in [ReplicaArchive::Shared, ReplicaArchive::Private] {
            let x0 = start(8, 3.0);
            let obj = Recorder::new(rastrigin, -5.12, 5.12, 8);
            let result = portfolio_replicas(&obj, 801, 3, x0.view(), 4, archive);
            let points = obj.points();
            let batches = obj.batches.lock().expect("batches").clone();
            assert_eq!(points[0], x0.to_vec());
            assert_eq!(points.iter().filter(|p| **p == x0.to_vec()).count(), 1);
            assert_eq!(batches[0], 1);
            assert!(batches.iter().all(|&n| (1..=4).contains(&n)));
            assert_eq!(result.batches, batches.len());
            assert!(points.len() <= 801);
            let mut first = 0;
            for &n in &batches {
                let batch = &points[first..first + n];
                for (i, p) in batch.iter().enumerate() {
                    assert!(
                        !batch[..i].contains(p),
                        "{archive:?} repeats a point in a batch"
                    );
                }
                first += n;
            }
            if archive == ReplicaArchive::Private {
                // Each replica pays for a row at each of the 200 steps after
                // the start; the few rows at points evaluated before are
                // answered without a call, and a step's batch holds the rest.
                let full = batches[1..].iter().filter(|&&n| n == 4).count();
                assert!(points.len() > 790 && full > 190, "{batches:?}");
            } else {
                // Steps where replicas follow one descent, such as its line
                // searches, and the end of the run, once some replicas have
                // spent their shares, ask for fewer points than there are
                // replicas.
                let full: usize = batches.iter().filter(|&&n| n == 4).sum();
                assert!(2 * full > points.len(), "{batches:?}");
            }
            assert_eq!(result.portfolio.n_evals, points.len());
            for p in &points {
                assert!(p.iter().all(|v| (-5.12..=5.12).contains(v)));
            }
            let best = points
                .iter()
                .map(|p| rastrigin(ArrayView1::from(p)))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(result.portfolio.best_val, best);
        }
    }

    #[test]
    fn replicas_replay_a_seed_and_shared_runs_differ_from_private_ones() {
        let x0 = start(10, 1.5);
        let run = |archive, seed| {
            let obj = Recorder::new(rosenbrock, -2.0, 2.0, 10);
            let result = portfolio_replicas(&obj, 1_200, seed, x0.view(), 4, archive);
            (obj.points(), result.portfolio.best_val)
        };
        for seed in 0..3u64 {
            let shared = run(ReplicaArchive::Shared, seed);
            assert_eq!(shared, run(ReplicaArchive::Shared, seed));
            let private = run(ReplicaArchive::Private, seed);
            assert_eq!(private, run(ReplicaArchive::Private, seed));
            assert_ne!(shared.0, private.0);
        }
    }

    fn distinct(points: &[Vec<f64>]) -> usize {
        let mut distinct = points.to_vec();
        distinct.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        distinct.dedup();
        distinct.len()
    }

    #[test]
    fn shared_replicas_pay_once_for_a_point_they_all_ask_for() {
        // Every replica opens with the same deterministic descent from x0,
        // so with a shared archive those evaluations are paid once. With
        // private archives each replica pays for them, and as the objective
        // is called once at a point, what the others pay are calls the run
        // does not make.
        let x0 = start(4, 1.2);
        let shared = Recorder::new(rosenbrock, -2.0, 2.0, 4);
        let result = portfolio_replicas(&shared, 2_000, 1, x0.view(), 4, ReplicaArchive::Shared);
        let points = shared.points();
        assert!(points.len() <= 2_000);
        assert_eq!(result.portfolio.n_evals, points.len());
        assert_eq!(distinct(&points), points.len());
        let private = Recorder::new(rosenbrock, -2.0, 2.0, 4);
        let result = portfolio_replicas(&private, 2_000, 1, x0.view(), 4, ReplicaArchive::Private);
        let calls = private.points();
        assert_eq!(result.portfolio.n_evals, calls.len());
        assert_eq!(distinct(&calls), calls.len());
        assert!(
            points.len() > calls.len(),
            "{} against {}",
            points.len(),
            calls.len()
        );
    }

    #[test]
    fn replicas_call_the_objective_once_at_a_point() {
        // From the box's minimiser at a corner, replicas come back to the
        // start and to stencils they have asked for before, and with
        // private archives reach each other's points. Each point is called
        // once, the start first.
        let sphere = |x: ArrayView1<f64>| x.iter().map(|v| (v - 0.3).powi(2)).sum::<f64>();
        for dim in [1usize, 6] {
            for archive in [ReplicaArchive::Shared, ReplicaArchive::Private] {
                for seed in 0..3u64 {
                    let x0 = Array1::from_elem(dim, 1.0);
                    let obj = Recorder::new(sphere, 1.0, 2.0, dim);
                    let result = portfolio_replicas(&obj, 200, seed, x0.view(), 4, archive);
                    let points = obj.points();
                    let batches = obj.batches.lock().expect("batches").clone();
                    assert_eq!(points[0], x0.to_vec());
                    assert_eq!(
                        distinct(&points),
                        points.len(),
                        "{dim}-D {archive:?} {seed}"
                    );
                    assert_eq!(result.portfolio.n_evals, points.len());
                    assert_eq!(batches.iter().sum::<usize>(), points.len());
                    assert_eq!(result.batches, batches.len());
                }
            }
        }
    }

    #[test]
    fn replicas_split_the_budget_and_skip_an_empty_share() {
        assert_eq!(
            (0..4).map(|r| replica_share(11, 4, r)).collect::<Vec<_>>(),
            [3, 3, 2, 2]
        );
        assert_eq!(
            (0..4).map(|r| replica_share(3, 4, r)).collect::<Vec<_>>(),
            [1, 1, 0, 0]
        );
        let x0 = start(3, 0.5);
        for budget in [1usize, 2, 3, 6] {
            let obj = Recorder::new(rosenbrock, -2.0, 2.0, 3);
            let result = portfolio_replicas(&obj, budget, 0, x0.view(), 4, ReplicaArchive::Shared);
            assert!(obj.calls.load(Ordering::Relaxed) <= budget);
            assert_eq!(result.portfolio.n_evals, obj.calls.load(Ordering::Relaxed));
            assert!(result.portfolio.best_val <= rosenbrock(x0.view()));
        }
    }

    #[test]
    fn a_nan_value_is_answered_like_any_other() {
        let x0 = start(3, 0.5);
        let calls = AtomicUsize::new(0);
        let obj = Recorder::new(
            |x: ArrayView1<f64>| {
                if calls.fetch_add(1, Ordering::Relaxed) == 40 {
                    f64::NAN
                } else {
                    rosenbrock(x)
                }
            },
            -2.0,
            2.0,
            3,
        );
        let result = portfolio_replicas(&obj, 400, 0, x0.view(), 3, ReplicaArchive::Shared);
        assert!(result.portfolio.best_val.is_finite());
        assert_eq!(result.portfolio.n_evals, obj.points().len());
    }

    #[test]
    #[should_panic(expected = "objective failed")]
    fn an_objective_that_panics_ends_the_run_with_its_panic() {
        let x0 = start(3, 0.5);
        let calls = AtomicUsize::new(0);
        let obj = Recorder::new(
            |x: ArrayView1<f64>| {
                assert!(
                    calls.fetch_add(1, Ordering::Relaxed) < 40,
                    "objective failed"
                );
                rosenbrock(x)
            },
            -2.0,
            2.0,
            3,
        );
        portfolio_replicas(&obj, 400, 0, x0.view(), 3, ReplicaArchive::Shared);
    }
}
