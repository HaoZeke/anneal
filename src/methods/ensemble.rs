//! Thread-replica ensembles of the hop loop with typed communication.
//!
//! One seed is one ensemble: `replicas` chains of [`cluster_hopping`] that
//! divide one aggregate budget, start from their own random clusters with
//! their own streams, and talk through exactly the channels the
//! [`EnsembleConfig`] names. Every channel is count- or table-valued and
//! none moves a chain: a shared exact minimum history (identity and visit
//! counts), the multiple-walker exchange of bias visits, gossip averaging of
//! the wells over a ring or random pairs, and the two-choice family restart.
//! The private and shared arms of a comparison differ only in the channel,
//! never in seeds, starts, budgets or lookups, which is the contract the
//! campaign records are read under.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::catalog::euclidean_gradient_norm;
use crate::descriptor_space::DescriptorSpace;
use crate::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, Config, Ledger, Outcome, run_with_history_at_checkpoints,
};
use crate::methods::minima_hopping::{
    HistoryHook, HistoryMembership, MinimumHistory, SharedMinimumHistory,
};
use crate::methods::warm_lbfgs::WarmLbfgs;
use crate::pes_exploration::{ExactStructureWitness, StructureContext};
use crate::shared_bias::{SharedDeposits, visit_deltas};

/// Whether replicas keep an exact minimum history, and whether it is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryMode {
    /// No history at all: the control with no lookup cost.
    None,
    /// One history per replica: the isolated control with the lookup cost.
    Private,
    /// One history for the ensemble, behind a lock.
    Shared,
}

impl HistoryMode {
    /// Record label.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "no",
            Self::Private => "private",
            Self::Shared => "shared",
        }
    }
}

/// Graph a gossip round draws its peer from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GossipTopology {
    /// The two ring neighbours, alternating: spectral gap Theta(1/N^2).
    Ring,
    /// A uniform peer: randomised gossip on the complete graph.
    Random,
}

/// Gossip averaging of the wells (DeGroot over a graph).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GossipConfig {
    /// Peer graph.
    pub topology: GossipTopology,
    /// Charged calls between rounds.
    pub interval: usize,
    /// Step toward the peer; one half is the pairwise average, less is
    /// the stubborn agent of Friedkin and Johnsen.
    pub weight: f64,
    /// Additive-increase, multiplicative-decrease on the interval: a round
    /// that follows no improvement doubles the interval, an improvement
    /// resets it. Mambrini and Sudholt's rule, TCP's shape.
    pub adaptive: bool,
    /// Wells sent per round, deepest first; `None` sends the whole table,
    /// which makes every receiver's index the union of all walkers' basins.
    pub top: Option<usize>,
}

/// What an ensemble runs and how its chains communicate.
#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleConfig {
    /// Chains per seed.
    pub replicas: usize,
    /// Aggregate charged budget divided among the replicas, remainder to
    /// the low indices.
    pub budget: usize,
    /// Exact minimum history channel.
    pub history: HistoryMode,
    /// Membership policy the history reports under.
    pub membership: HistoryMembership,
    /// Multiple-walker exchange of bias visits, with the foreign deposit
    /// weight (1/N holds the deposition rate at one walker's).
    pub shared_bias: Option<f64>,
    /// Gossip averaging of the wells.
    pub gossip: Option<GossipConfig>,
    /// Two-choice family restart after this many charged calls without
    /// improvement (Azar, Broder, Karlin, Upfal 1999).
    pub two_choice_stall: Option<usize>,
    /// Charged calls between checkpoints, which is the exchange lag and
    /// the resolution of the first-target record.
    pub checkpoint_interval: usize,
    /// Energy below which a replica has found the target.
    pub target: Option<f64>,
}

impl EnsembleConfig {
    /// Per-replica budgets, remainder to the low indices.
    pub fn budgets(&self) -> Vec<usize> {
        (0..self.replicas)
            .map(|replica| {
                self.budget / self.replicas + usize::from(replica < self.budget % self.replicas)
            })
            .collect()
    }

    /// Replica seeds for one ensemble seed, identical in every arm.
    pub fn replica_seeds(&self, seed: u64) -> Vec<u64> {
        (0..self.replicas)
            .map(|replica| seed.wrapping_add((replica as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)))
            .collect()
    }

    /// Checks the ensemble against the chain configuration.
    pub fn validate(&self, cfg: &Config) -> Result<(), String> {
        if self.replicas == 0 {
            return Err("an ensemble needs at least one replica".into());
        }
        if self.budget < self.replicas {
            return Err("the aggregate budget must give every replica at least one call".into());
        }
        if cfg.replicas > 1 {
            return Err("an ensemble runs single-rung chains; a ladder owns its own bias".into());
        }
        if self.checkpoint_interval == 0 {
            return Err("the checkpoint interval must be positive".into());
        }
        if let Some(weight) = self.shared_bias
            && !(weight.is_finite() && weight > 0.0)
        {
            return Err("the shared bias weight must be positive".into());
        }
        if let Some(gossip) = self.gossip {
            if gossip.interval == 0 {
                return Err("the gossip interval must be positive".into());
            }
            if !(gossip.weight > 0.0 && gossip.weight <= 1.0) {
                return Err("the gossip weight must be in (0, 1]".into());
            }
        }
        Ok(())
    }
}

/// Value and gradient of the objective at a point, owned by one replica.
pub type Objective<'a> = &'a dyn Fn(ArrayView1<f64>) -> (f64, Array1<f64>);

/// Builds a replica's objective inside the replica's own thread.
///
/// A potential handle that is `Send` but not `Sync` (an xtb or EAM engine
/// behind FFI) is constructed once per replica here and never shared.
pub type ObjectiveFactory<'a> =
    &'a (dyn Fn(usize) -> Box<dyn Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + 'a> + Sync);

/// A replica's start structure from its own stream, also used for restarts.
pub type StartFactory<'a> = &'a (dyn Fn(usize, &mut StdRng) -> Array1<f64> + Sync);

/// Whether two occupied states stand in the same packing family.
pub type SameFamily<'a> = &'a (dyn Fn(&[f64], &[f64]) -> bool + Sync);

/// The problem an ensemble runs: objective, starts, identity and families.
pub struct EnsembleProblem<'a, W: ExactStructureWitness + Sync + ?Sized> {
    /// Objective per replica.
    pub objective: ObjectiveFactory<'a>,
    /// Start structure per replica.
    pub start: StartFactory<'a>,
    /// Descriptor ordering the exact witness checks.
    pub descriptor: &'a DescriptorSpace,
    /// Species and identity domain of the structures.
    pub context: &'a StructureContext,
    /// Exact identity, called from every replica thread.
    pub witness: &'a W,
    /// Packing-family predicate for the two-choice restart; `false`
    /// everywhere disables it where families are not defined.
    pub same_family: SameFamily<'a>,
    /// Gradient norm below which a relaxation certifies a minimum.
    pub certificate: f64,
    /// Gradient norm below which a stalled relaxation is polished on
    /// toward the certificate rather than abandoned.
    pub polish_below: f64,
}

/// What one replica reports.
pub struct ReplicaReport {
    /// Replica index within the ensemble.
    pub replica: usize,
    /// Its own seed.
    pub seed: u64,
    /// The chain's outcome.
    pub outcome: Outcome,
    /// Charged calls it consumed.
    pub charged: usize,
    /// Aggregate charged calls when its best first crossed the target.
    pub first_target_calls: Option<usize>,
    /// Wall seconds of the chain.
    pub wall_seconds: f64,
    /// History observations, refusals and seconds.
    pub history_cost: (usize, usize, f64),
    /// Own hop visits published to the shared bias exchange.
    pub bias_published: u64,
    /// Restarts taken by the two-choice rule.
    pub two_choice_restarts: usize,
    /// Gossip interval at the end, which the adaptive rule may have grown.
    pub gossip_interval: usize,
}

/// What one ensemble reports.
pub struct EnsembleReport {
    /// Every replica, in index order.
    pub replicas: Vec<ReplicaReport>,
    /// Lowest energy across replicas.
    pub best: f64,
    /// Earliest aggregate call count at which any replica crossed the target.
    pub first_target_calls: Option<usize>,
    /// Charged calls consumed by the ensemble.
    pub aggregate_charged: usize,
    /// Wall seconds from first thread to last join.
    pub wall_seconds: f64,
    /// Per history: minima, accepted minima, visits.
    pub histories: Vec<(usize, usize, u64)>,
    /// Bias exchange: visits published and delivered.
    pub exchange: (u64, u64),
}

impl EnsembleReport {
    /// Whether any replica reached the target.
    pub fn solved(&self, target: Option<f64>) -> bool {
        target.is_some_and(|t| self.best < t)
    }
}

/// Warm L-BFGS relaxation with the fresh certificate and the boundary on
/// the ledger.
///
/// Zero steps is one charged evaluation. Otherwise the relaxation runs
/// `iters` steps, then one fresh evaluation; a gradient norm between
/// `certificate` and `polish_below` continues in bounded chunks of 500
/// steps with plain descent on the last few per cent, so a fixed step
/// count that satisfies one size does not stall a whisker above the bound
/// on another. Only a norm below `certificate` is a certificate. The LJ
/// campaign values are 1e-5 and 1e-3 in reduced units.
pub fn validated_relax(
    objective: Objective<'_>,
    opt: &mut WarmLbfgs,
    led: &mut Ledger,
    x: ArrayView1<f64>,
    iters: usize,
    certificate: f64,
    polish_below: f64,
) -> (f64, Array1<f64>) {
    if iters == 0 {
        let energy = if led.charge() {
            objective(x).0
        } else {
            f64::INFINITY
        };
        return (energy, x.to_owned());
    }
    let charged_before = led.spent();
    opt.forget();
    let (_, mut xr, _) = opt.minimize(x, iters, |v| led.charge().then(|| objective(v)));
    let mut boundary_energy = f64::INFINITY;
    let mut validated_gradient = None;
    if led.charge() {
        let (fresh_energy, mut g) = objective(xr.view());
        boundary_energy = fresh_energy;
        let mut gnorm = euclidean_gradient_norm(g.as_slice().expect("gradient is contiguous"));
        let mut chunks = 0;
        while (certificate..polish_below).contains(&gnorm) && chunks < 10 && led.remaining() > 0 {
            opt.forget();
            let (_, xc, _) = opt.minimize(xr.view(), 500, |v| led.charge().then(|| objective(v)));
            boundary_energy = f64::INFINITY;
            xr = xc;
            if !led.charge() {
                break;
            }
            let (fe, ge) = objective(xr.view());
            boundary_energy = fe;
            gnorm = euclidean_gradient_norm(ge.as_slice().expect("gradient is contiguous"));
            g = ge;
            chunks += 1;
            let mut descents = 0;
            while (certificate..3.0 * certificate).contains(&gnorm)
                && descents < 200
                && led.charge()
            {
                for (value, gradient) in xr.iter_mut().zip(g.iter()) {
                    *value -= 0.01 * gradient;
                }
                let (fe, ge) = objective(xr.view());
                boundary_energy = fe;
                gnorm = euclidean_gradient_norm(ge.as_slice().expect("gradient is contiguous"));
                g = ge;
                descents += 1;
            }
        }
        if gnorm < certificate {
            validated_gradient = Some(g);
        }
    }
    led.record_quench_boundary(
        charged_before,
        boundary_energy,
        xr.clone(),
        validated_gradient,
    );
    (boundary_energy, xr)
}

/// A small generator for peer draws, so the chain's own stream is untouched.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Runs one ensemble seed.
///
/// The witness is called from every replica thread, so a non-reentrant
/// matcher goes behind [`crate::methods::minima_hopping::SerializedWitness`].
pub fn run_ensemble<W: ExactStructureWitness + Sync + ?Sized>(
    cfg: &Config,
    ens: &EnsembleConfig,
    seed: u64,
    problem: &EnsembleProblem<'_, W>,
) -> Result<EnsembleReport, String> {
    ens.validate(cfg)?;
    if !(problem.certificate > 0.0 && problem.polish_below >= problem.certificate) {
        return Err("the certificate must be positive and below the polish bound".into());
    }
    let (descriptor, context, witness, same_family) = (
        problem.descriptor,
        problem.context,
        problem.witness,
        problem.same_family,
    );
    let started = Instant::now();
    let replicas = ens.replicas;
    let budgets = ens.budgets();
    let replica_seeds = ens.replica_seeds(seed);
    let history_count = match ens.history {
        HistoryMode::None => 0,
        HistoryMode::Private => replicas,
        HistoryMode::Shared => 1,
    };
    let histories: Vec<Mutex<MinimumHistory>> = (0..history_count)
        .map(|_| {
            MinimumHistory::new(cfg.record_gradient.max(1e-5))
                .map(Mutex::new)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let charged_total = AtomicUsize::new(0);
    let exchange = Mutex::new(SharedDeposits::new(replicas));
    let mailboxes: Vec<Mutex<Option<Vec<(Array1<f64>, f64)>>>> =
        (0..replicas).map(|_| Mutex::new(None)).collect();
    let occupied: Vec<Mutex<Option<Vec<f64>>>> = (0..replicas).map(|_| Mutex::new(None)).collect();
    let target = ens.target;

    let reports: Vec<Result<ReplicaReport, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = budgets
            .iter()
            .zip(&replica_seeds)
            .enumerate()
            .map(|(replica, (&replica_budget, &replica_seed))| {
                let history = match ens.history {
                    HistoryMode::None => None,
                    HistoryMode::Private => Some(&histories[replica]),
                    HistoryMode::Shared => Some(&histories[0]),
                };
                let (exchange, mailboxes, occupied, charged_total) =
                    (&exchange, &mailboxes, &occupied, &charged_total);
                let context = context.clone();
                scope.spawn(move || -> Result<ReplicaReport, String> {
                    let replica_started = Instant::now();
                    let objective = (problem.objective)(replica);
                    let objective: Objective<'_> = &*objective;
                    let mut ledger = Ledger::new(replica_budget);
                    let mut opt = WarmLbfgs::default();
                    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                        let before = led.spent();
                        let out = validated_relax(
                            objective,
                            &mut opt,
                            led,
                            x,
                            iters,
                            problem.certificate,
                            problem.polish_below,
                        );
                        charged_total.fetch_add(led.spent() - before, Ordering::SeqCst);
                        out
                    };
                    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
                        if !led.charge() {
                            return None;
                        }
                        charged_total.fetch_add(1, Ordering::SeqCst);
                        Some(objective(x).1)
                    };
                    let mut hook = history.map(|history| {
                        SharedMinimumHistory::new(
                            history,
                            descriptor,
                            context,
                            witness,
                            ens.membership,
                        )
                    });
                    let mut first_target_calls: Option<usize> = None;
                    let mut seen_visits: Vec<u64> = Vec::new();
                    let mut published = 0u64;
                    let base_interval = ens.gossip.map_or(0, |g| g.interval);
                    let mut gossip_interval = base_interval;
                    let mut next_gossip = base_interval;
                    let mut best_at_gossip = f64::INFINITY;
                    let mut gossip_side = replica % 2;
                    let mut draw = replica_seed ^ 0x5DEE_CE66_D1CE_B00C;
                    let mut best_seen = f64::INFINITY;
                    let mut charged_at_best = 0usize;
                    let mut restart_rng = StdRng::seed_from_u64(replica_seed ^ 0x7C0A_1CE5);
                    let mut two_choice_restarts = 0usize;
                    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
                        if first_target_calls.is_none()
                            && target.is_some_and(|t| snapshot.best_energy() < t)
                        {
                            first_target_calls = Some(charged_total.load(Ordering::SeqCst));
                        }
                        if snapshot.best_energy() < best_seen - 1e-9 {
                            best_seen = snapshot.best_energy();
                            charged_at_best = snapshot.charged();
                        }
                        // Two-choice restart: post the occupied state, and on a
                        // stall sample two peers; both in this family means
                        // the family is crowded and the chain leaves it.
                        if let Some(stall) = ens.two_choice_stall
                            && replicas > 1
                        {
                            if let Some(mine) = snapshot.current_state().as_slice() {
                                *occupied[replica].lock().expect("occupied mailbox") =
                                    Some(mine.to_vec());
                            }
                            if snapshot.charged().saturating_sub(charged_at_best) >= stall
                                && let Some(mine) = snapshot.current_state().as_slice()
                            {
                                let mut crowded = 0usize;
                                for _ in 0..2 {
                                    let k = (lcg(&mut draw) % (replicas as u64 - 1)) as usize;
                                    let peer = (replica + 1 + k) % replicas;
                                    let theirs =
                                        occupied[peer].lock().expect("occupied mailbox").clone();
                                    if let Some(theirs) = theirs
                                        && theirs.len() == mine.len()
                                        && same_family(mine, &theirs)
                                    {
                                        crowded += 1;
                                    }
                                }
                                if crowded == 2 {
                                    two_choice_restarts += 1;
                                    charged_at_best = snapshot.charged();
                                    let fresh = (problem.start)(replica, &mut restart_rng);
                                    return CheckpointAction::ExternalAdopt {
                                        state: fresh,
                                        action: "two-choice-restart".to_owned(),
                                        external_calls: 0,
                                    };
                                }
                            }
                        }
                        // Gossip: post the wells, take a peer's, step toward them.
                        if let Some(gossip) = ens.gossip
                            && replicas > 1
                            && snapshot.charged() >= next_gossip
                            && let Some(bias) = snapshot.bias()
                        {
                            if gossip.adaptive {
                                if snapshot.best_energy() < best_at_gossip - 1e-9 {
                                    gossip_interval = base_interval;
                                } else {
                                    gossip_interval = (gossip_interval * 2).min(base_interval * 16);
                                }
                                best_at_gossip = snapshot.best_energy();
                            }
                            next_gossip = snapshot.charged() + gossip_interval;
                            *mailboxes[replica].lock().expect("gossip mailbox") =
                                Some(match gossip.top {
                                    Some(count) => bias.deepest_wells(count),
                                    None => bias.wells(),
                                });
                            let peer = match gossip.topology {
                                GossipTopology::Ring => {
                                    gossip_side ^= 1;
                                    if gossip_side == 0 {
                                        (replica + 1) % replicas
                                    } else {
                                        (replica + replicas - 1) % replicas
                                    }
                                }
                                GossipTopology::Random => {
                                    let k = (lcg(&mut draw) % (replicas as u64 - 1)) as usize;
                                    (replica + 1 + k) % replicas
                                }
                            };
                            let wells = mailboxes[peer].lock().expect("gossip mailbox").clone();
                            if let Some(wells) = wells {
                                return CheckpointAction::MergeBias {
                                    wells,
                                    weight: gossip.weight,
                                    complete: gossip.top.is_none(),
                                };
                            }
                        }
                        // Multiple-walker exchange: publish own visit deltas,
                        // take everyone else's since the last look.
                        let Some(weight) = ens.shared_bias else {
                            return CheckpointAction::Continue;
                        };
                        let Some(bias) = snapshot.bias() else {
                            return CheckpointAction::Continue;
                        };
                        let index = bias.index();
                        let mine = visit_deltas(
                            |i| index.centre(i),
                            |i| index.visits(i),
                            bias.n_basins(),
                            &mut seen_visits,
                        );
                        published += mine.iter().map(|(_, count)| *count).sum::<u64>();
                        let mut exchange = exchange.lock().expect("shared deposit exchange");
                        exchange.publish(replica, mine);
                        let deposits = exchange.drain(replica);
                        drop(exchange);
                        if deposits.is_empty() {
                            CheckpointAction::Continue
                        } else {
                            CheckpointAction::DepositDescriptors { deposits, weight }
                        }
                    };
                    let mut rng = StdRng::seed_from_u64(replica_seed);
                    let start = (problem.start)(replica, &mut rng);
                    let outcome = run_with_history_at_checkpoints(
                        cfg,
                        start.view(),
                        &mut ledger,
                        &mut relax,
                        Some(&mut grad),
                        None,
                        hook.as_mut().map(|hook| hook as &mut dyn HistoryHook),
                        &mut rng,
                        ens.checkpoint_interval,
                        &mut checkpoint,
                    );
                    if first_target_calls.is_none() && target.is_some_and(|t| outcome.best < t) {
                        first_target_calls = Some(charged_total.load(Ordering::SeqCst));
                    }
                    Ok(ReplicaReport {
                        replica,
                        seed: replica_seed,
                        outcome,
                        charged: ledger.spent(),
                        first_target_calls,
                        wall_seconds: replica_started.elapsed().as_secs_f64(),
                        history_cost: hook.as_ref().map_or((0, 0, 0.0), |hook| hook.cost()),
                        bias_published: published,
                        two_choice_restarts,
                        gossip_interval,
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err("an ensemble replica panicked".to_string()))
            })
            .collect()
    });
    let replicas = reports.into_iter().collect::<Result<Vec<_>, _>>()?;
    let best = replicas
        .iter()
        .map(|report| report.outcome.best)
        .fold(f64::INFINITY, f64::min);
    let first_target_calls = replicas
        .iter()
        .filter_map(|report| report.first_target_calls)
        .min();
    let histories = histories
        .iter()
        .map(|history| {
            let history = history.lock().expect("minimum history");
            (
                history.minimum_count(),
                history.accepted_count(),
                history.total_visits(),
            )
        })
        .collect();
    let exchange = exchange
        .lock()
        .map(|exchange| exchange.counts())
        .unwrap_or((0, 0));
    Ok(EnsembleReport {
        replicas,
        best,
        first_target_calls,
        aggregate_charged: charged_total.load(Ordering::SeqCst),
        wall_seconds: started.elapsed().as_secs_f64(),
        histories,
        exchange,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
    use crate::methods::cluster_hopping::random_cluster_in_radius;
    use crate::methods::minima_hopping::SerializedWitness;

    /// Two wells: a tetrahedron and its mirror scaled by 1.7, so chains have
    /// somewhere to go and something to disagree about.
    fn two_well(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let a = Array1::from(vec![
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        ]);
        let b = a.mapv(|v| -1.7 * v);
        let da = &x - &a;
        let db = &x - &b;
        let ea = da.dot(&da);
        let eb = db.dot(&db) - 0.5;
        if ea < eb {
            (ea, 2.0 * da)
        } else {
            (eb, 2.0 * db)
        }
    }

    fn ensemble(
        history: HistoryMode,
        shared_bias: Option<f64>,
        gossip: Option<GossipConfig>,
    ) -> EnsembleConfig {
        EnsembleConfig {
            replicas: 3,
            budget: 30_000,
            history,
            membership: HistoryMembership::Accepted,
            shared_bias,
            gossip,
            two_choice_stall: Some(2_000),
            checkpoint_interval: 300,
            target: Some(-0.4),
        }
    }

    fn chain_config() -> Config {
        let mut cfg = Config::for_cluster(4);
        cfg.screen_steps = 1;
        cfg.relax_steps = 120;
        cfg.screen_margin = f64::INFINITY;
        cfg.return_screen = false;
        cfg
    }

    #[test]
    fn budgets_and_seeds_are_identical_across_arms_and_sum_to_the_budget() {
        let a = ensemble(HistoryMode::Private, None, None);
        let b = ensemble(HistoryMode::Shared, Some(0.25), None);
        assert_eq!(a.budgets(), b.budgets());
        assert_eq!(a.budgets().iter().sum::<usize>(), 30_000);
        assert_eq!(a.replica_seeds(7), b.replica_seeds(7));
        assert_ne!(a.replica_seeds(7), a.replica_seeds(8));
        let mut ladder = chain_config();
        ladder.replicas = 2;
        assert!(a.validate(&ladder).is_err());
        let mut bad = a.clone();
        bad.gossip = Some(GossipConfig {
            topology: GossipTopology::Ring,
            interval: 10,
            weight: 1.5,
            adaptive: false,
            top: None,
        });
        assert!(bad.validate(&chain_config()).is_err());
    }

    #[test]
    fn every_channel_runs_and_reports_its_traffic() {
        let cfg = chain_config();
        let descriptor = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
        let context = StructureContext::new(Some(vec![18; 4]), None, Some("two-well".into()));
        let witness = SerializedWitness(Mutex::new(|l: ArrayView1<f64>, r: ArrayView1<f64>| {
            l.iter()
                .zip(r.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
                < 0.5
        }));
        let same_family = |a: &[f64], b: &[f64]| {
            // Same well when the first coordinate has the same sign.
            (a[0] > 0.0) == (b[0] > 0.0)
        };
        let objective: ObjectiveFactory<'_> = &|_| Box::new(two_well);
        let start: StartFactory<'_> =
            &|_, rng| random_cluster_in_radius(4, cfg.start_radius(), cfg.min_separation, rng);
        let problem = EnsembleProblem {
            objective,
            start,
            descriptor: &descriptor,
            context: &context,
            witness: &witness,
            same_family: &same_family,
            certificate: 1e-5,
            polish_below: 1e-3,
        };
        let gossip = GossipConfig {
            topology: GossipTopology::Ring,
            interval: 1_000,
            weight: 0.5,
            adaptive: true,
            top: Some(8),
        };
        let shared = run_ensemble(
            &cfg,
            &ensemble(HistoryMode::Shared, Some(0.5), Some(gossip)),
            3,
            &problem,
        )
        .unwrap();
        assert_eq!(shared.replicas.len(), 3);
        assert!(shared.aggregate_charged >= 3 * 9_000);
        assert!(shared.replicas.iter().any(|r| r.outcome.gossip_rounds > 0));
        assert!(shared.exchange.0 > 0 && shared.exchange.1 > 0);
        assert_eq!(shared.histories.len(), 1);
        assert!(shared.histories[0].2 > 0);
        assert!(shared.solved(Some(-0.4)));
        let private =
            run_ensemble(&cfg, &ensemble(HistoryMode::None, None, None), 3, &problem).unwrap();
        assert!(private.histories.is_empty());
        assert_eq!(private.exchange, (0, 0));
        assert!(
            private
                .replicas
                .iter()
                .all(|r| r.outcome.gossip_rounds == 0)
        );
        assert!(
            private
                .replicas
                .iter()
                .all(|r| r.outcome.shared_deposits == 0)
        );
        assert!(
            private
                .replicas
                .iter()
                .all(|r| r.history_cost == (0, 0, 0.0))
        );
    }
}
