//! Record lines for thread-replica ensembles, shared by the LJ, water and
//! surface drivers so one summariser reads every campaign.

use anneal_core::methods::ensemble::{EnsembleConfig, EnsembleReport};
use ndarray::Array1;
use std::io::{self, Write};

/// Running totals over the seeds of one campaign arm.
#[derive(Default)]
pub struct Tally {
    /// Seeds whose ensemble reached the target.
    pub solved: usize,
    /// Seeds run.
    pub seeds: u64,
    /// Lowest energy over every seed.
    pub deepest: f64,
    /// Aggregate calls at first target, per solved seed.
    pub first_target: Vec<usize>,
}

impl Tally {
    /// Empty tally.
    pub fn new() -> Self {
        Self {
            deepest: f64::INFINITY,
            ..Self::default()
        }
    }
}

/// Prints one replica line per chain and one ensemble line per seed.
///
/// `verify` re-evaluates the returned coordinates read-only and returns the
/// energy and largest gradient component, or `None` when there is no state;
/// it asserts the record against the coordinates the way the seed loop does.
pub fn print_report(
    seed: u64,
    ens: &EnsembleConfig,
    report: &EnsembleReport,
    unit: &str,
    verify: &dyn Fn(usize, &Array1<f64>, f64) -> (f64, f64),
    tally: &mut Tally,
) {
    for run in &report.replicas {
        let verified = run
            .outcome
            .best_state
            .as_ref()
            .map(|x| verify(run.replica, x, run.outcome.best));
        let hit = ens.target.is_some_and(|t| run.outcome.best < t);
        println!(
            "    seed {seed} replica {} (seed {}): best {:.6}{unit}  hops {}  charged {}  basins {}  \
             history obs {} new {} refused {} secs {:.1}  shared_deposits {}  bias_published {}  \
             gossip {}  gossip_interval {}  two_choice_restarts {}  md {}/{}/{}  \
             escape {:.3} thr {:.4} same/known/new {}/{}/{}  first_target {}  wall {:.1}s  verified {}{}",
            run.replica,
            run.seed,
            run.outcome.best,
            run.outcome.hops,
            run.charged,
            run.outcome.basins,
            run.outcome.history_visits.0,
            run.outcome.history_visits.1,
            run.history_cost.1,
            run.history_cost.2,
            run.outcome.shared_deposits,
            run.bias_published,
            run.outcome.gossip_rounds,
            run.gossip_interval,
            run.two_choice_restarts,
            run.outcome.md_escape.0,
            run.outcome.md_escape.1,
            run.outcome.md_escape.2,
            run.outcome.escape_scale,
            run.outcome.escape_threshold,
            run.outcome.visit_counts.0,
            run.outcome.visit_counts.1,
            run.outcome.visit_counts.2,
            run.first_target_calls
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into()),
            run.wall_seconds,
            verified
                .map(|(e, gmax)| format!("{e:.6} |g| {gmax:.1e}"))
                .unwrap_or_else(|| "NO STATE".into()),
            if hit { "  SOLVED" } else { "" }
        );
    }
    let hit = report.solved(ens.target);
    tally.seeds += 1;
    if hit {
        tally.solved += 1;
    }
    if let Some(calls) = report.first_target_calls {
        tally.first_target.push(calls);
    }
    tally.deepest = tally.deepest.min(report.best);
    println!(
        "  seed {seed} ensemble: best {:.6}{unit}  aggregate charged {}  first_target_calls {}  \
         history minima/accepted/visits {:?}  bias exchange published {} delivered {}  wall {:.1}s{}",
        report.best,
        report.aggregate_charged,
        report
            .first_target_calls
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into()),
        report.histories,
        report.exchange.0,
        report.exchange.1,
        report.wall_seconds,
        if hit { "  SOLVED" } else { "" }
    );
    let _ = io::stdout().flush();
}

/// The campaign-arm footer the summariser and the note read.
pub fn print_tally(ens: &EnsembleConfig, tally: &Tally, reference: Option<f64>) {
    println!(
        "{}/{} solved ({} history, {} membership, {} replicas), deepest {:.6}, first_target_calls {:?}",
        tally.solved,
        tally.seeds,
        ens.history.name(),
        ens.membership.name(),
        ens.replicas,
        tally.deepest,
        tally.first_target
    );
    if let Some(r) = reference {
        println!("gap to reference {:+.6}", tally.deepest - r);
    }
}

/// Header naming every channel and the executable.
pub fn print_header(ens: &EnsembleConfig, witness: &str, mechanisms: &str, extra: &str) {
    println!(
        "  history ensembles: {} replicas, {} history, {} membership, shared bias {:?}, gossip {:?}, \
         two-choice stall {:?}, budgets {:?}, checkpoint {}, witness {witness}, {extra} \
         mechanisms {mechanisms}, executable sha256 {}",
        ens.replicas,
        ens.history.name(),
        ens.membership.name(),
        ens.shared_bias,
        ens.gossip,
        ens.two_choice_stall,
        ens.budgets(),
        ens.checkpoint_interval,
        executable_sha256()
    );
}

/// SHA-256 of the running executable, so a record names what produced it.
pub fn executable_sha256() -> String {
    use sha2::{Digest, Sha256};
    std::env::current_exe()
        .and_then(std::fs::read)
        .map(|bytes| format!("{:x}", Sha256::digest(&bytes)))
        .unwrap_or_else(|_| "unavailable".into())
}

/// Reads the ensemble's channels from the environment.
///
/// `HISTORY` shared, private or none (default private) with
/// `HISTORY_POLICY` accepted or observed-exclusion; `SHARED_BIAS=1` with
/// `SHARED_BIAS_WEIGHT`; `GOSSIP` ring or random with `GOSSIP_INTERVAL`,
/// `GOSSIP_WEIGHT`, `GOSSIP_TOP` (0 sends the whole table) and
/// `GOSSIP_ADAPTIVE=1`; `TWO_CHOICE_STALL` in charged calls;
/// `HISTORY_CHECKPOINT` for the exchange lag.
pub fn config_from_env(replicas: usize, budget: usize, target: Option<f64>) -> EnsembleConfig {
    use anneal_core::methods::ensemble::{GossipConfig, GossipTopology, HistoryMode};
    use anneal_core::methods::minima_hopping::HistoryMembership;
    fn parsed<T: std::str::FromStr>(name: &str) -> Option<T> {
        std::env::var(name).ok().and_then(|v| v.parse().ok())
    }
    let history = match std::env::var("HISTORY").as_deref() {
        Ok("shared") => HistoryMode::Shared,
        Ok("private") | Err(_) => HistoryMode::Private,
        Ok("none") => HistoryMode::None,
        Ok(other) => panic!("HISTORY={other:?}; expected shared, private or none"),
    };
    let membership = HistoryMembership::parse(std::env::var("HISTORY_POLICY").ok().as_deref())
        .unwrap_or_else(|error| panic!("{error}"));
    let shared_bias = std::env::var("SHARED_BIAS")
        .is_ok_and(|v| v == "1")
        .then(|| parsed::<f64>("SHARED_BIAS_WEIGHT").unwrap_or(1.0));
    let gossip = match std::env::var("GOSSIP").as_deref() {
        Ok("ring") => Some(GossipTopology::Ring),
        Ok("random") => Some(GossipTopology::Random),
        Ok("") | Err(_) => None,
        Ok(other) => panic!("GOSSIP={other:?}; expected ring or random"),
    }
    .map(|topology| GossipConfig {
        topology,
        interval: parsed::<usize>("GOSSIP_INTERVAL").unwrap_or(20_000).max(1),
        weight: parsed::<f64>("GOSSIP_WEIGHT").unwrap_or(0.5),
        adaptive: std::env::var("GOSSIP_ADAPTIVE").is_ok_and(|v| v == "1"),
        top: match parsed::<usize>("GOSSIP_TOP") {
            Some(0) => None,
            Some(count) => Some(count),
            None => Some(64),
        },
    });
    EnsembleConfig {
        replicas,
        budget,
        history,
        membership,
        shared_bias,
        gossip,
        two_choice_stall: parsed::<usize>("TWO_CHOICE_STALL").filter(|v| *v > 0),
        checkpoint_interval: parsed::<usize>("HISTORY_CHECKPOINT")
            .unwrap_or(1_000)
            .max(1),
        target,
    }
}
