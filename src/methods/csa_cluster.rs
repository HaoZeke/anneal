//! A bank of chains under an annealed diversity threshold.
//!
//! Ties [`crate::methods::bank`] to the cluster driver: the bank decides where
//! to search from and what to keep, [`crate::methods::cluster_hopping::run`]
//! does the searching, and [`crate::diversity`] narrows the threshold as the
//! budget runs down.
//!
//! The division of labour is the point. A single biased chain at 75 points
//! reaches the global minimum in 4 runs of 8 at three million charged
//! evaluations, and the four that fail all fail the same way: they settle in
//! the icosahedral funnel and the decahedron is never seen. Nothing in a single
//! chain notices that, because from inside the funnel the search looks healthy.
//! A bank does notice, because it holds solutions from more than one funnel at
//! once and spends its next start on the one it knows least about.
//!
//! The budget is split, not multiplied. A bank of eight running eight chains of
//! an eighth the length costs what one chain costs, which is the comparison
//! that matters; a bank that quietly spends eight times the budget has proved
//! nothing.

use crate::bias::{Fingerprint, SortedPairs};
use crate::diversity::DiversityAnnealer;
use crate::methods::bank::{Admission, Bank};
use crate::methods::cluster_hopping::{random_cluster, Config, GradFn, Ledger, Outcome, Relax};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// How the bank is run.
#[derive(Debug, Clone)]
pub struct BankConfig {
    /// Solutions held at once.
    pub capacity: usize,
    /// Charged evaluations given to each chain before it reports back.
    ///
    /// Short enough that a bad region is abandoned quickly and long enough that
    /// a chain settles into a minimum: a slice that ends mid-descent offers the
    /// bank a structure that says nothing about where it was going.
    pub slice: usize,
    /// Chains run from random starts before the bank is first used, to give
    /// `Dcut` a scale taken from the data.
    pub seeding: usize,
    /// Floor of the `Dcut` schedule, as a fraction of its start.
    pub dcut_floor: f64,
}

impl Default for BankConfig {
    fn default() -> Self {
        Self {
            capacity: 8,
            slice: 60_000,
            seeding: 8,
            dcut_floor: 0.1,
        }
    }
}

/// What a bank run did.
#[derive(Debug, Clone)]
pub struct BankOutcome {
    /// Lowest value found anywhere.
    pub best: f64,
    /// The structure attaining it.
    pub best_state: Option<Array1<f64>>,
    /// Chains run.
    pub slices: usize,
    /// Charged evaluations spent.
    pub charged: usize,
    /// Candidates that improved the member they resembled.
    pub improved: usize,
    /// Candidates that resembled nothing in the bank.
    pub novel: usize,
    /// Candidates discarded as near-copies.
    pub duplicates: usize,
    /// `Dcut` at the start and at the end.
    pub dcut: (f64, f64),
    /// Energies held in the bank at the end, ascending.
    pub bank: Vec<f64>,
}

/// One chain against its own sub-ledger, settled up against the caller's.
///
/// The sub-ledger is not an accounting nicety. Handing the campaign ledger to
/// the inner run makes every budget-aware mechanism inside it, the
/// budget-window temperature above all, see the whole campaign's budget rather
/// than the slice it was given.
fn slice_run<'g>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: &mut Option<&mut GradFn<'g>>,
    start: ArrayView1<f64>,
    budget: usize,
    rng: &mut StdRng,
) -> Outcome {
    let mut slice_ledger = Ledger::new(budget.min(ledger.remaining()));
    let out = crate::methods::cluster_hopping::run_with_gradient(
        cfg,
        start,
        &mut slice_ledger,
        relax,
        grad.as_deref_mut(),
        rng,
    );
    ledger.charge_many(slice_ledger.spent());
    if let Some(st) = slice_ledger.best_state.as_ref() {
        ledger.record(slice_ledger.best, st.view());
    }
    out
}

/// Runs a bank of chains until the ledger is spent.
///
/// `distance` decides when two solutions are the same, and is the caller's:
/// with a shape distance `Dcut` is a length and transfers between sizes, which
/// a threshold in descriptor space does not.
pub fn run<'g, D>(
    cfg: &Config,
    bank_cfg: &BankConfig,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut grad: Option<&mut GradFn<'g>>,
    mut distance: D,
    seed: u64,
) -> BankOutcome
where
    D: FnMut(ArrayView1<f64>, ArrayView1<f64>) -> f64,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let total = ledger.remaining();
    let mut slices = 0usize;
    let mut improved = 0usize;
    let mut duplicates = 0usize;

    // The threshold is set from the seeding population below. Nothing is
    // judged against this one: the seeding phase bypasses the rule entirely.
    let mut bank = Bank::new(bank_cfg.capacity, 1.0);

    // Seeding: independent chains from random starts, which is also how the
    // threshold gets a scale.
    for _ in 0..bank_cfg.seeding {
        if ledger.remaining() == 0 {
            break;
        }
        let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
        let out = slice_run(cfg, ledger, relax, &mut grad, start.view(), bank_cfg.slice, &mut rng);
        slices += 1;
        if let Some(s) = out.best_state.as_ref() {
            bank.seed(s.view(), out.best);
        }
    }

    let mean = bank.mean_distance(&mut distance);
    let mut schedule = match mean {
        Some(m) => DiversityAnnealer::from_initial(0.5 * m),
        // Without two distinguishable members there is no scale to take, and a
        // number invented here would be the hand-set constant this replaces.
        // The run continues as plain restarts, which is what it already is.
        None => DiversityAnnealer::from_initial(cfg.merge_radius),
    }
    .with_final_fraction(bank_cfg.dcut_floor);
    let dcut0 = schedule.current();

    while ledger.remaining() > 0 {
        let progress = 1.0 - ledger.remaining() as f64 / total.max(1) as f64;
        bank.dcut = schedule.threshold(progress);

        let i = match bank.next_start() {
            Some(i) => i,
            None => break,
        };
        bank.mark_used(i);
        let start = bank.members()[i].state.clone();
        // Started from the member, not from its exact coordinates: a chain
        // begun at a minimum with a bias that has not seen it yet spends its
        // first hops rediscovering where it is.
        let mut kick = start.clone();
        for v in kick.iter_mut() {
            *v += rng.random_range(-0.15..0.15);
        }

        let out = slice_run(cfg, ledger, relax, &mut grad, kick.view(), bank_cfg.slice, &mut rng);
        slices += 1;
        if let Some(s) = out.best_state.as_ref() {
            match bank.offer(s.view(), out.best, &mut distance) {
                Admission::Improved(_) => improved += 1,
                Admission::Duplicate(_) => duplicates += 1,
                _ => {}
            }
        }
    }

    let mut energies: Vec<f64> = bank.members().iter().map(|m| m.energy).collect();
    energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    BankOutcome {
        best: ledger.best,
        best_state: ledger.best_state.clone(),
        slices,
        charged: ledger.spent(),
        improved,
        novel: bank.novel,
        duplicates,
        dcut: (dcut0, bank.dcut),
        bank: energies,
    }
}

/// Euclidean distance between sorted pairwise-distance spectra.
///
/// The default when no shape matching is available. It is a distance in
/// descriptor space, so its threshold does not transfer between sizes; with the
/// `ira` feature the caller should pass a shape distance instead.
pub fn spectrum_distance(n_points: usize) -> impl FnMut(ArrayView1<f64>, ArrayView1<f64>) -> f64 {
    let f = SortedPairs { n_points };
    move |a, b| {
        let da = f.describe(a);
        let db = f.describe(b);
        if da.len() != db.len() {
            return f64::INFINITY;
        }
        da.iter()
            .zip(db.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A separable washboard: `sum -cos(2 pi v) + 0.01 v^2`, so there is a
    /// minimum near every integer coordinate and the tilt makes them differ in
    /// energy. Enough to exercise the accounting and the control flow without a
    /// potential.
    ///
    /// A single-minimum objective will not do here even though it serves the
    /// other tests. Every chain converges to the same answer, so a bank holding
    /// one member is correct behaviour and a test asking for two is testing its
    /// own premise.
    fn washboard(x: ArrayView1<f64>) -> f64 {
        x.iter()
            .map(|v| -(2.0 * std::f64::consts::PI * v).cos() + 0.01 * v * v)
            .sum()
    }

    fn toy_relax(ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize) -> (f64, Array1<f64>) {
        let mut cur = x.to_owned();
        for _ in 0..steps {
            if !ledger.charge() {
                break;
            }
            for v in cur.iter_mut() {
                let g = 2.0 * std::f64::consts::PI
                    * (2.0 * std::f64::consts::PI * *v).sin()
                    + 0.02 * *v;
                *v -= 0.02 * g;
            }
        }
        let e = washboard(cur.view());
        (e, cur)
    }

    fn small_config(n: usize) -> Config {
        let mut cfg = Config::for_cluster(n);
        cfg.max_hops = Some(40);
        cfg
    }

    /// The accounting claim the comparison rests on: a bank of chains costs
    /// what one chain costs, because the budget is split rather than handed to
    /// each of them.
    #[test]
    fn the_bank_spends_one_budget_not_one_per_chain() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 4,
            slice: 400,
            seeding: 4,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(5_000);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            11,
        );
        assert!(
            out.charged <= 5_000,
            "spent {} against a budget of 5000",
            out.charged
        );
        assert!(out.slices > 1, "only {} chains ran", out.slices);
    }

    #[test]
    fn the_threshold_is_taken_from_the_seeding_and_narrows() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 6,
            slice: 300,
            seeding: 6,
            dcut_floor: 0.1,
        };
        let mut ledger = Ledger::new(12_000);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            3,
        );
        assert!(out.dcut.0 > 0.0, "no starting threshold was set");
        assert!(
            out.dcut.1 <= out.dcut.0 + 1e-12,
            "the threshold went from {} to {}",
            out.dcut.0,
            out.dcut.1
        );
    }

    /// A bank that ends holding one solution has not held anything apart.
    #[test]
    fn the_bank_ends_holding_more_than_one_solution() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 5,
            slice: 300,
            seeding: 5,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(9_000);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            7,
        );
        assert!(
            out.bank.len() >= 2,
            "the bank ended with {} members",
            out.bank.len()
        );
        // Ascending, so a caller reading the first entry gets the best.
        assert!(out.bank.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    }

    #[test]
    fn an_exhausted_budget_stops_the_run() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 4,
            slice: 1_000,
            seeding: 4,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(500);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            5,
        );
        assert!(out.charged <= 500, "spent {}", out.charged);
    }
}
