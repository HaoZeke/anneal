//! Running a cluster search against an objective.
//!
//! [`crate::methods::cluster_hopping::run`] takes a relaxation and a gradient as
//! closures, which is the right interface for a driver: it does not care where
//! the energy comes from. It is the wrong interface for a caller, because every
//! caller then writes the same three things, and the campaign this crate
//! reports was run against potentials defined inside its own examples.
//!
//! This is the missing half. Hand it anything implementing
//! [`DifferentiableObjective<f64>`] and it builds the relaxation, charges every
//! evaluation to the ledger, counts what converged and runs the search.
//!
//! What that buys is provenance. rgpot reaches this crate as an
//! `eindir_objective_t`, wrapped into an `Objective<f64>`, so a potential from
//! there arrives at the cluster driver by the same route as one written here
//! and neither the driver nor this function can tell them apart.

use crate::methods::cluster_hopping::{optimize_with_gradient, Config, Ledger, Outcome};
use crate::quench::{QuenchPredictor, Verdict};
use crate::methods::warm_lbfgs::WarmLbfgs;
use eindir_core::gradient::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};

/// What a search did, beyond the outcome the driver reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RelaxStats {
    /// Relaxations that reached a point with a small gradient.
    pub converged: usize,
    /// Charged evaluations spent in screening passes.
    ///
    /// Split from the full relaxations because the two are different levers.
    /// Every mechanism in this crate that tried to change where the chain goes
    /// was measured and failed; the one that helped, the return screen, buys
    /// hops by not paying for relaxations that will be discarded. If throughput
    /// is what moves the number then knowing which pass the budget goes to is
    /// the first thing to establish, and it has never been measured here.
    pub screen_charged: usize,
    /// Charged evaluations spent in full relaxations.
    pub full_charged: usize,
    /// Charged evaluations spent confirming convergence.
    pub check_charged: usize,
    /// Relaxations that stopped at their iteration cap.
    ///
    /// A large share of these is not by itself wrong, because the screening
    /// pass is capped deliberately, but a run where nothing converges is not on
    /// the quenched landscape and every mechanism above it is acting on noise.
    pub capped: usize,
    /// Screening passes run.
    pub screens: usize,
    /// Descent steps those passes took, summed.
    ///
    /// Against `screens * screen_steps` this is what stopping on a decision
    /// bought, and it is the only number that says whether it bought anything.
    pub screen_steps_taken: usize,
}

impl RelaxStats {
    /// Relaxation calls made.
    pub fn total(&self) -> usize {
        self.converged + self.capped
    }

    /// Charged evaluations across both passes and the convergence check.
    pub fn charged(&self) -> usize {
        self.screen_charged + self.full_charged + self.check_charged
    }

    /// Share of the charged budget spent screening.
    pub fn screen_share(&self) -> f64 {
        let t = self.charged();
        if t == 0 {
            return 0.0;
        }
        self.screen_charged as f64 / t as f64
    }
}

/// Gradient magnitude below which a relaxation counts as converged.
///
/// Loose enough that a screening pass is not called converged and tight enough
/// that a genuine minimum is: on a Lennard-Jones cluster a quenched structure
/// comes back at about 1e-6.
pub const CONVERGED_GRADIENT: f64 = 1e-5;

/// Runs a cluster search on `objective` under `ledger`.
///
/// The relaxation is this crate's warm-started quasi-Newton one, and its
/// curvature is deliberately not carried between calls: measured on a cluster,
/// retaining it across a structural change costs more than it saves.
pub fn search<O>(objective: &O, cfg: &Config, ledger: &mut Ledger, seed: u64) -> (Outcome, RelaxStats)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut stats = RelaxStats::default();
    let mut opt = WarmLbfgs::default();

    // Split deliberately: the relaxation needs the optimizer mutably, the
    // gradient needs only the objective. Sharing the objective by reference is
    // what lets both closures exist at once.
    // The driver calls the screening pass with `screen_steps` and the full one
    // with `relax_steps`, so the iteration count identifies which is which.
    let screen_iters = cfg.screen_steps;
    let adaptive = cfg.adaptive_screen;
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let before = led.spent();
        let screening = iters <= screen_iters;
        // The screening pass stops when its own trajectory says the question is
        // settled; the full relaxation runs to its tolerance, because there the
        // answer is the structure and not a verdict about it.
        let mut pred = QuenchPredictor::new();
        let mut early = false;
        let target = led.best;
        let (f, xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                Some(objective.value_and_gradient(v))
            },
            |_, fv| {
                if !(screening && adaptive) {
                    return true;
                }
                pred.observe(fv);
                if pred.verdict(target) == Verdict::Undecided {
                    return true;
                }
                early = true;
                false
            },
        );
        let cost = led.spent() - before;
        if screening {
            stats.screen_charged += cost;
            stats.screen_steps_taken += pred.len();
            stats.screens += 1;
        } else {
            stats.full_charged += cost;
        }
        // The energy the caller sees is the extrapolated limit, not the value
        // at the point where the descent stopped.
        //
        // This is the whole of it. Stopping a screening quench after five steps
        // instead of twenty-five cut the cost of a hop from 31 charged
        // evaluations to 8 and quadrupled the hops, and solved nothing in three
        // seeds where the fixed-length screen solved three: the value at step
        // five is not the quenched energy, and a chain that accepts on it is
        // walking on the raw landscape rather than the transformed one that
        // basin hopping exists to walk on. The predictor already says where the
        // descent was going; the estimate is what the chain should move on.
        //
        // Safe against reporting a point that is not a minimum, because the
        // early stop only fires on a verdict, and a `Hopeless` verdict means
        // the predicted limit sits above the incumbent by more than its own
        // error. An extrapolated energy therefore cannot become the run's best.
        // The `Promising` branch does not use it at all: it goes on to the full
        // relaxation, which returns a real value.
        let f = if early {
            match pred.predict() {
                Some(p) if p.limit.is_finite() && p.limit < f => p.limit,
                _ => f,
            }
        } else {
            f
        };
        // A descent stopped on a verdict is known not to be converged, so
        // asking is spending an evaluation on an answer already in hand. The
        // check stays on every other path, where the answer is not known.
        if early {
            stats.capped += 1;
            return (f, xr);
        }
        // Charged like any other evaluation: asking whether a relaxation
        // converged is asking the potential a question, and a protocol that
        // counts evaluations has to count this one.
        let converged = if led.charge() {
            stats.check_charged += 1;
            let g = objective.grad(xr.view());
            g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < CONVERGED_GRADIENT
        } else {
            false
        };
        if converged {
            stats.converged += 1;
        } else {
            stats.capped += 1;
        }
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(objective.grad(x))
    };

    let out = optimize_with_gradient(cfg, ledger, &mut relax, Some(&mut grad), seed);
    (out, stats)
}

/// Work spent before a run first reached `target`, or how much it spent
/// without reaching it.
///
/// The statistic to report. A success rate at a fixed budget is this quantity
/// pushed through an arbitrary threshold: above the budget it saturates and
/// hides the margin, below it censors and hides how near the failures came.
/// Eight seeds in eight at twelve million evaluations and five in eight at
/// three million are the same method described twice, badly.
///
/// A first encounter time is a property of the method. It is what lets one
/// paper's result be compared with another's, and it is what makes a claim like
/// a seventyfold improvement mean something.
///
/// # Censoring
///
/// A run that never reached the target has not produced a first encounter time;
/// it has produced a lower bound. That is [`Encounter::Censored`], and it must
/// not be dropped or replaced by the budget: dropping the failures reports the
/// mean of the successes, which is smaller than the truth and gets smaller as
/// the method gets worse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Encounter {
    /// Charged evaluations spent when the target was first reached.
    Found {
        /// Charged evaluations at the first crossing.
        charged: usize,
        /// Hops at the first crossing.
        hops: usize,
    },
    /// The target was never reached; the run spent this much without it.
    Censored {
        /// Charged evaluations spent in total.
        charged: usize,
    },
}

impl Encounter {
    /// The charged count either way, which is the encounter time when found and
    /// a lower bound on it when censored.
    pub fn charged(&self) -> usize {
        match self {
            Encounter::Found { charged, .. } | Encounter::Censored { charged } => *charged,
        }
    }

    /// Whether the target was reached.
    pub fn found(&self) -> bool {
        matches!(self, Encounter::Found { .. })
    }
}

/// The first encounter with `target` in a run's improvement trace.
///
/// `target` is compared with a tolerance, since a published minimum is quoted
/// to six decimals and a relaxation lands near it rather than on it.
pub fn first_encounter(out: &Outcome, target: f64, tolerance: f64, spent: usize) -> Encounter {
    for &(hops, charged, _, e) in &out.improvements {
        if e < target + tolerance {
            return Encounter::Found { charged, hops };
        }
    }
    Encounter::Censored { charged: spent }
}

/// Median first encounter time under censoring, by Kaplan-Meier.
///
/// The median is the point where the survival function first falls to a half.
/// `None` when more than half the runs are censored, which is the honest answer:
/// the median has not been observed, and quoting the mean of the successes
/// instead reports a number that improves as the method gets worse.
pub fn median_encounter(runs: &[Encounter]) -> Option<usize> {
    if runs.is_empty() {
        return None;
    }
    let mut events: Vec<(usize, bool)> = runs
        .iter()
        .map(|e| (e.charged(), e.found()))
        .collect();
    events.sort_by_key(|(c, _)| *c);

    let mut at_risk = events.len() as f64;
    let mut survival = 1.0_f64;
    for (c, found) in events {
        if found {
            survival *= 1.0 - 1.0 / at_risk;
            if survival <= 0.5 {
                return Some(c);
            }
        }
        at_risk -= 1.0;
        if at_risk <= 0.0 {
            break;
        }
    }
    None
}

/// Checks that a reported result is what it claims to be.
///
/// Returns the energy of the returned structure and its largest gradient
/// component, both computed off the ledger and outside the driver. `None` when
/// no structure came back at all.
///
/// Worth having as a function rather than as a line in each example, because
/// checking only the energy is not enough: an arm of this crate once returned a
/// structure carrying the right energy with a gradient of 0.31, which is not a
/// minimum, and the energy check passed.
pub fn verify<O>(objective: &O, out: &Outcome) -> Option<(f64, f64)>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let x = out.best_state.as_ref()?;
    let (e, g) = objective.value_and_gradient(x.view());
    Some((e, g.iter().fold(0.0_f64, |a, v| a.max(v.abs()))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potentials::PairPotential;

    /// The search runs against a potential passed as a trait object, which is
    /// the whole point: an rgpot potential arrives the same way.
    #[test]
    fn it_searches_an_objective_given_as_a_trait_object() {
        let pot = PairPotential::lennard_jones(13);
        let dyn_pot: &dyn DifferentiableObjective<f64> = &pot;
        let mut cfg = Config::for_cluster(13);
        cfg.allocate_moves = true;
        cfg.return_screen = true;
        let mut ledger = Ledger::new(120_000);
        let (out, stats) = search(dyn_pot, &cfg, &mut ledger, 0);

        assert!(out.hops > 0, "no hops were taken");
        assert!(stats.total() > 0, "no relaxations were counted");
        assert!(
            stats.converged > 0,
            "nothing converged, so the chain is not on the quenched landscape"
        );
        assert!(ledger.spent() <= 120_000, "spent {}", ledger.spent());
    }

    /// The returned structure has to be a minimum carrying the reported energy,
    /// and `verify` is what says so.
    #[test]
    fn verify_reports_the_energy_and_the_gradient_of_what_came_back() {
        let pot = PairPotential::lennard_jones(13);
        let mut cfg = Config::for_cluster(13);
        cfg.return_screen = true;
        let mut ledger = Ledger::new(120_000);
        let (out, _) = search(&pot, &cfg, &mut ledger, 3);
        let (e, gmax) = verify(&pot, &out).expect("no structure came back");
        assert!(
            (e - out.best).abs() < 1e-6,
            "reported {} but the structure is {e}",
            out.best
        );
        assert!(gmax < 1e-3, "returned a structure with gradient {gmax:.2e}");
    }

    /// A run that reached the target reports where, and one that did not is
    /// censored rather than being given the budget as its time.
    #[test]
    fn an_encounter_is_found_or_censored() {
        let pot = PairPotential::lennard_jones(13);
        let mut cfg = Config::for_cluster(13);
        cfg.allocate_moves = true;
        cfg.return_screen = true;
        let mut ledger = Ledger::new(150_000);
        let (out, _) = search(&pot, &cfg, &mut ledger, 0);
        let e = first_encounter(&out, -44.326801, 1e-4, ledger.spent());
        match e {
            Encounter::Found { charged, hops } => {
                assert!(charged > 0 && charged <= ledger.spent());
                assert!(hops > 0 && hops <= out.hops);
            }
            Encounter::Censored { charged } => assert_eq!(charged, ledger.spent()),
        }

        // A target nothing can reach must censor at the spend, not report a
        // time.
        let never = first_encounter(&out, -1e9, 1e-4, ledger.spent());
        assert_eq!(never, Encounter::Censored { charged: ledger.spent() });
    }

    /// The median under censoring, and the refusal that keeps it honest.
    #[test]
    fn the_median_refuses_when_most_runs_are_censored() {
        let found = |c: usize| Encounter::Found { charged: c, hops: c / 30 };
        let cens = |c: usize| Encounter::Censored { charged: c };

        // Five found, spread; the median is the third.
        let all = vec![found(10), found(20), found(30), found(40), found(50)];
        assert_eq!(median_encounter(&all), Some(30));

        // One found early, four censored late: the survival function never
        // reaches a half, so there is no median to quote.
        let mostly = vec![found(10), cens(90), cens(91), cens(92), cens(93)];
        assert_eq!(median_encounter(&mostly), None);

        // Censoring must not be treated as a success: replacing the censored
        // runs with successes at the same times gives a median, which is
        // exactly the error this guards against.
        let wrong = vec![found(10), found(90), found(91), found(92), found(93)];
        assert!(median_encounter(&wrong).is_some());
    }

    /// A censored run late in the ordering must not shrink the median.
    #[test]
    fn late_censoring_does_not_flatter_the_median() {
        let found = |c: usize| Encounter::Found { charged: c, hops: 1 };
        let cens = |c: usize| Encounter::Censored { charged: c };
        let clean = vec![found(10), found(20), found(30), found(40)];
        let with_censor = vec![found(10), found(20), found(30), cens(1000)];
        let a = median_encounter(&clean).unwrap();
        let b = median_encounter(&with_censor).unwrap();
        assert!(b >= a, "censoring moved the median from {a} down to {b}");
    }

    /// LJ13 is the case with one answer everyone agrees on, so it is the one
    /// that says the plumbing did not quietly change the problem.
    #[test]
    fn it_finds_the_thirteen_point_icosahedron() {
        let pot = PairPotential::lennard_jones(13);
        let mut cfg = Config::for_cluster(13);
        cfg.allocate_moves = true;
        cfg.return_screen = true;
        let mut best = f64::INFINITY;
        for seed in 0..3 {
            let mut ledger = Ledger::new(150_000);
            let (out, _) = search(&pot, &cfg, &mut ledger, seed);
            best = best.min(out.best);
        }
        assert!(
            best < -44.326801 + 1e-4,
            "best {best} against the published -44.326801"
        );
    }

    /// The budget is the experiment, so nothing may run past it.
    #[test]
    fn it_stops_at_the_budget() {
        let pot = PairPotential::morse(13, 6.0);
        let cfg = Config::for_cluster(13);
        let mut ledger = Ledger::new(5_000);
        let (_, _) = search(&pot, &cfg, &mut ledger, 1);
        assert!(ledger.spent() <= 5_000, "spent {}", ledger.spent());
    }
}
