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
use crate::methods::warm_lbfgs::WarmLbfgs;
use eindir_core::gradient::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};

/// What a search did, beyond the outcome the driver reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RelaxStats {
    /// Relaxations that reached a point with a small gradient.
    pub converged: usize,
    /// Relaxations that stopped at their iteration cap.
    ///
    /// A large share of these is not by itself wrong, because the screening
    /// pass is capped deliberately, but a run where nothing converges is not on
    /// the quenched landscape and every mechanism above it is acting on noise.
    pub capped: usize,
}

impl RelaxStats {
    /// Relaxation calls made.
    pub fn total(&self) -> usize {
        self.converged + self.capped
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
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, xr, _) = opt.minimize(x, iters, |v| {
            if !led.charge() {
                return None;
            }
            Some(objective.value_and_gradient(v))
        });
        // Charged like any other evaluation: asking whether a relaxation
        // converged is asking the potential a question, and a protocol that
        // counts evaluations has to count this one.
        let converged = if led.charge() {
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
