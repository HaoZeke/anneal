//! A population resampled by estimated probability of improvement.
//!
//! The committor of rare-event theory is the probability that a trajectory
//! from a state reaches the product before returning to the reactant. Without
//! naming any product, the search-facing analogue is the probability that a
//! walk segment from a state produces a *new lower minimum* before the segment
//! ends: defined by energy and history alone, no order parameter, no target.
//!
//! Each walker carries a Beta posterior over that probability, updated by one
//! Bernoulli observation per segment. Resampling is Thompson: the walker with
//! the worst posterior draw is replaced by a clone of the one with the best.
//! The distinction from diffusion Monte Carlo weighting is the point.
//! Boltzmann weights `exp(-beta E)` clone the deepest walkers, and the
//! measured failure mode of this landscape is precisely a walker pinned at
//! the deepest wrong minimum; an improvement-committor posterior culls a
//! deep-but-dead walker that depth weighting would multiply.
//!
//! Segments are ordinary runs of the cluster driver with whatever mechanism
//! stack the caller configured, so everything measured to work (the composed
//! excursion, the depth bandit) rides along unchanged.

use crate::methods::cluster_hopping::{
    random_cluster, run_with_gradient, Config, GradFn, Ledger, Relax,
};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Outcome of a committor-population run.
#[derive(Debug, Clone)]
pub struct CommittorOutcome {
    /// Best quenched energy seen by any walker.
    pub best: f64,
    /// Resampling events performed.
    pub resamples: usize,
    /// Segments run.
    pub segments: usize,
    /// Improvement events observed.
    pub improvements: usize,
}

/// Runs `walkers` short chains under `cfg`, resampling by improvement
/// posterior after each round, until the ledger refuses.
pub fn committor_population(
    cfg: &Config,
    walkers: usize,
    segment_budget: usize,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    seed: u64,
) -> CommittorOutcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = cfg.n_points;
    // (structure, own best, alpha, beta)
    let mut pop: Vec<(Array1<f64>, f64, f64, f64)> = (0..walkers.max(2))
        .map(|_| {
            (
                random_cluster(n, 0.7, cfg.min_separation, &mut rng),
                f64::INFINITY,
                1.0,
                1.0,
            )
        })
        .collect();
    let mut best = f64::INFINITY;
    let mut resamples = 0usize;
    let mut segments = 0usize;
    let mut improvements = 0usize;
    let no_grad: Option<&mut GradFn<'_>> = None;
    let _ = no_grad;
    while ledger.remaining() > 0 {
        for w in 0..pop.len() {
            if ledger.remaining() == 0 {
                break;
            }
            let slice = segment_budget.min(ledger.remaining());
            let mut sub = Ledger::new(slice);
            let start = pop[w].0.clone();
            let out = run_with_gradient(
                cfg,
                start.view(),
                &mut sub,
                relax,
                None,
                &mut rng,
            );
            segments += 1;
            if !ledger.charge_many(sub.spent()) {
                // The slice overran the remaining budget; the overrun is
                // charged and the loop ends on the next check.
            }
            let improved = out.best < pop[w].1 - 1e-9;
            if let Some(x) = out.best_state {
                pop[w].0 = x;
            }
            if out.best < pop[w].1 {
                pop[w].1 = out.best;
            }
            if out.best < best {
                best = out.best;
            }
            if improved {
                improvements += 1;
                pop[w].2 += 1.0;
            } else {
                pop[w].3 += 1.0;
            }
        }
        // Thompson resampling on the improvement posteriors.
        let draws: Vec<f64> = pop
            .iter()
            .map(|(_, _, a, b)| crate::allocate::beta_draw(*a, *b, &mut rng))
            .collect();
        let (hi, _) = draws
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("population nonempty");
        let (lo, _) = draws
            .iter()
            .enumerate()
            .min_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("population nonempty");
        if hi != lo {
            let clone = pop[hi].clone();
            pop[lo] = clone;
            resamples += 1;
        }
    }
    CommittorOutcome {
        best,
        resamples,
        segments,
        improvements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dead walker has to lose its slot to a productive one, which is the
    /// property Boltzmann weighting does not have.
    #[test]
    fn the_dead_walker_is_culled() {
        let mut rng = StdRng::seed_from_u64(3);
        // Posteriors after six segments: walker 0 improved five times, walker
        // 1 never. Thompson draws must prefer 0 nearly always.
        let mut wins = 0;
        for _ in 0..200 {
            let a = crate::allocate::beta_draw(6.0, 2.0, &mut rng);
            let b = crate::allocate::beta_draw(1.0, 7.0, &mut rng);
            if a > b {
                wins += 1;
            }
        }
        assert!(wins > 180, "productive walker won only {wins}/200");
    }
}
