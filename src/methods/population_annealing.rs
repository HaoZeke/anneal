//! Population annealing on one Lennard-Jones ledger.
//!
//! Hukushima and Iba, Phys. Rev. E **67**, 066106 (2003), with the
//! inverse-temperature step of Machta, Phys. Rev. E **82**, 026704 (2010):
//! the next `β` is the one that puts the effective sample size at half the
//! population. One population is one seed. The caller charges every energy,
//! including the final quenches. An unquenched energy is not the reported
//! best.

use std::collections::HashSet;

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rayon::prelude::*;

use super::cluster_hopping::{self, Ledger};
use crate::scaling::{ScaleDecision, SuccessiveHalving};

/// The reference-script population: 100 walkers, `T` from 0.8 down to 0.05,
/// a Wales–Doye step of 0.38, and 500 evaluations per walker held back for
/// the final quenches.
#[derive(Clone, Debug)]
pub struct PopulationAnneal {
    /// Walkers in one population.
    pub walkers: usize,
    /// Ladder start.
    pub temperature_high: f64,
    /// Ladder floor.
    pub temperature_low: f64,
    /// Half-width of the coordinate displacement.
    pub step: f64,
    /// Evaluations kept per walker for the closing quench.
    pub quench_reserve_per_walker: usize,
    /// Iteration cap of one closing quench.
    pub quench_steps: usize,
    /// Container radius of the initial draw and of a displaced walker.
    pub radius: f64,
    /// Minimum pair separation of the initial draw.
    pub min_separation: f64,
}

impl PopulationAnneal {
    /// The constants of `experiments/reference/population_annealing.py`.
    ///
    /// The container is `2.5 N^{1/3}`, the same sphere that script draws in.
    /// The initial draw is this crate's non-overlapping cluster rather than
    /// a uniform box: two atoms on one site make the pair energy unusable.
    pub fn reference(n: usize) -> Self {
        Self {
            walkers: 100,
            temperature_high: 0.8,
            temperature_low: 0.05,
            step: 0.38,
            quench_reserve_per_walker: 500,
            quench_steps: 3000,
            radius: 2.5 * (n as f64).cbrt(),
            min_separation: 0.9,
        }
    }
}

/// What one population produced.
#[derive(Clone, Debug)]
pub struct PopulationReport {
    /// Deepest quenched energy. Non-finite when no quench finished.
    pub best_energy: f64,
    /// Deepest energy seen before the closing quenches.
    pub unquenched_best: f64,
    /// Metropolis generations completed.
    pub generations: usize,
    /// Temperature after the last accepted step.
    pub final_temperature: f64,
    /// Walkers whose quench returned a finite energy.
    pub quenched: usize,
    /// Coordinates of `best_energy`, empty when none finished.
    pub best_state: Vec<f64>,
}

/// Smallest positive `Δβ` whose effective sample size is at most `target_ess`.
///
/// A flat population has effective size equal to its length at every step,
/// and the search then returns the cap of 50. A spread population is bisected
/// until the size sits on the target.
pub fn choose_dbeta(energies: &[f64], target_ess: f64) -> f64 {
    if energies.is_empty() || !target_ess.is_finite() || target_ess <= 0.0 {
        return 0.0;
    }
    let floor = energies.iter().copied().fold(f64::INFINITY, f64::min);
    if !floor.is_finite() {
        return 0.0;
    }
    let shifted: Vec<f64> = energies.iter().map(|energy| energy - floor).collect();
    let ess = |dbeta: f64| effective_sample_size(&shifted, dbeta);
    if ess(50.0) > target_ess {
        return 50.0;
    }
    let mut low = 0.0;
    let mut high = 50.0;
    for _ in 0..40 {
        let mid = 0.5 * (low + high);
        if ess(mid) > target_ess {
            low = mid;
        } else {
            high = mid;
        }
    }
    high
}

fn effective_sample_size(shifted: &[f64], dbeta: f64) -> f64 {
    let mut total = 0.0;
    let mut weights = Vec::with_capacity(shifted.len());
    for energy in shifted {
        let weight = (-dbeta * energy).exp();
        if !weight.is_finite() {
            return 0.0;
        }
        weights.push(weight);
        total += weight;
    }
    if total <= 0.0 || !total.is_finite() {
        return 0.0;
    }
    let sum_sq: f64 = weights
        .iter()
        .map(|weight| {
            let p = weight / total;
            p * p
        })
        .sum();
    if sum_sq <= 0.0 { 0.0 } else { 1.0 / sum_sq }
}

fn draw_indices<R: Rng + ?Sized>(weights: &[f64], draws: usize, rng: &mut R) -> Vec<usize> {
    let fallback = weights.len().max(1);
    let dist = WeightedIndex::new(weights).ok();
    (0..draws)
        .map(|_| match dist.as_ref() {
            Some(dist) => dist.sample(rng),
            None => rng.random_range(0..fallback),
        })
        .collect()
}

/// One population. `energy_of` and `quench` charge `ledger` themselves and
/// return `None` when the budget is gone.
pub fn population_anneal<R, F, Q>(
    n: usize,
    ledger: &mut Ledger,
    cfg: &PopulationAnneal,
    mut energy_of: F,
    mut quench: Q,
    rng: &mut R,
) -> PopulationReport
where
    R: Rng + ?Sized,
    F: FnMut(&mut Ledger, ArrayView1<f64>) -> Option<f64>,
    Q: FnMut(&mut Ledger, ArrayView1<f64>, usize) -> Option<(f64, Array1<f64>)>,
{
    let alive_target = cfg.walkers.max(1);
    let mut states: Vec<Array1<f64>> = Vec::new();
    let mut energies: Vec<f64> = Vec::new();
    for _ in 0..alive_target {
        let start =
            cluster_hopping::random_cluster_in_radius(n, cfg.radius, cfg.min_separation, rng);
        let Some(energy) = energy_of(ledger, start.view()) else {
            break;
        };
        states.push(start);
        energies.push(energy);
    }
    let mut unquenched_best = energies.iter().copied().fold(f64::INFINITY, f64::min);
    if states.is_empty() {
        return PopulationReport {
            best_energy: f64::INFINITY,
            unquenched_best,
            generations: 0,
            final_temperature: cfg.temperature_high,
            quenched: 0,
            best_state: Vec::new(),
        };
    }
    let mut beta = 1.0 / cfg.temperature_high;
    let mut generations = 0usize;
    let reserve = cfg.quench_reserve_per_walker.saturating_mul(states.len());
    while ledger.spent().saturating_add(states.len()) < ledger.budget().saturating_sub(reserve) {
        let temperature = 1.0 / beta;
        let mut trial_states = Vec::with_capacity(states.len());
        let mut trial_energies = Vec::with_capacity(states.len());
        for (state, energy) in states.iter().zip(energies.iter()) {
            let mut proposal = state.clone();
            for coordinate in proposal.iter_mut() {
                *coordinate += rng.random_range(-cfg.step..cfg.step);
            }
            cluster_hopping::contain(&mut proposal, n, cfg.radius);
            let Some(proposed) = energy_of(ledger, proposal.view()) else {
                trial_states.push(state.clone());
                trial_energies.push(*energy);
                continue;
            };
            let accept = proposed <= *energy
                || rng.random::<f64>() < (-(proposed - *energy) / temperature).exp();
            if accept {
                trial_states.push(proposal);
                trial_energies.push(proposed);
            } else {
                trial_states.push(state.clone());
                trial_energies.push(*energy);
            }
        }
        states = trial_states;
        energies = trial_energies;
        unquenched_best =
            unquenched_best.min(energies.iter().copied().fold(f64::INFINITY, f64::min));
        if temperature > cfg.temperature_low {
            let dbeta = choose_dbeta(&energies, states.len() as f64 / 2.0);
            let previous = beta;
            beta = (beta + dbeta).min(1.0 / cfg.temperature_low);
            let floor = energies.iter().copied().fold(f64::INFINITY, f64::min);
            let weights: Vec<f64> = energies
                .iter()
                .map(|energy| (-(beta - previous) * (energy - floor)).exp())
                .collect();
            let index = draw_indices(&weights, states.len(), rng);
            let next_states: Vec<Array1<f64>> = index.iter().map(|i| states[*i].clone()).collect();
            let next_energies: Vec<f64> = index.iter().map(|i| energies[*i]).collect();
            states = next_states;
            energies = next_energies;
        }
        generations += 1;
    }
    let mut best_energy = f64::INFINITY;
    let mut best_state = Vec::new();
    let mut quenched = 0usize;
    for state in &states {
        let Some((energy, coordinates)) = quench(ledger, state.view(), cfg.quench_steps) else {
            break;
        };
        if !energy.is_finite() {
            continue;
        }
        quenched += 1;
        if energy < best_energy {
            best_energy = energy;
            best_state = coordinates.to_vec();
        }
    }
    PopulationReport {
        best_energy,
        unquenched_best,
        generations,
        final_temperature: 1.0 / beta,
        quenched,
        best_state,
    }
}

/// Fixed-work record from a strong scaling schedule.
///
/// The atom count, walker count, and generation count are the problem.
/// They do not change with the thread count.
#[derive(Clone, Debug)]
pub struct ThreadedReport {
    /// Deepest finite energy seen. Quenched when a quench ran.
    pub best_energy: f64,
    /// Energy evaluations, including the closing quenches.
    pub evaluations: usize,
    /// Metropolis or control rounds completed.
    pub generations: usize,
    /// Walkers whose quench returned a finite energy.
    pub quenched: usize,
    /// Successive-halving decisions emitted. Zero for the Metropolis schedule.
    pub decisions: usize,
}

/// Strong scaling schedule for one fixed population.
///
/// `generations` Metropolis sweeps run, then every walker is quenched for
/// `cfg.quench_steps` iterations. `threads` is the pool that evaluates one
/// sweep and the closing quenches. Proposal noise and one accept variate
/// are drawn on the calling thread before that sweep's energies, so the
/// trajectory does not depend on `threads`. The accept variate is drawn
/// for every walker. [`population_anneal`] draws it only when the trial is
/// worse, and it stops on a ledger budget rather than a generation count.
#[allow(clippy::too_many_arguments)]
pub fn population_anneal_threads<R, E, Q>(
    n: usize,
    cfg: &PopulationAnneal,
    generations: usize,
    threads: usize,
    energy_of: &E,
    quench: &Q,
    rng: &mut R,
) -> ThreadedReport
where
    R: Rng + ?Sized,
    E: Sync + Fn(ArrayView1<f64>) -> f64,
    Q: Sync + Fn(ArrayView1<f64>) -> Option<(f64, Array1<f64>, usize)>,
{
    let threads = threads.max(1);
    let alive_target = cfg.walkers.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    let mut states: Vec<Array1<f64>> = Vec::with_capacity(alive_target);
    let mut energies: Vec<f64> = Vec::with_capacity(alive_target);
    for _ in 0..alive_target {
        let start =
            cluster_hopping::random_cluster_in_radius(n, cfg.radius, cfg.min_separation, rng);
        let energy = energy_of(start.view());
        states.push(start);
        energies.push(energy);
    }
    let mut evaluations = states.len();
    let mut unquenched_best = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let mut beta = 1.0 / cfg.temperature_high;
    for _ in 0..generations {
        let temperature = 1.0 / beta;
        let mut proposals = Vec::with_capacity(states.len());
        let mut accept_u = Vec::with_capacity(states.len());
        for state in &states {
            let mut proposal = state.clone();
            for coordinate in proposal.iter_mut() {
                *coordinate += rng.random_range(-cfg.step..cfg.step);
            }
            cluster_hopping::contain(&mut proposal, n, cfg.radius);
            proposals.push(proposal);
            accept_u.push(rng.random::<f64>());
        }
        let proposed: Vec<f64> = pool.install(|| {
            proposals
                .par_iter()
                .map(|proposal| energy_of(proposal.view()))
                .collect()
        });
        evaluations += proposed.len();
        let mut trial_states = Vec::with_capacity(states.len());
        let mut trial_energies = Vec::with_capacity(states.len());
        for (index, (state, energy)) in states.iter().zip(energies.iter()).enumerate() {
            let accept = proposed[index] <= *energy
                || accept_u[index] < (-(proposed[index] - *energy) / temperature).exp();
            if accept {
                trial_states.push(proposals[index].clone());
                trial_energies.push(proposed[index]);
            } else {
                trial_states.push(state.clone());
                trial_energies.push(*energy);
            }
        }
        states = trial_states;
        energies = trial_energies;
        unquenched_best =
            unquenched_best.min(energies.iter().copied().fold(f64::INFINITY, f64::min));
        if temperature > cfg.temperature_low {
            let dbeta = choose_dbeta(&energies, states.len() as f64 / 2.0);
            let previous = beta;
            beta = (beta + dbeta).min(1.0 / cfg.temperature_low);
            let floor = energies.iter().copied().fold(f64::INFINITY, f64::min);
            let weights: Vec<f64> = energies
                .iter()
                .map(|energy| (-(beta - previous) * (energy - floor)).exp())
                .collect();
            let index = draw_indices(&weights, states.len(), rng);
            let next_states: Vec<Array1<f64>> = index.iter().map(|i| states[*i].clone()).collect();
            let next_energies: Vec<f64> = index.iter().map(|i| energies[*i]).collect();
            states = next_states;
            energies = next_energies;
        }
    }
    let quenched_rows: Vec<Option<(f64, Array1<f64>, usize)>> = pool.install(|| {
        states
            .par_iter()
            .map(|state| quench(state.view()))
            .collect()
    });
    let mut best_energy = f64::INFINITY;
    let mut quenched = 0usize;
    for row in quenched_rows {
        let Some((energy, _coordinates, calls)) = row else {
            continue;
        };
        evaluations += calls;
        if !energy.is_finite() {
            continue;
        }
        quenched += 1;
        if energy < best_energy {
            best_energy = energy;
        }
    }
    if !best_energy.is_finite() {
        best_energy = unquenched_best;
    }
    ThreadedReport {
        best_energy,
        evaluations,
        generations,
        quenched,
        decisions: 0,
    }
}

/// Strong scaling schedule for successive-halving control of a fixed roster.
///
/// Each round evaluates every live walker, then [`SuccessiveHalving`]
/// retires the worst fraction and the same number of new walkers is drawn.
/// The roster is returned to `cfg.walkers` before the next round, so the
/// evaluation count does not grow with `threads`. The reduction is two and
/// the first rung is one charged round.
pub fn population_halving_threads<R, E>(
    n: usize,
    cfg: &PopulationAnneal,
    generations: usize,
    threads: usize,
    energy_of: &E,
    rng: &mut R,
) -> ThreadedReport
where
    R: Rng + ?Sized,
    E: Sync + Fn(ArrayView1<f64>) -> f64,
{
    let threads = threads.max(1);
    let walkers = cfg.walkers.max(1);
    let target = u32::try_from(walkers).expect("walker count fits");
    let mut policy = SuccessiveHalving::new(1, 2.0, target).expect("halving parameters");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    struct Slot {
        id: u32,
        age: u64,
        state: Array1<f64>,
    }
    let mut next_id = target;
    let mut slots = Vec::with_capacity(walkers);
    for id in 0..target {
        slots.push(Slot {
            id,
            age: 0,
            state: cluster_hopping::random_cluster_in_radius(
                n,
                cfg.radius,
                cfg.min_separation,
                rng,
            ),
        });
    }
    let mut evaluations = 0usize;
    let mut decisions = 0usize;
    let mut best_energy = f64::INFINITY;
    for _ in 0..generations {
        let energies: Vec<f64> = pool.install(|| {
            slots
                .par_iter()
                .map(|slot| energy_of(slot.state.view()))
                .collect()
        });
        evaluations += energies.len();
        let mut retired = HashSet::new();
        let mut spawn = 0u32;
        for (slot, energy) in slots.iter().zip(energies.iter()) {
            if energy.is_finite() {
                best_energy = best_energy.min(*energy);
            }
            let charged = slot.age.saturating_add(1);
            for decision in policy.observe(slot.id, charged, *energy) {
                decisions += 1;
                match decision {
                    ScaleDecision::Retire(id) => {
                        retired.insert(id);
                    }
                    ScaleDecision::Spawn(count) => {
                        spawn = spawn.saturating_add(count);
                    }
                }
            }
        }
        for slot in &mut slots {
            slot.age = slot.age.saturating_add(1);
        }
        slots.retain(|slot| !retired.contains(&slot.id));
        for _ in 0..spawn {
            slots.push(Slot {
                id: next_id,
                age: 0,
                state: cluster_hopping::random_cluster_in_radius(
                    n,
                    cfg.radius,
                    cfg.min_separation,
                    rng,
                ),
            });
            next_id = next_id.saturating_add(1);
        }
    }
    ThreadedReport {
        best_energy,
        evaluations,
        generations,
        quenched: 0,
        decisions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::warm_lbfgs::WarmLbfgs;
    use crate::potentials::PairPotential;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn a_flat_population_takes_the_largest_temperature_step() {
        let energies = vec![-1.0; 8];
        assert_eq!(choose_dbeta(&energies, 4.0), 50.0);
        assert_eq!(effective_sample_size(&[0.0; 8], 10.0), 8.0);
    }

    #[test]
    fn a_spread_population_stops_at_half_its_size() {
        let energies = vec![0.0, 0.0, 4.0, 4.0];
        let step = choose_dbeta(&energies, 2.0);
        assert!(step > 0.0 && step < 50.0, "{step}");
        let shifted = vec![0.0, 0.0, 4.0, 4.0];
        let size = effective_sample_size(&shifted, step);
        assert!(
            (size - 2.0).abs() < 0.05,
            "effective size {size} at step {step}"
        );
    }

    #[test]
    fn a_quenched_population_lands_below_its_unquenched_best() {
        let n = 7usize;
        let mut cfg = PopulationAnneal::reference(n);
        cfg.walkers = 4;
        cfg.quench_reserve_per_walker = 40;
        cfg.quench_steps = 80;
        let potential = PairPotential::lennard_jones(n);
        let mut ledger = Ledger::new(800);
        let mut optimiser = WarmLbfgs::default();
        let mut rng = StdRng::seed_from_u64(2000);
        let report = population_anneal(
            n,
            &mut ledger,
            &cfg,
            |led, x| {
                if !led.charge() {
                    return None;
                }
                Some(potential.value_and_gradient(x).0)
            },
            |led, x, steps| {
                optimiser.forget();
                let (value, relaxed, _) = optimiser.minimize(x, steps, |v| {
                    if !led.charge() {
                        return None;
                    }
                    Some(potential.value_and_gradient(v))
                });
                value.is_finite().then_some((value, relaxed))
            },
            &mut rng,
        );
        assert!(report.generations > 0, "generations {}", report.generations);
        assert!(report.quenched > 0, "quenched {}", report.quenched);
        assert!(report.best_energy.is_finite());
        assert!(
            report.best_energy < report.unquenched_best,
            "quenched {} against unquenched {}",
            report.best_energy,
            report.unquenched_best
        );
        assert_eq!(report.best_state.len(), 3 * n);
        assert!(report.final_temperature <= cfg.temperature_high + 1e-12);
        assert!(ledger.spent() > 0);
    }

    fn tiny_cfg(n: usize) -> PopulationAnneal {
        let mut cfg = PopulationAnneal::reference(n);
        cfg.walkers = 4;
        cfg.quench_steps = 4;
        cfg
    }

    #[test]
    fn threaded_population_matches_across_thread_counts() {
        let n = 7usize;
        let cfg = tiny_cfg(n);
        let once = |threads: usize| {
            let pot = PairPotential::lennard_jones(n);
            let mut rng = StdRng::seed_from_u64(3);
            population_anneal_threads(
                n,
                &cfg,
                2,
                threads,
                &|x| pot.value_and_gradient(x).0,
                &|x| {
                    let mut optimiser = WarmLbfgs::default();
                    let (value, relaxed, calls) = optimiser
                        .minimize(x, cfg.quench_steps, |v| Some(pot.value_and_gradient(v)));
                    value.is_finite().then_some((value, relaxed, calls))
                },
                &mut rng,
            )
        };
        let one = once(1);
        let three = once(3);
        assert_eq!(one.best_energy.to_bits(), three.best_energy.to_bits());
        assert_eq!(one.evaluations, three.evaluations);
        assert_eq!(one.generations, three.generations);
        assert_eq!(one.quenched, three.quenched);
        assert!(one.quenched > 0);
        assert!(one.evaluations > 0);
    }

    #[test]
    fn threaded_halving_matches_across_thread_counts() {
        let n = 7usize;
        let cfg = tiny_cfg(n);
        let once = |threads: usize| {
            let pot = PairPotential::lennard_jones(n);
            let mut rng = StdRng::seed_from_u64(5);
            population_halving_threads(
                n,
                &cfg,
                3,
                threads,
                &|x| pot.value_and_gradient(x).0,
                &mut rng,
            )
        };
        let one = once(1);
        let four = once(4);
        assert_eq!(one.best_energy.to_bits(), four.best_energy.to_bits());
        assert_eq!(one.evaluations, four.evaluations);
        assert_eq!(one.decisions, four.decisions);
        assert_eq!(one.evaluations, cfg.walkers * 3);
        assert!(one.decisions > 0, "decisions {}", one.decisions);
    }
}
