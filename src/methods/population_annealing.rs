//! Population annealing on one Lennard-Jones ledger.
//!
//! Hukushima and Iba, Phys. Rev. E **67**, 066106 (2003), with the
//! inverse-temperature step of Machta, Phys. Rev. E **82**, 026704 (2010):
//! the next `β` is the one that puts the effective sample size at half the
//! population. One population is one seed. The caller charges every energy,
//! including the final quenches. An unquenched energy is not the reported
//! best.

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rayon::prelude::*;

use super::cluster_hopping::{self, Ledger};

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

/// Counts from one synchronous population-annealing generation schedule.
///
/// The schedule draws every proposal and every accept variate on the
/// calling thread, and evaluates the proposals together. The reference
/// [`population_anneal`] loop evaluates inside the same walk that draws
/// the next proposal. This schedule keeps those draws in two blocks so
/// the parallel evaluations do not change the generator stream.
#[derive(Clone, Debug)]
pub struct SyncPopulationReport {
    /// Deepest energy among the initial draw and the accepted proposals.
    pub best_energy: f64,
    /// Generations completed.
    pub generations: usize,
    /// Temperature after the last generation.
    pub final_temperature: f64,
    /// Energy evaluations charged. One per walker at the start, plus
    /// `steps_per_generation` per walker per generation.
    pub charged: usize,
    /// Walkers in the population.
    pub walkers: usize,
}

/// Synchronous population annealing on a fixed generation count.
///
/// `threads` is the pool width. `walkers` is the population. Each
/// generation gives every walker `steps_per_generation` Metropolis steps,
/// then one half-sample resampling step, the same temperature rule as
/// [`population_anneal`]. One thread walks the population in order. More
/// threads run one generation's walker steps together. Walker generators
/// are drawn on the calling thread, so a wider pool visits the same states.
#[allow(clippy::too_many_arguments)]
pub fn population_anneal_sync<E>(
    n: usize,
    walkers: usize,
    threads: usize,
    generations: usize,
    steps_per_generation: usize,
    step: f64,
    temperature_high: f64,
    temperature_low: f64,
    radius: f64,
    min_separation: f64,
    energy: E,
    seed: u64,
) -> SyncPopulationReport
where
    E: Fn(ArrayView1<f64>) -> f64 + Sync,
{
    let threads = threads.max(1);
    let walkers = walkers.max(1);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut states = Vec::with_capacity(walkers);
    for _ in 0..walkers {
        states.push(cluster_hopping::random_cluster_in_radius(
            n,
            radius,
            min_separation,
            &mut rng,
        ));
    }
    assert!(
        states.iter().all(|state| state.len() == 3 * n),
        "cluster draw returned a short configuration"
    );
    let pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool"),
        )
    } else {
        None
    };
    let mut energies = evaluate_all(&pool, &states, &energy);
    let mut charged = energies.len();
    let mut best_energy = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let mut beta = 1.0 / temperature_high;
    let steps_per_generation = steps_per_generation.max(1);
    for _ in 0..generations {
        let temperature = 1.0 / beta;
        let seeds: Vec<u64> = (0..states.len()).map(|_| rng.random()).collect();
        let updated = advance_walkers(
            &pool,
            &states,
            &energies,
            &seeds,
            steps_per_generation,
            temperature,
            step,
            n,
            radius,
            &energy,
        );
        charged += states.len() * steps_per_generation;
        for (index, (state, value)) in updated.into_iter().enumerate() {
            states[index] = state;
            energies[index] = value;
            if value < best_energy {
                best_energy = value;
            }
        }
        if temperature > temperature_low {
            let dbeta = choose_dbeta(&energies, states.len() as f64 / 2.0);
            let previous = beta;
            beta = (beta + dbeta).min(1.0 / temperature_low);
            let floor = energies.iter().copied().fold(f64::INFINITY, f64::min);
            let weights: Vec<f64> = energies
                .iter()
                .map(|value| (-(beta - previous) * (value - floor)).exp())
                .collect();
            let index = draw_indices(&weights, states.len(), &mut rng);
            let next_states: Vec<Array1<f64>> = index.iter().map(|i| states[*i].clone()).collect();
            let next_energies: Vec<f64> = index.iter().map(|i| energies[*i]).collect();
            states = next_states;
            energies = next_energies;
        }
    }
    SyncPopulationReport {
        best_energy,
        generations,
        final_temperature: 1.0 / beta,
        charged,
        walkers,
    }
}

fn advance_walkers<E>(
    pool: &Option<rayon::ThreadPool>,
    states: &[Array1<f64>],
    energies: &[f64],
    seeds: &[u64],
    steps: usize,
    temperature: f64,
    step: f64,
    n: usize,
    radius: f64,
    energy: &E,
) -> Vec<(Array1<f64>, f64)>
where
    E: Fn(ArrayView1<f64>) -> f64 + Sync,
{
    let one = |index: usize| -> (Array1<f64>, f64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seeds[index]);
        let mut state = states[index].clone();
        let mut value = energies[index];
        for _ in 0..steps {
            let mut proposal = state.clone();
            for coordinate in proposal.iter_mut() {
                *coordinate += rng.random_range(-step..step);
            }
            cluster_hopping::contain(&mut proposal, n, radius);
            let proposed = energy(proposal.view());
            let draw: f64 = rng.random();
            let accept = proposed <= value || draw < (-(proposed - value) / temperature).exp();
            if accept {
                state = proposal;
                value = proposed;
            }
        }
        (state, value)
    };
    if let Some(pool) = pool.as_ref() {
        pool.install(|| (0..states.len()).into_par_iter().map(one).collect())
    } else {
        (0..states.len()).map(one).collect()
    }
}

fn evaluate_all<E>(pool: &Option<rayon::ThreadPool>, states: &[Array1<f64>], energy: &E) -> Vec<f64>
where
    E: Fn(ArrayView1<f64>) -> f64 + Sync,
{
    if let Some(pool) = pool.as_ref() {
        pool.install(|| {
            states
                .par_iter()
                .map(|state| energy(state.view()))
                .collect()
        })
    } else {
        states.iter().map(|state| energy(state.view())).collect()
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

    #[test]
    fn synchronous_workers_keep_the_serial_population() {
        let n = 5usize;
        let potential = PairPotential::lennard_jones(n);
        let reference = PopulationAnneal::reference(n);
        let once = |threads: usize| {
            population_anneal_sync(
                n,
                4,
                threads,
                3,
                2,
                reference.step,
                reference.temperature_high,
                reference.temperature_low,
                reference.radius,
                reference.min_separation,
                |x| potential.value_and_gradient(x).0,
                11,
            )
        };
        let serial = once(1);
        let threaded = once(2);
        assert_eq!(serial.charged, threaded.charged);
        assert_eq!(serial.walkers, threaded.walkers);
        assert_eq!(serial.generations, threaded.generations);
        assert_eq!(serial.best_energy.to_bits(), threaded.best_energy.to_bits());
        assert_eq!(
            serial.final_temperature.to_bits(),
            threaded.final_temperature.to_bits()
        );
        assert_eq!(
            serial.charged,
            serial.walkers * (1 + serial.generations * 2)
        );
    }
}
