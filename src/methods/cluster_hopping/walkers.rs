// Copyright (c) anneal developers
//! Hyperband walker allocation over one charged ledger.
//!
//! Successive halving (Karnin, Koren and Somekh, ICML 2013) inside
//! Hyperband brackets (Li et al., JMLR 18, 2018). A bracket opens many
//! short chains and keeps the best fraction at each rung; the brackets
//! range from many short chains to one long one, so no walker count is
//! chosen. Survivors resume from their live state with their own bias, so
//! a slice continues a chain rather than restarting it. The opening probe
//! and every segment are charged to the caller's ledger.

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::fingerprint::{l2, points_of, radial_spectrum};
use super::{
    BasinBias, BiasCarry, ClusterFingerprint, Config, EnergyGradFn, GradFn, HopStop, Ledger,
    Outcome, Relax, SearchError, WalkerSchedule, continue_without_checkpoint,
    random_cluster_in_radius, random_rigid_cluster, run_full,
};

/// The smallest slice, in opening quenches.
///
/// A chain shorter than this has not left its first funnel and its best
/// energy says nothing about where it is going.
const MIN_SLICE_QUENCHES: usize = 100;

/// Brackets are capped so the largest opens at most `eta^MAX_BRACKET` chains.
const MAX_BRACKET: usize = 8;

/// Two walkers are on one minimum when their best energies agree to this
/// fraction and their radial spectra to [`SAME_SPECTRUM`] per point.
const SAME_ENERGY: f64 = 1e-7;
const SAME_SPECTRUM: f64 = 1e-3;

struct Walker {
    state: Array1<f64>,
    bias: Option<BasinBias<ClusterFingerprint>>,
    rng: StdRng,
    best: f64,
    best_state: Option<Array1<f64>>,
    gain: f64,
    alive: bool,
}

/// Largest bracket whose first slice is still [`MIN_SLICE_QUENCHES`] probes.
fn bracket_count(budget: usize, probe: usize, eta: usize) -> usize {
    let floor = MIN_SLICE_QUENCHES.saturating_mul(probe.max(1));
    let mut s_max = 0;
    for s in 1..=MAX_BRACKET {
        let per_bracket = budget / (s + 1);
        let first = per_bracket / ((s + 1) * eta.pow(s as u32));
        if first >= floor {
            s_max = s;
        } else {
            break;
        }
    }
    s_max
}

/// Walkers kept per rung of bracket `s`.
fn rung_sizes(s: usize, eta: usize) -> Vec<usize> {
    (0..=s)
        .map(|i| (eta.pow(s as u32) / eta.pow(i as u32)).max(1))
        .collect()
}

fn fresh_start<R: Rng + ?Sized>(cfg: &Config, start: ArrayView1<f64>, rng: &mut R) -> Array1<f64> {
    if cfg.move_library.is_molecular() {
        start.to_owned()
    } else if cfg.move_library.is_rigid_body() {
        random_rigid_cluster(cfg.n_points, cfg.min_separation, rng)
    } else {
        random_cluster_in_radius(cfg.n_points, cfg.start_radius(), cfg.min_separation, rng)
    }
}

fn spectrum(cfg: &Config, state: ArrayView1<f64>) -> Option<Vec<f64>> {
    let rigid = cfg.move_library.is_rigid_body().then_some(cfg.n_points);
    radial_spectrum(points_of(state, rigid).view())
}

fn same_minimum(cfg: &Config, a: &Walker, b: &Walker) -> bool {
    let scale = a.best.abs().max(b.best.abs()).max(1e-12);
    if (a.best - b.best).abs() > SAME_ENERGY * scale {
        return false;
    }
    match (&a.best_state, &b.best_state) {
        (Some(x), Some(y)) => match (spectrum(cfg, x.view()), spectrum(cfg, y.view())) {
            (Some(p), Some(q)) => l2(&p, &q) / (p.len().max(1) as f64).sqrt() < SAME_SPECTRUM,
            _ => true,
        },
        _ => true,
    }
}

/// Drop in the segment's best over its second half.
///
/// The improvement rate a survivor is credited with. The first half is
/// the descent from wherever the slice began, which every chain makes.
fn recent_gain(outcome: &Outcome, spent: usize) -> f64 {
    let half = spent / 2;
    let at_half = outcome
        .improvements
        .iter()
        .filter(|&&(_, at, _, _)| at <= half)
        .map(|&(_, _, _, energy)| energy)
        .fold(f64::INFINITY, f64::min);
    if at_half.is_finite() && outcome.best.is_finite() {
        (at_half - outcome.best).max(0.0)
    } else {
        0.0
    }
}

/// Indices of the `keep` survivors: lowest best energy less the last rung's
/// gain, one walker per minimum while distinct ones remain.
fn survivors(cfg: &Config, walkers: &[Walker], keep: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..walkers.len()).filter(|&i| walkers[i].alive).collect();
    order.sort_by(|&a, &b| {
        let score = |w: &Walker| w.best - w.gain;
        score(&walkers[a])
            .total_cmp(&score(&walkers[b]))
            .then(a.cmp(&b))
    });
    let mut kept: Vec<usize> = Vec::with_capacity(keep);
    let mut held: Vec<usize> = Vec::new();
    for &index in &order {
        if kept.len() == keep {
            break;
        }
        if kept
            .iter()
            .any(|&other| same_minimum(cfg, &walkers[index], &walkers[other]))
        {
            held.push(index);
        } else {
            kept.push(index);
        }
    }
    for index in held {
        if kept.len() == keep {
            break;
        }
        kept.push(index);
    }
    kept
}

#[allow(clippy::too_many_arguments)]
pub(super) fn hyperband<'g, R: Rng + ?Sized>(
    cfg: &Config,
    eta: usize,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut grad: Option<&mut GradFn<'g>>,
    mut energy_grad: Option<&mut EnergyGradFn<'g>>,
    rng: &mut R,
) -> Result<Outcome, SearchError> {
    let eta = eta.max(2);
    let mut single = cfg.clone();
    single.walkers = WalkerSchedule::Single;
    let mut improvements: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut hops = 0usize;
    let mut basins = 0usize;
    let mut final_state: Option<(f64, Array1<f64>)> = None;

    let before = ledger.spent();
    let (probe_energy, probed) = relax(ledger, start, cfg.relax_steps);
    let probe = ledger.spent().saturating_sub(before).max(1);
    let opening = if probe_energy.is_finite() && probed.len() == start.len() {
        probed
    } else {
        start.to_owned()
    };
    let s_max = bracket_count(ledger.remaining(), probe, eta);
    let mut first_start = Some(opening);

    let closed =
        |ledger: &Ledger| ledger.stopped() || ledger.target_reached() || ledger.remaining() == 0;
    'brackets: for s in (0..=s_max).rev() {
        if closed(ledger) {
            break;
        }
        let brackets_left = s + 1;
        let mut bracket_budget = ledger.remaining() / brackets_left;
        let sizes = rung_sizes(s, eta);
        let mut walkers: Vec<Walker> = (0..sizes[0])
            .map(|_| {
                let state = first_start
                    .take()
                    .unwrap_or_else(|| fresh_start(cfg, start, rng));
                Walker {
                    state,
                    bias: None,
                    rng: StdRng::seed_from_u64(rng.random::<u64>()),
                    best: f64::INFINITY,
                    best_state: None,
                    gain: 0.0,
                    alive: true,
                }
            })
            .collect();
        let mut active: Vec<usize> = (0..walkers.len()).collect();
        for (rung, &size) in sizes.iter().enumerate() {
            if rung > 0 {
                active = survivors(&single, &walkers, size);
                if active.is_empty() {
                    break;
                }
            }
            let rungs_left = sizes.len() - rung;
            let slice = (bracket_budget / (rungs_left * active.len())).max(1);
            for &index in &active {
                if closed(ledger) {
                    break 'brackets;
                }
                let walker = &mut walkers[index];
                let mut segment = ledger.slice(slice.min(ledger.remaining()));
                if let Some(target) = ledger.target() {
                    segment.stop_below(target);
                }
                let mut checkpoint = continue_without_checkpoint;
                let result = run_full(
                    &single,
                    walker.state.view(),
                    &mut segment,
                    &mut *relax,
                    grad.as_deref_mut(),
                    energy_grad.as_deref_mut(),
                    BiasCarry::Fill(&mut walker.bias),
                    None,
                    None,
                    None,
                    &mut checkpoint,
                    &mut walker.rng,
                );
                let offset = ledger.spent();
                let spent = segment.spent();
                ledger.settle_slice(spent);
                bracket_budget = bracket_budget.saturating_sub(spent);
                let outcome = match result {
                    Ok(outcome) => outcome,
                    Err(_) => {
                        walker.alive = false;
                        continue;
                    }
                };
                for &(hop, at, seen, energy) in &outcome.improvements {
                    if energy < ledger.best && improvements.len() < 512 {
                        improvements.push((hops + hop, offset + at, basins + seen, energy));
                    }
                }
                if let Some(state) = outcome.best_state.as_ref()
                    && outcome.best < ledger.best
                {
                    ledger.record(outcome.best, state.view());
                }
                hops += outcome.hops;
                basins += outcome.basins;
                walker.gain = recent_gain(&outcome, spent);
                if outcome.best < walker.best {
                    walker.best = outcome.best;
                    walker.best_state = outcome.best_state.clone();
                }
                if let Some(state) = outcome.final_state {
                    if state.len() == walker.state.len() {
                        walker.state = state;
                    }
                }
                if final_state
                    .as_ref()
                    .is_none_or(|(energy, _)| outcome.final_energy < *energy)
                {
                    final_state = Some((outcome.final_energy, walker.state.clone()));
                }
            }
        }
    }

    if !ledger.best.is_finite() && !probe_energy.is_finite() {
        return Err(SearchError::NoFiniteStart { attempts: 1 });
    }
    let (final_energy, final_state) = match final_state {
        Some((energy, state)) => (energy, Some(state)),
        None => (probe_energy, Some(start.to_owned())),
    };
    Ok(Outcome {
        best: ledger.best,
        best_state: ledger.best_state.clone(),
        final_state,
        final_energy,
        hops,
        basins,
        charged: ledger.spent(),
        improvements,
        stop: Some(HopStop::of_closed(ledger)),
        ..Outcome::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_brackets_shrink_with_the_budget() {
        assert_eq!(bracket_count(1_000, 100, 3), 0);
        let large = bracket_count(4_000_000, 300, 3);
        let small = bracket_count(400_000, 300, 3);
        assert!(large > small, "{large} {small}");
    }

    #[test]
    fn every_rung_keeps_a_third_and_the_last_one_walker() {
        assert_eq!(rung_sizes(3, 3), vec![27, 9, 3, 1]);
        assert_eq!(rung_sizes(0, 3), vec![1]);
    }

    #[test]
    fn the_first_slice_of_the_largest_bracket_clears_the_floor() {
        let (budget, probe, eta) = (4_000_000, 250, 3);
        let s = bracket_count(budget, probe, eta);
        let first = budget / (s + 1) / ((s + 1) * eta.pow(s as u32));
        assert!(first >= MIN_SLICE_QUENCHES * probe);
    }
}
