//! Inverted Gelman--Rubin for cooperative search.
//!
//! MCMC wants \(\hat R \to 1\): chains have mixed onto one stationary
//! distribution. Cooperative search splits that diagnostic:
//!
//! - explore-role chains must stay unmixed (\(\hat R\) large);
//! - a putative global minimum is certified only when occupant chains
//!   have mixed onto it *and* it is uniquely deepest *and* it is
//!   strictly more occupied than every competing basin.
//!
//! Mixing onto a lone icosahedral floor is a false certificate.

use std::cmp::Ordering;

/// Gelman--Rubin \(\hat R\) on one scalar series per chain.
///
/// Constant traces at the same value are mixed (\(\hat R = 0\)).
/// Constant traces at different values have \(W = 0\) and \(B > 0\),
/// which is unmixed (\(\hat R = \infty\)). The MCMC
/// [`crate::methods::GelmanRubin`] skip on \(W = 0\) would call that
/// mixed; inverted search cannot.
pub fn rhat_series(chains: &[Vec<f64>]) -> f64 {
    let usable: Vec<&[f64]> = chains
        .iter()
        .map(Vec::as_slice)
        .filter(|chain| chain.len() >= 2)
        .collect();
    if usable.len() < 2 {
        return f64::INFINITY;
    }
    let n = usable.iter().map(|chain| chain.len()).min().unwrap_or(0);
    if n < 2 {
        return f64::INFINITY;
    }
    let windows: Vec<&[f64]> = usable
        .iter()
        .map(|chain| &chain[chain.len() - n..])
        .collect();
    let m = windows.len() as f64;
    let means: Vec<f64> = windows
        .iter()
        .map(|chain| chain.iter().sum::<f64>() / n as f64)
        .collect();
    let vars: Vec<f64> = windows
        .iter()
        .zip(&means)
        .map(|(chain, mean)| {
            chain.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
        })
        .collect();
    let theta_bar: f64 = means.iter().sum::<f64>() / m;
    let b: f64 = (n as f64 / (m - 1.0))
        * means
            .iter()
            .map(|mean| (mean - theta_bar).powi(2))
            .sum::<f64>();
    let w: f64 = vars.iter().sum::<f64>() / m;
    // Repeated identical f64 values can leave a residual W after
    // (n * x) / n, so treat a vanished within-chain variance as zero.
    if w <= 1e-18 {
        return if b > 1e-18 {
            f64::INFINITY
        } else {
            0.0
        };
    }
    let var_hat = ((n as f64 - 1.0) / n as f64) * w + b / n as f64;
    (var_hat / w).sqrt()
}

/// MCMC mixing threshold. Below this, occupant chains have collapsed
/// onto one attractor. For explore that is failure; for an incumbent
/// that is certification only if the attractor wins the occupancy
/// contest against at least one competitor.
pub const MIXED_RHAT: f64 = 1.2;

/// Whether \(\hat R\) says the chains have mixed.
pub fn mixed(rhat: f64) -> bool {
    rhat.is_finite() && rhat < MIXED_RHAT
}

/// Occupancy and occupant mixing of one packing family or census basin.
#[derive(Debug, Clone, PartialEq)]
pub struct AttractorStrength {
    /// Lowest energy observed among occupants.
    pub energy: f64,
    /// Number of independent chains sitting on this attractor.
    pub occupancy: usize,
    /// Gelman--Rubin \(\hat R\) of occupant energy series.
    pub occupant_rhat: f64,
}

impl AttractorStrength {
    /// Occupant chains have mixed onto this attractor.
    pub fn mixed(&self) -> bool {
        mixed(self.occupant_rhat)
    }
}

/// Strict attraction order: more occupants win; equal occupancy
/// breaks toward the tighter occupant mix.
pub fn stronger(left: &AttractorStrength, right: &AttractorStrength) -> bool {
    match left.occupancy.cmp(&right.occupancy) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => {
            left.occupant_rhat.is_finite()
                && right.occupant_rhat.is_finite()
                && left.occupant_rhat < right.occupant_rhat
        }
    }
}

/// A putative global minimum is certified only when occupant chains
/// have mixed onto it, it is uniquely deepest, at least one competing
/// basin has also mixed (it is an attractor, not a flyby), and the
/// putative is strictly more occupied than every competitor. Mixing
/// onto a lone floor, or beating a single unmixed walk, is not a
/// certificate.
pub fn certified_global_minimum(
    putative: &AttractorStrength,
    competitors: &[AttractorStrength],
    uniquely_deepest: bool,
) -> bool {
    uniquely_deepest
        && putative.mixed()
        && competitors.iter().any(AttractorStrength::mixed)
        && competitors
            .iter()
            .all(|other| stronger(putative, other))
}

/// Explore-role chains have collapsed onto one attractor.
pub fn explore_collapsed(explore_series: &[Vec<f64>]) -> bool {
    mixed(rhat_series(explore_series))
}

/// Explore-role failure: mixed energy, or mixed leftover-SOAP / DECAF
/// packing labels among two or more assigned walks.
///
/// One occupied packing with several occupants is a collapse. Waiting
/// for a second family never opens that family. Distinct packing
/// labels stay unmixed and do not force Leave.
pub fn explore_must_leave(
    energy_explore: &[Vec<f64>],
    family_series: &[Vec<f64>],
    n_families: usize,
    n_assigned: usize,
) -> bool {
    let _ = n_families;
    explore_collapsed(energy_explore)
        || (n_assigned >= 2 && mixed(rhat_series(family_series)))
}

/// Inverted Gelman--Rubin evidence consumed by the catalog policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MixingEvidence {
    /// Explore-role (or lone-floor) series have mixed.
    pub explore_collapsed: bool,
    /// Incumbent attractor is uniquely deepest, occupant-mixed, and
    /// strictly more occupied than every competitor.
    pub certified_attractor: bool,
    /// Asynchronous successive halving discarded this walk at a rung.
    pub pruned: bool,
}

/// Combine attractor strengths and the explore-role series.
pub fn invert_mixing(
    attractors: &[AttractorStrength],
    explore_series: &[Vec<f64>],
) -> MixingEvidence {
    MixingEvidence {
        explore_collapsed: explore_collapsed(explore_series),
        certified_attractor: unique_deepest(attractors)
            .is_some_and(|index| {
                let putative = &attractors[index];
                let competitors: Vec<AttractorStrength> = attractors
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != index)
                    .map(|(_, attractor)| attractor.clone())
                    .collect();
                certified_global_minimum(putative, &competitors, true)
            }),
        pruned: false,
    }
}

fn unique_deepest(attractors: &[AttractorStrength]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    let mut unique = true;
    for (index, attractor) in attractors.iter().enumerate() {
        if !attractor.energy.is_finite() {
            continue;
        }
        match best {
            None => {
                best = Some((index, attractor.energy));
                unique = true;
            }
            Some((_, energy)) if attractor.energy < energy - 1e-10 => {
                best = Some((index, attractor.energy));
                unique = true;
            }
            Some((_, energy)) if (attractor.energy - energy).abs() <= 1e-10 => {
                unique = false;
            }
            _ => {}
        }
    }
    if unique { best.map(|(index, _)| index) } else { None }
}
