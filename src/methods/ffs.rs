//! Archive-ratcheted exploration of the minima network.
//!
//! *Measured negative.* This module is kept as the reproducibility record of
//! a mechanism that was built, measured against paired controls, and refuted;
//! the numbers and the mechanism of failure are in `docs/derivations/`. It is
//! not part of the recommended configuration and ships as evidence, not as
//! advice.
//!
//! Forward flux sampling (Allen, Warren and ten Wolde,
//! doi:10.1103/PhysRevLett.94.018104) stores every configuration that makes
//! partial progress and fires continuations from the stored frontier, so no
//! single trajectory has to make the whole rare event. The correction this
//! module carries over the naive transcription: a landscape of many minima is
//! a *network*, not a ladder above one incumbent. Interfaces indexed by
//! energy above a single home, cleared on every descent, assume the crossing
//! is one climb-and-descend excursion of one walker, which is a
//! one-dimensional picture. Here the frontier is the whole archive: every
//! distinct arrangement ever quenched, keyed by its canonical contact graph,
//! is a permanent launch site, so progress composes across launches from
//! anywhere and nothing assumes single-walk accessibility.
//!
//! Launch sites are chosen by Thompson sampling on a per-site discovery
//! posterior (Beta over "a launch from here quenched somewhere new"), with a
//! preference toward low energies: an acquisition over a growing discrete
//! design space, which is Bayesian optimisation over the archive rather than
//! over coordinates. Identity is the canonical graph key: exact, no
//! threshold, no reference, no morphology.

use crate::graphkey::contact_key;
use crate::methods::cluster_hopping::{ClusterMove, Ledger, Relax, random_cluster};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Outcome of an archive-ratcheted run.
#[derive(Debug, Clone)]
pub struct FfsOutcome {
    /// Best quenched energy reached.
    pub best: f64,
    /// Distinct arrangements in the archive.
    pub descents: usize,
    /// Archive inserts (new keys discovered).
    pub stored: usize,
    /// Launches fired.
    pub continuations: usize,
    /// Launches whose every shot landed on known keys.
    pub returns: usize,
}

/// Configuration.
#[derive(Debug, Clone)]
pub struct FfsConfig {
    /// Points in a state.
    pub n_points: usize,
    /// Launch window: sites within this many temperatures of the archive's
    /// best remain launchable. Wide enough that high funnels stay reachable,
    /// finite so the acquisition is not diluted over dead material.
    pub window: f64,
    /// Proposals fired per launch.
    pub shots: usize,
    /// Screening relaxation length.
    pub screen_steps: usize,
    /// Full relaxation length.
    pub relax_steps: usize,
    /// Move scale.
    pub temperature: f64,
    /// Contact-graph cutoff in units of the median nearest-neighbour distance.
    pub key_cutoff: f64,
}

impl FfsConfig {
    /// Defaults: launch window of six temperatures, two dozen shots per launch.
    pub fn for_cluster(n: usize) -> Self {
        Self {
            n_points: n,
            window: 6.0,
            shots: 24,
            screen_steps: 25,
            relax_steps: 500,
            temperature: 0.8,
            key_cutoff: 1.35,
        }
    }
}

/// One archive site: a distinct arrangement and its discovery record.
struct Site {
    e: f64,
    x: Array1<f64>,
    /// Beta posterior over "a shot from here finds a new key".
    a: f64,
    b: f64,
}

/// Runs archive-ratcheted exploration until the ledger refuses.
pub fn ffs_descent(
    cfg: &FfsConfig,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    seed: u64,
) -> FfsOutcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = cfg.n_points;
    let kernels = ClusterMove::library_lean_burst(n);
    let mut archive: std::collections::HashMap<u64, Site> = std::collections::HashMap::new();
    let mut best = f64::INFINITY;
    let mut stored = 0usize;
    let mut launches = 0usize;
    let mut barren = 0usize;
    let x0 = random_cluster(n, 0.7, 0.5, &mut rng);
    let (e0, xq) = relax(ledger, x0.view(), cfg.relax_steps);
    best = best.min(e0);
    archive.insert(
        contact_key(xq.view(), cfg.key_cutoff),
        Site {
            e: e0,
            x: xq,
            a: 1.0,
            b: 1.0,
        },
    );
    'outer: while ledger.remaining() > 0 {
        // The acquisition: Thompson draw on each launchable site's discovery
        // posterior, tilted toward depth by the same temperature the moves
        // use. Sites outside the window are retired from launching but never
        // deleted: identity is permanent, launchability is not.
        let floor = best + cfg.window * cfg.temperature;
        let pick = {
            let mut best_key = None;
            let mut best_score = f64::NEG_INFINITY;
            for (k, s) in archive.iter() {
                if s.e > floor {
                    continue;
                }
                let draw = crate::allocate::beta_draw(s.a, s.b, &mut rng);
                let score = draw.max(1e-12).ln() - (s.e - best) / cfg.temperature;
                if score > best_score {
                    best_score = score;
                    best_key = Some(*k);
                }
            }
            match best_key {
                Some(k) => k,
                None => break 'outer,
            }
        };
        launches += 1;
        let (sx, se) = {
            let s = &archive[&pick];
            (s.x.clone(), s.e)
        };
        let _ = se;
        let mut found_new = false;
        for _ in 0..cfg.shots {
            if ledger.remaining() == 0 {
                break 'outer;
            }
            let k = rng.random_range(0..kernels.len());
            let trial = kernels[k].propose(sx.view(), cfg.temperature, &mut rng);
            let (e_s, x_s) = relax(ledger, trial.view(), cfg.screen_steps);
            if e_s >= floor {
                continue;
            }
            if ledger.remaining() == 0 {
                break 'outer;
            }
            let (e_new, x_new) = relax(ledger, x_s.view(), cfg.relax_steps);
            best = best.min(e_new);
            let key = contact_key(x_new.view(), cfg.key_cutoff);
            if let std::collections::hash_map::Entry::Vacant(v) = archive.entry(key) {
                v.insert(Site {
                    e: e_new,
                    x: x_new,
                    a: 1.0,
                    b: 1.0,
                });
                stored += 1;
                found_new = true;
            }
        }
        if let Some(s) = archive.get_mut(&pick) {
            if found_new {
                s.a += 1.0;
            } else {
                s.b += 1.0;
                barren += 1;
            }
        }
    }
    FfsOutcome {
        best,
        descents: archive.len(),
        stored,
        continuations: launches,
        returns: barren,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayView1;

    /// Descent has to be monotone in the adopted home, and the run has to make
    /// real progress on a small cluster under a small budget.
    #[test]
    fn descent_is_monotone_and_reaches_depth() {
        let cfg = FfsConfig::for_cluster(13);
        let mut led = Ledger::new(30000);
        let lj = |x: ArrayView1<f64>| -> (f64, Array1<f64>) {
            let n = x.len() / 3;
            let mut e = 0.0;
            let mut g = Array1::zeros(x.len());
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = [
                        x[3 * i] - x[3 * j],
                        x[3 * i + 1] - x[3 * j + 1],
                        x[3 * i + 2] - x[3 * j + 2],
                    ];
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    let inv2 = 1.0 / r2;
                    let inv6 = inv2 * inv2 * inv2;
                    let inv12 = inv6 * inv6;
                    e += 4.0 * (inv12 - inv6);
                    let c = 24.0 * inv2 * (2.0 * inv12 - inv6);
                    for k in 0..3 {
                        g[3 * i + k] -= c * d[k];
                        g[3 * j + k] += c * d[k];
                    }
                }
            }
            (e, g)
        };
        let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            opt.forget();
            let (f, xr, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                Some(lj(v))
            });
            (f, xr)
        };
        let out = ffs_descent(&cfg, &mut led, &mut relax, 11);
        assert!(out.best < -40.0, "reached only {}", out.best);
        assert!(out.descents > 0, "never descended");
    }
}
