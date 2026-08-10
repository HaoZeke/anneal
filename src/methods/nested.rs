//! Nested search: the quenched landscape under a descending ceiling.
//!
//! *Measured negative.* This module is kept as the reproducibility record of
//! a mechanism that was built, measured against paired controls, and refuted;
//! the numbers and the mechanism of failure are in `docs/derivations/`. It is
//! not part of the recommended configuration and ships as evidence, not as
//! advice.
//!
//! # Where this comes from
//!
//! The eigensolver lineage splits hard spectra into windows and solves inside
//! each window with walls rather than gradients: spectrum slicing and
//! shift-invert in the EISPACK tradition, contour filters in its modern
//! descendants, the divide-and-merge of Cuppen (doi:10.1007/BF01396757) at the
//! core of ELPA (doi:10.1088/0953-8984/26/21/213201). The statistical form of
//! the same idea is nested sampling (Skilling, doi:10.1214/06-BA127): hold a
//! population of `K` states, repeatedly discard the worst and replace it with a
//! clone of a survivor evolved under the hard constraint `E < E_worst`. The
//! ceiling descends by order statistics, each replacement compresses the
//! reachable volume by about `K/(K+1)`, and no Metropolis ratio appears
//! anywhere. Applied to the quenched landscape of a cluster this is the
//! construction of Partay, Bartok and Csanyi (doi:10.1021/jp1012973).
//!
//! # Why it fits the measured obstruction
//!
//! Measured on the 38-point double funnel: the crossing into the funnel that
//! holds the answer completes from precursor structures *above* the incumbent
//! best, reached through accepted uphill moves, and under Metropolis every
//! intermediate of that excursion must survive its own test, so the excursion
//! survives with the product of its acceptance probabilities. Under a ceiling
//! there is no product: any state below the ceiling is free, so precursors
//! stay reachable for as long as the ceiling has not passed them. The
//! population keeps both funnels alive without being told funnels exist,
//! which is deflation by bookkeeping rather than by penalty.
//!
//! Nothing here reads an order parameter, a template or a morphology. The
//! moves are the library's own, the constraint is the energy the run measures
//! for itself, and the one dimensionless choice is the population size.

use crate::methods::cluster_hopping::{ClusterMove, Ledger, Relax, random_cluster};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Outcome of a nested search.
#[derive(Debug, Clone)]
pub struct NestedOutcome {
    /// Best quenched energy seen.
    pub best: f64,
    /// The structure that reached it.
    pub best_x: Array1<f64>,
    /// Ceiling replacements performed.
    pub replacements: usize,
    /// Constrained walk steps taken.
    pub steps: usize,
    /// Walk steps whose trial fell below the ceiling and was taken.
    pub taken: usize,
    /// The ceiling when the budget died.
    pub final_ceiling: f64,
    /// Fresh populations started after the posterior said the previous one had
    /// drained its reachable volume.
    pub repopulations: usize,
    /// The death record, for the volume-energy curve.
    pub curve: VolumeCurve,
}

/// Configuration for the nested search.
#[derive(Debug, Clone)]
pub struct NestedConfig {
    /// Points in a state.
    pub n_points: usize,
    /// Live points. Resolution of the compression: each replacement removes
    /// about `1/K` of the reachable volume, so larger is slower and finer.
    pub live: usize,
    /// Constrained walk steps per replacement.
    pub walk: usize,
    /// Relaxation steps for the screening pass of the walk.
    pub screen_steps: usize,
    /// Relaxation steps for a full quench.
    pub relax_steps: usize,
    /// Move scale handed to the kernels.
    pub temperature: f64,
}

impl NestedConfig {
    /// Defaults for an `n`-point cluster: the driver's own screen and quench
    /// lengths, the lean move library's scale, and a population sized to the
    /// cluster rather than tuned.
    pub fn for_cluster(n: usize) -> Self {
        Self {
            n_points: n,
            live: 32,
            walk: 8,
            screen_steps: 25,
            relax_steps: 500,
            temperature: 0.8,
        }
    }
}

/// Runs the nested search until the ledger refuses.
///
/// The walk under the ceiling uses the lean library: the arms measured to
/// carry improvements. A trial is screened with a short relaxation first and
/// promoted to a full quench only when the screened energy is not hopelessly
/// above the ceiling, which is the same economy the driver's screen buys.
pub fn nested_search(
    cfg: &NestedConfig,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    seed: u64,
) -> NestedOutcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = cfg.n_points;
    let kernels = ClusterMove::library_lean(n);
    let mut live: Vec<(f64, Array1<f64>)> = Vec::with_capacity(cfg.live);
    let mut best = f64::INFINITY;
    let mut best_x = Array1::zeros(3 * n);
    // The population starts as quenched random clusters, each paid for.
    for _ in 0..cfg.live.max(2) {
        let x0 = random_cluster(n, 0.7, 0.5, &mut rng);
        let (e, x) = relax(ledger, x0.view(), cfg.relax_steps);
        if e < best {
            best = e;
            best_x = x.clone();
        }
        live.push((e, x));
        if ledger.remaining() == 0 {
            break;
        }
    }
    let mut replacements = 0usize;
    let mut steps = 0usize;
    let mut taken = 0usize;
    let mut ceiling = f64::INFINITY;
    let mut curve = VolumeCurve::default();
    let mut repopulations = 0usize;
    'outer: loop {
        // The stopping rule, applied as a reallocation rather than a stop:
        // when the measured curve says the volume still reachable below the
        // incumbent best is negligible, this population has converged and
        // further replacements only re-sample it. The budget buys a fresh
        // compression instead.
        if replacements > 2 * cfg.live {
            if let Some(p) = curve.mass_below(best, 1e-3) {
                if p < 0.05 {
                    repopulations += 1;
                    curve = VolumeCurve::default();
                    for slot in live.iter_mut() {
                        let x0 = random_cluster(n, 0.7, 0.5, &mut rng);
                        let (e, x) = relax(ledger, x0.view(), cfg.relax_steps);
                        *slot = (e, x);
                        if e < best {
                            best = e;
                            best_x = slot.1.clone();
                        }
                        if ledger.remaining() == 0 {
                            break 'outer;
                        }
                    }
                }
            }
        }
        // The worst live point sets the ceiling and dies.
        let (w, _) = live
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.0
                    .partial_cmp(&b.1.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("population is nonempty");
        ceiling = live[w].0;
        curve.record(ceiling, live.len());
        // Clone a survivor and walk it under the ceiling.
        let s = loop {
            let c = rng.random_range(0..live.len());
            if c != w || live.len() == 1 {
                break c;
            }
        };
        let (mut e, mut x) = live[s].clone();
        for _ in 0..cfg.walk {
            steps += 1;
            let k = rng.random_range(0..kernels.len());
            let trial = kernels[k].propose(x.view(), cfg.temperature, &mut rng);
            let (e_screen, x_screen) = relax(ledger, trial.view(), cfg.screen_steps);
            if ledger.remaining() == 0 {
                break 'outer;
            }
            // Screened out: a partial descent already above the ceiling will
            // not come back under it with more relaxation often enough to pay
            // for trying.
            if e_screen >= ceiling {
                continue;
            }
            let (e_new, x_new) = relax(ledger, x_screen.view(), cfg.relax_steps);
            if ledger.remaining() == 0 {
                break 'outer;
            }
            // The constraint is the whole acceptance rule. Below the ceiling
            // every state is equally welcome, which is what keeps precursor
            // structures above the incumbent best alive.
            if e_new < ceiling {
                e = e_new;
                x = x_new;
                taken += 1;
                if e < best {
                    best = e;
                    best_x = x.clone();
                }
            }
        }
        live[w] = (e, x);
        replacements += 1;
    }
    NestedOutcome {
        best,
        best_x,
        replacements,
        steps,
        taken,
        final_ceiling: ceiling,
        repopulations,
        curve,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ceiling has to descend monotonically, which is the compression the
    /// method is named for.
    #[test]
    fn the_ceiling_descends() {
        let cfg = NestedConfig {
            n_points: 13,
            live: 8,
            walk: 4,
            screen_steps: 10,
            relax_steps: 200,
            temperature: 0.8,
        };
        let mut led = Ledger::new(30000);
        let lj = |x: ndarray::ArrayView1<f64>| -> (f64, Array1<f64>) {
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
        let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iters: usize| {
            opt.forget();
            let (f, xr, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                Some(lj(v))
            });
            (f, xr)
        };
        let out = nested_search(&cfg, &mut led, &mut relax, 7);
        assert!(
            out.replacements > 3,
            "only {} replacements",
            out.replacements
        );
        assert!(
            out.best <= out.final_ceiling,
            "best {} above the final ceiling {}",
            out.best,
            out.final_ceiling
        );
        // LJ13 under this little budget should still be descending well below
        // a random quench, which sits near -37.
        assert!(out.best < -38.0, "best only reached {}", out.best);
    }
}

/// A posterior over the landscape's volume-energy curve, fed by the deaths.
///
/// Each replacement kills the worst of `K` live points, and the reachable
/// volume contracts by a Beta(K, 1) factor, so `ln V` after death `i` is a sum
/// of independent Exponential(K) decrements: mean `i / K`, variance `i / K^2`,
/// exactly known. The pairs `(E_i, ln V_i)` are therefore direct, honestly
/// noised observations of the integrated density of minima -- the quantity
/// Wang-Landau style estimation reconstructs from biased histograms with a
/// gain schedule. Here it arrives as order statistics with analytic error.
///
/// What it is for: the run's own stopping rule. The volume still reachable
/// below the incumbent best is extrapolated from the curve's slope, and when
/// the posterior probability that meaningful volume remains falls below a
/// threshold, the compression has provably outrun the landscape and the budget
/// is better spent on a fresh population than on a dead one.
#[derive(Debug, Clone, Default)]
pub struct VolumeCurve {
    /// Death energies, in the order they occurred.
    pub deaths: Vec<f64>,
    /// Live points at each death.
    pub k_at: Vec<usize>,
}

impl VolumeCurve {
    /// Records a death at `energy` with `k` live points.
    pub fn record(&mut self, energy: f64, k: usize) {
        if energy.is_finite() && k > 0 {
            self.deaths.push(energy);
            self.k_at.push(k);
        }
    }

    /// Mean and standard deviation of `ln V` at death `i`.
    pub fn log_volume(&self, i: usize) -> (f64, f64) {
        let mut mean = 0.0;
        let mut var = 0.0;
        for j in 0..=i.min(self.k_at.len().saturating_sub(1)) {
            let k = self.k_at[j] as f64;
            mean -= 1.0 / k;
            var += 1.0 / (k * k);
        }
        (mean, var.sqrt())
    }

    /// The local slope `d ln V / dE` near the current ceiling, by least squares
    /// over the last `window` deaths, with its standard error.
    ///
    /// This is the reciprocal statistical temperature of the *minima*
    /// distribution at the ceiling, measured rather than assumed. A steep
    /// slope says the volume is draining fast per unit energy: many minima per
    /// energy still ahead. A flattening slope says compression has reached a
    /// floor.
    pub fn slope(&self, window: usize) -> Option<(f64, f64)> {
        let n = self.deaths.len();
        if n < 4 {
            return None;
        }
        let lo = n.saturating_sub(window.max(4));
        let pts: Vec<(f64, f64)> = (lo..n)
            .map(|i| (self.deaths[i], self.log_volume(i).0))
            .collect();
        let m = pts.len() as f64;
        let sx: f64 = pts.iter().map(|p| p.0).sum();
        let sy: f64 = pts.iter().map(|p| p.1).sum();
        let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
        let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
        let det = m * sxx - sx * sx;
        if det.abs() < 1e-12 {
            return None;
        }
        let b = (m * sxy - sx * sy) / det;
        let a = (sy - b * sx) / m;
        let mut ss = 0.0;
        for p in &pts {
            let r = p.1 - (a + b * p.0);
            ss += r * r;
        }
        let se = (ss / (m - 2.0).max(1.0) / (sxx - sx * sx / m).max(1e-12)).sqrt();
        Some((b, se))
    }

    /// Posterior probability that the volume still reachable below `floor`
    /// exceeds `eps` of the current volume, under the local linear model of
    /// `ln V(E)` continued past the ceiling.
    ///
    /// The decision this feeds: while the probability is high, keep
    /// compressing, because the measured curve says undrained volume remains
    /// below the incumbent; once it collapses, the population has converged
    /// onto structures the curve says are the last ones, and further
    /// replacements only re-sample them.
    pub fn mass_below(&self, floor: f64, eps: f64) -> Option<f64> {
        let (b, se) = self.slope(24)?;
        let last = *self.deaths.last()?;
        let gap = last - floor;
        if gap <= 0.0 {
            return Some(0.0);
        }
        // ln of the volume ratio between the floor and the ceiling under the
        // linear continuation. The slope is d ln V / dE with V shrinking as E
        // falls, so descending by gap leaves ln ratio = -b * gap, and the
        // reachable fraction exceeds eps when -b * gap > ln eps.
        let need = eps.ln();
        let z = (-b * gap - need) / (se * gap).max(1e-12);
        // One-sided normal tail.
        Some(0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2)))
    }
}

/// Abramowitz-Stegun 7.1.26, absolute error under 1.5e-7, plenty for a
/// stopping decision.
fn erf_approx(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

#[cfg(test)]
mod volume_tests {
    use super::*;

    /// The log-volume error bar has to match the analytic order-statistics
    /// variance, which is the whole point of feeding deaths rather than
    /// histograms.
    #[test]
    fn log_volume_carries_the_analytic_error() {
        let mut v = VolumeCurve::default();
        for i in 0..100 {
            v.record(-(i as f64), 20);
        }
        let (m, s) = v.log_volume(79);
        assert!((m + 80.0 / 20.0).abs() < 1e-9, "mean {m}");
        assert!((s - (80.0f64 / 400.0).sqrt()).abs() < 1e-9, "sd {s}");
    }

    /// A steep synthetic landscape has to read as "mass remains" and a
    /// flattened one as "drained", or the stopping rule stops the wrong runs.
    #[test]
    fn the_stopping_rule_separates_steep_from_flat() {
        // Steep: ln V falls 0.5 per unit energy; far below the ceiling there
        // is volume left.
        let mut steep = VolumeCurve::default();
        for i in 0..60 {
            // deaths marching down in energy, K=20: ln V_i = -i/20, and the
            // energies fall by 0.1 each, so slope = (1/20)/0.1 = 0.5 per unit E.
            steep.record(10.0 - 0.1 * i as f64, 20);
        }
        let p_steep = steep.mass_below(0.0, 1e-3).expect("no slope");
        // Flat: the same deaths bunched within a hair of one energy; below
        // them, nothing.
        let mut flat = VolumeCurve::default();
        for i in 0..60 {
            flat.record(10.0 - 1e-4 * i as f64, 20);
        }
        let p_flat = flat.mass_below(0.0, 1e-3).expect("no slope");
        assert!(
            p_steep > 0.9 && p_flat < 0.1,
            "steep {p_steep}, flat {p_flat}"
        );
    }
}
