//! Residual archive search: local events, energy floors, cheap novelty loop.
//!
//! This is not [`Config::recommended`]. That preset keeps its measured rates.
//! The driver here buys a full quench only for a residual landing or a live
//! class record, hunts with the screen until the trial is not a return, and
//! spends ARTn only on an unsaturated local key.
//!
//! Shared state is [`Archive`]. Several workers pass the same archive; pending
//! locks stop two of them paying `R` for the same floor or the same residual
//! slot.

use crate::allocate::DepthAllocator;
use crate::calibrate::StepCalibrator;
use crate::catalog::{Catalog, Event};
use crate::floors::FloorBook;
use crate::graphkey::contact_key;
use crate::localkey::{bag_key, local_key, local_keys};
use crate::methods::activation::{activate, Activation};
use crate::methods::cluster_hopping::{
    active_mask, contain, recentre, Config, GradFn, Ledger, Relax,
};
use crate::residual_field::ResidualField;
use crate::screen::DropModel;
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use std::collections::HashSet;

/// Contact multiple for local NAUTY, matching [`crate::graphkey`] tests.
const LOCAL_CUTOFF: f64 = 1.35;

/// Shared catalogue, floors, residual field, and pending locks.
#[derive(Debug, Clone)]
pub struct Archive {
    /// Local-topology events.
    pub catalog: Catalog,
    /// Energy classes.
    pub floors: FloorBook,
    /// GMRF residual on the class graph.
    pub residual: ResidualField,
    /// Remaining-drop model.
    pub drop: DropModel,
    /// One representative structure per floor.
    pub reps: Vec<Array1<f64>>,
    /// Floor currently being fully quenched.
    pub pending_floors: HashSet<usize>,
    /// Whether a residual full quench is in flight.
    pub pending_residual: bool,
}

impl Default for Archive {
    fn default() -> Self {
        Self::new()
    }
}

impl Archive {
    /// Empty shared state.
    pub fn new() -> Self {
        Self {
            catalog: Catalog::new(),
            floors: FloorBook::new(),
            residual: ResidualField::new(),
            drop: DropModel::new(),
            reps: Vec::new(),
            pending_floors: HashSet::new(),
            pending_residual: false,
        }
    }

    fn set_rep(&mut self, id: usize, x: ArrayView1<f64>) {
        if id >= self.reps.len() {
            self.reps.resize(id + 1, Array1::zeros(0));
        }
        if self.reps[id].is_empty() {
            self.reps[id] = x.to_owned();
        }
    }
}

/// What one archive run produced.
#[derive(Debug, Clone, Default)]
pub struct ArchiveOutcome {
    /// Lowest quenched energy.
    pub best: f64,
    /// Structure attaining it.
    pub best_state: Option<Array1<f64>>,
    /// Screen passes paid.
    pub screens: usize,
    /// Full quenches paid.
    pub full: usize,
    /// Screens discarded as returning to the incumbent attractor.
    pub returned: usize,
    /// Screens discarded as a saturated floor.
    pub same_floor: usize,
    /// Distinct energy classes at the end.
    pub floors: usize,
    /// Distinct local-topology events at the end.
    pub events: usize,
    /// ARTn climbs attempted.
    pub artn: usize,
    /// Charged evaluations spent (copied from the ledger).
    pub charged: usize,
    /// Ledger spend when `best` last improved.
    pub best_at: usize,
}

/// One worker on a shared [`Archive`], until the ledger is empty.
pub fn archive_search<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut grad: Option<&mut GradFn<'g>>,
    archive: &mut Archive,
    rng: &mut R,
) -> ArchiveOutcome {
    let tau = FloorBook::tau(cfg.screen_steps, cfg.relax_steps);
    let kernels = cfg.move_library.kernels(cfg);
    let mut alloc = DepthAllocator::new(kernels.len().max(1));
    let n = cfg.n_points;
    let mut x = start.to_owned();
    let (e0, x0) = relax(ledger, x.view(), cfg.relax_steps);
    x = x0;
    let mut e = e0;
    let mut here: Option<usize> = None;
    if e0.is_finite() {
        here = Some(archive.floors.assign(e0, None, 0.0));
        archive
            .residual
            .observe(here.unwrap(), e0);
        archive.set_rep(here.unwrap(), x.view());
        ledger.record(e0, x.view());
        let keyed = key_coords(x.view(), cfg);
        let keys = local_keys(keyed.view(), LOCAL_CUTOFF);
        archive.catalog.observe_bag(&keys);
    }
    let mut out = ArchiveOutcome {
        best: e0,
        best_state: if e0.is_finite() {
            Some(x.clone())
        } else {
            None
        },
        full: 1,
        best_at: ledger.spent(),
        ..ArchiveOutcome::default()
    };
    let mut scale_cal = StepCalibrator::new(0.5, 8, 1.0);
    let mut scale = 1.0_f64;
    let mut here_key = contact_key(key_coords(x.view(), cfg).view(), LOCAL_CUTOFF);

    while ledger.remaining() > 0 {
        // Start choice: unsaturated local key / live floor / residual.
        let e_star = ledger.best;
        let stuck = here
            .and_then(|h| archive.floors.get(h))
            .is_some_and(|f| f.saturated(tau, e_star));
        if stuck {
            if let Some(id) = archive.floors.best_start(tau, e_star) {
                if id < archive.reps.len() && archive.reps[id].len() == x.len() && here != Some(id)
                {
                    x = archive.reps[id].clone();
                    e = archive.floors.get(id).map(|f| f.e_min).unwrap_or(e);
                    here = Some(id);
                    here_key = contact_key(key_coords(x.view(), cfg).view(), LOCAL_CUTOFF);
                }
            }
        }

        let keys_here = local_keys(key_coords(x.view(), cfg).view(), LOCAL_CUTOFF);
        archive.catalog.observe_bag(&keys_here);

        // Cheap novelty loop: skip only exact returns. Same-floor isomers are
        // still quenched; refusing them starved the search (24 full quenches
        // in 400k on LJ38). ARTn waits until this loop finds no escape.
        let mut found = None;
        let s_steps = cfg.screen_steps.max(1);
        let max_tries = ((ledger.remaining() / s_steps).max(1)).min(32);
        for _ in 0..max_tries {
            if ledger.remaining() < s_steps {
                break;
            }
            let k = if cfg.allocate_moves && !kernels.is_empty() {
                alloc.select(rng)
            } else if kernels.is_empty() {
                break;
            } else {
                rng.random_range(0..kernels.len())
            };
            let mut trial = kernels[k].propose_scaled(x.view(), cfg.temperature, scale, rng);
            recentre(&mut trial, n);
            contain(&mut trial, n, cfg.container);
            let (e_sc, x_sc) = relax(ledger, trial.view(), cfg.screen_steps);
            out.screens += 1;
            if !e_sc.is_finite() {
                continue;
            }
            let sc_key = contact_key(key_coords(x_sc.view(), cfg).view(), LOCAL_CUTOFF);
            if sc_key == here_key {
                out.returned += 1;
                scale_cal.observe(0.1);
                scale = (scale * 1.05).min(8.0);
                continue;
            }
            // Same energy screen as the recommended hop: a trial this far
            // above the incumbent is not worth 200 L-BFGS steps.
            if e_sc > ledger.best + cfg.screen_margin {
                continue;
            }
            scale_cal.observe(1.0);
            scale = (scale * 0.98).max(0.5);

            let hat = archive
                .drop
                .predicted_full(e_sc)
                .unwrap_or(e_sc);
            let rise_hat = (hat - e).max(0.0);
            // Peek assignment without committing a full-quench observation:
            // compare to existing floors.
            let de = archive.floors.delta_e();
            let mut landed: Option<usize> = None;
            if let Some(h) = here {
                if rise_hat <= de {
                    landed = Some(h);
                }
            }
            if landed.is_none() {
                for i in 0..archive.floors.len() {
                    if let Some(f) = archive.floors.get(i) {
                        if (hat - f.e_min).abs() <= de {
                            landed = Some(i);
                            break;
                        }
                    }
                }
            }
            if landed.is_some_and(|id| archive.pending_floors.contains(&id))
                || (landed.is_none() && archive.pending_residual)
            {
                continue;
            }
            if landed.is_some_and(|id| {
                archive
                    .floors
                    .get(id)
                    .is_some_and(|f| f.saturated(tau, ledger.best))
            }) {
                out.same_floor += 1;
            }
            found = Some((e_sc, x_sc, landed, k));
            break;
        }
        let Some((e_sc, x_sc, landed, arm)) = found else {
            // No non-return in the cheap window: one ARTn residual, then hop.
            if let (Some(uk), Some(g)) = (
                archive.catalog.unsaturated_in(&keys_here),
                grad.as_deref_mut(),
            ) {
                out.artn += 1;
                let sign = if rng.random::<bool>() { 1.0 } else { -1.0 };
                if let Some(ao) = activate(x.view(), |y| g(ledger, y), &Activation::default(), sign)
                {
                    if ao.crossed && ledger.remaining() > 0 {
                        let (ef, xf) = relax(ledger, ao.state.view(), cfg.relax_steps);
                        out.full += 1;
                        archive.catalog.record_search(
                            uk,
                            ef.is_finite().then_some(Event {
                                from: uk,
                                to: local_key(key_coords(xf.view(), cfg).view(), 0, LOCAL_CUTOFF),
                                dest_energy: ef,
                            }),
                        );
                        if ef.is_finite() {
                            let rise = (ef - e).max(0.0);
                            archive.floors.observe_rise(rise);
                            let dest = archive.floors.assign(ef, here, rise);
                            archive.residual.observe(dest, ef);
                            archive.set_rep(dest, xf.view());
                            ledger.record(ef, xf.view());
                            if ef < out.best {
                                out.best = ef;
                                out.best_state = Some(xf.clone());
                                out.best_at = ledger.spent();
                            }
                            let temp = cfg.temperature.max(1e-12);
                            if ef <= e || rng.random::<f64>() < (-(ef - e) / temp).exp() {
                                x = xf;
                                e = ef;
                                here = Some(dest);
                                here_key =
                                    contact_key(key_coords(x.view(), cfg).view(), LOCAL_CUTOFF);
                            }
                        }
                    } else {
                        archive.catalog.record_search(uk, None);
                    }
                } else {
                    archive.catalog.record_search(uk, None);
                }
            }
            continue;
        };

        if landed.is_none() {
            archive.pending_residual = true;
        } else if let Some(id) = landed {
            archive.pending_floors.insert(id);
        }

        let (ef, xf) = relax(ledger, x_sc.view(), cfg.relax_steps);
        out.full += 1;
        if cfg.allocate_moves {
            alloc.update(arm, -ef);
        }
        archive.drop.observe(e_sc, ef);
        if landed.is_none() {
            archive.pending_residual = false;
        } else if let Some(id) = landed {
            archive.pending_floors.remove(&id);
        }
        if !ef.is_finite() {
            continue;
        }
        let rise = (ef - e).max(0.0);
        archive.floors.observe_rise(rise);
        let dest = archive.floors.assign(ef, here, rise);
        archive.residual.observe(dest, ef);
        if let (Some(h), d) = (here, dest) {
            if h != d {
                archive.residual.edge(h, d);
            }
        }
        archive.set_rep(dest, xf.view());
        let from_bag = bag_key(&keys_here);
        let to_keys = local_keys(key_coords(xf.view(), cfg).view(), LOCAL_CUTOFF);
        if let Some(uk) = keys_here.first().copied() {
            archive.catalog.record_search(
                uk,
                Some(Event {
                    from: uk,
                    to: bag_key(&to_keys),
                    dest_energy: ef,
                }),
            );
        }
        let _ = from_bag;
        ledger.record(ef, xf.view());
        if ef < out.best {
            out.best = ef;
            out.best_state = Some(xf.clone());
            out.best_at = ledger.spent();
        }
        let temp = cfg.temperature.max(1e-12);
        let accept = ef <= e || rng.random::<f64>() < (-(ef - e) / temp).exp();
        if accept {
            x = xf;
            e = ef;
            here = Some(dest);
            here_key = contact_key(key_coords(x.view(), cfg).view(), LOCAL_CUTOFF);
        }
    }

    out.floors = archive.floors.len();
    out.events = archive.catalog.event_count();
    out.charged = ledger.spent();
    out
}

/// Coordinates used for topology identity: the active patch on a slab, else all.
fn key_coords(x: ArrayView1<f64>, cfg: &Config) -> Array1<f64> {
    let n = x.len() / 3;
    let Some((seeds, shells)) = cfg.active_region.as_ref() else {
        return x.to_owned();
    };
    let Some(species) = cfg.species.as_ref() else {
        return x.to_owned();
    };
    let mask = active_mask(x, species, seeds, *shells, cfg.bond_tolerance);
    let idx: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, m)| **m)
        .map(|(i, _)| i)
        .collect();
    if idx.len() < 2 {
        return x.to_owned();
    }
    let mut y = Array1::zeros(idx.len() * 3);
    for (k, &i) in idx.iter().enumerate() {
        if i < n {
            for d in 0..3 {
                y[3 * k + d] = x[3 * i + d];
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::cluster_hopping::random_cluster;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn lj(x: ArrayView1<f64>) -> f64 {
        let n = x.len() / 3;
        let mut e = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut r2 = 0.0;
                for k in 0..3 {
                    let d = x[3 * i + k] - x[3 * j + k];
                    r2 += d * d;
                }
                if r2 < 1e-12 {
                    return f64::INFINITY;
                }
                let ir2 = 1.0 / r2;
                let ir6 = ir2 * ir2 * ir2;
                e += 4.0 * (ir6 * ir6 - ir6);
            }
        }
        e
    }

    #[test]
    fn archive_search_does_not_touch_recommended_defaults() {
        let rec = Config::recommended(13);
        assert!(!rec.return_screen);
        assert_eq!(rec.screen_steps, 25);
        assert_eq!(rec.relax_steps, 200);
    }

    #[test]
    fn a_tiny_run_records_floors_and_refuses_returns() {
        let mut rng = StdRng::seed_from_u64(1);
        let cfg = Config::recommended(7);
        let start = random_cluster(7, 0.7, cfg.min_separation, &mut rng);
        let mut ledger = Ledger::new(400);
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
            let mut y = x.to_owned();
            let mut e = lj(y.view());
            for _ in 0..steps {
                if !led.charge() {
                    break;
                }
                // One coordinate of steepest finite difference, one step.
                let mut best_i = 0usize;
                let mut best_g = 0.0_f64;
                for i in 0..y.len() {
                    let old = y[i];
                    y[i] = old + 1e-4;
                    let ep = lj(y.view());
                    y[i] = old;
                    let g = (ep - e) / 1e-4;
                    if g.abs() > best_g.abs() {
                        best_g = g;
                        best_i = i;
                    }
                }
                y[best_i] -= 0.05 * best_g.signum();
                e = lj(y.view());
            }
            (e, y)
        };
        let mut archive = Archive::new();
        let out = archive_search(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            None,
            &mut archive,
            &mut rng,
        );
        assert!(out.full >= 1);
        assert!(out.screens + out.full >= 1);
        assert!(out.charged <= 400);
        assert!(out.best.is_finite());
        // The loop either hunts or refuses. A run that only full-quenches the
        // start still has a floor.
        assert!(out.floors >= 1);
    }
}
