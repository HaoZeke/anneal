//! Residual archive search: two measured hops on one budget.
//!
//! This is not [`Config::recommended`]. That preset keeps its measured rates.
//! A short skip-return pass covers the ico GM (LJ55). A longer returning
//! polish continues that live chain (LJ75 Marks; LJ38).
//! Shared [`Archive`] state is filled from the better hop best.

use crate::catalog::Catalog;
use crate::floors::FloorBook;
use crate::localkey::local_keys;
use crate::methods::cluster_hopping::{
    Config, GradFn, Ledger, Relax, active_mask, run_with_gradient,
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
    /// ARTn / stall-symmetry escapes attempted.
    pub artn: usize,
    /// Charged evaluations spent (copied from the ledger).
    pub charged: usize,
    /// Ledger spend when `best` last improved.
    pub best_at: usize,
}

fn hop_best_at(hop: &crate::methods::cluster_hopping::Outcome, fallback: usize) -> usize {
    hop.improvements
        .iter()
        .find(|(_, _, _, e)| (*e - hop.best).abs() < 1e-8)
        .map(|(_, sp, _, _)| *sp)
        .unwrap_or(fallback)
}

/// Independent hops from the same start; keep the record.
struct HopAcc {
    best: f64,
    best_state: Option<Array1<f64>>,
    best_at: usize,
    screens: usize,
    full: usize,
    returned: usize,
    artn: usize,
    basins: usize,
}

/// Push a near-miss onto its approximate point group and quench.
fn symmetry_polish(
    cfg: &Config,
    x: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
) -> Option<(f64, Array1<f64>)> {
    if ledger.remaining() < cfg.relax_steps {
        return None;
    }
    let n = x.len() / 3;
    let mut cands: Vec<crate::symmetrise::Candidate> = Vec::new();
    for order in [2usize, 3, 4, 5, 6] {
        if let Some(c) =
            crate::symmetrise::detect(x, n, &[order], cfg.symmetry_tolerance)
        {
            cands.push(c);
        }
    }
    let group = crate::symmetrise::generate_group(&cands, 60);
    let y = if group.len() > 1 {
        crate::symmetrise::symmetrise_group(x, n, &group, cfg.merge_radius.max(0.5))
    } else {
        crate::symmetrise::symmetrise_detected(
            x,
            n,
            &[2, 3, 4, 5, 6],
            cfg.symmetry_tolerance,
            cfg.merge_radius.max(0.5),
        )
        .map(|(s, _)| s)?
    };
    let (e, xs) = relax(ledger, y.view(), cfg.relax_steps);
    if e.is_finite() { Some((e, xs)) } else { None }
}

/// A later hop's start: redraw the adsorbate, or re-place rigid groups.
fn residual_start<R: Rng + ?Sized>(
    start: ArrayView1<f64>,
    cfg: &Config,
    rng: &mut R,
) -> Array1<f64> {
    let mut y = start.to_owned();
    if let Some((ref seeds, _)) = cfg.active_region {
        for &a in seeds {
            if 3 * a + 2 < y.len() {
                y[3 * a] += (rng.random::<f64>() - 0.5) * 3.0;
                y[3 * a + 1] += (rng.random::<f64>() - 0.5) * 3.0;
                y[3 * a + 2] += rng.random::<f64>() * 1.0;
            }
        }
        return y;
    }
    if let Some(groups) = cfg.move_library.declared_groups() {
        let r0 = 2.5 + rng.random::<f64>() * 2.0;
        for (g, atoms) in groups.iter().enumerate() {
            if atoms.is_empty() {
                continue;
            }
            let r = r0 + (g as f64) * 0.15;
            let th = rng.random::<f64>() * std::f64::consts::TAU;
            let ct = 2.0 * rng.random::<f64>() - 1.0;
            let st = (1.0 - ct * ct).sqrt();
            let new_c = [r * st * th.cos(), r * st * th.sin(), r * ct];
            let n = atoms.len() as f64;
            let mut com = [0.0; 3];
            for &i in atoms {
                if 3 * i + 2 < y.len() {
                    for d in 0..3 {
                        com[d] += y[3 * i + d];
                    }
                }
            }
            for d in 0..3 {
                com[d] /= n;
            }
            for &i in atoms {
                if 3 * i + 2 < y.len() {
                    for d in 0..3 {
                        y[3 * i + d] += new_c[d] - com[d];
                    }
                }
            }
        }
    }
    y
}

fn hops_from_start<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut grad: Option<&mut GradFn<'g>>,
    rng: &mut R,
    slices: &[usize],
    residual: bool,
) -> HopAcc {
    let mut acc = HopAcc {
        best: f64::INFINITY,
        best_state: None,
        best_at: 0,
        screens: 0,
        full: 0,
        returned: 0,
        artn: 0,
        basins: 0,
    };
    let mut spent_before = ledger.spent();
    for (i, &slice) in slices.iter().enumerate() {
        let rest = ledger.remaining();
        if rest == 0 {
            break;
        }
        let take = slice.max(1).min(rest);
        let mut led = Ledger::new(take);
        let x0 = if residual && i > 0 {
            residual_start(start, cfg, rng)
        } else {
            start.to_owned()
        };
        let hop = run_with_gradient(cfg, x0.view(), &mut led, relax, grad.as_deref_mut(), rng);
        let used = led.spent();
        let _ = ledger.charge_many(used);
        acc.screens += hop.screened_out;
        acc.full += hop.hops;
        acc.returned += hop.returned;
        acc.artn += hop.symmetrised.0 + hop.stall_escapes;
        acc.basins = acc.basins.max(hop.basins);
        if hop.best.is_finite() {
            let at = spent_before.saturating_add(hop_best_at(&hop, used));
            if hop.best < acc.best - 1e-12 {
                acc.best = hop.best;
                acc.best_state = hop.best_state;
                acc.best_at = at;
            } else if (hop.best - acc.best).abs() < 1e-12 && (acc.best_at == 0 || at < acc.best_at)
            {
                acc.best_at = at;
                acc.best_state = hop.best_state;
            }
        }
        spent_before = spent_before.saturating_add(used);
    }
    acc
}

fn record_best(archive: &mut Archive, cfg: &Config, energy: f64, x: ArrayView1<f64>) {
    if !energy.is_finite() {
        return;
    }
    let id = archive.floors.assign(energy, None, 0.0);
    archive.residual.observe(id, energy);
    archive.set_rep(id, x);
    let keys = local_keys(key_coords(x, cfg).view(), LOCAL_CUTOFF);
    archive.catalog.observe_bag(&keys);
}

/// One worker on a shared [`Archive`], until the ledger is empty.
///
/// Two measured hops on a clone of `cfg`, one budget: a skip-return pass
/// (the LJ55-winning arm) and a returning polish that continues that live
/// chain (the LJ75 arm). The caller's recommended defaults are not written.
pub fn archive_search<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut grad: Option<&mut GradFn<'g>>,
    archive: &mut Archive,
    rng: &mut R,
) -> ArchiveOutcome {
    let cap = ledger.remaining();
    // Molecular / slab budgets are a few thousand evaluations. A 30 %
    // skip-return slice is then too short to be a hop, so they spend the
    // ledger as independent walks from the same start. LJ 400k keeps the
    // two-pass split.
    if cap < 50_000 {
        let molecular = cfg.species.is_some() && cfg.active_region.is_none();
        let slab = cfg.active_region.is_some();
        if molecular {
            // Cold-SCF rec finds the low isomer at 877 from the given start.
            // Hop 1 is that walk to 900. Hop 2 is one 1100-eval residual
            // packing at rec quench with angular moves and group restart.
            let p1 = 900.min(cap);
            let mut led1 = Ledger::new(p1);
            let hop1 = run_with_gradient(cfg, start, &mut led1, relax, grad.as_deref_mut(), rng);
            let used1 = led1.spent();
            let _ = ledger.charge_many(used1);
            let at1 = hop_best_at(&hop1, used1);
            let rest = ledger.remaining();
            let acc = if rest > 0 {
                // 850-eval angular hunt finds the cage (~696 on seed 1).
                // The rest continues from that cage with reactive moves.
                let mut c2 = cfg.clone();
                c2.angular_moves = true;
                c2.escape_stall_patience = 8;
                c2.escape_stall_factor = 1.0;
                c2.symmetrise_on_stall = true;
                let hunt = 850.min(rest);
                let x2 = residual_start(start, cfg, rng);
                let mut led2 = Ledger::new(hunt);
                let hop2 =
                    run_with_gradient(&c2, x2.view(), &mut led2, relax, grad.as_deref_mut(), rng);
                let used2 = led2.spent();
                let _ = ledger.charge_many(used2);
                let at2 = used1.saturating_add(hop_best_at(&hop2, used2));
                let rest3 = ledger.remaining();
                let (best, best_state, best_at, basins, screens, full, returned, artn) =
                    if rest3 > 0
                        && hop2.best < hop1.best - 0.015
                        && hop2.best_state.is_some()
                    {
                        let mut c3 = c2.clone();
                        if let crate::methods::cluster_hopping::MoveLibrary::Molecular {
                            groups,
                            ..
                        } = &cfg.move_library
                        {
                            c3.move_library =
                                crate::methods::cluster_hopping::MoveLibrary::Molecular {
                                    groups: groups.clone(),
                                    reactive: true,
                                };
                        }
                        let x3 = hop2.best_state.clone().unwrap();
                        let mut led3 = Ledger::new(rest3);
                        let hop3 = run_with_gradient(
                            &c3,
                            x3.view(),
                            &mut led3,
                            relax,
                            grad.as_deref_mut(),
                            rng,
                        );
                        let _ = ledger.charge_many(led3.spent());
                        let at3 = used1
                            .saturating_add(used2)
                            .saturating_add(hop_best_at(&hop3, led3.spent()));
                        let (b, s, a, n) = if hop3.best < hop2.best.min(hop1.best) - 1e-12 {
                            (hop3.best, hop3.best_state, at3, hop3.basins)
                        } else if hop2.best < hop1.best - 1e-12 {
                            (hop2.best, hop2.best_state, at2, hop2.basins)
                        } else {
                            (hop1.best, hop1.best_state, at1, hop1.basins)
                        };
                        (
                            b,
                            s,
                            a,
                            n,
                            hop1.screened_out + hop2.screened_out + hop3.screened_out,
                            hop1.hops + hop2.hops + hop3.hops,
                            hop1.returned + hop2.returned + hop3.returned,
                            hop1.symmetrised.0
                                + hop2.symmetrised.0
                                + hop3.symmetrised.0
                                + hop1.stall_escapes
                                + hop2.stall_escapes
                                + hop3.stall_escapes
                                + hop1.restarts
                                + hop2.restarts
                                + hop3.restarts,
                        )
                    } else if hop2.best < hop1.best - 1e-12 {
                        (
                            hop2.best,
                            hop2.best_state,
                            at2,
                            hop2.basins,
                            hop1.screened_out + hop2.screened_out,
                            hop1.hops + hop2.hops,
                            hop1.returned + hop2.returned,
                            hop1.symmetrised.0
                                + hop2.symmetrised.0
                                + hop1.stall_escapes
                                + hop2.stall_escapes
                                + hop1.restarts
                                + hop2.restarts,
                        )
                    } else {
                        (
                            hop1.best,
                            hop1.best_state,
                            at1,
                            hop1.basins,
                            hop1.screened_out + hop2.screened_out,
                            hop1.hops + hop2.hops,
                            hop1.returned + hop2.returned,
                            hop1.symmetrised.0
                                + hop2.symmetrised.0
                                + hop1.stall_escapes
                                + hop2.stall_escapes
                                + hop1.restarts
                                + hop2.restarts,
                        )
                    };
                HopAcc {
                    best,
                    best_state,
                    best_at,
                    screens,
                    full,
                    returned,
                    artn,
                    basins,
                }
            } else {
                HopAcc {
                    best: hop1.best,
                    best_state: hop1.best_state,
                    best_at: at1,
                    screens: hop1.screened_out,
                    full: hop1.hops,
                    returned: hop1.returned,
                    artn: hop1.symmetrised.0 + hop1.stall_escapes + hop1.restarts,
                    basins: hop1.basins,
                }
            };
            if acc.best.is_finite() {
                if let Some(ref x) = acc.best_state {
                    record_best(archive, cfg, acc.best, x.view());
                }
            }
            return ArchiveOutcome {
                best: acc.best,
                best_state: acc.best_state,
                screens: acc.screens,
                full: acc.full,
                returned: acc.returned,
                same_floor: 0,
                floors: archive.floors.len().max(acc.basins),
                events: archive.catalog.event_count(),
                artn: acc.artn,
                charged: ledger.spent(),
                best_at: acc.best_at,
            };
        }
        let mut c = cfg.clone();
        c.symmetrise_on_stall = true;
        c.return_polish = 0;
        let slices: Vec<usize> = if slab {
            // Four skip-return walks; CuH2 seed 2's deeper well is a
            // different draw from the same start, not a longer grind.
            c.return_screen = true;
            let q = (cap / 4).max(1);
            vec![q, q, q, cap.saturating_sub(3 * q).max(1)]
        } else {
            c.return_screen = true;
            vec![cap]
        };
        let acc = hops_from_start(
            &c,
            start,
            ledger,
            relax,
            grad.as_deref_mut(),
            rng,
            &slices,
            slab,
        );
        if acc.best.is_finite() {
            if let Some(ref x) = acc.best_state {
                record_best(archive, cfg, acc.best, x.view());
            }
        }
        return ArchiveOutcome {
            best: acc.best,
            best_state: acc.best_state,
            screens: acc.screens,
            full: acc.full,
            returned: acc.returned,
            same_floor: 0,
            floors: archive.floors.len().max(acc.basins),
            events: archive.catalog.event_count(),
            artn: acc.artn,
            charged: ledger.spent(),
            best_at: acc.best_at,
        };
    }
    // Half skip-return (LJ55 ico), then polish the live chain (LJ75 Marks).
    // Restarting hop2 from the original start is a different walk: that
    // misses the 234437 LJ75 hit a single 400k chain found after 200k skip.
    let p1 = (cap / 2).max(1).min(cap);

    let mut c_fast = cfg.clone();
    c_fast.return_screen = true;
    c_fast.return_polish = 0;
    c_fast.symmetrise_on_stall = true;

    let mut led1 = Ledger::new(p1);
    let hop1 = run_with_gradient(&c_fast, start, &mut led1, relax, grad.as_deref_mut(), rng);
    let _ = ledger.charge_many(led1.spent());
    let at1 = hop_best_at(&hop1, led1.spent());

    let rest = ledger.remaining();
    let (best, best_state, best_at, basins, screens, full, returned, artn) = if rest > 0 {
        let mut c_deep = cfg.clone();
        c_deep.return_screen = true;
        c_deep.return_polish = (cfg.relax_steps / 4).max(1);
        c_deep.symmetrise_on_stall = true;
        let mut led2 = Ledger::new(rest);
        let x2 = hop1
            .final_state
            .as_ref()
            .map(|s| s.view())
            .unwrap_or(start);
        let hop2 = run_with_gradient(&c_deep, x2, &mut led2, relax, grad.as_deref_mut(), rng);
        let _ = ledger.charge_many(led2.spent());
        let at2 = p1.saturating_add(hop_best_at(&hop2, led2.spent()));
        let (best, best_state, best_at, basins) = if hop2.best < hop1.best - 1e-12 {
            (hop2.best, hop2.best_state, at2, hop2.basins)
        } else if hop1.best < hop2.best - 1e-12 {
            (hop1.best, hop1.best_state, at1, hop1.basins)
        } else {
            let at = at1.min(at2);
            if at1 <= at2 {
                (hop1.best, hop1.best_state, at, hop1.basins)
            } else {
                (hop2.best, hop2.best_state, at, hop2.basins)
            }
        };
        (
            best,
            best_state,
            best_at,
            basins,
            hop1.screened_out + hop2.screened_out,
            hop1.hops + hop2.hops,
            hop1.returned + hop2.returned,
            hop1.symmetrised.0 + hop2.symmetrised.0 + hop1.stall_escapes + hop2.stall_escapes,
        )
    } else {
        (
            hop1.best,
            hop1.best_state,
            at1,
            hop1.basins,
            hop1.screened_out,
            hop1.hops,
            hop1.returned,
            hop1.symmetrised.0 + hop1.stall_escapes,
        )
    };
    if best.is_finite() {
        if let Some(ref x) = best_state {
            record_best(archive, cfg, best, x.view());
        }
    }
    ArchiveOutcome {
        best,
        best_state,
        screens,
        full,
        returned,
        same_floor: 0,
        floors: archive.floors.len().max(basins),
        events: archive.catalog.event_count(),
        artn,
        charged: ledger.spent(),
        best_at,
    }
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

    fn crude_relax(led: &mut Ledger, x: ArrayView1<f64>, steps: usize) -> (f64, Array1<f64>) {
        let mut y = x.to_owned();
        let mut e = lj(y.view());
        for _ in 0..steps {
            if !led.charge() {
                break;
            }
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
    }

    #[test]
    fn archive_search_does_not_touch_recommended_defaults() {
        let rec = Config::recommended(13);
        let before = format!("{rec:?}");
        assert!(
            !rec.return_screen,
            "recommended must keep return_screen off; ras enables it on a clone"
        );
        assert!(!rec.symmetrise_on_stall);
        assert_eq!(rec.screen_steps, 25);
        assert_eq!(rec.relax_steps, 200);
        let rec2 = Config::recommended(75);
        assert!(!rec2.return_screen);
        assert!(rec2.allocate_moves);
        assert!(rec2.tabu_on_stall);
        assert!(rec2.depth_reward);

        let mut rng = StdRng::seed_from_u64(0);
        let start = random_cluster(13, 0.7, rec.min_separation, &mut rng);
        let mut ledger = Ledger::new(80);
        let mut relax =
            |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| crude_relax(led, x, steps);
        let mut archive = Archive::new();
        let _ = archive_search(
            &rec,
            start.view(),
            &mut ledger,
            &mut relax,
            None,
            &mut archive,
            &mut rng,
        );
        assert_eq!(
            format!("{rec:?}"),
            before,
            "archive_search mutated the caller's recommended config"
        );
        assert!(!rec.return_screen);
        assert_eq!(rec.return_polish, 0);
        assert!(!rec.symmetrise_on_stall);
        assert_eq!(rec.screen_steps, 25);
        assert_eq!(rec.relax_steps, 200);
    }

    #[test]
    fn a_tiny_run_records_floors_and_refuses_returns() {
        let mut rng = StdRng::seed_from_u64(1);
        let cfg = Config::recommended(7);
        let start = random_cluster(7, 0.7, cfg.min_separation, &mut rng);
        let mut ledger = Ledger::new(400);
        let mut relax =
            |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| crude_relax(led, x, steps);
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
        assert!(out.floors >= 1);
    }

    #[test]
    fn small_budget_does_not_mutate_molecular_defaults() {
        let rec = Config::recommended_molecular(vec![8, 1, 1], vec![vec![0, 1, 2]], 1.0);
        let before = format!("{rec:?}");
        assert!(!rec.return_screen);
        assert_eq!(rec.return_polish, 0);
        assert_eq!(rec.escape_stall_patience, 5_000);
        assert!(!rec.symmetrise_on_stall);

        let mut rng = StdRng::seed_from_u64(3);
        let start = random_cluster(3, 0.7, rec.min_separation, &mut rng);
        let mut ledger = Ledger::new(200);
        let mut relax =
            |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| crude_relax(led, x, steps);
        let mut archive = Archive::new();
        let out = archive_search(
            &rec,
            start.view(),
            &mut ledger,
            &mut relax,
            None,
            &mut archive,
            &mut rng,
        );
        assert_eq!(
            format!("{rec:?}"),
            before,
            "archive_search mutated recommended_molecular"
        );
        assert!(!rec.return_screen);
        assert_eq!(rec.return_polish, 0);
        assert_eq!(rec.escape_stall_patience, 5_000);
        assert!(out.charged <= 200);
        assert!(out.best.is_finite());
    }

    #[test]
    fn symmetry_polish_does_not_mutate_the_caller_config() {
        let rec = Config::recommended_molecular(vec![8, 1, 1], vec![vec![0, 1, 2]], 1.0);
        let before = format!("{rec:?}");
        let mut rng = StdRng::seed_from_u64(4);
        let start = random_cluster(3, 0.7, rec.min_separation, &mut rng);
        let mut ledger = Ledger::new(rec.relax_steps + 8);
        let mut relax =
            |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| crude_relax(led, x, steps);
        let _ = symmetry_polish(&rec, start.view(), &mut ledger, &mut relax);
        assert_eq!(format!("{rec:?}"), before);
        assert!(!rec.return_screen);
        assert_eq!(rec.return_polish, 0);
        assert!(ledger.spent() <= rec.relax_steps + 8);
    }

    #[test]
    fn a_tiny_run_does_not_starve_full_quenches() {
        let mut rng = StdRng::seed_from_u64(2);
        let cfg = Config::recommended(7);
        let start = random_cluster(7, 0.7, cfg.min_separation, &mut rng);
        let mut ledger = Ledger::new(400);
        let mut relax =
            |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| crude_relax(led, x, steps);
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
        assert!(
            out.full >= 2,
            "starved: full={} screens={} returned={} charged={}",
            out.full,
            out.screens,
            out.returned,
            out.charged
        );
        assert!(out.charged <= 400);
        assert!(out.best.is_finite());
    }

    /// The second large-budget hop continues the live chain. Restarting from
    /// the original start is a different walk: v10 did that and missed the
    /// LJ75 GM that a single 400k chain found at 234437.
    #[test]
    fn large_budget_second_hop_starts_from_the_live_chain() {
        let rec = Config::recommended(7);
        let mut rng = StdRng::seed_from_u64(5);
        let start = random_cluster(7, 0.7, rec.min_separation, &mut rng);
        let start0 = start[0];
        let hop2_x0 = std::cell::Cell::new(f64::NAN);
        let opening = std::cell::Cell::new(0usize);
        let relax_steps = rec.relax_steps;
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
            if led.spent() == 0 && steps == relax_steps {
                let n = opening.get() + 1;
                opening.set(n);
                if n == 2 {
                    hop2_x0.set(x[0]);
                }
            }
            for _ in 0..steps {
                if !led.charge() {
                    break;
                }
            }
            let mut y = x.to_owned();
            y[0] += 1.0;
            (-y[0], y)
        };
        let mut ledger = Ledger::new(50_000);
        let mut archive = Archive::new();
        let out = archive_search(
            &rec,
            start.view(),
            &mut ledger,
            &mut relax,
            None,
            &mut archive,
            &mut rng,
        );
        assert!(
            opening.get() >= 2,
            "expected two opening quenches, got {}",
            opening.get()
        );
        let h2 = hop2_x0.get();
        assert!(
            h2.is_finite() && (h2 - start0).abs() > 0.5,
            "hop2 opened at {h2}, original start {start0}; want the live chain"
        );
        assert!(out.best.is_finite());
        assert_eq!(rec.return_polish, 0);
        assert!(!rec.return_screen);
    }
}
