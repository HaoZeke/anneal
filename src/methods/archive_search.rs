//! Residual archive search: local events, energy floors, cheap novelty loop.
//!
//! This is not [`Config::recommended`]. That preset keeps its measured rates.
//! The driver here buys a full quench only for a residual landing or a live
//! class record, hunts with the screen until the trial is not a return, and
//! spends ARTn only on an unsaturated local key or when the cheap loop cannot
//! open a non-return.
//!
//! Shared state is [`Archive`]. Several workers pass the same archive; pending
//! locks stop two of them paying `R` for the same floor or the same residual
//! slot.

use crate::allocate::DepthAllocator;
use crate::catalog::{Catalog, Event};
use crate::floors::FloorBook;
use crate::graphkey::contact_key;
use crate::localkey::{bag_key, local_keys};
use crate::methods::activation::{activate, Activation};
use crate::methods::cluster_hopping::{
    active_mask, connectivity_groups, connectivity_groups_z, contain, recentre, ClusterMove,
    Config, GradFn, Ledger, MoveLibrary, Relax,
};
use crate::residual_field::ResidualField;
use crate::screen::DropModel;
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use std::collections::HashSet;

/// Contact multiple for local NAUTY, matching [`crate::graphkey`] tests.
const LOCAL_CUTOFF: f64 = 1.35;

/// Cheap redraws of one hop before ARTn or a starve-gate purchase.
const MAX_REDRAW: usize = 16;

/// Fraction of stall restarts that leave `E_star` for a high-residual rep.
const DIVERSE_START: f64 = 0.25;

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
///
/// The caller’s [`Config`] is borrowed and never written. Purchase is driven
/// by the catalogue, the floor book, the drop model, and the residual field:
/// the cheap screen is redrawn until the trial is not an exact return, and
/// `R` is paid only for an uncalibrated drop, a live class, or residual `U`.
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
    let n = cfg.n_points;
    let mut kernels = cfg.move_library.kernels(cfg);
    let mut alloc = if cfg.allocate_moves {
        Some(DepthAllocator::new(kernels.len().max(1)))
    } else {
        None
    };

    let mut x = start.to_owned();
    let (e0, x0) = relax(ledger, x.view(), cfg.relax_steps);
    x = x0;
    let mut e = e0;
    let mut here: Option<usize> = None;
    if e0.is_finite() {
        here = Some(archive.floors.assign(e0, None, 0.0));
        archive.residual.observe(here.unwrap(), e0);
        archive.set_rep(here.unwrap(), x.view());
        ledger.record(e0, x.view());
        let keys = local_keys(key_coords(x.view(), cfg).view(), LOCAL_CUTOFF);
        archive.catalog.observe_bag(&keys);
    }
    let mut out = ArchiveOutcome {
        best: e0,
        best_state: if e0.is_finite() {
            Some(x.clone())
        } else {
            None
        },
        // The start always consumes one full quench slot: the ledger paid `R`.
        full: 1,
        best_at: ledger.spent(),
        ..ArchiveOutcome::default()
    };

    let mut here_coords = key_coords(x.view(), cfg);
    let mut here_key = contact_key(here_coords.view(), LOCAL_CUTOFF);
    let mut here_keys = local_keys(here_coords.view(), LOCAL_CUTOFF);
    let mut here_bag = bag_key(&here_keys);
    let mut returned_streak = 0usize;

    while ledger.remaining() > 0 {
        let s_steps = cfg.screen_steps.max(1);
        if ledger.remaining() < s_steps {
            // Leftover cannot pay a screen. One residual climb may still fit.
            let keys_artn = here_keys.clone();
            if archive.catalog.unsaturated_in(&keys_artn).is_some() {
                let _ = run_artn(
                    cfg,
                    ledger,
                    relax,
                    grad.as_deref_mut(),
                    archive,
                    &mut out,
                    &keys_artn,
                    &mut x,
                    &mut e,
                    &mut here,
                    &mut here_coords,
                    &mut here_key,
                    &mut here_keys,
                    &mut here_bag,
                    rng,
                );
            }
            break;
        }

        rebuild_molecular(cfg, x.view(), n, &mut kernels, &mut alloc);

        let e_star = ledger.best;
        let stall = here
            .and_then(|h| archive.floors.get(h))
            .is_some_and(|f| f.saturated(tau, e_star))
            || returned_streak >= MAX_REDRAW;
        if stall {
            if let Some((id, nx, ne)) = pick_diverse_start(archive, x.len(), tau, e_star, rng) {
                if here != Some(id) {
                    x = nx;
                    e = ne;
                    here = Some(id);
                    here_coords = key_coords(x.view(), cfg);
                    here_key = contact_key(here_coords.view(), LOCAL_CUTOFF);
                    here_keys = local_keys(here_coords.view(), LOCAL_CUTOFF);
                    here_bag = bag_key(&here_keys);
                    archive.catalog.observe_bag(&here_keys);
                    returned_streak = 0;
                }
            }
        }

        archive.catalog.observe_bag(&here_keys);
        // Recycle is a free topology hop, not a ledger-free outer continue:
        // apply the known landing and then hunt from there.
        let keys_now = here_keys.clone();
        let _ = try_recycle(
            cfg,
            archive,
            &keys_now,
            &mut x,
            &mut e,
            &mut here,
            &mut here_coords,
            &mut here_key,
            &mut here_keys,
            &mut here_bag,
            rng,
        );

        let cap = if archive.drop.calibrated() {
            MAX_REDRAW
        } else {
            4
        };
        let max_tries = ((ledger.remaining() / s_steps).max(1)).min(cap);
        let mut found = None;
        let mut last_screen: Option<(f64, Array1<f64>, Option<usize>, usize)> = None;
        let mut saw_nonreturn = false;

        for _ in 0..max_tries {
            if ledger.remaining() < s_steps {
                break;
            }
            let k = select_arm(cfg, &kernels, alloc.as_ref(), rng);
            let trial = propose_trial(cfg, &kernels, k, x.view(), n, rng);
            let (e_sc, x_sc) = relax(ledger, trial.view(), cfg.screen_steps);
            out.screens += 1;
            if !e_sc.is_finite() {
                continue;
            }
            let sc_coords = key_coords(x_sc.view(), cfg);
            let sc_key = contact_key(sc_coords.view(), LOCAL_CUTOFF);
            let sc_keys = local_keys(sc_coords.view(), LOCAL_CUTOFF);
            let sc_bag = bag_key(&sc_keys);
            if is_return(
                sc_key,
                here_key,
                sc_bag,
                here_bag,
                e_sc,
                e,
                cfg.screen_margin,
            ) {
                out.returned += 1;
                returned_streak = returned_streak.saturating_add(1);
                last_screen = Some((e_sc, x_sc, here, k));
                continue;
            }
            saw_nonreturn = true;
            returned_streak = 0;

            let hat = archive.drop.predicted_full(e_sc).unwrap_or(e_sc);
            let rise_hat = (hat - e).max(0.0);
            let landed = peek_floor(&archive.floors, hat, here, rise_hat);
            if pending_blocks(archive, landed) {
                continue;
            }
            last_screen = Some((e_sc, x_sc.clone(), landed, k));
            if should_buy(archive, landed, tau, ledger.best) {
                found = Some((e_sc, x_sc, landed, k));
                break;
            }
            out.same_floor += 1;
        }

        if found.is_none() && !archive.drop.calibrated() {
            // Starve gate: an uncalibrated drop model is not allowed to sit
            // on screens. Buy the last finite screen so the model sees a pair.
            found = last_screen.take();
        }

        if let Some((e_sc, x_sc, landed, arm)) = found {
            let keys_from = here_keys.clone();
            buy_full(
                cfg,
                ledger,
                relax,
                archive,
                &mut out,
                &mut alloc,
                arm,
                e_sc,
                x_sc,
                landed,
                &keys_from,
                &mut x,
                &mut e,
                &mut here,
                &mut here_coords,
                &mut here_key,
                &mut here_keys,
                &mut here_bag,
                rng,
            );
            continue;
        }

        // Residual ARTn: owed local search, or the cheap loop never left.
        let artn_due = archive.catalog.unsaturated_in(&here_keys).is_some() || !saw_nonreturn;
        if artn_due {
            let keys_artn = here_keys.clone();
            if run_artn(
                cfg,
                ledger,
                relax,
                grad.as_deref_mut(),
                archive,
                &mut out,
                &keys_artn,
                &mut x,
                &mut e,
                &mut here,
                &mut here_coords,
                &mut here_key,
                &mut here_keys,
                &mut here_bag,
                rng,
            ) {
                returned_streak = 0;
                continue;
            }
        }
    }

    out.floors = archive.floors.len();
    out.events = archive.catalog.event_count();
    out.charged = ledger.spent();
    out
}

fn should_buy(archive: &Archive, landed: Option<usize>, tau: f64, e_star: f64) -> bool {
    if !archive.drop.calibrated() {
        return true;
    }
    match landed {
        None => true,
        Some(id) => {
            let live = archive
                .floors
                .get(id)
                .map(|f| !f.saturated(tau, e_star))
                .unwrap_or(true);
            let residual_u = archive.residual.score(id) >= 0.5 * archive.residual.residual_score();
            live || residual_u
        }
    }
}

fn pending_blocks(archive: &Archive, landed: Option<usize>) -> bool {
    match landed {
        None => archive.pending_residual,
        Some(id) => archive.pending_floors.contains(&id),
    }
}

fn peek_floor(book: &FloorBook, hat: f64, here: Option<usize>, rise_hat: f64) -> Option<usize> {
    let de = book.delta_e();
    if let Some(h) = here {
        if rise_hat <= de {
            return Some(h);
        }
    }
    let mut best: Option<(usize, f64)> = None;
    for i in 0..book.len() {
        if let Some(f) = book.get(i) {
            let d = (hat - f.e_min).abs();
            if d <= de || (de == 0.0 && d == 0.0) {
                match best {
                    Some((_, bd)) if d >= bd => {}
                    _ => best = Some((i, d)),
                }
            }
        }
    }
    best.map(|(i, _)| i)
}

fn is_return(
    sc_key: u64,
    here_key: u64,
    sc_bag: u64,
    here_bag: u64,
    e_sc: f64,
    e_here: f64,
    margin: f64,
) -> bool {
    let same_topo = sc_key == here_key || sc_bag == here_bag;
    if !same_topo {
        return false;
    }
    e_sc + margin >= e_here
}

fn select_arm<R: Rng + ?Sized>(
    cfg: &Config,
    kernels: &[ClusterMove],
    alloc: Option<&DepthAllocator>,
    rng: &mut R,
) -> usize {
    if kernels.is_empty() {
        return 0;
    }
    if cfg.allocate_moves {
        if let Some(a) = alloc {
            return a.select(rng).min(kernels.len() - 1);
        }
    }
    rng.random_range(0..kernels.len())
}

fn propose_trial<R: Rng + ?Sized>(
    cfg: &Config,
    kernels: &[ClusterMove],
    k: usize,
    x: ArrayView1<f64>,
    n: usize,
    rng: &mut R,
) -> Array1<f64> {
    let mut trial = if kernels.is_empty() {
        x.to_owned()
    } else {
        kernels[k].propose_scaled(x, cfg.temperature, 1.0, rng)
    };
    match cfg.frozen.as_ref() {
        None => {
            recentre(&mut trial, n);
            contain(&mut trial, n, cfg.container);
        }
        Some(f) => {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for i in 0..n {
                if f.get(i).copied().unwrap_or(false) {
                    for d in 0..3 {
                        lo[d] = lo[d].min(trial[3 * i + d]);
                        hi[d] = hi[d].max(trial[3 * i + d]);
                    }
                }
            }
            if lo[0].is_finite() {
                for i in 0..n {
                    if !f.get(i).copied().unwrap_or(false) {
                        for d in 0..3 {
                            trial[3 * i + d] = trial[3 * i + d]
                                .clamp(lo[d] - cfg.container, hi[d] + cfg.container);
                        }
                    }
                }
            }
        }
    }
    trial
}

fn rebuild_molecular(
    cfg: &Config,
    x: ArrayView1<f64>,
    n: usize,
    kernels: &mut Vec<ClusterMove>,
    alloc: &mut Option<DepthAllocator>,
) {
    let MoveLibrary::Molecular { reactive, .. } = &cfg.move_library else {
        return;
    };
    let hop_frozen: Option<Vec<bool>> = match (&cfg.active_region, &cfg.species) {
        (Some((seeds, shells)), Some(z)) => Some(
            active_mask(x, z, seeds, *shells, cfg.bond_tolerance)
                .into_iter()
                .map(|a| !a)
                .collect(),
        ),
        _ => cfg.frozen.clone(),
    };
    let fresh = match cfg.species.as_ref() {
        Some(z) => connectivity_groups_z(x, z, cfg.bond_tolerance),
        None => connectivity_groups(x, n, cfg.covalent_cutoff),
    };
    let movable: Vec<Vec<usize>> = match hop_frozen.as_ref() {
        Some(f) => fresh
            .into_iter()
            .map(|g| {
                g.into_iter()
                    .filter(|&a| !f.get(a).copied().unwrap_or(false))
                    .collect::<Vec<usize>>()
            })
            .filter(|g: &Vec<usize>| !g.is_empty())
            .collect(),
        None => fresh,
    };
    if movable.is_empty() {
        return;
    }
    *kernels = MoveLibrary::Molecular {
        groups: movable,
        reactive: *reactive,
    }
    .kernels(cfg);
    if let Some(a) = alloc.as_ref() {
        if a.arms() != kernels.len().max(1) {
            *alloc = Some(DepthAllocator::new(kernels.len().max(1)));
        }
    }
}

fn pick_diverse_start<R: Rng + ?Sized>(
    archive: &Archive,
    xlen: usize,
    tau: f64,
    e_star: f64,
    rng: &mut R,
) -> Option<(usize, Array1<f64>, f64)> {
    if archive.reps.is_empty() {
        return None;
    }
    let residual_pick = rng.random::<f64>() < DIVERSE_START;
    let id = if residual_pick {
        let mut best = None;
        let mut best_s = f64::NEG_INFINITY;
        for i in 0..archive.floors.len() {
            if archive.pending_floors.contains(&i) {
                continue;
            }
            if i >= archive.reps.len() || archive.reps[i].len() != xlen {
                continue;
            }
            let s = archive.residual.score(i);
            if s > best_s {
                best_s = s;
                best = Some(i);
            }
        }
        best.or_else(|| archive.floors.best_start(tau, e_star))
    } else {
        archive.floors.best_start(tau, e_star)
    }?;
    if archive.pending_floors.contains(&id) {
        return None;
    }
    if id >= archive.reps.len() || archive.reps[id].len() != xlen {
        return None;
    }
    let e = archive.floors.get(id).map(|f| f.e_min)?;
    Some((id, archive.reps[id].clone(), e))
}

fn try_recycle<R: Rng + ?Sized>(
    cfg: &Config,
    archive: &mut Archive,
    keys_here: &[u64],
    x: &mut Array1<f64>,
    e: &mut f64,
    here: &mut Option<usize>,
    here_coords: &mut Array1<f64>,
    here_key: &mut u64,
    here_keys: &mut Vec<u64>,
    here_bag: &mut u64,
    rng: &mut R,
) -> bool {
    let mut dest: Option<(usize, f64)> = None;
    for &k in keys_here {
        if archive.catalog.unsaturated(k) {
            continue;
        }
        let Some(rec) = archive.catalog.get(k) else {
            continue;
        };
        for ev in &rec.events {
            for i in 0..archive.floors.len() {
                if archive.pending_floors.contains(&i) {
                    continue;
                }
                if Some(i) == *here {
                    continue;
                }
                let Some(f) = archive.floors.get(i) else {
                    continue;
                };
                if (f.e_min - ev.dest_energy).abs() > cfg.screen_margin.max(1e-6) {
                    continue;
                }
                if i < archive.reps.len() && archive.reps[i].len() == x.len() {
                    dest = Some((i, f.e_min));
                    break;
                }
            }
            if dest.is_some() {
                break;
            }
        }
        if dest.is_some() {
            break;
        }
    }
    let Some((id, dest_e)) = dest else {
        return false;
    };
    let temp = cfg.temperature.max(1e-12);
    let accept = dest_e <= *e || rng.random::<f64>() < (-(dest_e - *e) / temp).exp();
    if !accept {
        return false;
    }
    *x = archive.reps[id].clone();
    *e = dest_e;
    *here = Some(id);
    *here_coords = key_coords(x.view(), cfg);
    *here_key = contact_key(here_coords.view(), LOCAL_CUTOFF);
    *here_keys = local_keys(here_coords.view(), LOCAL_CUTOFF);
    *here_bag = bag_key(here_keys);
    archive.catalog.observe_bag(here_keys);
    true
}

fn buy_full<R: Rng + ?Sized>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    archive: &mut Archive,
    out: &mut ArchiveOutcome,
    alloc: &mut Option<DepthAllocator>,
    arm: usize,
    e_sc: f64,
    x_sc: Array1<f64>,
    landed: Option<usize>,
    keys_from: &[u64],
    x: &mut Array1<f64>,
    e: &mut f64,
    here: &mut Option<usize>,
    here_coords: &mut Array1<f64>,
    here_key: &mut u64,
    here_keys: &mut Vec<u64>,
    here_bag: &mut u64,
    rng: &mut R,
) {
    match landed {
        None => archive.pending_residual = true,
        Some(id) => {
            archive.pending_floors.insert(id);
        }
    }
    let (ef, xf) = relax(ledger, x_sc.view(), cfg.relax_steps);
    out.full += 1;
    if let Some(a) = alloc.as_mut() {
        a.update(arm, -ef);
    }
    archive.drop.observe(e_sc, ef);
    match landed {
        None => archive.pending_residual = false,
        Some(id) => {
            archive.pending_floors.remove(&id);
        }
    }
    if !ef.is_finite() {
        return;
    }
    let rise = (ef - *e).max(0.0);
    let temp = cfg.temperature.max(1e-12);
    let accept = ef <= *e || rng.random::<f64>() < (-(ef - *e) / temp).exp();
    if accept && ef > *e {
        archive.floors.observe_rise(ef - *e);
    }
    let dest = archive.floors.assign(ef, *here, rise);
    archive.residual.observe(dest, ef);
    if let (Some(h), d) = (*here, dest) {
        archive.residual.edge(h, d);
    }
    archive.set_rep(dest, xf.view());
    record_atom_events(archive, keys_from, xf.view(), cfg, ef);
    ledger.record(ef, xf.view());
    if ef < out.best {
        out.best = ef;
        out.best_state = Some(xf.clone());
        out.best_at = ledger.spent();
    }
    if accept {
        *x = xf;
        *e = ef;
        *here = Some(dest);
        *here_coords = key_coords(x.view(), cfg);
        *here_key = contact_key(here_coords.view(), LOCAL_CUTOFF);
        *here_keys = local_keys(here_coords.view(), LOCAL_CUTOFF);
        *here_bag = bag_key(here_keys);
        archive.catalog.observe_bag(here_keys);
    }
}

fn record_atom_events(
    archive: &mut Archive,
    from_keys: &[u64],
    xf: ArrayView1<f64>,
    cfg: &Config,
    ef: f64,
) {
    let to_keys = local_keys(key_coords(xf, cfg).view(), LOCAL_CUTOFF);
    let n = from_keys.len().min(to_keys.len());
    let mut recorded = false;
    for i in 0..n {
        if from_keys[i] == to_keys[i] {
            continue;
        }
        let ev = Event {
            from: from_keys[i],
            to: to_keys[i],
            dest_energy: ef,
        };
        archive.catalog.record_search(from_keys[i], Some(ev));
        recorded = true;
    }
    if !recorded {
        if let (Some(&fk), Some(&tk)) = (from_keys.first(), to_keys.first()) {
            if fk != tk {
                archive.catalog.record_search(
                    fk,
                    Some(Event {
                        from: fk,
                        to: tk,
                        dest_energy: ef,
                    }),
                );
            }
        }
    }
    archive.catalog.observe_bag(&to_keys);
}

fn run_artn<R: Rng + ?Sized>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'_>>,
    archive: &mut Archive,
    out: &mut ArchiveOutcome,
    keys_here: &[u64],
    x: &mut Array1<f64>,
    e: &mut f64,
    here: &mut Option<usize>,
    here_coords: &mut Array1<f64>,
    here_key: &mut u64,
    here_keys: &mut Vec<u64>,
    here_bag: &mut u64,
    rng: &mut R,
) -> bool {
    let Some(g) = grad else {
        return false;
    };
    let uk = archive
        .catalog
        .unsaturated_in(keys_here)
        .or_else(|| keys_here.first().copied());
    let Some(uk) = uk else {
        return false;
    };
    out.artn += 1;
    let sign = if rng.random::<bool>() { 1.0 } else { -1.0 };
    let act = Activation {
        step: cfg.escape_amplitude,
        max_steps: cfg.escape_max_climb,
        overshoot: cfg.escape_overshoot,
        lanczos_steps: cfg.escape_lanczos_steps,
        epsilon: cfg.escape_epsilon,
        ..Activation::default()
    };
    let climbed = activate(x.view(), |y| g(ledger, y), &act, sign);
    let Some(ao) = climbed else {
        archive.catalog.record_search(uk, None);
        return true;
    };
    if !ao.crossed || ledger.remaining() == 0 {
        archive.catalog.record_search(uk, None);
        return true;
    }
    let (ef, xf) = relax(ledger, ao.state.view(), cfg.relax_steps);
    out.full += 1;
    if !ef.is_finite() {
        archive.catalog.record_search(uk, None);
        return true;
    }
    let to_keys = local_keys(key_coords(xf.view(), cfg).view(), LOCAL_CUTOFF);
    let to = to_keys
        .iter()
        .copied()
        .find(|&k| k != uk)
        .or_else(|| to_keys.first().copied())
        .unwrap_or(uk);
    archive.catalog.record_search(
        uk,
        Some(Event {
            from: uk,
            to,
            dest_energy: ef,
        }),
    );
    let rise = (ef - *e).max(0.0);
    let temp = cfg.temperature.max(1e-12);
    let accept = ef <= *e || rng.random::<f64>() < (-(ef - *e) / temp).exp();
    if accept && ef > *e {
        archive.floors.observe_rise(ef - *e);
    }
    let dest = archive.floors.assign(ef, *here, rise);
    archive.residual.observe(dest, ef);
    if let (Some(h), d) = (*here, dest) {
        archive.residual.edge(h, d);
    }
    archive.set_rep(dest, xf.view());
    archive.catalog.observe_bag(&to_keys);
    ledger.record(ef, xf.view());
    if ef < out.best {
        out.best = ef;
        out.best_state = Some(xf.clone());
        out.best_at = ledger.spent();
    }
    if accept {
        *x = xf;
        *e = ef;
        *here = Some(dest);
        *here_coords = key_coords(x.view(), cfg);
        *here_key = contact_key(here_coords.view(), LOCAL_CUTOFF);
        *here_keys = local_keys(here_coords.view(), LOCAL_CUTOFF);
        *here_bag = bag_key(here_keys);
    }
    true
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
            "recommended must keep return_screen off; ras does not flip it"
        );
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
        // The residual loop records a live class, not only wrapper bookkeeping
        // of the hop best.
        assert!(out.floors >= 1);
        assert!(
            out.events > 0 || out.returned > 0,
            "expected a local event or a returned screen, got events={} returned={} full={} screens={}",
            out.events,
            out.returned,
            out.full,
            out.screens
        );
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
            "starved: full={} same_floor={} screens={} returned={} charged={}",
            out.full,
            out.same_floor,
            out.screens,
            out.returned,
            out.charged
        );
        assert!(
            out.same_floor < 40 || out.full >= 2,
            "same_floor refuse dwarfed purchase: same_floor={} full={}",
            out.same_floor,
            out.full
        );
        assert!(out.charged <= 400);
        assert!(out.best.is_finite());
    }

    #[test]
    fn drop_model_warmup_is_the_refuse_floor() {
        let d = DropModel::new();
        assert!(!d.calibrated());
        assert!(d.warmup() > 0);
        assert!(
            d.warmup() < 100,
            "warmup must come from DropModel, not a search knob"
        );
    }
}
