// Copyright (c) anneal developers
//! Published dynamic lattice search.
//!
//! Shao, Cheng and Cai, J. Comput. Chem. 25, 1693 (2004),
//! doi 10.1002/jcc.20096. The headings and equation numbers below
//! are those of that paper.
//!
//! The outer circulation is the section "Dynamic Lattice Searching
//! (DLS) Method": a random local minimum, then repeated dynamic-lattice
//! construction, dynamic-lattice searching, and local minimization.
//! A generation that does not lower the incumbent ends that run. The
//! next run starts from a fresh random local minimum. The paper repeats
//! this for `Nruns` (Table 1). Here the repeats stop when the charged
//! budget is spent, and the best minimum is kept.
//!
//! "DL Construction": recenter the minimum, read `Rmax`, and place
//! `Np` primary sites on the sphere of radius `R0 = Rmax + 0.2`. The
//! sites are one Mackay icosahedral shell, used only as a uniform set
//! of directions. The shell's topology is not read. A probe atom is
//! relaxed at each primary site with the cluster held fixed. The Mackay
//! directions are projected onto the unit sphere and then scaled to
//! `R0`. The paper states the rescaling and does not state the
//! projection. The probe stops when its root-mean-square gradient is
//! below 1e-3 or after 40 iterations. The paper does not state that
//! stop. Candidates closer than 0.05 collapse to the site nearer the
//! cluster centroid. A coordinate comparison breaks a remaining tie.
//! Atom labels are not read: the atoms are sorted by coordinates
//! before the probe and the pair sums. The pair sum over the whole
//! cluster is one charged evaluation.
//!
//! "DL Searching": the `Nmov` atoms of highest pair energy join the
//! lattice. Equations (2), (3) and (4) are the core energy, the
//! site-core energy and the site-site energy. Equation (5) is the
//! energy of one choice of `Nmov` sites. The simple greedy method
//! starts from a random choice, then moves the occupied site of
//! highest energy onto the vacant site of lowest energy while equation
//! (5) falls. The paper does not cap the number of steps. This search
//! stops after one pass over the sites. Identical occupations are
//! dropped before the quench. The paper does not say to drop them.
//! That greedy search is repeated `Ntry` times. The `Nbest` lowest
//! candidates are quenched, and the lowest quench is the generation's
//! result.
//!
//! Equation (1) is the reduced Lennard-Jones potential used here.
//! Table 1 supplies `Nmov`, `Np`, `Ntry` and `Nbest` from the cluster
//! size. A tie inside either selection cut, the `Nmov` atoms or the
//! greedy site, is broken by the lowest index or by a uniform draw.

use ndarray::ArrayView1;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::methods::cluster_hopping::{Ledger, Outcome, random_cluster_in_radius};
use crate::methods::lattice_search::{Quench, charge_oracle, merge_toward_centroid};
use crate::potentials::PairKind;

/// How a tied selection cut is broken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieCut {
    /// The lowest coordinate-ordered index among equal energies.
    ///
    /// Atoms and sites are ordered by their coordinates first, so this
    /// cut does not follow the caller's atom labels.
    LowestIndex,
    /// A uniform draw among equal energies.
    Uniform,
}

/// Table 1 parameters for one cluster size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table1 {
    /// Atoms moved in one greedy search.
    pub n_mov: usize,
    /// Primary sites on the sphere.
    pub n_primary: usize,
    /// Independent greedy starts.
    pub n_try: usize,
    /// Candidates quenched after those starts.
    pub n_best: usize,
}

/// Table 1 row that contains `n`.
///
/// Sizes above 309 use the last row of the table. The magic-number
/// footnote changes `Nruns`, which this budget replaces.
pub fn table1(n: usize) -> Table1 {
    const ROWS: &[(usize, Table1)] = &[
        (
            49,
            Table1 {
                n_mov: 10,
                n_primary: 92,
                n_try: 100,
                n_best: 4,
            },
        ),
        (
            79,
            Table1 {
                n_mov: 15,
                n_primary: 162,
                n_try: 200,
                n_best: 4,
            },
        ),
        (
            119,
            Table1 {
                n_mov: 15,
                n_primary: 252,
                n_try: 300,
                n_best: 5,
            },
        ),
        (
            149,
            Table1 {
                n_mov: 20,
                n_primary: 362,
                n_try: 350,
                n_best: 5,
            },
        ),
        (
            169,
            Table1 {
                n_mov: 20,
                n_primary: 492,
                n_try: 400,
                n_best: 5,
            },
        ),
        (
            189,
            Table1 {
                n_mov: 20,
                n_primary: 492,
                n_try: 450,
                n_best: 8,
            },
        ),
        (
            199,
            Table1 {
                n_mov: 20,
                n_primary: 492,
                n_try: 500,
                n_best: 10,
            },
        ),
        (
            229,
            Table1 {
                n_mov: 20,
                n_primary: 492,
                n_try: 550,
                n_best: 10,
            },
        ),
        (
            251,
            Table1 {
                n_mov: 20,
                n_primary: 642,
                n_try: 550,
                n_best: 10,
            },
        ),
        (
            309,
            Table1 {
                n_mov: 25,
                n_primary: 642,
                n_try: 700,
                n_best: 10,
            },
        ),
    ];
    for (limit, row) in ROWS {
        if n <= *limit {
            return *row;
        }
    }
    Table1 {
        n_mov: 40,
        n_primary: 812,
        n_try: 1000,
        n_best: 10,
    }
}

const ALPHA: f64 = 0.2;
const MERGE: f64 = 0.05;
const TIE_ABS: f64 = 0.0;

/// One published search from `start` until the ledger is spent.
///
/// `start` is the paired initial geometry. It is quenched, then the
/// circulation above runs. Later runs draw a fresh cluster in the
/// sphere of radius `radius`.
pub fn run<R: Rng + ?Sized>(
    n: usize,
    ledger: &mut Ledger,
    quench: &Quench,
    start: &[f64],
    radius: f64,
    min_separation: f64,
    rng: &mut R,
    tie: TieCut,
    target: Option<f64>,
) -> Outcome {
    let mut outcome = Outcome::default();
    if n < 2 || start.len() != 3 * n {
        outcome.best = ledger.best;
        outcome.charged = ledger.spent();
        return outcome;
    }
    let params = table1(n);
    let mut restarts = 0usize;
    let mut generations = 0usize;
    let mut improvements = 0usize;
    let mut current = match quench.relax(ledger, start) {
        Some(relaxed) => relaxed,
        None => {
            outcome.charged = ledger.spent();
            outcome.best = ledger.best;
            return outcome;
        }
    };
    ledger.record(current.energy, ArrayView1::from(&current.state));
    note(
        &mut outcome,
        &mut improvements,
        generations,
        restarts,
        ledger,
    );
    if hit(target, current.energy) {
        return finish(outcome, ledger, generations, restarts, improvements);
    }
    while ledger.remaining() > 0 {
        let improved = generation(n, &params, ledger, quench, &mut current, rng, tie, target);
        generations += 1;
        note(
            &mut outcome,
            &mut improvements,
            generations,
            restarts,
            ledger,
        );
        if hit(target, ledger.best) {
            break;
        }
        if improved {
            continue;
        }
        if ledger.remaining() == 0 {
            break;
        }
        let fresh = random_cluster_in_radius(n, radius, min_separation, rng);
        match quench.relax(ledger, fresh.as_slice().unwrap_or(&fresh.to_vec())) {
            Some(relaxed) => {
                current = relaxed;
                restarts += 1;
                ledger.record(current.energy, ArrayView1::from(&current.state));
                note(
                    &mut outcome,
                    &mut improvements,
                    generations,
                    restarts,
                    ledger,
                );
            }
            None => break,
        }
        if hit(target, ledger.best) {
            break;
        }
    }
    finish(outcome, ledger, generations, restarts, improvements)
}

fn finish(
    mut outcome: Outcome,
    ledger: &Ledger,
    generations: usize,
    restarts: usize,
    improvements: usize,
) -> Outcome {
    outcome.best = ledger.best;
    outcome.best_state = ledger.best_state.clone();
    outcome.charged = ledger.spent();
    outcome.hops = generations;
    outcome.basins = restarts;
    outcome.returned = improvements;
    outcome
}

fn note(
    outcome: &mut Outcome,
    improvements: &mut usize,
    generations: usize,
    restarts: usize,
    ledger: &Ledger,
) {
    let previous = outcome.improvements.last().map(|row| row.3);
    let dropped = previous.is_none_or(|energy| ledger.best < energy - 1e-12);
    if dropped && ledger.best.is_finite() {
        *improvements += 1;
        if outcome.improvements.len() < 512 {
            outcome
                .improvements
                .push((generations, ledger.spent(), restarts, ledger.best));
        }
    }
}

fn hit(target: Option<f64>, energy: f64) -> bool {
    target.is_some_and(|level| energy <= level)
}

fn generation<R: Rng + ?Sized>(
    n: usize,
    params: &Table1,
    ledger: &mut Ledger,
    quench: &Quench,
    current: &mut crate::methods::lattice_search::Relaxed,
    rng: &mut R,
    tie: TieCut,
    target: Option<f64>,
) -> bool {
    let before = current.energy;
    let Some((core, sites)) = build_lattice(n, params, ledger, &current.state, rng, tie) else {
        return false;
    };
    let n_mov = params
        .n_mov
        .min(sites.len())
        .min(n.saturating_sub(1).max(1));
    if n_mov == 0 || sites.len() < n_mov {
        return false;
    }
    let Some((edlc, edl)) = pair_tables(n, ledger, &core, &sites) else {
        return false;
    };
    let mut ranked: Vec<(f64, Vec<usize>)> = Vec::new();
    for _ in 0..params.n_try {
        if ledger.remaining() == 0 {
            break;
        }
        let chosen = greedy(&edlc, &edl, n_mov, rng, tie);
        let energy = selection_energy(&edlc, &edl, &chosen);
        consider(&mut ranked, params.n_best, energy, chosen);
    }
    let mut best_local: Option<crate::methods::lattice_search::Relaxed> = None;
    for (_, chosen) in &ranked {
        if ledger.remaining() == 0 {
            break;
        }
        let trial = assemble(&core, &sites, chosen);
        let Some(relaxed) = quench.relax(ledger, &trial) else {
            break;
        };
        ledger.record(relaxed.energy, ArrayView1::from(&relaxed.state));
        let replace = best_local
            .as_ref()
            .is_none_or(|held| relaxed.energy < held.energy);
        if replace {
            best_local = Some(relaxed);
        }
        if hit(target, ledger.best) {
            break;
        }
    }
    let Some(best_local) = best_local else {
        return false;
    };
    if best_local.energy < before - 1e-12 {
        *current = best_local;
        true
    } else {
        false
    }
}

fn build_lattice<R: Rng + ?Sized>(
    n: usize,
    params: &Table1,
    ledger: &mut Ledger,
    x: &[f64],
    rng: &mut R,
    tie: TieCut,
) -> Option<(Vec<[f64; 3]>, Vec<[f64; 3]>)> {
    let mut atoms: Vec<[f64; 3]> = (0..n).map(|i| point(x, i)).collect();
    atoms.sort_by(|a, b| coord_cmp(*a, *b));
    let centre = centroid(&atoms);
    let rmax = atoms
        .iter()
        .map(|p| dist(*p, centre))
        .fold(0.0_f64, f64::max);
    let radius = rmax + ALPHA;
    let shell = mackay_shell_count(params.n_primary);
    let mut primary = Vec::with_capacity(shell.len());
    for direction in shell {
        primary.push(add(centre, scale(direction, radius)));
    }
    let mut candidates = Vec::new();
    for site in primary {
        let placed = relax_probe(n, ledger, &atoms, site)?;
        if placed.iter().all(|v| v.is_finite()) {
            candidates.push(placed);
        }
    }
    let vacant = merge_toward_centroid(&candidates, centre, MERGE);
    // One full pair sum: N(N-1)/2 pairs, charged as one evaluation.
    if !charge_oracle(ledger, 1.0) {
        return None;
    }
    let sub = atom_energies(&atoms);
    let active = select_highest(&sub, params.n_mov.min(n), tie, rng);
    let mut sites = vacant;
    let mut core = Vec::new();
    for i in 0..n {
        if active.contains(&i) {
            sites.push(atoms[i]);
        } else {
            core.push(atoms[i]);
        }
    }
    Some((core, sites))
}

fn pair_tables(
    n: usize,
    ledger: &mut Ledger,
    core: &[[f64; 3]],
    sites: &[[f64; 3]],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n_core = core.len();
    let n_sites = sites.len();
    let full = ((n * n.saturating_sub(1)) / 2).max(1) as f64;
    // The core energy is a constant of the generation. It is not
    // evaluated here, so it is not charged. The site rows are.
    let n_edlc = n_sites * n_core;
    let n_edl = (n_sites * n_sites.saturating_sub(1)) / 2;
    let frac = (n_edlc + n_edl) as f64 / full;
    if !charge_oracle(ledger, frac) {
        return None;
    }
    let mut edlc = vec![0.0; n_sites];
    for (i, site) in sites.iter().enumerate() {
        for core_atom in core {
            edlc[i] += lj_pair(dist2(*site, *core_atom));
        }
    }
    let mut edl = vec![0.0; n_sites * n_sites];
    for i in 0..n_sites {
        for j in (i + 1)..n_sites {
            let value = lj_pair(dist2(sites[i], sites[j]));
            edl[i * n_sites + j] = value;
            edl[j * n_sites + i] = value;
        }
    }
    Some((edlc, edl))
}

fn greedy<R: Rng + ?Sized>(
    edlc: &[f64],
    edl: &[f64],
    n_mov: usize,
    rng: &mut R,
    tie: TieCut,
) -> Vec<usize> {
    let n_sites = edlc.len();
    let mut chosen = sample(n_sites, n_mov, rng);
    let mut energy = selection_energy(edlc, edl, &chosen);
    // The paper ends the sweep when the energy stops falling. The
    // iteration count is capped at the number of sites.
    for _ in 0..n_sites.max(1) {
        let occupied = site_terms(edlc, edl, &chosen, None);
        let from = pick(true, &occupied, tie, rng);
        let vacant_pool: Vec<usize> = (0..n_sites).filter(|i| !chosen.contains(i)).collect();
        if vacant_pool.is_empty() {
            break;
        }
        let mut vacant_terms = Vec::with_capacity(vacant_pool.len());
        for site in vacant_pool {
            let value = term(edlc, edl, site, &chosen, Some(from));
            vacant_terms.push((site, value));
        }
        let to = pick(false, &vacant_terms, tie, rng);
        let mut next = chosen.clone();
        if let Some(slot) = next.iter().position(|site| *site == from) {
            next[slot] = to;
        }
        next.sort_unstable();
        let next_energy = selection_energy(edlc, edl, &next);
        if next_energy < energy - 1e-12 {
            chosen = next;
            energy = next_energy;
        } else {
            break;
        }
    }
    chosen
}

fn consider(ranked: &mut Vec<(f64, Vec<usize>)>, n_best: usize, energy: f64, chosen: Vec<usize>) {
    if ranked.iter().any(|(_, held)| held == &chosen) {
        return;
    }
    ranked.push((energy, chosen));
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    ranked.truncate(n_best);
}

fn selection_energy(edlc: &[f64], edl: &[f64], chosen: &[usize]) -> f64 {
    let mut energy = 0.0;
    for (a, &i) in chosen.iter().enumerate() {
        energy += edlc[i];
        for &j in &chosen[a + 1..] {
            energy += edl[i * edlc.len() + j];
        }
    }
    energy
}

fn site_terms(
    edlc: &[f64],
    edl: &[f64],
    chosen: &[usize],
    skip: Option<usize>,
) -> Vec<(usize, f64)> {
    chosen
        .iter()
        .filter(|site| Some(**site) != skip)
        .map(|&site| (site, term(edlc, edl, site, chosen, Some(site))))
        .collect()
}

fn term(edlc: &[f64], edl: &[f64], site: usize, chosen: &[usize], skip: Option<usize>) -> f64 {
    let mut energy = edlc[site];
    let width = edlc.len();
    for &other in chosen {
        if Some(other) == skip || other == site {
            continue;
        }
        energy += edl[site * width + other];
    }
    energy
}

fn pick<R: Rng + ?Sized>(highest: bool, terms: &[(usize, f64)], tie: TieCut, rng: &mut R) -> usize {
    let extreme = terms.iter().fold(terms[0].1, |held, (_, energy)| {
        if highest {
            held.max(*energy)
        } else {
            held.min(*energy)
        }
    });
    let mut pool: Vec<usize> = terms
        .iter()
        .filter(|(_, energy)| {
            (*energy - extreme).abs() <= TIE_ABS || energy.total_cmp(&extreme).is_eq()
        })
        .map(|(site, _)| *site)
        .collect();
    pool.sort_unstable();
    match tie {
        TieCut::LowestIndex => pool[0],
        TieCut::Uniform => pool[rng.random_range(0..pool.len())],
    }
}

#[cfg(test)]
fn extreme_pool(highest: bool, terms: &[(usize, f64)]) -> Vec<usize> {
    let extreme = terms.iter().fold(terms[0].1, |held, (_, energy)| {
        if highest {
            held.max(*energy)
        } else {
            held.min(*energy)
        }
    });
    let mut pool: Vec<usize> = terms
        .iter()
        .filter(|(_, energy)| {
            (*energy - extreme).abs() <= TIE_ABS || energy.total_cmp(&extreme).is_eq()
        })
        .map(|(site, _)| *site)
        .collect();
    pool.sort_unstable();
    pool
}

/// Probabilities of the first greedy exchange, keyed by site coordinates.
#[cfg(test)]
fn first_exchange_probabilities<R: Rng + ?Sized>(
    edlc: &[f64],
    edl: &[f64],
    sites: &[[f64; 3]],
    n_mov: usize,
    rng: &mut R,
    tie: TieCut,
) -> Vec<([f64; 3], [f64; 3], f64)> {
    let chosen = sample(sites.len(), n_mov, rng);
    let occupied = site_terms(edlc, edl, &chosen, None);
    let from_pool = extreme_pool(true, &occupied);
    let mut out = Vec::new();
    for &from in &from_pool {
        let p_from = match tie {
            TieCut::LowestIndex => {
                if from == from_pool[0] {
                    1.0
                } else {
                    0.0
                }
            }
            TieCut::Uniform => 1.0 / from_pool.len() as f64,
        };
        if p_from == 0.0 {
            continue;
        }
        let vacant_pool: Vec<usize> = (0..sites.len()).filter(|i| !chosen.contains(i)).collect();
        let mut vacant_terms = Vec::with_capacity(vacant_pool.len());
        for site in vacant_pool {
            vacant_terms.push((site, term(edlc, edl, site, &chosen, Some(from))));
        }
        let to_pool = extreme_pool(false, &vacant_terms);
        for &to in &to_pool {
            let p_to = match tie {
                TieCut::LowestIndex => {
                    if to == to_pool[0] {
                        1.0
                    } else {
                        0.0
                    }
                }
                TieCut::Uniform => 1.0 / to_pool.len() as f64,
            };
            if p_to == 0.0 {
                continue;
            }
            out.push((sites[from], sites[to], p_from * p_to));
        }
    }
    out
}

fn sample<R: Rng + ?Sized>(n_sites: usize, n_mov: usize, rng: &mut R) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n_sites).collect();
    order.shuffle(rng);
    order.truncate(n_mov);
    order.sort_unstable();
    order
}

fn select_highest<R: Rng + ?Sized>(
    energies: &[f64],
    count: usize,
    tie: TieCut,
    rng: &mut R,
) -> Vec<usize> {
    let count = count.min(energies.len());
    if count == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..energies.len()).collect();
    order.sort_by(|&a, &b| energies[b].total_cmp(&energies[a]).then(a.cmp(&b)));
    if tie == TieCut::LowestIndex {
        order.truncate(count);
        return order;
    }
    let cutoff = energies[order[count - 1]];
    let mut locked: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&i| energies[i] > cutoff)
        .collect();
    let mut pool: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&i| energies[i].total_cmp(&cutoff).is_eq())
        .collect();
    pool.shuffle(rng);
    let need = count - locked.len();
    locked.extend(pool.into_iter().take(need));
    locked
}

fn assemble(core: &[[f64; 3]], sites: &[[f64; 3]], chosen: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(3 * (core.len() + chosen.len()));
    for p in core {
        out.extend_from_slice(p);
    }
    for &site in chosen {
        out.extend_from_slice(&sites[site]);
    }
    out
}

fn relax_probe(
    n: usize,
    ledger: &mut Ledger,
    fixed: &[[f64; 3]],
    start: [f64; 3],
) -> Option<[f64; 3]> {
    let full = ((n * n.saturating_sub(1)) / 2).max(1) as f64;
    let frac = (n as f64) / full;
    let mut x = start;
    let mut g = [0.0; 3];
    let mut e = probe(fixed, x, &mut g);
    if !charge_oracle(ledger, frac) {
        return None;
    }
    let mut s_hist: Vec<[f64; 3]> = Vec::new();
    let mut y_hist: Vec<[f64; 3]> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();
    for _ in 0..40 {
        let rms = (g.iter().map(|v| v * v).sum::<f64>() / 3.0).sqrt();
        if rms < 1e-3 || !e.is_finite() {
            break;
        }
        let mut d = g;
        let mut alpha = Vec::with_capacity(s_hist.len());
        for k in (0..s_hist.len()).rev() {
            let a = rho_hist[k] * dot3(s_hist[k], d);
            alpha.push(a);
            d = sub3(d, scale(y_hist[k], a));
        }
        alpha.reverse();
        let gamma = if let (Some(s), Some(y)) = (s_hist.last(), y_hist.last()) {
            let ys = dot3(*y, *s);
            let yy = dot3(*y, *y);
            if yy > 1e-16 { ys / yy } else { 0.1 }
        } else {
            0.1
        };
        d = scale(d, gamma);
        for k in 0..s_hist.len() {
            let b = rho_hist[k] * dot3(y_hist[k], d);
            d = add3(d, scale(s_hist[k], alpha[k] - b));
        }
        d = scale(d, -1.0);
        if dot3(d, g) > 0.0 {
            d = scale(g, -1.0);
        }
        let mut step = 0.2;
        let mut accepted = None;
        for _ in 0..8 {
            let trial = add3(x, scale(d, step));
            if !charge_oracle(ledger, frac) {
                return Some(x);
            }
            let mut gn = [0.0; 3];
            let en = probe(fixed, trial, &mut gn);
            if en.is_finite() && en <= e {
                accepted = Some((en, trial, gn));
                break;
            }
            step *= 0.4;
        }
        let Some((en, trial, gn)) = accepted else {
            break;
        };
        let s = sub3(trial, x);
        let y = sub3(gn, g);
        let ys = dot3(y, s);
        if ys > 1e-12 {
            if s_hist.len() == 5 {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / ys);
        }
        x = trial;
        e = en;
        g = gn;
    }
    Some(x)
}

fn probe(fixed: &[[f64; 3]], p: [f64; 3], g: &mut [f64; 3]) -> f64 {
    let mut energy = 0.0;
    *g = [0.0; 3];
    for q in fixed {
        let d = sub3(p, *q);
        let r2 = dot3(d, d);
        if r2 < 1e-16 {
            return f64::INFINITY;
        }
        let (value, coef) = PairKind::LennardJones.pair(r2);
        energy += value;
        *g = add3(*g, scale(d, coef));
    }
    energy
}

fn atom_energies(atoms: &[[f64; 3]]) -> Vec<f64> {
    let n = atoms.len();
    let mut energy = vec![0.0; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let value = lj_pair(dist2(atoms[i], atoms[j]));
            energy[i] += value;
            energy[j] += value;
        }
    }
    energy
}

fn lj_pair(r2: f64) -> f64 {
    PairKind::LennardJones.pair(r2).0
}

fn coord_cmp(a: [f64; 3], b: [f64; 3]) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0])
        .then(a[1].total_cmp(&b[1]))
        .then(a[2].total_cmp(&b[2]))
}

fn point(x: &[f64], i: usize) -> [f64; 3] {
    [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
}

fn centroid(atoms: &[[f64; 3]]) -> [f64; 3] {
    let mut c = [0.0; 3];
    if atoms.is_empty() {
        return c;
    }
    for p in atoms {
        c = add3(c, *p);
    }
    scale(c, 1.0 / atoms.len() as f64)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    dist2(a, b).sqrt()
}

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub3(a, b);
    dot3(d, d)
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    add3(a, b)
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Unit directions of the Mackay shell whose count is `count`.
///
/// Counts 92, 162, 252, 362, 492, 642 and 812 are the shells named
/// under "DL Construction".
pub fn mackay_shell_count(count: usize) -> Vec<[f64; 3]> {
    let n2 = count.saturating_sub(2) / 10;
    let shell = (n2 as f64).sqrt().round() as i32;
    mackay_shell(shell.max(1))
}

fn mackay_shell(shell: i32) -> Vec<[f64; 3]> {
    let vertices = icosahedron();
    let faces = icosa_faces(&vertices);
    let mut raw = Vec::new();
    if shell <= 1 {
        raw.extend_from_slice(&vertices);
    } else {
        let s = shell as f64;
        for face in faces {
            let (a, b, c) = (vertices[face[0]], vertices[face[1]], vertices[face[2]]);
            for i in 0..=shell {
                for j in 0..=(shell - i) {
                    let k = shell - i - j;
                    let p = add3(
                        add3(scale(a, i as f64 / s), scale(b, j as f64 / s)),
                        scale(c, k as f64 / s),
                    );
                    raw.push(p);
                }
            }
        }
    }
    let mut unique = Vec::new();
    for p in raw {
        let norm = dot3(p, p).sqrt();
        if norm < 1e-12 {
            continue;
        }
        let u = scale(p, 1.0 / norm);
        if unique.iter().all(|q| dist2(u, *q) > 1e-10) {
            unique.push(u);
        }
    }
    unique
}

fn icosahedron() -> [[f64; 3]; 12] {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut out = [[0.0; 3]; 12];
    let mut slot = 0;
    for cycle in 0..3 {
        for s1 in [-1.0, 1.0] {
            for s2 in [-1.0, 1.0] {
                let mut p = [0.0; 3];
                p[(cycle + 1) % 3] = s1;
                p[(cycle + 2) % 3] = s2 * phi;
                out[slot] = p;
                slot += 1;
            }
        }
    }
    out
}

fn icosa_faces(vertices: &[[f64; 3]; 12]) -> Vec<[usize; 3]> {
    let mut neighbors = vec![Vec::new(); 12];
    for i in 0..12 {
        for j in (i + 1)..12 {
            if (dist2(vertices[i], vertices[j]) - 4.0).abs() < 1e-6 {
                neighbors[i].push(j);
                neighbors[j].push(i);
            }
        }
    }
    let mut faces = Vec::new();
    for i in 0..12 {
        for &j in &neighbors[i] {
            for &k in &neighbors[i] {
                if j < k && neighbors[j].contains(&k) {
                    let mut face = [i, j, k];
                    face.sort_unstable();
                    if !faces.contains(&face) {
                        faces.push(face);
                    }
                }
            }
        }
    }
    faces
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// One lattice build charges the full atom pair sum.
    ///
    /// Without that charge the spent count is short by one. The
    /// assertion fails on the tree that skips it.
    #[test]
    fn the_atom_pair_sum_is_charged() {
        let n = 13;
        let mut rng = StdRng::seed_from_u64(11);
        let start = random_cluster_in_radius(n, 2.2, 0.5, &mut rng);
        let params = table1(n);
        let mut ledger = Ledger::new(5_000_000);
        let mut draw = StdRng::seed_from_u64(12);
        let built = build_lattice(
            n,
            &params,
            &mut ledger,
            start.as_slice().unwrap(),
            &mut draw,
            TieCut::LowestIndex,
        );
        assert!(built.is_some(), "the lattice was not built");
        assert!(
            ledger.spent() >= 654,
            "the atom pair sum was not charged: spent {}",
            ledger.spent()
        );
    }

    #[test]
    fn mackay_shells_match_the_published_counts() {
        for (shell, count) in [(3, 92), (4, 162), (5, 252), (6, 362), (8, 642)] {
            assert_eq!(mackay_shell(shell).len(), count, "shell {shell}");
        }
    }

    #[test]
    fn table1_uses_the_80_to_119_row_at_104() {
        let row = table1(104);
        assert_eq!(row.n_mov, 15);
        assert_eq!(row.n_primary, 252);
        assert_eq!(row.n_try, 300);
        assert_eq!(row.n_best, 5);
    }

    #[test]
    fn a_thirteen_atom_run_reaches_the_icosahedron() {
        let n = 13;
        let mut rng = StdRng::seed_from_u64(13);
        let start = random_cluster_in_radius(n, 2.2, 0.5, &mut rng);
        let quench = Quench {
            max_iterations: 200,
            rms_tolerance: 1.0e-4,
            ..Quench::default()
        };
        let mut ledger = Ledger::new(80_000);
        let outcome = run(
            n,
            &mut ledger,
            &quench,
            start.as_slice().unwrap(),
            2.2,
            0.5,
            &mut rng,
            TieCut::LowestIndex,
            Some(-44.3),
        );
        assert!(ledger.spent() > 0, "the search was not charged");
        assert!(
            outcome.best < -44.3,
            "best {} did not reach the thirteen-atom icosahedron",
            outcome.best
        );
    }

    fn relabel(x: &[f64], order: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        for (slot, &atom) in order.iter().enumerate() {
            out[3 * slot..3 * slot + 3].copy_from_slice(&x[3 * atom..3 * atom + 3]);
        }
        out
    }

    fn same_sites(left: &[[f64; 3]], right: &[[f64; 3]]) -> bool {
        if left.len() != right.len() {
            return false;
        }
        left.iter().zip(right.iter()).all(|(p, q)| {
            (p[0] - q[0]).abs() <= 1e-10
                && (p[1] - q[1]).abs() <= 1e-10
                && (p[2] - q[2]).abs() <= 1e-10
        })
    }

    fn same_law(left: &[([f64; 3], [f64; 3], f64)], right: &[([f64; 3], [f64; 3], f64)]) -> bool {
        if left.len() != right.len() {
            return false;
        }
        left.iter().zip(right.iter()).all(|(p, q)| {
            (p.0[0] - q.0[0]).abs() <= 1e-10
                && (p.0[1] - q.0[1]).abs() <= 1e-10
                && (p.0[2] - q.0[2]).abs() <= 1e-10
                && (p.1[0] - q.1[0]).abs() <= 1e-10
                && (p.1[1] - q.1[1]).abs() <= 1e-10
                && (p.1[2] - q.1[2]).abs() <= 1e-10
                && (p.2 - q.2).abs() <= 1e-10
        })
    }

    fn lattice_law(
        x: &[f64],
        tie: TieCut,
        build_seed: u64,
        draw_seed: u64,
    ) -> (Vec<[f64; 3]>, Vec<([f64; 3], [f64; 3], f64)>) {
        let n = x.len() / 3;
        let params = table1(n);
        let mut ledger = Ledger::new(2_000_000);
        let mut rng = StdRng::seed_from_u64(build_seed);
        let (core, sites) = build_lattice(n, &params, &mut ledger, x, &mut rng, tie)
            .expect("the probe fits in the ledger");
        let (edlc, edl) = pair_tables(n, &mut ledger, &core, &sites).expect("pair tables");
        let n_mov = params.n_mov.min(sites.len());
        let mut draw = StdRng::seed_from_u64(draw_seed);
        let law = first_exchange_probabilities(&edlc, &edl, &sites, n_mov, &mut draw, tie);
        (sites, law)
    }

    /// Relabelling a minimum must not move a lattice site or a greedy draw.
    #[test]
    fn lattice_sites_and_move_probabilities_ignore_labels() {
        use rand::seq::SliceRandom;
        let quench = Quench {
            max_iterations: 80,
            rms_tolerance: 1.0e-3,
            ..Quench::default()
        };
        let mut minima = Vec::new();
        for seed in [11_u64, 29, 47] {
            let n = 13;
            let mut rng = StdRng::seed_from_u64(seed);
            let start = random_cluster_in_radius(n, 2.2, 0.5, &mut rng);
            let mut ledger = Ledger::new(40_000);
            let relaxed = quench
                .relax(&mut ledger, start.as_slice().unwrap())
                .expect("quench");
            minima.push(relaxed.state);
        }
        for tie in [TieCut::LowestIndex, TieCut::Uniform] {
            for state in &minima {
                let n = state.len() / 3;
                let (sites, law) = lattice_law(state, tie, 1, 2);
                let mut orders = vec![(0..n).rev().collect::<Vec<_>>()];
                let mut rotated: Vec<usize> = (0..n).collect();
                rotated.rotate_left(n / 3);
                orders.push(rotated);
                let mut shuffled: Vec<usize> = (0..n).collect();
                shuffled.shuffle(&mut StdRng::seed_from_u64(5));
                orders.push(shuffled);
                for order in orders {
                    let moved = relabel(state, &order);
                    let (sites_b, law_b) = lattice_law(&moved, tie, 1, 2);
                    assert!(
                        same_sites(&sites, &sites_b),
                        "lattice sites moved under a relabelling"
                    );
                    assert!(
                        same_law(&law, &law_b),
                        "move probabilities moved under a relabelling"
                    );
                }
            }
        }
    }
}
