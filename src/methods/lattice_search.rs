//! Dynamic lattice search for pair-potential clusters.
//!
//! Shao, Cheng and Cai (*J. Comput. Chem.* **2004**, *25*, 1693,
//! <https://doi.org/10.1002/jcc.20096>) reach the hard Lennard-Jones sizes at
//! a few thousand local minimizations per hit by refusing to walk: from a
//! quenched structure they read the lattice of hollow sites the structure
//! itself defines, move the worst-bound atom to the best vacant site until
//! no such move lowers the pair energy, relax once, and restart from a fresh
//! random cluster when the construction stops improving. The lattice is a
//! function of the current structure, so nothing here is a template.
//!
//! Site energies are single-atom pair sums, one `N`th of a full evaluation,
//! and are charged to the ledger at that fraction so the reported cost is
//! comparable with every other method in the crate.

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::methods::cluster_hopping::{Ledger, Outcome, Relax, random_cluster};
use crate::potentials::PairKind;

/// Settings of one dynamic lattice search.
#[derive(Debug, Clone, Copy)]
pub struct LatticeSearchConfig {
    /// Points in the cluster.
    pub n_points: usize,
    /// The pair potential the site energies are read from.
    pub kind: PairKind,
    /// Separation below which two points are neighbours, in the potential's
    /// length units.
    pub neighbour_cutoff: f64,
    /// Relaxation steps per quench.
    pub relax_steps: usize,
    /// Constructions that fail to lower the energy before a restart.
    pub patience: usize,
    /// Cap on site moves in one construction.
    pub max_moves: usize,
    /// Minimum separation of a fresh random start.
    pub min_separation: f64,
}

impl LatticeSearchConfig {
    /// Settings for a reduced Lennard-Jones cluster of `n` points.
    pub fn lennard_jones(n: usize) -> Self {
        Self {
            n_points: n,
            kind: PairKind::LennardJones,
            neighbour_cutoff: 1.35 * PairKind::LennardJones.r_min(),
            relax_steps: 200,
            patience: 4,
            max_moves: 4 * n,
            min_separation: 0.5,
        }
    }

    /// Settings for a Morse cluster at range `rho`.
    pub fn morse(n: usize, rho: f64) -> Self {
        let kind = PairKind::Morse { rho };
        Self {
            n_points: n,
            kind,
            neighbour_cutoff: 1.35 * kind.r_min(),
            relax_steps: 200,
            patience: 4,
            max_moves: 4 * n,
            min_separation: 0.5,
        }
    }
}

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])
}

fn point(x: &[f64], i: usize) -> [f64; 3] {
    [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
}

/// Pair energy of a point at `p` in the field of every point of `x` except
/// `skip`.
fn site_energy(kind: PairKind, x: &[f64], skip: Option<usize>, p: [f64; 3]) -> f64 {
    let n = x.len() / 3;
    let mut e = 0.0;
    for j in 0..n {
        if Some(j) == skip {
            continue;
        }
        let r2 = dist2(p, point(x, j));
        if r2 > 1e-12 {
            e += kind.pair(r2).0;
        }
    }
    e
}

/// Hollow sites of `x`: apex positions over every triangle of mutual
/// neighbours, on the side away from the centroid, at the bond length from
/// all three, that overlap no point and no site already listed.
pub fn hollow_sites(x: &[f64], neighbour_cutoff: f64) -> Vec<[f64; 3]> {
    match median_bond(x) {
        Some(bond) => hollow_sites_with_bond(x, neighbour_cutoff, bond),
        None => Vec::new(),
    }
}

/// Median nearest-neighbour distance of `x`, if it has two points.
fn median_bond(x: &[f64]) -> Option<f64> {
    let n = x.len() / 3;
    let mut nearest = vec![f64::INFINITY; n];
    for a in 0..n {
        for b in (a + 1)..n {
            let d2 = dist2(point(x, a), point(x, b));
            nearest[a] = nearest[a].min(d2);
            nearest[b] = nearest[b].min(d2);
        }
    }
    let mut nn: Vec<f64> = nearest
        .iter()
        .filter(|d| d.is_finite())
        .map(|d| d.sqrt())
        .collect();
    if nn.is_empty() {
        return None;
    }
    nn.sort_by(|a, b| a.total_cmp(b));
    Some(nn[nn.len() / 2])
}

/// [`hollow_sites`] at a given bond length, for point sets whose own
/// nearest-neighbour distance is not the bond, such as a layer of
/// mutually exclusive sites.
pub fn hollow_sites_with_bond(x: &[f64], neighbour_cutoff: f64, bond: f64) -> Vec<[f64; 3]> {
    let n = x.len() / 3;
    let cut2 = neighbour_cutoff * neighbour_cutoff;
    let mut nb: Vec<Vec<usize>> = vec![Vec::new(); n];
    for a in 0..n {
        for b in (a + 1)..n {
            if dist2(point(x, a), point(x, b)) < cut2 {
                nb[a].push(b);
                nb[b].push(a);
            }
        }
    }
    let exclusion2 = (0.85 * bond) * (0.85 * bond);
    let merge2 = (0.3 * bond) * (0.3 * bond);
    let mut c = [0.0_f64; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k] / n as f64;
        }
    }
    let mut sites: Vec<[f64; 3]> = Vec::new();
    for a in 0..n {
        for &b in &nb[a] {
            if b <= a {
                continue;
            }
            for &d in &nb[b] {
                if d <= b || !nb[a].contains(&d) {
                    continue;
                }
                let (pa, pb, pd) = (point(x, a), point(x, b), point(x, d));
                let centre = [
                    (pa[0] + pb[0] + pd[0]) / 3.0,
                    (pa[1] + pb[1] + pd[1]) / 3.0,
                    (pa[2] + pb[2] + pd[2]) / 3.0,
                ];
                let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let v = [pd[0] - pa[0], pd[1] - pa[1], pd[2] - pa[2]];
                let normal = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let height2 = bond * bond - dist2(centre, pa);
                push_site(
                    x, c, centre, normal, height2, exclusion2, merge2, &mut sites,
                );
            }
        }
    }
    // Four-fold sites over squares: two second neighbours at about the
    // square diagonal that share two first neighbours which are themselves
    // a diagonal apart, as on a (100) facet.
    let diag_lo = 1.3 * bond;
    let diag_hi = 1.5 * bond;
    for a in 0..n {
        for b in (a + 1)..n {
            let dab = dist2(point(x, a), point(x, b)).sqrt();
            if !(diag_lo..diag_hi).contains(&dab) {
                continue;
            }
            let common: Vec<usize> = nb[a]
                .iter()
                .copied()
                .filter(|e| nb[b].contains(e))
                .collect();
            for (i, &p) in common.iter().enumerate() {
                for &q in &common[i + 1..] {
                    let dpq = dist2(point(x, p), point(x, q)).sqrt();
                    if !(diag_lo..diag_hi).contains(&dpq) {
                        continue;
                    }
                    let (pa, pb, pp, pq) = (point(x, a), point(x, b), point(x, p), point(x, q));
                    let centre = [
                        (pa[0] + pb[0] + pp[0] + pq[0]) / 4.0,
                        (pa[1] + pb[1] + pp[1] + pq[1]) / 4.0,
                        (pa[2] + pb[2] + pp[2] + pq[2]) / 4.0,
                    ];
                    let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let v = [pq[0] - pp[0], pq[1] - pp[1], pq[2] - pp[2]];
                    let normal = [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ];
                    let height2 = bond * bond - dist2(centre, pa);
                    push_site(
                        x, c, centre, normal, height2, exclusion2, merge2, &mut sites,
                    );
                }
            }
        }
    }
    sites
}

/// Place a site at `height2.sqrt()` above `centre` along `normal`, on the
/// side away from the centroid `c`, unless it overlaps a point or a site.
#[allow(clippy::too_many_arguments)]
fn push_site(
    x: &[f64],
    c: [f64; 3],
    centre: [f64; 3],
    normal: [f64; 3],
    height2: f64,
    exclusion2: f64,
    merge2: f64,
    sites: &mut Vec<[f64; 3]>,
) {
    let n = x.len() / 3;
    let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if norm < 1e-12 || height2 <= 0.0 {
        return;
    }
    let height = height2.sqrt();
    let outward = if (centre[0] - c[0]) * normal[0]
        + (centre[1] - c[1]) * normal[1]
        + (centre[2] - c[2]) * normal[2]
        >= 0.0
    {
        1.0
    } else {
        -1.0
    };
    let site = [
        centre[0] + outward * height * normal[0] / norm,
        centre[1] + outward * height * normal[1] / norm,
        centre[2] + outward * height * normal[2] / norm,
    ];
    if (0..n).any(|e| dist2(site, point(x, e)) < exclusion2) {
        return;
    }
    if sites.iter().any(|s| dist2(*s, site) < merge2) {
        return;
    }
    sites.push(site);
}

/// Occupation optimisation: at every step the lattice is rebuilt from the
/// current structure, the three worst-bound points are each tried against
/// every vacant site, and the single relocation with the largest drop in
/// that point's energy is taken; the sweep ends when no relocation lowers
/// any of them. Site energies are charged at their pair fraction and the
/// per-point energies at one evaluation per step.
pub fn optimise_occupation(
    cfg: &LatticeSearchConfig,
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
) -> (Array1<f64>, usize) {
    optimise_occupation_over(cfg, ledger, x, 3)
}

/// [`optimise_occupation`] trying the `movers` worst-bound points at each
/// step; `movers` at the point count is the full swap search.
pub fn optimise_occupation_over(
    cfg: &LatticeSearchConfig,
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
    movers: usize,
) -> (Array1<f64>, usize) {
    let n = cfg.n_points;
    let mut cur: Vec<f64> = x.to_vec();
    let frac = 2.0 / n.max(1) as f64;
    let mut moves = 0usize;
    while moves < cfg.max_moves {
        if !ledger.charge() {
            break;
        }
        let energies: Vec<f64> = (0..n)
            .map(|i| site_energy(cfg.kind, &cur, Some(i), point(&cur, i)))
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| energies[b].total_cmp(&energies[a]));
        let sites = hollow_sites(&cur, cfg.neighbour_cutoff);
        if sites.is_empty() {
            break;
        }
        let sites = match median_bond(&cur) {
            Some(bond) => best_coordinated(&cur, &sites, bond, OCCUPATION_CANDIDATE_SITES),
            None => sites,
        };
        let mut best: Option<(usize, [f64; 3], f64)> = None;
        for &atom in order.iter().take(movers.max(1)) {
            for site in &sites {
                if !ledger.charge_frac(frac) {
                    return (Array1::from(cur), moves);
                }
                let gain = energies[atom] - site_energy(cfg.kind, &cur, Some(atom), *site);
                if gain > 1e-9 && best.is_none_or(|(_, _, g)| gain > g) {
                    best = Some((atom, *site, gain));
                }
            }
        }
        let Some((atom, site, _)) = best else { break };
        for k in 0..3 {
            cur[3 * atom + k] = site[k];
        }
        moves += 1;
    }
    (Array1::from(cur), moves)
}

/// One greedy construction: move the worst-bound point to the best vacant
/// site while that lowers its energy. Returns the constructed coordinates
/// and how many moves were made; site energies are charged at their pair
/// fraction.
pub fn construct(
    cfg: &LatticeSearchConfig,
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
) -> (Array1<f64>, usize) {
    let n = cfg.n_points;
    let mut cur: Vec<f64> = x.to_vec();
    let mut sites = hollow_sites(&cur, cfg.neighbour_cutoff);
    let frac = 2.0 / n.max(1) as f64;
    let mut moves = 0usize;
    let mut last_moved: Option<usize> = None;
    while moves < cfg.max_moves && !sites.is_empty() {
        // Per-point energies: one full evaluation's worth of pair terms.
        if !ledger.charge() {
            break;
        }
        let energies: Vec<f64> = (0..n)
            .map(|i| site_energy(cfg.kind, &cur, Some(i), point(&cur, i)))
            .collect();
        let worst = (0..n)
            .filter(|&i| Some(i) != last_moved)
            .max_by(|&a, &b| energies[a].total_cmp(&energies[b]))
            .unwrap_or(0);
        let mut best: Option<(usize, f64)> = None;
        for (s, site) in sites.iter().enumerate() {
            if !ledger.charge_frac(frac) {
                return (Array1::from(cur), moves);
            }
            let e = site_energy(cfg.kind, &cur, Some(worst), *site);
            if best.is_none_or(|(_, be)| e < be) {
                best = Some((s, e));
            }
        }
        let Some((s, e)) = best else { break };
        if e >= energies[worst] - 1e-9 {
            break;
        }
        let vacated = point(&cur, worst);
        let target = sites.swap_remove(s);
        for k in 0..3 {
            cur[3 * worst + k] = target[k];
        }
        // The vacated position is a site again; the lattice is otherwise
        // kept, so a construction is a sequence of occupation swaps.
        sites.push(vacated);
        last_moved = Some(worst);
        moves += 1;
    }
    (Array1::from(cur), moves)
}

/// Rebuild the whole cluster on the lattice of occupied positions and hollow
/// sites of `x`: the fully coordinated interior stays, then each further
/// point takes the vacant lattice site with the lowest energy in the field
/// of the points already placed, and the occupation is then relaxed by
/// swaps. Every site energy is charged at its pair fraction.
pub fn reoccupy(cfg: &LatticeSearchConfig, ledger: &mut Ledger, x: ArrayView1<f64>) -> Array1<f64> {
    let n = cfg.n_points;
    let (placed, vacant) = place_successively_on(cfg, ledger, x);
    // Successive placement is greedy; a Metropolis walk over occupations
    // of the same lattice moves it off that ordering, and swaps then
    // finish the occupation.
    let walked = walk_occupation_on(
        cfg,
        ledger,
        placed.view(),
        &vacant,
        OCCUPATION_WALK_STEPS * n,
        OCCUPATION_WALK_TEMPERATURE,
        0x0cc_u64 ^ n as u64,
    );
    optimise_occupation_over(cfg, ledger, walked.view(), OCCUPATION_FINISH_MOVERS).0
}

/// Worst-bound points the finishing swap pass of a reoccupation tries.
pub const OCCUPATION_FINISH_MOVERS: usize = 8;

/// Vacant sites whose energies are read per mover or per placement: the
/// sites with the most points within the bond length, ranked without any
/// energy call, so the charged site energies go to the candidates that can
/// bind.
pub const OCCUPATION_CANDIDATE_SITES: usize = 40;

/// The sites of `sites` with the most points of `x` within `1.2 * bond`,
/// most first: the `keep` best and every site tied with the last of them.
fn best_coordinated(x: &[f64], sites: &[[f64; 3]], bond: f64, keep: usize) -> Vec<[f64; 3]> {
    let n = x.len() / 3;
    let r2 = (1.2 * bond) * (1.2 * bond);
    let mut ranked: Vec<(usize, [f64; 3])> = sites
        .iter()
        .map(|s| ((0..n).filter(|&i| dist2(*s, point(x, i)) < r2).count(), *s))
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0));
    // Every site tied with the last kept one stays: the cut is by
    // coordination, never by the arbitrary order of equals.
    let floor = ranked.get(keep.saturating_sub(1)).map_or(0, |(c, _)| *c);
    ranked
        .into_iter()
        .take_while(|(c, _)| *c >= floor)
        .map(|(_, s)| s)
        .collect()
}

/// The successive placement alone: the fully coordinated interior of `x`
/// stays, and every other point takes, in turn, the vacant lattice site
/// with the lowest energy in the field of the points placed so far.
pub fn place_successively(
    cfg: &LatticeSearchConfig,
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
) -> Array1<f64> {
    place_successively_on(cfg, ledger, x).0
}

/// The lattice of `x` grown from its interior: the fully coordinated points
/// stay, the first layer holds every hollow site over them (both stackings
/// of each facet), the second layer every hollow site over the interior and
/// that first layer, and the original surface points are kept as sites too.
/// A structure without an interior uses its own points and hollow sites and
/// seeds the site nearest its centroid. Returns the seed coordinates and
/// the vacant sites.
pub fn core_lattice(cfg: &LatticeSearchConfig, x: &[f64]) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let n = x.len() / 3;
    let cut2 = cfg.neighbour_cutoff * cfg.neighbour_cutoff;
    let mut coordination = vec![0usize; n];
    for a in 0..n {
        for b in (a + 1)..n {
            if dist2(point(x, a), point(x, b)) < cut2 {
                coordination[a] += 1;
                coordination[b] += 1;
            }
        }
    }
    let interior: Vec<usize> = (0..n).filter(|&i| coordination[i] >= 12).collect();
    let Some(bond) = median_bond(x) else {
        return ((0..n).map(|i| point(x, i)).collect(), Vec::new());
    };
    if interior.is_empty() {
        let mut c = [0.0_f64; 3];
        for i in 0..n {
            for k in 0..3 {
                c[k] += x[3 * i + k] / n as f64;
            }
        }
        let mut lattice: Vec<[f64; 3]> = (0..n).map(|i| point(x, i)).collect();
        lattice.extend(hollow_sites_with_bond(x, cfg.neighbour_cutoff, bond));
        let first = (0..lattice.len())
            .min_by(|&a, &b| dist2(lattice[a], c).total_cmp(&dist2(lattice[b], c)))
            .unwrap_or(0);
        let seed = lattice.swap_remove(first);
        return (vec![seed], lattice);
    }
    let mut core: Vec<f64> = Vec::with_capacity(3 * interior.len());
    for &i in &interior {
        core.extend_from_slice(&x[3 * i..3 * i + 3]);
    }
    let first_layer = hollow_sites_with_bond(&core, cfg.neighbour_cutoff, bond);
    let mut with_first = core.clone();
    for site in &first_layer {
        with_first.extend_from_slice(site);
    }
    let second_layer = hollow_sites_with_bond(&with_first, cfg.neighbour_cutoff, bond);
    let merge2 = (0.3 * bond) * (0.3 * bond);
    let mut vacant: Vec<[f64; 3]> = first_layer;
    vacant.extend(second_layer);
    for i in 0..n {
        if interior.contains(&i) {
            continue;
        }
        let p = point(x, i);
        if !vacant.iter().any(|s| dist2(*s, p) < merge2) {
            vacant.push(p);
        }
    }
    let seeds = interior.iter().map(|&i| point(x, i)).collect();
    (seeds, vacant)
}

/// [`place_successively`] returning the sites it left vacant as well.
pub fn place_successively_on(
    cfg: &LatticeSearchConfig,
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
) -> (Array1<f64>, Vec<[f64; 3]>) {
    let n = cfg.n_points;
    let xs = x.to_vec();
    let (seeds, mut vacant) = core_lattice(cfg, &xs);
    let mut placed: Vec<f64> = Vec::with_capacity(3 * n);
    for seed in &seeds {
        placed.extend_from_slice(seed);
    }
    let frac = 2.0 / n.max(1) as f64;
    while placed.len() / 3 < n && !vacant.is_empty() {
        let mut best: Option<(usize, f64)> = None;
        // Every vacant site is read here: a coordination cut misses the
        // facet sites a decahedral surface needs, and the placement is a
        // small share of the construction's charge.
        for (s, site) in vacant.iter().enumerate() {
            if !ledger.charge_frac(frac) {
                break;
            }
            let e = site_energy(cfg.kind, &placed, None, *site);
            if best.is_none_or(|(_, be)| e < be) {
                best = Some((s, e));
            }
        }
        let Some((s, _)) = best else { break };
        let site = vacant.swap_remove(s);
        placed.extend_from_slice(&site);
    }
    if placed.len() / 3 < n {
        // The lattice ran short of sites: keep the original for the rest.
        for i in placed.len() / 3..n {
            placed.extend_from_slice(&xs[3 * i..3 * i + 3]);
        }
    }
    (Array1::from(placed), vacant)
}

/// Occupation walk steps per point.
pub const OCCUPATION_WALK_STEPS: usize = 20;
/// Occupation walk final temperature in pair-well units.
pub const OCCUPATION_WALK_TEMPERATURE: f64 = 0.4;
/// The walk starts at this multiple of the final temperature and cools
/// geometrically.
pub const OCCUPATION_WALK_START_RATIO: f64 = 2.5;

/// Metropolis walk over occupations of the lattice of `x` (its points and
/// their hollow sites): each step proposes moving one point to one vacant
/// site and accepts by the change in its energy, cooling geometrically to
/// `temperature` from [`OCCUPATION_WALK_START_RATIO`] times it. Returns
/// the lowest-energy occupation seen. Each proposal is charged one pair
/// fraction, and each acceptance a second for the incremental update.
pub fn walk_occupation(
    cfg: &LatticeSearchConfig,
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
    steps: usize,
    temperature: f64,
    seed: u64,
) -> Array1<f64> {
    let vacant = hollow_sites(x.as_slice().unwrap_or(&x.to_vec()), cfg.neighbour_cutoff);
    walk_occupation_on(cfg, ledger, x, &vacant, steps, temperature, seed)
}

/// [`walk_occupation`] over the points of `x` and the given vacant sites.
#[allow(clippy::too_many_arguments)]
pub fn walk_occupation_on(
    cfg: &LatticeSearchConfig,
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
    vacant_sites: &[[f64; 3]],
    steps: usize,
    temperature: f64,
    seed: u64,
) -> Array1<f64> {
    let n = cfg.n_points;
    let xs = x.to_vec();
    let mut lattice: Vec<[f64; 3]> = (0..n).map(|i| point(&xs, i)).collect();
    lattice.extend_from_slice(vacant_sites);
    let m = lattice.len();
    if m <= n || !ledger.charge() {
        return x.to_owned();
    }
    let kind = cfg.kind;
    let frac = 2.0 / n.max(1) as f64;
    let mut cur = xs;
    let mut energies: Vec<f64> = (0..n)
        .map(|i| site_energy(kind, &cur, Some(i), point(&cur, i)))
        .collect();
    let mut total: f64 = energies.iter().sum::<f64>() / 2.0;
    let mut best_total = total;
    let mut best = cur.clone();
    let mut occupied: Vec<usize> = (0..n).collect();
    let mut vacant: Vec<usize> = (n..m).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let t_hi = OCCUPATION_WALK_START_RATIO * temperature;
    for step in 0..steps {
        if !ledger.charge_frac(frac) {
            break;
        }
        let progress = step as f64 / steps.max(1) as f64;
        let temperature = t_hi * (temperature / t_hi).powf(progress);
        let i = rng.random_range(0..n);
        let slot = rng.random_range(0..vacant.len());
        let site = lattice[vacant[slot]];
        let e_new = site_energy(kind, &cur, Some(i), site);
        let delta = e_new - energies[i];
        if delta > 0.0 && rng.random::<f64>() >= (-delta / temperature).exp() {
            continue;
        }
        if !ledger.charge_frac(frac) {
            break;
        }
        let old = point(&cur, i);
        for j in 0..n {
            if j == i {
                continue;
            }
            let pj = point(&cur, j);
            let (r2_old, r2_new) = (dist2(old, pj), dist2(site, pj));
            if r2_old > 1e-12 {
                energies[j] -= kind.pair(r2_old).0;
            }
            if r2_new > 1e-12 {
                energies[j] += kind.pair(r2_new).0;
            }
        }
        energies[i] = e_new;
        for k in 0..3 {
            cur[3 * i + k] = site[k];
        }
        let freed = occupied[i];
        occupied[i] = vacant[slot];
        vacant[slot] = freed;
        total += delta;
        if total < best_total - 1e-9 {
            best_total = total;
            best.clone_from(&cur);
        }
    }
    Array1::from(best)
}

/// Run the dynamic lattice search under `ledger`.
///
/// `relax` is the caller's charged quench. The search alternates construction
/// and relaxation from the current best, and restarts from a fresh random
/// cluster after `patience` constructions that fail to lower the energy.
pub fn run(cfg: &LatticeSearchConfig, ledger: &mut Ledger, relax: Relax<'_>, seed: u64) -> Outcome {
    let n = cfg.n_points;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut outcome = Outcome::default();
    let mut restarts = 0usize;
    let mut constructions = 0usize;
    let mut improvements = 0usize;
    let mut charged_at_best = 0usize;
    let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
    let (mut e, mut x) = relax(ledger, start.view(), cfg.relax_steps);
    ledger.record(e, x.view());
    let mut stalls = 0usize;
    while ledger.remaining() > 0 {
        // Constructions alternate: reoccupy the whole lattice from the
        // centre, then repair the worst-bound point.
        let (built, moves) = match constructions % 3 {
            0 => optimise_occupation(cfg, ledger, x.view()),
            1 => (reoccupy(cfg, ledger, x.view()), 1),
            _ => construct(cfg, ledger, x.view()),
        };
        constructions += 1;
        if ledger.remaining() == 0 {
            break;
        }
        let (e_new, x_new) = relax(ledger, built.view(), cfg.relax_steps);
        ledger.record(e_new, x_new.view());
        if moves > 0 && e_new < e - 1e-9 {
            e = e_new;
            x = x_new;
            stalls = 0;
            improvements += 1;
            if e < outcome.best {
                charged_at_best = ledger.spent();
            }
        } else {
            stalls += 1;
        }
        if stalls >= cfg.patience {
            let fresh = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
            let (e_fresh, x_fresh) = relax(ledger, fresh.view(), cfg.relax_steps);
            ledger.record(e_fresh, x_fresh.view());
            e = e_fresh;
            x = x_fresh;
            stalls = 0;
            restarts += 1;
        }
        outcome.best = outcome.best.min(ledger.best);
    }
    outcome.best = ledger.best;
    outcome.best_state = ledger.best_state.clone();
    outcome.hops = constructions;
    outcome.charged = ledger.spent();
    outcome.basins = restarts;
    outcome.returned = improvements;
    if charged_at_best > 0 {
        outcome
            .improvements
            .push((constructions, charged_at_best, restarts, outcome.best));
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::warm_lbfgs::WarmLbfgs;
    use crate::potentials::PairPotential;
    use ndarray::Array1;

    fn search(n: usize, budget: usize, seed: u64) -> Outcome {
        let cfg = LatticeSearchConfig::lennard_jones(n);
        let pot = PairPotential::lennard_jones(n);
        let mut opt = WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            opt.forget();
            let (f, xr, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                Some(pot.value_and_gradient(v))
            });
            (f, xr)
        };
        let mut ledger = Ledger::new(budget);
        run(&cfg, &mut ledger, &mut relax, seed)
    }

    fn fixture(name: &str) -> Array1<f64> {
        let path = format!("{}/tests/fixtures/{name}.xyz", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut coords = Vec::new();
        for line in text.lines().skip(2) {
            let f: Vec<f64> = line
                .split_whitespace()
                .skip(1)
                .map(|v| v.parse().expect("coordinate"))
                .collect();
            if f.len() == 3 {
                coords.extend(f);
            }
        }
        Array1::from(coords)
    }

    fn quench(n: usize, x: ArrayView1<f64>) -> f64 {
        let pot = PairPotential::lennard_jones(n);
        let mut opt = WarmLbfgs::default();
        let (f, _, _) = opt.minimize(x, 5000, |v| Some(pot.value_and_gradient(v)));
        f
    }

    /// The eight least coordinated points scattered onto a sphere outside
    /// the cluster.
    fn displace_surface(x: &Array1<f64>, cutoff: f64, seed: u64) -> Array1<f64> {
        use rand::{Rng, SeedableRng};
        let n = x.len() / 3;
        let xs = x.to_vec();
        let cut2 = cutoff * cutoff;
        let mut coordination = vec![0usize; n];
        let mut rmax: f64 = 0.0;
        for a in 0..n {
            rmax = rmax.max(dist2([0.0; 3], point(&xs, a)).sqrt());
            for b in (a + 1)..n {
                if dist2(point(&xs, a), point(&xs, b)) < cut2 {
                    coordination[a] += 1;
                    coordination[b] += 1;
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| coordination[i]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut out = xs;
        for &atom in order.iter().take(8) {
            let u: f64 = rng.random_range(-1.0..1.0);
            let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
            let r = 1.15 * rmax;
            let s = (1.0 - u * u).sqrt();
            out[3 * atom] = r * s * phi.cos();
            out[3 * atom + 1] = r * s * phi.sin();
            out[3 * atom + 2] = r * u;
        }
        Array1::from(out)
    }

    fn restores(name: &str, n: usize, reference: f64, seed: u64) -> (f64, f64) {
        let gm = fixture(name);
        assert_eq!(gm.len(), 3 * n);
        let cfg = LatticeSearchConfig::lennard_jones(n);
        let start = displace_surface(&gm, cfg.neighbour_cutoff, seed);
        let mut ledger = Ledger::new(400_000);
        let (built, _) = optimise_occupation(&cfg, &mut ledger, start.view());
        let occupied = quench(n, built.view());
        let mut ledger = Ledger::new(400_000);
        let rebuilt = reoccupy(&cfg, &mut ledger, gm.view());
        let reoccupied = quench(n, rebuilt.view());
        assert!(
            quench(n, gm.view()) < reference + 1e-3,
            "fixture {name} does not quench to {reference}"
        );
        (occupied, reoccupied)
    }

    #[test]
    fn occupation_optimisation_restores_displaced_surface_atoms_of_lj98() {
        let (occupied, reoccupied) = restores("lj98_gm", 98, -543.665361, 1);
        assert!(occupied < -543.665361 + 1e-3, "occupation gave {occupied}");
        assert!(
            reoccupied < -543.665361 + 1e-3,
            "reoccupation gave {reoccupied}"
        );
    }

    #[test]
    fn occupation_optimisation_restores_displaced_surface_atoms_of_lj104() {
        let (occupied, reoccupied) = restores("lj104_gm", 104, -582.086642, 1);
        assert!(occupied < -582.086642 + 1e-3, "occupation gave {occupied}");
        assert!(
            reoccupied < -582.086642 + 1e-3,
            "reoccupation gave {reoccupied}"
        );
    }

    #[test]
    fn a_tetrahedron_offers_sites_and_a_loose_point_takes_the_best() {
        let h = (2.0_f64 / 3.0).sqrt();
        let x = vec![
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.5,
            0.75_f64.sqrt(),
            0.0,
            0.5,
            0.75_f64.sqrt() / 3.0,
            h,
            5.0,
            5.0,
            5.0,
        ];
        let sites = hollow_sites(&x, 1.3);
        assert!(sites.len() >= 3, "{}", sites.len());
        let cfg = LatticeSearchConfig {
            n_points: 5,
            kind: PairKind::LennardJones,
            neighbour_cutoff: 1.3,
            relax_steps: 1,
            patience: 1,
            max_moves: 4,
            min_separation: 0.5,
        };
        let mut ledger = Ledger::new(1000);
        let (built, moves) = construct(&cfg, &mut ledger, ArrayView1::from(&x));
        assert!(moves >= 1);
        let e_before = site_energy(PairKind::LennardJones, &x, Some(4), point(&x, 4));
        let e_after = site_energy(
            PairKind::LennardJones,
            built.as_slice().unwrap(),
            Some(4),
            point(built.as_slice().unwrap(), 4),
        );
        assert!(e_after < e_before, "{e_after} against {e_before}");
        assert!(ledger.spent() > 0, "site energies are charged");
    }

    #[test]
    fn the_search_reaches_the_lj13_icosahedron() {
        let out = search(13, 20_000, 1);
        assert!(
            out.best < -44.3268 + 1e-3,
            "LJ13 stopped at {:.6}",
            out.best
        );
        assert!(out.hops > 0 && out.charged <= 20_000);
    }
}
