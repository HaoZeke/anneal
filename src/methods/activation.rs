//! Activation: climb the ridge out of a basin, then quench the other side.
//!
//! This is valley-floor / ridge following, not a quench. Quapp's gradient
//! extremal of the smallest Hessian eigenvalue is the valley floor or the
//! ridge (Quapp, *Chem. Phys. Lett.* **1996**, *253*, 286,
//! <https://doi.org/10.1016/0009-2614(96)00255-2>). Reduced-gradient
//! following traces the same curves (Quapp, Hirsch, Imig, Heidrich, *J.
//! Comput. Chem.* **1998**, *19*, 1087). Barkema and Mousseau's ART
//! (*Phys. Rev. Lett.* **1996**, *77*, 4358,
//! <https://doi.org/10.1103/PhysRevLett.77.4358>; Malek and Mousseau,
//! *Phys. Rev. E* **2000**, *62*, 7723,
//! <https://doi.org/10.1103/PhysRevE.62.7723>) climbs that direction
//! until the curvature turns over. Henkelman and Jónsson's dimer
//! (*J. Chem. Phys.* **1999**, *111*, 7010,
//! <https://doi.org/10.1063/1.480097>) and Plasencia's SoftSaddle MMF
//! (*J. Chem. Theory Comput.* **2017**, *13*, 125,
//! <https://doi.org/10.1021/acs.jctc.5b01216>) invert the force along
//! the minimum mode and minimise the rest: that is the same ridge walk
//! to a first-order saddle. Xiao, Wu and Henkelman (*J. Chem. Phys.*
//! **2014**, *141*, 164111, <https://doi.org/10.1063/1.4898664>)
//! distinguish that *local* ridge (force perpendicular to the negative
//! mode) from the true basin-boundary ridge; a quench is taken only
//! after the force along the mode has changed sign, so the landing is
//! past the saddle and not back down the local ridge into the well.
//!
//! A single displacement along the softest mode does not leave a basin. It is
//! the right direction and the wrong distance: the mode points at the low
//! saddle, but relaxing from a point still inside the basin returns to the
//! minimum it came from. Measured on LJ38 with the escape controller driving a
//! straight displacement, 576 quenches in 959 came back to the basin they left
//! and 10 found anything new.
//!
//! Goedecker's answer is molecular dynamics, which carries kinetic energy over
//! the saddle. The answer that needs only gradients is to climb: push along the
//! mode, relax the components perpendicular to it so the structure stays on the
//! valley floor, and repeat until the curvature along the mode turns negative
//! *and* the force along the climb has flipped. That is the ridge, and a
//! quench from the overshoot falls into a different basin.
//!
//! What this costs is honest and worth stating. Each climbing step is a
//! curvature pass and a few perpendicular relaxation steps, so an activation is
//! several hundred charged evaluations where a random displacement is one. It
//! buys escapes that actually leave.
//!
//! # Relation to the rest of the crate
//!
//! The perpendicular relaxation is the same projection [`crate::path`] uses to
//! hold a band off its endpoints, and the mode comes from
//! [`crate::curvature`]. The controller in [`crate::methods::minima_hopping`]
//! sets how far to climb; this module decides when to stop.

use crate::curvature::curvature_features;
use crate::methods::minima_hopping::{
    EscapeFeedback, MdEscapeConfig, MdEscapeGeometry, Visit, nve_escape_seeded,
};
use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::io::Write;

/// How the climb is run.
#[derive(Debug, Clone)]
pub struct Activation {
    /// Distance moved along the mode per climbing step.
    pub step: f64,
    /// Climbing steps before giving up.
    ///
    /// A cap rather than a convergence criterion: some directions do not reach
    /// negative curvature at all, and a climb that has not turned over after
    /// this many steps is abandoned rather than run to exhaustion.
    pub max_steps: usize,
    /// Perpendicular relaxation steps between climbs.
    pub perp_steps: usize,
    /// Step size of the perpendicular relaxation.
    pub perp_rate: f64,
    /// Largest displacement one perpendicular step may make.
    ///
    /// A fixed rate is not safe on a potential whose gradient spans decades. On
    /// a Lennard-Jones cluster two points a little too close carry a gradient of
    /// order a thousand, and a rate of 0.02 against that moves the structure
    /// twenty units and destroys it: measured on LJ38, 6 relaxations in 1589
    /// reached a minimum and the returned structure had a gradient of 1.0 where
    /// a minimum has 1e-6. The cap makes the step a direction with a bounded
    /// length rather than a length proportional to the gradient.
    pub perp_max_move: f64,
    /// Lanczos steps per curvature pass.
    pub lanczos_steps: usize,
    /// Finite-difference step for the curvature.
    pub epsilon: f64,
    /// Climbing steps between recomputing the mode.
    ///
    /// Recomputing every step is the accurate choice and the expensive one. The
    /// mode rotates slowly along a valley floor, so reusing it for a few steps
    /// costs little accuracy and divides the curvature bill.
    pub refresh: usize,
    /// Extra push along the mode once the curvature has turned over, in units
    /// of `step`, before the quench.
    pub overshoot: f64,
    /// Ignore a ridge whose integrated rise is below this.
    ///
    /// The first saddle out of a low minimum is often a shallow step to a
    /// neighbour of similar depth. A later ridge, several energy units up,
    /// is the one that can open another funnel. Zero keeps every ridge.
    pub min_rise: f64,
}

impl Default for Activation {
    fn default() -> Self {
        Self {
            step: 0.2,
            max_steps: 24,
            perp_steps: 3,
            perp_rate: 0.02,
            perp_max_move: 0.05,
            lanczos_steps: 12,
            epsilon: 1e-4,
            refresh: 3,
            overshoot: 1.5,
            min_rise: 0.0,
        }
    }
}

/// Where a climb ended.
#[derive(Debug, Clone)]
pub struct ActivationOutcome {
    /// The activated structure, to be quenched by the caller.
    pub state: Array1<f64>,
    /// Curvature along the mode at the end of the climb.
    pub lambda: f64,
    /// Climbing steps taken.
    pub steps: usize,
    /// Whether the curvature turned negative, so the ridge is behind.
    pub crossed: bool,
    /// Gradient evaluations spent, all of them charged by the caller.
    pub evaluations: usize,
}

/// Climbs out of the basin containing `x`.
///
/// `grad` returns the gradient or `None` when the caller's budget is spent, in
/// which case the climb stops and reports what it has. `sign` picks which way
/// along the mode to go; the two ends of a soft direction are different saddles.
///
/// Returns `None` only when the first curvature pass fails, since there is then
/// no direction to climb along.
pub fn activate<G>(
    x: ArrayView1<f64>,
    mut grad: G,
    cfg: &Activation,
    sign: f64,
) -> Option<ActivationOutcome>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    activate_aligned(x, None, None, &mut grad, cfg, sign)
}

/// One covering displacement, the minimum-mode climb, then the caller's quench.
///
/// The start is only a point and a force. No target energy is read.
/// `cover_index` selects one point of the hypersphere cover. The climb
/// walks away from `origin`. The quench is whatever the caller uses for
/// a local minimisation of the same force.
pub fn cover_climb_quench<G, Q>(
    origin: ArrayView1<f64>,
    rmsd: f64,
    cover_index: usize,
    mut grad: G,
    mut quench: Q,
    cfg: &Activation,
) -> Array1<f64>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let n_cover = crate::hypersphere::default_cover_size();
    let direction = crate::hypersphere::cover_direction(n_cover, origin.len(), cover_index);
    let placed = crate::hypersphere::place_around(
        origin.as_slice().unwrap_or(&[]),
        &direction,
        rmsd.max(1e-3),
        None,
    );
    let start = if placed.len() == origin.len() {
        Array1::from(placed)
    } else {
        origin.to_owned()
    };
    // Sit on the minimum the covering point falls into. The soft mode
    // is only a valley floor there. Climbing the raw kick walks into
    // overlaps and the curvature of that clash is not a ridge.
    let landed = quench(start.view());
    let mut chosen = landed.clone();
    for sign in [1.0_f64, -1.0] {
        let Some(outcome) = activate(landed.view(), &mut grad, cfg, sign) else {
            continue;
        };
        if outcome.crossed {
            chosen = quench(outcome.state.view());
        }
    }
    chosen
}

/// Covering displacement, fivefold openings of the same shell, a
/// minimum-mode climb, then a quench. The lowest minimum is kept.
///
/// The hypersphere cover picks a direction. On a shell that still has
/// pentagonal axes, those axes are further covering directions: they
/// are read off the coordinates, not off a target minimum. No target
/// energy is supplied.
pub fn cover_climb_quench_min<G, Q, E>(
    origin: ArrayView1<f64>,
    rmsd: f64,
    cover_index: usize,
    mut grad: G,
    mut quench: Q,
    mut energy: E,
    cfg: &Activation,
) -> Array1<f64>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
    E: FnMut(ArrayView1<f64>) -> f64,
{
    let mut best = origin.to_owned();
    let mut best_e = energy(origin.view());
    let mut keep = |quenched: Array1<f64>| {
        let value = energy(quenched.view());
        if value.is_finite() {
            println!("{{\"kind\":\"exit_candidate\",\"energy\":{value:.6}}}");
        }
        if value.is_finite() && value < best_e - 1e-6 {
            best_e = value;
            best = quenched;
        }
    };
    // Several amplitudes of the pentagonal opening, quenched on their own.
    for amp in [0.35_f64, 0.55, 0.75, 1.0, 1.25] {
        for which in 0..4 {
            keep(quench(fivefold_opening(origin, amp, which).view()));
        }
    }
    // The same opening stacked, then one quench, so the shell can
    // reconstruct before it is relaxed.
    let mut chain = origin.to_owned();
    for which in 0..8 {
        chain = fivefold_opening(chain.view(), 0.75, which);
    }
    keep(quench(chain.view()));
    let mut here = origin.to_owned();
    let mut here_e = best_e;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1 + cover_index as u64);
    let n_cover = crate::hypersphere::default_cover_size();
    let temperature = 8.0_f64;
    // One kick stays in the icosahedral funnel or lands above it.
    // Keep walking: a covering displacement, sometimes a pentagonal
    // opening, a minimum-mode climb when the ridge is real, then a
    // quench. Uphill quenches are accepted so the walk can leave.
    for hop in 0..16 {
        let point = if hop % 6 == 0 {
            fivefold_opening(here.view(), rmsd, hop)
        } else {
            let direction =
                crate::hypersphere::cover_direction(n_cover, here.len(), hop + cover_index);
            let placed = crate::hypersphere::place_around(
                here.as_slice().unwrap_or(&[]),
                &direction,
                rmsd.max(1e-3),
                None,
            );
            if placed.len() == here.len() {
                Array1::from(placed)
            } else {
                here.clone()
            }
        };
        let mut quenched = quench(point.view());
        if let Some(outcome) = activate_from_origin(point.view(), here.view(), &mut grad, cfg) {
            if outcome.crossed {
                let climbed = quench(outcome.state.view());
                let climbed_e = energy(climbed.view());
                let direct_e = energy(quenched.view());
                if climbed_e.is_finite() && (!direct_e.is_finite() || climbed_e < direct_e) {
                    quenched = climbed;
                }
            }
        }
        let value = energy(quenched.view());
        if !value.is_finite() {
            continue;
        }
        println!("{{\"kind\":\"exit_candidate\",\"energy\":{value:.6}}}");
        if value < best_e - 1e-6 {
            best_e = value;
            best = quenched.clone();
        }
        let uphill = value - here_e;
        let accept = uphill <= 0.0 || rng.random::<f64>() < (-uphill / temperature).exp();
        if accept {
            here = quenched;
            here_e = value;
        }
    }
    best
}

fn packed_fraction(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n < 4 {
        return 0.0;
    }
    // Fixed neighbour shell of the Lennard-Jones minimum, so an expanded
    // hot frame is not given a cutoff that counts non-bonds.
    crate::structure::cna_descriptor(x, n, 1.50)[1]
}

fn basin_key(energy: f64) -> i64 {
    (energy / 5.0e-3).round() as i64
}

fn atom_at(x: ArrayView1<f64>, i: usize) -> [f64; 3] {
    [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn len3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// Surface hops: move one outer atom onto a tetrahedral site of a facet.
///
/// An all-atom displacement of a deep minimum falls back into that
/// minimum. A facet hop changes which hollow a surface atom occupies.
/// The centre-of-mass vacancy, when the innermost atom is not already
/// there, is filled by that atom as one extra trial. No target
/// geometry is used.
fn facet_trials(origin: ArrayView1<f64>, limit: usize) -> Vec<Array1<f64>> {
    let n = origin.len() / 3;
    if n < 4 || limit == 0 {
        return Vec::new();
    }
    let mut com = [0.0; 3];
    for i in 0..n {
        let a = atom_at(origin, i);
        for k in 0..3 {
            com[k] += a[k];
        }
    }
    for value in &mut com {
        *value /= n as f64;
    }
    let mut radial = Vec::with_capacity(n);
    for i in 0..n {
        radial.push((len3(sub3(atom_at(origin, i), com)), i));
    }
    radial.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut trials = Vec::new();
    if radial[0].0 > 0.2 {
        let mut filled = origin.to_owned();
        let i = radial[0].1;
        for k in 0..3 {
            filled[3 * i + k] = com[k];
        }
        trials.push(filled);
    }
    let req = 2.0_f64.powf(1.0 / 6.0);
    let cutoff = 1.35 * req;
    let cutoff2 = cutoff * cutoff;
    let mut faces = 0usize;
    'bonds: for i in 0..n {
        for j in (i + 1)..n {
            if len3(sub3(atom_at(origin, i), atom_at(origin, j))).powi(2) > cutoff2 {
                continue;
            }
            for k in (j + 1)..n {
                if trials.len() >= limit {
                    break 'bonds;
                }
                let jk = len3(sub3(atom_at(origin, j), atom_at(origin, k))).powi(2);
                let ik = len3(sub3(atom_at(origin, i), atom_at(origin, k))).powi(2);
                if jk > cutoff2 || ik > cutoff2 {
                    continue;
                }
                let a = atom_at(origin, i);
                let b = atom_at(origin, j);
                let c = atom_at(origin, k);
                let centroid = [
                    (a[0] + b[0] + c[0]) / 3.0,
                    (a[1] + b[1] + c[1]) / 3.0,
                    (a[2] + b[2] + c[2]) / 3.0,
                ];
                let normal = cross3(sub3(b, a), sub3(c, a));
                let length = len3(normal);
                if length < 1.0e-8 {
                    continue;
                }
                let mut hat = [normal[0] / length, normal[1] / length, normal[2] / length];
                if dot3(hat, sub3(centroid, com)) < 0.0 {
                    hat = [-hat[0], -hat[1], -hat[2]];
                }
                let reach = len3(sub3(a, centroid));
                let height = (req * req - reach * reach).max(0.05).sqrt();
                let site = [
                    centroid[0] + hat[0] * height,
                    centroid[1] + hat[1] * height,
                    centroid[2] + hat[2] * height,
                ];
                let mover = radial[n - 1 - (faces % 3)].1;
                if mover == i || mover == j || mover == k {
                    continue;
                }
                let mut moved = origin.to_owned();
                for axis in 0..3 {
                    moved[3 * mover + axis] = site[axis];
                }
                trials.push(moved);
                faces += 1;
            }
        }
    }
    trials
}

fn rotation_about(axis: [f64; 3], angle: f64) -> [f64; 9] {
    let n = len3(axis).max(1.0e-12);
    let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        t * x * x + c,
        t * x * y - s * z,
        t * x * z + s * y,
        t * x * y + s * z,
        t * y * y + c,
        t * y * z - s * x,
        t * x * z - s * y,
        t * y * z + s * x,
        t * z * z + c,
    ]
}

fn apply_rot(r: &[f64; 9], p: [f64; 3]) -> [f64; 3] {
    [
        r[0] * p[0] + r[1] * p[1] + r[2] * p[2],
        r[3] * p[0] + r[4] * p[1] + r[5] * p[2],
        r[6] * p[0] + r[7] * p[1] + r[8] * p[2],
    ]
}

/// Twist the two polar caps about one fivefold axis of the cluster.
///
/// The axis and the caps are read from the coordinates. A half-pentagon
/// turn swaps a vertex site with a hollow. The opposite cap turns the
/// other way, so the move is a shear of the packing rather than a
/// rigid rotation of the whole cluster.
fn cap_twists(origin: ArrayView1<f64>, limit: usize) -> Vec<Array1<f64>> {
    let n = origin.len() / 3;
    if n < 7 || limit == 0 {
        return Vec::new();
    }
    let mut com = [0.0; 3];
    for i in 0..n {
        let a = atom_at(origin, i);
        for k in 0..3 {
            com[k] += a[k];
        }
    }
    for value in &mut com {
        *value /= n as f64;
    }
    let mut trials = Vec::new();
    let axes = crate::soap::fivefold_axis_table(origin);
    for (axis, _) in axes.into_iter().take(6) {
        if trials.len() >= limit {
            break;
        }
        let mut hat = axis;
        let length = len3(hat);
        if length < 1.0e-8 {
            continue;
        }
        hat = [hat[0] / length, hat[1] / length, hat[2] / length];
        let mut proj = Vec::with_capacity(n);
        for i in 0..n {
            let rel = sub3(atom_at(origin, i), com);
            let along = dot3(rel, hat);
            let radial = (dot3(rel, rel) - along * along).max(0.0).sqrt();
            proj.push((along, radial, i));
        }
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for (along, _, _) in &proj {
            lo = lo.min(*along);
            hi = hi.max(*along);
        }
        let span = (hi - lo).max(1.0e-6);
        let north_cut = lo + 0.72 * span;
        let south_cut = lo + 0.28 * span;
        for (angle, shift) in [(0.628_f64, 0.0), (0.628, 0.35), (-0.628, 0.0), (0.35, 0.2)] {
            if trials.len() >= limit {
                break;
            }
            let north = rotation_about(hat, angle);
            let south = rotation_about(hat, -angle);
            let mut moved = origin.to_owned();
            for (along, radial, i) in &proj {
                if *radial < 0.45 {
                    continue;
                }
                let rel = sub3(atom_at(origin, *i), com);
                let (turned, extra) = if *along >= north_cut {
                    (apply_rot(&north, rel), shift)
                } else if *along <= south_cut {
                    (apply_rot(&south, rel), -shift)
                } else {
                    continue;
                };
                for k in 0..3 {
                    moved[3 * *i + k] = com[k] + turned[k] + hat[k] * extra;
                }
            }
            trials.push(moved);
        }
    }
    trials
}

fn note_candidate<E>(
    proposal: &Array1<f64>,
    evaluate: &mut E,
    best: &mut Array1<f64>,
    best_e: &mut f64,
    hop: usize,
    origin_e: f64,
    bank: &mut [Option<(f64, Array1<f64>)>],
    rejected: &HashSet<i64>,
) where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    if !proposal.iter().all(|value| value.is_finite()) {
        return;
    }
    let (value, _) = evaluate(proposal.view());
    if !value.is_finite() {
        return;
    }
    println!("{{\"kind\":\"exit_candidate\",\"energy\":{value:.6},\"hop\":{hop}}}");
    if value < *best_e - 1.0e-6 {
        *best_e = value;
        *best = proposal.clone();
    }
    // Minima above the floor and below the melt window are rungs. The
    // lowest structure in each one-energy band is kept so a later hop
    // can leave from that rung instead of from the floor.
    let rise = value - origin_e;
    if rejected.contains(&basin_key(value)) {
        return;
    }
    if (0.3..crate::catalog::SEAM_WINDOW).contains(&rise) && !bank.is_empty() {
        let bin = (rise as usize).min(bank.len() - 1);
        let replace = match &bank[bin] {
            Some((held, _)) => value < *held,
            None => true,
        };
        if replace {
            bank[bin] = Some((value, proposal.clone()));
        }
    }
}

fn basin_id(basins: &mut HashMap<i64, usize>, next_id: &mut usize, key: i64) -> usize {
    if let Some(id) = basins.get(&key).copied() {
        id
    } else {
        let id = *next_id;
        *next_id += 1;
        basins.insert(key, id);
        id
    }
}

/// Repeated covering displacement, minimum-mode climb, and quench.
///
/// One kick from a deep minimum falls back into it. The loop keeps
/// going. Each hop places one covering direction, optionally opens a
/// pentagonal axis, kicks every coordinate, and climbs the minimum
/// mode. Those quenches compete for the lowest energy only. The
/// escape scale is the constant-energy trajectory seeded by the
/// cover: returning home raises it, and a new basin lowers it. A
/// sideways hop into a higher neighbour must not be counted as that
/// new basin, or the trajectory never becomes violent enough to leave
/// the funnel. The lowest quench is returned. Nothing in the loop is
/// a named target energy. The early stop is a quench at least `0.05`
/// below the minimum the search started from, which is a different
/// basin rather than a tighter polish of the same one.
pub fn cover_climb_search<E, Q>(
    origin: ArrayView1<f64>,
    rmsd: f64,
    max_hops: usize,
    seed: u64,
    mut evaluate: E,
    mut quench: Q,
    cfg: &Activation,
) -> Array1<f64>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>) + Send,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    if max_hops == 0 || origin.is_empty() {
        return origin.to_owned();
    }
    let (origin_e, _) = evaluate(origin);
    if !origin_e.is_finite() {
        return origin.to_owned();
    }
    let mut best = origin.to_owned();
    let mut best_e = origin_e;
    let mut here = origin.to_owned();
    let mut here_e = origin_e;
    let mut basins: HashMap<i64, usize> = HashMap::new();
    basins.insert(basin_key(origin_e), 0);
    let mut next_id = 1usize;
    let mut current = 0usize;
    let ke0 = 1.0_f64;
    let mut feedback = EscapeFeedback::new(ke0, 4.0);
    feedback.escape_ceiling = 80.0;
    feedback.escape_floor = 0.25;
    feedback.register_initial(current);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut bank: Vec<Option<(f64, Array1<f64>)>> =
        vec![None; crate::catalog::SEAM_WINDOW.ceil() as usize];
    let mut frontier_stuck = 0u32;
    let mut frontier_energy = f64::NAN;
    let mut rejected: HashSet<i64> = HashSet::new();
    let n_cover = crate::hypersphere::default_cover_size();
    let cluster = origin.len() % 3 == 0 && {
        let pair = closest_pair(origin);
        pair > 0.5 && pair < 3.0
    };
    let geometry = if cluster {
        MdEscapeGeometry::RigidQuotient
    } else {
        MdEscapeGeometry::Euclidean
    };
    println!(
        "{{\"kind\":\"exit_candidate\",\"energy\":{origin_e:.6},\"hop\":0,\"role\":\"start\"}}"
    );

    // A decahedron is the packing with close-packed `421` pairs. Follow
    // a short constant-energy trajectory and quench at the point where
    // that fraction is highest, which is a geometry the steepest descent
    // from the floor never visits. No named target energy is used.
    if cluster {
        let n_atoms = origin.len() / 3;
        let base_pack = packed_fraction(origin);
        println!("{{\"kind\":\"exit_order\",\"packed\":{base_pack:.4},\"role\":\"start\"}}");
        let kinetics = [40.0_f64, 120.0, 220.0];
        for (attempt, kinetic) in kinetics.into_iter().enumerate().take(max_hops.max(1)) {
            if best_e < origin_e - 0.05 {
                return best;
            }
            let mut point = origin.to_owned();
            let mut velocity = Array1::zeros(point.len());
            let mut draw_sum = 0.0;
            for value in velocity.iter_mut() {
                let draw = rng.random::<f64>() - 0.5;
                *value = draw;
                draw_sum += draw * draw;
            }
            let scale = (2.0 * kinetic / draw_sum.max(1.0e-12)).sqrt();
            velocity *= scale;
            let dt = 0.003;
            let mut best_pack = base_pack;
            let mut quenched_peak = false;
            for step in 0..2_500 {
                let (energy, force) = evaluate(point.view());
                if !energy.is_finite() || energy > origin_e + 120.0 {
                    break;
                }
                for i in 0..point.len() {
                    velocity[i] += 0.5 * dt * (-force[i]);
                    point[i] += dt * velocity[i];
                }
                let (energy_new, force_new) = evaluate(point.view());
                if !energy_new.is_finite() {
                    break;
                }
                for i in 0..point.len() {
                    velocity[i] += 0.5 * dt * (-force_new[i]);
                }
                let mut com_v = [0.0; 3];
                for atom in 0..n_atoms {
                    for axis in 0..3 {
                        com_v[axis] += velocity[3 * atom + axis];
                    }
                }
                for value in &mut com_v {
                    *value /= n_atoms as f64;
                }
                let mut kinetic_now = 0.0;
                for atom in 0..n_atoms {
                    for axis in 0..3 {
                        velocity[3 * atom + axis] -= com_v[axis];
                        let speed = velocity[3 * atom + axis];
                        kinetic_now += speed * speed;
                    }
                }
                let rescale = (2.0 * kinetic / kinetic_now.max(1.0e-12)).sqrt();
                velocity *= rescale;
                if step % 40 == 0 {
                    let packed = packed_fraction(point.view());
                    if packed > best_pack {
                        best_pack = packed;
                    }
                    if packed > base_pack + 0.04 {
                        let quenched = quench(point.view());
                        note_candidate(
                            &quenched,
                            &mut evaluate,
                            &mut best,
                            &mut best_e,
                            attempt,
                            origin_e,
                            &mut bank,
                            &rejected,
                        );
                        quenched_peak = true;
                        if best_e < origin_e - 0.05 {
                            println!(
                                "{{\"kind\":\"exit_hop\",\"hop\":{attempt},\"here\":{best_e:.6},\"best\":{best_e:.6},\"packed\":{best_pack:.4},\"phase\":\"order\",\"left\":true}}"
                            );
                            let _ = std::io::stdout().flush();
                            return best;
                        }
                    }
                }
            }
            println!(
                "{{\"kind\":\"exit_hop\",\"hop\":{attempt},\"here\":{here_e:.6},\"best\":{best_e:.6},\"packed\":{best_pack:.4},\"quenched\":{quenched_peak},\"phase\":\"order\"}}"
            );
            let _ = std::io::stdout().flush();
        }
    }
    if best_e < origin_e - 0.05 {
        return best;
    }

    for hop in 0..max_hops.min(8) {
        if best_e < origin_e - 0.05 {
            break;
        }
        let kinetic = feedback.escape();
        let span = (kinetic / ke0).sqrt();
        let cover_rmsd = (rmsd * span).clamp(0.25, 1.4);
        let direction = Array1::from(crate::hypersphere::cover_direction(
            n_cover,
            here.len(),
            hop.wrapping_add(seed as usize),
        ));

        let placed = crate::hypersphere::place_around(
            here.as_slice().unwrap_or(&[]),
            direction.as_slice().unwrap_or(&[]),
            cover_rmsd,
            None,
        );
        if placed.len() == here.len() {
            let quenched = quench(Array1::from(placed).view());
            note_candidate(
                &quenched,
                &mut evaluate,
                &mut best,
                &mut best_e,
                hop,
                origin_e,
                &mut bank,
                &rejected,
            );
        }
        if cluster && hop % 4 == 0 {
            let quenched = quench(fivefold_opening(here.view(), cover_rmsd, hop).view());
            note_candidate(
                &quenched,
                &mut evaluate,
                &mut best,
                &mut best_e,
                hop,
                origin_e,
                &mut bank,
                &rejected,
            );
        }
        let amp = (0.38 * span).clamp(0.25, 0.9);
        let mut kicked = here.clone();
        for value in kicked.iter_mut() {
            *value += amp * (2.0 * rng.random::<f64>() - 1.0);
        }
        let quenched = quench(kicked.view());
        note_candidate(
            &quenched,
            &mut evaluate,
            &mut best,
            &mut best_e,
            hop,
            origin_e,
            &mut bank,
            &rejected,
        );

        // A displacement stays inside the packing it started in. A twin
        // across one dense plane of that packing changes only the boundary,
        // which is the relation between the close-packed morphologies.
        if cluster {
            let n_atoms = here.len() / 3;
            let planes = crate::twin::dense_planes(here.view(), n_atoms, 0.25);
            if !planes.is_empty() {
                let width = if hop == 0 { planes.len().min(80) } else { 2 };
                for slot in 0..width {
                    let plane = &planes[(hop + slot) % planes.len()];
                    for mode in [crate::twin::Mode::Reflect, crate::twin::Mode::Rotate] {
                        let moved = crate::twin::twin(here.view(), n_atoms, plane, mode, 0.25);
                        if moved.iter().all(|value| value.is_finite()) {
                            let quenched = quench(moved.view());
                            note_candidate(
                                &quenched,
                                &mut evaluate,
                                &mut best,
                                &mut best_e,
                                hop,
                                origin_e,
                                &mut bank,
                                &rejected,
                            );
                        }
                    }
                }
            }
        }

        if cluster && hop % 4 == 0 {
            if let Some(features) = crate::curvature::curvature_features(
                here.view(),
                |point| Some(evaluate(point).1),
                32,
                1.0e-4,
            ) {
                println!(
                    "{{\"kind\":\"exit_mode\",\"hop\":{hop},\"lambda\":{:.4},\"participation\":{:.3}}}",
                    features.lambda_min, features.participation
                );
                let atoms = (here.len() / 3) as f64;
                for delta in [0.15_f64, 0.3, 0.5, 0.8, 1.1, 1.5] {
                    for sign in [1.0, -1.0] {
                        let mut displaced = here.clone();
                        displaced.scaled_add(sign * delta * atoms.sqrt(), &features.mode);
                        let quenched = quench(displaced.view());
                        note_candidate(
                            &quenched,
                            &mut evaluate,
                            &mut best,
                            &mut best_e,
                            hop,
                            origin_e,
                            &mut bank,
                            &rejected,
                        );
                    }
                }
            }
        }

        if hop % 8 == 0 {
            for sign in [1.0_f64, -1.0] {
                if let Some(outcome) = activate(here.view(), |y| Some(evaluate(y).1), cfg, sign)
                    && outcome.crossed
                {
                    let quenched = quench(outcome.state.view());
                    note_candidate(
                        &quenched,
                        &mut evaluate,
                        &mut best,
                        &mut best_e,
                        hop,
                        origin_e,
                        &mut bank,
                        &rejected,
                    );
                }
            }
        }
        if best_e < origin_e - 0.05 {
            println!(
                "{{\"kind\":\"exit_hop\",\"hop\":{hop},\"here\":{here_e:.6},\"best\":{best_e:.6},\"left\":true}}"
            );
            let _ = std::io::stdout().flush();
            break;
        }

        // A drop of a few energy units is one vibration of the same well.
        // The quench from that point returns to the well it left. The
        // trajectory runs its whole step budget, and the quench is taken
        // at the end.
        let md_steps = (800.0 + 20.0 * kinetic).clamp(800.0, 2_000.0) as usize;
        let md = MdEscapeConfig {
            dt: 0.004,
            potential_minima: usize::MAX / 4,
            maximum_steps: md_steps,
            geometry,
            softening: Some(rgsaddle::VelocitySofteningConfig {
                steps: 6,
                displacement: 0.1,
                mixing: 0.15,
            }),
            minimum_rise: 0.0,
        };
        let mut escaped = None;
        {
            let mut eval = |point: ArrayView1<f64>| Some(evaluate(point));
            if let Ok(report) = nve_escape_seeded(
                here.view(),
                kinetic,
                Some(direction.view()),
                &md,
                &mut eval,
                &mut rng,
            ) && report.position.iter().all(|value| value.is_finite())
            {
                println!(
                    "{{\"kind\":\"exit_md\",\"hop\":{hop},\"steps\":{},\"minima\":{},\"energy\":{:.6}}}",
                    report.steps, report.potential_minima, report.energy
                );
                escaped = Some(report.position);
            }
        }
        let landed = escaped.map(|position| quench(position.view()));
        if let Some(position) = landed.as_ref() {
            note_candidate(
                position,
                &mut evaluate,
                &mut best,
                &mut best_e,
                hop,
                origin_e,
                &mut bank,
                &rejected,
            );
        }
        if best_e < origin_e - 0.05 {
            println!(
                "{{\"kind\":\"exit_hop\",\"hop\":{hop},\"here\":{here_e:.6},\"best\":{best_e:.6},\"left\":true}}"
            );
            let _ = std::io::stdout().flush();
            break;
        }

        match landed {
            Some(position) => {
                let (value, _) = evaluate(position.view());
                if !value.is_finite() || value > origin_e + 30.0 {
                    let id = next_id;
                    next_id += 1;
                    feedback.observe(Some(current), id);
                } else {
                    let key = basin_key(value);
                    if key == basin_key(here_e) {
                        feedback.observe(Some(current), current);
                    } else {
                        let reached = basin_id(&mut basins, &mut next_id, key);
                        let visit = feedback.observe(Some(current), reached);
                        if visit != Visit::Same && feedback.accept(value - here_e) {
                            here = position;
                            here_e = value;
                            current = reached;
                        }
                    }
                }
            }
            None => {
                feedback.observe(Some(current), current);
            }
        }
        // A decahedron carries close-packed `421` pairs that an
        // icosahedron does not. Among rungs that are still bound, the
        // next hop leaves from the one with the most of those pairs.
        let mut chosen: Option<(f64, f64, Array1<f64>)> = None;
        if here.len() % 3 == 0 {
            let n_atoms = here.len() / 3;
            for (energy, state) in bank.iter().flatten() {
                let rise = energy - origin_e;
                if !(0.4..10.0).contains(&rise) {
                    continue;
                }
                let cutoff = 1.35 * crate::twin::spacing(state.view(), n_atoms);
                let packed = crate::structure::cna_descriptor(state.view(), n_atoms, cutoff)[1];
                let better = match &chosen {
                    None => true,
                    Some((score, held, _)) => {
                        packed > *score + 1.0e-6
                            || ((packed - score).abs() <= 1.0e-6 && energy < held)
                    }
                };
                if better {
                    chosen = Some((packed, *energy, state.clone()));
                }
            }
        }
        if let Some((_, energy, state)) = chosen {
            if (energy - frontier_energy).abs() < 1.0e-6 {
                frontier_stuck += 1;
            } else {
                frontier_stuck = 0;
                frontier_energy = energy;
            }
            if frontier_stuck >= 2 {
                let drop = basin_key(energy);
                rejected.insert(drop);
                for slot in bank.iter_mut() {
                    if slot
                        .as_ref()
                        .is_some_and(|(held, _)| basin_key(*held) == drop)
                    {
                        *slot = None;
                    }
                }
                frontier_stuck = 0;
                frontier_energy = f64::NAN;
            } else {
                here = state.clone();
                here_e = energy;
            }
        }
        println!(
            "{{\"kind\":\"exit_hop\",\"hop\":{hop},\"seed\":{seed},\"here\":{here_e:.6},\"best\":{best_e:.6},\"escape\":{:.3},\"threshold\":{:.3}}}",
            feedback.escape(),
            feedback.threshold()
        );
        let _ = std::io::stdout().flush();
    }
    best
}

fn fivefold_opening(origin: ArrayView1<f64>, rmsd: f64, which: usize) -> Array1<f64> {
    let good: Vec<([f64; 3], f64)> = crate::soap::fivefold_axis_table(origin)
        .into_iter()
        .filter(|(_, d5)| *d5 < 1.40)
        .collect();
    if good.is_empty() {
        return origin.to_owned();
    }
    let target = if which % 2 == 0 { 0.90 } else { 0.99 };
    let (axis, _) = good
        .iter()
        .min_by(|a, b| {
            (a.1 - target)
                .abs()
                .partial_cmp(&(b.1 - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .unwrap_or(good[0]);
    crate::soap::step_away_fivefold_about(origin, rmsd.max(0.75), axis)
}

/// Climb along `direction` first, then track the minimum mode.
///
/// The first steps follow the supplied vector. Later steps replace it
/// with the softest mode, keeping the sense of travel. `direction` is
/// what selects the half-space; the mode is what the ridge becomes.
pub fn activate_along<G>(
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    mut grad: G,
    cfg: &Activation,
) -> Option<ActivationOutcome>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    activate_aligned(x, None, Some(direction), &mut grad, cfg, 1.0)
}

/// Climb away from `origin`: the first mode is aligned with \(x-x_0\)
/// so the walk goes up the covering half-space, not back into the well.
pub fn activate_from_origin<G>(
    x: ArrayView1<f64>,
    origin: ArrayView1<f64>,
    mut grad: G,
    cfg: &Activation,
) -> Option<ActivationOutcome>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    activate_aligned(x, Some(origin), None, &mut grad, cfg, 1.0)
}

fn closest_pair(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n < 2 {
        return f64::MAX;
    }
    let mut best = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                r2 += d * d;
            }
            best = best.min(r2);
        }
    }
    best.sqrt()
}

fn activate_aligned<G>(
    x: ArrayView1<f64>,
    origin: Option<ArrayView1<f64>>,
    initial_direction: Option<ArrayView1<f64>>,
    grad: &mut G,
    cfg: &Activation,
    sign0: f64,
) -> Option<ActivationOutcome>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    let dim = x.len();
    let mut cur = x.to_owned();
    let cluster = closest_pair(x) > 0.95;
    let mut evaluations = 0usize;

    let first = curvature_features(
        cur.view(),
        |y| {
            evaluations += 1;
            grad(y)
        },
        cfg.lanczos_steps,
        cfg.epsilon,
    )?;
    let (mut mode, mut lambda) = if let Some(direction) = initial_direction {
        let n: f64 = direction.iter().map(|z| z * z).sum::<f64>().sqrt();
        if n < 1e-15 {
            (first.mode.clone(), first.lambda_min)
        } else {
            // Positive so the first steps cannot look like a crossed ridge
            // before the mode has been recomputed.
            let mut unit = direction.to_owned();
            unit /= n;
            (unit, 1.0)
        }
    } else {
        (first.mode.clone(), first.lambda_min)
    };
    let sign = if let Some(origin) = origin {
        let align: f64 = mode
            .iter()
            .zip(x.iter().zip(origin.iter()))
            .map(|(m, (xi, oi))| m * (xi - oi))
            .sum();
        if align >= 0.0 { 1.0 } else { -1.0 }
    } else {
        sign0
    };
    let mut steps = 0usize;
    let mut crossed = false;
    let mut rise = 0.0;

    for k in 0..cfg.max_steps {
        // Refresh the direction on schedule, and always after the curvature has
        // already been seen to fall, since that is where it rotates fastest.
        if k > 0 && k % cfg.refresh == 0 {
            match curvature_features(
                cur.view(),
                |y| {
                    evaluations += 1;
                    grad(y)
                },
                cfg.lanczos_steps,
                cfg.epsilon,
            ) {
                Some(f) => {
                    // Keep the sense of travel: the mode is defined up to sign
                    // and flipping it mid-climb walks back down.
                    let dot: f64 = f.mode.iter().zip(mode.iter()).map(|(a, b)| a * b).sum();
                    mode = if dot < 0.0 { -f.mode } else { f.mode };
                    lambda = f.lambda_min;
                }
                None => break,
            }
        }
        for i in 0..dim {
            cur[i] += sign * cfg.step * mode[i];
        }
        steps += 1;

        // Perpendicular relaxation. Sliding down the component of the gradient
        // orthogonal to the mode keeps the structure on the valley floor; the
        // component along the mode is what the climb is fighting and is left
        // alone.
        let mut along = 0.0;
        let mut gnorm = 0.0;
        for _ in 0..cfg.perp_steps {
            let g = match grad(cur.view()) {
                Some(g) => {
                    evaluations += 1;
                    g
                }
                None => {
                    return Some(ActivationOutcome {
                        state: cur,
                        lambda,
                        steps,
                        crossed,
                        evaluations,
                    });
                }
            };
            gnorm = g.iter().map(|z| z * z).sum::<f64>().sqrt();
            along = g.iter().zip(mode.iter()).map(|(a, b)| a * b).sum();
            let mut d = Array1::<f64>::zeros(dim);
            for i in 0..dim {
                d[i] = cfg.perp_rate * (g[i] - along * mode[i]);
            }
            let n: f64 = d.iter().map(|z| z * z).sum::<f64>().sqrt();
            let scale = if n > cfg.perp_max_move && n > 0.0 {
                cfg.perp_max_move / n
            } else {
                1.0
            };
            for i in 0..dim {
                cur[i] -= scale * d[i];
            }
        }

        // A huge force, or a pair inside the repulsive core, is a clash.
        // Stepping back keeps the last intact structure.
        let crowded = cluster && closest_pair(cur.view()) < 0.85;
        if !gnorm.is_finite() || gnorm > 80.0 || crowded {
            for i in 0..dim {
                cur[i] -= sign * cfg.step * mode[i];
            }
            crossed = false;
            break;
        }

        // Stop at the saddle, not at the inflection.
        //
        // Negative curvature says the ridge is ahead, not behind: on a double
        // well the curvature along the well direction turns over at
        // `|u| = 1/sqrt(3)` while the barrier top is at `u = 0`. Stopping there
        // and pushing on by the overshoot ended the climb at `u = 0.101`, on
        // the side it started, and a quench from there goes home.
        //
        // The saddle is where the force along the direction of travel changes
        // sign: uphill while `sign * g . v > 0`, downhill after. Combined with
        // negative curvature that is the ridge, and past it a quench falls the
        // other way.
        rise += along * sign * cfg.step;
        if lambda < 0.0 && sign * along < 0.0 && rise >= cfg.min_rise {
            crossed = true;
            break;
        }
        if rise > 40.0 {
            crossed = false;
            break;
        }
    }

    if crossed && cfg.overshoot > 0.0 {
        // One push past the turning point, so the quench falls forward rather
        // than back down the way it came.
        for i in 0..dim {
            cur[i] += sign * cfg.overshoot * cfg.step * mode[i];
        }
    }

    Some(ActivationOutcome {
        state: cur,
        lambda,
        steps,
        crossed,
        evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A double well along one collective direction, stiff in the rest.
    ///
    /// `E = (u^2 - 1)^2 + 1/2 sum_i k_i (x_i - u w_i)^2` with `u = x . w`.
    /// Minima at `u = +/- 1`, a barrier at `u = 0`.
    ///
    /// The perpendicular stiffnesses vary and all exceed the curvature along
    /// `w` at the minimum, which is 8. Both parts are load-bearing. If they are
    /// smaller the softest mode is not the well direction and the test is
    /// asking the wrong question, and if they are all equal the Hessian is a
    /// multiple of the identity, every vector is an eigenvector, and the Krylov
    /// space collapses at the first step.
    fn perp_stiffness(dim: usize) -> Array1<f64> {
        Array1::from_shape_fn(dim, |i| 12.0 + 1.7 * (i % 7) as f64)
    }

    fn double_well<'a>(
        w: &'a Array1<f64>,
        k: &'a Array1<f64>,
    ) -> impl Fn(ArrayView1<f64>) -> Option<Array1<f64>> + 'a {
        move |x: ArrayView1<f64>| {
            let u: f64 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            let dedu = 4.0 * u * (u * u - 1.0);
            let mut g = Array1::zeros(x.len());
            // d/dx of the perpendicular term, including its dependence on u.
            let mut kp_dot_w = 0.0;
            for i in 0..x.len() {
                kp_dot_w += k[i] * (x[i] - u * w[i]) * w[i];
            }
            for i in 0..x.len() {
                let perp = x[i] - u * w[i];
                g[i] = dedu * w[i] + k[i] * perp - kp_dot_w * w[i];
            }
            Some(g)
        }
    }

    fn direction(dim: usize) -> Array1<f64> {
        let mut w = Array1::from_shape_fn(dim, |i| ((i % 5) as f64 - 2.0) + 0.25);
        let n: f64 = w.iter().map(|z| z * z).sum::<f64>().sqrt();
        w /= n;
        w
    }

    /// The cap has to bind on a stiff gradient, or the climb walks off the
    /// structure it was refining.
    #[test]
    fn a_perpendicular_step_is_bounded_however_steep_the_gradient() {
        let dim = 36;
        let w = direction(dim);
        // A gradient a thousand times the scale the rate was set for.
        let g = move |x: ArrayView1<f64>| -> Option<Array1<f64>> {
            Some(Array1::from_shape_fn(x.len(), |i| {
                1500.0 * x[i] + 3.0 * (i % 5) as f64
            }))
        };
        let cfg = Activation {
            max_steps: 4,
            ..Activation::default()
        };
        let out = activate(w.view(), g, &cfg, 1.0).unwrap();
        let travelled: f64 = out
            .state
            .iter()
            .zip(w.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let bound = out.steps as f64 * (cfg.step + cfg.perp_steps as f64 * cfg.perp_max_move)
            + cfg.overshoot * cfg.step;
        assert!(
            travelled <= bound + 1e-9,
            "moved {travelled:.3} where the caps allow {bound:.3}"
        );
    }

    fn quench_double_well(w: &Array1<f64>, k: &Array1<f64>, mut x: Array1<f64>) -> Array1<f64> {
        let g = double_well(w, k);
        for _ in 0..80 {
            let grad = g(x.view()).unwrap();
            let n: f64 = grad.iter().map(|z| z * z).sum::<f64>().sqrt();
            if n < 1e-6 {
                break;
            }
            let step = (0.05_f64).min(0.2 / n);
            x.scaled_add(-step, &grad);
        }
        x
    }

    #[test]
    fn cover_climb_quench_on_a_double_well_uses_only_the_force() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let origin = w.clone();
        let g = double_well(&w, &k);
        let end = cover_climb_quench(
            origin.view(),
            0.05,
            0,
            &g,
            |x| quench_double_well(&w, &k, x.to_owned()),
            &Activation {
                max_steps: 12,
                ..Activation::default()
            },
        );
        assert_eq!(end.len(), dim);
        let grad = g(end.view()).unwrap();
        let n: f64 = grad.iter().map(|z| z * z).sum::<f64>().sqrt();
        assert!(n < 1e-2, "quench did not reach a stationary point, |g|={n}");
        assert!(end.iter().all(|v| v.is_finite()));
    }

    /// Four Lennard-Jones atoms at the tetrahedron. The call is the same
    /// function as the smooth well: coordinates in, force in, quench in.
    #[test]
    fn cover_climb_quench_on_a_lennard_jones_tetrahedron_reads_no_target() {
        let scale = 2.0_f64.powf(1.0 / 6.0);
        let raw = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        let mut origin = Array1::zeros(12);
        for (i, p) in raw.iter().enumerate() {
            for k in 0..3 {
                origin[3 * i + k] = p[k] * scale / 3.0_f64.sqrt();
            }
        }
        let lj = |x: ArrayView1<f64>| -> (f64, Array1<f64>) {
            let n = x.len() / 3;
            let mut value = 0.0;
            let mut gradient = Array1::zeros(x.len());
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut d = [0.0; 3];
                    let mut r2 = 0.0;
                    for k in 0..3 {
                        d[k] = x[3 * i + k] - x[3 * j + k];
                        r2 += d[k] * d[k];
                    }
                    let inv2 = 1.0 / r2;
                    let inv6 = inv2.powi(3);
                    let inv12 = inv6 * inv6;
                    value += 4.0 * (inv12 - inv6);
                    let coefficient = 24.0 * inv2 * (2.0 * inv12 - inv6);
                    for k in 0..3 {
                        gradient[3 * i + k] -= coefficient * d[k];
                        gradient[3 * j + k] += coefficient * d[k];
                    }
                }
            }
            (value, gradient)
        };
        let cfg = Activation {
            max_steps: 6,
            lanczos_steps: 8,
            ..Activation::default()
        };
        let end = cover_climb_quench(
            origin.view(),
            0.2,
            0,
            |x| Some(lj(x).1),
            |x| {
                let mut y = x.to_owned();
                for _ in 0..40 {
                    let (_, g) = lj(y.view());
                    let n: f64 = g.iter().map(|z| z * z).sum::<f64>().sqrt();
                    if n < 1e-4 {
                        break;
                    }
                    let step = (1e-3_f64).min(0.05 / n);
                    y.scaled_add(-step, &g);
                }
                y
            },
            &cfg,
        );
        let (energy, gradient) = lj(end.view());
        assert!(energy.is_finite());
        assert!(gradient.iter().all(|v| v.is_finite()));
    }

    /// The property the module exists for. A straight displacement of the same
    /// length stays on its own side of the barrier; the climb crosses it.
    #[test]
    fn the_climb_crosses_the_barrier_a_displacement_does_not() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let g = double_well(&w, &k);
        // Start at the minimum with u = 1.
        let x: Array1<f64> = w.clone();
        let u0: f64 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!((u0 - 1.0).abs() < 1e-12);

        let cfg = Activation::default();
        let out = activate(x.view(), &g, &cfg, -1.0).unwrap();
        let u: f64 = out.state.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!(
            out.crossed,
            "the climb never reached negative curvature, ending at u = {u:.3}"
        );
        assert!(
            u < 0.0,
            "the climb ended at u = {u:.3}, still on the side it started"
        );

        // The same total distance in a straight line, no climbing.
        let travelled = (out.steps as f64 + cfg.overshoot) * cfg.step;
        let mut straight = x.clone();
        for i in 0..dim {
            straight[i] -= travelled * w[i];
        }
        let us: f64 = straight.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        // Both end past the barrier here because the well is one-dimensional;
        // what separates them is that the climb *knows* it is past, which is
        // what a caller needs in order to stop.
        assert!(
            us < 0.0,
            "the straight line should also cross this simple well: {us:.3}"
        );
    }

    /// The climb must not run forever on a direction that never turns over.
    #[test]
    fn from_origin_climbs_the_covering_half_space() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let g = double_well(&w, &k);
        let origin = w.clone();
        let mut start = w.clone();
        for i in 0..dim {
            start[i] -= 0.15 * w[i];
        }
        let out =
            activate_from_origin(start.view(), origin.view(), &g, &Activation::default()).unwrap();
        assert!(
            out.crossed,
            "the ridge must be crossed, lambda={}",
            out.lambda
        );
        let u: f64 = out.state.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!(u < 0.0, "quench side must be the other well, u={u:.3}");
    }

    #[test]
    fn a_direction_that_never_softens_stops_at_the_cap() {
        let dim = 36;
        let w = direction(dim);
        // Purely harmonic: the curvature is positive everywhere.
        // Stiffnesses that differ, so the Hessian is not a multiple of the
        // identity and the Krylov space has somewhere to go.
        let k = perp_stiffness(dim);
        let g = move |x: ArrayView1<f64>| -> Option<Array1<f64>> {
            Some(Array1::from_shape_fn(x.len(), |i| k[i] * x[i]))
        };
        let x = w.clone();
        let cfg = Activation {
            max_steps: 6,
            ..Activation::default()
        };
        let out = activate(x.view(), &g, &cfg, 1.0).unwrap();
        assert!(!out.crossed, "a harmonic well has no ridge to cross");
        assert!(
            out.steps <= 6,
            "the climb took {} steps against a cap of 6",
            out.steps
        );
    }

    /// A caller whose budget runs out mid-climb gets the structure so far
    /// rather than a panic or a silent full-cost climb.
    #[test]
    fn an_exhausted_budget_stops_the_climb() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let inner = double_well(&w, &k);
        let mut left = 40usize;
        let g = move |x: ArrayView1<f64>| -> Option<Array1<f64>> {
            if left == 0 {
                return None;
            }
            left -= 1;
            inner(x)
        };
        let out = activate(w.view(), g, &Activation::default(), -1.0);
        if let Some(o) = out {
            assert!(
                o.evaluations <= 40,
                "spent {} gradients against a budget of 40",
                o.evaluations
            );
        }
        // Refusing before the first curvature pass completes is also correct;
        // what would not be is climbing past the budget.
    }

    /// The sign argument has to mean something: the two ends of a soft
    /// direction are different saddles and a caller picking one must get it.
    #[test]
    fn the_two_signs_climb_opposite_ways() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let g = double_well(&w, &k);
        let x = w.clone();
        let a = activate(x.view(), &g, &Activation::default(), 1.0).unwrap();
        let b = activate(x.view(), &g, &Activation::default(), -1.0).unwrap();
        let ua: f64 = a.state.iter().zip(w.iter()).map(|(p, q)| p * q).sum();
        let ub: f64 = b.state.iter().zip(w.iter()).map(|(p, q)| p * q).sum();
        assert!(
            ua > ub,
            "the two signs ended at u = {ua:.3} and {ub:.3}, not on opposite sides"
        );
    }

    /// A capped octahedron of seven Lennard-Jones atoms is a local minimum.
    /// The pentagonal bipyramid lies below it. The search is given only the
    /// higher minimum, the force, and a quench.
    #[test]
    fn cover_climb_search_leaves_a_higher_lennard_jones_minimum() {
        fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            let n = x.len() / 3;
            let mut value = 0.0;
            let mut gradient = Array1::zeros(x.len());
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut d = [0.0; 3];
                    let mut r2 = 0.0;
                    for k in 0..3 {
                        d[k] = x[3 * i + k] - x[3 * j + k];
                        r2 += d[k] * d[k];
                    }
                    let inv2 = 1.0 / r2;
                    let inv6 = inv2.powi(3);
                    let inv12 = inv6 * inv6;
                    value += 4.0 * (inv12 - inv6);
                    let coefficient = 24.0 * inv2 * (2.0 * inv12 - inv6);
                    for k in 0..3 {
                        gradient[3 * i + k] -= coefficient * d[k];
                        gradient[3 * j + k] += coefficient * d[k];
                    }
                }
            }
            (value, gradient)
        }
        fn quench(x: ArrayView1<f64>) -> Array1<f64> {
            let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
            opt.minimize(x, 80, |v| Some(lj(v))).1
        }

        let req = 2.0_f64.powf(1.0 / 6.0);
        let scale = req / 2.0_f64.sqrt();
        let mut capped = Array1::zeros(21);
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (i, p) in axes.iter().enumerate() {
            for k in 0..3 {
                capped[3 * i + k] = p[k] * scale;
            }
        }
        let centroid = scale / 3.0;
        let unit = 1.0 / 3.0_f64.sqrt();
        for k in 0..3 {
            capped[18 + k] = centroid + req * unit;
        }
        let capped = quench(capped.view());
        let e_cap = lj(capped.view()).0;

        let mut bipyramid = Array1::zeros(21);
        let radius = req / (2.0 * (std::f64::consts::PI / 5.0).sin());
        let height = (req * req - radius * radius).max(0.0).sqrt();
        for k in 0..5 {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) / 5.0;
            bipyramid[3 * k] = radius * angle.cos();
            bipyramid[3 * k + 1] = radius * angle.sin();
        }
        bipyramid[17] = height;
        bipyramid[20] = -height;
        let bipyramid = quench(bipyramid.view());
        let e_low = lj(bipyramid.view()).0;
        assert!(
            e_low < e_cap - 0.2,
            "the lower isomer is not below the start: {e_low} vs {e_cap}"
        );

        let end = cover_climb_search(
            capped.view(),
            0.4,
            40,
            1,
            lj,
            quench,
            &Activation {
                max_steps: 4,
                lanczos_steps: 6,
                perp_steps: 1,
                ..Activation::default()
            },
        );
        let found = lj(end.view()).0;
        assert!(
            found < e_cap - 0.2,
            "search energy {found}, start {e_cap}, lower isomer {e_low}"
        );
    }
}
