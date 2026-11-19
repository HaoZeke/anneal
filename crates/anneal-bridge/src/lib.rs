//! Stalled rigid-body archive walk.
//!
//! Compiled here, and called through function pointers, so the search crate
//! does not contain the loop. A molecular quench then keeps the
//! translation-rotation step it had without this walk.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Search operations. `ctx` is the search's own state; the pointers below
/// are its existing helpers.
pub struct Ops {
    pub count_entry: fn(),
    pub offer: fn(*mut u8, f64, ArrayView1<f64>, u64),
    pub remaining: fn(*mut u8) -> usize,
    pub stopped: fn(*mut u8) -> bool,
    pub best_energy: fn(*mut u8) -> f64,
    pub has_best: fn(*mut u8) -> bool,
    pub copy_best: fn(*mut u8) -> Array1<f64>,
    pub has_side: fn(*mut u8) -> bool,
    pub side_energy: fn(*mut u8) -> f64,
    pub copy_side: fn(*mut u8) -> Array1<f64>,
    pub initial_energy: fn(*mut u8) -> f64,
    pub copy_initial: fn(*mut u8) -> Array1<f64>,
    pub cage_key: fn(ArrayView1<f64>, usize, f64) -> u64,
    pub recentre: fn(&mut Array1<f64>, usize),
    pub shell_turn: fn(ArrayView1<f64>, usize, usize, f64) -> Array1<f64>,
    pub centres_clear: fn(ArrayView1<f64>, usize, f64) -> bool,
    pub centres_parted: fn(ArrayView1<f64>, ArrayView1<f64>, usize, f64) -> bool,
    pub identity: fn(ArrayView1<f64>, ArrayView1<f64>) -> bool,
    pub random_cluster: fn(usize, f64, &mut StdRng) -> Array1<f64>,
    pub contain: fn(*mut u8, &mut Array1<f64>),
    pub has_geometry: fn(*mut u8) -> bool,
    pub count_scan: fn(),
    pub water_sites: fn(*mut u8) -> Option<(usize, [usize; 2])>,
    pub lab_sites: fn(*mut u8, ArrayView1<f64>) -> Vec<Vec<[f64; 3]>>,
    pub donations: fn(&[Vec<[f64; 3]>], usize, [usize; 2]) -> Vec<Vec<(usize, usize)>>,
    pub frame_rotation: fn(([f64; 3], [f64; 3]), ([f64; 3], [f64; 3])) -> [[f64; 3]; 3],
    pub compose_rotation: fn([[f64; 3]; 3], [f64; 3]) -> [f64; 3],
    pub rewire_oo_max: fn() -> f64,
    pub propose: fn(*mut u8, usize, ArrayView1<f64>, f64, f64, &mut StdRng) -> Array1<f64>,
    pub relax: fn(*mut u8, ArrayView1<f64>, usize) -> (f64, Array1<f64>),
    pub record: fn(*mut u8, f64, ArrayView1<f64>),
    pub reset_bandit: fn(*mut u8, usize),
    pub bandit_select: fn(*mut u8, &mut StdRng) -> usize,
    pub bandit_update: fn(*mut u8, usize, bool),
    pub name_is: fn(*mut u8, usize, &str) -> bool,
    pub n_points: fn(*mut u8) -> usize,
    pub temperature: fn(*mut u8) -> f64,
    pub min_separation: fn(*mut u8) -> f64,
    pub relax_steps: fn(*mut u8) -> usize,
    pub shell_angle: fn(*mut u8) -> f64,
    pub n_kernels: fn(*mut u8) -> usize,
}

/// One stalled rigid archive walk.
///
/// The returned point replaces the hop's energy and coordinates when the
/// ledger holds something lower.
#[cold]
#[inline(never)]
pub fn walk(
    ops: &Ops,
    ctx: *mut u8,
    energy: f64,
    x: ArrayView1<f64>,
    hops: usize,
    cage_quantum: f64,
    orient_scale: f64,
    new_lowest: f64,
) -> Option<(f64, Array1<f64>)> {
    let n_points = (ops.n_points)(ctx);
    let temperature = (ops.temperature)(ctx);
    let min_separation = (ops.min_separation)(ctx);
    let relax_steps = (ops.relax_steps)(ctx);
    let shell_angle = (ops.shell_angle)(ctx);
    let n_kernels = (ops.n_kernels)(ctx);
    let rotate_arm = (0..n_kernels).find(|arm| (ops.name_is)(ctx, *arm, "rrot"));
    (ops.count_entry)();
    if (ops.has_side)(ctx) {
        let side = (ops.copy_side)(ctx);
        (ops.record)(ctx, (ops.side_energy)(ctx), side.view());
    }
    let key = (ops.cage_key)(x, n_points, cage_quantum);
    (ops.offer)(ctx, energy, x, key);
    if (ops.has_best)(ctx) {
        let best = (ops.copy_best)(ctx);
        let best_key = (ops.cage_key)(best.view(), n_points, cage_quantum);
        (ops.offer)(ctx, (ops.best_energy)(ctx), best.view(), best_key);
    }
    let (mut side_energy, mut side_state) = if (ops.has_best)(ctx) {
        let best = (ops.copy_best)(ctx);
        if best.len() == x.len() && (ops.best_energy)(ctx) <= energy {
            ((ops.best_energy)(ctx), best)
        } else {
            (energy, x.to_owned())
        }
    } else {
        (energy, x.to_owned())
    };
    // A fresh generator, not a draw on the walker's.
    let mut side_rng = StdRng::seed_from_u64(
        (hops as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(key)
            .wrapping_add(side_energy.to_bits()),
    );
    let side_scale = orient_scale;
    (ops.reset_bandit)(ctx, n_kernels);
    let mut branch_best = side_energy;
    let mut branch_stale = 0usize;
    let mut branch_idx = 0usize;
    while (ops.remaining)(ctx) > 0 && !(ops.stopped)(ctx) {
        let stale_cap = if (2..7).contains(&branch_idx) {
            500
        } else {
            350
        };
        if branch_stale >= stale_cap {
            branch_idx += 1;
            branch_stale = 0;
            (ops.reset_bandit)(ctx, n_kernels);
            if branch_idx >= 7 && !branch_idx.is_multiple_of(2) {
                let initial_e = (ops.initial_energy)(ctx);
                let initial_x = (ops.copy_initial)(ctx);
                side_energy = initial_e;
                side_state = initial_x;
                branch_best = side_energy;
                continue;
            }
            let angle = shell_angle;
            if branch_idx < 7 && branch_idx % 3 != 1 && angle > 0.0 {
                let base_energy = (ops.best_energy)(ctx).min(side_energy);
                let mut base_state = if (ops.has_best)(ctx) {
                    (ops.copy_best)(ctx)
                } else {
                    side_state.clone()
                };
                if branch_idx >= 4
                    && let Some(rot_idx) = rotate_arm
                {
                    base_state = (ops.propose)(
                        ctx,
                        rot_idx,
                        base_state.view(),
                        temperature,
                        8.0,
                        &mut side_rng,
                    );
                    (ops.recentre)(&mut base_state, n_points);
                }
                let sign = if (branch_idx / 3).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                let mut turned: Option<(f64, Array1<f64>)> = None;
                for axis in 0..3 {
                    if (ops.remaining)(ctx) == 0 || (ops.stopped)(ctx) {
                        break;
                    }
                    let trial = (ops.shell_turn)(base_state.view(), n_points, axis, sign * angle);
                    let (quenched, relaxed) = (ops.relax)(ctx, trial.view(), relax_steps);
                    if quenched.is_finite()
                        && relaxed.len() == side_state.len()
                        && (ops.centres_clear)(relaxed.view(), n_points, 0.5 * min_separation)
                    {
                        (ops.record)(ctx, quenched, relaxed.view());
                        if (quenched - base_energy).abs() > 0.15
                            && turned.as_ref().is_none_or(|(held, _)| quenched < *held)
                        {
                            turned = Some((quenched, relaxed));
                        }
                    }
                }
                if let Some((quenched, relaxed)) = turned {
                    side_energy = quenched;
                    side_state = relaxed;
                    branch_best = quenched;
                    continue;
                }
            }
            let trial = (ops.random_cluster)(n_points, min_separation, &mut side_rng);
            let (quenched, relaxed) = (ops.relax)(ctx, trial.view(), relax_steps.saturating_mul(2));
            if !quenched.is_finite() || relaxed.len() != side_state.len() {
                if (ops.remaining)(ctx) == 0 || (ops.stopped)(ctx) {
                    break;
                }
                branch_stale = stale_cap;
                continue;
            }
            (ops.record)(ctx, quenched, relaxed.view());
            side_energy = quenched;
            side_state = relaxed;
            branch_best = quenched;
            continue;
        }
        branch_stale += 1;
        let mut arm = (ops.bandit_select)(ctx, &mut side_rng);
        let mut trial = if (ops.name_is)(ctx, arm, "rcycle")
            && (branch_idx >= 7 || side_rng.random_bool(0.5))
        {
            let mut out = side_state.clone();
            if (ops.has_geometry)(ctx) {
                let fns = ChainFns {
                    count_scan: ops.count_scan,
                    water_sites: ops.water_sites,
                    lab_sites: ops.lab_sites,
                    donations: ops.donations,
                    frame_rotation: ops.frame_rotation,
                    compose_rotation: ops.compose_rotation,
                    rewire_oo_max: ops.rewire_oo_max,
                };
                reverse_directed(&fns, ctx, n_points, &mut out, &mut side_rng);
            }
            out
        } else {
            (ops.propose)(
                ctx,
                arm,
                side_state.view(),
                temperature,
                side_scale,
                &mut side_rng,
            )
        };
        if (ops.identity)(side_state.view(), trial.view()) {
            for _ in 0..n_kernels {
                arm = (arm + 1) % n_kernels;
                if !(ops.name_is)(ctx, arm, "rcycle") {
                    break;
                }
            }
            trial = (ops.propose)(
                ctx,
                arm,
                side_state.view(),
                temperature,
                side_scale,
                &mut side_rng,
            );
        }
        (ops.recentre)(&mut trial, n_points);
        (ops.contain)(ctx, &mut trial);
        let (quenched, relaxed) = (ops.relax)(ctx, trial.view(), relax_steps);
        if !quenched.is_finite() || relaxed.len() != side_state.len() {
            if (ops.remaining)(ctx) == 0 || (ops.stopped)(ctx) {
                break;
            }
            continue;
        }
        (ops.record)(ctx, quenched, relaxed.view());
        let improved_branch = quenched < branch_best - new_lowest;
        if improved_branch {
            branch_best = quenched;
            branch_stale = 0;
        }
        let side_temp =
            if (2..7).contains(&branch_idx) && side_energy < branch_best + 3.0 * temperature {
                temperature * (1.0 + (branch_stale as f64 / 100.0).min(2.0))
            } else {
                temperature
            };
        let rise = (quenched - side_energy) / side_temp.max(1e-12);
        let accept_side = rise <= 0.0 || side_rng.random::<f64>() < (-rise).exp();
        let parted = (ops.centres_parted)(
            side_state.view(),
            relaxed.view(),
            n_points,
            0.12 * min_separation,
        );
        let reward =
            improved_branch || (accept_side && parted && quenched < side_energy - new_lowest);
        (ops.bandit_update)(ctx, arm, reward);
        if accept_side {
            side_energy = quenched;
            side_state = relaxed;
        }
    }
    if (ops.has_best)(ctx) && (ops.best_energy)(ctx) < energy {
        Some(((ops.best_energy)(ctx), (ops.copy_best)(ctx)))
    } else {
        None
    }
}

/// Geometry queries the chain reversal needs. The bodies stay in the search
/// crate; only this control flow is compiled here.
pub struct ChainFns {
    pub count_scan: fn(),
    pub water_sites: fn(*mut u8) -> Option<(usize, [usize; 2])>,
    pub lab_sites: fn(*mut u8, ArrayView1<f64>) -> Vec<Vec<[f64; 3]>>,
    pub donations: fn(&[Vec<[f64; 3]>], usize, [usize; 2]) -> Vec<Vec<(usize, usize)>>,
    pub frame_rotation: fn(([f64; 3], [f64; 3]), ([f64; 3], [f64; 3])) -> [[f64; 3]; 3],
    pub compose_rotation: fn([[f64; 3]; 3], [f64; 3]) -> [f64; 3],
    pub rewire_oo_max: fn() -> f64,
}

/// Reverses a directed hydrogen-bond cycle, including a stem donor within the
/// caller's O–O window of the closing molecule.
///
/// `ctx` is the search state the function pointers close over. They are called
/// one at a time.
#[cold]
#[inline(never)]
pub fn reverse_directed(
    fns: &ChainFns,
    ctx: *mut u8,
    n: usize,
    x: &mut Array1<f64>,
    rng: &mut StdRng,
) -> bool {
    (fns.count_scan)();
    let Some((o, h)) = (fns.water_sites)(ctx) else {
        return false;
    };
    if n < 2 || x.len() != 6 * n {
        return false;
    }
    let sites = (fns.lab_sites)(ctx, x.view());
    let donated = (fns.donations)(&sites, o, h);
    let donors: Vec<usize> = (0..n).filter(|&i| !donated[i].is_empty()).collect();
    if donors.is_empty() {
        return false;
    }
    let mut path: Vec<(usize, usize)> = Vec::new();
    let mut at = donors[rng.random_range(0..donors.len())];
    let chain = loop {
        let back = path.last().map(|&(m, _)| m);
        let out: Vec<(usize, usize)> = donated[at]
            .iter()
            .copied()
            .filter(|&(j, _)| Some(j) != back)
            .collect();
        if out.is_empty() || path.len() > n {
            return false;
        }
        let (next, which) = out[rng.random_range(0..out.len())];
        path.push((at, which));
        if let Some(start) = path.iter().position(|&(m, _)| m == next) {
            let last = path[path.len() - 1].0;
            let window = (fns.rewire_oo_max)();
            let cut = (0..start)
                .find(|&c| dist(sites[path[c].0][o], sites[last][o]) < window)
                .unwrap_or(start);
            break path.split_off(cut);
        }
        at = next;
    };
    let mut turns = Vec::with_capacity(chain.len());
    for (s, &(k, which)) in chain.iter().enumerate() {
        let prev = chain[(s + chain.len() - 1) % chain.len()].0;
        let oxygen = sites[k][o];
        let bonded = unit(sub(sites[k][h[which]], oxygen));
        let other = unit(sub(sites[k][h[1 - which]], oxygen));
        let target = unit(sub(sites[prev][o], oxygen));
        if norm(cross(target, other)) < 1e-6 {
            return false;
        }
        turns.push((
            k,
            oxygen,
            (fns.frame_rotation)((bonded, other), (target, other)),
        ));
    }
    for (k, oxygen, turn) in turns {
        let centre = [x[3 * k], x[3 * k + 1], x[3 * k + 2]];
        let moved = add(oxygen, mat_vec(turn, sub(centre, oxygen)));
        let base = 3 * n + 3 * k;
        let w = (fns.compose_rotation)(turn, [x[base], x[base + 1], x[base + 2]]);
        for a in 0..3 {
            x[3 * k + a] = moved[a];
            x[base + a] = w[a];
        }
    }
    true
}

/// A fresh rigid cluster: `n` centres in a sphere plus a random rotation each.
///
/// Same draw as the search's `random_rigid_cluster` for [`StdRng`]. Compiled
/// here so taking that generic's address does not land in the search crate.
#[cold]
#[inline(never)]
pub fn rigid_draw(
    n: usize,
    min_sep: f64,
    rng: &mut StdRng,
    fold: fn(usize, &mut [f64]),
) -> Array1<f64> {
    let radius = 1.35 * min_sep * (n.max(1) as f64).cbrt();
    let centres = sphere_sample(n, radius, min_sep, rng);
    let mut x = Array1::zeros(6 * n);
    for i in 0..(3 * n).min(centres.len()) {
        x[i] = centres[i];
    }
    for i in 0..n {
        let base = 3 * n + 3 * i;
        x[base] = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        x[base + 1] = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        x[base + 2] = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
    }
    if let Some(slice) = x.as_slice_mut() {
        fold(n, slice);
    }
    x
}

fn sphere_sample(n: usize, radius: f64, min_sep: f64, rng: &mut StdRng) -> Array1<f64> {
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut tries = 0;
    while pts.len() < n && tries < 20_000 {
        tries += 1;
        let mut v = [0.0; 3];
        let mut norm_sq = 0.0;
        for slot in &mut v {
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            *slot = libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * std::f64::consts::PI * u2);
            norm_sq += *slot * *slot;
        }
        let scale = libm::sqrt(norm_sq).max(1e-12);
        let r = radius * libm::cbrt(rng.random::<f64>());
        let p = [v[0] / scale * r, v[1] / scale * r, v[2] / scale * r];
        if pts.iter().all(|q| {
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
                >= min_sep
        }) {
            pts.push(p);
        }
    }
    let mut out = Array1::zeros(3 * pts.len());
    for (i, p) in pts.iter().enumerate() {
        out[3 * i] = p[0];
        out[3 * i + 1] = p[1];
        out[3 * i + 2] = p[2];
    }
    out
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let s = 1.0 / norm(a);
    [a[0] * s, a[1] * s, a[2] * s]
}

fn mat_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
