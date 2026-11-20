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
    pub offer: fn(*mut u8, f64, ArrayView1<f64>, u64),
    pub remaining: fn(*mut u8) -> usize,
    pub stopped: fn(*mut u8) -> bool,
    pub spent: fn(*mut u8) -> usize,
    pub best_energy: fn(*mut u8) -> f64,
    pub has_best: fn(*mut u8) -> bool,
    pub copy_best: fn(*mut u8) -> Array1<f64>,
    pub cage_key: fn(ArrayView1<f64>, usize, f64) -> u64,
    pub recentre: fn(&mut Array1<f64>, usize),
    pub identity: fn(ArrayView1<f64>, ArrayView1<f64>) -> bool,
    pub contain: fn(*mut u8, &mut Array1<f64>),
    pub has_geometry: fn(*mut u8) -> bool,
    pub reverses_cycle: fn(*mut u8, usize) -> bool,
    pub water_sites: fn(*mut u8) -> Option<(usize, [usize; 2])>,
    pub lab_sites: fn(*mut u8, ArrayView1<f64>) -> Vec<Vec<[f64; 3]>>,
    pub donations: fn(&[Vec<[f64; 3]>], usize, [usize; 2]) -> Vec<Vec<(usize, usize)>>,
    pub frame_rotation: fn(([f64; 3], [f64; 3]), ([f64; 3], [f64; 3])) -> [[f64; 3]; 3],
    pub compose_rotation: fn([[f64; 3]; 3], [f64; 3]) -> [f64; 3],
    pub stem_window: fn(*mut u8) -> f64,
    pub propose: fn(*mut u8, usize, ArrayView1<f64>, f64, f64, &mut StdRng) -> Array1<f64>,
    pub relax: fn(*mut u8, ArrayView1<f64>, usize) -> (f64, Array1<f64>),
    pub record: fn(*mut u8, f64, ArrayView1<f64>),
    pub reset_bandit: fn(*mut u8, usize),
    pub bandit_select: fn(*mut u8, &mut StdRng) -> usize,
    pub bandit_update: fn(*mut u8, usize, bool),
    pub n_points: fn(*mut u8) -> usize,
    pub temperature: fn(*mut u8) -> f64,
    pub relax_steps: fn(*mut u8) -> usize,
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
    archive_end: usize,
) -> Option<(f64, Array1<f64>)> {
    let n_points = (ops.n_points)(ctx);
    let temperature = (ops.temperature)(ctx);
    let relax_steps = (ops.relax_steps)(ctx);
    let n_kernels = (ops.n_kernels)(ctx);
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
    while (ops.spent)(ctx) < archive_end && (ops.remaining)(ctx) > 0 && !(ops.stopped)(ctx) {
        let mut arm = (ops.bandit_select)(ctx, &mut side_rng);
        let mut trial = if (ops.reverses_cycle)(ctx, arm) {
            let mut out = side_state.clone();
            if (ops.has_geometry)(ctx) {
                let fns = ChainFns {
                    water_sites: ops.water_sites,
                    lab_sites: ops.lab_sites,
                    donations: ops.donations,
                    frame_rotation: ops.frame_rotation,
                    compose_rotation: ops.compose_rotation,
                    stem_window: ops.stem_window,
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
                if !(ops.reverses_cycle)(ctx, arm) {
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
            break;
        }
        (ops.record)(ctx, quenched, relaxed.view());
        let improved = quenched < branch_best - new_lowest;
        if improved {
            branch_best = quenched;
        }
        let rise = (quenched - side_energy) / temperature.max(1e-12);
        let accept = rise <= 0.0 || side_rng.random::<f64>() < (-rise).exp();
        (ops.bandit_update)(ctx, arm, improved);
        if accept {
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
    pub water_sites: fn(*mut u8) -> Option<(usize, [usize; 2])>,
    pub lab_sites: fn(*mut u8, ArrayView1<f64>) -> Vec<Vec<[f64; 3]>>,
    pub donations: fn(&[Vec<[f64; 3]>], usize, [usize; 2]) -> Vec<Vec<(usize, usize)>>,
    pub frame_rotation: fn(([f64; 3], [f64; 3]), ([f64; 3], [f64; 3])) -> [[f64; 3]; 3],
    pub compose_rotation: fn([[f64; 3]; 3], [f64; 3]) -> [f64; 3],
    pub stem_window: fn(*mut u8) -> f64,
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
            let window = (fns.stem_window)(ctx);
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

#[cfg(test)]
mod tests {
    use super::{ChainFns, reverse_directed};
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// A donation walk that never revisits is not a cycle, including when it
    /// is longer than the cluster the caller asked to reverse.
    #[test]
    fn a_walk_longer_than_the_cluster_is_not_reversed() {
        fn water(_: *mut u8) -> Option<(usize, [usize; 2])> {
            Some((0, [1, 2]))
        }
        fn sites(_: *mut u8, _: ndarray::ArrayView1<f64>) -> Vec<Vec<[f64; 3]>> {
            (0..4)
                .map(|i| {
                    let o = [i as f64, 0.0, 0.0];
                    vec![o, [o[0] + 0.1, 0.0, 0.0], [o[0], 0.1, 0.0]]
                })
                .collect()
        }
        fn donations(_: &[Vec<[f64; 3]>], _: usize, _: [usize; 2]) -> Vec<Vec<(usize, usize)>> {
            vec![vec![(1, 0)], vec![(2, 0)], vec![(3, 0)], vec![]]
        }
        fn frame(_: ([f64; 3], [f64; 3]), _: ([f64; 3], [f64; 3])) -> [[f64; 3]; 3] {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        }
        fn compose(_: [[f64; 3]; 3], w: [f64; 3]) -> [f64; 3] {
            w
        }
        fn window(_: *mut u8) -> f64 {
            0.0
        }
        let fns = ChainFns {
            water_sites: water,
            lab_sites: sites,
            donations,
            frame_rotation: frame,
            compose_rotation: compose,
            stem_window: window,
        };
        let mut x = Array1::zeros(12);
        let mut rng = StdRng::seed_from_u64(1);
        assert!(!reverse_directed(
            &fns,
            std::ptr::null_mut(),
            2,
            &mut x,
            &mut rng
        ));
    }
}
