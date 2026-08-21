//! Even covering of the unit hypersphere for saddle-search starts.
//!
//! Plasencia Gutiérrez, Argáez and Jónsson, *J. Chem. Theory Comput.*
//! **13**, 1 (2017), DOI 10.1021/acs.jctc.5b01216 (the Optbench /
//! SoftSaddle heptamer). Instead of Gaussian displacements, starting
//! points lie on \(S^{d-1}\) and a Thomson-like repulsion
//! \(\sum_{i\neq j} 1/\gamma_{ij}^{s}\) with geodesic
//! \(\gamma=\arccos(\mathbf{v}_i\cdot\mathbf{v}_j)\) and \(s=1\)
//! spaces them. SoftSaddle `triangulation_hight_v2` (Manuel Plasencia
//! Gutiérrez) is the source of the Gauss–Seidel update. Occupancy
//! ArchiveHole takes one covering direction per extra, placed at the
//! existing RMSD cap: the covering is not a larger kick.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// Repulsion exponent \(s\) in \(1/\gamma^{s}\). SoftSaddle uses 1.
pub const COVER_S: f64 = 1.0;
/// Relative energy change that ends the Gauss–Seidel sweep.
pub const COVER_ETOL: f64 = 1e-6;
/// SoftSaddle default iteration cap (`Nitr = 2E3`).
pub const COVER_NITR: usize = 2000;
/// Smallest adaptive step.
const A_MIN: f64 = 1e-14;
/// Largest adaptive step.
const A_MAX: f64 = 0.2;
/// Grow / shrink factor on a successful or failed trial step.
const A_SCALE: f64 = 2.5;
/// Seed so every replica shares one covering of \((n, d)\).
const COVER_SEED: u64 = 1;

static COVER_CACHE: OnceLock<Mutex<HashMap<(usize, usize), Arc<Vec<Vec<f64>>>>>> = OnceLock::new();

/// Covering size: two directions per wave replica so a later Leave of
/// the same extra is not the same point. `CATALOG_WAVE` when set.
pub fn default_cover_size() -> usize {
    std::env::var("CATALOG_WAVE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&n| n >= 2)
        .unwrap_or(48)
        .saturating_mul(2)
        .max(64)
}

/// Unit vectors of length `dim`, Thomson-spaced on \(S^{d-1}\).
pub fn cover_points<R: rand::Rng + ?Sized>(
    n: usize,
    dim: usize,
    max_iter: usize,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    if n == 0 || dim == 0 {
        return Vec::new();
    }
    let mut points = random_sphere(n, dim, rng);
    if n == 1 {
        return points;
    }
    relax_cover(&mut points, max_iter);
    points
}

/// Shared covering for `(n, dim)`, farthest-point ordered.
pub fn shared_cover(n: usize, dim: usize) -> Arc<Vec<Vec<f64>>> {
    let cache = COVER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(existing) = guard.get(&(n, dim)) {
        return Arc::clone(existing);
    }
    let mut rng = StdRng::seed_from_u64(COVER_SEED);
    let points = cover_points(n, dim, COVER_NITR, &mut rng);
    let order = farthest_order(&points);
    let ordered: Vec<Vec<f64>> = order.iter().map(|&i| points[i].clone()).collect();
    let shared = Arc::new(ordered);
    guard.insert((n, dim), Arc::clone(&shared));
    shared
}

/// Direction `index` of the shared covering, wrapping.
pub fn cover_direction(n: usize, dim: usize, index: usize) -> Vec<f64> {
    let cover = shared_cover(n.max(1), dim.max(1));
    if cover.is_empty() {
        return vec![0.0; dim];
    }
    cover[index % cover.len()].clone()
}

/// Greedy order: each next point maximises the minimum geodesic to
/// those already chosen (JCTC 2017 Fig. 4).
pub fn farthest_order(points: &[Vec<f64>]) -> Vec<usize> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut order = Vec::with_capacity(n);
    // SoftSaddle starts from the highest-energy (most crowded) point.
    let start = (0..n)
        .max_by(|&left, &right| {
            pair_energy(points, left)
                .partial_cmp(&pair_energy(points, right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    remaining.retain(|&i| i != start);
    order.push(start);
    while !remaining.is_empty() {
        let next = remaining
            .iter()
            .copied()
            .max_by(|&left, &right| {
                min_geodesic_to_set(points, left, &order)
                    .partial_cmp(&min_geodesic_to_set(points, right, &order))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("remaining is non-empty");
        remaining.retain(|&i| i != next);
        order.push(next);
    }
    order
}

/// Place `min + radius * direction` at all-atom RMSD `rmsd`. Translation
/// of the mobile set is removed so a covering direction is not a COM
/// hop. Frozen atoms stay at `min`.
pub fn place_around(
    min: &[f64],
    direction: &[f64],
    rmsd: f64,
    mobile: Option<&[usize]>,
) -> Vec<f64> {
    let n_at = min.len() / 3;
    if n_at == 0 || direction.len() != min.len() {
        return min.to_vec();
    }
    let keep = atom_mask(n_at, mobile);
    let mut dr = vec![0.0; min.len()];
    for i in 0..n_at {
        if !keep[i] {
            continue;
        }
        for ax in 0..3 {
            dr[3 * i + ax] = direction[3 * i + ax];
        }
    }
    strip_com(&mut dr, &keep);
    let n = n_at.max(1) as f64;
    let cur = (dr.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    if cur < 1e-15 {
        return min.to_vec();
    }
    let cap = rmsd.max(1e-6);
    let scale = cap / cur;
    let mut out = min.to_vec();
    for i in 0..n_at {
        if !keep[i] {
            continue;
        }
        for ax in 0..3 {
            out[3 * i + ax] += scale * dr[3 * i + ax];
        }
    }
    out
}

/// All-atom RMSD between `left` and `right`.
pub fn all_atom_rmsd(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.len() < 3 {
        return 0.0;
    }
    let n = (left.len() / 3) as f64;
    let s: f64 = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    (s / n).sqrt()
}

/// Smallest pairwise geodesic on the covering.
pub fn min_geodesic(points: &[Vec<f64>]) -> f64 {
    let n = points.len();
    let mut best = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            best = best.min(geodesic(&points[i], &points[j]));
        }
    }
    best
}

fn random_sphere<R: rand::Rng + ?Sized>(n: usize, dim: usize, rng: &mut R) -> Vec<Vec<f64>> {
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = vec![0.0; dim];
        for item in &mut v {
            *item = StandardNormal.sample(rng);
        }
        normalize(&mut v);
        points.push(v);
    }
    points
}

fn relax_cover(points: &mut [Vec<f64>], max_iter: usize) {
    let n = points.len();
    let dim = points[0].len();
    let mut dots = vec![0.0; n * n];
    let mut geo = vec![0.0; n * n];
    let mut pair = vec![0.0; n * n];
    refresh_pairs(points, &mut dots, &mut geo, &mut pair);
    let mut energy: Vec<f64> = vec![pair.iter().sum()];
    let mut step = vec![A_MAX; n];
    let mut d_e = f64::INFINITY;
    let mut iter = 0usize;
    while iter < max_iter && d_e > COVER_ETOL {
        iter += 1;
        let mut contrib = vec![0.0; n];
        for i in 0..n {
            contrib[i] = (0..n).map(|j| pair[i * n + j]).sum();
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&left, &right| {
            contrib[right]
                .partial_cmp(&contrib[left])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &j in &order {
            let mut grad = vec![0.0; dim];
            let mut failed = false;
            for i in 0..n {
                if i == j {
                    continue;
                }
                let dot = dots[i * n + j].clamp(-1.0, 1.0);
                let g = geo[i * n + j];
                let tang = (1.0 - dot * dot).sqrt();
                let w = COVER_S / ((tang + f64::EPSILON) * (g.powf(COVER_S + 1.0) + f64::EPSILON));
                let mut nrm2 = 0.0;
                for ax in 0..dim {
                    let gax = w * points[i][ax];
                    grad[ax] += gax;
                    nrm2 += gax * gax;
                }
                if iter < 5 && nrm2 > 1e8 {
                    failed = true;
                    break;
                }
            }
            if failed {
                for point in points.iter_mut() {
                    for item in point.iter_mut() {
                        *item += 1e-5 * (hash_unit(*item) - 0.5);
                    }
                    normalize(point);
                }
                refresh_pairs(points, &mut dots, &mut geo, &mut pair);
                break;
            }
            let radial: f64 = grad.iter().zip(points[j].iter()).map(|(g, v)| g * v).sum();
            for ax in 0..dim {
                grad[ax] -= radial * points[j][ax];
            }
            let u_old: f64 = (0..n).map(|i| pair[j * n + i]).sum();
            loop {
                let mut trial = vec![0.0; dim];
                for ax in 0..dim {
                    trial[ax] = points[j][ax] - step[j] * grad[ax];
                }
                normalize(&mut trial);
                let mut trial_pair = vec![0.0; n];
                let mut trial_dot = vec![0.0; n];
                let mut trial_geo = vec![0.0; n];
                for i in 0..n {
                    if i == j {
                        trial_dot[i] = 1.0;
                        trial_geo[i] = f64::INFINITY;
                        trial_pair[i] = 0.0;
                        continue;
                    }
                    let d = trial
                        .iter()
                        .zip(points[i].iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        .clamp(-1.0, 1.0);
                    trial_dot[i] = d;
                    trial_geo[i] = d.acos();
                    trial_pair[i] = 1.0 / (trial_geo[i].powf(COVER_S) + f64::EPSILON);
                }
                let u_new: f64 = trial_pair.iter().sum();
                if u_new <= u_old {
                    points[j] = trial;
                    for i in 0..n {
                        dots[j * n + i] = trial_dot[i];
                        dots[i * n + j] = trial_dot[i];
                        geo[j * n + i] = trial_geo[i];
                        geo[i * n + j] = trial_geo[i];
                        pair[j * n + i] = trial_pair[i];
                        pair[i * n + j] = trial_pair[i];
                    }
                    step[j] = (step[j] * A_SCALE).min(A_MAX);
                    break;
                }
                if step[j] > A_MIN {
                    step[j] = (step[j] / A_SCALE).max(A_MIN);
                } else {
                    break;
                }
            }
        }
        let u_now: f64 = pair.iter().sum();
        energy.push(u_now);
        if iter >= 10 {
            let older = energy[iter.saturating_sub(2)];
            d_e = (older - u_now).abs() / 10.0;
        }
        if iter % 20 == 0 {
            step.fill(A_MAX);
        }
    }
}

fn refresh_pairs(points: &[Vec<f64>], dots: &mut [f64], geo: &mut [f64], pair: &mut [f64]) {
    let n = points.len();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                dots[i * n + j] = 1.0;
                geo[i * n + j] = f64::INFINITY;
                pair[i * n + j] = 0.0;
                continue;
            }
            let d = points[i]
                .iter()
                .zip(points[j].iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            dots[i * n + j] = d;
            geo[i * n + j] = d.acos();
            pair[i * n + j] = 1.0 / (geo[i * n + j].powf(COVER_S) + f64::EPSILON);
        }
    }
}

fn pair_energy(points: &[Vec<f64>], index: usize) -> f64 {
    points
        .iter()
        .enumerate()
        .map(|(j, other)| {
            if j == index {
                0.0
            } else {
                1.0 / (geodesic(&points[index], other).powf(COVER_S) + f64::EPSILON)
            }
        })
        .sum()
}

fn min_geodesic_to_set(points: &[Vec<f64>], index: usize, chosen: &[usize]) -> f64 {
    chosen
        .iter()
        .map(|&j| geodesic(&points[index], &points[j]))
        .fold(f64::INFINITY, f64::min)
}

fn geodesic(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .clamp(-1.0, 1.0)
        .acos()
}

fn normalize(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n < 1e-15 {
        if let Some(first) = v.first_mut() {
            *first = 1.0;
        }
        return;
    }
    for item in v.iter_mut() {
        *item /= n;
    }
}

fn atom_mask(n_at: usize, mobile: Option<&[usize]>) -> Vec<bool> {
    match mobile {
        None => vec![true; n_at],
        Some(set) => {
            let mut keep = vec![false; n_at];
            for &i in set {
                if i < n_at {
                    keep[i] = true;
                }
            }
            keep
        }
    }
}

fn strip_com(dr: &mut [f64], keep: &[bool]) {
    let mut count = 0.0;
    let mut com = [0.0; 3];
    for (i, on) in keep.iter().enumerate() {
        if !on {
            continue;
        }
        count += 1.0;
        for ax in 0..3 {
            com[ax] += dr[3 * i + ax];
        }
    }
    if count < 1.0 {
        return;
    }
    for item in &mut com {
        *item /= count;
    }
    for (i, on) in keep.iter().enumerate() {
        if !on {
            continue;
        }
        for ax in 0..3 {
            dr[3 * i + ax] -= com[ax];
        }
    }
}

fn hash_unit(x: f64) -> f64 {
    let bits = x.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (bits >> 11) as f64 / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn covering_points_are_unit() {
        let mut rng = StdRng::seed_from_u64(3);
        let points = cover_points(8, 5, 80, &mut rng);
        assert_eq!(points.len(), 8);
        for p in &points {
            let n = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-9, "norm {n}");
        }
    }

    #[test]
    fn covering_separates_better_than_gaussian() {
        let mut rng = StdRng::seed_from_u64(11);
        let cover = cover_points(12, 3, 200, &mut rng);
        let mut rng_g = StdRng::seed_from_u64(11);
        let gauss = random_sphere(12, 3, &mut rng_g);
        let cover_min = min_geodesic(&cover);
        let gauss_min = min_geodesic(&gauss);
        assert!(
            cover_min > gauss_min,
            "Thomson min geodesic {cover_min} must beat Gaussian {gauss_min}"
        );
    }

    #[test]
    fn two_points_go_to_antipodes() {
        let mut rng = StdRng::seed_from_u64(2);
        let points = cover_points(2, 4, 80, &mut rng);
        let dot: f64 = points[0]
            .iter()
            .zip(points[1].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(dot < -0.99, "antipodes, dot={dot}");
    }

    #[test]
    fn farthest_order_is_a_permutation() {
        let mut rng = StdRng::seed_from_u64(5);
        let points = cover_points(7, 4, 40, &mut rng);
        let order = farthest_order(&points);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn place_matches_rmsd_and_drops_com() {
        let min = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.8, 0.0];
        let mut dir = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        normalize(&mut dir);
        let y = place_around(&min, &dir, 0.35, None);
        let r = all_atom_rmsd(&min, &y);
        assert!((r - 0.35).abs() < 1e-9, "rmsd {r}");
        let mut com = [0.0; 3];
        for i in 0..3 {
            for ax in 0..3 {
                com[ax] += y[3 * i + ax] - min[3 * i + ax];
            }
        }
        for ax in 0..3 {
            assert!(com[ax].abs() < 1e-9, "com[{ax}]={}", com[ax]);
        }
    }

    #[test]
    fn distinct_indices_are_distinct_points() {
        let dim = 9;
        let a = cover_direction(8, dim, 0);
        let b = cover_direction(8, dim, 1);
        let d: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        assert!(d > 0.2, "covering directions collapsed, d={d}");
    }

    #[test]
    fn frozen_atoms_stay_put() {
        let min = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let dir = cover_direction(4, 9, 0);
        let y = place_around(&min, &dir, 0.2, Some(&[0, 1]));
        assert!((y[6] - min[6]).abs() < 1e-15);
        assert!((y[7] - min[7]).abs() < 1e-15);
        assert!((y[8] - min[8]).abs() < 1e-15);
        assert!(all_atom_rmsd(&min, &y) > 0.0);
    }
}
