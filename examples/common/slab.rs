//! Shared slab geometry: read a .con, place free hydrogen, wrap frozen
//! substrate gradients.

use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::Path;

/// Minimum in-plane separation between independently placed H, in Angstrom.
const H_MIN_SEP: f64 = 1.5;

pub fn read_system(path: &str) -> (Array1<f64>, Vec<u32>, Vec<usize>, [f64; 9]) {
    let frame = readcon_core::iterators::read_first_frame(Path::new(path))
        .expect("failed to read the con file");
    let mut pos = Vec::new();
    let mut species = Vec::new();
    let mut seeds = Vec::new();
    for (i, a) in frame.atom_data.iter().enumerate() {
        pos.extend_from_slice(&[a.x, a.y, a.z]);
        species.push(readcon_core::helpers::symbol_to_atomic_number(&a.symbol) as u32);
        if !a.is_fixed() {
            seeds.push(i);
        }
    }
    let boxl = frame.header.boxl;
    let box_ = [boxl[0], 0.0, 0.0, 0.0, boxl[1], 0.0, 0.0, 0.0, boxl[2]];
    (Array1::from(pos), species, seeds, box_)
}

pub fn z_top_cu(base: &Array1<f64>, species: &[u32]) -> f64 {
    (0..species.len())
        .filter(|&i| species[i] != 1)
        .map(|i| base[3 * i + 2])
        .fold(f64::NEG_INFINITY, f64::max)
}

pub fn free_hydrogen(species: &[u32], free: &[usize]) -> Vec<usize> {
    free.iter().copied().filter(|&i| species[i] == 1).collect()
}

/// Atoms the hop is allowed to propose on: the free hydrogen, or the
/// full free set when the con has no H. The quench may still relax
/// free substrate atoms; the move library must not relocate them.
pub fn hop_adsorbate(species: &[u32], free: &[usize]) -> Vec<usize> {
    let hydrogen = free_hydrogen(species, free);
    if hydrogen.is_empty() {
        free.to_vec()
    } else {
        hydrogen
    }
}

/// One rigid group for an H2 (or single-H) adsorbate; otherwise each free
/// atom is its own group so multi-site hydrogen can hop independently.
pub fn adsorbate_groups(species: &[u32], free: &[usize]) -> Vec<Vec<usize>> {
    let n = species.len();
    if free.is_empty() || free.len() == n {
        return vec![(0..n).collect()];
    }
    let only_h = free.iter().all(|&i| species[i] == 1);
    if only_h && free.len() <= 2 {
        return vec![free.to_vec()];
    }
    free.iter().copied().map(|i| vec![i]).collect()
}

pub fn place_hydrogens(
    base: &Array1<f64>,
    species: &[u32],
    free: &[usize],
    box_: [f64; 9],
    seed: u64,
) -> Array1<f64> {
    let mut x = base.clone();
    let hydrogens = free_hydrogen(species, free);
    if hydrogens.is_empty() {
        return x;
    }
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(3));
    let z_top = z_top_cu(base, species);
    let lx = box_[0].abs().max(1.0);
    let ly = box_[4].abs().max(1.0);
    let mut placed: Vec<[f64; 2]> = Vec::with_capacity(hydrogens.len());
    for &i in &hydrogens {
        let mut xy = [0.0; 2];
        for _ in 0..10_000 {
            xy = [rng.random::<f64>() * lx, rng.random::<f64>() * ly];
            if placed.iter().all(|p| {
                let dx = xy[0] - p[0];
                let dy = xy[1] - p[1];
                dx * dx + dy * dy >= H_MIN_SEP * H_MIN_SEP
            }) {
                break;
            }
        }
        placed.push(xy);
        x[3 * i] = xy[0];
        x[3 * i + 1] = xy[1];
        x[3 * i + 2] = z_top + 2.0 + rng.random::<f64>() * 1.5;
    }
    x
}

/// Place the free adsorbate at a seed-dependent site above the slab.
///
/// A single H2 (or one H) is moved as a rigid body so the existing
/// molecular fixture keeps its start. Several free H, or a partly
/// mobile substrate, are placed independently.
pub fn displace_adsorbate(
    base: &Array1<f64>,
    species: &[u32],
    free: &[usize],
    box_: [f64; 9],
    seed: u64,
) -> Array1<f64> {
    if free.is_empty() {
        return base.clone();
    }
    let only_h = free.iter().all(|&i| species[i] == 1);
    if !(only_h && free.len() <= 2) {
        return place_hydrogens(base, species, free, box_, seed);
    }
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(3));
    let mut x = base.clone();
    let nfree = free.len() as f64;
    let mut c = [0.0; 3];
    for &i in free {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for v in c.iter_mut() {
        *v /= nfree;
    }
    let z_top = z_top_cu(base, species);
    let lx = box_[0].abs().max(1.0);
    let ly = box_[4].abs().max(1.0);
    let target = [
        rng.random::<f64>() * lx,
        rng.random::<f64>() * ly,
        z_top + 2.0 + rng.random::<f64>() * 5.0,
    ];
    let ang = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
    let (sa, ca) = ang.sin_cos();
    for &i in free {
        let rel = [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]];
        x[3 * i] = target[0] + ca * rel[0] - sa * rel[1];
        x[3 * i + 1] = target[1] + sa * rel[0] + ca * rel[1];
        x[3 * i + 2] = target[2] + rel[2];
    }
    x
}

pub fn symbol(z: u32) -> &'static str {
    match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        29 => "Cu",
        79 => "Au",
        _ => "X",
    }
}

pub fn search_arm() -> &'static str {
    let from_env = std::env::var("ANNEAL_SLAB_ARM").ok();
    match from_env.as_deref() {
        Some("plain") => "plain",
        Some("recommended") | None => "recommended",
        Some(value) => panic!("ANNEAL_SLAB_ARM must be plain or recommended; got {value}"),
    }
}

/// Zero the gradient on frozen substrate atoms. The potential is unchanged.
pub struct Mobile<'a, O> {
    pub inner: &'a O,
    pub active: Vec<bool>,
}

impl<O: Objective<f64>> Objective<f64> for Mobile<'_, O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.inner.eval(x)
    }
}

impl<O: Gradient<f64>> Gradient<f64> for Mobile<'_, O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let mut g = self.inner.grad(x);
        for (i, on) in self.active.iter().enumerate() {
            if !on {
                for k in 0..3 {
                    g[3 * i + k] = 0.0;
                }
            }
        }
        g
    }
}

impl<O: Objective<f64> + Gradient<f64>> DifferentiableObjective<f64> for Mobile<'_, O> {}

#[cfg(test)]
mod tests {
    use super::{adsorbate_groups, free_hydrogen};

    #[test]
    fn h2_stays_one_rigid_group() {
        let species = vec![29, 29, 1, 1];
        let free = vec![2, 3];
        assert_eq!(adsorbate_groups(&species, &free), vec![vec![2, 3]]);
        assert_eq!(free_hydrogen(&species, &free), vec![2, 3]);
    }

    #[test]
    fn six_h_hop_independently() {
        let species = vec![29, 29, 1, 1, 1, 1, 1, 1];
        let free = vec![2, 3, 4, 5, 6, 7];
        assert_eq!(
            adsorbate_groups(&species, &free),
            vec![vec![2], vec![3], vec![4], vec![5], vec![6], vec![7],]
        );
    }

    #[test]
    fn free_cu_and_h_are_each_a_group() {
        let species = vec![29, 29, 29, 1, 1];
        let free = vec![1, 2, 3, 4];
        assert_eq!(
            adsorbate_groups(&species, &free),
            vec![vec![1], vec![2], vec![3], vec![4]]
        );
    }

    #[test]
    fn hops_only_the_hydrogen() {
        let species = vec![29, 29, 29, 1, 1, 1];
        let free = vec![1, 2, 3, 4, 5];
        assert_eq!(super::hop_adsorbate(&species, &free), vec![3, 4, 5]);
    }
}
