//! Shared slab geometry: read a .con, place the free adsorbates above the
//! fixed substrate, wrap frozen substrate gradients. What moves is what the
//! fixture leaves free, and what is substrate is what it fixes. Rigid
//! adsorbate groups use the caller-provided `ANNEAL_SLAB_BOND_CUTOFF`
//! distance. With no cutoff, free atoms move independently.

#![allow(dead_code)]

use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::Path;

/// Minimum in-plane separation between independently placed groups, in
/// Angstrom.
const GROUP_MIN_SEP: f64 = 2.5;
fn configured_bond_cutoff() -> f64 {
    std::env::var("ANNEAL_SLAB_BOND_CUTOFF")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.0)
}

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

/// Highest z of the fixed atoms, the top of the substrate. A fixture with
/// nothing fixed uses every atom.
pub fn substrate_top(base: &Array1<f64>, free: &[usize]) -> f64 {
    let n = base.len() / 3;
    let fixed: Vec<usize> = (0..n).filter(|i| !free.contains(i)).collect();
    let over = if fixed.is_empty() {
        (0..n).collect()
    } else {
        fixed
    };
    over.iter()
        .map(|&i| base[3 * i + 2])
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Rigid groups of the free atoms: connected components under
/// `ANNEAL_SLAB_BOND_CUTOFF`, so grouping is an explicit input rather
/// than an element chart.
pub fn adsorbate_groups(base: &Array1<f64>, species: &[u32], free: &[usize]) -> Vec<Vec<usize>> {
    let _ = species;
    adsorbate_groups_with_cutoff(base, free, configured_bond_cutoff())
}

fn adsorbate_groups_with_cutoff(
    base: &Array1<f64>,
    free: &[usize],
    bond_cutoff: f64,
) -> Vec<Vec<usize>> {
    if free.is_empty() {
        return Vec::new();
    }
    let cutoff2 = bond_cutoff * bond_cutoff;
    let bonded = |a: usize, b: usize| {
        let mut d2 = 0.0;
        for k in 0..3 {
            let d = base[3 * a + k] - base[3 * b + k];
            d2 += d * d;
        }
        cutoff2 > 0.0 && d2 < cutoff2
    };
    let mut assigned = vec![false; free.len()];
    let mut groups = Vec::new();
    for s in 0..free.len() {
        if assigned[s] {
            continue;
        }
        let mut stack = vec![s];
        let mut group = Vec::new();
        assigned[s] = true;
        while let Some(i) = stack.pop() {
            group.push(free[i]);
            for j in 0..free.len() {
                if !assigned[j] && bonded(free[i], free[j]) {
                    assigned[j] = true;
                    stack.push(j);
                }
            }
        }
        group.sort_unstable();
        groups.push(group);
    }
    groups.sort_by_key(|g| g[0]);
    groups
}

/// Every free group placed rigidly at a seed-dependent site above the
/// substrate: random in-plane position with [`GROUP_MIN_SEP`] between
/// groups, a random in-plane rotation, and a height of two to three and a
/// half Angstrom over the top fixed atom.
pub fn place_adsorbates(
    base: &Array1<f64>,
    species: &[u32],
    free: &[usize],
    box_: [f64; 9],
    seed: u64,
) -> Array1<f64> {
    place_adsorbates_with_cutoff(base, species, free, box_, seed, configured_bond_cutoff())
}

fn place_adsorbates_with_cutoff(
    base: &Array1<f64>,
    species: &[u32],
    free: &[usize],
    box_: [f64; 9],
    seed: u64,
    bond_cutoff: f64,
) -> Array1<f64> {
    let mut x = base.clone();
    if free.is_empty() {
        return x;
    }
    let _ = species;
    let groups = adsorbate_groups_with_cutoff(base, free, bond_cutoff);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(3));
    let z_top = substrate_top(base, free);
    let lx = box_[0].abs().max(1.0);
    let ly = box_[4].abs().max(1.0);
    let mut placed: Vec<[f64; 2]> = Vec::with_capacity(groups.len());
    for group in &groups {
        let mut c = [0.0; 3];
        for &i in group {
            for k in 0..3 {
                c[k] += base[3 * i + k] / group.len() as f64;
            }
        }
        let mut xy = [0.0; 2];
        for _ in 0..10_000 {
            xy = [rng.random::<f64>() * lx, rng.random::<f64>() * ly];
            if placed.iter().all(|p| {
                let dx = xy[0] - p[0];
                let dy = xy[1] - p[1];
                dx * dx + dy * dy >= GROUP_MIN_SEP * GROUP_MIN_SEP
            }) {
                break;
            }
        }
        placed.push(xy);
        let z = z_top + 2.0 + rng.random::<f64>() * 1.5;
        let ang = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
        let (sa, ca) = ang.sin_cos();
        for &i in group {
            let rel = [
                base[3 * i] - c[0],
                base[3 * i + 1] - c[1],
                base[3 * i + 2] - c[2],
            ];
            x[3 * i] = xy[0] + ca * rel[0] - sa * rel[1];
            x[3 * i + 1] = xy[1] + sa * rel[0] + ca * rel[1];
            x[3 * i + 2] = z + rel[2];
        }
    }
    x
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
    use super::{adsorbate_groups_with_cutoff, place_adsorbates_with_cutoff, substrate_top};
    use ndarray::Array1;

    fn pair_on_two_fixed() -> (Array1<f64>, Vec<u32>, Vec<usize>) {
        let base = Array1::from(vec![
            0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 1.0, 1.0, 2.0, 1.74, 1.0, 2.0,
        ]);
        (base, vec![29, 29, 1, 1], vec![2, 3])
    }

    #[test]
    fn caller_cutoff_controls_rigid_groups() {
        let (base, _, free) = pair_on_two_fixed();
        assert_eq!(
            adsorbate_groups_with_cutoff(&base, &free, 0.8),
            vec![vec![2, 3]]
        );
        assert_eq!(
            adsorbate_groups_with_cutoff(&base, &free, 0.7),
            vec![vec![2], vec![3]]
        );
    }

    #[test]
    fn zero_cutoff_makes_free_atoms_independent() {
        let mut coords = vec![0.0, 0.0, 0.0, 2.5, 0.0, 0.0];
        for k in 0..6 {
            coords.extend_from_slice(&[3.0 * k as f64, 5.0, 2.0]);
        }
        let base = Array1::from(coords);
        let free = vec![2, 3, 4, 5, 6, 7];
        assert_eq!(
            adsorbate_groups_with_cutoff(&base, &free, 0.0),
            vec![vec![2], vec![3], vec![4], vec![5], vec![6], vec![7]]
        );
    }

    #[test]
    fn placement_keeps_a_group_rigid_and_above_the_substrate() {
        let (base, species, free) = pair_on_two_fixed();
        let box_ = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 20.0];
        let x = place_adsorbates_with_cutoff(&base, &species, &free, box_, 7, 0.8);
        let top = substrate_top(&base, &free);
        assert!((top - 0.0).abs() < 1e-12);
        let d = ((x[6] - x[9]).powi(2) + (x[7] - x[10]).powi(2) + (x[8] - x[11]).powi(2)).sqrt();
        assert!((d - 0.74).abs() < 1e-9, "bond length {d}");
        assert!(x[8] > top + 2.0 - 1e-9 && x[8] < top + 3.5 + 1e-9);
        assert_eq!(&x.to_vec()[..6], &base.to_vec()[..6]);
    }
}
