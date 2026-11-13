//! The bit-row common-neighbour count and the integer distance sort
//! against the direct paths they replace.
//!
//! The direct common-neighbour walk is the bool matrix and the per-pair
//! scan. The direct neighbour spectra are pairwise distances ordered by
//! [`f64::total_cmp`]. Icosahedral, decahedral, face-centred cubic and
//! disordered clusters are each compared at more than one cutoff.

use anneal_core::bias::{Fingerprint, SortedPairs};
use anneal_core::neighbors::NeighborTable;
use anneal_core::rigid_body::RigidBodyGeometry;
use anneal_core::structure::{TRAIL_BOND_CAP, atom_triplet_fracs, cna};
use ndarray::{Array1, ArrayView1};

const TRAIL_CAP: usize = TRAIL_BOND_CAP;

fn direct_adjacency(x: &[f64], n: usize, cutoff: f64) -> Vec<Vec<bool>> {
    let limit = cutoff * cutoff;
    let mut adj = vec![vec![false; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut distance = 0.0;
            for axis in 0..3 {
                let delta = x[3 * i + axis] - x[3 * j + axis];
                distance += delta * delta;
            }
            if distance < limit {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
    }
    adj
}

fn direct_trail(sub: &[Vec<bool>]) -> usize {
    let m = sub.len();
    if m == 0 {
        return 0;
    }
    let mut edges = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            if sub[i][j] {
                edges.push((i, j));
            }
        }
    }
    if edges.is_empty() || edges.len() > TRAIL_CAP {
        return 0;
    }
    let mut incident: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m];
    for (k, &(i, j)) in edges.iter().enumerate() {
        incident[i].push((k, j));
        incident[j].push((k, i));
    }
    let mut best = 0usize;
    for start in 0..m {
        let mut stack = vec![(start, 0u64, 0usize)];
        while let Some((node, used, length)) = stack.pop() {
            if length > best {
                best = length;
            }
            for &(k, next) in &incident[node] {
                if k < 64 && (used >> k) & 1 == 0 {
                    stack.push((next, used | (1 << k), length + 1));
                }
            }
        }
    }
    best
}

fn direct_cna(x: &[f64], n: usize, cutoff: f64) -> (usize, Vec<((usize, usize, usize), usize)>) {
    let adj = direct_adjacency(x, n, cutoff);
    let mut counts = Vec::new();
    let mut bonds = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if !adj[i][j] {
                continue;
            }
            bonds += 1;
            let common: Vec<usize> = (0..n).filter(|&k| adj[i][k] && adj[j][k]).collect();
            let r = common.len();
            let key = if r == 0 {
                (0, 0, 0)
            } else {
                let sub: Vec<Vec<bool>> = common
                    .iter()
                    .map(|&a| common.iter().map(|&b| adj[a][b]).collect())
                    .collect();
                let s: usize = sub
                    .iter()
                    .enumerate()
                    .map(|(a, row)| {
                        row.iter()
                            .enumerate()
                            .filter(|&(b, &bond)| bond && b > a)
                            .count()
                    })
                    .sum();
                (r, s, direct_trail(&sub))
            };
            match counts.iter_mut().find(|(stored, _)| *stored == key) {
                Some((_, count)) => *count += 1,
                None => counts.push((key, 1)),
            }
        }
    }
    (bonds, counts)
}

fn direct_fracs(x: &[f64], n: usize, cutoff: f64) -> Vec<[f64; 3]> {
    let adj = direct_adjacency(x, n, cutoff);
    let mut out = vec![[0.0; 3]; n];
    for i in 0..n {
        let mut n555 = 0usize;
        let mut n421 = 0usize;
        let mut n422 = 0usize;
        let mut nb = 0usize;
        for j in 0..n {
            if j == i || !adj[i][j] {
                continue;
            }
            nb += 1;
            let common: Vec<usize> = (0..n).filter(|&k| adj[i][k] && adj[j][k]).collect();
            let r = common.len();
            if r == 0 {
                continue;
            }
            let sub: Vec<Vec<bool>> = common
                .iter()
                .map(|&a| common.iter().map(|&b| adj[a][b]).collect())
                .collect();
            let s: usize = sub
                .iter()
                .enumerate()
                .map(|(a, row)| {
                    row.iter()
                        .enumerate()
                        .filter(|&(b, &bond)| bond && b > a)
                        .count()
                })
                .sum();
            let t = direct_trail(&sub);
            match (r, s, t) {
                (5, 5, 5) => n555 += 1,
                (4, 2, 1) => n421 += 1,
                (4, 2, 2) => n422 += 1,
                _ => {}
            }
        }
        if nb > 0 {
            let inv = 1.0 / nb as f64;
            out[i] = [n555 as f64 * inv, n421 as f64 * inv, n422 as f64 * inv];
        }
    }
    out
}

fn direct_neighbours(x: &[f64], n: usize, cutoff: f64) -> Vec<Vec<usize>> {
    let limit = cutoff * cutoff;
    let mut lists = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut distance = 0.0;
            for axis in 0..3 {
                let delta = x[3 * i + axis] - x[3 * j + axis];
                distance += delta * delta;
            }
            if distance < limit {
                lists[i].push(j);
                lists[j].push(i);
            }
        }
    }
    lists
}

fn direct_sorted_pairs(x: &[f64], n: usize) -> Vec<f64> {
    let mut distances = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut distance = 0.0;
            for axis in 0..3 {
                let delta = x[3 * i + axis] - x[3 * j + axis];
                distance += delta * delta;
            }
            distances.push(distance.sqrt());
        }
    }
    distances.sort_unstable_by(f64::total_cmp);
    distances
}

fn icosahedron() -> Array1<f64> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let vertices = [
        [0.0, 1.0, phi],
        [0.0, 1.0, -phi],
        [0.0, -1.0, phi],
        [0.0, -1.0, -phi],
        [1.0, phi, 0.0],
        [1.0, -phi, 0.0],
        [-1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, 1.0],
        [phi, 0.0, -1.0],
        [-phi, 0.0, -1.0],
    ];
    let scale = 1.0 / (1.0 + phi * phi).sqrt();
    let mut coordinates = Array1::<f64>::zeros(3 * 13);
    for (index, vertex) in vertices.iter().enumerate() {
        for axis in 0..3 {
            coordinates[3 * (index + 1) + axis] = scale * vertex[axis];
        }
    }
    coordinates
}

/// Pentagonal bipyramid: five ring atoms and two apices, the seven-atom
/// decahedron.
fn decahedron() -> Array1<f64> {
    let radius = 1.0 / (2.0 * (std::f64::consts::PI / 5.0).sin());
    let height = (1.0 - radius * radius).sqrt();
    let mut points = Vec::with_capacity(7);
    for site in 0..5 {
        let angle = site as f64 * std::f64::consts::TAU / 5.0;
        points.push([radius * angle.cos(), radius * angle.sin(), 0.0]);
    }
    points.push([0.0, 0.0, height]);
    points.push([0.0, 0.0, -height]);
    Array1::from_iter(points.into_iter().flatten())
}

fn fcc_ball(spacing: f64, n: usize) -> Array1<f64> {
    let cell = spacing * 2.0_f64.sqrt();
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut sites = Vec::new();
    for i in -4..=4 {
        for j in -4..=4 {
            for k in -4..=4 {
                for offset in &basis {
                    let cell_index = [i as f64, j as f64, k as f64];
                    sites.push([0, 1, 2].map(|axis| cell * (cell_index[axis] + offset[axis])));
                }
            }
        }
    }
    let radius2 =
        |point: &[f64; 3]| point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
    sites.sort_by(|left, right| radius2(left).total_cmp(&radius2(right)));
    Array1::from_iter(sites.iter().take(n).flatten().copied())
}

fn disordered(n: usize, seed: u64) -> Array1<f64> {
    let mut state = seed;
    let mut coordinates = Vec::with_capacity(3 * n);
    for index in 0..n {
        state = state
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(index as u64 + 1);
        let unit = |draw: u64| (draw >> 11) as f64 / (1u64 << 53) as f64;
        let radius = 1.6 * (n as f64).cbrt() * unit(state).cbrt();
        state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3);
        let z = 2.0 * unit(state) - 1.0;
        state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(5);
        let angle = std::f64::consts::TAU * unit(state);
        let ring = (1.0 - z * z).sqrt();
        coordinates.push(radius * ring * angle.cos());
        coordinates.push(radius * ring * angle.sin());
        coordinates.push(radius * z);
    }
    Array1::from(coordinates)
}

fn assert_cna(name: &str, coordinates: &Array1<f64>, cutoff: f64) {
    let n = coordinates.len() / 3;
    let slice = coordinates
        .as_slice()
        .expect("owned coordinates are contiguous");
    let (bonds, counts) = direct_cna(slice, n, cutoff);
    let fast = cna(coordinates.view(), n, cutoff);
    assert_eq!(fast.bonds, bonds, "{name} cutoff {cutoff}: bond count");
    assert_eq!(fast.counts, counts, "{name} cutoff {cutoff}: signatures");
    let again = cna(coordinates.view(), n, cutoff);
    assert_eq!(again, fast, "{name} cutoff {cutoff}: repeated call");
    let fracs = atom_triplet_fracs(coordinates.view(), n, cutoff);
    let direct = direct_fracs(slice, n, cutoff);
    assert_eq!(fracs.len(), direct.len());
    for (index, (left, right)) in fracs.iter().zip(direct.iter()).enumerate() {
        for axis in 0..3 {
            assert_eq!(
                left[axis].to_bits(),
                right[axis].to_bits(),
                "{name} atom {index} fraction {axis}"
            );
        }
    }
}

fn assert_neighbours(name: &str, coordinates: &Array1<f64>, cutoff: f64) {
    let n = coordinates.len() / 3;
    let slice = coordinates
        .as_slice()
        .expect("owned coordinates are contiguous");
    let direct = direct_neighbours(slice, n, cutoff);
    let table = NeighborTable::build(coordinates.view(), n, cutoff);
    for atom in 0..n {
        assert_eq!(
            table.neighbors(atom),
            direct[atom].as_slice(),
            "{name} cutoff {cutoff}: neighbours of {atom}"
        );
    }
    let sorted = SortedPairs { n_points: n }.describe(coordinates.view());
    let reference = direct_sorted_pairs(slice, n);
    assert_eq!(
        sorted.as_slice().unwrap(),
        reference.as_slice(),
        "{name}: sorted pair distances"
    );
}

fn rigid_reference(molecules: usize, state: ArrayView1<f64>) -> Vec<f64> {
    let sites = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]];
    let species = [8u32, 1, 1];
    let mut keys = Vec::new();
    for &left in &species {
        for &right in &species {
            keys.push((left.min(right), left.max(right)));
        }
    }
    keys.sort_unstable();
    keys.dedup();
    let block_of: Vec<usize> = species
        .iter()
        .flat_map(|&left| {
            let keys = &keys;
            species
                .iter()
                .map(move |&right| {
                    keys.binary_search(&(left.min(right), left.max(right)))
                        .expect("pair")
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let geometry = RigidBodyGeometry::new(sites.to_vec(), species.to_vec()).expect("water");
    let points = geometry.expand(molecules, state);
    let per = sites.len();
    let atoms = molecules * per;
    let mut blocks = vec![Vec::new(); keys.len()];
    for i in 0..atoms {
        for j in (i + 1)..atoms {
            let mut distance = 0.0;
            for axis in 0..3 {
                let delta = points[3 * i + axis] - points[3 * j + axis];
                distance += delta * delta;
            }
            let left = i % per;
            let right = j % per;
            blocks[block_of[left * per + right]].push(distance.sqrt());
        }
    }
    for block in &mut blocks {
        block.sort_unstable_by(f64::total_cmp);
    }
    blocks.into_iter().flatten().collect()
}

#[test]
fn cna_and_neighbour_lists_match_the_direct_path() {
    let structures = [
        ("icosahedron", icosahedron()),
        ("decahedron", decahedron()),
        ("fcc", fcc_ball(2.0_f64.powf(1.0 / 6.0), 38)),
        ("compressed-fcc", fcc_ball(0.75, 38)),
        ("disordered", disordered(48, 11)),
    ];
    for (name, coordinates) in &structures {
        for cutoff in [1.05, 1.2, 1.35, 1.8] {
            assert_cna(name, coordinates, cutoff);
            assert_neighbours(name, coordinates, cutoff);
        }
    }

    let geometry = RigidBodyGeometry::new(
        vec![[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        vec![8, 1, 1],
    )
    .expect("water geometry");
    let mut state = vec![0.0; 6 * 4];
    for molecule in 0..4 {
        state[3 * molecule] = molecule as f64 * 3.1;
        state[3 * molecule + 1] = (molecule as f64 - 1.5) * 0.4;
        state[3 * molecule + 2] = 0.2 * molecule as f64;
        state[12 + 3 * molecule] = 0.15 * molecule as f64;
        state[12 + 3 * molecule + 1] = -0.05 * molecule as f64;
    }
    let view = ArrayView1::from(state.as_slice());
    let described = geometry.describe(4, view);
    let reference = rigid_reference(4, view);
    assert_eq!(described.as_slice().unwrap(), reference.as_slice());
}
