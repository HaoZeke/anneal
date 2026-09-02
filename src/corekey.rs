//! Prospective superbasin identity from the topology of a structure's core.
//!
//! The isomers that pile up on an icosahedral shelf differ by where a few
//! surface atoms sit; their interiors are one Mackay core deformed by
//! symmetry operations. A key that reads only the interior contact network
//! therefore names the superbasin a structure belongs to on first sight,
//! before any revisit, and does so from topology alone: no reference
//! structure, no order parameter, no size rule.
//!
//! The key is the d-SEAMS construction carried over to clusters and
//! coloured: primitive rings of the core contact graph in the shortest-path
//! sense of Franzblau (*Phys. Rev. B* **1991**, *44*, 4925), each coloured
//! by its size and by the species and coordination of its atoms, joined
//! wherever two rings share a bond, and the coloured ring graph hashed by
//! Weisfeiler--Lehman refinement. Rings and the cages they close are what
//! separate packing families in a network (Goswami, Goswami and Singh,
//! *J. Chem. Inf. Model.* **2020**, *60*, 2169), and a network built from
//! rings is what a water cage census and a metal core census have in common.
//!
//! A key is not a proof of identity: two cores with the same refined
//! colouring hash together. It is a prospective label for a superbasin, and
//! the exact witnesses elsewhere in the crate decide identity where it
//! matters.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use ndarray::ArrayView1;

use crate::structure::{RING_CUTOFF_SCALE, median_nearest_neighbour};

/// Rings longer than this are not part of the key.
pub const MAX_RING: usize = 6;
/// Weisfeiler--Lehman refinement rounds.
pub const WL_ROUNDS: usize = 3;

/// Which atoms count as the core.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoreRule {
    /// Atoms whose coordination is within `slack` of the largest observed.
    NearMaximum {
        /// Allowed shortfall from the largest coordination.
        slack: usize,
    },
    /// Atoms with at least this coordination.
    AtLeast(usize),
}

impl Default for CoreRule {
    fn default() -> Self {
        Self::NearMaximum { slack: 1 }
    }
}

/// A superbasin key with the census behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoreKey {
    /// Weisfeiler--Lehman hash of the coloured core ring graph.
    pub key: u64,
    /// Atoms the core rule admitted.
    pub core_atoms: usize,
    /// Primitive rings of the core, up to [`MAX_RING`].
    pub rings: usize,
}

impl CoreKey {
    /// The key spread over four coordinates, so a Euclidean lookup with a
    /// radius below one treats equal keys as one basin and different keys
    /// as far apart.
    pub fn coordinates(&self) -> [f64; 4] {
        let k = self.key;
        [
            (k & 0xffff) as f64,
            ((k >> 16) & 0xffff) as f64,
            ((k >> 32) & 0xffff) as f64,
            ((k >> 48) & 0xffff) as f64,
        ]
    }
}

fn hash_one<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Contact adjacency lists at `cutoff`.
pub fn contact_neighbours(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Vec<Vec<usize>> {
    let c2 = cutoff * cutoff;
    let mut nb = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                r2 += d * d;
            }
            if r2 < c2 {
                nb[i].push(j);
                nb[j].push(i);
            }
        }
    }
    nb
}

/// Atoms the rule admits, by index.
pub fn core_atoms(neighbours: &[Vec<usize>], rule: CoreRule) -> Vec<usize> {
    let coordination: Vec<usize> = neighbours.iter().map(Vec::len).collect();
    let threshold = match rule {
        CoreRule::NearMaximum { slack } => coordination
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_sub(slack),
        CoreRule::AtLeast(k) => k,
    };
    (0..neighbours.len())
        .filter(|&i| coordination[i] >= threshold && coordination[i] > 0)
        .collect()
}

/// Breadth-first distances inside a graph given as adjacency lists.
fn distances_from(nb: &[Vec<usize>], source: usize) -> Vec<usize> {
    let mut dist = vec![usize::MAX; nb.len()];
    let mut queue = std::collections::VecDeque::new();
    dist[source] = 0;
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        for &v in &nb[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
    dist
}

/// Primitive rings of a graph up to `max_ring`, as sorted-by-walk member
/// lists starting from their smallest vertex.
///
/// A cycle is primitive when every pair of its vertices is joined along the
/// cycle by a shortest path of the whole graph, which is Franzblau's
/// criterion: no chord, and no shortcut through vertices off the ring.
pub fn primitive_rings(nb: &[Vec<usize>], max_ring: usize) -> Vec<Vec<usize>> {
    let n = nb.len();
    let dist: Vec<Vec<usize>> = (0..n).map(|s| distances_from(nb, s)).collect();
    let mut rings = Vec::new();
    let mut path = Vec::with_capacity(max_ring);
    for start in 0..n {
        path.clear();
        path.push(start);
        walk(nb, &dist, start, max_ring, &mut path, &mut rings);
    }
    rings
}

fn walk(
    nb: &[Vec<usize>],
    dist: &[Vec<usize>],
    start: usize,
    max_ring: usize,
    path: &mut Vec<usize>,
    rings: &mut Vec<Vec<usize>>,
) {
    let last = *path.last().expect("path holds its start");
    for &next in &nb[last] {
        if next == start && path.len() >= 3 {
            // Closed. Canonical direction: the second vertex is smaller than
            // the last, so each ring is recorded once.
            if path[1] < last && is_primitive(dist, path) {
                rings.push(path.clone());
            }
            continue;
        }
        if next <= start || path.contains(&next) || path.len() >= max_ring {
            continue;
        }
        // Along a shortest-path ring every vertex stays within half the
        // ring length of the start; prune deeper walks.
        if dist[start][next] > max_ring / 2 {
            continue;
        }
        path.push(next);
        walk(nb, dist, start, max_ring, path, rings);
        path.pop();
    }
}

fn is_primitive(dist: &[Vec<usize>], ring: &[usize]) -> bool {
    let len = ring.len();
    for i in 0..len {
        for j in (i + 1)..len {
            let along = (j - i).min(len - (j - i));
            if dist[ring[i]][ring[j]] < along {
                return false;
            }
        }
    }
    true
}

/// Superbasin key of a structure.
///
/// `species` is one atomic number per atom or empty for a single species;
/// `cutoff` is the contact distance. The core is taken by `rule` on the
/// full contact graph, rings are found on the core-induced subgraph, and
/// the coloured ring graph is refined and hashed.
pub fn core_key(x: ArrayView1<f64>, species: &[u32], cutoff: f64, rule: CoreRule) -> CoreKey {
    let n = x.len() / 3;
    let neighbours = contact_neighbours(x, n, cutoff);
    let core = core_atoms(&neighbours, rule);
    let mut index = vec![usize::MAX; n];
    for (local, &atom) in core.iter().enumerate() {
        index[atom] = local;
    }
    let core_nb: Vec<Vec<usize>> = core
        .iter()
        .map(|&atom| {
            let mut local: Vec<usize> = neighbours[atom]
                .iter()
                .filter_map(|&j| (index[j] != usize::MAX).then_some(index[j]))
                .collect();
            local.sort_unstable();
            local
        })
        .collect();
    let rings = primitive_rings(&core_nb, MAX_RING);
    // Atom colour: species and full-graph coordination.
    let atom_colour: Vec<u64> = core
        .iter()
        .map(|&atom| {
            let z = species.get(atom).copied().unwrap_or(0);
            hash_one(&(z, neighbours[atom].len()))
        })
        .collect();
    // Ring nodes coloured by size and the sorted colours of their atoms.
    let mut labels: Vec<u64> = rings
        .iter()
        .map(|ring| {
            let mut colours: Vec<u64> = ring.iter().map(|&v| atom_colour[v]).collect();
            colours.sort_unstable();
            hash_one(&(ring.len(), colours))
        })
        .collect();
    // Rings are adjacent when they share a bond.
    let mut ring_nb: Vec<Vec<usize>> = vec![Vec::new(); rings.len()];
    let bonds: Vec<std::collections::BTreeSet<(usize, usize)>> = rings
        .iter()
        .map(|ring| {
            ring.iter()
                .enumerate()
                .map(|(i, &a)| {
                    let b = ring[(i + 1) % ring.len()];
                    (a.min(b), a.max(b))
                })
                .collect()
        })
        .collect();
    for i in 0..rings.len() {
        for j in (i + 1)..rings.len() {
            if bonds[i].intersection(&bonds[j]).next().is_some() {
                ring_nb[i].push(j);
                ring_nb[j].push(i);
            }
        }
    }
    for _ in 0..WL_ROUNDS {
        labels = (0..rings.len())
            .map(|i| {
                let mut around: Vec<u64> = ring_nb[i].iter().map(|&j| labels[j]).collect();
                around.sort_unstable();
                hash_one(&(labels[i], around))
            })
            .collect();
    }
    labels.sort_unstable();
    let key = hash_one(&(core.len(), rings.len(), labels));
    CoreKey {
        key,
        core_atoms: core.len(),
        rings: rings.len(),
    }
}

/// The contact cutoff the key uses: [`RING_CUTOFF_SCALE`] times the median
/// nearest-neighbour distance, which sits between the first and second
/// shells of every packing measured here and, unlike a histogram minimum,
/// cannot slip a shell on a strained decahedron.
pub fn contact_cutoff(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    RING_CUTOFF_SCALE * median_nearest_neighbour(x, n)
}

/// [`core_key`] at the structure's own contact cutoff.
pub fn core_key_nn(x: ArrayView1<f64>, species: &[u32], rule: CoreRule) -> CoreKey {
    core_key(x, species, contact_cutoff(x), rule)
}

/// Common-neighbour signature counts of a structure: pairs of contacts whose
/// common neighbours form a five-ring (555, a local five-fold axis), a
/// four-chain with two bonds (421, fcc) and with two bonds in a different
/// arrangement (422, hcp), at the structure's own contact cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotifCounts {
    /// Contact pairs with five common neighbours bonded in a ring.
    pub five_fold: usize,
    /// Contact pairs with four common neighbours and two bonds among them
    /// that do not share an atom.
    pub fcc: usize,
    /// Contact pairs with four common neighbours and two bonds among them
    /// that share an atom.
    pub hcp: usize,
    /// Contact pairs.
    pub contacts: usize,
}

/// Coarse class of a structure from its five-fold pair density: none (fcc
/// and hcp packings), sparse (a single five-fold axis, as in a decahedron),
/// moderate (several local axes, as in the LJ98 tetrahedron) or dense
/// (icosahedral packing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotifClass {
    /// No five-fold pairs.
    None,
    /// Fewer than [`SPARSE_FIVE_FOLD_PER_ATOM`] five-fold pairs per atom.
    Sparse,
    /// Fewer than [`DENSE_FIVE_FOLD_PER_ATOM`] five-fold pairs per atom.
    Moderate,
    /// Five-fold pairs throughout.
    Dense,
}

impl MotifClass {
    /// The class as a small integer, for keys.
    pub fn index(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Sparse => 1,
            Self::Moderate => 2,
            Self::Dense => 3,
        }
    }
}

/// Common-neighbour counts of `x` at its contact cutoff.
pub fn motif_counts(x: ArrayView1<f64>) -> MotifCounts {
    let n = x.len() / 3;
    let nb = contact_neighbours(x, n, contact_cutoff(x));
    let mut counts = MotifCounts::default();
    for i in 0..n {
        for &j in &nb[i] {
            if j <= i {
                continue;
            }
            counts.contacts += 1;
            let common: Vec<usize> = nb[i]
                .iter()
                .copied()
                .filter(|k| nb[j].contains(k))
                .collect();
            let mut bonds: Vec<(usize, usize)> = Vec::new();
            for (a, &p) in common.iter().enumerate() {
                for &q in &common[a + 1..] {
                    if nb[p].contains(&q) {
                        bonds.push((p, q));
                    }
                }
            }
            match (common.len(), bonds.len()) {
                (5, 5) => {
                    let mut degree = vec![0usize; common.len()];
                    for &(p, q) in &bonds {
                        degree[common.iter().position(|&c| c == p).expect("member")] += 1;
                        degree[common.iter().position(|&c| c == q).expect("member")] += 1;
                    }
                    if degree.iter().all(|&d| d == 2) {
                        counts.five_fold += 1;
                    }
                }
                (4, 2) => {
                    let (a, b) = (bonds[0], bonds[1]);
                    let shared = a.0 == b.0 || a.0 == b.1 || a.1 == b.0 || a.1 == b.1;
                    if shared {
                        counts.hcp += 1;
                    } else {
                        counts.fcc += 1;
                    }
                }
                _ => {}
            }
        }
    }
    counts
}

/// Five-fold pairs per atom below which the class is sparse.
pub const SPARSE_FIVE_FOLD_PER_ATOM: f64 = 0.1;
/// Five-fold pairs per atom below which the class is moderate.
pub const DENSE_FIVE_FOLD_PER_ATOM: f64 = 0.25;

/// The coarse five-fold class of `x`.
pub fn motif_class(x: ArrayView1<f64>) -> MotifClass {
    let n = (x.len() / 3).max(1);
    let density = motif_counts(x).five_fold as f64 / n as f64;
    if density == 0.0 {
        MotifClass::None
    } else if density < SPARSE_FIVE_FOLD_PER_ATOM {
        MotifClass::Sparse
    } else if density < DENSE_FIVE_FOLD_PER_ATOM {
        MotifClass::Moderate
    } else {
        MotifClass::Dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn fixture(name: &str) -> Array1<f64> {
        let path = format!("{}/tests/fixtures/{name}.xyz", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut lines = text.lines();
        let n: usize = lines.next().unwrap().trim().parse().unwrap();
        lines.next();
        let mut x = Vec::with_capacity(3 * n);
        for line in lines.take(n) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            for f in &fields[1..4] {
                x.push(f.parse::<f64>().unwrap());
            }
        }
        Array1::from(x)
    }

    fn rotate_permute(x: &Array1<f64>) -> Array1<f64> {
        let n = x.len() / 3;
        let (s, c) = (0.7_f64.sin(), 0.7_f64.cos());
        let mut y = Array1::zeros(x.len());
        for i in 0..n {
            let j = (i * 7 + 3) % n;
            let (px, py, pz) = (x[3 * i], x[3 * i + 1], x[3 * i + 2]);
            y[3 * j] = c * px - s * py + 1.5;
            y[3 * j + 1] = s * px + c * py - 0.3;
            y[3 * j + 2] = pz + 0.8;
        }
        y
    }

    #[test]
    fn a_square_and_a_fused_triangle_pair_are_told_apart() {
        // Square 0-1-2-3 with no chord: one primitive 4-ring.
        let square = vec![vec![1, 3], vec![0, 2], vec![1, 3], vec![0, 2]];
        let rings = primitive_rings(&square, 6);
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 4);
        // Add the chord 0-2: two triangles, and the square is no longer primitive.
        let fused = vec![vec![1, 2, 3], vec![0, 2], vec![0, 1, 3], vec![0, 2]];
        let rings = primitive_rings(&fused, 6);
        assert_eq!(rings.len(), 2);
        assert!(rings.iter().all(|r| r.len() == 3));
    }

    #[test]
    fn the_key_is_invariant_to_rotation_translation_and_relabelling() {
        for name in ["lj38_fcc", "lj38_ico", "lj75_ico", "lj75_marks"] {
            let x = fixture(name);
            let y = rotate_permute(&x);
            let a = core_key_nn(x.view(), &[], CoreRule::default());
            let b = core_key_nn(y.view(), &[], CoreRule::default());
            assert_eq!(a, b, "{name}");
            assert!(a.core_atoms >= 6 && a.rings > 0, "{name}: {a:?}");
        }
    }

    #[test]
    fn the_core_separates_packing_families() {
        let ico38 = core_key_nn(fixture("lj38_ico").view(), &[], CoreRule::default());
        let fcc38 = core_key_nn(fixture("lj38_fcc").view(), &[], CoreRule::default());
        assert_ne!(ico38.key, fcc38.key);
        let ico75 = core_key_nn(fixture("lj75_ico").view(), &[], CoreRule::default());
        let marks75 = core_key_nn(fixture("lj75_marks").view(), &[], CoreRule::default());
        assert_ne!(ico75.key, marks75.key);
    }

    #[test]
    fn a_surface_relocation_keeps_the_core_key() {
        // Remove the least coordinated atom: the shelf isomer that lost a
        // surface atom keeps the Mackay core and therefore the key.
        let x = fixture("lj75_ico");
        let n = x.len() / 3;
        let cutoff = contact_cutoff(x.view());
        let nb = contact_neighbours(x.view(), n, cutoff);
        let loose = (0..n).min_by_key(|&i| nb[i].len()).unwrap();
        let mut y = Vec::with_capacity(3 * (n - 1));
        for i in 0..n {
            if i != loose {
                y.extend_from_slice(&[x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
            }
        }
        let y = Array1::from(y);
        let whole = core_key(x.view(), &[], cutoff, CoreRule::default());
        let less = core_key(y.view(), &[], cutoff, CoreRule::default());
        assert_eq!(whole.key, less.key);
        assert_eq!(whole.core_atoms, less.core_atoms);
    }

    #[test]
    fn coordinates_keep_equal_keys_together_and_different_keys_apart() {
        let a = CoreKey {
            key: 0x1234_5678_9abc_def0,
            core_atoms: 1,
            rings: 1,
        };
        let b = CoreKey {
            key: 0x1234_5678_9abc_def1,
            core_atoms: 1,
            rings: 1,
        };
        assert_eq!(a.coordinates(), a.coordinates());
        let gap: f64 = a
            .coordinates()
            .iter()
            .zip(b.coordinates())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        assert!(gap >= 1.0);
    }

    #[test]
    fn motif_counts_separate_the_fixture_packings() {
        for name in [
            "lj38_fcc",
            "lj38_ico",
            "lj75_ico",
            "lj75_marks",
            "lj98_gm",
            "lj104_gm",
        ] {
            let x = fixture(name);
            let counts = motif_counts(x.view());
            eprintln!("MOTIF {name}: {counts:?} class {:?}", motif_class(x.view()));
        }
        assert_eq!(motif_class(fixture("lj38_fcc").view()), MotifClass::None);
        assert_eq!(motif_class(fixture("lj38_ico").view()), MotifClass::Dense);
        assert_eq!(motif_class(fixture("lj75_ico").view()), MotifClass::Dense);
        assert_eq!(
            motif_class(fixture("lj75_marks").view()),
            MotifClass::Sparse
        );
        assert_eq!(motif_class(fixture("lj104_gm").view()), MotifClass::Sparse);
        assert_eq!(motif_class(fixture("lj98_gm").view()), MotifClass::Moderate);
        assert_eq!(
            motif_class(fixture("lj104_gm").view()),
            MotifClass::Decahedral
        );
    }
}
