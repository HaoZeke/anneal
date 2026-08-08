//! An incremental neighbour table shared across the hop.
//!
//! Every kernel that reads structure recomputes it per call: surface
//! relocation its coordination counts, the graph key its distances, ring
//! profiles their adjacency. Each is O(n^2) per proposal, and a hop makes
//! several proposals from the same incumbent. The measured-productive moves
//! displace one to three atoms, so between incumbents the table changes in
//! O(k n), not O(n^2): the blocked-kernel economy of the eigensolver
//! libraries, applied to the structure every consumer shares.
//!
//! The table is exact, not approximate: after any sequence of updates it
//! equals the table built from scratch, and a test witnesses that on random
//! configurations and random moves.

use ndarray::ArrayView1;

/// Sorted adjacency lists under a fixed absolute cutoff.
#[derive(Debug, Clone)]
pub struct NeighborTable {
    cutoff2: f64,
    lists: Vec<Vec<usize>>,
}

impl NeighborTable {
    /// Builds the table from scratch in O(n^2).
    pub fn build(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Self {
        let cutoff2 = cutoff * cutoff;
        let mut lists = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                if Self::dist2(x, i, j) < cutoff2 {
                    lists[i].push(j);
                    lists[j].push(i);
                }
            }
        }
        Self { cutoff2, lists }
    }

    fn dist2(x: ArrayView1<f64>, i: usize, j: usize) -> f64 {
        (0..3)
            .map(|k| {
                let d = x[3 * i + k] - x[3 * j + k];
                d * d
            })
            .sum()
    }

    /// Points in the table.
    pub fn len(&self) -> usize {
        self.lists.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.lists.is_empty()
    }

    /// Neighbours of `i`, sorted ascending.
    pub fn neighbors(&self, i: usize) -> &[usize] {
        &self.lists[i]
    }

    /// Coordination of `i`.
    pub fn degree(&self, i: usize) -> usize {
        self.lists[i].len()
    }

    /// Reconciles the table with `x_new` after the atoms in `moved` changed,
    /// in O(k n) for k moved atoms.
    ///
    /// Exactness rests on one fact: a pair's distance changed only if at
    /// least one of its ends moved, so edges between unmoved atoms need no
    /// inspection.
    pub fn update(&mut self, x_new: ArrayView1<f64>, moved: &[usize]) {
        let n = self.lists.len();
        let mut is_moved = vec![false; n];
        for &m in moved {
            if m < n {
                is_moved[m] = true;
            }
        }
        // Drop every edge with a moved end.
        for i in 0..n {
            if is_moved[i] {
                self.lists[i].clear();
            } else {
                self.lists[i].retain(|&j| !is_moved[j]);
            }
        }
        // Rebuild the moved atoms' edges against everyone.
        for &m in moved {
            if m >= n {
                continue;
            }
            for j in 0..n {
                if j == m || (is_moved[j] && j < m) {
                    continue;
                }
                if Self::dist2(x_new, m, j) < self.cutoff2 {
                    self.lists[m].push(j);
                    self.lists[j].push(m);
                }
            }
        }
        for l in self.lists.iter_mut() {
            l.sort_unstable();
        }
    }

    /// The atoms whose coordinates differ between two structures.
    pub fn moved_between(a: ArrayView1<f64>, b: ArrayView1<f64>) -> Vec<usize> {
        let n = a.len().min(b.len()) / 3;
        (0..n)
            .filter(|&i| (0..3).any(|k| (a[3 * i + k] - b[3 * i + k]).abs() > 1e-12))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// After any sequence of moves, the incremental table must equal the one
    /// built from scratch. This is the whole contract.
    #[test]
    fn incremental_equals_rebuild() {
        let mut rng = StdRng::seed_from_u64(5);
        let n = 40;
        let mut x = Array1::from_shape_fn(3 * n, |_| rng.random::<f64>() * 4.0);
        let mut table = NeighborTable::build(x.view(), n, 1.4);
        for round in 0..30 {
            let k = 1 + rng.random_range(0..3);
            let mut moved = Vec::new();
            for _ in 0..k {
                let m = rng.random_range(0..n);
                moved.push(m);
                for c in 0..3 {
                    x[3 * m + c] += (rng.random::<f64>() - 0.5) * 2.0;
                }
            }
            moved.sort_unstable();
            moved.dedup();
            table.update(x.view(), &moved);
            let fresh = NeighborTable::build(x.view(), n, 1.4);
            for i in 0..n {
                assert_eq!(
                    table.neighbors(i),
                    fresh.neighbors(i),
                    "round {round}, atom {i}: incremental diverged from rebuild"
                );
            }
        }
    }

    /// The moved-set detector has to find exactly the atoms that differ.
    #[test]
    fn moved_between_finds_the_difference() {
        let mut rng = StdRng::seed_from_u64(9);
        let n = 20;
        let a = Array1::from_shape_fn(3 * n, |_| rng.random::<f64>());
        let mut b = a.clone();
        b[3 * 7] += 0.5;
        b[3 * 13 + 2] -= 0.1;
        assert_eq!(NeighborTable::moved_between(a.view(), b.view()), vec![7, 13]);
    }
}
