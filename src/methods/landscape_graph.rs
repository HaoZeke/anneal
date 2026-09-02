//! Spectral referee over the explored landscape.
//!
//! The catalog and census name basins; observed transitions between them
//! form a weighted graph. The second Laplacian eigenvalue measures how
//! metastable the explored landscape is, the sign structure of its
//! eigenvector splits the basins into the two most weakly coupled
//! communities, and an absorbing-chain solve prices the expected number
//! of transitions to reach a chosen set of basins. Together these answer
//! the questions a coordinator faces at an epoch boundary: is the
//! ensemble confined, where is the seam, which pair of basins deserves a
//! bridge, and is local work still worth its price.
//!
//! Everything is dense arithmetic on a graph no larger than the catalog
//! capacity, so no external eigensolver or sparse machinery appears.

use std::collections::HashMap;

/// Invalid graph query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GraphError {
    /// The queried basin has never been observed.
    #[error("basin not present in the landscape graph")]
    UnknownBasin,
    /// The query needs at least two basins.
    #[error("landscape graph too small for a spectral question")]
    TooSmall,
    /// The absorbing set must be nonempty and proper.
    #[error("absorbing set empty or covering the whole graph")]
    BadAbsorbingSet,
}

/// The two most weakly coupled communities of the explored landscape.
#[derive(Debug, Clone)]
pub struct SpectralSplit {
    /// Second-smallest Laplacian eigenvalue: the algebraic connectivity.
    pub algebraic_connectivity: f64,
    /// Basins on the negative side of the eigenvector.
    pub left: Vec<u64>,
    /// Basins on the non-negative side.
    pub right: Vec<u64>,
    /// Cut weight over the smaller side's volume: the conductance of the
    /// split, small when the two communities rarely exchange.
    pub conductance: f64,
    /// The best-anchored representative on each side: the basin with the
    /// largest internal weight in its community. Bridges commission
    /// between representatives, not between arbitrary members.
    pub representatives: (u64, u64),
}

/// Weighted transition graph over census basins.
#[derive(Debug, Clone, Default)]
pub struct LandscapeGraph {
    basins: Vec<u64>,
    index: HashMap<u64, usize>,
    weights: Vec<Vec<f64>>,
}

impl LandscapeGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of basins observed.
    pub fn len(&self) -> usize {
        self.basins.len()
    }

    /// Whether no basin has been observed.
    pub fn is_empty(&self) -> bool {
        self.basins.is_empty()
    }

    /// Ensure a basin is present, without any edge.
    pub fn observe_basin(&mut self, basin: u64) -> usize {
        if let Some(&i) = self.index.get(&basin) {
            return i;
        }
        let i = self.basins.len();
        self.basins.push(basin);
        self.index.insert(basin, i);
        for row in &mut self.weights {
            row.push(0.0);
        }
        self.weights.push(vec![0.0; i + 1]);
        i
    }

    /// Record an observed transition between two basins with a weight.
    /// The graph is undirected: confinement is a property of the seam,
    /// not of the direction it was crossed in.
    pub fn observe_crossing(&mut self, a: u64, b: u64, weight: f64) {
        let i = self.observe_basin(a);
        let j = self.observe_basin(b);
        if i != j && weight > 0.0 {
            self.weights[i][j] += weight;
            self.weights[j][i] += weight;
        }
    }

    fn degree(&self, i: usize) -> f64 {
        self.weights[i].iter().sum()
    }

    /// The spectral split of the landscape, or an error below two basins.
    ///
    /// The eigenpair comes from power iteration on the reflected
    /// Laplacian `B = c I - L` restricted to the complement of the
    /// constant vector, with `c` a Gershgorin bound on the spectrum, so
    /// the dominant surviving direction is the Fiedler vector and the
    /// eigenvalue recovers as `c` minus the Rayleigh quotient. On a
    /// disconnected graph the connectivity comes out at zero and the
    /// split follows components, which is the right referee answer: the
    /// seam has never been crossed at all.
    pub fn spectral_split(&self) -> Result<SpectralSplit, GraphError> {
        let n = self.len();
        if n < 2 {
            return Err(GraphError::TooSmall);
        }
        let ceiling = 2.0 * (0..n).map(|i| self.degree(i)).fold(0.0_f64, f64::max) + 1.0;
        // Deterministic start with a gradient along the index order so
        // the projection onto the Fiedler direction cannot vanish by
        // symmetry for the graphs a catalog produces.
        let mut v: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let mut rayleigh = 0.0_f64;
        for _ in 0..2048 {
            // Project out the constant vector.
            let mean = v.iter().sum::<f64>() / n as f64;
            for value in &mut v {
                *value -= mean;
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm <= f64::EPSILON {
                break;
            }
            for value in &mut v {
                *value /= norm;
            }
            // w = (c I - L) v = c v - D v + W v
            let mut w = vec![0.0_f64; n];
            for i in 0..n {
                let mut acc = (ceiling - self.degree(i)) * v[i];
                for j in 0..n {
                    acc += self.weights[i][j] * v[j];
                }
                w[i] = acc;
            }
            let next_rayleigh = v.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>();
            let settled = (next_rayleigh - rayleigh).abs() <= 1e-13 * ceiling;
            rayleigh = next_rayleigh;
            v = w;
            if settled {
                break;
            }
        }
        let mean = v.iter().sum::<f64>() / n as f64;
        for value in &mut v {
            *value -= mean;
        }
        let algebraic_connectivity = (ceiling - rayleigh).max(0.0);

        let mut left = Vec::new();
        let mut right = Vec::new();
        for (i, &basin) in self.basins.iter().enumerate() {
            if v[i] < 0.0 {
                left.push(basin);
            } else {
                right.push(basin);
            }
        }
        if left.is_empty() || right.is_empty() {
            // A degenerate direction: split by the median instead so the
            // referee always names two sides.
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
            left = order[..n / 2].iter().map(|&i| self.basins[i]).collect();
            right = order[n / 2..].iter().map(|&i| self.basins[i]).collect();
        }

        let side = |basin: u64| left.contains(&basin);
        let mut cut = 0.0;
        let mut volume_left = 0.0;
        let mut volume_right = 0.0;
        for i in 0..n {
            let vol = self.degree(i);
            if side(self.basins[i]) {
                volume_left += vol;
            } else {
                volume_right += vol;
            }
            for j in (i + 1)..n {
                if side(self.basins[i]) != side(self.basins[j]) {
                    cut += self.weights[i][j];
                }
            }
        }
        let denominator = volume_left.min(volume_right);
        let conductance = if denominator > 0.0 {
            cut / denominator
        } else {
            0.0
        };

        let anchor = |members: &[u64]| -> u64 {
            members
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    self.degree(self.index[&a])
                        .total_cmp(&self.degree(self.index[&b]))
                })
                .unwrap_or(members[0])
        };
        Ok(SpectralSplit {
            algebraic_connectivity,
            representatives: (anchor(&left), anchor(&right)),
            conductance,
            left,
            right,
        })
    }

    /// Expected number of transitions from every basin to reach any
    /// basin of `targets`, treating observed crossing weights as rates
    /// of a discrete jump chain. Basins in `targets` price at zero; a
    /// basin with no path prices at infinity. This is the referee's
    /// stall arithmetic: when the cheapest passage to anywhere new
    /// exceeds the remaining budget, local work in the current
    /// community is not worth its price.
    pub fn passage_times(&self, targets: &[u64]) -> Result<HashMap<u64, f64>, GraphError> {
        let n = self.len();
        if n < 2 {
            return Err(GraphError::TooSmall);
        }
        for basin in targets {
            if !self.index.contains_key(basin) {
                return Err(GraphError::UnknownBasin);
            }
        }
        let absorbing: Vec<bool> = self.basins.iter().map(|b| targets.contains(b)).collect();
        if targets.is_empty() || absorbing.iter().all(|&a| a) {
            return Err(GraphError::BadAbsorbingSet);
        }
        // Basins with no path to the target set price at infinity, and
        // the linear system below is singular exactly on them, so
        // reachability separates first.
        let mut reachable = absorbing.clone();
        let mut frontier: Vec<usize> = (0..n).filter(|&i| absorbing[i]).collect();
        while let Some(i) = frontier.pop() {
            for j in 0..n {
                if self.weights[i][j] > 0.0 && !reachable[j] {
                    reachable[j] = true;
                    frontier.push(j);
                }
            }
        }
        let transient: Vec<usize> = (0..n).filter(|&i| !absorbing[i] && reachable[i]).collect();
        let mut times = HashMap::new();
        for (i, &basin) in self.basins.iter().enumerate() {
            times.insert(basin, if absorbing[i] { 0.0 } else { f64::INFINITY });
        }
        if transient.is_empty() {
            return Ok(times);
        }
        // Solve (I - Q) t = 1 over the reachable transient basins, where
        // Q is the jump chain restricted to them; every such basin leaks
        // toward absorption, so the system is nonsingular. Dense
        // elimination: the system is at most catalog capacity.
        let m = transient.len();
        let mut a = vec![vec![0.0_f64; m + 1]; m];
        for (row, &i) in transient.iter().enumerate() {
            let degree = self.degree(i);
            a[row][row] = 1.0;
            a[row][m] = 1.0;
            for (column, &j) in transient.iter().enumerate() {
                if i != j {
                    a[row][column] -= self.weights[i][j] / degree;
                }
            }
        }
        // Gaussian elimination with partial pivoting.
        for pivot in 0..m {
            let best = (pivot..m)
                .max_by(|&x, &y| a[x][pivot].abs().total_cmp(&a[y][pivot].abs()))
                .unwrap_or(pivot);
            a.swap(pivot, best);
            let head = a[pivot][pivot];
            for row in 0..m {
                if row != pivot {
                    let factor = a[row][pivot] / head;
                    if factor != 0.0 {
                        for column in pivot..=m {
                            a[row][column] -= factor * a[pivot][column];
                        }
                    }
                }
            }
        }
        for (row, &i) in transient.iter().enumerate() {
            times.insert(self.basins[i], a[row][m] / a[row][row]);
        }
        Ok(times)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn barbell() -> LandscapeGraph {
        // Two 4-cliques, basins 0..4 and 10..14, joined by one weak edge
        // between 3 and 10.
        let mut g = LandscapeGraph::new();
        for community in [0u64, 10] {
            for a in 0..4 {
                for b in (a + 1)..4 {
                    g.observe_crossing(community + a, community + b, 5.0);
                }
            }
        }
        g.observe_crossing(3, 10, 0.5);
        g
    }

    #[test]
    fn the_split_of_a_barbell_is_its_cliques() {
        let split = barbell().spectral_split().unwrap();
        let mut left = split.left.clone();
        let mut right = split.right.clone();
        left.sort_unstable();
        right.sort_unstable();
        let (low, high) = if left[0] < 10 {
            (left, right)
        } else {
            (right, left)
        };
        assert_eq!(low, vec![0, 1, 2, 3]);
        assert_eq!(high, vec![10, 11, 12, 13]);
        // The weak seam makes the landscape nearly disconnected.
        assert!(split.algebraic_connectivity < 0.5);
        assert!(
            split.conductance < 0.02,
            "conductance {}",
            split.conductance
        );
        // Representatives sit on opposite sides and are the anchored
        // members: the joint basins carry the extra seam weight. The
        // side order is not promised, only the pairing.
        let mut reps = [split.representatives.0, split.representatives.1];
        reps.sort_unstable();
        assert_eq!(reps, [3, 10]);
    }

    #[test]
    fn the_connectivity_of_a_path_is_exact() {
        // L(P3) has eigenvalues 0, 1, 3.
        let mut g = LandscapeGraph::new();
        g.observe_crossing(0, 1, 1.0);
        g.observe_crossing(1, 2, 1.0);
        let split = g.spectral_split().unwrap();
        assert!(
            (split.algebraic_connectivity - 1.0).abs() < 1e-9,
            "lambda2 = {}",
            split.algebraic_connectivity
        );
    }

    #[test]
    fn the_fiedler_split_does_not_depend_on_start_vector_overlap() {
        // The index-ramp vector used by the iterative implementation is
        // orthogonal to this graph's Fiedler vector [1, -1, -1, 1].  An
        // eigensolve must still recover the weak cut {0, 3} | {1, 2} and
        // lambda_2 = 2.  Integer weights keep the orthogonality exact in the
        // reflected-Laplacian iteration instead of seeding the missing mode
        // through decimal roundoff.
        let mut g = LandscapeGraph::new();
        for basin in 0..4 {
            g.observe_basin(basin);
        }
        g.observe_crossing(0, 3, 4.0);
        g.observe_crossing(1, 2, 4.0);
        g.observe_crossing(0, 1, 1.0);
        g.observe_crossing(3, 2, 1.0);

        let split = g.spectral_split().unwrap();
        let mut sides = [split.left.clone(), split.right.clone()];
        for side in &mut sides {
            side.sort_unstable();
        }
        sides.sort();

        assert!((split.algebraic_connectivity - 2.0).abs() < 1e-10);
        assert_eq!(sides, [vec![0, 3], vec![1, 2]]);
    }

    #[test]
    fn a_disconnected_landscape_reports_zero_connectivity_and_components() {
        let mut g = LandscapeGraph::new();
        g.observe_crossing(0, 1, 3.0);
        g.observe_crossing(10, 11, 3.0);
        let split = g.spectral_split().unwrap();
        assert!(split.algebraic_connectivity < 1e-9);
        assert!(split.conductance < 1e-12);
        let mut sides = [split.left.clone(), split.right.clone()];
        for side in &mut sides {
            side.sort_unstable();
        }
        sides.sort();
        assert_eq!(sides, [vec![0, 1], vec![10, 11]]);
    }

    #[test]
    fn passage_times_on_a_path_match_the_closed_form() {
        // Unbiased walk on 0 - 1 - 2 absorbing at 2: t(1) = 3, t(0) = 4.
        let mut g = LandscapeGraph::new();
        g.observe_crossing(0, 1, 1.0);
        g.observe_crossing(1, 2, 1.0);
        let times = g.passage_times(&[2]).unwrap();
        assert!((times[&2] - 0.0).abs() < 1e-12);
        assert!((times[&1] - 3.0).abs() < 1e-9, "t1 = {}", times[&1]);
        assert!((times[&0] - 4.0).abs() < 1e-9, "t0 = {}", times[&0]);
    }

    #[test]
    fn unreachable_targets_price_at_infinity() {
        let mut g = LandscapeGraph::new();
        g.observe_crossing(0, 1, 1.0);
        g.observe_basin(7);
        let times = g.passage_times(&[7]).unwrap();
        assert!(times[&0].is_infinite());
        assert!(times[&1].is_infinite());
        assert_eq!(times[&7], 0.0);
    }

    #[test]
    fn bad_queries_are_rejected() {
        let mut g = LandscapeGraph::new();
        g.observe_basin(0);
        assert_eq!(g.spectral_split().unwrap_err(), GraphError::TooSmall);
        g.observe_crossing(0, 1, 1.0);
        assert_eq!(g.passage_times(&[9]).unwrap_err(), GraphError::UnknownBasin);
        assert_eq!(
            g.passage_times(&[0, 1]).unwrap_err(),
            GraphError::BadAbsorbingSet
        );
    }
}
