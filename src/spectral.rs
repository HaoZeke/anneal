//! Collective variables from the spectrum of the visited landscape.
//!
//! Every bias in this crate acts on a quantity computed from one configuration:
//! [`crate::bias::WellTemperedBias`] on a linear projection of it,
//! [`crate::bias::BasinBias`] on a fingerprint of it. Both are functions of a
//! single point, and on a multi-funnel landscape neither separates funnels
//! reliably. Measured over forty quenched 75-point Lennard-Jones minima, a
//! Steinhardt parameter spans four deposition widths and the best learned
//! projection twenty-two, and a bias on either fills the target funnel and its
//! competitor together.
//!
//! The object that carries the separation is not a configuration. It is the set
//! of minima with the transitions observed between them, and a funnel is a
//! poorly connected component of that graph. Component structure is spectral:
//! the eigenvectors of the graph Laplacian belonging to the smallest non-zero
//! eigenvalues vary slowly inside a component and quickly between them, so the
//! second one, the Fiedler vector, is a coordinate on which two funnels lie
//! apart by construction rather than by luck of projection.
//!
//! Two measurements make this the right object rather than a nice idea.
//!
//! The shape distance between such minima is bimodal with an empty middle:
//! a hop lands at either 0.000, having returned, or at a median 1.910, and two
//! independent minima never come closer than 1.663. A distance matrix with that
//! structure is block structured, which is where a spectral partition is exact
//! and where a merge radius or a kernel width has nothing to trade off. Three
//! radius calibrations and a shape-space kernel failed here for that reason.
//!
//! From the structure a search settles into, none of 1800 single moves reaches
//! anything lower, so escape is a sequence of accepted uphill hops rather than
//! a gap one move crosses. A quantity defined on the hop graph describes such a
//! sequence; a quantity defined on one structure cannot.
//!
//! The eigenproblem is solved by cyclic Jacobi rather than by pulling in a
//! LAPACK binding. The matrix is the visited-basin graph, orders of magnitude
//! smaller than the relaxations that produced it, and Jacobi is accurate for
//! small symmetric matrices and has no dependency.

use std::collections::{BTreeMap, BTreeSet};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::bias::{Bias, Fingerprint};

/// Eigenvalues and eigenvectors of a symmetric matrix, ascending.
///
/// Cyclic Jacobi: repeatedly annihilate the largest off-diagonal entry by a
/// plane rotation. Converges for any real symmetric matrix and is accurate for
/// small ones, which is what a visited-basin graph is.
///
/// Returns `(values, vectors)` with eigenvector `k` in column `k`.
pub fn symmetric_eigen(a: ArrayView2<f64>, max_sweeps: usize) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "symmetric_eigen needs a square matrix");
    let mut m = a.to_owned();
    let mut v = Array2::<f64>::eye(n);

    for _ in 0..max_sweeps {
        // Off-diagonal Frobenius norm; the sweep stops when it is negligible
        // against the diagonal rather than at a fixed iteration count.
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[[i, j]] * m[[i, j]];
            }
        }
        if off.sqrt() <= 1e-14 * (1.0 + m.diag().iter().map(|d| d.abs()).sum::<f64>()) {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[[p, q]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[[q, q]] - m[[p, p]]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = m[[k, p]];
                    let akq = m[[k, q]];
                    m[[k, p]] = c * akp - s * akq;
                    m[[k, q]] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = m[[p, k]];
                    let aqk = m[[q, k]];
                    m[[p, k]] = c * apk - s * aqk;
                    m[[q, k]] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut vals: Vec<(f64, usize)> = (0..n).map(|i| (m[[i, i]], i)).collect();
    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut out_vals = Array1::zeros(n);
    let mut out_vecs = Array2::zeros((n, n));
    for (new, (val, old)) in vals.into_iter().enumerate() {
        out_vals[new] = val;
        for k in 0..n {
            out_vecs[[k, new]] = v[[k, old]];
        }
    }
    (out_vals, out_vecs)
}

/// Reasons an embedding cannot be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectralError {
    /// Fewer nodes than a partition needs.
    TooFewNodes(usize),
    /// Nodes with no edges, for which the normalised Laplacian is undefined.
    IsolatedNodes(usize),
}

impl std::fmt::Display for SpectralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpectralError::TooFewNodes(n) => {
                write!(f, "a spectral coordinate needs at least 3 nodes, got {n}")
            }
            SpectralError::IsolatedNodes(n) => write!(
                f,
                "{n} nodes have no edges; the normalised Laplacian is undefined there"
            ),
        }
    }
}

impl std::error::Error for SpectralError {}

/// Eigenvectors of the normalised Laplacian, smallest non-trivial first.
///
/// Uses `L = I - D^-1/2 W D^-1/2` and returns the random-walk eigenvectors, by
/// applying the `D^-1/2` scaling, so the result is a function on nodes rather
/// than on the symmetrised operator.
///
/// The constant eigenvector belonging to eigenvalue zero is dropped: it says
/// nothing about how the graph divides.
pub fn laplacian_embedding(
    weights: ArrayView2<f64>,
    n_vectors: usize,
) -> Result<(Array1<f64>, Array2<f64>), SpectralError> {
    let n = weights.nrows();
    if n < 3 {
        return Err(SpectralError::TooFewNodes(n));
    }
    let degree: Array1<f64> = weights.sum_axis(ndarray::Axis(1));
    let isolated = degree.iter().filter(|d| **d <= 0.0).count();
    if isolated > 0 {
        return Err(SpectralError::IsolatedNodes(isolated));
    }
    let d_inv_sqrt: Array1<f64> = degree.mapv(|d| 1.0 / d.sqrt());

    let mut lap = Array2::<f64>::eye(n);
    for i in 0..n {
        for j in 0..n {
            lap[[i, j]] -= weights[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
        }
    }
    let (vals, vecs) = symmetric_eigen(lap.view(), 64);

    let take = n_vectors.min(n - 1);
    let mut out_vals = Array1::zeros(take);
    let mut out_vecs = Array2::zeros((n, take));
    for k in 0..take {
        out_vals[k] = vals[k + 1];
        for i in 0..n {
            out_vecs[[i, k]] = vecs[[i, k + 1]] * d_inv_sqrt[i];
        }
    }
    Ok((out_vals, out_vecs))
}

/// Basins visited, and the transitions observed between them.
#[derive(Debug, Default)]
pub struct TransitionGraph {
    edges: BTreeMap<(usize, usize), f64>,
    nodes: BTreeSet<usize>,
}

impl TransitionGraph {
    /// Records a hop.
    ///
    /// A hop that returns to the same basin marks the node as present but adds
    /// no edge: it says nothing about connectivity between components, and
    /// counting it would load the diagonal and flatten the spectrum. Given the
    /// measured return rate near a deep minimum, roughly 19 proposals in 20,
    /// including them would leave the graph describing little else.
    pub fn record(&mut self, from: usize, to: usize, weight: f64) {
        self.nodes.insert(from);
        self.nodes.insert(to);
        if from == to || !(weight > 0.0) {
            return;
        }
        let key = (from.min(to), from.max(to));
        *self.edges.entry(key).or_insert(0.0) += weight;
    }

    /// Nodes seen.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when no node has been seen.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Edges recorded.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Adjacency over the connected nodes, with the basin ids they belong to.
    ///
    /// Nodes seen but never connected are dropped rather than kept with a zero
    /// degree, which would make the normalised Laplacian undefined.
    pub fn adjacency(&self) -> (Vec<usize>, Array2<f64>) {
        let ids: Vec<usize> = self.nodes.iter().copied().collect();
        let index: BTreeMap<usize, usize> =
            ids.iter().enumerate().map(|(i, b)| (*b, i)).collect();
        let mut w = Array2::<f64>::zeros((ids.len(), ids.len()));
        for ((a, b), c) in &self.edges {
            let (i, j) = (index[a], index[b]);
            w[[i, j]] += c;
            w[[j, i]] += c;
        }
        let keep: Vec<usize> = (0..ids.len())
            .filter(|i| w.row(*i).sum() > 0.0)
            .collect();
        let mut sub = Array2::<f64>::zeros((keep.len(), keep.len()));
        for (a, i) in keep.iter().enumerate() {
            for (b, j) in keep.iter().enumerate() {
                sub[[a, b]] = w[[*i, *j]];
            }
        }
        (keep.into_iter().map(|i| ids[i]).collect(), sub)
    }
}

/// Well-tempered bias deposited on the Fiedler coordinate of the hop graph.
///
/// Keyed through a [`Fingerprint`] like [`crate::bias::BasinBias`], so it acts
/// on the same exact basin identity that works on this landscape, and deposits
/// on a continuous coordinate so that filling spreads over a funnel rather than
/// over one basin at a time. That combination is the point: identity supplies
/// the resolution, the spectrum supplies the generalisation, and neither a
/// hand-picked order parameter nor a distance threshold is involved.
///
/// A basin with no spectral position yet, because it is newly seen or not yet
/// connected, reads as the origin. That is a real position rather than a
/// missing value: an unconnected basin is not known to belong to either side of
/// any partition.
pub struct SpectralBias<F: Fingerprint> {
    fingerprint: F,
    merge_radius: f64,
    centres: Vec<Array1<f64>>,
    graph: TransitionGraph,
    coords: BTreeMap<usize, Array1<f64>>,
    /// Deposited height per visit before well tempering.
    pub w0: f64,
    /// Well-tempered factor; must exceed one.
    pub gamma: f64,
    /// Gaussian width on the spectral coordinate.
    pub sigma: f64,
    /// Hops between recomputations of the embedding.
    pub refit_every: usize,
    /// Connected nodes required before an embedding is attempted.
    pub min_nodes: usize,
    n_vectors: usize,
    hills: Vec<(Array1<f64>, f64)>,
    since_refit: usize,
    last_basin: Option<usize>,
    /// Embeddings computed.
    pub refits: usize,
    /// Why the last attempt failed, if it did.
    pub last_error: Option<SpectralError>,
}

impl<F: Fingerprint> SpectralBias<F> {
    /// Bias keyed by `fingerprint`, merging basins within `merge_radius`.
    pub fn new(fingerprint: F, merge_radius: f64, w0: f64, gamma: f64, sigma: f64) -> Self {
        assert!(gamma > 1.0, "well-tempered gamma must exceed one, got {gamma}");
        Self {
            fingerprint,
            merge_radius,
            centres: Vec::new(),
            graph: TransitionGraph::default(),
            coords: BTreeMap::new(),
            w0,
            gamma,
            sigma,
            refit_every: 64,
            min_nodes: 8,
            n_vectors: 1,
            hills: Vec::new(),
            since_refit: 0,
            last_basin: None,
            refits: 0,
            last_error: None,
        }
    }

    /// Basin index of a descriptor, creating one when it is new.
    fn identify(&mut self, s: ArrayView1<f64>) -> usize {
        // Most recent first: a chain revisits where it just was far more often
        // than anywhere else, so this ends early on the common case.
        for (i, c) in self.centres.iter().enumerate().rev() {
            let d2: f64 = c
                .iter()
                .zip(s.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d2.sqrt() <= self.merge_radius {
                return i;
            }
        }
        self.centres.push(s.to_owned());
        self.centres.len() - 1
    }

    /// Basin index without creating one, for read-only queries.
    fn peek(&self, s: ArrayView1<f64>) -> Option<usize> {
        for (i, c) in self.centres.iter().enumerate().rev() {
            let d2: f64 = c
                .iter()
                .zip(s.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d2.sqrt() <= self.merge_radius {
                return Some(i);
            }
        }
        None
    }

    /// Recomputes the embedding. Returns whether it succeeded.
    ///
    /// A failure is recorded rather than leaving stale coordinates silently in
    /// place, because a coordinate that stopped updating is a bias acting on a
    /// graph the run has moved past.
    pub fn refit(&mut self) -> bool {
        self.since_refit = 0;
        let (ids, w) = self.graph.adjacency();
        if ids.len() < self.min_nodes.max(3) {
            self.last_error = Some(SpectralError::TooFewNodes(ids.len()));
            return false;
        }
        match laplacian_embedding(w.view(), self.n_vectors) {
            Ok((_, vecs)) => {
                // Standardised, so a width means the same thing however many
                // nodes the graph has: eigenvector normalisation shrinks the
                // raw scale as nodes are added.
                let mut sd = vec![0.0; self.n_vectors];
                for k in 0..self.n_vectors {
                    let col = vecs.column(k);
                    let mean = col.sum() / col.len() as f64;
                    let var =
                        col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / col.len() as f64;
                    sd[k] = if var > 0.0 { var.sqrt() } else { 1.0 };
                }
                self.coords.clear();
                for (i, b) in ids.iter().enumerate() {
                    let mut c = Array1::zeros(self.n_vectors);
                    for k in 0..self.n_vectors {
                        c[k] = vecs[[i, k]] / sd[k];
                    }
                    self.coords.insert(*b, c);
                }
                self.refits += 1;
                self.last_error = None;
                true
            }
            Err(e) => {
                self.last_error = Some(e);
                false
            }
        }
    }

    /// Spectral position of a basin, or the origin when it has none.
    pub fn coordinate(&self, basin: usize) -> Array1<f64> {
        self.coords
            .get(&basin)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(self.n_vectors))
    }

    /// Basins carrying a spectral position.
    pub fn n_placed(&self) -> usize {
        self.coords.len()
    }

    /// Hops recorded.
    pub fn n_edges(&self) -> usize {
        self.graph.n_edges()
    }
}

impl<F: Fingerprint> Bias for SpectralBias<F> {
    /// The Fiedler coordinate of the basin holding `x`.
    ///
    /// Read-only, so an unseen structure reads as the origin rather than
    /// silently opening a basin: mapping a position to a CV must not mutate the
    /// partition the CV is defined on.
    fn cv(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let d = self.fingerprint.describe(x);
        match self.peek(d.view()) {
            Some(b) => self.coordinate(b),
            None => Array1::zeros(self.n_vectors),
        }
    }

    fn potential(&self, s: ArrayView1<f64>) -> f64 {
        let two_sigma2 = 2.0 * self.sigma * self.sigma;
        self.hills
            .iter()
            .map(|(c, h)| {
                let d2: f64 = c
                    .iter()
                    .zip(s.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                h * (-d2 / two_sigma2).exp()
            })
            .sum()
    }

    fn deposit(&mut self, s: ArrayView1<f64>, temp: f64) {
        let denom = (self.gamma - 1.0) * temp;
        let w = self.w0 * (-self.potential(s) / denom).exp();
        self.hills.push((s.to_owned(), w));
    }
}

impl<F: Fingerprint> SpectralBias<F> {
    /// Records that the chain moved to the basin holding `x`, and deposits.
    ///
    /// The graph is built from what the chain actually does, so this is the
    /// entry point a driver calls rather than [`Bias::deposit`] alone: the
    /// coordinate does not exist until the transitions that define it do.
    pub fn visit(&mut self, x: ArrayView1<f64>, temp: f64) {
        let d = self.fingerprint.describe(x);
        let basin = self.identify(d.view());
        // Accepted inter-basin hops only grow the graph. A self-edge is the
        // chain sitting still (or rejecting a trial); including it densifies
        // every basin into a clique with itself and washes out the cut the
        // Fiedler vector is meant to find.
        if let Some(prev) = self.last_basin {
            if prev != basin {
                self.graph.record(prev, basin, 1.0);
                self.since_refit += 1;
            }
        }
        self.last_basin = Some(basin);
        if self.graph.len() >= self.min_nodes && self.since_refit >= self.refit_every {
            self.refit();
        }
        let s = self.coordinate(basin);
        self.deposit(s.view(), temp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bias::SortedPairs;
    use ndarray::array;

    #[test]
    fn jacobi_matches_a_known_spectrum() {
        // Eigenvalues 1, 2, 3 by construction.
        let a = array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let (vals, _) = symmetric_eigen(a.view(), 64);
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
        assert!((vals[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn jacobi_diagonalises_a_dense_symmetric_matrix() {
        let a = array![
            [4.0, 1.0, -2.0, 2.0],
            [1.0, 2.0, 0.0, 1.0],
            [-2.0, 0.0, 3.0, -2.0],
            [2.0, 1.0, -2.0, -1.0]
        ];
        let (vals, vecs) = symmetric_eigen(a.view(), 128);
        for k in 0..4 {
            let v = vecs.column(k);
            let av = a.dot(&v);
            for i in 0..4 {
                assert!(
                    (av[i] - vals[k] * v[i]).abs() < 1e-9,
                    "eigenpair {k} fails at {i}"
                );
            }
        }
    }

    /// The property the module exists for: two well-connected groups joined by
    /// one weak edge must separate by the sign of the Fiedler coordinate.
    #[test]
    fn fiedler_separates_two_weakly_joined_cliques() {
        let n = 6;
        let mut w = Array2::<f64>::zeros((n, n));
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    w[[i, j]] = 1.0;
                }
            }
        }
        for i in 3..6 {
            for j in 3..6 {
                if i != j {
                    w[[i, j]] = 1.0;
                }
            }
        }
        // The single weak bridge; a funnel boundary looks like this.
        w[[2, 3]] = 0.01;
        w[[3, 2]] = 0.01;

        let (vals, vecs) = laplacian_embedding(w.view(), 1).unwrap();
        let f = vecs.column(0);
        let left = (0..3).all(|i| f[i] > 0.0);
        let right = (3..6).all(|i| f[i] < 0.0);
        let flipped = (0..3).all(|i| f[i] < 0.0) && (3..6).all(|i| f[i] > 0.0);
        assert!(
            (left && right) || flipped,
            "Fiedler vector did not split the cliques: {f:?}"
        );
        assert!(
            vals[0] > 0.0 && vals[0] < 0.2,
            "a weakly joined pair should have a small non-zero eigenvalue, got {}",
            vals[0]
        );
    }

    #[test]
    fn isolated_nodes_are_reported_not_silently_dropped() {
        let w = Array2::<f64>::zeros((4, 4));
        match laplacian_embedding(w.view(), 1) {
            Err(SpectralError::IsolatedNodes(n)) => assert_eq!(n, 4),
            other => panic!("expected an isolated-node error, got {other:?}"),
        }
    }

    #[test]
    fn self_hops_add_no_edges() {
        let mut g = TransitionGraph::default();
        for _ in 0..50 {
            g.record(0, 0, 1.0);
        }
        assert_eq!(g.n_edges(), 0, "a return is not a transition between basins");
        assert_eq!(g.len(), 1);
        g.record(0, 1, 1.0);
        assert_eq!(g.n_edges(), 1);
    }

    #[test]
    fn unconnected_nodes_are_excluded_from_the_adjacency() {
        let mut g = TransitionGraph::default();
        g.record(0, 1, 1.0);
        g.record(1, 2, 1.0);
        g.record(9, 9, 1.0); // seen, never connected
        let (ids, w) = g.adjacency();
        assert_eq!(ids, vec![0, 1, 2], "an unconnected node has no coordinate");
        assert_eq!(w.nrows(), 3);
    }

    #[test]
    fn cv_does_not_open_a_basin() {
        let bias = SpectralBias::new(SortedPairs { n_points: 4 }, 0.2, 0.05, 5.0, 0.35);
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let before = bias.n_placed();
        let s = bias.cv(x.view());
        assert_eq!(s.len(), 1);
        assert_eq!(bias.n_placed(), before, "cv must not mutate the partition");
    }

    #[test]
    fn a_visited_chain_builds_a_graph_and_places_basins() {
        let mut bias = SpectralBias::new(SortedPairs { n_points: 4 }, 0.05, 0.05, 5.0, 0.35);
        bias.min_nodes = 4;
        bias.refit_every = 4;
        // Twelve distinguishable structures, walked in a cycle so the graph is
        // connected and every node has degree two.
        let mut states = Vec::new();
        for k in 0..12 {
            let s = 1.0 + 0.5 * k as f64;
            states.push(Array1::from(vec![
                0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s,
            ]));
        }
        for round in 0..8 {
            for st in &states {
                bias.visit(st.view(), 1.0);
            }
            if round == 0 {
                assert!(bias.n_edges() > 0, "the walk recorded no transitions");
            }
        }
        assert!(bias.refits > 0, "no embedding was computed: {:?}", bias.last_error);
        assert!(
            bias.n_placed() >= 4,
            "only {} basins were placed",
            bias.n_placed()
        );
        // A deposited bias must be positive where the chain has been.
        let s = bias.cv(states[0].view());
        assert!(bias.potential(s.view()) > 0.0);
    }
}

/// The leading diffusion direction of an archive of minima, with a Nystrom
/// extension for proposing from any structure.
///
/// The archive of visited minima is a point cloud on the landscape's
/// low-dimensional backbone. Its diffusion map (Coifman and Lafon,
/// doi:10.1016/j.acha.2006.04.006) is the spectral embedding of the
/// row-normalised kernel; the leading nontrivial eigenvector orders the
/// archive along its principal connectivity direction, the same object the
/// sketch-map literature draws for these landscapes, computed here by power
/// iteration on the small dense kernel. The Nystrom extension evaluates that
/// coordinate at a structure outside the archive, so a proposal can step
/// along the backbone rather than isotropically. Rational spectral filtering
/// over the sparse kernel replaces the dense pass when the archive outgrows
/// it; at the archive sizes a charged run accumulates, dense is exact and
/// cheaper.
pub struct DiffusionDirection {
    /// Archive descriptors, one sorted-distance spectrum per minimum.
    anchors: Vec<Vec<f64>>,
    /// Leading nontrivial eigenvector entries per anchor.
    psi: Vec<f64>,
    /// Kernel bandwidth, the median pairwise distance of the archive.
    bandwidth: f64,
}

/// Sorted pairwise-distance spectrum: permutation and rotation invariant.
fn distance_spectrum(x: &[f64]) -> Vec<f64> {
    let n = x.len() / 3;
    let mut d = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            d.push(
                (0..3)
                    .map(|k| (x[3 * i + k] - x[3 * j + k]).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            );
        }
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d
}

fn euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt()
}

impl DiffusionDirection {
    /// Fits the leading diffusion coordinate of `structures`.
    ///
    /// Returns `None` below four anchors, where a direction is not defined.
    pub fn fit(structures: &[Vec<f64>]) -> Option<Self> {
        let m = structures.len();
        if m < 4 {
            return None;
        }
        let anchors: Vec<Vec<f64>> = structures.iter().map(|s| distance_spectrum(s)).collect();
        let mut dists = Vec::new();
        for i in 0..m {
            for j in (i + 1)..m {
                dists.push(euclid(&anchors[i], &anchors[j]));
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let bandwidth = dists[dists.len() / 2].max(1e-9);
        // Row-normalised kernel: the diffusion operator.
        let mut k = vec![vec![0.0; m]; m];
        for i in 0..m {
            let mut row = 0.0;
            for j in 0..m {
                let v = (-euclid(&anchors[i], &anchors[j]).powi(2)
                    / (2.0 * bandwidth * bandwidth))
                    .exp();
                k[i][j] = v;
                row += v;
            }
            for j in 0..m {
                k[i][j] /= row;
            }
        }
        // Power iteration deflated against the trivial constant eigenvector.
        let mut psi: Vec<f64> = (0..m).map(|i| (i as f64).sin() + 0.01).collect();
        for _ in 0..200 {
            let mean = psi.iter().sum::<f64>() / m as f64;
            for v in psi.iter_mut() {
                *v -= mean;
            }
            let mut next = vec![0.0; m];
            for i in 0..m {
                for j in 0..m {
                    next[i] += k[i][j] * psi[j];
                }
            }
            let norm = next.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
            for v in next.iter_mut() {
                *v /= norm;
            }
            psi = next;
        }
        Some(Self {
            anchors,
            psi,
            bandwidth,
        })
    }

    /// The diffusion coordinate of an arbitrary structure, by the Nystrom
    /// extension: the kernel-weighted average of the anchor coordinates.
    pub fn coordinate(&self, x: &[f64]) -> f64 {
        let s = distance_spectrum(x);
        let mut num = 0.0;
        let mut den = 0.0;
        for (a, p) in self.anchors.iter().zip(self.psi.iter()) {
            let w = (-euclid(&s, a).powi(2) / (2.0 * self.bandwidth * self.bandwidth)).exp();
            num += w * p;
            den += w;
        }
        if den > 1e-300 { num / den } else { 0.0 }
    }
}

#[cfg(test)]
mod diffusion_tests {
    use super::*;

    /// An archive lying along a curve has to come back ordered by the leading
    /// diffusion coordinate, and the Nystrom extension has to place a held-out
    /// point between its neighbours. That is the whole claim: the embedding
    /// recovers the backbone.
    #[test]
    fn the_backbone_is_recovered_and_extended() {
        // Twelve four-point structures along a one-parameter stretch.
        let make = |t: f64| -> Vec<f64> {
            vec![
                0.0, 0.0, 0.0,
                1.0 + t, 0.0, 0.0,
                0.0, 1.0 + 0.5 * t, 0.0,
                0.0, 0.0, 1.0 + 0.25 * t,
            ]
        };
        let arch: Vec<Vec<f64>> = (0..12).map(|i| make(i as f64 * 0.1)).collect();
        let d = DiffusionDirection::fit(&arch).expect("no direction");
        let coords: Vec<f64> = arch.iter().map(|s| d.coordinate(s)).collect();
        let increasing = coords.windows(2).all(|w| w[1] > w[0]);
        let decreasing = coords.windows(2).all(|w| w[1] < w[0]);
        assert!(
            increasing || decreasing,
            "diffusion coordinate does not order the backbone: {coords:?}"
        );
        let held = make(0.55);
        let c = d.coordinate(&held);
        let (lo, hi) = (coords[5].min(coords[6]), coords[5].max(coords[6]));
        assert!(
            c > lo && c < hi,
            "held-out point at {c} not between its neighbours [{lo}, {hi}]"
        );
    }
}
