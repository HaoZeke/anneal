//! Local topology keys, after k-ART.
//!
//! A global contact-graph hash treats every isomer as unknown. k-ART hashes
//! the neighbour graph of *one atom* (the atom and everything within two
//! contact steps). Icosahedral isomers share those local keys; a Marks
//! decahedron adds fivefold-join keys the icosahedral catalogue does not have.
//!
//! The cutoff is a multiple of the structure's own median nearest-neighbour
//! distance, the same scale [`crate::graphkey`] uses for the global key.

use ndarray::ArrayView1;
use petgraph::graph::UnGraph;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};

/// Contact shells around the atom. Shrunk when a two-shell ball is the
/// whole cluster, so LJ13 still distinguishes centre from vertex.
pub const SHELLS: usize = 2;

/// Median nearest-neighbour distance of `x`.
pub fn median_nn(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n < 2 {
        return 1.0;
    }
    let mut nn = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d2: f64 = (0..3)
                .map(|k| {
                    let d = x[3 * i + k] - x[3 * j + k];
                    d * d
                })
                .sum();
            if d2 < best {
                best = d2;
            }
        }
        nn.push(best.sqrt());
    }
    nn.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    nn[n / 2]
}

fn dist(x: ArrayView1<f64>, i: usize, j: usize) -> f64 {
    (0..3)
        .map(|k| {
            let d = x[3 * i + k] - x[3 * j + k];
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

fn adjacency(x: ArrayView1<f64>, reach: f64) -> Vec<Vec<usize>> {
    let n = x.len() / 3;
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if dist(x, i, j) < reach {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

fn hash_graph(nodes: &[usize], adj: &[Vec<usize>]) -> u64 {
    let mut g: UnGraph<(), ()> = UnGraph::default();
    let mut ix = vec![None; adj.len()];
    let handles: Vec<_> = nodes
        .iter()
        .map(|&a| {
            let h = g.add_node(());
            ix[a] = Some(h);
            h
        })
        .collect();
    let _ = handles;
    let set: HashSet<usize> = nodes.iter().copied().collect();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for &a in nodes {
        for &b in &adj[a] {
            if b > a && set.contains(&b) {
                if let (Some(ha), Some(hb)) = (ix[a], ix[b]) {
                    g.add_edge(ha, hb, ());
                    edges.push((a, b));
                }
            }
        }
    }
    use nauty_pet::canon::IntoCanon;
    let canon = g.into_canon();
    let mut canon_edges: Vec<(usize, usize)> = canon
        .edge_indices()
        .map(|e| {
            let (u, v) = canon.edge_endpoints(e).expect("edge exists");
            let (u, v) = (u.index(), v.index());
            if u <= v { (u, v) } else { (v, u) }
        })
        .collect();
    canon_edges.sort_unstable();
    let mut h = DefaultHasher::new();
    canon.node_count().hash(&mut h);
    canon_edges.hash(&mut h);
    h.finish()
}

/// Canonical key of the two-shell contact graph around atom `i`.
pub fn local_key(x: ArrayView1<f64>, i: usize, cutoff: f64) -> u64 {
    let n = x.len() / 3;
    if n < 2 || i >= n {
        return 0;
    }
    let reach = cutoff * median_nn(x);
    let adj = adjacency(x, reach);
    let mut shells = SHELLS;
    loop {
        let mut seen = vec![false; n];
        let mut dist_hops = vec![usize::MAX; n];
        let mut q = VecDeque::new();
        seen[i] = true;
        dist_hops[i] = 0;
        q.push_back(i);
        let mut nodes = Vec::new();
        while let Some(u) = q.pop_front() {
            nodes.push(u);
            if dist_hops[u] >= shells {
                continue;
            }
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    dist_hops[v] = dist_hops[u] + 1;
                    q.push_back(v);
                }
            }
        }
        if nodes.len() < n || shells == 1 {
            nodes.sort_unstable();
            return hash_graph(&nodes, &adj);
        }
        shells -= 1;
    }
}

/// Local key of every atom, in atom order.
pub fn local_keys(x: ArrayView1<f64>, cutoff: f64) -> Vec<u64> {
    let n = x.len() / 3;
    (0..n).map(|i| local_key(x, i, cutoff)).collect()
}

/// Hash of the sorted multiset of local keys: the system's local-topology bag.
pub fn bag_key(keys: &[u64]) -> u64 {
    let mut v = keys.to_vec();
    v.sort_unstable();
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// How many keys of `a` also appear in `b` (multiset intersection size).
pub fn bag_overlap(a: &[u64], b: &[u64]) -> usize {
    let mut aa = a.to_vec();
    let mut bb = b.to_vec();
    aa.sort_unstable();
    bb.sort_unstable();
    let mut i = 0;
    let mut j = 0;
    let mut n = 0;
    while i < aa.len() && j < bb.len() {
        match aa[i].cmp(&bb[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn ico13() -> Array1<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut v = vec![[0.0, 0.0, 0.0]];
        for s in [1.0, -1.0] {
            for t in [phi, -phi] {
                v.push([0.0, s, t]);
                v.push([s, t, 0.0]);
                v.push([t, 0.0, s]);
            }
        }
        let mut x = Array1::zeros(39);
        for (i, p) in v.iter().enumerate() {
            for k in 0..3 {
                x[3 * i + k] = p[k] * 0.6;
            }
        }
        x
    }

    fn rotate_z(x: &Array1<f64>, ang: f64) -> Array1<f64> {
        let n = x.len() / 3;
        let (s, c) = ang.sin_cos();
        let mut y = x.clone();
        for i in 0..n {
            let px = x[3 * i];
            let py = x[3 * i + 1];
            y[3 * i] = c * px - s * py;
            y[3 * i + 1] = s * px + c * py;
        }
        y
    }

    #[test]
    fn centre_and_surface_keys_differ() {
        let x = ico13();
        let k0 = local_key(x.view(), 0, 1.35);
        let k1 = local_key(x.view(), 1, 1.35);
        assert_ne!(k0, k1, "centre and surface atoms share a local topology");
    }

    #[test]
    fn rotation_preserves_local_keys() {
        let x = ico13();
        let y = rotate_z(&x, 0.7);
        let a = local_keys(x.view(), 1.35);
        let b = local_keys(y.view(), 1.35);
        let mut aa = a.clone();
        let mut bb = b.clone();
        aa.sort_unstable();
        bb.sort_unstable();
        assert_eq!(aa, bb, "a rotation changed the local-key multiset");
        assert_eq!(bag_key(&a), bag_key(&b));
    }

    #[test]
    fn surface_atoms_share_a_key() {
        let x = ico13();
        let k1 = local_key(x.view(), 1, 1.35);
        let k2 = local_key(x.view(), 2, 1.35);
        assert_eq!(k1, k2, "two ico vertices should be the same local topology");
    }

    #[test]
    fn bag_overlap_is_full_on_a_copy() {
        let x = ico13();
        let a = local_keys(x.view(), 1.35);
        assert_eq!(bag_overlap(&a, &a), a.len());
    }
}
