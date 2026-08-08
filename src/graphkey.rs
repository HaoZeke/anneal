//! Exact basin identity by canonical contact-graph labelling.
//!
//! Measured here, no coordinate radius defines basin identity: at 0.7 the run
//! registers 365 basins and solves 55 of 72, at 2.0 it registers 7 and solves
//! 24, and there is no setting in between where basins correspond to anything.
//! The failure is structural, since a single isotropic length cannot separate
//! "the same packing, distorted" from "a different packing".
//!
//! The contact graph can. Two minima are the same arrangement exactly when
//! their bond graphs are isomorphic, and a canonical labelling (McKay's nauty,
//! doi:10.1016/j.jsc.2013.09.003) turns isomorphism into equality of keys: no
//! threshold, no reference structure, no morphology. The graph is built from
//! the structure's own nearest-neighbour scale, so the key transfers across
//! systems and sizes unchanged.
use ndarray::ArrayView1;
use petgraph::graph::UnGraph;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Canonical key of the contact graph at `cutoff` times the structure's own
/// nearest-neighbour distance.
///
/// Equal keys mean isomorphic bond networks. Distinct keys mean provably
/// different arrangements, which a distance threshold can never say.
pub fn contact_key(x: ArrayView1<f64>, cutoff: f64) -> u64 {
    let n = x.len() / 3;
    if n < 2 {
        return 0;
    }
    // The structure's own scale: the median nearest-neighbour distance.
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
    let scale = nn[n / 2];
    let reach = cutoff * scale;
    let mut g: UnGraph<(), ()> = UnGraph::default();
    let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            let d2: f64 = (0..3)
                .map(|k| {
                    let d = x[3 * i + k] - x[3 * j + k];
                    d * d
                })
                .sum();
            if d2.sqrt() < reach {
                g.add_edge(nodes[i], nodes[j], ());
            }
        }
    }
    use nauty_pet::canon::IntoCanon;
    let canon = g.into_canon();
    // The canonical form is hashed by its sorted edge list, which is a pure
    // function of the isomorphism class once the labelling is canonical.
    let mut edges: Vec<(usize, usize)> = canon
        .edge_indices()
        .map(|e| {
            let (a, b) = canon.edge_endpoints(e).expect("edge exists");
            let (a, b) = (a.index(), b.index());
            if a <= b { (a, b) } else { (b, a) }
        })
        .collect();
    edges.sort_unstable();
    let mut h = DefaultHasher::new();
    canon.node_count().hash(&mut h);
    edges.hash(&mut h);
    h.finish()
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

    /// A permutation of the points is the same arrangement, and the key has to
    /// say so exactly, which is what canonical labelling buys over any
    /// distance in coordinate space.
    #[test]
    fn a_permuted_structure_keeps_its_key() {
        let x = ico13();
        let k1 = contact_key(x.view(), 1.35);
        let mut y = Array1::zeros(x.len());
        let perm = [5usize, 2, 9, 0, 12, 7, 1, 11, 3, 8, 10, 4, 6];
        for (a, b) in perm.iter().enumerate() {
            for k in 0..3 {
                y[3 * a + k] = x[3 * b + k];
            }
        }
        assert_eq!(k1, contact_key(y.view(), 1.35));
    }

    /// A small distortion keeps the bond network and so the key; deleting the
    /// centre changes the network and must change it.
    #[test]
    fn distortion_keeps_and_surgery_changes() {
        let x = ico13();
        let k1 = contact_key(x.view(), 1.35);
        let mut y = x.clone();
        for v in y.iter_mut() {
            *v *= 1.02;
        }
        assert_eq!(k1, contact_key(y.view(), 1.35), "uniform scaling changed the key");
        let z = Array1::from_iter(x.iter().skip(3).cloned());
        assert_ne!(k1, contact_key(z.view(), 1.35), "removing the centre kept the key");
    }
}
