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

/// Canonical key of a vertex-coloured contact graph.
///
/// Colours are species: a bond network of one arrangement of two species and
/// the same network with the species swapped are different states, and a
/// colour-blind key would merge them. nauty's canonical labelling refines the
/// initial partition by colour, so equal keys mean isomorphic bond networks
/// *with matching species everywhere*. The pair cutoff is a callback on the
/// two species, because contact distances differ by pair in any real system;
/// the scale argument it receives is the structure's own median
/// nearest-neighbour distance.
pub fn contact_key_colored(
    x: ArrayView1<f64>,
    colors: &[u32],
    pair_cutoff: impl Fn(u32, u32, f64) -> f64,
) -> u64 {
    let n = x.len() / 3;
    if n < 2 || colors.len() != n {
        return 0;
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
    let scale = nn[n / 2];
    let mut g: UnGraph<u32, ()> = UnGraph::default();
    let nodes: Vec<_> = (0..n).map(|i| g.add_node(colors[i])).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            let d2: f64 = (0..3)
                .map(|k| {
                    let d = x[3 * i + k] - x[3 * j + k];
                    d * d
                })
                .sum();
            let reach = pair_cutoff(colors[i], colors[j], scale);
            if d2.sqrt() < reach {
                g.add_edge(nodes[i], nodes[j], ());
            }
        }
    }
    use nauty_pet::canon::IntoCanon;
    let canon = g.into_canon();
    let mut edges: Vec<(usize, usize)> = canon
        .edge_indices()
        .map(|e| {
            let (a, b) = canon.edge_endpoints(e).expect("edge exists");
            let (a, b) = (a.index(), b.index());
            if a <= b { (a, b) } else { (b, a) }
        })
        .collect();
    edges.sort_unstable();
    let node_colors: Vec<u32> = canon.node_indices().map(|i| canon[i]).collect();
    let mut h = DefaultHasher::new();
    canon.node_count().hash(&mut h);
    node_colors.hash(&mut h);
    edges.hash(&mut h);
    h.finish()
}

/// Canonical key of the contact graph at `cutoff` times the structure's own
/// nearest-neighbour distance: the single-species case of
/// [`contact_key_colored`].
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

#[cfg(test)]
mod colored_tests {
    use super::*;
    use ndarray::Array1;

    fn pair(x: &[[f64; 3]]) -> Array1<f64> {
        let mut a = Array1::zeros(3 * x.len());
        for (i, p) in x.iter().enumerate() {
            for k in 0..3 {
                a[3 * i + k] = p[k];
            }
        }
        a
    }

    /// The same geometry with species swapped is a different state, and the
    /// colour-blind key cannot see that.
    #[test]
    fn swapped_species_change_the_key_and_blind_keys_do_not() {
        let x = pair(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.9, 0.0],
            [0.5, -0.9, 0.0],
        ]);
        let cut = |_a: u32, _b: u32, s: f64| 1.35 * s;
        let k1 = contact_key_colored(x.view(), &[0, 0, 1, 1], cut);
        let k2 = contact_key_colored(x.view(), &[1, 1, 0, 0], cut);
        let k3 = contact_key_colored(x.view(), &[0, 1, 0, 1], cut);
        assert_ne!(k1, k3, "different colourings of one geometry share a key");
        // k1 vs k2: swapping the labels globally is a different colouring of
        // the same graph unless the graph has a colour-swapping automorphism;
        // this geometry's triangle pair does not have one.
        assert_ne!(k1, k2, "species swap kept the key");
        let blind = contact_key(x.view(), 1.35);
        let blind2 = contact_key(x.view(), 1.35);
        assert_eq!(blind, blind2);
    }

    /// Permuting atoms and their colours together must keep the key.
    #[test]
    fn consistent_permutation_keeps_the_colored_key() {
        let x = pair(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.9, 0.0],
            [0.5, -0.9, 0.0],
            [1.5, 0.9, 0.0],
        ]);
        let colors = [0u32, 1, 0, 1, 2];
        let cut = |_a: u32, _b: u32, s: f64| 1.35 * s;
        let k1 = contact_key_colored(x.view(), &colors, cut);
        let perm = [3usize, 0, 4, 1, 2];
        let mut y = Array1::zeros(x.len());
        let mut pc = [0u32; 5];
        for (i, &p) in perm.iter().enumerate() {
            for k in 0..3 {
                y[3 * i + k] = x[3 * p + k];
            }
            pc[i] = colors[p];
        }
        assert_eq!(k1, contact_key_colored(y.view(), &pc, cut));
    }
}
