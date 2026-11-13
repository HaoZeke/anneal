//! Changing morphology by twinning, at the cost of one boundary layer.
//!
//! The move set has had two kinds of proposal and neither crosses a funnel.
//! A displacement moves points and lets the quench find a nearby minimum, so
//! the reachable set is whatever a displacement reaches, and on 98 points that
//! does not include the tetrahedral funnel from anywhere in the icosahedral
//! one. Rebuilding the structure from a local order does cross funnels, and was
//! measured at 0 of 34 against a control at 2 of 8, because a rebuilt candidate
//! shares nothing with the incumbent and quenches far above it: the chain
//! rejects it, and the full quench it forced is paid anyway.
//!
//! What sits between them is the operation the packings themselves differ by.
//! Face-centred cubic, hexagonal, decahedral and icosahedral packings are the
//! same close packing with different stacking, and they are related by
//! reflections in their dense planes: a decahedron is five tetrahedra sharing
//! twin boundaries, and an icosahedron is twenty. So the move that turns one
//! morphology into another is not a rebuild, it is a twin.
//!
//! # Why this preserves the energy a rebuild throws away
//!
//! Reflecting the points on one side of a dense plane leaves every neighbour
//! relation on that side intact, and every relation on the other side intact,
//! and changes only the contacts across the plane. The cost is one boundary
//! layer rather than the whole structure, which is what makes the proposal
//! survive an acceptance test that a rebuilt candidate cannot.
//!
//! # Nothing here is specific to one potential
//!
//! A dense plane is found by projecting the points onto a candidate normal and
//! looking for a projection where many of them coincide. The normals come from
//! the structure's own neighbour directions. Both are properties of a point
//! set, so the move applies wherever a packing does, and it carries no
//! knowledge of what is being optimised.

use ndarray::{Array1, ArrayView1};
use rand::Rng;

/// A dense plane of the structure, in the frame of its centroid.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    /// Unit normal.
    pub normal: [f64; 3],
    /// Signed distance from the centroid along the normal.
    pub offset: f64,
    /// Points lying in the plane, within the layer tolerance.
    pub population: usize,
}

/// How the far half is produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Replace the far half by the mirror image of the near half.
    ///
    /// The construction that gives a twin its registry. Moving the far half by
    /// the twin law instead, which is what a two-fold rotation about the
    /// normal does for a close packing, is crystallographically the right
    /// operation and produces the wrong structure here: the two halves keep
    /// their own in-plane origins, so neighbours across the boundary land at
    /// distances that are not the packing's, and Lennard-Jones charges for
    /// each one. Measured, that proposal was accepted on 52 of 2228 draws
    /// against 0.63 for a surface move, and scored 1 of 6 against a control at
    /// 1 of 6.
    ///
    /// Mirroring the near half onto the far side cannot have that fault: every
    /// distance on the far side is a distance that already existed on the
    /// near side, and the boundary is the plane both halves were built
    /// against.
    Reflect,
    /// Rotate about the normal by a fifth of a turn.
    ///
    /// The disclination that turns close packing into a five-fold axis, which
    /// is what a decahedron and an icosahedron have and a lattice cannot.
    Rotate,
}

fn centroid(x: ArrayView1<f64>, n: usize) -> [f64; 3] {
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for v in c.iter_mut() {
        *v /= n.max(1) as f64;
    }
    c
}

/// Mean nearest-neighbour distance, the structure's own length scale.
pub fn spacing(x: ArrayView1<f64>, n: usize) -> f64 {
    if n < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = (0..3)
                .map(|k| {
                    let v = x[3 * i + k] - x[3 * j + k];
                    v * v
                })
                .sum();
            best = best.min(d);
        }
        total += best.sqrt();
    }
    total / n as f64
}

/// Median nearest-neighbour distance.
///
/// A stretched contact pulls the mean, and a partial dislocation is a
/// fraction of a typical bond, so the slip length is read from the median.
fn median_spacing(x: ArrayView1<f64>, n: usize) -> f64 {
    if n < 2 {
        return 1.0;
    }
    let mut nearest = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut d = 0.0;
            for k in 0..3 {
                let delta = x[3 * i + k] - x[3 * j + k];
                d += delta * delta;
            }
            best = best.min(d);
        }
        nearest.push(best.sqrt());
    }
    nearest.sort_by(|a, b| a.total_cmp(b));
    nearest[n / 2]
}

/// Directions worth testing as plane normals, taken from the structure.
///
/// The normals of a close packing's dense planes are the directions in which
/// the points stack, and those are visible in the neighbour vectors: a bond
/// direction that many pairs share is a lattice direction, and the planes are
/// normal to it. Deduplicated up to sign, since a plane and its opposite are
/// the same plane.
fn candidate_normals(x: ArrayView1<f64>, n: usize, cut: f64) -> Vec<[f64; 3]> {
    let mut out: Vec<[f64; 3]> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = [
                x[3 * j] - x[3 * i],
                x[3 * j + 1] - x[3 * i + 1],
                x[3 * j + 2] - x[3 * i + 2],
            ];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r > cut || r < 1e-9 {
                continue;
            }
            let u = [d[0] / r, d[1] / r, d[2] / r];
            // Same direction up to sign is the same plane normal.
            if !out
                .iter()
                .any(|v| (v[0] * u[0] + v[1] * u[1] + v[2] * u[2]).abs() > 0.995)
            {
                out.push(u);
            }
        }
    }
    out
}

/// Dense planes of `x`, most populated first.
///
/// A plane is dense when many points project to the same distance along its
/// normal, which is what a lattice plane is. The layer tolerance is a fraction
/// of the spacing, so the test is scale free.
pub fn dense_planes(x: ArrayView1<f64>, n: usize, layer: f64) -> Vec<Plane> {
    if n < 4 {
        return Vec::new();
    }
    let c = centroid(x, n);
    let s = spacing(x, n);
    let tol = layer * s;
    let mut out = Vec::new();
    for normal in candidate_normals(x, n, 1.35 * s) {
        let mut proj: Vec<f64> = (0..n)
            .map(|i| {
                (0..3)
                    .map(|k| (x[3 * i + k] - c[k]) * normal[k])
                    .sum::<f64>()
            })
            .collect();
        proj.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Layers, as runs of projections within the tolerance of each other.
        let mut start = 0usize;
        for k in 1..=proj.len() {
            if k == proj.len() || proj[k] - proj[start] > tol {
                let population = k - start;
                if population >= 3 {
                    let offset = proj[start..k].iter().sum::<f64>() / population as f64;
                    out.push(Plane {
                        normal,
                        offset,
                        population,
                    });
                }
                start = k;
            }
        }
    }
    out.sort_by_key(|item| std::cmp::Reverse(item.population));
    out
}

/// Twins `x` across `plane`.
///
/// Points on the far side of the plane are mapped onto the other side's
/// stacking; points within one layer of the plane are left where they are,
/// since they belong to the boundary itself and moving them tears it.
///
/// The result is a structure whose local order is unchanged everywhere except
/// across the boundary, which is the property this move exists for.
pub fn twin(x: ArrayView1<f64>, n: usize, plane: &Plane, mode: Mode, layer: f64) -> Array1<f64> {
    let c = centroid(x, n);
    let s = spacing(x, n);
    let tol = layer * s;
    let m = plane.normal;
    let mut y = x.to_owned();

    if mode == Mode::Reflect {
        // The near half, its mirror, and the far half's count. The far points
        // are replaced by the mirror images nearest the boundary, so the
        // structure keeps its size and the far side inherits a packing that
        // already exists rather than one assembled at the boundary.
        let signed: Vec<f64> = (0..n)
            .map(|i| (0..3).map(|k| (x[3 * i + k] - c[k]) * m[k]).sum::<f64>() - plane.offset)
            .collect();
        let above: Vec<usize> = (0..n).filter(|&i| signed[i] > tol).collect();
        let below: Vec<usize> = (0..n).filter(|&i| signed[i] < -tol).collect();
        if above.is_empty() || below.is_empty() {
            return y;
        }
        // The fuller side is the source and the thinner side is replaced.
        //
        // Requiring the near side to be the larger one meant returning the
        // input whenever the plane cut off-centre, which is most planes, and
        // the move did nothing at all: the test that asks whether it changed
        // the structure is what caught that.
        let (src_side, dst_side) = if below.len() >= above.len() {
            (&below, &above)
        } else {
            (&above, &below)
        };
        // Mirror images of the source side, closest to the boundary first,
        // since those are the ones a cluster of this size actually holds.
        let mut order: Vec<usize> = src_side.clone();
        order.sort_by(|&a, &b| {
            signed[a]
                .abs()
                .partial_cmp(&signed[b].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (slot, &src) in dst_side.iter().zip(order.iter()) {
            for k in 0..3 {
                // Reflection through the plane: the in-plane part is kept and
                // the component along the normal changes sign about the plane.
                let v = x[3 * src + k] - c[k];
                let along = signed[src] + plane.offset;
                y[3 * slot + k] = c[k] + v - 2.0 * (along - plane.offset) * m[k];
            }
        }
        return y;
    }

    for i in 0..n {
        let v = [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]];
        let d: f64 = (0..3).map(|k| v[k] * m[k]).sum::<f64>() - plane.offset;
        if d <= tol {
            continue;
        }
        // Split into the component along the normal, which the operation
        // leaves alone, and the component in the plane, which it acts on.
        let along: f64 = (0..3).map(|k| v[k] * m[k]).sum();
        let inplane = [
            v[0] - along * m[0],
            v[1] - along * m[1],
            v[2] - along * m[2],
        ];
        let mapped = match mode {
            // A mirror twin acts on the in-plane part by inverting it about
            // the plane's own axis: the stacking of the far side is exchanged
            // while its distance from the boundary is kept.
            Mode::Reflect => [-inplane[0], -inplane[1], -inplane[2]],
            Mode::Rotate => {
                let a = std::f64::consts::TAU / 5.0;
                let (sa, ca) = a.sin_cos();
                // Rodrigues about the normal, for a vector already in the
                // plane, which drops the term along the axis.
                let cross = [
                    m[1] * inplane[2] - m[2] * inplane[1],
                    m[2] * inplane[0] - m[0] * inplane[2],
                    m[0] * inplane[1] - m[1] * inplane[0],
                ];
                [
                    inplane[0] * ca + cross[0] * sa,
                    inplane[1] * ca + cross[1] * sa,
                    inplane[2] * ca + cross[2] * sa,
                ]
            }
        };
        for k in 0..3 {
            y[3 * i + k] = c[k] + mapped[k] + along * m[k];
        }
    }
    y
}

/// In-plane slides of the one close-packed sheet that stands clear of the rest.
///
/// A sheet is the set of atoms within a tenth of a bond of one triangular
/// face. It is dominant when the empty gap beyond it is at least twice the
/// gap beyond every other sheet. The move slides a diamond of that sheet,
/// two triangles sharing an edge, by `spacing / √2` along the in-plane
/// normal of the shared edge, and slides one neighbouring atom of the edge
/// by half a spacing along the edge. Both directions and both signs are
/// returned. Nothing here depends on how many atoms the cluster has.
pub fn dominant_layer_slides(x: ArrayView1<f64>) -> Vec<Array1<f64>> {
    isolated_layer_slides(x, true)
}

/// Edge slides of the most isolated sheet, without the factor-of-two test.
///
/// A slip leaves a packing whose two flattest sheets can sit within numerical
/// noise of each other. The sheet that is still the most isolated is the one
/// the edge slide has to finish. Callers that have not just opened a fault
/// use [`dominant_layer_slides`], which stays quiet when the faces tie.
pub fn most_isolated_layer_slides(x: ArrayView1<f64>) -> Vec<Array1<f64>> {
    isolated_layer_slides(x, false)
}

fn isolated_layer_slides(x: ArrayView1<f64>, require_dominant: bool) -> Vec<Array1<f64>> {
    let n = x.len() / 3;
    if n < 8 {
        return Vec::new();
    }
    let nn = spacing(x, n);
    if !(nn.is_finite() && nn > 0.0) {
        return Vec::new();
    }
    let cut2 = (1.25 * nn) * (1.25 * nn);
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = 0.0;
            for k in 0..3 {
                let delta = x[3 * i + k] - x[3 * j + k];
                d += delta * delta;
            }
            if d < cut2 && d > 1e-12 {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    let mut layers: Vec<(f64, Vec<usize>, [f64; 3], [f64; 3])> = Vec::new();
    let mut seen: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        for &j in &adj[i] {
            if j < i {
                continue;
            }
            for &k in &adj[i] {
                if k <= j || !adj[j].contains(&k) {
                    continue;
                }
                let e1 = [
                    x[3 * j] - x[3 * i],
                    x[3 * j + 1] - x[3 * i + 1],
                    x[3 * j + 2] - x[3 * i + 2],
                ];
                let e2 = [
                    x[3 * k] - x[3 * i],
                    x[3 * k + 1] - x[3 * i + 1],
                    x[3 * k + 2] - x[3 * i + 2],
                ];
                let mut normal = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let nl = (normal.iter().map(|v| v * v).sum::<f64>()).sqrt();
                if nl < 1e-8 {
                    continue;
                }
                for value in &mut normal {
                    *value /= nl;
                }
                let origin = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
                let mut signed = vec![0.0; n];
                for a in 0..n {
                    for t in 0..3 {
                        signed[a] += (x[3 * a + t] - origin[t]) * normal[t];
                    }
                }
                let mut layer: Vec<usize> = (0..n)
                    .filter(|&a| signed[a].abs() < 0.1 * nn)
                    .collect();
                if layer.len() < 6 || layer.len() * 2 >= n {
                    continue;
                }
                layer.sort_unstable();
                if seen.iter().any(|prev| prev == &layer) {
                    continue;
                }
                let thick = layer.iter().map(|&a| signed[a].abs()).fold(0.0, f64::max);
                let gap = (0..n)
                    .filter(|a| layer.binary_search(a).is_err())
                    .map(|a| signed[a].abs())
                    .fold(f64::INFINITY, f64::min);
                if !(gap.is_finite() && thick.is_finite()) {
                    continue;
                }
                let ratio = gap / thick.max(1e-9);
                seen.push(layer.clone());
                layers.push((ratio, layer, normal, origin));
            }
        }
    }
    if layers.is_empty() {
        return Vec::new();
    }
    layers.sort_by(|a, b| b.0.total_cmp(&a.0));
    let dominant = if layers.len() == 1 {
        layers[0].0 >= 2.0
    } else {
        layers[0].0 >= 2.0 * layers[1].0
    };
    if require_dominant && !dominant {
        return Vec::new();
    }
    let (_, layer, normal, _origin) = layers.remove(0);
    let in_layer = |a: usize| layer.binary_search(&a).is_ok();
    let mut slides = Vec::new();
    for ai in 0..layer.len() {
        let a = layer[ai];
        for &b in &adj[a] {
            if b < a || !in_layer(b) {
                continue;
            }
            let wings: Vec<usize> = adj[a]
                .iter()
                .copied()
                .filter(|&w| w != b && in_layer(w) && adj[b].contains(&w))
                .collect();
            for p in 0..wings.len() {
                for q in (p + 1)..wings.len() {
                    let atoms = [a, b, wings[p], wings[q]];
                    let mut edge_u = [0.0; 3];
                    for t in 0..3 {
                        edge_u[t] = x[3 * b + t] - x[3 * a + t];
                    }
                    let el = (edge_u.iter().map(|v| v * v).sum::<f64>()).sqrt();
                    if el < 1e-12 {
                        continue;
                    }
                    for value in &mut edge_u {
                        *value /= el;
                    }
                    let mut perp = [
                        normal[1] * edge_u[2] - normal[2] * edge_u[1],
                        normal[2] * edge_u[0] - normal[0] * edge_u[2],
                        normal[0] * edge_u[1] - normal[1] * edge_u[0],
                    ];
                    let pl = (perp.iter().map(|v| v * v).sum::<f64>()).sqrt();
                    if pl < 1e-12 {
                        continue;
                    }
                    for value in &mut perp {
                        *value /= pl;
                    }
                    let mut pendants = Vec::new();
                    for &vertex in &[a, b] {
                        for &pendant in &adj[vertex] {
                            if !atoms.contains(&pendant) && in_layer(pendant) {
                                pendants.push(pendant);
                            }
                        }
                    }
                    pendants.sort_unstable();
                    pendants.dedup();
                    for pendant in pendants {
                        for s4 in [-1.0, 1.0] {
                            for s5 in [-1.0, 1.0] {
                                let mut y = x.to_owned();
                                for &atom in &atoms {
                                    for t in 0..3 {
                                        y[3 * atom + t] +=
                                            s4 * perp[t] * nn / std::f64::consts::SQRT_2;
                                    }
                                }
                                for t in 0..3 {
                                    y[3 * pendant + t] += s5 * edge_u[t] * nn / 2.0;
                                }
                                slides.push(y);
                            }
                        }
                    }
                }
            }
        }
    }
    slides
}

/// In-plane slips of a stacking fault, once one sheet stands clear of the rest.
///
/// The diamond slide sends a patch along an edge. A fault leaves in the
/// plane, along a `<112>` direction. For every small sheet, and for a
/// connected patch of four atoms in it together with one neighbour, both
/// signs of all six in-plane directions are tried: the Shockley length
/// `spacing / √3` (the `a/√6` partial) and one full spacing. The list is
/// empty unless one sheet is an order of magnitude more isolated than the
/// next, so a tie and a near-tie produce nothing.
pub fn stacking_fault_slips(x: ArrayView1<f64>) -> Vec<Array1<f64>> {
    let n = x.len() / 3;
    if n < 8 {
        return Vec::new();
    }
    let nn = median_spacing(x, n);
    if !(nn.is_finite() && nn > 0.0) {
        return Vec::new();
    }
    let cut2 = (1.25 * nn) * (1.25 * nn);
    // A surface atom keeps at most two thirds of the twelve neighbours of a
    // close-packed interior atom. The interior count is the lattice's, not a
    // size of this cluster.
    let surface = 12 * 2 / 3;
    let mut adj = vec![Vec::new(); n];
    let mut coord = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = 0.0;
            for k in 0..3 {
                let delta = x[3 * i + k] - x[3 * j + k];
                d += delta * delta;
            }
            if d < cut2 && d > 1e-12 {
                adj[i].push(j);
                adj[j].push(i);
                coord[i] += 1;
                coord[j] += 1;
            }
        }
    }
    let mut sheets: Vec<(Vec<usize>, [f64; 3], f64)> = Vec::new();
    let mut seen: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        for &j in &adj[i] {
            if j < i {
                continue;
            }
            for &k in &adj[i] {
                if k <= j || !adj[j].contains(&k) {
                    continue;
                }
                let e1 = [
                    x[3 * j] - x[3 * i],
                    x[3 * j + 1] - x[3 * i + 1],
                    x[3 * j + 2] - x[3 * i + 2],
                ];
                let e2 = [
                    x[3 * k] - x[3 * i],
                    x[3 * k + 1] - x[3 * i + 1],
                    x[3 * k + 2] - x[3 * i + 2],
                ];
                let mut normal = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let nl = (normal.iter().map(|v| v * v).sum::<f64>()).sqrt();
                if nl < 1e-8 {
                    continue;
                }
                for value in &mut normal {
                    *value /= nl;
                }
                let mut layer = Vec::new();
                for a in 0..n {
                    let mut s = 0.0;
                    for t in 0..3 {
                        s += (x[3 * a + t] - x[3 * i + t]) * normal[t];
                    }
                    if s.abs() < 0.1 * nn {
                        layer.push(a);
                    }
                }
                if layer.len() < 6 || layer.len() * 2 >= n {
                    continue;
                }
                layer.sort_unstable();
                if seen.iter().any(|prev| prev == &layer) {
                    continue;
                }
                let mut thick = 0.0_f64;
                for &a in &layer {
                    let mut s = 0.0;
                    for t in 0..3 {
                        s += (x[3 * a + t] - x[3 * i + t]) * normal[t];
                    }
                    thick = thick.max(s.abs());
                }
                let mut gap = f64::INFINITY;
                for a in 0..n {
                    if layer.binary_search(&a).is_ok() {
                        continue;
                    }
                    let mut s = 0.0;
                    for t in 0..3 {
                        s += (x[3 * a + t] - x[3 * i + t]) * normal[t];
                    }
                    gap = gap.min(s.abs());
                }
                if !(gap.is_finite() && thick.is_finite()) {
                    continue;
                }
                let ratio = gap / thick.max(1e-9);
                seen.push(layer.clone());
                sheets.push((layer, normal, ratio));
            }
        }
    }
    if sheets.is_empty() {
        return Vec::new();
    }
    let mut ratios: Vec<f64> = sheets.iter().map(|sheet| sheet.2).collect();
    ratios.sort_by(|a, b| b.total_cmp(a));
    // The edge slide fires when one sheet is merely twice as isolated as
    // the next. This enumeration quenches every patch, so it waits until
    // the sheet is an order of magnitude clearer than every other: a flat
    // fault, not two faces that both stand proud. A tie, and a near-tie,
    // return nothing.
    let fault = if ratios.len() == 1 {
        ratios[0] >= 10.0
    } else {
        ratios[0] >= 10.0 * ratios[1]
    };
    if !fault {
        return Vec::new();
    }
    let mut slips = Vec::new();
    // A local fault is a patch in a sheet only a few atoms larger than the
    // patch. A bigger sheet is a face; it is slipped as a whole below, and
    // the diamond slide already moves its edges.
    for (layer, normal, _) in sheets
        .iter()
        .filter(|(layer, _, _)| (6..10).contains(&layer.len()))
    {
        let m = layer.len();
        for a in 0..m {
            for b in (a + 1)..m {
                for c in (b + 1)..m {
                    for d in (c + 1)..m {
                        let nodes = [layer[a], layer[b], layer[c], layer[d]];
                        if nodes.iter().any(|&atom| coord[atom] > surface) {
                            continue;
                        }
                        let mut got = vec![nodes[0]];
                        let mut stack = vec![nodes[0]];
                        while let Some(u) = stack.pop() {
                            for &v in &nodes {
                                if !got.contains(&v) && adj[u].contains(&v) {
                                    got.push(v);
                                    stack.push(v);
                                }
                            }
                        }
                        if got.len() != 4 {
                            continue;
                        }
                        let mut bond = None;
                        'bond: for &p in &nodes {
                            for &q in &adj[p] {
                                if got.contains(&q) {
                                    bond = Some((p, q));
                                    break 'bond;
                                }
                            }
                        }
                        let Some((p, q)) = bond else { continue };
                        let dirs = in_plane_directions(x, p, q, *normal);
                        let mut offs = Vec::new();
                        for &atom in &nodes {
                            for &v in &adj[atom] {
                                if !got.contains(&v) && coord[v] <= surface {
                                    offs.push(v);
                                }
                            }
                        }
                        offs.sort_unstable();
                        offs.dedup();
                        for off in offs {
                            let who = [nodes[0], nodes[1], nodes[2], nodes[3], off];
                            push_translations(&mut slips, x, &who, &dirs, nn);
                        }
                    }
                }
            }
        }
    }
    for (layer, normal, _) in &sheets {
        let mut bond = None;
        'face: for &p in layer {
            for &q in &adj[p] {
                if layer.binary_search(&q).is_ok() {
                    bond = Some((p, q));
                    break 'face;
                }
            }
        }
        let Some((p, q)) = bond else { continue };
        let dirs = in_plane_directions(x, p, q, *normal);
        push_translations(&mut slips, x, layer, &dirs, nn);
    }
    slips
}

/// Six in-plane directions: a bond and the two turns of 60°, then the three
/// turns of 30° that lie along `<112>`.
fn in_plane_directions(x: ArrayView1<f64>, p: usize, q: usize, normal: [f64; 3]) -> Vec<[f64; 3]> {
    let mut axis = [0.0; 3];
    for t in 0..3 {
        axis[t] = x[3 * q + t] - x[3 * p + t];
    }
    let al = (axis.iter().map(|v| v * v).sum::<f64>())
        .sqrt()
        .max(1e-12);
    for value in &mut axis {
        *value /= al;
    }
    let mut dirs = Vec::with_capacity(6);
    for deg in [0.0_f64, 60.0, 120.0, 30.0, 90.0, 150.0] {
        let (s, c) = deg.to_radians().sin_cos();
        let mut rotated = [0.0; 3];
        for t in 0..3 {
            let i1 = (t + 1) % 3;
            let i2 = (t + 2) % 3;
            let cross = normal[i1] * axis[i2] - normal[i2] * axis[i1];
            rotated[t] = axis[t] * c + cross * s;
        }
        let mut along = 0.0;
        for t in 0..3 {
            along += rotated[t] * normal[t];
        }
        for t in 0..3 {
            rotated[t] -= along * normal[t];
        }
        let rl = (rotated.iter().map(|v| v * v).sum::<f64>())
            .sqrt()
            .max(1e-12);
        for value in &mut rotated {
            *value /= rl;
        }
        dirs.push(rotated);
    }
    dirs
}

/// Both signs, full spacing first and the Shockley length `spacing / √3` after.
fn push_translations(
    slips: &mut Vec<Array1<f64>>,
    x: ArrayView1<f64>,
    who: &[usize],
    dirs: &[[f64; 3]],
    nn: f64,
) {
    for dir in dirs {
        for &sign in &[-1.0, 1.0] {
            for &length in &[nn, nn / 3.0_f64.sqrt()] {
                let mut y = x.to_owned();
                for &atom in who {
                    for t in 0..3 {
                        y[3 * atom + t] += sign * dir[t] * length;
                    }
                }
                slips.push(y);
            }
        }
    }
}

/// Twins `x` across one of its dense planes, chosen at random among the
/// densest.
///
/// The densest plane is the one whose boundary costs least, so the choice is
/// restricted to the top few rather than taken uniformly, and randomised
/// within them so repeated draws are different proposals.
pub fn propose<R: Rng + ?Sized>(x: ArrayView1<f64>, n: usize, rng: &mut R) -> Array1<f64> {
    let planes = dense_planes(x, n, 0.25);
    if planes.is_empty() {
        return x.to_owned();
    }
    let top = planes.len().min(8);
    let p = planes[rng.random_range(0..top)];
    let mode = if rng.random::<f64>() < 0.5 {
        Mode::Reflect
    } else {
        Mode::Rotate
    };
    twin(x.view(), n, &p, mode, 0.25)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice;
    use crate::structure::Template;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn packed(n: usize) -> Array1<f64> {
        let sites = lattice::grow(&Template::FaceCentredCubic.points(), n);
        let mut x = Array1::zeros(3 * n);
        for (i, s) in sites.iter().take(n).enumerate() {
            for k in 0..3 {
                x[3 * i + k] = 1.12 * s[k];
            }
        }
        x
    }

    fn coordination(x: ArrayView1<f64>, n: usize) -> Vec<usize> {
        let s = spacing(x, n);
        (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| {
                        if i == j {
                            return false;
                        }
                        let d: f64 = (0..3)
                            .map(|k| {
                                let v = x[3 * i + k] - x[3 * j + k];
                                v * v
                            })
                            .sum::<f64>()
                            .sqrt();
                        d < 1.2 * s
                    })
                    .count()
            })
            .collect()
    }

    #[test]
    fn an_unlayered_chain_has_no_sheet_to_slide() {
        let mut x = Array1::zeros(24);
        for i in 0..8 {
            x[3 * i] = i as f64;
        }
        assert!(dominant_layer_slides(x.view()).is_empty());
        assert!(stacking_fault_slips(x.view()).is_empty());
    }

    /// A close packing has dense planes, and they have to be found, or the
    /// move has nothing to act on.
    #[test]
    fn a_close_packing_has_dense_planes() {
        let x = packed(80);
        let planes = dense_planes(x.view(), 80, 0.25);
        assert!(!planes.is_empty(), "no dense plane in a close packing");
        assert!(
            planes[0].population >= 7,
            "densest plane holds only {} points",
            planes[0].population
        );
    }

    /// The claim the move rests on: twinning costs a boundary, not a
    /// structure. Measured as the number of points whose coordination changed,
    /// which has to be a minority.
    #[test]
    fn twinning_changes_only_a_boundary() {
        let n = 80;
        let x = packed(n);
        let planes = dense_planes(x.view(), n, 0.25);
        let before = coordination(x.view(), n);
        let y = twin(x.view(), n, &planes[0], Mode::Reflect, 0.25);
        let after = coordination(y.view(), n);
        let changed = (0..n).filter(|&i| before[i] != after[i]).count();
        assert!(
            changed * 2 < n,
            "{changed} of {n} points changed coordination, which is a rebuild rather than a twin"
        );
    }

    /// And it has to actually change the structure, or it is an identity move
    /// dressed up as a mechanism.
    #[test]
    fn twinning_produces_a_different_structure() {
        let n = 80;
        let x = packed(n);
        let planes = dense_planes(x.view(), n, 0.25);
        let y = twin(x.view(), n, &planes[0], Mode::Reflect, 0.25);
        let moved = (0..n)
            .filter(|&i| {
                (0..3)
                    .map(|k| (x[3 * i + k] - y[3 * i + k]).abs())
                    .fold(0.0_f64, f64::max)
                    > 1e-6
            })
            .count();
        assert!(moved > 0, "the twin moved nothing");
        assert!(moved < n, "the twin moved every point, so it is not a twin");
    }

    /// No overlaps, or the potential overflows and the relaxation counts a
    /// failure instead of work. This is the failure mode a naive reflection
    /// has, and the reason points inside the boundary layer are left alone.
    #[test]
    fn a_twin_produces_no_coincident_points() {
        let mut r = StdRng::seed_from_u64(4);
        for n in [38, 55, 98] {
            let x = packed(n);
            for _ in 0..12 {
                let y = propose(x.view(), n, &mut r);
                let s = spacing(y.view(), n);
                for i in 0..n {
                    for j in (i + 1)..n {
                        let d: f64 = (0..3)
                            .map(|k| {
                                let v = y[3 * i + k] - y[3 * j + k];
                                v * v
                            })
                            .sum::<f64>()
                            .sqrt();
                        assert!(
                            d > 0.35 * s,
                            "n={n}: points {i},{j} at {d} with spacing {s}"
                        );
                    }
                }
            }
        }
    }

    /// The fivefold rotation is what a lattice cannot do, so it has to produce
    /// something the mirror does not.
    #[test]
    fn the_rotation_and_the_reflection_differ() {
        let n = 80;
        let x = packed(n);
        let planes = dense_planes(x.view(), n, 0.25);
        let a = twin(x.view(), n, &planes[0], Mode::Reflect, 0.25);
        let b = twin(x.view(), n, &planes[0], Mode::Rotate, 0.25);
        let diff: f64 = (0..3 * n)
            .map(|i| (a[i] - b[i]) * (a[i] - b[i]))
            .sum::<f64>()
            .sqrt();
        assert!(diff > 1.0, "the two operations differ by only {diff}");
    }

    /// A structure with no dense plane must come back unchanged rather than
    /// mangled.
    #[test]
    fn a_structure_with_no_plane_is_returned_as_is() {
        let mut r = StdRng::seed_from_u64(9);
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let y = propose(x.view(), 2, &mut r);
        for (p, q) in x.iter().zip(y.iter()) {
            assert!((p - q).abs() < 1e-12);
        }
    }
}
