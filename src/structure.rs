//! Naming the local structure a point sits in.
//!
//! Two analyses that OVITO made standard, ported because this crate needs them
//! for its own reasons rather than for pictures.
//!
//! The Lennard-Jones sizes this crate is measured on fail by *morphology*: at
//! 75 points the search settles into an icosahedral funnel and the answer is a
//! Marks decahedron. Every continuous collective variable tried on that
//! distinction was too blunt, the fourth-order bond-order parameter separating
//! the two funnels by 0.023 against a deposition width four times larger. A
//! per-point structural classification is a sharper instrument: an icosahedron
//! and a decahedron differ in what their points *are*, not by a small number.
//!
//! # Common neighbour analysis
//!
//! Honeycutt and Andersen. Each bonded pair is labelled by three integers: how
//! many neighbours the two share, how many bonds are among those shared
//! neighbours, and the longest chain those bonds form. A pair inside an
//! icosahedral shell gives `555`; a face-centred cubic environment gives `421`
//! and hexagonal close packed gives `422`. A decahedron carries a fivefold axis
//! and close-packed facets together, so it shows `555` alongside a substantial
//! `421` population, which an icosahedron does not.
//!
//! # Polyhedral template matching
//!
//! Larsen, Schmidt and Schiøtz, *Robust structural identification via
//! polyhedral template matching*, Modelling Simul. Mater. Sci. Eng. 24, 055007
//! (2016). Common neighbour analysis decides through a bond cutoff, so it
//! degrades when the structure is warm or strained: a bond that falls just
//! outside the cutoff changes the triplet. Template matching instead compares
//! the neighbourhood to ideal templates after scaling and optimal rotation, and
//! reports the residual, so the answer degrades smoothly instead of flipping.
//!
//! What is implemented here is that comparison. The published method finds the
//! correspondence between a neighbourhood and a template through the topology
//! of the convex hull; this enumerates starting correspondences and refines
//! each by iterated closest point, which is a different way of solving the same
//! subproblem and is stated as such rather than claimed to be theirs. The
//! classification and the residual are the parts anything downstream uses.

use ndarray::{Array1, ArrayView1};

/// Unit vector, or `None` when the input has no direction.
fn normalise(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

/// Counts of common-neighbour triplets over the bonded pairs of a structure.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CnaCounts {
    /// `(r, s, t)` keys and how often each occurred.
    pub counts: Vec<((usize, usize, usize), usize)>,
    /// Bonded pairs found.
    pub bonds: usize,
}

impl CnaCounts {
    /// Occurrences of one triplet.
    pub fn get(&self, key: (usize, usize, usize)) -> usize {
        self.counts
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
            .unwrap_or(0)
    }

    /// Share of bonded pairs carrying a triplet.
    pub fn fraction(&self, key: (usize, usize, usize)) -> f64 {
        if self.bonds == 0 {
            return 0.0;
        }
        self.get(key) as f64 / self.bonds as f64
    }
}

/// Adjacency under a distance cutoff.
fn adjacency(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Vec<Vec<bool>> {
    let c2 = cutoff * cutoff;
    let mut adj = vec![vec![false; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            if dx * dx + dy * dy + dz * dz < c2 {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
    }
    adj
}

/// Bonds in the longest continuous chain among a set of common neighbours.
///
/// The third index, and the one that has to be counted the right way. It is the
/// longest *trail*: each bond is used at most once and a point may be passed
/// through more than once. Counting the longest simple path instead forbids
/// closing a ring, and the icosahedral signature is a closed five-membered
/// ring, so a perfect thirteen-point icosahedron reports `(5,5,4)` for its
/// twelve centre-to-surface pairs where the literature reports `555`, and
/// everything keyed on `555` reads zero.
fn longest_trail(sub: &[Vec<bool>]) -> usize {
    let m = sub.len();
    if m == 0 {
        return 0;
    }
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            if sub[i][j] {
                edges.push((i, j));
            }
        }
    }
    if edges.is_empty() {
        return 0;
    }
    let mut incident: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m];
    for (k, &(i, j)) in edges.iter().enumerate() {
        incident[i].push((k, j));
        incident[j].push((k, i));
    }
    // Exhaustive over bonds, which is harmless: the set is the common
    // neighbours of one pair and the bond count stays in single figures.
    let mut best = 0usize;
    for start in 0..m {
        let mut stack: Vec<(usize, u64, usize)> = vec![(start, 0, 0)];
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

/// Common-neighbour triplets over every bonded pair.
///
/// The default cutoff for a Lennard-Jones cluster sits between the first and
/// second neighbour shells, where the radial distribution is near zero, so the
/// bond set is insensitive to its exact value.
pub fn cna(x: ArrayView1<f64>, n: usize, cutoff: f64) -> CnaCounts {
    let adj = adjacency(x, n, cutoff);
    let mut counts: Vec<((usize, usize, usize), usize)> = Vec::new();
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
                    .map(|(a, row)| row.iter().enumerate().filter(|&(b, &v)| v && b > a).count())
                    .sum();
                (r, s, longest_trail(&sub))
            };
            match counts.iter_mut().find(|(k, _)| *k == key) {
                Some((_, v)) => *v += 1,
                None => counts.push((key, 1)),
            }
        }
    }
    CnaCounts { counts, bonds }
}

/// Per-atom fractions of 555 / 421 / 422 among that atom's bonds.
pub fn atom_triplet_fracs(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Vec<[f64; 3]> {
    let adj = adjacency(x, n, cutoff);
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
                .map(|(a, row)| row.iter().enumerate().filter(|&(b, &v)| v && b > a).count())
                .sum();
            let t = longest_trail(&sub);
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

/// A local structure a neighbourhood can be matched to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    /// Twelve neighbours, cuboctahedral.
    FaceCentredCubic,
    /// Twelve neighbours, anticuboctahedral.
    HexagonalClosePacked,
    /// Twelve neighbours, icosahedral.
    Icosahedral,
    /// Six neighbours, octahedral.
    SimpleCubic,
    /// Nothing matched within the residual cutoff.
    Other,
}

impl Template {
    /// Neighbours the template is built from.
    pub fn size(&self) -> usize {
        match self {
            Template::FaceCentredCubic | Template::HexagonalClosePacked | Template::Icosahedral => {
                12
            }
            Template::SimpleCubic => 6,
            Template::Other => 0,
        }
    }

    /// The ideal neighbour positions, scaled so the mean distance is one.
    pub fn points(&self) -> Vec<[f64; 3]> {
        let raw: Vec<[f64; 3]> = match self {
            Template::FaceCentredCubic => vec![
                [0.0, 1.0, 1.0],
                [0.0, -1.0, -1.0],
                [0.0, 1.0, -1.0],
                [0.0, -1.0, 1.0],
                [1.0, 0.0, 1.0],
                [-1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            Template::HexagonalClosePacked => {
                // Six in-plane, three above and three below, the stacking that
                // distinguishes it from the cubic packing.
                let s3 = 3.0_f64.sqrt();
                let a = 2.0_f64.sqrt();
                vec![
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.5, s3 / 2.0, 0.0],
                    [-0.5, -s3 / 2.0, 0.0],
                    [0.5, -s3 / 2.0, 0.0],
                    [-0.5, s3 / 2.0, 0.0],
                    [0.5, s3 / 6.0, a / s3],
                    [-0.5, s3 / 6.0, a / s3],
                    [0.0, -s3 / 3.0, a / s3],
                    [0.5, s3 / 6.0, -a / s3],
                    [-0.5, s3 / 6.0, -a / s3],
                    [0.0, -s3 / 3.0, -a / s3],
                ]
            }
            Template::Icosahedral => {
                let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
                vec![
                    [0.0, 1.0, p],
                    [0.0, 1.0, -p],
                    [0.0, -1.0, p],
                    [0.0, -1.0, -p],
                    [1.0, p, 0.0],
                    [1.0, -p, 0.0],
                    [-1.0, p, 0.0],
                    [-1.0, -p, 0.0],
                    [p, 0.0, 1.0],
                    [-p, 0.0, 1.0],
                    [p, 0.0, -1.0],
                    [-p, 0.0, -1.0],
                ]
            }
            Template::SimpleCubic => vec![
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            Template::Other => return Vec::new(),
        };
        normalise_scale(raw)
    }

    /// The templates a neighbourhood is tried against.
    pub fn all() -> [Template; 4] {
        [
            Template::FaceCentredCubic,
            Template::HexagonalClosePacked,
            Template::Icosahedral,
            Template::SimpleCubic,
        ]
    }
}

/// Scales a point set so its mean distance from the origin is one.
///
/// Scale invariance is what makes a residual comparable between a compressed
/// interior and an expanded surface, and it is why template matching survives
/// strain where a bond cutoff does not.
fn normalise_scale(mut v: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
    if v.is_empty() {
        return v;
    }
    let mean: f64 = v
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
        .sum::<f64>()
        / v.len() as f64;
    if mean > 1e-12 {
        for p in v.iter_mut() {
            for k in 0..3 {
                p[k] /= mean;
            }
        }
    }
    v
}

/// Optimal rotation taking `from` onto `to`, by the Kabsch construction.
fn kabsch(from: &[[f64; 3]], to: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let mut h = [[0.0_f64; 3]; 3];
    for (a, b) in from.iter().zip(to.iter()) {
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += a[i] * b[j];
            }
        }
    }
    // Rotation from the polar factor of H, obtained by a few Newton steps on
    // R <- (R + R^-T) / 2 starting from H. Enough for point sets this small,
    // and it avoids carrying a singular value decomposition for a 3x3.
    let mut r = h;
    for _ in 0..24 {
        let inv_t = match invert_transpose(&r) {
            Some(m) => m,
            None => break,
        };
        let mut next = [[0.0_f64; 3]; 3];
        let mut delta = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                next[i][j] = 0.5 * (r[i][j] + inv_t[i][j]);
                delta += (next[i][j] - r[i][j]).abs();
            }
        }
        r = next;
        if delta < 1e-12 {
            break;
        }
    }
    r
}

/// Inverse transpose of a 3x3, or `None` when it is singular.
fn invert_transpose(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let c = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
        ],
        [
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
        ],
        [
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ];
    // inverse = adj / det, and the transpose of that is c^T / det transposed
    // again, so this returns (M^-1)^T directly.
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = c[j][i] / det;
        }
    }
    Some(out)
}

fn apply(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[1][0] * v[1] + r[2][0] * v[2],
        r[0][1] * v[0] + r[1][1] * v[1] + r[2][1] * v[2],
        r[0][2] * v[0] + r[1][2] * v[1] + r[2][2] * v[2],
    ]
}

/// Rotation taking unit vector `a` onto unit vector `b`.
fn align(a: [f64; 3], b: [f64; 3]) -> [[f64; 3]; 3] {
    let v = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    if c < -1.0 + 1e-9 {
        // Antiparallel: a half turn about any perpendicular axis.
        let perp = if a[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let axis = normalise([
            a[1] * perp[2] - a[2] * perp[1],
            a[2] * perp[0] - a[0] * perp[2],
            a[0] * perp[1] - a[1] * perp[0],
        ])
        .unwrap_or([0.0, 0.0, 1.0]);
        return rotation_about(axis, std::f64::consts::PI);
    }
    // Rodrigues from the cross product, in the form that avoids the angle.
    let k = 1.0 / (1.0 + c);
    [
        [
            c + v[0] * v[0] * k,
            v[0] * v[1] * k - v[2],
            v[0] * v[2] * k + v[1],
        ],
        [
            v[1] * v[0] * k + v[2],
            c + v[1] * v[1] * k,
            v[1] * v[2] * k - v[0],
        ],
        [
            v[2] * v[0] * k - v[1],
            v[2] * v[1] * k + v[0],
            c + v[2] * v[2] * k,
        ],
    ]
}

/// Rotation matrix about a unit axis.
fn rotation_about(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [
            t * axis[0] * axis[0] + c,
            t * axis[0] * axis[1] - s * axis[2],
            t * axis[0] * axis[2] + s * axis[1],
        ],
        [
            t * axis[0] * axis[1] + s * axis[2],
            t * axis[1] * axis[1] + c,
            t * axis[1] * axis[2] - s * axis[0],
        ],
        [
            t * axis[0] * axis[2] - s * axis[1],
            t * axis[1] * axis[2] + s * axis[0],
            t * axis[2] * axis[2] + c,
        ],
    ]
}

fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                o[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    o
}

/// One assignment-and-refit pass from a given starting rotation.
fn icp(neigh: &[[f64; 3]], template: &[[f64; 3]], mut r: [[f64; 3]; 3]) -> f64 {
    let m = template.len();
    let mut best = f64::INFINITY;
    for _ in 0..16 {
        let mut used = vec![false; m];
        let mut pairs: Vec<([f64; 3], [f64; 3])> = Vec::with_capacity(m);
        let mut total = 0.0;
        for t in template.iter() {
            let tr = apply(&r, *t);
            let mut pick = usize::MAX;
            let mut pick_d = f64::INFINITY;
            for (k, nb) in neigh.iter().enumerate() {
                if used[k] {
                    continue;
                }
                let d = (nb[0] - tr[0]).powi(2) + (nb[1] - tr[1]).powi(2) + (nb[2] - tr[2]).powi(2);
                if d < pick_d {
                    pick_d = d;
                    pick = k;
                }
            }
            if pick == usize::MAX {
                return f64::INFINITY;
            }
            used[pick] = true;
            total += pick_d;
            pairs.push((*t, neigh[pick]));
        }
        let rms = (total / m as f64).sqrt();
        if rms < best - 1e-12 {
            best = rms;
        } else {
            break;
        }
        let from: Vec<[f64; 3]> = pairs.iter().map(|(a, _)| *a).collect();
        let to: Vec<[f64; 3]> = pairs.iter().map(|(_, b)| *b).collect();
        r = kabsch(&from, &to);
    }
    best
}

/// Residual between a neighbourhood and a template, after scaling and rotation.
///
/// The correspondence is the hard part. Iterated closest point solves it from a
/// starting rotation, and on these templates a single start is not enough: a
/// close-packed or icosahedral neighbourhood has an isotropic inertia tensor,
/// so there is no frame to align to and a start from the identity lands in a
/// local optimum. Measured on the 38-point global minimum, whose core is
/// close packed, that gave a best residual of 0.243 where an ideal environment
/// gives 1e-6, and nothing classified.
///
/// So the starts are enumerated instead: map each template point onto the
/// nearest neighbour in turn, spin about that axis through a few angles, and
/// take the best. That fixes one correspondence per start and lets the
/// assignment find the rest. The published method reaches the same
/// correspondence through the topology of the convex hull, which is exact where
/// this is a search; the residual is what anything downstream reads.
fn residual(neigh: &[[f64; 3]], template: &[[f64; 3]]) -> f64 {
    let m = template.len();
    if neigh.len() != m || m == 0 {
        return f64::INFINITY;
    }
    let first = match normalise(neigh[0]) {
        Some(v) => v,
        None => return f64::INFINITY,
    };
    let mut best = f64::INFINITY;
    for t in template.iter() {
        let tv = match normalise(*t) {
            Some(v) => v,
            None => continue,
        };
        let base = align(tv, first);
        // A spin about the now-fixed direction, since aligning one pair leaves
        // one angle free.
        for k in 0..6 {
            let spin = rotation_about(first, k as f64 * std::f64::consts::TAU / 6.0);
            let r = matmul(&spin, &base);
            let v = icp(neigh, template, r);
            if v < best {
                best = v;
            }
        }
    }
    best
}

/// The template a point's neighbourhood matches, and how well.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    /// Best-matching template, or [`Template::Other`] past the cutoff.
    pub template: Template,
    /// Scale-invariant root-mean-square residual to it.
    pub rmsd: f64,
}

/// Classifies every point by its local environment.
///
/// `cutoff` is the residual past which nothing is claimed. Larsen and coworkers
/// use a value near 0.1 for warm structures; smaller is stricter.
pub fn ptm(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Vec<Match> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Neighbours by distance, centred on the point.
        let mut d: Vec<(f64, [f64; 3])> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let v = [
                    x[3 * j] - x[3 * i],
                    x[3 * j + 1] - x[3 * i + 1],
                    x[3 * j + 2] - x[3 * i + 2],
                ];
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2], v)
            })
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut best = Match {
            template: Template::Other,
            rmsd: f64::INFINITY,
        };
        for t in Template::all() {
            let k = t.size();
            if d.len() < k {
                continue;
            }
            let neigh = normalise_scale(d.iter().take(k).map(|(_, v)| *v).collect());
            let rms = residual(&neigh, &t.points());
            if rms < best.rmsd {
                best = Match {
                    template: t,
                    rmsd: rms,
                };
            }
        }
        if best.rmsd > cutoff {
            best = Match {
                template: Template::Other,
                rmsd: best.rmsd,
            };
        }
        out.push(best);
    }
    out
}

/// Common-neighbour fractions as a morphology descriptor.
///
/// `[f555, f421, f422, f544, f433, bonds per point]`. These are the triplets
/// that separate the packings the cluster sizes here compete between: `555` is
/// the icosahedral signature, `421` close-packed cubic, `422` hexagonal, and a
/// decahedron carries `555` together with a substantial `421` population, which
/// an icosahedron does not.
///
/// Preferred over [`ptm_fractions`] for this purpose, and the reason is a
/// measurement rather than a preference. Template matching classifies only
/// points with a complete neighbour shell, which at 38 points is 6 of 38, so
/// its fraction vector is dominated by the surface and resolved 13 distinct
/// morphologies across 218 searches. Common-neighbour analysis reads every
/// bonded pair, including the surface, and costs less because no rotational
/// alignment is involved.
pub fn cna_descriptor(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Array1<f64> {
    let c = cna(x, n, cutoff);
    Array1::from(vec![
        c.fraction((5, 5, 5)),
        c.fraction((4, 2, 1)),
        c.fraction((4, 2, 2)),
        c.fraction((5, 4, 4)),
        c.fraction((4, 3, 3)),
        c.bonds as f64 / n.max(1) as f64 / 6.0,
    ])
}

/// Share of points carrying each template, as a descriptor.
///
/// Ordered as face-centred cubic, hexagonal close packed, icosahedral, simple
/// cubic, other, so it can be compared by distance like any other fingerprint.
pub fn ptm_fractions(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Array1<f64> {
    let m = ptm(x, n, cutoff);
    let mut out = Array1::zeros(5);
    for e in &m {
        let k = match e.template {
            Template::FaceCentredCubic => 0,
            Template::HexagonalClosePacked => 1,
            Template::Icosahedral => 2,
            Template::SimpleCubic => 3,
            Template::Other => 4,
        };
        out[k] += 1.0;
    }
    if n > 0 {
        out /= n as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// A thirteen-point icosahedron: a centre and twelve vertices.
    fn icosahedron13() -> Array1<f64> {
        let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let verts: [[f64; 3]; 12] = [
            [0.0, 1.0, p],
            [0.0, 1.0, -p],
            [0.0, -1.0, p],
            [0.0, -1.0, -p],
            [1.0, p, 0.0],
            [1.0, -p, 0.0],
            [-1.0, p, 0.0],
            [-1.0, -p, 0.0],
            [p, 0.0, 1.0],
            [-p, 0.0, 1.0],
            [p, 0.0, -1.0],
            [-p, 0.0, -1.0],
        ];
        // Edge length 2 before scaling; the centre-to-vertex distance is then
        // sqrt(1 + p^2), so scaling by its reciprocal puts vertices at 1.
        let s = 1.0 / (1.0 + p * p).sqrt();
        let mut x = Array1::<f64>::zeros(3 * 13);
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                x[3 * (i + 1) + k] = s * v[k];
            }
        }
        x
    }

    /// The centre of a perfect icosahedron. Its twelve centre-to-vertex pairs
    /// are the ones the literature reports as 555, and counting the third index
    /// as a simple path instead of a trail gives 554 and reads zero.
    #[test]
    fn an_icosahedron_centre_gives_the_five_five_five_signature() {
        let x = icosahedron13();
        // Cutoff between the first shell at 1 and the vertex-vertex distance,
        // which for this scaling is about 1.05.
        let c = cna(x.view(), 13, 1.2);
        assert!(
            c.get((5, 5, 5)) >= 12,
            "expected at least twelve 555 pairs, counts {:?}",
            c.counts
        );
        assert_eq!(c.get((5, 5, 4)), 0, "555 was miscounted as 554");
        let fr = atom_triplet_fracs(x.view(), 13, 1.2);
        assert!(
            fr[0][0] > 0.9,
            "centre atom 555 fraction {}, counts {:?}",
            fr[0][0],
            c.counts
        );
    }

    /// Template matching has to recognise the same structure, and by a
    /// different route: no bond cutoff is involved.
    #[test]
    fn template_matching_calls_an_icosahedron_centre_icosahedral() {
        let x = icosahedron13();
        let m = ptm(x.view(), 13, 0.1);
        assert_eq!(m[0].template, Template::Icosahedral, "rmsd {}", m[0].rmsd);
        assert!(m[0].rmsd < 1e-6, "an ideal centre gave rmsd {}", m[0].rmsd);
    }

    /// A face-centred cubic environment: the twelve nearest neighbours of an
    /// interior atom, which is the cuboctahedron.
    #[test]
    fn template_matching_calls_a_close_packed_environment_cubic() {
        let mut pts: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
        for p in Template::FaceCentredCubic.points() {
            pts.push(p);
        }
        let n = pts.len();
        let mut x = Array1::<f64>::zeros(3 * n);
        for (i, p) in pts.iter().enumerate() {
            for k in 0..3 {
                x[3 * i + k] = p[k];
            }
        }
        let m = ptm(x.view(), n, 0.1);
        assert_eq!(
            m[0].template,
            Template::FaceCentredCubic,
            "rmsd {}",
            m[0].rmsd
        );
        assert!(m[0].rmsd < 1e-6);
    }

    /// The property that motivates it over a bond cutoff: the answer degrades
    /// smoothly under noise rather than flipping.
    #[test]
    fn the_residual_grows_with_noise_rather_than_flipping() {
        let base = icosahedron13();
        let mut last = -1.0;
        for step in 0..4 {
            let amp = 0.01 * step as f64;
            let mut x = base.clone();
            for (i, v) in x.iter_mut().enumerate() {
                *v += amp * (((i * 29 + 7) % 13) as f64 / 6.0 - 1.0);
            }
            let m = ptm(x.view(), 13, 0.5);
            assert!(
                m[0].rmsd >= last,
                "residual fell from {last} to {} as noise rose",
                m[0].rmsd
            );
            last = m[0].rmsd;
        }
    }

    /// A structure with no recognisable environment is called Other rather than
    /// forced into the nearest template.
    #[test]
    fn a_random_environment_is_not_classified() {
        let n = 13;
        let mut x = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            let a = (i as f64) * 1.317;
            let r = 0.7 + 0.4 * ((i % 5) as f64);
            x[3 * i] = r * a.cos();
            x[3 * i + 1] = r * a.sin();
            x[3 * i + 2] = 0.61 * ((i % 4) as f64) - 0.9;
        }
        let m = ptm(x.view(), n, 0.08);
        assert_eq!(m[0].template, Template::Other, "rmsd {}", m[0].rmsd);
    }

    #[test]
    fn the_fractions_sum_to_one() {
        let x = icosahedron13();
        let f = ptm_fractions(x.view(), 13, 0.1);
        let s: f64 = f.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "fractions summed to {s}");
    }
}

/// Contact cutoff as a multiple of the structure's median nearest-neighbour
/// distance. 1.35 sits above first-neighbour LJ and below the second shell.
pub const RING_CUTOFF_SCALE: f64 = 1.35;

fn median_nearest_neighbour(x: ArrayView1<f64>, n: usize) -> f64 {
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
            let d2 = (0..3)
                .map(|k| {
                    let d = x[3 * i + k] - x[3 * j + k];
                    d * d
                })
                .sum::<f64>();
            if d2 < best {
                best = d2;
            }
        }
        nn.push(best.sqrt());
    }
    nn.sort_by(|a, b| a.total_cmp(b));
    nn[n / 2]
}

/// Franzblau census: global 3/4/5 counts and per-atom incidence.
#[derive(Clone, Debug)]
pub struct RingCensus {
    /// Primitive (triangles, squares, pentagons).
    pub profile: (usize, usize, usize),
    /// How many 3-, 4- and 5-rings contain each atom.
    pub atom: Vec<[u32; 3]>,
}

/// Primitive-ring profile of the contact graph: counts of 3-, 4- and 5-rings.
///
/// The shortest-path ring criterion of Franzblau
/// (doi:10.1103/PhysRevB.44.4925), the primitive of seams-core's network
/// analysis: a cycle counts only when no chord shortcuts it, so two fused
/// triangles do not masquerade as a square. Ring statistics separate packing
/// families by topology rather than by any order parameter, and cost nothing
/// but graph walks.
pub fn ring_profile(x: ArrayView1<f64>, n: usize, cutoff: f64) -> (usize, usize, usize) {
    ring_census(x, n, cutoff).profile
}

/// Per-atom primitive-ring incidence at `cutoff`.
pub fn ring_atom_incidence(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Vec<[u32; 3]> {
    ring_census(x, n, cutoff).atom
}

/// Franzblau census of the contact graph.
pub fn ring_census(x: ArrayView1<f64>, n: usize, cutoff: f64) -> RingCensus {
    let adj = adjacency(x, n, cutoff);
    let nb: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).filter(|&j| adj[i][j]).collect())
        .collect();
    let mut tri = 0usize;
    let mut sq = 0usize;
    let mut pent = 0usize;
    let mut atom = vec![[0u32; 3]; n];
    let bump = |atom: &mut [[u32; 3]], kind: usize, members: &[usize]| {
        for &i in members {
            atom[i][kind] = atom[i][kind].saturating_add(1);
        }
    };
    for a in 0..n {
        for &b in nb[a].iter().filter(|&&b| b > a) {
            for &c in nb[b].iter().filter(|&&c| c > a && c != a) {
                if adj[a][c] {
                    if b < c {
                        tri += 1;
                        bump(&mut atom, 0, &[a, b, c]);
                    }
                    continue;
                }
                // a-b-c open: close with one more step for a square, two for a
                // pentagon, requiring primitivity (no chord).
                for &d in nb[c].iter().filter(|&&d| d != b && d != a) {
                    if adj[a][d] && d > a && !adj[b][d] {
                        if b < d {
                            sq += 1;
                            bump(&mut atom, 1, &[a, b, c, d]);
                        }
                        continue;
                    }
                    if adj[a][d] || d <= a {
                        continue;
                    }
                    for &e in nb[d].iter() {
                        if e > a
                            && adj[a][e]
                            && e != b
                            && e != c
                            && !adj[b][d]
                            && !adj[b][e]
                            && !adj[c][e]
                            && b < e
                        {
                            pent += 1;
                            bump(&mut atom, 2, &[a, b, c, d, e]);
                        }
                    }
                }
            }
        }
    }
    // Each square is found once per traversal direction, each pentagon from
    // both end orderings; triangles are already unique by ordering.
    // Atom incidence stays the raw walk so a ring found once is not
    // rounded off the leave lens.
    RingCensus {
        profile: (tri, sq / 2, pent / 2),
        atom,
    }
}

/// Primitive-ring profile at [`RING_CUTOFF_SCALE`] times median nearest neighbour.
pub fn ring_profile_nn(x: ArrayView1<f64>, n: usize) -> (usize, usize, usize) {
    ring_census_nn(x, n).profile
}

/// Per-atom incidence at [`RING_CUTOFF_SCALE`] times median nearest neighbour.
pub fn ring_atom_incidence_nn(x: ArrayView1<f64>, n: usize) -> Vec<[u32; 3]> {
    ring_census_nn(x, n).atom
}

/// Franzblau census at [`RING_CUTOFF_SCALE`] times median nearest neighbour.
pub fn ring_census_nn(x: ArrayView1<f64>, n: usize) -> RingCensus {
    let cutoff = RING_CUTOFF_SCALE * median_nearest_neighbour(x, n);
    ring_census(x, n, cutoff)
}

#[cfg(test)]
mod ring_tests {
    use super::*;
    use ndarray::Array1;

    fn ico13() -> Array1<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut v = vec![[0.0, 0.0, 0.0]];
        for s in [1.0_f64, -1.0] {
            for t in [phi, -phi] {
                v.push([0.0, s, t]);
                v.push([s, t, 0.0]);
                v.push([t, 0.0, s]);
            }
        }
        let mut x = Array1::zeros(39);
        for (i, p) in v.iter().enumerate() {
            for k in 0..3 {
                x[3 * i + k] = p[k] * 0.55;
            }
        }
        x
    }

    fn fcc13() -> Array1<f64> {
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, -1.0],
        ];
        let mut x = Array1::zeros(39);
        for (i, p) in pts.iter().enumerate() {
            for k in 0..3 {
                x[3 * i + k] = p[k] * 0.75;
            }
        }
        x
    }

    /// The profile has to separate the two thirteen-point packings by
    /// topology and survive a permutation unchanged.
    #[test]
    fn rings_separate_packings_and_ignore_labelling() {
        let a = ring_profile(ico13().view(), 13, 1.2);
        let b = ring_profile(fcc13().view(), 13, 1.2);
        let census = ring_census(ico13().view(), 13, 1.2);
        assert_eq!(census.profile, a);
        let tri: u32 = census.atom.iter().map(|w| w[0]).sum();
        let pent_atoms = census.atom.iter().filter(|w| w[2] > 0).count();
        assert_eq!(tri as usize, 3 * a.0);
        assert!(
            a.2 == 0 || pent_atoms > 0,
            "a packing with pentagons must name the atoms that sit on them"
        );
        assert_ne!(a, b, "ico and fcc thirteen-point profiles agree: {a:?}");
        let x = ico13();
        let mut y = Array1::zeros(x.len());
        let perm = [4usize, 9, 1, 12, 0, 7, 3, 11, 2, 8, 10, 5, 6];
        for (i, p) in perm.iter().enumerate() {
            for k in 0..3 {
                y[3 * i + k] = x[3 * p + k];
            }
        }
        assert_eq!(a, ring_profile(y.view(), 13, 1.2));
    }
}
