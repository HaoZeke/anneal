//! Translation-rotation internal coordinates for molecular quenches.
//!
//! The coordinate system is TRIC (Wang and Song, J. Chem. Phys. 144,
//! 214108, 2016): inside each covalent fragment, its bonds, angles, proper
//! dihedrals and an improper torsion at each centre with three neighbours;
//! for each fragment, the translation of its centroid and a rotation vector
//! scaled by its radius of gyration. A linear fragment rotates as a line,
//! and the centre of a linear angle carries its Cartesian coordinates.
//! Covalent bonds are pairs closer than 1.2 times the sum of their Cordero
//! radii. The primitives are built at the start and rebuilt when an angle
//! goes linear, a fragment straightens, or a back-transformation misses.
//! The redundant primitives are spanned by delocalized coordinates, the
//! active left singular vectors `U` of the Wilson matrix `B` at the start
//! of the chart, kept fixed until a primitive goes linear, the local Gram
//! matrix `U^T B B^T U` loses rank, or a back-transformation misses. The
//! rotation of a fragment is measured from its geometry at the start of the
//! step, so the linear rotation rows of `B` are exact there.
//!
//! The step is a trust-radius Newton step on a model Hessian (Schlegel-type
//! constants per primitive class) projected into the delocalized chart and
//! updated there by BFGS. An interior positive-definite step solves by
//! Cholesky; only an indefinite or trust-capped step diagonalizes `H_d`.
//! The Cartesian point is recovered by iterating the pseudo-inverse of `B`
//! at the step start. A step that raises the energy is refused, the trust
//! radius shrinks, and the solver restarts from the accepted point. The
//! stop is the same as [`crate::methods::warm_lbfgs::WarmLbfgs`]: the
//! largest Cartesian gradient component below `gtol` at an accepted point.
//!
//! Units: the model Hessian is in electronvolt and angstrom, as the
//! molecular presets assume; `hessian_scale` converts it to another energy.
//!
//! # Measured status
//!
//! On 140 GFN2-xTB quenches, (H2O)20 and (NH3)4 starts from anneal's own
//! molecular search and raw ETKDG drug-like conformers at two gates, this
//! quench reached every gate with 48 to 86 percent fewer gradients than
//! [`crate::methods::warm_lbfgs::WarmLbfgs`]. geomeTRIC's TRIC saved 65 to
//! 88 percent, and this module's time outside the potential per call was
//! at most two fifths of geomeTRIC's (`bench/tric-quench`). It does not
//! call geomeTRIC.

use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::methods::cluster_hopping::covalent_radius;

const BOND_FACTOR: f64 = 1.2;
const LINEAR_COS: f64 = -0.996_194_698_091_745_5; // cos 175 degrees
const HARTREE_EV: f64 = 27.211_386_245_988;
const BOHR_A: f64 = 0.529_177_210_903;
const AU_STRETCH: f64 = HARTREE_EV / (BOHR_A * BOHR_A);
const SINGULAR_FLOOR: f64 = 1.0e-7;
const CURVATURE_FLOOR: f64 = 1.0e-4;
// Off-axis share of a fragment's spread below which it rotates as a line.
const LINEAR_SPREAD: f64 = 1.0e-2;
/// Timed probes [`TricOpt::step_seconds`] takes the least of.
const STEP_PROBES: usize = 3;

/// The optimizer every quench of a search uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalStep {
    /// Cartesian L-BFGS, [`crate::methods::warm_lbfgs::WarmLbfgs`].
    #[default]
    Lbfgs,
    /// [`TricOpt`] on a species-labelled Cartesian state; L-BFGS elsewhere.
    Tric,
    /// L-BFGS for the opening quench, then [`TricOpt`] for the rest of the
    /// search when one potential call of that quench took longer than one
    /// TRIC step outside the potential ([`TricOpt::step_seconds`]), and
    /// L-BFGS when it did not.
    ///
    /// A TRIC quench takes about half the calls of a Cartesian one, and each
    /// of its steps pays for a Wilson matrix and a dense eigendecomposition.
    /// Where a call costs less than that work, the halved call count costs
    /// more wall time than it saves.
    Auto,
}

impl LocalStep {
    /// True for the default, which serialization leaves out.
    pub fn is_lbfgs(&self) -> bool {
        matches!(self, Self::Lbfgs)
    }
}

#[derive(Debug, Clone)]
enum Primitive {
    Bond(usize, usize),
    Angle(usize, usize, usize),
    Dihedral(usize, usize, usize, usize),
    Translation {
        fragment: usize,
        axis: usize,
    },
    Rotation {
        fragment: usize,
        axis: usize,
    },
    /// The centre of a linear angle, which no angle or dihedral can bend.
    Cartesian {
        atom: usize,
        axis: usize,
    },
}

#[derive(Debug, Clone)]
struct Fragment {
    atoms: Vec<usize>,
    radius: f64,
    /// Collinear at the start: the spin about the axis is not a coordinate.
    linear: bool,
}

/// The primitive set of one quench, fixed at its start.
#[derive(Debug, Clone)]
pub struct Tric {
    n_atoms: usize,
    primitives: Vec<Primitive>,
    fragments: Vec<Fragment>,
    stiffness: Vec<f64>,
}

fn point(x: &[f64], atom: usize) -> [f64; 3] {
    [x[3 * atom], x[3 * atom + 1], x[3 * atom + 2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn wrap(angle: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut a = angle % tau;
    if a > std::f64::consts::PI {
        a -= tau;
    } else if a <= -std::f64::consts::PI {
        a += tau;
    }
    a
}

fn cos_angle(x: &[f64], i: usize, j: usize, k: usize) -> f64 {
    let u = sub(point(x, i), point(x, j));
    let v = sub(point(x, k), point(x, j));
    (dot(u, v) / (norm(u) * norm(v))).clamp(-1.0, 1.0)
}

fn centroid(x: &[f64], atoms: &[usize]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for &a in atoms {
        for k in 0..3 {
            c[k] += x[3 * a + k];
        }
    }
    scale(c, 1.0 / atoms.len() as f64)
}

/// Eigenvalues ascending and column eigenvectors of a symmetric matrix.
///
/// Householder tridiagonalisation and the implicit QL iteration, after
/// the EISPACK `tred2` and `tql2` routines as JAMA states them.
pub fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    if n == 0 {
        return (d, Vec::new());
    }
    // Store `v` column-major (`j * n + i`) so both EISPACK `tred2` and `tql2`
    // sweep contiguous unit-stride slices on every inner `0..n` / `0..i` loop.
    let mut v = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            v[c * n + r] = a[r * n + c];
        }
    }
    let at = |i: usize, j: usize| j * n + i;
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
    }
    for i in (1..n).rev() {
        let mut scale_sum = 0.0;
        let mut h = 0.0;
        for k in 0..i {
            scale_sum += d[k].abs();
        }
        if scale_sum == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
                v[at(j, i)] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale_sum;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale_sum * g;
            h -= f * g;
            d[i - 1] = f - g;
            for j in 0..i {
                e[j] = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[at(j, i)] = f;
                g = e[j] + v[at(j, j)] * f;
                for k in (j + 1)..i {
                    g += v[at(k, j)] * d[k];
                    e[k] += v[at(k, j)] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[at(k, j)] -= f * e[k] + g * d[k];
                }
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
            }
        }
        d[i] = h;
    }
    for i in 0..(n - 1) {
        v[at(n - 1, i)] = v[at(i, i)];
        v[at(i, i)] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[at(k, i + 1)] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[at(k, i + 1)] * v[at(k, j)];
                }
                for k in 0..=i {
                    v[at(k, j)] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[at(k, i + 1)] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
        v[at(n - 1, j)] = 0.0;
    }
    v[at(n - 1, n - 1)] = 1.0;
    e[0] = 0.0;

    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0;
    let mut tst1: f64 = 0.0;
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        if m > l {
            loop {
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for di in d.iter_mut().take(n).skip(l + 2) {
                    *di -= h;
                }
                f += h;
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    let (head, tail) = v.split_at_mut((i + 1) * n);
                    let col_i = &mut head[i * n..(i + 1) * n];
                    let col_next = &mut tail[..n];
                    for (vi, vn) in col_i.iter_mut().zip(col_next.iter_mut()) {
                        let hi = *vn;
                        *vn = s * *vi + c * hi;
                        *vi = c * *vi - s * hi;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    for i in 0..n.saturating_sub(1) {
        let mut k = i;
        let mut p = d[i];
        for (j, &dj) in d.iter().enumerate().take(n).skip(i + 1) {
            if dj < p {
                k = j;
                p = dj;
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            let (head, tail) = v.split_at_mut(k * n);
            head[i * n..(i + 1) * n].swap_with_slice(&mut tail[..n]);
        }
    }
    let mut out = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            out[r * n + c] = v[c * n + r];
        }
    }
    (d, out)
}

/// Rotation vector of the least-squares rotation taking `reference` onto
/// `current` over `atoms` with centroids `cr` and `cx` (Horn's quaternion).
fn rotation_vector_atoms(
    reference: &[f64],
    cr: [f64; 3],
    current: &[f64],
    cx: [f64; 3],
    atoms: &[usize],
) -> [f64; 3] {
    let mut s = [[0.0; 3]; 3];
    for &a in atoms {
        let p = sub(point(reference, a), cr);
        let q = sub(point(current, a), cx);
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += p[i] * q[j];
            }
        }
    }
    rotation_from_covariance(s)
}

#[cfg(test)]
fn rotation_vector(reference: &[[f64; 3]], current: &[[f64; 3]]) -> [f64; 3] {
    let mut s = [[0.0; 3]; 3];
    for (p, q) in reference.iter().zip(current) {
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += p[i] * q[j];
            }
        }
    }
    rotation_from_covariance(s)
}

fn rotation_from_covariance(s: [[f64; 3]; 3]) -> [f64; 3] {
    let (sxx, sxy, sxz) = (s[0][0], s[0][1], s[0][2]);
    let (syx, syy, syz) = (s[1][0], s[1][1], s[1][2]);
    let (szx, szy, szz) = (s[2][0], s[2][1], s[2][2]);
    let n = [
        sxx + syy + szz,
        syz - szy,
        szx - sxz,
        sxy - syx,
        syz - szy,
        sxx - syy - szz,
        sxy + syx,
        szx + sxz,
        szx - sxz,
        sxy + syx,
        -sxx + syy - szz,
        syz + szy,
        sxy - syx,
        szx + sxz,
        syz + szy,
        -sxx - syy + szz,
    ];
    let (_, vectors) = symmetric_eigen(&n, 4);
    let mut q = [vectors[3], vectors[7], vectors[11], vectors[15]];
    if q[0] < 0.0 {
        q = [-q[0], -q[1], -q[2], -q[3]];
    }
    let v = [q[1], q[2], q[3]];
    let sine = norm(v);
    if sine < 1e-15 {
        return scale(v, 2.0);
    }
    let angle = 2.0 * sine.atan2(q[0]);
    scale(v, angle / sine)
}

/// Sum of `d d^T` over the atoms about `c`, row-major.
fn scatter(x: &[f64], atoms: &[usize], c: [f64; 3]) -> Vec<f64> {
    let mut m = vec![0.0; 9];
    for &a in atoms {
        let d = sub(point(x, a), c);
        for s in 0..3 {
            for t in 0..3 {
                m[3 * s + t] += d[s] * d[t];
            }
        }
    }
    m
}

/// Unit direction of largest spread of centred points.
fn principal_axis(d: &[[f64; 3]]) -> [f64; 3] {
    let mut m = vec![0.0; 9];
    for di in d {
        for s in 0..3 {
            for t in 0..3 {
                m[3 * s + t] += di[s] * di[t];
            }
        }
    }
    let (_, v) = symmetric_eigen(&m, 3);
    [v[2], v[5], v[8]]
}

/// Rotation vector of the shortest rotation taking the axis of the centred
/// `reference` onto that of `current`; it has no component along the axis.
fn line_rotation(reference: &[[f64; 3]], current: &[[f64; 3]]) -> [f64; 3] {
    let from = principal_axis(reference);
    let mut to = principal_axis(current);
    let orient: f64 = reference
        .iter()
        .zip(current)
        .map(|(p, q)| dot(*p, from) * dot(*q, to))
        .sum();
    if orient < 0.0 {
        to = scale(to, -1.0);
    }
    let axis = cross(from, to);
    let sine = norm(axis);
    if sine < 1e-15 {
        return axis;
    }
    scale(axis, sine.atan2(dot(from, to)) / sine)
}

impl Tric {
    /// Primitives of `x` under covalent bonds of the labels `species`.
    pub fn new(species: &[u32], x: ArrayView1<f64>) -> Self {
        let x = x.to_vec();
        let n = species.len();
        assert_eq!(x.len(), 3 * n, "TRIC needs one species label per atom");
        let mut neighbours = vec![Vec::new(); n];
        let mut primitives = Vec::new();
        let mut stiffness = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let limit =
                    BOND_FACTOR * (covalent_radius(species[i]) + covalent_radius(species[j]));
                if limit > 0.0 && norm(sub(point(&x, i), point(&x, j))) < limit {
                    neighbours[i].push(j);
                    neighbours[j].push(i);
                    primitives.push(Primitive::Bond(i, j));
                    stiffness.push(0.45 * AU_STRETCH);
                }
            }
        }
        for j in 0..n {
            let around = &neighbours[j];
            let mut straight = false;
            for (a, &i) in around.iter().enumerate() {
                for &k in &around[a + 1..] {
                    if cos_angle(&x, i, j, k) > LINEAR_COS {
                        primitives.push(Primitive::Angle(i, j, k));
                        stiffness.push(0.16 * HARTREE_EV);
                    } else {
                        straight = true;
                    }
                }
            }
            if straight {
                for axis in 0..3 {
                    primitives.push(Primitive::Cartesian { atom: j, axis });
                    stiffness.push(0.05 * AU_STRETCH);
                }
            }
        }
        for j in 0..n {
            for &k in &neighbours[j] {
                if k < j {
                    continue;
                }
                for &i in &neighbours[j] {
                    if i == k || cos_angle(&x, i, j, k) <= LINEAR_COS {
                        continue;
                    }
                    for &l in &neighbours[k] {
                        if l == j || l == i || cos_angle(&x, j, k, l) <= LINEAR_COS {
                            continue;
                        }
                        primitives.push(Primitive::Dihedral(i, j, k, l));
                        stiffness.push(0.023 * HARTREE_EV);
                    }
                }
            }
        }
        // Bonds and angles alone lose the umbrella of a centre with three
        // neighbours as it flattens; an improper torsion keeps it.
        for j in 0..n {
            if let [i, k, l, ..] = neighbours[j][..]
                && cos_angle(&x, i, k, j) > LINEAR_COS
                && cos_angle(&x, k, j, l) > LINEAR_COS
            {
                primitives.push(Primitive::Dihedral(i, k, j, l));
                stiffness.push(0.045 * HARTREE_EV);
            }
        }
        let mut label = vec![usize::MAX; n];
        let mut fragments = Vec::new();
        for start in 0..n {
            if label[start] != usize::MAX {
                continue;
            }
            let id = fragments.len();
            let mut stack = vec![start];
            let mut atoms = Vec::new();
            label[start] = id;
            while let Some(a) = stack.pop() {
                atoms.push(a);
                for &b in &neighbours[a] {
                    if label[b] == usize::MAX {
                        label[b] = id;
                        stack.push(b);
                    }
                }
            }
            atoms.sort_unstable();
            let c = centroid(&x, &atoms);
            let spread = atoms
                .iter()
                .map(|&a| {
                    let d = sub(point(&x, a), c);
                    dot(d, d)
                })
                .sum::<f64>()
                / atoms.len() as f64;
            let linear = atoms.len() == 2 || {
                let (w, _) = symmetric_eigen(&scatter(&x, &atoms, c), 3);
                w[1] < LINEAR_SPREAD * w[2]
            };
            fragments.push(Fragment {
                atoms,
                radius: spread.sqrt().max(0.5),
                linear,
            });
        }
        for (fragment, frag) in fragments.iter().enumerate() {
            for axis in 0..3 {
                primitives.push(Primitive::Translation { fragment, axis });
                stiffness.push(0.05 * AU_STRETCH);
            }
            if frag.atoms.len() > 1 {
                for axis in 0..3 {
                    primitives.push(Primitive::Rotation { fragment, axis });
                    stiffness.push(0.05 * AU_STRETCH);
                }
            }
        }
        Self {
            n_atoms: n,
            primitives,
            fragments,
            stiffness,
        }
    }

    /// Number of primitives.
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// True when there are no primitives.
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// True when an angle or dihedral of `x` has gone linear, or a bent
    /// fragment has straightened, so the primitives need rebuilding.
    fn degenerate(&self, x: &[f64]) -> bool {
        let linear = |i, j, k| cos_angle(x, i, j, k) <= LINEAR_COS;
        let primitive = self.primitives.iter().any(|p| match *p {
            Primitive::Angle(i, j, k) => linear(i, j, k),
            Primitive::Dihedral(i, j, k, l) => linear(i, j, k) || linear(j, k, l),
            _ => false,
        });
        primitive
            || self.fragments.iter().any(|frag| {
                !frag.linear && frag.atoms.len() > 2 && {
                    let c = centroid(x, &frag.atoms);
                    let (w, _) = symmetric_eigen(&scatter(x, &frag.atoms, c), 3);
                    w[1] < LINEAR_SPREAD * w[2]
                }
            })
    }

    /// Primitive values at `x`, rotations measured from `reference`.
    fn values(&self, x: &[f64], reference: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.primitives.len()];
        self.values_into(x, reference, &mut out);
        out
    }

    fn values_into(&self, x: &[f64], reference: &[f64], out: &mut [f64]) {
        let mut centroids: Vec<Option<[f64; 3]>> = vec![None; self.fragments.len()];
        let mut rotations: Vec<Option<[f64; 3]>> = vec![None; self.fragments.len()];
        for (slot, p) in out.iter_mut().zip(&self.primitives) {
            *slot = match *p {
                Primitive::Bond(i, j) => norm(sub(point(x, i), point(x, j))),
                Primitive::Angle(i, j, k) => cos_angle(x, i, j, k).acos(),
                Primitive::Dihedral(i, j, k, l) => dihedral(x, i, j, k, l),
                Primitive::Translation { fragment, axis } => {
                    let cx = *centroids[fragment]
                        .get_or_insert_with(|| centroid(x, &self.fragments[fragment].atoms));
                    cx[axis]
                }
                Primitive::Cartesian { atom, axis } => x[3 * atom + axis],
                Primitive::Rotation { fragment, axis } => {
                    let frag = &self.fragments[fragment];
                    let theta = *rotations[fragment].get_or_insert_with(|| {
                        let cr = centroid(reference, &frag.atoms);
                        let cx =
                            *centroids[fragment].get_or_insert_with(|| centroid(x, &frag.atoms));
                        if frag.linear {
                            let p: Vec<[f64; 3]> = frag
                                .atoms
                                .iter()
                                .map(|&a| sub(point(reference, a), cr))
                                .collect();
                            let q: Vec<[f64; 3]> =
                                frag.atoms.iter().map(|&a| sub(point(x, a), cx)).collect();
                            line_rotation(&p, &q)
                        } else {
                            rotation_vector_atoms(reference, cr, x, cx, &frag.atoms)
                        }
                    });
                    frag.radius * theta[axis]
                }
            };
        }
    }

    fn difference(&self, to: &[f64], from: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.primitives.len()];
        self.difference_into(to, from, &mut out);
        out
    }

    fn difference_into(&self, to: &[f64], from: &[f64], out: &mut [f64]) {
        for ((slot, p), (&a, &b)) in out
            .iter_mut()
            .zip(&self.primitives)
            .zip(to.iter().zip(from))
        {
            *slot = match p {
                Primitive::Dihedral(..) => wrap(a - b),
                _ => a - b,
            };
        }
    }

    /// Sparse Wilson `B` in compressed-row form, primitives by Cartesian
    /// components. A rotation row is the derivative at zero rotation from `x`
    /// itself.
    fn wilson_sparse(&self, x: &[f64]) -> SparseWilson {
        let n3 = 3 * self.n_atoms;
        let rows = self.primitives.len();
        let mut offsets = Vec::with_capacity(rows + 1);
        let mut cols = Vec::with_capacity(rows * 9);
        let mut vals = Vec::with_capacity(rows * 9);
        offsets.push(0);
        let mut inertia: Vec<Option<[[f64; 3]; 3]>> = vec![None; self.fragments.len()];
        for p in &self.primitives {
            let mut put = |atom: usize, v: [f64; 3]| {
                for (k, &vk) in v.iter().enumerate() {
                    if vk != 0.0 {
                        cols.push(3 * atom + k);
                        vals.push(vk);
                    }
                }
            };
            match *p {
                Primitive::Bond(i, j) => {
                    let d = sub(point(x, i), point(x, j));
                    let u = scale(d, 1.0 / norm(d));
                    put(i, u);
                    put(j, scale(u, -1.0));
                }
                Primitive::Angle(i, j, k) => {
                    let a = sub(point(x, i), point(x, j));
                    let c = sub(point(x, k), point(x, j));
                    let (la, lc) = (norm(a), norm(c));
                    let u = scale(a, 1.0 / la);
                    let v = scale(c, 1.0 / lc);
                    let cos = dot(u, v).clamp(-1.0, 1.0);
                    let sin = (1.0 - cos * cos).sqrt();
                    if sin >= 1e-6 {
                        let di = scale(sub(scale(u, cos), v), 1.0 / (la * sin));
                        let dk = scale(sub(scale(v, cos), u), 1.0 / (lc * sin));
                        put(i, di);
                        put(k, dk);
                        put(
                            j,
                            scale([di[0] + dk[0], di[1] + dk[1], di[2] + dk[2]], -1.0),
                        );
                    }
                }
                Primitive::Dihedral(i, j, k, l) => {
                    let f = sub(point(x, i), point(x, j));
                    let g = sub(point(x, j), point(x, k));
                    let h = sub(point(x, l), point(x, k));
                    let a = cross(f, g);
                    let bb = cross(h, g);
                    let (aa, b2, lg) = (dot(a, a), dot(bb, bb), norm(g));
                    if aa >= 1e-12 && b2 >= 1e-12 && lg >= 1e-12 {
                        let fg = dot(f, g);
                        let hg = dot(h, g);
                        let di = scale(a, -lg / aa);
                        let dl = scale(bb, lg / b2);
                        let tj = sub(scale(a, fg / (aa * lg)), scale(bb, hg / (b2 * lg)));
                        put(i, di);
                        put(l, dl);
                        put(j, sub(tj, di));
                        put(k, sub(scale(tj, -1.0), dl));
                    }
                }
                Primitive::Cartesian { atom, axis } => {
                    cols.push(3 * atom + axis);
                    vals.push(1.0);
                }
                Primitive::Translation { fragment, axis } => {
                    let atoms = &self.fragments[fragment].atoms;
                    let w = 1.0 / atoms.len() as f64;
                    for &a in atoms {
                        cols.push(3 * a + axis);
                        vals.push(w);
                    }
                }
                Primitive::Rotation { fragment, axis } if self.fragments[fragment].linear => {
                    let frag = &self.fragments[fragment];
                    let c = centroid(x, &frag.atoms);
                    let d: Vec<[f64; 3]> =
                        frag.atoms.iter().map(|&a| sub(point(x, a), c)).collect();
                    let line = principal_axis(&d);
                    let s2: f64 = d.iter().map(|di| dot(*di, line).powi(2)).sum();
                    // theta = line x sum s_i dr_i / sum s_i^2, s_i = d_i . line.
                    let mut unit = [0.0; 3];
                    unit[axis] = 1.0;
                    let lever = cross(unit, line);
                    for (&a, di) in frag.atoms.iter().zip(&d) {
                        let s_i = dot(*di, line) / s2;
                        put(a, scale(lever, s_i * frag.radius));
                    }
                }
                Primitive::Rotation { fragment, axis } => {
                    let frag = &self.fragments[fragment];
                    let c = centroid(x, &frag.atoms);
                    let inv = *inertia[fragment].get_or_insert_with(|| {
                        let mut m = [0.0; 9];
                        for &a in &frag.atoms {
                            let d = sub(point(x, a), c);
                            let r2 = dot(d, d);
                            for s in 0..3 {
                                for t in 0..3 {
                                    m[3 * s + t] += if s == t { r2 } else { 0.0 } - d[s] * d[t];
                                }
                            }
                        }
                        pseudo_inverse3(&m)
                    });
                    for &a in &frag.atoms {
                        let d = sub(point(x, a), c);
                        // Row `axis` of I^-1 [d]x, so theta = I^-1 sum d x dr.
                        let row = inv[axis];
                        let v = [
                            row[1] * d[2] - row[2] * d[1],
                            row[2] * d[0] - row[0] * d[2],
                            row[0] * d[1] - row[1] * d[0],
                        ];
                        put(a, scale(v, frag.radius));
                    }
                }
            }
            offsets.push(cols.len());
        }
        SparseWilson {
            rows,
            n_cols: n3,
            offsets,
            cols,
            vals,
        }
    }

    #[cfg(test)]
    fn wilson(&self, x: &[f64]) -> Vec<f64> {
        let sparse = self.wilson_sparse(x);
        let mut dense = vec![0.0; sparse.rows * sparse.n_cols];
        for r in 0..sparse.rows {
            for idx in sparse.offsets[r]..sparse.offsets[r + 1] {
                dense[r * sparse.n_cols + sparse.cols[idx]] += sparse.vals[idx];
            }
        }
        dense
    }
}

struct SparseWilson {
    rows: usize,
    n_cols: usize,
    offsets: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

impl SparseWilson {
    fn mul(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.rows];
        for (r, slot) in out.iter_mut().enumerate() {
            let mut sum = 0.0;
            for idx in self.offsets[r]..self.offsets[r + 1] {
                sum += self.vals[idx] * x[self.cols[idx]];
            }
            *slot = sum;
        }
        out
    }

    fn transpose_mul(&self, y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_cols];
        for (r, &yr) in y.iter().enumerate() {
            if yr == 0.0 {
                continue;
            }
            for idx in self.offsets[r]..self.offsets[r + 1] {
                out[self.cols[idx]] += self.vals[idx] * yr;
            }
        }
        out
    }
}

fn dihedral(x: &[f64], i: usize, j: usize, k: usize, l: usize) -> f64 {
    let f = sub(point(x, i), point(x, j));
    let g = sub(point(x, j), point(x, k));
    let h = sub(point(x, l), point(x, k));
    let a = cross(f, g);
    let b = cross(h, g);
    let lg = norm(g);
    let y = dot(cross(b, a), g) / lg.max(1e-300);
    y.atan2(dot(a, b))
}

fn pseudo_inverse3(m: &[f64; 9]) -> [[f64; 3]; 3] {
    let (w, v) = symmetric_eigen(m, 3);
    let top = w[2].abs().max(1e-300);
    let mut out = [[0.0; 3]; 3];
    for k in 0..3 {
        if w[k] > 1e-8 * top {
            for s in 0..3 {
                for t in 0..3 {
                    out[s][t] += v[3 * s + k] * v[3 * t + k] / w[k];
                }
            }
        }
    }
    out
}

/// Right-looking blocked Cholesky factor `L` of symmetric `a` (`a = L L^T`),
/// delegating trailing updates to `ndarray`'s GEMM microkernel. `None` when
/// any pivot falls at or below `floor`.
fn cholesky_lower(a: &[f64], n: usize, floor: f64) -> Option<Vec<f64>> {
    const BLOCK: usize = 32;
    let mut l = Array2::<f64>::zeros((n, n));
    for r in 0..n {
        for c in 0..=r {
            l[(r, c)] = a[r * n + c];
        }
    }
    let mut k = 0;
    while k < n {
        let kb = (k + BLOCK).min(n);
        for i in k..kb {
            for j in k..i {
                let mut sum = l[(i, j)];
                for p in k..j {
                    sum -= l[(i, p)] * l[(j, p)];
                }
                l[(i, j)] = sum / l[(j, j)];
            }
            let mut diag = l[(i, i)];
            for p in k..i {
                let v = l[(i, p)];
                diag -= v * v;
            }
            if !(diag > floor) {
                return None;
            }
            l[(i, i)] = diag.sqrt();
        }
        if kb < n {
            for i in kb..n {
                for j in k..kb {
                    let mut sum = l[(i, j)];
                    for p in k..j {
                        sum -= l[(i, p)] * l[(j, p)];
                    }
                    l[(i, j)] = sum / l[(j, j)];
                }
            }
            let (_top, bot) = l.view_mut().split_at(ndarray::Axis(0), kb);
            let (l21_part, mut a22) = bot.split_at(ndarray::Axis(1), kb);
            let l21 = l21_part.slice(ndarray::s![.., k..kb]);
            let mut a22_sub = a22.slice_mut(ndarray::s![.., ..]);
            general_mat_mul(-1.0, &l21, &l21.t(), 1.0, &mut a22_sub);
            for r in 0..(n - kb) {
                for c in (r + 1)..(n - kb) {
                    a22_sub[(r, c)] = 0.0;
                }
            }
        }
        k = kb;
    }
    Some(l.into_raw_vec_and_offset().0)
}

/// Solves `L L^T x = rhs` from a row-major lower Cholesky factor `l`.
fn cholesky_solve(l: &[f64], n: usize, rhs: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = rhs[i];
        let li = &l[i * n..i * n + i];
        for (&lij, &yj) in li.iter().zip(&y[..i]) {
            sum -= lij * yj;
        }
        y[i] = sum / l[i * n + i];
    }
    for j in (0..n).rev() {
        let xj = y[j] / l[j * n + j];
        y[j] = xj;
        for i in 0..j {
            y[i] -= l[j * n + i] * xj;
        }
    }
    y
}

/// Delocalized internal coordinate chart built once from `B = U S V^T` at the
/// chart origin and held until a primitive degenerates, the local Gram matrix
/// loses rank, or back-transformation misses.
#[derive(Debug, Clone)]
struct Chart {
    rows: usize,
    cols: usize,
    m: usize,
    /// `rows` by `m`, row-major: active left singular vectors `U`.
    u: Vec<f64>,
    /// `cols` by `m`, row-major: `(U^T B(x_0))^T = V S` at the chart origin.
    origin_bd_t: Vec<f64>,
    /// Active singular values `S` at the chart origin.
    origin_s: Vec<f64>,
}

fn invert_spd(a: &[f64], n: usize, floor: f64) -> Option<Vec<f64>> {
    let l = cholesky_lower(a, n, floor)?;
    let mut inv = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    for col in 0..n {
        rhs[col] = 1.0;
        let solved = cholesky_solve(&l, n, &rhs);
        rhs[col] = 0.0;
        for row in 0..n {
            inv[row * n + col] = solved[row];
        }
    }
    Some(inv)
}

impl Chart {
    fn new(coords: &Tric, x0: &[f64], hessian_scale: f64) -> (Self, Vec<f64>, Vec<f64>) {
        let b = coords.wilson_sparse(x0);
        let rows = b.rows;
        let cols = b.n_cols;
        let mut btb = vec![0.0; cols * cols];
        for r in 0..rows {
            let start = b.offsets[r];
            let end = b.offsets[r + 1];
            for ia in start..end {
                let p = b.cols[ia];
                let bp = b.vals[ia];
                for ib in start..end {
                    let q = b.cols[ib];
                    if p <= q {
                        btb[p * cols + q] += bp * b.vals[ib];
                    }
                }
            }
        }
        for p in 0..cols {
            for q in 0..p {
                btb[p * cols + q] = btb[q * cols + p];
            }
        }
        let (w, vecs) = symmetric_eigen(&btb, cols);
        let keep: Vec<usize> = (0..cols).filter(|&k| w[k] > SINGULAR_FLOOR).collect();
        let m = keep.len();
        let mut v = vec![0.0; cols * m];
        let mut origin_bd_t = vec![0.0; cols * m];
        let mut s = vec![0.0; m];
        for (c, &k) in keep.iter().enumerate() {
            let sc = w[k].sqrt();
            s[c] = sc;
            for p in 0..cols {
                let vp = vecs[p * cols + k];
                v[p * m + c] = vp;
                origin_bd_t[p * m + c] = vp * sc;
            }
        }
        let mut u = vec![0.0; rows * m];
        let mut ku = vec![0.0; rows * m];
        for r in 0..rows {
            let ur = &mut u[r * m..(r + 1) * m];
            for idx in b.offsets[r]..b.offsets[r + 1] {
                let vp = &v[b.cols[idx] * m..(b.cols[idx] + 1) * m];
                let val = b.vals[idx];
                for (uc, &vc) in ur.iter_mut().zip(vp) {
                    *uc += val * vc;
                }
            }
            let k = (coords.stiffness[r] * hessian_scale).max(CURVATURE_FLOOR);
            let kur = &mut ku[r * m..(r + 1) * m];
            for ((uc, &sc), kuc) in ur.iter_mut().zip(&s).zip(kur) {
                *uc /= sc;
                *kuc = k * *uc;
            }
        }
        let hd = if m > 0 && rows > 0 {
            let u_mat = ArrayView2::from_shape((rows, m), &u).expect("u shape");
            let ku_mat = ArrayView2::from_shape((rows, m), &ku).expect("ku shape");
            let mut hd_mat = u_mat.t().dot(&ku_mat);
            for a in 0..m {
                hd_mat[(a, a)] = hd_mat[(a, a)].max(CURVATURE_FLOOR);
                for c in 0..a {
                    let avg = 0.5 * (hd_mat[(a, c)] + hd_mat[(c, a)]);
                    hd_mat[(a, c)] = avg;
                    hd_mat[(c, a)] = avg;
                }
            }
            hd_mat.into_raw_vec_and_offset().0
        } else {
            Vec::new()
        };
        let cd = invert_spd(&hd, m, CURVATURE_FLOOR * 0.5).unwrap_or_else(|| {
            let mut diag = vec![0.0; m * m];
            for a in 0..m {
                diag[a * m + a] = 1.0 / hd[a * m + a].max(CURVATURE_FLOOR);
            }
            diag
        });
        (
            Self {
                rows,
                cols,
                m,
                u,
                origin_bd_t,
                origin_s: s,
            },
            hd,
            cd,
        )
    }

    /// Delocalized vector `U^T dq` of a primitive displacement `dq`.
    fn project(&self, dq: &[f64]) -> Vec<f64> {
        if self.m == 0 || self.rows == 0 {
            return vec![0.0; self.m];
        }
        let u_mat = ArrayView2::from_shape((self.rows, self.m), &self.u).expect("u shape");
        let dq_view = ArrayView1::from(dq);
        u_mat.t().dot(&dq_view).into_raw_vec_and_offset().0
    }

    /// Primitive vector `U d` of a delocalized vector `d`.
    fn primitive(&self, d: &[f64]) -> Vec<f64> {
        if self.m == 0 || self.rows == 0 {
            return vec![0.0; self.rows];
        }
        let u_mat = ArrayView2::from_shape((self.rows, self.m), &self.u).expect("u shape");
        let d_view = ArrayView1::from(d);
        u_mat.dot(&d_view).into_raw_vec_and_offset().0
    }
}

/// Cached factored Gram matrix `L L^T = B_d(x_ref) B_d(x_ref)^T` and Jacobian
/// `B_d(x_ref)^T`, reused as preconditioner and back-transform map while atoms
/// stay within `BASIS_DRIFT_A` of `x_ref`.
#[derive(Debug, Clone)]
struct CachedGram {
    at: Vec<f64>,
    bd_t: Vec<f64>,
    l: Vec<f64>,
}

const BASIS_DRIFT_A: f64 = 0.15;

fn max_atom_drift(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() / 3;
    let mut best = 0.0_f64;
    for i in 0..n {
        let d2 = (a[3 * i] - b[3 * i]).powi(2)
            + (a[3 * i + 1] - b[3 * i + 1]).powi(2)
            + (a[3 * i + 2] - b[3 * i + 2]).powi(2);
        if d2 > best {
            best = d2;
        }
    }
    best.sqrt()
}

impl CachedGram {
    fn build(chart: &Chart, coords: &Tric, x: &[f64]) -> Option<Self> {
        let m = chart.m;
        let cols = chart.cols;
        let b = coords.wilson_sparse(x);
        let mut bd_t = vec![0.0; cols * m];
        for r in 0..b.rows {
            let ur = &chart.u[r * m..(r + 1) * m];
            for idx in b.offsets[r]..b.offsets[r + 1] {
                let dst = &mut bd_t[b.cols[idx] * m..(b.cols[idx] + 1) * m];
                let val = b.vals[idx];
                for (d, &uc) in dst.iter_mut().zip(ur) {
                    *d += val * uc;
                }
            }
        }
        let bd_mat = ArrayView2::from_shape((cols, m), &bd_t).expect("bd_t shape");
        let gram_mat = bd_mat.t().dot(&bd_mat);
        let gram = gram_mat.as_slice().expect("contiguous gram");
        let l = cholesky_lower(gram, m, SINGULAR_FLOOR)?;
        Some(Self {
            at: x.to_vec(),
            bd_t,
            l,
        })
    }

    /// Exact internal gradient at `x` (when `x == self.at`) or PCG-refined
    /// against `B(x)` using `self.l` as preconditioner.
    fn gradient(&self, chart: &Chart, coords: &Tric, x: &[f64], g: &[f64]) -> Option<Vec<f64>> {
        let m = chart.m;
        if x == self.at.as_slice() {
            let bd_mat = ArrayView2::from_shape((chart.cols, m), &self.bd_t).expect("bd_t");
            let bg = bd_mat.t().dot(&ArrayView1::from(g));
            return Some(cholesky_solve(&self.l, m, bg.as_slice().expect("bg")));
        }
        let b = coords.wilson_sparse(x);
        let rhs = chart.project(&b.mul(g));
        let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rhs_norm < 1e-30 {
            return Some(vec![0.0; m]);
        }
        let apply = |v: &[f64]| -> Vec<f64> {
            let uv = chart.primitive(v);
            let bt_uv = b.transpose_mul(&uv);
            let b_bt_uv = b.mul(&bt_uv);
            chart.project(&b_bt_uv)
        };
        let mut sol = cholesky_solve(&self.l, m, &rhs);
        let ax = apply(&sol);
        let mut r: Vec<f64> = rhs
            .iter()
            .zip(&ax)
            .map(|(rhs_i, a_i)| rhs_i - a_i)
            .collect();
        let tol = 1e-5 * rhs_norm;
        if r.iter().map(|v| v * v).sum::<f64>().sqrt() <= tol {
            return Some(sol);
        }
        let mut z = cholesky_solve(&self.l, m, &r);
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let mut p = z.clone();
        for _ in 0..12 {
            let ap = apply(&p);
            let pap: f64 = p.iter().zip(&ap).map(|(pi, api)| pi * api).sum();
            let pp: f64 = p.iter().map(|pi| pi * pi).sum();
            if !(pap > SINGULAR_FLOOR * pp) {
                return None;
            }
            let alpha = rz / pap;
            for (si, &pi) in sol.iter_mut().zip(&p) {
                *si += alpha * pi;
            }
            for (ri, &api) in r.iter_mut().zip(&ap) {
                *ri -= alpha * api;
            }
            if r.iter().map(|v| v * v).sum::<f64>().sqrt() <= tol {
                return Some(sol);
            }
            z = cholesky_solve(&self.l, m, &r);
            let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            if rz <= 0.0 {
                break;
            }
            let beta = rz_new / rz;
            for (pi, &zi) in p.iter_mut().zip(&z) {
                *pi = zi + beta * *pi;
            }
            rz = rz_new;
        }
        None
    }

    fn cartesian(&self, cols: usize, m: usize, dq_d: &[f64]) -> Vec<f64> {
        let w = cholesky_solve(&self.l, m, dq_d);
        let bd_mat = ArrayView2::from_shape((cols, m), &self.bd_t).expect("bd_t");
        bd_mat
            .dot(&ArrayView1::from(&w))
            .into_raw_vec_and_offset()
            .0
    }
}

enum StepBasis<'a> {
    Origin(&'a Chart),
    Gram(&'a Chart, &'a CachedGram),
}

impl<'a> StepBasis<'a> {
    fn origin(chart: &'a Chart) -> Self {
        Self::Origin(chart)
    }

    fn origin_gradient(chart: &Chart, g: &[f64]) -> Vec<f64> {
        let m = chart.m;
        let bd_mat = ArrayView2::from_shape((chart.cols, m), &chart.origin_bd_t).expect("bd_t");
        let bg = bd_mat.t().dot(&ArrayView1::from(g));
        bg.iter()
            .zip(&chart.origin_s)
            .map(|(&r, &sc)| r / (sc * sc))
            .collect()
    }

    fn cartesian(&self, dq_d: &[f64]) -> Vec<f64> {
        match *self {
            Self::Origin(chart) => {
                let m = chart.m;
                let w: Vec<f64> = dq_d
                    .iter()
                    .zip(&chart.origin_s)
                    .map(|(&d, &sc)| d / (sc * sc))
                    .collect();
                let bd_mat =
                    ArrayView2::from_shape((chart.cols, m), &chart.origin_bd_t).expect("bd_t");
                bd_mat
                    .dot(&ArrayView1::from(&w))
                    .into_raw_vec_and_offset()
                    .0
            }
            Self::Gram(chart, gram) => gram.cartesian(chart.cols, chart.m, dq_d),
        }
    }
}

/// Powell's dogleg step on positive-definite `hd` and its inverse `cd`, within
/// `trust`. Avoids `O(m^3)` eigendecompositions on trust-capped steps.
fn dogleg_step(hd: &[f64], cd: &[f64], gd: &[f64], m: usize, trust: f64) -> (Vec<f64>, f64) {
    let pb: Vec<f64> = (0..m)
        .map(|r| {
            -cd[r * m..(r + 1) * m]
                .iter()
                .zip(gd)
                .map(|(c, g)| c * g)
                .sum::<f64>()
        })
        .collect();
    let pb_norm = pb.iter().map(|v| v * v).sum::<f64>().sqrt();
    let step_d = if pb_norm <= trust {
        pb
    } else {
        let gnorm2: f64 = gd.iter().map(|v| v * v).sum();
        if gnorm2 == 0.0 {
            vec![0.0; m]
        } else {
            let ghg: f64 = (0..m)
                .map(|r| {
                    gd[r]
                        * hd[r * m..(r + 1) * m]
                            .iter()
                            .zip(gd)
                            .map(|(h, g)| h * g)
                            .sum::<f64>()
                })
                .sum();
            let alpha_u = gnorm2 / ghg.max(CURVATURE_FLOOR * gnorm2);
            let gnorm = gnorm2.sqrt();
            if alpha_u * gnorm >= trust {
                let scale = -trust / gnorm;
                gd.iter().map(|&g| scale * g).collect()
            } else {
                let pu: Vec<f64> = gd.iter().map(|&g| -alpha_u * g).collect();
                let diff: Vec<f64> = pb.iter().zip(&pu).map(|(b, u)| b - u).collect();
                let a: f64 = diff.iter().map(|v| v * v).sum();
                let b: f64 = pu.iter().zip(&diff).map(|(u, d)| u * d).sum();
                let c: f64 = pu.iter().map(|v| v * v).sum::<f64>() - trust * trust;
                let tau = if a > 0.0 {
                    ((-b + (b * b - a * c).max(0.0).sqrt()) / a).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                pu.iter().zip(&diff).map(|(u, d)| u + tau * d).collect()
            }
        }
    };
    let h_step_dot_step: f64 = (0..m)
        .map(|r| {
            step_d[r]
                * hd[r * m..(r + 1) * m]
                    .iter()
                    .zip(&step_d)
                    .map(|(h, s)| h * s)
                    .sum::<f64>()
        })
        .sum();
    let g_dot_step: f64 = gd.iter().zip(&step_d).map(|(g, s)| g * s).sum();
    (step_d, g_dot_step + 0.5 * h_step_dot_step)
}

#[derive(Debug, Clone)]
struct TricState {
    at: Vec<f64>,
    coords: Tric,
    chart: Chart,
    cached_gram: Option<CachedGram>,
    hd: Vec<f64>,
    cd: Vec<f64>,
    trust: f64,
    previous: Option<(Vec<f64>, Vec<f64>)>,
    rebuilt: bool,
    at_origin: bool,
}

/// Quasi-Newton relaxation in translation-rotation internal coordinates.
#[derive(Debug, Clone)]
pub struct TricOpt {
    species: Vec<u32>,
    /// Largest Cartesian gradient component at which a relaxation stops.
    pub gtol: f64,
    /// First trust radius, angstrom in the internal metric.
    pub trust: f64,
    /// Largest trust radius.
    pub max_trust: f64,
    /// Smallest trust radius; a refusal below it ends the relaxation.
    pub min_trust: f64,
    /// Energy unit of the model Hessian per electronvolt.
    pub hessian_scale: f64,
    /// Back-transformation iterations per step.
    pub max_backtransform: usize,
    state: std::cell::RefCell<Option<TricState>>,
}

impl TricOpt {
    /// A relaxation over atoms labelled by atomic number.
    pub fn new(species: Vec<u32>) -> Self {
        Self {
            species,
            gtol: 1e-6,
            trust: 0.1,
            max_trust: 0.3,
            min_trust: 1e-5,
            hessian_scale: 1.0,
            max_backtransform: 50,
            state: std::cell::RefCell::new(None),
        }
    }

    /// Discards any chart and Hessian held from an interrupted descent.
    pub fn forget(&self) {
        self.state.borrow_mut().take();
    }

    /// Atoms the species labels cover.
    pub fn atoms(&self) -> usize {
        self.species.len()
    }

    /// Expected call-count advantage `(N_LBFGS / N_TRIC - 1)` at `x`:
    /// dihedral-bearing molecules have torsional modes two decades softer than
    /// stretches (~16x fewer calls at `1e-3` eV/Å), whereas stretch-and-bend
    /// clusters without dihedrals save ~4x (`bench/tric-quench`).
    pub fn call_saving(&self, x: ArrayView1<f64>) -> f64 {
        let coords = Tric::new(&self.species, x);
        let has_dihedrals = coords
            .primitives
            .iter()
            .any(|p| matches!(p, Primitive::Dihedral(..)));
        if has_dihedrals { 16.0 } else { 3.0 }
    }

    /// Wall seconds one amortized step at `x` spends outside the potential:
    /// one chart build amortized over a quench plus the per-step Jacobian,
    /// dogleg and back-transformation, with `energy` and `gradient` standing
    /// in for the potential, which is not called.
    ///
    /// The least of `STEP_PROBES` probes is kept. A single wall-clock
    /// sample is the cold first step of the process, and one preemption
    /// inside it can exceed the potential call it is compared against.
    pub fn step_seconds(&self, x: ArrayView1<f64>, energy: f64, gradient: ArrayView1<f64>) -> f64 {
        let mut probe = self.clone();
        probe.gtol = 0.0;
        let answer = (energy, gradient.to_owned());
        (0..STEP_PROBES)
            .map(|_| {
                probe.forget();
                let t0 = std::time::Instant::now();
                let (_, x1, _) = probe.minimize(x, 2, |_| Some(answer.clone()));
                let chart_s = t0.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                probe.minimize(x1.view(), 5, |_| Some(answer.clone()));
                let steps_s = t1.elapsed().as_secs_f64() / 4.0;
                chart_s / 32.0 + steps_s
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Relaxes `x0`; `fg` returning `None` ends the relaxation on the best
    /// accepted point. Returns that point's energy, the point, and the calls
    /// made.
    pub fn minimize<F>(
        &self,
        x0: ArrayView1<f64>,
        max_calls: usize,
        fg: F,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        self.minimize_watched(x0, max_calls, fg, |_, _| true)
    }

    /// As [`TricOpt::minimize`], with `watch(calls, energy)` consulted at
    /// every accepted point; `false` stops there.
    pub fn minimize_watched<F, W>(
        &self,
        x0: ArrayView1<f64>,
        max_calls: usize,
        mut fg: F,
        mut watch: W,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        W: FnMut(usize, f64) -> bool,
    {
        let start = x0.to_owned();
        let saved = self.state.borrow_mut().take();
        // A zero-step quench is the raw energy, as for warm L-BFGS: the
        // driver reads a start and a first-stage screen through it.
        let Some((mut e, mut g)) = fg(x0) else {
            return (f64::INFINITY, start, 0);
        };
        let mut calls = 1usize;
        let mut x = start;
        if max_calls == 0 || !e.is_finite() || max_abs(&g) < self.gtol || !watch(calls, e) {
            return (e, x, calls);
        }
        let x_slice = x.as_slice().expect("contiguous point");
        let (
            mut coords,
            mut chart,
            mut cached_gram,
            mut hd,
            mut cd,
            mut trust,
            mut previous,
            mut rebuilt,
            mut at_origin,
        ) = match saved.filter(|s| s.at == x_slice) {
            Some(s) => (
                s.coords,
                s.chart,
                s.cached_gram,
                s.hd,
                s.cd,
                s.trust,
                s.previous,
                s.rebuilt,
                s.at_origin,
            ),
            None => {
                let coords = Tric::new(&self.species, x.view());
                let (chart, hd, cd) = Chart::new(&coords, x_slice, self.hessian_scale);
                (coords, chart, None, hd, cd, self.trust, None, true, true)
            }
        };
        while calls < max_calls {
            let xs = x.as_slice().expect("contiguous point").to_vec();
            // A linear angle or a straightened fragment has no well-defined
            // primitive value, so the chart is rebuilt from the current point.
            if !rebuilt && coords.degenerate(&xs) {
                coords = Tric::new(&self.species, x.view());
                (chart, hd, cd) = Chart::new(&coords, &xs, self.hessian_scale);
                cached_gram = None;
                previous = None;
                rebuilt = true;
                at_origin = true;
            }
            if chart.m == 0 {
                break;
            }
            let gs = g.as_slice().expect("contiguous gradient");
            if !at_origin {
                let stale = cached_gram
                    .as_ref()
                    .is_none_or(|cg| max_atom_drift(&xs, &cg.at) > BASIS_DRIFT_A);
                if stale {
                    cached_gram = CachedGram::build(&chart, &coords, &xs);
                    if cached_gram.is_none() {
                        coords = Tric::new(&self.species, x.view());
                        (chart, hd, cd) = Chart::new(&coords, &xs, self.hessian_scale);
                        previous = None;
                        rebuilt = true;
                        at_origin = true;
                        if chart.m == 0 {
                            break;
                        }
                    }
                }
            }
            let gd = if at_origin {
                StepBasis::origin_gradient(&chart, gs)
            } else if let Some(gd) = cached_gram
                .as_ref()
                .and_then(|cg| cg.gradient(&chart, &coords, &xs, gs))
            {
                gd
            } else {
                // Refresh `cached_gram` at `xs` before discarding `hd`/`cd`.
                cached_gram = CachedGram::build(&chart, &coords, &xs);
                if let Some(gd) = cached_gram
                    .as_ref()
                    .and_then(|cg| cg.gradient(&chart, &coords, &xs, gs))
                {
                    gd
                } else {
                    coords = Tric::new(&self.species, x.view());
                    (chart, hd, cd) = Chart::new(&coords, &xs, self.hessian_scale);
                    cached_gram = None;
                    previous = None;
                    rebuilt = true;
                    at_origin = true;
                    if chart.m == 0 {
                        break;
                    }
                    StepBasis::origin_gradient(&chart, gs)
                }
            };
            let basis = if at_origin {
                StepBasis::origin(&chart)
            } else {
                StepBasis::Gram(&chart, cached_gram.as_ref().expect("cached gram"))
            };
            let m = chart.m;
            if let Some((step_taken, gd_old)) = previous.take() {
                bfgs_pair(&mut hd, &mut cd, m, &step_taken, &gd, &gd_old);
            }
            let (step_d, predicted) = dogleg_step(&hd, &cd, &gd, m, trust);
            let planned = step_d.iter().map(|v| v * v).sum::<f64>().sqrt();
            let q0 = coords.values(&xs, &xs);
            let (trial, missed) = self.back_transform(&coords, &chart, &basis, &xs, &q0, &step_d);
            if !rebuilt && missed > 0.25 * planned {
                if cached_gram.as_ref().is_some_and(|cg| cg.at != xs)
                    && let Some(fresh) = CachedGram::build(&chart, &coords, &xs)
                {
                    let fresh_basis = StepBasis::Gram(&chart, &fresh);
                    let (retry_trial, retry_missed) =
                        self.back_transform(&coords, &chart, &fresh_basis, &xs, &q0, &step_d);
                    cached_gram = Some(fresh);
                    if retry_missed <= 0.25 * planned {
                        let Some((e_new, g_new)) = fg(ArrayView1::from(&retry_trial)) else {
                            break;
                        };
                        calls += 1;
                        let rise = e_new - e;
                        if !e_new.is_finite() || rise > 0.0 {
                            trust = (0.25 * planned.min(trust)).max(0.0);
                            if trust < self.min_trust {
                                break;
                            }
                            continue;
                        }
                        let ratio = if predicted < 0.0 {
                            rise / predicted
                        } else {
                            1.0
                        };
                        if ratio > 0.75 && planned > 0.8 * trust {
                            trust = (2.0 * trust).min(self.max_trust);
                        } else if ratio < 0.25 {
                            trust = (0.5 * trust).max(self.min_trust);
                        }
                        let q_new = coords.values(&retry_trial, &xs);
                        let s_d = chart.project(&coords.difference(&q_new, &q0));
                        previous = Some((s_d, gd));
                        rebuilt = false;
                        at_origin = false;
                        x = Array1::from_vec(retry_trial);
                        e = e_new;
                        g = g_new;
                        if max_abs(&g) < self.gtol || !watch(calls, e) {
                            break;
                        }
                        continue;
                    }
                }
                coords = Tric::new(&self.species, x.view());
                (chart, hd, cd) = Chart::new(&coords, &xs, self.hessian_scale);
                cached_gram = None;
                previous = None;
                rebuilt = true;
                at_origin = true;
                continue;
            }
            let Some((e_new, g_new)) = fg(ArrayView1::from(&trial)) else {
                break;
            };
            calls += 1;
            let rise = e_new - e;
            if !e_new.is_finite() || rise > 0.0 {
                trust = (0.25 * planned.min(trust)).max(0.0);
                if trust < self.min_trust {
                    break;
                }
                continue;
            }
            let ratio = if predicted < 0.0 {
                rise / predicted
            } else {
                1.0
            };
            if ratio > 0.75 && planned > 0.8 * trust {
                trust = (2.0 * trust).min(self.max_trust);
            } else if ratio < 0.25 {
                trust = (0.5 * trust).max(self.min_trust);
            }
            let q_new = coords.values(&trial, &xs);
            let s_d = chart.project(&coords.difference(&q_new, &q0));
            previous = Some((s_d, gd));
            rebuilt = false;
            at_origin = false;
            x = Array1::from_vec(trial);
            e = e_new;
            g = g_new;
            if max_abs(&g) < self.gtol || !watch(calls, e) {
                break;
            }
        }
        *self.state.borrow_mut() = Some(TricState {
            at: x.as_slice().expect("contiguous point").to_vec(),
            coords,
            chart,
            cached_gram,
            hd,
            cd,
            trust,
            previous,
            rebuilt,
            at_origin,
        });
        (e, x, calls)
    }

    /// The Cartesian point whose delocalized displacement from `x0` matches
    /// `step_d`, and its active-chart residual norm.
    fn back_transform(
        &self,
        coords: &Tric,
        chart: &Chart,
        basis: &StepBasis<'_>,
        x0: &[f64],
        q0: &[f64],
        step_d: &[f64],
    ) -> (Vec<f64>, f64) {
        let planned = step_d.iter().map(|v| v * v).sum::<f64>().sqrt();
        let res_tol = (1e-4 * planned).max(1e-10);
        let mut x = x0.to_vec();
        let mut best = (f64::INFINITY, x.clone());
        let mut q_cur = vec![0.0; q0.len()];
        let mut diff_q = vec![0.0; q0.len()];
        let mut res_d = vec![0.0; step_d.len()];
        for _ in 0..self.max_backtransform {
            coords.values_into(&x, x0, &mut q_cur);
            coords.difference_into(&q_cur, q0, &mut diff_q);
            let got_d = chart.project(&diff_q);
            for ((slot, &a), &b) in res_d.iter_mut().zip(step_d).zip(&got_d) {
                *slot = a - b;
            }
            let size = res_d.iter().map(|v| v * v).sum::<f64>().sqrt();
            if size < best.0 {
                best = (size, x.clone());
            } else if size > 2.0 * best.0 {
                break;
            }
            if size < res_tol {
                break;
            }
            let dx = basis.cartesian(&res_d);
            if max_abs(&dx) < 1e-10 {
                break;
            }
            for (xi, d) in x.iter_mut().zip(&dx) {
                *xi += d;
            }
        }
        (best.1, best.0)
    }
}

fn max_abs<'a>(values: impl IntoIterator<Item = &'a f64>) -> f64 {
    values.into_iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

fn bfgs_pair(h: &mut [f64], c: &mut [f64], n: usize, s: &[f64], g_new: &[f64], g_old: &[f64]) {
    let y: Vec<f64> = g_new.iter().zip(g_old).map(|(a, b)| a - b).collect();
    let ys: f64 = y.iter().zip(s).map(|(a, b)| a * b).sum();
    let hs: Vec<f64> = (0..n)
        .map(|r| {
            h[r * n..(r + 1) * n]
                .iter()
                .zip(s)
                .map(|(hrc, sc)| hrc * sc)
                .sum()
        })
        .collect();
    let shs: f64 = hs.iter().zip(s).map(|(a, b)| a * b).sum();
    if ys <= 1e-12 || shs <= 1e-12 {
        return;
    }
    let inv_ys = 1.0 / ys;
    let inv_shs = 1.0 / shs;
    for r in 0..n {
        let yr = y[r] * inv_ys;
        let hsr = hs[r] * inv_shs;
        let row = &mut h[r * n..(r + 1) * n];
        for ((hrc, &yc), &hsc) in row.iter_mut().zip(&y).zip(&hs) {
            *hrc += yr * yc - hsr * hsc;
        }
    }
    let cy: Vec<f64> = (0..n)
        .map(|r| {
            c[r * n..(r + 1) * n]
                .iter()
                .zip(&y)
                .map(|(crc, yc)| crc * yc)
                .sum()
        })
        .collect();
    let ycy: f64 = y.iter().zip(&cy).map(|(yi, cyi)| yi * cyi).sum();
    let beta = inv_ys * (1.0 + inv_ys * ycy);
    for r in 0..n {
        let sr = s[r];
        let cyr = cy[r];
        let row = &mut c[r * n..(r + 1) * n];
        for ((crc, &sc), &cyc) in row.iter_mut().zip(s).zip(&cy) {
            *crc += -inv_ys * (sr * cyc + cyr * sc) + beta * sr * sc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water_dimer() -> (Vec<u32>, Vec<f64>) {
        (
            vec![8, 1, 1, 8, 1, 1],
            vec![
                0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 2.9, 0.1, 0.2, 3.3, 0.9, 0.5, 3.2,
                -0.6, 0.8,
            ],
        )
    }

    #[test]
    fn eigen_reconstructs_a_symmetric_matrix() {
        let n = 5;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = ((i * 7 + j * 3) % 11) as f64 - 5.0;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let (w, v) = symmetric_eigen(&a, n);
        assert!(w.windows(2).all(|p| p[0] <= p[1]));
        for i in 0..n {
            for j in 0..n {
                let rebuilt: f64 = (0..n).map(|k| v[i * n + k] * w[k] * v[j * n + k]).sum();
                assert!((rebuilt - a[i * n + j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn rotation_vector_recovers_a_known_rotation() {
        let theta = [0.3, -0.2, 0.1];
        let angle = norm(theta);
        let axis = scale(theta, 1.0 / angle);
        let rotate = |p: [f64; 3]| {
            let (c, s) = (angle.cos(), angle.sin());
            let k = cross(axis, p);
            let along = scale(axis, dot(axis, p) * (1.0 - c));
            [
                p[0] * c + k[0] * s + along[0],
                p[1] * c + k[1] * s + along[1],
                p[2] * c + k[2] * s + along[2],
            ]
        };
        let p = [[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [-1.0, -1.5, 0.3]];
        let mean = scale(
            [
                p[0][0] + p[1][0] + p[2][0],
                p[0][1] + p[1][1] + p[2][1],
                p[0][2] + p[1][2] + p[2][2],
            ],
            1.0 / 3.0,
        );
        let p: Vec<[f64; 3]> = p.iter().map(|a| sub(*a, mean)).collect();
        let q: Vec<[f64; 3]> = p.iter().map(|a| rotate(*a)).collect();
        let got = rotation_vector(&p, &q);
        for k in 0..3 {
            assert!((got[k] - theta[k]).abs() < 1e-10, "{got:?}");
        }
    }

    #[test]
    fn wilson_rows_match_finite_differences() {
        let (species, mut x) = water_dimer();
        x.extend([4.0, 4.0, 0.0, 4.9, 4.4, 0.6, 5.3, 3.4, 1.0, 5.9, 3.9, 2.0]);
        let (ammonia_species, ammonia) = flat_ammonia();
        x.extend(
            ammonia
                .iter()
                .enumerate()
                .map(|(i, v)| v + if i % 3 == 0 { -5.0 } else { 0.0 }),
        );
        let species: Vec<u32> = species
            .into_iter()
            .chain([6, 1, 6, 1])
            .chain(ammonia_species)
            .collect();
        let tric = Tric::new(&species, ArrayView1::from(&x));
        assert!(
            tric.primitives
                .iter()
                .any(|p| matches!(p, Primitive::Dihedral(..)))
        );
        let b = tric.wilson(&x);
        let n3 = x.len();
        let h = 1e-6;
        for c in 0..n3 {
            let mut up = x.clone();
            let mut down = x.clone();
            up[c] += h;
            down[c] -= h;
            let delta = tric.difference(&tric.values(&up, &x), &tric.values(&down, &x));
            for (r, d) in delta.iter().enumerate() {
                let numeric = d / (2.0 * h);
                assert!(
                    (numeric - b[r * n3 + c]).abs() < 1e-5,
                    "row {r} ({:?}) column {c}: {numeric} against {}",
                    tric.primitives[r],
                    b[r * n3 + c]
                );
            }
        }
    }

    /// A water and a hydroxyl whose second hydrogen has left it.
    fn split_water() -> (Vec<u32>, Vec<f64>) {
        let (species, mut x) = water_dimer();
        x.truncate(9);
        x.extend([3.0, 0.2, 0.1, 3.95, 0.35, -0.05, 2.7, 1.5, 0.4]);
        (species.into_iter().take(3).chain([8, 1, 1]).collect(), x)
    }

    #[test]
    fn a_two_atom_fragment_rotates_as_a_line() {
        let (species, x) = split_water();
        let tric = Tric::new(&species, ArrayView1::from(&x));
        assert!(tric.fragments.iter().any(|f| f.linear && f.atoms == [3, 4]));
        let b = tric.wilson(&x);
        let n3 = x.len();
        let h = 1e-6;
        for c in 0..n3 {
            let mut up = x.clone();
            let mut down = x.clone();
            up[c] += h;
            down[c] -= h;
            let delta = tric.difference(&tric.values(&up, &x), &tric.values(&down, &x));
            for (r, d) in delta.iter().enumerate() {
                let numeric = d / (2.0 * h);
                assert!(
                    (numeric - b[r * n3 + c]).abs() < 1e-5,
                    "row {r} ({:?}) column {c}: {numeric} against {}",
                    tric.primitives[r],
                    b[r * n3 + c]
                );
            }
        }
    }

    /// An ammonia flattened to within 0.05 angstrom of planar.
    fn flat_ammonia() -> (Vec<u32>, Vec<f64>) {
        let mut x = vec![0.0, 0.0, 0.05];
        for k in 0..3 {
            let phi = 2.0 * std::f64::consts::PI * k as f64 / 3.0 + 0.1 * k as f64;
            x.extend([1.01 * phi.cos(), 1.01 * phi.sin(), 0.0]);
        }
        (vec![7, 1, 1, 1], x)
    }

    fn back_transform_residual(species: Vec<u32>, x: &[f64]) -> (f64, f64) {
        let tric = Tric::new(&species, ArrayView1::from(x));
        let (chart, _, _) = Chart::new(&tric, x, 1.0);
        let basis = StepBasis::origin(&chart);
        let step: Vec<f64> = (0..chart.m)
            .map(|k| 0.05 * ((k * 7 % 5) as f64 - 2.0))
            .collect();
        let q0 = tric.values(x, x);
        let dq = chart.primitive(&step);
        let target: Vec<f64> = q0.iter().zip(&dq).map(|(a, b)| a + b).collect();
        let (reached, _) =
            TricOpt::new(species).back_transform(&tric, &chart, &basis, x, &q0, &step);
        let residual = tric.difference(&target, &tric.values(&reached, x));
        let size = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
        (size, dq.iter().map(|v| v * v).sum::<f64>().sqrt())
    }

    #[test]
    fn a_flattened_ammonia_keeps_its_umbrella() {
        let (species, x) = flat_ammonia();
        let tric = Tric::new(&species, ArrayView1::from(&x));
        assert!(
            tric.primitives
                .iter()
                .any(|p| matches!(p, Primitive::Dihedral(_, _, 0, _)))
        );
        let (size, planned) = back_transform_residual(species, &x);
        assert!(size < 1e-3 * planned, "residual {size} of a {planned} step");
    }

    #[test]
    fn a_step_with_a_two_atom_fragment_reaches_its_target() {
        let (species, x) = split_water();
        let tric = Tric::new(&species, ArrayView1::from(&x));
        let (chart, _, _) = Chart::new(&tric, &x, 1.0);
        let basis = StepBasis::origin(&chart);
        let step: Vec<f64> = (0..chart.m)
            .map(|k| 0.05 * ((k * 7 % 5) as f64 - 2.0))
            .collect();
        let q0 = tric.values(&x, &x);
        let dq = chart.primitive(&step);
        let target: Vec<f64> = q0.iter().zip(&dq).map(|(a, b)| a + b).collect();
        let opt = TricOpt::new(species);
        let (reached, _) = opt.back_transform(&tric, &chart, &basis, &x, &q0, &step);
        let residual = tric.difference(&target, &tric.values(&reached, &x));
        let size = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
        let planned = dq.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(size < 1e-3 * planned, "residual {size} of a {planned} step");
    }

    #[test]
    fn a_step_probe_times_one_step() {
        let (species, x) = water_dimer();
        let x = Array1::from_vec(x);
        let gradient = Array1::from_elem(x.len(), 0.1);
        let seconds = TricOpt::new(species).step_seconds(x.view(), -1.0, gradient.view());
        assert!(seconds.is_finite() && seconds > 0.0);
    }

    #[test]
    fn a_zero_step_quench_is_the_raw_energy() {
        let (species, x) = water_dimer();
        let opt = TricOpt::new(species);
        let fg = |v: ArrayView1<f64>| {
            let e = v.iter().map(|c| c * c).sum::<f64>();
            Some((e, v.mapv(|c| 2.0 * c)))
        };
        let (e, y, calls) = opt.minimize(ArrayView1::from(&x), 0, fg);
        assert_eq!(calls, 1);
        assert_eq!(e, x.iter().map(|c| c * c).sum::<f64>());
        assert_eq!(y.as_slice().unwrap(), &x[..]);
    }

    #[test]
    fn a_bend_pulled_straight_still_converges() {
        // O=C=O springs whose rest lengths make the molecule linear, from 140
        // degrees, so the angle primitive degenerates on the way.
        let half = 70.0_f64.to_radians();
        let x = vec![
            -1.16 * half.sin(),
            -1.16 * half.cos(),
            0.0,
            0.0,
            0.0,
            0.0,
            1.16 * half.sin(),
            -1.16 * half.cos(),
            0.1,
        ];
        let rest = [(0, 1, 1.16), (1, 2, 1.16), (0, 2, 2.32)];
        let fg = |v: ArrayView1<f64>| {
            let v = v.as_slice().unwrap();
            let mut e = 0.0;
            let mut g = Array1::zeros(9);
            for &(i, j, r0) in &rest {
                let d = sub(point(v, i), point(v, j));
                let r = norm(d);
                e += 10.0 * (r - r0).powi(2);
                let f = 20.0 * (r - r0) / r;
                for k in 0..3 {
                    g[3 * i + k] += f * d[k];
                    g[3 * j + k] -= f * d[k];
                }
            }
            Some((e, g))
        };
        let mut opt = TricOpt::new(vec![8, 6, 8]);
        opt.gtol = 1e-4;
        let (e, xr, calls) = opt.minimize(ArrayView1::from(&x), 500, fg);
        let (_, g) = fg(xr.view()).unwrap();
        assert!(max_abs(&g) < 1e-4, "stopped at {e} after {calls} calls");
    }

    #[test]
    fn harmonic_bonds_relax_in_few_calls() {
        let (species, reference) = water_dimer();
        let x: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.08 * ((i * 37 % 11) as f64 / 11.0 - 0.5))
            .collect();
        let fg = |v: ArrayView1<f64>| {
            let mut e = 0.0;
            let mut g = Array1::zeros(v.len());
            for i in 0..6 {
                for j in (i + 1)..6 {
                    let d0 = norm(sub(point(&reference, i), point(&reference, j)));
                    let d = sub(
                        point(v.as_slice().unwrap(), i),
                        point(v.as_slice().unwrap(), j),
                    );
                    let r = norm(d);
                    e += (r - d0).powi(2);
                    let f = 2.0 * (r - d0) / r;
                    for k in 0..3 {
                        g[3 * i + k] += f * d[k];
                        g[3 * j + k] -= f * d[k];
                    }
                }
            }
            Some((e, g))
        };
        let mut opt = TricOpt::new(species);
        opt.gtol = 1e-5;
        let (e, x_end, calls) = opt.minimize(ArrayView1::from(&x), 200, fg);
        let (_, g) = fg(x_end.view()).unwrap();
        assert!(max_abs(&g) < 1e-5, "gradient {}", max_abs(&g));
        assert!(e < 1e-8, "energy {e}");
        assert!(calls < 200, "{calls} calls");
    }

    #[test]
    fn a_continued_descent_matches_one_unbroken_pass() {
        let (species, reference) = water_dimer();
        let x: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.12 * ((i * 37 % 11) as f64 / 11.0 - 0.5))
            .collect();
        let fg = |v: ArrayView1<f64>| {
            let mut e = 0.0;
            let mut g = Array1::zeros(v.len());
            for i in 0..6 {
                for j in (i + 1)..6 {
                    let d0 = norm(sub(point(&reference, i), point(&reference, j)));
                    let d = sub(
                        point(v.as_slice().unwrap(), i),
                        point(v.as_slice().unwrap(), j),
                    );
                    let r = norm(d);
                    e += (r - d0).powi(2);
                    let f = 2.0 * (r - d0) / r;
                    for k in 0..3 {
                        g[3 * i + k] += f * d[k];
                        g[3 * j + k] -= f * d[k];
                    }
                }
            }
            Some((e, g))
        };
        let mut unbroken = TricOpt::new(species.clone());
        unbroken.gtol = 1e-6;
        let (e_one, x_one, calls_one) = unbroken.minimize(ArrayView1::from(&x), 200, fg);

        let mut split = TricOpt::new(species);
        split.gtol = 1e-6;
        let (_, x_mid, calls_a) = split.minimize(ArrayView1::from(&x), 5, fg);
        assert_eq!(calls_a, 5);
        let (e_two, x_two, calls_b) = split.minimize(x_mid.view(), 200, fg);
        // `calls_b` counts the opening evaluation at `x_mid` (which `warm_relax_with`
        // answers from its paid cache without charging the potential).
        assert_eq!(calls_a + calls_b - 1, calls_one);
        assert!((e_one - e_two).abs() < 1e-14);
        assert_eq!(x_one, x_two);
    }
}
