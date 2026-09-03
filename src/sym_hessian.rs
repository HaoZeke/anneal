//! Two uses of a structure's symmetry for its curvature.
//!
//! Both start from the same fact. A Hessian is a property of the geometry, so
//! any rigid motion plus relabelling that carries one geometry onto another
//! carries the curvature with it. Where that motion is a symmetry of a single
//! structure the Hessian commutes with it and block-diagonalises; where it
//! relates two different structures it transports curvature between them.
//!
//! # Block-diagonalisation
//!
//! Let `G` be a set of rotations that map the structure onto itself, each with
//! its site permutation `sigma_R`. The representation on displacement space is
//!
//! ```text
//! (D(R) v)_{sigma_R(i)} = R v_i
//! ```
//!
//! which is orthogonal, being a permutation of rotation blocks. The model
//! Hessian depends only on pair separations and pair directions, both of which
//! `D(R)` carries consistently, so `H D(R) = D(R) H` exactly. The average
//! `P = (1/|G|) sum_R D(R)` is then an orthogonal projector commuting with `H`,
//! and the quadratic form splits with no error at all:
//!
//! ```text
//! g^T H^-1 g = (P g)^T H^-1 (P g) + (Q g)^T H^-1 (Q g),   Q = I - P
//! ```
//!
//! Verified symbolically against a C3v arrangement: `||P^2 - P|| = 8e-17`,
//! `max ||H D(R) - D(R) H|| = 6e-16`, and the split residual on the depth is
//! exactly zero in double precision.
//!
//! ## What the split buys, measured
//!
//! No serial work, and a shorter critical path nobody needs. From
//! `examples/sym_block_cg.rs`, solving to a relative residual of `1e-10`,
//! averaged over ten right-hand sides:
//!
//! | structure                   | `\|G\|` | blocks | plain | sum of blocks | busiest block |
//! |-----------------------------|---------|--------|-------|---------------|---------------|
//! | icosahedron, Ih             | 120     | 8      | 9.0   | 9.0           | 2.0           |
//! | cuboctahedron + centre, Oh  | 48      | 9      | 15.0  | 15.0          | 3.0           |
//! | truncated octahedron, Oh    | 48      | 10     | 21.8  | 29.0          | 5.0           |
//! | random blob, 12 points, C1  | 1       | 1      | 34.0  | 34.0          | 34.0          |
//! | random blob, 24 points, C1  | 1       | 1      | 46.7  | 46.7          | 46.7          |
//!
//! The depths sum to the plain depth to `7e-14`, so the decomposition is right.
//! It is just not cheaper. Serial cost is unchanged on two of the three
//! symmetric cases and 1.33 times worse on the third, because a Krylov space
//! that has to be restarted per block pays its first application again in every
//! one. The critical path does fall, by 4.5 to 5.0 times, which would matter on
//! a machine with one core per block; a cluster search has thousands of
//! independent hops to spend cores on first, and spending eight of them to save
//! seven operator applications is not the trade anyone is short of.
//!
//! The condition number is what would have to move and does not: 93 on the
//! worst block of an icosahedron against 150 on the whole operator, 101 against
//! 122 for a truncated octahedron. An operator whose spectrum is this flat has
//! no room, and conjugate gradients already exploits clustering without being
//! told where the clusters are.
//!
//! The C1 control does exactly nothing, to the bit: same block, same iteration
//! count, split error `0.0`.
//!
//! # Transport by correspondence
//!
//! Given a permutation `p` and a rotation `R` taking a target structure onto a
//! reference, with `(M v)_{slot} = R v_{p(slot)}`, the reference curvature acts
//! at the target as `M^T H_A M`. For an exactly related pair this is not an
//! approximation: `||H_B - M^T H_A M|| = 3e-16` on an eight-point test, and
//! [`Transport`] reproduces that through the IRA correspondence rather than a
//! constructed one.
//!
//! What is worth transporting is not the model Hessian, which is free to
//! rebuild anywhere, but the correction `H_true - H_model` measured once at a
//! relaxed minimum. At a 38-point Lennard-Jones minimum that correction carries
//! `||H_true - H_model||_F / ||H_true||_F = 0.977`: the stretch-only operator
//! supplies two per cent of the curvature and the correction is nearly the whole
//! Hessian.
//!
//! ## Transfer quality against IRA distance
//!
//! From `examples/curvature_transport.rs`, 240 targets around a relaxed LJ38
//! minimum, as `||M^T H_A M - H_y||_F / ||H_y||_F`:
//!
//! | IRA distance | transported | identity correspondence | model Hessian |
//! |--------------|-------------|-------------------------|---------------|
//! | 0.0013-0.0042| 0.016       | 0.016                   | 0.977         |
//! | 0.0042-0.0122| 0.051       | 0.051                   | 0.977         |
//! | 0.0124-0.0532| 0.214       | 0.214                   | 0.979         |
//! | 0.0545-0.1653| 0.611       | 0.611                   | 0.989         |
//! | 0.168-1.645  | 0.915       | 0.911                   | 0.998         |
//! | 1.649-2.359  | 0.725-0.888 | 0.786-0.886             | 0.981-0.996   |
//!
//! Transport is mechanically sound and degrades about linearly, near `6 d` for
//! `d < 0.1`, saturating past `d ~ 0.15`. It beats rebuilding the model at every
//! distance measured, which is the result the operator norm supports.
//!
//! ## Why it is not worth wiring in anyway
//!
//! Two numbers kill it, and neither is about the transport being wrong.
//!
//! The correspondence is free exactly where it works. Below an IRA distance of
//! 0.17 the transported and the identity correspondences agree to three decimal
//! places, because a small perturbation leaves IRA returning the identity
//! permutation and a near-identity rotation. Where IRA does contribute, past a
//! distance of 1.6, it moves the operator error from 0.820 to 0.725 on an
//! operator that is already three quarters wrong. At 0.41 ms a call, against a
//! measured 52 charged evaluations per hop for IRA where the cheap descriptor
//! costs 31, that is a bill with nothing behind it.
//!
//! And the depth does not want a better Hessian. Scored after the one global
//! scale the downstream regression absorbs, the median relative depth error
//! over the same 240 targets runs: exact Hessian at the target 0.103, free model
//! Hessian 0.159, plain `1/2 |g|^2` 0.189, transported 0.557. Spearman rank
//! correlation with the true depth is 0.999 for the model and for the gradient
//! norm, 0.991 for the exact Hessian. A Hessian with no transport error at all
//! buys 0.06 in median error and loses rank correlation, which says the depth
//! prediction is limited by anharmonicity and not by the curvature model. There
//! is nothing here for a better operator to win.
//!
//! ## Why the truer operator predicts worse
//!
//! The depth inverts the operator, and the two differ in exactly the place an
//! inverse is most sensitive. Off the translations, which is all
//! [`crate::model_hessian::depth`] deflates, the four softest modes of the exact
//! Hessian at the LJ38 reference are `-4.3e-8, 3.1e-8, 1.1e-7, 3.3`: the three
//! rigid rotations survive, so the operator the solve is asked to invert is
//! singular. The model's four are `0.05, 0.05, 0.05, 1.1`, the first three being
//! `FLOOR * K0` exactly. The shift that [`crate::model_hessian::FLOOR`] carries
//! for the bending terms the stretch-only form omits is also what keeps the
//! rotations out of the denominator, and that is what makes the crude operator
//! the better-behaved one here. Transporting the true curvature transports the
//! singularity with it.
//!
//! The machinery is kept because it is correct and the measurement is the point:
//! [`Transport`] and [`TransportedCurvature`] carry curvature exactly, and the
//! table above tells a caller with a different quantity in mind, one that does
//! not invert the operator, where the correspondence stops being usable.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::model_hessian;
use crate::potentials::PairKind;
use crate::symmetrise::Rot;

/// A group element together with the site permutation it induces.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteSymmetry {
    /// Rotation about the structure's centroid, row-major.
    pub rotation: Rot,
    /// `image[i]` is the site the operation carries site `i` onto.
    pub image: Vec<usize>,
    /// Largest distance between a rotated point and the site it was matched to.
    ///
    /// Zero for an exact symmetry; this is what decides whether the operation is
    /// admitted at all.
    pub residual: f64,
}

/// Group elements that carry the structure onto itself, with their site maps.
///
/// An element is admitted only when every rotated point lands within `tol` of
/// some site and the resulting map is a bijection. Both conditions matter: a
/// near-miss axis produces a map that sends two points to the same site, and
/// the average over such a set is not a projector.
///
/// Filtering is safe because the admitted elements form a subgroup. If `R` and
/// `S` each permute the sites exactly then so does `RS`, with site map
/// `sigma_R sigma_S`, so the average over what survives the filter still
/// satisfies the rearrangement theorem. That is what lets this be used on the
/// output of an approximate detector without checking closure again.
pub fn site_symmetries(x: ArrayView1<f64>, n: usize, group: &[Rot], tol: f64) -> Vec<SiteSymmetry> {
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    let mut centre = [0.0f64; 3];
    for i in 0..n {
        for k in 0..3 {
            centre[k] += x[3 * i + k];
        }
    }
    for c in centre.iter_mut() {
        *c /= n as f64;
    }
    for r in group {
        let mut image = vec![usize::MAX; n];
        let mut taken = vec![false; n];
        let mut worst = 0.0f64;
        let mut ok = true;
        for i in 0..n {
            let p = [
                x[3 * i] - centre[0],
                x[3 * i + 1] - centre[1],
                x[3 * i + 2] - centre[2],
            ];
            let q = [
                r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + centre[0],
                r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + centre[1],
                r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + centre[2],
            ];
            let mut best = f64::INFINITY;
            let mut best_j = usize::MAX;
            for j in 0..n {
                let d: f64 = (0..3)
                    .map(|k| {
                        let v = q[k] - x[3 * j + k];
                        v * v
                    })
                    .sum();
                if d < best {
                    best = d;
                    best_j = j;
                }
            }
            let d = best.sqrt();
            if d > tol || best_j == usize::MAX || taken[best_j] {
                ok = false;
                break;
            }
            taken[best_j] = true;
            image[i] = best_j;
            worst = worst.max(d);
        }
        if ok {
            out.push(SiteSymmetry {
                rotation: *r,
                image,
                residual: worst,
            });
        }
    }
    out
}

impl SiteSymmetry {
    /// The composition `self . other`, whose site map is `self.image` after
    /// `other.image` and whose rotation is the matrix product.
    ///
    /// Follows from the representation: `(D(A) D(B) v)_{a(b(i))} = A B v_i`.
    pub fn compose(&self, other: &Self) -> Self {
        let mut rotation = [[0.0f64; 3]; 3];
        for (i, row) in rotation.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3)
                    .map(|k| self.rotation[i][k] * other.rotation[k][j])
                    .sum();
            }
        }
        let image = other.image.iter().map(|&j| self.image[j]).collect();
        Self {
            rotation,
            image,
            residual: self.residual.max(other.residual),
        }
    }

    /// The inverse element: the transposed rotation and the inverted site map.
    pub fn inverse(&self) -> Self {
        let mut rotation = [[0.0f64; 3]; 3];
        for (i, row) in rotation.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = self.rotation[j][i];
            }
        }
        let mut image = vec![0usize; self.image.len()];
        for (i, &j) in self.image.iter().enumerate() {
            image[j] = i;
        }
        Self {
            rotation,
            image,
            residual: self.residual,
        }
    }

    /// Whether two elements are the same operation.
    pub fn matches(&self, other: &Self) -> bool {
        self.image == other.image
            && (0..3)
                .all(|i| (0..3).all(|j| (self.rotation[i][j] - other.rotation[i][j]).abs() < 1e-6))
    }
}

/// Conjugacy classes of the site group, as index lists into `sym`.
///
/// The class sums span the centre of the commutant, which is what makes the
/// isotypic decomposition reachable without a character table. Costs `|G|^2`
/// element comparisons, which at the sixty rotations of an icosahedral group is
/// nothing next to one operator application.
pub fn conjugacy_classes(sym: &[SiteSymmetry]) -> Vec<Vec<usize>> {
    let m = sym.len();
    let mut assigned = vec![usize::MAX; m];
    let mut classes: Vec<Vec<usize>> = Vec::new();
    for r in 0..m {
        if assigned[r] != usize::MAX {
            continue;
        }
        let id = classes.len();
        let mut members = Vec::new();
        for t in sym {
            let conj = t.compose(&sym[r]).compose(&t.inverse());
            if let Some(k) = sym.iter().position(|s| s.matches(&conj))
                && assigned[k] == usize::MAX
            {
                assigned[k] = id;
                members.push(k);
            }
        }
        members.sort_unstable();
        classes.push(members);
    }
    classes
}

/// Orthonormal bases of the isotypic components, as columns of each entry.
///
/// The model Hessian lies in the commutant of the representation, and the
/// centre of the commutant is spanned by the class sums. A generic combination
/// of the class sums therefore has exactly the isotypic components as its
/// eigenspaces, and any operator in the commutant is block diagonal across them.
/// That is the whole block-diagonalisation, not only its totally symmetric
/// piece: the totally symmetric block is the isotypic component of the trivial
/// representation and is one of these.
///
/// `seed` picks the combination. Only a measure-zero set of coefficients fails
/// to separate the components, and an accidental degeneracy shows up as two
/// components fused into one, which costs sharpness rather than correctness:
/// the fused space is still invariant.
pub fn isotypic_bases(sym: &[SiteSymmetry], n: usize, seed: u64) -> Vec<Array2<f64>> {
    let dim = 3 * n;
    if sym.len() <= 1 {
        return vec![Array2::<f64>::eye(dim)];
    }
    let classes = conjugacy_classes(sym);
    let mut rng = seed | 1;
    let mut coeff = Vec::with_capacity(classes.len());
    for _ in 0..classes.len() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        coeff.push((rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
    }
    // The central element, as a dense matrix. Formed column by column through
    // the same site action the projector uses, so there is one definition of
    // `D(R)` in this module and not two.
    let mut z = Array2::<f64>::zeros((dim, dim));
    let mut e = Array1::<f64>::zeros(dim);
    for c in 0..dim {
        e[c] = 1.0;
        for (k, members) in classes.iter().enumerate() {
            for &idx in members {
                let s = &sym[idx];
                for (i, &img) in s.image.iter().enumerate() {
                    for a in 0..3 {
                        z[[3 * img + a, c]] += coeff[k]
                            * (s.rotation[a][0] * e[3 * i]
                                + s.rotation[a][1] * e[3 * i + 1]
                                + s.rotation[a][2] * e[3 * i + 2]);
                    }
                }
            }
        }
        e[c] = 0.0;
    }
    // A central element of a real orthogonal representation's commutant is
    // symmetric up to rounding; symmetrised so the Jacobi solver is given what
    // it assumes.
    let zt = z.t().to_owned();
    z = (&z + &zt) * 0.5;
    let (vals, vecs) = crate::spectral::symmetric_eigen(z.view(), 128);
    let spread = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let gap = 1e-8 * spread.max(1.0);
    let mut out: Vec<Array2<f64>> = Vec::new();
    let mut start = 0usize;
    for i in 1..=dim {
        if i == dim || (vals[i] - vals[start]).abs() > gap {
            let mut b = Array2::<f64>::zeros((dim, i - start));
            for (c, col) in (start..i).enumerate() {
                for r in 0..dim {
                    b[[r, c]] = vecs[[r, col]];
                }
            }
            out.push(b);
            start = i;
        }
    }
    out
}

/// Averages `v` over the site symmetries: the projector onto the totally
/// symmetric subspace.
///
/// `P v = (1/|G|) sum_R D(R) v`. Idempotent and symmetric when `sym` is closed,
/// which [`site_symmetries`] guarantees.
pub fn project_symmetric(sym: &[SiteSymmetry], v: ArrayView1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(v.len());
    if sym.is_empty() {
        out.assign(&v);
        return out;
    }
    for s in sym {
        for (i, &img) in s.image.iter().enumerate() {
            for k in 0..3 {
                out[3 * img + k] += s.rotation[k][0] * v[3 * i]
                    + s.rotation[k][1] * v[3 * i + 1]
                    + s.rotation[k][2] * v[3 * i + 2];
            }
        }
    }
    out /= sym.len() as f64;
    out
}

/// Cost and outcome of one conjugate-gradient solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgReport {
    /// Operator applications used.
    pub iterations: usize,
    /// `|r| / |b|` at the last iterate.
    pub relative_residual: f64,
    /// `1/2 b^T d` for the solve's own right-hand side, the depth it predicts.
    pub quadratic_form: f64,
}

/// Conjugate gradients on the model Hessian, with `filter` applied to every
/// iterate and to every operator product.
///
/// The filter is how an invariant subspace is enforced. In exact arithmetic a
/// Krylov space started inside an invariant subspace stays there, and in
/// floating point it does not; the same re-projection appears in
/// [`crate::curvature`] for the rigid modes and for the same reason.
///
/// Stops on `|r| <= tol |b|` rather than on an iteration count, so the
/// iteration count is the measured quantity.
pub fn cg_filtered<F>(
    x: ArrayView1<f64>,
    n: usize,
    b: ArrayView1<f64>,
    tol: f64,
    max_iters: usize,
    mut filter: F,
) -> CgReport
where
    F: FnMut(&mut Array1<f64>),
{
    let scale = model_hessian::spacing(x, n);
    let mut rhs = b.to_owned();
    filter(&mut rhs);
    let b_norm = rhs.dot(&rhs).sqrt();
    if b_norm.is_nan() || b_norm <= 0.0 {
        return CgReport {
            iterations: 0,
            relative_residual: 0.0,
            quadratic_form: 0.0,
        };
    }
    let mut d = Array1::<f64>::zeros(3 * n);
    let mut r = rhs.clone();
    let mut p = r.clone();
    let mut rr = r.dot(&r);
    let mut used = 0usize;
    for _ in 0..max_iters {
        let mut ap = model_hessian::apply(x, n, p.view(), scale);
        filter(&mut ap);
        used += 1;
        let pap = p.dot(&ap);
        if pap.is_nan() || pap <= 1e-300 {
            break;
        }
        let alpha = rr / pap;
        d.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        let rr_new = r.dot(&r);
        if rr_new.sqrt() <= tol * b_norm {
            rr = rr_new;
            break;
        }
        let beta = rr_new / rr;
        p = &r + &(p.mapv(|v| v * beta));
        rr = rr_new;
    }
    CgReport {
        iterations: used,
        relative_residual: rr.sqrt() / b_norm,
        quadratic_form: 0.5 * rhs.dot(&d),
    }
}

/// Removes the translational component, which the operator annihilates.
fn deflate(v: &mut Array1<f64>, n: usize) {
    if n == 0 {
        return;
    }
    let mut mean = [0.0f64; 3];
    for i in 0..n {
        for k in 0..3 {
            mean[k] += v[3 * i + k];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    for i in 0..n {
        for k in 0..3 {
            v[3 * i + k] -= mean[k];
        }
    }
}

/// The plain solve set against the symmetry-split one.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitReport {
    /// One solve on the whole space.
    pub plain: CgReport,
    /// The solve confined to the totally symmetric subspace.
    pub symmetric: CgReport,
    /// The solve confined to its orthogonal complement.
    pub remainder: CgReport,
    /// Group elements that carried the structure onto itself.
    pub group_order: usize,
    /// Relative disagreement between the plain depth and the sum of the blocks.
    ///
    /// Zero in exact arithmetic, since the two subspaces are invariant. A number
    /// far from zero says the element set was not closed, so the projector was
    /// not a projector.
    pub depth_error: f64,
}

impl SplitReport {
    /// Operator applications the split solve costs, against the plain one.
    ///
    /// Above one, the split is more expensive.
    pub fn work_ratio(&self) -> f64 {
        let plain = self.plain.iterations.max(1) as f64;
        (self.symmetric.iterations + self.remainder.iterations) as f64 / plain
    }
}

/// Solves for the depth twice: once plainly, once split by symmetry.
///
/// Both arms deflate the translations, which the operator annihilates, so the
/// comparison is between two solves of the same nonsingular problem.
pub fn split_depth(
    x: ArrayView1<f64>,
    n: usize,
    gradient: ArrayView1<f64>,
    sym: &[SiteSymmetry],
    tol: f64,
    max_iters: usize,
) -> SplitReport {
    let plain = cg_filtered(x, n, gradient, tol, max_iters, |v| deflate(v, n));

    // Projected but not deflated here: `cg_filtered` runs the filter over its own
    // right-hand side, and deflating twice moves it by a rounding-sized amount
    // that shows up as one extra iteration at a tolerance near 1e-10. On a
    // structure with no symmetry the projector is exactly the identity, so
    // leaving the deflation to the filter makes the two arms bitwise identical
    // rather than nearly so, which is what the control case has to check.
    let g_sym = project_symmetric(sym, gradient);
    let symmetric = cg_filtered(x, n, g_sym.view(), tol, max_iters, |v| {
        let mut w = project_symmetric(sym, v.view());
        deflate(&mut w, n);
        v.assign(&w);
    });

    let mut g_rem = gradient.to_owned();
    g_rem -= &g_sym;
    let remainder = cg_filtered(x, n, g_rem.view(), tol, max_iters, |v| {
        let s = project_symmetric(sym, v.view());
        *v -= &s;
        deflate(v, n);
    });

    let total = symmetric.quadratic_form + remainder.quadratic_form;
    let depth_error = if plain.quadratic_form.abs() > 0.0 {
        (total - plain.quadratic_form).abs() / plain.quadratic_form.abs()
    } else {
        0.0
    };
    SplitReport {
        plain,
        symmetric,
        remainder,
        group_order: sym.len(),
        depth_error,
    }
}

/// The plain solve set against the fully block-diagonalised one.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockReport {
    /// One solve on the whole space.
    pub plain: CgReport,
    /// One solve per isotypic component, in the order the bases came in.
    pub blocks: Vec<CgReport>,
    /// Dimension of each block before the translations are deflated.
    pub block_dims: Vec<usize>,
    /// Relative disagreement between the plain depth and the sum of the blocks.
    pub depth_error: f64,
}

impl BlockReport {
    /// Operator applications the block solve costs, against the plain one.
    pub fn work_ratio(&self) -> f64 {
        let plain = self.plain.iterations.max(1) as f64;
        self.blocks.iter().map(|b| b.iterations).sum::<usize>() as f64 / plain
    }

    /// Applications on the busiest block, which is what a machine with one
    /// core per block would wait for.
    pub fn critical_path(&self) -> usize {
        self.blocks.iter().map(|b| b.iterations).max().unwrap_or(0)
    }
}

/// Solves for the depth twice: once plainly, once on each isotypic block.
///
/// The blocks partition the space, so the depths sum to the plain depth exactly
/// and the comparison is between two routes to the same number.
pub fn block_depth(
    x: ArrayView1<f64>,
    n: usize,
    gradient: ArrayView1<f64>,
    bases: &[Array2<f64>],
    tol: f64,
    max_iters: usize,
) -> BlockReport {
    let plain = cg_filtered(x, n, gradient, tol, max_iters, |v| deflate(v, n));
    let mut blocks = Vec::with_capacity(bases.len());
    let mut dims = Vec::with_capacity(bases.len());
    let mut total = 0.0;
    for b in bases {
        // `B B^T` is the orthogonal projector onto the block, applied without
        // forming the dense projector: two thin products at `O(dim * m)` each.
        let rhs = b.dot(&b.t().dot(&gradient));
        let rep = cg_filtered(x, n, rhs.view(), tol, max_iters, |v| {
            let w = b.dot(&b.t().dot(&*v));
            v.assign(&w);
            deflate(v, n);
        });
        total += rep.quadratic_form;
        dims.push(b.ncols());
        blocks.push(rep);
    }
    let depth_error = if plain.quadratic_form.abs() > 0.0 {
        (total - plain.quadratic_form).abs() / plain.quadratic_form.abs()
    } else {
        0.0
    };
    BlockReport {
        plain,
        blocks,
        block_dims: dims,
        depth_error,
    }
}

/// The model Hessian as a dense matrix, for spectral diagnosis.
///
/// Materialised only to read condition numbers off. Everything the solver does
/// stays matrix free; a `3n` by `3n` matrix at the sizes a cluster search
/// visits is a diagnostic, not a data structure to carry.
pub fn dense_model(x: ArrayView1<f64>, n: usize) -> Array2<f64> {
    let scale = model_hessian::spacing(x, n);
    let mut h = Array2::<f64>::zeros((3 * n, 3 * n));
    let mut e = Array1::<f64>::zeros(3 * n);
    for c in 0..(3 * n) {
        e[c] = 1.0;
        let col = model_hessian::apply(x, n, e.view(), scale);
        for r in 0..(3 * n) {
            h[[r, c]] = col[r];
        }
        e[c] = 0.0;
    }
    h
}

/// Orthonormal basis of the totally symmetric subspace, as columns.
///
/// Built by diagonalising the projector rather than by orthogonalising its
/// columns: the eigenvalues are exactly one and zero, so the dimension of the
/// subspace is read off rather than chosen by a rank tolerance on a
/// Gram-Schmidt sweep.
pub fn symmetric_basis(sym: &[SiteSymmetry], n: usize) -> Array2<f64> {
    let dim = 3 * n;
    let mut p = Array2::<f64>::zeros((dim, dim));
    let mut e = Array1::<f64>::zeros(dim);
    for c in 0..dim {
        e[c] = 1.0;
        let col = project_symmetric(sym, e.view());
        for r in 0..dim {
            p[[r, c]] = col[r];
        }
        e[c] = 0.0;
    }
    // Symmetrised before diagonalising. The projector is symmetric in exact
    // arithmetic and the Jacobi routine expects that; the average over the group
    // leaves rounding of order 1e-16 in the asymmetry, which is harmless here and
    // undefined behaviour for the solver.
    let pt = p.t().to_owned();
    p = (&p + &pt) * 0.5;
    let (vals, vecs) = crate::spectral::symmetric_eigen(p.view(), 128);
    let keep: Vec<usize> = (0..dim).filter(|&i| vals[i] > 0.5).collect();
    let mut basis = Array2::<f64>::zeros((dim, keep.len()));
    for (c, &i) in keep.iter().enumerate() {
        for r in 0..dim {
            basis[[r, c]] = vecs[[r, i]];
        }
    }
    basis
}

/// Eigenvalues of `h` restricted to the span of `basis`, ascending.
///
/// `basis` holds orthonormal columns, so the restriction `B^T H B` carries the
/// eigenvalues of `H` on that subspace whenever the subspace is invariant. The
/// low end is the part worth looking at: the depth solve inverts this operator,
/// so a mode near zero is what decides whether the answer means anything.
pub fn restricted_spectrum(h: ArrayView2<f64>, basis: ArrayView2<f64>) -> Vec<f64> {
    if basis.ncols() == 0 {
        return Vec::new();
    }
    let small = basis.t().dot(&h).dot(&basis);
    let (vals, _) = crate::spectral::symmetric_eigen(small.view(), 128);
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Spectral condition number of `h` restricted to the span of `basis`.
///
/// `basis` holds orthonormal columns; the restriction is `B^T H B`, whose
/// eigenvalues are the eigenvalues of `H` on that subspace when the subspace is
/// invariant. Returns `None` for an empty subspace or one on which the operator
/// is singular.
pub fn restricted_condition(h: ArrayView2<f64>, basis: ArrayView2<f64>) -> Option<f64> {
    let m = basis.ncols();
    if m == 0 {
        return None;
    }
    let small = basis.t().dot(&h).dot(&basis);
    let (vals, _) = crate::spectral::symmetric_eigen(small.view(), 128);
    let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if lo.is_nan() || lo <= 0.0 {
        return None;
    }
    Some(hi / lo)
}

/// Orthonormal basis of the translations, which the operator annihilates.
///
/// Excluded from every condition number here, since including them reports the
/// conditioning of a problem nobody solves: the depth solve deflates them.
fn translation_basis(n: usize) -> Array2<f64> {
    let dim = 3 * n;
    let mut b = Array2::<f64>::zeros((dim, 3));
    let s = 1.0 / (n as f64).sqrt();
    for i in 0..n {
        for k in 0..3 {
            b[[3 * i + k, k]] = s;
        }
    }
    b
}

/// Orthonormal basis of the complement of the translations inside `basis`.
///
/// Modified Gram-Schmidt against the translations, then a second pass to drop
/// what the removal left below rounding.
fn without_translations(basis: ArrayView2<f64>, n: usize) -> Array2<f64> {
    let t = translation_basis(n);
    let dim = basis.nrows();
    let mut cols: Vec<Array1<f64>> = Vec::new();
    for c in 0..basis.ncols() {
        let mut v = basis.column(c).to_owned();
        for k in 0..t.ncols() {
            let tk = t.column(k);
            let d = v.dot(&tk);
            v.scaled_add(-d, &tk);
        }
        for u in &cols {
            let d = v.dot(u);
            v.scaled_add(-d, u);
        }
        let nrm = v.dot(&v).sqrt();
        if nrm > 1e-8 {
            v /= nrm;
            cols.push(v);
        }
    }
    let mut out = Array2::<f64>::zeros((dim, cols.len()));
    for (c, v) in cols.iter().enumerate() {
        for r in 0..dim {
            out[[r, c]] = v[r];
        }
    }
    out
}

/// Condition numbers of the model Hessian on the whole space and on the
/// totally symmetric subspace, translations removed from both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conditioning {
    /// Condition number on everything but the translations.
    pub full: f64,
    /// Condition number on the totally symmetric subspace.
    pub symmetric: f64,
    /// Dimension of the totally symmetric subspace, translations removed.
    pub symmetric_dim: usize,
    /// Dimension solved on without the split.
    pub full_dim: usize,
}

/// Measures how much the symmetry projection improves the conditioning.
///
/// `None` when the symmetric subspace collapses to the translations alone,
/// which is what a structure with no symmetry gives and which carries no
/// solve at all.
pub fn conditioning(x: ArrayView1<f64>, n: usize, sym: &[SiteSymmetry]) -> Option<Conditioning> {
    let h = dense_model(x, n);
    let dim = 3 * n;
    let all = Array2::<f64>::eye(dim);
    let full_basis = without_translations(all.view(), n);
    let sym_basis = without_translations(symmetric_basis(sym, n).view(), n);
    let full = restricted_condition(h.view(), full_basis.view())?;
    let symmetric = restricted_condition(h.view(), sym_basis.view())?;
    Some(Conditioning {
        full,
        symmetric,
        symmetric_dim: sym_basis.ncols(),
        full_dim: full_basis.ncols(),
    })
}

/// The exact Hessian of a pairwise cluster potential, dense.
///
/// ```text
/// A_ij = V''(r) u u^T + (V'(r)/r) (I - u u^T)
/// H_ij = -A_ij  (i != j),   H_ii = sum_{j != i} A_ij
/// ```
///
/// Verified against symbolic differentiation of `4 (r^-12 - r^-6)`: the largest
/// entry-wise residual is 4e-17.
///
/// This is the expensive object transport exists to avoid rebuilding. Forming
/// it here costs `O(N^2)` arithmetic and no potential evaluations because the
/// pair form is known in closed form; a caller holding only gradients pays
/// `6N` of them to finite-difference the same matrix, which is 228 evaluations
/// at 38 points and 450 at 75.
pub fn pair_hessian(x: ArrayView1<f64>, n: usize, kind: PairKind) -> Array2<f64> {
    let mut h = Array2::<f64>::zeros((3 * n, 3 * n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0f64; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                d[k] = x[3 * i + k] - x[3 * j + k];
                r2 += d[k] * d[k];
            }
            if r2 <= 0.0 {
                continue;
            }
            let (vpp, tension) = kind.pair_curvature(r2);
            let inv_r2 = 1.0 / r2;
            for a in 0..3 {
                for b in 0..3 {
                    let uu = d[a] * d[b] * inv_r2;
                    let delta = if a == b { 1.0 } else { 0.0 };
                    let block = vpp * uu + tension * (delta - uu);
                    h[[3 * i + a, 3 * i + b]] += block;
                    h[[3 * j + a, 3 * j + b]] += block;
                    h[[3 * i + a, 3 * j + b]] -= block;
                    h[[3 * j + a, 3 * i + b]] -= block;
                }
            }
        }
    }
    h
}

/// A correspondence carrying displacements at a target onto a reference frame.
///
/// `source[slot]` is the target site occupying reference slot `slot`, and
/// `rotation` takes target coordinates onto reference coordinates. Together they
/// are the orthogonal map
///
/// ```text
/// (M v)_slot = R v_{source[slot]}
/// ```
///
/// which is the convention [`crate::shape::match_shapes`] already returns, no
/// transposes and no index shifts: the library reorders its own output to
/// C-style before returning.
#[derive(Debug, Clone, PartialEq)]
pub struct Transport {
    source: Vec<usize>,
    rotation: [f64; 9],
}

impl Transport {
    /// A transport from a permutation and a row-major rotation.
    ///
    /// `None` when the permutation is not a bijection, which is a correspondence
    /// that puts two target sites in one reference slot and no orthogonal map at
    /// all. IRA returns such a thing on occasion and reports success, so the
    /// check belongs here rather than in the caller.
    pub fn new(source: Vec<usize>, rotation: [f64; 9]) -> Option<Self> {
        let n = source.len();
        let mut seen = vec![false; n];
        for &s in &source {
            if s >= n || seen[s] {
                return None;
            }
            seen[s] = true;
        }
        Some(Self { source, rotation })
    }

    /// Sites the correspondence covers.
    pub fn n_points(&self) -> usize {
        self.source.len()
    }

    /// The correspondence IRA reports between a reference and a target.
    ///
    /// `m` must come from `match_shapes(reference, target)`, in that order.
    #[cfg(feature = "ira")]
    pub fn from_match(m: &crate::shape::Match) -> Option<Self> {
        Self::new(m.permutation.clone()?, m.rotation)
    }

    /// `M v`: a target displacement expressed in the reference's frame and order.
    pub fn to_reference(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let n = self.source.len();
        let mut out = Array1::<f64>::zeros(3 * n);
        for (slot, &src) in self.source.iter().enumerate() {
            for k in 0..3 {
                out[3 * slot + k] = self.rotation[3 * k] * v[3 * src]
                    + self.rotation[3 * k + 1] * v[3 * src + 1]
                    + self.rotation[3 * k + 2] * v[3 * src + 2];
            }
        }
        out
    }

    /// `M^T w`: a reference-frame vector brought back to the target's frame.
    pub fn from_reference(&self, w: ArrayView1<f64>) -> Array1<f64> {
        let n = self.source.len();
        let mut out = Array1::<f64>::zeros(3 * n);
        for (slot, &src) in self.source.iter().enumerate() {
            for k in 0..3 {
                out[3 * src + k] = self.rotation[k] * w[3 * slot]
                    + self.rotation[3 + k] * w[3 * slot + 1]
                    + self.rotation[6 + k] * w[3 * slot + 2];
            }
        }
        out
    }
}

/// Curvature measured once at a reference structure and carried elsewhere.
///
/// Holds the correction `H_true - H_model` rather than `H_true`. Transporting
/// the whole Hessian would be pointless: the model part is a function of the
/// geometry and costs nothing to rebuild at the target, so carrying it across
/// can only introduce the error of an imperfect correspondence into a term that
/// had none. What the model cannot supply is the part the stretch-only form
/// omits, chiefly the transverse tension `V'(r)/r`, and that is what travels.
///
/// Dense, `3N` by `3N`. At 38 points that is 114 by 114, about 104 kilobytes;
/// at 75 points, 405 kilobytes. One per reference minimum, not one per hop.
#[derive(Debug, Clone)]
pub struct TransportedCurvature {
    reference: Array1<f64>,
    correction: Array2<f64>,
}

impl TransportedCurvature {
    /// The correction between a measured Hessian and the model at the same point.
    pub fn from_hessian(reference: ArrayView1<f64>, measured: ArrayView2<f64>) -> Self {
        let n = reference.len() / 3;
        let model = dense_model(reference, n);
        let correction = &measured - &model;
        Self {
            reference: reference.to_owned(),
            correction,
        }
    }

    /// Points the reference holds.
    pub fn n_points(&self) -> usize {
        self.reference.len() / 3
    }

    /// The reference structure the correction was measured at.
    pub fn reference(&self) -> ArrayView1<'_, f64> {
        self.reference.view()
    }

    /// Frobenius norm of the correction, which is how much the model was missing.
    pub fn correction_norm(&self) -> f64 {
        self.correction.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// The transported operator applied to `v` at `target`.
    ///
    /// `H_model(target) v + M^T Delta M v`. The model term is rebuilt at the
    /// target, so a correspondence that is merely adequate degrades only the
    /// correction.
    pub fn apply(
        &self,
        target: ArrayView1<f64>,
        v: ArrayView1<f64>,
        t: &Transport,
        scale: f64,
    ) -> Array1<f64> {
        let n = self.n_points();
        let mut out = model_hessian::apply(target, n, v, scale);
        let mapped = t.to_reference(v);
        let corrected = self.correction.dot(&mapped);
        let back = t.from_reference(corrected.view());
        out += &back;
        out
    }

    /// Depth at `target` from the transported operator.
    ///
    /// Same conjugate-gradient solve as [`crate::model_hessian::depth`], on the
    /// corrected operator. The correction can be indefinite, since a true
    /// Hessian at a point that is not a minimum is, so the solve stops on a
    /// non-positive curvature rather than continuing into a direction the
    /// quadratic model does not bound.
    pub fn depth(
        &self,
        target: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
        t: &Transport,
        iters: usize,
    ) -> f64 {
        let n = self.n_points();
        if gradient.len() != 3 * n {
            return 0.0;
        }
        let scale = model_hessian::spacing(target, n);
        let mut b = gradient.to_owned();
        deflate(&mut b, n);
        let mut d = Array1::<f64>::zeros(3 * n);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rr = r.dot(&r);
        if rr.is_nan() || rr <= 0.0 {
            return 0.0;
        }
        for _ in 0..iters.max(1) {
            let mut ap = self.apply(target, p.view(), t, scale);
            deflate(&mut ap, n);
            let pap = p.dot(&ap);
            if pap.is_nan() || pap <= 1e-300 {
                break;
            }
            let alpha = rr / pap;
            d.scaled_add(alpha, &p);
            r.scaled_add(-alpha, &ap);
            let rr_new = r.dot(&r);
            if rr_new <= 1e-24 * rr {
                break;
            }
            let beta = rr_new / rr;
            p = &r + &(p.mapv(|v| v * beta));
            rr = rr_new;
        }
        0.5 * b.dot(&d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetrise::{detect_all, generate_group};

    /// Twelve vertices of a regular icosahedron, unit circumradius scaled so
    /// nearest neighbours sit near one.
    fn icosahedron() -> (Array1<f64>, usize) {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for &s1 in &[1.0, -1.0] {
            for &s2 in &[1.0, -1.0] {
                pts.push([0.0, s1, s2 * phi]);
                pts.push([s1, s2 * phi, 0.0]);
                pts.push([s2 * phi, 0.0, s1]);
            }
        }
        let mut x = Array1::zeros(3 * pts.len());
        for (i, p) in pts.iter().enumerate() {
            for k in 0..3 {
                x[3 * i + k] = p[k] / 2.0;
            }
        }
        let n = pts.len();
        (x, n)
    }

    /// The twenty-four vertices of a truncated octahedron: every permutation of
    /// `(0, +-1, +-2)`.
    fn truncated_octahedron() -> (Array1<f64>, usize) {
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for &a in &[1.0f64, -1.0] {
            for &b in &[2.0f64, -2.0] {
                pts.push([0.0, a, b]);
                pts.push([0.0, b, a]);
                pts.push([a, 0.0, b]);
                pts.push([b, 0.0, a]);
                pts.push([a, b, 0.0]);
                pts.push([b, a, 0.0]);
            }
        }
        let mut x = Array1::zeros(3 * pts.len());
        for (i, p) in pts.iter().enumerate() {
            for k in 0..3 {
                x[3 * i + k] = p[k];
            }
        }
        let n = pts.len();
        (x, n)
    }

    /// A blob with no symmetry at all, the control.
    fn asymmetric(n: usize) -> Array1<f64> {
        let mut rng = 20260806u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        Array1::from((0..3 * n).map(|_| next() * 5.0).collect::<Vec<_>>())
    }

    fn group_of(x: ArrayView1<f64>, n: usize) -> Vec<SiteSymmetry> {
        let cands = detect_all(x, n, &[2, 3, 5], 0.05);
        let group = generate_group(&cands, 200);
        site_symmetries(x, n, &group, 1e-6)
    }

    fn probe_gradient(n: usize) -> Array1<f64> {
        let mut rng = 99991u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        Array1::from((0..3 * n).map(|_| next()).collect::<Vec<_>>())
    }

    /// An icosahedron has to be found to carry a group larger than the identity,
    /// or nothing downstream is testing symmetry at all.
    #[test]
    fn an_icosahedron_carries_a_nontrivial_site_group() {
        let (x, n) = icosahedron();
        let sym = group_of(x.view(), n);
        assert!(
            sym.len() >= 10,
            "icosahedron admitted only {} site symmetries",
            sym.len()
        );
        for s in &sym {
            assert!(
                s.residual < 1e-6,
                "an admitted operation left a site residual of {}",
                s.residual
            );
        }
    }

    /// The average over the site symmetries has to be an orthogonal projector,
    /// or the split below is splitting along something that is not a subspace.
    #[test]
    fn the_group_average_is_an_idempotent_projector() {
        for (x, n) in [icosahedron(), truncated_octahedron()] {
            let sym = group_of(x.view(), n);
            let v = probe_gradient(n);
            let p1 = project_symmetric(&sym, v.view());
            let p2 = project_symmetric(&sym, p1.view());
            let err = (&p2 - &p1).iter().fold(0.0f64, |a, z| a.max(z.abs()));
            assert!(
                err < 1e-10,
                "P^2 - P reached {err} on a {n}-point structure with {} elements",
                sym.len()
            );
            // Symmetry: <Pu, v> = <u, Pv>, which is what makes the complement
            // orthogonal and the depth split exact.
            let u = asymmetric(n);
            let pu = project_symmetric(&sym, u.view());
            let pv = project_symmetric(&sym, v.view());
            let asym = (pu.dot(&v) - u.dot(&pv)).abs();
            assert!(asym < 1e-10, "the projector is not self-adjoint: {asym}");
        }
    }

    /// The model Hessian has to commute with every admitted operation. This is
    /// the whole premise: without it the two subspaces are not invariant and
    /// solving in them separately answers a different question.
    #[test]
    fn the_model_hessian_commutes_with_the_site_group() {
        let (x, n) = icosahedron();
        let sym = group_of(x.view(), n);
        let scale = model_hessian::spacing(x.view(), n);
        let v = probe_gradient(n);
        let hv = model_hessian::apply(x.view(), n, v.view(), scale);
        let mut worst = 0.0f64;
        for s in &sym {
            let one = [s.clone()];
            let dv = project_symmetric(&one, v.view());
            let hdv = model_hessian::apply(x.view(), n, dv.view(), scale);
            let dhv = project_symmetric(&one, hv.view());
            worst = worst.max((&hdv - &dhv).iter().fold(0.0f64, |a, z| a.max(z.abs())));
        }
        let magnitude = hv.iter().fold(0.0f64, |a, z| a.max(z.abs()));
        assert!(
            worst < 1e-9 * magnitude.max(1.0),
            "||H D - D H|| reached {worst} against an operator of size {magnitude}"
        );
    }

    /// The depth has to be the sum of the two blocks, exactly. A projector that
    /// commutes with the operator splits the quadratic form with no remainder,
    /// and any disagreement means the element set was not closed.
    #[test]
    fn the_depth_splits_exactly_across_the_two_blocks() {
        for (x, n) in [icosahedron(), truncated_octahedron()] {
            let sym = group_of(x.view(), n);
            let g = probe_gradient(n);
            let r = split_depth(x.view(), n, g.view(), &sym, 1e-12, 400);
            assert!(
                r.depth_error < 1e-8,
                "the two blocks summed to a depth {} away from the plain solve on \
                 a {n}-point structure carrying {} elements",
                r.depth_error,
                r.group_order
            );
        }
    }

    /// A structure with no symmetry must see the split do nothing: the group is
    /// the identity alone, the symmetric block is everything, and the remainder
    /// carries no work.
    #[test]
    fn an_asymmetric_structure_gets_no_split() {
        let n = 20;
        let x = asymmetric(n);
        let sym = group_of(x.view(), n);
        assert_eq!(
            sym.len(),
            1,
            "a random blob was found to carry {} site symmetries",
            sym.len()
        );
        let g = probe_gradient(n);
        let r = split_depth(x.view(), n, g.view(), &sym, 1e-10, 400);
        assert_eq!(
            r.symmetric.iterations, r.plain.iterations,
            "the projection changed the iteration count on a structure with no symmetry"
        );
        assert_eq!(
            r.remainder.iterations, 0,
            "the complement of the whole space carried {} iterations",
            r.remainder.iterations
        );
    }

    /// The symmetric block has to be better conditioned than the full operator,
    /// which is the mechanism the split was supposed to exploit. Recorded with
    /// the measured margin, because the margin is small enough that it does not
    /// pay for the second Krylov space.
    #[test]
    fn the_symmetric_block_is_better_conditioned() {
        let (x, n) = icosahedron();
        let sym = group_of(x.view(), n);
        let c = conditioning(x.view(), n, &sym).expect("icosahedron has a symmetric subspace");
        assert!(
            c.symmetric < c.full,
            "the symmetric block conditioned at {} against {} for the full operator",
            c.symmetric,
            c.full
        );
        assert!(
            c.symmetric_dim < c.full_dim,
            "the symmetric subspace has dimension {} of {}",
            c.symmetric_dim,
            c.full_dim
        );
    }

    /// The split never saves operator applications. Stated as a test so the
    /// negative result cannot quietly stop being true: if some later change
    /// makes the split pay, this fails and the claim in the module
    /// documentation gets revisited rather than left standing.
    #[test]
    fn the_a1_split_never_saves_work() {
        for (x, n) in [icosahedron(), truncated_octahedron()] {
            let sym = group_of(x.view(), n);
            let g = probe_gradient(n);
            let r = split_depth(x.view(), n, g.view(), &sym, 1e-10, 400);
            assert!(
                r.work_ratio() >= 1.0,
                "the A1 split took {} applications against {} for the plain solve on \
                 {n} points, which would make it worth wiring in",
                r.symmetric.iterations + r.remainder.iterations,
                r.plain.iterations
            );
        }
    }

    /// The isotypic blocks partition the space, so the depths sum to the plain
    /// depth. This is the correctness claim for the class-sum construction: if
    /// the eigenspaces were not invariant under the operator the blocks would
    /// disagree with the whole.
    #[test]
    fn the_isotypic_blocks_reproduce_the_plain_depth() {
        for (x, n) in [icosahedron(), truncated_octahedron()] {
            let sym = group_of(x.view(), n);
            let bases = isotypic_bases(&sym, n, 0x5EED);
            let total: usize = bases.iter().map(|b| b.ncols()).sum();
            assert_eq!(
                total,
                3 * n,
                "the {} blocks cover {total} of {} dimensions",
                bases.len(),
                3 * n
            );
            assert!(
                bases.len() >= 8,
                "a group of order {} produced only {} blocks",
                sym.len(),
                bases.len()
            );
            let g = probe_gradient(n);
            let r = block_depth(x.view(), n, g.view(), &bases, 1e-10, 400);
            assert!(
                r.depth_error < 1e-8,
                "the blocks summed to a depth {} away from the plain solve",
                r.depth_error
            );
        }
    }

    /// The full block-diagonalisation shortens the critical path without
    /// shortening the serial work. Both halves of that are the measured result
    /// and both are pinned: a decomposition that started saving serial
    /// applications would change the verdict in the module documentation.
    #[test]
    fn the_isotypic_split_shortens_the_critical_path_but_not_the_work() {
        let (x, n) = icosahedron();
        let sym = group_of(x.view(), n);
        let bases = isotypic_bases(&sym, n, 0x5EED);
        let g = probe_gradient(n);
        let r = block_depth(x.view(), n, g.view(), &bases, 1e-10, 400);
        assert!(
            r.work_ratio() >= 1.0,
            "the blocks took {} applications against {} for the plain solve",
            r.blocks.iter().map(|b| b.iterations).sum::<usize>(),
            r.plain.iterations
        );
        assert!(
            (r.critical_path() as f64) * 3.0 <= r.plain.iterations as f64,
            "the busiest block took {} applications against {} for the plain solve, \
             which is less than the threefold latency cut the decomposition is worth",
            r.critical_path(),
            r.plain.iterations
        );
    }

    /// A structure with no symmetry gets one block covering everything, so the
    /// isotypic route degenerates to the plain solve rather than doing something
    /// arbitrary.
    #[test]
    fn an_asymmetric_structure_gets_one_isotypic_block() {
        let n = 16;
        let x = asymmetric(n);
        let sym = group_of(x.view(), n);
        let bases = isotypic_bases(&sym, n, 0x5EED);
        assert_eq!(
            bases.len(),
            1,
            "a random blob decomposed into {} blocks",
            bases.len()
        );
        assert_eq!(bases[0].ncols(), 3 * n);
    }

    /// The site group has to be a group: closed, with inverses, and with the
    /// conjugacy classes partitioning it. The projector and the class sums both
    /// rest on that and neither checks it.
    #[test]
    fn the_admitted_elements_form_a_group() {
        let (x, n) = icosahedron();
        let sym = group_of(x.view(), n);
        for a in &sym {
            assert!(
                sym.iter().any(|s| s.matches(&a.inverse())),
                "an element's inverse is missing from the site group"
            );
            for b in &sym {
                assert!(
                    sym.iter().any(|s| s.matches(&a.compose(b))),
                    "the site group is not closed under composition"
                );
            }
        }
        let classes = conjugacy_classes(&sym);
        let covered: usize = classes.iter().map(|c| c.len()).sum();
        assert_eq!(
            covered,
            sym.len(),
            "the conjugacy classes cover {covered} of {} elements",
            sym.len()
        );
    }

    /// The exact pair Hessian has to reproduce a central difference of the
    /// analytic gradient, or the object transport carries is not a Hessian.
    #[test]
    fn the_pair_hessian_matches_a_finite_difference_of_the_gradient() {
        use crate::potentials::PairPotential;
        let n = 6;
        let mut x = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = (i % 3) as f64 * 1.12;
            x[3 * i + 1] = ((i / 3) % 2) as f64 * 1.09;
            x[3 * i + 2] = 0.07 * i as f64;
        }
        let pot = PairPotential::lennard_jones(n);
        let h = pair_hessian(x.view(), n, PairKind::LennardJones);
        let eps = 1e-5;
        let mut worst = 0.0f64;
        for c in 0..(3 * n) {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[c] += eps;
            xm[c] -= eps;
            let gp = pot.value_and_gradient(xp.view()).1;
            let gm = pot.value_and_gradient(xm.view()).1;
            for r in 0..(3 * n) {
                let fd = (gp[r] - gm[r]) / (2.0 * eps);
                worst = worst.max((fd - h[[r, c]]).abs() / (1.0 + h[[r, c]].abs()));
            }
        }
        assert!(
            worst < 1e-6,
            "the analytic pair Hessian differs from a central difference by {worst} relative"
        );
    }

    /// Transport onto a rotated and relabelled copy has to reproduce the
    /// operator exactly. This is the correctness claim for the index
    /// convention, and it is the one that fails silently if the permutation is
    /// read the wrong way round.
    #[test]
    fn transport_onto_a_relabelled_copy_reproduces_the_operator() {
        let n = 9;
        let a = asymmetric(n);
        // A rotation by 0.7 about a fixed axis, and a fixed derangement.
        let axis = {
            let mut v = [0.3f64, -0.5, 0.81];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            for c in v.iter_mut() {
                *c /= l;
            }
            v
        };
        let (s, c) = 0.7f64.sin_cos();
        let t = 1.0 - c;
        let rot = [
            t * axis[0] * axis[0] + c,
            t * axis[0] * axis[1] - s * axis[2],
            t * axis[0] * axis[2] + s * axis[1],
            t * axis[0] * axis[1] + s * axis[2],
            t * axis[1] * axis[1] + c,
            t * axis[1] * axis[2] - s * axis[0],
            t * axis[0] * axis[2] - s * axis[1],
            t * axis[1] * axis[2] + s * axis[0],
            t * axis[2] * axis[2] + c,
        ];
        // `source[slot]` is the target site holding reference slot `slot`.
        let source: Vec<usize> = (0..n).map(|i| (i * 4 + 3) % n).collect();
        // Target coordinates: `R^T` applied to the reference, then relabelled,
        // so that `R` carries the target back onto the reference and the
        // transport's own convention is the thing under test.
        let mut b = Array1::<f64>::zeros(3 * n);
        for (slot, &src) in source.iter().enumerate() {
            for k in 0..3 {
                b[3 * src + k] = rot[k] * a[3 * slot]
                    + rot[3 + k] * a[3 * slot + 1]
                    + rot[6 + k] * a[3 * slot + 2];
            }
        }
        let transport = Transport::new(source, rot).expect("a cycle is a bijection");

        // The map has to be orthogonal before anything built on it means
        // something.
        let probe = probe_gradient(n);
        let round = transport.from_reference(transport.to_reference(probe.view()).view());
        let round_err = (&round - &probe).iter().fold(0.0f64, |m, z| m.max(z.abs()));
        assert!(round_err < 1e-14, "M^T M is not the identity: {round_err}");

        // Zero correction, so the operator being compared is the model Hessian
        // itself carried across the correspondence.
        let carried = TransportedCurvature::from_hessian(a.view(), dense_model(a.view(), n).view());
        assert!(
            carried.correction_norm() < 1e-10,
            "a correction against the model itself came out at {}",
            carried.correction_norm()
        );
        let scale = model_hessian::spacing(b.view(), n);
        let direct = model_hessian::apply(b.view(), n, probe.view(), scale);
        let via = carried.apply(b.view(), probe.view(), &transport, scale);
        let err = (&via - &direct).iter().fold(0.0f64, |m, z| m.max(z.abs()));
        let mag = direct.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        assert!(
            err < 1e-12 * mag.max(1.0),
            "transporting onto a relabelled copy moved the operator by {err} against a \
             response of size {mag}"
        );

        // And with a real correction: the exact pair Hessian at the reference,
        // carried across, has to equal the exact pair Hessian at the target.
        let carried = TransportedCurvature::from_hessian(
            a.view(),
            pair_hessian(a.view(), n, PairKind::LennardJones).view(),
        );
        let exact = pair_hessian(b.view(), n, PairKind::LennardJones).dot(&probe);
        let via = carried.apply(b.view(), probe.view(), &transport, scale);
        let err = (&via - &exact).iter().fold(0.0f64, |m, z| m.max(z.abs()));
        let mag = exact.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        assert!(
            err < 1e-9 * mag.max(1.0),
            "the transported true Hessian differs from the target's own by {err} against \
             a response of size {mag}"
        );
    }

    /// A permutation that is not a bijection has to be refused rather than used,
    /// since IRA returns one on occasion while reporting success.
    #[test]
    fn a_non_bijective_correspondence_is_refused() {
        assert!(Transport::new(vec![0, 1, 1], [0.0; 9]).is_none());
        assert!(Transport::new(vec![0, 1, 5], [0.0; 9]).is_none());
        assert!(Transport::new(vec![2, 0, 1], [0.0; 9]).is_some());
    }
}
