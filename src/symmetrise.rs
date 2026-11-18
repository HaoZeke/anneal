//! Detecting the approximate point-group symmetry a structure has.
//!
//! The functions here measure. Candidate axes come from the structure itself,
//! the deviation of each candidate operation is the root-mean-square distance
//! between a point and its matched image, and [`generate_group`] closes a set
//! of detected operations into the group they generate. Nothing here moves a
//! point: the routines that pushed a structure onto the symmetry it nearly
//! has, completed orbits, or built a core out of a detected group are gone
//! from this crate, because they construct a candidate from a recognised
//! motif rather than search for one. The block-diagonalisation of the Hessian
//! in [`crate::sym_hessian`] is the consumer: it needs the group, not a
//! symmetrised structure.
//!
//! [`crate::movekernel::Symmetrise`] is a different thing and is unchanged.

use ndarray::ArrayView1;

/// An approximate rotational symmetry a structure was found to have.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Unit vector along the rotation axis, through the centroid.
    pub axis: [f64; 3],
    /// Rotation order; the operation is by `2 pi / order`.
    ///
    /// Order 1 with `improper` set is a plane: the reflection in the plane
    /// through the centroid whose normal is `axis`.
    pub order: usize,
    /// Whether the operation carries a reflection.
    ///
    /// The group a structure has is not always a rotation group. The 98-point
    /// global minimum is tetrahedral and the 38-point one is octahedral, and
    /// both of those are defined by their mirror planes: the proper rotations
    /// alone give T and O, which are index-two subgroups missing exactly the
    /// operations that pick out the structure. A generator set closed only
    /// under proper rotations cannot express either, so the scheme could
    /// symmetrise toward a subgroup of the target and never toward the target.
    pub improper: bool,
    /// Root-mean-square distance between a point and its matched image.
    ///
    /// Zero for an exact symmetry. This is what "approximate" is measured by,
    /// and what decides whether symmetrising is worth doing.
    pub deviation: f64,
}

/// Centroid of a flattened `(n, 3)` point set.
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

/// Rotates `v` about a unit `axis` by `angle`, by Rodrigues' formula.
fn rotate(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    // Host `sin` and `cos` of the same angle are not one function on glibc
    // and libSystem, and this rotation places the symmetrised points.
    let s = libm::sin(angle);
    let co = libm::cos(angle);
    let dot = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    let cross = [
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    ];
    [
        v[0] * co + cross[0] * s + axis[0] * dot * (1.0 - co),
        v[1] * co + cross[1] * s + axis[1] * dot * (1.0 - co),
        v[2] * co + cross[2] * s + axis[2] * dot * (1.0 - co),
    ]
}

fn normalise(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-9 {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

/// Root-mean-square distance between each point and its nearest image under a
/// rotation of `order` about `axis`.
///
/// The matching is nearest-image, which is what makes this a measure of
/// approximate symmetry rather than of exact symmetry: an exactly symmetric
/// structure permutes onto itself and every distance is zero.
pub fn deviation(x: ArrayView1<f64>, n: usize, axis: [f64; 3], order: usize) -> f64 {
    deviation_under(x, n, axis, order, f64::INFINITY).unwrap_or(f64::INFINITY)
}

/// [`deviation`], or `None` once the partial sum already exceeds `cap`.
///
/// Callers that keep only a value strictly below `cap` get the same
/// decision and the same bits. The early return is past `cap`, with a
/// margin so a later addition cannot round the reported root back under it.
fn deviation_under(
    x: ArrayView1<f64>,
    n: usize,
    axis: [f64; 3],
    order: usize,
    cap: f64,
) -> Option<f64> {
    if n == 0 || order < 2 {
        return None;
    }
    Cloud::from_view(x, n).rotation_deviation(axis, order, cap)
}

/// How far `x` is from being symmetric under the plane whose normal is `axis`.
///
/// The same statistic as [`deviation`] for a rotation: each point is reflected
/// and matched to its nearest partner, and the root-mean-square of those
/// distances is what "approximately symmetric" means here.
pub fn plane_deviation(x: ArrayView1<f64>, n: usize, axis: [f64; 3]) -> f64 {
    plane_deviation_under(x, n, axis, f64::INFINITY).unwrap_or(f64::INFINITY)
}

fn plane_deviation_under(x: ArrayView1<f64>, n: usize, axis: [f64; 3], cap: f64) -> Option<f64> {
    if n == 0 {
        return None;
    }
    Cloud::from_view(x, n).plane_deviation(axis, cap)
}

/// Centred coordinates in separate arrays, so a nearest-image scan reads
/// contiguous columns.
struct Cloud {
    xs: Vec<f64>,
    ys: Vec<f64>,
    zs: Vec<f64>,
}

impl Cloud {
    fn from_view(x: ArrayView1<f64>, n: usize) -> Self {
        let c = centroid(x, n);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut zs = Vec::with_capacity(n);
        for atom in 0..n {
            xs.push(x[3 * atom] - c[0]);
            ys.push(x[3 * atom + 1] - c[1]);
            zs.push(x[3 * atom + 2] - c[2]);
        }
        Self { xs, ys, zs }
    }

    fn point(&self, index: usize) -> [f64; 3] {
        [self.xs[index], self.ys[index], self.zs[index]]
    }

    fn nearest_squared_scalar(&self, image: [f64; 3]) -> f64 {
        let mut best = f64::INFINITY;
        for index in 0..self.xs.len() {
            let dx = image[0] - self.xs[index];
            let dy = image[1] - self.ys[index];
            let dz = image[2] - self.zs[index];
            let distance = dx * dx + dy * dy + dz * dz;
            if distance < best {
                best = distance;
            }
        }
        best
    }

    /// # Safety
    ///
    /// AVX2 must be enabled.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn nearest_squared_avx2(&self, image: [f64; 3]) -> f64 {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_min_pd, _mm256_mul_pd, _mm256_set1_pd,
            _mm256_storeu_pd, _mm256_sub_pd,
        };
        let ix = _mm256_set1_pd(image[0]);
        let iy = _mm256_set1_pd(image[1]);
        let iz = _mm256_set1_pd(image[2]);
        let n = self.xs.len();
        let mut index = 0usize;
        let mut best_lanes = _mm256_set1_pd(f64::INFINITY);
        while index + 4 <= n {
            unsafe {
                let dx = _mm256_sub_pd(_mm256_loadu_pd(self.xs.as_ptr().add(index)), ix);
                let dy = _mm256_sub_pd(_mm256_loadu_pd(self.ys.as_ptr().add(index)), iy);
                let dz = _mm256_sub_pd(_mm256_loadu_pd(self.zs.as_ptr().add(index)), iz);
                let distance = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                    _mm256_mul_pd(dz, dz),
                );
                best_lanes = _mm256_min_pd(best_lanes, distance);
            }
            index += 4;
        }
        let mut lane = [0.0; 4];
        unsafe { _mm256_storeu_pd(lane.as_mut_ptr(), best_lanes) };
        let mut best = lane[0];
        if lane[1] < best {
            best = lane[1];
        }
        if lane[2] < best {
            best = lane[2];
        }
        if lane[3] < best {
            best = lane[3];
        }
        while index < n {
            let dx = image[0] - self.xs[index];
            let dy = image[1] - self.ys[index];
            let dz = image[2] - self.zs[index];
            let distance = dx * dx + dy * dy + dz * dz;
            if distance < best {
                best = distance;
            }
            index += 1;
        }
        best
    }

    fn nearest_one(&self, image: [f64; 3]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { self.nearest_squared_avx2(image) };
        }
        self.nearest_squared_scalar(image)
    }

    /// Nearest squared distance of every image.
    ///
    /// Each image keeps a four-wide minimum, then the lowest lane, then the
    /// tail, which is the scalar scan's minimum for ordinary distances.
    /// A deviation asks for one image per atom, and scanning the cloud once
    /// per handful of images reads each coordinate a quarter as often.
    fn nearest_squared_batch(&self, images: &[[f64; 3]], out: &mut [f64]) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            unsafe { self.nearest_squared_batch_avx2(images, out) };
            return;
        }
        for (image, slot) in images.iter().zip(out.iter_mut()) {
            *slot = self.nearest_squared_scalar(*image);
        }
    }

    /// # Safety
    ///
    /// AVX2 must be enabled. `out` is at least as long as `images`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn nearest_squared_batch_avx2(&self, images: &[[f64; 3]], out: &mut [f64]) {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_min_pd, _mm256_mul_pd, _mm256_set1_pd,
            _mm256_storeu_pd, _mm256_sub_pd,
        };
        let n = self.xs.len();
        let mut base = 0usize;
        while base < images.len() {
            let width = (images.len() - base).min(4);
            let mut best = [_mm256_set1_pd(f64::INFINITY); 4];
            let mut index = 0usize;
            while index + 4 <= n {
                unsafe {
                    let x = _mm256_loadu_pd(self.xs.as_ptr().add(index));
                    let y = _mm256_loadu_pd(self.ys.as_ptr().add(index));
                    let z = _mm256_loadu_pd(self.zs.as_ptr().add(index));
                    for lane in 0..width {
                        let image = images[base + lane];
                        let dx = _mm256_sub_pd(x, _mm256_set1_pd(image[0]));
                        let dy = _mm256_sub_pd(y, _mm256_set1_pd(image[1]));
                        let dz = _mm256_sub_pd(z, _mm256_set1_pd(image[2]));
                        let distance = _mm256_add_pd(
                            _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                            _mm256_mul_pd(dz, dz),
                        );
                        best[lane] = _mm256_min_pd(best[lane], distance);
                    }
                }
                index += 4;
            }
            for lane in 0..width {
                let image = images[base + lane];
                let mut packed = [0.0; 4];
                unsafe { _mm256_storeu_pd(packed.as_mut_ptr(), best[lane]) };
                let mut acc = packed[0];
                if packed[1] < acc {
                    acc = packed[1];
                }
                if packed[2] < acc {
                    acc = packed[2];
                }
                if packed[3] < acc {
                    acc = packed[3];
                }
                let mut tail = index;
                while tail < n {
                    let dx = image[0] - self.xs[tail];
                    let dy = image[1] - self.ys[tail];
                    let dz = image[2] - self.zs[tail];
                    let distance = dx * dx + dy * dy + dz * dz;
                    if distance < acc {
                        acc = distance;
                    }
                    tail += 1;
                }
                out[base + lane] = acc;
            }
            base += width;
        }
    }

    /// Adds these images onto `total` in order.
    ///
    /// Returns whether the running sum is still under `cap`. A finite cap
    /// stops at the same atom a one-image scan would, after the batch that
    /// contains it has been measured.
    fn accumulate(&self, images: &[[f64; 3]], cap: f64, total: &mut f64) -> bool {
        let n = self.xs.len();
        let stop = cap.is_finite();
        let mut distance = [0.0; 4];
        self.nearest_squared_batch(images, &mut distance[..images.len()]);
        for lane in 0..images.len() {
            *total += distance[lane];
            if stop && *total > cap * cap * n as f64 {
                let partial = (*total / n as f64).sqrt();
                if partial >= cap * (1.0 + 1.0e-6) {
                    return false;
                }
            }
        }
        true
    }

    fn rotation_deviation(&self, axis: [f64; 3], order: usize, cap: f64) -> Option<f64> {
        let n = self.xs.len();
        if n == 0 || order < 2 {
            return None;
        }
        let angle = 2.0 * std::f64::consts::PI / order as f64;
        let mut total = 0.0;
        let stop = cap.is_finite();
        if stop {
            for index in 0..n {
                total += self.nearest_one(rotate(self.point(index), axis, angle));
                if total > cap * cap * n as f64 {
                    let partial = (total / n as f64).sqrt();
                    if partial >= cap * (1.0 + 1.0e-6) {
                        return None;
                    }
                }
            }
            return Some((total / n as f64).sqrt());
        }
        let mut index = 0usize;
        let mut stored = [[0.0; 3]; 4];
        while index < n {
            let width = (n - index).min(4);
            for lane in 0..width {
                stored[lane] = rotate(self.point(index + lane), axis, angle);
            }
            if !self.accumulate(&stored[..width], cap, &mut total) {
                return None;
            }
            index += width;
        }
        Some((total / n as f64).sqrt())
    }

    fn plane_deviation(&self, axis: [f64; 3], cap: f64) -> Option<f64> {
        let n = self.xs.len();
        if n == 0 {
            return None;
        }
        let m = mirror_matrix(axis);
        let mut total = 0.0;
        let stop = cap.is_finite();
        if stop {
            for index in 0..n {
                let v = self.point(index);
                total += self.nearest_one([
                    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
                ]);
                if total > cap * cap * n as f64 {
                    let partial = (total / n as f64).sqrt();
                    if partial >= cap * (1.0 + 1.0e-6) {
                        return None;
                    }
                }
            }
            return Some((total / n as f64).sqrt());
        }
        let mut index = 0usize;
        let mut stored = [[0.0; 3]; 4];
        while index < n {
            let width = (n - index).min(4);
            for lane in 0..width {
                let v = self.point(index + lane);
                stored[lane] = [
                    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
                ];
            }
            if !self.accumulate(&stored[..width], cap, &mut total) {
                return None;
            }
            index += width;
        }
        Some((total / n as f64).sqrt())
    }
}

/// Axes worth testing, taken from the structure rather than at random.
///
/// Two families. The principal axes of the inertia tensor, which is where a
/// rotation axis of a symmetric body must lie; and the directions from the
/// centroid to each point and to each pair midpoint, which is where an axis
/// through a vertex or an edge lies. That covers the axes of the point groups
/// clusters actually adopt without enumerating directions blindly.
fn candidate_axes(x: ArrayView1<f64>, n: usize) -> Vec<[f64; 3]> {
    let c = centroid(x, n);
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(3 + n + n);
    // Inertia tensor about the centroid, then its eigenvectors.
    let mut t = ndarray::Array2::<f64>::zeros((3, 3));
    for a in 0..n {
        let v = [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]];
        let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        for i in 0..3 {
            for j in 0..3 {
                t[[i, j]] += if i == j { r2 } else { 0.0 } - v[i] * v[j];
            }
        }
    }
    let (_, vecs) = crate::spectral::symmetric_eigen(t.view(), 64);
    for j in 0..3 {
        if let Some(u) = normalise([vecs[[0, j]], vecs[[1, j]], vecs[[2, j]]]) {
            out.push(u);
        }
    }
    // Through each point.
    for a in 0..n {
        if let Some(u) = normalise([x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]]) {
            out.push(u);
        }
    }
    out
}

/// The best approximate rotational symmetry of `x`, over the given orders.
///
/// `None` when nothing is even approximately symmetric, which is the signal to
/// leave the structure alone.
///
/// Proper rotations only, because [`symmetrise`] averages a point around its
/// orbit under a rotation and a reflection has no such orbit. Mirror planes are
/// found by [`detect_all`], which exists to build a group rather than to push a
/// structure along one operation.
pub fn detect(x: ArrayView1<f64>, n: usize, orders: &[usize], tolerance: f64) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    let cloud = Cloud::from_view(x, n);
    for axis in candidate_axes(x, n) {
        for &order in orders {
            let Some(d) = cloud.rotation_deviation(axis, order, tolerance) else {
                continue;
            };
            if d < tolerance && best.map(|b| d < b.deviation).unwrap_or(true) {
                best = Some(Candidate {
                    axis,
                    order,
                    improper: false,
                    deviation: d,
                });
            }
        }
    }
    best
}

/// All approximate symmetries of `x`, proper and improper.
///
/// [`detect`] returns only the best one, which is enough to symmetrise toward a
/// single operation and not enough to build a group: a tetrahedral structure
/// needs a three-fold axis and a plane together, and either alone generates a
/// subgroup that is not the answer.
pub fn detect_all(
    x: ArrayView1<f64>,
    n: usize,
    orders: &[usize],
    tolerance: f64,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    let cloud = Cloud::from_view(x, n);
    for axis in candidate_axes(x, n) {
        for &order in orders {
            let Some(d) = cloud.rotation_deviation(axis, order, tolerance) else {
                continue;
            };
            if d < tolerance {
                out.push(Candidate {
                    axis,
                    order,
                    improper: false,
                    deviation: d,
                });
            }
        }
        let Some(d) = cloud.plane_deviation(axis, tolerance) else {
            continue;
        };
        if d < tolerance {
            out.push(Candidate {
                axis,
                order: 1,
                improper: true,
                deviation: d,
            });
        }
    }
    out.sort_by(|a, b| {
        a.deviation
            .partial_cmp(&b.deviation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Reflection in the plane through the centroid with normal `axis`.
pub fn mirror_matrix(axis: [f64; 3]) -> Rot {
    let [x, y, z] = axis;
    [
        [1.0 - 2.0 * x * x, -2.0 * x * y, -2.0 * x * z],
        [-2.0 * y * x, 1.0 - 2.0 * y * y, -2.0 * y * z],
        [-2.0 * z * x, -2.0 * z * y, 1.0 - 2.0 * z * z],
    ]
}

/// A rotation as a 3x3 matrix, row-major.
pub type Rot = [[f64; 3]; 3];

fn rot_matrix(axis: [f64; 3], angle: f64) -> Rot {
    let s = libm::sin(angle);
    let c = libm::cos(angle);
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

fn mul(a: &Rot, b: &Rot) -> Rot {
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

fn same(a: &Rot, b: &Rot) -> bool {
    (0..3).all(|i| (0..3).all(|j| (a[i][j] - b[i][j]).abs() < 1e-6))
}

/// The rotation group generated by a set of approximate symmetries.
///
/// A point group is not an axis, and this is the difference between the scheme
/// and a single rotation. The 98-point global minimum is tetrahedral, and the
/// tetrahedral rotation group is generated by a three-fold and a two-fold axis
/// together: averaging orbits under either one alone does not make a structure
/// tetrahedral, it makes it axially symmetric, which is a different and much
/// weaker constraint.
///
/// Closed by repeated multiplication until nothing new appears or `cap` is
/// reached. The cap is a guard against a set of axes that are not quite
/// commensurate generating an unbounded pseudo-group out of rounding; the
/// icosahedral rotation group has sixty elements, so anything past that is a
/// sign the detection was loose rather than a real group.
pub fn generate_group(cands: &[Candidate], cap: usize) -> Vec<Rot> {
    let identity: Rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut group: Vec<Rot> = vec![identity];
    let mut gens: Vec<Rot> = Vec::new();
    for c in cands {
        if c.improper {
            // A reflection, which is what lifts a rotation group to the group
            // the structure actually has: proper rotations alone give T where
            // the 98-point minimum is Td, and O where the 38-point minimum is
            // Oh. Both are index-two subgroups missing exactly the operations
            // that define the target.
            gens.push(mirror_matrix(c.axis));
        } else {
            let angle = 2.0 * std::f64::consts::PI / c.order.max(2) as f64;
            gens.push(rot_matrix(c.axis, angle));
        }
    }
    let mut changed = true;
    while changed && group.len() < cap {
        changed = false;
        let snapshot = group.clone();
        for a in &snapshot {
            for g in &gens {
                let p = mul(a, g);
                if !group.iter().any(|q| same(q, &p)) {
                    group.push(p);
                    changed = true;
                    if group.len() >= cap {
                        break;
                    }
                }
            }
            if group.len() >= cap {
                break;
            }
        }
    }
    group
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_vector_nearest_image_matches_the_scalar_scan() {
        let cloud = Cloud::from_view(octahedron(1.3).view(), 6);
        let queries = [[0.2, -0.4, 1.1], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let mut batched = [0.0; 3];
        cloud.nearest_squared_batch(&queries, &mut batched);
        for (query, got) in queries.iter().zip(batched) {
            assert_eq!(
                cloud.nearest_squared_scalar(*query).to_bits(),
                got.to_bits()
            );
        }
        let wide = Cloud::from_view(
            Array1::from((0..30).map(|i| (i as f64) * 0.17).collect::<Vec<_>>()).view(),
            10,
        );
        // Ten points so the scan has a vector body and a scalar tail, and
        // five images so one batch is full and one is a remainder.
        let images = [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.5, 0.25],
            [4.0, -2.0, 0.5],
            [0.3, 0.3, 0.3],
        ];
        let mut wide_batched = [0.0; 5];
        wide.nearest_squared_batch(&images, &mut wide_batched);
        for (image, got) in images.iter().zip(wide_batched) {
            assert_eq!(wide.nearest_squared_scalar(*image).to_bits(), got.to_bits());
        }
    }

    fn octahedron(scale: f64) -> Array1<f64> {
        Array1::from(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
            .iter()
            .flat_map(|v| v.iter().map(|c| c * scale))
            .collect::<Vec<f64>>(),
        )
    }

    /// A deterministic jitter, so a test says the same thing every run.
    fn jitter(x: &Array1<f64>, amp: f64) -> Array1<f64> {
        let mut y = x.clone();
        for (i, v) in y.iter_mut().enumerate() {
            *v += amp * (((i * 37 + 11) % 17) as f64 / 8.0 - 1.0);
        }
        y
    }

    #[test]
    fn an_exact_symmetry_has_zero_deviation() {
        let x = octahedron(1.0);
        let d = deviation(x.view(), 6, [0.0, 0.0, 1.0], 4);
        assert!(d < 1e-12, "a four-fold axis of an octahedron gave {d}");
    }

    #[test]
    fn a_symmetry_the_structure_lacks_has_a_large_deviation() {
        let x = octahedron(1.0);
        // No five-fold axis anywhere in an octahedron.
        let d = deviation(x.view(), 6, [0.0, 0.0, 1.0], 5);
        assert!(d > 0.1, "a five-fold axis gave {d}, which is too small");
    }

    /// The groups the hard structures actually have, which proper rotations alone
    /// cannot express. Tetrahedral order is 24 and octahedral is 48; the
    /// rotation subgroups are 12 and 24, so a generator set closed under proper
    /// rotations stops exactly halfway.
    #[test]
    fn a_reflection_generator_reaches_the_full_groups() {
        let s3 = 1.0 / 3.0_f64.sqrt();
        // Tetrahedral. The rotation subgroup T needs a three-fold axis along a
        // body diagonal and a two-fold along a coordinate axis; a diagonal
        // mirror then doubles it to Td. A three-fold with a mirror containing
        // it gives C3v of order six instead, which is what the first attempt
        // at this test generated and what makes the choice of generators worth
        // stating rather than assuming.
        let td = generate_group(
            &[
                Candidate {
                    axis: [s3, s3, s3],
                    order: 3,
                    improper: false,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [0.0, 0.0, 1.0],
                    order: 2,
                    improper: false,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt(), 0.0],
                    order: 1,
                    improper: true,
                    deviation: 0.0,
                },
            ],
            200,
        );
        assert_eq!(
            td.len(),
            24,
            "tetrahedral group came out with {} elements",
            td.len()
        );
        assert!(
            td.iter().any(|m| determinant(m) < 0.0),
            "no improper element in a group that is half improper"
        );

        // Octahedral: a four-fold axis and a mirror plane normal to another.
        let oh = generate_group(
            &[
                Candidate {
                    axis: [0.0, 0.0, 1.0],
                    order: 4,
                    improper: false,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [1.0, 0.0, 0.0],
                    order: 1,
                    improper: true,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [s3, s3, s3],
                    order: 3,
                    improper: false,
                    deviation: 0.0,
                },
            ],
            200,
        );
        assert_eq!(
            oh.len(),
            48,
            "octahedral group came out with {} elements",
            oh.len()
        );
    }

    /// Without the reflection the same generators must give the proper
    /// subgroup, which is the statement of what was missing before.
    #[test]
    fn proper_generators_alone_stop_at_the_rotation_subgroup() {
        let s3 = 1.0 / 3.0_f64.sqrt();
        let t = generate_group(
            &[
                Candidate {
                    axis: [s3, s3, s3],
                    order: 3,
                    improper: false,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [0.0, 0.0, 1.0],
                    order: 2,
                    improper: false,
                    deviation: 0.0,
                },
            ],
            200,
        );
        assert_eq!(
            t.len(),
            12,
            "rotation subgroup came out with {} elements",
            t.len()
        );
        assert!(
            t.iter().all(|m| determinant(m) > 0.0),
            "an improper element appeared with no reflection generator"
        );
    }

    /// A plane the structure has must be detected, and one it lacks must not
    /// be, or the group is built from operations that are not there.
    #[test]
    fn plane_detection_matches_the_structure() {
        // A square in the xy plane: mirror in z is exact, mirror in a plane at
        // an odd angle is not.
        let x = ndarray::Array1::from(vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
        ]);
        let flat = plane_deviation(x.view(), 4, [0.0, 0.0, 1.0]);
        assert!(flat < 1e-9, "the plane of a planar square read as {flat}");
        let odd = plane_deviation(x.view(), 4, [0.3, 0.1, 0.95_f64.sqrt()]);
        assert!(odd > 1e-3, "a plane the square lacks read as {odd}");
    }

    fn determinant(m: &Rot) -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    #[test]
    fn detection_finds_an_axis_the_structure_has() {
        let x = jitter(&octahedron(1.0), 0.03);
        let cand = detect(x.view(), 6, &[2, 3, 4, 5, 6], 0.5).expect("nothing detected");
        // Whatever it found has to be a genuine symmetry of the ideal
        // structure, which is the claim; asserting a particular axis is not.
        // An octahedron has three four-fold axes through opposite vertices,
        // four three-fold axes through opposite faces and six two-fold axes
        // through opposite edge midpoints, so [110] with order two is as
        // correct an answer as [001] with order four, and the detector is
        // free to return whichever fits the jitter better.
        let exact = deviation(octahedron(1.0).view(), 6, cand.axis, cand.order);
        assert!(
            exact < 1e-9,
            "detected order {} about {:?}, which the ideal structure lacks: {exact}",
            cand.order,
            cand.axis
        );
        assert!(cand.deviation < 0.2, "deviation {}", cand.deviation);
    }

    /// The centre of mass must not move: a symmetrisation that translates the
    /// cluster is a translation with extra steps.
    /// The tetrahedral rotation group has twelve elements and is generated by
    /// a three-fold and a two-fold axis. This is the case the whole addition
    /// exists for: the 98-point global minimum is tetrahedral, and neither
    /// generator alone produces it.
    #[test]
    fn a_three_fold_and_a_two_fold_generate_the_tetrahedral_group() {
        let s3 = 1.0 / 3.0_f64.sqrt();
        let gens = vec![
            Candidate {
                axis: [s3, s3, s3],
                order: 3,
                improper: false,
                deviation: 0.0,
            },
            Candidate {
                axis: [0.0, 0.0, 1.0],
                order: 2,
                improper: false,
                deviation: 0.0,
            },
        ];
        let g = generate_group(&gens, 120);
        assert_eq!(
            g.len(),
            12,
            "the tetrahedral rotation group has twelve elements, got {}",
            g.len()
        );
    }

    /// And the octahedral group has twenty-four, from the same construction.
    #[test]
    fn the_four_fold_axes_generate_the_octahedral_group() {
        let gens = vec![
            Candidate {
                axis: [1.0, 0.0, 0.0],
                order: 4,
                improper: false,
                deviation: 0.0,
            },
            Candidate {
                axis: [0.0, 1.0, 0.0],
                order: 4,
                improper: false,
                deviation: 0.0,
            },
        ];
        let g = generate_group(&gens, 120);
        assert_eq!(
            g.len(),
            24,
            "octahedral rotations number 24, got {}",
            g.len()
        );
    }

    /// The cap is a guard: axes that are not quite commensurate would otherwise
    /// generate an unbounded pseudo-group out of rounding.
    #[test]
    fn incommensurate_axes_are_capped_rather_than_run_away() {
        let gens = vec![
            Candidate {
                axis: [1.0, 0.0, 0.0],
                order: 5,
                improper: false,
                deviation: 0.0,
            },
            Candidate {
                axis: [0.31, 0.77, 0.56],
                order: 3,
                improper: false,
                deviation: 0.0,
            },
        ];
        let g = generate_group(&gens, 60);
        assert!(g.len() <= 60, "group ran to {}", g.len());
    }
}
