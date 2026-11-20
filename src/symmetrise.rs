//! Making a structure exactly symmetric about the symmetry it nearly has.
//!
//! Oakley, Johnston and Wales, *Symmetrisation schemes for global optimisation
//! of atomic clusters*, Phys. Chem. Chem. Phys. 15, 3965 (2013).
//!
//! High-symmetry structures are over-represented at both ends of the energy
//! distribution, so the global minimum of a cluster is more often symmetric
//! than a typical minimum is. That is a searchable fact: a structure that is
//! nearly symmetric can be pushed onto the symmetry it nearly has, and if the
//! answer is symmetric the push lands near it. The paper reports the mean first
//! encounter time for the 98-point Lennard-Jones cluster, whose global minimum
//! is tetrahedral, improving by more than seventyfold.
//!
//! # What this is not
//!
//! [`crate::movekernel::Symmetrise`] already exists and is a different thing.
//! It picks a random axis and a random rotation order, then blends each point a
//! fraction of the way toward its image. Nothing about it refers to the
//! structure's own symmetry, so on a structure that is nearly tetrahedral it is
//! overwhelmingly likely to symmetrise about an axis the structure has no
//! relationship to, and the result is a perturbation with extra steps.
//!
//! The scheme here measures first. Candidate axes come from the structure, the
//! deviation from each candidate symmetry is computed, and the operation is
//! applied about the best one and applied fully rather than blended. A
//! structure with no approximate symmetry is left alone, which is the case
//! where the random version does its damage.

use ndarray::{Array1, ArrayView1};

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
    rotate_by(v, axis, Turn::of(angle))
}

/// Sine and cosine of a rotation angle, taken once for every point the
/// rotation moves.
#[derive(Debug, Clone, Copy)]
struct Turn {
    s: f64,
    co: f64,
}

impl Turn {
    fn of(angle: f64) -> Self {
        // Host `sin` and `cos` of the same angle are not one function on
        // glibc and libSystem, and this rotation places the symmetrised
        // points.
        Self {
            s: libm::sin(angle),
            co: libm::cos(angle),
        }
    }
}

/// [`rotate`] with the sine and cosine already taken.
fn rotate_by(v: [f64; 3], axis: [f64; 3], turn: Turn) -> [f64; 3] {
    let Turn { s, co } = turn;
    rotate_trig(v, axis, s, co)
}

/// Rodrigues rotation with a sine and cosine already taken once.
fn rotate_trig(v: [f64; 3], axis: [f64; 3], s: f64, co: f64) -> [f64; 3] {
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

fn squared_offset(image: [f64; 3], x: f64, y: f64, z: f64) -> f64 {
    let dx = image[0] - x;
    let dy = image[1] - y;
    let dz = image[2] - z;
    dx * dx + dy * dy + dz * dz
}

/// Sine and cosine of one order, from the same `libm` calls as a direct scan.
///
/// `sin(pi / order)` and `sin`/`cos` of `2 pi / order` are computed once.
/// The double-angle formulas are not those calls, so they are not used.
#[derive(Clone, Copy)]
struct OrderTurn {
    order: usize,
    /// `4 * sin(pi / order)^2`, the squared-chord scale.
    chord_scale: f64,
    /// `16 * sin(pi / order)^2`, the leave-the-site scale.
    leave_scale: f64,
    /// `sin(2 pi / order)`.
    sine: f64,
    /// `cos(2 pi / order)`.
    cosine: f64,
}

impl OrderTurn {
    fn new(order: usize) -> Option<Self> {
        if order < 2 {
            return None;
        }
        let half = libm::sin(std::f64::consts::PI / order as f64);
        let angle = 2.0 * std::f64::consts::PI / order as f64;
        Some(Self {
            order,
            chord_scale: 4.0 * half * half,
            leave_scale: 16.0 * half * half,
            sine: libm::sin(angle),
            cosine: libm::cos(angle),
        })
    }
}

/// Whether `sqrt(total / n) >= cap * (1 + 1e-6)` after `total > cap^2 * n`.
///
/// The square of that gate is `edge`. The computed square root does not
/// decrease as `total` grows, and every sum strictly above `edge` is
/// already on the reject side of the comparison. Sums between `cap^2 * n`
/// and `edge`, including `edge` itself, keep the square root.
fn sum_past_cap(total: f64, n: usize, cap: f64) -> bool {
    let limit = cap * cap * (n as f64);
    if !(total > limit) {
        return false;
    }
    let gate = cap * (1.0 + 1.0e-6);
    let edge = gate * gate * (n as f64);
    if total > edge {
        return true;
    }
    (total / n as f64).sqrt() >= gate
}

/// Squared displacement when the image lies strictly inside half the gap.
///
/// The nearest other atom is `gap` away. An image closer than half that is
/// closer to the source atom than to any other, and this squared length is
/// the one the nearest-image scan records for the source. A tie, or a
/// displacement that only meets the half gap, falls through to the scan,
/// which keeps the earlier index.
fn image_on_self(here: [f64; 3], image: [f64; 3], gap: f64) -> Option<f64> {
    if !gap.is_finite() || gap <= 0.0 {
        return None;
    }
    let self_d2 = squared_offset(image, here[0], here[1], here[2]);
    (self_d2 < gap * 0.25).then_some(self_d2)
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
    /// Squared nearest-neighbour gaps, filled the first time a turn has to
    /// be told apart from the identity.
    gaps: std::cell::RefCell<Option<Vec<f64>>>,
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
        Self {
            xs,
            ys,
            zs,
            gaps: std::cell::RefCell::new(None),
        }
    }

    /// Squared distance from each atom to its nearest other atom.
    ///
    /// Built only after a residual has passed, or when a scan with no cap
    /// would otherwise score a turn that maps every atom to itself.
    fn nearest_gaps(&self) -> Vec<f64> {
        let n = self.xs.len();
        let mut nn_sq = vec![f64::INFINITY; n];
        for left in 0..n {
            for right in (left + 1)..n {
                let dx = self.xs[left] - self.xs[right];
                let dy = self.ys[left] - self.ys[right];
                let dz = self.zs[left] - self.zs[right];
                let distance = dx * dx + dy * dy + dz * dz;
                if distance < nn_sq[left] {
                    nn_sq[left] = distance;
                }
                if distance < nn_sq[right] {
                    nn_sq[right] = distance;
                }
            }
        }
        nn_sq
    }

    fn gaps(&self) -> std::cell::Ref<'_, Vec<f64>> {
        if self.gaps.borrow().is_none() {
            let computed = self.nearest_gaps();
            *self.gaps.borrow_mut() = Some(computed);
        }
        std::cell::Ref::map(self.gaps.borrow(), |slot| slot.as_ref().expect("gaps"))
    }

    fn perp_sq_at(&self, index: usize, axis: [f64; 3]) -> f64 {
        let along = self.xs[index] * axis[0] + self.ys[index] * axis[1] + self.zs[index] * axis[2];
        let radius_sq = self.xs[index] * self.xs[index]
            + self.ys[index] * self.ys[index]
            + self.zs[index] * self.zs[index]
            - along * along;
        radius_sq.max(0.0)
    }

    /// An atom can leave its site when its chord reaches half its own
    /// nearest-neighbour gap. Otherwise the triangle inequality puts the
    /// image closer to the source than to any other atom.
    ///
    /// The farthest atom is tested first. A low order moves that atom, so
    /// the scan of the rest does not run. A high order whose turn is shorter
    /// than every gap fails here and never reaches the residual scan.
    fn any_atom_can_leave(&self, axis: [f64; 3], turn: &OrderTurn) -> bool {
        if turn.order < 2 || self.xs.is_empty() {
            return false;
        }
        // chord^2 = 4 * r_perp^2 * sin^2, and (gap / 2)^2 = gap^2 / 4,
        // so chord >= gap / 2 is 16 * r_perp^2 * sin^2 >= gap^2.
        let scale = turn.leave_scale;
        let gaps = self.gaps();
        let can_leave = |index: usize| {
            let gap = gaps[index];
            gap.is_finite() && gap > 0.0 && scale * self.perp_sq_at(index, axis) >= gap
        };
        let mut far = 0usize;
        let mut far_sq = -1.0;
        for index in 0..self.xs.len() {
            let radius = self.perp_sq_at(index, axis);
            if radius > far_sq {
                far_sq = radius;
                far = index;
            }
        }
        if can_leave(far) {
            return true;
        }
        (0..self.xs.len()).any(|index| index != far && can_leave(index))
    }

    /// Nearest atom, eight coordinates at a time.
    ///
    /// Strict `<` keeps the earlier index on a tie. The distances in a
    /// chunk do not depend on each other, so the compiler can autovectorise
    /// them. The index update stays in program order, then the tail.
    fn nearest_index(&self, image: [f64; 3]) -> (usize, f64) {
        let n = self.xs.len();
        let mut index = 0usize;
        let mut best_index = 0usize;
        let mut best = f64::INFINITY;
        while index + 8 <= n {
            let mut dist = [0.0; 8];
            for lane in 0..8 {
                dist[lane] = squared_offset(
                    image,
                    self.xs[index + lane],
                    self.ys[index + lane],
                    self.zs[index + lane],
                );
            }
            for lane in 0..8 {
                if dist[lane] < best {
                    best = dist[lane];
                    best_index = index + lane;
                }
            }
            index += 8;
        }
        while index < n {
            let distance = squared_offset(image, self.xs[index], self.ys[index], self.zs[index]);
            if distance < best {
                best = distance;
                best_index = index;
            }
            index += 1;
        }
        (best_index, best)
    }

    #[cfg(test)]
    fn nearest_index_scalar(&self, image: [f64; 3]) -> (usize, f64) {
        let mut best_index = 0usize;
        let mut best = f64::INFINITY;
        for index in 0..self.xs.len() {
            let distance = squared_offset(image, self.xs[index], self.ys[index], self.zs[index]);
            if distance < best {
                best = distance;
                best_index = index;
            }
        }
        (best_index, best)
    }

    /// Root-mean-square distance of a rotation that permutes off-axis atoms.
    ///
    /// The residual scan is the vectorised one. A turn that cannot move any
    /// atom off its own site is rejected before that scan: its residual is
    /// the chord length, and a high order makes that smaller than the
    /// acceptance tolerance without being a symmetry. A residual that does
    /// pass is kept only when every atom that could have moved did so.
    /// The nearest-image residual of a turn that maps every atom to itself
    /// is the chord length. Once the sum of those squared chords reaches
    /// `cap`, the ordinary scan can reject the turn on its own and the
    /// neighbour list is not needed. A turn that stays under the cap cannot
    /// be rejected that way, so it is kept only when some atom can reach a
    /// different atom.
    fn identity_stays_under(&self, axis: [f64; 3], turn: &OrderTurn, cap: f64) -> bool {
        if !cap.is_finite() {
            return true;
        }
        let scale = turn.chord_scale;
        let limit = cap * cap * self.xs.len() as f64;
        let mut total = 0.0;
        for index in 0..self.xs.len() {
            total += scale * self.perp_sq_at(index, axis);
            if total >= limit {
                return false;
            }
        }
        true
    }

    fn symmetry_deviation(&self, axis: [f64; 3], order: usize, cap: f64) -> Option<f64> {
        let turn = OrderTurn::new(order)?;
        self.deviation_with(axis, &turn, cap)
    }

    fn deviation_with(&self, axis: [f64; 3], turn: &OrderTurn, cap: f64) -> Option<f64> {
        if self.identity_stays_under(axis, turn, cap) && !self.any_atom_can_leave(axis, turn) {
            return None;
        }
        self.exchange_residual(axis, turn, cap)
    }

    /// Residual and the off-axis exchange, from one nearest-image pass.
    ///
    /// The distance sum is the residual. An atom that can leave its site and
    /// whose nearest image is itself is not a permutation of the cloud, and
    /// neither is a turn that lands every atom on itself. The cap stops the
    /// pass at the same threshold as the distance scan. Sine and cosine of
    /// the turn are computed once.
    ///
    /// A finite cap that the distances already exceed returns before the
    /// neighbour list is built. The list is only needed for the exchange.
    fn exchange_residual(&self, axis: [f64; 3], turn: &OrderTurn, cap: f64) -> Option<f64> {
        let n = self.xs.len();
        if n == 0 || turn.order < 2 {
            return None;
        }
        if cap.is_finite()
            && self.gaps.borrow().is_none()
            && self.residual_past_cap(axis, turn, cap)
        {
            return None;
        }
        let scale = turn.chord_scale;
        let s = turn.sine;
        let co = turn.cosine;
        let gaps = self.gaps();
        let mut total = 0.0;
        let mut moved = false;
        let stop = cap.is_finite();
        for index in 0..n {
            let gap = gaps[index];
            let chord_sq = scale * self.perp_sq_at(index, axis);
            // chord^2 < (gap / 2)^2. An atom that close to the axis has no
            // other site it can reach, so staying put is the permutation.
            let axial = !gap.is_finite() || gap <= 0.0 || chord_sq < gap * 0.25;
            let here = self.point(index);
            let image = rotate_by(here, axis, Turn { s, co });
            let (partner, distance) = if let Some(self_d2) = image_on_self(here, image, gap) {
                (index, self_d2)
            } else {
                self.nearest_index(image)
            };
            total += distance;
            if !axial && partner == index {
                return None;
            }
            if partner != index {
                moved = true;
            }
            if stop && sum_past_cap(total, n, cap) {
                return None;
            }
        }
        moved.then_some((total / n as f64).sqrt())
    }

    /// Whether the nearest-image sum is already past `cap`.
    ///
    /// Distances only: no neighbour list and no partner index. Each chunk's
    /// squared lengths do not depend on each other, so the compiler can
    /// autovectorise them. The minimum uses the same `<` as the index scan,
    /// so a tie has the same squared length and the partial sum matches the
    /// one-pass residual. A turn that stays under the cap is not decided
    /// here.
    fn residual_past_cap(&self, axis: [f64; 3], turn: &OrderTurn, cap: f64) -> bool {
        let n = self.xs.len();
        if n == 0 || turn.order < 2 || !cap.is_finite() {
            return false;
        }
        let mut total = 0.0;
        for index in 0..n {
            let image = rotate_trig(self.point(index), axis, turn.sine, turn.cosine);
            total += self.nearest_squared_wide(image);
            if sum_past_cap(total, n, cap) {
                return true;
            }
        }
        false
    }

    /// Smallest squared distance, four coordinates at a time.
    ///
    /// Each lane keeps its own minimum, so the four lengths in a chunk do
    /// not depend on each other and the compiler can autovectorise them.
    /// Strict `<` makes the lane minimum the squared length the index scan
    /// keeps. The tail joins the first lane.
    #[inline(never)]
    fn nearest_squared_wide(&self, image: [f64; 3]) -> f64 {
        let mut lane0 = f64::INFINITY;
        let mut lane1 = f64::INFINITY;
        let mut lane2 = f64::INFINITY;
        let mut lane3 = f64::INFINITY;
        let mut xs = self.xs.chunks_exact(4);
        let mut ys = self.ys.chunks_exact(4);
        let mut zs = self.zs.chunks_exact(4);
        for ((x, y), z) in xs.by_ref().zip(ys.by_ref()).zip(zs.by_ref()) {
            let d0 = squared_offset(image, x[0], y[0], z[0]);
            let d1 = squared_offset(image, x[1], y[1], z[1]);
            let d2 = squared_offset(image, x[2], y[2], z[2]);
            let d3 = squared_offset(image, x[3], y[3], z[3]);
            if d0 < lane0 {
                lane0 = d0;
            }
            if d1 < lane1 {
                lane1 = d1;
            }
            if d2 < lane2 {
                lane2 = d2;
            }
            if d3 < lane3 {
                lane3 = d3;
            }
        }
        for ((&x, &y), &z) in xs
            .remainder()
            .iter()
            .zip(ys.remainder())
            .zip(zs.remainder())
        {
            let distance = squared_offset(image, x, y, z);
            if distance < lane0 {
                lane0 = distance;
            }
        }
        if lane1 < lane0 {
            lane0 = lane1;
        }
        if lane2 < lane0 {
            lane0 = lane2;
        }
        if lane3 < lane0 {
            lane0 = lane3;
        }
        lane0
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
        let turn = Turn::of(2.0 * std::f64::consts::PI / order as f64);
        let mut total = 0.0;
        let stop = cap.is_finite();
        if stop {
            for index in 0..n {
                total += self.nearest_one(rotate_by(self.point(index), axis, turn));
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
                stored[lane] = rotate_by(self.point(index + lane), axis, turn);
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
    let turns: Vec<OrderTurn> = orders
        .iter()
        .filter_map(|&order| OrderTurn::new(order))
        .collect();
    for axis in candidate_axes(x, n) {
        for turn in &turns {
            let Some(d) = cloud.deviation_with(axis, turn, tolerance) else {
                continue;
            };
            if d < tolerance && best.map(|b| d < b.deviation).unwrap_or(true) {
                best = Some(Candidate {
                    axis,
                    order: turn.order,
                    improper: false,
                    deviation: d,
                });
            }
        }
    }
    best
}

/// Lowest nearest-image residual about one axis, among orders that exchange atoms.
///
/// A high order whose turn is shorter than the nearest-neighbour gap matches
/// every atom to itself and is not a candidate. `None` when no requested
/// order moves an atom onto a different atom.
pub fn best_order_about(
    x: ArrayView1<f64>,
    n: usize,
    axis: [f64; 3],
    orders: &[usize],
) -> Option<usize> {
    let cloud = Cloud::from_view(x, n);
    let mut best_order = None;
    let mut best = f64::INFINITY;
    for &order in orders {
        let Some(deviation) = cloud.symmetry_deviation(axis, order, f64::INFINITY) else {
            continue;
        };
        if deviation < best {
            best = deviation;
            best_order = Some(order);
        }
    }
    best_order
}

/// Best proper rotation of each requested order.
///
/// One centred cloud and one axis list serve every order. An order whose
/// rotation cannot exchange atoms is omitted. The survivors are in the
/// order they were requested, which is the order a later group closure sees.
pub fn detect_orders(
    x: ArrayView1<f64>,
    n: usize,
    orders: &[usize],
    tolerance: f64,
) -> Vec<Candidate> {
    let cloud = Cloud::from_view(x, n);
    let axes = candidate_axes(x, n);
    let mut out = Vec::new();
    for &order in orders {
        let Some(turn) = OrderTurn::new(order) else {
            continue;
        };
        let mut best: Option<Candidate> = None;
        for &axis in &axes {
            let Some(d) = cloud.deviation_with(axis, &turn, tolerance) else {
                continue;
            };
            if d < tolerance && best.as_ref().map(|b| d < b.deviation).unwrap_or(true) {
                best = Some(Candidate {
                    axis,
                    order: turn.order,
                    improper: false,
                    deviation: d,
                });
            }
        }
        if let Some(candidate) = best {
            out.push(candidate);
        }
    }
    out
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
    let turns: Vec<OrderTurn> = orders
        .iter()
        .filter_map(|&order| OrderTurn::new(order))
        .collect();
    for axis in candidate_axes(x, n) {
        for turn in &turns {
            let Some(d) = cloud.deviation_with(axis, turn, tolerance) else {
                continue;
            };
            if d < tolerance {
                out.push(Candidate {
                    axis,
                    order: turn.order,
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

/// Makes `x` exactly symmetric under the rotation `cand` describes.
///
/// Each point is followed around its orbit under the rotation, the orbit is
/// averaged after rotating every member back to the first one's frame, and
/// every member is replaced by the rotated average. A point whose orbit does
/// not close, meaning some image has no partner within `pair_cutoff`, is left
/// where it is rather than dragged onto a partner it does not have.
///
/// Applied fully rather than blended. A partial push toward a symmetry the
/// structure nearly has produces a structure that is neither, and the point of
/// the scheme is to land in the symmetric basin so the relaxation can take it
/// from there.
pub fn symmetrise(x: ArrayView1<f64>, n: usize, cand: &Candidate, pair_cutoff: f64) -> Array1<f64> {
    let mut out = x.to_owned();
    if n == 0 || cand.order < 2 {
        return out;
    }
    let c = centroid(x, n);
    let angle = 2.0 * std::f64::consts::PI / cand.order as f64;
    let step = Turn::of(angle);
    let rel =
        |a: usize| -> [f64; 3] { [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]] };

    let mut done = vec![false; n];
    for a in 0..n {
        if done[a] {
            continue;
        }
        // Walk the orbit: from each member, rotate once and take the nearest
        // point as the next member.
        let mut orbit = vec![a];
        let mut cur = a;
        let mut closed = true;
        for _ in 1..cand.order {
            let w = rotate_by(rel(cur), cand.axis, step);
            let mut best = usize::MAX;
            let mut best_d = f64::INFINITY;
            for b in 0..n {
                let u = rel(b);
                let d =
                    ((w[0] - u[0]).powi(2) + (w[1] - u[1]).powi(2) + (w[2] - u[2]).powi(2)).sqrt();
                if d < best_d {
                    best_d = d;
                    best = b;
                }
            }
            if best_d > pair_cutoff || best == usize::MAX {
                closed = false;
                break;
            }
            orbit.push(best);
            cur = best;
        }
        if !closed {
            done[a] = true;
            continue;
        }
        // A point on the axis maps to itself; its orbit collapses and there is
        // nothing to average.
        if orbit.iter().any(|&b| done[b]) {
            for &b in &orbit {
                done[b] = true;
            }
            continue;
        }

        // Average in the first member's frame: rotate member k back by k steps.
        let mut mean = [0.0_f64; 3];
        for (k, &b) in orbit.iter().enumerate() {
            let back = rotate(rel(b), cand.axis, -(k as f64) * angle);
            for i in 0..3 {
                mean[i] += back[i];
            }
        }
        let m = orbit.len() as f64;
        for v in mean.iter_mut() {
            *v /= m;
        }
        // Place every member at the rotated average.
        for (k, &b) in orbit.iter().enumerate() {
            let p = rotate(mean, cand.axis, k as f64 * angle);
            for i in 0..3 {
                out[3 * b + i] = c[i] + p[i];
            }
            done[b] = true;
        }
    }
    out
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

fn act(r: &Rot, v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
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

/// Makes `x` symmetric under a whole group, not one rotation.
///
/// Each point's orbit is collected by applying every group element and taking
/// the nearest partner, then the orbit is averaged in the frame of the element
/// that produced it and every member is replaced. A point whose orbit does not
/// close under some element is left alone rather than dragged onto a partner it
/// does not have.
pub fn symmetrise_group(
    x: ArrayView1<f64>,
    n: usize,
    group: &[Rot],
    pair_cutoff: f64,
) -> Array1<f64> {
    symmetrise_group_within(x, n, group, pair_cutoff, &[])
}

/// As [`symmetrise_group`], but an orbit may not pair atoms of different
/// `classes`. An empty `classes` slice is the unlabelled group. A labelled
/// orbit that cannot close inside one class is left unmoved.
pub fn symmetrise_group_within(
    x: ArrayView1<f64>,
    n: usize,
    group: &[Rot],
    pair_cutoff: f64,
    classes: &[u32],
) -> Array1<f64> {
    let mut out = x.to_owned();
    if n == 0 || group.len() < 2 {
        return out;
    }
    let c = centroid(x, n);
    let rel =
        |a: usize| -> [f64; 3] { [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]] };

    let mut done = vec![false; n];
    for a in 0..n {
        if done[a] {
            continue;
        }
        // Partner of `a` under each group element. Different classes
        // cannot be images of each other: a species label is not a
        // point-group orbit.
        let mut members: Vec<(usize, usize)> = Vec::with_capacity(group.len());
        let mut closed = true;
        for (gi, g) in group.iter().enumerate() {
            let w = act(g, rel(a));
            let mut best = usize::MAX;
            let mut best_d = f64::INFINITY;
            for b in 0..n {
                if classes.len() == n && classes[a] != classes[b] {
                    continue;
                }
                let u = rel(b);
                let d =
                    ((w[0] - u[0]).powi(2) + (w[1] - u[1]).powi(2) + (w[2] - u[2]).powi(2)).sqrt();
                if d < best_d {
                    best_d = d;
                    best = b;
                }
            }
            if best_d > pair_cutoff || best == usize::MAX {
                closed = false;
                break;
            }
            members.push((gi, best));
        }
        if !closed {
            done[a] = true;
            continue;
        }
        // Average in `a`'s frame: undo each element by using its transpose,
        // which is its inverse for a rotation.
        let mut mean = [0.0_f64; 3];
        for (gi, b) in &members {
            let g = &group[*gi];
            let gt: Rot = [
                [g[0][0], g[1][0], g[2][0]],
                [g[0][1], g[1][1], g[2][1]],
                [g[0][2], g[1][2], g[2][2]],
            ];
            let back = act(&gt, rel(*b));
            for k in 0..3 {
                mean[k] += back[k];
            }
        }
        let m = members.len() as f64;
        for v in mean.iter_mut() {
            *v /= m;
        }
        for (gi, b) in &members {
            if done[*b] {
                continue;
            }
            let p = act(&group[*gi], mean);
            for k in 0..3 {
                out[3 * b + k] = c[k] + p[k];
            }
            done[*b] = true;
        }
        done[a] = true;
    }
    out
}

/// Core symmetrisation after Oakley, Johnston and Wales: detect the
/// approximate point group of the innermost `core_fraction` of the points,
/// close it into a group, and make the core exactly symmetric under it.
/// Surface points are left where they are; the quench that follows places
/// them. Returns `None` when the core carries no approximate symmetry or the
/// symmetrised core is the input.
///
/// Detecting on the core is what makes the scheme fire on a realistic
/// minimum: a 75-point decahedral or icosahedral core with a disordered
/// surface fails a whole-cluster tolerance that its core passes easily.
pub fn symmetrise_core(
    x: ArrayView1<f64>,
    n: usize,
    tolerance: f64,
    pair_cutoff: f64,
    core_fraction: f64,
) -> Option<Array1<f64>> {
    if n < 4 {
        return None;
    }
    let c = centroid(x, n);
    let mut radii: Vec<(f64, usize)> = (0..n)
        .map(|a| {
            let v = [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]];
            ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(), a)
        })
        .collect();
    radii.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
    let m = ((n as f64) * core_fraction.clamp(0.1, 1.0))
        .round()
        .max(4.0) as usize;
    let m = m.min(n);
    let core_idx: Vec<usize> = radii[..m].iter().map(|p| p.1).collect();
    let mut core = Array1::<f64>::zeros(3 * m);
    for (k, &a) in core_idx.iter().enumerate() {
        for d in 0..3 {
            core[3 * k + d] = x[3 * a + d];
        }
    }
    let orders = orders_held(m);
    let cands = detect_all(core.view(), m, &orders, tolerance);
    if cands.is_empty() {
        return None;
    }
    let mut gens: Vec<Candidate> = Vec::new();
    for cand in &cands {
        let seen = gens.iter().any(|g| {
            let dot =
                g.axis[0] * cand.axis[0] + g.axis[1] * cand.axis[1] + g.axis[2] * cand.axis[2];
            dot.abs() > PARALLEL_COSINE && g.improper == cand.improper
        });
        if !seen {
            gens.push(*cand);
        }
        if gens.len() >= 3 {
            break;
        }
    }
    let group = generate_group(&gens, 48);
    let sym_core = if group.len() > 1 {
        symmetrise_group(core.view(), m, &group, pair_cutoff)
    } else {
        symmetrise(core.view(), m, &cands[0], pair_cutoff)
    };
    let moved = sym_core
        .iter()
        .zip(core.iter())
        .any(|(a, b)| (a - b).abs() > 1e-9);
    if !moved {
        return None;
    }
    let mut out = x.to_owned();
    for (k, &a) in core_idx.iter().enumerate() {
        for d in 0..3 {
            out[3 * a + d] = sym_core[3 * k + d];
        }
    }
    Some(out)
}

/// Orbit completion under a point group: every atom is moved onto the
/// symmetry-equivalent site set of the group, so partial orbits close.
///
/// Oakley, Johnston and Wales (Phys. Chem. Chem. Phys. 15, 3965, 2013)
/// symmetrise a cluster with the point group of its core: the images of
/// every atom under every operation are collected into sites, the sites
/// with the most images are the ones the structure already nearly has,
/// and the atoms are placed on the `n` best-supported sites, nearest
/// first. Surface atoms that sat off any orbit move onto empty orbit
/// positions; a structure already symmetric is unchanged. Sites closer
/// than `min_separation` to a chosen site are skipped so the result has
/// no overlapping atoms; the caller quenches and applies its acceptance
/// rule. Returns `None` when nothing moves.
pub fn complete_orbits(
    x: ArrayView1<f64>,
    n: usize,
    group: &[Rot],
    pair_cutoff: f64,
    min_separation: f64,
) -> Option<Array1<f64>> {
    if n == 0 || group.len() < 2 {
        return None;
    }
    complete_orbits_about(x, n, group, centroid(x, n), pair_cutoff, min_separation)
}

/// Squared distance is inside `pair_cutoff` after the same root test as a
/// direct comparison. A value more than a few ulps above the squared
/// cutoff cannot round back to a root at or below it, so the far pairs
/// skip the root. The accepted site is unchanged.
fn within_cutoff(d2: f64, pair_cutoff: f64) -> bool {
    let limit = pair_cutoff * pair_cutoff;
    let slack = limit * (64.0 * f64::EPSILON);
    if d2 > limit + slack {
        return false;
    }
    d2.sqrt() <= pair_cutoff
}

/// [`complete_orbits`] with the group acting about `c` rather than the
/// centroid, for a group read off a core whose centre is not the
/// centroid of the whole structure.
pub fn complete_orbits_about(
    x: ArrayView1<f64>,
    n: usize,
    group: &[Rot],
    c: [f64; 3],
    pair_cutoff: f64,
    min_separation: f64,
) -> Option<Array1<f64>> {
    if n == 0 || group.len() < 2 {
        return None;
    }
    let rel =
        |a: usize| -> [f64; 3] { [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]] };
    // Sites: merged images, with their support (images landing on them).
    let mut sites: Vec<([f64; 3], usize, usize)> = Vec::new();
    for a in 0..n {
        let r = rel(a);
        for g in group {
            let w = act(g, r);
            let mut hit = None;
            for (i, (p, _, _)) in sites.iter().enumerate() {
                let dx = w[0] - p[0];
                let dy = w[1] - p[1];
                let dz = w[2] - p[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if within_cutoff(d2, pair_cutoff) {
                    hit = Some(i);
                    break;
                }
            }
            match hit {
                Some(i) => {
                    let (p, count, _) = &mut sites[i];
                    let m = *count as f64;
                    for k in 0..3 {
                        p[k] = (p[k] * m + w[k]) / (m + 1.0);
                    }
                    *count += 1;
                }
                None => sites.push((w, 1, a)),
            }
        }
    }
    // Best-supported sites first; ties by distance from the centre so the
    // core is filled before the surface, as the cluster is.
    sites.sort_by(|p, q| {
        q.1.cmp(&p.1).then_with(|| {
            let rp = p.0.iter().map(|v| v * v).sum::<f64>();
            let rq = q.0.iter().map(|v| v * v).sum::<f64>();
            rp.partial_cmp(&rq).unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    let mut chosen: Vec<[f64; 3]> = Vec::with_capacity(n);
    for (p, _, _) in &sites {
        if chosen.len() == n {
            break;
        }
        let clear = chosen.iter().all(|q| {
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
                >= min_separation
        });
        if clear {
            chosen.push(*p);
        }
    }
    if chosen.len() < n {
        return None;
    }
    // Assign atoms to sites nearest first, so the move is the smallest
    // permutation-free displacement onto the symmetric site set.
    let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(n * n);
    for a in 0..n {
        let r = rel(a);
        for (i, p) in chosen.iter().enumerate() {
            // Squared length sorts in the same order as the length, and a
            // stable sort keeps an equal pair where it was. The root is
            // taken only for the pairs that are actually assigned.
            let d2 = (r[0] - p[0]).powi(2) + (r[1] - p[1]).powi(2) + (r[2] - p[2]).powi(2);
            pairs.push((d2, a, i));
        }
    }
    pairs.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut atom_done = vec![false; n];
    let mut site_done = vec![false; n];
    let mut out = x.to_owned();
    let mut moved = false;
    let mut assigned = 0;
    for (d, a, i) in pairs {
        if atom_done[a] || site_done[i] {
            continue;
        }
        atom_done[a] = true;
        site_done[i] = true;
        assigned += 1;
        for k in 0..3 {
            out[3 * a + k] = c[k] + chosen[i][k];
        }
        if d.sqrt() > 1e-9 {
            moved = true;
        }
        if assigned == n {
            break;
        }
    }
    moved.then_some(out)
}

/// Orbit completion with the point group detected on the core.
///
/// The core's group (as in [`symmetrise_core`]) is applied to every atom
/// through [`complete_orbits`]; this is the whole-cluster symmetrisation
/// that Oakley, Johnston and Wales report as the productive scheme on
/// the 98-point cluster, where the tetrahedral core is found long before
/// its surface completes.
pub fn orbit_complete_core(
    x: ArrayView1<f64>,
    n: usize,
    tolerance: f64,
    pair_cutoff: f64,
    core_fraction: f64,
    min_separation: f64,
) -> Option<Array1<f64>> {
    if n < 4 {
        return None;
    }
    let c = centroid(x, n);
    let mut radii: Vec<(f64, usize)> = (0..n)
        .map(|a| {
            let v = [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]];
            ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(), a)
        })
        .collect();
    radii.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
    let m = ((n as f64) * core_fraction.clamp(0.1, 1.0))
        .round()
        .max(4.0) as usize;
    let m = m.min(n);
    let mut core = Array1::<f64>::zeros(3 * m);
    for (k, &(_, a)) in radii[..m].iter().enumerate() {
        for d in 0..3 {
            core[3 * k + d] = x[3 * a + d];
        }
    }
    let orders = orders_held(m);
    let cands = detect_all(core.view(), m, &orders, tolerance);
    if cands.is_empty() {
        return None;
    }
    let mut gens: Vec<Candidate> = Vec::new();
    for cand in &cands {
        let seen = gens.iter().any(|g| {
            let dot =
                g.axis[0] * cand.axis[0] + g.axis[1] * cand.axis[1] + g.axis[2] * cand.axis[2];
            dot.abs() > PARALLEL_COSINE && g.improper == cand.improper
        });
        if !seen {
            gens.push(*cand);
        }
        if gens.len() >= 3 {
            break;
        }
    }
    let group = generate_group(&gens, 48);
    if group.len() < 2 {
        return None;
    }
    complete_orbits(x, n, &group, pair_cutoff, min_separation)
}

/// How [`shell_core_group`] finds a core and reads its point group, after
/// the `SYMMETRY` routine in Wales's global minimisation program (GMIN).
///
/// The centre is the centroid weighted by `exp(-decay r)`, iterated until
/// it moves less than `centre_step`, then moved to the plain centroid of
/// the points inside the widest gap in distance from it. Points whose
/// sorted distances from the centre differ by less than `shell_width` form
/// a shell, and shells are added from the inside until [`detect_all`] at
/// `tolerance` finds an operation, giving up past `max_points` points.
/// Each operation found is refined to the orthogonal map that best carries
/// the core onto the partners it matches, and the refined set is closed by
/// [`close_group`] at `closure`, at most `cap` elements.
///
/// The core is small and found by its own radial structure, so a disordered
/// surface does not enter detection; and the closure is tolerant, so
/// operations that are only approximately commensurate still close into
/// the group they approximate rather than running to the cap.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ShellCore {
    /// Rate of the exponential weights of the first centre, per length.
    pub decay: f64,
    /// Movement below which the weighted centre counts as converged.
    pub centre_step: f64,
    /// Largest gap in distance from the centre within one shell.
    pub shell_width: f64,
    /// Most points the core may have.
    pub max_points: usize,
    /// Largest deviation, as in [`detect_all`], of an operation of the core.
    pub tolerance: f64,
    /// Largest difference in any matrix element between two products that
    /// [`close_group`] counts as one operation.
    pub closure: f64,
    /// Most elements of the closed group.
    pub cap: usize,
}

impl ShellCore {
    /// GMIN's 75-point `SYMMETRISE` settings where it has one (`DISTFAC`
    /// 2.4, `SYMTOL1` 0.01, `MATDIFF` 0.27), a 0.1 detection tolerance, and
    /// the 120 elements of the full icosahedral group, at `length`.
    pub fn scaled(length: f64) -> Self {
        Self {
            decay: 2.4 / length,
            centre_step: 1e-4 * length,
            shell_width: 0.01 * length,
            max_points: 20,
            tolerance: 0.1 * length,
            closure: 0.27,
            cap: 120,
        }
    }
}

/// A point group read off a core, with the centre it acts about.
#[derive(Debug, Clone, PartialEq)]
pub struct CoreGroup {
    /// The closed group, identity first.
    pub group: Vec<Rot>,
    /// Indices of the core's points.
    pub core: Vec<usize>,
    /// The centre the group acts about.
    pub centre: [f64; 3],
}

/// Determinant magnitude below which [`inverse`] reports a matrix as singular.
const SINGULAR_DETERMINANT: f64 = 1e-12;
/// Newton steps of [`refine_operation`]'s polar factor, which converges quadratically.
const POLAR_STEPS: usize = 60;
/// Most steps of [`shell_core_group`]'s weighted-centre iteration before it settles for the last centre.
const CENTRE_STEPS: usize = 200;
/// Fewest points [`shell_core_group`] reads a group from, the smallest set that need not be planar.
const MIN_CORE_POINTS: usize = 4;

/// Rotation orders a cloud of `n` points can carry.
///
/// A cycle longer than the point count cannot send an atom onto a
/// different atom, so detection stops at `n`. One point has no turn.
pub(crate) fn orders_held(n: usize) -> Vec<usize> {
    if n < 2 {
        return Vec::new();
    }
    (2..=n).collect()
}
/// Cosine above which two detected operations' directions count as one, so only the first is kept.
const PARALLEL_COSINE: f64 = 0.995;

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn point(x: ArrayView1<f64>, a: usize) -> [f64; 3] {
    [x[3 * a], x[3 * a + 1], x[3 * a + 2]]
}

fn centroid_of(x: ArrayView1<f64>, idx: &[usize]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for &a in idx {
        for k in 0..3 {
            c[k] += x[3 * a + k];
        }
    }
    for v in c.iter_mut() {
        *v /= idx.len().max(1) as f64;
    }
    c
}

fn transpose(a: &Rot) -> Rot {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[j][i];
        }
    }
    o
}

fn inverse(a: &Rot) -> Option<Rot> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < SINGULAR_DETERMINANT {
        return None;
    }
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (i1, i2) = ((j + 1) % 3, (j + 2) % 3);
            let (j1, j2) = ((i + 1) % 3, (i + 2) % 3);
            o[i][j] = (a[i1][j1] * a[i2][j2] - a[i1][j2] * a[i2][j1]) / det;
        }
    }
    Some(o)
}

fn operation(c: &Candidate) -> Rot {
    if c.improper {
        mirror_matrix(c.axis)
    } else {
        rot_matrix(c.axis, 2.0 * std::f64::consts::PI / c.order.max(2) as f64)
    }
}

/// The orthogonal map nearest `g` that carries the points of `core`
/// (relative to the centre) onto the partners `g` matches them to: the
/// orthogonal polar factor of `sum q p^T` over each point `p` and the
/// point `q` nearest its image, by Newton's iteration, which keeps the
/// sign of the determinant. `g` itself when that sum is singular.
pub fn refine_operation(g: &Rot, core: &[[f64; 3]]) -> Rot {
    let mut h = [[0.0; 3]; 3];
    for p in core {
        let w = act(g, *p);
        let q = core
            .iter()
            .min_by(|a, b| {
                distance(**a, w)
                    .partial_cmp(&distance(**b, w))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(w);
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += q[i] * p[j];
            }
        }
    }
    let mut x = h;
    for _ in 0..POLAR_STEPS {
        let Some(inv) = inverse(&x) else { return *g };
        let it = transpose(&inv);
        let mut next = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                next[i][j] = 0.5 * (x[i][j] + it[i][j]);
            }
        }
        x = next;
    }
    x
}

/// The group generated by `gens`, counting two products as one operation
/// when every matrix element agrees to `closure`, at most `cap` elements.
///
/// [`generate_group`] compares to `1e-6`, which is right for exact
/// operations and wrong for approximate ones: two products of slightly
/// incommensurate generators differ by more than that, so the closure
/// never stops and the group runs to its cap. GMIN compares the same way
/// at `MATDIFF`.
pub fn close_group(gens: &[Rot], closure: f64, cap: usize) -> Vec<Rot> {
    let same =
        |a: &Rot, b: &Rot| (0..3).all(|i| (0..3).all(|j| (a[i][j] - b[i][j]).abs() <= closure));
    let identity: Rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut group = vec![identity];
    for g in gens {
        if !group.iter().any(|q| same(q, g)) {
            group.push(*g);
        }
    }
    let mut changed = true;
    while changed && group.len() < cap {
        changed = false;
        let snapshot = group.clone();
        'outer: for a in &snapshot {
            for b in &snapshot {
                let p = mul(a, b);
                if !group.iter().any(|q| same(q, &p)) {
                    group.push(p);
                    changed = true;
                    if group.len() >= cap {
                        break 'outer;
                    }
                }
            }
        }
    }
    group
}

/// The point group of the innermost shells of `x`, as [`ShellCore`]
/// describes. `None` when no core of at most `spec.max_points` points has
/// an operation.
pub fn shell_core_group(x: ArrayView1<f64>, n: usize, spec: &ShellCore) -> Option<CoreGroup> {
    if n < MIN_CORE_POINTS {
        return None;
    }
    let mut c = centroid(x, n);
    for _ in 0..CENTRE_STEPS {
        let mut num = [0.0; 3];
        let mut den = 0.0;
        for a in 0..n {
            let w = libm::exp(-spec.decay * distance(point(x, a), c));
            for k in 0..3 {
                num[k] += w * x[3 * a + k];
            }
            den += w;
        }
        let next = [num[0] / den, num[1] / den, num[2] / den];
        let step = distance(next, c);
        c = next;
        if step < spec.centre_step {
            break;
        }
    }
    let by_radius = |c: [f64; 3]| {
        let mut radii: Vec<(f64, usize)> = (0..n).map(|a| (distance(point(x, a), c), a)).collect();
        radii.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
        radii
    };
    let radii = by_radius(c);
    let mut gap = -1.0;
    let mut inside = 1;
    for j in 1..n {
        let d = radii[j].0 - radii[j - 1].0;
        if d > gap {
            gap = d;
            inside = j;
        }
    }
    let inner: Vec<usize> = radii[..inside].iter().map(|p| p.1).collect();
    let c = centroid_of(x, &inner);
    let radii = by_radius(c);
    let mut m = 0;
    while m < n {
        let mut e = m + 1;
        while e < n && radii[e].0 - radii[e - 1].0 < spec.shell_width {
            e += 1;
        }
        m = e;
        if m > spec.max_points {
            return None;
        }
        if m < MIN_CORE_POINTS {
            continue;
        }
        let core: Vec<usize> = radii[..m].iter().map(|p| p.1).collect();
        let rel: Vec<[f64; 3]> = core
            .iter()
            .map(|&a| [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]])
            .collect();
        let mut cx = Array1::<f64>::zeros(3 * m);
        for (k, p) in rel.iter().enumerate() {
            for d in 0..3 {
                cx[3 * k + d] = p[d];
            }
        }
        let orders = orders_held(m);
        let cands = detect_all(cx.view(), m, &orders, spec.tolerance);
        if cands.is_empty() {
            continue;
        }
        let mut picked: Vec<Candidate> = Vec::new();
        for cand in &cands {
            let seen = picked.iter().any(|g| {
                let dot =
                    g.axis[0] * cand.axis[0] + g.axis[1] * cand.axis[1] + g.axis[2] * cand.axis[2];
                dot.abs() > PARALLEL_COSINE && g.improper == cand.improper
            });
            if !seen {
                picked.push(*cand);
            }
        }
        let gens: Vec<Rot> = picked
            .iter()
            .map(|g| refine_operation(&operation(g), &rel))
            .collect();
        let group = close_group(&gens, spec.closure, spec.cap);
        return Some(CoreGroup {
            group,
            core,
            centre: c,
        });
    }
    None
}

/// Orbit completion with the point group of the innermost shells
/// ([`shell_core_group`]), every atom placed about the core's centre by
/// [`complete_orbits_about`]. `None` when the core has no group or nothing
/// moves.
pub fn orbit_complete_shells(
    x: ArrayView1<f64>,
    n: usize,
    spec: &ShellCore,
    pair_cutoff: f64,
    min_separation: f64,
) -> Option<Array1<f64>> {
    let core = shell_core_group(x, n, spec)?;
    complete_orbits_about(x, n, &core.group, core.centre, pair_cutoff, min_separation)
}

/// Detects and applies in one step, or returns `None` when there is no
/// approximate symmetry worth using.
pub fn symmetrise_detected(
    x: ArrayView1<f64>,
    n: usize,
    orders: &[usize],
    tolerance: f64,
    pair_cutoff: f64,
) -> Option<(Array1<f64>, Candidate)> {
    let cand = detect(x, n, orders, tolerance)?;
    Some((symmetrise(x, n, &cand, pair_cutoff), cand))
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

    /// The rotation as it was, taking the sine and cosine for every point.
    fn rotate_taking_trig_per_point(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
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

    /// The deviation as it was: every image rotated with its own sine and
    /// cosine, the same nearest-image scan and the same early stop.
    fn deviation_taking_trig_per_point(
        cloud: &Cloud,
        axis: [f64; 3],
        order: usize,
        cap: f64,
    ) -> Option<f64> {
        let n = cloud.xs.len();
        if n == 0 || order < 2 {
            return None;
        }
        let angle = 2.0 * std::f64::consts::PI / order as f64;
        let mut total = 0.0;
        if cap.is_finite() {
            for index in 0..n {
                total += cloud.nearest_one(rotate_taking_trig_per_point(
                    cloud.point(index),
                    axis,
                    angle,
                ));
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
                stored[lane] = rotate_taking_trig_per_point(cloud.point(index + lane), axis, angle);
            }
            if !cloud.accumulate(&stored[..width], cap, &mut total) {
                return None;
            }
            index += width;
        }
        Some((total / n as f64).sqrt())
    }

    #[test]
    fn one_sine_and_cosine_per_rotation_leaves_every_deviation_unchanged() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for n in [1usize, 3, 4, 7, 13, 38, 49, 98] {
            let x = Array1::from((0..3 * n).map(|_| 4.0 * next()).collect::<Vec<_>>());
            let cloud = Cloud::from_view(x.view(), n);
            for axis in candidate_axes(x.view(), n) {
                for order in [0usize, 1, 2, 3, 4, 5, 6, 7] {
                    for cap in [0.05, 0.3, 1.0, 3.0, f64::INFINITY] {
                        let now = cloud.rotation_deviation(axis, order, cap);
                        let before = deviation_taking_trig_per_point(&cloud, axis, order, cap);
                        assert_eq!(
                            now.map(f64::to_bits),
                            before.map(f64::to_bits),
                            "n {n} order {order} cap {cap}"
                        );
                    }
                }
                for angle in [0.3, -1.7, 2.0 * std::f64::consts::PI / 5.0] {
                    let v = [next(), next(), next()];
                    assert_eq!(
                        rotate(v, axis, angle).map(f64::to_bits),
                        rotate_taking_trig_per_point(v, axis, angle).map(f64::to_bits)
                    );
                }
            }
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
    fn a_regular_octagon_keeps_its_eightfold_axis() {
        let mut coords = Vec::with_capacity(24);
        for step in 0..8 {
            let angle = std::f64::consts::TAU * step as f64 / 8.0;
            coords.push(angle.cos());
            coords.push(angle.sin());
            coords.push(0.0);
        }
        let x = Array1::from(coords);
        let found = detect(x.view(), 8, &[8], 0.05).expect("octagon");
        assert_eq!(found.order, 8);
        assert!(found.deviation < 1e-8, "{}", found.deviation);
    }

    #[test]
    fn a_turn_shorter_than_the_nearest_pair_is_not_a_symmetry() {
        let mut coords = Vec::with_capacity(24);
        for step in 0..8 {
            let angle = std::f64::consts::TAU * step as f64 / 8.0;
            coords.push(3.0 * angle.cos());
            coords.push(3.0 * angle.sin());
            coords.push(0.0);
        }
        let x = Array1::from(coords);
        let rms = deviation(x.view(), 8, [0.0, 0.0, 1.0], 40);
        assert!(
            rms < 0.5,
            "a fortieth of a turn is a small nearest-image residual, got {rms}"
        );
        assert!(detect(x.view(), 8, &[40], 0.5).is_none());
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

    /// The point of the operation: a nearly symmetric structure comes back much
    /// more symmetric than it went in.
    #[test]
    fn symmetrising_reduces_the_deviation() {
        let x = jitter(&octahedron(1.0), 0.05);
        let cand = detect(x.view(), 6, &[2, 3, 4], 0.5).expect("nothing detected");
        let before = cand.deviation;
        let y = symmetrise(x.view(), 6, &cand, 1.0);
        let after = deviation(y.view(), 6, cand.axis, cand.order);
        assert!(
            after < before * 0.2,
            "deviation went from {before} to {after}"
        );
    }

    /// And it must not move a structure that is already exact.
    #[test]
    fn an_exactly_symmetric_structure_is_left_alone() {
        let x = octahedron(1.1);
        let cand = Candidate {
            axis: [0.0, 0.0, 1.0],
            order: 4,
            improper: false,
            deviation: 0.0,
        };
        let y = symmetrise(x.view(), 6, &cand, 1.0);
        for (p, q) in x.iter().zip(y.iter()) {
            assert!((p - q).abs() < 1e-9, "{p} moved to {q}");
        }
    }

    /// A structure with no approximate symmetry is left alone rather than
    /// pushed onto one it does not have. This is the case where symmetrising
    /// about a random axis does damage.
    #[test]
    fn a_structure_with_no_symmetry_is_not_forced_into_one() {
        let n = 7;
        let mut x = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            let a = (i as f64) * 1.317;
            let r = 0.8 + 0.31 * ((i % 5) as f64);
            x[3 * i] = r * a.cos();
            x[3 * i + 1] = r * a.sin();
            x[3 * i + 2] = 0.53 * ((i % 4) as f64) - 0.9;
        }
        assert!(
            detect(x.view(), n, &[2, 3, 4, 5, 6], 0.05).is_none(),
            "a generic structure was reported as nearly symmetric"
        );
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

    /// Averaging over the whole group beats averaging over one axis, which is
    /// the claim that makes the group version worth its cost.
    #[test]
    fn the_group_symmetrises_further_than_one_axis() {
        let x = jitter(&octahedron(1.0), 0.05);
        let one = detect(x.view(), 6, &[2, 3, 4], 0.5).expect("nothing detected");
        let single = symmetrise(x.view(), 6, &one, 1.0);

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
        let group = generate_group(&gens, 120);
        let full = symmetrise_group(x.view(), 6, &group, 1.0);

        // Measured against a symmetry the single-axis pass did not use.
        let axis = [0.0, 0.0, 1.0];
        let d_single = deviation(single.view(), 6, axis, 4);
        let d_full = deviation(full.view(), 6, axis, 4);
        assert!(
            d_full < d_single,
            "group gave {d_full} against single axis {d_single}"
        );
    }

    /// A structure already symmetric under the group must not move.
    #[test]
    fn the_group_leaves_an_exact_structure_alone() {
        let x = octahedron(1.1);
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
        let group = generate_group(&gens, 120);
        let y = symmetrise_group(x.view(), 6, &group, 1.0);
        for (p, q) in x.iter().zip(y.iter()) {
            assert!((p - q).abs() < 1e-9, "{p} moved to {q}");
        }
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

    #[test]
    fn the_centroid_is_preserved() {
        let x = jitter(&octahedron(1.0), 0.04);
        let cand = detect(x.view(), 6, &[2, 3, 4], 0.5).unwrap();
        let y = symmetrise(x.view(), 6, &cand, 1.0);
        let cx = centroid(x.view(), 6);
        let cy = centroid(y.view(), 6);
        for k in 0..3 {
            assert!((cx[k] - cy[k]).abs() < 1e-9, "centroid moved on axis {k}");
        }
    }
}

#[cfg(test)]
mod orbit_tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn a_far_pair_skips_the_root_and_a_near_pair_keeps_it() {
        let cutoff = 0.7;
        let mut state = 1u64;
        for _ in 0..4000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            let distance = cutoff * (0.2 + 2.2 * unit);
            let d2 = distance * distance;
            let fast = super::within_cutoff(d2, cutoff);
            let slow = d2.sqrt() <= cutoff;
            assert_eq!(fast, slow, "distance {distance}");
        }
        assert!(super::within_cutoff(cutoff * cutoff, cutoff));
        assert!(!super::within_cutoff(
            (cutoff * 2.0) * (cutoff * 2.0),
            cutoff
        ));
    }

    /// A regular octahedron with one vertex pushed off its site is put back
    /// on it by orbit completion under the octahedron's own C4 axes, and a
    /// structure already on its orbits is left alone.
    #[test]
    fn orbit_completion_returns_a_displaced_vertex_to_its_site() {
        let mut x = Array1::from(vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ]);
        let axes = [
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
        let group = generate_group(&axes, 48);
        assert!(group.len() >= 8);
        assert!(complete_orbits(x.view(), 6, &group, 0.2, 0.5).is_none());
        x[15] = 0.15;
        x[17] = -0.9;
        let y = complete_orbits(x.view(), 6, &group, 0.3, 0.5).expect("a vertex moved");
        let target = [0.0, 0.0, -1.0];
        let c = centroid(y.view(), 6);
        for k in 0..3 {
            assert!(
                (y[15 + k] - c[k] - target[k]).abs() < 0.05,
                "vertex not on its site"
            );
        }
    }
}

#[cfg(test)]
mod shell_tests {
    use super::*;
    use ndarray::Array1;

    /// Points of the centred icosahedron, the centre and its twelve vertices.
    const ICOSAHEDRON_POINTS: usize = 13;
    /// Points of the one-sided surface over the icosahedron.
    const SURFACE_POINTS: usize = 10;
    /// Golden-ratio conjugate, so successive surface points spread evenly in azimuth.
    const GOLDEN_FRACTION: f64 = 0.618_034;
    /// Smallest radius of a surface point, outside the icosahedron's first shell.
    const SURFACE_RADIUS: f64 = 1.9;
    /// Radius added per step of the surface's [`SURFACE_RADII`] radii.
    const SURFACE_RADIUS_STEP: f64 = 0.05;
    /// Distinct radii the surface points cycle through.
    const SURFACE_RADII: usize = 3;
    /// Lowest height fraction of a surface point, which keeps the surface on one side.
    const SURFACE_Z_MIN: f64 = 0.35;
    /// Spread of the surface's height fractions above [`SURFACE_Z_MIN`].
    const SURFACE_Z_SPAN: f64 = 0.6;
    /// Stride, coprime to [`SURFACE_POINTS`], that deals the heights out of order.
    const SURFACE_Z_STRIDE: usize = 7;
    /// Multiplier of the fixed jitter pattern over the coordinate index.
    const JITTER_STRIDE: usize = 37;
    /// Offset of the fixed jitter pattern.
    const JITTER_OFFSET: usize = 11;
    /// Period of the fixed jitter pattern, whose half maps it onto `[-1, 1]`.
    const JITTER_PERIOD: usize = 17;
    /// Jitter amplitude that moves the core off its exact orbit without hiding its group.
    const JITTER: f64 = 0.003;
    /// Angle in radians by which the second two-fold misses perpendicular.
    const TILT: f64 = 0.03;
    /// Most elements the exact closure may generate, reached because it never repeats.
    const EXACT_CAP: usize = 48;
    /// Element comparison of [`generate_group`], right for exact operations only.
    const EXACT_CLOSURE: f64 = 1e-6;
    /// Error added to a vertex direction before the five-fold is built from it.
    const DIRECTION_ERROR: [f64; 2] = [0.04, 0.03];
    /// Largest element difference between the refined and the exact operation.
    const REFINE_TOLERANCE: f64 = 1e-9;
    /// Largest distance of the found centre from the icosahedron's centre.
    const CENTRE_ERROR: f64 = 0.01;
    /// Smallest offset of the whole structure's centroid from the icosahedron's centre.
    const CENTROID_OFFSET: f64 = 0.3;
    /// Merge radius of the shell completion.
    const MERGE_RADIUS: f64 = 0.3;
    /// Minimum separation of the shell completion.
    const MIN_SEPARATION: f64 = 0.5;
    /// Largest fraction of its radial spread the completed core may keep.
    const SPREAD_RATIO: f64 = 0.1;

    fn icosahedron() -> Vec<[f64; 3]> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let s = 1.0 / (1.0 + phi * phi).sqrt();
        let mut out = Vec::new();
        for a in [-1.0, 1.0] {
            for b in [-1.0, 1.0] {
                out.push([0.0, a * s, b * phi * s]);
                out.push([a * s, b * phi * s, 0.0]);
                out.push([b * phi * s, 0.0, a * s]);
            }
        }
        out
    }

    /// A centred icosahedron of radius one, each coordinate moved by up to
    /// `amp`, then a disordered surface on one side only, so the centroid
    /// of the whole is well away from the core's centre.
    fn centred_icosahedron_under_a_one_sided_surface(amp: f64) -> (Array1<f64>, usize) {
        let mut points = vec![[0.0; 3]];
        points.extend(icosahedron());
        for i in 0..SURFACE_POINTS {
            let a = 2.0 * std::f64::consts::PI * (i as f64) * GOLDEN_FRACTION;
            let r = SURFACE_RADIUS + SURFACE_RADIUS_STEP * ((i % SURFACE_RADII) as f64);
            let z = SURFACE_Z_MIN
                + SURFACE_Z_SPAN * ((i * SURFACE_Z_STRIDE % SURFACE_POINTS) as f64)
                    / SURFACE_POINTS as f64;
            let rho = (1.0 - z * z).sqrt();
            points.push([r * rho * a.cos(), r * rho * a.sin(), r * z]);
        }
        let mut x = Array1::from(points.iter().flatten().copied().collect::<Vec<f64>>());
        for (i, v) in x.iter_mut().enumerate().take(3 * ICOSAHEDRON_POINTS) {
            *v += amp
                * (((i * JITTER_STRIDE + JITTER_OFFSET) % JITTER_PERIOD) as f64
                    / (JITTER_PERIOD / 2) as f64
                    - 1.0);
        }
        (x, points.len())
    }

    fn spread(x: ArrayView1<f64>) -> f64 {
        let c = point(x, 0);
        let r: Vec<f64> = (1..ICOSAHEDRON_POINTS)
            .map(|a| distance(point(x, a), c))
            .collect();
        r.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - r.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Two two-fold axes a few degrees from perpendicular are close to a
    /// dihedral group of order ten with the five-fold, and nowhere near any
    /// exact group: compared to `1e-6` their products never repeat, so an
    /// exact closure runs to its cap, while a tolerant one stops at ten.
    #[test]
    fn a_tolerant_closure_stops_at_the_group_the_operations_approximate() {
        let tilt = TILT;
        let near = [
            Candidate {
                axis: [0.0, 0.0, 1.0],
                order: 5,
                improper: false,
                deviation: 0.0,
            },
            Candidate {
                axis: [tilt.cos(), 0.0, tilt.sin()],
                order: 2,
                improper: false,
                deviation: 0.0,
            },
        ];
        assert_eq!(generate_group(&near, EXACT_CAP).len(), EXACT_CAP);
        let gens: Vec<Rot> = near.iter().map(operation).collect();
        assert_eq!(close_group(&gens, 0.27, 120).len(), 10);
        assert_eq!(close_group(&gens, EXACT_CLOSURE, 120).len(), 120);
    }

    /// Refinement replaces an operation by the orthogonal map that carries
    /// the points onto the partners it matched, so a five-fold axis read a
    /// few degrees off comes back as the exact five-fold of the points.
    #[test]
    fn refinement_recovers_the_exact_operation_of_the_points() {
        let mut core = vec![[0.0; 3]];
        core.extend(icosahedron());
        let v = core[1];
        let exact = rot_matrix(v, 2.0 * std::f64::consts::PI / 5.0);
        let off = normalise([v[0] + DIRECTION_ERROR[0], v[1] - DIRECTION_ERROR[1], v[2]]).unwrap();
        let rough = rot_matrix(off, 2.0 * std::f64::consts::PI / 5.0);
        let refined = refine_operation(&rough, &core);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (refined[i][j] - exact[i][j]).abs() < REFINE_TOLERANCE,
                    "element ({i}, {j}) is {} against {}",
                    refined[i][j],
                    exact[i][j]
                );
            }
        }
    }

    /// The case the shell core exists for: a surface heavy on one side
    /// moves the centroid and fills the innermost fraction with surface
    /// points, while the weighted centre stays on the core and the first
    /// shell that carries an operation is the icosahedron itself.
    #[test]
    fn the_shell_core_reads_the_core_group_through_a_one_sided_surface() {
        let (x, n) = centred_icosahedron_under_a_one_sided_surface(JITTER);
        let core = shell_core_group(x.view(), n, &ShellCore::scaled(1.0)).expect("a core group");
        assert_eq!(core.core.len(), ICOSAHEDRON_POINTS);
        assert!(
            core.core.iter().all(|&a| a < ICOSAHEDRON_POINTS),
            "a surface point in the core"
        );
        assert_eq!(core.group.len(), 60, "the icosahedral rotations");
        assert!(distance(core.centre, point(x.view(), 0)) < CENTRE_ERROR);
        assert!(distance(centroid(x.view(), n), point(x.view(), 0)) > CENTROID_OFFSET);
    }

    /// Completion about the core's centre puts the jittered core back on
    /// one radius and leaves the count of points as it was.
    #[test]
    fn completion_on_the_shell_core_closes_the_core_onto_its_orbit() {
        let (x, n) = centred_icosahedron_under_a_one_sided_surface(JITTER);
        let y = orbit_complete_shells(
            x.view(),
            n,
            &ShellCore::scaled(1.0),
            MERGE_RADIUS,
            MIN_SEPARATION,
        )
        .expect("the jittered core moves");
        assert_eq!(y.len(), x.len());
        let (before, after) = (spread(x.view()), spread(y.view()));
        assert!(
            after < SPREAD_RATIO * before,
            "radial spread {before} became {after}"
        );
    }

    /// Shells are whole: a first shell with more points than the core may
    /// have is not cut down to size, and there is no shell group.
    #[test]
    fn a_first_shell_past_the_largest_core_gives_no_group() {
        let (x, n) = centred_icosahedron_under_a_one_sided_surface(0.0);
        let spec = ShellCore {
            max_points: ICOSAHEDRON_POINTS - 1,
            ..ShellCore::scaled(1.0)
        };
        assert!(shell_core_group(x.view(), n, &spec).is_none());
        let spec = ShellCore {
            max_points: ICOSAHEDRON_POINTS,
            ..spec
        };
        assert!(shell_core_group(x.view(), n, &spec).is_some());
    }

    /// Residual and the exchange come from one nearest image. The split
    /// scan measures the residual, then walks the cloud again.
    #[test]
    fn one_pass_residual_matches_the_image_that_supplies_the_exchange() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut unit = || {
            state = state
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(0x6a09_e667_f3bc_c909);
            (state >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
        };
        let mut mismatches = 0u32;
        let mut example = String::new();
        for n in [2usize, 3, 5, 7, 9, 17] {
            let mut coords = Vec::with_capacity(3 * n);
            for _ in 0..3 * n {
                coords.push(unit() * 2.0);
            }
            if n >= 4 {
                coords[3] = coords[0];
                coords[4] = coords[1];
                coords[5] = coords[2];
            }
            let x = Array1::from(coords);
            let cloud = Cloud::from_view(x.view(), n);
            for order in 2..=n.min(8) {
                for cap in [0.05_f64, 0.2, 1.0, f64::INFINITY] {
                    let axis = normalise([unit(), unit(), unit()]).unwrap_or([0.0, 0.0, 1.0]);
                    let got = cloud.symmetry_deviation(axis, order, cap);
                    let scalar = scalar_exchange(&cloud, axis, order, cap);
                    if !same_option_bits(got, scalar) {
                        mismatches += 1;
                        if example.is_empty() {
                            example = format!(
                                "n={n} order={order} cap={cap} got={got:?} scalar={scalar:?}"
                            );
                        }
                    }
                    let split = split_exchange(&cloud, axis, order, cap);
                    if !same_option_bits(got, split) {
                        mismatches += 1;
                        if example.is_empty() {
                            example = format!(
                                "split n={n} order={order} cap={cap} got={got:?} split={split:?}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(mismatches, 0, "{example}");
    }

    fn scalar_exchange(cloud: &Cloud, axis: [f64; 3], order: usize, cap: f64) -> Option<f64> {
        let n = cloud.xs.len();
        if n == 0 || order < 2 {
            return None;
        }
        let turn = OrderTurn::new(order)?;
        if cloud.identity_stays_under(axis, &turn, cap) && !cloud.any_atom_can_leave(axis, &turn) {
            return None;
        }
        let angle = 2.0 * std::f64::consts::PI / order as f64;
        let sine = libm::sin(std::f64::consts::PI / order as f64);
        let scale = 4.0 * sine * sine;
        let gaps = cloud.gaps();
        let mut total = 0.0;
        let mut moved = false;
        let limit = cap * cap * n as f64;
        let stop = cap.is_finite();
        for index in 0..n {
            let gap = gaps[index];
            let chord_sq = scale * cloud.perp_sq_at(index, axis);
            let axial = !gap.is_finite() || gap <= 0.0 || chord_sq < gap * 0.25;
            let (partner, distance) =
                cloud.nearest_index_scalar(rotate(cloud.point(index), axis, angle));
            total += distance;
            if !axial && partner == index {
                return None;
            }
            if partner != index {
                moved = true;
            }
            if stop && total > limit {
                let partial = (total / n as f64).sqrt();
                if partial >= cap * (1.0 + 1.0e-6) {
                    return None;
                }
            }
        }
        moved.then_some((total / n as f64).sqrt())
    }

    /// Distance-only residual, then a second walk for the exchange.
    fn split_exchange(cloud: &Cloud, axis: [f64; 3], order: usize, cap: f64) -> Option<f64> {
        if order < 2 || cloud.xs.is_empty() {
            return None;
        }
        let turn = OrderTurn::new(order)?;
        if cloud.identity_stays_under(axis, &turn, cap) && !cloud.any_atom_can_leave(axis, &turn) {
            return None;
        }
        let deviation = cloud.rotation_deviation(axis, order, cap)?;
        let angle = 2.0 * std::f64::consts::PI / order as f64;
        let sine = libm::sin(std::f64::consts::PI / order as f64);
        let scale = 4.0 * sine * sine;
        let gaps = cloud.gaps();
        let mut moved = false;
        for index in 0..cloud.xs.len() {
            let gap = gaps[index];
            let chord_sq = scale * cloud.perp_sq_at(index, axis);
            let axial = !gap.is_finite() || gap <= 0.0 || chord_sq < gap * 0.25;
            let (partner, _) = cloud.nearest_index_scalar(rotate(cloud.point(index), axis, angle));
            if !axial && partner == index {
                return None;
            }
            if partner != index {
                moved = true;
            }
        }
        moved.then_some(deviation)
    }

    #[test]
    fn wide_nearest_image_matches_the_scalar_scan() {
        let mut state = 0x1234_5678_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u32 << 31) as f64
        };
        for n in [1usize, 3, 7, 8, 9, 17, 33, 55] {
            let mut coords = Vec::with_capacity(3 * n);
            for _ in 0..3 * n {
                coords.push(next() * 4.0 - 2.0);
            }
            if n >= 4 {
                coords[3] = coords[0];
                coords[4] = coords[1];
                coords[5] = coords[2];
            }
            let x = Array1::from(coords);
            let cloud = Cloud::from_view(x.view(), n);
            for _ in 0..24 {
                let image = [next() * 6.0 - 3.0, next() * 6.0 - 3.0, next() * 6.0 - 3.0];
                let scalar = cloud.nearest_index_scalar(image);
                let chosen = cloud.nearest_index(image);
                assert_eq!(chosen.0, scalar.0, "n={n}");
                assert_eq!(chosen.1.to_bits(), scalar.1.to_bits(), "n={n}");
                let wide = cloud.nearest_squared_wide(image);
                assert_eq!(wide.to_bits(), scalar.1.to_bits(), "n={n}");
            }
        }
    }

    #[test]
    fn an_image_inside_half_the_gap_is_the_source_atom() {
        let mut state = 0x9e37_79b9_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u32 << 31) as f64
        };
        for n in [4usize, 12, 33, 55] {
            let mut coords = Vec::with_capacity(3 * n);
            for _ in 0..3 * n {
                coords.push(next() * 3.0 - 1.5);
            }
            let x = Array1::from(coords);
            let cloud = Cloud::from_view(x.view(), n);
            let gaps = cloud.gaps().clone();
            let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
            for axis in axes {
                let axis = normalise(axis).expect("axis");
                for order in [2usize, 3, 4, 6, 12] {
                    let angle = 2.0 * std::f64::consts::PI / order as f64;
                    for index in 0..n {
                        let here = cloud.point(index);
                        let image = rotate(here, axis, angle);
                        let Some(self_d2) = image_on_self(here, image, gaps[index]) else {
                            continue;
                        };
                        let (partner, distance) = cloud.nearest_index_scalar(image);
                        assert_eq!(partner, index, "n={n} order={order}");
                        assert_eq!(distance.to_bits(), self_d2.to_bits());
                    }
                }
            }
        }
        let x = Array1::from(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let cloud = Cloud::from_view(x.view(), 2);
        let here = cloud.point(0);
        let image = [here[0] + 0.999, here[1], here[2]];
        let self_d2 = image_on_self(here, image, 4.0).expect("inside half the gap");
        let (partner, distance) = cloud.nearest_index_scalar(image);
        assert_eq!(partner, 0);
        assert_eq!(distance.to_bits(), self_d2.to_bits());
        let outside = [here[0] + 1.001, here[1], here[2]];
        assert!(image_on_self(here, outside, 4.0).is_none());
    }

    /// A turn the distances already reject never builds the neighbour list.
    #[test]
    fn a_miss_leaves_the_neighbour_list_unbuilt() {
        let mut state = 0x1234_5678_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u32 << 31) as f64
        };
        for n in [8usize, 17, 33] {
            let mut coords = Vec::with_capacity(3 * n);
            for _ in 0..3 * n {
                coords.push(next() * 4.0 - 2.0);
            }
            let x = Array1::from(coords);
            let cloud = Cloud::from_view(x.view(), n);
            let axis = normalise([0.2, -0.4, 0.9]).expect("axis");
            let got = cloud.symmetry_deviation(axis, 2, 0.05);
            assert!(got.is_none(), "n={n} {got:?}");
            assert!(cloud.gaps.borrow().is_none(), "n={n}");
        }
    }

    /// A sum strictly above the squared gate is the square-root comparison.
    #[test]
    fn a_sum_strictly_above_the_squared_gate_matches_the_square_root() {
        let caps = [1.0e-6, 1.0e-4, 1.0e-2, 0.05, 0.1, 0.2, 1.0, 10.0, 1.0e3];
        for n in 1..=128 {
            for cap in caps {
                let gate = cap * (1.0 + 1.0e-6);
                let edge = gate * gate * (n as f64);
                if !edge.is_finite() || edge <= 0.0 {
                    continue;
                }
                let above = edge.next_up();
                let root = (above / n as f64).sqrt();
                assert!(root >= gate, "n={n} cap={cap} root={root} gate={gate}");
                for scale in [0.5, 1.0, 2.0, 16.0] {
                    let total = edge * scale;
                    let original =
                        total > cap * cap * (n as f64) && (total / n as f64).sqrt() >= gate;
                    assert_eq!(
                        sum_past_cap(total, n, cap),
                        original,
                        "n={n} cap={cap} scale={scale}"
                    );
                }
            }
        }
    }

    /// `detect`, `detect_all` and `detect_orders` keep the base split's bits.
    ///
    /// The reference is the distance residual followed by the exchange walk,
    /// and the scalar mirror for a plane. Accepted deviations match by
    /// `to_bits`, including a ring that is symmetric, duplicated sites, and
    /// the random clouds the microbench times.
    #[test]
    fn detection_matches_the_base_split_on_the_corpus() {
        let mut clouds: Vec<(String, Array1<f64>, usize)> = Vec::new();
        for (name, n, seed) in [
            ("random33", 33usize, 1u64),
            ("random55", 55, 2),
            ("random38", 38, 7),
        ] {
            clouds.push((name.to_string(), bench_cloud(n, seed), n));
        }
        clouds.push(("ring12".to_string(), bench_ring(12, 2.2), 12));
        let (lattice, lattice_n) = bench_lattice(2);
        clouds.push(("lattice".to_string(), lattice, lattice_n));
        let mut duplicated = bench_cloud(6, 3).to_vec();
        duplicated.extend_from_within(..);
        clouds.push(("duplicated".to_string(), Array1::from(duplicated), 12));

        let orders: Vec<usize> = (2..=12).collect();
        let high: Vec<usize> = (2..=38).collect();
        let tolerance = 0.1;
        for (name, x, n) in &clouds {
            let orders = if name == "random38" { &high } else { &orders };
            assert_detection_bits(name, x.view(), *n, orders, tolerance);
        }
    }

    fn same_option_bits(left: Option<f64>, right: Option<f64>) -> bool {
        match (left, right) {
            (None, None) => true,
            (Some(left), Some(right)) => left.to_bits() == right.to_bits(),
            _ => false,
        }
    }

    fn assert_candidate_bits(label: &str, got: &Candidate, expect: &Candidate) {
        assert_eq!(got.order, expect.order, "{label} order");
        assert_eq!(got.improper, expect.improper, "{label} improper");
        for axis in 0..3 {
            assert_eq!(
                got.axis[axis].to_bits(),
                expect.axis[axis].to_bits(),
                "{label} axis {axis}"
            );
        }
        assert_eq!(
            got.deviation.to_bits(),
            expect.deviation.to_bits(),
            "{label} deviation"
        );
    }

    fn assert_detection_bits(
        name: &str,
        x: ArrayView1<f64>,
        n: usize,
        orders: &[usize],
        tolerance: f64,
    ) {
        let cloud = Cloud::from_view(x, n);
        let axes = candidate_axes(x, n);
        for &axis in &axes {
            for &order in orders {
                let got = cloud.symmetry_deviation(axis, order, tolerance);
                let split = split_exchange(&cloud, axis, order, tolerance);
                assert!(
                    same_option_bits(got, split),
                    "{name} order={order} got={got:?} split={split:?}"
                );
            }
            let plane = cloud.plane_deviation(axis, tolerance);
            let scalar = scalar_plane(&cloud, axis, tolerance);
            assert!(
                same_option_bits(plane, scalar),
                "{name} plane got={plane:?} scalar={scalar:?}"
            );
        }

        let expect_best = base_detect(x, n, orders, tolerance);
        match (detect(x, n, orders, tolerance), expect_best) {
            (None, None) => {}
            (Some(got), Some(expect)) => {
                assert_candidate_bits(&format!("{name} detect"), &got, &expect)
            }
            (got, expect) => panic!("{name} detect got={got:?} expect={expect:?}"),
        }

        let got_orders = detect_orders(x, n, orders, tolerance);
        let expect_orders = base_detect_orders(x, n, orders, tolerance);
        assert_eq!(
            got_orders.len(),
            expect_orders.len(),
            "{name} detect_orders"
        );
        for (index, (got, expect)) in got_orders.iter().zip(&expect_orders).enumerate() {
            assert_candidate_bits(&format!("{name} detect_orders {index}"), got, expect);
        }

        let got_all = detect_all(x, n, orders, tolerance);
        let expect_all = base_detect_all(x, n, orders, tolerance);
        assert_eq!(got_all.len(), expect_all.len(), "{name} detect_all");
        for (index, (got, expect)) in got_all.iter().zip(&expect_all).enumerate() {
            assert_candidate_bits(&format!("{name} detect_all {index}"), got, expect);
        }
    }

    fn base_detect(
        x: ArrayView1<f64>,
        n: usize,
        orders: &[usize],
        tolerance: f64,
    ) -> Option<Candidate> {
        let cloud = Cloud::from_view(x, n);
        let mut best: Option<Candidate> = None;
        for axis in candidate_axes(x, n) {
            for &order in orders {
                let Some(deviation) = split_exchange(&cloud, axis, order, tolerance) else {
                    continue;
                };
                if deviation < tolerance
                    && best
                        .as_ref()
                        .map(|found| deviation < found.deviation)
                        .unwrap_or(true)
                {
                    best = Some(Candidate {
                        axis,
                        order,
                        improper: false,
                        deviation,
                    });
                }
            }
        }
        best
    }

    fn base_detect_orders(
        x: ArrayView1<f64>,
        n: usize,
        orders: &[usize],
        tolerance: f64,
    ) -> Vec<Candidate> {
        let cloud = Cloud::from_view(x, n);
        let axes = candidate_axes(x, n);
        let mut out = Vec::new();
        for &order in orders {
            let mut best: Option<Candidate> = None;
            for &axis in &axes {
                let Some(deviation) = split_exchange(&cloud, axis, order, tolerance) else {
                    continue;
                };
                if deviation < tolerance
                    && best
                        .as_ref()
                        .map(|found| deviation < found.deviation)
                        .unwrap_or(true)
                {
                    best = Some(Candidate {
                        axis,
                        order,
                        improper: false,
                        deviation,
                    });
                }
            }
            if let Some(candidate) = best {
                out.push(candidate);
            }
        }
        out
    }

    fn base_detect_all(
        x: ArrayView1<f64>,
        n: usize,
        orders: &[usize],
        tolerance: f64,
    ) -> Vec<Candidate> {
        let cloud = Cloud::from_view(x, n);
        let mut out = Vec::new();
        for axis in candidate_axes(x, n) {
            for &order in orders {
                let Some(deviation) = split_exchange(&cloud, axis, order, tolerance) else {
                    continue;
                };
                if deviation < tolerance {
                    out.push(Candidate {
                        axis,
                        order,
                        improper: false,
                        deviation,
                    });
                }
            }
            let Some(deviation) = scalar_plane(&cloud, axis, tolerance) else {
                continue;
            };
            if deviation < tolerance {
                out.push(Candidate {
                    axis,
                    order: 1,
                    improper: true,
                    deviation,
                });
            }
        }
        out.sort_by(|left, right| {
            left.deviation
                .partial_cmp(&right.deviation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    fn scalar_plane(cloud: &Cloud, axis: [f64; 3], cap: f64) -> Option<f64> {
        let n = cloud.xs.len();
        if n == 0 {
            return None;
        }
        let mirror = mirror_matrix(axis);
        let mut total = 0.0;
        let limit = cap * cap * n as f64;
        for index in 0..n {
            let point = cloud.point(index);
            let image = [
                mirror[0][0] * point[0] + mirror[0][1] * point[1] + mirror[0][2] * point[2],
                mirror[1][0] * point[0] + mirror[1][1] * point[1] + mirror[1][2] * point[2],
                mirror[2][0] * point[0] + mirror[2][1] * point[1] + mirror[2][2] * point[2],
            ];
            total += cloud.nearest_squared_scalar(image);
            if cap.is_finite() && total > limit {
                let partial = (total / n as f64).sqrt();
                if partial >= cap * (1.0 + 1.0e-6) {
                    return None;
                }
            }
        }
        Some((total / n as f64).sqrt())
    }

    fn bench_cloud(n: usize, seed: u64) -> Array1<f64> {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
        };
        let radius = 1.15 * (n as f64).cbrt();
        Array1::from(
            (0..3 * n)
                .map(|_| 2.0 * radius * next())
                .collect::<Vec<_>>(),
        )
    }

    fn bench_ring(n: usize, radius: f64) -> Array1<f64> {
        let mut coords = vec![0.0; 3 * n];
        for atom in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (atom as f64) / (n as f64);
            coords[3 * atom] = radius * angle.cos();
            coords[3 * atom + 1] = radius * angle.sin();
        }
        Array1::from(coords)
    }

    fn bench_lattice(max_r2: i32) -> (Array1<f64>, usize) {
        let mut coords = Vec::new();
        for i in -2..=2 {
            for j in -2..=2 {
                for k in -2..=2 {
                    let r2 = i * i + j * j + k * k;
                    if r2 > 0 && r2 <= max_r2 {
                        coords.push(i as f64);
                        coords.push(j as f64);
                        coords.push(k as f64);
                    }
                }
            }
        }
        let n = coords.len() / 3;
        (Array1::from(coords), n)
    }
}
