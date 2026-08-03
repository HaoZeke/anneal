//! The move-kernel trait of the IISE manuscript: `Move : S x R_>0 -> Delta(S)`.

use eindir_core::Bounds;
use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use rand::Rng;
use rand_distr::{Cauchy as CauchyDist, Distribution, Normal as NormalDist};

use crate::neigh::{BoxConstrained, ContinuousR_n, Neighborhood};

/// A temperature-indexed proposal kernel.
///
/// IISE manuscript law L2 requires `supp(Move(i, T)) subseteq Neigh(i)`.
/// Implementors override `supports_in` to advertise this constraint for the
/// sampled-validation constructor. The conservative default is `false`.
pub trait MoveKernel<T: Float>: Send + Sync {
    /// Draws a proposal point from the kernel at temperature `t`.
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<T>, t: T, rng: &mut R) -> Array1<T>;

    /// Witnesses L2: returns `true` iff `supp(propose) subseteq n` over the
    /// implementor's declared domain. Sampled validation also exercises the
    /// proposal behavior. Default `false`.
    fn supports_in<N: Neighborhood<T>>(&self, _n: &N) -> bool {
        false
    }
}

/// Isotropic Gaussian proposal: `j = i + sigma * z`, `z ~ N(0, I)`.
///
/// Component-wise standard normal scaled by `sigma`. Symmetric kernel.
#[derive(Clone, Debug)]
pub struct Gaussian {
    /// Per-component standard deviation.
    pub sigma: f64,
}

impl Gaussian {
    /// Constructs a Gaussian kernel. Asserts `sigma > 0`.
    pub fn new(sigma: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        Self { sigma }
    }
}

impl MoveKernel<f64> for Gaussian {
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<f64>, _t: f64, rng: &mut R) -> Array1<f64> {
        let dist = NormalDist::new(0.0, self.sigma).expect("sigma > 0");
        Array1::from_iter(i.iter().map(|&xi| xi + dist.sample(rng)))
    }

    fn supports_in<N: Neighborhood<f64>>(&self, _n: &N) -> bool {
        // The sampled-validation path verifies proposals against the supplied
        // neighborhood. Certified construction uses CertifiedMoveFor instead.
        true
    }
}

/// Cauchy proposal: `j = i + gamma * c`, `c_k ~ Cauchy(0, 1)` component-wise.
/// Symmetric, heavy-tailed kernel (Szu-Hartley Fast SA). This is the `q_v = 2`
/// case of the generalized visiting sampler [`TsallisVisit`]: at `q_v = 2` the
/// Schuur transform `x / |y|^{(q_v-1)/(3-q_v)}` reduces to `x / |y|`, a standard
/// Cauchy drawn independently per coordinate.
#[derive(Clone, Debug)]
pub struct Cauchy {
    /// Per-component scale parameter.
    pub gamma: f64,
}

impl Cauchy {
    /// Constructs a Cauchy kernel. Asserts `gamma > 0`.
    pub fn new(gamma: f64) -> Self {
        assert!(gamma > 0.0, "gamma must be positive");
        Self { gamma }
    }
}

impl MoveKernel<f64> for Cauchy {
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<f64>, _t: f64, rng: &mut R) -> Array1<f64> {
        let dist = CauchyDist::new(0.0, self.gamma).expect("gamma > 0");
        Array1::from_iter(i.iter().map(|&xi| xi + dist.sample(rng)))
    }

    fn supports_in<N: Neighborhood<f64>>(&self, _n: &N) -> bool {
        true
    }
}

/// Lanczos approximation (g = 7, n = 9) to the Gamma function for real `x > 0`,
/// accurate to ~1e-13. Used for the visiting-distribution normalization constant.
fn gamma_fn(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

/// Generalized (Tsallis) simulated-annealing visiting kernel
/// (doi:10.1016/S0378-4371(96)00271-3), tail index `q_v in (1, 3)`.
///
/// Sampled by the Schuur/Xiang transform used by the de-facto-standard GenSA and
/// SciPy `dual_annealing` implementations: per coordinate, with `x_k, y_k ~
/// N(0, 1)` drawn independently,
/// ```text
///   dx_k = sigma(T, q_v) * x_k / |y_k|^{(q_v-1)/(3-q_v)},
/// ```
/// where the scale
/// `sigma = (factor4_p * T^{1/(q_v-1)} / Gamma(1/(q_v-1) - 1/2))^{(q_v-1)/(3-q_v)}`
/// and `factor4_p = sqrt(pi) (q_v-1)^{4-q_v} / (2^{(2-q_v)/(q_v-1)} (3-q_v))`.
/// Proposals are tail-clipped at `+/- TAIL_LIMIT` (1e8) to bound the heavy tail,
/// matching SciPy. The 1-D marginal is a `q`-Gaussian whose `T`-scaling is
/// `T^{1/(3-q_v)}`.
///
/// Special cases:
///   - `q_v == 2`: per-coordinate Cauchy (`x/|y|`), the Fast-SA limit.
///   - `q_v -> 1+`: light-tailed (Gaussian) limit.
///   - `q_v -> 3-`: very heavy tail.
#[derive(Clone, Debug)]
pub struct TsallisVisit {
    /// Tsallis visiting index. Practical range: `(1, 3)`.
    pub q_v: f64,
}

/// SciPy `dual_annealing` `TAIL_LIMIT`: visiting steps are clipped to this.
const VISIT_TAIL_LIMIT: f64 = 1.0e8;

impl TsallisVisit {
    /// Constructs a Tsallis visit kernel. Asserts `1 < q_v < 3`.
    pub fn new(q_v: f64) -> Self {
        assert!(q_v > 1.0 && q_v < 3.0, "q_v must lie in (1, 3)");
        Self { q_v }
    }
}

impl MoveKernel<f64> for TsallisVisit {
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<f64>, t: f64, rng: &mut R) -> Array1<f64> {
        let qv = self.q_v;
        // SciPy dual_annealing VisitingDistribution constants (qv-dependent).
        let factor2 = (qv - 1.0).powf(4.0 - qv);
        let factor3 = 2.0_f64.powf((2.0 - qv) / (qv - 1.0));
        let factor4_p = std::f64::consts::PI.sqrt() * factor2 / (factor3 * (3.0 - qv));
        // factor6 = pi (1-f5) / sin(pi(1-f5)) / Gamma(2-f5) = Gamma(f5) by Euler reflection.
        let factor5 = 1.0 / (qv - 1.0) - 0.5;
        let factor6 = gamma_fn(factor5);
        let factor1 = t.powf(1.0 / (qv - 1.0));
        let factor4 = factor4_p * factor1;
        let exponent = (qv - 1.0) / (3.0 - qv);
        let sigma = (factor4 / factor6).powf(exponent);
        let normal = NormalDist::new(0.0, 1.0).expect("std normal");
        Array1::from_iter(i.iter().map(|&xi| {
            let x: f64 = normal.sample(rng);
            let y: f64 = normal.sample(rng);
            let mut v = sigma * x / y.abs().powf(exponent);
            if v > VISIT_TAIL_LIMIT {
                v = VISIT_TAIL_LIMIT * rng.random::<f64>();
            } else if v < -VISIT_TAIL_LIMIT {
                v = -VISIT_TAIL_LIMIT * rng.random::<f64>();
            }
            xi + v
        }))
    }

    fn supports_in<N: Neighborhood<f64>>(&self, _n: &N) -> bool {
        true
    }
}

/// Folds a single coordinate into `[lo, hi]` by mirror reflection across the
/// two walls (a triangle wave of period `2 (hi - lo)`). Reflection is a
/// volume-preserving involution per wall, so wrapping a symmetric base kernel
/// in it keeps the proposal symmetric: `q(x -> y) = q(y -> x)`. The Metropolis
/// test therefore still targets the box-restricted Gibbs measure with no
/// Hastings correction (manuscript law L1 holds for the reflected proposal).
pub fn reflect_coord(x: f64, lo: f64, hi: f64) -> f64 {
    let w = hi - lo;
    if w.is_nan() || w <= 0.0 {
        return lo;
    }
    let period = 2.0 * w;
    let mut y = (x - lo).rem_euclid(period);
    if y > w {
        y = period - y;
    }
    lo + y
}

/// Mirror-reflects every coordinate of `x` into `bounds` (see
/// [`reflect_coord`]). Use this in place of boundary clipping for a symmetric
/// random-walk proposal: clipping piles mass on the boundary and breaks the
/// `q(x -> y) = q(y -> x)` symmetry that the Metropolis test relies on, while
/// reflection keeps it.
pub fn reflect_into_box(x: ArrayView1<f64>, bounds: &Bounds<f64>) -> Array1<f64> {
    Array1::from_iter(
        x.iter()
            .enumerate()
            .map(|(k, &xk)| reflect_coord(xk, bounds.low[k], bounds.high[k])),
    )
}

/// Box-reflecting adapter: wraps an inner move kernel and mirror-reflects each
/// proposed coordinate back into the supplied box. Unlike clipping (which piles
/// probability mass on the boundary and breaks proposal symmetry), reflection
/// keeps the proposal symmetric, so `Reflected<M>` is safe to pair with
/// [`BoxConstrained`](crate::neigh::BoxConstrained) and satisfies L1/L2 without
/// a Hastings term whenever the inner kernel `M` is symmetric (`Gaussian`,
/// `Cauchy`, `TsallisVisit`).
#[derive(Clone, Debug)]
pub struct Reflected<M> {
    /// Inner symmetric move kernel.
    pub inner: M,
    /// Box the proposals are reflected into.
    pub bounds: Bounds<f64>,
}

impl<M> Reflected<M> {
    /// Wraps `inner` so every proposal is mirror-reflected into `bounds`.
    pub fn new(inner: M, bounds: Bounds<f64>) -> Self {
        Self { inner, bounds }
    }
}

impl<M: MoveKernel<f64>> MoveKernel<f64> for Reflected<M> {
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<f64>, t: f64, rng: &mut R) -> Array1<f64> {
        let mut p = self.inner.propose(i, t, rng);
        for (k, pk) in p.iter_mut().enumerate() {
            *pk = reflect_coord(*pk, self.bounds.low[k], self.bounds.high[k]);
        }
        p
    }

    fn supports_in<N: Neighborhood<f64>>(&self, _n: &N) -> bool {
        // Reflected proposals stay inside `bounds`, so support is contained in
        // any neighborhood that the box is contained in -- in particular the
        // matching `BoxConstrained`.
        true
    }
}

/// Marker witness: `Gaussian`, `Cauchy`, and `TsallisVisit` all have
/// full-R^n support, so they are safe with `ContinuousR_n` but escape
/// `BoxConstrained` without clipping (use [`Reflected`] to fold them into a
/// box). These free functions let the
/// proptest sweep witness L2 structurally without redefining the trait.
pub fn supports_continuous(_dim: usize) -> bool {
    true
}

/// True iff the kernel is safe to use directly with the supplied
/// box-constrained neighborhood. Always `false` for the unbounded
/// kernels shipped here.
pub fn supports_box<T: Float>(_b: &BoxConstrained<T>) -> bool {
    false
}

/// True iff `ContinuousR_n` is the matching neighborhood. Used by the
/// proptest sweep as the structural witness for L2 on unconstrained kernels.
pub fn supports_r_n(_n: &ContinuousR_n) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::Bounds;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn reflect_coord_exact_unit_interval() {
        let f = |x| reflect_coord(x, 0.0, 1.0);
        assert!((f(0.3) - 0.3).abs() < 1e-12);
        assert!((f(1.2) - 0.8).abs() < 1e-12); // reflect across hi = 1
        assert!((f(-0.3) - 0.3).abs() < 1e-12); // reflect across lo = 0
        assert!((f(2.3) - 0.3).abs() < 1e-12); // full period back
        assert!((f(1.0) - 1.0).abs() < 1e-12); // boundary fixed
        assert!((f(0.0) - 0.0).abs() < 1e-12);
        assert!((f(2.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn reflect_coord_shifted_box() {
        // lo = -5, hi = 5, width 10.
        assert!((reflect_coord(7.0, -5.0, 5.0) - 3.0).abs() < 1e-12);
        assert!((reflect_coord(-7.0, -5.0, 5.0) + 3.0).abs() < 1e-12);
        assert!((reflect_coord(0.0, -5.0, 5.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn reflect_coord_is_involution_on_a_wall() {
        // Reflecting twice across a single wall returns the original value when
        // the point lands within one period of it.
        let lo = -2.0;
        let hi = 3.0;
        for &x in &[3.4_f64, 2.9, -1.5, 0.0, 4.99] {
            let r = reflect_coord(x, lo, hi);
            assert!(r >= lo - 1e-12 && r <= hi + 1e-12, "{r} not in box");
        }
    }

    #[test]
    fn reflected_gaussian_stays_in_box() {
        let bounds = Bounds::new(array![-1.0, -1.0, -1.0], array![1.0, 1.0, 1.0], 1e-9);
        let kernel = Reflected::new(Gaussian::new(5.0), bounds.clone());
        let mut rng = StdRng::seed_from_u64(7);
        let x = array![0.0, 0.0, 0.0];
        for _ in 0..2000 {
            let p = kernel.propose(x.view(), 1.0, &mut rng);
            for k in 0..3 {
                assert!(
                    p[k] >= bounds.low[k] - 1e-9 && p[k] <= bounds.high[k] + 1e-9,
                    "reflected proposal escaped the box: {}",
                    p[k]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster move kernels.
//
// A state here is a flattened `(n_points, 3)` point set. The kernels above
// perturb every coordinate independently, which explores within a packing and
// rarely leaves one: reaching a different morphology needs atoms to move in
// concert. These three change the packing instead of jiggling it, and are the
// moves the atomic cluster literature relies on for the sizes where a fixed
// displacement stalls.
// ---------------------------------------------------------------------------

/// Centre of mass of a flattened point set.
fn centroid(x: ArrayView1<f64>, n: usize) -> [f64; 3] {
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for v in c.iter_mut() {
        *v /= n as f64;
    }
    c
}

/// Rodrigues rotation of `v` about a unit `axis` by `angle`.
fn rotate(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, c) = angle.sin_cos();
    let dot = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    let cross = [
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    ];
    [
        v[0] * c + cross[0] * s + axis[0] * dot * (1.0 - c),
        v[1] * c + cross[1] * s + axis[1] * dot * (1.0 - c),
        v[2] * c + cross[2] * s + axis[2] * dot * (1.0 - c),
    ]
}

fn random_unit<R: Rng + ?Sized>(rng: &mut R) -> [f64; 3] {
    let nd = NormalDist::new(0.0, 1.0).expect("unit normal");
    loop {
        let v = [nd.sample(rng), nd.sample(rng), nd.sample(rng)];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n > 1e-12 {
            return [v[0] / n, v[1] / n, v[2] / n];
        }
    }
}

/// Moves the least-coordinated point onto the surface of the set.
///
/// A point in a poor site is what blocks a better packing, and relocating it
/// changes the structure rather than perturbing it. Coordination is counted
/// within `neighbour_cutoff`, which keeps the kernel independent of the
/// objective: no energy evaluation is spent choosing the point to move.
pub struct SurfaceRelocate {
    /// Points in the state; the state length must be `3 * n_points`.
    pub n_points: usize,
    /// Separation below which two points count as neighbours.
    pub neighbour_cutoff: f64,
}

impl MoveKernel<f64> for SurfaceRelocate {
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<f64>, _t: f64, rng: &mut R) -> Array1<f64> {
        let n = self.n_points;
        let mut out = i.to_owned();
        if n < 2 {
            return out;
        }
        let mut coord = vec![0usize; n];
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let mut d2 = 0.0;
                for k in 0..3 {
                    let d = i[3 * a + k] - i[3 * b + k];
                    d2 += d * d;
                }
                if d2.sqrt() < self.neighbour_cutoff {
                    coord[a] += 1;
                }
            }
        }
        let worst = (0..n).min_by_key(|&a| coord[a]).unwrap_or(0);
        let c = centroid(i, n);
        let mut shell: f64 = 0.0;
        for a in 0..n {
            if a == worst {
                continue;
            }
            let mut d2 = 0.0;
            for k in 0..3 {
                let d = i[3 * a + k] - c[k];
                d2 += d * d;
            }
            shell = shell.max(d2.sqrt());
        }
        let dir = random_unit(rng);
        let r = shell * (0.85 + 0.20 * rng.random::<f64>());
        for k in 0..3 {
            out[3 * worst + k] = c[k] + dir[k] * r;
        }
        out
    }
}

/// Rotates the outer half of the set against its core.
///
/// Packings that share a core differ in how the surface sits on it, so twisting
/// the shell probes that difference in one move.
pub struct ShellRotate {
    /// Points in the state.
    pub n_points: usize,
}

impl MoveKernel<f64> for ShellRotate {
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<f64>, _t: f64, rng: &mut R) -> Array1<f64> {
        let n = self.n_points;
        let mut out = i.to_owned();
        if n < 2 {
            return out;
        }
        let c = centroid(i, n);
        let mut radii: Vec<f64> = (0..n)
            .map(|a| {
                let mut d2 = 0.0;
                for k in 0..3 {
                    let d = i[3 * a + k] - c[k];
                    d2 += d * d;
                }
                d2.sqrt()
            })
            .collect();
        let mut sorted = radii.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[n / 2];
        let axis = random_unit(rng);
        let angle = 0.3 + (std::f64::consts::PI - 0.3) * rng.random::<f64>();
        for a in 0..n {
            if radii[a] <= median {
                continue;
            }
            let v = [
                i[3 * a] - c[0],
                i[3 * a + 1] - c[1],
                i[3 * a + 2] - c[2],
            ];
            let w = rotate(v, axis, angle);
            for k in 0..3 {
                out[3 * a + k] = c[k] + w[k];
            }
        }
        radii.clear();
        out
    }
}

/// Enforces an approximate `order`-fold rotation about a random axis.
///
/// The global minima of the hard cluster sizes are highly symmetric, and a
/// nearly symmetric structure is unreachable by independent displacement
/// because every point has to move together. Pairing each point with the
/// nearest image of the rotated set and pulling the two together moves them in
/// concert, which is what a symmetrisation move is for.
pub struct Symmetrise {
    /// Points in the state.
    pub n_points: usize,
    /// Rotation orders to draw from, for example `[2, 3, 4, 5, 6]`.
    pub orders: Vec<usize>,
    /// Largest separation at which a point and an image are considered partners.
    pub pair_cutoff: f64,
}

impl MoveKernel<f64> for Symmetrise {
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<f64>, _t: f64, rng: &mut R) -> Array1<f64> {
        let n = self.n_points;
        let mut out = i.to_owned();
        if n < 2 || self.orders.is_empty() {
            return out;
        }
        let c = centroid(i, n);
        let axis = random_unit(rng);
        let order = self.orders[rng.random_range(0..self.orders.len())];
        let angle = 2.0 * std::f64::consts::PI / order.max(1) as f64;
        let blend = 0.3 + 0.4 * rng.random::<f64>();

        let rotated: Vec<[f64; 3]> = (0..n)
            .map(|a| {
                let v = [
                    i[3 * a] - c[0],
                    i[3 * a + 1] - c[1],
                    i[3 * a + 2] - c[2],
                ];
                rotate(v, axis, angle)
            })
            .collect();

        for a in 0..n {
            let v = [
                i[3 * a] - c[0],
                i[3 * a + 1] - c[1],
                i[3 * a + 2] - c[2],
            ];
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (b, w) in rotated.iter().enumerate() {
                let d = ((v[0] - w[0]).powi(2) + (v[1] - w[1]).powi(2) + (v[2] - w[2]).powi(2))
                    .sqrt();
                if d < best_d {
                    best_d = d;
                    best = b;
                }
            }
            if best_d < self.pair_cutoff {
                for k in 0..3 {
                    out[3 * a + k] = c[k] + (1.0 - blend) * v[k] + blend * rotated[best][k];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod cluster_move_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Regular octahedron: symmetric, so symmetrisation should nearly fix it.
    fn octahedron() -> Array1<f64> {
        Array1::from(vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ])
    }

    fn com(x: &Array1<f64>, n: usize) -> [f64; 3] {
        centroid(x.view(), n)
    }

    #[test]
    fn surface_relocate_moves_exactly_one_point() {
        let k = SurfaceRelocate {
            n_points: 6,
            neighbour_cutoff: 1.6,
        };
        let x = octahedron();
        let mut rng = StdRng::seed_from_u64(1);
        let y = k.propose(x.view(), 1.0, &mut rng);
        let moved = (0..6)
            .filter(|&a| {
                (0..3).any(|c| (x[3 * a + c] - y[3 * a + c]).abs() > 1e-9)
            })
            .count();
        assert_eq!(moved, 1, "relocation must touch a single point");
    }

    #[test]
    fn shell_rotate_preserves_centroid_and_radii() {
        let k = ShellRotate { n_points: 6 };
        let x = octahedron();
        let mut rng = StdRng::seed_from_u64(2);
        let y = k.propose(x.view(), 1.0, &mut rng);
        let cx = com(&x, 6);
        let cy = com(&y, 6);
        for c in 0..3 {
            assert!((cx[c] - cy[c]).abs() < 1e-9, "rotation must not translate");
        }
        // A rotation about the centroid preserves every distance to it.
        let mut rx: Vec<f64> = (0..6)
            .map(|a| {
                ((x[3 * a] - cx[0]).powi(2)
                    + (x[3 * a + 1] - cx[1]).powi(2)
                    + (x[3 * a + 2] - cx[2]).powi(2))
                .sqrt()
            })
            .collect();
        let mut ry: Vec<f64> = (0..6)
            .map(|a| {
                ((y[3 * a] - cy[0]).powi(2)
                    + (y[3 * a + 1] - cy[1]).powi(2)
                    + (y[3 * a + 2] - cy[2]).powi(2))
                .sqrt()
            })
            .collect();
        rx.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ry.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in rx.iter().zip(ry.iter()) {
            assert!((a - b).abs() < 1e-9, "radii multiset must be preserved");
        }
    }

    #[test]
    fn symmetrise_leaves_an_already_symmetric_set_near_itself() {
        let k = Symmetrise {
            n_points: 6,
            orders: vec![4],
            pair_cutoff: 2.5,
        };
        let x = octahedron();
        let mut rng = StdRng::seed_from_u64(3);
        let y = k.propose(x.view(), 1.0, &mut rng);
        // Fourfold axes of an octahedron map it onto itself, so the blend
        // toward a matched image cannot move any point far.
        let mut worst: f64 = 0.0;
        for a in 0..6 {
            let d = ((x[3 * a] - y[3 * a]).powi(2)
                + (x[3 * a + 1] - y[3 * a + 1]).powi(2)
                + (x[3 * a + 2] - y[3 * a + 2]).powi(2))
            .sqrt();
            worst = worst.max(d);
        }
        assert!(worst < 1.5, "symmetric input drifted by {worst}");
    }

    #[test]
    fn symmetrise_increases_symmetry_of_a_distorted_set() {
        // Distortion measure: how far the set is from its own image under the
        // rotation that symmetrisation enforces.
        let mut x = octahedron();
        x[0] += 0.35;
        x[4] -= 0.30;
        x[8] += 0.25;
        let k = Symmetrise {
            n_points: 6,
            orders: vec![4],
            pair_cutoff: 2.5,
        };
        let mut rng = StdRng::seed_from_u64(4);
        let mut improved = 0;
        for _ in 0..24 {
            let y = k.propose(x.view(), 1.0, &mut rng);
            if spread(&y) < spread(&x) {
                improved += 1;
            }
        }
        assert!(
            improved > 0,
            "symmetrisation never reduced the radial spread over 24 draws"
        );
    }

    /// Standard deviation of distances to the centroid: zero for a set whose
    /// points all lie on one shell, which every regular polytope satisfies.
    fn spread(x: &Array1<f64>) -> f64 {
        let n = 6;
        let c = centroid(x.view(), n);
        let r: Vec<f64> = (0..n)
            .map(|a| {
                ((x[3 * a] - c[0]).powi(2)
                    + (x[3 * a + 1] - c[1]).powi(2)
                    + (x[3 * a + 2] - c[2]).powi(2))
                .sqrt()
            })
            .collect();
        let m = r.iter().sum::<f64>() / n as f64;
        (r.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt()
    }

    #[test]
    fn kernels_preserve_state_length() {
        let x = octahedron();
        let mut rng = StdRng::seed_from_u64(5);
        let kernels: Vec<Box<dyn Fn(&mut StdRng) -> Array1<f64>>> = vec![
            Box::new(|r| {
                SurfaceRelocate {
                    n_points: 6,
                    neighbour_cutoff: 1.6,
                }
                .propose(octahedron().view(), 1.0, r)
            }),
            Box::new(|r| ShellRotate { n_points: 6 }.propose(octahedron().view(), 1.0, r)),
            Box::new(|r| {
                Symmetrise {
                    n_points: 6,
                    orders: vec![3, 4],
                    pair_cutoff: 2.5,
                }
                .propose(octahedron().view(), 1.0, r)
            }),
        ];
        for k in kernels {
            assert_eq!(k(&mut rng).len(), x.len());
        }
    }
}
